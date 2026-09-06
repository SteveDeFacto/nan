#!/usr/bin/env python3
"""
worker.py -- the shielded GPU worker. Runs on the UNTRUSTED host, holds the card.

This is the component docs/shielded-inference.md calls "the worker", and it is
deliberately the least trusted thing in the system. It sees:

  * public weights, in their native GGUF quantisation, and
  * masked activations: x + r over Z_M for a one-time pad r it never receives.

It does not see, and cannot be asked to compute, anything else. No softmax, no
norms, no rope, no sampling, no attention. Those are refused by name in
protocol.py's denylist and the refusal is diagnostic, not a generic error.

TRUST POSTURE -- read this before "improving" anything here
-----------------------------------------------------------
The operator of this process is assumed hostile and assumed to have replaced it
entirely. Nothing in this file is a security control for the TENANT: confidentiality
comes from the mask, and honesty comes from Freivalds verification in the TEE. What
the admission rules buy is that a worker following them cannot be turned into a
general-purpose execution and exfiltration primitive on the GPU host -- which
protects the HOST OPERATOR and every other tenant of that box, not the tenant whose
tokens are passing through.

So: a bug here cannot leak plaintext (there is none to leak), but a bug here CAN
turn a GPU box into an attacker's shell. That is why it fails closed on every path.

WHAT IT ACTUALLY COMPUTES
-------------------------
FIELD_GEMM: y = (x+r) . W over Z_M, with W dequantised from q8_0 and re-encoded to
RNS residues in registers, and the CRT recombination fused into the epilogue. The
kernel is kernels/fused_field_gemm.py; this file is its server. The TEE subtracts
u = r.W to recover x.W exactly.

CONCURRENCY
-----------
One thread per connection, one ShieldedWorkerState per connection, and a single
global lock around kernel launches. Per-connection state means a hostile or broken
peer cannot reach another peer's buffers even by index confusion: bids are scoped
to the connection and freed with it. VRAM, however, is shared, so the lock also
serialises allocation accounting.
"""

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))

import numpy as np

# "cuda" (the GPU half by definition) or "cpu" (a reference/test mode: the
# pre-encoded float64 path is exact on either; no throughput claim is posted).
DEVICE = "cuda"
import torch

from protocol import (CMD_ALLOC_BUFFER, CMD_FIELD_GEMM, CMD_FIELD_GEMM24, CMD_FREE_BUFFER,
                      CMD_GET_TENSOR, CMD_GRAPH_INSTALL, CMD_GRAPH_RECOMPUTE,
                      CMD_HELLO, CMD_SET_TENSOR, PROTO_VERSION, ProtocolViolation,
                      ReservationLedger, ShieldedWorkerState)
from field import M_MOD, Q0, Q1, Q2, crt_host
import wire
from fused_field_gemm import QK, field_gemm

GPU_LOCK = threading.Lock()

# Measured once at startup and reported in HELLO. The box advertises this to the
# fleet, and it has to be a MEASUREMENT of the field GEMM on this card, not a spec
# sheet: the shielded tier's throughput is the fused masked kernel's, which reads
# 1.0625 B/weight of q8_0 and is bandwidth-bound at the shapes that matter. A
# vendor FP16 tensor-core figure would overstate it by a wide margin and would be
# describing an operation this worker never performs.
FIELD_GMACS = 0.0


def measure_field_throughput(m=32, K=4096, N=4096, iters=12):
    """Field GEMM throughput on this card, in G-MAC/s. Best-effort: a failure
    here must never stop the worker serving, it just leaves the figure unposted
    and the box advertises capacity without a throughput claim."""
    try:
        import numpy as _np
        from fused_field_gemm import make_masked_activation, make_weights
        rng = _np.random.default_rng(0)
        wq, wd = make_weights(K, N, rng)
        _, _, x_res = make_masked_activation(m, K, rng)
        wq_t = torch.from_numpy(wq).cuda()
        wd_t = torch.from_numpy(wd).cuda()
        xr = [torch.from_numpy(p).cuda() for p in x_res]
        for _ in range(3):
            field_gemm(xr, wq_t, wd_t, m, N)      # warm up, and compile the shape
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            field_gemm(xr, wq_t, wd_t, m, N)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        del wq_t, wd_t, xr
        torch.cuda.empty_cache()
        return (m * K * N * iters) / dt / 1e9 if dt > 0 else 0.0
    except Exception as e:                                   # noqa: BLE001
        print(f"[shielded-worker] throughput probe failed ({type(e).__name__}: {e}); "
              f"serving anyway, capacity posts without a throughput figure", flush=True)
        return 0.0


# A CUDA context, once broken, stays broken: every later call on it fails the same
# way. These are the signatures of that, as opposed to a bad request we should
# merely refuse. Matched on TEXT because torch surfaces them through several
# exception types (RuntimeError, AcceleratorError) whose names have moved between
# versions, and a missed match here costs a restart while a false match costs
# nothing but one.
_FATAL_DEVICE_MARKERS = (
    "mps server",                 # cudaErrorMpsRpcFailure and friends
    "mps client",
    "cuda error",
    "device-side assert",
    "unspecified launch failure",
    "illegal memory access",
    "no cuda gpus are available",
    "initialization error",
)


def _is_fatal_device_error(e: BaseException) -> bool:
    text = f"{type(e).__name__}: {e}".lower()
    return any(m in text for m in _FATAL_DEVICE_MARKERS)


class Node:
    """A resolved, validated FIELD_GEMM. Resolution happens once at install; the
    per-token doorbell then does no parsing at all, which is both the fast path
    and the safe one -- there is no attacker-supplied structure left to
    misinterpret at compute time."""

    __slots__ = ("nid", "wq", "wd", "wf", "xs", "y", "K", "N", "max_m")

    def __init__(self, nid, wq, wd, xs, y, K, N, max_m, wf=None):
        self.nid, self.wq, self.wd, self.xs, self.y = nid, wq, wd, xs, y
        self.K, self.N, self.max_m = K, N, max_m
        self.wf = wf     # pre-encoded (N,K) int8, when the peer sent "w"

    def gemm(self, xr, m):
        """(m,N) int32 products of the residue planes xr (3 x (m,K) int8)."""
        if self.wf is None:
            return field_gemm(xr, self.wq, self.wd, m, self.N)
        # The pre-encoded path, exact in float64 (K * 125 * 119 << 2^53), then
        # CRT on the device. Reference speed, not production speed: the C++
        # worker is where the (N,K) layout is fast.
        w = self.wf.to(torch.float64).t()
        res = []
        for p, q in zip(xr, (Q0, Q1, Q2)):
            a = (p[:m].to(torch.float64) @ w).to(torch.int64)
            res.append(torch.remainder(a, q))
        return crt_torch(res[0], res[1], res[2]).to(torch.int32)


def crt_torch(r0, r1, r2):
    """Garner recombination on device tensors, balanced in (-M/2, M/2]."""
    inv01 = pow(Q0 % Q1, -1, Q1)
    inv012 = pow((Q0 * Q1) % Q2, -1, Q2)
    t1 = torch.remainder((r1 - r0) * inv01, Q1)
    x = r0 + Q0 * t1
    t2 = torch.remainder((r2 - torch.remainder(x, Q2)) * inv012, Q2)
    x = x + (Q0 * Q1) * t2
    return torch.where(x > M_MOD // 2, x - M_MOD, x)


class Connection:
    def __init__(self, sock, addr, vram_bytes, log, ledger=None):
        self.sock = sock
        self.addr = addr
        self.log = log
        # 1.3 reservations are accounted here (one ledger per process) but not
        # claimed from the driver: this fixture leaves the claim to the CUDA
        # worker, which is what runs on the fleet.
        self.state = ShieldedWorkerState(vram_bytes=vram_bytes, ledger=ledger)
        self.storage = {}      # bid -> uint8 cuda tensor
        self.nodes = []
        self.recomputes = 0
        self.gemm_ms = 0.0

    # -- buffer helpers ----------------------------------------------------
    def _bytes_view(self, bid, offset, nbytes):
        """A uint8 view of a region. protocol.py has already bounds-checked it;
        this re-derives the same bound from the real allocation rather than
        trusting that, because the two must never be able to disagree."""
        buf = self.storage.get(bid)
        if buf is None:
            raise ProtocolViolation(f"no storage for buffer {bid}")
        if offset < 0 or nbytes < 0 or offset + nbytes > buf.numel():
            raise ProtocolViolation(f"region outside storage for buffer {bid}")
        return buf[offset:offset + nbytes]

    def _typed(self, bid, offset, shape, dtype, role=None):
        """A typed view of a region, with the element count checked against the
        shape. Alignment is enforced: a misaligned view() would either throw deep
        inside torch or, worse, silently reinterpret."""
        itemsize = torch.empty(0, dtype=dtype).element_size()
        n = 1
        for s in shape:
            n *= s
        nbytes = n * itemsize
        if role is not None:
            declared = self.state.buffers.get(bid)
            if declared is None or declared.role != role:
                raise ProtocolViolation(
                    f"buffer {bid} has role {declared.role if declared else None!r}, want {role!r}")
        if offset % itemsize:
            raise ProtocolViolation(f"offset {offset} misaligned for {dtype}")
        return self._bytes_view(bid, offset, nbytes).view(dtype).view(*shape)

    # -- command handlers --------------------------------------------------
    def handle(self, cmd, payload):
        # Admission FIRST, always. Everything below may assume the frame is
        # structurally legal and in-bounds; nothing below may assume it is
        # semantically sane.
        res = self.state.handle(cmd, payload)

        if cmd == CMD_HELLO:
            # vram_free is read from the driver, not from our own accounting. The
            # box advertises this number to the fleet, and the honest figure is
            # what the card actually has -- including whatever the host operator
            # is doing with it behind our back, which on a desktop is a running
            # X server. Our ledger cannot see that; mem_get_info can.
            if DEVICE == "cuda":
                free_b, total_b = torch.cuda.mem_get_info(0)
                props = torch.cuda.get_device_properties(0)
                dev = dict(device=props.name, vram_total=props.total_memory, vram_free=int(free_b),
                           sm_count=props.multi_processor_count, capability=f"{props.major}.{props.minor}")
            else:
                dev = dict(device="cpu (reference)", vram_total=self.state.vram_bytes, vram_free=self.state.vram_bytes,
                           sm_count=0, capability="0.0")
            info = {
                "version": list(PROTO_VERSION),
                "vram_budget": self.state.vram_bytes,
                "vram_reserved": self.state.ledger.reserved,
                "vram_reserve": self.state.reserve,
                "field_gmac_per_s": round(FIELD_GMACS, 1),
                "worker": "shielded/worker.py",
                **dev,
            }
            return json.dumps(info).encode()

        if cmd == CMD_ALLOC_BUFFER:
            bid = res["bid"]
            size = struct.unpack_from("<Q", payload, 0)[0]
            try:
                self.storage[bid] = torch.empty(size, dtype=torch.uint8, device=DEVICE)
            except torch.cuda.OutOfMemoryError as e:
                # The accounting said it fits and the driver disagreed. Fail the
                # connection rather than leaving the state machine's view of VRAM
                # diverged from the card's.
                raise ProtocolViolation(f"device allocation of {size} failed: {e}")
            return struct.pack("<Q", bid)

        if cmd == CMD_FREE_BUFFER:
            bid = struct.unpack_from("<Q", payload, 0)[0]
            self.storage.pop(bid, None)
            return b""

        if cmd == CMD_SET_TENSOR:
            bid, offset, nbytes = struct.unpack_from("<QQQ", payload, 0)
            data = payload[24:]
            if len(data) != nbytes:
                raise ProtocolViolation(
                    f"SET_TENSOR declared {nbytes} bytes, frame carries {len(data)}")
            view = self._bytes_view(bid, offset, nbytes)
            src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            view.copy_(src, non_blocking=False)
            return b""

        if cmd == CMD_GET_TENSOR:
            bid, offset, nbytes = struct.unpack_from("<QQQ", payload, 0)
            view = self._bytes_view(bid, offset, nbytes)
            return view.cpu().numpy().tobytes()

        if cmd == CMD_GRAPH_INSTALL:
            spec = json.loads(payload.decode("utf-8"))
            self._resolve(spec)
            return json.dumps({"nodes": len(self.nodes)}).encode()

        if cmd == CMD_GRAPH_RECOMPUTE:
            node_idx, m = wire.unpack_recompute(payload)
            return self._recompute(node_idx, m)

        if cmd in (CMD_FIELD_GEMM, CMD_FIELD_GEMM24):
            ids, m, K, at = res["nodes"], res["m"], res["K"], res["planes_at"]
            planes = torch.frombuffer(bytearray(payload[at:]), dtype=torch.int8).view(3, m, K).to(DEVICE)
            xr = [planes[p] for p in range(3)]
            outs = []
            t0 = time.perf_counter()
            with GPU_LOCK:
                for i in ids:
                    node = self.nodes[i]
                    outs.append(node.gemm(xr, m).contiguous())
                torch.cuda.synchronize()
            self.gemm_ms += (time.perf_counter() - t0) * 1e3
            self.recomputes += len(ids)
            if cmd == CMD_FIELD_GEMM24:
                # 1.2: the same products as 3-byte little-endian two's
                # complement. Balanced in (-M/2, M/2], M < 2^24, so the low
                # three bytes of the int32 form ARE the value: drop the
                # top byte of each little-endian int32.
                reply = b"".join(o.cpu().numpy().astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
                                 for o in outs)
                if len(reply) != res["reply_bytes"]:
                    raise ProtocolViolation("internal: packed reply size disagrees with the rule")
                return reply
            return b"".join(o.cpu().numpy().tobytes() for o in outs)

        return b""

    # -- graph resolution --------------------------------------------------
    def _resolve(self, spec):
        """Bind every allowlisted node to real storage, or refuse the graph.

        protocol.py validated the OP of each node. This validates the BINDINGS:
        shapes, sizes, alignment, and -- the invariant that matters -- that
        weights come from a 'weights' buffer and activations from an
        'activations' buffer. Without that last check a graph could declare an
        activation region as its weight operand, and the worker would happily
        treat a masked activation as public data.
        """
        for i, n in enumerate(spec["nodes"]):
            if n.get("op") != "FIELD_GEMM":
                # VIEW/RESHAPE/PERMUTE/TRANSPOSE/CONT/CPY are metadata-only and
                # carry no compute; they are allowed through admission but this
                # worker resolves them to nothing. Keeping them in the allowlist
                # matters for the C++ port, where sched does emit them.
                self.nodes.append(None)
                continue
            K, N, max_m = int(n["K"]), int(n["N"]), int(n["max_m"])
            if K <= 0 or N <= 0 or max_m <= 0:
                raise ProtocolViolation(f"node {i}: non-positive shape")
            if K % QK:
                raise ProtocolViolation(f"node {i}: K={K} is not a multiple of {QK}")
            if max_m > 4096:
                raise ProtocolViolation(f"node {i}: max_m={max_m} exceeds 4096")
            if "w" in n:
                # Pre-encoded (N,K) int8 -- what the engine backend sends. The
                # int8 lane bound is a correctness gate, not a style check.
                wf = self._typed(n["w"]["bid"], n["w"]["offset"], (N, K), torch.int8, "weights")
                if int(wf.abs().max()) > 119:
                    raise ProtocolViolation(f"node {i}: a fixed weight exceeds the int8 lane")
                wq = wd = None
            else:
                wq = self._typed(n["wq"]["bid"], n["wq"]["offset"], (K, N), torch.int8, "weights")
                wd = self._typed(n["wd"]["bid"], n["wd"]["offset"], (K // QK, N), torch.float16, "weights")
                wf = None
            xs = []
            xoff = n["x"]["offset"]
            plane = max_m * K
            for p in range(3):
                xs.append(self._typed(n["x"]["bid"], xoff + p * plane, (max_m, K),
                                      torch.int8, "activations"))
            y = self._typed(n["y"]["bid"], n["y"]["offset"], (max_m, N), torch.int32, "activations")
            self.nodes.append(Node(n.get("id", f"node{i}"), wq, wd, xs, y, K, N, max_m, wf))
        if not any(nd is not None for nd in self.nodes):
            raise ProtocolViolation("graph contains no computable node")

    def _recompute(self, node_idx, m):
        if node_idx >= len(self.nodes):
            raise ProtocolViolation(f"recompute of node {node_idx}, graph has {len(self.nodes)}")
        node = self.nodes[node_idx]
        if node is None:
            raise ProtocolViolation(f"node {node_idx} is metadata-only; nothing to compute")
        if m < 1 or m > node.max_m:
            raise ProtocolViolation(f"m={m} outside [1,{node.max_m}] for node {node_idx}")
        t0 = time.perf_counter()
        with GPU_LOCK:
            xr = [p[:m] for p in node.xs]
            out = node.gemm(xr, m)
            node.y[:m].copy_(out)
            torch.cuda.synchronize()
        self.gemm_ms += (time.perf_counter() - t0) * 1e3
        self.recomputes += 1
        return struct.pack("<I", m)

    # -- loop --------------------------------------------------------------
    def serve(self):
        try:
            while True:
                try:
                    cmd, payload = wire.recv_request(self.sock)
                except ConnectionError:
                    break
                try:
                    resp = self.handle(cmd, payload)
                except ProtocolViolation as e:
                    self.log(f"VIOLATION from {self.addr}: {e}")
                    self.sock.sendall(wire.build_response(wire.STATUS_VIOLATION, str(e).encode()))
                    break
                except Exception as e:  # noqa: BLE001
                    # Any unexpected exception is treated as a violation too: an
                    # unhandled error means we no longer know what state we are
                    # in, and guessing is exactly what fails-closed forbids.
                    self.log(f"INTERNAL from {self.addr}: {type(e).__name__}: {e}")
                    self.sock.sendall(wire.build_response(
                        wire.STATUS_VIOLATION, f"internal: {type(e).__name__}: {e}".encode()))
                    # A DEAD CUDA CONTEXT IS FATAL TO THE PROCESS, not just to this
                    # connection. Observed on 2026-08-25: the MPS server went away,
                    # every subsequent request failed with cudaErrorMpsRpcFailure,
                    # and the worker went on accepting connections and failing all
                    # of them for hours. Its launcher already restarts it on exit
                    # (metal/enclave-metal.mjs) -- that supervision was simply
                    # unreachable, because the process never died. A context cannot
                    # be repaired in place, so the honest response is to stop and
                    # let a fresh process rebuild one. Availability is the thing
                    # this tier explicitly does not promise; serving garbage
                    # forever is worse than a restart.
                    if _is_fatal_device_error(e):
                        self.log(f"FATAL: the CUDA context is gone ({type(e).__name__}). "
                                 f"Exiting so the launcher can restart with a fresh one.")
                        os._exit(70)          # EX_SOFTWARE; bypasses threads holding GPU_LOCK
                    break
                self.sock.sendall(wire.build_response(wire.STATUS_OK, resp))
        finally:
            # Per-connection VRAM dies with the connection. Nothing secret was
            # ever in it, but a leaked buffer is a denial-of-service on the card.
            self.storage.clear()
            torch.cuda.empty_cache()
            self.state.release()
            try:
                self.sock.close()
            except OSError:
                pass


def serve(host, port, vram_gb, quiet=False):
    def log(msg):
        if not quiet:
            print(f"[shielded-worker] {msg}", flush=True)

    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device; the shielded worker is the GPU half by definition")
        props = torch.cuda.get_device_properties(0)
        budget = int(vram_gb * (1 << 30)) if vram_gb else int(props.total_memory * 0.85)
        log(f"{props.name}, sm_{props.major}{props.minor}, "
            f"{props.total_memory / 2**30:.1f} GiB total, budget {budget / 2**30:.1f} GiB")
        global FIELD_GMACS
        FIELD_GMACS = measure_field_throughput()
        if FIELD_GMACS:
            log(f"field GEMM throughput {FIELD_GMACS:.0f} G-MAC/s (measured, masked path)")
    else:
        budget = int((vram_gb or 4) * (1 << 30))
        log(f"CPU reference mode (float64 field GEMM, exact, slow), budget {budget / 2**30:.1f} GiB")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    log(f"listening on {host}:{port}"
        + (f" (guest reaches it at 10.0.2.2:{port})" if host in ("127.0.0.1", "0.0.0.0") else ""))

    ledger = ReservationLedger(budget)
    while True:
        sock, addr = srv.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn = Connection(sock, addr, budget, log, ledger)
        t = threading.Thread(target=conn.serve, daemon=True, name=f"conn-{addr[1]}")
        t.start()


def main():
    ap = argparse.ArgumentParser(description="shielded GPU worker (untrusted host side)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; 127.0.0.1 is reachable from a slirp guest at 10.0.2.2")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SHIELDED_PORT", "9500")))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="cpu = exact float64 reference mode for tests; no GPU touched")
    ap.add_argument("--vram-gb", type=float, default=0.0, help="0 = 85%% of the card")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    global DEVICE
    DEVICE = a.device
    serve(a.host, a.port, a.vram_gb, a.quiet)


if __name__ == "__main__":
    main()
