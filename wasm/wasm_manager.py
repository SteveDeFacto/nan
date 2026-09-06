#!/usr/bin/env python3
"""wasm-manager: per-tenant WebAssembly app manager for Enclave on Tinfoil.

Replaces the runsc/microVM backends. Nested container runtimes need privileged
namespace/mount/exec operations that Tinfoil confines away; a Wasm runtime needs
none of them. Each tenant app is a `wasmtime serve` subprocess bound to a unique
host loopback port. Isolation is provided by three independent layers, all
unprivileged: the Wasm sandbox (memory-safe, no ambient authority), the OS
process boundary (separate PID, own process group), and curated WASI (no
filesystem, no host env, no network beyond the served HTTP socket).

It speaks the SAME HTTP contract the supervisor already uses for the "vm"
backend, so the supervisor needs no change:

  POST   /vms   {image, cpuShare, gpuShare?, gpuTflops?, cpuGflops?, name?, appPort?}
                 (cpuGflops in GFLOPS; legacy cpuTflops accepted as x1000)
                 -> 201 {id, status, endpoint, hostPort, ...}
  DELETE /vms/:id        -> {id, deleted: true}
  GET    /vms/:id | /vms | /health | /capacity | /catalog | /debug/env

Plus the encrypted-volume tenant plane (per-deployment token, NOT the control
token; see the ENC_* block): GET /encvol/:id, POST /encvol/:id/{unlock|sync|lock}.

Notes:
- `image` is reinterpreted as a Wasm APP REFERENCE:
    * `ipfs://<cid>` — the normal (and, through the supervisor, the ONLY) form:
      fetched from IPFS_GATEWAY and VERIFIED: we pull the DAG as a CAR, check
      every block hashes to its CID, and reassemble the file rooted at the
      requested CID (see ipfs_fetch.py). A tampering gateway fails the hash
      check, so "what ran == this exact CID" holds without trusting the gateway.
      The verified CID is what the supervisor folds into attestation.
    * a catalog id — only if a catalog file exists (WASM_CATALOG; none ships in
      the image: the only baked .wasm is nn-demo.wasm, the boot probe's fixture,
      baked under FIXTURES_DIR — outside the wasm-cache mount — and launched
      by the probes directly without going through this resolution).
    * an absolute path to a .wasm already under APPS_DIR (internal/debug; the
      supervisor's approval gate never forwards these).
- Apps must be wasi:http components (what `wasmtime serve` runs). A WASIX/wasmer
  socket-server launcher can be added behind the same LAUNCHER seam later.
- Attached model volumes (MODEL_VOLUMES) that carry a GGUF are preloaded as
  host wasi-nn graphs for GPU tenants (see _gguf_path / _stage_nn_graph);
  volumes named in MODEL_VOLUMES_SD preload through the stable-diffusion.cpp
  backend instead (image checkpoints - see _sd_checkpoint_path). A
  volume may ship a single *.gguf OR a llama.cpp split family
  ("<prefix>-NNNNN-of-MMMMM.gguf"); the whole family is staged together so
  models larger than HF's 50GB per-file cap load as one graph.
"""
import collections
import functools
import hashlib
import hmac
import http.server
import ipaddress
import json
import math
import os
import pathlib
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid

try:
    import ipfs_fetch   # local module (same dir): fetch + verify a wasm by IPFS CID
except Exception as _e:   # optional feature — never let a missing module take down the manager
    ipfs_fetch = None
    print(f"[wasm-manager] run-by-CID disabled: {_e}", flush=True)

# ---- config ---------------------------------------------------------------- #
PORT         = int(os.environ.get("WASM_MANAGER_PORT", "8091"))   # same port the supervisor expects
# Control-plane token (opt-in): tenants hold outbound HTTP to loopback (`serve`
# grants -Shttp), so with the tenant data plane advertising this port the
# control API must not stay open - one tenant could DELETE another's vm. When
# set (the enclave config passes the shared SECRET), every control route
# demands it; /health and the tenant data plane (/enc/*, its own per-deployment
# tokens) stay open.
#
# FAIL CLOSED: with neither VMMGR_TOKEN nor SECRET set, the control plane no
# longer stays legacy-open — a tenant holds outbound HTTP to loopback and could
# DELETE another tenant's vm, so a box with no token must DENY. Explicitly running
# open (local dev) is opt-in: VMMGR_ALLOW_UNAUTHENTICATED=1.
# DERIVED, not the raw fleet SECRET. This token rides as a bearer header on
# every control request the supervisor makes, and the raw SECRET is the master
# the fleet's OTHER credentials come from (HMAC(SECRET,"enclave dns-txt v1"),
# HMAC(SECRET,"enclave secrets v1")) - so putting it on the wire of every
# launch/kill call made one loopback-observable header worth the whole keyring.
# Same labelled-HMAC discipline as those two; supervisor.js derives identically.
_VMMGR_EXPLICIT = os.environ.get("VMMGR_TOKEN") or ""
_SECRET_RAW     = os.environ.get("SECRET") or ""
VMMGR_TOKEN = _VMMGR_EXPLICIT or (
    hmac.new(_SECRET_RAW.encode("utf-8", "surrogateescape"), b"enclave vmmgr v1", hashlib.sha256).hexdigest()
    if _SECRET_RAW else "")
# ROLLOUT WINDOW CLOSED (2026-07-27). While supervisor and wasm-manager were
# rolling independently this also accepted the RAW SECRET, so a staggered update
# could not take the control plane down mid-flight. Both live enclaves now run
# post-derivation supervisors (they advertise the rev-8 `rateCap` capability,
# which shipped long after the derivation), so the master is no longer accepted
# anywhere on the wire. Re-adding it would mean one observed loopback header is
# again worth HMAC(SECRET,"enclave dns-txt v1") + HMAC(SECRET,"enclave secrets
# v1") as well as this plane; if a future staggered rollout needs a window, set
# VMMGR_TOKEN explicitly on both sides instead of reopening this.
VMMGR_ALLOW_UNAUTH = os.environ.get("VMMGR_ALLOW_UNAUTHENTICATED", "").strip().lower() in ("1", "true", "yes", "on")
# /health is intentionally OPEN (no control token) for liveness probes — the
# container healthcheck curls it — but its FULL body leaks capacity (including
# what OTHER tenants hold committed), model-volume names/listings, GPU specs and
# the verbose GPU-probe diagnostics to any loopback-reaching caller, and a
# tenant CAN reach loopback (the wasmtime egress patch carves it out so apps can
# dial /encvol). So the detailed half is withheld from unauthenticated callers.
#
# DEFAULT ON since 2026-07-27. It was off pending "is every real consumer
# authenticated?" — they are: supervisor.js's vmReq attaches X-Vmmgr-Token to
# EVERY manager call, /health included, so the supervisor still gets the full
# body; the healthcheck only needs the 200. Every config in the tree
# (enclaves/*, metal/guest/gsup.mjs) already set 1, so this only changes what a
# third-party operator gets by writing no config at all — which is exactly the
# case that should be safe by default. WASM_HEALTH_MINIMAL=0 restores the old
# open body for local debugging.
HEALTH_MINIMAL = os.environ.get("WASM_HEALTH_MINIMAL", "1").lower() in ("1", "true", "on")
WASMTIME     = os.environ.get("WASMTIME_BIN", "wasmtime")
APPS_DIR     = pathlib.Path(os.environ.get("WASM_APPS_DIR", "/opt/enclave/apps"))
CATALOG_PATH = pathlib.Path(os.environ.get("WASM_CATALOG", str(APPS_DIR / "catalog.json")))
# The baked probe fixture lives OUTSIDE APPS_DIR: the persistent wasm-cache
# volume mounts over APPS_DIR, and on 2026-08-23 (v0.5.486) the initially
# empty mount shadowed the baked nn-demo.wasm — the arbiter capability probe
# read that as "toolchain unproven" and silently hard-capped every GPU tenant
# to its share-sized SM slice. Nothing a tenant mount can shadow may ever
# gate a capability, so the fixture gets its own un-mounted directory.
FIXTURES_DIR = pathlib.Path(os.environ.get("WASM_FIXTURES_DIR", "/opt/enclave/fixtures"))


def _fixture_wasm() -> pathlib.Path:
    """The boot/capability probes' fixture. Prefer the un-shadowable
    FIXTURES_DIR copy; fall back to the legacy APPS_DIR location so this
    manager still finds the fixture inside a pre-move image (dev checkouts,
    metal's separately-pinned manager). Callers keep their own is_file()
    check — with neither copy present the fallback path simply won't exist."""
    p = FIXTURES_DIR / "nn-demo.wasm"
    return p if p.is_file() else APPS_DIR / "nn-demo.wasm"
HOST_IP      = os.environ.get("WASM_HOST_IP", "127.0.0.1")
NODE_VCPUS   = int(os.environ.get("NODE_VCPUS", "16"))
NODE_RAM_GB  = int(os.environ.get("NODE_RAM_GB", "64"))
# Does this node have a GPU attached? Catalog apps that declare a GPU need
# (vram_mb or gpu_gflops > 0) are refused at launch on CPU-only nodes
# (enclaves/cpu/tinfoil-config.yml sets NODE_HAS_GPU=0). Apps without a GPU
# need are CPU-only and run anywhere the routing sends them.
NODE_HAS_GPU = os.environ.get("NODE_HAS_GPU", "0").lower() in ("1", "true", "on")
# CPU-only nodes: how much node RAM to hold back from mmap-backed ggml
# preloads (base system + other tenants' KV/compute + slack), and the escape
# hatch back to the strict per-share rule. See _build_cmd's budget block for
# why mmap'd weights are budgeted against the NODE rather than the share.
CPU_NN_RESERVE_GB = float(os.environ.get("WASM_CPU_NN_RESERVE_GB", "6") or 6)
CPU_NN_BUDGET = (os.environ.get("WASM_CPU_NN_BUDGET", "node") or "node").strip().lower()
# Deployments buy SHARES: cpuShare is this manager's admission unit and sets
# the guest linear-memory ceiling (wasmtime -W max-memory-size = cpuShare ×
# NODE_RAM_GB). The app's catalog specs (mem_mb etc.) only set the minimum
# share, so the cap is always >= what the app declared. Direct callers may
# still pass an explicit memMb to cap lower.
MIN_MEM_MB   = int(os.environ.get("WASM_APP_MIN_MEM_MB", "64"))
# Readiness window for a spawned tenant's port. The loop exits EARLY on
# port-open or process death, so this only bounds the slowest legitimate case:
# the FIRST launch of a big component per CVM boot, where wasmtime must
# cranelift-compile it cold (llm-chat is 123MB; under TDX that can far exceed
# the old 20s - observed live 2026-07-05 as deterministic "failed" adopts
# while everything else was healthy). Later launches hit wasmtime's compile
# cache and open the port in seconds.
READY_SECS   = float(os.environ.get("WASM_READY_TIMEOUT", "150"))
# how long launch() waits for a wanted model volume to finish mounting before
# failing the launch (node boot races the Modelwrap dm-verity mounts). Sized
# so VOL_READY_SECS + READY_SECS stays under the supervisor's 300s spawn
# budget - beyond it, fail fast and let the supervisor's backoff retry.
VOL_READY_SECS = float(os.environ.get("WASM_VOL_READY_TIMEOUT", "90"))
# how long the boot preload may take before the tenant is declared broken
# (_watch_preloads). Generous: a split 60GB family lifts EROFS -> VRAM.
NN_PRELOAD_VERIFY_SECS = float(os.environ.get("WASM_NN_PRELOAD_VERIFY_TIMEOUT", "900"))
# boot warmup (app-config `warmup` key): how long the background GET may take.
# Generous on purpose - its whole job is pulling big weights into VRAM.
WARMUP_SECS  = float(os.environ.get("WASM_WARMUP_TIMEOUT", "600"))
MOCK         = os.environ.get("WASM_MOCK", "") not in ("", "0", "false")
LOG_DIR      = pathlib.Path(os.environ.get("WASM_LOG_DIR", "/tmp/enclave-wasm-logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
# run-by-CID: fetch an app's bytes from IPFS and verify they hash to the CID.
IPFS_GATEWAY   = os.environ.get("IPFS_GATEWAY", "https://ipfs.enclave.host").rstrip("/")
# cap on a fetched app: models ride inside the wasm (llm-chat 0.2 embeds a
# 460MB q4f16 LLM), so the ceiling is set by fetch/compile budgets, not code.
# Note wasm32's 4GiB linear memory still bounds what an EMBEDDED model can be:
# include_bytes + the load() copy means ~1.5-2GB of model is the practical top.
WASM_MAX_BYTES = int(os.environ.get("WASM_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))
# a 2GB CAR at gateway speeds (~3.5MB/s) needs ~10min. The supervisor's
# prefetch call gives up at 300s, but this fetch keeps running and fills the
# cache - the supervisor's backed-off retry then hits the cache. This budget
# just has to outlast the whole fetch so the cache actually fills.
IPFS_TIMEOUT   = float(os.environ.get("IPFS_FETCH_TIMEOUT", "660"))
# firewall: per-version ports config from the catalog. Logical ports are LABELS
# (each deployment binds a remapped actual), so classic low numbers are allowed —
# a DNS app may advertise udp:53. The ceiling matches what the public relay
# binds (RELAY_PORTS=1-49999; its ephemeral range is pinned above the ceiling);
# reserved keeps labels off infrastructure (supervisor 8080, this manager 8091).
# Privileged actuals are never attempted: logical < 1024 is ALWAYS remapped to
# a free high port (unprivileged processes can't bind them).
PORT_MIN_DECL  = 1
PORT_MAX_DECL  = 49999
PRIV_PORT_MAX  = 1023
RESERVED_PORTS = {8080, 8091}
AUDIT_SECS     = float(os.environ.get("WASM_AUDIT_INTERVAL", "10"))
# CROSS-TENANT loopback policing (see _audit_peers). Tenants share the CVM's
# network namespace, so one tenant CAN open a TCP connection straight to another
# tenant's assigned loopback port - around /x/:id, which is where a PRIVATE
# deployment's owner-only check and the deployer's WAF live. There is no
# per-address gate available at the runtime today (see _build_cmd's SECURITY
# note), so this is measure-and-kill, exactly like the port firewall and the
# /data ceiling above it.
#   1/true/on (default) - kill the tenant that dialled a sibling
#   warn                - record it on the record and log, kill nothing
#   0/false/off         - don't look
WASM_PEER_AUDIT = (os.environ.get("WASM_PEER_AUDIT", "1") or "1").strip().lower()
PEER_AUDIT_ON   = WASM_PEER_AUDIT not in ("0", "false", "off", "no")
PEER_AUDIT_KILL = PEER_AUDIT_ON and WASM_PEER_AUDIT != "warn"
# wasi-nn GPU interface: a deployment that BUYS a GPU share (gpuShare > 0)
# is launched with `-S nn`, so the guest can run inference through the host's
# backends — ONNX Runtime for ONNX graphs, llama.cpp for GGUF, and
# stable-diffusion.cpp for image checkpoints; ExecutionTarget::Gpu maps to
# CUDA, ::Cpu to the CPU. Enforcement of the share is
# the SAME mechanism as the worker backend's PTX children: the tenant's
# wasmtime process is launched with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (SM cap)
# and CUDA_MPS_PINNED_DEVICE_MEM_LIMIT (VRAM cap = gpuShare × GPU_VRAM_GB), so
# the MPS daemon hardware-enforces both. Tenants that didn't buy a GPU share
# don't get the flag at all — a component importing wasi:nn then fails to
# instantiate, which is the admission control ("pay for a share to use the
# card"). WASM_NN=0 is the fleet-wide kill-switch (same shape as WASM_P3).
NN_ENABLED   = os.environ.get("WASM_NN", "1").lower() not in ("0", "false", "no")
GPU_VRAM_GB  = float(os.environ.get("GPU_VRAM_GB", "141"))
# The card outranks config: GPU_VRAM_GB above is only the fallback for when the
# card can't be asked (CPU node, mock, driver hiccup). This container holds the
# GPU, so probe memory.total at boot and size the MPS caps — and the gpuVramGb
# /health reports upward to the supervisor — from the hardware itself.
GPU_VRAM_SRC = "env" if "GPU_VRAM_GB" in os.environ else "default"

def _probe_card_vram_gb():
    """(smallest attached card's memory.total in GiB, card count) - nvidia-smi
    reports MiB, one line per card. None on probe failure."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        mib = [float(x) for x in r.stdout.split() if x]
        gb = round(min(mib) / 1024, 1) if mib else 0.0
        return (gb, len(mib)) if 1 <= gb <= 8192 else None
    except Exception:                                    # noqa: BLE001
        return None

# cards on this node: the VRAM ledger's budget is per-node (GB x cards); the
# per-card packing itself is the supervisor allocator's job
GPU_CARDS = max(1, int(os.environ.get("GPU_COUNT", "1")))
if NODE_HAS_GPU and not MOCK:
    _vram = _probe_card_vram_gb()
    if _vram:
        GPU_VRAM_GB, GPU_VRAM_SRC = _vram[0], "nvidia-smi"
        GPU_CARDS = max(GPU_CARDS, _vram[1])
        print(f"[gpu] card VRAM probed: {GPU_VRAM_GB} GB x {GPU_CARDS} card(s) "
              f"(nvidia-smi memory.total)", flush=True)
    else:
        print(f"[gpu] card VRAM probe failed - using {GPU_VRAM_SRC} {GPU_VRAM_GB} GB", flush=True)
MPS_PIPE_DIR = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
# CUDA readiness probe (see _nn_probe_loop): a wasi-nn load() is a SYNCHRONOUS
# host call, so a CUDA init that HANGS (rather than errors) eats a runtime
# thread forever - a few retried GPU requests then wedge the whole tenant,
# including its CPU paths. Launching GPU tenants is therefore gated on a boot
# probe that does cuInit + primary-context retain (the MPS attach point) in a
# throwaway subprocess with the exact tenant env and a hard timeout, and
# bisects which layer breaks: full env -> without the pinned-VRAM limit ->
# without MPS. Result drives launches: full/nopin = go (nopin drops only the
# never-validated CUDA_MPS_PINNED_DEVICE_MEM_LIMIT; VRAM stays accounted by
# the supervisor's allocator), anything else = GPU launches are refused with
# the probe's diagnosis instead of hanging apps.
# --- attached model volumes (Tinfoil Modelwrap) --------------------------- #
# The enclave can carry read-only, ATTESTED model volumes: tinfoil-config.yml
# declares `models:` whose weights become dm-verity+EROFS images mounted at
# MODEL_VOLUME_ROOT/mpk-<root_hash> (the dm-verity root is on the kernel
# cmdline, so the enclave measurement commits to the exact bytes). A deployment
# attaches one or more by name; launch() preopens each into the guest as a
# read-only /models/<name> dir. This is how big models reach a tenant without
# riding the app wasm (no include_bytes, no IPFS fetch, no 4GiB linear-memory
# or hostcall-fuel ceiling on the weights). MODEL_VOLUME_ROOT is scanned for
# mpk-* mounts; MODEL_VOLUMES adds/overrides explicit name:path pairs (for
# local dev without a real Modelwrap mount). Names must be [a-z0-9-]+.
MODEL_VOLUME_ROOT = pathlib.Path(os.environ.get("MODEL_VOLUME_ROOT", "/tinfoil/mpk"))
_MODEL_VOLUMES_ENV = os.environ.get("MODEL_VOLUMES", "").strip()
# Volumes that preload through the stable-diffusion.cpp backend
# (-S nn-graph=sd::<dir>) instead of ggml/llama.cpp: comma-separated volume
# names. EXPLICIT by design - an image-diffusion GGUF (FLUX quant) is
# indistinguishable from an LLM GGUF by extension, and preloading a 13 GB
# checkpoint into the wrong backend fails only at load time. MODEL_VOLUMES'
# optional third field still picks the file within the volume.
_SD_VOLUMES_ENV = os.environ.get("MODEL_VOLUMES_SD", "").strip()
_SD_VOLUMES = {v.strip() for v in _SD_VOLUMES_ENV.split(",") if v.strip()}
# Working-set allowance for the te-on-cpu AUTO rule below (launch()): what a
# 1024px sdcpp pipeline needs on the GPU beyond resident weights - attention/
# compute buffers with FA plus the tiled-VAE decode. Measured live: z-image
# (10.4 GB weights) peaks ~13 GB at 1024px => ~2.6 GB; rounded up.
SD_TE_AUTO_HEADROOM = 3 << 30  # 3 GiB
_VOL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

NN_PROBE_TIMEOUT = float(os.environ.get("WASM_NN_PROBE_TIMEOUT", "75"))   # worker.py validated ~60s CC init; match its patience
# Long-patience budget for the "is it hung or just glacial?" e2e variant:
# under CC, first-time cuBLAS/cuDNN kernel loading can legitimately take
# minutes (encrypted bounce-buffered copies), which only LOOKS like a hang.
NN_PROBE_LONG = float(os.environ.get("WASM_NN_PROBE_LONG", "600"))
# Control experiment for the bisect's endgame: the worker container's manager
# (same box, shared localhost) can spawn ITS validated CUDA path - an
# MPS-capped cupy child - on request. If that works while ORT hangs, the fault
# is container/ORT-side; if it hangs too, GPU compute init is broken node-wide
# under CC and the escalation target is the platform, not this stack.
WORKER_MGR_URL = os.environ.get("WORKER_MGR_URL", "http://127.0.0.1:8090").rstrip("/")
_NN_PROBE = {"state": "probing" if (NN_ENABLED and NODE_HAS_GPU and not MOCK) else "off",
             "mode": None, "detail": "", "attempts": 0, "stage": None,
             "env": {}, "args": []}   # state: probing|ok|failed|off; mode: full|nopin; stage: what's running RIGHT NOW; env/args: extra tenant env + wasmtime flags the probe adopted

_NN_PROBE_SRC = r"""
import ctypes, sys
try:
    cu = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print(f"libcuda.so.1 unavailable ({e}): nvidia runtime not applied to this container?", flush=True); sys.exit(4)
def ck(name, rc):
    if rc != 0:
        print(f"{name} rc={rc}", flush=True); sys.exit(2)
ck("cuInit", cu.cuInit(0))
n = ctypes.c_int(0)
ck("cuDeviceGetCount", cu.cuDeviceGetCount(ctypes.byref(n)))
if n.value < 1:
    print("no CUDA devices visible", flush=True); sys.exit(3)
dev = ctypes.c_int(0)
ck("cuDeviceGet", cu.cuDeviceGet(ctypes.byref(dev), 0))
ctx = ctypes.c_void_p()
ck("cuDevicePrimaryCtxRetain", cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev))
print("ok", flush=True)
"""
# WASIp3 (component-model async): wasmtime 45 accepts `-S p3` on both `run`
# and `serve`, and no longer marks it experimental. The flag widens the API
# SURFACE only — wasip2 components ignore it, wasip3 components need it to
# instantiate — while network reach stays gated by the same tcp/udp/
# inherit-network grants and the bind audit polices what is actually bound
# (the egress and loopback patches cover the p3 code paths too: they hook
# wasi-http's p3/request.rs and wasi's p3 sockets, so the cross-tenant wall
# and dedicated-IP egress hold for both worlds).
# WASM_P3=0 drops the flag fleet-wide (operator kill-switch, e.g. if a
# wasmtime upgrade regresses p3) without rebuilding the image. Whether the
# flag is actually EMITTED is decided per binary by _p3_supported() — the
# loopback lesson applied in advance: never pass an option this wasmtime
# does not have.
_P3_ENV_ENABLED = os.environ.get("WASM_P3", "1").lower() not in ("0", "false", "no")

# App scratch filesystem: each deployment gets its own private, writable /data
# preopen (wasi:filesystem via `wasmtime --dir`), so off-the-shelf code that
# expects to read/write files ports to wasm with no changes -- the point is to
# make apps EASIER TO CONVERT, not to store anything. It is a RAM-backed scratch
# dir on the enclave's (already encrypted) ramdisk: strictly ephemeral, torn down
# with the deployment, no persistence, no host paths exposed. Isolation is the
# wasi capability model -- an app sees ONLY its own dir; a path escaping the
# preopen (`/data/../../etc/...`) is refused by the runtime, and no preopen means
# no visibility at all. A per-app size cap bounds RAM use, since `-W
# max-memory-size` covers linear memory only, NOT files; we can't mount a sized
# tmpfs (the enclave blocks mounts), so the audit sweep polices it -- same shape
# as the port bind audit. Global kill-switch WASM_FS=0; per-app opt-out is
# `storage_mb: 0` in the catalog.
FS_ENABLED     = os.environ.get("WASM_FS", "1").lower() not in ("0", "false", "no")
FS_DIR         = pathlib.Path(os.environ.get("WASM_FS_DIR", "/tmp/enclave-wasm-fs"))  # base for per-deployment scratch dirs (ramdisk)
FS_GUEST_PATH  = os.environ.get("WASM_FS_GUEST", "/data")                          # where it shows up inside the guest
DEF_STORAGE_MB = int(os.environ.get("WASM_APP_STORAGE_MB", "256"))                 # per-app /data ceiling; catalog can override
# Storage is measure-and-kill on the 10s audit poll (a sized tmpfs isn't
# available — the enclave blocks mounts), so between sweeps a tenant can write
# past its /data + /enc caps and, since those scratch dirs live in the CVM's
# RAM (encrypted ramdisk), OOM the whole CVM. The real fix is a cgroup
# memory.max on the tenant group + a sized mount (orchestration outside this
# file). What we CAN add here is admission-time accounting: charge each app's
# guest linear memory + /data cap + encrypted-volume caps against the node's
# RAM and refuse a deployment whose SUM would oversubscribe it. It's OPT-IN and
# OFF by default because it tightens admission and could 429 a deployment that
# fits under the pure cpuShare dial today — enable per node once sized.
ACCOUNT_STORAGE_RAM = os.environ.get("WASM_ACCOUNT_STORAGE_RAM", "0").lower() in ("1", "true", "on")
RAM_ACCT_HEADROOM   = float(os.environ.get("WASM_RAM_HEADROOM", "0.9"))   # fraction of node RAM tenants may reserve

# ---- dead-man lease on every tenant (see POST /vms/lease) -------------------
# Teardown used to be purely INSTRUCTED: the supervisor decided a tenant should
# die and sent a DELETE. Every link in that chain fails open - if the DELETE is
# lost, the supervisor forgets, or it crashes between stopping and releasing,
# this process keeps the tenant alive forever holding its whole slice, and
# neither side can see the other's view. That stranded 17.6 GB of resident ggml
# weights on kryptos twice (2026-08-03), and an instructed reclaimer could not
# fix it because it reclaims over the very channel that failed.
#   So the default on silence is RELEASE, not HOLD. Every tenant carries a
# lease the supervisor must keep renewing by naming it in a heartbeat; anything
# whose lease lapses this process reaps ITSELF, with nobody's permission.
TENANT_LEASE_TTL = float(os.environ.get("WASM_TENANT_LEASE_TTL_SEC", "300"))
# ...but "the supervisor says this tenant is not mine" and "the supervisor is
# not talking" are DIFFERENT facts, and only the first is evidence about a
# tenant. The data path does not run through this API - a manager-API outage
# leaves tenants serving perfectly - so reaping on silence would turn a blip in
# a control channel nobody was using into a fleet-wide outage. Enforcement is
# therefore gated on having heard from the supervisor RECENTLY: no heartbeat at
# all, no reaping, however long the silence lasts.
TENANT_LEASE_SILENCE = float(os.environ.get("WASM_TENANT_LEASE_SILENCE_SEC", "180"))
_lease_last_beat = None    # None = never heard one: the lease is INERT until the
                           # first heartbeat, so an older supervisor that knows
                           # nothing about leases can never trigger a mass reap
# --- VRAM-reservation ledger (the GPU sibling of WASM_ACCOUNT_STORAGE_RAM) --- #
# CUDA_MPS_PINNED_DEVICE_MEM_LIMIT is a CAP, not a reservation: it bounds what
# a tenant may pin but reserves nothing, so physical device memory is only
# yours while you hold it. The supervisor's card allocator PLANS slices; this
# process SPAWNS them - and the two views can diverge (2026-07-25, cc1f4f3f: a
# leaked duplicate tenant pinned ~30 GiB the planner had handed out once, and
# the replacement failed context allocation against memory nobody accounted).
# So the process owner keeps its own physical ledger. Calibrated to admit
# exactly what the planner plans when the views agree - same per-slice
# CTX_OVERHEAD_GB term, no headroom - so it binds ONLY on divergence.
ACCOUNT_VRAM         = os.environ.get("WASM_ACCOUNT_VRAM", "1") == "1"
VRAM_CTX_OVERHEAD_GB = float(os.environ.get("CTX_OVERHEAD_GB", "0.5"))    # same env the supervisor prices per slice
VRAM_RESERVE_GB      = float(os.environ.get("WASM_VRAM_RESERVE_GB", "0")) # node-global device users the planner never sees (SD preloads)
# The ledger above is ARITHMETIC: it sums the shares this process handed out.
# Memory held by anything it did not spawn - a tenant generation orphaned by an
# in-place container update, a wedged MPS server retaining dead clients'
# allocations - is invisible to it, and on 2026-08-18 ~104 GiB of exactly that
# let a 51 GB tenant pass every fit check and die at its first generation
# (kryptos H200: ledger said 27 GB unreserved, the card had 35 GB total free).
# So the ledger now carries the DEVICE's own count beside its arithmetic
# (nvidia-smi memory.free), logs when the two views drift apart, and the
# admission gate refuses a launch the card physically cannot hold even when
# the arithmetic says it fits. Probe failure fails OPEN to ledger-only
# admission - a missing nvidia-smi must not refuse a whole node's launches.
VRAM_DEV_GATE        = os.environ.get("WASM_VRAM_DEV_GATE", "1") == "1"   # 0 = report only, never refuse on the device's word
VRAM_DEV_TTL_S       = float(os.environ.get("WASM_VRAM_DEV_TTL_S", "5"))
VRAM_DIVERGE_WARN_GB = float(os.environ.get("WASM_VRAM_DIVERGE_WARN_GB", "8"))
# MPS bounce orders: the recovery lever for the residue the gate above only
# DETECTS. The mps-control container consumes a request file off the shared
# pipe-dir volume (it has no HTTP surface) and bounces the whole MPS stack -
# every live GPU tenant's CUDA context dies and the supervisor respawns them,
# so ordering one is never free. Boot is the one moment it is nearly free:
# this process just started, its tenant table is empty, so device memory
# beyond VRAM_RESERVE_GB belongs to a generation that no longer exists -
# exactly what an in-place container update strands.
MPS_BOOT_BOUNCE      = os.environ.get("WASM_MPS_BOOT_BOUNCE", "1") == "1"
MPS_BOOT_BOUNCE_GB   = float(os.environ.get("WASM_MPS_BOOT_BOUNCE_GB", "8"))
if FS_ENABLED:
    FS_DIR.mkdir(parents=True, exist_ok=True)

# Encrypted volumes (rclone crypt over S3): user-held-key confidential storage,
# the simplified successor to enclave-vault (no wallets, no on-chain ACL). The
# owner encrypts a directory CLIENT-SIDE with `rclone crypt` and pushes the
# ciphertext to any S3-compatible bucket; the version's config (encVolumes)
# names the endpoint/bucket - never a key. The tenant starts immediately with
# an EMPTY /enc/<name> preopen (same ramdisk mechanism as /data) plus a
# per-deployment bearer token (ENCLAVE_ENC_TOKEN + ENCLAVE_ENC_API), and the
# app itself delivers the crypt password over the in-enclave-terminated TLS:
# POST /encvol/<vid>/unlock runs rclone (env-configured, secrets never in
# argv) to pull + decrypt into the preopen. Plaintext exists only on the
# CVM's encrypted ramdisk; the host/bucket only ever saw ciphertext. /sync
# pushes local edits back (creds held in RAM from unlock; readOnly opts out),
# /lock wipes. Caps: --max-transfer on the pull, then the storage audit
# polices post-unlock growth per volume (same kill policy as /data).
#
# An entry may also opt OUT of the crypt layer ("encrypted": false): a plain
# S3 mount for data that isn't secret (public datasets, published models,
# app-served assets). Same preopen, token plane, unlock/sync/lock lifecycle
# and caps - the unlock just takes S3 credentials only (none at all for a
# public-read bucket) and syncs the bucket bytes verbatim. The bucket host
# sees everything, so this is a convenience mount, not confidential storage.
ENC_ENABLED    = os.environ.get("WASM_ENC", "1").lower() not in ("0", "false", "no")
ENC_DIR        = pathlib.Path(os.environ.get("WASM_ENC_DIR", "/tmp/enclave-wasm-enc"))  # per-deployment staging (ramdisk)
ENC_GUEST_ROOT = os.environ.get("WASM_ENC_GUEST", "/enc")                # /enc/<name> inside the guest
ENC_DEF_MB     = int(os.environ.get("WASM_ENC_DEF_MB", "1024"))          # per-volume plaintext ceiling default
ENC_MAX_MB     = int(os.environ.get("WASM_ENC_MAX_MB", "4096"))          # what a config maxMb may ask up to
ENC_MAX_VOLS   = int(os.environ.get("WASM_ENC_MAX_VOLS", "8"))
ENC_SYNC_SECS  = float(os.environ.get("WASM_ENC_SYNC_TIMEOUT", "1800"))  # one rclone pull/push budget
RCLONE_BIN     = os.environ.get("RCLONE_BIN", "rclone")
# test hook: lets an endpoint of "local:/abs/path" use rclone's local backend
# instead of S3 so the whole pipeline runs without a bucket. NEVER set in the
# enclave configs - a local source would read the manager's own filesystem.
ENC_ALLOW_LOCAL = os.environ.get("WASM_ENC_LOCAL_SRC", "").lower() in ("1", "true", "on")
# SSRF guard for the encVolumes S3 endpoint. The endpoint host rides the
# (public, approved) version config and is dialled by the rclone child, so an
# endpoint like http://127.0.0.1:8090 (worker), http://169.254.169.254 (cloud
# metadata) or any RFC1918 host would let a deployment pivot rclone into the
# CVM's own loopback/private services. We HARD-REJECT non-public endpoint hosts
# by default (see _is_blocked_host). A genuinely in-CVM/private S3 endpoint is
# not how this feature is meant to be used (ciphertext lives on an EXTERNAL
# bucket), but an operator who deliberately runs a private/in-cluster bucket can
# opt back in with WASM_ENC_ALLOW_PRIVATE_ENDPOINT=1 (warns, does not block).
ENC_ALLOW_PRIVATE_EP = os.environ.get("WASM_ENC_ALLOW_PRIVATE_ENDPOINT", "").lower() in ("1", "true", "on")
if ENC_ENABLED:
    ENC_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_apps = {}    # id -> record


# ---- helpers --------------------------------------------------------------- #
def _load_catalog() -> dict:
    """Map of app-id -> {file, name, description, vram_mb?, gpu_gflops?,
    mem_mb?, cpu_gflops?, storage_mb?}. Baked-in + attested. The four resource
    fields are the app's EXACT minimums (memory in MB, compute in GFLOPS =
    1/1000 TFLOPS; any GPU axis > 0 marks a GPU app), mirroring EnclaveAppCatalog;
    shares are calculated from them."""
    try:
        data = json.loads(CATALOG_PATH.read_text())
        return {a["id"]: a for a in data.get("apps", [])}
    except Exception:
        return {}


def _check_component(data: bytes):
    """Reject anything that isn't runnable before we try to run it (same
    preamble check as the upload gateway; gives a clear error vs a wasmtime
    crash). Layer field: 0 = core module, 1 = component. Components pass and
    core modules do not, with no exceptions: the wasm64 carve-out is gone
    now that a >4 GiB guest is a memory64 COMPONENT (built by
    wasm/Dockerfile.wasm64p2-build), which keeps the whole socket/HTTP
    surface a preview1 module never had."""
    if len(data) < 8 or data[0:4] != b"\x00asm":
        raise ValueError("fetched bytes are not a WebAssembly file")
    layer = data[6] | (data[7] << 8)
    if layer == 0:
        raise ValueError("fetched a core wasm module, not a wasi:http component")
    if layer != 1:
        raise ValueError(f"unrecognized wasm layer {layer} (expected a component)")


def _module_mem64(data: bytes) -> bool:
    """Does this CORE module (layer 0) declare a 64-bit linear memory? Reads
    the memory section (id 5): the first memory's limits flags carry the
    memory64 bit (0x04). A real section walk, not a byte scan — unlike the
    thread markers there is no import-name string to key on, and the flag
    byte sits at a grammar-determined offset. Components return False (their
    memories live in nested core modules; a component that is mem64 inside
    still speaks the component ABI outside, which is not this feature).
    Anything unparseable returns False and flows to the existing refusals."""
    if len(data) < 8 or data[0:4] != b"\x00asm" or (data[6] | (data[7] << 8)) != 0:
        return False
    i = 8
    while i < len(data):
        sid = data[i]
        try:
            size, i = _uleb(data, i + 1)
        except (IndexError, ValueError):
            return False
        if sid == 5:                       # memory section
            try:
                count, j = _uleb(data, i)
                if count == 0:
                    return False
                flags, j = _uleb(data, j)
            except (IndexError, ValueError):
                return False
            return bool(flags & 0x04)      # limits flag bit 2: 64-bit index
        i += size
    return False


# Which wasi world contract does a component target? Read from the BINARY —
# the top-level component import (10) / export (11) sections carry the world's
# interface names as plain length-prefixed strings — never from catalog
# metadata: a version config can say anything, the bytes are what runs. The
# EXPORT decides, in the same order `wasmtime serve` tries instantiation:
# a component exporting both worlds is served as p3 (ServicePre::new first,
# p2 ProxyPre fallback), so classification must agree or the claim gate and
# the runtime would disagree about the same bytes.
#   wasi:http/handler@0.3.*           -> "0.3"  (wasi:http/service world)
#   wasi:http/incoming-handler@0.2.*  -> "0.2"  (wasi:http/proxy world)
#   wasi:cli/run@0.3.* / @0.2.*       -> run-mode app, same versions
# Mixed IMPORTS are normal and say nothing: rustc's wasm32-wasip3 std still
# imports WASIp2 APIs ("that's ok since it's all component-model-level
# imports anyway" — rustc target spec), and serve links both API sets into
# one linker. A component exporting neither world classifies as None: the
# caller decides whether that is an error (it is not here — the audit sweep
# and wasmtime itself refuse what cannot serve).
_WASI_NAME_RE = re.compile(rb"[A-Za-z0-9:/@.+\-]+")


def _uleb(data: bytes, i: int):
    r = s = 0
    while True:
        b = data[i]; i += 1
        r |= (b & 0x7F) << s
        if not (b & 0x80):
            return r, i
        s += 7
        if s > 35:
            raise ValueError("uleb128 too long")


def _component_wasi_names(path: pathlib.Path):
    """(imports, exports): wasi:* interface names named by the component's own
    top-level import/export sections. Seeks past every other section, so a
    big component costs a handful of reads, not a full load. Extraction scans
    for length-prefixed `wasi:`-strings inside those two payloads rather than
    decoding the full entry grammar — entries carry externdescs whose encoding
    has churned across component-model revisions, while name strings have not;
    a scan inside the RIGHT sections cannot see linear memory or data (those
    live in nested core-module sections we never open)."""
    imports, exports = set(), set()
    with open(path, "rb") as f:
        pre = f.read(8)
        if len(pre) < 8 or pre[0:4] != b"\x00asm" or (pre[6] | (pre[7] << 8)) != 1:
            return imports, exports
        while True:
            head = f.read(6)                       # section id + worst-case u32 leb
            if len(head) < 2:
                break
            sid = head[0]
            try:
                size, j = _uleb(head, 1)
            except (IndexError, ValueError):
                break
            f.seek(j - len(head), 1)               # rewind the over-read (SEEK_CUR)
            if sid in (10, 11):                    # component import / export section
                payload = f.read(size)
                names = set()
                for m in re.finditer(rb"wasi:", payload):
                    p = m.start()
                    for back in range(1, 6):       # LEB length sits just before the string
                        if p - back < 0:
                            break
                        try:
                            ln, q = _uleb(payload, p - back)
                        except (IndexError, ValueError):
                            continue
                        if q == p and p + ln <= len(payload):
                            s = payload[p:p + ln]
                            if _WASI_NAME_RE.fullmatch(s):
                                names.add(s.decode())
                            break
                (imports if sid == 10 else exports).update(names)
            else:
                f.seek(size, 1)
    return imports, exports


def _component_contract(path: pathlib.Path) -> dict:
    """{"wasi": "0.2"|"0.3"|None, "world": <the export that decided>|None}"""
    _imports, exports = _component_wasi_names(path)
    for prefix, ver in (("wasi:http/handler@0.3.", "0.3"),
                       ("wasi:http/incoming-handler@0.2.", "0.2"),
                       ("wasi:cli/run@0.3.", "0.3"),
                       ("wasi:cli/run@0.2.", "0.2")):
        hit = sorted(e for e in exports if e.startswith(prefix))
        if hit:
            return {"wasi": ver, "world": hit[0]}
    return {"wasi": None, "world": None}


def _resolve_cid(cid: str) -> pathlib.Path:
    """Fetch `cid` from IPFS, verify the bytes hash to it, cache under APPS_DIR, run."""
    # REJECT a non-CID, never sanitize one into a filename. The cache is
    # content-addressed, so its key has to BE the content address: stripping
    # characters made the key a lossy transform, and two different catalog CIDs
    # that differ only outside [A-Za-z0-9] collapsed onto one file. The second
    # one then took a cache HIT and ran the FIRST one's bytes, skipping the
    # hash verification entirely — the CID would no longer name what runs,
    # which is the single assumption the deploy gate rests on. The catalog
    # bounds a version's cid by LENGTH only (MAX_CID), so any publisher could
    # declare "bafy.REAL" next to a real "bafyREAL". Real CIDs are alphanumeric
    # either way (base58btc for v0, base32 for v1), so nothing legitimate is
    # turned away by requiring it.
    if not re.fullmatch(r"[A-Za-z0-9]{10,100}", cid or ""):
        raise ValueError(f"bad ipfs cid '{cid}' (a CID is 10-100 alphanumeric characters)")
    p = (APPS_DIR / f"ipfs-{cid}.wasm").resolve()
    if p.is_file():
        return p                                   # content-addressed cache hit
    if ipfs_fetch is None:
        raise ValueError("run-by-CID not available in this build (ipfs_fetch missing)")
    try:
        data = ipfs_fetch.fetch_verified(cid, IPFS_GATEWAY, WASM_MAX_BYTES, IPFS_TIMEOUT)
    except ValueError:
        raise                                      # verification / size errors already have clear messages
    except Exception as e:                          # network / gateway errors -> ValueError so launch() reports it
        raise ValueError(f"ipfs fetch failed for {cid}: {e}")
    _check_component(data)
    APPS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)                                  # atomic publish into the cache
    return p


# Per-deployment config (ENCLAVE_CONFIG): the approved catalog version's config
# JSON. Two shapes reach us, and the chain record is the source of both:
#
#   inline  - the version's `config` field, sent verbatim on /vms (<= 4096 bytes,
#             the catalog's on-chain ceiling).
#   by CID  - rev-7 `configCid`, sent as `configCid` on /vms and fetched HERE
#             through the same hash-verifying path as the wasm bytes.
#
# The CID form is NOT the retired deployer-pinned configCid. That one let the
# DEPLOYER name the bytes at deploy time, so what ran was never what the owner
# ruled on. This CID is a field of the version RECORD — publisher-set,
# immutable, approval-covered — and fetch_verified re-hashes every block back to
# it, so the config the guest sees is the config the owner approved, exactly as
# with the wasm. It exists because the inline field physically cannot grow: a
# 1MB string is 32,768 SSTOREs (~655M gas) against a 400M block limit.
#
# Must parse as JSON and fit the ceiling, or the launch fails loudly rather than
# silently serving app defaults with the wrong shape.
CONFIG_MAX_BYTES = int(os.environ.get("ENCLAVE_CONFIG_MAX_BYTES", str(1024 * 1024)))

# How much config may ride the guest's ENCLAVE_CONFIG env var. This is a KERNEL
# limit, not a policy one: the value goes to wasmtime as one argv string, and
# execve refuses any single argument over MAX_ARG_STRLEN = 32 pages = 131072
# bytes with E2BIG. Anything larger is delivered by FILE only (see
# CONFIG_GUEST_PATH). Held well under the wall so the "ENCLAVE_CONFIG=" prefix,
# the secrets substitution (which can grow the text) and any page-size
# difference cannot creep over it.
#
# Before rev 7 this could not be reached — the chain capped configs at 4096 —
# so CONFIG_MAX_BYTES sat at 256KB, twice the kernel wall, and a config between
# 128KB and 256KB would have passed validation and then died at spawn with a
# bare E2BIG. Raising the ceiling is what makes that reachable, hence the split.
CONFIG_ENV_MAX_BYTES = int(os.environ.get("ENCLAVE_CONFIG_ENV_MAX_BYTES", str(64 * 1024)))

# Where a config too big for the env var shows up inside the guest. The file is
# written for EVERY config, whatever its size, so an app has one mechanism that
# always works; ENCLAVE_CONFIG stays populated whenever it fits, so every app
# written before rev 7 keeps working untouched.
CONFIG_GUEST_PATH = os.environ.get("WASM_CONFIG_GUEST", "/config")
CONFIG_FILE_NAME = "config.json"
# How far past the staged config /config may grow before the audit calls it
# abuse. Zero would be right in principle (the dir is 0500 and its contents are
# fixed at launch), but a filesystem can round a file's block usage, so leave
# one block of slack rather than killing a healthy tenant over du arithmetic.
CONFIG_DIR_SLACK_BYTES = 4096


def _rm_tree_rw(path):
    """rmtree a dir we deliberately made read-only (/config is 0500 so a tenant
    cannot write there). rmtree needs WRITE on the directory to unlink its
    children, so restore the bit first — without this the cleanup silently
    fails under ignore_errors and a config holding substituted secrets outlives
    the deployment that was entitled to them."""
    try:
        os.chmod(path, 0o700)
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try: os.chmod(os.path.join(root, d), 0o700)
                except OSError: pass
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=True)


def _validate_config(text: str) -> str:
    if len(text.encode("utf-8")) > CONFIG_MAX_BYTES:
        raise ValueError(f"config exceeds {CONFIG_MAX_BYTES} bytes")
    try:
        json.loads(text)                            # must parse; the app merges it over its defaults
    except Exception as e:
        raise ValueError(f"config is not valid JSON: {e}")
    return text


def _resolve_config_cid(cid: str) -> str:
    """Fetch a rev-7 version's config JSON from IPFS and verify it hashes to `cid`.

    Same trust rule as _resolve_cid for the wasm: the gateway is untrusted, so
    the bytes are only accepted because they reproduce the CID the approved
    version record names. Bounded by CONFIG_MAX_BYTES *during* reconstruction,
    so a hostile gateway cannot make us materialize a huge DAG before the check.
    """
    if not re.fullmatch(r"[A-Za-z0-9]{10,100}", cid or ""):
        raise ValueError(f"bad config cid '{cid}' (a CID is 10-100 alphanumeric characters)")
    if ipfs_fetch is None:
        raise ValueError("config-by-CID not available in this build (ipfs_fetch missing)")
    try:
        data = ipfs_fetch.fetch_verified(cid, IPFS_GATEWAY, CONFIG_MAX_BYTES, IPFS_TIMEOUT)
    except ValueError:
        raise                                       # verification / size errors already read clearly
    except Exception as e:
        raise ValueError(f"ipfs fetch failed for config {cid}: {e}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"config at {cid} is not UTF-8: {e}")
    return _validate_config(text)


# Per-deployment secrets (relay-staged owner env vars, fetched by the
# supervisor at provision; relay/secrets.js owns the trust model). MIRRORS the
# relay's wire contract — the relay is the authority, this re-check just makes
# a malformed hand-off fail the launch loudly instead of baking a bad guest
# env. Values are guest-only --env vars, same handling class as ENCLAVE_CONFIG:
# never the process env, never a log line. ENCLAVE_* names are refused so a
# secret can't shadow a platform channel (ENCLAVE_CONFIG, ENCLAVE_ENC_TOKEN, …).
SECRETS_MAX_KEYS, SECRETS_MAX_VALUE, SECRETS_MAX_TOTAL = 64, 4096, 16384
_SECRET_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


# Every hostname this deployment legitimately answers on, comma-separated: its
# own <label>.app.enclave.host plus any custom domain its owner attached and
# proved (relay/domains.js). Handed to the guest as ENCLAVE_HOSTS so an app can
# build the things only IT can build — a Host allowlist, CSRF trusted origins,
# a websocket Origin check — without hardcoding a hostname it cannot know at
# publish time. Absolute URLs and redirects should still come from the REQUEST
# Host (which the in-enclave TLS bridge passes through untouched); this list is
# for deciding which Hosts are legitimate, not for picking one.
#
# Spawn-time, like secrets: a newly attached domain reaches the guest at the
# next start. Routing, TLS and the cross-tenant Host refusal are all live
# immediately, so the gap is an app-level allowlist and nothing structural.
HOSTS_MAX_BYTES = 4096
_HOSTNAME_RE = re.compile(r"^[a-z0-9.-]{1,253}$")


def _validate_hosts(text) -> str:
    """Comma-separated hostnames, normalized and filtered. Never fatal: a bad
    entry is dropped rather than failing the launch, because this is platform
    metadata about a running app, not the app's own configuration."""
    out = []
    for h in str(text or "").lower().split(","):
        h = h.strip().rstrip(".")
        if h and _HOSTNAME_RE.match(h) and h not in out:
            out.append(h)
    joined = ",".join(out)
    return joined[:HOSTS_MAX_BYTES] if len(joined) <= HOSTS_MAX_BYTES else ""


# Config placeholders: $NAME / ${NAME} inside any STRING value of the config
# JSON resolves to the deployment's secret of that name at launch, so a PUBLIC
# on-chain config can reference private values ("credentials": {"secretAccessKey":
# "$S3_SECRET_ACCESS_KEY"}) without any app changes. Substitution walks the
# PARSED JSON and replaces within string values only - a secret holding quotes
# or backslashes is re-serialized safely, never spliced into raw JSON text.
# Only names actually stored as secrets substitute; anything else keeps its
# literal $ (configs may legitimately contain dollar signs), and $$ escapes a
# literal $ before a real secret name. The supervisor passes config RAW, so
# the owner-visible record keeps the placeholder - values exist only in the
# guest env, same exposure class as the secrets themselves.
_SECRET_REF_RE = re.compile(r"\$(\$)|\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _subst_secrets(text: str, secrets: dict) -> str:
    if not secrets or "$" not in text:
        return text

    def rep(m):
        if m.group(1):
            return "$"                              # $$ -> literal $
        name = m.group(2) or m.group(3)
        v = secrets.get(name)
        return v if v is not None else m.group(0)   # unknown name: keep literal

    def walk(x):
        if isinstance(x, str):
            return _SECRET_REF_RE.sub(rep, x)
        if isinstance(x, list):
            return [walk(i) for i in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    return json.dumps(walk(json.loads(text)), separators=(",", ":"))


def _validate_secrets(d) -> dict:
    if not isinstance(d, dict):
        raise ValueError("secrets must be an object of NAME: value")
    if len(d) > SECRETS_MAX_KEYS:
        raise ValueError(f"too many secrets (max {SECRETS_MAX_KEYS})")
    total = 0
    for k, v in d.items():
        if not isinstance(k, str) or not _SECRET_KEY_RE.match(k):
            raise ValueError(f"secret name {k!r} is not an env-var name")
        if k.upper().startswith("ENCLAVE_"):
            raise ValueError(f"secret name {k!r}: the ENCLAVE_ prefix is reserved")
        if not isinstance(v, str):
            raise ValueError(f"secret {k!r} must be a string value")
        if "\0" in v or "\n" in v or "\r" in v:
            raise ValueError(f"secret {k!r} contains a NUL or newline")
        vb = len(v.encode("utf-8"))
        if vb > SECRETS_MAX_VALUE:
            raise ValueError(f"secret {k!r} is {vb} bytes (max {SECRETS_MAX_VALUE})")
        total += len(k.encode("utf-8")) + vb
    if total > SECRETS_MAX_TOTAL:
        raise ValueError(f"secrets total {total} bytes (max {SECRETS_MAX_TOTAL})")
    return dict(d)


# --- SSRF host classifier (mirror of net-guard.mjs; DO NOT import it — it's JS) #
# Kept in sync BY HAND with net-guard.mjs's blockedV4/blockedV6. Policy: allow
# only globally-routable unicast; refuse loopback, private (RFC1918/CGNAT),
# link-local, unique-local, documentation/benchmark, multicast and reserved —
# the ranges an app could use to pivot into the CVM's own loopback/private-
# network services. v4-mapped/-compat and NAT64 IPv6 are unwrapped so
# `::ffff:127.0.0.1` can't sneak loopback past the v6 path.
_BLOCKED_V4 = [ipaddress.ip_network(c) for c in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
    "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24", "192.88.99.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4")]
_BLOCKED_V6 = [ipaddress.ip_network(c) for c in (
    "100::/64", "2001:db8::/32", "fc00::/7", "fe80::/10", "ff00::/8")]


def _ip_blocked(ip_str: str) -> bool:
    """True if the IP literal `ip_str` is non-global (raises ValueError if it
    isn't an IP at all)."""
    ip = ipaddress.ip_address(ip_str)
    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in net for net in _BLOCKED_V4)
    n = int(ip)
    if n < (1 << 96):                                    # ::, ::1, v4-compat + v4-mapped (all non-global)
        return True
    if (0x64ff9b << 64) <= n < (0x64ff9b << 64) + (1 << 32):   # 64:ff9b::/96 NAT64 — judge embedded v4
        return _ip_blocked(str(ipaddress.IPv4Address(n & 0xffffffff)))
    return any(ip in net for net in _BLOCKED_V6)


def _is_blocked_host(host: str) -> bool:
    """True if `host` (an IP literal or a domain) is a destination we must not
    let the encVolumes rclone child dial. Literal private/loopback/link-local
    IPs and localhost names are blocked outright. Unlike net-guard.mjs (which
    defers a domain to a post-DNS re-check on the relay), THIS path has no
    downstream re-check, so we also resolve a domain here and block it if ANY
    resolved address is non-global. A domain that fails to resolve is allowed
    through — rclone will fail on its own, and a transient DNS miss must not
    fail an otherwise-legit deploy (residual DNS-rebinding risk noted)."""
    if not host:
        return True
    h = host.strip().lower().rstrip(".")
    if h == "localhost" or h.endswith(".localhost"):
        return True
    lit = h[1:-1] if h.startswith("[") and h.endswith("]") else h   # unwrap [v6]
    try:
        return _ip_blocked(lit)
    except ValueError:
        pass                                             # not an IP literal -> a domain name
    try:
        infos = socket.getaddrinfo(h, None)
    except OSError:
        return False                                     # unresolvable now: defer to rclone
    for info in infos:
        try:
            if _ip_blocked(info[4][0].split("%", 1)[0]):   # drop any IPv6 zone id
                return True
        except ValueError:
            continue
    return False


# --- encrypted volumes (rclone crypt over S3) -------------------------------- #
_ENC_BUCKET_RE   = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_ENC_FILENAME_ENC = ("standard", "off", "obfuscate")


def _parse_enc_volumes(cfg: dict) -> list:
    """Validate the config's encVolumes into internal specs. Every field here is
    NON-SECRET (it rides the approved, public version config): where the
    ciphertext lives and how it was packed. The crypt password and any S3
    credentials only ever arrive at unlock time, straight into RAM.
    "encrypted": false marks a PLAIN volume - same shape minus the crypt
    fields; the bucket then holds the bytes verbatim."""
    entries = cfg.get("encVolumes")
    if not entries:
        return []
    if not ENC_ENABLED:
        raise ValueError("encrypted volumes are disabled on this node (WASM_ENC=0)")
    if shutil.which(RCLONE_BIN) is None:
        raise ValueError("encrypted volumes unavailable: this build has no rclone")
    if not isinstance(entries, list) or len(entries) > ENC_MAX_VOLS:
        raise ValueError(f"encVolumes must be a list of at most {ENC_MAX_VOLS} entries")
    specs, seen = [], set()
    for e in entries:
        if not isinstance(e, dict):
            raise ValueError("encVolumes entries must be objects")
        name = str(e.get("name") or "").strip()
        if not _VOL_NAME_RE.match(name):
            raise ValueError(f"encVolumes: bad volume name '{name}' (want {_VOL_NAME_RE.pattern})")
        if name in seen:
            raise ValueError(f"encVolumes: duplicate volume '{name}'")
        seen.add(name)
        endpoint = str(e.get("endpoint") or "").strip().rstrip("/")
        if endpoint.startswith("local:"):
            if not ENC_ALLOW_LOCAL:
                raise ValueError(f"encVolumes '{name}': local: endpoints are a test hook (WASM_ENC_LOCAL_SRC), not deployable")
        elif not (endpoint.startswith("https://") or endpoint.startswith("http://")):
            raise ValueError(f"encVolumes '{name}': endpoint must be an http(s) S3 endpoint URL")
        else:
            # SSRF guard: the rclone child dials this endpoint, so a private/
            # loopback/link-local host would pivot into the CVM's own services
            # (worker:8090, supervisor:8080, cloud metadata, RFC1918). Hard-
            # reject unless explicitly opted in (see WASM_ENC_ALLOW_PRIVATE_ENDPOINT).
            _ep_host = urllib.parse.urlparse(endpoint).hostname or ""
            if _is_blocked_host(_ep_host):
                if not ENC_ALLOW_PRIVATE_EP:
                    raise ValueError(f"encVolumes '{name}': endpoint host '{_ep_host}' is a private/"
                                     f"loopback/link-local address (SSRF-blocked). Set "
                                     f"WASM_ENC_ALLOW_PRIVATE_ENDPOINT=1 only if this node's S3 endpoint "
                                     f"is deliberately in-CVM/private.")
                print(f"[enc] WARNING: '{name}' endpoint host '{_ep_host}' is private/loopback "
                      f"(WASM_ENC_ALLOW_PRIVATE_ENDPOINT=1 permits it) — SSRF risk", flush=True)
        bucket = str(e.get("bucket") or "").strip().strip("/")
        if not _ENC_BUCKET_RE.match(bucket):
            raise ValueError(f"encVolumes '{name}': bad bucket name")
        path = str(e.get("path") or "").strip().strip("/")
        if path and any(seg in ("", ".", "..") for seg in path.split("/")):
            raise ValueError(f"encVolumes '{name}': bad path prefix")
        encrypted = bool(e.get("encrypted", True))
        if not encrypted and ("filenameEncryption" in e or "directoryNameEncryption" in e):
            raise ValueError(f"encVolumes '{name}': filenameEncryption/directoryNameEncryption apply to the "
                             f"crypt layer, which \"encrypted\": false turns off - drop them")
        fenc = str(e.get("filenameEncryption") or "standard").strip()
        if fenc not in _ENC_FILENAME_ENC:
            raise ValueError(f"encVolumes '{name}': filenameEncryption must be one of {_ENC_FILENAME_ENC}")
        try:
            max_mb = int(e.get("maxMb") or ENC_DEF_MB)
        except (TypeError, ValueError):
            raise ValueError(f"encVolumes '{name}': maxMb must be an integer")
        if not 1 <= max_mb <= ENC_MAX_MB:
            raise ValueError(f"encVolumes '{name}': maxMb must be 1..{ENC_MAX_MB}")
        # unlock + keyId are UI METADATA, passed through to the app untouched:
        # the manager only ever takes an opaque password, however the app
        # produced it. "wallet" tells the UI to lead with signature-derived
        # keys (see the encrypted-volumes app); keyId is the stable label the
        # wallet signs over, so renaming a volume doesn't silently derive a
        # different key (default: the volume name).
        unlock = str(e.get("unlock") or "password").strip()
        if unlock not in ("password", "wallet"):
            raise ValueError(f"encVolumes '{name}': unlock must be 'password' or 'wallet'")
        key_id = str(e.get("keyId") or name).strip()
        if not _VOL_NAME_RE.match(key_id):
            raise ValueError(f"encVolumes '{name}': bad keyId (want {_VOL_NAME_RE.pattern})")
        specs.append({"name": name, "endpoint": endpoint, "bucket": bucket, "path": path,
                      "unlock": unlock, "keyId": key_id, "encrypted": encrypted,
                      "provider": str(e.get("provider") or "Other").strip() or "Other",
                      "region": str(e.get("region") or "").strip(),
                      "filenameEncryption": fenc,
                      "directoryNameEncryption": bool(e.get("directoryNameEncryption", True)),
                      "maxMb": max_mb, "readOnly": bool(e.get("readOnly", False))})
    return specs


def _rclone_obscure(secret: str) -> str:
    """rclone config wants password fields OBSCURED (its reversible masking).
    Piped via stdin - never argv - and verified to roundtrip byte-exact."""
    r = subprocess.run([RCLONE_BIN, "obscure", "-"], input=secret.encode(),
                       capture_output=True, timeout=30)
    if r.returncode != 0 or not r.stdout.strip():
        raise ValueError(f"rclone obscure failed: {(r.stderr or b'').decode('utf-8', 'replace').strip()[:200]}")
    return r.stdout.decode().strip()


def _enc_src_remote(spec: dict) -> str:
    """The encsrc-remote path (bucket, plus any key prefix) a volume syncs
    against - the crypt remote's REMOTE= target, and, for a plain
    ("encrypted": false) volume, the sync target itself."""
    if spec["endpoint"].startswith("local:"):        # test hook (ENC_ALLOW_LOCAL)
        remote = f"encsrc:{spec['endpoint'][len('local:'):]}/{spec['bucket']}"
    else:
        remote = f"encsrc:{spec['bucket']}"
    if spec["path"]:
        remote += "/" + spec["path"]
    return remote


def _enc_remote(spec: dict) -> str:
    """What rclone actually syncs: the crypt remote, or - plain volume - the
    S3 backend directly."""
    return "encvol:" if spec["encrypted"] else _enc_src_remote(spec)


def _enc_rclone_env(spec: dict, creds: dict) -> dict:
    """The rclone process environment for one volume: an env-defined S3 remote
    (encsrc), and - unless the volume is plain - a crypt remote (encvol)
    layered on it. Everything secret rides the ENVIRONMENT of the child,
    nothing in argv, nothing on disk (RCLONE_CONFIG=/dev/null keeps rclone
    from reading or writing a config)."""
    env = dict(os.environ)
    env["RCLONE_CONFIG"] = "/dev/null"
    if spec["endpoint"].startswith("local:"):        # test hook (ENC_ALLOW_LOCAL)
        env["RCLONE_CONFIG_ENCSRC_TYPE"] = "local"
    else:
        env["RCLONE_CONFIG_ENCSRC_TYPE"] = "s3"
        env["RCLONE_CONFIG_ENCSRC_PROVIDER"] = spec["provider"]
        env["RCLONE_CONFIG_ENCSRC_ENDPOINT"] = spec["endpoint"]
        if spec["region"]:
            env["RCLONE_CONFIG_ENCSRC_REGION"] = spec["region"]
        if creds.get("accessKeyId"):
            env["RCLONE_CONFIG_ENCSRC_ACCESS_KEY_ID"] = str(creds["accessKeyId"])
            env["RCLONE_CONFIG_ENCSRC_SECRET_ACCESS_KEY"] = str(creds.get("secretAccessKey") or "")
            if creds.get("sessionToken"):
                env["RCLONE_CONFIG_ENCSRC_SESSION_TOKEN"] = str(creds["sessionToken"])
        else:
            env["RCLONE_CONFIG_ENCSRC_ENV_AUTH"] = "false"   # anonymous: public-read bucket
    if not spec["encrypted"]:
        return env
    env["RCLONE_CONFIG_ENCVOL_TYPE"] = "crypt"
    env["RCLONE_CONFIG_ENCVOL_REMOTE"] = _enc_src_remote(spec)
    env["RCLONE_CONFIG_ENCVOL_PASSWORD"] = _rclone_obscure(str(creds["password"]))
    if creds.get("salt"):
        env["RCLONE_CONFIG_ENCVOL_PASSWORD2"] = _rclone_obscure(str(creds["salt"]))
    env["RCLONE_CONFIG_ENCVOL_FILENAME_ENCRYPTION"] = spec["filenameEncryption"]
    env["RCLONE_CONFIG_ENCVOL_DIRECTORY_NAME_ENCRYPTION"] = "true" if spec["directoryNameEncryption"] else "false"
    return env


def _enc_rclone_sync(src: str, dst: str, env: dict, max_mb: int = 0, crypt: bool = True) -> tuple:
    """One rclone sync. Returns (ok, error_message). Two failure shapes:
    a nonzero exit (network, auth, content MAC mismatch), and - crucially,
    crypt only - exit 0 with 'Skipping undecryptable' NOTICEs: under encrypted
    file names a WRONG PASSWORD decrypts nothing and rclone happily syncs an
    empty set, so undecryptable names must fail the unlock, not silently
    produce an empty volume."""
    cmd = [RCLONE_BIN, "sync", src, dst, "--transfers", "8", "--checkers", "8",
           "--retries", "2", "--contimeout", "15s"]
    if max_mb:
        cmd += ["--max-transfer", f"{max_mb}M"]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, timeout=ENC_SYNC_SECS,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False, f"rclone sync timed out after {int(ENC_SYNC_SECS)}s"
    err = (r.stderr or b"").decode("utf-8", "replace")
    if crypt and "undecryptable" in err.lower():
        return False, "volume did not decrypt (wrong password/salt, or filenameEncryption doesn't match how it was pushed)"
    if r.returncode != 0:
        tail = err.strip()[-800:] or f"rclone exited {r.returncode}"
        return False, tail
    return True, ""


def _enc_public(rec: dict) -> list:
    """Refresh + return the public per-volume view (rides the /vms record and
    GET /encvol/<vid>). Sizes are refreshed lazily here rather than per-write."""
    enc = rec.get("_enc")
    if not enc:
        return []
    for name, vol in enc["vols"].items():
        if vol["pub"]["status"] in ("unlocked", "pushing"):
            vol["pub"]["bytes"] = _dir_size(vol["dir"])
    return [v["pub"] for v in enc["vols"].values()]


def _enc_wipe_dir(vol: dict):
    """Drop a volume's plaintext but KEEP the directory inode: it is a live
    wasi preopen - the guest holds an fd to it - so we empty it, never rm it."""
    d = pathlib.Path(vol["dir"])
    for child in d.iterdir() if d.exists() else []:
        try:
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        except OSError:
            pass


def _enc_unlock_worker(rec: dict, vol: dict, creds: dict):
    """Background pull: rclone fetches the bucket contents - decrypting through
    the crypt remote, or verbatim for a plain volume - into the volume's
    preopened dir. On ANY failure the dir is wiped - a partial tree that LOOKS
    unlocked is worse than an empty one."""
    spec = vol["spec"]
    try:
        env = _enc_rclone_env(spec, creds)
    except (ValueError, subprocess.TimeoutExpired, OSError) as e:
        with _lock:
            vol["pub"]["status"], vol["pub"]["error"] = "locked", str(e)
        return
    ok, err = _enc_rclone_sync(_enc_remote(spec), vol["dir"], env, spec["maxMb"],
                               crypt=spec["encrypted"])
    with _lock:
        if ok:
            vol["pub"]["status"], vol["pub"]["error"] = "unlocked", None
            vol["pub"]["bytes"] = _dir_size(vol["dir"])
            # keep the rclone env in RAM for /sync push-back; readOnly drops it
            vol["env"] = None if spec["readOnly"] else env
        else:
            _enc_wipe_dir(vol)
            vol["pub"]["status"], vol["pub"]["error"] = "locked", err
    print(f"[enc] {rec['id']}/{spec['name']} unlock {'ok' if ok else 'failed'}", flush=True)


def _enc_push_worker(rec: dict, vol: dict):
    """Background push: sync the (possibly app-edited) local tree back to the
    bucket through the same remote. Local data stays intact either way."""
    ok, err = _enc_rclone_sync(vol["dir"], _enc_remote(vol["spec"]), vol["env"],
                               crypt=vol["spec"]["encrypted"])
    with _lock:
        vol["pub"]["status"] = "unlocked"
        vol["pub"]["error"] = None if ok else err
        if ok:
            vol["pub"]["lastPush"] = time.time()
    print(f"[enc] {rec['id']}/{vol['spec']['name']} push {'ok' if ok else 'failed'}", flush=True)


# --- attached model volumes ------------------------------------------------ #
_VOL_SIZE_CACHE = {}   # path -> (mtime_ns, bytes): du is expensive on a big model dir


def _dir_bytes(path: pathlib.Path) -> int:
    try:
        st = path.stat()
        cached = _VOL_SIZE_CACHE.get(str(path))
        if cached and cached[0] == st.st_mtime_ns:
            return cached[1]
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        _VOL_SIZE_CACHE[str(path)] = (st.st_mtime_ns, total)
        return total
    except OSError:
        return 0


def _vol_gguf_selection() -> dict:
    """name -> gguf filename explicitly selected in MODEL_VOLUMES via the
    optional third field ("name:/path:file.gguf"). This is how a multi-quant
    HF repo volume (e.g. Qwen/Qwen2.5-0.5B-Instruct-GGUF ships NINE *.gguf)
    names the one file that preloads; single-gguf volumes need none."""
    sel = {}
    for pair in _MODEL_VOLUMES_ENV.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        name, _, rest = pair.partition(":")
        _, _, file = rest.partition(":")
        if file.strip():
            sel[name.strip()] = file.strip()
    return sel


_GGUF_SPLIT_RE = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$")


def _split_family(gguf):
    """All parts of the split-GGUF family `gguf` belongs to (llama.cpp's
    "<prefix>-NNNNN-of-MMMMM.gguf" convention, forced on >50GB models by HF's
    per-file cap), sorted 00001 first - or None if the name isn't a split part
    or any sibling is missing. llama.cpp opens part 00001 and derives the
    sibling paths from its file name, so a family only loads complete."""
    m = _GGUF_SPLIT_RE.match(gguf.name)
    if not m:
        return None
    prefix, count = m.group(1), int(m.group(3))
    parts = [gguf.parent / f"{prefix}-{i:05d}-of-{m.group(3)}.gguf"
             for i in range(1, count + 1)]
    return parts if count >= 1 and all(x.is_file() for x in parts) else None


def _onnx_volume(host_path) -> bool:
    """True when the volume carries ONNX graphs the wasmtime toolchain can
    preload (-S nn-graph=onnx::<dir> registers EVERY *.onnx up to 3 levels
    deep as "<volume>/<component>" named graphs - diffusers layouts carry
    several models per volume). Depth-capped for the same layouts the
    toolchain walks: model.onnx / sub/model.onnx / sub/dir/file.onnx."""
    p = pathlib.Path(host_path)
    try:
        for pat in ("*.onnx", "*/*.onnx", "*/*/*.onnx"):
            for f in p.glob(pat):
                if f.is_file():
                    return True
    except OSError:
        pass
    return False


# sdcpp component-file mode (sd2 toolchain, 2025+ DiT families): Z-Image/
# Qwen-Image volumes ship split components the backend resolves through
# these node-global envs (paths relative to the volume dir). The manager
# mirrors the backend's validation so launch args stay honest.
_SD_COMPONENT_ENV_VARS = (
    "ENCLAVE_SD_DIFFUSION_FILE", "ENCLAVE_SD_CLIP_L_FILE",
    "ENCLAVE_SD_CLIP_G_FILE", "ENCLAVE_SD_T5XXL_FILE",
    "ENCLAVE_SD_LLM_FILE", "ENCLAVE_SD_VAE_FILE",
)

# NOT in _SD_COMPONENT_ENV_VARS on purpose: the component vars are
# VALIDATORS (every set one must resolve in a generation volume), while the
# upscale var is a DETECTOR - the file only exists in upscaler volumes, and
# its presence is what marks one (mirrors the sdcpp backend exactly).
_SD_UPSCALE_ENV = "ENCLAVE_SD_UPSCALE_FILE"


def _sd_component_files() -> dict:
    return {var: v for var in _SD_COMPONENT_ENV_VARS
            if (v := os.environ.get(var, "").strip())}


def _sd_layout(name: str, host_path):
    """How an MODEL_VOLUMES_SD volume preloads on this node, mirroring the
    sdcpp backend exactly: ("upscaler", None) when ENCLAVE_SD_UPSCALE_FILE
    resolves inside the volume (an ESRGAN upscale volume - decided FIRST,
    because its single model file would otherwise misread as a
    single-checkpoint generation volume), ("components", None) when
    ENCLAVE_SD_DIFFUSION_FILE selects component mode, ("checkpoint", path)
    for the single-file convention, (None, None) when the backend would
    refuse - the manager then mounts WITHOUT preloading instead of aborting
    the tenant launch. Every SET ENCLAVE_SD_*_FILE component env must
    resolve inside a GENERATION volume (the backend validates all of them,
    and they are node-global - which is why a component-layout volume and a
    single-checkpoint volume cannot both preload on one node yet; upscaler
    volumes are exempt, the component rules don't apply to them)."""
    p = pathlib.Path(host_path)
    upscale_rel = os.environ.get(_SD_UPSCALE_ENV, "").strip()
    if upscale_rel and (p / upscale_rel).is_file():
        return ("upscaler", None)
    comps = _sd_component_files()
    if not all((p / rel).is_file() for rel in comps.values()):
        return (None, None)
    if "ENCLAVE_SD_DIFFUSION_FILE" in comps:
        return ("components", None)
    ckpt = _sd_checkpoint_path(name, p)
    return ("checkpoint", ckpt) if ckpt else (None, None)


def _sd_checkpoint_path(name: str, host_path):
    """The image checkpoint an MODEL_VOLUMES_SD volume preloads through the
    sdcpp backend: the MODEL_VOLUMES-selected file when given, else
    model.safetensors / model.gguf, else the single top-level
    *.safetensors/*.gguf/*.ckpt in the dir. None = nothing unambiguous (the
    sdcpp backend would refuse the same way; failing here keeps the launch
    args honest)."""
    p = pathlib.Path(host_path)
    sel = _vol_gguf_selection().get(name)
    if sel:
        f = p / sel
        return f if f.is_file() else None
    for preferred in ("model.safetensors", "model.gguf"):
        f = p / preferred
        if f.is_file():
            return f
    ckpts = [x for x in p.glob("*.safetensors") if x.is_file()]
    ckpts += [x for x in p.glob("*.gguf") if x.is_file()]
    ckpts += [x for x in p.glob("*.ckpt") if x.is_file()]
    return ckpts[0] if len(ckpts) == 1 else None


def _gguf_path(name: str, host_path):
    """The concrete GGUF a volume preloads: the MODEL_VOLUMES-selected file
    when given, else model.gguf, else the single *.gguf, else part 00001 of
    the single complete split family covering every *.gguf in the dir. None =
    not a (preloadable) gguf volume - including multi-quant repos with no
    selection, where any pick would be a guess. A selection naming ANY part of
    a split family selects the family (normalized to part 00001)."""
    p = pathlib.Path(host_path)
    sel = _vol_gguf_selection().get(name)
    if sel:
        f = p / sel
        if not f.is_file():
            return None
        fam = _split_family(f)
        return fam[0] if fam else f
    preferred = p / "model.gguf"
    if preferred.is_file():
        return preferred
    ggufs = [x for x in p.glob("*.gguf") if x.is_file()]
    if len(ggufs) == 1:
        return ggufs[0]
    if len(ggufs) > 1:
        fam = _split_family(ggufs[0])
        if fam and {x.name for x in fam} == {x.name for x in ggufs}:
            return fam[0]
    return None


def _model_volumes() -> dict:
    """Discover attached model volumes. Two sources, env wins (friendly names):
      1. scan MODEL_VOLUME_ROOT for `mpk-*` mounts (Tinfoil Modelwrap); the
         mount's dir name IS the volume name (e.g. mpk-0900ca6b...).
      2. MODEL_VOLUMES="name:/path[:file.gguf],name2:/path2" - explicit
         name->path, for friendly aliases of the mpk mounts and for local dev;
         the optional third field picks the gguf out of a multi-quant repo.
    Returns {name: {"name", "path", "bytes", "onnx": bool, "gguf": bool,
    "sd": bool, "files": [top-level]}}.
    Only existing NON-EMPTY directories with a servable name are returned: an
    empty dir is a mount point whose dm-verity image hasn't mounted yet (the
    Modelwrap fetch at enclave boot), not an attached volume. Advertising or
    launching against one bakes a no-preload tenant that stays broken for its
    whole life - load_by_name() can never find a graph the boot preload never
    registered (seen live 2026-07-18: qwen3.5-9b NotFound on a healthy mount)."""
    out = {}
    def add(name, path):
        name = str(name).strip()
        p = pathlib.Path(path)
        if not _VOL_NAME_RE.match(name) or not p.is_dir():
            return
        try:
            top = sorted(x.name for x in p.iterdir())[:32]
        except OSError:
            top = []
        if not top:
            return   # bare mount point: the volume image isn't mounted (yet)
        onnx = _onnx_volume(p)
        # a GGUF volume doubles as a host-preloaded wasi-nn graph (the ggml
        # backend) when one unambiguous file exists or MODEL_VOLUMES picks it;
        # MODEL_VOLUMES_SD volumes preload through sdcpp instead
        sd = name in _SD_VOLUMES and _sd_layout(name, p)[0] is not None
        gguf = not sd and _gguf_path(name, p) is not None
        out[name] = {"name": name, "path": str(p), "bytes": _dir_bytes(p),
                     "onnx": onnx, "gguf": gguf, "sd": sd, "files": top}
    if MODEL_VOLUME_ROOT.is_dir():
        try:
            for child in MODEL_VOLUME_ROOT.iterdir():
                if child.is_dir() and child.name.startswith("mpk-"):
                    add(child.name, child)
        except OSError:
            pass
    for pair in _MODEL_VOLUMES_ENV.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        name, _, rest = pair.partition(":")
        path, _, _ = rest.partition(":")
        add(name, path)
    return out


# guest mount point for an attached volume: /models/<name> (read-only; the
# underlying dm-verity/EROFS mount is physically read-only anyway)
VOL_GUEST_ROOT = os.environ.get("VOL_GUEST_ROOT", "/models")


def _staged_bytes(stage) -> int:
    """Weights bytes of a staged nn-graph dir: the sum of its (symlinked)
    model files, split families included - the ggml preload-order sort key.
    stat() follows the links into the dm-verity mount; a vanished file counts
    0 rather than failing the launch (the preload will say so loudly)."""
    total = 0
    for pat in ("*.gguf", "*.safetensors", "*.ckpt"):
        for p in stage.glob(pat):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _stage_nn_graph(name: str, gguf):
    """wasmtime's -S nn-graph loads a DIRECTORY, registers the graph under the
    dir BASENAME, and wants model.gguf (or a single *.gguf, or one complete
    split family) inside. Modelwrap mounts are named mpk-<root_hash> and
    multi-quant HF repos carry many *.gguf, so neither is directly loadable:
    stage a symlink dir named after the VOLUME - <FS_DIR>/nn-graph/<name>/ -
    and hand wasmtime that. A single-file model stages as model.gguf; a SPLIT
    model stages every part under its REAL basename (llama.cpp derives the
    sibling paths from part 00001's name, so the split names must survive).
    The staging dir holds no bytes; reads resolve inside the dm-verity mount.
    Re-linked atomically on every launch - and stale *.gguf links from a
    previous selection are pruned - so a changed MODEL_VOLUMES selection takes
    effect and concurrent launches never see a missing or ambiguous file."""
    d = FS_DIR / "nn-graph" / name
    fam = _split_family(gguf)
    # sd checkpoints stage under model.<their real suffix> (the sdcpp backend
    # accepts model.safetensors / model.gguf / a single file); LLM ggufs keep
    # the model.gguf / split-family contract.
    targets = {x.name: x for x in fam} if fam else {f"model{gguf.suffix}": gguf}
    # A VISION volume pairs its weights with a projector (*mmproj*.gguf), and the
    # ggml backend looks for that projector IN THIS DIRECTORY, beside the model.
    # Staging only the model therefore makes the projector INVISIBLE to the host
    # while the guest can plainly see it in its own /models mount - which reads as
    # "the volume carries no *mmproj*.gguf, or this node's toolchain is too old"
    # when in fact the file is right there and the toolchain is fine. Stage it
    # under its real name; the host matches this same *mmproj* convention from the
    # other side, and its "exactly one MODEL" pick already excludes projectors.
    # GGUF only: sd checkpoints have no such pairing.
    if gguf.suffix == ".gguf":
        for extra in sorted(gguf.parent.glob("*.gguf")):
            if "mmproj" in extra.name.lower():
                targets[extra.name] = extra
    try:
        d.mkdir(parents=True, exist_ok=True)
        for pat in ("*.gguf", "*.safetensors", "*.ckpt"):
            for stale in d.glob(pat):
                if stale.name not in targets:
                    stale.unlink()
        for link_name, src in targets.items():
            tmp = d / f".{link_name}.{os.getpid()}"
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
            tmp.symlink_to(src)
            os.replace(tmp, d / link_name)
        return d
    except OSError as e:
        print(f"[nn-graph] staging volume '{name}' failed: {e}", flush=True)
        return None


# --- preload capability probe ---------------------------------------------- #
# Which `-S nn-graph=<kind>::` preload kinds THIS wasmtime toolchain
# implements, probed ONCE (lazily, before the first launch that wants them)
# with throwaway serve processes. Gating launches on this makes manager and
# toolchain rollouts order-independent: emitting onnx:: to a pre-preload
# wasmtime ABORTS the tenant at startup (upstream semantics look for
# <dir>/model.onnx - "No such file or directory"), and sd:: to a build
# without the sdcpp feature dies with "unknown graph encoding: sd".
#
# onnx is a POSITIVE-signal probe: an 84-byte Identity model staged at
# graph/sub/model.onnx preloads ("wasi-nn graph preload done") only with the
# multi-graph tree walk. sd discriminates ERROR text on an empty dir: the
# sdcpp backend complains it wants a checkpoint ("expected model.gguf..."),
# an unsupported build says "unknown graph encoding". Unknown output = not
# supported (fail safe: tenants just keep the guest-load contract).
_ONNX_PROBE_MODEL = bytes.fromhex(
    "0808120d656e636c6176652d70726f62653a3b0a100a017812017922084964656e74697479"
    "120570726f62655a0f0a0178120a0a08080112040a020801620f0a0179120a0a0808011204"
    "0a02080142040a00100d"
)
_PRELOAD_SUPPORT = {"state": "unprobed", "onnx": False, "sd": False, "nvenc": False,
                    "detail": ""}
_PRELOAD_PROBE_LOCK = threading.Lock()


def _probe_serve_output(extra_args, env_extra, timeout=45.0):
    """Launch a throwaway `wasmtime serve` with `extra_args` on the boot
    fixture and return its combined stdout+stderr until exit, preload-done,
    or timeout. Only used by _preload_support."""
    import select
    wasm = _fixture_wasm()
    if MOCK or not wasm.is_file():
        return None
    port = _free_port()
    cmd = [WASMTIME, "serve", "-Scli", "-Shttp", "-Snn", *extra_args,
           "--addr", f"{HOST_IP}:{port}", str(wasm)]
    # scrub the node-global sdcpp component-file envs: the sd leg probes an
    # EMPTY dir to reach the backend's checkpoint-picker error ("expected
    # model..."), and a leaked ENCLAVE_SD_*_FILE diverts it into env-file
    # validation with error text the classifier doesn't know - misreading a
    # capable toolchain as unsupported (happened live on v0.5.133, the first
    # release with these envs set fleet-wide).
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("ENCLAVE_SD_") and k.endswith("_FILE"))}
    env.update(env_extra)
    try:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, preexec_fn=_preexec)
    except Exception as e:                                       # noqa: BLE001
        return f"spawn failed: {e}"
    out = []
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            r, _, _ = select.select([proc.stdout], [], [], 0.25)
            if r:
                line = proc.stdout.readline()
                if line:
                    out.append(line.strip())
            if proc.poll() is not None:
                out.extend(x.strip() for x in (proc.stdout.read() or "").splitlines())
                break
            # the preload-done line means serve came up and stayed up
            if any("preload done" in x for x in out):
                break
        return "\n".join(out)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:                                        # noqa: BLE001
            pass


def _nn_budgets(gpu_share, cpu_share, vram_bytes, nn_resident_other=0):
    """Which memory a preloaded graph must fit here, and what to call it.

    Returns (share_ram_bytes, node_ram_bytes, budget_bytes, budget_kind,
             ggml_budget_bytes, ggml_budget_kind, gpu_tenant).

    Split out of _volume_args so it can be driven directly: this is the gate
    that decides whether a mounted volume is preloaded at all, and getting it
    wrong is silent - the model simply never appears and the app reports it
    unfit. test/nn-budgets.test.mjs pins the tenant shapes.

    A SHIELDED node needs no special case here, which is worth stating because
    it does not look that way from the tenant's side. Such a box sells GPU
    shares (its supervisor adopts the card at runtime and advertises gpu:true)
    and its tenants carry a real vram_bytes reservation - but the card is on
    the untrusted HOST, reached per matmul, so the guest's manager is told
    NODE_HAS_GPU=0 and every tenant already takes the RAM path below. The
    reservation is an offload budget, not residency, and nothing on a shielded
    box is resident on the card. NODE_HAS_GPU means LOCAL card throughout this
    file - the thing whose OOM calls ggml_abort and kills the tenant - and no
    flavor sets it beside a shielded worker.
    """
    # CPU-ONLY NODE: the same question, but the resource behaves nothing
    # like VRAM, so the answer is not "a slice of the node per share".
    # ggml MMAPS a GGUF (llama.cpp: "mmap = true", "CPU_Mapped model buffer
    # size = 20190.50 MiB"): the weights are FILE-BACKED page cache the
    # kernel reclaims under pressure, not memory the tenant holds. What the
    # tenant actually allocates is its KV cache and compute buffers - 512
    # MiB for a 27B at 8k context - plus its linear memory, which
    # `-W max-memory-size` already caps. So charging 20 GB of reclaimable
    # page cache against a share was wrong twice over: it refused models a
    # node has ample room for, and it measured the wrong bytes.
    #
    # What DOES matter is that the weights fit the NODE: a model larger
    # than RAM page-thrashes forever rather than running slowly. So the
    # ggml budget is the node's usable RAM (minus a reserve for the base
    # system, other tenants' KV, and slack), independent of share. The
    # share still governs what the tenant gets: CPU time and its own
    # allocations. Operators who want the strict per-share rule anyway can
    # set WASM_CPU_NN_BUDGET=share.
    #
    # sd/onnx are NOT mmap-backed the same way (sdcpp builds anonymous
    # buffers, ORT sessions allocate per request), so they keep the
    # per-share budget on a CPU node - reclaimable and resident are
    # different promises and only ggml makes the first one.
    # ...and node RAM is SHARED, so the node's usable pool is net of what
    # other live tenants already hold resident. Without that term the budget
    # is per-deployment only: two tenants each clear a 23 GiB check on a
    # 29 GB box and the pair thrashes. Subtracting it also keeps the RAM
    # ledger and this gate reading the same node - _rec_ram_mb charges the
    # very bytes we deduct here.
    share_ram_bytes = int(cpu_share * NODE_RAM_GB * (1 << 30))
    node_ram_bytes = max(0, int((NODE_RAM_GB - CPU_NN_RESERVE_GB) * (1 << 30)) - max(0, nn_resident_other))
    # WHICH budget applies is a property of the TENANT, not the node: a GPU
    # box also hosts 0-GPU tenants, and they run on cores. Keying this on
    # NODE_HAS_GPU gave them a VRAM budget of zero, so every volume was
    # skipped "exceeds the VRAM budget" and the app could never load.
    gpu_tenant = NODE_HAS_GPU and gpu_share > 0
    # ...and a SHIELDED share is not a local card, which changes the answer
    # again for ggml. The hard VRAM gate below exists because a CUDA OOM
    # inside compute calls ggml_abort and takes the whole wasmtime process
    # down - so on a LOCAL card a model that cannot fit must never be
    # preloaded, let alone probed. A shielded tenant has no local card at
    # all (CUDA_VISIBLE_DEVICES="", ENCLAVE_GGML_N_GPU_LAYERS=0): its
    # weights are mmap'd page cache in the CVM exactly like a CPU tenant's,
    # and the card is reached per-matmul over the masked-offload protocol,
    # where being too big is SLOW, not fatal. The backend already degrades
    # on its own - placement is a policy (SHIELDED_MIN_MACS/MAX_M), a
    # refused reservation reads as a dead link and computes in the enclave,
    # and with no calibration it claims nothing and every matmul stays
    # inside. Pricing its ggml graphs against the offload reservation
    # skipped the volume entirely, so load_by_name() failed instantly and
    # the app reported "unfit" for a model the node holds comfortably
    # (2026-08-31, eyesoff-ai on metal0: a 24 GB model refused against a
    # 2.3 GB reservation while 58 GB of node RAM sat free).
    if gpu_tenant:
        budget_bytes, budget_kind = vram_bytes, "VRAM"
        ggml_budget_bytes, ggml_budget_kind = vram_bytes, "VRAM"
    else:
        budget_bytes, budget_kind = share_ram_bytes, "RAM"
        ggml_budget_bytes = share_ram_bytes if CPU_NN_BUDGET == "share" else node_ram_bytes
        ggml_budget_kind = "RAM"
    return (share_ram_bytes, node_ram_bytes, budget_bytes, budget_kind,
            ggml_budget_bytes, ggml_budget_kind, gpu_tenant)


def _preload_support() -> dict:
    with _PRELOAD_PROBE_LOCK:
        if _PRELOAD_SUPPORT["state"] != "unprobed":
            return _PRELOAD_SUPPORT
        detail = []
        try:
            graph = FS_DIR / "preload-probe" / "graph"
            (graph / "sub").mkdir(parents=True, exist_ok=True)
            (graph / "sub" / "model.onnx").write_bytes(_ONNX_PROBE_MODEL)
            empty = FS_DIR / "preload-probe" / "empty"
            empty.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _PRELOAD_SUPPORT.update(state="failed", detail=f"fixture: {e}")
            print(f"[preload-probe] fixture staging failed: {e}", flush=True)
            return _PRELOAD_SUPPORT
        o = _probe_serve_output(["-S", f"nn-graph=onnx::{graph}"],
                                {"ENCLAVE_ONNX_PRELOAD_TARGET": "cpu"})
        onnx_ok = bool(o) and "preload done" in o
        detail.append(f"onnx: {'ok' if onnx_ok else (o or 'no fixture/mock').splitlines()[-1][:120]}")
        # ENCLAVE_SD_USE_GPU=0 skips the strict-GPU check so the probe reaches
        # the checkpoint picker deterministically on CPU and GPU nodes alike
        s = _probe_serve_output(["-S", f"nn-graph=sd::{empty}"],
                                {"ENCLAVE_SD_USE_GPU": "0"})
        sd_ok = bool(s) and "unknown graph encoding" not in s and "expected model" in s
        detail.append(f"sd: {'ok' if sd_ok else (s or 'no fixture/mock').splitlines()[-1][:120]}")
        # nvenc reads no directory at all - the graph IS the encoder - so an
        # empty dir is the real fixture rather than a stand-in. Two answers mean
        # the backend is THERE: "preload done" on a node with an encoder, and
        # its own "no hardware video encoder" where there is none. Only
        # "unknown graph encoding" means this toolchain lacks the feature.
        # Distinguishing them matters: a GPU node that answers the second way
        # has a driver problem, not a rollout problem.
        n = _probe_serve_output(["-S", f"nn-graph=nvenc::{empty}"], {})
        nvenc_ok = bool(n) and "unknown graph encoding" not in n and (
            "preload done" in n or "no hardware video encoder" in n)
        nvenc_here = bool(n) and "preload done" in n
        detail.append(f"nvenc: {'ok' if nvenc_ok else (n or 'no fixture/mock').splitlines()[-1][:120]}"
                      + ("" if nvenc_here or not nvenc_ok else " (backend present, no encoder on this node)"))
        _PRELOAD_SUPPORT.update(state="probed", onnx=onnx_ok, sd=sd_ok,
                                nvenc=nvenc_ok and nvenc_here,
                                detail="; ".join(detail))
        print(f"[preload-probe] onnx={onnx_ok} sd={sd_ok} nvenc={_PRELOAD_SUPPORT['nvenc']} "
              f"({_PRELOAD_SUPPORT['detail']})", flush=True)
        return _PRELOAD_SUPPORT


def _stage_onnx_dir(name: str, host_path):
    """Stage a WHOLE volume dir for -S nn-graph=onnx::<dir> (and sd::<dir>
    component-layout volumes): unlike the gguf case there is no file to
    pick - the onnx toolchain walks the whole tree, and the sdcpp backend
    resolves the ENCLAVE_SD_*_FILE names inside it - so the stage is ONE
    symlink to the mount, named after the VOLUME (the mpk-<hash> mount name
    must not leak into graph names). Atomic re-link per launch, like
    _stage_nn_graph."""
    d = FS_DIR / "nn-graph" / name
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.is_symlink() and d.is_dir():
            shutil.rmtree(d)  # a stale gguf-style staging dir from a re-typed volume
        tmp = d.parent / f".{name}.{os.getpid()}"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(host_path)
        os.replace(tmp, d)
        return d
    except OSError as e:
        print(f"[nn-graph] staging onnx volume '{name}' failed: {e}", flush=True)
        return None


# --- GPU request-level arbiter (work-conserving fair share) ----------------- #
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is a STATIC partition, fixed the moment a
# tenant's wasmtime attaches to MPS: a 5%-share tenant is locked to ~5% of the
# SMs even when the rest of the card is idle, and MPS has no weight/fair-share
# scheduler to do better (set_active_thread_percentage only affects FUTURE
# clients). So idle capacity is wasted by construction. The arbiter replaces
# the SM cap as the *fairness* mechanism: GPU tenants launch with the FULL SM
# budget (WASM_NN_ARB_SM_PCT, default 100) and instead take short-lived TURNS
# on the card — the patched toolchain (wasmtime-nn-arbiter.patch) acquires a
# grant over this Unix socket around each natural compute quantum (one ggml
# decode turn, one ORT run, one sd denoise step) and releases it after. The
# scheduler is weighted fair queuing over VIRTUAL TIME: a grant's wall-clock
# hold is charged to the tenant as held/gpuShare, and the next grant goes to
# the waiting tenant with the smallest virtual time — so an idle card grants
# instantly (work-conserving burst to 100%), and a contended card divides GPU
# TIME in proportion to purchased shares. The pinned-VRAM cap is UNTOUCHED:
# device memory cannot be reclaimed from a burster, so it can never be
# work-conserving and stays sized by the share.
#
# FAIL-OPEN, both sides, always: arbitration is a fairness upgrade, not a
# wall. The toolchain client runs computes unarbitrated the moment the socket
# misbehaves (and reconnects with backoff, so a manager restart re-arms it);
# this server never fails a launch, revokes the slot of any grant held past
# WASM_NN_ARB_MAX_HOLD_SECS (a wedged CUDA call must not freeze the queue — a
# known GPU failure mode here is a D-state ioctl), and a tenant that dies
# mid-grant releases by disconnect. Cross-tenant floors degrade to the
# hardware scheduler in every failure mode — i.e. to plain uncapped sharing,
# never to an outage.
#
# ROLLOUT (launcher and wasmtime move by different hands — see the loopback
# probe's doctrine): OFF unless the operator sets WASM_NN_ARBITER=1, and even
# then tenants keep today's hard MPS caps until _arbiter_support() PROVES the
# binary carries the client (it launches a throwaway `wasmtime serve -Snn`
# pointed at a probe socket and requires an actual hello). Unproven means
# hard caps — behavior is bit-identical to the pre-arbiter fleet.
#
# Queues: HELLO names a queue; grants are exclusive per queue (WASM_NN_ARB_CONC,
# default 1). Today every tenant lands on queue "0" — wasi-nn tenants share
# card 0 (the pinned-VRAM cap is written for device 0). If per-tenant card
# packing lands later, the launcher sets ENCLAVE_NN_ARB_QUEUE per card and
# this file needs no change.
NN_ARB_ENABLED  = os.environ.get("WASM_NN_ARBITER", "").strip().lower() in ("1", "true", "on")
NN_ARB_SOCK     = os.environ.get("WASM_NN_ARB_SOCK", "/tmp/enclave-nn-arb.sock")
NN_ARB_SM_PCT   = min(100, max(1, int(os.environ.get("WASM_NN_ARB_SM_PCT", "100") or 100)))
NN_ARB_MAX_HOLD = float(os.environ.get("WASM_NN_ARB_MAX_HOLD_SECS", "30") or 30)
NN_ARB_CONC     = max(1, int(os.environ.get("WASM_NN_ARB_CONC", "1") or 1))
# Hesitation window (below): how long a freed slot waits for a more-deserving
# tenant's next request before settling for a waiter. Sized to cover a guest's
# gap between compute quanta (sample + stream a token: ~sub-ms to a few ms).
NN_ARB_GRACE    = float(os.environ.get("WASM_NN_ARB_GRACE_MS", "15") or 15) / 1000.0


class NnArbScheduler:
    """The pure scheduler state machine — no I/O, no locking, injectable clock,
    so tests drive it directly. The socket server below is a thin shell that
    translates connection events into these calls under one lock and writes
    out the grants they return (lists of (conn_id, req_id)).

    Virtual-time bookkeeping: each tenant carries `vt`, charged held/weight at
    release; the queue's `vclock` rides up to the granted tenant's vt so a
    tenant that was idle re-enters at max(own vt, vclock) — idleness banks no
    credit (else a long-idle tenant could monopolize the card for minutes to
    "catch up"). FIFO within a tenant; min-vt across tenants.

    HESITATION (the part that makes weights actually bite): clients are
    synchronous — hold, release, compute the next step, re-request — so at
    the instant a tenant releases, it is never in the waiting list, and
    naive work-conserving dispatch hands the slot to whoever IS waiting.
    Every release then alternates 1:1 and a 50% share equals a 5% share
    under contention. So dispatch hesitates: when some recently-releasing
    tenant is MORE underserved (lower vt) than every waiter, the slot is
    held for it for NN_ARB_GRACE (its next request typically lands within a
    millisecond) before settling for the waiter. Bounded work-conservation
    loss (grace ms per handoff, only under contention), exact long-run
    share-proportional split. conc=1 only — with multiple slots the spare
    capacity already absorbs the race."""

    def __init__(self, conc=None, max_hold=None, grace=None, clock=time.monotonic):
        self.conc = max(1, int(conc if conc is not None else NN_ARB_CONC))
        self.max_hold = float(max_hold if max_hold is not None else NN_ARB_MAX_HOLD)
        self.grace = float(grace if grace is not None else NN_ARB_GRACE)
        # how recently a tenant must have released to be worth hesitating for
        self.phantom_window = max(0.25, self.grace * 4)
        self.clock = clock
        self.conns = {}     # conn_id -> {"tenant", "queue"}
        self.queues = {}    # name -> {"vclock", "tenants", "active", "waiting", "hold"}
        self.stats = {"grants": 0, "revokes": 0, "hesitations": 0}

    def _q(self, name):
        q = self.queues.get(name)
        if q is None:
            q = self.queues[name] = {"vclock": 0.0, "tenants": {}, "active": {},
                                     "waiting": [], "hold": None}
        return q

    def _busy(self, q, tenant) -> bool:
        return (any(w["tenant"] == tenant for w in q["waiting"])
                or any(a["tenant"] == tenant for a in q["active"].values()))

    def hello(self, conn_id, tenant, weight, queue):
        tenant, queue = str(tenant or conn_id)[:128], str(queue or "0")[:64]
        self.conns[conn_id] = {"tenant": tenant, "queue": queue}
        q = self._q(queue)
        t = q["tenants"].setdefault(tenant, {"vt": 0.0, "weight": 0.01, "conns": 0})
        t["conns"] += 1
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 0.0
        if w > 0:
            t["weight"] = max(1e-4, min(1.0, w))

    def acquire(self, conn_id, req_id):
        c = self.conns.get(conn_id)
        if c is None:
            return []
        q = self._q(c["queue"])
        # the request a hesitation was waiting for: clear the hold, dispatch
        # picks this tenant by min-vt on its own merits
        if q["hold"] is not None and q["hold"]["tenant"] == c["tenant"]:
            q["hold"] = None
        t = q["tenants"].get(c["tenant"])
        if t is not None:
            # release->re-acquire gap, EWMA'd: the hesitation PREDICTOR (see
            # _phantom). A hot decode loop re-requests in sub-ms; a tenant
            # that came back after seconds must not be hesitated for.
            lr = t.get("last_rel")
            if lr is not None:
                gap = max(0.0, self.clock() - lr)
                ew = t.get("gap_ewma")
                t["gap_ewma"] = gap if ew is None else 0.7 * ew + 0.3 * gap
            if not self._busy(q, c["tenant"]):
                t["vt"] = max(t["vt"], q["vclock"])
        q["waiting"].append({"conn": conn_id, "req": req_id, "tenant": c["tenant"]})
        return self._dispatch(q)

    def release(self, conn_id, req_id):
        c = self.conns.get(conn_id)
        if c is None:
            return []
        q = self._q(c["queue"])
        a = q["active"].pop((conn_id, req_id), None)
        if a is None:
            return []            # duplicate, or already revoked — ignored
        self._charge(q, a)
        t = q["tenants"].get(c["tenant"])
        if t is not None:
            t["last_rel"] = self.clock()   # hesitation eligibility (voluntary rel only)
        return self._dispatch(q)

    def disconnect(self, conn_id):
        c = self.conns.pop(conn_id, None)
        if c is None:
            return []
        q = self._q(c["queue"])
        q["waiting"] = [w for w in q["waiting"] if w["conn"] != conn_id]
        for key in [k for k in q["active"] if k[0] == conn_id]:
            self._charge(q, q["active"].pop(key))
        t = q["tenants"].get(c["tenant"])
        if t is not None:
            t["conns"] -= 1
            if t["conns"] <= 0 and not self._busy(q, c["tenant"]):
                del q["tenants"][c["tenant"]]
                if q["hold"] is not None and q["hold"]["tenant"] == c["tenant"]:
                    q["hold"] = None       # never hesitate for the departed
        return self._dispatch(q)

    def tick(self):
        """Timer duties: revoke grants held past max_hold (a wedged CUDA call
        must not freeze the queue — charged and dropped, the eventual rel is
        ignored) and resolve expired hesitations. Returns (grants, revoked)
        where revoked = [(queue, tenant, held_secs)]."""
        grants, revoked = [], []
        now = self.clock()
        for name, q in self.queues.items():
            over = [k for k, a in q["active"].items() if now - a["t0"] > self.max_hold]
            for key in over:
                a = q["active"].pop(key)
                self._charge(q, a)
                self.stats["revokes"] += 1
                revoked.append((name, a["tenant"], now - a["t0"]))
            if over or (q["hold"] is not None and q["hold"]["until"] <= now):
                grants += self._dispatch(q)
        return grants, revoked

    def _charge(self, q, a):
        t = q["tenants"].get(a["tenant"])
        if t is not None:
            t["vt"] += max(0.0, self.clock() - a["t0"]) / t["weight"]

    def _phantom(self, q, below_vt, now):
        """The most underserved tenant worth hesitating for: connected, not
        waiting or active, released voluntarily within the window, strictly
        lower vt than the best waiter — AND predicted to actually come back
        within the grace (its release->re-acquire gap EWMA fits it). The
        prediction gate is load-bearing: without it, any tenant that touched
        the card in the last window and sits marginally below the dominant
        tenant's vt taxes EVERY dispatch with a dead hold — observed live
        2026-08-01 as llm-chat decode slowdown minutes after arming. A hot
        loop re-requests in sub-ms and keeps its protection; a sparse caller
        proves nothing and gets ordinary min-vt service when it shows up."""
        best, best_vt = None, below_vt
        for name, t in q["tenants"].items():
            if (t["conns"] > 0 and t["vt"] < best_vt
                    and now - t.get("last_rel", -1e18) < self.phantom_window
                    and t.get("gap_ewma") is not None
                    and t["gap_ewma"] <= self.grace
                    and not self._busy(q, name)):
                best, best_vt = name, t["vt"]
        return best

    def _dispatch(self, q):
        grants = []
        now = self.clock()
        while q["waiting"] and len(q["active"]) < self.conc:
            if q["hold"] is not None:
                if q["hold"]["until"] > now:
                    break              # slot spoken for (hesitation pending)
                q["hold"] = None
            # min() keeps the FIRST minimal element, so ties (same tenant, or
            # equal-vt tenants) resolve in arrival order
            best = min(q["waiting"], key=lambda w: q["tenants"][w["tenant"]]["vt"])
            if self.conc == 1:
                ph = self._phantom(q, q["tenants"][best["tenant"]]["vt"], now)
                if ph is not None:
                    q["hold"] = {"tenant": ph, "until": now + self.grace}
                    self.stats["hesitations"] += 1
                    break
            q["waiting"].remove(best)
            q["vclock"] = max(q["vclock"], q["tenants"][best["tenant"]]["vt"])
            q["active"][(best["conn"], best["req"])] = {"tenant": best["tenant"],
                                                        "t0": self.clock()}
            self.stats["grants"] += 1
            grants.append((best["conn"], best["req"]))
        return grants

    def snapshot(self):
        return {"stats": dict(self.stats),
                "queues": {name: {"active": len(q["active"]),
                                  "waiting": len(q["waiting"]),
                                  "tenants": {t: {"weight": v["weight"],
                                                  "vt": round(v["vt"], 3)}
                                              for t, v in q["tenants"].items()}}
                           for name, q in self.queues.items()}}


class _NnArbServer:
    """The socket shell: newline-JSON over a Unix stream socket.
      client -> {"op":"hello","v":1,"tenant","weight","queue"}   (first line)
      server -> {"ok":true}
      client -> {"op":"acq","id":N}    ... server -> {"ok":true,"id":N} on grant
      client -> {"op":"rel","id":N}    (fire-and-forget)
    Disconnect = release everything the connection holds or awaits. The hello
    weight is advisory: when the tenant id matches a live record, the RECORD's
    gpuShare is authoritative (the env is manager-set either way — guests
    cannot reach this socket, wasmtime preopens no path to it)."""

    def __init__(self, path):
        self.path = path
        self.sched = NnArbScheduler()
        self.lock = threading.Lock()
        self.socks = {}                 # conn_id -> (socket, write_lock)
        self._n = 0
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(64)
        threading.Thread(target=self._accept_loop, daemon=True, name="nn-arb-accept").start()
        threading.Thread(target=self._tick_loop, daemon=True, name="nn-arb-tick").start()

    def _accept_loop(self):
        while True:
            try:
                s, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._conn_loop, args=(s,), daemon=True).start()

    def _conn_loop(self, s):
        cid = None
        try:
            f = s.makefile("r", encoding="utf-8", newline="\n")
            try:
                hello = json.loads(f.readline() or "null")
            except ValueError:
                hello = None
            if not isinstance(hello, dict) or hello.get("op") != "hello":
                s.close()
                return
            tenant = str(hello.get("tenant") or "")[:128]
            weight = hello.get("weight")
            queue = hello.get("queue")
            with _lock:
                rec = _apps.get(tenant)
                if rec and rec.get("gpuShare"):
                    weight = rec["gpuShare"]
                if rec:
                    queue = _nn_arb_queue(rec)
            with self.lock:
                self._n += 1
                cid = self._n
                self.socks[cid] = (s, threading.Lock())
                self.sched.hello(cid, tenant, weight, queue)
            self._send(cid, {"ok": True})
            for line in f:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                op, rid = msg.get("op"), msg.get("id")
                if not isinstance(rid, int):
                    continue
                with self.lock:
                    grants = (self.sched.acquire(cid, rid) if op == "acq"
                              else self.sched.release(cid, rid) if op == "rel" else [])
                for g in grants:
                    self._send(g[0], {"ok": True, "id": g[1]})
        except Exception:                                        # noqa: BLE001
            pass                       # a broken conn is just a disconnect
        finally:
            if cid is not None:
                with self.lock:
                    self.socks.pop(cid, None)
                    grants = self.sched.disconnect(cid)
                for g in grants:
                    self._send(g[0], {"ok": True, "id": g[1]})
            try:
                s.close()
            except OSError:
                pass

    def _send(self, cid, obj):
        with self.lock:
            ent = self.socks.get(cid)
        if not ent:
            return
        sock, wlock = ent
        try:
            with wlock:
                # COMPACT separators are WIRE FORMAT, not style: the toolchain
                # client parses grant frames with a substring scan that reads
                # the digits immediately after '"id":'. Default json.dumps
                # spacing ('"id": 7') made it parse an empty string, drop
                # EVERY grant, and ride the 120s fail-open watchdog on every
                # decode step — live 2026-08-01 as "forever warming" llm-chat
                # minutes after the v0.5.306 knob flip. Pinned by the wire
                # test's raw-bytes assertion; keep both in step.
                sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        except OSError:
            # let the conn's own reader thread observe the close and clean up
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _tick_loop(self):
        # 10ms cadence: hesitation holds (NN_ARB_GRACE ~15ms) must resolve
        # promptly or the grace becomes added latency for the waiter. The
        # sweep itself is O(active grants) — a handful of dict entries.
        n = 0
        last_stats = {}
        while True:
            time.sleep(0.01)
            n += 1
            if n % 6000 == 0:            # once a minute, only when something moved
                with self.lock:
                    s = dict(self.sched.stats)
                if s != last_stats:
                    print(f"[nn-arb] stats: {s}", flush=True)
                    last_stats = s
            with self.lock:
                grants, revoked = self.sched.tick()
            for name, tenant, held in revoked:
                print(f"[nn-arb] REVOKED queue {name}: '{tenant}' held a grant "
                      f"{held:.1f}s (> {self.sched.max_hold:.0f}s) — slot freed, "
                      f"tenant keeps running unarbitrated until it releases", flush=True)
            for g in grants:
                self._send(g[0], {"ok": True, "id": g[1]})

    def snapshot(self):
        with self.lock:
            out = self.sched.snapshot()
        out["sock"] = self.path
        return out


_NN_ARB = None                  # the running _NnArbServer, or None
_NN_ARB_SUPPORT = {"state": "unprobed", "supported": False, "detail": ""}
_NN_ARB_PROBE_LOCK = threading.Lock()


def _arbiter_support() -> dict:
    """Does THIS wasmtime carry the nn-arbiter client (wasmtime-nn-arbiter.patch)?

    POSITIVE-signal probe, probed once, lazily, from the launch path (same
    doctrine as the preload probe): launch a throwaway `wasmtime serve -Snn`
    on the boot fixture with ENCLAVE_NN_ARBITER pointing at a probe socket,
    and call it supported ONLY when the runtime actually connects and says
    hello — the exact behavior tenants will rely on, not a help-text marker.
    Unproven means HARD CAPS: tenants keep the share-sized SM percentage and
    behavior is bit-identical to the pre-arbiter fleet. That direction matters
    because raising SM caps WITHOUT a working client would quietly stop
    enforcing the very floors tenants pay for."""
    with _NN_ARB_PROBE_LOCK:
        if _NN_ARB_SUPPORT["state"] != "unprobed":
            return _NN_ARB_SUPPORT
        wasm = _fixture_wasm()
        if MOCK or not wasm.is_file():
            _NN_ARB_SUPPORT.update(state="probed", supported=False,
                                   detail="mock" if MOCK else "no nn-demo.wasm fixture")
            return _NN_ARB_SUPPORT
        sock_path = str(FS_DIR / "nn-arb-probe.sock")
        srv = proc = None
        try:
            FS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(sock_path)
            srv.listen(1)
            srv.settimeout(1.0)
            env = dict(os.environ)
            env.update({"ENCLAVE_NN_ARBITER": sock_path,
                        "ENCLAVE_NN_ARB_TENANT": "capability-probe",
                        "ENCLAVE_NN_ARB_WEIGHT": "1",
                        "ENCLAVE_NN_ARB_QUEUE": "probe"})
            proc = subprocess.Popen([WASMTIME, "serve", "-Scli", "-Shttp", "-Snn",
                                     "--addr", f"{HOST_IP}:{_free_port()}", str(wasm)],
                                    env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    preexec_fn=_preexec)
            deadline, got = time.time() + 30, ""
            while time.time() < deadline:
                if proc.poll() is not None:
                    got = f"serve exited rc={proc.returncode} before connecting"
                    break
                try:
                    c, _ = srv.accept()
                except socket.timeout:
                    continue
                c.settimeout(5.0)
                try:
                    got = (c.makefile("r", encoding="utf-8").readline() or "").strip()
                    c.sendall(b'{"ok":true}\n')
                except OSError:
                    pass
                finally:
                    c.close()
                break
            ok = '"hello"' in got
            _NN_ARB_SUPPORT.update(state="probed", supported=ok,
                                   detail="hello received" if ok
                                   else (got or "no connect within 30s"))
        except Exception as e:                                   # noqa: BLE001
            _NN_ARB_SUPPORT.update(state="failed", supported=False, detail=f"probe error: {e}")
        finally:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:                                # noqa: BLE001
                    pass
            if srv is not None:
                srv.close()
            try:
                os.unlink(sock_path)
            except OSError:
                pass
        print(f"[nn-arb] toolchain support: {_NN_ARB_SUPPORT['supported']} "
              f"({_NN_ARB_SUPPORT['detail']})", flush=True)
        return _NN_ARB_SUPPORT


def _start_nn_arbiter():
    """Boot-time (main). Never raises: a failed bind leaves _NN_ARB None and
    every launch keeps hard caps."""
    global _NN_ARB
    if _NN_ARB is not None:
        return
    try:
        _NN_ARB = _NnArbServer(NN_ARB_SOCK)
        print(f"[nn-arb] serving on {NN_ARB_SOCK} (conc={NN_ARB_CONC}, "
              f"sm={NN_ARB_SM_PCT}%, max-hold={NN_ARB_MAX_HOLD:.0f}s)", flush=True)
    except Exception as e:                                       # noqa: BLE001
        _NN_ARB = None
        print(f"[nn-arb] failed to start ({e}) — GPU tenants keep hard MPS caps", flush=True)


def _nn_arbiter_live() -> bool:
    """Arbitration governs NEW GPU launches: knob on, server up, toolchain
    proven. May block once (~30s) on the lazy probe — launch path only; the
    health path reads _NN_ARB_SUPPORT without triggering it."""
    if not (NN_ARB_ENABLED and _NN_ARB is not None):
        return False
    return bool(_arbiter_support().get("supported"))


def _nn_arb_queue(rec: dict) -> str:
    """One fairness queue per worker or whole pool, bound to the launch record."""
    if (rec.get("shielded") or {}).get("pooled"):
        return "shielded:pool"
    endpoint = str((rec.get("shielded") or {}).get("endpoint") or "")
    return ("shielded:" + endpoint) if endpoint else "0"


def _nn_arb_arm(env: dict, rec: dict, gpu_share: float) -> None:
    """Arm one tenant's runtime as an arbiter client — the ONE place the four
    ENCLAVE_NN_ARB* vars are written, shared by the CUDA branch and the
    shielded branch so the two can never drift. Grant weight = the launch-time
    gpuShare (re-checked against the RECORD by the arbiter at hello). Shielded
    workers have independent queues; tenants on the same worker share turns.
    A no-op unless
    the arbiter is live, which keeps OFF bit-identical to the pre-arbiter
    fleet."""
    if not _nn_arbiter_live():
        return
    env["ENCLAVE_NN_ARBITER"] = NN_ARB_SOCK
    env["ENCLAVE_NN_ARB_TENANT"] = rec["id"]
    env["ENCLAVE_NN_ARB_WEIGHT"] = str(gpu_share)
    # Independent host workers must not serialize behind the same GPU queue.
    env["ENCLAVE_NN_ARB_QUEUE"] = _nn_arb_queue(rec)
    rec["nnArbiter"] = True


def _nn_arb_public() -> dict:
    out = {"enabled": NN_ARB_ENABLED, "smPct": NN_ARB_SM_PCT,
           "probe": dict(_NN_ARB_SUPPORT)}
    if _NN_ARB is not None:
        out.update(_NN_ARB.snapshot())
    return out














_EGRESS_FS = None   # does this wasmtime carry the transparent-egress shim (-S egress)?

def _egress_supported() -> bool:
    """Probe (once) whether the wasmtime toolchain has the enclave transparent-egress
    shim: `-S egress=<host>:<port>` routes ALL guest outbound (wasi:sockets TCP
    connect AND the wasi:http outgoing handler) through the enclave's loopback
    SOCKS front, so an UNMODIFIED app leaves from the deployment's dedicated IPv6
    (wasmtime-egress.patch, phase 2). When present the manager makes egress
    transparent and drops the raw -Sinherit-network in run mode; on older
    toolchains it falls back to phase-1 (guest-visible ENCLAVE_EGRESS only)."""
    global _EGRESS_FS
    if _EGRESS_FS is None:
        try:
            r = subprocess.run([WASMTIME, "run", "-S", "help"],
                               capture_output=True, text=True, timeout=10)
            _EGRESS_FS = "egress=" in (r.stdout or "") + (r.stderr or "")
        except Exception:
            _EGRESS_FS = False
        print(f"[egress] wasmtime -S egress (transparent) support: {_EGRESS_FS}", flush=True)
    return _EGRESS_FS


def _parse_egress_url(url: str):
    """The supervisor hands us the per-deployment ENCLAVE_EGRESS verbatim: a
    `socks5h://<id>:<token>@<host>:<port>` URL. For TRANSPARENT egress we reuse
    its parts host-side — the endpoint on the `-S egress` flag and `<id>:<token>`
    in $ENCLAVE_EGRESS_CRED (guest-invisible). Returns {endpoint, cred} or None if it
    isn't a usable socks URL (then we leave egress as the guest-visible env only).
    Parsing the existing field means no supervisor<->manager protocol change."""
    try:
        u = urllib.parse.urlparse(url)
        if not u.scheme.startswith("socks5") or not u.username or not u.password or not u.hostname or not u.port:
            return None
        # username/password are percent-encoded in the URL; decode for SOCKS auth.
        uid = urllib.parse.unquote(u.username)
        tok = urllib.parse.unquote(u.password)
        if not uid or not tok:
            return None
        return {"endpoint": f"{u.hostname}:{u.port}", "cred": f"{uid}:{tok}"}
    except Exception:
        return None




def _no_card_env() -> dict:
    """The process env for an nn tenant that bought NO GPU share on a GPU box.

    Sibling of _nn_tenant_env, and the reason it exists: that one CAPS a tenant's
    use of the card (MPS SM% + pinned VRAM, both computed from the share), and a
    0-GPU tenant has no share to compute from - so instead of an uncapped
    tenant, there must be one with no card at all. Without this it would inherit
    the manager's environment, which on a GPU node carries the MPS pipe and an
    unrestricted view of the device: an unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
    means ALL the SMs, and an ExecutionTarget::Gpu request (ORT's CUDA EP, an
    sdcpp graph) would take the whole card on a box that sells it by the slice.

    Hide the device, unjoin MPS, and pin ggml to zero offloaded layers - the
    last one also stops a node-global ENCLAVE_GGML_N_GPU_LAYERS from reaching a
    tenant that never bought a card. A Gpu request then fails loudly, which is
    the honest answer, instead of quietly spending someone else's slice."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    env["ENCLAVE_GGML_N_GPU_LAYERS"] = "0"
    return env


def _nn_cfg_int(enclave_config, key: str, lo: int, hi: int):
    """An integer knob from the per-deployment config JSON, or None. Bools
    are excluded deliberately (json true would int() to 1), and out-of-range
    values are ignored rather than clamped - a config asking for something
    the platform won't grant should behave as if it never asked."""
    if not enclave_config:
        return None
    try:
        v = json.loads(enclave_config).get(key)
    except (ValueError, AttributeError):
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = int(v)
    return v if lo <= v <= hi else None


# Where the shielded ggml backend and its calibration live inside the guest
# image. Both are ordinary files in the measured image, not secrets: the backend
# is code we ship, and calibration is two public constants per site derived from
# the public weights.
SHIELDED_BACKEND_SO = os.environ.get("SHIELDED_BACKEND_SO", "/opt/enclave/shielded/libggml-shielded.so")
SHIELDED_CALIB_DIR  = os.environ.get("SHIELDED_CALIB_DIR", "/opt/enclave/shielded/calib")


@functools.lru_cache(maxsize=1)
def _shielded_pool_available() -> bool:
    """Probe the measured module, not just this manager's support for a field."""
    try:
        # ggml normally loads this module after its own registry symbols are
        # global. The capability function needs none of them; resolve lazily
        # here so the probe does not require initializing an inference engine.
        code = "import ctypes,os,sys; b=ctypes.CDLL(sys.argv[1],mode=os.RTLD_LAZY); sys.exit(0 if b.ggml_backend_shielded_pool_version()==1 else 1)"
        return subprocess.run([sys.executable, "-c", code, SHIELDED_BACKEND_SO],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _shielded_profile_tail(rec: dict) -> None:
    """Echo the backend's profile lines from a shielded tenant's log to our own
    stdout, i.e. the guest console, i.e. the host's journal. The tenant's log is
    a file the owner reads through /logs; the operator tuning the tier needs the
    same few lines without a wallet. Counters and timings only -- the backend
    prints nothing else under that prefix -- polled every 2 s, until the tenant
    exits."""
    path = rec.get("_log")
    if not path:
        return
    def run():
        pos = 0
        buf = b""
        while True:
            time.sleep(2.0)
            try:
                with open(path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except OSError:
                chunk = b""
            if chunk:
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    if ln.startswith(b"[shielded] profile:"):
                        print(f"[shielded] {rec['id']} {ln.decode('utf-8', 'replace')}", flush=True)
            proc = rec.get("_proc")
            if proc is not None and proc.poll() is not None:
                return
            if rec.get("status") in ("stopped", "failed", "removed"):
                return
    threading.Thread(target=run, daemon=True, name=f"shielded-profile-{rec.get('id')}").start()


def _shielded_shm_fields(worker: dict):
    """Only guest-derived, card-bound BAR mappings can enter a native process."""
    if "shmPath" not in worker and "shmBytes" not in worker:
        return None
    card_id, path, size = worker.get("cardId"), worker.get("shmPath"), worker.get("shmBytes")
    if (type(card_id) is not int or not 0 <= card_id < 16 or
            path != f"/dev/enclave-shielded-shm/card-{card_id}" or
            type(size) is not int or not 8 * 1048576 <= size <= 64 * 1048576 or size & (size - 1)):
        raise ValueError("invalid shielded shared-memory mapping")
    return path, size


def _shielded_tenant_env(spec: dict, model_volume: str = "") -> dict:
    """The env a tenant runs with when its GPU share is a SHIELDED card.

    The sibling of _nn_tenant_env, and deliberately none of the same things: there
    is no local CUDA device to cap, no MPS pipe to join, and no VRAM limit to pin,
    because the card is on the untrusted host and the tenant reaches it over the
    masked-offload protocol. What it gets instead is the worker's address and the
    shielded ggml backend, which the engine loads through GGML_BACKEND_PATH --
    already honoured by ggml_backend_load_all_from_path(), which enclave_llama.c
    calls at init, so no engine change is involved.

    ENCLAVE_GGML_N_GPU_LAYERS is explicitly ZERO here, and that is load-bearing
    rather than tidy. The ggml backend's whole job is to claim matmuls through
    ggml_backend_sched; a nonzero offload count would tell llama.cpp to move whole
    layers to a CUDA device that does not exist on this box, and the tenant would
    fail to launch instead of quietly using the shielded path.
    """
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    env.pop("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", None)
    env["CUDA_VISIBLE_DEVICES"] = ""          # there is no local card; do not let one be found
    env["ENCLAVE_GGML_N_GPU_LAYERS"] = "0"

    env.pop("SHIELDED_WORKERS", None)
    env.pop("SHIELDED_VSOCK_PORT", None)
    for name in ("SHIELDED_SHM", "SHIELDED_SHM_BYTES", "SHIELDED_SHM_RING"):
        env.pop(name, None)
    workers = (spec or {}).get("workers")
    if (spec or {}).get("pooled"):
        if not isinstance(workers, list) or not 1 <= len(workers) <= 16:
            raise ValueError("shielded pool needs 1..16 worker reservations")
        records, endpoints, devices, card_ids, vsock_ports = [], set(), set(), set(), set()
        total = 0
        for worker in workers:
            if not isinstance(worker, dict):
                raise ValueError("invalid shielded pool worker")
            endpoint = str(worker.get("endpoint") or "")
            host, _, port = endpoint.rpartition(":")
            if not re.fullmatch(r"[A-Za-z0-9._:-]{1,127}", host) or not port.isdecimal() or not 1 <= int(port) <= 65535:
                raise ValueError("invalid shielded pool endpoint")
            vport = int(worker.get("vsockPort") or 0)
            reserve = int(float(worker.get("vramGb") or 0) * (1 << 30))
            device = str(worker.get("deviceUuid") or "").lower()
            card_id = worker.get("cardId")
            endpoint = f"{host}:{int(port)}"
            if type(card_id) is not int or not 0 <= card_id < 16 or endpoint in endpoints or (device and device in devices) or card_id in card_ids or (vport > 0 and vport in vsock_ports) or not 0 <= vport <= (1 << 30) or not 0 < reserve <= (1 << 50):
                raise ValueError("invalid or duplicate shielded pool reservation")
            endpoints.add(endpoint); devices.add(device); card_ids.add(card_id); vsock_ports.add(vport)
            shm = _shielded_shm_fields(worker)
            records.append(f"{host}|{int(port)}|{vport}|{reserve}" + (f"|{shm[0]}|{shm[1]}" if shm else ""))
            total += reserve
        advertised = int(float(spec.get("vramGb") or 0) * (1 << 30))
        if abs(total - advertised) > len(workers):
            raise ValueError("shielded pool total differs from worker reservations")
        env["SHIELDED_WORKERS"] = "\n".join(records)
        # Keep the first address for the engine's shielded-capability detection.
        spec = {**spec, "endpoint": workers[0]["endpoint"], "vsockPort": workers[0].get("vsockPort", 0)}
    elif workers is not None:
        raise ValueError("worker reservations require a shielded pool")
    else:
        shm = _shielded_shm_fields(spec or {})
        if shm:
            env["SHIELDED_SHM"], size = shm
            env["SHIELDED_SHM_BYTES"] = str(size)
            env["SHIELDED_SHM_RING"] = "-1"  # first free ring, bounded by this BAR
    endpoint = str((spec or {}).get("endpoint") or "")
    host, _, port = endpoint.rpartition(":")
    env["SHIELDED_HOST"] = host or "10.0.2.2"
    env["SHIELDED_PORT"] = port or "9500"
    # The worker also listens on AF_VSOCK when the host runs it that way; the
    # backend tries it first and falls back to TCP. The guest's boot probe only
    # posts the port when the guest itself has a vsock device, so a tenant is
    # never told about a transport it cannot open.
    vport = int((spec or {}).get("vsockPort") or 0)
    if vport > 0:
        env["SHIELDED_VSOCK_PORT"] = str(vport)
    env["GGML_BACKEND_PATH"] = SHIELDED_BACKEND_SO
    # The tenant's slice of the card, RESERVED at HELLO (protocol 1.3): the
    # backend packs it into the handshake and the worker holds that much device
    # memory for the connection until it closes, refusing the HELLO when the
    # card cannot give it (the backend then computes in the enclave and
    # reconnects with backoff). This is the same number the supervisor sized the
    # share from (rec.shielded.vramGb = share x the card's budget) and the same
    # bytes ENCLAVE_VRAM_BYTES tells the engine it may use, so what the fleet
    # sold, what the engine allocates and what the worker holds are one figure.
    # A share of 0 (or an old supervisor sending no vramGb) reserves nothing:
    # the 4-byte HELLO every worker accepts, capped at the budget as before.
    reserve_bytes = int(float((spec or {}).get("vramGb") or 0) * (1 << 30))
    if reserve_bytes > 0:
        env["SHIELDED_RESERVE_BYTES"] = str(reserve_bytes)
    else:
        env.pop("SHIELDED_RESERVE_BYTES", None)

    # Calibration is per MODEL, so it is named after the volume the tenant serves.
    # Without it the backend claims nothing and every matmul stays in the enclave:
    # correct, and slow, rather than wrong.
    if model_volume:
        cal = os.path.join(SHIELDED_CALIB_DIR, f"{model_volume}.calib")
        if os.path.exists(cal):
            env["SHIELDED_CALIB"] = cal
        else:
            print(f"[shielded] no calibration for {model_volume} at {cal}; "
                  f"the tenant will run its matmuls in the enclave", flush=True)
    env.setdefault("WASMTIME_LOG", "wasmtime_wasi_nn=debug")
    # The backend's per-term profile (counters and milliseconds only, never a
    # value that crossed or was masked), printed every 4096 exchanges to the
    # tenant's stderr and echoed to the console by _shielded_profile_tail. It
    # is the only way to learn where a token's time goes INSIDE the CVM: the
    # host-loopback figure and the deployed one have disagreed, and the host
    # cannot see the guest's split of wire, mask, verify and CPU half any
    # other way.
    env.setdefault("SHIELDED_PROFILE", "1")
    # The backend says what it registered and what it claimed. Cheap -- a line per
    # weight at graph build, nothing per token -- and without it a shielded tenant
    # that silently claims NOTHING is indistinguishable from one that is working:
    # both serve tokens, one just quietly ignores the card it is billing for.
    env.setdefault("SHIELDED_VERBOSE", "1")
    # Host-configured tuning is applied last. Routing, reservation and calibration
    # fields remain bound to the launch spec; other tuning overrides defaults. The
    # one that exists today is SHIELDED_SPIN_US: the wire layer's bounded
    # MSG_DONTWAIT poll before it blocks on a reply, the guest-side answer to a
    # vhost-vsock exchange that costs 152 us in the CVM against 46 on the host.
    #
    # ONLY names starting with SHIELDED_ get through, and only printable string
    # values of bounded length. This spec travels host config -> fw_cfg -> the
    # guest's verdict file -> the supervisor -> here, and every hop is host-
    # influenced; an operator may tune the backend the tenant already talks to,
    # not inject arbitrary environment (LD_PRELOAD, WASMTIME_*, a model path)
    # into a tenant through a tuning knob. Anything else is dropped and named
    # in the log, never applied.
    te = (spec or {}).get("tenantEnv")
    if isinstance(te, dict):
        for k, v in te.items():
            k = str(k)
            if k in {"SHIELDED_WORKERS", "SHIELDED_HOST", "SHIELDED_PORT", "SHIELDED_VSOCK_PORT", "SHIELDED_RESERVE_BYTES", "SHIELDED_CALIB", "SHIELDED_SHM", "SHIELDED_SHM_BYTES", "SHIELDED_SHM_RING"}:
                continue  # routing, budget and model calibration come from the launch spec
            if not re.fullmatch(r"SHIELDED_[A-Z0-9_]{0,63}", k):
                print(f"[shielded] tenantEnv: dropping {k[:40]!r} (only SHIELDED_* names may be set from host config)", flush=True)
                continue
            if not isinstance(v, (str, int, float)) or isinstance(v, bool):
                print(f"[shielded] tenantEnv: dropping {k} (value must be a string)", flush=True)
                continue
            v = str(v)
            if len(v) > 256 or not all(0x20 <= ord(c) <= 0x7E for c in v):
                print(f"[shielded] tenantEnv: dropping {k} (value must be printable ASCII, at most 256 bytes)", flush=True)
                continue
            env[k] = v
    return env


def _nn_tenant_env(gpu_share: float, pinned: bool) -> dict:
    """The MPS cap env a GPU tenant's wasmtime process runs with. `pinned`
    adds the per-client VRAM limit; dropped when the probe found it poisonous
    (mode "nopin") - the SM cap is the validated, load-bearing one."""
    env = dict(os.environ)
    env["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE_DIR
    # Under a live arbiter the SM percentage stops being the fairness
    # mechanism (turn-taking is — see the nn-arb section) and becomes the
    # burst ceiling; without one it IS the enforcement of the sold share.
    if _nn_arbiter_live():
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(NN_ARB_SM_PCT)
    else:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(max(1, round(gpu_share * 100)))
    if pinned:
        env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = f"0={max(1, int(gpu_share * GPU_VRAM_GB * 1024))}M"
    # host-side wasi-nn traces into the tenant's log file (owner-readable via
    # the deployment logs endpoint) - names the backend step a hang died in
    env.setdefault("WASMTIME_LOG", "wasmtime_wasi_nn=debug")
    # ggml (GGUF) graphs: offload the whole model to the tenant's GPU share by
    # default. Load-bearing beyond tuning: the preload registry hardcodes
    # ExecutionTarget::Cpu, so WITHOUT this env a preloaded GGUF would run
    # pure-CPU on an H200 tenant; with it set nonzero the backend also REFUSES
    # to run if the CUDA module/driver didn't actually load (strict-GPU, no
    # silent fallback). setdefault: a dashboard env on the manager container
    # overrides per node.
    env.setdefault("ENCLAVE_GGML_N_GPU_LAYERS", "-1")
    # Recurrent cell pinning is safe again as of the mm27 engine (fleet
    # v0.5.386): mm14's pin + skip-reorder now only applies at n_seqs==1,
    # so single-stream keeps CUDA-graph reuse and concurrent hybrid-SSM
    # sequences use stock find_slot packing. The 2026-08-04 fleet-wide
    # LLAMA_RS_PIN_CELLS=0 override that mitigated the mm14 concurrency
    # abort (GGML_ASSERT(cell.has_seq_id) under multi-user load) is
    # therefore gone; the env stays honored by the engine as a kill switch
    # should pinning ever need to come off again per node.
    # FUSED ATTENTION. Background: ORT's sm_90 flash/memory-efficient attention
    # kernels compute launch heuristics that integer-divide by the device SM
    # budget; under a small MPS partition (a 2-4% slice of an H200) the
    # denominator floors to zero -> SIGFPE / decode hang mid-compute (observed
    # live 2026-07-05; sm_86 is fine - different kernel family). We used to
    # disable flash + memory-efficient attention AND force ORT_GRAPH_OPT_LEVEL
    # basic (so Level3 couldn't re-fuse the decomposed attention into those
    # kernels). Since v0.5.58 nan-onnxruntime PATCHES the division sites
    # (wasm/onnxruntime-sm90-mps.patch: flash/lean num_SMs clamp), so the fused
    # kernels are safe again - and much faster on long contexts / big models.
    # Default is now FUSED ON. Revert WITHOUT a release by setting
    # ENCLAVE_FUSED_ATTENTION=0 on the wasm-manager container (Tinfoil dashboard) -
    # it re-applies the conservative unfused knobs. ORT_DISABLE_MATMUL4BITS_KERNEL
    # (also from the patch) is a SEPARATE, unrelated switch for the fp16 M=1
    # GEMV corruption - production dodges that with fp32-activation models.
    if os.environ.get("ENCLAVE_FUSED_ATTENTION", "1").strip().lower() in ("0", "false", "no", "off"):
        env.setdefault("ORT_DISABLE_FLASH_ATTENTION", "1")
        env.setdefault("ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION", "1")
        env.setdefault("ORT_GRAPH_OPT_LEVEL", "basic")
    # else: leave the attention knobs unset -> ORT uses its fused kernels
    # (flash + memory-efficient, Level3 fusion) on the patched runtime.
    # whatever the probe's GPU bisect adopted (e.g. CUDA_MODULE_LOADING=EAGER
    # when lazy loading deadlocks under MPS) applies to every tenant
    env.update(_NN_PROBE.get("env") or {})
    return env


def _nn_probe_once(env: dict) -> tuple:
    """(ok, detail). Runs the cuInit/primary-ctx probe in a subprocess under a
    hard timeout - a HANG is a result here, not a failure mode. Deliberately
    NOT subprocess.run: its timeout path does kill()+wait(), and a child stuck
    in an UNINTERRUPTIBLE kernel ioctl (D-state - how GPU driver hangs look
    under CC) never reaps, blocking the whole probe forever. We poll with a
    deadline and ABANDON an unkillable child rather than wait on it."""
    try:
        proc = subprocess.Popen(["python3", "-c", _NN_PROBE_SRC], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, preexec_fn=_preexec)
    except Exception as e:                                       # noqa: BLE001
        return (False, f"probe spawn failed: {e}")
    deadline = time.time() + NN_PROBE_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            out = (proc.stdout.read() or "").strip() if proc.stdout else ""
            return (proc.returncode == 0 and out.endswith("ok"), out or f"rc={proc.returncode}")
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:                                            # noqa: BLE001
        pass
    for _ in range(10):                                          # 5s of grace to reap
        if proc.poll() is not None:
            return (False, f"HUNG >{NN_PROBE_TIMEOUT:.0f}s (killed)")
        time.sleep(0.5)
    return (False, f"HUNG >{NN_PROBE_TIMEOUT:.0f}s and UNKILLABLE (kernel-stuck GPU ioctl?) - abandoned")


def _proc_hang_dump(pid) -> str:
    """Compact thread dump of a HUNG process, readable without root from the
    same user: per-thread state + kernel wait channel. This is the ground
    truth the whole bisect exists to reach - a D-state thread's wchan names
    the kernel/driver function the hang lives in (platform's bug); an S-state
    futex points back at userspace (ours)."""
    out = []
    base = f"/proc/{pid}/task"
    try:
        maps_n = sum(1 for _ in open(f"/proc/{pid}/maps"))
    except Exception:                                            # noqa: BLE001
        maps_n = -1
    try:
        for tid in sorted(os.listdir(base), key=int):
            try:
                comm = open(f"{base}/{tid}/comm").read().strip()
                state = open(f"{base}/{tid}/stat").read().rsplit(") ", 1)[-1].split()[0]
                try:
                    wchan = open(f"{base}/{tid}/wchan").read().strip() or "0"
                except Exception:                                # noqa: BLE001
                    wchan = "?"
                out.append((state, f"{comm}:{state}:{wchan}"))
            except Exception:                                    # noqa: BLE001
                continue
    except Exception as e:                                       # noqa: BLE001
        return f"dump failed: {e}"
    # D-state threads always shown; the rest deduped by (comm prefix, wchan)
    ds = [s for st, s in out if st == "D"]
    others, seen = [], set()
    for st, s in out:
        if st == "D":
            continue
        key = s.rsplit(":", 1)[-1] + s[:4]
        if key not in seen:
            seen.add(key)
            others.append(s)
    shown = ds + others[: max(0, 14 - len(ds))]
    return f"maps={maps_n} threads({len(out)})=[" + ", ".join(shown) + "]"


def _nn_probe_e2e(env, targets=("cpu", "gpu"), timeout=None, extra_args=()) -> tuple:
    """({target: ok}, detail). The ORT layer, end to end: serve the baked-in
    nn-demo with the real tenant env and run ONE inference per target through
    it. The cuInit probe can pass while ORT's session creation still hangs (it
    exercises cudart/cublas/cuDNN and the CC data path, not just the driver
    attach), so only this stage proves a GPU deployment will actually answer.
    Each call is a FRESH wasmtime process = a fresh CUDA init. On a hang, the
    detail carries a thread dump of the wedged process (state + kernel wchan)."""
    timeout = timeout or NN_PROBE_TIMEOUT
    wasm = _fixture_wasm()
    if not wasm.is_file():
        return ({t: True for t in targets}, "e2e skipped (nn-demo.wasm not baked in)")
    port = _free_port()
    cmd = [WASMTIME, "serve", "-Scli", "-Shttp", *_p3_flags(), "-Snn", *extra_args,
           "--addr", f"{HOST_IP}:{port}", str(wasm)]
    try:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=_preexec)
    except Exception as e:                                       # noqa: BLE001
        return ({t: False for t in targets}, f"e2e spawn failed: {e}")
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not _port_open(port):
            if proc.poll() is not None:
                return ({t: False for t in targets}, f"e2e wasmtime exited rc={proc.returncode} before serving")
            time.sleep(0.2)
        if not _port_open(port):
            return ({t: False for t in targets}, "e2e serve socket never opened")
        parts, results = [], {}
        for tgt in targets:
            t0 = time.time()
            try:
                with urllib.request.urlopen(f"http://{HOST_IP}:{port}/?target={tgt}",
                                            timeout=timeout) as r:
                    body = json.loads(r.read() or b"{}")
                results[tgt] = bool(body.get("ok"))
                parts.append(f"{tgt}: {'ok' if results[tgt] else body.get('error', 'not ok')} ({time.time() - t0:.1f}s)")
            except Exception as e:                               # noqa: BLE001
                results[tgt] = False
                dump = _proc_hang_dump(proc.pid) if proc.poll() is None else f"process exited rc={proc.returncode}"
                parts.append(f"{tgt}: HUNG/failed after {time.time() - t0:.1f}s ({e.__class__.__name__}: {e}) {dump}")
        return (results, "; ".join(parts))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:                                        # noqa: BLE001
            try:
                proc.kill()
            except Exception:                                    # noqa: BLE001
                pass


# Runtime-API bisect: walks the exact CUDA calls ORT's CUDA provider makes at
# session init, journaling each STEP to a file BEFORE executing it - when one
# hangs, the journal's last line names the call. The two calls the passing
# probes never exercised are the prime suspects: cudaHostAlloc (PINNED host
# memory needs shared/unencrypted pages under TDX) and cuBLAS/cuDNN library
# init (their kernel-module upload is the heavyweight step).
_NN_RT_SRC = r"""
import ctypes, sys
log = open(sys.argv[1], "w", buffering=1)
def step(name): log.write("STEP " + name + "\n")
def ck(name, rc):
    if rc != 0:
        log.write(f"FAIL {name} rc={rc}\n"); sys.exit(2)
if len(sys.argv) > 2 and sys.argv[2] == "threads":
    # mimic wasmtime's thread pressure OUTSIDE wasmtime: ~50 threads on the
    # 16-vCPU TDX guest before any CUDA call. If the walk spins HERE too, the
    # hang is thread-count x driver x TDX - wasmtime fully exonerated.
    import threading, time as _t
    step("spawn 48 sleeper threads")
    for _ in range(48):
        threading.Thread(target=_t.sleep, args=(3600,), daemon=True).start()
step("dlopen libcudart.so.12")
try:
    rt = ctypes.CDLL("libcudart.so.12")
except OSError as e:
    log.write(f"SKIP no cudart ({e})\n"); sys.exit(3)
step("cudaSetDevice(0)");         ck("cudaSetDevice", rt.cudaSetDevice(0))
step("cudaFree(0) [ctx init]");   ck("cudaFree0", rt.cudaFree(None))
p = ctypes.c_void_p()
step("cudaMalloc 1MB [device]");  ck("cudaMalloc", rt.cudaMalloc(ctypes.byref(p), 1 << 20))
h = ctypes.c_void_p()
step("cudaHostAlloc 1MB [PINNED host - TDX shared pages]")
ck("cudaHostAlloc", rt.cudaHostAlloc(ctypes.byref(h), 1 << 20, 0))
step("cudaMemcpy pinned H2D 1MB"); ck("cudaMemcpy", rt.cudaMemcpy(p, h, 1 << 20, 1))
step("dlopen libcublas.so.12")
cb = ctypes.CDLL("libcublas.so.12")
bh = ctypes.c_void_p()
step("cublasCreate [cublas init]"); ck("cublasCreate", cb.cublasCreate_v2(ctypes.byref(bh)))
one = ctypes.c_float(1.0); zero = ctypes.c_float(0.0)
step("cublasSgemm 64x64 [kernel-module load + compute]")
ck("cublasSgemm", cb.cublasSgemm_v2(bh, 0, 0, 64, 64, 64, ctypes.byref(one),
                                    p, 64, p, 64, ctypes.byref(zero), p, 64))
step("cudaDeviceSynchronize");    ck("sync", rt.cudaDeviceSynchronize())
step("dlopen libcudnn.so.9")
dn = ctypes.CDLL("libcudnn.so.9")
dh = ctypes.c_void_p()
step("cudnnCreate [cudnn init]"); ck("cudnnCreate", dn.cudnnCreate(ctypes.byref(dh)))
log.write("ok\n")
"""


def _nn_probe_rt(env: dict, threaded=False) -> tuple:
    """(ok, detail). Runs the runtime-API bisect with a hard deadline; on a
    hang, reports the exact CUDA call it died in (journal's last STEP).
    threaded=True first spawns ~50 sleeper threads (wasmtime-like pressure)."""
    jpath = LOG_DIR / f"nn-rt-{uuid.uuid4().hex[:6]}.log"
    argv = ["python3", "-c", _NN_RT_SRC, str(jpath)] + (["threads"] if threaded else [])
    try:
        proc = subprocess.Popen(argv, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=_preexec)
    except Exception as e:                                       # noqa: BLE001
        return (False, f"rt probe spawn failed: {e}")
    deadline = time.time() + NN_PROBE_TIMEOUT
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.5)
    hung = proc.poll() is None
    if hung:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:                                        # noqa: BLE001
            pass
    try:
        lines = [l.strip() for l in jpath.read_text().splitlines() if l.strip()]
    except Exception:                                            # noqa: BLE001
        lines = []
    finally:
        try:
            jpath.unlink()
        except Exception:                                        # noqa: BLE001
            pass
    if not hung and lines and lines[-1] == "ok":
        return (True, f"ok ({len(lines) - 1} steps)")
    last = lines[-1] if lines else "(no journal)"
    return (False, (f"HUNG at '{last}' after {NN_PROBE_TIMEOUT:.0f}s" if hung
                    else f"stopped at '{last}' rc={proc.returncode}"))


def _nn_probe_gdb(env, extra_args=()) -> str:
    """Symbol-level stacks of the hang: spawn the e2e wasmtime UNDER gdb (gdb
    as parent dodges ptrace-scope), trigger one gpu load, let it wedge, SIGINT
    gdb (it stops the inferior; batch mode then runs the queued commands), and
    harvest `thread apply all bt`. The full dump goes to the manager log; the
    CUDA/ORT-relevant frames come back for the public trail."""
    if not shutil.which("gdb"):
        return "gdb not in image"
    wasm = _fixture_wasm()
    if not wasm.is_file():
        return "no nn-demo.wasm"
    port = _free_port()
    cmd = ["gdb", "--batch", "-q",
           "-ex", "set pagination off", "-ex", "set confirm off", "-ex", "run",
           "-ex", "thread apply all bt 24",
           "--args", WASMTIME, "serve", "-Scli", "-Shttp", *_p3_flags(), "-Snn", *extra_args,
           "--addr", f"{HOST_IP}:{port}", str(wasm)]
    try:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, preexec_fn=_preexec)
    except Exception as e:                                       # noqa: BLE001
        return f"gdb spawn failed: {e}"
    out = ""
    try:
        deadline = time.time() + 40
        while time.time() < deadline and not _port_open(port):
            if proc.poll() is not None:
                out, _ = proc.communicate(timeout=10)
                return f"gdb/wasmtime exited rc={proc.returncode} before serving: {(out or '')[-200:]}"
            time.sleep(0.3)
        if _port_open(port):
            threading.Thread(target=lambda: urllib.request.urlopen(
                f"http://{HOST_IP}:{port}/?target=gpu", timeout=120).read(),
                daemon=True).start()
            time.sleep(45)   # let the init wedge properly before snapping
        try:
            os.kill(proc.pid, signal.SIGINT)   # gdb only: stops the inferior, then bt runs
        except Exception:                                        # noqa: BLE001
            pass
        try:
            out, _ = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate(timeout=10)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:                                        # noqa: BLE001
            pass
    out = out or ""
    print("[nn-probe] gdb full dump (tail):\n" + out[-20000:], flush=True)
    # Dedupe by symbol (address-stripped): N identical spinner threads must not
    # crowd out the one interesting stack (v0.5.21's capture was 12 copies of
    # ORT's WorkerLoop). Wide match incl. rust `ort::`/wasmtime symbols.
    frames, seen = [], set()
    for l in out.splitlines():
        ls = l.strip()
        if not re.match(r"#\d+ ", ls):
            continue
        if not re.search(r"cuda|cublas|cudnn|onnx|ort|wasi|nn_|wasmtime", ls, re.I):
            continue
        key = re.sub(r"0x[0-9a-f]+", "", ls).split(" in ", 1)[-1]
        if key in seen:
            continue
        seen.add(key)
        frames.append(ls)
    if not frames:
        frames = [l.strip() for l in out.splitlines() if re.match(r"#\d+ ", l.strip())][:12]
    return ("frames: " + " | ".join(frames[:16])[:1100]) if frames else \
        f"no frames captured (gdb said: {out[-220:]})"


def _nn_probe_worker_control() -> tuple:
    """(ok, detail). Ask the worker manager (if present on this box) to spawn
    one MPS-capped cupy child - the platform's VALIDATED CUDA path - and tear
    it down. Purely diagnostic: discriminates 'this container/ORT is broken'
    from 'GPU compute under CC is broken node-wide'."""
    tid = "nn-probe-control"
    try:
        req = urllib.request.Request(f"{WORKER_MGR_URL}/tenants",
                                     data=json.dumps({"id": tid, "gpuShare": 0.01}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=100) as r:
            body = json.loads(r.read() or b"{}")
        ok = body.get("status") == "running"
        detail = (f"ok (sm_granted={body.get('sm_granted')}, device={body.get('device')})" if ok
                  else f"{body.get('status')}: {body.get('error') or 'no error detail'}")
    except Exception as e:                                       # noqa: BLE001
        return (False, f"unreachable/failed ({e.__class__.__name__}: {e})")
    finally:
        try:
            req = urllib.request.Request(f"{WORKER_MGR_URL}/tenants/{tid}", method="DELETE")
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:                                        # noqa: BLE001
            pass
    return (ok, detail)


def _nn_probe_loop():
    """Crash guard: a probe that dies must VERDICT, not stay 'probing' forever
    (every GPU deploy 503s on that state)."""
    try:
        _nn_probe_run()
    except Exception:                                            # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        _NN_PROBE.update(state="failed", stage="crashed",
                         detail="probe crashed: " + " | ".join(tb.strip().splitlines()[-3:]))
        print("[nn-probe] CRASHED:\n" + tb, flush=True)


def _nn_probe_run():
    """Boot-time GPU readiness bisect. Writes _NN_PROBE; launches gate on it.
    Layer 1: raw driver (cuInit + primary-ctx retain = the MPS attach point),
    bisecting the env. Layer 2: ORT end to end through the real runtime.
    Progress is published LIVE (stage + growing detail) so an outside observer
    can always tell a slow probe from a stuck one.
    WASM_NN_PROBE=0 skips everything and declares ok (operator escape hatch)."""
    if os.environ.get("WASM_NN_PROBE", "1").lower() in ("0", "false", "off"):
        _NN_PROBE.update(state="ok", mode="full", stage="done", detail="probe skipped (WASM_NN_PROBE=0)")
        print("[nn-probe] skipped by WASM_NN_PROBE=0 - GPU launches ungated", flush=True)
        return
    share = 0.01   # smallest grain; the probe only needs A context, not capacity
    steps = [("full",  _nn_tenant_env(share, pinned=True),
              "tenant env (SM cap + pinned VRAM limit)"),
             ("nopin", _nn_tenant_env(share, pinned=False),
              "without CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"),
             ("nomps", {k: v for k, v in os.environ.items() if not k.startswith("CUDA_MPS")},
              "without any MPS env (diagnostic only - tenants NEVER run uncapped)")]
    history = []
    mode = None
    # up to 3 rounds of the full env first: rides out MPS-daemon boot races the
    # same way worker.py's children retry their CUDA init.
    def note(entry):   # publish progress LIVE: detail grows as the bisect runs
        history.append(entry)
        _NN_PROBE["detail"] = "; ".join(history)

    for attempt in range(3):
        _NN_PROBE["attempts"] = attempt + 1
        _NN_PROBE["stage"] = f"cuInit full env #{attempt + 1}"
        ok, detail = _nn_probe_once(steps[0][1])
        note(f"full#{attempt + 1}: {detail}")
        print(f"[nn-probe] full env attempt {attempt + 1}: {'ok' if ok else detail}", flush=True)
        if ok:
            mode = "full"
            break
        time.sleep(5)
    if mode is None:
        for m, env, label in steps[1:]:
            _NN_PROBE["stage"] = f"cuInit {m}"
            ok, detail = _nn_probe_once(env)
            note(f"{m}: {detail}")
            print(f"[nn-probe] {label}: {'ok' if ok else detail}", flush=True)
            if ok and m == "nopin":
                # the pinned-VRAM var is the poison: run tenants SM-capped only
                # (VRAM stays admission-accounted by the supervisor's allocator).
                print("[nn-probe] WARNING: CUDA_MPS_PINNED_DEVICE_MEM_LIMIT hangs/fails CUDA "
                      "init on this node - GPU tenants run with the SM cap only.", flush=True)
                mode = "nopin"
                break
            if ok and m == "nomps":
                # CUDA works, MPS attach doesn't: refusing is the honest move -
                # uncapped tenants would break the share product.
                _NN_PROBE.update(state="failed", mode=None, stage="done",
                                 detail="MPS attach breaks CUDA init in this container (bare CUDA works). "
                                        + "; ".join(history))
                return
        if mode is None:
            _NN_PROBE.update(state="failed", mode=None, stage="done", detail="; ".join(history))
            return
    # driver layer passed - walk the runtime-API calls ORT will make, so a
    # later e2e hang is pre-attributed to an exact CUDA call (diagnostic; the
    # e2e stage below remains the launch gate)
    base = _nn_tenant_env(share, pinned=(mode == "full"))
    _NN_PROBE["stage"] = "runtime-API bisect (cudaMalloc/pinned/cublas/cudnn)"
    rt_ok, rt_detail = _nn_probe_rt(base)
    note(f"rtapi: {rt_detail}")
    print(f"[nn-probe] runtime-API bisect: {rt_detail}", flush=True)
    _NN_PROBE["stage"] = f"ORT e2e base ({mode}): cpu then gpu, {NN_PROBE_TIMEOUT:.0f}s each"
    res, e2e_detail = _nn_probe_e2e(base)
    note(f"e2e[{mode}]: {e2e_detail}")
    print(f"[nn-probe] ORT end-to-end ({mode}): {e2e_detail}", flush=True)
    if all(res.values()):
        _NN_PROBE.update(state="ok", mode=mode, stage="done", detail="; ".join(history))
        return
    if not res.get("cpu", False):
        # ORT can't even run the CPU provider here - nothing GPU-specific to bisect
        _NN_PROBE.update(state="failed", mode=mode, stage="done",
                         detail="ORT fails on the CPU provider itself - " + "; ".join(history))
        return
    # GPU-only failure. The v0.5.15-0.5.20 exoneration ladder cleared MPS, the
    # pinned limit, lazy loading, slow-CC-load, pooling VA, signal traps, and
    # CoW init individually (kryptos hangs are a userspace SPIN during ORT's
    # CUDA init: one R-state thread, no D-state, driver event thread healthy).
    # This path now EXTRACTS rather than guesses:
    #   rtapi-threaded - the plain-process CUDA walk under wasmtime-like
    #                    thread pressure; a spin HERE fully exonerates wasmtime
    #   gdb            - symbol-level stacks of the actual hang
    #   bare           - the one remaining heal candidate (all flags off)
    _NN_PROBE["stage"] = "rtapi under thread pressure (48 sleepers)"
    tok, tdetail = _nn_probe_rt(base, threaded=True)
    note(f"rtapi-threaded: {tdetail}")
    print(f"[nn-probe] rtapi threaded: {tdetail}", flush=True)
    _NN_PROBE["stage"] = "gdb stack capture of the hang (~2.5 min)"
    gdetail = _nn_probe_gdb(base)
    note(f"gdb: {gdetail}")
    print(f"[nn-probe] gdb frames: {gdetail}", flush=True)
    BARE = ["-O", "pooling-allocator=n", "-O", "signals-based-traps=n", "-O", "memory-init-cow=n"]
    _NN_PROBE["stage"] = "ORT e2e gpu variant 'bare' (all wasmtime flags off)"
    vres, vdetail = _nn_probe_e2e(base, targets=("gpu",), extra_args=BARE)
    note(f"bare: {vdetail}")
    print(f"[nn-probe] gpu variant bare: {vdetail}", flush=True)
    if vres.get("gpu"):
        _NN_PROBE.update(args=BARE)
        note("ADOPTED bare: nn tenants run with pooling, signal traps, and CoW init all off")
        _NN_PROBE.update(state="ok", stage="done", detail="; ".join(history))
        return
    # Every tenant-shaped variant hung. Endgame diagnostics (adopt NOTHING -
    # both would compromise the share caps - but name the guilty layer):
    #   control - the worker container's VALIDATED cupy-under-MPS path
    #   nomps   - ORT with no MPS env at all
    _NN_PROBE["stage"] = "control: worker manager cupy-under-MPS tenant"
    ctl_ok, ctl_detail = _nn_probe_worker_control()
    note(f"control[worker-cupy]: {ctl_detail}")
    print(f"[nn-probe] control worker-cupy: {ctl_detail}", flush=True)
    _NN_PROBE["stage"] = "ORT e2e gpu without MPS (diagnostic only)"
    nomps_env = {k: v for k, v in base.items() if not k.startswith("CUDA_MPS")}
    nomps_res, nomps_detail = _nn_probe_e2e(nomps_env, targets=("gpu",))
    note(f"nomps[diagnostic]: {nomps_detail}")
    print(f"[nn-probe] gpu without MPS (diagnostic): {nomps_detail}", flush=True)
    nomps_ok = bool(nomps_res.get("gpu"))
    if ctl_ok and nomps_ok:
        verdict = ("the MPS+ORT interaction in THIS container is the fault: cupy-under-MPS works "
                   "(worker) and ORT works here without MPS, but ORT under MPS hangs")
    elif ctl_ok:
        verdict = ("this container's GPU compute path is the fault: the worker's cupy-under-MPS "
                   "control works, but ORT hangs here with AND without MPS")
    elif nomps_ok:
        verdict = ("MPS is broken node-wide for real compute init: even the validated worker path "
                   "fails, while ORT works without MPS")
    else:
        verdict = ("GPU compute init is broken NODE-WIDE under CC (the validated worker cupy path "
                   "and ORT, with and without MPS, all fail) - escalate to the platform/driver level")
    _NN_PROBE.update(state="failed", mode=mode, stage="done",
                     detail=f"driver layer ok ({mode}); ORT CUDA hung in every tenant variant. "
                            f"VERDICT: {verdict}. Trail: " + "; ".join(history))


def _resolve_wasm(ref: str) -> pathlib.Path:
    """Resolve an app reference to a .wasm path: a catalog id (baked-in, attested),
    `ipfs://<cid>` (fetched + verified against the CID), or a path INSIDE APPS_DIR."""
    if ref.startswith("ipfs://"):
        cid = ref[len("ipfs://"):].split("/", 1)[0].split("?", 1)[0].strip()
        return _resolve_cid(cid)
    cat = _load_catalog()
    if ref in cat:
        p = (APPS_DIR / cat[ref]["file"]).resolve()
    else:
        p = pathlib.Path(ref).resolve()
    # containment: only allow paths under APPS_DIR
    if APPS_DIR.resolve() not in p.parents and p != APPS_DIR.resolve():
        raise ValueError(f"app '{ref}' is not in the catalog and not under {APPS_DIR}")
    if not p.is_file():
        raise ValueError(f"wasm module not found for app '{ref}' ({p})")
    return p


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST_IP, 0))
        return s.getsockname()[1]


def _port_open(port: int, timeout=0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((HOST_IP, port))
            return True
        except OSError:
            return False


def _used_cpu_share() -> float:
    """Admission is by cpuShare only: this manager owns the node's vCPU+RAM pool.
    GPU shares are accounted by the supervisor's card allocator, not here."""
    return round(sum(r["cpuShare"] for r in _apps.values() if r["status"] in ("starting", "running")), 4)


def _rec_ram_mb(rec) -> int:
    """Worst-case CVM RAM a running tenant can pin: guest linear memory + its
    ramdisk caps (/data + each encrypted volume's plaintext ceiling) + any model
    weights the host PRELOADED for it into node RAM. All of it lives in the
    CVM's RAM, so the SUM across tenants oversubscribing node RAM is the OOM the
    storage audit only catches AFTER the fact. Used only when
    WASM_ACCOUNT_STORAGE_RAM is on.

    The preloaded weights (CPU nn only - on a GPU node they land in VRAM and the
    VRAM ledger owns them) are counted at FULL size even though they are
    file-backed page cache the kernel could in principle reclaim. Reclaiming
    them is not free capacity: it is the serving tenant page-thrashing its model
    back in on every token. A box holding a 19.7 GiB model has ~19.7 GiB less to
    sell, and until this term existed it advertised those bytes as available -
    28 GB 'free' on a node with 5 GB actually free (2026-07-27, metal0)."""
    mb = int(rec.get("mem_mb") or 0) + int(rec.get("storageMb") or 0) + int(rec.get("nnResidentMb") or 0)
    enc = rec.get("_enc")
    if enc:
        mb += sum(int(v["spec"].get("maxMb") or 0) for v in enc["vols"].values())
    return mb


def _nn_resident_bytes(exclude: str | None = None) -> int:
    """Model weights other live tenants already hold resident in node RAM.
    The per-deployment ggml budget subtracts this: the budget bounds what ONE
    tenant may map, but node RAM is shared, so without the cross-tenant term two
    deployments each pass a 23 GiB check and together thrash a 29 GB box."""
    return sum(int(r.get("nnResidentMb") or 0) for vid, r in _apps.items()
               if vid != exclude and r["status"] in ("starting", "running")) * (1 << 20)


def _volumes_public() -> list:
    """Attached model volumes for advertisement (no host paths leaked)."""
    return [{"name": v["name"], "bytes": v["bytes"], "onnx": v["onnx"], "gguf": v["gguf"],
             "files": v["files"]}
            for v in sorted(_model_volumes().values(), key=lambda x: x["name"])]


def _nn_available() -> bool:
    """Does this node offer wasi-nn to tenants?

    GPU node: only once the boot probe says the card + MPS path is healthy —
    advertising nn on a box whose CUDA init is broken sells deployments that
    die at load. CPU-only node: there is no card to probe, and the ggml backend
    runs on the cores a tenant already buys, so the interface is available
    whenever it is enabled. What a CPU node can actually PRELOAD is a per
    deployment answer (its RAM budget, see _build_cmd), reported per tenant in
    nnPreloads/nnSkipped rather than as a node-wide claim."""
    if not NN_ENABLED:
        return False
    if MOCK:
        return True
    return _NN_PROBE["state"] == "ok" if NODE_HAS_GPU else True


def _ram_budget() -> dict | None:
    """The RAM-reservation ledger (WASM_ACCOUNT_STORAGE_RAM): worst-case MB
    committed by starting/running tenants vs the node ceiling - the SAME sum
    the admission check at provision time enforces. None when the accounting
    is off."""
    if not ACCOUNT_STORAGE_RAM:
        return None
    committed = sum(_rec_ram_mb(r) for r in _apps.values()
                    if r["status"] in ("starting", "running"))
    budget = int(NODE_RAM_GB * 1024 * RAM_ACCT_HEADROOM)
    return {"ramBudgetMb": budget, "ramCommittedMb": committed,
            "ramFreeMb": max(0, budget - committed),
            # broken out because it is the term that surprises: a box can be
            # 85% committed with every tenant idle, purely from resident weights
            "ramNnResidentMb": _nn_resident_bytes() // (1 << 20)}


def _lease_public() -> dict:
    """Dead-man lease state, for /capacity -> /availability. `armed` false means
    no heartbeat has ever arrived and nothing can be reaped on lease grounds -
    which is the honest thing to say about a box where the switch is not live
    yet, rather than letting it look protected."""
    now = time.time()
    return {"armed": _lease_last_beat is not None,
            "enforcing": _lease_last_beat is not None and (now - _lease_last_beat) <= TENANT_LEASE_SILENCE,
            "lastBeatAgoSec": None if _lease_last_beat is None else round(now - _lease_last_beat, 1),
            "ttlSec": TENANT_LEASE_TTL}


def _rec_vram_gb(rec) -> float:
    """Worst-case device memory a tenant can pin: its sold slice (the MPS
    pinned cap = gpuShare x card VRAM) plus the per-process CUDA context
    overhead the supervisor's allocator prices identically (CTX_OVERHEAD_GB).
    0 for CPU tenants. Used only when WASM_ACCOUNT_VRAM is on."""
    share = float(rec.get("gpuShare") or 0)
    return share * GPU_VRAM_GB + VRAM_CTX_OVERHEAD_GB if share > 0 else 0.0


_dev_mem_cache = {"at": 0.0, "val": None}


def _gpu_dev_mem() -> dict | None:
    """The device's OWN memory count: nvidia-smi memory.free/memory.total
    summed across cards, in GB. Cached VRAM_DEV_TTL_S - /availability polls
    land every few seconds and admission must not fork a probe per launch.
    None on CPU nodes, in mock mode, and on probe failure."""
    if not NODE_HAS_GPU or MOCK:
        return None
    now = time.time()
    if now - _dev_mem_cache["at"] < VRAM_DEV_TTL_S:
        return _dev_mem_cache["val"]
    val = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        rows = [ln.split(",") for ln in r.stdout.strip().splitlines() if "," in ln]
        free = sum(float(a) for a, _ in rows) / 1024
        total = sum(float(b) for _, b in rows) / 1024
        if rows and 0 <= free <= total:
            val = {"vramDevFreeGb": round(free, 2), "vramDevTotalGb": round(total, 2)}
    except Exception:                                    # noqa: BLE001
        val = None
    _dev_mem_cache.update(at=now, val=val)
    return val


def _gpu_dev_procs() -> list:
    """Per-PID device attribution (nvidia-smi compute-apps): what is ACTUALLY
    holding the card, whether or not any ledger sold it. Control-plane only.
    This is the table the 2026-08-18 incident could not produce without a
    shell nobody has on a CVM. usedMiB is None where the driver withholds it
    (MPS/CC report [N/A] for some clients - the row still names the holder)."""
    if not NODE_HAS_GPU or MOCK:
        return []
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-compute-apps=pid,process_name,used_gpu_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        procs = []
        for ln in r.stdout.strip().splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    used = int(float(parts[-1]))
                except ValueError:
                    used = None
                procs.append({"pid": int(parts[0]),
                              "name": ",".join(parts[1:-1]), "usedMiB": used})
        return procs
    except Exception:                                    # noqa: BLE001
        return []


def _request_mps_bounce(reason: str) -> bool:
    """Order the mps-control container to bounce the MPS stack: write the
    request file it polls off the shared pipe-dir volume (atomically - the
    consumer must never read a half-written order). The answer lands beside
    it in enclave-bounce-result within seconds. Costly by design: a bounce
    kills every live GPU tenant's CUDA context (their watchdogs abort, the
    supervisor respawns them onto the fresh daemon); the daemon's own
    cooldown refuses back-to-back orders."""
    try:
        d = pathlib.Path(MPS_PIPE_DIR)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / ".enclave-bounce-request.tmp"
        tmp.write_text(json.dumps({"at": time.time(), "reason": reason[:400]}))
        tmp.replace(d / "enclave-bounce-request")
        print(f"[mps] bounce ordered: {reason}", flush=True)
        return True
    except OSError as e:
        print(f"[mps] bounce order FAILED to write: {e}", flush=True)
        return False


def _mps_bounce_result() -> str | None:
    """The daemon's answer to the last consumed order ('<ts> ok' /
    'refused-cooldown' / 'failed-daemon-down'), or None before any."""
    try:
        txt = (pathlib.Path(MPS_PIPE_DIR) / "enclave-bounce-result").read_text().strip()
        return txt or None
    except OSError:
        return None


def _boot_bounce_check():
    """One look at the device as this process starts: the tenant table is
    empty, so anything the card holds beyond VRAM_RESERVE_GB belongs to a
    generation that no longer exists - the residue an in-place container
    update strands (2026-08-18: ~104 GiB on kryptos, reclaimed only by a
    full CVM restart because nothing ordered a bounce). Order one now,
    BEFORE the first launch admits against memory the card cannot deliver.
    Documented caveat: a bounce also kills PTX-worker tenants that survived
    a wasm-only update - WASM_MPS_BOOT_BOUNCE=0 opts out for a fleet that
    runs long-lived worker tenants."""
    dev = _gpu_dev_mem()
    if not dev:
        return
    with _lock:
        if _apps:
            return
    held = dev["vramDevTotalGb"] - dev["vramDevFreeGb"] - VRAM_RESERVE_GB
    if held <= MPS_BOOT_BOUNCE_GB:
        return
    _request_mps_bounce(f"boot: {held:.1f} GB held with an empty tenant table "
                        f"(orphaned generation from an in-place update?)")


_diverge_last = {"at": 0.0}


def _warn_divergence(div_gb: float):
    """A large positive divergence = the device holds memory this ledger never
    sold: THE signature of an orphaned tenant generation or a wedged MPS
    server (2026-08-18). Logged with the per-PID table, rate-limited; the
    admission gate is the enforcement - this line is how an operator finds
    out before a tenant does."""
    if div_gb <= VRAM_DIVERGE_WARN_GB:
        return
    now = time.time()
    if now - _diverge_last["at"] < 300:
        return
    _diverge_last["at"] = now
    print(f"[vram] device/ledger divergence: ledger-free exceeds device-free by "
          f"{div_gb:.1f} GB (warn threshold {VRAM_DIVERGE_WARN_GB} GB) - something "
          f"outside the tenant table is holding the card; compute-apps: "
          f"{json.dumps(_gpu_dev_procs())}", flush=True)


def _vram_budget() -> dict | None:
    """The VRAM-reservation ledger (WASM_ACCOUNT_VRAM): worst-case GB pinned
    by starting/running GPU tenants vs the node's device memory - the SAME
    sum the admission check at create time enforces. None when the accounting
    is off or the node has no GPU. Carries the device-measured count beside
    the arithmetic when the card answers; vramDivergenceGb > 0 means the card
    holds memory this ledger never sold."""
    if not (ACCOUNT_VRAM and NODE_HAS_GPU):
        return None
    committed = round(sum(_rec_vram_gb(r) for r in _apps.values()
                          if r["status"] in ("starting", "running")), 2)
    budget = round(GPU_VRAM_GB * GPU_CARDS - VRAM_RESERVE_GB, 2)
    out = {"vramBudgetGb": budget, "vramCommittedGb": committed,
           "vramFreeGb": round(max(0.0, budget - committed), 2)}
    dev = _gpu_dev_mem()
    if dev:
        out.update(dev)
        out["vramDivergenceGb"] = round(out["vramFreeGb"] - dev["vramDevFreeGb"], 2)
        _warn_divergence(out["vramDivergenceGb"])
    return out


def _capacity() -> dict:
    used = _used_cpu_share()
    free = round(max(0.0, 1.0 - used), 4)
    cap = {"cpuShareFree": free, "usedCpuShare": used,
           "maxShare": free, "usedShare": used,   # deprecated aliases (one release)
           "vcpusFree": round(NODE_VCPUS * free, 2),
           "ramGbFree": round(NODE_RAM_GB * free, 2),
           "apps": len(_apps)}
    # Two ledgers govern admission: the cpuShare pool AND (when accounting is
    # on) the RAM-reservation budget. ADVERTISE whichever is tighter, each as
    # a fraction of its own pool - otherwise the dashboard/availability/claim
    # routing read 30%+ share free while the next 911 MB app is refused
    # (2026-07-19: 60-app fleet hit the RAM ceiling at cpuShareFree 0.34).
    # The raw share ledger stays visible as sharePoolFree.
    ram = _ram_budget()
    if ram:
        ram_free_frac = round(ram["ramFreeMb"] / ram["ramBudgetMb"], 4) if ram["ramBudgetMb"] else 0.0
        eff = min(free, ram_free_frac)
        cap.update(ram)
        cap["sharePoolFree"] = free
        cap["cpuShareFree"] = cap["maxShare"] = eff
        cap["usedCpuShare"] = cap["usedShare"] = round(1.0 - eff, 4)
        cap["vcpusFree"] = round(NODE_VCPUS * eff, 2)
        cap["ramGbFree"] = round(NODE_RAM_GB * eff, 2)
    # VRAM ledger: gpuShareFree here is the largest additional single-card
    # slice THIS ledger still admits (net of the per-slice context overhead);
    # the supervisor folds min(its card allocator, this) into /availability,
    # so a physical-vs-planned divergence surfaces as reduced capacity
    # instead of a claim that fails at provision time. The DEVICE's own count
    # bounds it from below for the same reason (2026-08-18): a card holding
    # orphaned memory must advertise what it can actually take, not what the
    # arithmetic wishes it could.
    vram = _vram_budget()
    if vram:
        cap.update(vram)
        eff_gb = vram["vramFreeGb"]
        if VRAM_DEV_GATE and vram.get("vramDevFreeGb") is not None:
            eff_gb = min(eff_gb, vram["vramDevFreeGb"])
        slice_gb = eff_gb - VRAM_CTX_OVERHEAD_GB
        cap["gpuShareFree"] = round(min(1.0, max(0.0, slice_gb / GPU_VRAM_GB)) if GPU_VRAM_GB else 0.0, 4)
    cap["tenantLease"] = _lease_public()
    return cap


# CPU noisy-neighbour control (cgroup v2). Two knobs with DELIBERATELY
# different defaults, because they are not the same kind of risk:
#
#   cpu.weight  ON BY DEFAULT, proportional to the tenant's PURCHASED cpuShare.
#               A weight does NOT cap anything: it only divides CONTENDED CPU
#               between tenants, so an app still bursts to every idle core.
#               What it buys is the invariant the fleet actually sells — under
#               contention you get the share you paid for, and no neighbour can
#               take it from you. That mattered less when a tenant was one
#               single-threaded wasm process; it is load-bearing the moment a
#               guest can run on many cores at once (concurrency today via
#               `serve`'s per-request parallelism, real parallelism when the
#               engine grows it — see docs/wasm-parallelism.md). Turning this
#               on later, AFTER parallel guests exist, would be closing the
#               door behind the horse.
#   cpu.max     STILL OPT-IN, default OFF. A HARD ceiling throttles a bursty
#               app even on an otherwise idle node, which is a real regression
#               for legitimate workloads. Only an operator should ask for it.
#
# Operator overrides:
#   WASM_CPU_WEIGHT=<1..10000>  pin EVERY tenant to this fixed weight instead of
#                               the proportional default (escape hatch / A-B).
#   WASM_CPU_WEIGHT=0           disable the weight entirely (pre-2026-08 behaviour).
#   WASM_CPU_MAX_PCT=<1..100>   HARD ceiling: at most this % of the whole node's
#                               vCPUs (cgroup cpu.max). Can throttle bursty apps
#                               — use deliberately.
# Both need the cpu controller available in a cgroup-v2 subtree the manager can
# write. We SELF-CONFIGURE, so this is NOT per-enclave or per-boot work: on the
# first launch we move the manager's own processes into a leaf child (so the
# manager's cgroup becomes an inner node, satisfying the cgroup-v2 "no internal
# processes" rule), enable `+cpu` on its subtree_control, and nest per-tenant
# cgroups under enclave-tenants/. The ONLY external requirement is that the CVM
# launched the manager with the cpu controller DELEGATED to its cgroup (systemd
# `Delegate=cpu`, or the container runtime's cgroup delegation) — a one-time
# image setting inherited by every enclave. WASM_CGROUP_PARENT is an optional
# override: point it at a ready-made cpu-enabled subtree to skip self-config. If
# cpu isn't delegated, or placement fails for ANY reason, we WARN and leave the
# tenant uncapped — never fail a launch over it.
_CPU_WEIGHT        = os.environ.get("WASM_CPU_WEIGHT", "").strip()
_CPU_MAX_PCT       = os.environ.get("WASM_CPU_MAX_PCT", "").strip()
_CGROUP_PARENT_ENV = os.environ.get("WASM_CGROUP_PARENT", "").strip()
# "0" is the explicit opt-OUT of the proportional weight; anything else
# non-empty pins a fixed weight; empty = the proportional default.
_CPU_WEIGHT_OFF    = _CPU_WEIGHT in ("0", "off", "no", "false")
# cgroup work now runs by default (the weight), not only when a knob is set.
_CPU_CGROUP_ON     = bool(_CPU_MAX_PCT) or not _CPU_WEIGHT_OFF


def _cpu_weight_for(cpu_share: float) -> int:
    """cgroup-v2 cpu.weight for a tenant holding `cpu_share` of the node.

    The whole node is 10000, so a share maps straight onto it: two tenants at
    0.25 and 0.75 divide a contended node 25/75, which is exactly what they
    bought. Clamped into cgroup's legal 1..10000. A share of 0 (direct callers
    that never set one) falls back to cgroup's own default of 100 rather than
    weight 1 — an unpriced tenant should be ordinary, not starved."""
    if _CPU_WEIGHT and not _CPU_WEIGHT_OFF:
        try:
            return max(1, min(10000, int(_CPU_WEIGHT)))
        except ValueError:
            return 100
    if cpu_share <= 0:
        return 100
    return max(1, min(10000, round(cpu_share * 10000)))


def _available_parallelism_for(cpu_share: float) -> int:
    """How many threads a tenant holding `cpu_share` can ACTUALLY run at once.

    Handed to the tenant's wasmtime as ENCLAVE_AVAILABLE_PARALLELISM, where it
    becomes the answer to the shared-everything-threads
    `thread.available_parallelism` intrinsic AND the ceiling on how many SET
    worker threads that process may have live (docs/wasm-parallelism.md).

    The honest answer is the tenant's SLICE, not the node's core count. A guest
    that sizes a pool from a 32-core box while holding 0.25 just builds 32
    threads to contend over 8 cores' worth of cgroup weight — and, worse, OS
    threads are a NODE-WIDE kernel resource (threads-max, pid space) that
    cpu.weight does not bound, so an over-sized pool is a cost the tenant's
    neighbours pay too.

    Rounds UP and floors at 1: a guest reading 0 typically divides by it, and a
    small-share tenant should still be able to express a pool. Unpriced callers
    (share 0) get 1 — ordinary sequential behaviour, matching how
    `_cpu_weight_for` treats them as ordinary rather than starved."""
    if cpu_share <= 0:
        return 1
    return max(1, min(NODE_VCPUS, math.ceil(cpu_share * NODE_VCPUS)))


def _nn_threads_for(enclave_config, cpu_share: float):
    """Optional ggml compute-thread limit, bounded by this tenant's CPU share.

    Shielded GPU inference also runs background mask-refill threads. A smaller
    compute pool can leave those threads time to prepare the next GPU call.
    Absent/invalid configuration preserves the engine's existing default.
    """
    requested = _nn_cfg_int(enclave_config, "nnThreads", 1, 512)
    return min(requested, _available_parallelism_for(cpu_share)) if requested is not None else None


_cpu_cgroup_parent = None      # resolved lazily on first launch; False once known-unavailable


def _cpu_cgroup_base():
    """Resolve (once) a writable, cpu-enabled cgroup-v2 parent to nest tenants
    under. Returns a pathlib.Path or None. Never raises."""
    global _cpu_cgroup_parent
    if _cpu_cgroup_parent is not None:
        return _cpu_cgroup_parent or None
    _cpu_cgroup_parent = False
    try:
        if _CGROUP_PARENT_ENV:
            base = pathlib.Path(_CGROUP_PARENT_ENV)
            if base.is_dir():
                _cpu_cgroup_parent = base
            else:
                print(f"[cpu] WASM_CGROUP_PARENT {base} is not a directory — CPU limits off", flush=True)
            return _cpu_cgroup_parent or None
        rel = ""
        for l in pathlib.Path("/proc/self/cgroup").read_text().splitlines():
            if l.startswith("0::"):                      # cgroup v2 line: "0::/path"
                rel = l[3:]
                break
        mgr = pathlib.Path("/sys/fs/cgroup") / rel.lstrip("/")
        ctrl = mgr / "cgroup.controllers"
        if not ctrl.exists():
            print("[cpu] cgroup v2 not found under the manager's cgroup — CPU limits off", flush=True)
            return None
        if "cpu" not in ctrl.read_text().split():
            print("[cpu] cpu controller not delegated to the manager's cgroup — launch the manager "
                  "with cgroup cpu delegation (systemd Delegate=cpu) or set WASM_CGROUP_PARENT; "
                  "CPU limits off", flush=True)
            return None
        base = mgr / "enclave-tenants"
        base.mkdir(exist_ok=True)
        sub = mgr / "cgroup.subtree_control"
        try:
            sub.write_text("+cpu")                       # so children get cpu.* files
        except OSError:
            # cgroup-v2 "no internal processes" rule: mgr can't hand a controller
            # to its children while it directly holds processes. Move everything
            # in mgr into a leaf child so mgr becomes an inner node, then retry.
            leaf = mgr / "mgr"
            leaf.mkdir(exist_ok=True)
            try:
                for pid in (mgr / "cgroup.procs").read_text().split():
                    try:
                        (leaf / "cgroup.procs").write_text(pid)   # one PID per write in v2
                    except OSError:
                        pass
                sub.write_text("+cpu")
            except OSError as e:
                print(f"[cpu] could not enable cpu controller ({e}); set WASM_CGROUP_PARENT to a "
                      f"cpu-enabled subtree — CPU limits off", flush=True)
                return None
        _cpu_cgroup_parent = base
    except Exception as e:                                        # noqa: BLE001
        print(f"[cpu] cgroup setup failed ({e}) — CPU limits off", flush=True)
        _cpu_cgroup_parent = False
    return _cpu_cgroup_parent or None


def _apply_cpu_cgroup(vid: str, pid: int, cpu_share: float = 0.0):
    """Best-effort: move `pid` (a setsid group leader) into a per-tenant cgroup
    and set cpu.weight (proportional to `cpu_share`, on by default) / cpu.max
    (opt-in). Never raises — a tenant that cannot be placed runs unweighted
    rather than not at all. Returns the cgroup dir (for teardown) or None."""
    if not _CPU_CGROUP_ON:
        return None
    base = _cpu_cgroup_base()
    if base is None:
        return None
    cg = base / vid
    try:
        cg.mkdir(exist_ok=True)
        if not _CPU_WEIGHT_OFF:
            try:
                (cg / "cpu.weight").write_text(str(_cpu_weight_for(cpu_share)))
            except (ValueError, OSError) as e:
                print(f"[cpu] {vid}: cpu.weight not applied: {e}", flush=True)
        if _CPU_MAX_PCT:
            try:
                pct = max(1, min(100, int(_CPU_MAX_PCT)))
                period = 100000
                quota = max(1000, int(period * NODE_VCPUS * pct / 100))
                (cg / "cpu.max").write_text(f"{quota} {period}")
            except (ValueError, OSError) as e:
                print(f"[cpu] {vid}: cpu.max not applied: {e}", flush=True)
        (cg / "cgroup.procs").write_text(str(pid))   # moves the whole process group
        return cg
    except OSError as e:
        print(f"[cpu] {vid}: cgroup placement failed ({e}) — tenant runs uncapped", flush=True)
        try:
            cg.rmdir()
        except OSError:
            pass
        return None


def _preexec():
    """preexec: put the app in its own session so teardown can kill the whole
    group cleanly, and cap open files.

    We deliberately do NOT cap RLIMIT_AS. `wasmtime` reserves an enormous
    *virtual* address space (multi-TiB PROT_NONE guard/pooling regions) for fast
    linear-memory bounds-checking while touching almost no physical RAM, and on a
    many-core host it also reserves a worker-thread stack per CPU. Any RLIMIT_AS
    small enough to bound real memory instead makes those reservations fail,
    killing the runtime at startup (the "memory allocation of N bytes failed"
    abort). The guest's real memory is bounded on its linear memory via
    `wasmtime -W max-memory-size` in launch(); that is the only memory a tenant
    can grow, so it is the meaningful per-app cap."""
    os.setsid()
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
    except (ValueError, OSError):
        pass


# ---- lifecycle ------------------------------------------------------------- #
def _parse_ports(entries):
    """Parse a firewall config (list of 'http' | 'http:N' | 'tcp:N' | 'udp:N').

    Empty / just 'http' -> classic serve mode: `wasmtime serve` on a manager-
    assigned port, no wasi:sockets, the sandbox we've always had.
    Anything else -> run mode: the app is a long-running command component that
    binds its DECLARED ports itself via wasi:sockets ('http:N' = it serves HTTP
    on N and the supervisor proxies /x/:id there)."""
    http_port, tcp, udp, norm = None, set(), set(), []
    for e in entries or []:
        s = str(e).strip().lower()
        if not s or s == "http":
            continue
        m = re.fullmatch(r"(http|tcp|udp):(\d{1,5})", s)
        if not m:
            raise ValueError(f"bad port spec '{e}' (use http[:N] | tcp:N | udp:N)")
        n = int(m.group(2))
        if not (PORT_MIN_DECL <= n <= PORT_MAX_DECL) or n in RESERVED_PORTS:
            raise ValueError(f"port {n} not allowed (labels are {PORT_MIN_DECL}-{PORT_MAX_DECL}, "
                             f"excluding {sorted(RESERVED_PORTS)})")
        if m.group(1) == "http":
            if http_port is not None:
                raise ValueError("only one http:N entry allowed")
            http_port = n
        elif m.group(1) == "tcp":
            tcp.add(n)
        else:
            udp.add(n)
        norm.append(f"{m.group(1)}:{n}")
    declared = tcp | udp | ({http_port} if http_port else set())
    return {"serve": not declared, "http": http_port, "tcp": tcp, "udp": udp,
            "declared": declared, "norm": norm}


def _port_free(p: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST_IP, p))
            return True
        except OSError:
            return False


def _alloc_ports(pspec) -> dict:
    """Map each LOGICAL port entry ('tcp:5432') to an ACTUAL loopback port.

    Declared ports are the app's stable interface (what the bridge URL and
    ENCLAVE_PORTS reference); the actual bind is per-deployment, so two tenants can
    both run "the 5432 app" at the same time with no conflict — the URL routes
    by deployment id, never by the raw port. We prefer the logical number when
    it's free (apps that hardcode their port keep working while they can);
    otherwise the OS assigns a free one. ENCLAVE_PORTS always carries the truth."""
    with _lock:
        claimed = set()
        for r in _apps.values():
            if r["status"] in ("starting", "running"):
                claimed |= set((r.get("portMap") or {}).values())
    out = {}
    for entry in pspec["norm"]:
        logical = int(entry.split(":")[1])
        # privileged labels (<1024, e.g. udp:53) are never bound literally
        keep = logical > PRIV_PORT_MAX and logical not in claimed and _port_free(logical)
        actual = logical if keep else _free_port()
        claimed.add(actual)
        out[entry] = actual
    return out


# Does THIS wasmtime binary know `-S loopback-allow` (wasm/wasmtime-loopback.patch)?
#
# The launcher and the runtime ride in one image but come from TWO inputs, moved
# by two different hands: wasm_manager.py is rebuilt by CI on every push that
# touches wasm/, while the binary arrives through Dockerfile.wasm's
# WASMTIME_IMAGE digest, which only the MANUAL toolchain workflow moves. On
# 2026-07-28 they came apart — the launcher started emitting the flag hours
# before a patched binary existed anywhere — and an unknown -S option is not a
# soft landing: wasmtime answers `error: unknown -S / --wasi option:
# loopback-allow` and exits 2 before it has even read the module. EVERY tenant
# launch on the fleet failed, each claim was taken and handed straight back, and
# from outside it read only as "queued", forever.
#
# So ask the binary instead of assuming it. One `-S help` exec against the very
# file that runs every tenant, cached for the process.
#
# ONLY POSITIVE EVIDENCE OF ABSENCE suppresses the flag. A probe that cannot run
# at all (no binary on PATH, a timeout) leaves it in place: an environment with
# no wasmtime launches no tenants either, and "I could not check" must never be
# the thing that quietly takes a wall down. The suppression is loud, and
# /health carries it so the fleet can be asked from outside which boxes are
# missing the patch — see test/wasm-loopback-flag.test.mjs, which pins that the
# flag is still emitted whenever support is not disproven.
_LOOPBACK_FLAG = None      # None = not probed yet; True/False = the binary's answer


def _loopback_flag_supported() -> bool:
    """Does this wasmtime speak `-S loopback-allow=<port[+port...]>`?

    The question is the SEPARATOR, not the option name. `-S` splits its own
    argument on commas before any option sees the value, so the comma-joined
    list this launcher used to emit reached wasmtime as
    `loopback-allow=<first>` plus a bogus option named `<second>`, and an
    unknown -S option is a hard parse error before the module is even read.
    Every tenant on the box died at launch — 2026-07-28 with the option name
    unknown, then again 2026-07-30 (release v0.5.296) with the name known and
    the separator wrong. So probe for the `+` FORM in the help text: a binary
    advertising `<port[,port...]>` carries the older patch and cannot be handed
    a multi-port list at all.

    UNPROVEN MEANS DO NOT PASS. That is the opposite of the usual instinct for
    a security control, and it is deliberate: a wall that refuses every launch
    protects nothing, it just takes the box down. The suppression is loud, and
    /health carries it so the fleet can be asked from outside which boxes are
    missing the patch.
    """
    global _LOOPBACK_FLAG
    if _LOOPBACK_FLAG is not None:
        return _LOOPBACK_FLAG
    if MOCK:
        _LOOPBACK_FLAG = True
        return _LOOPBACK_FLAG
    try:
        r = subprocess.run([WASMTIME, "serve", "-S", "help"],
                           capture_output=True, text=True, timeout=10)
        # the patch documents its own separator; that string is the contract
        _LOOPBACK_FLAG = "port[+port" in ((r.stdout or "") + (r.stderr or ""))
    except Exception as e:
        print(f"wasm-manager: could not probe `-S loopback-allow` ({e}); "
              f"launching WITHOUT the loopback wall", flush=True)
        _LOOPBACK_FLAG = False
    if not _LOOPBACK_FLAG:
        print("wasm-manager ERROR: this wasmtime has no `-S loopback-allow` — the cross-tenant "
              "loopback wall is NOT enforced on this box. The image was assembled from a "
              "wasm_manager.py newer than its WASMTIME_IMAGE pin: rebuild the wasmtime toolchain "
              "with wasm/wasmtime-loopback.patch and repoint wasm/Dockerfile.wasm. Launching "
              "tenants WITHOUT the flag — passing it would fail every launch outright.", flush=True)
    return _LOOPBACK_FLAG


# Does THIS wasmtime binary serve WASIp3? TWO flags make that true, and the
# probe proves BOTH — same doctrine as the loopback probe, applied BEFORE the
# outage this time: the launcher and the binary move by different hands, and
# an unknown option is exit 2 before the module is read, on every tenant.
#   -S p3                       links the WASIp3 API surface
#   -W component-model-async    enables the async/`stream` component-model
#                               feature in the ENGINE — without it a p3
#                               component fails instantiation with "`stream`
#                               requires the component model async feature"
#                               (found by the app side's first real p3 build,
#                               2026-07-31: -Sp3 alone serves nothing)
# Probe tokens are option name PLUS value hint as the help printer renders
# them contiguously ("p3[=y|n]", "component-model-async[=y|n]"): bare names
# could match prose, and the exact token cannot match the sibling
# component-model-async-stackful/-bytes options. Unproven means DO NOT PASS —
# dropping the pair costs nothing on the p2 majority, and wasip3 deployments
# are refused at claim and at launch with a readable error instead of
# exit-2ing the whole box.
_P3_FLAG = None            # None = not probed yet; True/False = the binary's answer


def _p3_supported() -> bool:
    global _P3_FLAG
    if _P3_FLAG is not None:
        return _P3_FLAG
    if MOCK:
        _P3_FLAG = True
        return _P3_FLAG
    def _help(group):
        r = subprocess.run([WASMTIME, "serve", group, "help"],
                           capture_output=True, text=True, timeout=10)
        return (r.stdout or "") + (r.stderr or "")
    try:
        _P3_FLAG = ("p3[=y|n]" in _help("-S")
                    and "component-model-async[=y|n]" in _help("-W"))
    except Exception as e:
        print(f"wasm-manager: could not probe the p3 flag pair ({e}); launching without them", flush=True)
        _P3_FLAG = False
    if not _P3_FLAG:
        print("wasm-manager: this wasmtime does not advertise BOTH `-S p3` and "
              "`-W component-model-async` — wasip2 tenants launch unchanged, wasip3 "
              "versions are refused at claim/launch (/health carries `p3`).", flush=True)
    return _P3_FLAG


def _p3_active() -> bool:
    """The box serves WASIp3: the binary speaks it AND the operator switch is on."""
    return _P3_ENV_ENABLED and _p3_supported()


def _p3_flags() -> list:
    # both or neither: -Sp3 without the engine feature is a p3 surface whose
    # streams cannot instantiate — worse than absent, it looks half-alive
    return ["-Sp3", "-W", "component-model-async"] if _p3_active() else []


def _p3_tuning(enclave_config, wasi_contract) -> list:
    """serve-mode instance knobs for a wasip3 component, from the version
    config's `p3` object. wasmtime 45 reuses instances for p3 (defaults: 128
    requests per instance, 16 concurrent, 1s idle hold) where p2 stays one
    instance per request; a publisher whose guest state must not be shared
    that widely — or whose nn compute() is sync and would stall its instance's
    other in-flight requests — dials these down per VERSION:
      {"p3": {"maxConcurrent": 1, "maxReuse": 256, "idleSeconds": 5}}
    Clamped hard (concurrent 1-64, reuse 1-1024, idle 0-120s): these shape one
    tenant's own process, but unbounded values are still a resource lever.
    Emitted ONLY for a component classified 0.3 AND only when this binary's
    serve --help lists the option — the flags are younger than -Sp3 itself,
    and a bad per-version value must degrade to wasmtime's defaults with a
    printed warning, never brick the version."""
    if wasi_contract != "0.3" or not enclave_config:
        return []
    try:
        t = (json.loads(enclave_config) or {}).get("p3")
    except Exception:
        return []
    if not isinstance(t, dict):
        return []
    help_text = _serve_long_help()
    args = []
    for key, flag, lo, hi, suffix in (
            ("maxConcurrent", "--max-instance-concurrent-reuse-count", 1, 64, ""),
            ("maxReuse", "--max-instance-reuse-count", 1, 1024, ""),
            ("idleSeconds", "--idle-instance-timeout", 0, 120, "s")):
        if key not in t:
            continue
        if flag not in help_text:
            print(f"wasm-manager: this wasmtime lacks {flag}; ignoring p3.{key}", flush=True)
            continue
        try:
            v = max(lo, min(hi, int(t[key])))
        except (TypeError, ValueError):
            print(f"wasm-manager: version config p3.{key} is not an integer; ignoring", flush=True)
            continue
        args += [flag, f"{v}{suffix}"]
    return args


# Cooperative threads (wasip3 🧵): a coop-linked guest (wasi-libc
# ENABLE_COOP_THREADS + lld `--cooperative-threading`) implements pthreads /
# std::thread on component-model tasks — threads interleave on the instance's
# async runtime (concurrent, not parallel; the process/cpu model is untouched).
# Serving one takes wasmtime >= the 49 dev line AND `-W component-model-
# threading`. HELP TEXT IS A LIAR HERE, which is why this probe is not the
# usual token match: wasmtime 47 ADVERTISES the option yet cannot parse the
# `thread.new-indirect` canon builtin every coop guest is linked around (its
# binary reader predates the encoding — "failed to parse WebAssembly module").
# So ask the binary the only honest way: COMPILE the smallest component that
# uses the builtin. Pass = the engine that will run tenants accepted the exact
# construct pthread_create needs. Unproven means DO NOT PASS (the loopback
# doctrine): the flag is dropped, and thread-needing apps are refused at
# launch with a readable error instead of exit-2ing the box or dying in
# missing-import noise.
_THREADS_ENV_ENABLED = os.environ.get("WASM_COOP_THREADS", "1").lower() not in ("0", "false", "no")
_THREADS_FLAG = None       # None = not probed yet; True/False = the binary's answer

# The probe component: nothing but a funcref table and one `thread.new-indirect`
# canon builtin against it — the intrinsic pthread_create lowers to. Text WAT on
# purpose (wasmtime parses .wat natively): no binary artifact to ship or drift.
_THREADS_PROBE_WAT = """(component
  (core type $start (func (param i32)))
  (core module $m
    (table (export "t") 1 1 funcref)
  )
  (core instance $i (instantiate $m))
  (alias core export $i "t" (core table $tbl))
  (core func $spawn (canon thread.new-indirect $start $tbl))
)
"""


def _threads_supported() -> bool:
    global _THREADS_FLAG
    if _THREADS_FLAG is not None:
        return _THREADS_FLAG
    if MOCK:
        _THREADS_FLAG = True
        return _THREADS_FLAG
    try:
        with tempfile.TemporaryDirectory() as td:
            wat = pathlib.Path(td) / "thread-probe.wat"
            wat.write_text(_THREADS_PROBE_WAT)
            r = subprocess.run([WASMTIME, "compile",
                                "-W", "component-model-threading,component-model-async",
                                str(wat), "-o", str(pathlib.Path(td) / "probe.cwasm")],
                               capture_output=True, text=True, timeout=30)
        _THREADS_FLAG = r.returncode == 0
    except Exception as e:
        print(f"wasm-manager: could not probe cooperative-threading support ({e}); "
              f"launching without the flag", flush=True)
        _THREADS_FLAG = False
    if not _THREADS_FLAG:
        print("wasm-manager: this wasmtime cannot compile the thread.new-indirect probe — "
              "coop-thread guests are refused at launch with a readable error "
              "(/health carries `coopThreads`).", flush=True)
    return _THREADS_FLAG


def _threads_active() -> bool:
    """The box serves coop-thread guests: p3 is live (threads are a p3
    feature), the binary proved the builtin, and the operator switch is on."""
    return _THREADS_ENV_ENABLED and _p3_active() and _threads_supported()


def _needs_coop_threads(wasm) -> bool:
    """Does this component spawn cooperative threads? The coop runtime's core
    module imports `[thread-new-indirect-v0]` (and siblings) — length-prefixed
    names sitting verbatim in the binary, so a raw scan is exact the same way
    the wasi:-string world scan is. Calibrated on real builds: a coop-linked
    guest carries the marker ~12x, a non-coop build of the same source zero.
    Prefix-matched (`[thread-`) so a future `-v1` revision still routes right;
    a data-segment false positive costs only an unneeded engine flag."""
    try:
        return b"[thread-" in pathlib.Path(wasm).read_bytes()
    except OSError:
        return False


def _threads_flags(needs_threads) -> list:
    # per-tenant, not blanket: the 🧵 surface is experimental engine area, so
    # only guests that carry the intrinsic marker get it switched on.
    return (["-W", "component-model-threading"]
            if needs_threads and _threads_active() else [])


# Shared-everything threads (SET ⚡): the OTHER threads model, and the one that
# buys real cores. A SET guest (wasi-libc ENABLE_SET_THREADS + set-componentize,
# built by Dockerfile.wasipsetc-build) spawns REAL OS threads that run in
# PARALLEL inside one component instance over shared linear memory — measured
# 15.8x on 16 cores. Coop threads (above) interleave on one core; these do not.
# The distinction is kept sharp in docs/wasm-parallelism.md, and it drives a
# SEPARATE capability because the engine surface, the toolchain and the
# thread-count blast radius all differ.
#
# Serving one takes the enclave wasmtime with wasm/wasmtime-set-threads.patch
# AND `-W shared-everything-threads` (plus the shared threads/component-model-
# threading/shared-memory flags the spawn intrinsics ride on). Same HELP-IS-A-
# LIAR discipline as coop: prove it by COMPILING the exact `thread.spawn-indirect`
# construct the toolchain emits, never by grepping help text. Unproven ⇒ drop
# the flag and refuse SET guests readably at launch (never exit-2 the box).
_SET_ENV_ENABLED = os.environ.get("WASM_SET_THREADS", "1").lower() not in ("0", "false", "no")
_SET_FLAG = None       # None = not probed yet; True/False = the binary's answer

# SET epoch machinery: `-W set-epochs=n` (wasm/wasmtime-set-epochs.patch) drops
# the epoch-interruption codegen that SET otherwise compiles into every guest
# function entry and loop back-edge — measured ~10% guest MIPS on risc-box, the
# last engine-side cost of declaring SET. The platform's stop of last resort is
# killing the tenant's process, parked workers still stop via the parking spot,
# and a spinning worker's teardown is bounded by the join timeout + reaper
# detach, so per-tenant epoch checks buy stop PROMPTNESS the fleet does not
# need. WASM_SET_EPOCHS=1 is the kill-switch back to epoch-armed SET. The flag
# is PROBED (old engine ⇒ omitted, launches keep working), so manager and
# engine can be released in either order.
_SET_EPOCHS_ENV_KEEP = os.environ.get("WASM_SET_EPOCHS", "0").lower() in ("1", "true", "yes")
_SET_EPOCHS_FLAG = None  # None = not probed yet; True/False = the binary's answer

# The probe: a shared-memory core module and one `thread.spawn-indirect` canon
# builtin over its exported plain funcref table — the exact construct
# set-componentize wires and pthread_create lowers to. Text WAT (wasmtime parses
# .wat natively): no binary artifact to ship or drift. If this compiles, the
# engine carries the spawn path; a help-advertised-but-unpatched wasmtime fails
# it (the intrinsic bails at translation without the patch).
_SET_PROBE_WAT = """(component
  (core type $start (func (param i32)))
  (core module $m
    (memory (export "memory") 1 1 shared)
    (table (export "t") 1 1 funcref)
  )
  (core instance $i (instantiate $m))
  (alias core export $i "t" (core table $tbl))
  (core func $spawn (canon thread.spawn-indirect $start (table $tbl)))
)
"""


def _set_supported() -> bool:
    global _SET_FLAG
    if _SET_FLAG is not None:
        return _SET_FLAG
    if MOCK:
        _SET_FLAG = True
        return _SET_FLAG
    try:
        with tempfile.TemporaryDirectory() as td:
            wat = pathlib.Path(td) / "set-probe.wat"
            wat.write_text(_SET_PROBE_WAT)
            r = subprocess.run([WASMTIME, "compile",
                                "-W", "threads,shared-everything-threads,"
                                      "component-model-threading,shared-memory",
                                str(wat), "-o", str(pathlib.Path(td) / "probe.cwasm")],
                               capture_output=True, text=True, timeout=30)
        _SET_FLAG = r.returncode == 0
    except Exception as e:
        print(f"wasm-manager: could not probe shared-everything-threads support ({e}); "
              f"launching without the flag", flush=True)
        _SET_FLAG = False
    if not _SET_FLAG:
        print("wasm-manager: this wasmtime cannot compile the thread.spawn-indirect probe — "
              "SET (shared-everything-threads) guests are refused at launch with a readable "
              "error (/health carries `set`).", flush=True)
    return _SET_FLAG


def _set_epochs_flag_supported() -> bool:
    """Does this wasmtime parse `-W set-epochs`? Same probe-not-help discipline
    as `_set_supported`: compile the spawn-intrinsic probe WITH the flag. An
    engine without wasmtime-set-epochs.patch rejects the unknown option before
    it looks at the module, so the failure direction is safe — unproven means
    the flag is OMITTED and SET tenants launch epoch-armed as before, never a
    fleet of launches dying on an unrecognized option."""
    global _SET_EPOCHS_FLAG
    if _SET_EPOCHS_FLAG is not None:
        return _SET_EPOCHS_FLAG
    if MOCK:
        _SET_EPOCHS_FLAG = True
        return _SET_EPOCHS_FLAG
    try:
        with tempfile.TemporaryDirectory() as td:
            wat = pathlib.Path(td) / "set-epochs-probe.wat"
            wat.write_text(_SET_PROBE_WAT)
            r = subprocess.run([WASMTIME, "compile",
                                "-W", "threads,shared-everything-threads,"
                                      "component-model-threading,shared-memory,"
                                      "set-epochs=n",
                                str(wat), "-o", str(pathlib.Path(td) / "probe.cwasm")],
                               capture_output=True, text=True, timeout=30)
        _SET_EPOCHS_FLAG = r.returncode == 0
    except Exception as e:
        print(f"wasm-manager: could not probe -W set-epochs support ({e}); "
              f"SET tenants launch epoch-armed", flush=True)
        _SET_EPOCHS_FLAG = False
    return _SET_EPOCHS_FLAG


def _set_active() -> bool:
    """The box serves SET guests: the binary proved the spawn intrinsic and the
    operator switch is on. Unlike coop threads, SET rides on wasip2 (its libc is
    the wasip2 flavor), so it does NOT gate on p3 being live."""
    return _SET_ENV_ENABLED and _set_supported()


def _needs_set_threads(wasm) -> bool:
    """Does this component spawn SET threads? set-componentize wires the spawn
    canon under the import string `[set-spawn-indirect]`, which sits verbatim in
    the component bytes — the same raw-scan exactness as the coop `[thread-`
    marker and the wasi:-string world scan. A component that skipped
    componentization carries the libc's `-1` stub instead and does NOT match, so
    it correctly routes as a plain (single-threaded-fallback) app."""
    try:
        return b"[set-spawn-indirect]" in pathlib.Path(wasm).read_bytes()
    except OSError:
        return False


def _set_flags(needs_set) -> list:
    # per-tenant, not blanket: SET is experimental engine surface AND spawns
    # node-wide OS threads, so only guests that carry the marker arm it. The
    # four flags are the spawn intrinsics' full dependency set (threads +
    # shared-everything-threads + component-model-threading + shared-memory);
    # the engine's live-thread cap (ENCLAVE_MAX_SET_THREADS, derived from the
    # tenant's ENCLAVE_AVAILABLE_PARALLELISM) is what bounds the blast radius.
    #
    # set-epochs=n on top: SET without the per-backedge epoch-check tax (see
    # the _SET_EPOCHS_FLAG comment above). Probed, and WASM_SET_EPOCHS=1
    # restores epoch-armed SET fleet-wide without a release.
    if not (needs_set and _set_active()):
        return []
    flags = ("threads,shared-everything-threads,"
             "component-model-threading,shared-memory")
    if not _SET_EPOCHS_ENV_KEEP and _set_epochs_flag_supported():
        flags += ",set-epochs=n"
    return ["-W", flags]


# wasm64 / memory64: 64-bit linear memory, the only guest class allowed past
# the 4 GiB line. It is a COMPONENT (layer 1) whose main core module carries
# a 64-bit memory — an ordinary wasip2 app with ports, sockets and HTTP that
# happens to address more than 4 GiB — built by Dockerfile.wasm64p2-build
# from C or Rust. Nothing about launch changes for it except the engine's
# two memory64 switches and the ceiling lift.
#
# It used to be a preview1 CORE MODULE instead (clang --target=wasm64-wasip1
# against the marshalling wasi-libc in wasm/wasi-libc-mem64.patch, which the
# component build still applies), because no memory64 component toolchain
# existed anywhere. That class could only ever be a portless compute guest —
# preview1 has no socket surface on this engine — and it is gone: the door
# refuses every core module now, and a >4 GiB guest gets the full surface.
#
# TWO probes, because the class needs two engine features. Plain memory64 is
# default-enabled in the pinned engine, proven below by compiling the
# construct with NO flags (the exit-2 doctrine: a flag that is not needed is
# a flag that is not passed). The component-model half is NOT default-on and
# is passed explicitly; _cm64_supported probes it. _mem64_advertised is the
# AND — what /health claims — while launch keeps the two failures apart so
# each says what actually went wrong.
_MEM64_ENV_ENABLED = os.environ.get("WASM_MEM64", "1").lower() not in ("0", "false", "no")
_MEM64_FLAG = None     # None = not probed yet; True/False = the binary's answer

# The probe: a 64-bit memory plus a load through a 64-bit address — the exact
# construct every wasm64 guest opens with. Compiled with NO feature flags:
# passing proves the default engine configuration serves these guests, which
# is precisely what launch does.
_MEM64_PROBE_WAT = """(module
  (memory i64 1)
  (func (export "_start") (drop (i64.load (i64.const 0))))
)
"""


def _mem64_supported() -> bool:
    global _MEM64_FLAG
    if _MEM64_FLAG is not None:
        return _MEM64_FLAG
    if MOCK:
        _MEM64_FLAG = True
        return _MEM64_FLAG
    try:
        with tempfile.TemporaryDirectory() as td:
            wat = pathlib.Path(td) / "mem64-probe.wat"
            wat.write_text(_MEM64_PROBE_WAT)
            r = subprocess.run([WASMTIME, "compile", str(wat),
                                "-o", str(pathlib.Path(td) / "probe.cwasm")],
                               capture_output=True, text=True, timeout=30)
        _MEM64_FLAG = r.returncode == 0
    except Exception as e:
        print(f"wasm-manager: could not probe memory64 support ({e}); "
              f"wasm64 guests will be refused", flush=True)
        _MEM64_FLAG = False
    if not _MEM64_FLAG:
        print("wasm-manager: this wasmtime cannot compile the memory64 probe — "
              "wasm64 guests are refused at launch with a readable error "
              "(/health carries `mem64`).", flush=True)
    return _MEM64_FLAG


_CM64_FLAG = None

# A memory64 COMPONENT: a 64-bit canonical memory carrying a string argument.
# The engine parses this only with `component-model-memory64` on ("64-bit
# memories require the `cm64` feature", probed on the 49 line), which is
# exactly the switch launch passes for such apps — so compiling it under the
# launch flags proves the whole path, not just the core-module half.
_CM64_PROBE_WAT = """(component
  (core module $m
    (memory (export "memory") i64 1)
    (func (export "cabi_realloc") (param i64 i64 i64 i64) (result i64) (i64.const 0))
    (func (export "f") (param i64 i64)))
  (core instance $i (instantiate $m))
  (func (export "f") (param "s" string)
    (canon lift (core func $i "f") (memory $i "memory") (realloc (func $i "cabi_realloc")) string-encoding=utf8))
)
"""


def _cm64_supported() -> bool:
    """Can this wasmtime run a memory64 component (a 64-bit canonical memory)?"""
    global _CM64_FLAG
    if _CM64_FLAG is not None:
        return _CM64_FLAG
    if MOCK:
        _CM64_FLAG = True
        return _CM64_FLAG
    try:
        with tempfile.TemporaryDirectory() as td:
            wat = pathlib.Path(td) / "cm64-probe.wat"
            wat.write_text(_CM64_PROBE_WAT)
            r = subprocess.run([WASMTIME, "compile", "-W", "memory64,component-model-memory64",
                                str(wat), "-o", str(pathlib.Path(td) / "probe.cwasm")],
                               capture_output=True, text=True, timeout=30)
        _CM64_FLAG = r.returncode == 0
    except Exception as e:
        print(f"wasm-manager: could not probe component-model memory64 ({e}); "
              f"memory64 components will be refused", flush=True)
        _CM64_FLAG = False
    if not _CM64_FLAG:
        print("wasm-manager: this wasmtime cannot compile the memory64 COMPONENT probe — "
              "memory64 components are refused at launch with a readable error.", flush=True)
    return _CM64_FLAG


def _mem64_active() -> bool:
    """The box serves wasm64 guests: the binary proved memory64 under its
    default configuration and the operator switch is on."""
    return _MEM64_ENV_ENABLED and _mem64_supported()


def _mem64_advertised() -> bool:
    """What /health's `mem64` claims, which is NARROWER than _mem64_active().

    The >4 GiB guest class is a memory64 COMPONENT (a wasip2 app on a 64-bit
    linear memory), and running one needs the engine's component-model
    memory64 support on top of plain memory64 — two different engine
    features, two different probes. A box that proved only the first would
    advertise `mem64`, win the claim for a component it cannot start, and
    refuse at launch: the deployment queues on the wrong box instead of
    landing on a capable one. So the capability the fleet advertises is the
    AND of both probes.

    Launch keeps the two apart on purpose (a plain-memory64 failure and a
    component-memory64 failure say different things), so this is the
    advertisement only."""
    return _mem64_active() and _cm64_supported()


def _component_mem64(data: bytes) -> bool:
    """Is this COMPONENT (layer 1) built on a 64-bit linear memory? Walks the
    component's sections, applies `_module_mem64` to every core module
    (section id 1) that defines a memory, and descends into nested
    components (id 4): ANY 64-bit memory anywhere makes the component
    memory64. "Any", not "the first": a wasm64 app ships composed under a
    wasm32 pass-through (the engine's host ABI is 32-bit; the app's WASI
    calls cross a component-to-component adapter), so the top level holds a
    32-bit proxy module beside a nested component whose core is 64-bit, and
    the engine needs `component-model-memory64` for that nested memory as
    much as for a top-level one. A memory64 component is the run-mode shape
    wasm64 ships in from 2026-09-02 on (a wasip2 app whose guest can address
    >4 GiB): it keeps the whole socket/HTTP surface and the only launch
    difference is the engine's memory64 switches plus the RAM ceiling.
    Lockstep with the gateway's component_mem64 and the CLI/site
    componentMem64."""
    if len(data) < 8 or data[0:4] != b"\x00asm" or (data[6] | (data[7] << 8)) != 1:
        return False
    i = 8
    while i < len(data):
        sid = data[i]
        try:
            size, j = _uleb(data, i + 1)
        except (IndexError, ValueError):
            return False
        inner = data[j:j + size]
        if sid == 1:                       # core module
            # shims and fixup modules the encoder adds have no memory: skipped
            if len(inner) >= 8 and inner[0:4] == b"\x00asm" and _module_has_memory(inner) and _module_mem64(inner):
                return True
        elif sid == 4:                     # nested component: a full binary
            if _component_mem64(inner):
                return True
        i = j + size
    return False


def _module_has_memory(data: bytes) -> bool:
    """Does this core module define a memory (section id 5 with count > 0)?"""
    i = 8
    while i < len(data):
        sid = data[i]
        try:
            size, i = _uleb(data, i + 1)
        except (IndexError, ValueError):
            return False
        if sid == 5:
            try:
                count, _ = _uleb(data, i)
            except (IndexError, ValueError):
                return False
            return count > 0
        i += size
    return False


def _needs_cm64(wasm) -> bool:
    """A memory64 COMPONENT (see `_component_mem64`)."""
    try:
        return _component_mem64(pathlib.Path(wasm).read_bytes())
    except OSError:
        return False


_SERVE_HELP = None


def _serve_long_help() -> str:
    """`wasmtime serve --help` text, cached for the process (probe substrate
    for long options the way `-S help` is for -S ones)."""
    global _SERVE_HELP
    if _SERVE_HELP is not None:
        return _SERVE_HELP
    if MOCK:
        _SERVE_HELP = ("--max-instance-concurrent-reuse-count "
                       "--max-instance-reuse-count --idle-instance-timeout")
        return _SERVE_HELP
    try:
        r = subprocess.run([WASMTIME, "serve", "--help"],
                           capture_output=True, text=True, timeout=10)
        _SERVE_HELP = (r.stdout or "") + (r.stderr or "")
    except Exception:
        _SERVE_HELP = ""
    return _SERVE_HELP


def _build_cmd(pspec, wasm, serve_port: int, mem_bytes: int, port_map=None, fsdir=None,
               nn=False, enclave_config=None, vol_mounts=None, egress=None, egress_transparent=None,
               enc=None, gpu_share: float = 0.0, nn_report=None, secrets=None,
               cpu_share: float = 0.0, nn_resident_other: int = 0, hosts="", wasi_contract=None,
               threads=False, set_threads=False, cfgdir=None,
               shielded_vram_gb: float = 0.0, cm64=False):
    """The wasmtime invocation for a ports spec. Returns (cmd, host_port, wait_ports).
    `nn_report`, when a dict, is filled with the wasi-nn preload plan:
    {"emitted": [vol...], "skipped": {vol: why}, "stages": {vol: "kind::dir"},
     "residentBytes": int}
    - what launch() records on the tenant so the preload watchdog and the log
    verifier can hold the boot to it.

    serve mode: `wasmtime serve` owns the one HTTP listener; no sockets granted.
    run mode:   `wasmtime run` with wasi:sockets granted (-Stcp/-Sudp/
                -Sinherit-network/-Sallow-ip-name-lookup, verified against
                wasmtime 45). The app binds the ACTUAL ports from the mapping;
                ENCLAVE_PORTS tells it which ("tcp:5432=31245" = logical=actual —
                bind the actual). The grant is coarse (wasmtime can't allowlist
                per port), so the audit sweep enforces the firewall: bind an
                unassigned low port and the app is killed.
    mem64 mode: `wasmtime run` with NO socket surface at all. A wasm64 guest
                is a preview1 CORE module and preview1 has no socket API on
                this engine (the legacy tcplisten/listenfd implementation is
                deleted in the 49 line — probed: `-Spreview2=n` is refused),
                so granting -Stcp/-Sinherit-network would hand capabilities
                nothing inside the guest can reach. It gets compute + /data +
                /config + volumes + stdio, nothing else — which also means a
                wasm64 tenant physically cannot dial a sibling's loopback
                port (the wall the run-mode note below worries about).
    Both modes add -Sp3 (when this binary proves it and WASM_P3 is not 0) so
    apps may target the WASIp3 async APIs as well as wasip2; socket
    permissions are identical either way. `wasi_contract` is the launch-time
    classification of the component's own bytes ("0.2"/"0.3"/None); a 0.3
    serve app additionally gets the publisher's `p3` instance knobs.
    `fsdir`, when set, is preopened as the guest's /data (a private ramdisk
    scratch space); the app sees only that subtree.
    `nn`, when set, grants wasi-nn (`-S nn`): the deployment bought a GPU share,
    so the guest may run ONNX inference through the host runtime (the MPS caps
    ride in the process env, set by launch(), not here)."""
    fs_args = ["--dir", f"{fsdir}::{FS_GUEST_PATH}"] if fsdir else []
    # nn tenants also carry whatever wasmtime flags the boot probe adopted
    # (e.g. -O pooling-allocator=n: serve's default pooling allocator reserves
    # ~6TB of virtual address space, which CUDA init chokes on under TDX).
    # hostcall-fuel: wasmtime caps guest<->host copies per request at 128MiB
    # by default (a DoS guard); an embedded LLM blows through it twice - the
    # model bytes at load() (traps with a bare wasm backtrace, no message at
    # default log levels) and the per-step logits reads on a big vocab
    # (151936 x 4B x 400 tokens = 240MB). 4GiB keeps the guard while clearing
    # any model that passes WASM_MAX_BYTES.
    nn_args = ["-Snn", "-S", "hostcall-fuel=4294967296", *(_NN_PROBE.get("args") or [])] if nn else []
    # per-deployment config: a wasi --env var the guest reads (the app decides
    # what to do with it). Value is the verified config JSON; only forwarded to
    # the GUEST, never to the wasmtime process env (that carries the CUDA/ORT
    # knobs). Kept out of the log line: a config may hold an API key.
    #
    # Two channels, because one argv string cannot hold more than
    # MAX_ARG_STRLEN (128KiB) and a rev-7 config CID can carry up to
    # CONFIG_MAX_BYTES:
    #   ENCLAVE_CONFIG      the JSON itself, whenever it fits CONFIG_ENV_MAX_BYTES.
    #                       Every app written before rev 7 reads only this, so it
    #                       stays populated for every config that can fit.
    #   ENCLAVE_CONFIG_FILE the path to the same JSON inside the guest, ALWAYS
    #                       set when there is a config. This is the only channel
    #                       for a config past the env ceiling, so an app that
    #                       wants large configs reads the file and gets both.
    # An app checking the file first and the env second works at every size.
    # The guest's memory ceiling, in MiB, as a GUEST-FACING variable. This is
    # `-W max-memory-size` below, the cap the engine enforces on the app's
    # linear memory: the wasm32 clamp or the mem64 lift, whichever applied.
    #
    # It has to be an explicit --env like the config and the ports, because a
    # guest inherits NOTHING from this process (there is no -Sinherit-env, on
    # purpose). Do not confuse it with ENCLAVE_AVAILABLE_PARALLELISM, which is
    # set on the wasmtime PROCESS because the ENGINE reads that one; nothing
    # reads this one but the app, so process-level would be invisible.
    #
    # Why an app wants it: the cap is enforced but unknowable from inside, so
    # an app that sizes a heap, a cache or an emulated machine's RAM can only
    # discover it by dying at it. It is a ceiling, not a budget — everything
    # the guest holds lives under it, so an app must leave room for the rest
    # of itself.
    cfg_args = ["--env", f"ENCLAVE_MEM_MB={max(1, mem_bytes >> 20)}"]
    if enclave_config:
        if len(enclave_config.encode("utf-8")) <= CONFIG_ENV_MAX_BYTES:
            cfg_args += ["--env", "ENCLAVE_CONFIG=" + enclave_config]
        if cfgdir:
            cfg_args += ["--dir", f"{cfgdir}::{CONFIG_GUEST_PATH}",
                         "--env", f"ENCLAVE_CONFIG_FILE={CONFIG_GUEST_PATH}/{CONFIG_FILE_NAME}"]
        elif len(enclave_config.encode("utf-8")) > CONFIG_ENV_MAX_BYTES:
            # Refuse rather than start an app that would silently see no config
            # at all: past the env ceiling the file IS the only channel.
            raise ValueError(
                f"config is {len(enclave_config.encode('utf-8'))} bytes, over the "
                f"{CONFIG_ENV_MAX_BYTES}-byte env ceiling, and no config dir could be created")
    # per-deployment secrets: owner-staged env vars (_validate_secrets), one
    # guest --env each, sorted for a deterministic cmdline. Same discipline as
    # ENCLAVE_CONFIG: guest-only, never the process env, never a log line (the
    # cmd is never printed; /proc argv is host-side only — no other tenant can
    # read it, wasm guests hold no host fs).
    for _sk in sorted(secrets or {}):
        cfg_args += ["--env", f"{_sk}={secrets[_sk]}"]
    # dedicated-IP egress: a per-deployment SOCKS URL minted by the supervisor
    # (see egress.js). Forwarded verbatim to the GUEST only; it carries a bearer
    # token, so — like ENCLAVE_CONFIG — it never reaches the wasmtime process env or
    # a log line. Set in both modes: a `serve` app makes outbound calls too.
    if egress:
        cfg_args += ["--env", "ENCLAVE_EGRESS=" + egress]
    # the hostnames this deployment answers on (see _validate_hosts). Public
    # information — it is in DNS and in the CT logs — so unlike the values
    # above it is safe in a log line; it rides here to stay with the guest env.
    if hosts:
        cfg_args += ["--env", "ENCLAVE_HOSTS=" + hosts]
    # attached model volumes: preopen each mount as a guest /models/<name> dir.
    # Read-only in practice (dm-verity/EROFS mounts are physically read-only);
    # ENCLAVE_MODELS lists the mounted names so the app can discover them without
    # probing the filesystem.
    vol_mounts = vol_mounts or {}
    vol_args = []
    for name, host_path in vol_mounts.items():
        vol_args += ["--dir", f"{host_path}::{VOL_GUEST_ROOT}/{name}"]
    if vol_mounts:
        vol_args += ["--env", "ENCLAVE_MODELS=" + ",".join(vol_mounts.keys())]
    # encrypted volumes: preopen each (initially empty) staging dir as
    # /enc/<name> - a LIVE preopen, so the plaintext rclone decrypts into it
    # after unlock appears to the guest with no restart. ENCLAVE_ENC lists the
    # names; ENCLAVE_ENC_API + ENCLAVE_ENC_TOKEN are how the app (the only
    # holder of the token besides this manager) drives unlock/sync/lock over
    # loopback. Like ENCLAVE_CONFIG, the token is guest-only env.
    if enc:
        enc_mounts, enc_api, enc_token = enc
        for name, host_path in enc_mounts.items():
            vol_args += ["--dir", f"{host_path}::{ENC_GUEST_ROOT}/{name}"]
        vol_args += ["--env", "ENCLAVE_ENC=" + ",".join(enc_mounts.keys()),
                     "--env", "ENCLAVE_ENC_API=" + enc_api,
                     "--env", "ENCLAVE_ENC_TOKEN=" + enc_token]
    # GGUF volumes double as HOST-PRELOADED wasi-nn graphs (the ggml/llama.cpp
    # backend in our wasmtime): -S nn-graph=ggml::<dir> loads the model ONCE at
    # process start, registered under the dir BASENAME; the guest load_by_name()s
    # it and the weights never enter guest linear memory - model size is bounded
    # by the tenant's share, not wasm32's 4 GiB. Gated on `nn` like the
    # interface itself (no GPU share, no wasi-nn). wasmtime wants the dir named
    # after the graph with one unambiguous model inside (model.gguf, a single
    # *.gguf, or one complete split family) - true for neither Modelwrap
    # mounts (dir = mpk-<root_hash>) nor multi-quant HF repos - so every
    # volume preloads through a STAGED symlink dir named after the volume
    # (_stage_nn_graph); MODEL_VOLUMES' third field picks the file (any part
    # of a split family selects the whole family).
    if nn:
        # gate the NEWER preload kinds on what this toolchain implements
        # (_preload_support) - emitting a kind the wasmtime can't parse or
        # walk aborts the tenant at startup, and manager/toolchain images
        # roll independently. ggml predates the probe and stays ungated.
        support = _preload_support()
        # ggml AND sd graphs collect here and emit AFTER the loop,
        # SMALLEST-FIRST across both kinds: wasmtime preloads -S nn-graph
        # flags in order at boot, both backends load weights into the same
        # VRAM share, and residency is first-come-first-served - so
        # smallest-first puts the models most likely to fit in VRAM before a
        # big one claims - or fails to claim - the rest. Pairs with the
        # apps' smallest-first boot-warmup ladders (llm-chat 0.7.0,
        # image-generator 0.2.0) and the preload's per-graph
        # skip-on-failure (wasmtime-nn-ggml.patch): a small deployment
        # serves its small models and reports the big ones unfit instead of
        # dying at boot. onnx preloads stay inline: they register on the CPU
        # (sessions build per request) and hold no VRAM at boot.
        #
        # The emission below additionally STOPS at the tenant's VRAM budget
        # (gpu_share x card VRAM - the same number launch() puts in the MPS
        # cap): preloading weights that cannot fit is a guaranteed slow OOM,
        # so over-budget volumes are never emitted at all. They stay mounted
        # (and in ENCLAVE_MODELS), so a guest load_by_name() fails INSTANTLY
        # and the apps report "unfit" without a doomed multi-GB load.
        # ENCLAVE_VRAM_BYTES hands the guest the same budget so its warmup
        # ladder can skip the probe entirely. Weights-only accounting on
        # purpose: contexts/compute come and go per request - the budget
        # gate only refuses CERTAIN failures, borderline models still get
        # the honest probe.
        # A SHIELDED share is a slice of the worker's card, not of this node's --
        # and this node has no card at all, so GPU_VRAM_GB is a fleet default
        # (141 GiB, an H200) describing hardware that is not here. Sizing a
        # tenant's budget from it told eyesoff it had 0.85 x 141 = 119.85 GiB of
        # VRAM on a box whose shielded card offers 5.5, and the engine sized its
        # allocation against the fiction and died in llama_decode. The supervisor
        # already sends the real number with the share; use it.
        vram_gb = shielded_vram_gb if shielded_vram_gb > 0 else gpu_share * GPU_VRAM_GB
        vram_bytes = int(vram_gb * (1 << 30)) if gpu_share > 0 else 0
        if vram_bytes:
            vol_args += ["--env", f"ENCLAVE_VRAM_BYTES={vram_bytes}"]
        (share_ram_bytes, node_ram_bytes, budget_bytes, budget_kind,
         ggml_budget_bytes, ggml_budget_kind, gpu_tenant) = _nn_budgets(
            gpu_share, cpu_share, vram_bytes, nn_resident_other)
        # What an APP should price ggml SERVING against, and what kind of memory
        # that is. ENCLAVE_VRAM_BYTES could not answer this on its own: it means
        # "your slice of a card", which on a local GPU tenant IS the serve
        # budget but on a shielded one is only the offload RESERVATION - so an
        # app pricing against it refused models the node can serve on cores.
        # Apps prefer these two and fall back to ENCLAVE_VRAM_BYTES, so an older
        # guest keeps exactly its current behaviour.
        vol_args += ["--env", f"ENCLAVE_NN_SERVE_BYTES={ggml_budget_bytes}",
                     "--env", f"ENCLAVE_NN_SERVE_KIND={ggml_budget_kind}"]
        # Forward the node's ggml context tuning to the GUEST too: with the
        # window and KV cache type known, an app can price a model's KV cache
        # (weights + n_ctx x kv-bytes/token + working set) and refuse models
        # the share certainly cannot SERVE - not just cannot load. That
        # matters because a CUDA OOM inside compute ABORTS the wasmtime
        # process (ggml_abort - no error ever reaches the guest), so a
        # "let's try it" probe of a too-big model kills the whole tenant.
        # MAX_SESSIONS rides along so the app's serve pricing can multiply
        # the per-session cost by the host's concurrent-context cap.
        nn_fwd = {k: os.environ.get(k, "").strip()
                  for k in ("ENCLAVE_GGML_N_CTX", "ENCLAVE_GGML_KV_CACHE_TYPE",
                            "ENCLAVE_GGML_KV_CACHE_TYPE_V", "ENCLAVE_GGML_MAX_SESSIONS")}
        # a deployment-config nnCtx override changes what the HOST allocates
        # (see the launch-time env in _spawn_and_wait); quote the same number
        # to the guest so its serve pricing matches reality
        nc = _nn_cfg_int(enclave_config, "nnCtx", 8192, 262144)
        if nc is not None:
            nn_fwd["ENCLAVE_GGML_N_CTX"] = str(nc)
        for k, val in nn_fwd.items():
            if val:
                vol_args += ["--env", f"{k}={val}"]
        # The batched (pooled) ggml toolchain serves all of a model's
        # concurrent sessions from ONE shared KV pool. This manager ships in
        # the SAME image as that wasmtime, so the flag is a build-time truth,
        # not a node option: guests price one persistent pool per model
        # (accumulating across models) instead of per-session windows.
        vol_args += ["--env", "ENCLAVE_GGML_POOLED=1"]
        vram_stages = []  # (bytes, name, kind, stage)
        emitted = []      # volume names whose graphs went on the cmdline, in order
        skipped = {}      # volume -> why it did NOT preload
        stages = {}       # volume -> "kind::stagedir" as the -S flag spells it
        for name, host_path in vol_mounts.items():
            # MODEL_VOLUMES_SD volumes preload through the sdcpp backend
            # (image txt2img pipelines: safetensors/ckpt checkpoints, FLUX
            # gguf quants); everything else with a GGUF is an LLM for ggml.
            if name in _SD_VOLUMES:
                if not support["sd"]:
                    skipped[name] = "toolchain lacks sd preload"
                    print(f"[nn-graph] sd volume '{name}': toolchain lacks sd preload - mounting only", flush=True)
                    continue
                mode, ckpt = _sd_layout(name, host_path)
                if mode in ("components", "upscaler"):
                    # split-component volumes (Z-Image/Qwen-Image-class) and
                    # upscaler volumes (ESRGAN) stage WHOLE-DIR: the backend
                    # resolves the env-named files inside it - same
                    # one-symlink shape as onnx
                    stage = _stage_onnx_dir(name, host_path)
                    if stage:
                        vram_stages.append((_staged_bytes(stage), name, "sd", stage))
                    else:
                        skipped[name] = "staging failed"
                    continue
                if not ckpt:
                    skipped[name] = "no unambiguous checkpoint"
                    print(f"[nn-graph] sd volume '{name}': no unambiguous checkpoint "
                          f"and the ENCLAVE_SD_*_FILE envs don't resolve here - mounting only", flush=True)
                    continue
                stage = _stage_nn_graph(name, ckpt)
                if stage:
                    vram_stages.append((_staged_bytes(stage), name, "sd", stage))
                else:
                    skipped[name] = "staging failed"
                continue
            gguf = _gguf_path(name, host_path)
            if gguf:
                stage = _stage_nn_graph(name, gguf)
                if stage:
                    vram_stages.append((_staged_bytes(stage), name, "ggml", stage))
                else:
                    skipped[name] = "staging failed"
                continue
            # ONNX volumes preload too (every *.onnx registers as
            # "<volume>/<component>"; guests load_by_name and skip the
            # per-request byte lift entirely). Guest load() of the same
            # bytes converges on the same content-hash session cache, so
            # apps built against the old contract keep working unchanged.
            if _onnx_volume(host_path):
                if not support["onnx"]:
                    skipped[name] = "toolchain lacks onnx preload"
                    print(f"[nn-graph] onnx volume '{name}': toolchain lacks onnx preload - mounting only", flush=True)
                    continue
                stage = _stage_onnx_dir(name, host_path)
                if stage:
                    vol_args += ["-S", f"nn-graph=onnx::{stage}"]
                    emitted.append(name)
                else:
                    skipped[name] = "staging failed"
                continue
            # mounted, but nothing to preload: a data volume, or a layout no
            # picker resolves. Recorded so the plan covers EVERY mounted
            # volume - the heal sweep keys off this being exhaustive.
            skipped[name] = "no preloadable model file (data volume or ambiguous layout)"
        resident = 0
        for _bytes, _name, kind, stage in sorted(vram_stages):
            # mmap-backed ggml graphs answer to the node's RAM wherever the
            # weights land there (a 0-GPU tenant, and every tenant of a box
            # with no local card); everything else to the tenant's own budget.
            # Both figures come from _nn_budgets, so this only picks between
            # them - ggml_budget_bytes IS the tenant budget on a GPU node.
            cap = ggml_budget_bytes if kind == "ggml" else budget_bytes
            cap_kind = "node RAM" if (cap == node_ram_bytes and kind == "ggml") else (
                ggml_budget_kind if kind == "ggml" else budget_kind)
            if cap_kind == "node RAM" and nn_resident_other > 0:
                # say WHY the pool is small - otherwise a shrunk budget reads as
                # a mis-sized node instead of a neighbour holding a model
                cap_kind += f" net of {nn_resident_other / 2**30:.1f} GB held by other tenants"
            if cap and resident + _bytes > cap:
                skipped[_name] = (f"exceeds the {cap_kind} budget ({_bytes / 2**30:.1f} GB weights, "
                                  f"{resident / 2**30:.1f}/{cap / 2**30:.1f} GB already claimed)")
                print(f"[nn-graph] volume '{_name}' ({_bytes / 2**30:.1f} GB weights) skipped: "
                      f"{resident / 2**30:.1f} GB already claimed of the "
                      f"{cap / 2**30:.1f} GB {cap_kind} budget - mounting only", flush=True)
                continue
            resident += _bytes
            vol_args += ["-S", f"nn-graph={kind}::{stage}"]
            emitted.append(_name)
            stages[_name] = f"{kind}::{stage}"
        # Video encode, for a tenant that bought a GPU share. Not a volume:
        # there is no model, the graph IS an encoder session, so this rides the
        # -Snn grant and gpuShare the tenant already has rather than adding a
        # launch flag of its own. It costs nothing until the guest calls
        # load_by_name("nvenc") - opening no session, holding no VRAM - so
        # granting it to every GPU tenant is cheaper than making it a purchase.
        #
        # NOT wrapped in the MPS arbiter, deliberately. The arbiter exists
        # because MPS statically partitions SMs; NVENC is a separate
        # fixed-function engine, and one 720p stream measured 4-5% encoder
        # utilization against 8-9% overall. Taking an arbiter turn per frame
        # would queue encodes behind inference tenants that are not competing
        # for the same silicon. OPEN: whether encode wants its own arbiter
        # class, and what the per-card concurrent-session cap should be - it is
        # a real limit (unrestricted on datacenter parts, capped on consumer
        # ones) and it bounds how many streaming deployments fit on a node.
        if gpu_tenant and support.get("nvenc"):
            try:
                nvdir = FS_DIR / "preload-probe" / "empty"
                nvdir.mkdir(parents=True, exist_ok=True)
                vol_args += ["-S", f"nn-graph=nvenc::{nvdir}"]
                emitted.append("nvenc")
            except OSError as e:
                skipped["nvenc"] = f"staging failed: {e}"
        elif gpu_tenant:
            skipped["nvenc"] = "no hardware encoder on this node, or toolchain lacks the nvenc backend"
        # The guest can't see wasmtime's graph registry, so tell it what the
        # host PRELOADED: the volume names whose graphs went on the cmdline.
        # A load_by_name() NotFound then has an honest reading app-side -
        # name listed = the boot preload was attempted (and failed, loudly,
        # in this tenant's log); name missing = the host never tried (the
        # skip reason - over budget / no unambiguous file - is in nnSkipped
        # on the manager record). Set only under nn, like ENCLAVE_VRAM_BYTES:
        # apps treat an absent env as "older manager, no signal".
        vol_args += ["--env", "ENCLAVE_NN_PRELOADS=" + ",".join(emitted)]
        if isinstance(nn_report, dict):
            nn_report["emitted"] = list(emitted)
            nn_report["skipped"] = dict(skipped)
            nn_report["stages"] = dict(stages)
            # What this tenant will hold resident in NODE RAM once booted -
            # which follows where the weights LAND, not what the node has. A
            # GPU tenant's preloads sit in VRAM and the VRAM ledger owns them
            # (charging RAM too would sell the box short twice); a 0-GPU tenant
            # maps them into host RAM even on a GPU box, and that RAM is as
            # unavailable to the next tenant as it is anywhere else.
            # A shielded box reaches this the same way a CPU one does: its
            # manager is told NODE_HAS_GPU=0, so gpu_tenant is false and its
            # weights - which really are mmap'd in the CVM - are counted.
            nn_report["residentBytes"] = 0 if gpu_tenant else resident
    # enclave transparent egress (phase 2): `-S egress=<host>:<port>` makes the
    # patched wasmtime funnel ALL guest outbound through the loopback SOCKS front
    # (credential in $ENCLAVE_EGRESS_CRED, set host-side by _spawn_and_wait), so an
    # UNMODIFIED app leaves from the deployment's dedicated IPv6. Added in BOTH
    # modes: `serve` intercepts the wasi:http outgoing handler, `run` the
    # wasi:sockets connect. In run mode it ALSO closes the raw bypass — we drop
    # `-Sinherit-network` so the guest can no longer reach the network directly.
    egress_args = ["-S", f"egress={egress_transparent}"] if egress_transparent else []
    # WHICH LOOPBACK PORTS THIS TENANT MAY DIAL (`-S loopback-allow`, the patched
    # wasmtime's per-address rule; see wasm/wasmtime-loopback.patch).
    #
    # Tenants share ONE network namespace and each listens on 127.0.0.1:<its
    # port>, so unrestricted loopback let a guest connect straight into a SIBLING
    # tenant's app - around the supervisor's /x/:id, which is where a PRIVATE
    # deployment's owner-token check and the deployer's WAF live. Nothing at the
    # services can close that: another tenant's app is not ours to gate.
    #
    # The list is what this deployment legitimately reaches, and nothing else:
    #   - its OWN assigned ports (an app that talks to itself)
    #   - the manager's control port, ONLY when it has encrypted volumes (that is
    #     the /encvol plane behind ENCLAVE_ENC_API, the one loopback service a
    #     tenant is supposed to use; without encVolumes it has no business there)
    #   - the SOCKS front, ONLY when this tenant was given an egress URL to dial
    #     (a phase-1 guest dials it explicitly; the transparent path recognises
    #     the front before this rule applies, but naming it is free and keeps the
    #     two paths honest about the same set)
    # Empty is a real, strict answer: a serve-mode app with neither volumes nor
    # egress gets NO loopback at all.
    _lb = set(int(p) for p in ([serve_port] if serve_port else []) + list((port_map or {}).values()) if p)
    if enc:
        _lb.add(PORT)
    for _u in (egress_transparent, egress):
        if not _u:
            continue
        # egress_transparent is already "<host>:<port>"; the guest-visible
        # ENCLAVE_EGRESS is a full socks5h:// URL (never trust it to parse - an
        # unparseable one simply contributes no port)
        _ep = (_parse_egress_url(_u) or {}).get("endpoint", "") if str(_u).startswith("socks5") else str(_u)
        try:
            _lb.add(int(_ep.rsplit(":", 1)[1]))
        except (ValueError, IndexError):
            pass
    # '+'-joined, NOT comma: `-S` eats commas (see _loopback_flag_supported)
    lb_args = ["-S", "loopback-allow=" + "+".join(str(p) for p in sorted(_lb))] if _loopback_flag_supported() else []
    # A memory64 COMPONENT runs exactly like any other component, plus the
    # engine's two memory64 switches: `memory64` (default-on in the 49 pin,
    # stated anyway) and `component-model-memory64`, which the engine
    # refuses a 64-bit canonical memory without ("64-bit memories require
    # the `cm64` feature", probed). Nothing else changes: sockets, egress,
    # loopback wall, p3/thread flags all ride as for a wasm32 component.
    cm64_args = ["-W", "memory64,component-model-memory64"] if cm64 else []
    # Shared memories cannot relocate when growing. Reserve virtual address
    # space up to this deployment's existing cap, so a single >4 GiB grow
    # works as reliably as incremental growth. This does not raise the cap
    # or commit the reserved pages to physical RAM.
    if cm64 and (threads or set_threads):
        cm64_args += ["-O", f"memory-reservation={mem_bytes}"]
    if pspec["serve"]:
        return ([WASMTIME, "serve", "-Scli", "-Shttp", *_p3_flags(), *_p3_tuning(enclave_config, wasi_contract),
                 *_threads_flags(threads), *_set_flags(set_threads),
                 *nn_args, *fs_args, *cfg_args, *vol_args, *cm64_args,
                 *egress_args, *lb_args, "-W", f"max-memory-size={mem_bytes}",
                 "--addr", f"{HOST_IP}:{serve_port}", str(wasm)],
                serve_port, [serve_port])
    port_map = port_map or {}
    enclave_ports = ",".join(f"{e}={port_map[e]}" for e in pspec["norm"])
    # inbound binds (declared tcp:N/udp:N) still need the socket-address check to
    # permit them: `-Sinherit-network` allows all, while `-S egress` installs a
    # check that permits TCP bind/connect + UDP bind but DENIES raw UDP egress.
    # So we grant EXACTLY ONE of them — inherit-network (no egress) OR egress.
    #
    # SECURITY (known, accepted here — defense in depth is elsewhere): when a
    # port-serving app does NOT buy transparent egress, `-Sinherit-network`
    # hands the guest the CVM's shared loopback namespace. A malicious tenant
    # can then reach the enclave's own loopback services — supervisor:8080,
    # worker:8090, this manager:8091 — bypassing the
    # egress net-guard AND per-request billing (an SSRF-to-localhost).
    #
    # AND IT REACHES SIBLINGS, not just platform services — the half this note
    # used to leave out. Every tenant's app listens on 127.0.0.1:<actual>, so one
    # tenant can dial another's port and land INSIDE its app, around /x/:id where
    # a PRIVATE deployment's owner-token check and the deployer's WAF live. That
    # is not closed at any service, because a tenant app is not ours to gate.
    # Transparent egress does NOT fix it either: the `-S egress` carve-out dials
    # literal loopback DIRECT by design (wasmtime-egress.patch), which is what
    # keeps /encvol reachable. Until the per-address check below exists,
    # _audit_peers polices it the same way this file polices binds and storage —
    # it watches every tenant's own sockets for a connection to a live sibling's
    # port and kills the caller. Detection, not prevention: a short-lived dial
    # between sweeps can complete first. We
    # deliberately do NOT try to fix this by dropping the flag, because with the
    # STOCK wasmtime CLI there is no middle ground: the WASI socket-address
    # check is all-or-nothing — `-Sinherit-network` sets it to allow-all, and
    # its ABSENCE defaults it to DENY-all (bind included). `-Stcp`/`-Sudp` only
    # gate whether a socket may be created; they do not permit any address. So
    # dropping `-Sinherit-network` without a replacement check would make the
    # guest's bind() to its OWN assigned loopback port fail, breaking EVERY
    # port-serving app (minecraft/IRC/DNS/…). A `WASM_NO_INHERIT_NET` opt-in
    # would therefore be a footgun that silently kills those apps when enabled,
    # not a safe toggle, so it is intentionally NOT added.
    # The ONLY correct fix is a per-address socket_addr_check that permits bind
    # to the deployment's assigned loopback actual(s) and DENIES connect to the
    # internal service ports / private ranges. wasmtime's CLI cannot express
    # that; it requires EITHER extending the existing `-S egress` patch
    # (wasmtime-egress.patch) with a "local-bind-only, deny-arbitrary-egress"
    # mode usable WITHOUT a live SOCKS backend, OR driving wasmtime through its
    # embedder API (WasiCtxBuilder::socket_addr_check) instead of the CLI, OR a
    # per-tenant network namespace (delicate; must still expose the assigned
    # loopback port to the bridge). Until then, the billing/SSRF exposure is
    # closed at the SERVICES, and those gates are now FAIL-CLOSED rather than
    # optional, which is what makes accepting this bearable: an unset
    # WORKER_TOKEN denies every worker request (not "disables auth"), an unset
    # VMMGR_TOKEN/SECRET denies every control route here, and both opt-outs are
    # explicit env flags that warn loudly. The one plane a tenant is SUPPOSED to
    # reach - /encvol - authenticates with that deployment's own 192-bit token,
    # matched against its own record, so it cannot touch another tenant's
    # volumes. See also the bind audit (_audit_rec), which still kills a guest
    # that binds an unassigned policed port.
    net_args = egress_args if egress_transparent else ["-Sinherit-network"]
    # lb_args rides in BOTH modes, and in run mode it is the half that matters
    # most: `-Sinherit-network` installs an allow-ALL address check, and the
    # patched CLI replaces it with one that still permits every bind and every
    # off-box connect but refuses loopback outside this deployment's own set.
    # That is what keeps a port-serving app off its neighbours' sockets without
    # taking away the raw network it was granted.
    cmd = [WASMTIME, "run", "-Scli", *_p3_flags(), *_threads_flags(threads), *_set_flags(set_threads), *nn_args, "-Stcp", "-Sudp",
           *net_args, *lb_args, "-Sallow-ip-name-lookup", *fs_args, *cfg_args, *vol_args, *cm64_args,
           "-W", f"max-memory-size={mem_bytes}",
           "--env", "ENCLAVE_PORTS=" + enclave_ports, str(wasm)]
    http_entry = f"http:{pspec['http']}" if pspec["http"] else None
    host_port = port_map.get(http_entry, 0) if http_entry else 0
    if host_port:
        wait = [host_port]
    else:
        tcp_actuals = sorted(port_map[e] for e in pspec["norm"] if e.startswith("tcp:"))
        wait = tcp_actuals[:1]                               # udp-only: no waitable port
    return cmd, host_port, wait


def launch(app_ref: str, name: str, cpu_share: float, gpu_share: float = 0.0,
           mem_mb: int = 0, pspec=None, storage_mb=None, config="", volumes=None,
           egress="", secrets=None, hosts="", config_cid="", shielded=None) -> dict:
    pspec = pspec or _parse_ports([])
    if storage_mb is None:
        storage_mb = DEF_STORAGE_MB
    # the guest memory ceiling is the deployment's slice of the node's RAM
    # (cpuShare × NODE_RAM_GB); an explicit memMb (direct callers) caps lower.
    # Clamped to 4 GiB for wasm32: its linear memory is hard-limited there
    # anyway, and the wasmtime generation before the 49 pin also refused
    # `-W max-memory-size` above its memory reservation ("maximum memory size
    # ... exceeds the configured memory reservation"), which killed every
    # launch with cpuShare > ~6% of a 64 GB node. Larger CPU shares still buy
    # proportional vCPU time. A wasm64 guest is the one class the clamp does
    # NOT apply to — mem_mb_raw (its full slice) replaces the ceiling after
    # the launch-time sniff proves the bytes really carry a 64-bit memory
    # (the 49 engine accepts >4 GiB caps; probed live before this shipped).
    WASM32_MAX_MEM_MB = 4096
    if not mem_mb or mem_mb <= 0:
        mem_mb = int(cpu_share * NODE_RAM_GB * 1024)
    mem_mb_raw = max(MIN_MEM_MB, int(mem_mb))
    mem_mb = min(WASM32_MAX_MEM_MB, mem_mb_raw)
    port = _free_port() if pspec["serve"] else 0
    port_map = {} if pspec["serve"] else _alloc_ports(pspec)   # logical entry -> actual bind
    vid = "app_" + uuid.uuid4().hex[:9]
    log_path = LOG_DIR / f"{vid}.log"
    assigned = set(port_map.values()) | ({port} if port else set())
    # Per-deployment scratch fs: private /data on the ramdisk. `storage_mb: 0`
    # (or WASM_FS=0) opts out; a mkdir failure is non-fatal (run without /data
    # rather than fail the deploy).
    fsdir = None
    if FS_ENABLED and storage_mb > 0:
        cand = FS_DIR / vid
        try:
            cand.mkdir(parents=True, exist_ok=True)
            fsdir = cand
        except OSError as e:
            print(f"[fs] {vid} could not create scratch dir: {e}", flush=True)
    # wasi-nn goes to EVERY tenant that asks; what differs is where the weights
    # land. Holding GPU share gets the card, MPS-capped and VRAM-budgeted (env
    # below). Holding none gets the ggml CPU backend on the cores the tenant
    # already bought, weights mapped into node RAM (_build_cmd budgets them).
    #
    # That second case is available ON A GPU NODE TOO, which it was not before.
    # Gating the interface on gpu_share meant a model-volume app dialled to 0%
    # GPU could not run on the fleet's biggest machines at all: it links against
    # wasi:nn/tensor, finds no implementation and dies at startup. Live proof
    # 2026-07-27 - a CPU-dialled llm-chat was moved onto a GPU box, which
    # claimed the lease and handed it back four seconds later. The card is
    # still never handed out for free: a 0-GPU tenant gets no CUDA_MPS_* env
    # and no VRAM budget, so it runs on cores exactly as it would on a CPU box.
    # Preferring CPU-only boxes for this work is a RANKING decision (the deploy
    # console demotes GPU boxes for it), not a capability one - an owner who
    # deliberately moves such a deployment onto a GPU box gets what they asked.
    nn = NN_ENABLED
    rec = {"id": vid, "name": name or vid, "app": app_ref,
           "cpuShare": cpu_share, "gpuShare": gpu_share, "nn": nn,
           # A GPU share served by a card on the UNTRUSTED host, reached by masked
           # offload. Recorded rather than inferred so /vms says plainly which kind
           # of card a tenant got -- the two are priced the same and are not the
           # same thing.
           **({"shielded": shielded} if shielded else {}),
           "hostPort": port,
           "endpoint": f"http://{HOST_IP}:{port}" if port else None, "status": "starting",
           "createdAt": time.time(), "_proc": None, "_log": str(log_path),
           "error": None, "mem_mb": mem_mb,   # exact guest memory cap (floor MIN_MEM_MB)
           "storageMb": storage_mb if fsdir else 0,   # 0 = no /data (opted out or disabled)
           "storageBytes": 0,                         # last measured /data usage (audit sweep)
           # a FULL lease at birth, so a launch that races the supervisor's
           # heartbeat is never mistaken for an unvouched tenant
           "leaseUntil": time.time() + TENANT_LEASE_TTL,
           "ports": pspec["norm"],           # logical (the app's advertised interface)
           "portMap": port_map,              # logical entry -> actual loopback bind
           "boundPorts": [],                 # actuals confirmed bound (the bridge checks this)
           "_assigned": assigned,            # what the audit allows
           "_fsdir": str(fsdir) if fsdir else None}
    with _lock:
        _apps[vid] = rec

    if MOCK:
        # Stand up a trivial responder so the full supervisor path is testable.
        mock_port = port or _free_port()
        rec["hostPort"] = mock_port
        rec["_proc"] = _mock_server(mock_port, vid)
        rec["status"] = "running"
        return rec

    try:
        wasm = _resolve_wasm(app_ref)
    except ValueError as e:
        rec["status"], rec["error"] = "failed", str(e)
        return rec

    # World contract, from the bytes that will run — never from metadata (a
    # version config declares `wasi` for CLAIM routing; this is the launch-time
    # truth it is held to). A wasip3 component on a box whose runtime cannot
    # speak p3 must fail HERE with a readable error: without this check it
    # would instantiate-fail inside wasmtime with missing-import noise (p3
    # probed off) — or, worse, exit 2 on a flag (never: the probe owns that).
    try:
        contract = _component_contract(wasm)
    except OSError as e:
        contract = {"wasi": None, "world": None}
        print(f"[wasi] {vid} contract classification failed ({e}); launching as-is", flush=True)
    rec["wasi"] = contract["wasi"]
    rec["wasiWorld"] = contract["world"]
    if contract["wasi"] == "0.3" and not _p3_active():
        why = "WASM_P3=0 (operator switch)" if _p3_supported() else "this wasmtime lacks -S p3"
        rec["status"] = "failed"
        rec["error"] = (f"app targets WASIp3 ({contract['world']}) but this box does not "
                        f"serve p3 ({why}); a p3-capable enclave must claim it")
        return rec
    # Cooperative threads (🧵): same doctrine, one step further into the
    # binary. A coop-linked guest that launches on a box whose engine lacks
    # the builtin dies as "failed to parse WebAssembly module" — noise. Fail
    # HERE with the readable truth instead.
    needs_threads = _needs_coop_threads(wasm)
    rec["threads"] = needs_threads
    if needs_threads and not _threads_active():
        why = ("WASM_COOP_THREADS=0 (operator switch)" if not _THREADS_ENV_ENABLED
               else "p3 is not live on this box" if not _p3_active()
               else "this wasmtime cannot run the thread.new-indirect builtin")
        rec["status"] = "failed"
        rec["error"] = (f"app uses cooperative threads (wasip3 \U0001f9f5) but this box "
                        f"cannot serve them ({why}); a thread-capable enclave must claim it")
        return rec
    # Shared-everything threads (SET ⚡): same doctrine as coop, one model over.
    # A SET-linked guest carries the `[set-spawn-indirect]` marker and needs the
    # patched engine; on a box without it, spawning would refuse at runtime and
    # the app would silently fall back to one thread — so fail HERE with the
    # readable truth instead, the way p3 and coop do.
    needs_set = _needs_set_threads(wasm)
    rec["set"] = needs_set
    if needs_set and not _set_active():
        why = ("WASM_SET_THREADS=0 (operator switch)" if not _SET_ENV_ENABLED
               else "this wasmtime cannot run the thread.spawn-indirect builtin")
        rec["status"] = "failed"
        rec["error"] = (f"app uses shared-everything threads (SET ⚡) but this box "
                        f"cannot serve them ({why}); a SET-capable enclave must claim it")
        return rec
    # wasm64 / memory64: same doctrine, structural sniff. Two extra rules of
    # its own: (a) it is a COMPUTE guest — a preview1 core module can neither
    # serve wasi:http (no component ABI) nor open sockets (the engine's
    # legacy p1 socket surface is deleted; probed), so a version that
    # declared ports promises an interface the guest cannot provide and its
    # launch would sit waiting for a bind that never comes. Refuse that in
    # class whose memory ceiling is its full RAM slice rather than the
    # wasm32 4 GiB clamp — that lift happens HERE, after the bytes proved
    # the 64-bit memory and before the RAM-budget admission reads
    # rec["mem_mb"], so what the ledger charges is what the guest gets.
    needs_cm64 = _needs_cm64(wasm)
    rec["mem64"] = needs_cm64
    rec["cm64"] = needs_cm64
    if needs_cm64 and not _mem64_active():
        why = ("WASM_MEM64=0 (operator switch)" if not _MEM64_ENV_ENABLED
               else "this wasmtime cannot run memory64")
        rec["status"] = "failed"
        rec["error"] = (f"app is a memory64 component but this box cannot "
                        f"serve it ({why}); a mem64-capable enclave must claim it")
        return rec
    if needs_cm64 and not _cm64_supported():
        rec["status"] = "failed"
        rec["error"] = ("app is a memory64 component but this wasmtime cannot run a 64-bit "
                        "canonical memory (no component-model-memory64); a newer engine must claim it")
        return rec
    if needs_cm64 and mem_mb_raw > rec["mem_mb"]:
        rec["mem_mb"] = mem_mb_raw

    # per-deployment config: the JSON the guest receives as ENCLAVE_CONFIG. The
    # supervisor passes it one of two ways, both read off the chain record:
    #   config     - inline (<= 4096 bytes on-chain): the version's own field,
    #                or a deployer's inline override from the options envelope
    #   config_cid - fetched here and accepted only because the bytes re-hash to
    #                the CID the record names (see _resolve_config_cid); either
    #                the VERSION's configCid (catalog rev 7) or the DEPLOYER's
    #                (the options envelope's configCid — same split, deployment
    #                side, so an override can exceed that 4096-byte field)
    # Who chose the CID changes nothing here: the gateway is untrusted either
    # way, and the hash is what makes the bytes acceptable. A deployer-chosen
    # one reaches only its own deployment's guest, exactly like the inline
    # override it replaces.
    # Re-validated either way, so a malformed record fails the launch cleanly
    # rather than the tenant on first request. If both arrive the CID wins and
    # the inline field is ignored — which is precisely the envelope's split
    # (there `config` is the routing manifest, never the guest's document) and,
    # from the catalog, a pairing publishVersionCfg cannot produce.
    enclave_config = None
    if config_cid:
        try:
            enclave_config = _resolve_config_cid(config_cid)
        except ValueError as e:
            rec["status"], rec["error"] = "failed", str(e)
            return rec
    elif config:
        try:
            enclave_config = _validate_config(config)
        except ValueError as e:
            rec["status"], rec["error"] = "failed", str(e)
            return rec

    # per-deployment secrets: relay-staged owner env vars, handed over by the
    # supervisor. Values live only in this call chain and the guest env — the
    # record keeps a COUNT for the owner view, never a name or value.
    if secrets:
        try:
            secrets = _validate_secrets(secrets)
        except ValueError as e:
            rec["status"], rec["error"] = "failed", str(e)
            return rec
        rec["secretsCount"] = len(secrets)
    else:
        secrets = None

    # $NAME config placeholders resolve BEFORE any config consumer runs, so
    # substitution reaches everything the config feeds: the guest's
    # ENCLAVE_CONFIG, encVolumes creds fields, warmup, volume lists.
    if enclave_config and secrets:
        enclave_config = _subst_secrets(enclave_config, secrets)

    # Config drop: a private dir holding just the RESOLVED config JSON (post
    # substitution — the guest must see the same text either way), preopened as
    # the guest's /config. Staged here, after every transform, so the file and
    # ENCLAVE_CONFIG can never disagree.
    #
    # Separate from /data on purpose: /data is the app's own writable scratch
    # with a quota, and the config must not be something the app can fill up,
    # delete, or lose to a storage sweep. Unlike /data, a failure here is FATAL
    # for a config past the env ceiling — there is no other channel — which
    # _build_cmd enforces; below the ceiling the env var still carries it.
    cfgdir, cfg_bytes = None, 0
    if enclave_config:
        cand = FS_DIR / f"{vid}-cfg"
        try:
            cand.mkdir(parents=True, exist_ok=True)
            cand.chmod(0o700)
            # 0400, and written before the preopen exists: a resolved config can
            # hold substituted secrets, the same exposure class as the argv value
            f = cand / CONFIG_FILE_NAME
            f.write_text(enclave_config, encoding="utf-8")
            f.chmod(0o400)
            cfg_bytes = f.stat().st_size
            # 0500 LAST: the guest's /config is configuration, not scratch, and
            # dropping the write bit is what actually makes it read-only —
            # wasmtime's --dir has no read-only mode, so like the model volumes
            # this leans on the filesystem. _audit_storage is the backstop for a
            # runtime running as root, where the mode bits are advisory.
            cand.chmod(0o500)
            cfgdir = cand
        except OSError as e:
            # a failure AFTER write_text leaves a secret-substituted file on
            # disk that nothing would ever clean up (_cfgdir stays None, so
            # teardown skips it) — remove the whole dir on any partial stage
            _rm_tree_rw(cand)
            cfgdir, cfg_bytes = None, 0
            print(f"[cfg] {vid} could not stage the config dir: {e}", flush=True)
    rec["_cfgdir"] = str(cfgdir) if cfgdir else None
    rec["_cfgBytes"] = cfg_bytes

    # attached model volumes: the request may name them two ways - an explicit
    # /vms `volumes` list (direct callers) and/or a `volumes` array in the
    # version's config JSON (owner-approved with the version; how catalog apps
    # attach volumes). Union both. A wanted volume that isn't attached (or is
    # still a bare mount point - _model_volumes refuses those) gets a BOUNDED
    # WAIT first: at node boot the supervisor resumes leased deployments while
    # the Modelwrap dm-verity mounts are still landing, and launching in that
    # window would bake a tenant whose model can never load (the wasmtime
    # graph registry is sealed at process boot). Beyond the wait the launch
    # fails with a clear reason - the supervisor backs off and retries, so a
    # slow mount delays the deployment instead of breaking it. Stays well
    # inside the supervisor's SPAWN_TIMEOUT_MS (300s) minus READY_SECS.
    want = list(volumes or [])
    if enclave_config:
        try:
            cfg_vols = json.loads(enclave_config).get("volumes")
            if isinstance(cfg_vols, list):
                want += cfg_vols
        except Exception:
            pass
    vol_mounts = {}
    if want:
        wanted = []
        for vname in want:
            vname = str(vname).strip()
            if vname and vname not in wanted:
                wanted.append(vname)
        have = _model_volumes()
        missing = [v for v in wanted if v not in have]
        if missing:
            print(f"[vol] {vid} waiting up to {VOL_READY_SECS:.0f}s for volume(s) "
                  f"{', '.join(missing)} to finish mounting", flush=True)
            deadline = time.time() + VOL_READY_SECS
            while missing and time.time() < deadline:
                time.sleep(2)
                have = _model_volumes()
                missing = [v for v in wanted if v not in have]
        if missing:
            rec["status"], rec["error"] = "failed", (
                f"volume(s) {', '.join(missing)} not attached to this enclave after "
                f"{VOL_READY_SECS:.0f}s (available: {', '.join(sorted(have)) or 'none'}) - "
                f"still mounting, or not in this enclave's tinfoil-config")
            return rec
        for vname in wanted:
            vol_mounts[vname] = have[vname]["path"]
    rec["volumes"] = list(vol_mounts.keys())

    # S3-backed volumes (rclone over S3, crypt-layered unless "encrypted":
    # false): stage an EMPTY dir per volume and spawn right away - the app
    # itself (or anything holding the per-deployment token) unlocks over
    # loopback and the contents appear under the already-preopened
    # /enc/<name>. Unlike /data, a failure to stage is a failed LAUNCH: an
    # app deployed around a volume must not silently run without the mount.
    enc = None
    if enclave_config:
        try:
            enc_specs = _parse_enc_volumes(json.loads(enclave_config))
        except ValueError as e:
            rec["status"], rec["error"] = "failed", str(e)
            return rec
        if enc_specs:
            base = ENC_DIR / vid
            try:
                vols = {}
                for spec in enc_specs:
                    d = base / spec["name"]
                    d.mkdir(parents=True, exist_ok=True)
                    vols[spec["name"]] = {
                        "spec": spec, "dir": str(d), "env": None,
                        "pub": {"name": spec["name"], "status": "locked", "error": None,
                                "bytes": 0, "maxMb": spec["maxMb"], "readOnly": spec["readOnly"],
                                "endpoint": spec["endpoint"], "bucket": spec["bucket"],
                                "path": spec["path"], "unlock": spec["unlock"],
                                "keyId": spec["keyId"], "encrypted": spec["encrypted"]}}
            except OSError as e:
                rec["status"], rec["error"] = "failed", f"encrypted volume staging failed: {e}"
                shutil.rmtree(base, ignore_errors=True)
                return rec
            rec["_enc"] = {"token": os.urandom(24).hex(), "dir": str(base), "vols": vols}
            rec["encVolumes"] = _enc_public(rec)
            enc = ({name: v["dir"] for name, v in vols.items()},
                   f"http://{HOST_IP}:{PORT}/encvol/{vid}", rec["_enc"]["token"])

    # OPT-IN RAM-budget accounting (default off — no admission change). Charge
    # this deployment's linear memory + /data cap + encrypted-volume caps
    # against node RAM and refuse if the fleet SUM would oversubscribe it. This
    # bounds the tmpfs-OOM window at admission WITHOUT the measure-and-kill
    # audit ever having to kill a legitimate app mid-write.
    if ACCOUNT_STORAGE_RAM:
        new_mb = _rec_ram_mb(rec)
        with _lock:
            committed = sum(_rec_ram_mb(r) for r in _apps.values()
                            if r["id"] != vid and r["status"] in ("starting", "running"))
        budget_mb = int(NODE_RAM_GB * 1024 * RAM_ACCT_HEADROOM)
        if committed + new_mb > budget_mb:
            rec["status"], rec["error"] = "failed", (
                f"insufficient RAM budget: this deployment reserves {new_mb} MB (linear memory + "
                f"/data + encrypted-volume caps); {committed} MB of a {budget_mb} MB ceiling is already "
                f"committed by live tenants (their reservations plus "
                f"{_nn_resident_bytes() // (1 << 20)} MB of model weights they hold resident) "
                f"(WASM_ACCOUNT_STORAGE_RAM)")
            _rm_fsdir(rec)
            with _lock:
                _apps.pop(vid, None)
            return rec

    ctx = {"pspec": pspec, "wasm": wasm, "port": port, "port_map": port_map, "fsdir": fsdir,
           "nn": nn, "enclave_config": enclave_config, "vol_mounts": vol_mounts, "gpu_share": gpu_share,
           "log_path": log_path, "egress": egress, "enc": enc, "secrets": secrets,
           "hosts": _validate_hosts(hosts), "wasi": contract["wasi"],
           "threads": needs_threads, "set": needs_set,
           "cm64": needs_cm64}
    return _spawn_and_wait(rec, ctx)


def _warmup_path(enclave_config) -> str:
    """The app config's optional `warmup` key: a path the manager GETs ONCE,
    in the background, the moment the app's port opens - so a model-serving
    app pulls its weights into device memory at DEPLOYMENT BOOT instead of on
    the first visitor (llm-chat ships "warmup": "/warmup"). Serve-mode apps
    only (it is an HTTP request). Absent/malformed = no poke."""
    if not enclave_config:
        return ""
    try:
        p = json.loads(enclave_config).get("warmup")
    except Exception:
        return ""
    if isinstance(p, str) and p.startswith("/") and len(p) <= 128:
        return p
    return ""


def _fire_warmup(host_port: int, path: str, log_path):
    """Fire-and-forget GET from a daemon thread, long timeout (WARMUP_SECS) -
    a cold model load is legitimately slow and holding the launch for it would
    blow the adopt deadline. The outcome lands in the tenant's own log."""
    def run():
        url = f"http://{HOST_IP}:{host_port}{path}"
        try:
            req = urllib.request.Request(url, headers={"user-agent": "wasm-manager-warmup"})
            with urllib.request.urlopen(req, timeout=WARMUP_SECS) as resp:
                body = resp.read(512)
                msg = f"[warmup] GET {path} -> {resp.status} {body[:200]!r}"
        except Exception as e:                                       # noqa: BLE001
            msg = f"[warmup] GET {path} failed: {e}"
        try:
            with open(log_path, "ab") as f:
                f.write(msg.encode() + b"\n")
        except OSError:
            print(msg, flush=True)
    threading.Thread(target=run, daemon=True, name="warmup").start()


_NN_FAIL_RE = re.compile(r"wasi-nn: preload of (\S+) FAILED \((.*)\); graph skipped")


def _watch_preloads(rec, log_path):
    """Hold a serving tenant's boot to its preload plan (rec._nnStages). The
    patched wasmtime preloads -S nn-graph AFTER binding the port, so 'running'
    flips while the graphs are still lifting into VRAM - and a graph that
    fails loads NEVER (skip-on-failure keeps the process up, the guest gets
    load_by_name NotFound forever). This watcher tails the tenant log for the
    preload markers: per-graph FAILED lines land in rec.nnFailed (apps and the
    console read it), and when EVERY expected graph failed - or the preload
    never finishes - the tenant is a zombie that can only hand out NotFound,
    so it is killed and failed for the supervisor's bounded relaunch (3
    deaths, then the lease is handed back with the reason on the record).
    Bails out quietly when the toolchain predates the preload markers."""
    stages = rec.get("_nnStages") or {}
    if not stages or MOCK:
        return
    proc, log_p = rec.get("_proc"), pathlib.Path(log_path)
    def run():
        pos = 0
        seen_marker = False
        failed = {}                          # "kind::stagedir" -> error text
        t0 = time.time()
        while True:
            time.sleep(2)
            if proc is None or proc.poll() is not None:
                return                       # died: the crash paths own it
            if rec.get("_proc") is not proc or rec["status"] != "running":
                return                       # superseded or already failed
            try:
                with open(log_p, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except OSError:
                chunk = b""
            for line in chunk.decode(errors="replace").splitlines():
                if "wasi-nn graph(s)..." in line:
                    seen_marker = True
                m = _NN_FAIL_RE.search(line)
                if m:
                    failed[m.group(1)] = m.group(2)
                if "wasi-nn graph preload done" in line:
                    bad = {n: failed[s] for n, s in stages.items() if s in failed}
                    if bad:
                        rec["nnFailed"] = bad
                        print(f"[nn-graph] {rec['id']} preload FAILED for "
                              f"{', '.join(sorted(bad))}", flush=True)
                    if bad and set(bad) >= set(stages):
                        rec["status"] = "failed"
                        rec["error"] = ("wasi-nn boot preload failed for every model: "
                                        + "; ".join(f"{n}: {e}" for n, e in sorted(bad.items())))
                        _kill(rec)
                    return
            if not seen_marker and time.time() - t0 > 30:
                return                       # toolchain predates the markers
            if time.time() - t0 > NN_PRELOAD_VERIFY_SECS:
                rec["status"] = "failed"
                rec["error"] = (f"wasi-nn boot preload did not finish within "
                                f"{NN_PRELOAD_VERIFY_SECS:.0f}s (expected: {', '.join(sorted(stages))})")
                _kill(rec)
                return
    threading.Thread(target=run, daemon=True, name="nn-preload-watch").start()


def _spawn_and_wait(rec, ctx):
    """Build the wasmtime command from a prepared context and spawn it, waiting
    for readiness."""
    pspec, wasm, port, port_map, fsdir, nn, enclave_config, vol_mounts, gpu_share, log_path = (
        ctx["pspec"], ctx["wasm"], ctx["port"], ctx["port_map"], ctx["fsdir"], ctx["nn"],
        ctx["enclave_config"], ctx["vol_mounts"], ctx["gpu_share"], ctx["log_path"])
    # the CPU share is the preload budget on a CPU-only node (see _build_cmd);
    # rec is the record launch() built, so read it from there rather than
    # widening the ctx contract
    cpu_share = float(rec.get("cpuShare") or 0)
    egress = ctx.get("egress", "")
    # enclave transparent egress (phase 2): if the supervisor enabled egress (the
    # per-deployment socks5h URL rides `egress`) AND this toolchain carries the
    # -S egress shim, make it TRANSPARENT — the endpoint goes on the wasmtime
    # cmdline and the SOCKS credential into the process env (guest-invisible,
    # host-process-env only, never the guest). On older toolchains _egress_supported()
    # is False and we fall back to phase-1: the guest-visible ENCLAVE_EGRESS only,
    # with raw -Sinherit-network still granted in run mode.
    egress_transparent, egress_env = None, {}
    if egress and _egress_supported():
        parsed = _parse_egress_url(egress)
        if parsed:
            egress_transparent = parsed["endpoint"]
            egress_env["ENCLAVE_EGRESS_CRED"] = parsed["cred"]
    # `-W max-memory-size` caps the guest's linear memory (the only RAM a tenant
    # can grow) - the real per-app memory ceiling, enforced by the runtime.
    mem_bytes = max(rec["mem_mb"], 1) * 1024 * 1024
    nn_report = {}
    # _build_cmd refuses rather than assembling a command it knows is wrong (a
    # config past the env ceiling with no file channel). Turn that into a FAILED
    # record like every other launch refusal — uncaught it would escape to the
    # HTTP handler as a traceback and a dropped connection, and the supervisor
    # would see a transport error instead of the reason.
    try:
        cmd, host_port, wait_ports = _build_cmd(pspec, wasm, port, mem_bytes, port_map, fsdir, nn,
                                                enclave_config, vol_mounts, egress, egress_transparent,
                                                ctx.get("enc"), gpu_share=gpu_share, nn_report=nn_report,
                                                secrets=ctx.get("secrets"), cpu_share=cpu_share,
                                                nn_resident_other=_nn_resident_bytes(exclude=rec["id"]),
                                                hosts=ctx.get("hosts", ""), wasi_contract=ctx.get("wasi"),
                                                shielded_vram_gb=float((rec.get("shielded") or {}).get("vramGb") or 0),
                                                threads=ctx.get("threads", False),
                                                set_threads=ctx.get("set", False),
                                                cfgdir=rec.get("_cfgdir"),
                                                cm64=ctx.get("cm64", False))
    except ValueError as e:
        rec["status"], rec["error"] = "failed", str(e)
        print(f"[launch] {rec['id']} refused: {e}", flush=True)
        return rec
    # the preload plan, public on the record: what the boot preload will
    # attempt (nnPreloads) and why the rest won't (nnSkipped). The watchdog
    # sweep compares this against what a launch would emit NOW; the log
    # verifier holds the actual boot to it.
    rec["nnPreloads"] = nn_report.get("emitted", [])
    rec["nnSkipped"] = nn_report.get("skipped", {})
    rec["_nnStages"] = nn_report.get("stages", {})
    # Weights this tenant holds in node RAM - part of its RAM reservation from
    # here on, so /capacity stops advertising a resident model as free memory.
    # Re-read on EVERY spawn (a restart can preload a different set).
    rec["nnResidentMb"] = int(nn_report.get("residentBytes", 0)) // (1 << 20)
    rec["hostPort"] = host_port
    rec["endpoint"] = f"http://{HOST_IP}:{host_port}" if host_port else None
    # GPU tenants: the wasmtime process itself is the CUDA process (ORT holds the
    # context), so the MPS caps go in ITS environment (SM% + VRAM from the share).
    env = None
    shielded_spec = rec.get("shielded")
    if gpu_share > 0 and shielded_spec:
        # The card is on the untrusted host: no device to cap, no MPS pipe to
        # join. Checked BEFORE the CUDA branch because NODE_HAS_GPU is false here
        # and the two must never both apply.
        # vol_mounts is {name: host_path}, not a list of records. Calibration is
        # per MODEL and this tenant attaches one volume in the shielded case; take
        # the first NAME in attach order rather than indexing a dict by 0.
        vol = next(iter(vol_mounts), "") if vol_mounts else ""
        try:
            if shielded_spec.get("pooled") and not _shielded_pool_available():
                raise ValueError("shielded model-layer backend is unavailable")
            env = _shielded_tenant_env(shielded_spec, vol)
        except (ValueError, TypeError, OverflowError) as e:
            rec["status"], rec["error"] = "failed", str(e)
            return rec
        rec["shieldedEndpoint"] = shielded_spec.get("endpoint")
        # Same weighted turns as the CUDA branch below, same queue: shielded
        # tenants all funnel into ONE host worker whose g_gpu mutex serializes
        # them FCFS, so the arbiter is what makes a 50% share worth more than a
        # 5% one under contention. The toolchain client dials at launch (that
        # is what the capability probe observes) but only takes turns once its
        # gpu flag knows SHIELDED_HOST graphs are GPU graphs — armed now,
        # active on the wasmtime repin that carries that gate, fail-open
        # (today's unarbitrated behavior) everywhere in between.
        _nn_arb_arm(env, rec, gpu_share)
        print(f"[shielded] {rec['id']}: gpuShare={gpu_share} served by the shielded card at "
              f"{shielded_spec.get('endpoint')}"
              + (f", calibration for {vol}" if env.get("SHIELDED_CALIB") else ", NO calibration")
              + (", arbited" if rec.get("nnArbiter") else ""),
              flush=True)
        _shielded_profile_tail(rec)
    elif nn and NODE_HAS_GPU and gpu_share > 0:
        # MPS caps belong to tenants that BOUGHT a card slice. A 0-GPU nn
        # tenant gets none - on a CPU box there is no card at all, and on a GPU
        # box it runs on cores by choice, so handing it CUDA_MPS_* (SM 1%, a
        # 1 MB pinned limit computed from a zero GPU share) would cap a card it
        # never touches and would strand it the moment ggml probed for one.
        env = _nn_tenant_env(gpu_share, pinned=_NN_PROBE.get("mode") != "nopin")
        rec["mpsPct"] = max(1, round(gpu_share * 100))
        # Work-conserving fair share (see the nn-arb section): the tenant's
        # runtime takes weighted turns on the card instead of living inside a
        # static SM slice. Grant weight = the RECORD's gpuShare (re-checked by
        # the arbiter at hello). All wasi-nn tenants share queue "0" today —
        # per-card queues become a launcher-side env change if per-tenant card
        # packing lands.
        _nn_arb_arm(env, rec, gpu_share)
        if rec.get("nnArbiter"):
            rec["mpsPct"] = NN_ARB_SM_PCT
        # sdcpp text-encoder placement (wasm/sd-shim, ENCLAVE_SD_TE_ON_CPU):
        # an explicit deployment-config `sdTeOnCpu` wins; else AUTO - when the
        # attached SD volumes' resident weights plus the ~3 GB 1024px working
        # set exceed the share, run the text encoder on the CPU. Conditioning
        # happens ONCE per prompt (a few seconds of CPU prefill on the
        # Qwen3-4B / Qwen2.5-VL encoders), and the alternative on a too-small
        # share is an sd.cpp CUDA OOM mid-pipeline, which ABORTS the tenant -
        # measured live 2026-07-18: z-image on a 12 GB share crashed the
        # deployment on every 512px+ request. Weights SUM because every
        # preloaded SD volume is resident; a node-level env is inherited
        # untouched when neither the config flag nor the auto rule applies.
        sd_bytes = sum(_staged_bytes(pathlib.Path(hp))
                       for n, hp in (vol_mounts or {}).items() if n in _SD_VOLUMES)
        te_flag = None
        if enclave_config:
            try:
                te_flag = json.loads(enclave_config).get("sdTeOnCpu")
            except (ValueError, AttributeError):
                te_flag = None
        if te_flag is None and sd_bytes and gpu_share > 0:
            vram = gpu_share * GPU_VRAM_GB * (1 << 30)
            if sd_bytes + SD_TE_AUTO_HEADROOM > vram:
                te_flag = True
                print(f"[sd] '{name}': te-on-cpu AUTO - {sd_bytes / 2**30:.1f} GB of SD "
                      f"weights + ~{SD_TE_AUTO_HEADROOM / 2**30:.0f} GB working set exceed "
                      f"the {vram / 2**30:.1f} GB share; text encoder runs on the CPU",
                      flush=True)
        if te_flag is not None:
            env["ENCLAVE_SD_TE_ON_CPU"] = "1" if te_flag else "0"
        # Fused-attention quarantine, PER VOLUME: ENCLAVE_ONNX_UNFUSED_VOLUMES
        # names model volumes whose ONNX sessions must not use ORT's fused
        # attention family - flash, memory-efficient AND the TRT fused/cross/
        # flash kernels (one step beyond ENCLAVE_FUSED_ATTENTION=0, which
        # leaves TRT on) - falling back to the unfused MATH path. The switches
        # are process-wide ORT envs, but each deployment is its own wasmtime
        # process and a quarantined volume's graphs are that process's ONNX
        # sessions, so scoping by ATTACHED VOLUME is per-model in practice;
        # every other deployment keeps the fused kernels. First user:
        # sd-turbo (Olive fp16 export: UNet epsilon 100% NaN -> black images
        # under MPS on sm_90 with the defaults, seen live 2026-07-14).
        quarantined = {v.strip()
                       for v in os.environ.get("ENCLAVE_ONNX_UNFUSED_VOLUMES", "").split(",")
                       if v.strip()}
        if quarantined & set((vol_mounts or {}).keys()):
            for k in ("ORT_DISABLE_FLASH_ATTENTION",
                      "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION",
                      "ORT_DISABLE_FUSED_ATTENTION",
                      "ORT_DISABLE_FUSED_CROSS_ATTENTION",
                      "ORT_DISABLE_TRT_FLASH_ATTENTION"):
                env.setdefault(k, "1")
            # basic keeps Level3 from re-fusing decomposed patterns into the
            # kernels we just disabled (moot for pre-fused Olive graphs,
            # load-bearing for plain exports)
            env.setdefault("ORT_GRAPH_OPT_LEVEL", "basic")
    elif nn and NODE_HAS_GPU:
        # A 0-GPU tenant on a GPU box: wasi-nn without the card. The comment
        # above this launch path promises such a tenant "runs on cores exactly
        # as it would on a CPU box" - _no_card_env is what makes that true by
        # construction rather than by the backends happening to default to CPU.
        env = _no_card_env()
    # CPU parallelism is a property of EVERY tenant, GPU or not, so it lands
    # after the MPS/no-card branches above (each of which REPLACES `env`).
    #
    # Inert on today's fleet: the shared-everything-threads intrinsics that
    # read this are in wasm/wasmtime-set-threads.patch.wip, deliberately not in
    # Dockerfile.wasmtime's chain. It is set NOW because it has to be in place
    # BEFORE they land — absent it the engine falls back to the node's core
    # count, so a tenant's `thread.available_parallelism` answer and its SET
    # worker-thread ceiling would both be sized from hardware it does not own.
    # OS threads are a node-wide kernel resource that cpu.weight cannot bound,
    # which is why that ceiling has to track the purchased share.
    # See docs/wasm-parallelism.md.
    env = dict(os.environ) if env is None else env
    env["ENCLAVE_AVAILABLE_PARALLELISM"] = str(_available_parallelism_for(cpu_share))
    rec["availableParallelism"] = _available_parallelism_for(cpu_share)
    # (the guest's own memory ceiling rides as a --env flag in _build_cmd,
    # not here: this env is the wasmtime PROCESS's, which the guest never
    # inherits — no -Sinherit-env — so a guest-facing value set here would
    # reach nothing.)
    if nn and enclave_config:
        nt = _nn_threads_for(enclave_config, cpu_share)
        if nt is not None:
            env["ENCLAVE_GGML_N_THREADS"] = str(nt)
        # Optional bounded wait for a privacy-pad batch already in flight.
        # No extra threads, CPU allocation, or pad reuse. Zero disables it.
        pw = _nn_cfg_int(enclave_config, "nnShieldedPadWaitUs", 0, 50000)
        if pw is not None:
            env["SHIELDED_PAD_WAIT_US"] = str(pw)
        # Recurrent-snapshot depth for speculative rewind (the shim's
        # ENCLAVE_GGML_N_RS_SEQ, read at ggml server-context creation):
        # deployment-config `nnRsSeq`, wasmtime PROCESS env like the MPS
        # caps. Per-deployment because each unit of depth costs one full
        # recurrent-state copy in the tenant's own VRAM/RAM share, and
        # absent/0 leaves the context byte-for-byte unchanged - so only the
        # deployment that wants rewind-based speculation pays for it.
        rs = _nn_cfg_int(enclave_config, "nnRsSeq", 1, 16)
        if rs is not None:
            env = env if env is not None else dict(os.environ)
            env["ENCLAVE_GGML_N_RS_SEQ"] = str(rs)
        # Context-window override, same pattern: `nnCtx` shrinks (or grows,
        # within the node ceiling) THIS deployment's unified KV pool. The
        # trade is the tenant's own to make - a smaller window frees VRAM
        # for things like snapshot depth or an MTP head, a larger one buys
        # longer conversations at the price of pool memory. The guest-side
        # forward in _build_cmd quotes the same override so app fit math
        # prices what the host actually allocated.
        nc = _nn_cfg_int(enclave_config, "nnCtx", 8192, 262144)
        if nc is not None:
            env = env if env is not None else dict(os.environ)
            env["ENCLAVE_GGML_N_CTX"] = str(nc)
        # MTP-head opt-out (`nnLoadMtp: false` -> the shim skips loading the
        # nextn tensors, reclaiming their VRAM for deployments that draft
        # via prompt-lookup or not at all). Engines predating the env read
        # ignore it - the knob is inert until the next toolchain cut, the
        # same shipping pattern as every other engine knob here.
        try:
            lmtp = json.loads(enclave_config).get("nnLoadMtp")
        except (ValueError, AttributeError):
            lmtp = None
        if lmtp is False:
            env = env if env is not None else dict(os.environ)
            env["ENCLAVE_GGML_LOAD_MTP"] = "0"
        # MTP device-sampling (`nnMtpDevSample: true` -> the head's draft steps
        # are sampled ON DEVICE, so llama never copies the 248K-vocab logits
        # row to host; that copy measured 77% of a head step, 3.67 -> 0.77 ms
        # on a 9b). It suppresses the p_min confidence gate, which needs the
        # raw row - harmless at draft_tokens 1, a real change above that.
        # Inert on engines predating the env read, like every knob here.
        try:
            mds = json.loads(enclave_config).get("nnMtpDevSample")
        except (ValueError, AttributeError):
            mds = None
        if mds is True:
            env = env if env is not None else dict(os.environ)
            env["ENCLAVE_MTP_DEV_SAMPLE"] = "1"
        # Device top-k (`nnDevTopk: true|<K>` -> the serving context arms a
        # [top_k(K)] backend sampler chain per sequence; small decodes then
        # return top-K ids+logits computed ON DEVICE instead of extracting
        # full 248K-vocab rows through the CC-forced-synchronous D2H and
        # scanning them on vCPUs - ~2.5 ms of a k=1 speculative round.
        # true = 256 (the app's HOST_TOPK). Prefill and wide batches keep
        # the classic path inside llama. Inert on engines predating mm26.
        try:
            dtk = json.loads(enclave_config).get("nnDevTopk")
        except (ValueError, AttributeError):
            dtk = None
        if dtk is True:
            dtk = 256
        if isinstance(dtk, int) and not isinstance(dtk, bool) and 16 <= dtk <= 4096:
            env = env if env is not None else dict(os.environ)
            env["ENCLAVE_GGML_DEV_TOPK"] = str(dtk)
        # Cohort batching window (`nnCohortMs`, mm30): how long a decode
        # leader holds the batch for the OTHER sequences that are stepping,
        # so the ubatch's participating-sequence set stays identical step to
        # step - the precondition for llama reusing its graph and hence for
        # ggml-cuda replaying a captured one. Without it two concurrent chats
        # each ran ~5x slower than one chat alone. Per-deployment because the
        # right window is the guest's per-token turnaround, which depends on
        # the app and on how many vCPUs the deployment bought; 0 restores the
        # pre-mm30 merge-whatever-is-queued behaviour. Inert on engines
        # predating mm30, like every knob here.
        cms = _nn_cfg_int(enclave_config, "nnCohortMs", 0, 200)
        if cms is not None:
            env = env if env is not None else dict(os.environ)
            env["ENCLAVE_GGML_COHORT_MS"] = str(cms)
    if egress_env:
        # SOCKS credential for transparent egress: wasmtime PROCESS env only
        # (guest-invisible — no -Sinherit-env,
        # and the token never touches the cmdline or a log line).
        env = env if env is not None else dict(os.environ)
        env.update(egress_env)
    logf = open(log_path, "wb")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf,
                                stderr=logf, preexec_fn=_preexec, env=env)
    except Exception as e:
        rec["status"], rec["error"] = "failed", f"spawn: {e}"
        logf.close()
        return rec
    rec["_proc"] = proc
    # CPU fair-share proportional to the purchased share (on by default) plus
    # the opt-in hard cap. See _apply_cpu_cgroup.
    cg = _apply_cpu_cgroup(rec["id"], proc.pid, float(rec.get("cpuShare") or 0))
    if cg:
        rec["_cgroup"] = str(cg)

    # readiness: a waitable port accepts, or the process dies first.
    # (udp-only apps have no waitable port: a short grace, then alive == running.)
    deadline = time.time() + (READY_SECS if wait_ports else 2.0)
    while time.time() < deadline:
        if proc.poll() is not None:
            rec["status"] = "failed"
            rec["error"] = _log_tail(log_path) or f"wasmtime exited {proc.returncode}"
            return rec
        if wait_ports and _port_open(wait_ports[0]):
            rec["status"] = "running"
            _audit_rec(rec)          # populate boundPorts right away (the bridge checks it)
            _watch_preloads(rec, log_path)
            wp = _warmup_path(enclave_config)
            if wp and pspec["serve"]:
                _fire_warmup(host_port, wp, log_path)
            return rec
        time.sleep(0.1)
    if wait_ports:
        # timed out: keep it but flag; supervisor can decide
        rec["status"] = "running" if _port_open(wait_ports[0]) else "failed"
    else:
        rec["status"] = "running" if proc.poll() is None else "failed"
    if rec["status"] == "failed":
        rec["error"] = rec.get("error") or ("did not open port in time; " + (_log_tail(log_path) or ""))
        _kill(rec)
    else:
        _audit_rec(rec)
        _watch_preloads(rec, log_path)
        wp = _warmup_path(enclave_config)
        if wp and pspec["serve"]:
            _fire_warmup(host_port, wp, log_path)
    return rec




# --- firewall enforcement: audit what each app actually bound ---------------- #
def _sock_inodes(pid) -> set:
    """The socket inodes `pid` holds open (/proc/<pid>/fd). Unprivileged: the
    manager spawned these processes, so it can read their fd table."""
    inodes = set()
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                ln = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if ln.startswith("socket:["):
                inodes.add(ln[8:-1])
    except OSError:
        return set()
    return inodes


def _bound_ports(pid) -> set:
    """Ports bound by `pid`: its socket inodes matched against
    /proc/net/{tcp,tcp6,udp,udp6}. TCP counts only LISTEN (st=0A); UDP counts
    unconnected binds."""
    inodes = _sock_inodes(pid)
    if not inodes:
        return set()
    ports = set()
    for name in ("tcp", "tcp6", "udp", "udp6"):
        try:
            lines = pathlib.Path(f"/proc/net/{name}").read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) < 10 or f[9] not in inodes:
                continue
            if name.startswith("tcp") and f[3] != "0A":     # LISTEN only
                continue
            ports.add(int(f[1].rsplit(":", 1)[1], 16))
    return ports


def _audit_rec(rec):
    """Enforce the per-port firewall on one app. The wasmtime sockets grant is
    all-or-nothing, so this is the fine-grained half: any bind in the policed
    space (<= PORT_MAX_DECL, or reserved) that wasn't assigned kills the app.
    Ephemeral outbound ports (32768+) are out of scope on purpose."""
    proc = rec.get("_proc")
    pid = getattr(proc, "pid", None)
    if pid is None or (hasattr(proc, "poll") and proc.poll() is not None):
        return
    bound = _bound_ports(pid)
    assigned = set(rec.get("_assigned") or [])
    rec["boundPorts"] = sorted(bound & assigned)   # actuals the bridge may target
    policed = {p for p in bound if p <= PORT_MAX_DECL or p in RESERVED_PORTS}
    extra = policed - assigned
    if extra:
        rec["status"] = "failed"
        rec["error"] = (f"firewall: bound unassigned port(s) {sorted(extra)}; app killed. "
                        f"Apps must bind the ACTUAL ports from ENCLAVE_PORTS (logical=actual), not hardcode.")
        print(f"[audit] {rec['id']} killed: unassigned ports {sorted(extra)}", flush=True)
        _kill(rec)


# --- cross-tenant loopback policing ----------------------------------------- #
def _is_loopback_hex(addr_hex: str) -> bool:
    """Is a /proc/net address a loopback one? The kernel prints each 4-byte word
    in HOST order, so IPv4 127.0.0.1 reads "0100007F" (last byte first) and ::1
    reads "00000000000000000000000001000000"; ::ffff:127.0.0.1 is the v4-mapped
    form. Anything else - including this box's own public address - is not
    loopback and is none of this audit's business (that traffic left through the
    egress front and was SSRF-checked there)."""
    a = (addr_hex or "").lower()
    if len(a) == 8:                                   # IPv4
        return a[6:8] == "7f"
    if len(a) == 32:                                  # IPv6
        if a == "00000000000000000000000001000000":   # ::1
            return True
        if a[16:24] == "0000ffff":                    # ::ffff:a.b.c.d
            return a[30:32] == "7f"
    return False


def _peer_ports(inodes: set, rows: list) -> set:
    """Remote LOOPBACK ports the sockets in `inodes` are connected to. `rows` is
    the /proc/net/{tcp,tcp6} body (header dropped). LISTEN sockets (st=0A) carry
    a null peer and are skipped; every other state counts, so a half-open or
    just-refused dial reads the same as a completed one - the intent is what is
    being policed. `inodes` comes from the tenant's own fd table, so a socket it
    has already closed (a TIME_WAIT with no owner) is invisible here: this sees
    connections that are still OPEN when the sweep runs."""
    out = set()
    for line in rows:
        f = line.split()
        if len(f) < 10 or f[9] not in inodes or f[3] == "0A":
            continue
        addr, _, port_hex = f[2].rpartition(":")
        if not _is_loopback_hex(addr):
            continue
        try:
            out.add(int(port_hex, 16))
        except ValueError:
            continue
    return out


def _tenant_port_owner() -> dict:
    """actual loopback port -> the id of the LIVE tenant it belongs to."""
    owner = {}
    with _lock:
        recs = [(r["id"], r.get("_assigned") or [], r.get("hostPort"))
                for r in _apps.values() if r["status"] in ("starting", "running")]
    for vid, assigned, host_port in recs:
        for p in list(assigned) + ([host_port] if host_port else []):
            try:
                owner[int(p)] = vid
            except (TypeError, ValueError):
                continue
    return owner


def _audit_peers(rec, port_owner: dict):
    """Kill a tenant that opened a loopback connection to ANOTHER tenant's port.

    Tenants share one network namespace, and the runtime's egress carve-out
    dials literal loopback direct (wasmtime-egress.patch), so the /x/:id proxy -
    where a PRIVATE deployment's owner check and the deployer's WAF live - can be
    walked around by connecting to the sibling's port on 127.0.0.1. Nothing
    legitimate does this: an app that wants another app calls its PUBLIC address,
    which leaves through the egress front. So a hit is a deliberate reach across
    the tenant boundary and the app dies, the same measure-and-kill answer the
    port firewall and the storage ceilings give.

    Deliberately not port-scan detection: a CONNECT to a live sibling's port is
    the thing itself, no heuristics and no threshold."""
    if not PEER_AUDIT_ON:
        return
    proc = rec.get("_proc")
    pid = getattr(proc, "pid", None)
    if pid is None or (hasattr(proc, "poll") and proc.poll() is not None):
        return
    foreign = {p: vid for p, vid in port_owner.items() if vid != rec["id"]}
    if not foreign:
        return
    inodes = _sock_inodes(pid)
    if not inodes:
        return
    rows = []
    for name in ("tcp", "tcp6"):
        try:
            rows += pathlib.Path(f"/proc/net/{name}").read_text().splitlines()[1:]
        except OSError:
            continue
    hits = sorted({(p, foreign[p]) for p in _peer_ports(inodes, rows) if p in foreign})
    if not hits:
        return
    who = ", ".join(f"{p} ({vid})" for p, vid in hits)
    rec["peerDials"] = [{"port": p, "deployment": vid} for p, vid in hits]
    print(f"[audit] {rec['id']} dialled another tenant on loopback: {who}"
          + ("" if PEER_AUDIT_KILL else " (WASM_PEER_AUDIT=warn: not killed)"), flush=True)
    if not PEER_AUDIT_KILL:
        return
    rec["status"] = "failed"
    rec["error"] = (f"isolation: connected to another deployment's loopback port ({who}); app killed. "
                    f"Reach other apps at their public address - the in-CVM loopback is this "
                    f"enclave's own service plane, not a tenant network.")
    _kill(rec)


def _dir_size(path) -> int:
    """Bytes used under `path` (files only, symlinks not followed)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f), follow_symlinks=False).st_size
            except OSError:
                pass
    return total


def _audit_storage(rec):
    """Enforce the per-app /data ceiling. We can't mount a sized tmpfs (the
    enclave blocks mounts), so -- like the port firewall -- we measure and kill
    on breach. `storageBytes` is refreshed each sweep so callers can see usage."""
    # /config is policed FIRST and on its own terms, because it exists even when
    # /data does not (storage_mb 0, or WASM_FS off) — so folding it into the
    # /data cap would leave it unmeasured in exactly the configurations that
    # have no cap at all. It is a preopen on the shared ramdisk, and the dir is
    # mode 0500 so a tenant should not be able to write there anyway; this is
    # the measure-and-kill backstop for when it can (a root-run runtime ignores
    # the mode bits). Its legitimate size is fixed at launch, so anything past
    # a slack margin over what we staged is the app writing into it.
    cfgdir, staged = rec.get("_cfgdir"), rec.get("_cfgBytes") or 0
    if cfgdir:
        cfg_used = _dir_size(cfgdir)
        if cfg_used > staged + CONFIG_DIR_SLACK_BYTES:
            rec["status"] = "failed"
            rec["error"] = (f"storage: /config is read-only and holds the app's configuration; "
                            f"it grew to {cfg_used} bytes (staged {staged}). App killed.")
            print(f"[audit] {rec['id']} killed: /config grew to {cfg_used} bytes (staged {staged})", flush=True)
            _kill(rec)
            return

    fsdir, cap_mb = rec.get("_fsdir"), rec.get("storageMb") or 0
    if not fsdir or cap_mb <= 0:
        return
    used = _dir_size(fsdir)
    rec["storageBytes"] = used
    if used > cap_mb * 1024 * 1024:
        rec["status"] = "failed"
        rec["error"] = (f"storage: /data used {used // (1024*1024)}MiB exceeds the {cap_mb}MiB cap; "
                        f"app killed. Raise storage_mb in the catalog if the app needs more scratch space.")
        print(f"[audit] {rec['id']} killed: storage {used} bytes > {cap_mb}MiB", flush=True)
        _kill(rec)


def _audit_enc(rec):
    """Enforce each encrypted volume's plaintext ceiling. The rclone pull is
    already capped by --max-transfer; this polices what the app WRITES into
    the live /enc/<name> preopen afterwards - same measure-and-kill shape as
    /data (a sized tmpfs is not available in the enclave)."""
    enc = rec.get("_enc")
    if not enc:
        return
    for name, vol in enc["vols"].items():
        if vol["pub"]["status"] == "locked":
            continue
        used = _dir_size(vol["dir"])
        vol["pub"]["bytes"] = used
        if used > vol["spec"]["maxMb"] * 1024 * 1024:
            rec["status"] = "failed"
            rec["error"] = (f"storage: volume '{name}' holds {used // (1024*1024)}MiB, over its "
                            f"{vol['spec']['maxMb']}MiB cap (encVolumes maxMb); app killed.")
            print(f"[audit] {rec['id']} killed: enc volume {name} {used} bytes > {vol['spec']['maxMb']}MiB", flush=True)
            _kill(rec)
            return


# skip reasons that stay true for the tenant's lifetime - never heal-restart
# for these (a budget or toolchain skip is a decision, not a race).
# Matched by PREFIX and deliberately kind-agnostic: the budget skip names the
# resource it ran out of (VRAM on a GPU node, RAM on a CPU-only one), and
# spelling one of them here would make the other look like a race - the sweep
# would restart the tenant, the launch would skip the volume again for the same
# arithmetic reason, forever. Seen live 2026-07-27, a restart every ~70s.
_NN_SKIP_TERMINAL = ("exceeds the ", "toolchain lacks")


def _heal_candidates(rec) -> list:
    """Wanted volumes this tenant did NOT preload for a reason that could
    have been a race (not on the cmdline, not a terminal skip, not a boot
    preload failure). Cheap - record lookups only, no filesystem."""
    if not rec.get("nn") or rec["status"] != "running" or MOCK:
        return []
    if time.time() - rec.get("createdAt", 0) < 60:
        return []              # boot grace: _watch_preloads owns early failures
    emitted = set(rec.get("nnPreloads") or [])
    skipped = rec.get("nnSkipped") or {}
    failed = rec.get("nnFailed") or {}
    return [v for v in (rec.get("volumes") or [])
            if v not in emitted and v not in failed
            and not str(skipped.get(v, "")).startswith(_NN_SKIP_TERMINAL)]


def _heal_preloads(rec, cands, vols):
    """The frozen-preload self-heal: wasmtime registers -S nn-graph ONLY at
    process start, so a tenant that booted while its model volume was still
    mounting can never serve that model - load_by_name() NotFound for its
    whole life (seen live 2026-07-18, qwen3.5-9b). launch() now refuses to
    boot into that state; this sweep is the backstop for any race that still
    slips through: when a wanted volume is preloadable NOW but wasn't in this
    tenant's boot plan, kill + fail the tenant - the supervisor's bounded
    relaunch (3 deaths) sends it back through the fixed launch path, which
    preloads properly or fails with the reason on the record."""
    ready = [v for v in cands
             if v in vols and (vols[v]["sd"] if v in _SD_VOLUMES
                               else (vols[v]["gguf"] or vols[v]["onnx"]))]
    if not ready:
        return
    print(f"[nn-graph] {rec['id']} volume(s) {', '.join(ready)} became preloadable "
          f"after boot; restarting the tenant to preload them", flush=True)
    rec["status"] = "failed"
    rec["error"] = (f"model volume(s) {', '.join(ready)} became ready after this tenant "
                    f"booted, and wasmtime preloads graphs only at process start - "
                    f"killed for relaunch so the model actually loads")
    _kill(rec)


def _lease_expired(now: float) -> list:
    """PURE. Which live tenants' leases have lapsed, given the clock and what we
    know about the supervisor. Returns [] whenever we must not act:

      - no heartbeat ever received  -> the lease is inert (old supervisor)
      - no heartbeat RECENTLY       -> silence is not evidence about any tenant

    Only when the supervisor is demonstrably talking does an unrenewed lease
    mean what it looks like: this tenant is one nobody owns any more.
    """
    if _lease_last_beat is None:
        return []
    if now - _lease_last_beat > TENANT_LEASE_SILENCE:
        return []
    return [r["id"] for r in _apps.values()
            if r["status"] in ("starting", "running")
            and float(r.get("leaseUntil") or 0) < now]


def _lease_reap():
    now = time.time()
    with _lock:
        dead = _lease_expired(now)
    for vid in dead:
        print(f"[lease] {vid}: lease expired - the supervisor stopped vouching for it; reaping "
              f"(its slice and any resident model weights go back to the pool)", flush=True)
        try:
            teardown(vid)
        except Exception as e:
            print(f"[lease] {vid}: teardown failed ({e}) - retrying next sweep", flush=True)


def _audit_sweep():
    while True:
        time.sleep(AUDIT_SECS)
        try:
            _lease_reap()
        except Exception:
            pass
        with _lock:
            recs = [r for r in _apps.values() if r["status"] == "running"]
        heal = [(r, c) for r in recs if (c := _heal_candidates(r))]
        if heal:
            try:
                vols = _model_volumes()      # one scan serves every tenant
                for r, c in heal:
                    _heal_preloads(r, c, vols)
            except Exception:
                pass
        port_owner = _tenant_port_owner() if PEER_AUDIT_ON else {}
        for r in recs:
            try:
                _audit_rec(r)
                if r["status"] == "running":
                    _audit_storage(r)
                if r["status"] == "running":
                    _audit_enc(r)
                if r["status"] == "running":
                    _audit_peers(r, port_owner)
            except Exception:
                pass


def _mock_server(port: int, vid: str):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"enclave-wasm-ok {vid}".encode())
        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer((HOST_IP, port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _log_tail(path, n=800) -> str:
    try:
        return path.read_text(errors="replace")[-n:].strip()
    except Exception:
        return ""


def _kill(rec):
    p = rec.get("_proc")
    if p is None:
        return
    try:
        if MOCK:
            p.shutdown()
            return
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def _rm_fsdir(rec):
    d = rec.get("_fsdir")
    if d:
        shutil.rmtree(d, ignore_errors=True)   # ephemeral scratch: nothing to preserve
    c = rec.get("_cfgdir")
    if c:
        # the staged config dies with the deployment: post-substitution it can
        # hold secrets, so it must not outlive the tenant that was entitled to
        # them. _rm_tree_rw because we made the dir read-only on purpose.
        _rm_tree_rw(c)
    enc = rec.get("_enc")
    if enc:
        # plaintext + any retained rclone credentials die with the deployment
        for vol in enc["vols"].values():
            vol["env"] = None
        shutil.rmtree(enc["dir"], ignore_errors=True)
    cg = rec.get("_cgroup")
    if cg:
        # per-tenant cgroup dir (opt-in CPU limits); removable only once empty,
        # i.e. after _kill reaped the process group.
        try:
            pathlib.Path(cg).rmdir()
        except OSError:
            pass




def teardown(vid: str) -> bool:
    with _lock:
        rec = _apps.pop(vid, None)
    if rec is None:
        return False
    _kill(rec)
    _rm_fsdir(rec)
    return True


def _refresh_status(rec: dict) -> None:
    """A tenant that died on its own (fatal signal, OOM-kill, crash) must not
    keep reporting "running": the supervisor routes traffic and RENEWS leases
    on this status (observed live: a SIGFPE'd app served ECONNREFUSED for an
    hour while its lease kept being paid)."""
    proc = rec.get("_proc")
    if rec.get("status") == "running" and proc is not None:
        code = proc.poll()
        if code is not None:
            rec["status"] = "failed"
            rec["error"] = (f"app process died: signal {-code}" if code < 0
                            else f"app process exited (code {code})")
            print(f"[audit] {rec['id']} died: exit={code}", flush=True)


def _public(rec: dict) -> dict:
    _refresh_status(rec)
    if rec.get("_enc"):
        _enc_public(rec)                    # refresh per-volume sizes in place
    return {k: v for k, v in rec.items() if not k.startswith("_")}


_ENC_ROUTE_RE = re.compile(r"^/encvol/([^/?]+)(?:/(unlock|sync|lock))?$")
# constant-time compares take BYTES here — see _authorized below
_b = lambda s: (s or "").encode("utf-8", "surrogateescape")


# ---- HTTP contract --------------------------------------------------------- #
class Handler(http.server.BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def log_message(self, *a):
        pass

    def _ctrl_authed(self) -> bool:
        """Control-plane gate (see VMMGR_TOKEN). Timing-safe; header X-Vmmgr-Token
        or Authorization: Bearer. Fail closed when no token is configured unless
        VMMGR_ALLOW_UNAUTHENTICATED is explicitly set."""
        if not VMMGR_TOKEN:
            return VMMGR_ALLOW_UNAUTH
        tok = self.headers.get("X-Vmmgr-Token") or ""
        if not tok:
            m = re.match(r"^Bearer\s+(\S+)$", self.headers.get("Authorization") or "")
            tok = m.group(1) if m else ""
        # bytes, not str: compare_digest raises TypeError on a str with any
        # character above U+007F, and headers arrive latin-1-decoded — one high
        # byte in the header would raise inside the auth check rather than fail it
        return hmac.compare_digest(_b(tok), _b(VMMGR_TOKEN))

    # --- encrypted volumes: the tenant plane ------------------------------- #
    # /encvol/<vid>[/<action>] is NOT control-plane: it authenticates with the
    # deployment's own token (ENCLAVE_ENC_TOKEN), which only the guest holds -
    # the same posture as the old /enc data plane. The password/credentials in
    # an unlock body exist in RAM for the duration of the request + the rclone
    # child's environment; they are never logged, never persisted.
    def _enc_route(self):
        """Match an /encvol route; returns (rec, action) after auth, or None
        after having already sent the error response."""
        m = _ENC_ROUTE_RE.match(self.path)
        if not m:
            return None
        vid, action = m.group(1), m.group(2)
        with _lock:
            rec = _apps.get(vid)
        enc = rec.get("_enc") if rec else None
        if not enc:
            self._json(404, {"error": "no such deployment or no encrypted volumes"})
            return None
        tok = self.headers.get("X-Enc-Token") or ""
        if not tok:
            b = re.match(r"^Bearer\s+(\S+)$", self.headers.get("Authorization") or "")
            tok = b.group(1) if b else ""
        # TENANT-REACHABLE, and that is the point of the bytes. ENCLAVE_ENC_API +
        # ENCLAVE_ENC_TOKEN are handed to the GUEST, and the wasmtime egress
        # patch carves loopback out of the SOCKS front precisely so an app can
        # dial this plane — so the header below is untrusted guest input, and a
        # str compare_digest would raise TypeError out of this auth check on any
        # byte above U+007F rather than returning false. The egress carve-out's
        # safety argument is literally "the in-CVM services are token-gated
        # fail-closed"; this is one of those gates. Keep it bytes.
        if not hmac.compare_digest(_b(tok), _b(enc["token"])):
            self._json(401, {"error": "volume token required"})
            return None
        return rec, action

    def _enc_post(self, rec, action):
        b = self._body()
        enc = rec["_enc"]
        name = str(b.get("name") or "").strip()
        vol = enc["vols"].get(name)
        if not vol:
            return self._json(404, {"error": f"no encrypted volume '{name}' on this deployment"})
        pub = vol["pub"]
        if action == "unlock":
            password = b.get("password")
            if vol["spec"]["encrypted"]:
                if not isinstance(password, str) or not password:
                    return self._json(400, {"error": "password required"})
            elif password or b.get("salt"):
                # a password on a plaintext store is a misconfiguration, not a
                # convenience: whoever sent it believes something is encrypted.
                return self._json(400, {"error": f"volume '{name}' is plain (\"encrypted\": false): "
                                                 f"it takes S3 credentials only, no password/salt"})
            creds = {"password": password, "salt": b.get("salt"),
                     "accessKeyId": b.get("accessKeyId"),
                     "secretAccessKey": b.get("secretAccessKey"),
                     "sessionToken": b.get("sessionToken")}
            with _lock:
                if pub["status"] in ("syncing", "pushing"):
                    return self._json(409, {"error": f"volume is busy ({pub['status']})"})
                pub["status"], pub["error"] = "syncing", None
                vol["env"] = None
            threading.Thread(target=_enc_unlock_worker, args=(rec, vol, creds), daemon=True).start()
            return self._json(202, {"name": name, "status": "syncing"})
        if action == "sync":
            with _lock:
                if pub["status"] != "unlocked":
                    return self._json(409, {"error": f"volume is {pub['status']}, not unlocked"})
                if vol["spec"]["readOnly"] or not vol["env"]:
                    return self._json(403, {"error": "read-only volume: no credentials retained for push"})
                pub["status"] = "pushing"
            threading.Thread(target=_enc_push_worker, args=(rec, vol), daemon=True).start()
            return self._json(202, {"name": name, "status": "pushing"})
        # lock: wipe the plaintext + drop retained credentials
        with _lock:
            if pub["status"] in ("syncing", "pushing"):
                return self._json(409, {"error": f"volume is busy ({pub['status']})"})
            _enc_wipe_dir(vol)
            vol["env"] = None
            pub["status"], pub["error"], pub["bytes"] = "locked", None, 0
        return self._json(200, {"name": name, "status": "locked"})



    def do_GET(self):
        if self.path == "/health":
            # Bare liveness is always open (supervisor probe). The detailed
            # fields below disclose capacity/models/GPU/probe internals; when
            # WASM_HEALTH_MINIMAL is on they are withheld from unauthenticated
            # callers (a tenant can reach loopback). Default off = unchanged.
            live = {"ok": True, "runtime": "wasmtime",
                    "version": _wasmtime_version(), "mock": MOCK}
            if HEALTH_MINIMAL and not self._ctrl_authed():
                return self._json(200, live)
            return self._json(200, {**live,
                                    "nn": _nn_available(),
                                    "nnProbe": dict(_NN_PROBE),
                                    "shieldedPool": _shielded_pool_available(),
                                    # false = this binary predates wasmtime-loopback.patch, so
                                    # tenants run WITHOUT the cross-tenant wall (the launcher
                                    # cannot pass a flag that would fail every launch outright)
                                    "loopbackWall": _loopback_flag_supported(),
                                    # this box serves WASIp3 components (binary probed AND
                                    # WASM_P3 not switched off) — the supervisor forwards it
                                    # to /availability and the claim gate keys on it
                                    "p3": _p3_active(),
                                    # this box serves cooperative-thread (🧵) guests: p3 live
                                    # AND the thread.new-indirect compile probe passed AND
                                    # WASM_COOP_THREADS not switched off — same forwarding,
                                    # same claim-gate role, for versions marked `threads`
                                    "coopThreads": _threads_active(),
                                    # this box serves shared-everything-thread (⚡) guests: the
                                    # thread.spawn-indirect compile probe passed AND
                                    # WASM_SET_THREADS not switched off. Unlike coopThreads it
                                    # does NOT require p3 (SET rides wasip2). Forwarded and
                                    # claim-gated for versions marked `set`.
                                    "set": _set_active(),
                                    # this box serves the >4 GiB guest class — memory64
                                    # components: BOTH the flagless memory64 probe and the
                                    # component-model memory64 probe passed, AND WASM_MEM64
                                    # is not switched off. Forwarded and claim-gated for
                                    # versions marked `mem64` (see _mem64_advertised: the
                                    # AND is what keeps a component off a box that could
                                    # only parse a bare 64-bit memory).
                                    "mem64": _mem64_advertised(),
                                    # catalog rev 7: this box resolves a version's configCid —
                                    # fetching the config from IPFS and accepting it only
                                    # because it re-hashes to the CID the approved record
                                    # names — and delivers past the argv ceiling by file.
                                    # Forwarded to /availability and claim-gated: an older
                                    # manager ignores the field and would serve the routing
                                    # manifest as the config, so this has to be sayable.
                                    "configCid": ipfs_fetch is not None,
                                    "configMaxBytes": CONFIG_MAX_BYTES,
                                    "configEnvMaxBytes": CONFIG_ENV_MAX_BYTES,
                                    **({"gpuVramGb": GPU_VRAM_GB, "gpuVramSource": GPU_VRAM_SRC,
                                        # request-level GPU fair-share (nn-arb): enabled = the
                                        # operator knob; probe = whether the toolchain's client
                                        # was proven (unproven = tenants keep hard MPS caps)
                                        "nnArbiter": _nn_arb_public()}
                                       if NODE_HAS_GPU else {}),
                                    "volumes": _volumes_public(),
                                    "capacity": _capacity()})
        if _ENC_ROUTE_RE.match(self.path):                 # tenant plane: own token
            hit = self._enc_route()
            if hit:
                rec, _action = hit
                with _lock:
                    vols = _enc_public(rec)
                self._json(200, {"id": rec["id"], "volumes": vols})
            return None
        if not self._ctrl_authed():
            return self._json(401, {"error": "control token required"})
        if self.path == "/capacity":
            return self._json(200, _capacity())
        # The attribution table for a VRAM incident, in one authenticated GET:
        # the device's own memory count, the per-PID compute-app list, the
        # reservation ledger and every tenant's pin. The 2026-08-18 hunt for
        # ~104 GiB of orphaned device memory had to be reconstructed from
        # share arithmetic because none of this was reachable from outside;
        # the supervisor now proxies it at /v1/admin/gpu (ADMIN_TOKEN).
        if self.path == "/gpu":
            with _lock:
                tenants = [{"id": r["id"], "name": r.get("name"),
                            "status": r.get("status"),
                            "gpuShare": float(r.get("gpuShare") or 0),
                            "pinnedGb": round(_rec_vram_gb(r), 2)}
                           for r in _apps.values()
                           if float(r.get("gpuShare") or 0) > 0]
            return self._json(200, {"dev": _gpu_dev_mem(), "procs": _gpu_dev_procs(),
                                    "ledger": _vram_budget(), "tenants": tenants,
                                    "bounce": _mps_bounce_result()})
        if self.path == "/volumes":
            return self._json(200, {"volumes": _volumes_public()})
        if self.path == "/catalog":
            cat = _load_catalog()
            return self._json(200, {"apps": [
                {"id": a["id"], "name": a.get("name", a["id"]),
                 "description": a.get("description", ""),
                 "vramMb": int(a.get("vram_mb", 0)),
                 "gpuGflops": int(a.get("gpu_gflops", 0)),
                 "memMb": int(a.get("mem_mb", 0)),
                 "cpuGflops": int(a.get("cpu_gflops", 0)),
                 "gpu": int(a.get("vram_mb", 0)) > 0 or int(a.get("gpu_gflops", 0)) > 0} for a in cat.values()]})
        if self.path == "/debug/env":
            return self._json(200, _debug_env())
        if self.path == "/vms":
            with _lock:
                return self._json(200, {"vms": [_public(r) for r in _apps.values()]})
        # Tenant log tail: the wasmtime process's stdout+stderr (stage markers,
        # panics, CUDA/ORT aborts). The supervisor exposes it owner-only at
        # /v1/deployments/:id/logs - a crashed app's last words are the owner's
        # ONLY debugging evidence on this backend.
        m = re.match(r"^/vms/([^/?]+)/logs(?:\?(.*))?$", self.path)
        if m:
            vid = m.group(1)
            q = dict(p.split("=", 1) for p in (m.group(2) or "").split("&") if "=" in p)
            try:
                tail = min(2000, max(1, int(q.get("tail") or 200)))
            except ValueError:
                tail = 200
            with _lock:
                rec = _apps.get(vid)
            if not rec:
                return self._json(404, {"error": "not found"})
            p = pathlib.Path(rec.get("_log") or str(LOG_DIR / f"{vid}.log"))
            try:
                lines = p.read_bytes().decode("utf-8", "replace").splitlines()[-tail:]
            except OSError:
                lines = []
            proc = rec.get("_proc")
            exit_code = proc.poll() if proc is not None else None
            return self._json(200, {"id": vid, "lines": lines,
                                    "exited": exit_code is not None,
                                    "exitCode": exit_code,
                                    "status": rec.get("status"), "error": rec.get("error")})
        if self.path.startswith("/vms/"):
            vid = self.path[len("/vms/"):]
            with _lock:
                rec = _apps.get(vid)
            return self._json(200, _public(rec)) if rec else self._json(404, {"error": "not found"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if _ENC_ROUTE_RE.match(self.path):                 # tenant plane: own token
            hit = self._enc_route()
            if hit:
                rec, action = hit
                if not action:
                    return self._json(405, {"error": "POST /encvol/<vid>/{unlock|sync|lock}"})
                self._enc_post(rec, action)
            return None
        if not self._ctrl_authed():
            return self._json(401, {"error": "control token required"})
        # Order an MPS bounce (operator lever, proxied by the supervisor at
        # POST /v1/admin/gpu/bounce-mps): the reclaim for device memory a dead
        # tenant generation's MPS servers still hold. 202 = the order is on
        # the shared volume; the daemon consumes it within seconds and its
        # cooldown refuses back-to-back orders. Every live GPU tenant's CUDA
        # context dies with the bounce - the supervisor respawns them.
        if self.path == "/gpu/bounce-mps":
            if not NODE_HAS_GPU or MOCK:
                return self._json(422, {"error": "no GPU on this node"})
            b = self._body()
            ok = _request_mps_bounce("operator: " + str(b.get("reason") or "admin api")[:300])
            return self._json(202 if ok else 500, {
                "requested": ok,
                "note": ("consumed by the mps-control daemon within seconds; every live GPU "
                         "tenant's CUDA context dies with the bounce and the supervisor "
                         "respawns them - a recovery action, not a free one"),
                "lastResult": _mps_bounce_result()})
        # Prefetch: resolve + verify + cache an app's bytes WITHOUT launching.
        # The supervisor calls this before claiming an on-chain deployment so
        # a lease is never burned racing a 100MB+ IPFS fetch against the spawn
        # window - after this, the launch's fetch is a local cache hit.
        if self.path == "/prefetch":
            b = self._body()
            ref = str(b.get("image") or b.get("app") or "").strip()
            if not ref.startswith("ipfs://"):
                return self._json(400, {"error": "prefetch takes an ipfs://<cid> app reference"})
            try:
                t0 = time.time()
                p = _resolve_wasm(ref)
                return self._json(200, {"ok": True, "bytes": p.stat().st_size,
                                        "seconds": round(time.time() - t0, 1)})
            except ValueError as e:
                return self._json(422, {"error": str(e)})
            except Exception as e:
                return self._json(502, {"error": f"prefetch failed: {e}"})
        # The supervisor's heartbeat: the ids it still legitimately owns. Every
        # named tenant's lease is extended; every live tenant it did NOT name is
        # left to lapse and _lease_reap tears it down. This is the whole
        # dead-man switch - the supervisor never has to successfully instruct a
        # teardown for one to happen, it only has to stop vouching, which is
        # also what a crashed or wedged supervisor does for free.
        if self.path == "/vms/lease":
            global _lease_last_beat
            b = self._body()
            ids = {str(x) for x in (b.get("ids") or []) if str(x)}
            now = time.time()
            ttl = TENANT_LEASE_TTL
            extended, unvouched = [], []
            with _lock:
                _lease_last_beat = now
                for r in _apps.values():
                    if r["status"] not in ("starting", "running"):
                        continue
                    key = str(r.get("name") or r["id"])
                    if key in ids:
                        r["leaseUntil"] = now + ttl
                        extended.append(key)
                    else:
                        unvouched.append({"id": key, "expiresIn": round(float(r.get("leaseUntil") or 0) - now, 1)})
            return self._json(200, {"extended": extended, "unvouched": unvouched, "ttlSec": ttl})
        if self.path != "/vms":
            return self._json(404, {"error": "not found"})
        b = self._body()
        app_ref = b.get("image") or b.get("app")
        if not app_ref:
            return self._json(400, {"error": "missing app reference (image)"})
        # deployments buy shares: cpuShare (the admission unit; "share" is the
        # legacy alias) sets the memory cap; gpuShare rides along for GPU
        # catalog apps (the card pool itself is the supervisor's allocator).
        # The app's catalog specs set minimums, checked below.
        def _share(*keys, default=0.0):
            for k in keys:
                if b.get(k) is not None:
                    try:
                        return min(max(float(b[k]), 0.0), 1.0)
                    except (TypeError, ValueError):
                        pass
            return default
        try:
            mem_mb = max(0, int(b.get("memMb") or 0))
        except (TypeError, ValueError):
            mem_mb = 0
        cpu_share = _share("cpuShare", "share",
                           default=min(1.0, mem_mb / (NODE_RAM_GB * 1024.0)) if mem_mb else 0.05)
        gpu_share = _share("gpuShare")
        # The two shares are INDEPENDENT (ledger rev 13). This used to 422 any
        # gpuShare under the cpuShare, mirroring the old create() rule; the
        # pools are separate (this card's VRAM+compute vs the node's vCPU+RAM)
        # and the supervisor reserves them separately, so a CPU-heavy GPU app
        # buying a sliver of card is a legal record and must launch here.
        # An in-place restart / version switch must never DUPLICATE a tenant:
        # the supervisor tears the old instance down before re-creating, but
        # that teardown is best-effort over HTTP. If it was missed, the OLD
        # process keeps its whole share pinned (a 27b GPU tenant holds
        # ~30 GiB of weights+KV) while routing follows the new record - the
        # replacement then fails context allocation against memory it cannot
        # see (live 2026-07-25). The process owner dedups by deployment name
        # as the backstop; before the capacity check, so the duplicate's own
        # reservation cannot 429 its replacement.
        name = str(b.get("name") or "").strip()
        if name:
            def _live(r):
                if r.get("status") in ("starting", "running"):
                    return True
                p = r.get("_proc")
                return p is not None and getattr(p, "poll", lambda: 0)() is None
            with _lock:
                dupes = [r["id"] for r in _apps.values() if r.get("name") == name and _live(r)]
            for old in dupes:
                print(f"[vms] {name}: reaping duplicate live tenant {old} before create", flush=True)
                teardown(old)
        if _used_cpu_share() + cpu_share > 1.0 + 1e-6:
            return self._json(429, {"error": "insufficient capacity", "capacity": _capacity()})
        # VRAM-reservation ledger: refuse to spawn a GPU tenant the device
        # cannot physically hold. Same 429 contract as the cpu pool - the
        # supervisor backs the claim off and the routing re-plans.
        if gpu_share > 0:
            v = _vram_budget()
            if v:
                ask_gb = gpu_share * GPU_VRAM_GB + VRAM_CTX_OVERHEAD_GB
                if ask_gb > v["vramFreeGb"] + 1e-6:
                    return self._json(429, {
                        "error": (f"insufficient device memory: this launch would pin {ask_gb:.1f} GB "
                                  f"but only {v['vramFreeGb']:.1f} GB of {v['vramBudgetGb']:.1f} GB is unreserved"),
                        "capacity": _capacity()})
                # The device's own count outranks the arithmetic when it says
                # LESS. Ledger-free counts only what THIS process handed out;
                # memory held by anything else - an orphaned tenant generation,
                # a wedged MPS server - is invisible to it and very visible to
                # the tenant that later dies allocating against it
                # (2026-08-18: a 51 GB spec passed every arithmetic check onto
                # a card with 35 GB physically free, loaded its weights, and
                # SIGABRT'd at the first lazy context allocation). Device-free
                # can only over-admit (a reserved-but-not-yet-loaded sibling
                # hasn't touched the card yet) - the ledger check above already
                # covers that direction, so the pair is safe from both sides.
                dev_free = v.get("vramDevFreeGb") if VRAM_DEV_GATE else None
                if dev_free is not None and ask_gb > dev_free + 1e-6:
                    print(f"[vms] refusing GPU launch for {name}: asks {ask_gb:.1f} GB but the "
                          f"device reports {dev_free:.1f} GB physically free (ledger thought "
                          f"{v['vramFreeGb']:.1f} GB; divergence {v.get('vramDivergenceGb')} GB)",
                          flush=True)
                    return self._json(429, {
                        "error": (f"insufficient device memory: this launch would pin {ask_gb:.1f} GB "
                                  f"but the device reports only {dev_free:.1f} GB physically free "
                                  f"(the ledger's {v['vramFreeGb']:.1f} GB says something untracked "
                                  f"holds the card)"),
                        "capacity": _capacity()})
        try:
            pspec = _parse_ports(b.get("ports") or [])
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        # No cross-tenant port conflicts by construction: declared ports are LOGICAL;
        # _alloc_ports gives each deployment its own actual bind (ENCLAVE_PORTS tells the
        # app which), so two tenants can both run "the tcp:5432 app" simultaneously.
        cat = _load_catalog()
        meta = cat.get(app_ref, {})
        min_vram = int(meta.get("vram_mb", 0))
        min_ggf = int(meta.get("gpu_gflops", 0))
        min_mem = int(meta.get("mem_mb", 0))
        min_cgf = int(meta.get("cpu_gflops", 0))
        if (min_vram > 0 or min_ggf > 0) and (not NODE_HAS_GPU or gpu_share <= 0):
            return self._json(422, {"error": f"app '{app_ref}' requires a GPU ({min_vram} MB VRAM / {min_ggf / 1000} TFLOPS); "
                                             + ("ask for GPU resources" if NODE_HAS_GPU else "this node is CPU-only")})
        # GPU tenants launch only after the CUDA/MPS probe passed: a hanging
        # CUDA init inside a tenant eats runtime threads until the whole app
        # wedges, so failing the deploy loudly here is the honest alternative.
        if gpu_share > 0 and NN_ENABLED and NODE_HAS_GPU and not MOCK and _NN_PROBE["state"] != "ok":
            msg = ("GPU interface warming up (CUDA readiness probe still running); retry shortly"
                   if _NN_PROBE["state"] == "probing"
                   else f"GPU interface unavailable on this node: {_NN_PROBE['detail'] or 'probe failed'}")
            return self._json(503, {"error": msg, "nnProbe": dict(_NN_PROBE)})
        if min_mem and (mem_mb or int(cpu_share * NODE_RAM_GB * 1024)) < min_mem:
            return self._json(422, {"error": f"app '{app_ref}' declares a minimum of {min_mem} MB RAM; the request asks for less"})
        # compute minimums: the ask arrives GPU in TFLOPS (gpuTflops) but CPU in
        # GFLOPS (cpuGflops) - a whole node is ~1000 GFLOPS, so TFLOPS is too
        # coarse a grain for CPU. cpuTflops is the legacy pre-GFLOPS field.
        def _num(k):
            try:
                return max(0.0, float(b.get(k) or 0))
            except (TypeError, ValueError):
                return 0.0
        ask_cgf = _num("cpuGflops") or _num("cpuTflops") * 1000
        if min_cgf and round(ask_cgf) < min_cgf:
            return self._json(422, {"error": f"app '{app_ref}' declares a minimum of {min_cgf} CPU GFLOPS; the request asks for less"})
        if min_ggf and round(_num("gpuTflops") * 1000) < min_ggf:
            return self._json(422, {"error": f"app '{app_ref}' declares a minimum of {min_ggf / 1000} GPU TFLOPS; the request asks for less"})
        storage_mb = int(meta.get("storage_mb", DEF_STORAGE_MB))   # per-app /data cap; 0 opts out
        config = str(b.get("config") or "").strip()                # per-deployment ENCLAVE_CONFIG (the version's config, inline; validated in launch)
        config_cid = str(b.get("configCid") or "").strip()         # rev-7 alternative to `config`: the CID holding it (fetched + hash-checked in launch)
        egress = str(b.get("egress") or "").strip()                # per-deployment ENCLAVE_EGRESS (opaque SOCKS URL, forwarded verbatim)
        secrets = b.get("secrets") or None                         # relay-staged owner secrets (guest --env; validated in launch, never logged)
        hosts = str(b.get("hosts") or "").strip()                  # every hostname this deployment answers on -> guest ENCLAVE_HOSTS
        req_vols = b.get("volumes") or []                          # attached model volumes by name
        if not isinstance(req_vols, list):
            return self._json(400, {"error": "volumes must be a list of volume names"})
        shielded = b.get("shielded") if isinstance(b.get("shielded"), dict) else None
        rec = launch(app_ref, name, cpu_share, gpu_share, mem_mb, pspec, storage_mb, config, req_vols, egress, secrets, hosts,
                     config_cid, shielded=shielded)
        code = 201 if rec["status"] in ("starting", "running") else 500
        return self._json(code, _public(rec))

    def do_DELETE(self):
        if not self._ctrl_authed():
            return self._json(401, {"error": "control token required"})
        if self.path.startswith("/vms/"):
            vid = self.path[len("/vms/"):]
            return self._json(200, {"id": vid, "deleted": teardown(vid)})
        return self._json(404, {"error": "not found"})


def _wasmtime_version() -> str:
    if MOCK:
        return "mock"
    try:
        r = subprocess.run([WASMTIME, "--version"], capture_output=True, text=True, timeout=10)
        return (r.stdout or r.stderr or "").strip().splitlines()[0] if (r.stdout or r.stderr) else ""
    except Exception as e:
        return f"err: {e}"


def _debug_env() -> dict:
    out = {"runtime": "wasmtime", "mock": MOCK, "apps_dir": str(APPS_DIR),
           "catalog": sorted(_load_catalog().keys()), "version": _wasmtime_version(),
           # p3: what the box actually serves; p3_env/p3_binary: why, when it doesn't
           # (the old field reported the env switch alone and would claim p3 on a
           # binary that rejects the flag)
           "p3": _p3_active(), "p3_env": _P3_ENV_ENABLED, "p3_binary": _p3_supported(),
           "coop_threads": _threads_active(), "coop_threads_env": _THREADS_ENV_ENABLED,
           "coop_threads_binary": _threads_supported(),
           "loopback_wall": _loopback_flag_supported(),
           "nn": _nn_available(),
           "nn_probe": dict(_NN_PROBE), "gpu_vram_gb": GPU_VRAM_GB, "gpu_vram_source": GPU_VRAM_SRC,
           "mps_pipe": MPS_PIPE_DIR if (NN_ENABLED and NODE_HAS_GPU) else None,
           "fs": FS_ENABLED, "fs_guest": FS_GUEST_PATH if FS_ENABLED else None,
           # rev-7 large configs (this is the snake_case operator surface; the
           # camelCase twin on /health is what the supervisor reads)
           "config_cid": ipfs_fetch is not None,
           "config_max_bytes": CONFIG_MAX_BYTES,
           "config_env_max_bytes": CONFIG_ENV_MAX_BYTES,
           "config_guest": CONFIG_GUEST_PATH,
           "default_storage_mb": DEF_STORAGE_MB if FS_ENABLED else 0,
           "enc": ENC_ENABLED and shutil.which(RCLONE_BIN) is not None,
           "enc_guest": ENC_GUEST_ROOT if ENC_ENABLED else None,
           "nn_preload": dict(_PRELOAD_SUPPORT),
           "nn_arbiter": _nn_arb_public()}
    try:
        out["uname"] = " ".join(os.uname())
    except Exception as e:
        out["uname"] = f"err: {e}"
    # does `wasmtime serve` exist in this build?
    if not MOCK:
        try:
            r = subprocess.run([WASMTIME, "serve", "--help"], capture_output=True, text=True, timeout=10)
            out["serve_available"] = (r.returncode == 0)
        except Exception as e:
            out["serve_available"] = f"err: {e}"
    return out


def main():
    # Clear stale scratch dirs from a previous run: /data is strictly ephemeral,
    # and a manager restart has already lost track of any prior deployments.
    # NOT gated on FS_ENABLED: the staged <vid>-cfg config drops are written
    # whenever a deployment HAS a config, /data or no /data, and a resolved
    # config can hold substituted secrets. Gating this sweep the way the /data
    # one is gated would let those outlive the tenant across a restart.
    for child in FS_DIR.iterdir() if FS_DIR.exists() else []:
        if child.is_dir():
            _rm_tree_rw(child)      # the -cfg drops are 0500; a plain rmtree would leave them
    if ENC_ENABLED:
        for child in ENC_DIR.iterdir() if ENC_DIR.exists() else []:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    httpd = http.server.ThreadingHTTPServer((HOST_IP if HOST_IP else "0.0.0.0", PORT), Handler)
    threading.Thread(target=_audit_sweep, daemon=True).start()   # firewall bind + storage audit
    if _NN_PROBE["state"] == "probing":
        threading.Thread(target=_nn_probe_loop, daemon=True).start()   # gates GPU launches
    # No NODE_HAS_GPU in this gate: a shielded (metal) box has NODE_HAS_GPU=0 —
    # the card lives on the untrusted host — yet its GPU tenants still contend
    # for one serialized worker and want the same weighted turns. The env is an
    # explicit operator opt-in either way (gsup arms it only on shielded boxes;
    # the CPU fleet never sets it), so this loosening arms nothing by itself.
    if NN_ARB_ENABLED and NN_ENABLED and not MOCK:
        _start_nn_arbiter()          # GPU work-conserving fair share (tenants connect at launch)
    if MPS_BOOT_BOUNCE and NODE_HAS_GPU and not MOCK:
        _boot_bounce_check()         # reclaim a stranded generation BEFORE the first launch
    print(f"wasm-manager on :{PORT} runtime=wasmtime mock={MOCK} apps_dir={APPS_DIR} "
          f"p3={_p3_active()} coopThreads={_threads_active()} set={_set_active()} fs={FS_ENABLED} nn={_NN_PROBE['state']}", flush=True)
    if not VMMGR_TOKEN:
        if VMMGR_ALLOW_UNAUTH:
            print("wasm-manager WARNING: no VMMGR_TOKEN/SECRET and VMMGR_ALLOW_UNAUTHENTICATED=1 — "
                  "control plane is UNAUTHENTICATED by explicit configuration (a loopback-reaching "
                  "tenant can create/delete any vm).", flush=True)
        else:
            print("wasm-manager WARNING: no VMMGR_TOKEN/SECRET — control plane is FAIL-CLOSED "
                  "(control routes deny every request). Set SECRET/VMMGR_TOKEN to operate, or "
                  "VMMGR_ALLOW_UNAUTHENTICATED=1 to explicitly run open (local dev only).", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for vid in list(_apps):
            teardown(vid)


if __name__ == "__main__":
    main()
