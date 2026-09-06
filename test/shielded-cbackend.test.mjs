/*
 * The C half of the shielded tier: the field encoding all three implementations
 * must agree on, and -- when a worker is reachable -- one real masked GEMM.
 *
 * The encoding test is the load-bearing one and runs anywhere gcc does. Python is
 * the reference, metal/guest/shielded.mjs mirrors it in float32, and this mirrors
 * it again for the engine. A divergence between any two does not fail loudly at
 * run time: the unmasking subtraction just returns noise. So it gets a test that
 * fails loudly here, over vectors that deliberately include fp16 subnormals,
 * which real GGUF scales hit and a naive converter gets wrong.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { existsSync, writeFileSync } from "node:fs";
import net from "node:net";

const repo = join(dirname(fileURLToPath(import.meta.url)), "..");
const dir = join(repo, "wasm", "ggml-shielded");

let built = null;
function build() {
  if (built !== null) return built;
  const r = spawnSync("make", ["-s"], { cwd: dir, encoding: "utf8", timeout: 300_000 });
  built = r.status === 0;
  if (!built) console.error("[shielded-c] make failed:", r.stderr || r.stdout);
  return built;
}

test("the C field encoding matches Python, subnormals included", (t) => {
  if (!build()) return t.skip("no toolchain for the C backend");
  const py = JSON.parse(execFileSync("python3", [join(repo, "shielded", "field.py")],
    { encoding: "utf8", timeout: 300_000 }).trim().split("\n").pop());

  const consts = JSON.parse(execFileSync(join(dir, "field-selftest"), ["--constants"],
    { encoding: "utf8", timeout: 60_000 }).trim());
  assert.equal(consts.M_MOD, py.M_MOD);
  assert.equal(consts.HALF_M, py.HALF_M);
  assert.deepEqual(consts.primes, py.primes);
  assert.equal(consts.QK, py.QK);
  assert.equal(consts.FRAC, py.FRAC);
  assert.equal(consts.WEIGHT_BYTE_LIMIT, py.WEIGHT_BYTE_LIMIT);

  const v = py.vectors;
  const stdin = v.half_bits.map((h, i) => `${h} ${v.quant[i]}`).join("\n") + "\n";
  const got = execFileSync(join(dir, "field-selftest"), {
    input: stdin, encoding: "utf8", timeout: 60_000 }).trim().split("\n").map(Number);

  assert.ok(v.w_fixed.length >= 512, "vector set shrank");
  assert.equal(got.length, v.w_fixed.length);
  const bad = got.reduce((n, x, i) => n + (x === v.w_fixed[i] ? 0 : 1), 0);
  assert.equal(bad, 0,
    `${bad}/${got.length} encodings differ between wasm/ggml-shielded and shielded/field.py`);
});

// The refill kernels are the pad generator: a wrong column there is silent noise
// after the unmasking subtraction, exactly like a wrong encoding. The oracle
// covers the four-row path and the row-blocked path a wider batch takes.
test("the AVX-512 refill kernels match a scalar exact product on every shape", (t) => {
  if (!build()) return t.skip("no toolchain for the C backend");
  const r = spawnSync(join(dir, "refill-selftest"), { encoding: "utf8", timeout: 600_000 });
  if (r.status === 77) return t.skip("no AVX-512 VNNI on this host");
  assert.equal(r.status, 0, `refill-selftest failed: ${r.signal || r.status} ${r.stderr || ""}`);
});

// Dealt pads (shielded/dealer/PLAN.md): a shipment minted from synthetic weights
// reads back against an exact oracle, a flipped byte is refused, a foreign key
// sees nothing, the ledger window advances before use, and the link's own
// import path yields the same pads. No worker, no AVX-512 requirement.
test("dealt pads: mint, read back, tamper, exhaustion, ledger window, link import", (t) => {
  if (!build()) return t.skip("no toolchain for the C backend");
  const r = spawnSync(join(dir, "dealt-selftest"), { encoding: "utf8", timeout: 300_000 });
  assert.equal(r.status, 0, `dealt-selftest failed: ${r.signal || r.status} ${r.stderr || ""}`);
  assert.match(r.stdout, /dealt-selftest: ok/);
});

// The scales are half of THE encoding, so their fp32->fp16 conversion is part of
// the contract too. Truncating instead of rounding to nearest-even lands one ulp
// low on about half of all blocks and fails NOWHERE -- the two sides simply derive
// different weights and unmasking returns noise. Caught exactly once, here.
test("the C fp16 conversion rounds like numpy, subnormals and overflow included", (t) => {
  if (!build()) return t.skip("no toolchain for the C backend");
  const out = execFileSync("python3", ["-c", `
import json, subprocess, sys
import numpy as np
rng = np.random.default_rng(3)
vals = np.concatenate([
    rng.uniform(-0.05, 0.05, 20000),
    rng.uniform(-1e-6, 1e-6, 20000),          # subnormal in fp16
    rng.uniform(-70000, 70000, 4000),         # overflows fp16
    np.array([0.0, -0.0, 6e-8, 5.96e-8, 6.104e-5, 0.00390625, 65504.0, 65520.0]),
]).astype(np.float64)
p = subprocess.run([${JSON.stringify(join(dir, "half-selftest"))}],
                   input=chr(10).join(repr(float(v)) for v in vals),
                   capture_output=True, text=True, timeout=600)
c = np.array([int(x) for x in p.stdout.split()], dtype=np.uint16)
with np.errstate(over="ignore"):
    ref = vals.astype(np.float32).astype(np.float16).view(np.uint16)
print(json.dumps({"checked": int(c.size), "mismatch": int((c != ref).sum())}))
`], { encoding: "utf8", timeout: 900_000 }).trim().split("\n").pop();
  const v = JSON.parse(out);
  assert.ok(v.checked > 40000, "sample shrank");
  assert.equal(v.mismatch, 0, `${v.mismatch}/${v.checked} fp16 conversions differ from numpy`);
});

// The C deliberately picks the exponent PER OUTPUT COLUMN where tee.py picks one
// per tensor -- a per-tensor exponent has to be sized for the single largest
// weight and quantises 13.5% of this model's weights to zero (39-41% on some
// tensors), which measurably costs the model its answer. So the two are no longer
// expected to agree in general. What must still hold is that the shared ARITHMETIC
// is identical, and a single-column weight is exactly the case where per-column
// and per-tensor mean the same thing -- so N=1 pins them together, and the byte
// lane is checked directly for the general case.
test("the C exponent agrees with tee.py where the two mean the same thing", (t) => {
  if (!build()) return t.skip("no toolchain for the C backend");
  const out = execFileSync("python3", ["-c", `
import json, subprocess, sys
sys.path.insert(0, ${JSON.stringify(join(repo, "shielded"))})
import numpy as np
from tee import PublicWeight, QK
from field import WEIGHT_BYTE_LIMIT
rng = np.random.default_rng(11)
bad, wide = [], []

def run_c(wd, wq, K, N):
    nl = chr(10)
    inp = ("%d %d" % (K, N) + nl
           + " ".join(str(int(x)) for x in wd.view(np.uint16).ravel()) + nl
           + " ".join(str(int(x)) for x in wq.ravel()) + nl)
    p = subprocess.run([${JSON.stringify(join(dir, "prepare-selftest"))}],
                       input=inp, capture_output=True, text=True, timeout=600)
    return [int(x) for x in p.stdout.split()]

# N == 1: per-column IS per-tensor, so the two must agree exactly.
for trial, K in enumerate([64, 128, 256]):
    wd = (rng.uniform(1e-6, 0.02, size=(K // QK, 1))
          * (10.0 ** rng.integers(-2, 1, size=(K // QK, 1)))).astype(np.float16)
    wq = rng.integers(-127, 128, size=(K, 1)).astype(np.int8)
    pw = PublicWeight("t%d" % trial, wq, wd)
    fw = run_c(wd, wq, K, 1)
    if len(fw) != 1 or fw[0] != pw.f_w:
        bad.append({"K": K, "py_fw": int(pw.f_w), "c_fw": fw})

# N > 1: every column's own exponent must keep its encoded weights inside the
# byte lane, which is what the residue identity (and the fast kernel) rests on.
for K, N in [(64, 32), (256, 48)]:
    wd = (rng.uniform(1e-6, 0.02, size=(K // QK, N))
          * (10.0 ** rng.integers(-3, 1, size=(K // QK, N)))).astype(np.float16)
    wq = rng.integers(-127, 128, size=(K, N)).astype(np.int8)
    fw = np.array(run_c(wd, wq, K, N), dtype=np.int64)
    if fw.size != N: wide.append({"K": K, "N": N, "got": int(fw.size)}); continue
    true = np.repeat(wd.astype(np.float64), QK, axis=0)[:K] * wq.astype(np.float64)
    enc = np.floor(true * (2.0 ** fw)[None, :] + 0.5)
    if np.abs(enc).max() > WEIGHT_BYTE_LIMIT:
        wide.append({"K": K, "N": N, "peak": float(np.abs(enc).max())})
print(json.dumps({"bad": bad, "wide": wide}))
`], { encoding: "utf8", timeout: 900_000 }).trim().split("\n").pop();
  const v = JSON.parse(out);
  assert.deepEqual(v.bad, [], "single-column: the C and tee.py disagree on the exponent");
  assert.deepEqual(v.wide, [], "a per-column exponent pushed weights outside the int8 lane");
});

const reachable = (host, port) => new Promise((res) => {
  const s = net.connect({ host, port });
  const done = (v) => { s.destroy(); res(v); };
  s.setTimeout(1500);
  s.on("connect", () => done(true));
  s.on("error", () => done(false));
  s.on("timeout", () => done(false));
});

test("one real masked GEMM through the C stack, asserted four ways", async (t) => {
  if (!build()) return t.skip("no toolchain for the C backend");
  const host = process.env.SHIELDED_HOST || "127.0.0.1";
  const port = Number(process.env.SHIELDED_PORT || 9500);
  if (!(await reachable(host, port)))
    return t.skip(`no shielded worker at ${host}:${port} (needs a CUDA box)`);

  const out = execFileSync(join(dir, "shielded-probe"),
    ["--host", host, "--port", String(port)],
    { encoding: "utf8", timeout: 300_000 }).trim().split("\n").pop();
  const v = JSON.parse(out);
  // Every claim, not a subset: a product that came back exact from a worker that
  // also accepts a denylisted op is not one to trust.
  assert.equal(v.exact, true, "unmasked product diverged from the int64 reference");
  assert.equal(v.verified, true, "Freivalds rejected an honest product");
  assert.equal(v.lie_rejected, true, "Freivalds ACCEPTED a single-element lie");
  assert.equal(v.denylist_refused, true, "the worker ran a denylisted op");
  assert.equal(v.verify_fail, 0);
  // Protocol 1.2: against a worker that offers it, the packed (int24) reply
  // must unmask to the int32 form's y exactly; against a 1.1 worker the
  // probe reports width 4 and this is vacuously true.
  assert.equal(v.packed_identical, true, `reply width ${v.reply_width}: packed and int32 replies disagree`);
  assert.ok(v.field_headroom > 1, `field wrapped: peak |y| ${v.peak_abs_y}`);
});

// The integration this whole tier exists for: an ordinary ggml graph, split by
// ggml_backend_sched, with the linear ops executing on an untrusted GPU under
// one-time pads and the nonlinear ones staying in the enclave. Needs an engine
// checkout AND a live worker, so it skips rather than fails without them.
test("ggml_backend_sched offloads the matmuls and keeps the rest in the enclave", async (t) => {
  const ggmlSrc = process.env.GGML_SRC || join(process.env.HOME || "", "Projects", "llama.cpp");
  const ggmlLib = process.env.GGML_LIB || join(process.env.HOME || "", "Projects", "llamacpp-lib");
  if (!existsSync(join(ggmlSrc, "ggml", "include", "ggml.h")) || !existsSync(ggmlLib))
    return t.skip("no ggml checkout to build the backend against");

  const mk = spawnSync("make", ["-s", "ggml"], {
    cwd: dir, encoding: "utf8", timeout: 600_000,
    env: { ...process.env, GGML_SRC: ggmlSrc, GGML_LIB: ggmlLib } });
  if (mk.status !== 0) return t.skip(`ggml backend did not build: ${(mk.stderr || "").slice(0, 300)}`);

  const host = process.env.SHIELDED_HOST || "127.0.0.1";
  const port = Number(process.env.SHIELDED_PORT || 9500);
  const live = await reachable(host, port);

  const calib = join(repo, "wasm", "ggml-shielded", "test.calib");
  writeFileSync(calib,
    "# shielded-calib 1\nsite blk.0.ffn_gate.weight 8 0\nsite blk.0.ffn_down.weight 8 0\n");

  const out = execFileSync(join(dir, "ggml-test"), [], {
    encoding: "utf8", timeout: 900_000,
    // The size floor is a placement policy for real models; this graph's
    // 512x256 matmuls would sit under it and the split would say nothing.
    env: { ...process.env, SHIELDED_CALIB: calib, SHIELDED_HOST: host, SHIELDED_PORT: String(port),
           SHIELDED_MIN_MACS: "0" },
  }).trim().split("\n").pop();
  const v = JSON.parse(out);

  // The split is the assertion. Two matmuls on the shielded backend and nothing
  // else there -- a run where sched quietly put everything on the CPU would
  // otherwise read as a pass, and so would one where SiLU leaked onto the GPU.
  assert.equal(v.sched_ok, true, "a non-matmul landed on the shielded backend");
  assert.equal(v.sched_shielded_nodes, 2, "sched did not place both matmuls on the shielded backend");
  assert.equal(v.verify_fail, 0, "a product failed verification");
  if (live) assert.ok(v.offloaded_nodes > 0, "a worker was reachable but nothing was offloaded");
  else assert.ok(v.local_nodes > 0, "no worker, so the nodes should have run locally");

  // Against ggml's own f32 matmul the shielded path is a fixed-point
  // approximation, so the bound comes from the encoding rather than from taste:
  // the weight quantum dominates, and this stays far inside it.
  assert.ok(v.rel < 0.05, `shielded result drifted ${v.rel} from the CPU backend`);
});
