// GPU request-level arbiter (work-conserving fair share) — wasm_manager.py.
//
// The contract under test, in order of what it would cost to break:
//   1. OFF is bit-identical to the pre-arbiter fleet: no knob = share-sized
//      MPS SM caps and no ENCLAVE_NN_ARBITER env (the launcher must never arm
//      tenants against a toolchain that was not proven).
//   2. The scheduler is work-conserving: an idle queue grants instantly.
//   3. Contention divides GPU TIME by gpuShare (weighted fair queuing over
//      virtual time), and idling banks no credit.
//   4. Crash-safety: a revoked (wedged) grant frees the slot; a dead
//      connection releases everything it held or awaited.
//   5. The socket server speaks the wire protocol the toolchain client
//      (wasmtime-nn-arbiter.patch) implements, and takes the weight from the
//      RECORD when the tenant id is known (hello weight is advisory).
//
//   run: node --test test/nn-arbiter.test.mjs   (needs python3)

import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MGR = path.join(REPO, "wasm", "wasm_manager.py");

// Fixture-layout scaffolding for the _fixture_wasm() tests: resolution only
// cares that a file named nn-demo.wasm exists, so a stub byte suffices.
function mkdtempEmpty() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "nn-fix-"));
}
function mkdtempWithFixture() {
  const d = mkdtempEmpty();
  fs.writeFileSync(path.join(d, "nn-demo.wasm"), "\0asm");
  return d;
}

function run(pyBody, envExtra = {}) {
  const code = `
import importlib.util, sys, json, os, socket, tempfile, threading, time
spec = importlib.util.spec_from_file_location("wm", ${JSON.stringify(MGR)})
m = importlib.util.module_from_spec(spec); sys.modules["wm"] = m
spec.loader.exec_module(m)
${pyBody}
`;
  const out = execFileSync("python3", ["-c", code], {
    env: { ...process.env, NODE_HAS_GPU: "1", WASM_NN: "1",
           CUDA_MPS_PIPE_DIRECTORY: "/tmp/nvidia-mps", GPU_VRAM_GB: "141",
           ...envExtra },
    encoding: "utf8", stdio: ["ignore", "pipe", "pipe"],
  });
  return JSON.parse(out.trim().split("\n").pop());
}

test("OFF by default: share-sized SM cap, no arbiter env — bit-identical to today", () => {
  const r = run(`
env = m._nn_tenant_env(0.25, pinned=True)
print(json.dumps({"enabled": m.NN_ARB_ENABLED,
                  "sm": env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"],
                  "armed": "ENCLAVE_NN_ARBITER" in env,
                  "live": m._nn_arbiter_live()}))
`);
  assert.equal(r.enabled, false, "WASM_NN_ARBITER must default off");
  assert.equal(r.sm, "25", "without the arbiter the SM cap IS the sold share");
  assert.equal(r.armed, false);
  assert.equal(r.live, false);
});

test("live arbiter: SM cap becomes the burst ceiling (VRAM pin untouched)", () => {
  const r = run(`
m.NN_ARB_ENABLED = True
m._NN_ARB = object()                       # server "running"
m._NN_ARB_SUPPORT.update(state="probed", supported=True)   # toolchain proven
env = m._nn_tenant_env(0.25, pinned=True)
# GPU_VRAM_GB may be probed off a real local card — derive the expectation
print(json.dumps({"sm": env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"],
                  "pin": env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"],
                  "want": "0=%dM" % max(1, int(0.25 * m.GPU_VRAM_GB * 1024))}))
`);
  assert.equal(r.sm, "100", "arbitrated tenants burst to the full SM budget");
  assert.equal(r.pin, r.want,
    "the VRAM pin can never be work-conserving and stays share-sized");
});

test("unproven toolchain keeps hard caps even with the knob on", () => {
  const r = run(`
m.NN_ARB_ENABLED = True
m._NN_ARB = object()
m._NN_ARB_SUPPORT.update(state="probed", supported=False)  # probe said no
env = m._nn_tenant_env(0.25, pinned=True)
print(json.dumps({"sm": env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"],
                  "live": m._nn_arbiter_live()}))
`);
  assert.equal(r.sm, "25", "unproven means hard caps — the loopback doctrine");
  assert.equal(r.live, false);
});

test("scheduler: work-conserving, share-proportional, no banked idle credit", () => {
  const r = run(`
clk = [0.0]

s = m.NnArbScheduler(conc=1, max_hold=30, clock=lambda: clk[0])
s.hello(1, "solo", 0.05, "0")
instant = s.acquire(1, 0) == [(1, 0)]      # idle queue grants a 5% tenant instantly

# Steady contention, 1s holds: both tenants keep their next request queued at
# all times; whoever is granted holds 1s, releases, immediately re-queues.
# Over 600 turns the grant split must approach the share split .5:.1 = 5:1.
def contended_counts(w1, w2, turns):
    sched = m.NnArbScheduler(conc=1, max_hold=1e9, clock=lambda: clk[0])
    sched.hello(1, "a", w1, "0"); sched.hello(2, "b", w2, "0")
    gr = sched.acquire(1, 1)
    sched.acquire(2, 1)
    n, rid = {1: 0, 2: 0}, {1: 1, 2: 1}
    order = []
    for _ in range(turns):
        (conn, req) = gr[0]
        n[conn] += 1
        order.append(conn)
        clk[0] += 1.0                      # the grant holds the card 1s
        rid[conn] += 1
        nxt = sched.release(conn, req)     # release -> next grant fires...
        # ...and re-queue self; a hesitation means the grant comes from HERE
        own = sched.acquire(conn, rid[conn] * 1000 + conn)
        gr = nxt or own or sched._dispatch(sched._q("0"))
    return n, order, sched

n, _, _ = contended_counts(0.5, 0.1, 600)

# No banked idle credit: equal weights, but "busy" worked alone for 500s
# while "sleepy" idled. When sleepy shows up it must ALTERNATE with busy from
# now on (vclock clamps its vt), not take 250 catch-up turns in a row.
sched = m.NnArbScheduler(conc=1, max_hold=1e9, clock=lambda: clk[0])
sched.hello(1, "busy", 0.5, "0"); sched.hello(2, "sleepy", 0.5, "0")
gr = sched.acquire(1, 1)
rid = {1: 1, 2: 0}
for _ in range(500):                       # busy grinds alone
    (conn, req) = gr[0]
    clk[0] += 1.0
    rid[1] += 1
    nxt = sched.release(conn, req)
    own = sched.acquire(1, rid[1])
    gr = nxt or own or sched._dispatch(sched._q("0"))
sched.acquire(2, 900001)                   # sleepy arrives NOW
order = []
for _ in range(6):
    (conn, req) = gr[0]
    order.append(conn)
    clk[0] += 1.0
    rid[conn] += 1
    nxt = sched.release(conn, req)
    own = sched.acquire(conn, conn * 100000 + rid[conn])
    gr = nxt or own or sched._dispatch(sched._q("0"))
sleepy_burst = max(len(list(g)) for _, g in __import__("itertools").groupby(order))

print(json.dumps({"instant": instant, "big": n[1], "small": n[2],
                  "sleepy_burst": sleepy_burst}))
`);
  assert.equal(r.instant, true, "an idle queue must grant without waiting");
  const ratio = r.big / r.small;
  assert.ok(ratio > 4.4 && ratio < 5.6,
    `GPU time must split ~5:1 for shares .5:.1 (got ${r.big}:${r.small})`);
  assert.ok(r.sleepy_burst <= 2,
    `a returning idler competes from NOW, no banked credit (longest run ${r.sleepy_burst})`);
});

test("a sparse tenant never taxes a hot one with dead holds", () => {
  const r = run(`
clk = [0.0]
s = m.NnArbScheduler(conc=1, max_hold=1e9, grace=0.015, clock=lambda: clk[0])
s.hello(1, "hot", 0.5, "0"); s.hello(2, "sparse", 0.5, "0")
# hot grinds 100ms turns; sparse does ONE op and goes quiet. Under the old
# hesitation rule sparse (recent, marginally lower vt) put a dead 15ms hold
# in front of EVERY hot dispatch — a per-token decode tax, seen live
# 2026-08-01. The predictor (gap EWMA <= grace) must refuse to hesitate for
# a tenant that has never demonstrated a quick return.
g = s.acquire(1, 1)
s.acquire(2, 1)                            # sparse queues once
clk[0] += 0.1
g = s.release(1, 1)                        # sparse takes its one turn
assert g == [(2, 1)], g
clk[0] += 0.01
s.release(2, 1)                            # ...and goes quiet forever
instant = []
rid = 10
for _ in range(5):                         # hot keeps grinding
    got = s.acquire(1, rid)
    instant.append(got == [(1, rid)])      # must grant IMMEDIATELY, no hold
    clk[0] += 0.1
    s.release(1, rid)
    rid += 1
print(json.dumps({"instant": instant}))
`);
  assert.deepEqual(r.instant, [true, true, true, true, true],
    "every hot acquire must grant instantly once sparse has gone quiet");
});

test("revoke frees the slot; disconnect releases everything", () => {
  const r = run(`
clk = [0.0]
s = m.NnArbScheduler(conc=1, max_hold=30, clock=lambda: clk[0])
s.hello(1, "wedged", 0.5, "0"); s.hello(2, "healthy", 0.5, "0")
s.acquire(1, 1)                            # wedged holds...
s.acquire(2, 1)                            # healthy waits
clk[0] = 31.0                              # ...past the lease
grants, revoked = s.tick()
revoke_ok = grants == [(2, 1)] and revoked[0][1] == "wedged"
late_rel = s.release(1, 1)                 # the wedged rel arrives late: ignored
# disconnect: conn 2 dies holding the grant and with another acq queued
s.acquire(2, 2)
g = s.disconnect(2)
gone = not s._q("0")["active"] and not s._q("0")["waiting"]
print(json.dumps({"revoke_ok": revoke_ok, "late_rel": late_rel == [],
                  "gone": gone}))
`);
  assert.ok(r.revoke_ok, "a wedged grant must not freeze the queue");
  assert.ok(r.late_rel, "a release after revoke is ignored, not double-freed");
  assert.ok(r.gone, "a dead connection leaves nothing behind");
});

test("wire protocol e2e: hello/acq/grant/rel over the real socket, record weight wins", () => {
  const r = run(`
sock_path = tempfile.mktemp(prefix="nn-arb-test-", suffix=".sock")
# a live record for tenant "vm-a": the record's gpuShare must override hello
m._apps["vm-a"] = {"id": "vm-a", "status": "running", "gpuShare": 0.5,
                   "cpuShare": 0.1}
srv = m._NnArbServer(sock_path)

def client(tenant, weight):
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(sock_path)
    c.sendall((json.dumps({"op": "hello", "v": 1, "tenant": tenant,
                           "weight": weight, "queue": "0"}) + "\\n").encode())
    f = c.makefile("r")
    assert json.loads(f.readline())["ok"] is True
    return c, f

a, fa = client("vm-a", 0.001)   # lies small; record says 0.5
b, fb = client("vm-b", 0.1)

a.sendall(b'{"op":"acq","id":1}\\n')
raw = fa.readline()
# WIRE-SHAPE PIN, byte-exact: the Rust client (wasmtime-nn-arbiter.patch)
# parses grants by reading the digits IMMEDIATELY after '"id":'. Pythonic
# json.dumps spacing ('"id": 1') made it drop every grant and ride the 120s
# fail-open watchdog per decode step (live 2026-08-01). Parse like the
# client does, not with a JSON parser that would forgive the regression.
first = raw == '{"ok":true,"id":1}\\n'

b.sendall(b'{"op":"acq","id":7}\\n')
import select
r, _, _ = select.select([b], [], [], 0.3)
blocked = not r                           # conc=1: b must wait for a's rel

a.sendall(b'{"op":"rel","id":1}\\n')
handoff = fb.readline() == '{"ok":true,"id":7}\\n'

# a dies (no rel for nothing held, but with a queued acq) -> b unaffected
a.sendall(b'{"op":"acq","id":2}\\n')
time.sleep(0.1)
a.close()
time.sleep(0.2)
with srv.lock:
    w = srv.sched.queues["0"]["tenants"].get("vm-a", {}).get("weight")
snap = srv.snapshot()
print(json.dumps({"first": first, "blocked": blocked, "handoff": handoff,
                  "recordWeight": w, "grants": snap["stats"]["grants"]}))
`);
  assert.ok(r.first, "idle queue: first acq grants immediately");
  assert.ok(r.blocked, "conc=1: the second tenant queues");
  assert.ok(r.handoff, "rel hands the card to the waiter");
  assert.equal(r.recordWeight, 0.5,
    "the deployment record's gpuShare outranks the hello weight");
  assert.equal(r.grants, 2);
});

// The v0.5.486 regression (2026-08-23): the persistent wasm-cache volume
// mounts an initially EMPTY dir over APPS_DIR, which shadowed the baked
// nn-demo.wasm — _arbiter_support() read the missing fixture as "toolchain
// unproven" and hard-capped every GPU tenant to its share-sized SM slice
// (eyesoff 70->45 tok/s). The fixture now lives in FIXTURES_DIR, which no
// tenant-facing volume may ever mount over.

test("fixture survives an empty mount over APPS_DIR (the v0.5.486 regression)", () => {
  const r = run(`
fixtures = os.environ["WASM_FIXTURES_DIR"] # baked, outside the mount
p = m._fixture_wasm()
print(json.dumps({"path": str(p), "found": p.is_file(),
                  "inFixtures": str(p).startswith(fixtures)}))
`, {
    WASM_APPS_DIR: mkdtempEmpty(),
    WASM_FIXTURES_DIR: mkdtempWithFixture(),
  });
  assert.ok(r.found, "an empty cache mount must not hide the probe fixture");
  assert.ok(r.inFixtures, "the fixture resolves from FIXTURES_DIR");
});

test("legacy layout: no fixtures dir falls back to the APPS_DIR copy", () => {
  const r = run(`
p = m._fixture_wasm()
print(json.dumps({"found": p.is_file(),
                  "inApps": str(p).startswith(os.environ["WASM_APPS_DIR"])}))
`, {
    WASM_APPS_DIR: mkdtempWithFixture(),
    WASM_FIXTURES_DIR: path.join(os.tmpdir(), `nn-fix-none-${process.pid}`),
  });
  assert.ok(r.found, "a pre-move image still finds its APPS_DIR fixture");
  assert.ok(r.inApps);
});

test("no fixture anywhere: arbiter probe reports it and keeps hard caps", () => {
  const r = run(`
s = m._arbiter_support()
print(json.dumps({"supported": s["supported"], "detail": s["detail"]}))
`, {
    WASM_APPS_DIR: mkdtempEmpty(),
    WASM_FIXTURES_DIR: mkdtempEmpty(),
  });
  assert.equal(r.supported, false, "no fixture = unproven = hard caps");
  assert.equal(r.detail, "no nn-demo.wasm fixture",
    "the fallback must be SAYABLE, not a silent cap");
});

// --- shielded (metal) arming -------------------------------------------------
// A shielded box's GPU tenants funnel into ONE host worker whose g_gpu mutex
// serializes them FCFS; _nn_arb_arm is the shared arming point that gives them
// the same weighted turns as the CUDA branch. NODE_HAS_GPU is 0 on those boxes,
// so these run the metal shape of the environment.

test("shielded tenants arm with the same arbiter env — and still no MPS caps", () => {
  const r = run(`
m._nn_arbiter_live = lambda: True
env = m._shielded_tenant_env({"endpoint": "10.0.2.2:9500"})
rec = {"id": "dep-shielded", "shielded": {"endpoint": "10.0.2.2:9500"}}
m._nn_arb_arm(env, rec, 0.35)
print(json.dumps({
  "sock": env.get("ENCLAVE_NN_ARBITER"), "tenant": env.get("ENCLAVE_NN_ARB_TENANT"),
  "weight": env.get("ENCLAVE_NN_ARB_WEIGHT"), "queue": env.get("ENCLAVE_NN_ARB_QUEUE"),
  "armed": rec.get("nnArbiter", False),
  "mps": [k for k in env if k.startswith("CUDA_MPS_")],
  "cvd": env.get("CUDA_VISIBLE_DEVICES"), "ngl": env.get("ENCLAVE_GGML_N_GPU_LAYERS"),
}))
`, { NODE_HAS_GPU: "0" });
  assert.equal(r.sock, "/tmp/enclave-nn-arb.sock");
  assert.equal(r.tenant, "dep-shielded");
  assert.equal(r.weight, "0.35");
  assert.equal(r.queue, "shielded:10.0.2.2:9500");
  assert.equal(r.armed, true);
  assert.deepEqual(r.mps, [], "the card is on the untrusted host: arming must not re-introduce MPS env");
  assert.equal(r.cvd, "", "no local card may be found");
  assert.equal(r.ngl, "0", "ngl stays 0 — the toolchain's SHIELDED_HOST gate carries the gpu flag");
});

test("shielded tenants stay bit-identical when the arbiter is not live", () => {
  const r = run(`
m._nn_arbiter_live = lambda: False
env = m._shielded_tenant_env({"endpoint": "10.0.2.2:9500"})
rec = {"id": "dep-shielded", "shielded": {"endpoint": "10.0.2.2:9500"}}
m._nn_arb_arm(env, rec, 0.35)
print(json.dumps({"arb": [k for k in env if k.startswith("ENCLAVE_NN_ARB")],
                  "armed": rec.get("nnArbiter", False)}))
`, { NODE_HAS_GPU: "0" });
  assert.deepEqual(r.arb, [], "not live = not a single arbiter var");
  assert.equal(r.armed, false);
});

test("shielded GPUs run independently while tenants sharing a worker queue fairly", () => {
  const r = run(`
m._nn_arbiter_live = lambda: True
records = [{"id": str(i), "shielded": {"endpoint": ep}} for i, ep in
           enumerate(["10.0.2.2:9501", "10.0.2.2:9502", "10.0.2.2:9501"])]
s = m.NnArbScheduler(conc=1, max_hold=30, grace=0)
queues = []
for i, rec in enumerate(records):
    env = {}; m._nn_arb_arm(env, rec, .5)
    queues.append(env["ENCLAVE_NN_ARB_QUEUE"])
    s.hello(i + 1, rec["id"], .5, queues[-1])
print(json.dumps({"queues": queues, "first": s.acquire(1, 1),
                  "second": s.acquire(2, 1), "same": s.acquire(3, 1)}))
`);
  assert.equal(r.queues[0], r.queues[2]); assert.notEqual(r.queues[0], r.queues[1]);
  assert.deepEqual(r.first, [[1, 1]]); assert.deepEqual(r.second, [[2, 1]]);
  assert.deepEqual(r.same, []);
});

test("socket server binds shielded queue to the tenant record", () => {
  const r = run(`
p = tempfile.mktemp(prefix="nn-arb-record-", suffix=".sock")
m._apps["bound"] = {"id": "bound", "gpuShare": .25, "shielded": {"endpoint": "10.0.2.2:9502"}}
srv = m._NnArbServer(p)
c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(2); c.connect(p)
c.sendall(b'{"op":"hello","tenant":"bound","weight":1,"queue":"bypass"}\\n')
f = c.makefile("r"); assert json.loads(f.readline())["ok"]
with srv.lock:
    queues = list(srv.sched.queues)
    weight = srv.sched.queues[queues[0]]["tenants"]["bound"]["weight"]
print(json.dumps({"queues": queues, "weight": weight}))
`);
  assert.deepEqual(r.queues, ["shielded:10.0.2.2:9502"]); assert.equal(r.weight, .25);
});
