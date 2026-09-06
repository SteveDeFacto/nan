// ENCLAVE_AVAILABLE_PARALLELISM: how many threads a tenant can ACTUALLY run
// at once, derived from the cpuShare it bought.
//
// Two things read it inside the tenant's wasmtime (see docs/wasm-parallelism.md
// and wasm/wasmtime-set-threads.patch.wip):
//
//   * `thread.available_parallelism`, the shared-everything-threads intrinsic
//     a guest sizes its pool from;
//   * the ceiling on how many SET worker threads that process may have live.
//
// The second is why this is a safety property and not a hint. OS threads are a
// NODE-WIDE kernel resource (threads-max, pid space) and cgroup cpu.weight
// bounds CPU share, not thread count — so a tenant sizing a pool from the
// node's core count instead of its own slice spends a resource its neighbours
// need. Measured before the engine-side cap existed: 200 guest spawn calls
// produced 234 OS threads with no refusal.
//
//   run: node --test test/wasm-available-parallelism.test.mjs   (needs python3)

import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MGR = path.join(REPO, "wasm", "wasm_manager.py");

function mgr(expr, env = {}) {
  const code = `
import importlib.util, sys, json
spec = importlib.util.spec_from_file_location("wm", ${JSON.stringify(MGR)})
m = importlib.util.module_from_spec(spec); sys.modules["wm"] = m
spec.loader.exec_module(m)
print(json.dumps(${expr}))
`;
  const out = execFileSync("python3", ["-c", code], {
    env: { ...process.env, WASM_MANAGER_PORT: "8091", NODE_HAS_GPU: "0", ...env },
    encoding: "utf8", stdio: ["ignore", "pipe", "pipe"],
  });
  return JSON.parse(out.trim().split("\n").pop());
}

test("parallelism is the purchased share of the node's vCPUs", () => {
  const e = { NODE_VCPUS: "16" };
  assert.equal(mgr("m._available_parallelism_for(1.0)", e), 16);
  assert.equal(mgr("m._available_parallelism_for(0.5)", e), 8);
  assert.equal(mgr("m._available_parallelism_for(0.25)", e), 4);
});

test("rounds UP, so a small share still gets a usable thread count", () => {
  // 0.1 of 16 vCPUs is 1.6. Rounding down to 1 would tell a guest it cannot
  // parallelize at all; rounding up keeps a small tenant able to use the
  // share it bought.
  assert.equal(mgr("m._available_parallelism_for(0.1)", { NODE_VCPUS: "16" }), 2);
  assert.equal(mgr("m._available_parallelism_for(0.01)", { NODE_VCPUS: "16" }), 1);
});

test("never returns 0 — a guest reading it typically divides by it", () => {
  assert.equal(mgr("m._available_parallelism_for(0)", { NODE_VCPUS: "16" }), 1);
  assert.equal(mgr("m._available_parallelism_for(-1)", { NODE_VCPUS: "16" }), 1);
});

test("cannot exceed the node, however much share is claimed", () => {
  // The bound is a real kernel resource, so an over-claimed share must not
  // hand out more parallelism than the box actually has.
  assert.equal(mgr("m._available_parallelism_for(2.0)", { NODE_VCPUS: "16" }), 16);
  assert.equal(mgr("m._available_parallelism_for(1.5)", { NODE_VCPUS: "8" }), 8);
});

test("tracks the node size it is told about", () => {
  assert.equal(mgr("m._available_parallelism_for(0.5)", { NODE_VCPUS: "64" }), 32);
  assert.equal(mgr("m._available_parallelism_for(0.5)", { NODE_VCPUS: "2" }), 1);
});

test("ggml thread tuning leaves room for GPU helpers without exceeding the tenant share", () => {
  const e = { NODE_VCPUS: "16" };
  assert.equal(mgr(`m._nn_threads_for('{"nnThreads":4}', 0.53)`, e), 4);
  assert.equal(mgr(`m._nn_threads_for('{"nnThreads":16}', 0.53)`, e), 9);
  assert.equal(mgr(`m._nn_threads_for('{"nnThreads":512}', 0.01)`, e), 1);
  assert.equal(mgr(`m._nn_threads_for('{"nnThreads":512}', 1.0)`, e), 16);
});

test("omitted or invalid ggml thread tuning preserves the engine default", () => {
  for (const config of ['{}', '{"nnThreads":0}', '{"nnThreads":-1}',
    '{"nnThreads":513}', '{"nnThreads":true}', '{"nnThreads":"4"}', '[]', 'invalid']) {
    assert.equal(mgr(`m._nn_threads_for(${JSON.stringify(config)}, 0.53)`, { NODE_VCPUS: "16" }), null);
  }
});
