// nnThreadsBatch: the prefill (batch) thread count, separate from nnThreads.
// The shielded tier wants FEW decode threads (its masking refill threads are the
// binding resource) and MANY prefill threads (prefill runs in the enclave on
// cores by policy). Pinned here: the manager key -> engine env seam, the cap,
// and that every llama context the shim creates takes the batch count.
//
//   run: node --test test/nn-threads-batch.test.mjs

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const manager = fs.readFileSync(path.join(ROOT, "wasm/wasm_manager.py"), "utf8");
const shim = fs.readFileSync(path.join(ROOT, "wasm/llama-shim/enclave_llama.c"), "utf8");

test("nnThreadsBatch reaches the shim as ENCLAVE_GGML_N_THREADS_BATCH, capped at the tenant's parallelism", () => {
  const m = manager.match(/_nn_cfg_int\(enclave_config, "nnThreadsBatch", 1, 512\)[\s\S]*?env\["ENCLAVE_GGML_N_THREADS_BATCH"\] = str\(min\(ntb, _available_parallelism_for\(cpu_share\)\)\)/);
  assert.ok(m, "the manager must map nnThreadsBatch to ENCLAVE_GGML_N_THREADS_BATCH under the parallelism cap");
  assert.match(shim, /getenv\("ENCLAVE_GGML_N_THREADS_BATCH"\)/, "the shim must read the same env");
});

test("every llama context the shim creates takes the batch thread count separately", () => {
  const sites = shim.match(/p\.n_threads = ell_n_threads\(\);\n\s*p\.n_threads_batch = ell_n_threads_batch\(\);/g) || [];
  assert.equal(sites.length, 3, "ell_new, ell_new_server and ell_mtp_new each set both counts");
  assert.equal((shim.match(/p\.n_threads = p\.n_threads_batch = /g) || []).length, 0, "no context may tie prefill threads to decode threads");
  assert.match(shim, /static int32_t ell_n_threads_batch\(void\) \{[\s\S]*?return ell_n_threads\(\);\n\}/, "unset = the decode count (no behaviour change for old configs)");
});
