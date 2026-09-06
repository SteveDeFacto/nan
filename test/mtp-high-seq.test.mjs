// Real-model regression for device sampling after prefix-cache slot expansion.
// Build wasm/ggml-shielded/bench-spec against the shim under test, then set
// ENCLAVE_MTP_TEST_BENCH and ENCLAVE_MTP_TEST_MODEL (a small MTP GGUF).
// LD_LIBRARY_PATH selects the matching engine/shim. Set BACKENDS to that
// release's libggml-cpu.so (the bench loads modules explicitly); no GPU is required.
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";

const bench = process.env.ENCLAVE_MTP_TEST_BENCH;
const model = process.env.ENCLAVE_MTP_TEST_MODEL;
const skip = !bench || !model ? "requires a built bench-spec and an MTP GGUF" : false;

for (const [seq, device, lastPrefill] of [
  [0, true, false], [8, true, false], [13, true, false], [31, true, false], [13, false, false],
  [13, true, true], [13, false, true],
]) {
  test(`MTP preserves greedy output on sequence ${seq}, device sampling=${device}, last-row prefill=${lastPrefill}`, { skip }, () => {
    const output = execFileSync(bench, [model, "The capital of France is", "16"], {
      encoding: "utf8", timeout: 120_000, maxBuffer: 8 * 1024 * 1024,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, CUDA_VISIBLE_DEVICES: "", N_GPU_LAYERS: "0",
        THREADS: "2", K: "1", P_MIN: "0", SEQ_ID: String(seq), N_SEQS: String(seq + 1),
        ENCLAVE_MTP_DEV_SAMPLE: device ? "1" : "0", PREFILL_LAST: lastPrefill ? "1" : "0" },
    });
    const result = JSON.parse(output);
    assert.equal(result.text_identical, true);
    assert.equal(result.obs_fail, 0);
    assert.ok(result.drafted > 0, "the head must produce proposals on IDs beyond seven");
    assert.ok(result.accepted > 0, "the target must accept at least one proposal");
  });
}
