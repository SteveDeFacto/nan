import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('the CPU module and full engine toolchain use the same source and tensor ABI', () => {
  const runtime = readFileSync(new URL('../wasm/Dockerfile.wasm', import.meta.url), 'utf8');
  const engine = readFileSync(new URL('../.github/workflows/llamacpp-toolchain.yml', import.meta.url), 'utf8');
  const cpuPin = /^ARG GGML_CPU_COMMIT=([a-f0-9]{40})$/m.exec(runtime)?.[1];
  const enginePin = /^  LLAMA_COMMIT: ([a-f0-9]{40})$/m.exec(engine)?.[1];
  assert.ok(cpuPin && enginePin, 'both source revisions must be explicit');
  assert.equal(cpuPin, enginePin, 'a CPU-only module must not introduce a second ggml ABI');
  for (const source of [runtime, engine]) {
    assert.match(source, /-DGGML_NATIVE=OFF/);
    assert.match(source, /-DGGML_BACKEND_DL=ON/);
    assert.match(source, /CMAKE_C_FLAGS[^\n]*-DGGML_MAX_NAME=128/);
    assert.match(source, /CMAKE_CXX_FLAGS[^\n]*-DGGML_MAX_NAME=128/);
  }
});
