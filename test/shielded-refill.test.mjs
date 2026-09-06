import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

test('AVX-512 refills match an int64 field oracle across row/column tails and output strides', {
  skip: process.arch !== 'x64',
}, t => {
  const dir = mkdtempSync(join(tmpdir(), 'shielded-refill-'));
  const source = fileURLToPath(new URL('../wasm/ggml-shielded/', import.meta.url));
  try {
    writeFileSync(join(dir, 'test.c'), `
#include "shielded-simd.h"
#include "shielded-field.h"
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
static uint32_t state = 721;
static uint32_t next(void) { state = state * 1664525U + 1013904223U; return state; }
static void check(int64_t K, int N, int b, int extremes) {
    const int stride = N + 3;
    int8_t *w = malloc((size_t)K * N);
    int32_t *r = malloc((size_t)b * K * sizeof *r);
    uint8_t *planes = malloc((size_t)3 * b * K);
    int32_t *u = malloc((size_t)b * stride * sizeof *u);
    int32_t *acc = malloc((size_t)12 * N * sizeof *acc);
    assert(w && r && planes && u && acc);
    for (int64_t i = 0; i < K * N; i++) w[i] = extremes >= 2 ? (extremes == 2 ? 119 : -119) : extremes ? (i / K % 2 ? 119 : -119) : (int)(next() % 239) - 119;
    for (int64_t i = 0; i < (int64_t)b * K; i++) r[i] = extremes >= 2 ? -1 : extremes ? (i / K % 3 == 0 ? SH_HALF_M : i / K % 3 == 1 ? -SH_HALF_M : 0) : (int64_t)(next() % SH_M_MOD) - SH_HALF_M;
    sh_simd_avx512_pad_planes(r, (size_t)b * K, planes, planes + (size_t)b * K, planes + (size_t)2 * b * K);
    for (int i = 0; i < b * stride; i++) u[i] = INT32_MIN;
    sh_simd_avx512_refill(planes, b, w, K, N, u, stride, acc);
    for (int row = 0; row < b; row++) {
        for (int j = 0; j < N; j++) {
            int64_t exact = 0;
            for (int64_t k = 0; k < K; k++) exact += (int64_t)r[(size_t)row * K + k] * w[(size_t)j * K + k];
            assert(u[row * stride + j] == sh_balanced(exact));
        }
        for (int j = N; j < stride; j++) assert(u[row * stride + j] == INT32_MIN);
    }
    free(w); free(r); free(planes); free(u); free(acc);
}
int main(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx512vnni") || !__builtin_cpu_supports("avx512bw") ||
        !__builtin_cpu_supports("avx512dq") || !__builtin_cpu_supports("avx512vl")) return 77;
    const int sizes[] = {1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 5119, 5120, 17408, 70000};
    const int widths[] = {1, 2, 3, 7, 8, 16, 17};
    for (unsigned i = 0; i < sizeof sizes / sizeof *sizes; i++)
        for (int b = 1; b <= 9; b++)
            for (int extremes = 0; extremes <= 3; extremes++)
                for (unsigned n = 0; n < sizeof widths / sizeof *widths; n++) check(sizes[i], widths[n], b, extremes);
    return 0;
}
`);
    execFileSync('cc', ['-std=c11', '-O3', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl', '-mavx512vnni',
      '-DSH_SIMD_AVX512', '-c', join(source, 'shielded-simd.c'), '-o', join(dir, 'simd.o')], { timeout: 60_000 });
    execFileSync('cc', ['-std=c11', '-O2', '-I', source, join(dir, 'test.c'), join(dir, 'simd.o'),
      join(source, 'shielded-field.c'), '-lm', '-o', join(dir, 'test')], { timeout: 30_000 });
    const result = spawnSync(join(dir, 'test'), { encoding: 'utf8', timeout: 30_000 });
    if (result.status === 77) { t.skip('AVX-512 VNNI hardware unavailable; optimized object compiled'); return; }
    assert.equal(result.status, 0, result.stderr || String(result.error));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
