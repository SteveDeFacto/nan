/* Refill oracle: every AVX-512 refill path (the four-row kernel and the
 * row-blocked kernel a batch past four rows takes) against a scalar exact
 * product, over the shapes that matter: K below, at and past the 64-byte
 * vector, the 2048-byte slab boundary (and exact multiples of it, which the
 * production 6144/10240 widths are), the real 5120/6144/17408 widths and a
 * width past 65536; N from one column through one and two full tiles; batches 1..33
 * (every four-row tail and a blocked batch past two tiles); random planes
 * plus the three extreme fillings (alternating-sign rows, all-max, all-min)
 * that overflow a naive accumulator. A mismatch aborts; exit 77 means this
 * host has no AVX-512 VNNI and the test is skipped, never passed.
 *
 *   make refill-selftest && ./refill-selftest
 */
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
    const int sizes[] = {1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 2048, 4096, 5119, 5120, 6144, 17408, 70000};
    const int widths[] = {1, 2, 3, 10, 16, 17, 32}; /* partial tiles, one full tile, two full tiles */
    for (unsigned i = 0; i < sizeof sizes / sizeof *sizes; i++)
        for (int b = 1; b <= 33; b++)
            for (int extremes = 0; extremes <= 3; extremes++)
                for (unsigned j = 0; j < sizeof widths / sizeof *widths; j++) check(sizes[i], widths[j], b, extremes);
    return 0;
}
