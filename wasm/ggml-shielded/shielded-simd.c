/*
 * shielded-simd.c -- the hot loops of the trusted half.
 *
 * Compiled TWICE by the Makefile: once with the AVX-512 VNNI target
 * (-DSH_SIMD_AVX512, suffix _avx512) and once generic (suffix _generic), and
 * shielded-tee.c picks at run time. The .so has to load on any x86-64 -- a
 * SIGILL inside the engine is not a degraded mode -- and the same C body has to
 * be the reference for its vectorised twin, which is why nearly everything
 * here is plain loops the compiler vectorises rather than intrinsics. The one
 * exception is the refill inner product, where vpdpbusd is the whole point.
 * On aarch64 (the phone anchor) a third build, -DSH_SIMD_NEON (suffix _neon),
 * gives that one loop to SDOT; the x86 builds do not see a token of it.
 *
 * Every function here is arithmetic on values the TEE already holds. Nothing
 * here touches a socket, and nothing here decides what crosses to the worker.
 */
#include "shielded-field.h"
#include "shielded-simd.h"

/* OP-TEE TAs have no libm and no <math.h>. The ONLY libm user in this file is
 * lrintf in FN(encode); on aarch64 __builtin_lrintf lowers to FCVTNS -- round
 * to nearest, ties to even -- with no library call, which is precisely what
 * lrintf does under the default rounding mode. SH_NO_LIBM selects that form so
 * the file builds in S-EL0. Default builds are unchanged and still call lrintf.
 * (The anchor's TA never calls encode at all: it takes activations already in
 * field form. This exists so the file COMPILES there, not so it is used.) */
#ifdef SH_NO_LIBM
/* __builtin_lrintf still lowers to a libcall at -O3, so name the instruction:
 * FCVTNS is round-to-nearest-ties-even, exactly lrintf's default-mode result. */
#if defined(__aarch64__)
static inline long sh_lrintf_nolibm(float v) {
    long r; __asm__("fcvtns %x0, %s1" : "=r"(r) : "w"(v)); return r;
}
#define sh_lrintf(v) sh_lrintf_nolibm(v)
#else
#define sh_lrintf(v) __builtin_lrintf(v)
#endif
#else
#include <math.h>
#define sh_lrintf(v) lrintf(v)
#endif
#include <stdlib.h>
#include <string.h>

#ifdef SH_SIMD_AVX512
#include <immintrin.h>
#define FN(name) sh_simd_avx512_##name
#elif defined(SH_SIMD_NEON)
#include <arm_neon.h>
#define FN(name) sh_simd_neon_##name
#else
#define FN(name) sh_simd_generic_##name
#endif

#define Q0 SH_Q0
#define Q1 SH_Q1
#define Q2 SH_Q2
#define M_MOD SH_M_MOD
/* Garner constants: inv(Q0) mod Q1 and inv(Q0*Q1) mod Q2. Asserted against
 * sh_crt's int64 truth in simd_agree (shielded-tee.c), so a typo here fails loudly
 * at load rather than as verification failures on the first token. */
#define INV01  217
#define INV012 10

/* v mod q for |v| < 2^28, branch-free and vectorisable: one float estimate of
 * the quotient, then two corrections. The estimate is within 1 of the truth
 * for the whole range, so two corrections are exact. */
static inline int32_t modq(int32_t v, int32_t q, float inv) {
    int32_t t = (int32_t)((float)v * inv);
    int32_t r = v - t * q;
    r += (r < 0) ? q : 0;
    r -= (r >= q) ? q : 0;
    return r;
}

static inline int32_t crt_balanced(int32_t a0, int32_t a1, int32_t a2) {
    const int32_t r0 = modq(a0, Q0, 1.0f / Q0);
    const int32_t r1 = modq(a1, Q1, 1.0f / Q1);
    const int32_t r2 = modq(a2, Q2, 1.0f / Q2);
    const int32_t t1 = modq((r1 - r0) * INV01, Q1, 1.0f / Q1);
    int32_t x = r0 + Q0 * t1;                                   /* < Q0*Q1 */
    const int32_t t2 = modq((r2 - modq(x, Q2, 1.0f / Q2)) * INV012, Q2, 1.0f / Q2);
    x += Q0 * Q1 * t2;                                          /* < M */
    return x > (int32_t)(M_MOD / 2) ? x - (int32_t)M_MOD : x;
}

/* Unsigned residue planes of a pad, [0,q). The pad is in [0,M). */
void FN(pad_planes)(const int32_t *r, size_t n, uint8_t *p0, uint8_t *p1, uint8_t *p2) {
    for (size_t i = 0; i < n; i++) {
        const int32_t v = r[i];
        p0[i] = (uint8_t)modq(v, Q0, 1.0f / Q0);
        p1[i] = (uint8_t)modq(v, Q1, 1.0f / Q1);
        p2[i] = (uint8_t)modq(v, Q2, 1.0f / Q2);
    }
}

/* Balanced residue planes of (x + r) mod M -- what crosses to the worker. x is
 * the plaintext field element (any int64; a value outside the field wraps here
 * and Freivalds catches the consequence), r the one-time pad in [0,M). */
void FN(mask_planes)(const int64_t *x, const int32_t *r, size_t n, int8_t *p0, int8_t *p1, int8_t *p2) {
    const double invM = 1.0 / (double)M_MOD;
    for (size_t i = 0; i < n; i++) {
        int64_t v = x[i] + r[i];
        int64_t t = (int64_t)((double)v * invM);
        v -= t * M_MOD;
        v += (v < 0) ? M_MOD : 0;
        v -= (v >= M_MOD) ? M_MOD : 0;
        const int32_t w = (int32_t)v;
        int32_t a0 = modq(w, Q0, 1.0f / Q0), a1 = modq(w, Q1, 1.0f / Q1), a2 = modq(w, Q2, 1.0f / Q2);
        a0 -= (a0 > Q0 / 2) ? Q0 : 0;
        a1 -= (a1 > Q1 / 2) ? Q1 : 0;
        a2 -= (a2 > Q2 / 2) ? Q2 : 0;
        p0[i] = (int8_t)a0; p1[i] = (int8_t)a1; p2[i] = (int8_t)a2;
    }
}

/* y = balanced(ym - u): both operands already balanced in (-M/2, M/2]. */
void FN(unmask)(const int32_t *ym, const int32_t *u, size_t n, int64_t *y) {
    for (size_t i = 0; i < n; i++) {
        int32_t v = ym[i] - u[i];
        v += (v <= -(int32_t)(M_MOD / 2)) ? (int32_t)M_MOD : 0;
        v -= (v > (int32_t)(M_MOD / 2)) ? (int32_t)M_MOD : 0;
        y[i] = v;
    }
}

/* x_field = round(src * scale). */
#ifdef SH_SIMD_AVX512
/* lrintf is a libm call the vectoriser will not touch (errno, FP exceptions),
 * and 896 of them cost 4.4 us per exchange -- 14 ms of a 64-token decode.
 * vcvtps2dq rounds under the same MXCSR mode lrintf does, so the two agree
 * bit for bit wherever the int32 conversion is exact; lanes at or beyond 2^30
 * (never seen from a sane model, but the outlier channels ARE encoded here
 * before they are pulled out) take the scalar path so nothing saturates. */
void FN(encode)(const float *src, size_t n, float scale, int64_t *x) {
    const __m512 sc  = _mm512_set1_ps(scale);
    const __m512 lim = _mm512_set1_ps(1073741824.0f);           /* 2^30 */
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(src + i), sc);
        const __mmask16 big = _mm512_cmp_ps_mask(_mm512_abs_ps(v), lim, _CMP_NLT_UQ);   /* |v| >= 2^30, or NaN */
        if (__builtin_expect(big != 0, 0)) {
            for (int t = 0; t < 16; t++) x[i + t] = (int64_t)sh_lrintf(src[i + t] * scale);
            continue;
        }
        const __m512i q = _mm512_cvtps_epi32(v);
        _mm512_storeu_si512((void *)(x + i),     _mm512_cvtepi32_epi64(_mm512_castsi512_si256(q)));
        _mm512_storeu_si512((void *)(x + i + 8), _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(q, 1)));
    }
    for (; i < n; i++) x[i] = (int64_t)sh_lrintf(src[i] * scale);
}
#else
void FN(encode)(const float *src, size_t n, float scale, int64_t *x) {
    for (size_t i = 0; i < n; i++) x[i] = (int64_t)sh_lrintf(src[i] * scale);
}
#endif

/* dst[j] = y[j] * inv[j], the per-column descale. */
void FN(descale)(const int64_t *y, const float *inv, size_t n, float *dst) {
    for (size_t i = 0; i < n; i++) dst[i] = (float)y[i] * inv[i];
}

/* sum_j y[j] * s[j*stride + rep] mod P2. |y| < 2^24 and |s| < 2^20, so the
 * int64 sum is exact up to N = 2^19 terms; chunk beyond that. */
int64_t FN(fv_dot)(const int64_t *y, const int64_t *s, int stride, int rep, int64_t n) {
    const int64_t P2 = SH_FV_P2;
    int64_t total = 0;
    for (int64_t k0 = 0; k0 < n; k0 += 262144) {
        const int64_t k1 = k0 + 262144 < n ? k0 + 262144 : n;
        int64_t acc = 0;
        for (int64_t j = k0; j < k1; j++) acc += y[j] * s[j * stride + rep];
        acc %= P2; if (acc < 0) acc += P2;
        total = (total + acc) % P2;
    }
    return total;
}

/* sum_k x[k] * st[k*stride + rep] mod P2 for the rhs, where st < 2^31 and x is
 * the plaintext activation: chunks of 32 keep the int64 accumulator exact for
 * |x| < 2^26 (a legal activation is below M/2 < 2^23; one beyond that has
 * already wrapped and fails the check whatever this sums); chunks of 128
 * keep the accumulator under 2^62. */
int64_t FN(fv_dot_x)(const int64_t *x, const int64_t *st, int stride, int rep, int64_t n) {
    const int64_t P2 = SH_FV_P2;
    int64_t total = 0;
    for (int64_t k0 = 0; k0 < n; k0 += 32) {
        const int64_t k1 = k0 + 32 < n ? k0 + 32 : n;
        int64_t acc = 0;
        for (int64_t k = k0; k < k1; k++) acc += x[k] * st[k * stride + rep];
        acc %= P2; if (acc < 0) acc += P2;
        total = (total + acc) % P2;
    }
    return total;
}

/* st[k*REPS+rep] = sum_j W[j][k] * s[j*REPS+rep] mod P2, W in (N,K). Row-wise
 * axpy so the weight streams once; the int64 accumulator holds 2^44 at N=2^17. */
void FN(fv_prepare)(const int8_t *W, int64_t K, int64_t N, const int64_t *s, int reps, int64_t *st) {
    /* st[k*reps+rep] = sum_j W[j][k] * s[j*reps+rep] mod P2, W in (N,K).
     *
     * Accumulated per rep in a CONTIGUOUS double row and folded once at the
     * end: exact, since |s| < 2^20, |w| <= 119 and N <= 2^17 keep every partial
     * below 2^44 < 2^53, and it vectorises (int8 -> double, FMA), where the
     * original int64 axpy into a stride-`reps` output did not. This runs at
     * weight registration for every node, i.e. inside the engine's context
     * creation: it was 3.6 s of the 0.5B's `sched_reserve` and 32 s of the
     * 4B's, serial, and most of a tenant's launch-to-ready time. */
    enum { BLK = 2048 };
    for (int64_t i = 0; i < K * reps; i++) st[i] = 0;
    double acc[BLK];
    for (int rep = 0; rep < reps; rep++) {
        for (int64_t k0 = 0; k0 < K; k0 += BLK) {
            const int64_t n = k0 + BLK < K ? BLK : K - k0;
            for (int64_t i = 0; i < n; i++) acc[i] = 0.0;
            for (int64_t j = 0; j < N; j++) {
                const double sj = (double)s[j * reps + rep];
                const int8_t *w = W + j * K + k0;
                for (int64_t i = 0; i < n; i++) acc[i] += sj * (double)w[i];
            }
            for (int64_t i = 0; i < n; i++) {
                int64_t v = (int64_t)acc[i] % SH_FV_P2; if (v < 0) v += SH_FV_P2;
                st[(k0 + i) * reps + rep] = v;
            }
        }
    }
}

/* THE REFILL: u[b][j] = sum_k r[b][k] * W[j][k] over Z_M, from the unsigned
 * residue planes of r, for `b` pads at once. Each weight row is read once per
 * batch and dotted against every pad row in all three planes, so the batch is
 * what amortises the weight stream; the pad planes stay in L1/L2.
 *
 * Accumulators: K * 250 * 119 < 2^31 for K < 72k. No saturation anywhere. */
#ifdef SH_SIMD_AVX512
/* A dry pool commonly needs only one or two pads for speculative decode.
 * Do not compute duplicate rows for those requests. Independent K lanes
 * retain enough accumulators to hide VNNI latency even at a single row. The
 * four-row background refill below also reuses each plane across columns. */
#define SH_SMALL_REFILL(ROWS, LANES) \
static inline void refill_rows##ROWS(const uint8_t *planes, int b, int b0, \
        const int8_t *W, int64_t K, int64_t N, int32_t *acc) { \
    const uint8_t *pl[3][ROWS]; \
    for (int p = 0; p < 3; p++) \
        for (int r = 0; r < ROWS; r++) \
            pl[p][r] = planes + ((size_t)p * b + b0 + r) * K; \
    for (int64_t j = 0; j < N; j++) { \
        __m512i a[3][ROWS][LANES]; \
        for (int p = 0; p < 3; p++) \
            for (int r = 0; r < ROWS; r++) \
                for (int lane = 0; lane < LANES; lane++) a[p][r][lane] = _mm512_setzero_si512(); \
        int64_t k = 0; \
        for (; k + 64 * LANES <= K; k += 64 * LANES) { \
            for (int lane = 0; lane < LANES; lane++) { \
                const int64_t at = k + 64 * lane; \
                const __m512i w = _mm512_loadu_si512((const void *)(W + j * K + at)); \
                for (int p = 0; p < 3; p++) \
                    for (int r = 0; r < ROWS; r++) \
                        a[p][r][lane] = _mm512_dpbusd_epi32(a[p][r][lane], \
                            _mm512_loadu_si512((const void *)(pl[p][r] + at)), w); \
            } \
        } \
        for (; k < K; k += 64) { \
            const int n = K - k < 64 ? (int)(K - k) : 64; \
            const __mmask64 mask = n == 64 ? ~(__mmask64)0 : (((__mmask64)1 << n) - 1); \
            const __m512i w = _mm512_maskz_loadu_epi8(mask, W + j * K + k); \
            for (int p = 0; p < 3; p++) \
                for (int r = 0; r < ROWS; r++) \
                    a[p][r][0] = _mm512_dpbusd_epi32(a[p][r][0], \
                        _mm512_maskz_loadu_epi8(mask, pl[p][r] + k), w); \
        } \
        for (int p = 0; p < 3; p++) \
            for (int r = 0; r < ROWS; r++) { \
                __m512i sum = a[p][r][0]; \
                for (int lane = 1; lane < LANES; lane++) sum = _mm512_add_epi32(sum, a[p][r][lane]); \
                acc[(p * 4 + r) * N + j] = _mm512_reduce_add_epi32(sum); \
            } \
    } \
}
SH_SMALL_REFILL(1, 4)
SH_SMALL_REFILL(2, 1)
SH_SMALL_REFILL(3, 1)
#undef SH_SMALL_REFILL

static inline void refill_rows4(const uint8_t *planes,int b,int b0,
        const int8_t *W,int64_t K,int64_t N,int32_t *acc) {
    const int64_t K64=K&~(int64_t)63;
    const __mmask64 tail=(K&63)?(((__mmask64)1<<(K&63))-1):0;
    const uint8_t *pl[3][4];
    for(int p=0;p<3;p++)for(int r=0;r<4;r++) {
        const int row=b0+r<b?b0+r:b-1;
        pl[p][r]=planes+((size_t)p*b+row)*K;
    }
    for(int64_t j0=0;j0<N;j0+=16) {
        __m512i saved[16/4][3][4][4];
        for(int t=0;t<16/4;t++)for(int p=0;p<3;p++)for(int c=0;c<4;c++)for(int r=0;r<4;r++)saved[t][p][c][r]=_mm512_setzero_si512();
        const int8_t *wp[16];
        for(int c=0;c<16;c++)wp[c]=W+(j0+c<N?j0+c:j0)*K;
        for(int64_t k0=0;k0<K;k0+=2048) {
            const int64_t k_end=k0+2048<K64?k0+2048:K64;
            for(int t=0;t<16/4;t++)for(int p=0;p<3;p++) {
                __m512i a[4][4];
                for(int c=0;c<4;c++)for(int r=0;r<4;r++)a[c][r]=saved[t][p][c][r];
                int64_t k=k0;
                for(;k<k_end;k+=64) {
                    __m512i x[4];
                    for(int c=0;c<4;c++)x[c]=_mm512_loadu_si512((const void*)(wp[t*4+c]+k));
                    for(int r=0;r<4;r++) {
                        const __m512i v=_mm512_loadu_si512((const void*)(pl[p][r]+k));
                        for(int c=0;c<4;c++)a[c][r]=_mm512_dpbusd_epi32(a[c][r],v,x[c]);
                    }
                }
                if(tail && k==K64 && K64<k0+2048) {
                    __m512i x[4];
                    for(int c=0;c<4;c++)x[c]=_mm512_maskz_loadu_epi8(tail,wp[t*4+c]+k);
                    for(int r=0;r<4;r++) {
                        const __m512i v=_mm512_maskz_loadu_epi8(tail,pl[p][r]+k);
                        for(int c=0;c<4;c++)a[c][r]=_mm512_dpbusd_epi32(a[c][r],v,x[c]);
                    }
                }
                for(int c=0;c<4;c++)for(int r=0;r<4;r++)saved[t][p][c][r]=a[c][r];
            }
        }
        for(int t=0;t<16/4;t++)for(int p=0;p<3;p++)for(int c=0;c<4;c++)if(j0+t*4+c<N)
            for(int r=0;r<4;r++)acc[(p*4+r)*N+j0+t*4+c]=_mm512_reduce_add_epi32(saved[t][p][c][r]);
    }
}

/* ROW-BLOCKED refill: every row of the batch against one weight tile before
 * the next tile, so the weight stream is paid once per BATCH rather than once
 * per four rows (the wrapper's four-row loop below re-read W for each group
 * of four, so a 32-row batch streamed the matrix eight times). Sixteen output
 * columns per tile, 2048-byte K slabs. Per slab the weight tile (32 KiB) stays
 * in L1 across every (row group, plane); the four mask rows of one (row group,
 * plane) (8 KiB) stay in L1 across the four column quads; the running int32
 * sums round-trip L2 once per slab (2 KiB per 512 dpbusd). Each tile's sums
 * are reduced and CRT-balanced straight into u, so no [3][4][N] accumulator is
 * touched. Duplicate tail rows (b not a multiple of four) are computed against
 * the last real row and never written. Field arithmetic, planes, tails and
 * the output layout are those of refill_rows4. */
#define SH_BLK_K 2048
static void refill_rows_blocked(const uint8_t *planes, int b, const int8_t *W,
        int64_t K, int64_t N, int32_t *u, int64_t u_stride) {
    const int G = (b + 3) / 4;
    const int64_t K64 = K & ~(int64_t)63;
    const __mmask64 tail = (K & 63) ? (((__mmask64)1 << (K & 63)) - 1) : 0;
    __m512i *saved = (__m512i *)aligned_alloc(64, (size_t)G * 3 * 4 * 16 * sizeof(__m512i));
    if (!saved) { /* out of memory: fall back to the four-row path */
        for (int b0 = 0; b0 < b; b0 += 4) {
            int32_t *acc = (int32_t *)malloc((size_t)12 * N * sizeof(int32_t));
            if (!acc) return;
            refill_rows4(planes, b, b0, W, K, N, acc);
            const int rows = b - b0 < 4 ? b - b0 : 4;
            for (int r = 0; r < rows; r++) {
                const int32_t *a0 = acc + (0 * 4 + r) * N, *a1 = acc + (1 * 4 + r) * N, *a2 = acc + (2 * 4 + r) * N;
                int32_t *o = u + (int64_t)(b0 + r) * u_stride;
                for (int64_t j = 0; j < N; j++) o[j] = crt_balanced(a0[j], a1[j], a2[j]);
            }
            free(acc);
        }
        return;
    }
    const uint8_t *pl[3][4];
    for (int64_t j0 = 0; j0 < N; j0 += 16) {
        const int8_t *wp[16];
        for (int c = 0; c < 16; c++) wp[c] = W + (j0 + c < N ? j0 + c : j0) * K;
        for (size_t i = 0; i < (size_t)G * 3 * 4 * 16; i++) saved[i] = _mm512_setzero_si512();
        for (int64_t k0 = 0; k0 < K; k0 += SH_BLK_K) {
            const int64_t k_end = k0 + SH_BLK_K < K64 ? k0 + SH_BLK_K : K64;
            const int do_tail = tail && k0 <= K64 && K64 < k0 + SH_BLK_K;
            for (int g = 0; g < G; g++) {
                for (int p = 0; p < 3; p++) for (int r = 0; r < 4; r++) {
                    const int row = g * 4 + r < b ? g * 4 + r : b - 1;
                    pl[p][r] = planes + ((size_t)p * b + row) * K;
                }
                for (int p = 0; p < 3; p++) {
                    for (int t = 0; t < 4; t++) {
                        __m512i *sv = saved + (((size_t)g * 3 + p) * 4 + t) * 16;
                        __m512i a[4][4];
                        for (int c = 0; c < 4; c++) for (int r = 0; r < 4; r++) a[c][r] = sv[c * 4 + r];
                        int64_t k = k0;
                        for (; k < k_end; k += 64) {
                            __m512i x[4];
                            for (int c = 0; c < 4; c++) x[c] = _mm512_loadu_si512((const void *)(wp[t * 4 + c] + k));
                            for (int r = 0; r < 4; r++) {
                                const __m512i v = _mm512_loadu_si512((const void *)(pl[p][r] + k));
                                for (int c = 0; c < 4; c++) a[c][r] = _mm512_dpbusd_epi32(a[c][r], v, x[c]);
                            }
                        }
                        if (do_tail) {
                            __m512i x[4];
                            for (int c = 0; c < 4; c++) x[c] = _mm512_maskz_loadu_epi8(tail, wp[t * 4 + c] + K64);
                            for (int r = 0; r < 4; r++) {
                                const __m512i v = _mm512_maskz_loadu_epi8(tail, pl[p][r] + K64);
                                for (int c = 0; c < 4; c++) a[c][r] = _mm512_dpbusd_epi32(a[c][r], v, x[c]);
                            }
                        }
                        for (int c = 0; c < 4; c++) for (int r = 0; r < 4; r++) sv[c * 4 + r] = a[c][r];
                    }
                }
            }
        }
        for (int g = 0; g < G; g++) for (int r = 0; r < 4; r++) {
            const int row = g * 4 + r;
            if (row >= b) break;
            int32_t *o = u + (int64_t)row * u_stride;
            for (int t = 0; t < 4; t++) for (int c = 0; c < 4; c++) {
                const int64_t j = j0 + t * 4 + c;
                if (j >= N) continue;
                const int32_t a0 = _mm512_reduce_add_epi32(saved[(((size_t)g * 3 + 0) * 4 + t) * 16 + c * 4 + r]);
                const int32_t a1 = _mm512_reduce_add_epi32(saved[(((size_t)g * 3 + 1) * 4 + t) * 16 + c * 4 + r]);
                const int32_t a2 = _mm512_reduce_add_epi32(saved[(((size_t)g * 3 + 2) * 4 + t) * 16 + c * 4 + r]);
                o[j] = crt_balanced(a0, a1, a2);
            }
        }
    }
    free(saved);
}
#elif defined(SH_SIMD_NEON)
/* SDOT is signed x signed; the planes are unsigned residues. The exact identity
 *   sum_k x[k]*w[k] = sum_k (x[k]-128)*w[k] + 128*sum_k w[k]
 * with x-128 in [-128,127] (no saturation) makes it one SDOT per 16 bytes, and
 * the row sum is public arithmetic on the public weights, once per row for the
 * twelve dots. Measured 6.0x over the scalar loop on an A78 (anchor REPORT 4). */
static inline void refill_rows4(const uint8_t *planes, int b, int b0,
                                const int8_t *W, int64_t K, int64_t N, int32_t *acc) {
    const int64_t K16 = K & ~(int64_t)15;
    const uint8x16_t bias = vdupq_n_u8(128);
    const uint8_t *pl[3][4];
    for (int p = 0; p < 3; p++)
        for (int r = 0; r < 4; r++) {
            const int row = b0 + r < b ? b0 + r : b - 1;          /* clamp: duplicates are discarded */
            pl[p][r] = planes + ((size_t)p * b + row) * K;
        }
    for (int64_t j = 0; j < N; j++) {
        const int8_t *w = W + j * K;
        int32_t wsum = 0;
        for (int64_t k = 0; k < K; k++) wsum += w[k];
        const int32_t corr = 128 * wsum;
        for (int p = 0; p < 3; p++)
            for (int r = 0; r < 4; r++) {
                const uint8_t *x = pl[p][r];
                int32x4_t a = vdupq_n_s32(0);
                int64_t k = 0;
                for (; k < K16; k += 16)
                    a = vdotq_s32(a, vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(x + k), bias)), vld1q_s8(w + k));
                int32_t t = vaddvq_s32(a);
                for (; k < K; k++) t += ((int32_t)x[k] - 128) * (int32_t)w[k];
                acc[(p * 4 + r) * N + j] = t + corr;
            }
    }
}
#else
static inline void refill_rows4(const uint8_t *planes, int b, int b0,
                                const int8_t *W, int64_t K, int64_t N, int32_t *acc) {
    for (int64_t j = 0; j < N; j++) {
        const int8_t *w = W + j * K;
        for (int p = 0; p < 3; p++)
            for (int r = 0; r < 4; r++) {
                const int row = b0 + r < b ? b0 + r : b - 1;
                const uint8_t *x = planes + ((size_t)p * b + row) * K;
                int32_t a = 0;
                for (int64_t k = 0; k < K; k++) a += (int32_t)x[k] * (int32_t)w[k];
                acc[(p * 4 + r) * N + j] = a;
            }
    }
}
#endif

void FN(refill)(const uint8_t *planes, int b, const int8_t *W, int64_t K, int64_t N,
                int32_t *u, int64_t u_stride, int32_t *acc) {
#ifdef SH_SIMD_AVX512
    /* a batch past four rows: one weight stream for the whole batch */
    if (b > 4) { refill_rows_blocked(planes, b, W, K, N, u, u_stride); return; }
#endif
    for (int b0 = 0; b0 < b; b0 += 4) {
        const int rows = b - b0 < 4 ? b - b0 : 4;
#ifdef SH_SIMD_AVX512
        if (rows == 1) refill_rows1(planes, b, b0, W, K, N, acc);
        else if (rows == 2) refill_rows2(planes, b, b0, W, K, N, acc);
        else if (rows == 3) refill_rows3(planes, b, b0, W, K, N, acc);
        else
#endif
        refill_rows4(planes, b, b0, W, K, N, acc);
        for (int r = 0; r < rows; r++) {
            const int32_t *a0 = acc + (0 * 4 + r) * N, *a1 = acc + (1 * 4 + r) * N, *a2 = acc + (2 * 4 + r) * N;
            int32_t *o = u + (int64_t)(b0 + r) * u_stride;
            for (int64_t j = 0; j < N; j++) o[j] = crt_balanced(a0[j], a1[j], a2[j]);
        }
    }
}

/* ---------------------------------------------------------------------------
 * The request-path Freivalds dots.
 *
 * fv_dot/fv_dot_x above are the reference forms and stay as written; these
 * are what the online check runs. Two things changed, both measured on the
 * 0.5B's lm_head (N=151936, two reps):
 *  - s is int32 in [rep][n] rows rather than int64 interleaved [n][rep]. The
 *    old layout made each rep's pass read every other 8-byte word of a 2.4 MB
 *    array (so all of it, twice); the new one streams 0.6 MB per rep with a
 *    unit stride that vectorises to vpmovsxdq + vpmullq.
 *  - all reps come out of ONE pass over y, and unmask_fv writes y and dots it
 *    in the same pass, so the 1.2 MB int64 y is written once and never
 *    re-read on the request path.
 * Bounds are the same as the reference: |y| < 2^24 and s < 2^20 keep 2^18
 * terms under 2^62: |x| < 2^26 and st < 2^31 with chunks of 32 (see fv_dot_x).
 * ------------------------------------------------------------------------ */
static inline int64_t fv_fold(int64_t total, int64_t acc) {
    acc %= SH_FV_P2; if (acc < 0) acc += SH_FV_P2;
    return (total + acc) % SH_FV_P2;
}

void FN(fv_dots)(const int64_t *y, const int32_t *s, int reps, int64_t n, int64_t *out) {
    for (int r = 0; r < reps; r++) out[r] = 0;
    for (int64_t k0 = 0; k0 < n; k0 += 262144) {
        const int64_t k1 = k0 + 262144 < n ? k0 + 262144 : n;
        if (reps == 2) {
            const int32_t *s0 = s, *s1 = s + n;
            int64_t a0 = 0, a1 = 0;
            for (int64_t j = k0; j < k1; j++) {
                const int64_t v = y[j];
                a0 += v * (int64_t)s0[j];
                a1 += v * (int64_t)s1[j];
            }
            out[0] = fv_fold(out[0], a0); out[1] = fv_fold(out[1], a1);
        } else {
            for (int r = 0; r < reps; r++) {
                const int32_t *sr = s + (size_t)r * n;
                int64_t a = 0;
                for (int64_t j = k0; j < k1; j++) a += y[j] * (int64_t)sr[j];
                out[r] = fv_fold(out[r], a);
            }
        }
    }
}

void FN(fv_dots_x)(const int64_t *x, const int32_t *st, int reps, int64_t n, int64_t *out) {
    for (int r = 0; r < reps; r++) out[r] = 0;
    for (int64_t k0 = 0; k0 < n; k0 += 32) {
        const int64_t k1 = k0 + 32 < n ? k0 + 32 : n;
        if (reps == 2) {
            const int32_t *t0 = st, *t1 = st + n;
            int64_t a0 = 0, a1 = 0;
            for (int64_t k = k0; k < k1; k++) {
                const int64_t v = x[k];
                a0 += v * (int64_t)t0[k];
                a1 += v * (int64_t)t1[k];
            }
            out[0] = fv_fold(out[0], a0); out[1] = fv_fold(out[1], a1);
        } else {
            for (int r = 0; r < reps; r++) {
                const int32_t *tr = st + (size_t)r * n;
                int64_t a = 0;
                for (int64_t k = k0; k < k1; k++) a += x[k] * (int64_t)tr[k];
                out[r] = fv_fold(out[r], a);
            }
        }
    }
}

/* unmask and the lhs dot in one pass: y[j] = balanced(ym[j] - u[j]) is stored
 * AND accumulated against every rep of s from the register it was just
 * computed in. Arithmetic identical to unmask followed by fv_dots. */
void FN(unmask_fv)(const int32_t *ym, const int32_t *u, const int32_t *s, int reps, int64_t n,
                   int64_t *y, int64_t *out) {
    for (int r = 0; r < reps; r++) out[r] = 0;
    for (int64_t k0 = 0; k0 < n; k0 += 262144) {
        const int64_t k1 = k0 + 262144 < n ? k0 + 262144 : n;
        if (reps == 2) {
            const int32_t *s0 = s, *s1 = s + n;
            int64_t a0 = 0, a1 = 0;
            for (int64_t j = k0; j < k1; j++) {
                int32_t v = ym[j] - u[j];
                v += (v <= -(int32_t)(M_MOD / 2)) ? (int32_t)M_MOD : 0;
                v -= (v > (int32_t)(M_MOD / 2)) ? (int32_t)M_MOD : 0;
                y[j] = v;
                a0 += (int64_t)v * (int64_t)s0[j];
                a1 += (int64_t)v * (int64_t)s1[j];
            }
            out[0] = fv_fold(out[0], a0); out[1] = fv_fold(out[1], a1);
        } else {
            FN(unmask)(ym + k0, u + k0, (size_t)(k1 - k0), y + k0);
            for (int r = 0; r < reps; r++) {
                const int32_t *sr = s + (size_t)r * n;
                int64_t a = 0;
                for (int64_t j = k0; j < k1; j++) a += y[j] * (int64_t)sr[j];
                out[r] = fv_fold(out[r], a);
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * The packed reply (protocol 1.2, FIELD_GEMM24): one 3-byte little-endian
 * two's-complement value per output instead of an int32. The worker's every
 * product is balanced in (-M/2, M/2] and M = 14457349 < 2^24, so 24 bits
 * carry it exactly; the reply of the 0.5B's lm_head drops from 608 KB to
 * 456 KB, and in the CVM every byte of it crosses vhost-vsock. These read
 * the narrow values straight out of the reply buffer -- no widening pass,
 * which would write and re-read the 608 KB the format just saved.
 * ------------------------------------------------------------------------ */
static inline int32_t ld24(const uint8_t *p) {
    /* Assemble into the top 24 bits and arithmetic-shift down: the sign
     * extension is the shift, not a branch. */
    const uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    return (int32_t)(v << 8) >> 8;
}

#ifdef SH_SIMD_AVX512
/* 16 packed values (48 bytes) -> 16 sign-extended int32 lanes. Reads 52
 * bytes: the last 16-byte load starts at byte 36. Callers stop the vector
 * loop two values short of the end so the over-read stays inside the
 * caller's buffer (see the i + 18 <= n bound below). */
static inline __m512i ld24x16(const uint8_t *p) {
    const __m128i shuf = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    __m512i v = _mm512_castsi128_si512(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p)), shuf));
    v = _mm512_inserti32x4(v, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 12)), shuf), 1);
    v = _mm512_inserti32x4(v, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 24)), shuf), 2);
    v = _mm512_inserti32x4(v, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 36)), shuf), 3);
    return _mm512_srai_epi32(_mm512_slli_epi32(v, 8), 8);
}
/* balanced(ym - u) on 16 lanes: the same two corrections unmask applies. */
static inline __m512i unmask16(__m512i ym, __m512i u) {
    const __m512i half = _mm512_set1_epi32((int32_t)(M_MOD / 2)), nhalf = _mm512_set1_epi32(-(int32_t)(M_MOD / 2));
    const __m512i mm = _mm512_set1_epi32((int32_t)M_MOD);
    __m512i v = _mm512_sub_epi32(ym, u);
    v = _mm512_mask_add_epi32(v, _mm512_cmple_epi32_mask(v, nhalf), v, mm);
    v = _mm512_mask_sub_epi32(v, _mm512_cmpgt_epi32_mask(v, half), v, mm);
    return v;
}
#endif

void FN(unmask24)(const uint8_t *ym, const int32_t *u, size_t n, int64_t *y) {
    size_t i = 0;
#ifdef SH_SIMD_AVX512
    for (; i + 18 <= n; i += 16) {
        const __m512i v = unmask16(ld24x16(ym + 3 * i), _mm512_loadu_si512((const void *)(u + i)));
        _mm512_storeu_si512((void *)(y + i),     _mm512_cvtepi32_epi64(_mm512_castsi512_si256(v)));
        _mm512_storeu_si512((void *)(y + i + 8), _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(v, 1)));
    }
#endif
    for (; i < n; i++) {
        int32_t v = ld24(ym + 3 * i) - u[i];
        v += (v <= -(int32_t)(M_MOD / 2)) ? (int32_t)M_MOD : 0;
        v -= (v > (int32_t)(M_MOD / 2)) ? (int32_t)M_MOD : 0;
        y[i] = v;
    }
}

/* unmask24 fused with the lhs dot, arithmetic identical to unmask24 followed
 * by fv_dots. Same chunking and bounds as unmask_fv. */
void FN(unmask24_fv)(const uint8_t *ym, const int32_t *u, const int32_t *s, int reps, int64_t n,
                     int64_t *y, int64_t *out) {
    for (int r = 0; r < reps; r++) out[r] = 0;
    for (int64_t k0 = 0; k0 < n; k0 += 262144) {
        const int64_t k1 = k0 + 262144 < n ? k0 + 262144 : n;
        if (reps == 2) {
            const int32_t *s0 = s, *s1 = s + n;
            int64_t a0 = 0, a1 = 0;
            int64_t j = k0;
#ifdef SH_SIMD_AVX512
            /* The products are |v| < 2^24 times s < 2^20: 2^44 each, and a
             * lane sees at most 2^18 / 8 of them per chunk. Multiplied as
             * signed 32x32 -> 64 (vpmuldq) on the sign-extended halves. */
            __m512i c0l = _mm512_setzero_si512(), c0h = c0l, c1l = c0l, c1h = c0l;
            for (; j + 18 <= k1; j += 16) {
                const __m512i v = unmask16(ld24x16(ym + 3 * j), _mm512_loadu_si512((const void *)(u + j)));
                const __m512i vl = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(v));
                const __m512i vh = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(v, 1));
                _mm512_storeu_si512((void *)(y + j), vl);
                _mm512_storeu_si512((void *)(y + j + 8), vh);
                const __m512i t0 = _mm512_loadu_si512((const void *)(s0 + j)), t1 = _mm512_loadu_si512((const void *)(s1 + j));
                c0l = _mm512_add_epi64(c0l, _mm512_mul_epi32(vl, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(t0))));
                c0h = _mm512_add_epi64(c0h, _mm512_mul_epi32(vh, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(t0, 1))));
                c1l = _mm512_add_epi64(c1l, _mm512_mul_epi32(vl, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(t1))));
                c1h = _mm512_add_epi64(c1h, _mm512_mul_epi32(vh, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(t1, 1))));
            }
            a0 = _mm512_reduce_add_epi64(_mm512_add_epi64(c0l, c0h));
            a1 = _mm512_reduce_add_epi64(_mm512_add_epi64(c1l, c1h));
#endif
            for (; j < k1; j++) {
                int32_t v = ld24(ym + 3 * j) - u[j];
                v += (v <= -(int32_t)(M_MOD / 2)) ? (int32_t)M_MOD : 0;
                v -= (v > (int32_t)(M_MOD / 2)) ? (int32_t)M_MOD : 0;
                y[j] = v;
                a0 += (int64_t)v * (int64_t)s0[j];
                a1 += (int64_t)v * (int64_t)s1[j];
            }
            out[0] = fv_fold(out[0], a0); out[1] = fv_fold(out[1], a1);
        } else {
            FN(unmask24)(ym + 3 * k0, u + k0, (size_t)(k1 - k0), y + k0);
            for (int r = 0; r < reps; r++) {
                const int32_t *sr = s + (size_t)r * n;
                int64_t a = 0;
                for (int64_t j = k0; j < k1; j++) a += y[j] * (int64_t)sr[j];
                out[r] = fv_fold(out[r], a);
            }
        }
    }
}

/* The outlier term: y[row][j] += x_tee[row][c] * Wc[c][j], in the TEE.
 * Blocked over j so a stretch of y stays in L1 while every channel is added
 * to it: channel-major, a site with 8 outliers read and wrote its 39 KB
 * int64 y eight times per exchange. */
void FN(outlier_add)(const int64_t *x_tee, const int8_t *wc, int nout, int64_t N, int64_t *y) {
    /* Accumulated in DOUBLE, which is exact here and vectorises where the int64
     * multiply does not: every product |x_tee * w| is an integer below 2^47
     * (|x| < 2^40 checked below, |w| <= 119) and at most 64 of them are
     * summed, so nothing rounds. The int64 form (`y[j] += xv * w[j]`) compiled
     * to vpmullq at best and cost 0.6 ms per token on Qwen2.5-0.5B once the
     * per-column calibration held back 32 channels of lm_head (4.9 M MACs per
     * token); this is ~4x cheaper and bit-identical. */
    enum { BLK = 256 };
    int64_t xmax = 0;
    for (int c = 0; c < nout; c++) { const int64_t a = x_tee[c] < 0 ? -x_tee[c] : x_tee[c]; if (a > xmax) xmax = a; }
    if (xmax >= ((int64_t)1 << 40) || nout > 64) {           /* out of the exact range: the plain form */
        for (int c = 0; c < nout; c++) {
            const int64_t xv = x_tee[c];
            if (!xv) continue;
            const int8_t *w = wc + (size_t)c * N;
            for (int64_t j = 0; j < N; j++) y[j] += xv * w[j];
        }
        return;
    }
    double acc[BLK];
    for (int64_t j0 = 0; j0 < N; j0 += BLK) {
        const int64_t n = j0 + BLK < N ? BLK : N - j0;
        for (int64_t i = 0; i < n; i++) acc[i] = 0.0;
        for (int c = 0; c < nout; c++) {
            const double xv = (double)x_tee[c];
            if (xv == 0.0) continue;
            const int8_t *w = wc + (size_t)c * N + j0;
            for (int64_t i = 0; i < n; i++) acc[i] += xv * (double)w[i];
        }
        int64_t *o = y + j0;
        for (int64_t i = 0; i < n; i++) o[i] += (int64_t)acc[i];
    }
}
