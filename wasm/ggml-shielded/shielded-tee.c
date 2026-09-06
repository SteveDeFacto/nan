#define _GNU_SOURCE
#include "shielded-tee.h"
#include "shielded-field.h"

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <time.h>
#include <unistd.h>
#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#define SH_ALIGN 64
static int64_t align_up(int64_t x) { return (x + SH_ALIGN - 1) & ~(int64_t)(SH_ALIGN - 1); }

static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
/* Profile counters, read by the backend under SHIELDED_PROFILE:
 * 0 mask  1 wire  2 refill-on-path  3 unmask  4 verify  5 calls  6 pads missed  7 pads used */
double sh_prof[8];

/* ---------------------------------------------------------------------------
 * SIMD dispatch. The two builds of shielded-simd.c are checked against each
 * other once, on this CPU, before the fast one is trusted: a vectorised loop
 * that disagrees with its scalar twin is a bug that would otherwise surface as
 * verification failures on every request.
 * ------------------------------------------------------------------------ */
#define SIMD_TABLE(sfx, nm) { nm, sh_simd_##sfx##_pad_planes, sh_simd_##sfx##_mask_planes, \
    sh_simd_##sfx##_unmask, sh_simd_##sfx##_encode, sh_simd_##sfx##_descale, sh_simd_##sfx##_fv_dot, \
    sh_simd_##sfx##_fv_dot_x, sh_simd_##sfx##_fv_prepare, sh_simd_##sfx##_refill, sh_simd_##sfx##_outlier_add, \
    sh_simd_##sfx##_fv_dots, sh_simd_##sfx##_fv_dots_x, sh_simd_##sfx##_unmask_fv, \
    sh_simd_##sfx##_unmask24, sh_simd_##sfx##_unmask24_fv }
#if !defined(__aarch64__)
static const sh_simd simd_avx512  = SIMD_TABLE(avx512, "avx512-vnni");
#endif
static const sh_simd simd_generic = SIMD_TABLE(generic, "generic");
#if defined(__aarch64__)
static const sh_simd simd_neon    = SIMD_TABLE(neon, "neon-sdot");
#endif

const sh_simd *sh_simd_generic(void) { return &simd_generic; }

/* The request path keeps the Freivalds vectors as int32 rows, one per rep
 * ([rep][n]); the reference fv_dot takes them int64 and interleaved ([n][rep]).
 * Both forms are derived from the same values and checked against each other. */
static void fv_rows_i32(const int64_t *inter, int reps, int64_t n, int32_t *rows) {
    for (int64_t j = 0; j < n; j++)
        for (int r = 0; r < reps; r++) rows[(size_t)r * n + j] = (int32_t)inter[j * reps + r];
}

static bool simd_agree(const sh_simd *a, const sh_simd *b) {
    enum { K = 96 + 32, N = 37, B = 5 };   /* K not a multiple of 64: exercises the tail */
    uint64_t seed = 0x243f6a8885a308d3ull;
#define RND() (seed ^= seed << 13, seed ^= seed >> 7, seed ^= seed << 17, seed)
    int32_t r[B * K]; int64_t x[B * K]; int8_t W[N * K];
    for (int i = 0; i < B * K; i++) { r[i] = (int32_t)(RND() % (uint64_t)SH_M_MOD); x[i] = (int64_t)(RND() % (1u << 26)) - (1 << 25); }
    for (int i = 0; i < N * K; i++) W[i] = (int8_t)((int)(RND() % 239) - 119);
    uint8_t pa[3 * B * K], pb[3 * B * K]; int8_t ma[3 * B * K], mb[3 * B * K];
    a->pad_planes(r, B * K, pa, pa + B * K, pa + 2 * B * K);
    b->pad_planes(r, B * K, pb, pb + B * K, pb + 2 * B * K);
    if (memcmp(pa, pb, sizeof pa)) return false;
    a->mask_planes(x, r, B * K, ma, ma + B * K, ma + 2 * B * K);
    b->mask_planes(x, r, B * K, mb, mb + B * K, mb + 2 * B * K);
    if (memcmp(ma, mb, sizeof ma)) return false;
    int32_t ua[B * N], ub[B * N], acc[12 * N];
    a->refill(pa, B, W, K, N, ua, N, acc);
    b->refill(pb, B, W, K, N, ub, N, acc);
    if (memcmp(ua, ub, sizeof ua)) return false;
    /* and both against the int64 truth */
    for (int bb = 0; bb < B; bb++)
        for (int j = 0; j < N; j++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++) s += (int64_t)r[bb * K + k] * W[j * K + k];
            if (ua[bb * N + j] != (int32_t)sh_balanced(s)) return false;
        }
    int64_t ya[B * N], yb[B * N];
    a->unmask(ua, ub, B * N, ya); b->unmask(ua, ub, B * N, yb);
    if (memcmp(ya, yb, sizeof ya)) return false;
    int64_t s[N * 2], sta[K * 2], stb[K * 2];
    for (int i = 0; i < N * 2; i++) s[i] = 1 + (int64_t)(RND() % (SH_FV_S_RANGE - 1));
    a->fv_prepare(W, K, N, s, 2, sta); b->fv_prepare(W, K, N, s, 2, stb);
    if (memcmp(sta, stb, sizeof sta)) return false;
    for (int rep = 0; rep < 2; rep++) {
        if (a->fv_dot(ya, s, 2, rep, N) != b->fv_dot(ya, s, 2, rep, N)) return false;
        if (a->fv_dot_x(x, sta, 2, rep, K) != b->fv_dot_x(x, sta, 2, rep, K)) return false;
    }
    /* The request-path forms: fused, int32 rows. Against each other AND against
     * the reference dots, so a layout slip fails here and not as a verification
     * failure on the first token. */
    int32_t s32[2 * N], st32[2 * K];
    fv_rows_i32(s, 2, N, s32); fv_rows_i32(sta, 2, K, st32);
    int64_t da[2], db[2], dxa[2], dxb[2], fa[2], fb[2], yfa[B * N], yfb[B * N];
    a->fv_dots(ya, s32, 2, N, da);       b->fv_dots(ya, s32, 2, N, db);
    a->fv_dots_x(x, st32, 2, K, dxa);    b->fv_dots_x(x, st32, 2, K, dxb);
    a->unmask_fv(ua, ub, s32, 2, N, yfa, fa); b->unmask_fv(ua, ub, s32, 2, N, yfb, fb);
    if (memcmp(yfa, ya, N * sizeof(int64_t)) || memcmp(yfb, ya, N * sizeof(int64_t))) return false;
    for (int rep = 0; rep < 2; rep++) {
        if (da[rep] != db[rep] || da[rep] != a->fv_dot(ya, s, 2, rep, N)) return false;
        if (fa[rep] != fb[rep] || fa[rep] != da[rep]) return false;
        if (dxa[rep] != dxb[rep] || dxa[rep] != a->fv_dot_x(x, sta, 2, rep, K)) return false;
    }
    /* The packed reply: the same balanced values as 3-byte little-endian
     * two's complement (what a 1.2 worker sends) must unmask to exactly what
     * the int32 form did, in both the plain and the fused form, from both
     * builds. B*N = 185 values: the vector loop, its two-short stop and the
     * scalar tail are all exercised. Values near +-M/2 are included so the
     * balancing corrections fire. */
    {
        uint8_t packed[3 * B * N];
        for (int i = 0; i < B * N; i++) {
            int32_t v = ua[i];
            if (i % 7 == 0) v = (int32_t)(SH_M_MOD / 2) - (i % 3);
            if (i % 11 == 0) v = -(int32_t)(SH_M_MOD / 2) + 1 + (i % 3);
            ua[i] = v;
            packed[3 * i] = (uint8_t)v; packed[3 * i + 1] = (uint8_t)(v >> 8); packed[3 * i + 2] = (uint8_t)(v >> 16);
        }
        int64_t y32[B * N], y24a[B * N], y24b[B * N], f32[2], f24a[2], f24b[2];
        a->unmask(ua, ub, B * N, y32);
        a->unmask24(packed, ub, B * N, y24a); b->unmask24(packed, ub, B * N, y24b);
        if (memcmp(y32, y24a, sizeof y32) || memcmp(y32, y24b, sizeof y32)) return false;
        a->unmask_fv(ua, ub, s32, 2, N, y32, f32);
        a->unmask24_fv(packed, ub, s32, 2, N, y24a, f24a); b->unmask24_fv(packed, ub, s32, 2, N, y24b, f24b);
        if (memcmp(y32, y24a, N * sizeof(int64_t)) || memcmp(y32, y24b, N * sizeof(int64_t))) return false;
        if (f32[0] != f24a[0] || f32[1] != f24a[1] || f32[0] != f24b[0] || f32[1] != f24b[1]) return false;
        /* and the whole B*N run through the fused form as one long row */
        a->unmask24_fv(packed, ub, s32, 1, B * N, y24a, f24a); b->unmask24_fv(packed, ub, s32, 1, B * N, y24b, f24b);
        a->unmask(ua, ub, B * N, y32);
        if (memcmp(y32, y24a, sizeof y32) || memcmp(y32, y24b, sizeof y32) || f24a[0] != f24b[0]) return false;
    }
    /* encode: the vector path rounds under MXCSR exactly as lrintf does, including
     * ties to even, and its scalar tail must agree with itself. */
    float fsrc[B * K]; int64_t ea[B * K], eb[B * K];
    for (int i = 0; i < B * K; i++) fsrc[i] = ((float)(int)(RND() % 200001) - 100000.0f) / 8.0f;   /* .0/.125 steps: exact ties */
    a->encode(fsrc, B * K, 4.0f, ea); b->encode(fsrc, B * K, 4.0f, eb);
    if (memcmp(ea, eb, sizeof ea)) return false;
#undef RND
    return true;
}

const sh_simd *sh_simd_get(void) {
    static const sh_simd *chosen = NULL;
    if (chosen) return chosen;
    const char *off = getenv("SHIELDED_NO_SIMD");
#if defined(__aarch64__)
    /* The phone anchor: SDOT (FEAT_DotProd) is the whole refill win; checked
     * against the generic build exactly as AVX-512 is on x86. */
    bool want_neon = !(off && *off && strcmp(off, "0"));
    if (want_neon) want_neon = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    if (want_neon && !simd_agree(&simd_neon, &simd_generic)) {
        fprintf(stderr, "[shielded] the NEON kernels disagree with the generic ones on this CPU; using generic\n");
        want_neon = false;
    }
    chosen = want_neon ? &simd_neon : &simd_generic;
    return chosen;
#else
    bool want_avx512 = !(off && *off && strcmp(off, "0"));
    if (want_avx512) {
        __builtin_cpu_init();
        want_avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                      __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
                      __builtin_cpu_supports("avx512vnni");
    }
    if (want_avx512 && !simd_agree(&simd_avx512, &simd_generic)) {
        fprintf(stderr, "[shielded] the AVX-512 kernels disagree with the generic ones on this CPU; using generic\n");
        want_avx512 = false;
    }
    chosen = want_avx512 ? &simd_avx512 : &simd_generic;
    return chosen;
#endif
}
const sh_simd *sh_link_simd(void) { return sh_simd_get(); }

/* ---------------------------------------------------------------------------
 * OS entropy. Both the pad seed and the Freivalds secret come from here and
 * nowhere else -- see rule 4 in the header, and the commit that had to fix it.
 * ------------------------------------------------------------------------ */
static bool os_random(void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    while (n) {
        ssize_t r = getrandom(p, n, 0);
        if (r < 0) return false;
        p += r; n -= (size_t)r;
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * ChaCha20 keystream, for the pad bank.
 *
 * Pads are the one value that never has to agree across languages: only the
 * TEE generates them and only the TEE consumes them, so the requirement is
 * cryptographic strength and speed, not reproducibility. ChaCha20 keeps this
 * file dependency-free, which matters for something linked into the engine
 * inside the measurement.
 * ------------------------------------------------------------------------ */
#define ROTL32(v, c) (((v) << (c)) | ((v) >> (32 - (c))))
#define QR(a, b, c, d) ( \
    a += b, d ^= a, d = ROTL32(d, 16), \
    c += d, b ^= c, b = ROTL32(b, 12), \
    a += b, d ^= a, d = ROTL32(d, 8),  \
    c += d, b ^= c, b = ROTL32(b, 7))

static void chacha20_block(const uint32_t key[8], uint64_t counter, uint32_t out[16]) {
    static const uint32_t C[4] = { 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574 };
    uint32_t s[16];
    s[0] = C[0]; s[1] = C[1]; s[2] = C[2]; s[3] = C[3];
    for (int i = 0; i < 8; i++) s[4 + i] = key[i];
    s[12] = (uint32_t)counter; s[13] = (uint32_t)(counter >> 32);
    s[14] = 0; s[15] = 0;
    uint32_t x[16]; memcpy(x, s, sizeof x);
    for (int i = 0; i < 10; i++) {
        QR(x[0], x[4], x[ 8], x[12]); QR(x[1], x[5], x[ 9], x[13]);
        QR(x[2], x[6], x[10], x[14]); QR(x[3], x[7], x[11], x[15]);
        QR(x[0], x[5], x[10], x[15]); QR(x[1], x[6], x[11], x[12]);
        QR(x[2], x[7], x[ 8], x[13]); QR(x[3], x[4], x[ 9], x[14]);
    }
    for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
}

typedef struct {
    pthread_mutex_t mu;
    uint32_t key[8];
    uint64_t counter;      /* strictly monotonic; the machine-checkable form of "never reused" */
    uint64_t issued_hi;
    uint64_t capacity;
} sh_maskbank;

static bool maskbank_init(sh_maskbank *b) {
    pthread_mutex_init(&b->mu, NULL);
    b->counter = 0; b->issued_hi = 0; b->capacity = UINT64_C(1) << 40;
    return os_random(b->key, sizeof b->key);
}

/* Fill `n` pad values uniform over [0, M). Drawn as uint64 and reduced, so the
 * modulo bias is ~2^-40 rather than the ~2^-8 a uint32 draw would carry. One
 * issuance index covers one call, however many values it produces; the index
 * never repeats, so no two calls share keystream. */
static int maskbank_issue(sh_maskbank *b, int32_t *dst, size_t n) {
    pthread_mutex_lock(&b->mu);
    if (b->counter >= b->capacity) { pthread_mutex_unlock(&b->mu); return SH_ERR_EXHAUST; }
    const uint64_t index = b->counter++;
    if (b->counter <= b->issued_hi) { pthread_mutex_unlock(&b->mu); return SH_ERR_EXHAUST; }
    b->issued_hi = b->counter;
    pthread_mutex_unlock(&b->mu);
    uint32_t blk[16];
    uint64_t ctr = index << 24;      /* room for 2^24 blocks under one index */
    size_t produced = 0;
    while (produced < n) {
        chacha20_block(b->key, ctr++, blk);
        for (int i = 0; i + 1 < 16 && produced < n; i += 2) {
            const uint64_t v = ((uint64_t)blk[i + 1] << 32) | blk[i];
            dst[produced++] = (int32_t)(v % (uint64_t)SH_M_MOD);
        }
    }
    return SH_OK;
}

/* ---------------------------------------------------------------------------
 * Nodes and groups
 * ------------------------------------------------------------------------ */
typedef struct {
    char     name[64];
    const int8_t *w;            /* (N,K) borrowed */
    int64_t   K, N;
    int32_t   max_m;
    int       group;
    int64_t   w_off, x_off, y_off;
    int64_t   u_off;            /* this node's columns within its group's u rows */
    int64_t  *s, *s_tilde;      /* Freivalds, reference layout: (N,REPS) and (K,REPS) mod P2 */
    int32_t  *s32, *st32;       /* the same values as the request path reads them: [REPS][N], [REPS][K] */
} sh_node;

/* THE RING. `depth` slots of (r, u). Around it, in ring order:
 *
 *     [head-held, head)                 HELD: handed to the request in flight
 *     [head, head+count)                READY: generated, waiting
 *     [head+count, +generating)         RESERVED: a refill thread is writing
 *
 * A request takes from the front by advancing head -- the pad is CONSUMED at
 * that moment, whatever happens to the request afterwards (rule 1: a pad that
 * left the ready run never comes back). It holds the slot by POINTER until its
 * unmask is done and then releases it; a refill may only reserve a slot that
 * is neither ready, held nor reserved, which is what `held` in the deficit
 * guarantees. Refill threads reserve slot indices under the lock and write
 * the ring OUTSIDE it, so a lm_head refill (608 KB of u) no longer holds the
 * mutex against the request path; `ready` lets them finish out of order while
 * `count` still advances in ring order. Before this, take_pads memcpy'd r and
 * u out of the ring under the lock on every exchange -- 608 KB per lm_head
 * token -- and the refill thread memcpy'd them in under the same lock. */
typedef struct {
    int64_t   K;
    int       nodes[SH_GROUP_MAX];
    int       n_nodes;
    int64_t   u_len;            /* sum of N over the group */
    int32_t   max_m;
    int       depth, count, head, generating, held;
    uint8_t  *ready;            /* per slot: written by a refill, not yet counted */
    int32_t  *r_store;          /* depth x K, each in [0,M) */
    int32_t  *u_store;          /* depth x u_len, balanced */
    uint64_t pads_used, pads_missed;
    double on_path_ms;          /* request-thread counters; refill threads do not write them */
} sh_group;

struct sh_link {
    sh_pipe   *pipe;
    char       host[128];
    int        port, vsock_port;
    uint64_t   reserve_bytes;
    bool       shm_configured;
    char       shm_path[256];
    size_t     shm_bytes;
    bool       verify;
    /* Bytes per reply value: 4 (FIELD_GEMM, protocol 1.1) or 3 (FIELD_GEMM24,
     * 1.2). Decided at start from the worker's HELLO; SHIELDED_REPLY32=1
     * forces the wide form against a worker that offers both. */
    int        ywidth;
    sh_node   *nodes;   size_t n_nodes,  cap_nodes;
    sh_group  *groups;  size_t n_groups, cap_groups;
    int64_t    wbytes, abytes;
    sh_maskbank bank;
    const sh_simd *simd;

    pthread_mutex_t pool_mu;
    pthread_cond_t  need_refill;   /* a deficit appeared */
    pthread_cond_t  pool_filled;   /* a pad was published (start waits on it) */
    pthread_t *threads; int n_threads; bool threads_running, stop;
    int threads_env;               /* SHIELDED_REFILL_THREADS, or -1 = derive from the weights */
    int pool_depth, refill_batch, target_ms, warm_ms, pad_wait_us;
    int64_t Kmax, Nmax, ulen_max;

    /* request-path scratch (caller thread) */
    const int32_t **rp; const int32_t **up; int *slots; size_t p_cap;   /* per-row pad pointers, m of each */
    int32_t *r;       size_t r_cap;      /* pads generated ON the path, pool dry */
    int32_t *u;       size_t u_cap;
    int8_t  *planes;  size_t planes_cap;
    uint8_t *gplanes; size_t gplanes_cap;
    int32_t *acc;     size_t acc_cap;
    uint8_t *hdr;     size_t hdr_cap;

    uint64_t   exchanges, macs, verify_fail, pads_used, pads_missed;
    uint64_t   pads_waited;
    double     pad_wait_ms;
    double     last_wire_us;
    char       transport[192];
    char       err[256];
};

const char *sh_link_transport(const sh_link *l) { return l && l->transport[0] ? l->transport : "not connected"; }
double sh_link_last_wire_us(const sh_link *l) { return l ? l->last_wire_us : 0.0; }

const char *sh_link_last_error(const sh_link *l) { return l ? l->err : ""; }

void sh_link_stats(const sh_link *l, uint64_t *e, uint64_t *m, uint64_t *v) {
    if (!l) return;
    if (e) *e = l->exchanges;
    if (m) *m = l->macs;
    if (v) *v = l->verify_fail;
}
void sh_link_pool_stats(const sh_link *l, uint64_t *consumed, uint64_t *missed) {
    if (!l) return;
    if (consumed) *consumed = l->pads_used;
    if (missed) *missed = l->pads_missed;
}
void sh_link_pad_wait_stats(const sh_link *l, uint64_t *waited, double *wait_ms) {
    if (waited) *waited = l ? l->pads_waited : 0;
    if (wait_ms) *wait_ms = l ? l->pad_wait_ms : 0;
}
void sh_link_node_pool_stats(const sh_link *l, int node, uint64_t *consumed,
                             uint64_t *missed, double *on_path_ms) {
    const sh_group *g = l && node >= 0 && (size_t)node < l->n_nodes
                      ? &l->groups[l->nodes[node].group] : NULL;
    if (consumed) *consumed = g ? g->pads_used : 0;
    if (missed) *missed = g ? g->pads_missed : 0;
    if (on_path_ms) *on_path_ms = g ? g->on_path_ms : 0;
}
int sh_link_refill_threads(const sh_link *l) { return l ? l->n_threads : 0; }
int sh_link_reply_width(const sh_link *l) { return l && l->pipe ? l->ywidth : 0; }

static int env_int(const char *name, int dflt, int lo, int hi) {
    const char *e = getenv(name);
    if (!e || !*e) return dflt;
    int v = atoi(e);
    return v < lo ? lo : v > hi ? hi : v;
}

sh_link *sh_link_open(const char *host, int port, bool verify, int *err) {
    sh_link *l = (sh_link *)calloc(1, sizeof *l);
    if (!l) { if (err) *err = SH_ERR_NOMEM; return NULL; }
    snprintf(l->host, sizeof l->host, "%s", host);
    l->port = port; l->verify = verify; l->ywidth = 4;
    l->vsock_port = env_int("SHIELDED_VSOCK_PORT", -1, -1, 1 << 30);
    const char *r = getenv("SHIELDED_RESERVE_BYTES");
    if (r && *r) { char *end = NULL; unsigned long long v = strtoull(r, &end, 10); if (end && *end == 0) l->reserve_bytes = v; }
    l->simd = sh_simd_get();
    if (!maskbank_init(&l->bank)) { free(l); if (err) *err = SH_ERR_IO; return NULL; }
    pthread_mutex_init(&l->pool_mu, NULL);
    pthread_cond_init(&l->need_refill, NULL);
    pthread_condattr_t filled_attr;
    pthread_condattr_init(&filled_attr);
    pthread_condattr_setclock(&filled_attr, CLOCK_MONOTONIC);
    pthread_cond_init(&l->pool_filled, &filled_attr);
    pthread_condattr_destroy(&filled_attr);
    l->threads_env  = env_int("SHIELDED_REFILL_THREADS", -1, 0, 64);
    l->pool_depth   = env_int("SHIELDED_POOL_DEPTH", -1, -1, 4096);   /* -1: 4 x the widest max_m, at least 16 */
    l->refill_batch = env_int("SHIELDED_REFILL_BATCH", 4, 1, 64);
    l->target_ms    = env_int("SHIELDED_REFILL_TARGET_MS", 6, 1, 10000);
    l->warm_ms      = env_int("SHIELDED_WARM_MS", 5000, 0, 600000);
    l->pad_wait_us  = env_int("SHIELDED_PAD_WAIT_US", 0, 0, 50000);
    if (err) *err = SH_OK;
    return l;
}

void sh_link_configure(sh_link *l, int vsock_port, uint64_t reserve_bytes, int refill_threads) {
    if (!l || l->pipe || l->threads_running) return;
    l->vsock_port = vsock_port;
    l->reserve_bytes = reserve_bytes;
    if (refill_threads >= 0 && refill_threads <= 64) l->threads_env = refill_threads;
}

void sh_link_configure_shm(sh_link *l, const char *path, uint64_t bytes) {
    if (!l || l->pipe || l->threads_running) return;
    l->shm_configured = true;
    l->shm_path[0] = 0; l->shm_bytes = 0;
    if (!path || !*path || strlen(path) >= sizeof l->shm_path ||
        bytes < SH_RING_BYTES || bytes > SH_RING_MAX_FILE) return;
    snprintf(l->shm_path, sizeof l->shm_path, "%s", path);
    l->shm_bytes = (size_t)bytes;
}

static void stop_threads(sh_link *l) {
    if (!l->threads_running) return;
    pthread_mutex_lock(&l->pool_mu);
    l->stop = true;
    pthread_cond_broadcast(&l->need_refill);
    pthread_cond_broadcast(&l->pool_filled);
    pthread_mutex_unlock(&l->pool_mu);
    for (int i = 0; i < l->n_threads; i++) pthread_join(l->threads[i], NULL);
    free(l->threads); l->threads = NULL;
    l->threads_running = false; l->stop = false;
}

static void free_pools(sh_link *l) {
    for (size_t g = 0; g < l->n_groups; g++) {
        free(l->groups[g].r_store); free(l->groups[g].u_store); free(l->groups[g].ready);
        l->groups[g].r_store = NULL; l->groups[g].u_store = NULL; l->groups[g].ready = NULL;
        l->groups[g].count = l->groups[g].head = l->groups[g].generating = l->groups[g].held = 0;
    }
}

void sh_link_close(sh_link *l) {
    if (!l) return;
    stop_threads(l);
    free_pools(l);
    for (size_t i = 0; i < l->n_nodes; i++) {
        free(l->nodes[i].s); free(l->nodes[i].s_tilde); free(l->nodes[i].s32); free(l->nodes[i].st32);
    }
    free(l->nodes); free(l->groups);
    free(l->rp); free(l->up); free(l->slots);
    free(l->r); free(l->u); free(l->planes); free(l->gplanes); free(l->acc); free(l->hdr);
    sh_pipe_close(l->pipe);
    pthread_mutex_destroy(&l->pool_mu); pthread_cond_destroy(&l->need_refill); pthread_cond_destroy(&l->pool_filled);
    pthread_mutex_destroy(&l->bank.mu);
    free(l);
}

/* fv_prepare over row ranges on several threads: the products of the ranges
 * are independent and sum mod P2. Registration is serial in the engine's
 * context creation, so this is the one place the link spends threads before
 * the pool exists; the count follows the machine, not the pool policy. */
typedef struct { const sh_simd *simd; const int8_t *W; int64_t K, N; const int64_t *s; int reps; int64_t *st; } fv_job;
static void *fv_job_main(void *arg) { fv_job *j = (fv_job *)arg; j->simd->fv_prepare(j->W, j->K, j->N, j->s, j->reps, j->st); return NULL; }
static void fv_prepare_parallel(sh_link *l, const int8_t *W, int64_t K, int64_t N, const int64_t *s, int reps, int64_t *st) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int nt = ncpu > 1 ? (int)(ncpu > 16 ? 16 : ncpu) : 1;
    if (N < 64 || (int64_t)nt * 8 > N) nt = 1;
    if (nt == 1) { l->simd->fv_prepare(W, K, N, s, reps, st); return; }
    fv_job *jobs = (fv_job *)calloc((size_t)nt, sizeof *jobs);
    pthread_t *th = (pthread_t *)calloc((size_t)nt, sizeof *th);
    int64_t *part = (int64_t *)malloc((size_t)nt * K * reps * sizeof(int64_t));
    if (!jobs || !th || !part) { free(jobs); free(th); free(part); l->simd->fv_prepare(W, K, N, s, reps, st); return; }
    int made = 0;
    for (int t = 0; t < nt; t++) {
        const int64_t j0 = N * t / nt, j1 = N * (t + 1) / nt;
        jobs[t] = (fv_job){ l->simd, W + j0 * K, K, j1 - j0, s + j0 * reps, reps, part + (size_t)t * K * reps };
        if (pthread_create(&th[t], NULL, fv_job_main, &jobs[t]) == 0) made++; else fv_job_main(&jobs[t]);
    }
    for (int t = 0; t < made; t++) pthread_join(th[t], NULL);
    for (int64_t i = 0; i < K * reps; i++) {
        int64_t v = 0;
        for (int t = 0; t < nt; t++) v = (v + part[(size_t)t * K * reps + i]) % SH_FV_P2;
        st[i] = v;
    }
    free(jobs); free(th); free(part);
}

static int fv_prepare(sh_link *l, sh_node *nd) {
    const int64_t K = nd->K, N = nd->N;
    nd->s       = (int64_t *)malloc((size_t)N * SH_FV_REPS * sizeof(int64_t));
    nd->s_tilde = (int64_t *)malloc((size_t)K * SH_FV_REPS * sizeof(int64_t));
    nd->s32     = (int32_t *)malloc((size_t)N * SH_FV_REPS * sizeof(int32_t));
    nd->st32    = (int32_t *)malloc((size_t)K * SH_FV_REPS * sizeof(int32_t));
    if (!nd->s || !nd->s_tilde || !nd->s32 || !nd->st32) return SH_ERR_NOMEM;
    /* s from the OS CSPRNG. Predictable s == forgeable results; see rule 4. */
    uint64_t *raw = (uint64_t *)malloc((size_t)N * SH_FV_REPS * sizeof(uint64_t));
    if (!raw) return SH_ERR_NOMEM;
    if (!os_random(raw, (size_t)N * SH_FV_REPS * sizeof(uint64_t))) { free(raw); return SH_ERR_IO; }
    for (int64_t i = 0; i < N * SH_FV_REPS; i++)
        nd->s[i] = 1 + (int64_t)(raw[i] % (uint64_t)(SH_FV_S_RANGE - 1));
    free(raw);
    fv_prepare_parallel(l, nd->w, K, N, nd->s, SH_FV_REPS, nd->s_tilde);
    /* s < 2^20 and s_tilde < P2 < 2^31: both fit the int32 rows the online
     * check streams. The int64 forms stay for sh_link_verify's reference path. */
    fv_rows_i32(nd->s, SH_FV_REPS, N, nd->s32);
    fv_rows_i32(nd->s_tilde, SH_FV_REPS, K, nd->st32);
    return SH_OK;
}

int sh_link_add_weight(sh_link *l, const char *name, const int8_t *w_fixed,
                       int64_t K, int64_t N, int32_t max_m, int share_x_with) {
    if (K % SH_QK != 0 || K % 16 != 0) {
        snprintf(l->err, sizeof l->err, "K=%lld not a multiple of %d", (long long)K, SH_QK); return SH_ERR_PROTO;
    }
    /* A weight added after start (a split that first shows a weight on a
     * later graph, a retry that registers more) changes the groups the refill
     * threads are writing -- u_len is their row stride into u_store. Stop them
     * and drop the rings first; the ready pads are discarded, never re-issued
     * (the bank's counter is monotonic), and the caller restarts the link,
     * which rebuilds the pools. ASan-confirmed heap overflow without this. */
    if (l->threads_running) { stop_threads(l); free_pools(l); }
    if (l->n_nodes == l->cap_nodes) {
        size_t cap = l->cap_nodes ? l->cap_nodes * 2 : 16;
        sh_node *nn = (sh_node *)realloc(l->nodes, cap * sizeof *nn);
        if (!nn) return SH_ERR_NOMEM;
        l->nodes = nn; l->cap_nodes = cap;
    }
    sh_node *nd = &l->nodes[l->n_nodes];
    memset(nd, 0, sizeof *nd);
    snprintf(nd->name, sizeof nd->name, "%s", name);
    nd->w = w_fixed; nd->K = K; nd->N = N; nd->max_m = max_m;

    /* Rejecting here is what keeps the residue identity honest: a weight above
     * the byte lane would wrap in every plane on the worker and unmask to noise. */
    {   /* a min/max scan (vectorises) rather than an early-exit compare loop */
        int lo = 0, hi = 0;
        for (int64_t i = 0; i < K * N; i++) { const int v = w_fixed[i]; lo = v < lo ? v : lo; hi = v > hi ? v : hi; }
        if (hi > SH_WEIGHT_BYTE_LIMIT || lo < -SH_WEIGHT_BYTE_LIMIT) {
            snprintf(l->err, sizeof l->err, "%s: fixed weight %d exceeds the int8 lane (+-%d)",
                     name, hi > SH_WEIGHT_BYTE_LIMIT ? hi : lo, SH_WEIGHT_BYTE_LIMIT);
            return SH_ERR_RANGE;
        }
    }

    nd->w_off = align_up(l->wbytes);
    l->wbytes = nd->w_off + K * N;
    nd->x_off = align_up(l->abytes);
    nd->y_off = align_up(nd->x_off + 3 * (int64_t)max_m * K);
    l->abytes = nd->y_off + (int64_t)max_m * N * 4;

    if (share_x_with >= 0) {
        if ((size_t)share_x_with >= l->n_nodes) { snprintf(l->err, sizeof l->err, "share_x_with out of range"); return SH_ERR_PROTO; }
        sh_group *g = &l->groups[l->nodes[share_x_with].group];
        if (g->K != K || g->n_nodes >= SH_GROUP_MAX) {
            snprintf(l->err, sizeof l->err, "node %zu cannot share x with %d", l->n_nodes, share_x_with);
            return SH_ERR_PROTO;
        }
        nd->group = l->nodes[share_x_with].group;
        nd->u_off = g->u_len;
        g->nodes[g->n_nodes++] = (int)l->n_nodes;
        g->u_len += N;
        if (max_m < g->max_m) g->max_m = max_m;
    } else {
        if (l->n_groups == l->cap_groups) {
            size_t cap = l->cap_groups ? l->cap_groups * 2 : 16;
            sh_group *ng = (sh_group *)realloc(l->groups, cap * sizeof *ng);
            if (!ng) return SH_ERR_NOMEM;
            l->groups = ng; l->cap_groups = cap;
        }
        sh_group *g = &l->groups[l->n_groups];
        memset(g, 0, sizeof *g);
        g->K = K; g->nodes[0] = (int)l->n_nodes; g->n_nodes = 1; g->u_len = N; g->max_m = max_m;
        nd->group = (int)l->n_groups++;
        nd->u_off = 0;
    }
    if (K > l->Kmax) l->Kmax = K;
    if (N > l->Nmax) l->Nmax = N;
    if (l->groups[nd->group].u_len > l->ulen_max) l->ulen_max = l->groups[nd->group].u_len;

    if (l->verify) {
        int rc = fv_prepare(l, nd);
        if (rc != SH_OK) return rc;
    }
    return (int)l->n_nodes++;
}

/* ---------------------------------------------------------------------------
 * Pad generation: r from the bank, u = r.W for every node of the group.
 * Runs on the refill threads, and on the request path only when the pool is
 * dry (counted, because that number should be ~0 in steady state).
 * ------------------------------------------------------------------------ */
typedef struct {
    uint8_t *planes; int32_t *acc;
} gen_scratch;

static int gen_scratch_init(const sh_link *l, gen_scratch *s, int b) {
    s->planes = (uint8_t *)malloc((size_t)3 * b * l->Kmax);
    s->acc    = (int32_t *)malloc((size_t)12 * l->Nmax * sizeof(int32_t));
    return s->planes && s->acc ? SH_OK : SH_ERR_NOMEM;
}

static int generate(sh_link *l, const sh_group *g, int b, int32_t *r_out, int32_t *u_out, gen_scratch *s) {
    const int64_t K = g->K;
    int rc = maskbank_issue(&l->bank, r_out, (size_t)b * K);
    if (rc != SH_OK) return rc;
    l->simd->pad_planes(r_out, (size_t)b * K, s->planes, s->planes + (size_t)b * K, s->planes + (size_t)2 * b * K);
    for (int i = 0; i < g->n_nodes; i++) {
        const sh_node *nd = &l->nodes[g->nodes[i]];
        l->simd->refill(s->planes, b, nd->w, K, nd->N, u_out + nd->u_off, g->u_len, s->acc);
    }
    return SH_OK;
}

/* Slots a refill may write: everything that is not ready, held or reserved. */
static int group_deficit(const sh_group *g) { return g->depth - g->count - g->held - g->generating; }

static void *refill_main(void *arg) {
    sh_link *l = (sh_link *)arg;
    gen_scratch s;
    const int B = l->refill_batch;
    if (gen_scratch_init(l, &s, B) != SH_OK) return NULL;
    /* Private staging, used only when the reserved slots wrap around the ring
     * end; otherwise the batch is generated straight into the ring. */
    int32_t *r = (int32_t *)malloc((size_t)B * l->Kmax * sizeof(int32_t));
    int32_t *u = (int32_t *)malloc((size_t)B * l->ulen_max * sizeof(int32_t));
    if (!r || !u) { free(r); free(u); free(s.planes); free(s.acc); return NULL; }
    for (;;) {
        pthread_mutex_lock(&l->pool_mu);
        sh_group *g = NULL; int deficit = 0;
        for (;;) {
            if (l->stop) { pthread_mutex_unlock(&l->pool_mu); goto done; }
            /* Which group, and whether to bother. Every partial batch still
             * streams the group's weights, while a full four-row batch
             * amortises that stream across four pads. In
             * steady state every group loses exactly one pad per token, and
             * refilling each deficit-1 group as it appeared kept two cores
             * busy at a quarter efficiency for the whole decode. So: a group
             * that is LOW (fewer than a batch ready or coming) is refilled at
             * once with whatever fits -- at start-up that is every group in
             * turn, so the first token finds a pad in each -- and otherwise a
             * group is refilled only once a whole batch fits, largest deficit
             * first. Lowest-first among the low ones. */
            int best_low = -1;      /* ready+coming of the lowest low group seen, or -1 */
            for (size_t i = 0; i < l->n_groups; i++) {
                sh_group *c = &l->groups[i];
                const int d = group_deficit(c);
                if (d <= 0) continue;
                const int coming = c->count + c->generating;
                if (coming < B) {
                    if (best_low < 0 || coming < best_low) { g = c; deficit = d; best_low = coming; }
                } else if (best_low < 0 && d >= B && d > deficit) { g = c; deficit = d; }
            }
            if (g) break;
            pthread_cond_wait(&l->need_refill, &l->pool_mu);
        }
        const int b = deficit < B ? deficit : B;
        const int first = (g->head + g->count + g->generating) % g->depth;
        g->generating += b;
        pthread_mutex_unlock(&l->pool_mu);

        const bool direct = first + b <= g->depth;
        int32_t *r_out = direct ? g->r_store + (size_t)first * g->K     : r;
        int32_t *u_out = direct ? g->u_store + (size_t)first * g->u_len : u;
        const int rc = generate(l, g, b, r_out, u_out, &s);
        if (rc == SH_OK && !direct) {
            for (int i = 0; i < b; i++) {
                const int slot = (first + i) % g->depth;
                memcpy(g->r_store + (size_t)slot * g->K,     r + (size_t)i * g->K,     (size_t)g->K * sizeof(int32_t));
                memcpy(g->u_store + (size_t)slot * g->u_len, u + (size_t)i * g->u_len, (size_t)g->u_len * sizeof(int32_t));
            }
        }

        pthread_mutex_lock(&l->pool_mu);
        if (rc == SH_OK) {
            for (int i = 0; i < b; i++) g->ready[(first + i) % g->depth] = 1;
            /* Count forward in ring order over whatever is finished. */
            while (g->generating > 0 && g->ready[(g->head + g->count) % g->depth]) {
                g->ready[(g->head + g->count) % g->depth] = 0;
                g->count++; g->generating--;
            }
            pthread_cond_broadcast(&l->pool_filled);
        } else {
            /* Bank exhausted: the reservation is abandoned and nothing more
             * happens until the process restarts. */
            g->generating -= b;
            l->stop = true;
            pthread_cond_broadcast(&l->pool_filled);
        }
        pthread_mutex_unlock(&l->pool_mu);
    }
done:
    free(r); free(u); free(s.planes); free(s.acc);
    return NULL;
}

/* How many refill threads the registered weights need.
 *
 * Each pad of a group costs three residue planes of u = r.W, i.e. 3.K.N int8
 * MACs, and the request path consumes one pad per group per token, so refill
 * has to sustain 3 x (the offloaded MACs of one token) per token time. Measured
 * (2026-08-26, EPYC 9115, AVX-512 VNNI, batch 4): ~250 G-MAC/s per core, and
 * the 0.5B decodes at ~6 ms/token on this box, which is the default target.
 *   0.5B: 1.36 G MAC/token -> 5.4 core-ms/token -> 1 thread, clamped up to 2
 *   4B:   12  G MAC/token -> 48 core-ms/token   -> 10 threads
 * The 1.25 is headroom for sharing the cores with the engine's own threads.
 * Clamped to [2, ncores/2]; SHIELDED_REFILL_THREADS overrides the whole thing. */
#define SH_REFILL_GMACS_PER_CORE 250.0
static int derive_threads(const sh_link *l) {
    if (l->threads_env >= 0) return l->threads_env;
    double macs = 0;
    for (size_t i = 0; i < l->n_nodes; i++) macs += (double)l->nodes[i].K * (double)l->nodes[i].N;
    const double core_ms = 3.0 * macs / (SH_REFILL_GMACS_PER_CORE * 1e9) * 1e3;
    /* Pads per token is the batch width, not 1: a tenant serving several users
     * (or verifying a speculative draft) takes m per group per step, and a
     * step is only ~2-4x longer than a single token at m=8. Sized for half
     * the widest batch the graph may present, which over-provisions a
     * single-user decode by idle threads that cost nothing, and stops a
     * batched one from generating pads on the request path. */
    int32_t mmax = 1;
    for (size_t i = 0; i < l->n_groups; i++) if (l->groups[i].max_m > mmax) mmax = l->groups[i].max_m;
    const double m_eff = mmax > 2 ? mmax / 2.0 : 1.0;
    int want = (int)(core_ms * m_eff * 1.25 / (double)l->target_ms + 0.999);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int hi = ncpu > 4 ? (int)(ncpu / 2) : 2;
    if (want < 2) want = 2;
    if (want > hi) want = hi;
    return want;
}

static int start_pools(sh_link *l) {
    stop_threads(l);
    free_pools(l);
    for (size_t i = 0; i < l->n_groups; i++) {
        sh_group *g = &l->groups[i];
        /* A batched step takes m pads from a group at once, and the pool is
         * refilled between steps: a depth below ~4m starves it (bench-batch at
         * m=8 with the old fixed 16: hundreds of on-path refills). Sized from
         * what the graph may ask for; SHIELDED_POOL_DEPTH overrides. */
        g->depth = l->pool_depth > 0 ? l->pool_depth : (g->max_m * 4 > 16 ? g->max_m * 4 : 16);
        g->r_store = (int32_t *)malloc((size_t)g->depth * g->K * sizeof(int32_t));
        g->u_store = (int32_t *)malloc((size_t)g->depth * g->u_len * sizeof(int32_t));
        g->ready   = (uint8_t *)calloc((size_t)g->depth, 1);
        if (!g->r_store || !g->u_store || !g->ready) return SH_ERR_NOMEM;
    }
    l->n_threads = derive_threads(l);
    if (l->n_threads > 0) {
        l->threads = (pthread_t *)calloc((size_t)l->n_threads, sizeof(pthread_t));
        if (!l->threads) return SH_ERR_NOMEM;
        l->stop = false;
        int made = 0;
        for (int i = 0; i < l->n_threads; i++) if (pthread_create(&l->threads[i], NULL, refill_main, l) == 0) made++; else break;
        l->n_threads = made;                       /* join exactly what was created */
        if (!made) { free(l->threads); l->threads = NULL; l->threads_running = false; return SH_OK; }
        l->threads_running = true;

        /* Warm the pool before the first exchange. Returning with every ring
         * empty made the first token generate 49 pads on the request path
         * (19.6 ms of refill-on-path in a 64-token decode, all in token one).
         * Bounded: a pathological box still starts, it just pays on the path. */
        struct timespec dl; clock_gettime(CLOCK_MONOTONIC, &dl);
        dl.tv_sec += l->warm_ms / 1000; dl.tv_nsec += (long)(l->warm_ms % 1000) * 1000000L;
        if (dl.tv_nsec >= 1000000000L) { dl.tv_sec++; dl.tv_nsec -= 1000000000L; }
        pthread_mutex_lock(&l->pool_mu);
        pthread_cond_broadcast(&l->need_refill);
        for (;;) {
            bool warm = true;
            for (size_t i = 0; i < l->n_groups && warm; i++) warm = l->groups[i].count > 0;
            if (warm || l->stop) break;
            if (pthread_cond_timedwait(&l->pool_filled, &l->pool_mu, &dl) == ETIMEDOUT) break;
        }
        pthread_mutex_unlock(&l->pool_mu);
    }
    return SH_OK;
}

/* Take up to m ready pads from the front of group g's ring, by slot index.
 * They are consumed here and held until release_pads. */
static int take_pads(sh_link *l, sh_group *g, int m, int *slots) {
    pthread_mutex_lock(&l->pool_mu);
    const int before = g->count < m ? g->count : m;
    /* An in-flight full refill can finish sooner than a separate, partial
     * request-thread refill. Only wait when the missing pads are ALREADY
     * reserved, and release the mutex while their producers write/publish.
     * A timeout consumes only the ready prefix; unpublished and held slots
     * remain owned exactly as before. Default zero preserves the old path. */
    if (l->pad_wait_us > 0 && !l->stop && g->count < m &&
        g->count + g->generating >= m) {
        const double started = now_ms();
        struct timespec dl; clock_gettime(CLOCK_MONOTONIC, &dl);
        dl.tv_nsec += (long)l->pad_wait_us * 1000L;
        if (dl.tv_nsec >= 1000000000L) { dl.tv_sec++; dl.tv_nsec -= 1000000000L; }
        while (!l->stop && g->count < m && g->generating > 0) {
            if (pthread_cond_timedwait(&l->pool_filled, &l->pool_mu, &dl) != 0) break;
        }
        l->pad_wait_ms += now_ms() - started;
    }
    int take = g->count < m ? g->count : m;
    l->pads_waited += (uint64_t)(take - before);
    for (int i = 0; i < take; i++) slots[i] = (g->head + i) % g->depth;
    g->head = (g->head + take) % g->depth;
    g->count -= take;
    g->held += take;
    pthread_mutex_unlock(&l->pool_mu);
    return take;
}

/* The slots may be overwritten from here on; the deficit they open is what
 * wakes a refill thread. */
static void release_pads(sh_link *l, sh_group *g, int n) {
    if (n <= 0) return;
    pthread_mutex_lock(&l->pool_mu);
    g->held -= n;
    pthread_cond_signal(&l->need_refill);
    pthread_mutex_unlock(&l->pool_mu);
}

/* --- start: connect, upload public weights, install the vetted graph ------ */
static int json_append(char **buf, size_t *len, size_t *cap, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    for (;;) {
        size_t avail = *cap - *len;
        va_list cp; va_copy(cp, ap);
        int n = vsnprintf(*buf + *len, avail, fmt, cp);
        va_end(cp);
        if (n < 0) { va_end(ap); return SH_ERR_NOMEM; }
        if ((size_t)n < avail) { *len += (size_t)n; va_end(ap); return SH_OK; }
        size_t ncap = (*cap ? *cap : 1024) * 2;
        while (ncap < *len + (size_t)n + 1) ncap *= 2;
        char *nb = (char *)realloc(*buf, ncap);
        if (!nb) { va_end(ap); return SH_ERR_NOMEM; }
        *buf = nb; *cap = ncap;
    }
}

/* One integer field of the HELLO reply, for a LOG LINE only (-1 when absent);
 * the two workers space their JSON differently, so walk past the colon. */
static double hello_num(const char *hello, const char *key) {
    const char *p = strstr(hello, key);
    if (!p) return -1.0;
    p += strlen(key);
    while (*p == ' ' || *p == ':') p++;
    return strtod(p, NULL);
}

int sh_link_start(sh_link *l) {
    int err = SH_OK;
    if (l->pipe) { sh_pipe_close(l->pipe); l->pipe = NULL; }
    /* vsock first when the guest was told the worker listens on one, TCP as the
     * fallback: a guest without the vsock driver, or a host without the
     * device, still reaches the card over slirp -- more slowly, not not at all. */
    int vport = l->vsock_port;
    /* Unset means "if this guest has a vsock device, the worker listens on the
     * same port number there" -- which is what the launcher does -- so a guest
     * with the driver gets the fast path with no configuration at all. 0
     * disables it explicitly. A port nothing answers on fails the connect at
     * once and the link takes TCP. */
    if (vport < 0) vport = access("/dev/vsock", F_OK) == 0 ? l->port : 0;
    if (vport > 0) {
        l->pipe = sh_pipe_open("vsock", vport, &err);
        if (l->pipe) snprintf(l->transport, sizeof l->transport, "vsock:%d", vport);
        else snprintf(l->transport, sizeof l->transport, "tcp %s:%d (vsock port %d unreachable)", l->host, l->port, vport);
    } else {
        snprintf(l->transport, sizeof l->transport, "tcp %s:%d", l->host, l->port);
    }
    if (!l->pipe) l->pipe = sh_pipe_open(l->host, l->port, &err);
    if (!l->pipe) { snprintf(l->err, sizeof l->err, "connect %s:%d failed", l->host, l->port); return err; }

    uint8_t pay[256]; sh_reply rep;
    /* The tenant's slice of the card, reserved at HELLO (protocol 1.3). The
     * manager sets SHIELDED_RESERVE_BYTES from the share the deployment bought;
     * the worker holds that much device memory for this connection until it
     * closes, and refuses the HELLO when the card cannot give it -- which this
     * link treats like any other failed start: the backend computes in the
     * enclave and reconnects with backoff, so the tenant runs until the memory
     * is there. Unset or 0 sends the 4-byte HELLO every worker accepts and
     * reserves nothing. Nothing in the reply is trusted beyond the version:
     * the reservation is the untrusted host's own bookkeeping, and the
     * tenant's privacy does not depend on it. */
    const uint64_t reserve = l->reserve_bytes;
    size_t n = sh_pack_hello(pay, 1, reserve);
    int rc = sh_pipe_call(l->pipe, SH_CMD_HELLO, pay, n, &rep);
    if (rc != SH_OK) { snprintf(l->err, sizeof l->err, "HELLO%s: %s", reserve ? " (with reservation)" : "", sh_pipe_last_error(l->pipe)); return rc; }
    /* The one-frame exchange is protocol 1.1; an older worker would refuse it
     * as an unknown command, so say what is wrong here rather than there. */
    int hello_minor = -1;
    {
        /* HELLO is JSON from either worker; only the version array matters here,
         * and the two serialisers space it differently. */
        int major = -1, minor = -1;
        char hello[1024] = { 0 };   /* the 1.3 reply carries a few more integers; only the version is acted on */
        if (rep.data) snprintf(hello, sizeof hello, "%.*s", (int)(rep.len < sizeof hello - 1 ? rep.len : sizeof hello - 1), (const char *)rep.data);
        const char *p = strstr(hello, "\"version\"");
        if (p) {
            p += 9;
            while (*p && (*p < '0' || *p > '9')) p++;
            major = atoi(p);
            while (*p >= '0' && *p <= '9') p++;
            while (*p && (*p < '0' || *p > '9')) p++;
            minor = atoi(p);
        }
        if (major != 1 || minor < 1) {
            sh_reply_free(&rep);
            snprintf(l->err, sizeof l->err, "worker speaks protocol %d.%d; this link needs 1.1 (FIELD_GEMM)", major, minor);
            return SH_ERR_PROTO;
        }
        /* 1.2 adds FIELD_GEMM24: the same request, the products as 3-byte
         * values. Every product is balanced below 2^23.8, so nothing is lost
         * and 25% of every reply's bytes are -- 152 KB of the 0.5B's lm_head
         * exchange, the largest of its 49. A 1.1 worker never sees command
         * 13; SHIELDED_REPLY32=1 keeps the int32 form against a 1.2 worker
         * (the A/B, and the escape hatch). What crosses is unchanged either
         * way: the same masked products, narrower. */
        const char *w32 = getenv("SHIELDED_REPLY32");
        l->ywidth = (minor >= 2 && !(w32 && *w32 && strcmp(w32, "0"))) ? 3 : 4;
        {
            size_t tl = strlen(l->transport);
            snprintf(l->transport + tl, sizeof l->transport - tl, " proto 1.%d reply int%d", minor, l->ywidth * 8);
        }
        hello_minor = minor;
        /* What the worker says it holds: this connection's reservation and the
         * sum over every live tenant, under SHIELDED_VERBOSE. Logged, never
         * acted on (see the reserve comment above). A 1.2 worker sends neither. */
        const char *vb = getenv("SHIELDED_VERBOSE");
        if (vb && *vb && strcmp(vb, "0")) {
            const double MB = 1024.0 * 1024.0;
            fprintf(stderr, "[shielded] link: proto 1.%d, asked to reserve %.0f MiB; worker holds %.0f MiB for this link, "
                            "%.0f MiB for all tenants, driver free %.0f MiB of a %.0f MiB budget\n",
                    minor, (double)reserve / MB,
                    hello_num(hello, "\"vram_reserve\"") / MB, hello_num(hello, "\"vram_reserved\"") / MB,
                    hello_num(hello, "\"vram_free\"") / MB, hello_num(hello, "\"vram_budget\"") / MB);
        }
    }
    sh_reply_free(&rep);

    const struct { int64_t size; const char *role; } bufs[2] = {
        { l->wbytes, "weights" }, { l->abytes, "activations" } };
    for (int i = 0; i < 2; i++) {
        n = sh_pack_alloc(pay, (uint64_t)bufs[i].size, bufs[i].role);
        rc = sh_pipe_call(l->pipe, SH_CMD_ALLOC_BUFFER, pay, n, &rep);
        if (rc != SH_OK) { snprintf(l->err, sizeof l->err, "ALLOC(%s): %s",
                                    bufs[i].role, sh_pipe_last_error(l->pipe)); return rc; }
        sh_reply_free(&rep);
    }

    /* Weights are PUBLIC: they cross in the clear, by design. */
    for (size_t i = 0; i < l->n_nodes; i++) {
        sh_node *nd = &l->nodes[i];
        const size_t bytes = (size_t)(nd->K * nd->N);
        const size_t CHUNK = 32u << 20;
        for (size_t off = 0; off < bytes; off += CHUNK) {
            size_t part = bytes - off < CHUNK ? bytes - off : CHUNK;
            uint8_t hdr[24];
            sh_pack_set_tensor_header(hdr, 1, (uint64_t)(nd->w_off + (int64_t)off), part);
            sh_frame f = { SH_CMD_SET_TENSOR, hdr, 24, (const uint8_t *)nd->w + off, part };
            rc = sh_pipe_exchange(l->pipe, &f, 1, &rep);
            if (rc != SH_OK) { snprintf(l->err, sizeof l->err, "upload %s: %s", nd->name, sh_pipe_last_error(l->pipe)); return rc; }
            sh_reply_free(&rep);
        }
    }

    char *js = NULL; size_t jl = 0, jc = 0;
    json_append(&js, &jl, &jc, "{\"nodes\":[");
    for (size_t i = 0; i < l->n_nodes; i++) {
        sh_node *nd = &l->nodes[i];
        json_append(&js, &jl, &jc,
            "%s{\"op\":\"FIELD_GEMM\",\"id\":\"%s\",\"w\":{\"bid\":1,\"offset\":%lld},"
            "\"x\":{\"bid\":2,\"offset\":%lld},\"y\":{\"bid\":2,\"offset\":%lld},"
            "\"K\":%lld,\"N\":%lld,\"max_m\":%d}",
            i ? "," : "", nd->name, (long long)nd->w_off,
            (long long)nd->x_off, (long long)nd->y_off,
            (long long)nd->K, (long long)nd->N, nd->max_m);
    }
    /* The exchange this link uses returns its products in-band, so declared
     * outputs exist only to satisfy install's rule that something be readable. */
    json_append(&js, &jl, &jc, "],\"outputs\":[");
    for (size_t i = 0; i < l->n_nodes; i++)
        json_append(&js, &jl, &jc, "%s{\"bid\":2,\"offset\":%lld,\"nbytes\":%lld}",
                    i ? "," : "", (long long)l->nodes[i].y_off, (long long)(l->nodes[i].N * 4));
    json_append(&js, &jl, &jc, "]}");
    rc = sh_pipe_call(l->pipe, SH_CMD_GRAPH_INSTALL, js, jl, &rep);
    free(js);
    if (rc != SH_OK) { snprintf(l->err, sizeof l->err, "GRAPH_INSTALL: %s", sh_pipe_last_error(l->pipe)); return rc; }
    sh_reply_free(&rep);

    /* The shared-memory ring (shielded-wire.h): opened only when per-link
     * config or the legacy environment names one AND the worker speaks 1.2.
     * Anything short of a
     * granted ring leaves the link on the socket exactly as before; the
     * transport string says which, so the profile line can be read. */
    {
        const char *shm = l->shm_configured ? l->shm_path : getenv("SHIELDED_SHM");
        if (shm && *shm && !env_int("SHIELDED_SHM_DISABLE", 0, 0, 1)) {
            if (hello_minor < 2) {
                snprintf(l->transport + strlen(l->transport), sizeof l->transport - strlen(l->transport),
                         " (shm ring: worker speaks 1.%d, needs 1.2)", hello_minor);
            } else {
                int index = l->shm_configured ? -1 : env_int("SHIELDED_SHM_RING", 0, -1, 7);
                const char *nb = getenv("SHIELDED_SHM_BYTES");
                size_t bytes = l->shm_configured ? l->shm_bytes : (nb && *nb ? (size_t)strtoull(nb, NULL, 10) : 0);
                int arc = index < 0 ? sh_pipe_shm_attach_available(l->pipe, shm, bytes, &index)
                                    : sh_pipe_shm_attach(l->pipe, shm, index, bytes);
                if (arc == SH_OK)
                    snprintf(l->transport + strlen(l->transport), sizeof l->transport - strlen(l->transport),
                             " + shm ring %d (%s)%s", index, shm,
                             sh_pipe_ring_stream_load(l->pipe) ? " stream-load" : "");
                else if (arc == SH_ERR_IO)
                    snprintf(l->transport + strlen(l->transport), sizeof l->transport - strlen(l->transport), " (shm ring unavailable: %s)", sh_pipe_last_error(l->pipe));
                else {
                    /* The worker answered SHM_ATTACH with nonsense: the same
                     * peer serves the socket, so do not trust it either. */
                    snprintf(l->err, sizeof l->err, "SHM_ATTACH: %s", sh_pipe_last_error(l->pipe));
                    return arc;
                }
            }
        }
    }

    return start_pools(l);
}

/* --- Freivalds over an unrelated prime ------------------------------------ */
static bool fv_check(const sh_link *l, const sh_node *nd, const int64_t *x, const int64_t *y, int32_t m) {
    for (int32_t row = 0; row < m; row++) {
        int64_t lhs[SH_FV_REPS], rhs[SH_FV_REPS];
        l->simd->fv_dots(y + (int64_t)row * nd->N, nd->s32, SH_FV_REPS, nd->N, lhs);
        l->simd->fv_dots_x(x + (int64_t)row * nd->K, nd->st32, SH_FV_REPS, nd->K, rhs);
        for (int rep = 0; rep < SH_FV_REPS; rep++)
            if (lhs[rep] != rhs[rep]) return false;
    }
    return true;
}

int sh_link_gemm_local(sh_link *l, const int *nodes, size_t n_nodes,
                       const int64_t *x_field, int32_t m, int64_t **y_out) {
    for (size_t i = 0; i < n_nodes; i++) {
        const sh_node *nd = &l->nodes[nodes[i]];
        const int64_t K = nd->K, N = nd->N;
        int64_t *y = y_out[i];
        for (int32_t row = 0; row < m; row++) {
            const int64_t *xr = x_field + (int64_t)row * K;
            int64_t *yr = y + (int64_t)row * N;
            for (int64_t j = 0; j < N; j++) {
                const int8_t *w = nd->w + j * K;
                int64_t acc = 0;
                for (int64_t k = 0; k < K; k++) acc += xr[k] * w[k];
                yr[j] = sh_balanced(acc);
            }
        }
        l->macs += (uint64_t)m * (uint64_t)K * (uint64_t)N;
    }
    return SH_OK;
}

bool sh_link_is_live(const sh_link *l) { return l && l->pipe; }

const int8_t *sh_link_weight(const sh_link *l, int node) {
    if (!l || node < 0 || (size_t)node >= l->n_nodes) return NULL;
    return l->nodes[node].w;
}

bool sh_link_verify(const sh_link *l, int node, const int64_t *x, const int64_t *y, int32_t m) {
    if (!l || node < 0 || (size_t)node >= l->n_nodes) return false;
    return fv_check(l, &l->nodes[node], x, y, m);
}

static int ensure(void **p, size_t *cap, size_t want) {
    if (*cap >= want) return SH_OK;
    void *n = realloc(*p, want);
    if (!n) return SH_ERR_NOMEM;
    *p = n; *cap = want;
    return SH_OK;
}

int sh_link_gemm(sh_link *l, const int *nodes, size_t n_nodes,
                 const int64_t *x_field, int32_t m, int64_t **y_out) {
    if (!n_nodes) return SH_OK;
    if (n_nodes > SH_GROUP_MAX) { snprintf(l->err, sizeof l->err, "too many nodes in one exchange"); return SH_ERR_PROTO; }
    sh_group *g = &l->groups[l->nodes[nodes[0]].group];
    for (size_t i = 0; i < n_nodes; i++)
        if (l->nodes[nodes[i]].group != l->nodes[nodes[0]].group) {
            snprintf(l->err, sizeof l->err, "exchanged nodes do not share an activation"); return SH_ERR_PROTO;
        }
    if (m < 1 || m > g->max_m) {
        snprintf(l->err, sizeof l->err, "m=%d outside this group's [1,%d]", m, g->max_m); return SH_ERR_PROTO;
    }
    const int64_t K = g->K;
    int rc;
    /* Steady state allocates nothing: every buffer here has grown to its
     * largest use by the second token. */
    if ((rc = ensure((void **)&l->planes, &l->planes_cap, (size_t)3 * m * K)) != SH_OK) return rc;
    if ((rc = ensure((void **)&l->hdr,    &l->hdr_cap,    8 + 4 * n_nodes)) != SH_OK) return rc;
    if (l->p_cap < (size_t)m) {
        const int32_t **rp = (const int32_t **)realloc(l->rp, (size_t)m * sizeof *rp);
        if (rp) l->rp = rp;
        const int32_t **up = (const int32_t **)realloc(l->up, (size_t)m * sizeof *up);
        if (up) l->up = up;
        int *slots = (int *)realloc(l->slots, (size_t)m * sizeof *slots);
        if (slots) l->slots = slots;
        if (!rp || !up || !slots) return SH_ERR_NOMEM;
        l->p_cap = (size_t)m;
    }

    /* ONE pad per plaintext row -- rule 2 -- from the pool, or made here if the
     * pool is dry. Shared-x nodes see the same pad by construction: they are
     * in the same group and read the same masked planes. Pool pads are used IN
     * PLACE and held until the unmask below; the rows made here live in the
     * link's private scratch. Either way each row's pad is one pointer. */
    double t0 = now_ms();
    const int took = l->threads_running ? take_pads(l, g, m, l->slots) : 0;
    for (int i = 0; i < took; i++) {
        l->rp[i] = g->r_store + (size_t)l->slots[i] * K;
        l->up[i] = g->u_store + (size_t)l->slots[i] * g->u_len;
    }
    if (took < m) {
        const int miss = m - took;
        if ((rc = ensure((void **)&l->r,       &l->r_cap,       (size_t)miss * K * sizeof(int32_t))) != SH_OK) goto fail;
        if ((rc = ensure((void **)&l->u,       &l->u_cap,       (size_t)miss * g->u_len * sizeof(int32_t))) != SH_OK) goto fail;
        if ((rc = ensure((void **)&l->gplanes, &l->gplanes_cap, (size_t)3 * miss * K)) != SH_OK) goto fail;
        if ((rc = ensure((void **)&l->acc,     &l->acc_cap,     (size_t)12 * l->Nmax * sizeof(int32_t))) != SH_OK) goto fail;
        gen_scratch s = { l->gplanes, l->acc };
        double tg = now_ms();
        rc = generate(l, g, miss, l->r, l->u, &s);
        const double elapsed_ms = now_ms() - tg;
        sh_prof[2] += elapsed_ms;
        g->on_path_ms += elapsed_ms;
        if (rc != SH_OK) { snprintf(l->err, sizeof l->err, "pad bank exhausted; stall the request"); goto fail; }
        for (int i = 0; i < miss; i++) {
            l->rp[took + i] = l->r + (size_t)i * K;
            l->up[took + i] = l->u + (size_t)i * g->u_len;
        }
        l->pads_missed += (uint64_t)miss;
        g->pads_missed += (uint64_t)miss;
        sh_prof[6] += miss;
    }
    l->pads_used += (uint64_t)m; sh_prof[7] += m;
    g->pads_used += (uint64_t)m;

    for (int32_t row = 0; row < m; row++)
        l->simd->mask_planes(x_field + (size_t)row * K, l->rp[row], (size_t)K,
                             l->planes + (size_t)row * K, l->planes + ((size_t)m + row) * K, l->planes + ((size_t)2 * m + row) * K);
    const size_t hn = sh_pack_field_gemm(l->hdr, (uint32_t)n_nodes, (uint32_t)m, nodes);
    double t1 = now_ms(); sh_prof[0] += t1 - t0;

    {
        sh_frame f = { l->ywidth == 3 ? SH_CMD_FIELD_GEMM24 : SH_CMD_FIELD_GEMM, l->hdr, hn, l->planes, (size_t)3 * m * K };
        sh_reply rep;
        /* The reply length is OURS to know, from the request and the width
         * this exchange uses -- never anything the worker says. The ring
         * carries the int32 form only; the socket the negotiated one. A ring
         * miss (no reply in the spin budget, a length that is not `want`)
         * sends the SAME frame on the socket -- the same ciphertext under the
         * same pad, which tells the host nothing a retransmit would not. */
        const bool via_ring = sh_pipe_ring_live(l->pipe);
        const size_t yw = via_ring ? 4 : (size_t)l->ywidth;
        f.cmd = yw == 3 ? SH_CMD_FIELD_GEMM24 : SH_CMD_FIELD_GEMM;
        size_t want = 0;
        for (size_t i = 0; i < n_nodes; i++) want += (size_t)m * l->nodes[nodes[i]].N * yw;
        rc = via_ring ? sh_pipe_ring_exchange(l->pipe, &f, want, &rep) : SH_ERR_IO;
        if (rc == SH_ERR_IO) rc = sh_pipe_exchange(l->pipe, &f, 1, &rep);
        double t2 = now_ms(); sh_prof[1] += t2 - t1; l->last_wire_us = (t2 - t1) * 1000.0;
        if (rc != SH_OK) {
            snprintf(l->err, sizeof l->err, "exchange: %s", sh_pipe_last_error(l->pipe));
            /* SH_ERR_PROTO is reserved for THIS link's pre-flight refusals
             * above (a property of the node). Anything wrong that came back
             * over the wire -- an oversize frame, a bad status -- is the
             * worker misbehaving, and the caller must treat it like a
             * refusal: take the link down and reconnect, not keep shipping
             * pads to a peer that answers nonsense. */
            if (rc == SH_ERR_PROTO) rc = SH_ERR_VIOLATION;
            goto fail;
        }
        l->exchanges++;

        /* `want` was computed before the exchange from the request and the
         * width; a mismatch is a lying peer, not a shorter read. */
        if (rep.len != want) {
            snprintf(l->err, sizeof l->err, "worker returned %zu bytes, expected %zu", rep.len, want);
            rc = SH_ERR_VIOLATION; goto fail;           /* the worker's fault: reconnect, see above */
        }

        /* Unmask, then verify, then hand back -- never the other way round.
         * With verification on, the unmask and the lhs dot are one pass. */
        size_t off = 0;
        for (size_t i = 0; i < n_nodes; i++) {
            const sh_node *nd = &l->nodes[nodes[i]];
            const uint8_t *ym = rep.data + off;         /* int32 or packed int24 rows, yw bytes each */
            off += (size_t)m * nd->N * yw;
            int64_t *y = y_out[i];
            bool ok = true;
            if (l->verify) {
                /* Profile split: [3] is the fused unmask + lhs pass over y,
                 * [4] the rhs pass over x. */
                for (int32_t row = 0; row < m; row++) {
                    int64_t lhs[SH_FV_REPS], rhs[SH_FV_REPS];
                    double t3 = now_ms();
                    if (yw == 3)
                        l->simd->unmask24_fv(ym + (size_t)row * nd->N * 3, l->up[row] + nd->u_off, nd->s32, SH_FV_REPS, nd->N,
                                             y + (size_t)row * nd->N, lhs);
                    else
                        l->simd->unmask_fv((const int32_t *)ym + (size_t)row * nd->N, l->up[row] + nd->u_off, nd->s32, SH_FV_REPS, nd->N,
                                           y + (size_t)row * nd->N, lhs);
                    double t4 = now_ms(); sh_prof[3] += t4 - t3;
                    l->simd->fv_dots_x(x_field + (size_t)row * K, nd->st32, SH_FV_REPS, K, rhs);
                    sh_prof[4] += now_ms() - t4;
                    for (int rep_i = 0; rep_i < SH_FV_REPS; rep_i++) ok = ok && lhs[rep_i] == rhs[rep_i];
                }
            } else {
                double t3 = now_ms();
                for (int32_t row = 0; row < m; row++) {
                    if (yw == 3)
                        l->simd->unmask24(ym + (size_t)row * nd->N * 3, l->up[row] + nd->u_off, (size_t)nd->N, y + (size_t)row * nd->N);
                    else
                        l->simd->unmask((const int32_t *)ym + (size_t)row * nd->N, l->up[row] + nd->u_off, (size_t)nd->N, y + (size_t)row * nd->N);
                }
                sh_prof[3] += now_ms() - t3;
            }
            l->macs += (uint64_t)m * (uint64_t)nd->K * (uint64_t)nd->N;
            if (!ok) {
                l->verify_fail++;
                snprintf(l->err, sizeof l->err,
                         "%s: verification FAILED -- the worker lied or the field wrapped. "
                         "Abort the request; do not sample, stream, or cache this.", nd->name);
                rc = SH_ERR_VERIFY; goto fail;
            }
        }
        sh_prof[5] += 1;
    }
    release_pads(l, g, took);
    return SH_OK;
fail:
    /* Whatever failed, the pads were consumed the moment they were taken:
     * releasing the slots lets a refill overwrite them, never re-issues them. */
    release_pads(l, g, took);
    return rc;
}
