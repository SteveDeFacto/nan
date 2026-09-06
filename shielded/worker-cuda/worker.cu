/*
 * worker.cu -- the shielded GPU worker, in C++/CUDA. Runs on the UNTRUSTED
 * host, holds the card.
 *
 * This is the production form of shielded/worker.py: the same wire protocol,
 * the same admission rules (protocol.py is the reference and the tests pin it),
 * the same trust posture. Read the header of worker.py before touching the
 * rules here; this file adds nothing to them and removes nothing from them.
 *
 * WHY IT EXISTS
 * -------------
 * Decode is a chain of ~2 masked exchanges per layer, and the chain is strictly
 * sequential: layer 2's input is a nonlinear function of layer 1's output, so
 * nothing can be pipelined across it. The per-token cost is therefore
 *
 *     (exchanges per token) x (cost of one exchange)
 *
 * and the Python worker's cost per exchange -- five framed commands, each going
 * through the interpreter, torch's dispatcher and a Triton launch -- measured
 * 0.35 ms on loopback. At ~50 exchanges per token that is 17 ms before the GPU
 * has done any work, against a whole-token budget of 14 ms. This worker does one
 * exchange in one frame read straight into pinned memory, one captured graph
 * (the upload and one fused kernel for every node of the exchange, writing
 * the products into mapped host memory), and one send from where they landed,
 * with no interpreter anywhere on the path. Measured on the 0.5B gate|up
 * exchange over TCP loopback: 67 us round trip with a memcpy, an H2D copy, a
 * kernel per node, a D2H copy and a stream sync; 42.5 us this way, of which
 * 24 is the kernel itself.
 *
 * WHAT IT COMPUTES
 * ----------------
 * FIELD_GEMM: y = (x+r) . W over Z_M for a one-time pad r it never receives.
 * The activation arrives as three int8 residue planes (one per byte prime), the
 * weight is the fixed-point int8 w_fixed the TEE also holds (|w| <= 119, so
 * w mod q == w for every prime and one weight serves all three planes), and the
 * kernel accumulates each plane with dp4a and recombines the three residues by
 * Garner CRT in the epilogue. One int32 per output. The TEE subtracts u = r.W.
 *
 * The weight layout is GGUF's own: one row per OUTPUT, K contiguous bytes. A
 * decode GEMV is then N dot products over contiguous rows, which is the shape a
 * GPU reads at full bandwidth, and the TEE never transposes anything.
 *
 * TWO WAYS TO SUPPLY A WEIGHT
 * ---------------------------
 * A node may carry "w" (the encoded int8 matrix, (N,K)) or the legacy pair
 * "wq"/"wd" (q8_0 quants (K,N) and fp16 scales (K/32,N)), in which case this
 * worker runs THE shared encoding (shielded-field.c, the same object the TEE
 * links) at install time. The legacy form is what metal/guest/shielded-probe.mjs
 * sends at boot; the engine backend sends "w" because it already has it.
 *
 * FAIL CLOSED. Every malformed frame, unknown command, unlisted op, out-of-range
 * region or undeclared read terminates the connection with a named reason. A
 * dead CUDA context terminates the PROCESS (exit 70) so the launcher can build a
 * fresh one -- see the note in worker.py about the MPS server going away.
 */
#include <cuda_runtime.h>
#include <immintrin.h>

#include <arpa/inet.h>
#include <errno.h>
#include <linux/vm_sockets.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

extern "C" {
#include "shielded-field.h"
}

/* ---------------------------------------------------------------------------
 * Protocol constants. Mirror protocol.py exactly.
 * ------------------------------------------------------------------------ */
enum : uint8_t {
    CMD_HELLO = 0, CMD_ALLOC_BUFFER = 1, CMD_FREE_BUFFER = 2, CMD_BUFFER_GET_BASE = 3,
    CMD_GET_ALIGNMENT = 4, CMD_GET_MAX_SIZE = 5, CMD_GET_DEVICE_MEMORY = 6, CMD_DEVICE_COUNT = 7,
    CMD_SET_TENSOR = 8, CMD_GET_TENSOR = 9, CMD_GRAPH_INSTALL = 10, CMD_GRAPH_RECOMPUTE = 11,
    CMD_FIELD_GEMM = 12,
    CMD_FIELD_GEMM24 = 13,   /* 1.2: the same request, 3-byte reply values */
    CMD_SHM_ATTACH = 14,     /* 1.2: bind a shared-memory ring to this connection */
    CMD_COUNT = 15,
};
/* 1.2: SHM_ATTACH -- the shared-memory ring. A 1.1 link never sends it.
 * 1.3: HELLO carries an optional u64 reservation and the reply names the
 *      card's reservations. A 4-byte HELLO is a 1.2 link: reserves nothing. */
static const int PROTO_MAJOR = 1, PROTO_MINOR = 3, PROTO_PATCH = 0;
static const uint8_t STATUS_OK = 0, STATUS_VIOLATION = 1;
static const size_t MAX_FRAME = (size_t)256 << 20;
static const size_t SH_HDR = 9;

static const char *OP_ALLOWLIST[] = { "FIELD_GEMM", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "CONT", "CPY" };
static const std::pair<const char *, const char *> OP_DENYLIST[] = {
    { "SOFT_MAX", "nonlinear on secret data; TEE-only" },
    { "RMS_NORM", "nonlinear on secret data; TEE-only" },
    { "NORM", "nonlinear on secret data; TEE-only" },
    { "SILU", "nonlinear on secret data; TEE-only" },
    { "GELU", "nonlinear on secret data; TEE-only" },
    { "ROPE", "consumes token positions; TEE-only" },
    { "FLASH_ATTN_EXT", "activation-activation product; TwinShield m=1 is broken" },
    { "MUL_MAT", "plain matmul would run on UNMASKED data; use FIELD_GEMM" },
    { "ARGSORT", "sampling-adjacent; TEE-only" },
    { "GET_ROWS", "embedding gather keyed by a secret token id; TEE-only" },
};

struct Violation {
    std::string why;
    explicit Violation(std::string w) : why(std::move(w)) {}
};
static std::string fmt(const char *f, ...) {
    char buf[1024]; va_list ap; va_start(ap, f); vsnprintf(buf, sizeof buf, f, ap); va_end(ap);
    return buf;
}
#define VIOLATE(...) throw Violation(fmt(__VA_ARGS__))

/* Device memory comes from the reserved budget pool (defined with it below). */
static cudaError_t dmalloc(void **p, size_t n);
static void dfree(void *p);

static bool g_quiet = false;
#ifdef SH_XPROF
static double g_xp[16]; static uint64_t g_xpn; static std::chrono::steady_clock::time_point g_xpt;
#define XP(i) do { auto _t = std::chrono::steady_clock::now(); if (i) g_xp[i] += std::chrono::duration<double, std::micro>(_t - g_xpt).count(); g_xpt = _t; } while (0)
#define XPN() (g_xpn++)
#else
#define XP(i) do {} while (0)
#define XPN() do {} while (0)
#endif
static void logf(const char *f, ...) {
    if (g_quiet) return;
    char buf[1024]; va_list ap; va_start(ap, f); vsnprintf(buf, sizeof buf, f, ap); va_end(ap);
    fprintf(stdout, "[shielded-worker] %s\n", buf); fflush(stdout);
}

/* ---------------------------------------------------------------------------
 * The card's RATED FP16 tensor throughput, derived rather than declared.
 *
 * This is the SIZING unit -- the number an app's declared TFLOPS requirement is
 * divided by to decide what share of the card it must buy -- and it therefore
 * has to be on the same basis as every other box in the fleet, which quotes the
 * vendor's dense FP16 tensor figure. It is NOT a claim about what the masked
 * path delivers; that is `field_gmac_per_s`, measured, reported separately, and
 * roughly 25x smaller. Conflating the two is what broke placement on metal0: at
 * a masked 1.6 TFLOPS, an app declaring 5 TFLOPS computed a 312% share and
 * could not be placed on the box at all.
 *
 * Rated dense FP16 = SMs x (FLOP per SM per clock, an architecture constant) x
 * clock. Checked against the published figures: RTX 3070 40.6, 2080 Ti 53.8,
 * A100 312, RTX 4090 165.2, H100 SXM 989.4 -- the last being exactly the
 * fleet's own GPU_TFLOPS default, so a shielded box lands on the same scale as
 * a passed-through one by construction.
 *
 * Two honest caveats, both recorded rather than smoothed over. cudaDeviceProp
 * reports the MAX boost clock while vendor spec sheets quote the REFERENCE
 * boost, so this reads a little above the sticker (a 3070 measures 2100 MHz
 * against a rated 1725, i.e. 49 rather than 40.6). And an architecture missing
 * from the table returns 0, which the supervisor treats as "no rated figure"
 * and falls back rather than inventing one.
 * ------------------------------------------------------------------------ */
static int fp16_flops_per_sm_clock(int major, int minor) {
    switch (major) {
        case 7:  return minor == 0 ? 1024 : 512;      /* Volta / Turing */
        case 8:  return minor == 0 ? 2048 : 512;      /* A100 / Ampere consumer + Ada */
        case 9:  return 4096;                          /* Hopper */
        case 10: case 12: return 4096;                 /* Blackwell; refine when one is measured */
        default: return 0;                             /* unknown: say so, do not guess */
    }
}
static double rated_fp16_tflops(const cudaDeviceProp &p) {
    const int per = fp16_flops_per_sm_clock(p.major, p.minor);
    if (!per) return 0.0;
    return (double)p.multiProcessorCount * per * ((double)p.clockRate * 1e3) / 1e12;
}

/* ---------------------------------------------------------------------------
 * CUDA errors. A bad request is refused; a broken context kills the process.
 * ------------------------------------------------------------------------ */
static bool fatal_cuda(cudaError_t e) {
    switch (e) {
        case cudaErrorIllegalAddress: case cudaErrorLaunchFailure: case cudaErrorAssert:
        case cudaErrorHardwareStackError: case cudaErrorIllegalInstruction:
        case cudaErrorMisalignedAddress: case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc: case cudaErrorECCUncorrectable: case cudaErrorMpsRpcFailure:
        case cudaErrorMpsServerNotReady: case cudaErrorMpsConnectionFailed:
        case cudaErrorUnknown: case cudaErrorDeviceUninitialized:
            return true;
        default: return false;
    }
}
static void ck(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return;
    const char *msg = cudaGetErrorString(e);
    if (fatal_cuda(e)) {
        fprintf(stderr, "[shielded-worker] FATAL: %s: %s. The CUDA context is gone; exiting so "
                        "the launcher can restart with a fresh one.\n", what, msg);
        fflush(stderr);
        _exit(70);
    }
    VIOLATE("internal: %s: %s", what, msg);
}

/* ---------------------------------------------------------------------------
 * The kernel.
 *
 * One launch covers up to 8 nodes of the exchange (gate|up is 2, q|k|v is 3):
 * the node table travels in the launch parameters and each block finds its
 * node by blockIdx. Fusing matters because the card's gap between two
 * dependent launches is ~3-4 us, which at two launches per exchange and 49
 * exchanges per token was 0.3 ms of a 6 ms token spent on nothing. The table
 * is in PARAMETER space on purpose: an earlier form read it from device memory
 * and the dependent load sat in front of every weight load, which on the
 * K=896 decode shapes (two loop iterations per warp) cost 40% of the kernel.
 *
 * A block is 8 warps and owns RB = WR x G consecutive output rows: G row
 * groups of WR rows each, and the KS = 8/G warps of a group split K between
 * them in interleaved 512-byte chunks. A warp therefore streams WR weight rows
 * AT ONCE and applies every activation value it loads to all WR rows. That is
 * the m >= 2 fix: the old kernel had one warp per output row, so every 4 bytes
 * of weight cost 3m x 4 bytes of shared-memory reads of X -- at m = 8 that is
 * 24 bytes of shared memory per weight byte, 10.7 TB/s at the card's 448 GB/s,
 * i.e. the card's ENTIRE shared-memory bandwidth, so m = 8 ran at 91 GB/s.
 * Register-blocking WR rows divides that traffic by WR.
 *
 * The activation is read through L1 (__ldg) rather than staged in shared
 * memory: it is small (3mK bytes), every block on an SM shares the same L1
 * lines, and not staging it removes the per-block copy that was the OTHER
 * m = 8 cost (N/8 blocks x 3mK bytes) as well as the shared-memory cap on K.
 * Any K that is a multiple of 16 works: lanes past the last 16-byte chunk
 * simply drop out of the loop.
 *
 * G is chosen per launch so there are enough blocks to cover the SMs while
 * each block's output write is as wide as possible: the product goes STRAIGHT
 * INTO MAPPED HOST MEMORY (no D2H copy, no copy-engine latency), as one
 * contiguous RB x 4-byte write per activation row per block. Measured, the
 * mapped destination costs the same as device memory on every shape here
 * (lm_head, 608 KB of output: 331 vs 329 us), which is what makes the copy
 * free to drop.
 *
 * Accumulator range: |x_p| <= 125, |w| <= 119, so K terms reach K * 14875 --
 * under 2^31 for any K below 144k. No chunking, no saturation.
 * ------------------------------------------------------------------------ */
#define KQ0 251
#define KQ1 241
#define KQ2 239
#define KM  ((long long)KQ0 * KQ1 * KQ2)
#define KINV01  217   /* inv(251 mod 241) mod 241 -- checked against the host at startup */
#define KINV012 10    /* inv(251*241 mod 239) mod 239 */

/* Garner, entirely in int32: r0 + 251*t1 < 60491 and the final value is
 * below M = 14457349, so nothing here needs 64 bits (a 64-bit remainder is a
 * ~100-instruction library call on the card, and this runs once per output). */
__device__ __forceinline__ int32_t crt3(int32_t a0, int32_t a1, int32_t a2) {
    int r0 = a0 % KQ0; if (r0 < 0) r0 += KQ0;
    int r1 = a1 % KQ1; if (r1 < 0) r1 += KQ1;
    int r2 = a2 % KQ2; if (r2 < 0) r2 += KQ2;
    int t1 = (r1 - r0) % KQ1; if (t1 < 0) t1 += KQ1;
    t1 = (t1 * KINV01) % KQ1;
    int x = r0 + KQ0 * t1;
    int t2 = (r2 - x) % KQ2; if (t2 < 0) t2 += KQ2;
    t2 = (t2 * KINV012) % KQ2;
    x += KQ0 * KQ1 * t2;
    if (x > (int)(KM / 2)) x -= (int)KM;
    return x;
}

/* The nodes of one launch, by value in parameter space. blk0[i] is the first
 * block belonging to node i. Indexed only with compile-time constants inside
 * the kernel (see the unrolled selects) so nothing is copied to local memory. */
static const int GEMM_TAB_NODES = 8;
struct GemmTab {
    const int8_t *W[GEMM_TAB_NODES];   /* (N,K) int8 */
    uint8_t *Y[GEMM_TAB_NODES];        /* [m][N] destination: int32, or 3-byte values when pack */
    int N[GEMM_TAB_NODES];
    int blk0[GEMM_TAB_NODES];
    int n;
    int pack;                          /* FIELD_GEMM24: the epilogue writes int24 (see pack24 below) */
};

/* Row-blocking per activation-row count: WR = 4 weight rows per warp at once
 * for every m (swept 1/2/4/8 per class on the 0.5B and 4B shapes; 4 won or
 * tied everywhere), 3 x MR x 4 accumulators. The launch bound asks for two
 * resident blocks per SM (<= 128 registers) up to m = 4; at m = 5..8 the 60-96
 * accumulators need the whole register file and a forced cap only spills
 * (m = 8 gate|up: 40 us at one block per SM, 44 with two). On Volta, m = 3..4
 * also benefits from the extra registers: the 27B m = 4 gate|up and down
 * kernels took 40-54% less time in both V100 microbenchmarks. */
static const int GEMM_WR = 4;
template <int MR> struct RowsFor {
    static const int WR = GEMM_WR;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    static const int MINB = MR <= 2 ? 2 : 1;
#else
    static const int MINB = MR <= 4 ? 2 : 1;
#endif
};
static inline int gemm_rows_per_block(int mr, int g) { (void)mr; return GEMM_WR * g; }

/* Reduce NV (a power of two, <= 32) per-lane partial sums across the warp
 * at once. Each stage swaps half of the live values with the partner lane and
 * halves the live count, so 32 values cost 16+8+4+2+1 = 31 shuffles instead of
 * 32 x 5 = 160. Afterwards lane l holds the warp total of value l / (32/NV).
 * This is not a nicety: at m = 8 a warp carries 96 partial sums, and the
 * straightforward per-value butterfly (480 shuffles per warp, at one warp-
 * shuffle per clock per SM) cost MORE than the whole K = 896 dot-product loop
 * -- 67 us vs 33 us for the loop alone on the 0.5B gate|up x2 m = 8 shape. */
template <int OFF, int C, int O, int NVP>
__device__ __forceinline__ void warp_reduce_stage(int (&v)[NVP], int lane) {
    /* Every index below is a compile-time constant, which is what keeps v in
     * registers: a runtime-indexed local array lands in local memory. */
    if constexpr (O >= 1) {
        if constexpr (C > 1) {
            constexpr int h = C / 2;
            const bool up = (lane & O) != 0;
#pragma unroll
            for (int i = 0; i < h; i++) {
                const int send = up ? v[OFF + i] : v[OFF + i + h];
                const int keep = up ? v[OFF + i + h] : v[OFF + i];
                v[OFF + i] = keep + __shfl_xor_sync(0xffffffffu, send, O);
            }
            warp_reduce_stage<OFF, h, O / 2>(v, lane);
        } else {
            v[OFF] += __shfl_xor_sync(0xffffffffu, v[OFF], O);
            warp_reduce_stage<OFF, 1, O / 2>(v, lane);
        }
    }
}
template <int CH, int OFF, int NVP>
__device__ __forceinline__ void warp_reduce_chunk(int (&v)[NVP], int lane) {
    warp_reduce_stage<OFF, CH, 16>(v, lane);
}
template <int N> struct Pow2Ceil { static const int v = N <= 1 ? 1 : 2 * Pow2Ceil<(N + 1) / 2>::v; };
template <> struct Pow2Ceil<1> { static const int v = 1; };

template <int MR, int WR, int G>
__global__ void __launch_bounds__(256, RowsFor<MR>::MINB)
field_gemm_kernel(const GemmTab tab, int K,
                  const int8_t *__restrict__ X, long long xstride, long long pstride) {
    constexpr int KS = 8 / G, RB = WR * G;
    __shared__ int red[8][3 * MR * WR];

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rg = warp / KS, ks = warp - rg * KS;

    /* Which node is this block's? Selects over parameter-space constants. */
    int ni = 0;
#pragma unroll
    for (int i = 1; i < GEMM_TAB_NODES; i++) if (i < tab.n && (int)blockIdx.x >= tab.blk0[i]) ni = i;
    const int8_t *W = tab.W[0]; uint8_t *Y = tab.Y[0]; int N = tab.N[0]; int blk0 = 0;
#pragma unroll
    for (int i = 1; i < GEMM_TAB_NODES; i++)
        if (ni == i) { W = tab.W[i]; Y = tab.Y[i]; N = tab.N[i]; blk0 = tab.blk0[i]; }

    const int jbase = (blockIdx.x - blk0) * RB;
    const int j0 = jbase + rg * WR;                       /* this warp's first row */
    const int K16 = K >> 4;

    const int4 *wrow[WR];
#pragma unroll
    for (int w = 0; w < WR; w++) {
        const int j = min(j0 + w, N - 1);                 /* clamp: a row past N is computed and dropped */
        wrow[w] = reinterpret_cast<const int4 *>(W + (size_t)j * K);
    }
    const int4 *xp = reinterpret_cast<const int4 *>(X);
    const int xs16 = (int)(xstride >> 4), ps16 = (int)(pstride >> 4);

    /* Partial sums, flat and padded so they reduce in chunks of <= 32. */
    constexpr int NV = 3 * MR * WR;
    constexpr int NVP = NV > 32 ? ((NV + 31) / 32) * 32 : Pow2Ceil<NV>::v;
    constexpr int CH = NVP > 32 ? 32 : NVP;             /* chunk size */
    int acc[NVP];
#pragma unroll
    for (int i = 0; i < NVP; i++) acc[i] = 0;

    if constexpr (MR <= 4) {
        /* Two K chunks per trip with every load issued before any dp4a: on
         * the K=896 decode shapes a warp has only two chunks, so without this
         * the second chunk's loads wait behind the first chunk's arithmetic
         * (0.5B gate|up m=4: 30.0 -> 29.1 us). Not at m >= 5: the extra
         * registers push those instantiations into spills. */
        for (int k16 = lane + 32 * ks; k16 < K16; k16 += 64 * KS) {
            const int k16b = k16 + 32 * KS;
            const bool two = k16b < K16;
            int4 wv[WR], wv2[WR];
#pragma unroll
            for (int w = 0; w < WR; w++) {
                wv[w] = __ldg(wrow[w] + k16);
                wv2[w] = two ? __ldg(wrow[w] + k16b) : make_int4(0, 0, 0, 0);
            }
#pragma unroll
            for (int p = 0; p < 3; p++)
#pragma unroll
                for (int r = 0; r < MR; r++) {
                    const int4 x = __ldg(xp + p * ps16 + r * xs16 + k16);
                    const int4 x2 = two ? __ldg(xp + p * ps16 + r * xs16 + k16b) : make_int4(0, 0, 0, 0);
#pragma unroll
                    for (int w = 0; w < WR; w++) {
                        int a = acc[(p * MR + r) * WR + w];
                        a = __dp4a(wv[w].x, x.x, a); a = __dp4a(wv[w].y, x.y, a);
                        a = __dp4a(wv[w].z, x.z, a); a = __dp4a(wv[w].w, x.w, a);
                        a = __dp4a(wv2[w].x, x2.x, a); a = __dp4a(wv2[w].y, x2.y, a);
                        a = __dp4a(wv2[w].z, x2.z, a); a = __dp4a(wv2[w].w, x2.w, a);
                        acc[(p * MR + r) * WR + w] = a;
                    }
                }
        }
    } else
    for (int k16 = lane + 32 * ks; k16 < K16; k16 += 32 * KS) {
        int4 wv[WR];
#pragma unroll
        for (int w = 0; w < WR; w++) wv[w] = __ldg(wrow[w] + k16);
#pragma unroll
        for (int p = 0; p < 3; p++)
#pragma unroll
            for (int r = 0; r < MR; r++) {
                const int4 x = __ldg(xp + p * ps16 + r * xs16 + k16);
#pragma unroll
                for (int w = 0; w < WR; w++) {
                    int a = acc[(p * MR + r) * WR + w];
                    a = __dp4a(wv[w].x, x.x, a); a = __dp4a(wv[w].y, x.y, a);
                    a = __dp4a(wv[w].z, x.z, a); a = __dp4a(wv[w].w, x.w, a);
                    acc[(p * MR + r) * WR + w] = a;
                }
            }
    }
    /* Warp totals into shared memory: after reducing chunk c, lane l holds
     * value c*CH + l/(32/CH). At most three chunks (NV <= 96). */
    static_assert(NVP <= 96, "accumulator count");
    warp_reduce_chunk<CH, 0>(acc, lane);
    if (lane % (32 / CH) == 0 && lane / (32 / CH) < NV) red[warp][lane / (32 / CH)] = acc[0];
    if constexpr (NVP > 32) {
        warp_reduce_chunk<32, 32>(acc, lane);
        if (32 + lane < NV) red[warp][32 + lane] = acc[32];
    }
    if constexpr (NVP > 64) {
        warp_reduce_chunk<32, 64>(acc, lane);
        if (64 + lane < NV) red[warp][64 + lane] = acc[64];
    }
    __syncthreads();

    /* Epilogue: thread t -> (activation row r, block row jj); consecutive
     * threads write consecutive outputs, so each (r, block) is one contiguous
     * RB x 4-byte write. MR * RB <= 256: one output per thread. */
    static_assert(MR * RB <= 256, "one epilogue output per thread");
    __shared__ uint8_t pk[3 * MR * RB];          /* the packed form, staged (see below) */
    const int t = threadIdx.x;
    const int r = t / RB, jj = t - r * RB;
    const int j = jbase + jj;
    int32_t v = 0;
    if (t < MR * RB && j < N) {
        const int g = jj / WR, w = jj - g * WR;
        int s0 = 0, s1 = 0, s2 = 0;
#pragma unroll
        for (int k = 0; k < KS; k++) {
            s0 += red[g * KS + k][(0 * MR + r) * WR + w];
            s1 += red[g * KS + k][(1 * MR + r) * WR + w];
            s2 += red[g * KS + k][(2 * MR + r) * WR + w];
        }
        v = crt3(s0, s1, s2);
        if (!tab.pack) reinterpret_cast<int32_t *>(Y)[(long long)r * N + j] = v;
    }
    if (!tab.pack) return;
    /* The packed reply, from the epilogue. NOT as three byte stores per
     * thread: to mapped host memory those leave the card as partial-sector
     * writes, and the 0.5B gate|up exchange at m = 4 measured 490 us against
     * 82 for the int32 form (m = 1 was a wash; the pattern is transport-
     * dependent, not a property of the shape). So the block's 3-byte values
     * are staged in shared memory and each row's run leaves as whole,
     * aligned 4-byte words -- fewer of them than the int32 form writes. A
     * row whose destination is not word-aligned (an odd N, m > 1) or whose
     * run is cut by N (the last block) takes the byte stores; those rows are
     * a rounding error of the reply. */
    if (t < MR * RB) { uint8_t *o = pk + 3 * t; o[0] = (uint8_t)v; o[1] = (uint8_t)(v >> 8); o[2] = (uint8_t)(v >> 16); }
    __syncthreads();
    constexpr int WPR = 3 * RB / 4;                /* words per row: RB is a multiple of 4 */
    const bool full = jbase + RB <= N;
    for (int q = t; q < MR * WPR; q += 256) {
        const int rr = q / WPR, ww = q - rr * WPR;
        uint8_t *row = Y + 3 * ((long long)rr * N + jbase);
        const uint8_t *src = pk + 3 * rr * RB + 4 * ww;
        if (full && (((size_t)row) & 3) == 0) {
            *reinterpret_cast<uint32_t *>(row + 4 * ww) =
                (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
        } else {
#pragma unroll
            for (int b = 0; b < 4; b++)
                if ((4 * ww + b) / 3 + jbase < N) row[4 * ww + b] = src[b];
        }
    }
}

/* FIELD_GEMM24's other card form: the products stay int32 in device memory
 * and this packs them into the mapped reply as whole 4-byte words. Word k
 * holds bytes 4k..4k+3 of the packed stream, which come from at most two
 * values: a = 4k/3 and b = (4k+3)/3. The destination is 3E rounded up to a
 * word, so the last word may carry up to three bytes past the reply; the
 * reply length sent is exactly 3E. */
__global__ void pack24_kernel(const int32_t *__restrict__ y, uint32_t *__restrict__ o, long long E) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long nw = (3 * E + 3) / 4;
    if (k >= nw) return;
    const long long a = (4 * k) / 3, b = (4 * k + 3) / 3;
    const uint32_t va = (uint32_t)y[a], vb = b < E ? (uint32_t)y[b] : 0u;
    uint32_t w;
    switch (k % 3) {
        case 0:  w = (va & 0xffffffu) | (vb << 24); break;
        case 1:  w = ((va >> 8) & 0xffffu) | (vb << 16); break;
        default: w = ((va >> 16) & 0xffu) | (vb << 8); break;
    }
    o[k] = w;
}
static void pack24_launch(const int32_t *y, uint8_t *o, long long E, cudaStream_t s) {
    const long long nw = (3 * E + 3) / 4;
    const int blocks = (int)((nw + 255) / 256);
    pack24_kernel<<<blocks, 256, 0, s>>>(y, (uint32_t *)o, E);
    ck(cudaGetLastError(), "pack launch");
}

/* The host-side pack, for the CPU form and the self-test's reference:
 * E int32 values -> 3E bytes, little-endian two's complement. Four values
 * per 16-byte shuffle where SSSE3 is there; the last groups and a machine
 * without it take the byte loop. */
__attribute__((target("ssse3")))
static void pack24_host_ssse3(const int32_t *y, uint8_t *o, long long E) {
    const __m128i shuf = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    long long i = 0;
    /* Each 16-byte store spills 4 bytes past its 12; the next store covers
     * them, and the loop stops where a spill would pass 3E. */
    for (; i + 8 <= E; i += 4)
        _mm_storeu_si128((__m128i *)(o + 3 * i), _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(y + i)), shuf));
    for (; i < E; i++) { const int32_t v = y[i]; o[3 * i] = (uint8_t)v; o[3 * i + 1] = (uint8_t)(v >> 8); o[3 * i + 2] = (uint8_t)(v >> 16); }
}
static void pack24_host(const int32_t *y, uint8_t *o, long long E) {
    static int ssse3 = -1;
#if defined(__CUDA_ARCH__)
    ssse3 = 0;                         /* host-only code; the device pass still parses it */
#else
    if (ssse3 < 0) { __builtin_cpu_init(); ssse3 = __builtin_cpu_supports("ssse3") ? 1 : 0; }
#endif
    if (ssse3) { pack24_host_ssse3(y, o, E); return; }
    for (long long i = 0; i < E; i++) { const int32_t v = y[i]; o[3 * i] = (uint8_t)v; o[3 * i + 1] = (uint8_t)(v >> 8); o[3 * i + 2] = (uint8_t)(v >> 16); }
}
static inline int32_t unpack24(const uint8_t *p) {
    const uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    return (int32_t)(v << 8) >> 8;
}

/* How a FIELD_GEMM24 reply is produced. All three send identical bytes.
 *   epilogue  the GEMM kernel writes the 3-byte values into the mapped reply
 *   kernel    int32 into device memory, then pack24_kernel into the reply
 *   cpu       int32 into the mapped reply as for FIELD_GEMM, packed on the
 *             host after the sync (pack24_host, ~20 us for lm_head's 152k)
 * Measured 2026-08-26 (RTX 3070, idle box, xtimer medians of 3, wire term
 * of the exchange, int24 minus int32):
 *              0.5B gate|up m=1   gate|up m=4   down m=1   lm_head m=1
 *   epilogue    -0.9 / +0.8 us    -8.3 / -0.2   +0.1/+2.6  -21.0 / +7.2   (tcp / vsock loopback)
 *   kernel      +4.4 / +5.7       +13.5 / +12.9 +4.2/-1.4  +61.1 / +46.6
 *   cpu         -1.2 / -0.9       -5.9 / +1.5   +0.5/-0.2  -20.4 / -26.8
 * The appended pack kernel loses everywhere: one more launch in the graph
 * plus a second pass over the products. The epilogue and the CPU pack are
 * within noise of each other and of the int32 form on the host loopback --
 * there the 152 KB saved on lm_head is worth ~15 us at most -- so the
 * epilogue is the default: no host pass, nothing to schedule, and the
 * byte saving lands where it is paid for, on the guest's vhost-vsock,
 * which this box cannot measure from outside the CVM.
 * SHIELDED_WORKER_PACK=epilogue|kernel|cpu overrides it for an A/B run. */
enum PackMode { PACK_EPILOGUE = 0, PACK_KERNEL = 1, PACK_CPU = 2 };
static const PackMode PACK_DEFAULT = PACK_EPILOGUE;
static PackMode pack_mode() {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("SHIELDED_WORKER_PACK");
        v = PACK_DEFAULT;
        if (e && !strcmp(e, "epilogue")) v = PACK_EPILOGUE;
        else if (e && !strcmp(e, "kernel")) v = PACK_KERNEL;
        else if (e && !strcmp(e, "cpu")) v = PACK_CPU;
    }
    return (PackMode)v;
}

template <int MR, int G>
static void launch_g(const GemmTab &tab, int nblocks, int K, const int8_t *X,
                     long long xstride, long long pstride, cudaStream_t s) {
    field_gemm_kernel<MR, RowsFor<MR>::WR, G><<<nblocks, 256, 0, s>>>(tab, K, X, xstride, pstride);
}

/* One planned launch: <= 8 nodes, <= 8 activation rows, a common G. */
struct GemmPlan { int mr, g, blocks; GemmTab tab; };

/* The G every node of a launch shares: the largest of {8,4,2,1} that still
 * gives the launch at least one block per SM. Measured on the 0.5B down shape
 * (N=896): G=4 (56 blocks) beat G=1 (224 blocks) 8.4 vs 9.7 us at m=1 and 21.7
 * vs 28 us at m=8; the wider block writes wider output rows and splits K less. */
static int g_sm_count = 46;
static GemmPlan gemm_plan(const int8_t *const *W, uint8_t *const *Y, const int *N, int nn, int mr, int pack = 0) {
    GemmPlan pl; pl.mr = mr; pl.g = 1; pl.blocks = 0;
    pl.tab.pack = pack;
    for (int g = 8; g >= 1; g >>= 1) {
        const int rpb = gemm_rows_per_block(mr, g);
        int blocks = 0;
        for (int i = 0; i < nn; i++) blocks += (N[i] + rpb - 1) / rpb;
        pl.g = g; pl.blocks = blocks;
        if (blocks >= g_sm_count || g == 1) break;
    }
    const int rpb = gemm_rows_per_block(mr, pl.g);
    int blocks = 0;
    for (int i = 0; i < GEMM_TAB_NODES; i++) {
        if (i < nn) {
            pl.tab.W[i] = W[i]; pl.tab.Y[i] = Y[i]; pl.tab.N[i] = N[i]; pl.tab.blk0[i] = blocks;
            blocks += (N[i] + rpb - 1) / rpb;
        } else { pl.tab.W[i] = nullptr; pl.tab.Y[i] = nullptr; pl.tab.N[i] = 0; pl.tab.blk0[i] = 0x7fffffff; }
    }
    pl.tab.n = nn;
    return pl;
}
static void gemm_launch_planned(const GemmPlan &pl, int K, const int8_t *X,
                                long long xstride, long long pstride, cudaStream_t s) {
#define LG(MR) case MR: switch (pl.g) { \
        case 8: launch_g<MR, 8>(pl.tab, pl.blocks, K, X, xstride, pstride, s); break; \
        case 4: launch_g<MR, 4>(pl.tab, pl.blocks, K, X, xstride, pstride, s); break; \
        case 2: launch_g<MR, 2>(pl.tab, pl.blocks, K, X, xstride, pstride, s); break; \
        default: launch_g<MR, 1>(pl.tab, pl.blocks, K, X, xstride, pstride, s); break; } break;
    switch (pl.mr) { LG(1) LG(2) LG(3) LG(4) LG(5) LG(6) LG(7) LG(8) default: break; }
#undef LG
    ck(cudaGetLastError(), "kernel launch");
}

/* y[m][N] = (planes . W) for m rows of one node, in passes of up to 8 rows.
 * Used by the legacy doorbell, the self-test and the throughput probe; the
 * exchange path plans its own launches. Completion is the stream's: a kernel-
 * raised flag in mapped memory was measured and bought nothing over
 * cudaStreamSynchronize with the driver spinning (45.6 vs 45.1 us/exchange),
 * and the system-scope fence it needs is expensive on the card. */
static void field_gemm_launch(const int8_t *W, int K, int N, const int8_t *X, int m,
                              long long xstride, long long pstride, int32_t *Y, cudaStream_t s) {
    for (int row0 = 0; row0 < m; row0 += 8) {
        const int mr = std::min(m - row0, 8);
        uint8_t *y = (uint8_t *)(Y + (size_t)row0 * N);
        const GemmPlan pl = gemm_plan(&W, &y, &N, 1, mr);
        gemm_launch_planned(pl, K, X + row0 * xstride, xstride, pstride, s);
    }
}

/* ---------------------------------------------------------------------------
 * Startup self-test: the kernel against an int64 reference, and the CRT
 * constants against the host's. A worker whose arithmetic is wrong is caught by
 * Freivalds in the TEE anyway -- but it would be caught on every request, which
 * is a very expensive way to learn about a typo in KINV01.
 *
 * Covers every MR instantiation the exchange path can pick (m = 1..5, 8), K
 * across the shapes the fleet serves (96 exercises lanes past the end of K,
 * 896/2560/4864 the 0.5B and 4B models), N that is not a multiple of any block
 * size, multi-node fused launches through the same table path the exchange
 * uses, and an m > 8 two-pass case.
 * ------------------------------------------------------------------------ */
static bool selftest_one(int K, int N, int m, int nn, cudaStream_t s, uint64_t &seed, int pack = 0) {
    std::vector<int8_t> W((size_t)nn * N * K), X((size_t)3 * m * K);
    std::vector<int64_t> xr((size_t)m * K);
    auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (auto &w : W) w = (int8_t)((int)(rnd() % 239) - 119);
    for (int i = 0; i < m * K; i++) {
        xr[i] = (int64_t)(rnd() % (uint64_t)SH_M_MOD);
        for (int p = 0; p < 3; p++) X[(size_t)p * m * K + i] = sh_residue(xr[i], sh_primes[p]);
    }
    std::vector<int32_t> ref((size_t)nn * m * N);
    for (int q = 0; q < nn; q++)
        for (int r = 0; r < m; r++)
            for (int j = 0; j < N; j++) {
                int64_t acc = 0;
                for (int k = 0; k < K; k++) acc += xr[(size_t)r * K + k] * W[((size_t)q * N + j) * K + k];
                ref[((size_t)q * m + r) * N + j] = (int32_t)sh_balanced(acc);
            }
    int8_t *dW = nullptr, *dX = nullptr; int32_t *dY = nullptr; uint8_t *dP = nullptr;
    const size_t E = ref.size(), pbytes = ((3 * E + 3) / 4) * 4;
    ck(dmalloc((void **)&dW, W.size()), "selftest malloc");
    ck(dmalloc((void **)&dX, X.size()), "selftest malloc");
    ck(dmalloc((void **)&dY, E * 4), "selftest malloc");
    ck(dmalloc((void **)&dP, pbytes), "selftest malloc");
    ck(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice), "selftest copy");
    ck(cudaMemcpy(dX, X.data(), X.size(), cudaMemcpyHostToDevice), "selftest copy");
    ck(cudaMemset(dY, 0x7f, E * 4), "selftest clear");
    ck(cudaMemset(dP, 0x7f, pbytes), "selftest clear");
    /* The memset runs on the legacy stream; the launches below do not wait for it. */
    ck(cudaDeviceSynchronize(), "selftest sync");
    /* The exchange path's shape: fused launches per pass of <= 8 rows.
     * pack = 1: the epilogue writes 3-byte values into dP; pack = 2: int32
     * into dY, then pack24_kernel over the whole flat output. Both are
     * unpacked here and held to the same int64 reference as the int32 form,
     * and the host pack of that reference must reproduce the card's bytes. */
    for (int row0 = 0; row0 < m; row0 += 8) {
        const int mr = std::min(m - row0, 8);
        const int8_t *Ws[GEMM_TAB_NODES]; uint8_t *Ys[GEMM_TAB_NODES]; int Ns[GEMM_TAB_NODES];
        for (int q = 0; q < nn; q++) {
            Ws[q] = dW + (size_t)q * N * K; Ns[q] = N;
            Ys[q] = pack == 1 ? dP + 3 * (((size_t)q * m + row0) * N) : (uint8_t *)(dY + ((size_t)q * m + row0) * N);
        }
        const GemmPlan pl = gemm_plan(Ws, Ys, Ns, nn, mr, pack == 1);
        gemm_launch_planned(pl, K, dX + (size_t)row0 * K, K, (long long)m * K, s);
    }
    if (pack == 2) pack24_launch(dY, dP, (long long)E, s);
    ck(cudaStreamSynchronize(s), "selftest sync");
    std::vector<int32_t> got(E);
    if (pack) {
        std::vector<uint8_t> pk(pbytes);
        ck(cudaMemcpy(pk.data(), dP, pbytes, cudaMemcpyDeviceToHost), "selftest readback");
        for (size_t i = 0; i < E; i++) got[i] = unpack24(pk.data() + 3 * i);
        std::vector<uint8_t> hp(3 * E);
        pack24_host(ref.data(), hp.data(), (long long)E);
        if (memcmp(hp.data(), pk.data(), 3 * E)) {
            fprintf(stderr, "[shielded-worker] SELFTEST FAILED K=%d N=%d m=%d nodes=%d pack=%d: host pack differs from the card's\n", K, N, m, nn, pack);
            dfree(dW); dfree(dX); dfree(dY); dfree(dP);
            return false;
        }
    } else {
        ck(cudaMemcpy(got.data(), dY, E * 4, cudaMemcpyDeviceToHost), "selftest readback");
    }
    dfree(dW); dfree(dX); dfree(dY); dfree(dP);
    /* The int64 product is far outside Z_M here; balanced() folds it, which is
     * exactly what the GPU's residue arithmetic does implicitly. */
    for (size_t i = 0; i < ref.size(); i++)
        if (got[i] != ref[i]) {
            fprintf(stderr, "[shielded-worker] SELFTEST FAILED K=%d N=%d m=%d nodes=%d pack=%d at %zu: gpu %d ref %d\n",
                    K, N, m, nn, pack, i, got[i], ref[i]);
            return false;
        }
    return true;
}

static bool selftest() {
    uint64_t seed = 0x9e3779b97f4a7c15ull;
    auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    cudaStream_t s; ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "selftest stream");
    bool ok = true;
    const int Ks[] = { 32, 96, 896, 2560, 4864 }, ms[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    for (int K : Ks)
        for (int m : ms)
            if (ok) ok = selftest_one(K, 37, m, 1, s, seed);
    /* Fused launches: N large enough to pick G > 1 with a ragged tail, the
     * full 8-node table, and an m > 8 two-pass case. */
    if (ok) ok = selftest_one(96, 1234, 1, 2, s, seed);
    if (ok) ok = selftest_one(96, 1234, 8, 3, s, seed);
    if (ok) ok = selftest_one(96, 4103, 3, 2, s, seed);
    if (ok) ok = selftest_one(96, 301, 2, 8, s, seed);
    if (ok) ok = selftest_one(96, 37, 11, 2, s, seed);
    if (ok) ok = selftest_one(32, 1, 7, 1, s, seed);      /* smallest K, N below a warp's rows, MR=7 */
    if (ok) ok = selftest_one(96, 3, 6, 3, s, seed);
    /* The packed reply, both card forms: 3E not a multiple of 4 (the tail
     * word), multi-node fused tables, m > 8 two-pass, a single value. */
    for (int pack = 1; pack <= 2 && ok; pack++) {
        ok = ok && selftest_one(896, 37, 1, 1, s, seed, pack);
        ok = ok && selftest_one(96, 1234, 3, 2, s, seed, pack);
        ok = ok && selftest_one(96, 4103, 8, 3, s, seed, pack);
        ok = ok && selftest_one(96, 301, 2, 8, s, seed, pack);
        ok = ok && selftest_one(96, 37, 11, 2, s, seed, pack);
        ok = ok && selftest_one(32, 1, 1, 1, s, seed, pack);
        ok = ok && selftest_one(32, 5, 1, 1, s, seed, pack);
    }
    cudaStreamDestroy(s);
    if (!ok) return false;
    /* CRT constants vs the host's Garner. */
    for (int t = 0; t < 1000; t++) {
        const int64_t v = (int64_t)(rnd() % (uint64_t)SH_M_MOD);
        const int64_t h = sh_crt(sh_residue(v, SH_Q0), sh_residue(v, SH_Q1), sh_residue(v, SH_Q2));
        if (h != sh_balanced(v)) { fprintf(stderr, "[shielded-worker] host CRT disagrees with itself\n"); return false; }
    }
    return true;
}

/* Field GEMM throughput on this card, G-MAC/s, on the decode-shaped kernel.
 * Reported in HELLO and advertised to the fleet as this card's rate for the
 * operation it actually performs -- not a spec-sheet FP16 number. */
static double measure_gmacs() {
    const int K = 4096, N = 4096, m = 8, iters = 20;
    int8_t *dW = nullptr, *dX = nullptr; int32_t *dY = nullptr;
    if (dmalloc((void **)&dW, (size_t)N * K) != cudaSuccess) return 0.0;
    if (dmalloc((void **)&dX, (size_t)3 * m * K) != cudaSuccess) { dfree(dW); return 0.0; }
    if (dmalloc((void **)&dY, (size_t)m * N * 4) != cudaSuccess) { dfree(dW); dfree(dX); return 0.0; }
    cudaMemset(dW, 1, (size_t)N * K); cudaMemset(dX, 1, (size_t)3 * m * K);
    cudaDeviceSynchronize();
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    for (int i = 0; i < 3; i++) field_gemm_launch(dW, K, N, dX, m, K, (long long)m * K, dY, s);
    cudaStreamSynchronize(s);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) field_gemm_launch(dW, K, N, dX, m, K, (long long)m * K, dY, s);
    cudaStreamSynchronize(s);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cudaStreamDestroy(s);
    dfree(dW); dfree(dX); dfree(dY);
    return dt > 0 ? (double)m * K * N * iters / dt / 1e9 : 0.0;
}

#ifdef SH_XPROF
/* Temporary: --kbench times the kernel alone (cudaEvents) per shape, per G,
 * with Y in device memory and in mapped host memory. Compiled out of the
 * shipped worker. */
static int kbench() {
    struct Shape { int K, N, m, nn; const char *name; };
    std::vector<Shape> shapes = {
        {896,4864,1,2,"0.5B gate|up x2"},{4864,896,1,1,"0.5B down"},{896,151936,1,1,"0.5B lm_head"},
        {896,4864,2,2,"0.5B gate|up x2 m2"},{896,4864,4,2,"0.5B gate|up x2 m4"},{896,4864,8,2,"0.5B gate|up x2 m8"},
        {4864,896,4,1,"0.5B down m4"},{4864,896,8,1,"0.5B down m8"},
        {2560,9728,1,2,"4B gate|up x2"},{9728,2560,1,1,"4B down"},{2560,151936,1,1,"4B lm_head"},
        {2560,9728,4,2,"4B gate|up x2 m4"},{2560,9728,8,2,"4B gate|up x2 m8"},
        {9728,2560,4,1,"4B down m4"},{9728,2560,8,1,"4B down m8"},
        {4096,4096,8,1,"4096^2 m8 (HELLO shape)"}};
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    printf("%-26s %6s %4s | %28s | %28s\n", "shape", "MB", "m", "Y device: us (GB/s) G=1/2/4/8", "Y mapped: us (GB/s) G=1/2/4/8");
    for (auto &sh : shapes) {
        const size_t wb = (size_t)sh.K * sh.N * sh.nn;
        int8_t *dW, *dX; int32_t *dY; int32_t *hY, *dhY;
        if (dmalloc((void **)&dW, wb) != cudaSuccess) { printf("%-26s alloc fail\n", sh.name); continue; }
        dmalloc((void **)&dX, (size_t)3 * sh.m * sh.K); dmalloc((void **)&dY, (size_t)sh.m * sh.N * sh.nn * 4);
        cudaHostAlloc((void **)&hY, (size_t)sh.m * sh.N * sh.nn * 4, cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **)&dhY, hY, 0);
        cudaMemset(dW, 3, wb); cudaMemset(dX, 1, (size_t)3 * sh.m * sh.K); cudaDeviceSynchronize();
        printf("%-26s %6.1f %4d |", sh.name, wb / 1e6, sh.m);
        for (int which = 0; which < 2; which++) {
            for (int g = 1; g <= 8; g <<= 1) {
                const int8_t *Ws[2]; uint8_t *Ys[2]; int Ns[2];
                for (int i = 0; i < sh.nn; i++) { Ws[i] = dW + (size_t)i * sh.K * sh.N; Ys[i] = (uint8_t *)((which ? dhY : dY) + (size_t)i * sh.m * sh.N); Ns[i] = sh.N; }
                GemmPlan pl = gemm_plan(Ws, Ys, Ns, sh.nn, sh.m);
                /* force this G */
                pl.g = g; { const int rpb = gemm_rows_per_block(sh.m, g); int b = 0; for (int i = 0; i < sh.nn; i++) { pl.tab.blk0[i] = b; b += (sh.N + rpb - 1) / rpb; } pl.blocks = b; }
                for (int i = 0; i < 3; i++) gemm_launch_planned(pl, sh.K, dX, sh.K, (long long)sh.m * sh.K, s);
                cudaStreamSynchronize(s);
                const int iters = 30;
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; i++) gemm_launch_planned(pl, sh.K, dX, sh.K, (long long)sh.m * sh.K, s);
                cudaEventRecord(e1, s); cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                const double us = ms * 1e3 / iters;
                printf(" %6.1f(%3.0f)", us, wb / us / 1e3);
            }
            printf(" |");
        }
        printf("\n");
        dfree(dW); dfree(dX); dfree(dY); cudaFreeHost(hY);
    }
    return 0;
}
#endif

/* ---------------------------------------------------------------------------
 * A JSON reader for the install spec: objects, arrays, strings, numbers, the
 * three literals. Nothing else is needed and nothing else is accepted.
 * ------------------------------------------------------------------------ */
struct JVal {
    enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
    bool b = false; double num = 0; std::string str;
    std::vector<JVal> arr; std::vector<std::pair<std::string, JVal>> obj;
    const JVal *get(const char *k) const {
        if (kind != OBJ) return nullptr;
        for (auto &kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    int64_t i64(const char *k) const {
        const JVal *v = get(k);
        if (!v || v->kind != NUM) VIOLATE("graph spec: missing or non-numeric %s", k);
        return (int64_t)v->num;
    }
};
struct JParser {
    const char *p, *end;
    int depth = 0;                  /* nesting; a spec is a few levels deep, a stack overflow is not a violation */
    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
    bool eat(char c) { ws(); if (p < end && *p == c) { p++; return true; } return false; }
    JVal parse() {
        ws();
        if (p >= end) VIOLATE("malformed graph spec: truncated");
        if (depth > 32) VIOLATE("malformed graph spec: nesting deeper than 32");
        struct Depth { int &d; Depth(int &x) : d(x) { d++; } ~Depth() { d--; } } guard(depth);
        JVal v;
        if (*p == '{') {
            p++; v.kind = JVal::OBJ;
            if (eat('}')) return v;
            for (;;) {
                ws(); if (p >= end || *p != '"') VIOLATE("malformed graph spec: key");
                std::string k = parse_str();
                if (!eat(':')) VIOLATE("malformed graph spec: colon");
                JVal val = parse();
                v.obj.emplace_back(std::move(k), std::move(val));
                if (eat(',')) continue;
                if (eat('}')) return v;
                VIOLATE("malformed graph spec: object");
            }
        }
        if (*p == '[') {
            p++; v.kind = JVal::ARR;
            if (eat(']')) return v;
            for (;;) {
                v.arr.push_back(parse());
                if (eat(',')) continue;
                if (eat(']')) return v;
                VIOLATE("malformed graph spec: array");
            }
        }
        if (*p == '"') { v.kind = JVal::STR; v.str = parse_str(); return v; }
        if (end - p >= 4 && !strncmp(p, "true", 4)) { p += 4; v.kind = JVal::BOOL; v.b = true; return v; }
        if (end - p >= 5 && !strncmp(p, "false", 5)) { p += 5; v.kind = JVal::BOOL; return v; }
        if (end - p >= 4 && !strncmp(p, "null", 4)) { p += 4; return v; }
        char *e = nullptr;
        v.num = strtod(p, &e);
        if (e == p) VIOLATE("malformed graph spec: token");
        p = e; v.kind = JVal::NUM; return v;
    }
    std::string parse_str() {
        p++; std::string s;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                p++; if (p >= end) break;
                switch (*p) { case 'n': s += '\n'; break; case 't': s += '\t'; break;
                              case 'u': p += 4; s += '?'; break; default: s += *p; }
            } else s += *p;
            p++;
        }
        if (p >= end) VIOLATE("malformed graph spec: string");
        p++; return s;
    }
};

/* ---------------------------------------------------------------------------
 * Socket helpers. Frames: | cmd u8 | size u64 LE | payload |, responses
 * | status u8 | size u64 LE | payload |. A violation is the last frame.
 * ------------------------------------------------------------------------ */
/* The next frame usually follows the reply by the TEE's share of a token
 * (~50-100 us): spin on a non-blocking receive for that long before blocking,
 * so the wake-up on the host is not on the round trip. This thread serves one
 * connection and nothing else. SHIELDED_WORKER_SPIN_US, default 0 = off:
 * measured -1.6 us on TCP loopback and +2.7 us on vsock loopback, i.e. not
 * worth a spinning core until a guest measurement says otherwise. */
static int g_spin_us = -1;
static bool read_exact(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    if (g_spin_us < 0) { const char *e = getenv("SHIELDED_WORKER_SPIN_US"); g_spin_us = (e && *e) ? std::max(0, atoi(e)) : 0; }
    if (g_spin_us > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int spins = 0; n; spins++) {
            ssize_t r = recv(fd, p, n, MSG_DONTWAIT);
            if (r > 0) { p += r; n -= (size_t)r; continue; }
            if (r == 0) return false;
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return false;
            if ((spins & 63) == 63 &&
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() > g_spin_us) break;
        }
    }
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        p += r; n -= (size_t)r;
    }
    return true;
}
static bool write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n) {
        ssize_t r = write(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        p += r; n -= (size_t)r;
    }
    return true;
}
/* Header + body in one call; falls back to plain writes for whatever a short
 * writev left behind. */
static bool writev_all(int fd, const void *hdr, size_t hn, const void *body, size_t bn) {
    struct iovec iov[2];
    iov[0].iov_base = (void *)hdr; iov[0].iov_len = hn;
    iov[1].iov_base = (void *)body; iov[1].iov_len = bn;
    ssize_t r;
    do { r = writev(fd, iov, 2); } while (r < 0 && errno == EINTR);
    if (r < 0) return false;
    if ((size_t)r >= hn + bn) return true;
    if ((size_t)r < hn) return write_all(fd, (const uint8_t *)hdr + r, hn - r) && write_all(fd, body, bn);
    return write_all(fd, (const uint8_t *)body + (r - hn), bn - (r - hn));
}
static uint64_t rd_u64(const uint8_t *p) { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i); return v; }
static uint32_t rd_u32(const uint8_t *p) { uint32_t v = 0; for (int i = 0; i < 4; i++) v |= (uint32_t)p[i] << (8 * i); return v; }
static void wr_u64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void wr_u32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }

/* ---------------------------------------------------------------------------
 * The shared-memory ring (--shm <file>). Mirrors wasm/ggml-shielded/
 * shielded-wire.h: the file holds size / 8 MiB rings; ring i at i * 8 MiB;
 * per ring a request sequence line, a reply sequence line, the two socket
 * headers byte for byte, a 2 MiB request slot and a 6 MiB reply slot.
 *
 * Why it exists: in the SEV-SNP guest a vhost-vsock exchange costs 152 us,
 * of which this worker's card path is ~28 and the socket ~10 -- the rest is
 * VM exits and interrupts. On the host the file is a plain shared mapping;
 * in the guest it is the BAR of an ivshmem-plain device backed by the same
 * file. Both sides poll; nobody sleeps on the exchange.
 *
 * Trust: the ring is written by the guest and this worker only ever COPIES
 * a request out of it into its pinned staging (a snapshot) before any
 * admission check reads a field, so the guest editing the slot mid-flight
 * cannot move a bound after it was checked. The reply is written last, then
 * the sequence line (release). Everything else -- allowlist, K, max_m, the
 * exact-length rule -- is the unchanged field_gemm().
 * ------------------------------------------------------------------------ */
static const size_t RING_BYTES = (size_t)8 << 20, RING_OFF_REQ = 0, RING_OFF_REP = 64,
                    RING_OFF_RQH = 128, RING_OFF_RPH = 192, RING_OFF_RQP = 4096,
                    RING_OFF_RPP = (size_t)2 << 20;
static const size_t RING_REQ_CAP = RING_OFF_RPP - RING_OFF_RQP, RING_REP_CAP = RING_BYTES - RING_OFF_RPP;
static const size_t RING_MAX_FILE = (size_t)64 << 20;
static uint8_t *g_shm = nullptr; static size_t g_shm_len = 0; static int g_nrings = 0;
static std::atomic<int> g_ring_owner[RING_MAX_FILE / RING_BYTES];
/* How long a ring-serving thread spins with no traffic before it backs off
 * to a 1 ms poll: a decode token is ~5-10 ms of continuous exchanges, so a
 * live link never sees the back-off and an idle one costs no core. */
static int g_shm_idle_ms = 200;
static inline uint64_t ring_ld(const uint8_t *at) { return __atomic_load_n((const uint64_t *)at, __ATOMIC_ACQUIRE); }
static inline void ring_st(uint8_t *at, uint64_t v) { _mm_sfence(); __atomic_store_n((uint64_t *)at, v, __ATOMIC_RELEASE); }

/* ---------------------------------------------------------------------------
 * Per-connection state: the admission rules of protocol.py, plus the storage
 * they gate. Buffers, graph and scratch die with the connection; only the
 * process-wide card survives.
 * ------------------------------------------------------------------------ */
static std::mutex g_gpu;                 /* one kernel stream at a time */
static long long g_vram_budget = 0;
static double g_gmacs = 0.0;
static std::chrono::steady_clock::time_point g_gmacs_at;
static cudaDeviceProp g_props;

/* THE BUDGET IS THE CARD THE FLEET SEES; A TENANT RESERVES ITS SHARE AT HELLO.
 *
 * --vram-gb is the part of the card dedicated to Enclave. The fleet sees a
 * card of exactly that size and nothing else (the supervisor adopts it as
 * CARD_VRAM_GB); the rest of the card is the host's, and the budget is 100%
 * free until an app has reserved a share of it. The worker holds NO device
 * memory for the budget at start-up: the only start-up requirement is that
 * the card is at least the budget (a smaller card is a misconfiguration and
 * exits 75, naming both figures); a card with less than the budget free at
 * start is a warning, because whatever holds it may leave.
 *
 * Why memory is reserved at all: a game started on the production 3070 on
 * 2026-08-26 and took 6.3 GB while the worker was taking device memory from
 * the driver lazily, so the tenant's next allocation failed -- a tenant that
 * had bought that memory lost it to a neighbour that started later. The first
 * fix claimed the whole budget at start (one process holding 6.5 GiB at idle
 * whether or not anyone was running), which made the fleet's "free" figure a
 * lie in the other direction. Now the claim happens at PLACEMENT: a 1.3 HELLO
 * carries the tenant's reservation (the guest sends its VRAM share), and the
 * worker, under the GPU mutex, refuses it if the sum of live reservations
 * would exceed the budget, else takes it from the driver through the
 * stream-ordered allocator's default pool -- the pool then keeps what it
 * holds (release threshold up; explicit trims down to the sum of
 * reservations plus whatever unreserved links hold), so memory a tenant
 * frees stays with this process for that tenant, and a claim
 * the driver cannot honour (something else holds the card) is refused with
 * the free figure. A refused HELLO is a dead link to the guest: it computes
 * in the enclave and reconnects with backoff, so the tenant runs until the
 * memory is there. At disconnect the reservation is released and the pool is
 * trimmed to what the remaining tenants reserved: the memory is visibly back
 * in the driver (nvidia-smi, cudaMemGetInfo) the moment the link closes.
 *
 * The per-connection cap is the reservation when there is one, else the
 * budget (a 4-byte HELLO: exactly the pre-1.3 behaviour). The cap counts the
 * DEVICE bytes a link holds -- activations buffers and the installed graph's
 * node weights -- and a separate host-RAM ledger bounds the 'weights' buffers
 * (host-resident until install) by the budget, as before. The largest
 * consumer used to be the one thing uncounted: node weights.
 *
 * The fleet's free figure is min(budget - vram_reserved, vram_free): the
 * supervisor sends a 4-byte HELLO and reads both from the reply. */
static long long g_reserved = 0;         /* sum of live reservations; under g_gpu */
static long long g_floating = 0;         /* device bytes held by links WITHOUT a reservation; under g_gpu */
static cudaError_t dmalloc(void **p, size_t n) {
    cudaError_t e = cudaMallocAsync(p, n ? n : 1, 0);
    if (e == cudaSuccess) e = cudaStreamSynchronize(0);
    return e;
}
static void dfree(void *p) { if (p) { cudaFreeAsync(p, 0); cudaStreamSynchronize(0); } }
static cudaMemPool_t default_pool() {
    cudaMemPool_t pool = nullptr;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    return pool;
}
static long long pool_reserved_now() {
    cuuint64_t r = 0;
    cudaMemPoolGetAttribute(default_pool(), cudaMemPoolAttrReservedMemCurrent, &r);
    return (long long)r;
}
/* What the pool keeps for us: everything, while anything is reserved or an
 * unreserved link holds device memory; nothing when idle. The pool releases
 * in whole chunks (32 MiB on this driver) and a threshold below the chunk
 * drops the chunk -- measured 2026-08-27: a 16 MiB reservation over a 16 MiB
 * threshold left the pool holding 0 -- so the threshold is not the ledger.
 * The ledger is g_reserved + g_floating, and it is enforced by explicit trims
 * (pool_trim) whenever something is given back; the threshold only decides
 * whether a synchronisation may hand memory to the driver on its own.
 * Called under g_gpu. */
static void pool_retune() {
    cuuint64_t keep = (g_reserved + g_floating > 0) ? UINT64_MAX : 0;
    cudaMemPoolSetAttribute(default_pool(), cudaMemPoolAttrReleaseThreshold, &keep);
}
/* Give the driver everything above the ledger: what the remaining tenants
 * reserved, plus what unreserved links hold, stays (at least; whole chunks). */
static void pool_trim() {
    pool_retune();
    cudaMemPoolTrimTo(default_pool(), (size_t)(g_reserved + g_floating));
}
/* Take R more bytes from the driver for the pool, or leave the pool as it
 * was. The pool serves a new allocation from its cache first, so one malloc
 * of R need not grow it: hold what is handed out until the pool's reserved
 * figure reaches the new threshold, then free the holds -- the threshold now
 * keeps them. Called under g_gpu; on failure *freeb is the driver's figure. */
static bool claim_reservation(long long R, size_t *freeb, long long *held) {
    const long long target = g_reserved + R + g_floating;
    g_reserved += R;
    pool_retune();
    std::vector<void *> holds;
    bool ok = true;
    for (int i = 0; i < 16 && ok; i++) {
        const long long have = pool_reserved_now();
        if (have >= target) break;
        void *p = nullptr;
        if (cudaMallocAsync(&p, (size_t)(target - have), 0) != cudaSuccess) { cudaGetLastError(); ok = false; break; }
        holds.push_back(p);
    }
    for (void *p : holds) cudaFreeAsync(p, 0);
    if (cudaStreamSynchronize(0) != cudaSuccess) ok = false;
    *held = pool_reserved_now();
    if (ok && *held < target) ok = false;          /* the pool let the holds go: not a reservation */
    if (!ok) {
        g_reserved -= R;
        pool_trim();
        size_t totalb = 0; cudaMemGetInfo(freeb, &totalb);
    }
    return ok;
}
/* The link is gone and its memory freed: hand the reservation back to the
 * driver now, not at some later synchronisation. Called under g_gpu. */
static void release_reservation(long long R) {
    g_reserved -= R;
    pool_trim();
}

struct Buffer {
    uint64_t bid = 0, size = 0;
    std::string role;
    bool consumed = false;               /* weights: host copy released at install */
    int8_t *dev = nullptr;               /* activations: device-resident */
    std::vector<uint8_t> host;           /* weights: host-resident until install */
};
struct Node {
    std::string id;
    bool gemm = false;
    int8_t *w = nullptr;                 /* (N,K) int8 on the device */
    size_t wbytes = 0;                   /* counted against the link's device cap */
    int64_t K = 0, N = 0; int max_m = 0;
    uint64_t xbid = 0, xoff = 0, ybid = 0, yoff = 0;
};

struct Conn {
    int fd;
    std::string peer;
    bool hello_done = false, installed = false;
    std::map<uint64_t, Buffer> buffers;
    uint64_t next_bid = 1;
    long long allocated = 0;                 /* DEVICE bytes: activations buffers + node weights */
    long long host_allocated = 0;            /* host bytes: 'weights' buffers until install */
    long long reserve = 0;                   /* this link's reservation from HELLO, 0 = none */
    std::vector<Node> nodes;
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> outputs;
    cudaStream_t stream = nullptr;
    /* Staging for the one-frame exchange. The FIELD_GEMM frame is read straight
     * into pinned memory (h_in), so the planes go to the card with no memcpy;
     * the product is written by the kernel into MAPPED pinned memory (h_out),
     * so it comes back with no D2H copy and is sent from where it landed. */
    uint8_t *h_in = nullptr;  size_t h_in_cap = 0;
    int8_t  *d_x = nullptr;   size_t d_x_cap = 0;
    uint8_t *h_out = nullptr; size_t h_out_cap = 0; int32_t *d_out = nullptr;
    /* FIELD_GEMM24 staging: the int32 products on the device (PACK_KERNEL)
     * and the host-packed reply (PACK_CPU). Unused by the default form. */
    int32_t *d_y32 = nullptr; size_t d_y32_cap = 0;
    std::vector<uint8_t> h_pack;
    /* A response that lives in h_out rather than in a std::string. */
    const void *resp_ptr = nullptr; size_t resp_len = 0;
    /* The exchange's card work (upload + launches), captured once per
     * (m, node list) and replayed with one cudaGraphLaunch: 42.5 vs 45.0 us
     * per 0.5B gate|up exchange against issuing the copy and the launch
     * separately. The captured pointers are h_in, d_x and h_out, so the cache
     * is dropped whenever any of them is reallocated; bounded at 256 entries
     * (a 0.5B decode uses three). */
    std::map<std::vector<uint32_t>, cudaGraphExec_t> graphs;
    void drop_graphs() { for (auto &kv : graphs) cudaGraphExecDestroy(kv.second); graphs.clear(); }
    uint64_t exchanges = 0, recomputes = 0;
    double gemm_ms = 0;
    int64_t kmax = 0;                        /* widest K installed: bounds a FIELD_GEMM frame */
    /* The ring this connection owns after SHM_ATTACH, or none. */
    uint8_t *ring = nullptr; int ring_index = -1; uint64_t ring_seen = 0;
    uint64_t ring_exchanges = 0;

    Conn(int f, std::string p) : fd(f), peer(std::move(p)) {}
    ~Conn() {
        if (ring_index >= 0) g_ring_owner[ring_index].store(0);
        if (stream) cudaStreamSynchronize(stream);          /* nothing in flight before the pool takes the memory back */
        for (auto &kv : buffers) if (kv.second.dev) dfree(kv.second.dev);
        for (auto &n : nodes) if (n.w) dfree(n.w);
        if (d_x) dfree(d_x);
        if (d_y32) dfree(d_y32);
        drop_graphs();
        if (h_in) cudaFreeHost(h_in);
        if (h_out) cudaFreeHost(h_out);
        if (stream) cudaStreamDestroy(stream);
        {
            std::lock_guard<std::mutex> lk(g_gpu);
            if (!reserve) g_floating -= allocated;
            release_reservation(reserve);            /* also trims what an unreserved link freed */
        }
    }

    /* The device bytes this link may hold: its reservation, else the budget. */
    long long cap() const { return reserve > 0 ? reserve : g_vram_budget; }
    /* Account device bytes against the cap, or refuse. A link without a
     * reservation also moves the pool's threshold: its usage is not reserved
     * memory, and must not push a reserver's cache back to the driver. */
    void charge_device(long long nbytes, const char *what) {
        if (nbytes > cap() || allocated + nbytes > cap())
            VIOLATE("%s of %lld bytes exceeds the link's %s (%lld of %lld held)", what, nbytes,
                    reserve > 0 ? "reservation" : "budget", allocated, cap());
        allocated += nbytes;
        if (!reserve) { std::lock_guard<std::mutex> lk(g_gpu); g_floating += nbytes; pool_retune(); }
    }
    /* An unreserved link's freed bytes go back to the driver at once (the
     * caller frees, then uncharges); a reserver's stay in the pool for it. */
    void uncharge_device(long long nbytes) {
        allocated -= nbytes;
        if (!reserve) { std::lock_guard<std::mutex> lk(g_gpu); g_floating -= nbytes; pool_trim(); }
    }

    Buffer &region_ok(uint64_t bid, uint64_t off, uint64_t nbytes) {
        auto it = buffers.find(bid);
        if (it == buffers.end()) VIOLATE("reference to unknown buffer %llu", (unsigned long long)bid);
        Buffer &b = it->second;
        if (off > b.size || nbytes > b.size - off)
            VIOLATE("region [%llu,%llu) outside buffer %llu of %llu",
                    (unsigned long long)off, (unsigned long long)(off + nbytes),
                    (unsigned long long)bid, (unsigned long long)b.size);
        return b;
    }
    void ensure_host_in(size_t n) {
        if (h_in_cap >= n) return;
        if (h_in) cudaFreeHost(h_in);
        ck(cudaHostAlloc((void **)&h_in, n, cudaHostAllocDefault), "pinned alloc"); h_in_cap = n;
        drop_graphs();
    }
    void ensure_host_out(size_t n) {
        if (h_out_cap >= n) return;
        if (h_out) cudaFreeHost(h_out);
        ck(cudaHostAlloc((void **)&h_out, n, cudaHostAllocMapped), "pinned alloc"); h_out_cap = n;
        ck(cudaHostGetDevicePointer((void **)&d_out, h_out, 0), "pinned map");
        drop_graphs();
    }
    void ensure_dx(size_t n) {
        if (d_x_cap >= n) return;
        if (d_x) dfree(d_x);
        ck(dmalloc((void **)&d_x, n), "device scratch alloc"); d_x_cap = n;
        drop_graphs();
    }
    void ensure_dy32(size_t n) {
        if (d_y32_cap >= n) return;
        if (d_y32) dfree(d_y32);
        ck(dmalloc((void **)&d_y32, n), "device product alloc"); d_y32_cap = n;
        drop_graphs();
    }

    std::string hello(const uint8_t *p, size_t n) {
        if (hello_done) VIOLATE("duplicate HELLO");
        if (n < 4) VIOLATE("truncated u32");
        const uint32_t major = rd_u32(p);
        if (major != (uint32_t)PROTO_MAJOR) VIOLATE("protocol major %u != %d", major, PROTO_MAJOR);
        /* 1.3: an optional u64 reservation. Checked against the budget before
         * any arithmetic on it: a u64 above the budget is refused as-is, so
         * the sum below cannot wrap. */
        uint64_t want = 0;
        if (n >= 12) want = rd_u64(p + 4);
        if (want > (uint64_t)g_vram_budget)
            VIOLATE("reservation %llu exceeds the budget: %lld reserved of %lld",
                    (unsigned long long)want, g_reserved, g_vram_budget);
        hello_done = true;
        size_t freeb = 0, totalb = 0;
        {
            std::lock_guard<std::mutex> lk(g_gpu);
            if (want > 0) {
                if (g_reserved + (long long)want > g_vram_budget)
                    VIOLATE("reservation %llu exceeds the budget: %lld reserved of %lld",
                            (unsigned long long)want, g_reserved, g_vram_budget);
                long long held = 0;
                if (!claim_reservation((long long)want, &freeb, &held))
                    VIOLATE("cannot reserve %llu: the card has %zu free (the pool holds %lld against %lld reserved)",
                            (unsigned long long)want, freeb, held, g_reserved);
                reserve = (long long)want;
            }
            cudaMemGetInfo(&freeb, &totalb);
            /* Re-measured on HELLO when the last figure is older than 20 s: the
             * guest asks every 30 s and advertises the answer, and a card that
             * is being time-sliced with a game answers ~40% of its idle figure
             * (904 against 2150 G-MAC/s on 2026-08-26). A few ms of card time
             * per half minute; a contended card is not capacity. */
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - g_gmacs_at).count() > 20.0) {
                const double g = measure_gmacs();
                if (g > 0) { g_gmacs = g; g_gmacs_at = now; }
                pool_trim();                          /* the benchmark's scratch, back to the driver (under g_gpu) */
            }
        }
        /* card_tflops is the RATED sizing figure; field_gmac_per_s is the MEASURED
         * masked throughput. Both cross, because they answer different
         * questions, and the inputs to the derived one cross too so a reader can
         * recompute it instead of trusting an untrusted worker's arithmetic. */
        /* 1.3: vram_reserved is the sum of live reservations AFTER this one,
         * vram_reserve this link's own. The fleet's free figure is
         * min(vram_budget - vram_reserved, vram_free). */
        return fmt("{\"version\":[%d,%d,%d],\"device\":\"%s\",\"vram_total\":%llu,\"vram_free\":%llu,"
                   "\"vram_budget\":%lld,\"vram_reserved\":%lld,\"vram_reserve\":%lld,"
                   "\"sm_count\":%d,\"capability\":\"%d.%d\","
                   "\"clock_khz\":%d,\"card_tflops\":%.1f,"
                   "\"field_gmac_per_s\":%.1f,\"worker\":\"shielded/worker-cuda\"}",
                   PROTO_MAJOR, PROTO_MINOR, PROTO_PATCH, g_props.name,
                   (unsigned long long)g_props.totalGlobalMem, (unsigned long long)freeb,
                   g_vram_budget, g_reserved, reserve, g_props.multiProcessorCount, g_props.major, g_props.minor,
                   g_props.clockRate, rated_fp16_tflops(g_props), g_gmacs);
    }

    std::string alloc(const uint8_t *p, size_t n) {
        if (n < 12) VIOLATE("truncated u64");
        const uint64_t size = rd_u64(p);
        const uint32_t rl = rd_u32(p + 8);
        if (12 + rl > n) VIOLATE("truncated role");
        std::string role((const char *)p + 12, rl);
        if (role != "weights" && role != "activations") VIOLATE("unknown buffer role '%s'", role.c_str());
        /* Both halves: a size above 2^63 wrapped the sum negative and passed. */
        if (size > (uint64_t)g_vram_budget) VIOLATE("allocation exceeds device memory");
        Buffer b; b.bid = next_bid++; b.size = size; b.role = role;
        if (role == "activations") {
            charge_device((long long)size, "allocation");
            if (dmalloc((void **)&b.dev, size ? size : 1) != cudaSuccess) {
                uncharge_device((long long)size);
                VIOLATE("device allocation of %llu failed", (unsigned long long)size);
            }
        } else {
            /* Host-resident until install; bounded by the budget as it always
             * was, so a link cannot pin the host's RAM either. */
            if (host_allocated + (long long)size > g_vram_budget) VIOLATE("allocation exceeds device memory");
            b.host.assign(size, 0);
            host_allocated += (long long)size;
        }
        const uint64_t bid = b.bid;
        buffers[bid] = std::move(b);
        std::string r(8, '\0'); wr_u64((uint8_t *)&r[0], bid);
        return r;
    }

    std::string free_buf(const uint8_t *p, size_t n) {
        if (n < 8) VIOLATE("truncated u64");
        const uint64_t bid = rd_u64(p);
        auto it = buffers.find(bid);
        if (it == buffers.end()) VIOLATE("free of unknown buffer %llu", (unsigned long long)bid);
        if (it->second.dev) { dfree(it->second.dev); uncharge_device((long long)it->second.size); }
        else if (!it->second.consumed) host_allocated -= (long long)it->second.size;
        buffers.erase(it);
        return "";
    }

    std::string set_tensor(const uint8_t *p, size_t n) {
        if (n < 24) VIOLATE("truncated u64");
        const uint64_t bid = rd_u64(p), off = rd_u64(p + 8), nbytes = rd_u64(p + 16);
        Buffer &b = region_ok(bid, off, nbytes);
        if (!b.dev && b.host.empty() && nbytes) VIOLATE("buffer %llu was consumed at install", (unsigned long long)bid);
        if (n - 24 != nbytes) VIOLATE("SET_TENSOR declared %llu bytes, frame carries %zu",
                                      (unsigned long long)nbytes, n - 24);
        if (b.dev) {
            std::lock_guard<std::mutex> lk(g_gpu);
            ck(cudaMemcpy(b.dev + off, p + 24, nbytes, cudaMemcpyHostToDevice), "SET_TENSOR copy");
        } else {
            memcpy(b.host.data() + off, p + 24, nbytes);
        }
        return "";
    }

    std::string get_tensor(const uint8_t *p, size_t n) {
        if (n < 24) VIOLATE("truncated u64");
        const uint64_t bid = rd_u64(p), off = rd_u64(p + 8), nbytes = rd_u64(p + 16);
        Buffer &b = region_ok(bid, off, nbytes);
        if (!b.dev && b.host.empty() && nbytes) VIOLATE("buffer %llu was consumed at install", (unsigned long long)bid);
        if (!installed) VIOLATE("GET_TENSOR before any graph was installed");
        if (!outputs.count(std::make_tuple(bid, off, nbytes)))
            VIOLATE("GET_TENSOR region (%llu,%llu,%llu) is not a declared graph output",
                    (unsigned long long)bid, (unsigned long long)off, (unsigned long long)nbytes);
        std::string out(nbytes, '\0');
        if (b.dev) {
            std::lock_guard<std::mutex> lk(g_gpu);
            ck(cudaMemcpy(&out[0], b.dev + off, nbytes, cudaMemcpyDeviceToHost), "GET_TENSOR copy");
        } else {
            memcpy(&out[0], b.host.data() + off, nbytes);
        }
        return out;
    }

    /* Bind every allowlisted node to real storage, or refuse the graph. The
     * invariant that matters: weights come from a 'weights' buffer and
     * activations from an 'activations' buffer, so a masked activation can never
     * be treated as public data by declaring it a weight operand. */
    std::string graph_install(const uint8_t *p, size_t n) {
        if (installed) VIOLATE("graph already installed; reconnect to replace");
        std::vector<Node> nn;
        try {
            return graph_install_nodes(p, n, nn);
        } catch (...) {
            /* Whatever this install put on the card comes off again and out
             * of the ledger: the violation closes the link, and the destructor
             * must not find these bytes charged twice or held at all. */
            for (auto &x : nn) if (x.w) { dfree(x.w); x.w = nullptr; }
            for (auto &x : nn) if (x.wbytes) { uncharge_device((long long)x.wbytes); x.wbytes = 0; }
            throw;
        }
    }
    std::string graph_install_nodes(const uint8_t *p, size_t n, std::vector<Node> &nn) {
        /* NUL-terminated copy: strtod reads to a non-digit, and a spec whose
         * last byte is a digit would otherwise read past the frame. */
        const std::string spec_s((const char *)p, n);
        JParser jp{ spec_s.data(), spec_s.data() + n };
        JVal spec = jp.parse();
        const JVal *jn = spec.get("nodes");
        if (!jn || jn->kind != JVal::ARR || jn->arr.empty()) VIOLATE("graph spec has no nodes");
        for (size_t i = 0; i < jn->arr.size(); i++) {
            const JVal &nd = jn->arr[i];
            const JVal *op = nd.get("op");
            const std::string ops = (op && op->kind == JVal::STR) ? op->str : "";
            for (auto &d : OP_DENYLIST) if (ops == d.first) VIOLATE("node %zu: op %s refused (%s)", i, ops.c_str(), d.second);
            bool allowed = false;
            for (auto a : OP_ALLOWLIST) if (ops == a) allowed = true;
            if (!allowed) VIOLATE("node %zu: op '%s' not in allowlist", i, ops.c_str());
            Node node;
            const JVal *id = nd.get("id");
            node.id = (id && id->kind == JVal::STR) ? id->str : fmt("node%zu", i);
            if (ops != "FIELD_GEMM") { nn.push_back(std::move(node)); continue; }   /* metadata-only */
            node.gemm = true;
            node.K = nd.i64("K"); node.N = nd.i64("N"); node.max_m = (int)nd.i64("max_m");
            if (node.K > kmax) kmax = node.K;
            if (node.K <= 0 || node.N <= 0 || node.max_m <= 0) VIOLATE("node %zu: non-positive shape", i);
            /* Bounded before N*K is formed: a spec can name any int64, and an
             * overflowed size reached std::vector as a length_error that took
             * the whole process down (fuzzed, 2026-08-26). Refuse, do not die. */
            if (node.K > (1 << 20) || node.N > (1 << 24) || node.K * node.N > cap())
                VIOLATE("node %zu: shape %lldx%lld exceeds the card", i, (long long)node.N, (long long)node.K);
            if (node.K % SH_QK) VIOLATE("node %zu: K=%lld is not a multiple of %d", i, (long long)node.K, SH_QK);
            if (node.max_m > 4096) VIOLATE("node %zu: max_m=%d exceeds 4096", i, node.max_m);
            const JVal *x = nd.get("x"), *y = nd.get("y");
            if (!x || !y) VIOLATE("node %zu: missing x/y binding", i);
            node.xbid = (uint64_t)x->i64("bid"); node.xoff = (uint64_t)x->i64("offset");
            node.ybid = (uint64_t)y->i64("bid"); node.yoff = (uint64_t)y->i64("offset");
            Buffer &xb = region_ok(node.xbid, node.xoff, (uint64_t)3 * node.max_m * node.K);
            Buffer &yb = region_ok(node.ybid, node.yoff, (uint64_t)node.max_m * node.N * 4);
            if (xb.role != "activations" || yb.role != "activations")
                VIOLATE("node %zu: x/y must bind an 'activations' buffer", i);
            if (node.xoff % 16 || node.yoff % 4) VIOLATE("node %zu: misaligned x/y offset", i);

            std::vector<int8_t> wfix((size_t)node.N * node.K);
            if (const JVal *w = nd.get("w")) {
                Buffer &wb = region_ok((uint64_t)w->i64("bid"), (uint64_t)w->i64("offset"), wfix.size());
                if (wb.role != "weights") VIOLATE("node %zu: w must bind a 'weights' buffer", i);
                memcpy(wfix.data(), wb.host.data() + (uint64_t)w->i64("offset"), wfix.size());
                for (size_t t = 0; t < wfix.size(); t++)
                    if (wfix[t] > SH_WEIGHT_BYTE_LIMIT || wfix[t] < -SH_WEIGHT_BYTE_LIMIT)
                        VIOLATE("node %zu: weight %d exceeds the int8 lane", i, (int)wfix[t]);
            } else {
                const JVal *wq = nd.get("wq"), *wd = nd.get("wd");
                if (!wq || !wd) VIOLATE("node %zu: missing weight binding", i);
                const int64_t nb = node.K / SH_QK;
                Buffer &qb = region_ok((uint64_t)wq->i64("bid"), (uint64_t)wq->i64("offset"), (uint64_t)node.K * node.N);
                Buffer &db = region_ok((uint64_t)wd->i64("bid"), (uint64_t)wd->i64("offset"), (uint64_t)nb * node.N * 2);
                if (qb.role != "weights" || db.role != "weights") VIOLATE("node %zu: wq/wd must bind a 'weights' buffer", i);
                if ((uint64_t)wd->i64("offset") % 2) VIOLATE("node %zu: misaligned wd offset", i);
                const int8_t *q = (const int8_t *)qb.host.data() + (uint64_t)wq->i64("offset");
                const uint16_t *d = (const uint16_t *)(db.host.data() + (uint64_t)wd->i64("offset"));
                /* THE shared encoding, run here by the same object the TEE links. */
                for (int64_t k = 0; k < node.K; k++)
                    for (int64_t j = 0; j < node.N; j++) {
                        const int64_t v = sh_encode_weight_fixed(d[(k / SH_QK) * node.N + j], q[k * node.N + j]);
                        if (v > SH_WEIGHT_BYTE_LIMIT || v < -SH_WEIGHT_BYTE_LIMIT)
                            VIOLATE("node %zu: fixed weight %lld exceeds the int8 lane", i, (long long)v);
                        wfix[(size_t)j * node.K + k] = (int8_t)v;
                    }
            }
            /* The node's device weights count against the link's cap like any
             * activations buffer. The charge lands in nn before the upload, so
             * a refusal further down this install frees and uncharges it. */
            charge_device((long long)wfix.size(), fmt("node %zu weights", i).c_str());
            node.wbytes = wfix.size();
            nn.push_back(std::move(node));
            Node &nw = nn.back();
            {
                std::lock_guard<std::mutex> lk(g_gpu);
                if (dmalloc((void **)&nw.w, wfix.size()) != cudaSuccess) {
                    nw.w = nullptr;
                    VIOLATE("node %zu: device allocation of %zu weight bytes failed", i, wfix.size());
                }
                ck(cudaMemcpy(nw.w, wfix.data(), wfix.size(), cudaMemcpyHostToDevice), "weight upload");
            }
        }
        bool any = false;
        for (auto &x : nn) any |= x.gemm;
        if (!any) VIOLATE("graph contains no computable node");
        const JVal *jo = spec.get("outputs");
        std::set<std::tuple<uint64_t, uint64_t, uint64_t>> outs;
        if (jo && jo->kind == JVal::ARR)
            for (auto &o : jo->arr) {
                const uint64_t bid = (uint64_t)o.i64("bid"), off = (uint64_t)o.i64("offset"), nb = (uint64_t)o.i64("nbytes");
                /* Outputs are products: they live in an activations buffer.
                 * A weights buffer's host copy is freed below; declaring it
                 * readable used to let GET_TENSOR walk a null pointer. */
                if (region_ok(bid, off, nb).role != "activations") VIOLATE("output must bind an 'activations' buffer");
                outs.insert(std::make_tuple(bid, off, nb));
            }
        if (outs.empty()) VIOLATE("graph declares no outputs; nothing could be read back");
        /* The weights buffers' host copies are no longer needed once every node
         * has its device-resident encoding. */
        for (auto &kv : buffers) if (!kv.second.dev && !kv.second.consumed) {
            std::vector<uint8_t>().swap(kv.second.host);
            kv.second.consumed = true; host_allocated -= (long long)kv.second.size;
        }
        nodes = std::move(nn); outputs = std::move(outs); installed = true;
        return fmt("{\"nodes\":%zu}", nodes.size());
    }

    Node &node_ok(uint32_t idx) {
        if (!installed) VIOLATE("RECOMPUTE with no installed graph");
        if (idx >= nodes.size()) VIOLATE("recompute of node %u, graph has %zu", idx, nodes.size());
        Node &nd = nodes[idx];
        if (!nd.gemm) VIOLATE("node %u is metadata-only; nothing to compute", idx);
        return nd;
    }

    /* Legacy doorbell: planes already SET_TENSOR'd into the node's x region. */
    std::string recompute(const uint8_t *p, size_t n) {
        if (n < 8) VIOLATE("truncated u32");
        const uint32_t idx = rd_u32(p), m = rd_u32(p + 4);
        Node &nd = node_ok(idx);
        if (m < 1 || (uint64_t)m > (uint64_t)nd.max_m) VIOLATE("m=%u outside [1,%d] for node %u", m, nd.max_m, idx);
        Buffer &xb = buffers[nd.xbid]; Buffer &yb = buffers[nd.ybid];
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(g_gpu);
            field_gemm_launch(nd.w, (int)nd.K, (int)nd.N, xb.dev + nd.xoff, (int)m,
                              nd.K, (long long)nd.max_m * nd.K,
                              (int32_t *)(yb.dev + nd.yoff), stream);
            ck(cudaStreamSynchronize(stream), "recompute sync");
        }
        gemm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        recomputes++;
        std::string r(4, '\0'); wr_u32((uint8_t *)&r[0], m);
        return r;
    }

    /* Record one exchange's card work as a graph: the planes upload, then per
     * pass of <= 8 rows one fused launch per <= 8 nodes. A failure mid-capture
     * ends the capture before the violation propagates, so the stream is left
     * usable for the reply. */
    cudaGraphExec_t capture_exchange(const uint8_t *planes, size_t xbytes, Node *const *nds, uint32_t nn,
                                     uint32_t m, int K, bool packed, PackMode pm) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        cudaGraph_t g = nullptr;
        try {
            ck(cudaMemcpyAsync(d_x, planes, xbytes, cudaMemcpyHostToDevice, stream), "planes upload");
            /* Where the GEMM writes and how wide. The packed reply's default
             * form has the epilogue write 3-byte values straight into the
             * mapped reply; PACK_KERNEL has it write int32 to device memory
             * and appends the pack; PACK_CPU is the FIELD_GEMM capture. */
            const bool epi = packed && pm == PACK_EPILOGUE;
            uint8_t *ybase = (packed && pm == PACK_KERNEL) ? (uint8_t *)d_y32 : (uint8_t *)d_out;
            const size_t yw = epi ? 3 : 4;
            const int8_t *Ws[GEMM_TAB_NODES]; uint8_t *Ys[GEMM_TAB_NODES]; int Ns[GEMM_TAB_NODES];
            size_t E = 0;
            for (uint32_t row0 = 0; row0 < m; row0 += 8) {
                const int mr = std::min((int)(m - row0), 8);
                size_t yoff = 0;
                for (uint32_t i0 = 0; i0 < nn; i0 += GEMM_TAB_NODES) {
                    const int cnt = std::min((int)(nn - i0), GEMM_TAB_NODES);
                    for (int i = 0; i < cnt; i++) {
                        const Node &nd = *nds[i0 + i];
                        Ws[i] = nd.w; Ns[i] = (int)nd.N;
                        Ys[i] = ybase + yoff + (size_t)row0 * nd.N * yw;
                        yoff += (size_t)m * nd.N * yw;
                    }
                    gemm_launch_planned(gemm_plan(Ws, Ys, Ns, cnt, mr, epi), K, d_x + (size_t)row0 * K,
                                        K, (long long)m * K, stream);
                }
                E = yoff / yw;
            }
            if (packed && pm == PACK_KERNEL) pack24_launch(d_y32, (uint8_t *)d_out, (long long)E, stream);
        } catch (...) {
            cudaStreamEndCapture(stream, &g);
            if (g) cudaGraphDestroy(g);
            throw;
        }
        ck(cudaStreamEndCapture(stream, &g), "end capture");
        cudaGraphExec_t ge = nullptr;
        const cudaError_t e = cudaGraphInstantiate(&ge, g, 0);
        cudaGraphDestroy(g);
        ck(e, "graph instantiate");
        return ge;
    }

    /* The one-frame exchange: | n u32 | m u32 | node u32[n] | planes int8[3][m][K] |
     * -> | y int32[m][N_0] | y int32[m][N_1] | ... |
     *
     * Every node must be an installed FIELD_GEMM with one common K; the planes
     * are the shared masked activation. The response is exactly the products of
     * the nodes named, and nothing else -- the same rule GET_TENSOR enforces
     * through declared outputs, here enforced by construction.
     *
     * The frame sits in h_in (pinned; serve() read it there, so there is no
     * memcpy). The card's work is ONE upload of the planes and ONE fused kernel
     * per pass of <= 8 rows and <= 8 nodes, writing the products into mapped
     * host memory (no D2H copy; the reply is sent from where they landed),
     * replayed as a captured graph. Measured on the 0.5B gate|up exchange
     * (K=896, 2 nodes, m=1, TCP loopback): 67 us round trip before, 42.5
     * after, of which 24 is the kernel and ~3.5 the upload and launch gaps. */
    void field_gemm(const uint8_t *p, size_t n, bool packed) {
        if (n < 8) VIOLATE("truncated u32");
        const uint32_t nn = rd_u32(p), m = rd_u32(p + 4);
        if (nn < 1 || nn > 64) VIOLATE("FIELD_GEMM names %u nodes", nn);
        if (n < 8 + 4 * (size_t)nn) VIOLATE("truncated node list");
        Node *nds[64];
        int64_t K = -1; size_t ybytes = 0;
        for (uint32_t i = 0; i < nn; i++) {
            Node &nd = node_ok(rd_u32(p + 8 + 4 * i));
            if (K < 0) K = nd.K;
            if (nd.K != K) VIOLATE("FIELD_GEMM nodes disagree on K");
            if (m < 1 || (uint64_t)m > (uint64_t)nd.max_m) VIOLATE("m=%u outside [1,%d] for node %s", m, nd.max_m, nd.id.c_str());
            nds[i] = &nd;
            ybytes += (size_t)m * nd.N * (packed ? 3 : 4);
        }
        const size_t xbytes = (size_t)3 * m * K;
        if (n != 8 + 4 * (size_t)nn + xbytes)
            VIOLATE("FIELD_GEMM payload is %zu bytes, expected %zu", n, 8 + 4 * (size_t)nn + xbytes);
        if (p != h_in) VIOLATE("internal: FIELD_GEMM frame not in pinned staging");
        const uint8_t *planes = h_in + 8 + 4 * (size_t)nn;

        /* FIELD_GEMM24 (protocol 1.2): 3 bytes per product. The values are
         * the same balanced (-M/2, M/2] integers, M < 2^24, so nothing is
         * lost and nothing new crosses; the reply is 25% smaller -- 152 KB on
         * the 0.5B's lm_head, where every byte is paid for on the guest's
         * vhost-vsock. */
        const PackMode pm = pack_mode();
        const size_t E = packed ? ybytes / 3 : 0;
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(g_gpu);
            ensure_dx(xbytes);
            /* The reply staging is rounded up to a word so pack24_kernel's
             * last word fits, and holds the int32 form for the CPU pack. */
            ensure_host_out(packed ? std::max((ybytes + 3) & ~(size_t)3, E * 4) : ybytes);
            if (packed && pm == PACK_KERNEL) ensure_dy32(E * 4);
            XP(0);
            std::vector<uint32_t> key(nn + 2); key[0] = packed ? (uint32_t)pm + 1 : 0; key[1] = m;
            for (uint32_t i = 0; i < nn; i++) key[i + 2] = rd_u32(p + 8 + 4 * i);
            auto git = graphs.find(key);
            if (git == graphs.end()) {
                if (graphs.size() >= 256) drop_graphs();
                git = graphs.emplace(key, capture_exchange(planes, xbytes, nds, nn, m, (int)K, packed, pm)).first;
            }
            ck(cudaGraphLaunch(git->second, stream), "graph launch");
            XP(3);
            ck(cudaStreamSynchronize(stream), "exchange sync");
            XP(5);
            if (packed && pm == PACK_CPU) {
                if (h_pack.size() < ybytes) h_pack.resize(ybytes);
                pack24_host((const int32_t *)h_out, h_pack.data(), (long long)E);
            }
            XP(6);
        }
        gemm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        exchanges++; XPN();
        resp_ptr = (packed && pm == PACK_CPU) ? (const void *)h_pack.data() : (const void *)h_out;
        resp_len = ybytes;
    }

    /* SHM_ATTACH: | ring u32 | -> | granted u8 | ring_bytes u64 | req_cap u64 | rep_cap u64 |.
     * Needs an installed graph (the ring carries FIELD_GEMM and nothing else),
     * one ring per connection, one connection per ring. A refusal is an
     * answer with granted = 0, not a violation: the link keeps the socket. */
    std::string shm_attach(const uint8_t *p, size_t n) {
        if (n < 4) VIOLATE("truncated u32");
        if (!installed) VIOLATE("SHM_ATTACH before GRAPH_INSTALL");
        if (ring) VIOLATE("duplicate SHM_ATTACH");
        const uint32_t index = rd_u32(p);
        bool granted = false;
        if (g_shm && index < (uint32_t)g_nrings) {
            int expect = 0;
            granted = g_ring_owner[index].compare_exchange_strong(expect, 1);
        }
        if (granted) {
            ring = g_shm + (size_t)index * RING_BYTES; ring_index = (int)index;
            ring_seen = ring_ld(ring + RING_OFF_REQ);      /* whatever is there is stale */
            logf("%s attached shm ring %u", peer.c_str(), index);
        }
        std::string r(25, '\0');
        r[0] = granted ? 1 : 0;
        wr_u64((uint8_t *)&r[1], RING_BYTES); wr_u64((uint8_t *)&r[9], RING_REQ_CAP); wr_u64((uint8_t *)&r[17], RING_REP_CAP);
        return r;
    }

    /* One ring request: snapshot the header, bound it, copy the frame into
     * the pinned staging, run the unchanged handler, publish the reply.
     * Returns false when the connection must close (a violation, as on the
     * socket: the reason is left in the reply slot with status 1). */
    bool service_ring(uint64_t seq) {
        uint8_t *r = ring;
        const uint8_t cmd = r[RING_OFF_RQH];
        const uint64_t size = rd_u64(r + RING_OFF_RQH + 1);
        std::string resp; bool violation = false;
        resp_ptr = nullptr; resp_len = 0;
        try {
            if (cmd != CMD_FIELD_GEMM) VIOLATE("ring carries command %u; only FIELD_GEMM rides the ring", cmd);
            const uint64_t cap = 8 + 4 * 64 + (uint64_t)3 * 4096 * (uint64_t)kmax;
            if (size > RING_REQ_CAP || size > cap)
                VIOLATE("ring frame of %llu bytes exceeds %llu", (unsigned long long)size, (unsigned long long)std::min<uint64_t>(cap, RING_REQ_CAP));
            ensure_host_in((size_t)size);
            memcpy(h_in, r + RING_OFF_RQP, (size_t)size);        /* the snapshot */
            resp = handle(cmd, h_in, (size_t)size);
            if (resp_len > RING_REP_CAP) VIOLATE("ring reply of %zu bytes exceeds %zu", resp_len, RING_REP_CAP);
        } catch (const Violation &v) {
            logf("VIOLATION from %s (ring): %s", peer.c_str(), v.why.c_str());
            resp = v.why; violation = true; resp_ptr = nullptr;
        } catch (const std::exception &e) {
            logf("VIOLATION from %s (ring): internal %s", peer.c_str(), e.what());
            resp = fmt("internal: %s", e.what()); violation = true; resp_ptr = nullptr;
        }
        const void *rp = resp_ptr ? resp_ptr : resp.data();
        size_t rl = resp_ptr ? resp_len : resp.size();
        if (rl > RING_REP_CAP) rl = RING_REP_CAP;
        if (rl) memcpy(r + RING_OFF_RPP, rp, rl);
        r[RING_OFF_RPH] = violation ? STATUS_VIOLATION : STATUS_OK;
        wr_u64(r + RING_OFF_RPH + 1, rl);
        ring_st(r + RING_OFF_REP, seq);
        ring_exchanges++;
        return !violation;
    }

    /* With a ring attached this thread serves BOTH: it spins on the ring's
     * request line and polls the socket (control frames, the fallback path)
     * every few spins; after g_shm_idle_ms without traffic it sleeps in 1 ms
     * polls. Returns with a socket header in h, or false to close. */
    bool next_header(uint8_t *h) {
        size_t got = 0;
        auto last = std::chrono::steady_clock::now();
        for (unsigned spins = 0;; spins++) {
            const uint64_t seq = ring_ld(ring + RING_OFF_REQ);
            if (seq != ring_seen) {
                ring_seen = seq;
                if (!service_ring(seq)) return false;
                last = std::chrono::steady_clock::now();
                continue;
            }
            if ((spins & 31) == 0) {
                ssize_t r = recv(fd, h + got, SH_HDR - got, MSG_DONTWAIT);
                if (r > 0) { got += (size_t)r; if (got == SH_HDR) return true; last = std::chrono::steady_clock::now(); continue; }
                if (r == 0) return false;
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return false;
            }
            _mm_pause();
            if ((spins & 4095) == 4095 &&
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last).count() > g_shm_idle_ms) {
                struct pollfd pfd = { fd, POLLIN, 0 };
                poll(&pfd, 1, 1);
            }
        }
    }

    std::string handle(uint8_t cmd, const uint8_t *p, size_t n) {
        if (cmd == CMD_HELLO) return hello(p, n);
        if (!hello_done) VIOLATE("command before HELLO");
        switch (cmd) {
            case CMD_ALLOC_BUFFER:    return alloc(p, n);
            case CMD_FREE_BUFFER:     return free_buf(p, n);
            case CMD_SET_TENSOR:      return set_tensor(p, n);
            case CMD_GET_TENSOR:      return get_tensor(p, n);
            case CMD_GRAPH_INSTALL:   return graph_install(p, n);
            case CMD_GRAPH_RECOMPUTE: return recompute(p, n);
            case CMD_FIELD_GEMM:      field_gemm(p, n, false); return std::string();
            case CMD_FIELD_GEMM24:    field_gemm(p, n, true);  return std::string();
            case CMD_SHM_ATTACH:      return shm_attach(p, n);
            case CMD_BUFFER_GET_BASE: case CMD_GET_ALIGNMENT: case CMD_GET_MAX_SIZE:
            case CMD_GET_DEVICE_MEMORY: case CMD_DEVICE_COUNT:
                return "";
            default: VIOLATE("unhandled command %u", cmd);
        }
    }

    void serve() {
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
        std::vector<uint8_t> buf;
        for (;;) {
            uint8_t h[SH_HDR];
            if (ring ? !next_header(h) : !read_exact(fd, h, SH_HDR)) break;
            const uint8_t cmd = h[0];
            const uint64_t size = rd_u64(h + 1);
            std::string resp; bool violation = false;
            if (cmd >= CMD_COUNT || size > MAX_FRAME) {
                resp = cmd >= CMD_COUNT ? fmt("unknown command %u", cmd) : fmt("frame of %llu bytes exceeds cap", (unsigned long long)size);
                violation = true;
                /* drain nothing: the peer is closed on immediately after the reply */
            } else {
                uint8_t *body = nullptr;
                if (cmd == CMD_FIELD_GEMM || cmd == CMD_FIELD_GEMM24) {
                    /* Straight into pinned staging: the planes go to the card
                     * from here. Bounded by what an installed graph can ask
                     * for (64 nodes, 4096 rows of the widest K) so a 9-byte
                     * header cannot pin 256 MB of host RAM. */
                    const uint64_t cap = installed ? 8 + 4 * 64 + (uint64_t)3 * 4096 * (uint64_t)kmax : (uint64_t)16 << 20;
                    try {
                        if (size > cap) VIOLATE("FIELD_GEMM frame of %llu bytes exceeds %llu", (unsigned long long)size, (unsigned long long)cap);
                        ensure_host_in((size_t)size);
                        body = h_in;
                    } catch (const Violation &v) {
                        logf("VIOLATION from %s: %s", peer.c_str(), v.why.c_str());
                        resp = v.why; violation = true;           /* replied with a reason, then closed */
                    }
                } else {
                    buf.resize((size_t)size);
                    body = buf.data();
                }
                resp_ptr = nullptr; resp_len = 0;
                if (violation) { /* nothing was read; the reply below names why */ }
                else if (size && !read_exact(fd, body, (size_t)size)) break;
                else try {
                    resp = handle(cmd, body, (size_t)size);
                } catch (const Violation &v) {
                    logf("VIOLATION from %s: %s", peer.c_str(), v.why.c_str());
                    resp = v.why; violation = true; resp_ptr = nullptr;
                } catch (const std::exception &e) {
                    /* bad_alloc, length_error, ...: the frame was malformed in a
                     * way the checks above did not name. Same outcome as a
                     * violation -- refuse and close -- never std::terminate,
                     * which is a DoS on every other tenant of this card. */
                    logf("VIOLATION from %s: internal %s", peer.c_str(), e.what());
                    resp = fmt("internal: %s", e.what()); violation = true; resp_ptr = nullptr;
                }
            }
            const void *rp = resp_ptr ? resp_ptr : resp.data();
            const size_t rl = resp_ptr ? resp_len : resp.size();
            uint8_t rh[SH_HDR]; rh[0] = violation ? STATUS_VIOLATION : STATUS_OK; wr_u64(rh + 1, rl);
            if (cmd == CMD_FIELD_GEMM || cmd == CMD_FIELD_GEMM24) XP(0);
            /* Header and body in one writev: one syscall, one segment on the wire. */
            if (!writev_all(fd, rh, SH_HDR, rp, rl)) break;
            if (cmd == CMD_FIELD_GEMM || cmd == CMD_FIELD_GEMM24) XP(7);
            if (violation) break;
        }
        close(fd);
#ifdef SH_XPROF
        if (g_xpn) { fprintf(stderr, "[xprof] n=%llu memcpy %.1f h2d-call %.1f launches %.1f d2h-call %.1f sync %.1f pack %.1f send %.1f us/exchange\n",
            (unsigned long long)g_xpn, g_xp[1]/g_xpn, g_xp[2]/g_xpn, g_xp[3]/g_xpn, g_xp[4]/g_xpn, g_xp[5]/g_xpn, g_xp[6]/g_xpn, g_xp[7]/g_xpn); memset(g_xp,0,sizeof g_xp); g_xpn=0; }
#endif
        if (exchanges || recomputes)
            logf("%s closed: %llu exchanges (%llu over the ring), %llu recomputes, %.1f ms on the card",
                 peer.c_str(), (unsigned long long)exchanges, (unsigned long long)ring_exchanges, (unsigned long long)recomputes, gemm_ms);
    }
};

/* ---------------------------------------------------------------------------
 * Listeners: TCP (a slirp guest reaches 127.0.0.1 at 10.0.2.2) and, when asked,
 * AF_VSOCK -- the host is CID 2 to any guest, and a vsock round trip is a
 * fraction of slirp's, which at ~50 exchanges per token is the difference
 * between transport being a rounding error and being the budget.
 * ------------------------------------------------------------------------ */
static void accept_loop(int srv, const char *kind) {
    for (;;) {
        sockaddr_storage sa; socklen_t sl = sizeof sa;
        int fd = accept(srv, (sockaddr *)&sa, &sl);
        if (fd < 0) { if (errno == EINTR) continue; logf("%s accept failed: %s", kind, strerror(errno)); continue; }
        std::string peer = kind;
        if (sa.ss_family == AF_INET) {
            char ip[64]; inet_ntop(AF_INET, &((sockaddr_in *)&sa)->sin_addr, ip, sizeof ip);
            peer = fmt("%s:%d", ip, ntohs(((sockaddr_in *)&sa)->sin_port));
            int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        } else if (sa.ss_family == AF_VSOCK) {
            peer = fmt("vsock:%u:%u", ((sockaddr_vm *)&sa)->svm_cid, ((sockaddr_vm *)&sa)->svm_port);
        }
        std::thread([fd, peer]() {
            auto c = std::make_unique<Conn>(fd, peer);
            c->serve();
        }).detach();
    }
}

/* Knobs from a file beside the binary, applied as environment defaults.
 *
 * The launcher spawns the worker with its own fixed environment, and changing
 * that means restarting the launcher, which means restarting the CVM and its
 * tenants (and, on a metal box, re-issuing every certificate). A worker
 * restart alone costs a tenant one reconnect. So the tunables that only the
 * host side needs -- the read spin, for one -- live in worker.conf next to the
 * binary: KEY=VALUE per line, '#' comments, environment wins over the file.
 * Nothing here reaches the TEE; the file is host configuration of the host's
 * own untrusted half. */
static void load_conf_beside_binary(void) {
    char exe[4096]; ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return;
    exe[n] = 0;
    char *slash = strrchr(exe, '/');
    if (!slash) return;
    std::string path(exe, (size_t)(slash - exe));
    path += "/worker.conf";
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || !*p) continue;
        char *eq = strchr(p, '=');
        if (!eq) continue;
        *eq = 0;
        char *v = eq + 1; char *e = v + strlen(v);
        while (e > v && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ')) *--e = 0;
        if (setenv(p, v, 0) == 0 && !getenv("SHIELDED_WORKER_QUIET_CONF")) logf("worker.conf: %s=%s", p, v);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    load_conf_beside_binary();
    const char *host = "127.0.0.1";
    int port = getenv("SHIELDED_PORT") ? atoi(getenv("SHIELDED_PORT")) : 9500;
    int vsock_port = 0;
    double vram_gb = 0.0;
    const char *shm_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vsock-port") && i + 1 < argc) vsock_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vram-gb") && i + 1 < argc) vram_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--shm") && i + 1 < argc) shm_path = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) g_quiet = true;
#ifdef SH_XPROF
        else if (!strcmp(argv[i], "--kbench")) { cudaSetDevice(0); cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost); return kbench(); }
#endif
        else { fprintf(stderr, "usage: shielded-worker [--host H] [--port P] [--vsock-port P] [--vram-gb G] [--shm FILE] [--quiet]\n"); return 2; }
    }
    if (shm_path) {
        /* The launcher creates and sizes the file (it is also the ivshmem
         * backing store of the CVM); this side only maps what exists. */
        if (const char *e = getenv("SHIELDED_SHM_IDLE_MS")) g_shm_idle_ms = std::max(1, atoi(e));
        int sfd = open(shm_path, O_RDWR | O_CLOEXEC);
        struct stat st{};
        if (sfd < 0 || fstat(sfd, &st) != 0) { fprintf(stderr, "--shm %s: %s\n", shm_path, strerror(errno)); return 2; }
        size_t len = std::min<size_t>((size_t)st.st_size, RING_MAX_FILE) / RING_BYTES * RING_BYTES;
        if (len < RING_BYTES) { fprintf(stderr, "--shm %s: %lld bytes, needs at least %zu\n", shm_path, (long long)st.st_size, RING_BYTES); return 2; }
        void *m = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
        close(sfd);
        if (m == MAP_FAILED) { fprintf(stderr, "--shm mmap %s: %s\n", shm_path, strerror(errno)); return 2; }
        g_shm = (uint8_t *)m; g_shm_len = len; g_nrings = (int)(len / RING_BYTES);
        logf("shm ring file %s: %d ring(s) of %zu MiB (request slot %zu KiB, reply slot %zu MiB)",
             shm_path, g_nrings, RING_BYTES >> 20, RING_REQ_CAP >> 10, RING_REP_CAP >> 20);
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        fprintf(stderr, "no CUDA device; the shielded worker is the GPU half by definition\n"); return 1;
    }
    cudaSetDevice(0);
    /* Spin on synchronisation: a blocking wait costs tens of microseconds to
     * wake, which at decode is a large fraction of the whole exchange. */
    cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost);
    cudaGetDeviceProperties(&g_props, 0);
    g_sm_count = std::max(1, g_props.multiProcessorCount);
    g_vram_budget = vram_gb > 0 ? (long long)(vram_gb * (double)(1ull << 30)) : (long long)(g_props.totalGlobalMem * 0.85);
    logf("%s, sm_%d%d, %.1f GiB total, budget %.1f GiB, %d SMs @ %.2f GHz -> %.1f TFLOPS fp16 rated",
         g_props.name, g_props.major, g_props.minor,
         g_props.totalGlobalMem / 1073741824.0, g_vram_budget / 1073741824.0,
         g_props.multiProcessorCount, g_props.clockRate / 1e6, rated_fp16_tflops(g_props));
    /* Nothing is claimed here: tenants reserve at HELLO (see g_reserved). A
     * card smaller than the budget can never honour it -- misconfiguration,
     * exit 75 so the launcher's retry surfaces it -- while a card with less
     * free than the budget right now may be held by something that leaves. */
    if ((long long)g_props.totalGlobalMem < g_vram_budget) {
        fprintf(stderr, "[shielded-worker] the budget of %.1f GiB exceeds the card: %s has %.1f GiB in total. "
                        "Lower --vram-gb (metal/config.json shieldedWorker.vramGb); exiting 75.\n",
                g_vram_budget / 1073741824.0, g_props.name, g_props.totalGlobalMem / 1073741824.0);
        return 75;
    }
    {
        size_t freeb = 0, totalb = 0; cudaMemGetInfo(&freeb, &totalb);
        if ((long long)freeb < g_vram_budget)
            logf("warning: the card has %.1f GiB free against a %.1f GiB budget; reservations beyond what is free will be refused until it is",
                 freeb / 1073741824.0, g_vram_budget / 1073741824.0);
    }
    pool_retune();                       /* threshold 0: the selftest's scratch goes back to the driver */
    try {
        if (!selftest()) return 1;
    } catch (const Violation &v) { fprintf(stderr, "selftest: %s\n", v.why.c_str()); return 1; }
    g_gmacs = measure_gmacs(); g_gmacs_at = std::chrono::steady_clock::now();
    /* The self-test and the benchmark leave their scratch cached in the pool;
     * with nothing reserved that cache is memory the fleet counts as free and
     * a neighbour cannot get (201 MB observed idle, 2026-08-27). Hand it back. */
    { std::lock_guard<std::mutex> lk(g_gpu); pool_trim(); }
    if (g_gmacs > 0) logf("field GEMM throughput %.0f G-MAC/s (measured, masked path)", g_gmacs);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) { fprintf(stderr, "bad host %s\n", host); return 2; }
    if (bind(srv, (sockaddr *)&sa, sizeof sa) < 0 || listen(srv, 8) < 0) { perror("bind"); return 1; }
    logf("listening on %s:%d%s", host, port,
         (!strcmp(host, "127.0.0.1") || !strcmp(host, "0.0.0.0")) ? fmt(" (guest reaches it at 10.0.2.2:%d)", port).c_str() : "");

    if (vsock_port > 0) {
        int vs = socket(AF_VSOCK, SOCK_STREAM, 0);
        if (vs < 0) { logf("vsock unavailable (%s); TCP only", strerror(errno)); }
        else {
            sockaddr_vm vm{}; vm.svm_family = AF_VSOCK; vm.svm_cid = VMADDR_CID_ANY; vm.svm_port = (unsigned)vsock_port;
            if (bind(vs, (sockaddr *)&vm, sizeof vm) < 0 || listen(vs, 8) < 0) {
                logf("vsock bind failed (%s); TCP only", strerror(errno)); close(vs);
            } else {
                logf("listening on vsock port %d (guest reaches it at CID 2)", vsock_port);
                std::thread(accept_loop, vs, "vsock").detach();
            }
        }
    }
    accept_loop(srv, "tcp");
    return 0;
}
