/*
 * shielded-wire.h -- the socket half of the shielded protocol, in C.
 *
 * Mirrors shielded/wire.py. Request framing is | cmd u8 | size u64 LE | payload |,
 * response framing is | status u8 | size u64 LE | payload |, and status 1 means
 * the worker refused: it is always the last frame on that connection.
 *
 * Three failure modes have to stay distinguishable, which is the whole reason the
 * response carries a status byte rather than just closing:
 *   SH_ERR_VIOLATION  the worker refused us -- OUR protocol bug, loud in tests
 *   SH_ERR_IO         the socket died -- a liveness event, retryable
 *   (a lying worker)  not visible here at all; Freivalds in shielded-tee catches it
 * Collapsing them, as stock ggml-rpc does, makes the first two indistinguishable
 * in production, and the tier rests on telling them apart.
 */
#ifndef SHIELDED_WIRE_H
#define SHIELDED_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SH_CMD_HELLO            0
#define SH_CMD_ALLOC_BUFFER     1
#define SH_CMD_FREE_BUFFER      2
#define SH_CMD_SET_TENSOR       8
#define SH_CMD_GET_TENSOR       9
#define SH_CMD_GRAPH_INSTALL   10
#define SH_CMD_GRAPH_RECOMPUTE 11
/* The one-frame exchange: | n u32 | m u32 | node u32[n] | planes int8[3][m][K] |
 * answered by the products of exactly those nodes, | y int32[m][N_i] |...
 * A decode exchange is one write and one read instead of five of each, and
 * the batch is whatever m is -- no power-of-two bucketing, because the reply
 * is defined by the request rather than matched against a declared region. */
#define SH_CMD_FIELD_GEMM      12
/* Protocol 1.2: the SAME request as FIELD_GEMM, answered with 3-byte little-
 * endian two's-complement values, | y int24[m][N_i] |... -- 3*m*N_i bytes per
 * node. Every product is balanced in (-M/2, M/2] with M < 2^24, so the value
 * is the same; only the reply shrinks (25%, 152 KB on the 0.5B's lm_head).
 * Sent only to a worker whose HELLO says minor >= 2. */
#define SH_CMD_FIELD_GEMM24    13
/* Protocol 1.2: bind a shared-memory ring (SHIELDED_SHM) to this connection.
 * | ring u32 | -> | granted u8 | ring_bytes u64 | req_cap u64 | rep_cap u64 |.
 * granted == 0 is an answer, not a violation: the link keeps the socket. See
 * sh_pipe_shm_attach below and shielded/PROTOCOL.md "shm ring". */
#define SH_CMD_SHM_ATTACH      14

#define SH_OK             0
#define SH_ERR_IO        -1
#define SH_ERR_VIOLATION -2
#define SH_ERR_PROTO     -3
#define SH_ERR_NOMEM     -4

typedef struct sh_pipe sh_pipe;

/* One outbound frame in a pipelined batch. Both segments are borrowed, never
 * copied. The second exists for SET_TENSOR, whose payload is a fixed header
 * followed by tensor bytes: without it every masked activation would be
 * memcpy'd into a staging buffer once per node per token, which at decode is
 * pure overhead against a payload that is already the right bytes in the right
 * order. Frame size is len + len2; leave payload2 NULL for everything else. */
typedef struct {
    uint8_t      cmd;
    const void  *payload;
    size_t       len;
    const void  *payload2;
    size_t       len2;
} sh_frame;

/* One inbound response. `data` points into a buffer the PIPE owns and reuses
 * on the next exchange: read it before exchanging again, and never free it.
 * sh_reply_free only clears the view, so existing call sites stay correct.
 * The buffer grows monotonically to the largest reply seen and dies with the
 * pipe -- which is what makes a decode exchange allocation-free. */
typedef struct {
    uint8_t  status;
    uint8_t *data;
    size_t   len;
} sh_reply;

sh_pipe *sh_pipe_open(const char *host, int port, int *err);
void     sh_pipe_close(sh_pipe *p);

/* The last violation reason the worker sent, or "" -- diagnostics only. */
const char *sh_pipe_last_error(const sh_pipe *p);

/* Write every frame in ONE writev and then read all n responses. This is what
 * makes a masked exchange (SET_TENSOR, RECOMPUTE, GET_TENSOR) cost one RTT
 * instead of three; at 32 layers that is the difference between transport being
 * a rounding error and the second-largest term in the token budget.
 * Replies borrow the pipe's buffer (see sh_reply); a batch of up to 16 frames
 * allocates nothing. */
int sh_pipe_exchange(sh_pipe *p, const sh_frame *frames, size_t n, sh_reply *out);

/* Convenience: one frame, one response. */
int sh_pipe_call(sh_pipe *p, uint8_t cmd, const void *payload, size_t len, sh_reply *out);

void sh_reply_free(sh_reply *r);

/* --- the shared-memory ring (SHIELDED_SHM) -----------------------------------
 * In the CVM an exchange over vhost-vsock costs 152 us, of which the socket is
 * ~10 and the card ~28: the rest is two virtqueue kicks (VM exits) and two
 * wakeups, the guest's from HLT via an injected interrupt (REPORT.md 13.13).
 * A ring in memory both sides map and POLL pays none of that: measured in an
 * SEV-SNP guest on an ivshmem BAR mapped write-back + decrypted, the 0.5B
 * gate|up exchange's transport is 0.97 us (scratchpad/shm-ring/DESIGN.md).
 *
 * Layout of one ring (SH_RING_BYTES, ring i of a file at i * SH_RING_BYTES):
 *   0     u64 req_seq   TEE -> worker, its own cache line, written LAST (release)
 *   64    u64 rep_seq   worker -> TEE, own cache line, written LAST (release)
 *   128   | cmd u8 | len u64 |      the socket request header, byte for byte
 *   192   | status u8 | len u64 |   the socket response header, byte for byte
 *   4096  request payload (SH_RING_REQ_CAP)
 *   2 MiB reply payload   (SH_RING_REP_CAP)
 * The TEE writes ONLY what the socket frame carries (the masked planes and the
 * FIELD_GEMM header), and reads a reply only after rep_seq matches the sequence
 * IT chose and the peer's len equals the length IT expects. The host may write
 * anything: every such failure is handled as a hostile socket peer would be
 * (refuse; fall back to the socket; a bad status takes the link down). */
#define SH_RING_BYTES     ((size_t)8 << 20)
#define SH_RING_OFF_REQ   0
#define SH_RING_OFF_REP   64
#define SH_RING_OFF_RQH   128
#define SH_RING_OFF_RPH   192
#define SH_RING_OFF_RQP   4096
#define SH_RING_OFF_RPP   ((size_t)2 << 20)
#define SH_RING_REQ_CAP   (SH_RING_OFF_RPP - SH_RING_OFF_RQP)
#define SH_RING_REP_CAP   (SH_RING_BYTES - SH_RING_OFF_RPP)
#define SH_RING_MAX_FILE  ((size_t)64 << 20)

/* Map `path` (a file, /dev/shmring, or a PCI resource2_wc) and bind ring
 * `index` of it to this connection with SHM_ATTACH. `bytes` is used when the
 * node reports no size (a char device); 0 = fstat. Returns SH_OK with the ring
 * live, SH_ERR_IO when the ring cannot be used (the pipe stays valid on the
 * socket), SH_ERR_VIOLATION/PROTO when the worker answered nonsense. */
int  sh_pipe_shm_attach(sh_pipe *p, const char *path, int index, size_t bytes);
/* First available ring, bounded by the mapping and compiled geometry. */
int  sh_pipe_shm_attach_available(sh_pipe *p, const char *path, size_t bytes, int *index);
int  sh_pipe_ring_live(const sh_pipe *p);
/* One FIELD_GEMM frame over the ring. `want` is the reply length the CALLER
 * expects; a reply of any other length is refused before a byte of it is read.
 * SH_ERR_IO = the ring did not carry it (busy, timed out, not live): send the
 * same frame on the socket. After 3 consecutive misses the ring is disabled. */
int  sh_pipe_ring_exchange(sh_pipe *p, const sh_frame *f, size_t want, sh_reply *out);

/* Payload builders. Each writes little-endian into `dst` and returns the length.
 * Buffers are caller-provided so the hot path allocates nothing. */
/* HELLO: u32 major, then (protocol 1.3) an optional u64 reserve_bytes -- the
 * device memory this connection asks the worker to hold for it, out of the
 * worker's budget, until it disconnects. 0 packs the old 4-byte form, which
 * every worker accepts and which reserves nothing (the per-connection cap is
 * then the whole budget, as before 1.3). Returns 4 or 12. */
size_t sh_pack_hello(void *dst, uint32_t major, uint64_t reserve_bytes);
size_t sh_pack_alloc(void *dst, uint64_t size, const char *role);
size_t sh_pack_region(void *dst, uint64_t bid, uint64_t offset, uint64_t nbytes);
size_t sh_pack_set_tensor_header(void *dst, uint64_t bid, uint64_t offset, uint64_t nbytes);
size_t sh_pack_recompute(void *dst, uint32_t node, uint32_t m);
/* Header of a FIELD_GEMM frame; the planes follow as payload2. Returns 8 + 4n. */
size_t sh_pack_field_gemm(void *dst, uint32_t n_nodes, uint32_t m, const int *nodes);

#ifdef __cplusplus
}
#endif
#endif
