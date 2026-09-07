/*
 * shielded-tee.h -- the trusted half, in C: pads, refill, Freivalds, and the
 * link that drives one shielded worker.
 *
 * Mirrors shielded/tee.py. What the worker sees is public weights and masked
 * activations; what never crosses is the pad, the Freivalds secret, or any
 * plaintext activation. The four rules most easily broken by accident, all of
 * which this file is responsible for:
 *
 *  1. PADS ARE ONE-TIME. Issuance is a monotonic counter that STALLS at capacity
 *     and never wraps. Two activations under one pad hand the adversary their
 *     difference, and successive decode activations differ by very little.
 *  2. ONE PLAINTEXT GETS ONE PAD. Weights fed by the same activation (q/k/v from
 *     one attn_norm, gate/up from one ffn_norm) share the pad, because masking
 *     one x three times under three pads is three encryptions of one value for
 *     no benefit. Such weights form a GROUP and are exchanged together.
 *  3. VERIFY BEFORE USE, over the INTEGERS. A product that exceeds M/2 wraps and
 *     stays congruent mod M, so a mod-M check accepts it and it decodes to
 *     garbage with no error signal. The check runs modulo an unrelated prime and
 *     catches a lying worker AND a field wrap in the same two dot products.
 *  4. THE FREIVALDS SECRET COMES FROM THE OS CSPRNG. A worker that can predict s
 *     solves d.s == 0 over three outputs and forges freely.
 *
 * THE POOL. u = r.W is the one term that can never be offloaded, and at three
 * residue planes it costs the TEE three times the work of the matmul it
 * unmasks. It is also independent of the activation, so it is computed AHEAD of
 * the request: background threads keep a pool of (r, r.W) pairs per group, and
 * the exchange itself only masks, sends, unmasks and verifies. Refill is a
 * throughput term on spare cores, not a latency term on the decode chain.
 */
#ifndef SHIELDED_TEE_H
#define SHIELDED_TEE_H

#include <stdbool.h>
#include "shielded-pads.h"
#include <stddef.h>
#include <stdint.h>

#include "shielded-wire.h"
#include "shielded-simd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SH_FV_S_RANGE (1 << 20)
#define SH_FV_REPS    2            /* ~2^-40 soundness per check */
#define SH_GROUP_MAX  8            /* weights sharing one activation */

typedef struct sh_link sh_link;

/* Errors beyond the wire's. */
#define SH_ERR_VERIFY   -10   /* worker lied, or the field wrapped */
#define SH_ERR_EXHAUST  -11   /* pad bank dry: stall the request, never wrap */
#define SH_ERR_RANGE    -12   /* weights do not fit the int8 lane */

/* Tunables read from the environment at open:
 *   SHIELDED_REFILL_THREADS  background refill threads; unset = derived from the
 *                            registered weights at start (see derive_threads in
 *                            shielded-tee.c), clamped to [2, ncores/2]
 *   SHIELDED_REFILL_TARGET_MS the token time the derived count must keep up
 *                            with (default 6, the measured 0.5B decode)
 *   SHIELDED_POOL_DEPTH      ready pads kept per group (default 16)
 *   SHIELDED_REFILL_BATCH    pads generated per batch (default 4)
 *   SHIELDED_WARM_MS         longest sh_link_start waits for one ready pad per
 *                            group before the first exchange (default 5000)
 * and at start:
 *   SHIELDED_VSOCK_PORT      try AF_VSOCK to the host on this port before TCP;
 *                            unset = the TCP port number when /dev/vsock exists, 0 = never */
/* Dealt pads (shielded-pads.h, shielded/dealer/PLAN.md), all read at open and
 * required TOGETHER - a partial set refuses to open rather than minting:
 *   SHIELDED_PAD_SOURCE      directory of .pads shipments (the operator's bank)
 *   SHIELDED_PAD_SEED        64 hex: the 32-byte seed r is derived from
 *   SHIELDED_PAD_SEED_ID     32 hex: names the seed; shipments carry it
 *   SHIELDED_PAD_SK          64 hex: this consumer's X25519 pad secret key
 *   SHIELDED_PAD_LEDGER      path of the ledger file (P1: local; P2: platform)
 *   SHIELDED_PAD_MODEL_DIGEST optional 64 hex: only shipments minted for this
 *                            model (the digest the dealer printed) are used
 *   SHIELDED_PAD_CHECK       1 = verify every imported pad against the weights
 *                            (Freivalds mod M; one weight pass per node at
 *                            registration): a wrong dealer is refused at
 *                            import, before any use
 *   SHIELDED_PAD_WINDOW      indices reserved per ledger call (default 64)
 *   SHIELDED_PAD_WAIT_MS     how long a refill waits for a missing index
 *                            before the link stops (default 10000)
 *   SHIELDED_PAD_SOURCE=http://host:port/v1/pads/shipments
 *                            a bank instead of a directory (shielded-bank.c):
 *                            shipments are fetched into SHIELDED_PAD_CACHE
 *                            (default /tmp/shielded-pads-<seed_id>), at most
 *                            SHIELDED_PAD_CACHE_MB (2048) ahead of the window
 *   SHIELDED_PAD_PRUNE       unlink spent shipments in the source directory
 *                            (default 1 for a bank cache, 0 for a directory)
 *   SHIELDED_PAD_WINDOW_URL  an http:// window agent answering
 *                            POST {want, seed_id} with {lo, hi, iat, sig};
 *                            requires SHIELDED_PAD_LEDGER_PK (64 hex, the
 *                            platform's Ed25519 ledger key) and replaces the
 *                            ledger file, like the pVM's provider does */
sh_link *sh_link_open(const char *host, int port, bool verify, int *err);
/* Explicit per-link transport/reservation for a pooled tenant. Call before start;
 * avoids mutating process-wide environment while other links refill/reconnect. */
void sh_link_configure(sh_link *l, int vsock_port, uint64_t reserve_bytes, int refill_threads);
/* Per-card optional BAR, before start. Empty path explicitly keeps sockets. */
void sh_link_configure_shm(sh_link *l, const char *path, uint64_t bytes);
void     sh_link_close(sh_link *l);
const char *sh_link_last_error(const sh_link *l);

/* Register one weight as a FIELD_GEMM node, before connecting.
 * `w_fixed` is (N,K) int8 -- THE encoding, one row per output, borrowed for the
 * life of the link. `share_x_with` is an earlier node fed by the SAME
 * activation, or -1. Returns the node index, or negative on error. */
int sh_link_add_weight(sh_link *l, const char *name, const int8_t *w_fixed,
                       int64_t K, int64_t N, int32_t max_m, int share_x_with);

/* Ship the public weights, install the vetted graph, start the refill threads.
 * Restartable: a weight added after start means a fresh connection carrying
 * the whole set. */
int sh_link_start(sh_link *l);

/* One masked exchange: every node in `nodes` must belong to one group (share
 * the activation). `x_field` is the PLAINTEXT field-encoded activation, (m,K)
 * int64, balanced. `y_out[i]` receives (m,N_i) int64, exact and verified.
 * On any failure the result is SH_ERR_VERIFY and y_out must be DISCARDED by
 * the caller: rows of already-checked nodes, and the unmasked rows of the
 * failing one, may have been written before the check fired. */
int sh_link_gemm(sh_link *l, const int *nodes, size_t n_nodes,
                 const int64_t *x_field, int32_t m, int64_t **y_out);

/* The same product, computed in the TEE in plain int64, with no worker
 * involved. Numerically IDENTICAL to sh_link_gemm -- the offloaded path is
 * exact -- which is what makes it a fallback rather than a degraded mode. */
int sh_link_gemm_local(sh_link *l, const int *nodes, size_t n_nodes,
                       const int64_t *x_field, int32_t m, int64_t **y_out);

/* True once the worker is connected and the graph installed. */
bool sh_link_is_live(const sh_link *l);
/* Dealt mode: the ring is fed from shipments and the link never mints; an
 * EXHAUST from sh_link_gemm then means the bank is behind and the caller
 * must refuse to proceed rather than compute in the clear. */
bool sh_link_is_dealt(const sh_link *l);

/* The encoded weight, (N,K), for the caller's TEE-side outlier term. */
const int8_t *sh_link_weight(const sh_link *l, int node);

/* Run the integrity check directly on a candidate (x, y) pair. Exposed so the
 * probe can assert BOTH directions against the same code the online path uses. */
bool sh_link_verify(const sh_link *l, int node, const int64_t *x, const int64_t *y, int32_t m);

/* Counters, for the probe and for tests. */
void sh_link_stats(const sh_link *l, uint64_t *exchanges, uint64_t *macs, uint64_t *verify_fail);

/* Dealt pads. The group table this link would bind a shipment against
 * (ordinal, first node's name, K, u_len); returns the group count. */
int sh_link_group_table(const sh_link *l, sh_pads_group *out, uint32_t cap);
/* The dealer's mint: with the model's weights registered (no worker needed),
 * write one shipment of `count` indices from `index0` for every group, r from
 * `seed`, cells boxed to `consumer_pk`. */
/* Ledger windows from the host instead of SHIELDED_PAD_LEDGER: the pVM's owner
 * app asks the platform (relay/pads.mjs) and the payload verifies the signed
 * window before answering. Must return SH_OK with lo == the previous hi. */
typedef int (*sh_window_fn)(void *ctx, uint64_t want, uint64_t *lo, uint64_t *hi);
void sh_link_set_window_provider(sh_link *l, sh_window_fn fn, void *ctx);
int sh_link_dealt_selftest(sh_link *l, int rows, int32_t *r_out, int32_t *u_out);
/* The same shipment minted through the link's worker (needs SHIELDED_ZERO_PADS=1). */
int sh_link_mint_shipment_worker(sh_link *l, const uint8_t seed[32], const uint8_t seed_id[16], const uint8_t model_digest[32],
                                 uint64_t index0, uint64_t count, const uint8_t consumer_pk[32], const char *path);
int sh_link_mint_shipment(sh_link *l, const uint8_t seed[32], const uint8_t seed_id[16], const uint8_t model_digest[32],
                          uint64_t index0, uint64_t count, const uint8_t consumer_pk[32], const char *path);
/* Pool health: pads consumed, and how many had to be generated on the
 * request path because the pool was dry (the number that should be ~0). */
void sh_link_pool_stats(const sh_link *l, uint64_t *consumed, uint64_t *missed);
/* Optional wait for already-reserved pads: pads obtained and total wait time. */
void sh_link_pad_wait_stats(const sh_link *l, uint64_t *waited, double *wait_ms);
/* The same counters for this node's shared-activation group, plus time spent
 * generating missing pads on the request path. Public weight metadata only. */
void sh_link_node_pool_stats(const sh_link *l, int node, uint64_t *consumed,
                             uint64_t *missed, double *on_path_ms);
/* Refill threads actually running (derived or from the environment), for logs. */
int  sh_link_refill_threads(const sh_link *l);
const char *sh_link_refill_priority(const sh_link *l);
/* Bytes per reply value after start: 4 (FIELD_GEMM) or 3 (FIELD_GEMM24); 0 before. */
int  sh_link_reply_width(const sh_link *l);

/* The wall time of the last exchange's wire phase (send, wait, receive), in
 * microseconds: what the backend watches to notice a card that is being
 * shared with something else. */
double sh_link_last_wire_us(const sh_link *l);

/* How the worker is reached ("vsock:9500", "tcp 10.0.2.2:9500 ..."), for logs. */
const char *sh_link_transport(const sh_link *l);

/* The SIMD table this process runs on, for callers with hot loops of their own. */
const sh_simd *sh_link_simd(void);

#ifdef __cplusplus
}
#endif
#endif
