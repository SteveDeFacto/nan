/* Dealt pads: shipments of u = r.W minted OFF the trusted half.
 *
 * shielded/dealer/PLAN.md is the design. A shipment (`.pads`) carries, for a
 * range of pad indices and every weight group of one model, the unblinding
 * vectors u; the mask r is never shipped - both the dealer and the consumer
 * derive it from a shared 32-byte seed with sh_pad_r(). Each (index, group)
 * cell is an independent ChaCha20-Poly1305 box under a per-shipment key that
 * is itself boxed to the consumer's X25519 pad key, so an operator's bank
 * holds ciphertext and sizes only and a consumer can start on any row.
 *
 * Layout, little-endian:
 *   sh_pads_hdr (fixed 256 bytes) | groups[] (sh_pads_group, 80 bytes each)
 *   | cells: row-major by index, group order as in groups[], each cell =
 *     16-byte Poly1305 tag + 3*u_len bytes of u+M/2 in [0,M) (M ~ 2^23.8)
 */
#ifndef SHIELDED_PADS_H
#define SHIELDED_PADS_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SH_PADS_MAGIC     "ENCLPAD1"
#define SH_PADS_VERSION   2u          /* 2: ChaCha20-Poly1305 cells (1 was XSalsa20-Poly1305) */
#define SH_PADS_NAME_MAX  64
#define SH_PADS_CELL_TAG  16          /* Poly1305 */
#define SH_PADS_HDR_BYTES 256

typedef struct {
    uint32_t group;                   /* consumer's ordinal at mint time */
    uint32_t K;
    uint64_t u_len;                   /* sum of N over the group's nodes */
    char     name[SH_PADS_NAME_MAX];  /* the group's first node, for binding by name */
} sh_pads_group;

/* Fixed header. Everything but key_box/hdr_box is covered by hdr_box, the
 * secretbox of the SHA-512 of (fixed fields with those two zeroed || group
 * table) under the shipment key. */
typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t group_count;
    uint8_t  model_digest[32];
    uint8_t  seed_id[16];
    uint64_t index0, index_count;
    uint64_t data_off;                /* first cell */
    uint64_t row_bytes;               /* one index, all groups */
    uint8_t  dealer_pk[32];           /* ephemeral X25519 public key of this shipment */
    uint8_t  key_box[48];             /* crypto_box(shipment key) to the consumer's pad key */
    uint8_t  hdr_box[80];             /* ChaCha20-Poly1305 of sha512(header || groups): tag || ct */
    uint8_t  pad[256 - 8 - 4 - 4 - 32 - 16 - 8 - 8 - 8 - 8 - 32 - 48 - 80];
} sh_pads_hdr;

/* The engine's ChaCha20 block (64-bit counter, zero nonce). */
void sh_chacha20_block(const uint32_t key[8], uint64_t counter, uint32_t out[16]);
/* r for (seed, group, index): ChaCha20 keyed by the seed, block counter
 * (group << 48) | (index << 24) | block, values uniform over [0, M) by the
 * same uint64 draw the engine's own mask bank uses. Both sides call this. */
void sh_pad_r(const uint8_t seed[32], uint32_t group, uint64_t index, int64_t K, int32_t *r_out);

/* --- writer (the dealer) -------------------------------------------------- */
typedef struct sh_pads_writer sh_pads_writer;
/* Creates the file, mints a shipment key, boxes it to `consumer_pk`, writes
 * the header and group table, and reserves the cell area. */
sh_pads_writer *sh_pads_writer_open(const char *path, const uint8_t model_digest[32],
                                    const uint8_t seed_id[16], const sh_pads_group *groups, uint32_t n_groups,
                                    uint64_t index0, uint64_t index_count, const uint8_t consumer_pk[32], int *err);
/* Cells may be written in any order; each is boxed independently. `u` is the
 * balanced row (u_len values in (-M/2, M/2]). */
int  sh_pads_writer_cell(sh_pads_writer *w, uint64_t index, uint32_t group, const int32_t *u);
int  sh_pads_writer_close(sh_pads_writer *w);

/* --- reader (the consumer: the pVM, or the CVM engine) -------------------- */
typedef struct sh_pads_reader sh_pads_reader;
/* Scans `dir` for shipments of `seed_id` readable with `consumer_sk` and keeps
 * the ones whose header verifies. Rescans on demand when an index is missing. */
sh_pads_reader *sh_pads_reader_open(const char *dir, const uint8_t seed_id[16], const uint8_t consumer_sk[32], int *err);
/* Binds the consumer's group table: every group must appear in every shipment
 * with the same K and u_len (matched by name); records the ordinal map. */
int  sh_pads_reader_bind(sh_pads_reader *r, const sh_pads_group *groups, uint32_t n_groups);
/* Reads and opens one cell into `u_out` (u_len balanced values). Returns
 * SH_ERR_EXHAUST when no shipment on disk covers the index (after a rescan),
 * SH_ERR_VERIFY when the box does not open (tampered or wrong key). */
int  sh_pads_reader_cell(sh_pads_reader *r, uint32_t group, uint64_t index, int32_t *u_out);
/* Pin the model: shipments whose header digest differs are ignored (the
 * dealer prints the digest it recorded; SHIELDED_PAD_MODEL_DIGEST carries it). */
void sh_pads_reader_require_digest(sh_pads_reader *r, const uint8_t model_digest[32]);
/* The highest index any shipment on disk covers, plus one (0 = none). */
uint64_t sh_pads_reader_extent(const sh_pads_reader *r);
void sh_pads_reader_close(sh_pads_reader *r);

/* --- ledger window ---------------------------------------------------------
 * Reserve-before-use: the mark advances to lo+W BEFORE any index in [lo, lo+W)
 * is consumed, so a restart resumes at the mark and can never replay. The P1
 * ledger is a local file holding the mark; P2 moves it to the platform. */
int  sh_pads_window_reserve(const char *ledger_path, uint64_t want, uint64_t *lo, uint64_t *hi);

/* --- the platform's seed box and signed windows (relay/pads.mjs) ------------
 * seed box: X25519(epk, pad key) -> HKDF-SHA512(shared, salt = epk || pad_pk,
 * info "enclave-pads-seed-box") -> ChaCha20-Poly1305 (RFC 8439, no AAD). */
int  sh_pads_seed_open(const uint8_t epk[32], const uint8_t nonce[12], const uint8_t *box, size_t box_len,
                       const uint8_t pad_sk[32], const uint8_t pad_pk[32], uint8_t seed_out[32]);
/* A ledger window: Ed25519 over "enclave-pads-window\n<seed_id hex>\n<lo>\n<hi>\n<iat>". */
bool sh_pads_window_verify(const uint8_t ledger_pk[32], const char *seed_id_hex, uint64_t lo, uint64_t hi, uint64_t iat, const uint8_t sig[64]);
/* Ed25519 detached signature by the pVM's transport key over the request
 * line set "enclave-pads-<kind>\n<field>\n...\n<nonce hex>" (fields already text). */
void sh_pads_request_sign(const uint8_t transport_sk[64], const char *kind, const char *const *fields, size_t n_fields, const char *nonce_hex, uint8_t sig_out[64]);

/* Hex helpers for the env/CLI surface. */
bool sh_pads_hex2bin(const char *hex, uint8_t *out, size_t n);
void sh_pads_bin2hex(const uint8_t *in, size_t n, char *out /* 2n+1 */);

#ifdef __cplusplus
}
#endif
#endif
