/* The model's own MTP head as a draft model, for the phone engine.
 *
 * A straight extraction of the shim's driver (wasm/llama-shim/enclave_llama.c,
 * "MTP (multi-token prediction) driver") without the device sampler and the
 * server plumbing: single sequence, greedy, plain C over llama.h plus the
 * fork's two nextn entry points bound by their mangled names. Why it matters
 * here: on the phone every token costs ~100 dependent round trips through the
 * shielded link, so tokens per round trip is the lever - the head proposes k
 * tokens on the phone's CPU (one small layer), the target verifies k+1 rows
 * in ONE chain, and the accepted prefix arrives for the price of one token.
 * Only accepted tokens are mirrored into the head (observe), so rejected
 * proposals never pollute its attention. */
#ifndef ANCHOR_MTP_H
#define ANCHOR_MTP_H
#include <stdint.h>
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct anchor_mtp anchor_mtp;

/* NULL when the model has no nextn layer or libllama lacks the fork's nextn
 * symbols (then decode plainly). `n_batch` bounds one observe (k+1). Enables
 * nextn hidden-row output on the target context. */
anchor_mtp *anchor_mtp_new(struct llama_model *model, struct llama_context *target, uint32_t n_ctx, uint32_t n_batch, int n_threads);
void        anchor_mtp_free(anchor_mtp *m);
/* The head context, e.g. to attach the target's thread pool (one pool for both: the target's
 * workers busy-poll after its graph and would otherwise contend with the head's fresh threads). */
struct llama_context *anchor_mtp_ctx(anchor_mtp *m);
/* Accumulated microseconds inside draft: seq_rm, decode, argmax (+ nextn copy), and observe's decode. */
void        anchor_mtp_timers(anchor_mtp *m, double *rm_us, double *decode_us, double *argmax_us, double *observe_us);
/* After the target decoded `n_rows` rows: copy their nextn hidden rows. */
int         anchor_mtp_harvest(anchor_mtp *m, struct llama_context *target, int32_t n_rows);
/* Mirror `n` committed tokens at pos0.. into the head (needs a harvest of >= n rows
 * from the decode that produced them); the last row's h becomes the draft seed. */
int32_t     anchor_mtp_observe(anchor_mtp *m, int32_t pos0, const int32_t *tokens, int32_t n);
/* Propose up to k tokens after id_last at n_past (greedy on the head; p_min > 0
 * stops when the head's confidence drops). Returns how many. */
int32_t     anchor_mtp_draft(anchor_mtp *m, int32_t id_last, int32_t n_past, int32_t k, float p_min, int32_t *tokens_out);

#ifdef __cplusplus
}
#endif
#endif
