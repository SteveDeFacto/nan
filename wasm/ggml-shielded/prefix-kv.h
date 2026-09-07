/* Shared-prefix KV artifacts (shielded/dealer/PLAN.md, P4).
 *
 * The shared prefix of a chat (system prompt, tool schemas) is public text,
 * so the platform prefills it once, in the clear, and publishes the resulting
 * KV cache for (model, prefix) as a signed file; a consumer loads it instead
 * of prefilling: no pad rows for the prefix, no minutes of phone prefill, a
 * one-second re-park after a restart. The KV file itself is llama's
 * `llama_state_seq_save_file` of the prefix sequence; the trust is the
 * sidecar `<file>.sig`:
 *
 *   enclave-prefix-kv-v1\n
 *   model <model digest hex, 64>\n
 *   prefix-sha512 <hex, 128>\n
 *   tokens <n>\n
 *   file-sha512 <hex, 128>\n
 *   sig <hex, 128>\n
 *
 * `sig` is Ed25519 (TweetNaCl) over the first five lines exactly as written
 * (including their newlines), by the platform's prefix key, which consumers
 * pin the way they pin the ledger key. Nothing here is secret. */
#ifndef SHIELDED_PREFIX_KV_H
#define SHIELDED_PREFIX_KV_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Writes `<kv_path>.sig`. `sk` is the 64-byte TweetNaCl Ed25519 secret. */
int sh_prefix_kv_sign(const char *kv_path, const uint8_t model_digest[32], const char *prefix, size_t prefix_len,
                      uint64_t n_tokens, const uint8_t sk[64], char *err, size_t err_cap);
/* Verifies `<kv_path>.sig` against the pinned key, this model and this exact
 * prefix text, and the file's own hash. 0 = usable (n_tokens filled in),
 * -1 = not usable (err says why). Never loads anything into llama. */
int sh_prefix_kv_verify(const char *kv_path, const uint8_t pk[32], const uint8_t model_digest[32], const char *prefix, size_t prefix_len,
                        uint64_t *n_tokens_out, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif
