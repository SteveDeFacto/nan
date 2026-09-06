/*
 * ggml-shielded.h -- a GGML backend that offloads linear ops to a GPU whose host
 * is fully untrusted.
 *
 * This is the piece REPORT.md calls "the production engine" and lists as open
 * item 1: without it the shielded tier is a protocol and a worker with nothing
 * able to drive them, and a metal box advertises a shielded pool that nothing can
 * buy. shielded/model.py is the specification; this is the same op placement in
 * the engine, so an ordinary ggml graph -- llama.cpp, whisper.cpp, sd.cpp -- gets
 * the tier without knowing it exists.
 *
 * HOW THE SPLIT HAPPENS. ggml_backend_sched already partitions a graph across a
 * priority-ordered backend list and inserts the copies. This backend claims ONLY
 * matmuls against q8_0 weights it has calibration for; everything else -- softmax,
 * norms, SiLU, rope, attention, sampling -- fails supports_op and lands on the CPU
 * backend, inside the enclave. The classifier is a pure function of the tensor, so
 * it is deterministic across decode steps, which is what stops sched reallocating
 * and forcing a full graph resend every token.
 *
 * WHAT CROSSES. Public weights, once, in the clear. Then one-time-padded
 * activations and their masked products, and nothing else, ever. Plaintext
 * activations, the pads, the KV cache, the Freivalds secret and the sampling state
 * never leave the CVM.
 *
 * WHAT THIS IS NOT. It is not a confidentiality boundary by itself -- the masks
 * are. The worker is assumed hostile and is expected to read every byte it gets.
 */
#ifndef GGML_SHIELDED_H
#define GGML_SHIELDED_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t     ggml_backend_shielded_init(void);
GGML_BACKEND_API bool               ggml_backend_is_shielded(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_shielded_reg(void);

/* Where the worker is, and which calibration to trust. Both default from the
 * environment (SHIELDED_HOST, SHIELDED_PORT, SHIELDED_CALIB) so an engine can
 * enable the tier as launch configuration without an app-visible API -- which is
 * the point: existing catalog guests keep their wasi-nn contract unchanged. */
GGML_BACKEND_API void ggml_backend_shielded_configure(const char *host, int port,
                                                      const char *calib_path);

/* Capability probe used by the manager before admitting a pooled tenant. */
GGML_BACKEND_API int ggml_backend_shielded_pool_version(void);
/* Dealt pads: mint one .pads shipment from the registered weights (single
 * link). Hex arguments: seed 64, seed_id 32, model digest 64, consumer X25519
 * public key 64. Returns SH_OK or an sh error. */
GGML_BACKEND_API int ggml_backend_shielded_mint(const char *seed_hex, const char *seed_id_hex, const char *digest_hex,
                                                uint64_t index0, uint64_t count, const char *consumer_pk_hex, const char *path);

/* Counters for the boot probe and the supervisor's verdict. */
GGML_BACKEND_API void ggml_backend_shielded_stats(uint64_t *offloaded_nodes,
                                                  uint64_t *local_nodes,
                                                  uint64_t *macs,
                                                  uint64_t *verify_fail);

#ifdef __cplusplus
}
#endif
#endif
