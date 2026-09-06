/* enclave_llama.c - see enclave_llama.h for the contract. Built against the
 * PINNED llama.cpp checkout by the enclave-llamacpp toolchain workflow (and by
 * hand for local smokes):
 *
 *   cc -shared -fPIC -Wl,-soname,libenclave_llama.so \
 *      -I<llama.cpp>/include -I<llama.cpp>/ggml/include \
 *      enclave_llama.c -L<llama.cpp>/build/bin -lllama -lggml \
 *      -o libenclave_llama.so
 *
 * The -soname is load-bearing: the wasmtime binary NEEDs "libenclave_llama.so"
 * by that bare name, and in the manager image it is resolved by ldconfig from
 * /usr/local/lib - a soname-less lib is not reliably cached there.
 */
/* RTLD_DEFAULT (the mtp nextn binding) is a GNU extension; must be defined
 * before the FIRST libc header anywhere in the include graph */
#define _GNU_SOURCE 1

#include "enclave_llama.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* CPU threads for text-model compute. llama_context_default_params() leaves
 * n_threads at GGML_DEFAULT_N_THREADS (4) - a fine number for the GPU fleet,
 * where the card does the work and the CPU only orchestrates, and a silent
 * 4x-to-16x throughput cliff on a box that serves the model on CORES: nobody
 * sets it, llama never rethinks it, and a 16-vCPU CVM prefills a 27b on 4
 * threads (measured on metal0, 2026-08-31: a whole first turn on ~1 core).
 * ENCLAVE_GGML_N_THREADS overrides; unset = every online CPU, which is the
 * right default in a CVM because the guest's vCPUs are the share the operator
 * already bought - there is no other tenant inside this boundary to leave
 * cores for. */
static int32_t ell_n_threads(void) {
    const char *e = getenv("ENCLAVE_GGML_N_THREADS");
    if (e && *e) {
        long v = strtol(e, NULL, 10);
        if (v > 0 && v <= 512) { return (int32_t)v; }
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int32_t)n : 4;
}

/* Threads for BATCH compute (prefill: any ubatch of 32+ tokens), llama's
 * n_threads_batch. Decode wants few threads on the shielded tier - the
 * masking refill threads are the binding resource there and every compute
 * thread past four measured as pure contention - but prefill runs in the
 * enclave on cores by policy and scales with them (metal0, 27B: ~15 tok/s
 * on 4 threads). ENCLAVE_GGML_N_THREADS_BATCH overrides; unset = the decode
 * count, so nothing changes for a deployment that sets only nnThreads. */
static int32_t ell_n_threads_batch(void) {
    const char *e = getenv("ENCLAVE_GGML_N_THREADS_BATCH");
    if (e && *e) {
        long v = strtol(e, NULL, 10);
        if (v > 0 && v <= 512) { return (int32_t)v; }
    }
    return ell_n_threads();
}

void ell_init(void) {
    /* GGML_BACKEND_DL builds ship the compute backends (cpu, cuda) as
     * dlopened modules so the wasmtime binary carries no DT_NEEDED on
     * libcuda.so.1 (the driver exists only at runtime, injected by the
     * nvidia container runtime - and never on CPU-flavor nodes). Load them
     * from ENCLAVE_GGML_BACKEND_DIR (NULL = executable dir + cwd). A module
     * whose own deps are unresolvable is skipped silently in release builds;
     * ell_gpu_devices() is how callers check that a GPU actually arrived.
     *
     * Guarded: the sd shim (enclave_sd) shares this process AND this ggml -
     * whichever init runs first loads the modules; loading again would
     * register duplicate devices in ggml's registry. */
    if (ggml_backend_dev_count() == 0) {
        ggml_backend_load_all_from_path(getenv("ENCLAVE_GGML_BACKEND_DIR"));
    }
    llama_backend_init();
}

int32_t ell_gpu_devices(void) {
    int32_t n = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            n++;
        }
    }
    return n;
}

/* D2H path probe: the speculation post-mortem's last open question is
 * whether logits copies out of the GPU run at pinned or pageable speed
 * inside the CVM (ggml-cuda's host buffer type FALLS BACK to plain malloc
 * with only a host-side warning when cudaMallocHost fails - plausible under
 * SEV-SNP, invisible from the guest). Allocates size_mb on the device, then
 * times synchronous tensor_get into (a) the device's pinned host buffer if
 * it allocates, (b) plain malloc. min-of-3 each. out = [pinned_ok,
 * pinned_us, pageable_us]. Returns 0 ok, -1 no GPU / alloc failure. */
/* Graph-stage perf + small-batch slot state, from llama_graph_perf (mm11):
 * out = [build_us, sched_alloc_us, set_inputs_us, slot_state] with the
 * micros read-and-clear and slot_state persistent (0 never tried, 1 the
 * small-batch graph slot is active, -1 its reserve failed on this box).
 * The discriminator for whether the mm10 slot actually engages inside the
 * CVM - its failure warning never leaves host logs. */
int32_t ell_graph_perf(void *ctx, int64_t *out) {
    if (!ctx) return -1;
    llama_graph_perf((struct llama_context *)ctx, out);
    return 0;
}

int32_t ell_graph_perf2(void *ctx, int64_t *out) {
    /* mm21: out[7] = graph_perf plus [4]=memory init_batch us,
     * [5]=graph_compute call us, [6]=output reserve+extract us
     * (read-and-clear) - the decode-stage decomposition behind the
     * CPU-orchestration-bound finding (sync waits measured 0.18 ms/token,
     * so the ~15 ms/token lives in llama's CPU path; these name it). */
    if (!ctx) return -1;
    llama_graph_perf2((struct llama_context *)ctx, out);
    return 0;
}

int32_t ell_d2h_probe(int32_t size_mb, int64_t *out) {
    if (size_mb < 1)  size_mb = 1;
    if (size_mb > 16) size_mb = 16;
    const size_t sz = (size_t) size_mb * 1024 * 1024;
    out[0] = 0; out[1] = -1; out[2] = -1;

    ggml_backend_dev_t dev = NULL;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = ggml_backend_dev_get(i);
            break;
        }
    }
    if (!dev) return -1;

    struct ggml_init_params ip = { /*mem_size*/ ggml_tensor_overhead() * 2, /*mem_buffer*/ NULL, /*no_alloc*/ true };
    struct ggml_context * gctx = ggml_init(ip);
    if (!gctx) return -1;
    struct ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, (int64_t) (sz / sizeof(float)));
    ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, ggml_backend_dev_buffer_type(dev));
    if (!dbuf) { ggml_free(gctx); return -1; }

    ggml_backend_buffer_t pbuf = NULL;
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (host_buft) {
        pbuf = ggml_backend_buft_alloc_buffer(host_buft, sz);
    }
    void * pageable = malloc(sz);
    if (!pageable) { if (pbuf) ggml_backend_buffer_free(pbuf); ggml_backend_buffer_free(dbuf); ggml_free(gctx); return -1; }

    /* one warm copy primes lazy init on whichever path exists */
    ggml_backend_tensor_get(t, pageable, 0, sz);

    if (pbuf) {
        void * pinned = ggml_backend_buffer_get_base(pbuf);
        int64_t best = INT64_MAX;
        for (int r = 0; r < 3; r++) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_get(t, pinned, 0, sz);
            const int64_t el = ggml_time_us() - t0;
            if (el < best) best = el;
        }
        out[0] = 1; out[1] = best;
    }
    {
        int64_t best = INT64_MAX;
        for (int r = 0; r < 3; r++) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_get(t, pageable, 0, sz);
            const int64_t el = ggml_time_us() - t0;
            if (el < best) best = el;
        }
        out[2] = best;
    }

    free(pageable);
    if (pbuf) ggml_backend_buffer_free(pbuf);
    ggml_backend_buffer_free(dbuf);
    ggml_free(gctx);
    return 0;
}

void *ell_load_model(const char *path, int32_t n_gpu_layers) {
    struct llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = n_gpu_layers;
    /* upstream ddd4ec14 (#26296) made MTP-head loading OPT-IN and default
     * OFF; without this the nextn tensors load as "unused ... ignoring",
     * ell_mtp_available() reports 0, and speculation silently degrades to
     * plain decode (observed live 2026-08-01, first build on the new pin).
     * Default on, matching pre-#26296 behavior: the head's cost is priced
     * into the fit math (kv_layers+1) and callers without MTP models are
     * unaffected (no nextn tensors to load).
     *
     * ENCLAVE_GGML_LOAD_MTP=0 (deployment-config nnLoadMtp:false) is the
     * per-deployment opt-OUT: the head weights cost real VRAM (~260 MB on
     * the 27b) and only pay when a config drafts with the head - which
     * loses to prompt-lookup on CVM hardware (per-round head-step +
     * harvest launch costs, measured 2026-08-03). Opt-out keeps absence a
     * deliberate choice: a drafting config against a headless load
     * degrades loudly to plain decode via ell_mtp_available()=0. */
    const char *lm = getenv("ENCLAVE_GGML_LOAD_MTP");
    p.load_mtp = !(lm && lm[0] == '0' && lm[1] == '\0');
    return llama_model_load_from_file(path, p);
}

void ell_free_model(void *model) { llama_model_free((struct llama_model *)model); }

int32_t ell_n_vocab(void *model) {
    return llama_vocab_n_tokens(llama_model_get_vocab((const struct llama_model *)model));
}

/* ---- host tokenizer ------------------------------------------------------
 * The GGUF already carries the model's tokenizer, fully built the moment the
 * model loads. Exporting it saves every guest request the multi-MB
 * tokenizer.json read + parse (~600ms of TTFT measured in-fleet 2026-08-01).
 * Both calls are pure vocab lookups - no context, no decode turn. */

/* Tokenize UTF-8 text (parse_special on: template markers like <|im_start|>
 * must map to their single special ids, exactly as the guest-side tokenizer
 * did). Returns the token count, or -needed when out_cap is too small, or
 * INT32_MIN on hard failure. */
int32_t ell_tokenize(void *model, const char *text, int32_t text_len,
                     int32_t *out_ids, int32_t out_cap) {
    const struct llama_vocab *v = llama_model_get_vocab((const struct llama_model *)model);
    int32_t n = llama_tokenize(v, text, text_len, out_ids, out_cap,
                               /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) {
        /* llama returns -needed on overflow */
        return n == INT32_MIN ? INT32_MIN : n;
    }
    return n;
}

/* The raw bytes of one token (special tokens render their text form).
 * Returns byte count, or -needed when buf is too small. */
int32_t ell_token_piece(void *model, int32_t id, char *buf, int32_t cap) {
    const struct llama_vocab *v = llama_model_get_vocab((const struct llama_model *)model);
    return llama_token_to_piece(v, id, buf, cap, /*lstrip=*/0, /*special=*/true);
}

/* ell_kv_type code -> ggml_type; unknown falls back to F16 (the llama default). */
static enum ggml_type ell_ggml_kv_type(int32_t t) {
    switch (t) {
        case ELL_KV_Q8_0: return GGML_TYPE_Q8_0;
        case ELL_KV_Q4_0: return GGML_TYPE_Q4_0;
        case ELL_KV_F32:  return GGML_TYPE_F32;
        case ELL_KV_F16:
        default:          return GGML_TYPE_F16;
    }
}

void *ell_new_context(void *model, uint32_t n_ctx, uint32_t n_batch,
                      int32_t type_k, int32_t type_v, int32_t flash_attn) {
    struct llama_context_params p = llama_context_default_params();
    p.n_threads = ell_n_threads();
    p.n_threads_batch = ell_n_threads_batch();
    p.n_ctx = n_ctx;
    if (n_batch) { p.n_batch = n_batch; }
    p.type_k = ell_ggml_kv_type(type_k);
    p.type_v = ell_ggml_kv_type(type_v);
    switch (flash_attn) {
        case ELL_FA_ENABLED:  p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  break;
        case ELL_FA_DISABLED: p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; break;
        case ELL_FA_AUTO:
        default:              p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;     break;
    }
    return llama_init_from_model((struct llama_model *)model, p);
}

static void ell_topk_release(void *ctx); /* mm26 registry, defined below */

void ell_free_context(void *ctx) {
    llama_free((struct llama_context *)ctx);
    /* device top-k chains must outlive the context; freed after it */
    ell_topk_release(ctx);
}

void ell_reset(void *ctx) {
    llama_memory_clear(llama_get_memory((struct llama_context *)ctx), true);
}

int32_t ell_decode(void *ctx, void *model, const int32_t *tokens, int32_t n, float *logits_out) {
    struct llama_context *lctx = (struct llama_context *)ctx;
    if (n <= 0 || (uint32_t)n > llama_n_batch(lctx)) {
        return -1;
    }
    /* llama_batch_get_one wants a mutable pointer but does not write; the
     * cast is safe against the pinned revision (verified at pin time). */
    struct llama_batch batch = llama_batch_get_one((llama_token *)tokens, n);
    int32_t rc = llama_decode(lctx, batch);
    if (rc != 0) {
        return rc;
    }
    const float *logits = llama_get_logits_ith(lctx, -1);
    if (!logits) {
        return -2;
    }
    memcpy(logits_out, logits, (size_t)ell_n_vocab(model) * sizeof(float));
    return 0;
}

/* ---- device top-k (mm26) --------------------------------------------------
 * ENCLAVE_GGML_DEV_TOPK=<K>: arm every server sequence with a backend
 * sampler chain of [top_k(K)]. llama then computes the top-K candidate ids
 * and their logits ON DEVICE for every output row of a small decode and
 * skips the full-vocab logits extraction entirely (needs_raw_logits) - the
 * per-row cost drops from a ~1 MB forced-sync D2H + a host-side 248K scan
 * to two K-sized async copies and a CUB radix sort. Wide batches (prefill:
 * more output rows than LLAMA_SAMPLER_MAX_ROWS, default 8) fall back to the
 * raw-logits path inside llama, and the *_topk entry points report that
 * fallback to the caller instead of failing (return 0 = full rows written).
 * Chains must outlive the context: registered here, freed by
 * ell_free_context after llama_free. */
#define ELL_TOPK_MAX_CTX 16
#define ELL_TOPK_MAX_SEQ 16
struct ell_topk_reg {
    void *ctx;
    int32_t k;
    int32_t n_seq;
    struct llama_sampler *chains[ELL_TOPK_MAX_SEQ];
};
static struct ell_topk_reg ell_topk_regs[ELL_TOPK_MAX_CTX];

static struct ell_topk_reg *ell_topk_find(void *ctx) {
    for (int i = 0; i < ELL_TOPK_MAX_CTX; i++) {
        if (ell_topk_regs[i].ctx == ctx) { return &ell_topk_regs[i]; }
    }
    return NULL;
}

static void ell_topk_release(void *ctx) {
    struct ell_topk_reg *r = ctx ? ell_topk_find(ctx) : NULL;
    if (!r) { return; }
    for (int32_t s = 0; s < r->n_seq; s++) {
        if (r->chains[s]) { llama_sampler_free(r->chains[s]); }
    }
    memset(r, 0, sizeof(*r));
}

int32_t ell_server_topk_k(void *ctx) {
    struct ell_topk_reg *r = ell_topk_find(ctx);
    return r ? r->k : 0;
}

void *ell_new_server(void *model, uint32_t n_ctx, uint32_t n_batch, uint32_t n_seq_max,
                     int32_t type_k, int32_t type_v, int32_t flash_attn) {
    struct llama_context_params p = llama_context_default_params();
    p.n_threads = ell_n_threads();
    p.n_threads_batch = ell_n_threads_batch();
    p.n_ctx = n_ctx;
    if (n_batch) { p.n_batch = n_batch; }
    if (n_seq_max) { p.n_seq_max = n_seq_max; }
    /* n_ubatch by env, not by parameter: see the header. Only vision models
     * whose image chunks decode non-causally need it above the default, and
     * every model pays for it in compute buffers, so it stays opt-in. */
    const char *ub = getenv("ENCLAVE_GGML_N_UBATCH");
    if (ub && *ub) {
        long v = strtol(ub, NULL, 10);
        if (v > 0) {
            p.n_ubatch = (uint32_t)v;
            if (p.n_batch < p.n_ubatch) { p.n_batch = p.n_ubatch; }
        }
    }
    /* Recurrent-state snapshots for speculative rewind, env-read like
     * n_ubatch above (a parameter would silently change this function's ABI
     * for a wasmtime built against an older tarball). Each unit of depth
     * widens every recurrent layer's R/S tensors by one full copy (~1.2 GB
     * on the 27b), buying llama_memory_seq_rm rollback of that many trailing
     * tokens - the no-branch speculative verify. llama clamps the value to 0
     * itself on archs without rollback support (currently qwen3.5/moe), and
     * 0 (unset) is byte-for-byte today's context. */
    const char *rs = getenv("ENCLAVE_GGML_N_RS_SEQ");
    if (rs && *rs) {
        long v = strtol(rs, NULL, 10);
        if (v > 0) {
            p.n_rs_seq = (uint32_t)(v > 16 ? 16 : v);
        }
    }
    /* ONE pool of n_ctx tokens shared by every sequence (vs. the split
     * per-stream layout): a long conversation and several short ones coexist
     * without pre-partitioning, which is the sizing model the platform's
     * capacity gates price. */
    p.kv_unified = true;
    p.type_k = ell_ggml_kv_type(type_k);
    p.type_v = ell_ggml_kv_type(type_v);
    switch (flash_attn) {
        case ELL_FA_ENABLED:  p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  break;
        case ELL_FA_DISABLED: p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; break;
        case ELL_FA_AUTO:
        default:              p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;     break;
    }
    /* device top-k arming (see the mm26 block above). Chains are sampler
     * CHAINS (bare samplers are refused by llama), built per sequence, and
     * must outlive the context. A context that refuses the sampler graph
     * (extra compute buffers) falls back to an unarmed context cleanly. */
    int32_t tk = 0;
    const char *tke = getenv("ENCLAVE_GGML_DEV_TOPK");
    if (tke && *tke) {
        long v = strtol(tke, NULL, 10);
        if (v >= 16 && v <= 4096) { tk = (int32_t)v; }
    }
    int32_t ns_tk = (int32_t)(n_seq_max ? n_seq_max : 1);
    if (ns_tk > ELL_TOPK_MAX_SEQ) { ns_tk = ELL_TOPK_MAX_SEQ; }
    struct llama_sampler *tkchains[ELL_TOPK_MAX_SEQ] = {0};
    struct llama_sampler_seq_config tkcfg[ELL_TOPK_MAX_SEQ];
    if (tk > 0) {
        for (int32_t s = 0; s < ns_tk; s++) {
            struct llama_sampler_chain_params cp = llama_sampler_chain_default_params();
            cp.no_perf = true;
            tkchains[s] = llama_sampler_chain_init(cp);
            if (!tkchains[s]) { tk = 0; break; }
            llama_sampler_chain_add(tkchains[s], llama_sampler_init_top_k(tk));
            tkcfg[s].seq_id  = s;
            tkcfg[s].sampler = tkchains[s];
        }
    }
    if (tk > 0) {
        p.samplers   = tkcfg;
        p.n_samplers = (size_t)ns_tk;
    }
    struct llama_context *lctx = llama_init_from_model((struct llama_model *)model, p);
    if (!lctx && tk > 0) {
        p.samplers = NULL;
        p.n_samplers = 0;
        for (int32_t s = 0; s < ns_tk; s++) {
            if (tkchains[s]) { llama_sampler_free(tkchains[s]); tkchains[s] = NULL; }
        }
        tk = 0;
        lctx = llama_init_from_model((struct llama_model *)model, p);
    }
    if (lctx && tk > 0) {
        struct ell_topk_reg *r = ell_topk_find(NULL); /* free slot */
        if (r) {
            r->ctx = lctx;
            r->k = tk;
            r->n_seq = ns_tk;
            memcpy(r->chains, tkchains, sizeof(tkchains));
        }
        /* no free registry slot: keep serving, report unarmed via
         * ell_server_topk_k == 0; chains leak, bounded by process life */
    } else if (tk > 0) {
        for (int32_t s = 0; s < ns_tk; s++) {
            if (tkchains[s]) { llama_sampler_free(tkchains[s]); }
        }
    }
    return lctx;
}

int32_t ell_decode_batch(void *ctx, void *model, int32_t n_items,
                         const int32_t *seq_ids, const int32_t *counts,
                         const int32_t *positions, const int32_t *tokens_flat,
                         float *logits_flat) {
    struct llama_context *lctx = (struct llama_context *)ctx;
    if (n_items <= 0) {
        return -1;
    }
    int32_t total = 0;
    for (int32_t i = 0; i < n_items; i++) {
        if (counts[i] <= 0 || positions[i] < 0) {
            return -1;
        }
        total += counts[i];
    }
    if ((uint32_t)total > llama_n_batch(lctx)) {
        return -1;
    }
    struct llama_batch batch = llama_batch_init(total, 0, 1);
    int32_t cursor = 0;
    for (int32_t i = 0; i < n_items; i++) {
        for (int32_t t = 0; t < counts[i]; t++) {
            batch.token[cursor]     = tokens_flat[cursor];
            batch.pos[cursor]       = positions[i] + t;
            batch.n_seq_id[cursor]  = 1;
            batch.seq_id[cursor][0] = seq_ids[i];
            batch.logits[cursor]    = (int8_t)(t == counts[i] - 1);
            cursor++;
        }
    }
    batch.n_tokens = total;
    int32_t rc = llama_decode(lctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        return rc;
    }
    const size_t row = (size_t)ell_n_vocab(model);
    cursor = 0;
    for (int32_t i = 0; i < n_items; i++) {
        cursor += counts[i];
        const float *logits = llama_get_logits_ith(lctx, cursor - 1);
        if (!logits) {
            llama_batch_free(batch);
            return -2;
        }
        memcpy(logits_flat + (size_t)i * row, logits, row * sizeof(float));
    }
    llama_batch_free(batch);
    return 0;
}

void ell_seq_remove(void *ctx, int32_t seq_id) {
    llama_memory_seq_rm(llama_get_memory((struct llama_context *)ctx), seq_id, -1, -1);
}

int32_t ell_decode_seq_full(void *ctx, void *model, int32_t seq_id, int32_t pos0,
                            const int32_t *tokens, int32_t n, float *logits_out) {
    struct llama_context *lctx = (struct llama_context *)ctx;
    if (n <= 0 || pos0 < 0 || (uint32_t)n > llama_n_batch(lctx)) {
        return -1;
    }
    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t t = 0; t < n; t++) {
        batch.token[t]     = tokens[t];
        batch.pos[t]       = pos0 + t;
        batch.n_seq_id[t]  = 1;
        batch.seq_id[t][0] = seq_id;
        batch.logits[t]    = 1;
    }
    batch.n_tokens = n;
    int32_t rc = llama_decode(lctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        return rc;
    }
    const size_t row = (size_t)ell_n_vocab(model);
    for (int32_t t = 0; t < n; t++) {
        const float *lg = llama_get_logits_ith(lctx, t);
        if (!lg) {
            llama_batch_free(batch);
            return -2;
        }
        memcpy(logits_out + (size_t)t * row, lg, row * sizeof(float));
    }
    llama_batch_free(batch);
    return 0;
}

/* mm26 shared readback: after a decode on an armed context, copy each of
 * the n output rows' device-computed top-k (ids + logits) into k_cap-strided
 * buffers. Returns k_eff (>0), or 0 when the sampled path did not engage
 * (wide batch, unarmed context) - the caller then reads full logits rows
 * itself. Never re-decodes. */
static int32_t ell_topk_readback(struct llama_context *lctx, int32_t n,
                                 const int32_t *idxs,
                                 int32_t k_cap, int32_t *ids_out, float *vals_out) {
    /* idxs[i] = the BATCH TOKEN INDEX of row i's output (the _ith getters
     * follow llama_get_logits_ith semantics: token index, not output row -
     * they only coincide when every token is an output) */
    int32_t cnt0 = (int32_t)llama_get_sampled_candidates_count_ith(lctx, idxs[0]);
    if (cnt0 <= 0) {
        return 0;
    }
    int32_t k_eff = cnt0 < k_cap ? cnt0 : k_cap;
    for (int32_t i = 0; i < n; i++) {
        const llama_token *ids = llama_get_sampled_candidates_ith(lctx, idxs[i]);
        const float *vals = llama_get_sampled_logits_ith(lctx, idxs[i]);
        int32_t ci = (int32_t)llama_get_sampled_candidates_count_ith(lctx, idxs[i]);
        int32_t li = (int32_t)llama_get_sampled_logits_count_ith(lctx, idxs[i]);
        if (!ids || !vals || ci < k_eff || li < k_eff) {
            return -4; /* partial sampled state: should not happen */
        }
        memcpy(ids_out + (size_t)i * k_cap, ids, (size_t)k_eff * sizeof(int32_t));
        memcpy(vals_out + (size_t)i * k_cap, vals, (size_t)k_eff * sizeof(float));
    }
    return k_eff;
}

/* ell_decode_seq_full with a device top-k readback (mm26). On an armed
 * context whose sampled path engaged, writes k_eff ids+logits per row at
 * k_cap stride and returns k_eff. Otherwise falls back IN THE SAME CALL to
 * copying full logits rows into vals_out (vocab stride - the caller sizes
 * vals_out for n*vocab floats either way) and returns 0. */
int32_t ell_decode_seq_topk(void *ctx, void *model, int32_t seq_id, int32_t pos0,
                            const int32_t *tokens, int32_t n, int32_t k_cap,
                            int32_t *ids_out, float *vals_out) {
    struct llama_context *lctx = (struct llama_context *)ctx;
    if (n <= 0 || pos0 < 0 || k_cap <= 0 || (uint32_t)n > llama_n_batch(lctx)) {
        return -1;
    }
    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t t = 0; t < n; t++) {
        batch.token[t]     = tokens[t];
        batch.pos[t]       = pos0 + t;
        batch.n_seq_id[t]  = 1;
        batch.seq_id[t][0] = seq_id;
        batch.logits[t]    = 1;
    }
    batch.n_tokens = n;
    int32_t rc = llama_decode(lctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        return rc;
    }
    int32_t k_eff;
    {
        /* every token is an output row: idx i = token i */
        int32_t *idxs = malloc((size_t)n * sizeof(int32_t));
        for (int32_t t = 0; t < n; t++) { idxs[t] = t; }
        k_eff = ell_topk_readback(lctx, n, idxs, k_cap, ids_out, vals_out);
        free(idxs);
    }
    if (k_eff == 0) {
        const size_t row = (size_t)ell_n_vocab(model);
        for (int32_t t = 0; t < n; t++) {
            const float *lg = llama_get_logits_ith(lctx, t);
            if (!lg) {
                llama_batch_free(batch);
                return -2;
            }
            memcpy(vals_out + (size_t)t * row, lg, row * sizeof(float));
        }
    }
    llama_batch_free(batch);
    return k_eff;
}

/* ell_decode_batch with a device top-k readback (mm26); same contract as
 * ell_decode_seq_topk (k_eff > 0 = k_cap-strided ids+vals; 0 = full logits
 * rows in vals_out at vocab stride). */
int32_t ell_decode_batch_topk(void *ctx, void *model, int32_t n_items,
                              const int32_t *seq_ids, const int32_t *counts,
                              const int32_t *positions, const int32_t *tokens_flat,
                              int32_t k_cap, int32_t *ids_out, float *vals_out) {
    struct llama_context *lctx = (struct llama_context *)ctx;
    if (n_items <= 0 || k_cap <= 0) {
        return -1;
    }
    int32_t total = 0;
    for (int32_t i = 0; i < n_items; i++) {
        if (counts[i] <= 0 || positions[i] < 0) {
            return -1;
        }
        total += counts[i];
    }
    if ((uint32_t)total > llama_n_batch(lctx)) {
        return -1;
    }
    struct llama_batch batch = llama_batch_init(total, 0, 1);
    int32_t cursor = 0;
    for (int32_t i = 0; i < n_items; i++) {
        for (int32_t t = 0; t < counts[i]; t++) {
            batch.token[cursor]     = tokens_flat[cursor];
            batch.pos[cursor]       = positions[i] + t;
            batch.n_seq_id[cursor]  = 1;
            batch.seq_id[cursor][0] = seq_ids[i];
            batch.logits[cursor]    = (int8_t)(t == counts[i] - 1);
            cursor++;
        }
    }
    batch.n_tokens = total;
    int32_t rc = llama_decode(lctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        return rc;
    }
    int32_t k_eff;
    {
        /* each item outputs only its LAST token */
        int32_t *idxs = malloc((size_t)n_items * sizeof(int32_t));
        int32_t c = 0;
        for (int32_t i = 0; i < n_items; i++) { c += counts[i]; idxs[i] = c - 1; }
        k_eff = ell_topk_readback(lctx, n_items, idxs, k_cap, ids_out, vals_out);
        free(idxs);
    }
    if (k_eff == 0) {
        const size_t row = (size_t)ell_n_vocab(model);
        cursor = 0;
        for (int32_t i = 0; i < n_items; i++) {
            cursor += counts[i];
            const float *lg = llama_get_logits_ith(lctx, cursor - 1);
            if (!lg) {
                llama_batch_free(batch);
                return -2;
            }
            memcpy(vals_out + (size_t)i * row, lg, row * sizeof(float));
        }
    }
    llama_batch_free(batch);
    return k_eff;
}

int32_t ell_seq_rewind(void *ctx, int32_t seq_id, int32_t n_keep) {
    /* llama_memory_seq_rm REFUSES a partial removal it cannot honor (recurrent/
     * hybrid state keeps no per-token history) and mutates nothing on refusal -
     * propagate that so callers never continue against a tail they believe
     * gone. Full-range removals (n_keep 0) always succeed. With
     * ENCLAVE_GGML_N_RS_SEQ > 0 (ell_rewind_depth > 0) a partial removal of up
     * to that many trailing tokens SUCCEEDS on rollback-capable hybrid archs:
     * the recurrent state restores from the per-token snapshot groups the last
     * decode wrote. */
    bool ok = llama_memory_seq_rm(llama_get_memory((struct llama_context *)ctx),
                                  seq_id, n_keep, -1);
    return ok ? 0 : -1;
}

/* ---- hidden-state export (mm24: medusa-head training harvest) ----------
 * Frozen-trunk draft heads train on (h_t -> future token) pairs, and h_t
 * is just inference output: toggle embeddings, run a chunk, read rows.
 * llama outputs ALL positions when embeddings are enabled, so a prefill
 * chunk yields one row per token at prefill speed - the whole training
 * corpus harvests at ~20x decode speed. */
void ell_set_embeddings(void *ctx, int32_t on) {
    llama_set_embeddings((struct llama_context *)ctx, on != 0);
}

int32_t ell_n_embd(void *model) {
    return llama_model_n_embd((const struct llama_model *)model);
}

/* copy position i's last-layer hidden row (n_embd floats) after a decode
 * that ran with embeddings enabled; returns 0, or -1 when the row is
 * unavailable (embeddings off, i out of range). */
int32_t ell_hidden_row(void *ctx, int32_t i, float *out, int32_t cap) {
    struct llama_context *lctx = (struct llama_context *)ctx;
    const float *e = llama_get_embeddings_ith(lctx, i);
    if (!e || cap < 0) {
        return -1;
    }
    /* caller passes cap = n_embd; trust but bound */
    memcpy(out, e, (size_t)cap * sizeof(float));
    return 0;
}

int32_t ell_cuda_sync_stats(int64_t out[2]) {
    /* cumulative [sync_us, sync_calls] from the CUDA module's sync-instr
     * counters (mm20). The module is dlopened RTLD_LOCAL, so the getter
     * travels ggml's backend proc registry rather than the dynamic symbol
     * table. Absence (older module, no GPU) reports -1/-1 - callers surface
     * it as "engine predates the instrument", never an error. */
    typedef int64_t (*fn_t)(int64_t *);
    static fn_t fn;
    static int looked;
    if (!looked) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (reg) {
                fn = (fn_t)ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_cuda_sync_stats");
            }
            break;
        }
        looked = 1;
    }
    if (!fn) {
        out[0] = -1;
        out[1] = -1;
        return -1;
    }
    int64_t calls = 0;
    out[0] = fn(&calls);
    out[1] = calls;
    return 0;
}

int32_t ell_rewind_depth(void *ctx) {
    /* the EFFECTIVE snapshot depth of this context (llama zeroes the request
     * on archs without rollback), i.e. how many trailing tokens of the most
     * recent decode ell_seq_rewind can drop on a recurrent/hybrid model */
    return (int32_t)llama_n_rs_seq((struct llama_context *)ctx);
}

void ell_seq_copy(void *ctx, int32_t src_seq, int32_t dst_seq) {
    llama_memory_t mem = llama_get_memory((struct llama_context *)ctx);
    /* rm_all on dst is the always-supported removal shape (recurrent included),
     * then adopt src's cells. For attention KV this is metadata (shared cells);
     * for recurrent state it is copy-on-write: dst shares src's state cell and
     * diverges into a free cell only when dst next decodes. */
    llama_memory_seq_rm(mem, dst_seq, -1, -1);
    llama_memory_seq_cp(mem, src_seq, dst_seq, -1, -1);
}

int32_t ell_model_recurrent(void *model) {
    const struct llama_model *m = (const struct llama_model *)model;
    return (llama_model_is_recurrent(m) || llama_model_is_hybrid(m)) ? 1 : 0;
}

/* ---- MTP (multi-token prediction) driver ---------------------------------
 * The model's own trained next-token head drafts for it: near-zero proposal
 * cost, no separate draft model. Single-head, non-shared-memory mode only
 * (qwen3.5/3.6); the head is dense attention even on hybrid trunks, so its
 * tiny KV rewinds freely. Head-KV hygiene beats upstream's driver: only
 * ACCEPTED tokens are ever mirrored into the head (observe), so rejected
 * proposals never pollute its attention. */

/* The nextn API lives in llama's C++ staging header (src/llama-ext.h), so
 * libllama exports it MANGLED. Binding the two functions by their Itanium
 * mangled names (verified against the PINNED build - `nm -D libllama.so`)
 * keeps the shim plain C and the toolchain unchanged; a pin bump that
 * changes them makes ell_mtp_new return NULL, which callers treat as
 * "model has no MTP" - speculation degrades to plain decode, fail-safe. */
typedef void   (*ell_nextn_set_fn)(struct llama_context *, bool, bool);
typedef float *(*ell_nextn_ith_fn)(struct llama_context *, int32_t);
static ell_nextn_set_fn ell_nextn_set;
static ell_nextn_ith_fn ell_nextn_ith;

static int ell_mtp_bind(void) {
    if (ell_nextn_set && ell_nextn_ith) {
        return 1;
    }
    ell_nextn_set = (ell_nextn_set_fn)dlsym(
        RTLD_DEFAULT, "_Z26llama_set_embeddings_nextnP13llama_contextbb");
    ell_nextn_ith = (ell_nextn_ith_fn)dlsym(
        RTLD_DEFAULT, "_Z30llama_get_embeddings_nextn_ithP13llama_contexti");
    return ell_nextn_set != NULL && ell_nextn_ith != NULL;
}

/* Every head sequence needs a sampler, including IDs used by prefix parks.
 * The chains must outlive the context, so ell_mtp owns the array. */
struct ell_mtp {
    int32_t dev_sample;
    struct llama_sampler **smpl;
    int32_t n_smpl;
    struct llama_context *head;
    void *model;
    int32_t n_embd;
    int32_t n_seq;
    float *pending_h;     /* [n_seq][n_embd]: h at each seq's last mirrored pos */
    float **verify_h;     /* per seq: harvested target nextn rows */
    int32_t *verify_rows;
    int32_t *verify_cap;
    struct llama_batch batch; /* token+embd batch, token side malloc'd (see init) */
    int32_t batch_cap;
};

int32_t ell_mtp_available(void *model) {
    return llama_model_n_layer_nextn((const struct llama_model *)model) > 0 ? 1 : 0;
}

void *ell_mtp_new(void *model, void *target_ctx, uint32_t n_ctx, uint32_t n_batch,
                  uint32_t n_seq_max, int32_t type_k, int32_t type_v, int32_t flash_attn) {
    struct llama_model *lm = (struct llama_model *)model;
    if (n_seq_max > INT32_MAX || llama_model_n_layer_nextn(lm) <= 0 || !ell_mtp_bind()) {
        return NULL;
    }
    struct llama_context_params p = llama_context_default_params();
    p.n_threads = ell_n_threads();
    p.n_threads_batch = ell_n_threads_batch();
    p.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    p.n_ctx = n_ctx;
    if (n_batch) { p.n_batch = n_batch; }
    if (n_seq_max) { p.n_seq_max = n_seq_max; }
    p.kv_unified = true;
    p.type_k = ell_ggml_kv_type(type_k);
    p.type_v = ell_ggml_kv_type(type_v);
    switch (flash_attn) {
        case ELL_FA_ENABLED:  p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  break;
        case ELL_FA_DISABLED: p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; break;
        default:              p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;     break;
    }
    /* ENCLAVE_MTP_DEV_SAMPLE=1: sample the draft steps ON DEVICE. llama skips
     * the 248K-vocab logits copy to host entirely when every output sequence
     * has a backend sampler chain (needs_raw_logits, llama-context.cpp) - and
     * that copy is 77% of a head step: measured 3.67 -> 0.77 ms on a 9b, all
     * of it transfer rather than the one-block head's arithmetic.
     *
     * OFF by default. It suppresses the p_min confidence gate, because the
     * gate needs the raw row the whole point is not to fetch - harmless at
     * k=1 (one proposal, nothing to truncate) but a real behaviour change at
     * larger k, where forcing full-length drafts measured WORSE (acceptance
     * 75%% -> 57%% on the fleet). The chains must be sampler CHAINS and must
     * outlive the context. Falls back to the host argmax path if the backend
     * refuses the sampler graph (it needs extra compute buffers). */
    const char *ds_env = getenv("ENCLAVE_MTP_DEV_SAMPLE");
    int32_t want_dev = (ds_env && *ds_env == '1') ? 1 : 0;
    int32_t ns_dev = (int32_t)(n_seq_max ? n_seq_max : 1);
    struct llama_sampler **chains = NULL;
    struct llama_sampler_seq_config *scfg = NULL;
    if (want_dev) {
        chains = calloc((size_t)ns_dev, sizeof(*chains));
        scfg = calloc((size_t)ns_dev, sizeof(*scfg));
        if (!chains || !scfg) { want_dev = 0; }
    }
    if (want_dev) {
        for (int32_t s = 0; s < ns_dev; s++) {
            struct llama_sampler_chain_params cp = llama_sampler_chain_default_params();
            cp.no_perf = true;
            chains[s] = llama_sampler_chain_init(cp);
            if (!chains[s]) { want_dev = 0; break; }
            llama_sampler_chain_add(chains[s], llama_sampler_init_greedy());
            scfg[s].seq_id  = s;
            scfg[s].sampler = chains[s];
        }
    }
    if (want_dev) {
        p.samplers   = scfg;
        p.n_samplers = (size_t)ns_dev;
    }
    struct llama_context *head = llama_init_from_model(lm, p);
    if (!head && want_dev) {
        p.samplers = NULL;
        p.n_samplers = 0;
        for (int32_t s = 0; s < ns_dev; s++) {
            if (chains[s]) { llama_sampler_free(chains[s]); chains[s] = NULL; }
        }
        want_dev = 0;
        head = llama_init_from_model(lm, p);
    }
    free(scfg);
    struct ell_mtp *m = head ? calloc(1, sizeof(*m)) : NULL;
    if (!m && head) { llama_free(head); head = NULL; }
    if (!m || !want_dev) {
        if (chains) {
            for (int32_t s = 0; s < ns_dev; s++) { if (chains[s]) { llama_sampler_free(chains[s]); } }
            free(chains);
            chains = NULL;
        }
        if (!m) {
            return NULL;
        }
    }
    m->head = head;
    m->dev_sample = want_dev;
    m->smpl = chains;
    m->n_smpl = want_dev ? ns_dev : 0;
    m->model = model;
    m->n_embd = llama_model_n_embd(lm);
    m->n_seq = (int32_t)(n_seq_max ? n_seq_max : 1);
    m->pending_h = calloc((size_t)m->n_seq * m->n_embd, sizeof(float));
    m->verify_h = calloc(m->n_seq, sizeof(float *));
    m->verify_rows = calloc(m->n_seq, sizeof(int32_t));
    m->verify_cap = calloc(m->n_seq, sizeof(int32_t));
    m->batch_cap = (int32_t)(n_batch ? n_batch : 512);
    /* llama_batch_init allocates only ONE of token/embd (embd_dim nonzero =
     * embd); MTP pairs (h, x) so it needs both - add the token side by hand
     * and free it by hand (llama_batch_free would free it too, but keep
     * ownership explicit and symmetric with the upstream driver). */
    m->batch = llama_batch_init(m->batch_cap, m->n_embd, 1);
    m->batch.token = malloc(sizeof(llama_token) * (size_t)m->batch_cap);
    /* target must emit nextn hidden rows (unmasked); the head emits its own
     * (masked) to feed proposals forward */
    ell_nextn_set((struct llama_context *)target_ctx, true, false);
    ell_nextn_set(head, true, true);
    return m;
}

void ell_mtp_free(void *mp) {
    struct ell_mtp *m = (struct ell_mtp *)mp;
    if (!m) { return; }
    free(m->batch.token);
    m->batch.token = NULL;
    llama_batch_free(m->batch);
    llama_free(m->head);
    for (int32_t s = 0; s < m->n_smpl; s++) {
        if (m->smpl[s]) { llama_sampler_free(m->smpl[s]); m->smpl[s] = NULL; }
    }
    free(m->smpl);
    for (int32_t s = 0; s < m->n_seq; s++) { free(m->verify_h[s]); }
    free(m->verify_h);
    free(m->verify_rows);
    free(m->verify_cap);
    free(m->pending_h);
    free(m);
}

void ell_mtp_harvest(void *mp, void *target_ctx, int32_t seq, int32_t n_rows) {
    struct ell_mtp *m = (struct ell_mtp *)mp;
    if (seq < 0 || seq >= m->n_seq || n_rows <= 0) { return; }
    if (m->verify_cap[seq] < n_rows) {
        m->verify_h[seq] = realloc(m->verify_h[seq],
                                   (size_t)n_rows * m->n_embd * sizeof(float));
        m->verify_cap[seq] = n_rows;
    }
    for (int32_t i = 0; i < n_rows; i++) {
        const float *h = ell_nextn_ith((struct llama_context *)target_ctx, i);
        if (!h) { m->verify_rows[seq] = 0; return; }
        memcpy(m->verify_h[seq] + (size_t)i * m->n_embd, h,
               (size_t)m->n_embd * sizeof(float));
    }
    m->verify_rows[seq] = n_rows;
}

static int32_t ell_mtp_observe_impl(struct ell_mtp *m, int32_t seq, int32_t pos0,
                                    const int32_t *tokens, int32_t n);

int32_t ell_mtp_observe(void *mp, int32_t seq, int32_t pos0,
                        const int32_t *tokens, int32_t n) {
    return ell_mtp_observe_impl((struct ell_mtp *)mp, seq, pos0, tokens, n);
}

static int32_t ell_mtp_observe_impl(struct ell_mtp *m, int32_t seq, int32_t pos0,
                                    const int32_t *tokens, int32_t n) {
    if (!m || seq < 0 || seq >= m->n_seq || n <= 0 || n > m->batch_cap || pos0 < 0) {
        return -1;
    }
    if (m->verify_rows[seq] < n) {
        return -1; /* rows 0..n-2 pair tokens 1..n-1; row n-1 becomes pending */
    }
    /* drop anything at/after pos0 (a previous round's proposals) - the head
     * KV is plain attention, partial removal always succeeds */
    llama_memory_seq_rm(llama_get_memory(m->head), seq, pos0, -1);
    const size_t row = (size_t)m->n_embd;
    for (int32_t j = 0; j < n; j++) {
        m->batch.token[j]     = tokens[j];
        m->batch.pos[j]       = pos0 + j;
        m->batch.n_seq_id[j]  = 1;
        m->batch.seq_id[j][0] = seq;
        m->batch.logits[j]    = 0;
        const float *h = (j == 0) ? m->pending_h + (size_t)seq * row
                                  : m->verify_h[seq] + (size_t)(j - 1) * row;
        memcpy(m->batch.embd + (size_t)j * row, h, row * sizeof(float));
    }
    m->batch.n_tokens = n;
    int32_t rc = llama_decode(m->head, m->batch);
    if (rc != 0) {
        return rc;
    }
    memcpy(m->pending_h + (size_t)seq * row,
           m->verify_h[seq] + (size_t)(n - 1) * row, row * sizeof(float));
    return 0;
}

int32_t ell_mtp_draft(void *mp, int32_t seq, int32_t id_last, int32_t n_past,
                      int32_t k, float p_min, int32_t *tokens_out) {
    struct ell_mtp *m = (struct ell_mtp *)mp;
    if (!m || seq < 0 || seq >= m->n_seq || k <= 0 || n_past < 0) {
        return 0;
    }
    /* proposals live at n_past.. and are dropped by the next observe(); still
     * start clean in case the previous round never observed (early exit) */
    llama_memory_seq_rm(llama_get_memory(m->head), seq, n_past, -1);
    const size_t row = (size_t)m->n_embd;
    const int32_t n_vocab = ell_n_vocab(m->model);
    int32_t tok = id_last;
    const float *h = m->pending_h + (size_t)seq * row;
    int32_t n = 0;
    for (int32_t i = 0; i < k; i++) {
        m->batch.token[0]     = tok;
        m->batch.pos[0]       = n_past + i;
        m->batch.n_seq_id[0]  = 1;
        m->batch.seq_id[0][0] = seq;
        m->batch.logits[0]    = 1;
        memcpy(m->batch.embd, h, row * sizeof(float));
        m->batch.n_tokens = 1;
        if (llama_decode(m->head, m->batch) != 0) {
            break;
        }
        int32_t best = 0;
        float lmax = 0.0f;
        const float *lg = NULL;
        if (m->dev_sample) {
            /* the row never left the device - llama sampled it there */
            best = (int32_t)llama_get_sampled_token_ith(m->head, 0);
            if (best < 0) { break; }
        } else {
            lg = llama_get_logits_ith(m->head, 0);
            if (!lg) { break; }
            lmax = lg[0];
            for (int32_t v = 1; v < n_vocab; v++) {
                if (lg[v] > lmax) { lmax = lg[v]; best = v; }
            }
        }
        /* confidence gate (p = 1/sum(exp(l-max))). p_min <= 0 means "always
         * draft the full k": the gate can never trip, so skip the pass
         * outright. Full-length drafts also keep every verify pass the same
         * ubatch shape, which is what lets the CUDA backend keep replaying
         * its captured graph instead of re-warming.
         *
         * The sum is EXACT but cheap (mm19): exp() only for logits within
         * 16 of the max - anything below contributes < 1.2e-7 each, and the
         * whole 248K tail is bounded by +0.03, absorbed as slack. The naive
         * full-vocab exp() sum measured ~2.7 ms per drafted token on
         * SNP-throttled vCPUs (fleet, 2026-08-03); this pass is comparisons
         * plus a handful of exp() calls. */
        if (p_min > 0.0f && lg) {
            const float floor_l = lmax - 16.0f;
            double sum = 0.03; /* tail slack: 248K * 1.2e-7 */
            for (int32_t v = 0; v < n_vocab; v++) {
                if (lg[v] >= floor_l) { sum += exp((double)(lg[v] - lmax)); }
            }
            if ((float)(1.0 / sum) < p_min) {
                break; /* not confident - stop proposing */
            }
        }
        tokens_out[n++] = best;
        tok = best;
        h = ell_nextn_ith(m->head, 0);
        if (!h) { break; }
    }
    return n;
}

/* observe-fold (mm19): the previous round's ACCEPTED tokens ride the next
 * draft call - one shim entry, one head-context session, instead of a
 * separate mtp_accept round trip (measured 9.5-14 ms/round on the fleet,
 * mostly boundary and sync). Semantically identical to
 * ell_mtp_observe(obs_*) followed by ell_mtp_draft(...): the observe
 * mirrors accepted tokens into the head KV and re-seeds pending_h, then
 * the draft chain runs exactly as before. obs_n 0 = plain draft. Returns
 * the draft count; a FAILED observe returns 0 drafts (the caller falls
 * back to plain decode for the round and the next explicit observe
 * resynchronizes the head). */
int32_t ell_mtp_draft2(void *mp, int32_t seq, int32_t id_last, int32_t n_past,
                       int32_t k, float p_min, int32_t *tokens_out,
                       int32_t obs_pos0, const int32_t *obs_tokens, int32_t obs_n) {
    struct ell_mtp *m = (struct ell_mtp *)mp;
    if (!m) { return 0; }
    if (obs_n > 0) {
        if (ell_mtp_observe_impl(m, seq, obs_pos0, obs_tokens, obs_n) != 0) {
            return 0;
        }
    }
    return ell_mtp_draft(mp, seq, id_last, n_past, k, p_min, tokens_out);
}

void ell_mtp_reset(void *mp, int32_t seq) {
    struct ell_mtp *m = (struct ell_mtp *)mp;
    if (seq < 0 || seq >= m->n_seq) { return; }
    llama_memory_seq_rm(llama_get_memory(m->head), seq, -1, -1);
    memset(m->pending_h + (size_t)seq * m->n_embd, 0,
           (size_t)m->n_embd * sizeof(float));
    m->verify_rows[seq] = 0;
}

/* ---- vision (multimodal input) -------------------------------------------
 * Thin wrapper over libmtmd: the caller passes raw image FILE bytes and a
 * position, mtmd does preprocessing, the vision encoder, the projector, the
 * model's marker tokens and the mask/M-RoPE handling, and the caller learns
 * only how many positions the sequence advanced. See the header for the
 * contract and for why these are optional symbols. */

struct ell_mtmd {
    mtmd_context *ctx;
};

int32_t ell_mtmd_caps_file(const char *mmproj_path) {
    if (!mmproj_path) { return -1; }
    struct mtmd_caps c = mtmd_get_cap_from_file(mmproj_path);
    if (!c.inp_vision && !c.inp_audio) {
        return -1; /* readable GGUFs that project nothing are not mmprojs */
    }
    return (c.inp_vision ? 1 : 0) | (c.inp_audio ? 2 : 0);
}

void *ell_mtmd_new(void *model, const char *mmproj_path, int32_t n_threads,
                   int32_t use_gpu, int32_t image_max_tokens) {
    if (!model || !mmproj_path) { return NULL; }
    struct mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu = use_gpu != 0;
    /* the tenant log is the deployment's log: keep per-image encode timings
     * out of it, the backend already times the whole eval */
    p.print_timings = false;
    if (n_threads > 0) { p.n_threads = n_threads; }
    if (image_max_tokens > 0) { p.image_max_tokens = image_max_tokens; }
    mtmd_context *c = mtmd_init_from_file(mmproj_path,
                                          (const struct llama_model *)model, p);
    if (!c) { return NULL; }
    if (!mtmd_support_vision(c)) {
        /* an audio-only projector against an image verb: refuse at load
         * rather than per request */
        mtmd_free(c);
        return NULL;
    }
    struct ell_mtmd *m = calloc(1, sizeof(*m));
    if (!m) { mtmd_free(c); return NULL; }
    m->ctx = c;
    return m;
}

void ell_mtmd_free(void *mp) {
    struct ell_mtmd *m = (struct ell_mtmd *)mp;
    if (!m) { return; }
    mtmd_free(m->ctx);
    free(m);
}

int32_t ell_mtmd_eval_image(void *mp, void *lctx, int32_t seq_id, int32_t pos0,
                            const uint8_t *bytes, uint32_t len, int32_t n_batch,
                            int32_t *n_pos_out) {
    struct ell_mtmd *m = (struct ell_mtmd *)mp;
    struct llama_context *c = (struct llama_context *)lctx;
    if (!m || !c || !bytes || len == 0 || pos0 < 0 || n_batch <= 0 || !n_pos_out) {
        return -1;
    }
    /* placeholder=false: actually decode the pixels (a placeholder bitmap
     * carries only geometry, for cache lookups we do not do) */
    struct mtmd_helper_bitmap_wrapper w =
        mtmd_helper_bitmap_init_from_buf(m->ctx, bytes, (size_t)len, false);
    if (w.video_ctx) {
        /* video needs the C++ helper loop and an ffmpeg binary in the enclave;
         * neither exists here, and a half-decoded video is worse than a clear
         * refusal */
        mtmd_helper_video_free(w.video_ctx);
        if (w.bitmap) { mtmd_bitmap_free(w.bitmap); }
        return 2;
    }
    if (!w.bitmap) {
        return 2;
    }
    if (mtmd_bitmap_is_audio(w.bitmap)) {
        mtmd_bitmap_free(w.bitmap);
        return 2;
    }
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (!chunks) {
        mtmd_bitmap_free(w.bitmap);
        return 1;
    }
    /* Text is JUST the media marker, so mtmd emits exactly the chunks this
     * model wants around one image and nothing else: the caller renders its
     * own chat template and feeds the surrounding text itself. add_special is
     * off for the same reason (no BOS in the middle of a conversation).
     * Zero-init, then EVERY field named: ddd4ec14 added text_len (tokenize
     * reads exactly that many bytes, no NUL fallback) and an uninitialized
     * field here made every image on the fleet "not readable" - the marker
     * string came out garbage-length, tokenize failed, and the shim's rc 2
     * blamed the decoder. A future field added to this struct must default
     * to zero, not stack garbage. */
    struct mtmd_input_text txt = {0};
    const char *marker = mtmd_default_marker();
    txt.text = marker;
    txt.text_len = strlen(marker);
    txt.add_special = false;
    txt.parse_special = true;
    const mtmd_bitmap *bmps[1];
    bmps[0] = w.bitmap;
    int32_t rc = mtmd_tokenize(m->ctx, chunks, &txt, bmps, 1);
    mtmd_bitmap_free(w.bitmap);
    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        return 2; /* 1 = marker/bitmap count mismatch (impossible here), 2 = preprocessing */
    }
    /* A non-causal image chunk (gemma-style) must land in ONE ubatch or the
     * mask is wrong, and mtmd's helper splits at n_batch without knowing that.
     * Catch it here with a distinct code instead of returning a quietly
     * degraded answer. */
    const uint32_t n_ubatch = llama_n_ubatch(c);
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk *ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            continue;
        }
        if (mtmd_decode_use_non_causal(m->ctx, ch) &&
            mtmd_input_chunk_get_n_tokens(ch) > (size_t)n_ubatch) {
            mtmd_input_chunks_free(chunks);
            return 3;
        }
    }
    llama_pos new_n_past = pos0;
    /* logits_last=false: the image is never the end of a prompt, the caller's
     * text turn follows and produces the row it samples from */
    rc = mtmd_helper_eval_chunks(m->ctx, c, chunks, (llama_pos)pos0,
                                 (llama_seq_id)seq_id, n_batch, false, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (rc != 0) {
        return 1;
    }
    *n_pos_out = (int32_t)(new_n_past - (llama_pos)pos0);
    return 0;
}

/* mm33: ONE video file -> sampled frames into seq_id at pos0. Mirrors
 * ell_mtmd_eval_image, except the bitmaps come out of libmtmd's video helper
 * (an ffmpeg/ffprobe subprocess pair) and the tokenize text carries one
 * marker per frame with the helper's timestamp texts spliced in between, so
 * mtmd emits exactly the chunks this model wants around a frame sequence.
 * Frames are read EAGERLY and capped: the helper's own lazy-bitmap path would
 * happily hand a two-hour clip to the encoder, and here every frame is an
 * encoder pass plus its share of the KV pool. */
int32_t ell_mtmd_eval_video(void *mp, void *lctx, int32_t seq_id, int32_t pos0,
                            const uint8_t *bytes, uint32_t len, int32_t n_batch,
                            int32_t fps_milli, int32_t max_frames, int32_t timestamp_ms,
                            const char *ffmpeg_dir,
                            int32_t *n_pos_out, int32_t *n_frames_out) {
    struct ell_mtmd *m = (struct ell_mtmd *)mp;
    struct llama_context *c = (struct llama_context *)lctx;
    if (!m || !c || !bytes || len == 0 || pos0 < 0 || n_batch <= 0 || !n_pos_out || !n_frames_out) {
        return -1;
    }
    if (max_frames <= 0) { max_frames = 8; }
    if (max_frames > 64) { max_frames = 64; }
    if (!mtmd_helper_support_video(m->ctx)) {
        return 4;
    }
    struct mtmd_helper_video_init_params p = mtmd_helper_video_init_params_default();
    if (fps_milli > 0) { p.fps_target = (float)fps_milli / 1000.0f; }
    p.ffmpeg_bin_dir = (ffmpeg_dir && ffmpeg_dir[0]) ? ffmpeg_dir : NULL;
    p.timestamp_interval_ms = timestamp_ms > 0 ? (int64_t)timestamp_ms : 0;
    mtmd_helper_video *v = mtmd_helper_video_init_from_buf(m->ctx, bytes, (size_t)len, p);
    if (!v) {
        return 2; /* ffprobe missing, or the bytes are not a video */
    }
    const char *marker = mtmd_default_marker();
    const size_t mlen = strlen(marker);
    const mtmd_bitmap **bmps = (const mtmd_bitmap **)calloc((size_t)max_frames, sizeof(*bmps));
    size_t cap = 256 + (size_t)max_frames * (mlen + 32);
    char *text = (char *)malloc(cap);
    size_t tl = 0;
    int32_t nf = 0;
    int32_t rc = 0;
    if (!bmps || !text) {
        rc = 1;
    }
    while (rc == 0) {
        mtmd_bitmap *bm = NULL;
        char *tx = NULL;
        int32_t r = mtmd_helper_video_read_next(v, &bm, &tx);
        if (r == -1) { break; }          /* EOF */
        if (r != 0) { rc = 2; break; }   /* decode error */
        if (tx) {
            size_t n = strlen(tx);
            if (tl + n + mlen + 1 > cap) {
                cap = (cap + n + mlen) * 2;
                char *nt = (char *)realloc(text, cap);
                if (!nt) { free(tx); rc = 1; break; }
                text = nt;
            }
            memcpy(text + tl, tx, n);
            tl += n;
            free(tx);
        }
        if (bm) {
            if (nf >= max_frames) { mtmd_bitmap_free(bm); break; }   /* cap: stop reading */
            bmps[nf++] = bm;
            if (tl + mlen + 1 > cap) {
                cap = (cap + mlen) * 2;
                char *nt = (char *)realloc(text, cap);
                if (!nt) { rc = 1; break; }
                text = nt;
            }
            memcpy(text + tl, marker, mlen);
            tl += mlen;
        }
    }
    mtmd_helper_video_free(v);
    if (rc == 0 && nf == 0) { rc = 2; }
    mtmd_input_chunks *chunks = NULL;
    if (rc == 0) {
        text[tl] = 0;
        chunks = mtmd_input_chunks_init();
        if (!chunks) { rc = 1; }
    }
    if (rc == 0) {
        struct mtmd_input_text txt = {0};
        txt.text = text;
        txt.text_len = tl;
        txt.add_special = false;
        txt.parse_special = true;
        if (mtmd_tokenize(m->ctx, chunks, &txt, bmps, (size_t)nf) != 0) { rc = 2; }
    }
    for (int32_t i = 0; i < nf; i++) { mtmd_bitmap_free((mtmd_bitmap *)bmps[i]); }
    free(bmps);
    free(text);
    if (rc != 0) {
        if (chunks) { mtmd_input_chunks_free(chunks); }
        return rc;
    }
    const uint32_t n_ubatch = llama_n_ubatch(c);
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk *ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) { continue; }
        if (mtmd_decode_use_non_causal(m->ctx, ch) &&
            mtmd_input_chunk_get_n_tokens(ch) > (size_t)n_ubatch) {
            mtmd_input_chunks_free(chunks);
            return 3;
        }
    }
    llama_pos new_n_past = pos0;
    rc = mtmd_helper_eval_chunks(m->ctx, c, chunks, (llama_pos)pos0,
                                 (llama_seq_id)seq_id, n_batch, false, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (rc != 0) {
        return 1;
    }
    *n_pos_out = (int32_t)(new_n_past - (llama_pos)pos0);
    *n_frames_out = nf;
    return 0;
}
