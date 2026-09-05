/*
 * shielded-run -- drive a whole GGUF through the shielded ggml backend.
 *
 * Exists because engine builds link their backends statically, so
 * ggml_backend_load_all() short-circuits on `if (!ggml_backend_reg_count())` and
 * GGML_BACKEND_PATH is silently ignored. This loads the module explicitly, then
 * runs an ordinary llama.cpp generation on top of it.
 *
 * What it is for is the measurement, not the tokens: run it once with a worker
 * and once without (SHIELDED_PORT pointing nowhere). The completions must be
 * CHARACTER-IDENTICAL, because the offloaded path is exact -- so any difference
 * between them is an offload bug, and any difference from a plain CPU run is the
 * fixed-point encoding, which is a separate and much larger effect. Keeping those
 * two apart is the whole point; conflating them is how "the GPU is wrong" and
 * "the encoding is lossy" get mistaken for each other.
 */
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <thread>
#include <chrono>
typedef void (*stats_fn)(uint64_t*, uint64_t*, uint64_t*, uint64_t*);
static stats_fn shielded_stats = nullptr;

int main(int argc, char **argv) {
    const char *backend = getenv("SHIELDED_SO");
    const char *model_path = argc > 1 ? argv[1] : nullptr;
    const char *prompt = argc > 2 ? argv[2] : "The capital of France is";
    int n_predict = argc > 3 ? atoi(argv[3]) : 8;
    if (!model_path) { fprintf(stderr, "usage: shielded-run <model.gguf> [prompt] [n]\n"); return 2; }

    /* The CPU backend is a separate module in a shared-library build of ggml, and
     * once any module is registered ggml_backend_load_all() stops looking. Load
     * it explicitly, first, so the in-enclave half of the graph has somewhere
     * to run. */
    if (const char *cpu_so = getenv("GGML_CPU_SO")) {
        fprintf(stderr, "[run] cpu backend: %s\n", ggml_backend_load(cpu_so) ? "loaded" : "FAILED TO LOAD");
    }
    if (backend) {
        ggml_backend_reg_t r = ggml_backend_load(backend);
        fprintf(stderr, "[run] shielded backend: %s\n", r ? "loaded" : "FAILED TO LOAD");
        if (!r) return 2;
        /* The counters live in the module ggml dlopened, so reach them the
         * same way rather than linking against it. */
        void *h = dlopen(backend, RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen(backend, RTLD_NOW);
        shielded_stats = h ? (stats_fn)dlsym(h, "ggml_backend_shielded_stats") : nullptr;
        fprintf(stderr, "[run] stats symbol: %s\n", shielded_stats ? "resolved" : "NOT RESOLVED");
    }
    llama_backend_init();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++)
        fprintf(stderr, "[run] device %zu: %s\n", i, ggml_backend_dev_name(ggml_backend_dev_get(i)));

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                       /* no in-enclave card; ACCEL is separate */
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 8; cp.n_threads_batch = 8;
    /* SHIELDED_RUN_THREADS: the CPU backend's thread count. On a phone every
     * scheduler split spins the CPU threadpool up and down (no persistent
     * pool here), so this is the knob that exposes that cost. */
    if (const char *t = getenv("SHIELDED_RUN_THREADS")) { cp.n_threads = cp.n_threads_batch = atoi(t) > 0 ? atoi(t) : 8; }
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 2; }
    /* SHIELDED_RUN_THREADPOOL=1: one persistent CPU threadpool for the whole
     * run. Without it every scheduler split (~150 per token once the matmuls
     * are offloaded) spins the CPU backend's threads up and down; on a phone
     * that is seconds per token (REPORT.md section 12). */
    ggml_threadpool *tp = nullptr;
    if (const char *e = getenv("SHIELDED_RUN_THREADPOOL"); e && *e && strcmp(e, "0")) {
        /* ggml_threadpool_new lives in the CPU backend MODULE (loaded above via
         * GGML_CPU_SO), not in a linked library: resolve it from the process. */
        typedef ggml_threadpool *(*tp_new_fn)(ggml_threadpool_params *);
        /* bionic's RTLD_DEFAULT does not see a library another library dlopened
         * RTLD_LOCAL; open the same module ourselves (same handle) and ask it. */
        void *cpu_h = getenv("GGML_CPU_SO") ? dlopen(getenv("GGML_CPU_SO"), RTLD_NOW) : nullptr;
        tp_new_fn tp_new = (tp_new_fn)dlsym(cpu_h ? cpu_h : RTLD_DEFAULT, "ggml_threadpool_new");
        if (!tp_new) fprintf(stderr, "[run] persistent threadpool: ggml_threadpool_new not found (CPU backend module not loaded?)\n");
        else {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(cp.n_threads);
            tp = tp_new(&tpp);
            llama_attach_threadpool(ctx, tp, tp);
            fprintf(stderr, "[run] persistent threadpool: %d threads\n", cp.n_threads);
        }
    }
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 2; }

    std::vector<llama_token> toks(256);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), (int)toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 2; }
    toks.resize(n);
    fprintf(stderr, "[run] %d prompt tokens\n", n);

    llama_batch batch = llama_batch_get_one(toks.data(), n);
    const int64_t t_pp0 = ggml_time_us();
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode(prompt) failed\n"); return 2; }
    const int64_t t_pp1 = ggml_time_us();

    // Test-only opt-in: compare every logit, not just the selected text, across
    // single-card, pooled and exact CPU runs of the same public prompt.
    FILE *logits_out = nullptr;
    if (const char *path = getenv("SHIELDED_RUN_LOGITS")) {
        logits_out = fopen(path, "wb");
        if (!logits_out) { perror("logits output"); return 2; }
    }
    std::string out;
    llama_token cur = 0;
    int n_gen = 0;
    const int64_t t_tg0 = ggml_time_us();
    for (int i = 0; i < n_predict; i++) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        if (logits_out && fwrite(logits, sizeof(float), (size_t)n_vocab, logits_out) != (size_t)n_vocab) {
            perror("logits write"); return 2;
        }
        int best = 0; float bv = logits[0];
        for (int t = 1; t < n_vocab; t++) if (logits[t] > bv) { bv = logits[t]; best = t; }
        cur = best;
        if (llama_vocab_is_eog(vocab, cur)) break;
        char buf[256];
        int L = llama_token_to_piece(vocab, cur, buf, sizeof buf, 0, false);
        if (L > 0) out.append(buf, L);
        llama_batch b1 = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, b1)) { fprintf(stderr, "decode failed at %d\n", i); break; }
        n_gen++;
        if (const char *delay = getenv("SHIELDED_RUN_STEP_MS")) {
            const int ms = atoi(delay);
            if (ms > 0 && ms <= 1000) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
    }
    const int64_t t_tg1 = ggml_time_us();

    uint64_t off = 0, loc = 0, macs = 0, vf = 0;
    if (shielded_stats) shielded_stats(&off, &loc, &macs, &vf);
    printf("\n=== shielded ===\n");
    printf("prompt      : %s\n", prompt);
    printf("completion  : %s\n", out.c_str());
    printf("offloaded   : %llu nodes\n", (unsigned long long)off);
    printf("local       : %llu nodes\n", (unsigned long long)loc);
    printf("GMAC        : %.2f\n", (double)macs / 1e9);
    printf("verify fail : %llu\n", (unsigned long long)vf);
    printf("prefill     : %d tokens in %.1f ms (%.1f tok/s)\n", n,
           (t_pp1 - t_pp0) / 1e3, n * 1e6 / (double)(t_pp1 - t_pp0));
    if (n_gen)
        printf("decode      : %d tokens in %.1f ms = %.1f ms/tok (%.2f tok/s)\n", n_gen,
               (t_tg1 - t_tg0) / 1e3, (t_tg1 - t_tg0) / 1e3 / n_gen, n_gen * 1e6 / (double)(t_tg1 - t_tg0));
    if (logits_out) fclose(logits_out);
    llama_free(ctx); llama_model_free(model);
    return vf == 0 ? 0 : 1;
}
