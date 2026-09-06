/* bench-spec -- real self-drafting speculative decoding through the engine's
 * own primitives (libenclave_llama), so what is measured is the accept loop
 * the wasi-nn host actually runs, not a reimplementation of it.
 *
 * The round mirrors wasmtime-nn-ggml.patch's mtp_round + the guest's
 * greedy accept:
 *   draft   ell_mtp_draft2(seq, id_last, n_past, K, P_MIN, out, obs...)
 *           with the PREVIOUS round's accepted tokens folded in as the
 *           observe (the mm19 observe-fold).
 *   verify  ONE ell_decode_seq_full of [id_last, d1..dk] at n_past.., logits
 *           for every row (one llama_decode of k+1 rows: THE batch-width
 *           amortiser the shielded tier is after).
 *   accept  the longest prefix of drafts matching the target's greedy pick,
 *           plus the target's own next token from the first mismatching row.
 *   rewind  ell_seq_rewind(seq, n_past + a + 1) drops the rejected tail. The
 *           MTP models are hybrid (qwen35: deltanet + attention) whose
 *           recurrent state keeps no per-token history, so this is only
 *           possible with the no-branch verify: ENCLAVE_GGML_N_RS_SEQ=K
 *           snapshots (set here when unset) let llama roll back up to K
 *           trailing tokens of the last decode. ell_rewind_depth reports
 *           what the context actually got; < K is fatal for this tool.
 *   harvest ell_mtp_harvest(k+1 rows) right after the verify, before any
 *           other target decode overwrites the nextn rows.
 *
 * Greedy speculation is lossless: the tool decodes the same prompt plainly
 * first and asserts the token stream is identical (exit 3 otherwise).
 *
 * Conventions match bench-run: BACKENDS, N_GPU_LAYERS, THREADS, LABEL, one
 * JSON line on stdout. K (default 4), P_MIN (default 0 = always full k).
 */
#include "enclave_llama.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>

static int env_int(const char *k, int d) { const char *v = getenv(k); return v ? atoi(v) : d; }

static void json_text(const std::string &s) {
    for (char c : s) { if (c == '"') printf("\\\""); else if (c == '\\') printf("\\\\"); else if (c == '\n') printf(" "); else if ((unsigned char)c >= 32) putchar(c); }
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : nullptr;
    const char *prompt     = argc > 2 ? argv[2] : "Explain in one paragraph why the sky is blue.";
    const int   n_predict  = argc > 3 ? atoi(argv[3]) : 64;
    if (!model_path) { fprintf(stderr, "usage: bench-spec <model.gguf> [prompt] [n]   (env K, P_MIN)\n"); return 2; }
    const int   K     = env_int("K", 4);
    const float p_min = getenv("P_MIN") ? (float)atof(getenv("P_MIN")) : 0.0f;
    const int   threads = env_int("THREADS", 8);
    const int   seq = env_int("SEQ_ID", 0);
    const int   n_seqs = env_int("N_SEQS", seq + 1);
    const uint32_t n_ctx = 1024, n_batch = 1024;
    if (K < 1 || K > 16) { fprintf(stderr, "K must be 1..16\n"); return 2; }
    if (seq < 0 || seq >= n_seqs || n_seqs > 1024) { fprintf(stderr, "invalid SEQ_ID/N_SEQS\n"); return 2; }
    /* The shim's supported setting reaches both contexts; do not inspect its
     * private struct layout to find the head. High IDs exercise prefix parks. */
    { char b[16]; snprintf(b, sizeof b, "%d", threads); setenv("ENCLAVE_GGML_N_THREADS", b, 1); }
    /* the no-branch verify needs K snapshots of recurrent state (see header) */
    if (!getenv("ENCLAVE_GGML_N_RS_SEQ")) { char b[8]; snprintf(b, sizeof b, "%d", K); setenv("ENCLAVE_GGML_N_RS_SEQ", b, 1); }

    if (const char *list = getenv("BACKENDS")) {
        std::string s(list);
        size_t p = 0;
        while (p <= s.size()) {
            size_t c = s.find(':', p);
            std::string one = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!one.empty())
                fprintf(stderr, "[bench] backend %s: %s\n", one.c_str(),
                        ggml_backend_load(one.c_str()) ? "loaded" : "FAILED");
            if (c == std::string::npos) break;
            p = c + 1;
        }
    }
    llama_backend_init();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++)
        fprintf(stderr, "[bench] device %zu: %s\n", i, ggml_backend_dev_name(ggml_backend_dev_get(i)));

    /* ell_load_model: load_mtp on (llama's default is OFF, which drops the
     * nextn tensors as "unused" and makes ell_mtp_available report 0) */
    void *model = ell_load_model(model_path, env_int("N_GPU_LAYERS", 0));
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const llama_vocab *vocab = llama_model_get_vocab((llama_model *)model);
    const int nv = ell_n_vocab(model);
    fprintf(stderr, "[bench] mtp_available=%d n_layer_nextn=%d recurrent=%d\n",
            ell_mtp_available(model), llama_model_n_layer_nextn((llama_model *)model), ell_model_recurrent(model));
    if (!ell_mtp_available(model)) { fprintf(stderr, "no MTP head in this GGUF (n_layer_nextn=0)\n"); return 4; }

    void *ctx = ell_new_server(model, n_ctx, n_batch, n_seqs, 0, 0, 0);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 2; }
    llama_set_n_threads((llama_context *)ctx, threads, threads);
    const int depth = ell_rewind_depth(ctx);
    fprintf(stderr, "[bench] rewind_depth=%d (need >= %d)\n", depth, K);
    if (ell_model_recurrent(model) && depth < K) {
        fprintf(stderr, "hybrid model without enough rewind depth: the no-branch verify cannot rewind K rejected tokens\n");
        return 2;
    }

    std::vector<int32_t> toks(512);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), (int)toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 2; }
    toks.resize(n);
    auto argmax = [&](const float *lg) { int b = 0; for (int t = 1; t < nv; t++) if (lg[t] > lg[b]) b = t; return b; };
    auto piece = [&](int t) { char buf[256]; int L = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, false); return L > 0 ? std::string(buf, L) : std::string(); };

    std::vector<float> logits((size_t)(K + 1) * nv);
    std::vector<float> plog((size_t)n * nv);

    /* ---- 1. plain greedy reference (before the MTP head flips nextn output
     * on the target: this is the engine WITHOUT speculation) ---- */
    std::vector<int32_t> plain;
    double plain_ms = 0, plain_prefill_ms = 0;
    {
        const int64_t a = ggml_time_us();
        if (ell_decode_seq_full(ctx, model, seq, 0, toks.data(), n, plog.data())) { fprintf(stderr, "plain prefill failed\n"); return 2; }
        const int64_t b = ggml_time_us();
        plain_prefill_ms = (b - a) / 1e3;
        int t = argmax(plog.data() + (size_t)(n - 1) * nv);
        int pos = n;
        for (int i = 0; i < n_predict; i++) {
            plain.push_back(t);
            if (llama_vocab_is_eog(vocab, t)) break;
            if (ell_decode_seq_full(ctx, model, seq, pos, &t, 1, logits.data())) { fprintf(stderr, "plain decode failed\n"); return 2; }
            pos++;
            t = argmax(logits.data());
        }
        plain_ms = (ggml_time_us() - b) / 1e3;
        ell_seq_remove(ctx, seq);
    }
    const int plain_gen = (int)plain.size();

    /* ---- 2. speculative ---- */
    void *mtp = ell_mtp_new(model, ctx, n_ctx, n_batch, n_seqs, 0, 0, 0);
    if (!mtp) { fprintf(stderr, "ell_mtp_new failed\n"); return 2; }
    std::vector<int32_t> spec; std::string text;
    int rounds = 0, drafted = 0, accepted = 0, obs_fail = 0;
    double draft_ms = 0, verify_ms = 0, spec_ms = 0, spec_prefill_ms = 0;
    {
        const int64_t a = ggml_time_us();
        if (ell_decode_seq_full(ctx, model, seq, 0, toks.data(), n, plog.data())) { fprintf(stderr, "spec prefill failed\n"); return 2; }
        ell_mtp_harvest(mtp, ctx, seq, n);
        const int64_t b = ggml_time_us();
        spec_prefill_ms = (b - a) / 1e3;
        int id_last = argmax(plog.data() + (size_t)(n - 1) * nv);
        int n_past = n;
        /* the observe folded into the next draft: the prompt first, then each
         * round's accepted tokens */
        std::vector<int32_t> obs(toks.begin(), toks.end());
        int obs_pos0 = 0;
        std::vector<int32_t> drafts(K), ids;
        spec.push_back(id_last);
        while ((int)spec.size() < n_predict && !llama_vocab_is_eog(vocab, id_last)) {
            const int64_t d0 = ggml_time_us();
            int k = ell_mtp_draft2(mtp, seq, id_last, n_past, K, p_min, drafts.data(),
                                   obs_pos0, obs.data(), (int)obs.size());
            const int64_t d1 = ggml_time_us();
            draft_ms += (d1 - d0) / 1e3;
            if (k == 0 && !obs.empty()) {
                /* draft2 returns 0 on a failed observe: re-sync explicitly
                 * (the engine's recovery path) and take a plain step */
                if (ell_mtp_observe(mtp, seq, obs_pos0, obs.data(), (int)obs.size()) != 0) obs_fail++;
            }
            ids.assign(1, id_last);
            ids.insert(ids.end(), drafts.begin(), drafts.begin() + k);
            const int64_t v0 = ggml_time_us();
            if (ell_decode_seq_full(ctx, model, seq, n_past, ids.data(), (int)ids.size(), logits.data())) {
                fprintf(stderr, "verify decode failed at round %d\n", rounds); return 2;
            }
            ell_mtp_harvest(mtp, ctx, seq, (int)ids.size());
            verify_ms += (ggml_time_us() - v0) / 1e3;
            int acc = 0;
            for (; acc < k; acc++) if (argmax(logits.data() + (size_t)acc * nv) != drafts[acc]) break;
            const int t_new = argmax(logits.data() + (size_t)acc * nv);
            /* drop the rejected tail; the next decode continues at n_past+acc+1 */
            if (acc < k && ell_seq_rewind(ctx, seq, n_past + acc + 1) != 0) {
                fprintf(stderr, "rewind refused at round %d (acc %d of %d)\n", rounds, acc, k); return 2;
            }
            obs.assign(ids.begin(), ids.begin() + acc + 1);
            obs_pos0 = n_past;
            rounds++; drafted += k; accepted += acc;
            /* emit: the accepted drafts, then the target's own token */
            for (int i = 0; i < acc && (int)spec.size() < n_predict; i++) spec.push_back(drafts[i]);
            if ((int)spec.size() < n_predict) spec.push_back(t_new);
            n_past += acc + 1;
            id_last = t_new;
            if (acc < k) { bool eog = false; for (int i = 0; i < acc; i++) if (llama_vocab_is_eog(vocab, drafts[i])) eog = true; if (eog) break; }
        }
        spec_ms = (ggml_time_us() - b) / 1e3;
    }
    /* the plain reference stops AT the eog token; make the spec stream match */
    for (size_t i = 0; i < spec.size(); i++) {
        if (llama_vocab_is_eog(vocab, spec[i])) { spec.resize(i + 1); break; }
    }
    const int spec_gen = (int)spec.size();
    const bool same = (spec == plain);
    /* where the streams part, when they do: a CUDA verify pass (m > 1) runs
     * different kernels than the m = 1 step, so a near-tie argmax can flip -
     * the loop is only lossless when verify and plain logits are bit-equal */
    int first_diff = -1;
    for (size_t i = 0; i < spec.size() || i < plain.size(); i++) {
        if (i >= spec.size() || i >= plain.size() || spec[i] != plain[i]) { first_diff = (int)i; break; }
    }
    for (int t : spec) if (!llama_vocab_is_eog(vocab, t)) text += piece(t);

    printf("{\"label\":\"%s\",\"k\":%d,\"p_min\":%.2f,\"rewind_depth\":%d,\"prompt_tokens\":%d,"
           "\"generated\":%d,\"rounds\":%d,\"drafted\":%d,\"accepted\":%d,"
           "\"mean_accepted_per_round\":%.3f,\"mean_tokens_per_round\":%.3f,\"acceptance_rate\":%.3f,"
           "\"draft_ms_per_round\":%.3f,\"verify_ms_per_round\":%.3f,"
           "\"spec_prefill_ms\":%.1f,\"decode_ms_per_tok\":%.3f,\"decode_tok_s\":%.2f,"
           "\"plain_generated\":%d,\"plain_prefill_ms\":%.1f,\"plain_ms_per_tok\":%.3f,\"plain_tok_s\":%.2f,"
           "\"speedup\":%.3f,\"obs_fail\":%d,\"text_identical\":%s,\"first_diff_token\":%d,\"text\":\"",
           getenv("LABEL") ? getenv("LABEL") : "?", K, p_min, depth, n,
           spec_gen, rounds, drafted, accepted,
           rounds ? (double)accepted / rounds : 0, rounds ? (double)(accepted + rounds) / rounds : 0,
           drafted ? (double)accepted / drafted : 0,
           rounds ? draft_ms / rounds : 0, rounds ? verify_ms / rounds : 0,
           spec_prefill_ms, spec_gen > 1 ? spec_ms / (spec_gen - 1) : 0, spec_gen > 1 ? (spec_gen - 1) * 1e3 / spec_ms : 0,
           plain_gen, plain_prefill_ms, plain_gen > 1 ? plain_ms / (plain_gen - 1) : 0, plain_gen > 1 ? (plain_gen - 1) * 1e3 / plain_ms : 0,
           (spec_ms > 0 && plain_ms > 0 && spec_gen > 1 && plain_gen > 1) ? (plain_ms / (plain_gen - 1)) / (spec_ms / (spec_gen - 1)) : 0,
           obs_fail, same ? "true" : "false", first_diff);
    json_text(text);
    printf("\"}\n");
    fflush(stdout); /* the result must survive a teardown crash below */
    if (!same) {
        std::string pt; for (int t : plain) if (!llama_vocab_is_eog(vocab, t)) pt += piece(t);
        fprintf(stderr, "[bench] TEXT MISMATCH\n plain: %s\n spec:  %s\n", pt.c_str(), text.c_str());
    }
    if (const char *sh = getenv("SHIELDED_SO_FOR_STATS")) {
        void *h = dlopen(sh, RTLD_NOW | RTLD_NOLOAD);
        if (h) {
            typedef void (*stats_fn)(uint64_t*, uint64_t*, uint64_t*, uint64_t*);
            stats_fn f = (stats_fn)dlsym(h, "ggml_backend_shielded_stats");
            uint64_t off=0, loc=0, macs=0, vf=0;
            if (f) { f(&off,&loc,&macs,&vf);
                fprintf(stderr, "[bench] shielded: offloaded=%llu local=%llu GMAC=%.2f verify_fail=%llu\n",
                        (unsigned long long)off,(unsigned long long)loc,(double)macs/1e9,(unsigned long long)vf); }
        }
    }
    ell_mtp_free(mtp);
    ell_free_context(ctx);
    ell_free_model(model);
    return same ? 0 : 3;
}
