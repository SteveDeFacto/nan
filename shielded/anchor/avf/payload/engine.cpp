/*
 * engine.cpp -- the Shielded inference engine INSIDE the protected VM.
 *
 * shielded-run.cpp's flow, with the three things a Microdroid guest cannot do
 * the normal-world way taken from the owner instead:
 *   - the model arrives as bytes over vsock into a memfd (no filesystem the
 *     owner could populate), loaded through /proc/self/fd/N;
 *   - the worker link is an ACCEPTED fd, adopted by the trusted half via
 *     sh_pipe_adopt_fd (the module's build renames sh_pipe_open to the hook);
 *   - the calibration comes from the APK's assets, which the attestation's
 *     codeHash covers.
 * Built as libengine.so with ordinary DT_NEEDED on libllama/libggml; the
 * bootstrap payload dlopens those from the APK first (RTLD_GLOBAL), then this.
 */
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <android/log.h>

static int g_ctl = -1;
static void outf(const char *fmt, ...) {
    char line[4096]; va_list ap; va_start(ap, fmt); int n = vsnprintf(line, sizeof line - 1, fmt, ap); va_end(ap);
    if (n < 0) return; if ((size_t)n > sizeof line - 2) n = sizeof line - 2;
    line[n] = '\n'; line[n + 1] = 0;
    fputs(line, stdout); fflush(stdout);
    __android_log_print(ANDROID_LOG_INFO, "anchor-engine", "%.*s", n, line);
    if (g_ctl >= 0) { const char *p = line; size_t left = (size_t)n + 1; while (left) { ssize_t w = write(g_ctl, p, left); if (w <= 0) break; p += w; left -= (size_t)w; } }
}
/* llama's load chatter stays off the control channel; its warnings and errors
 * go to stderr (the engine.err file in the encrypted store), which the owner
 * sees as a tail when a step fails. */
static void quiet_log(enum ggml_log_level level, const char *text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) { fputs(text, stderr); fflush(stderr); }
}

typedef void (*stats_fn)(uint64_t *offloaded, uint64_t *local, uint64_t *macs, uint64_t *verify_fail);
typedef void (*adopt_fn)(int fd);
typedef ggml_threadpool *(*tp_new_fn)(ggml_threadpool_params *);

/* ---- dealt pads: ledger windows through the owner app --------------------
 * The engine never talks to the platform itself; it writes a signed PADWIN
 * request on the control socket (the owner app POSTs it to /v1/pads/reserve)
 * and verifies the signed window the app relays back. The control socket is
 * ours to read for the whole run: the payload's loop is parked in engine_main. */
#include "anchor_pads.h"
#include "shielded-pads.h"
#include "prefix-kv.h"
#include "anchor_mtp.h"
#include <atomic>
#include <thread>
extern "C" {
#include "tweetnacl.h"
}
#include <sys/random.h>
typedef int (*win_fn)(void *, uint64_t, uint64_t *, uint64_t *);
typedef void (*set_win_fn)(win_fn, void *);
static int ctl_read_line(char *buf, size_t cap) {
    size_t n = 0;
    while (n + 1 < cap) {
        char c; ssize_t r = read(g_ctl, &c, 1);
        if (r <= 0) return -1;
        if (c == '\n') break;
        buf[n++] = c;
    }
    buf[n] = 0;
    return (int)n;
}
/* The usage receipt the operator is paid on: pads consumed and tokens served,
 * signed by the transport key, relayed by the owner app to the platform
 * (PADS receipt). Nothing here is secret; the signature is what makes it
 * the pVM's word rather than the operator's. */
static void pads_receipt(const anchor_pads *p, uint64_t pads_used, uint64_t tokens) {
    uint8_t nb[16]; if (getrandom(nb, sizeof nb, 0) != (ssize_t)sizeof nb) return;
    char nonce[33], used[24], toks[24], sig_hex[129]; uint8_t sig[64];
    sh_pads_bin2hex(nb, 16, nonce);
    snprintf(used, sizeof used, "%llu", (unsigned long long)pads_used);
    snprintf(toks, sizeof toks, "%llu", (unsigned long long)tokens);
    const char *fields[4] = { p->name, p->seed_id_hex, used, toks };
    sh_pads_request_sign(p->transport_sk, "receipt", fields, 4, nonce, sig);
    sh_pads_bin2hex(sig, 64, sig_hex);
    outf("RECEIPT %s %s %s %s %s %s", p->name, p->seed_id_hex, used, toks, nonce, sig_hex);
}

static int pads_window(void *ctx, uint64_t want, uint64_t *lo, uint64_t *hi) {
    const anchor_pads *p = (const anchor_pads *)ctx;
    uint8_t nb[16]; if (getrandom(nb, sizeof nb, 0) != (ssize_t)sizeof nb) return -1;
    char nonce[33], wants[24], sig_hex[129]; uint8_t sig[64];
    sh_pads_bin2hex(nb, 16, nonce);
    snprintf(wants, sizeof wants, "%llu", (unsigned long long)want);
    const char *fields[3] = { p->name, p->seed_id_hex, wants };
    sh_pads_request_sign(p->transport_sk, "reserve", fields, 3, nonce, sig);
    sh_pads_bin2hex(sig, 64, sig_hex);
    outf("PADWIN %s %s %s", wants, nonce, sig_hex);
    /* the app answers PADWIN <lo> <hi> <iat> <sig> (or PADWIN fail <why>); other lines are the app's chatter */
    for (int tries = 0; tries < 64; tries++) {
        char line[512];
        if (ctl_read_line(line, sizeof line) < 0) return -1;
        if (strncmp(line, "PADWIN ", 7)) continue;
        unsigned long long l = 0, h = 0, iat = 0; char sh[129] = "";
        if (sscanf(line + 7, "%llu %llu %llu %128s", &l, &h, &iat, sh) != 4) { outf("ENGINE pads: window refused: %s", line + 7); return -1; }
        uint8_t wsig[64];
        if (!sh_pads_hex2bin(sh, wsig, 64) || !sh_pads_window_verify(p->ledger_pk, p->seed_id_hex, l, h, iat, wsig)) { outf("ENGINE pads: window signature REJECTED"); return -1; }
        *lo = l; *hi = h;
        return 0;
    }
    return -1;
}

extern "C" int engine_main(int ctl_fd, int worker_fd, int model_fd, const char *lib_dir, const char *calib_path,
                           const char *prompt, int n_predict, int n_threads, const anchor_pads *pads) {
    g_ctl = ctl_fd;
    setvbuf(stdout, NULL, _IONBF, 0);
    /* The backend's own diagnostics ("[shielded] ...") go to stderr, which
     * nothing in the VM collects: keep them in the encrypted store and hand
     * the tail to the owner when a step fails, so a refusal explains itself. */
    static char err_path[512];
    { const char *es = getenv("ANCHOR_ENCRYPTED_STORE"); snprintf(err_path, sizeof err_path, "%s/engine.err", es && *es ? es : "/data/local/tmp"); }
    if (!freopen(err_path, "w", stderr)) { outf("ENGINE stderr not captured (%s: %s)", err_path, strerror(errno)); err_path[0] = 0; }
    else outf("ENGINE stderr -> %s", err_path);
    setvbuf(stderr, NULL, _IONBF, 0);
    auto dump_err = [&]() {
        if (!err_path[0]) return;
        FILE *f = fopen(err_path, "r"); if (!f) return;
        std::vector<std::string> lines; char line[512];
        while (fgets(line, sizeof line, f)) { line[strcspn(line, "\n")] = 0; if (line[0]) { lines.push_back(line); if (lines.size() > 400) lines.erase(lines.begin()); } }
        fclose(f);
        size_t from = lines.size() > 12 ? lines.size() - 12 : 0;
        for (size_t i = from; i < lines.size(); i++) outf("ENGINE stderr: %s", lines[i].c_str());
    };
    llama_log_set(quiet_log, nullptr);
    setenv("SHIELDED_HOST", "adopted-fd", 1); setenv("SHIELDED_PORT", "0", 1);
    /* The contention detector models an exchange's cost from the weight
     * bytes it moves (a microsecond wire inside the box). On the phone the
     * wire is milliseconds by construction, so every exchange looks "8x
     * slower than it should be" and the detector parks the whole model on
     * the phone's CPU. Not a signal here: never conclude contention. */
    setenv("SHIELDED_CONTENTION_ABS_X", "1000000000", 0);
    setenv("SHIELDED_CONTENTION_X", "1000000000", 0);
    /* Spin on the reply instead of blocking: a vCPU that halts between
     * exchanges pays the phone's wake-up latency ~100 times per token. */
    /* no SHIELDED_SPIN_US here: a 20 ms spin-poll on the link thread took the wire from 5.4 to 18 ms
     * per exchange on the phone (run24): the spinning vCPU starves the compute threads */
    setenv("SHIELDED_CALIB", calib_path, 1);
    setenv("SHIELDED_PROFILE", "1", 1);

    std::string cpu_so = std::string(lib_dir) + "/libggml-cpu.so", sh_so = std::string(lib_dir) + "/libggml-shielded.so";
    if (!ggml_backend_load(cpu_so.c_str())) { outf("ENGINE cpu backend failed to load"); return 2; }
    ggml_backend_reg_t r = ggml_backend_load(sh_so.c_str());
    if (!r) { outf("ENGINE shielded backend failed to load"); return 2; }
    void *sh_h = dlopen(sh_so.c_str(), RTLD_NOW);
    stats_fn stats = sh_h ? (stats_fn)dlsym(sh_h, "ggml_backend_shielded_stats") : nullptr;
    typedef void (*pads_used_fn)(uint64_t *, uint64_t *);
    pads_used_fn pads_used = sh_h ? (pads_used_fn)dlsym(sh_h, "ggml_backend_shielded_pads_used") : nullptr;
    adopt_fn adopt = sh_h ? (adopt_fn)dlsym(sh_h, "sh_pipe_adopt_fd") : nullptr;
    if (!adopt) { outf("ENGINE the shielded module has no sh_pipe_adopt_fd (built without the hook?)"); return 2; }
    adopt(worker_fd);                                 /* the first sh_pipe_open (inside the first graph) gets this */
    if (pads) {
        set_win_fn set_win = sh_h ? (set_win_fn)dlsym(sh_h, "ggml_backend_shielded_set_window_provider") : nullptr;
        if (!set_win) { outf("ENGINE pads: the shielded module has no window provider hook"); return 2; }
        set_win(pads_window, (void *)pads);
        outf("ENGINE dealt pads: seed %s, windows through the owner (%s)", pads->seed_id_hex, getenv("SHIELDED_PAD_SOURCE") ? getenv("SHIELDED_PAD_SOURCE") : "no bank dir");
    }
    outf("ENGINE backends loaded, worker fd %d adopted, stats=%s, calib %s (%s)", worker_fd, stats ? "yes" : "no",
         access(calib_path, R_OK) == 0 ? "readable" : "MISSING", calib_path);

    llama_backend_init();
    /* ANCHOR_MTP_K > 0: decode with the model's own MTP head as the draft
     * (anchor_mtp.h). k proposals + the sampled token ride ONE exchange chain
     * as k+1 rows (the shielded link takes up to 8), the target's logits
     * verify them, rejected rows roll back through n_rs_seq recurrent-state
     * snapshots. Greedy on both sides, so the text equals plain greedy. */
    int mtp_k = 0; { const char *e = getenv("ANCHOR_MTP_K"); if (e) mtp_k = atoi(e); if (mtp_k > 7) mtp_k = 7; if (mtp_k < 0) mtp_k = 0; }
    float mtp_pmin = 0.0f; { const char *e = getenv("ANCHOR_MTP_PMIN"); if (e) mtp_pmin = (float)atof(e); }
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    mp.load_mtp = mtp_k > 0;   /* the nextn head is opt-in since the fork's ddd4ec1 pin; without it the head tensors load as "unused" */
    char path[64]; snprintf(path, sizeof path, "/proc/self/fd/%d", model_fd);
    const long t_load0 = ggml_time_us();
    llama_model *model = llama_model_load_from_file(path, mp);
    if (!model) { outf("ENGINE model load failed from %s", path); return 2; }
    outf("ENGINE model loaded in %.1f s", (ggml_time_us() - t_load0) / 1e6);
    const llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    if (mtp_k > 0) cp.n_rs_seq = (uint32_t)mtp_k;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { outf("ENGINE context failed"); return 2; }
    anchor_mtp *mtp = mtp_k > 0 ? anchor_mtp_new(model, ctx, 512, 8, n_threads) : nullptr;
    /* ANCHOR_BOOST_THREADS=n: n threads that only spin. The decode is ~100 short compute
     * bursts per token between link waits, and the phone's governor answers that duty cycle
     * with 0.4 GHz on the mid cores (measured: 402 MHz during decode, 2.1 GHz with burners,
     * token rate +57% even with the burners competing). Spinning vCPUs keep the cores'
     * utilization high, so the compute threads run at full clock. Off by default. */
    std::atomic<bool> boost_on{true}; std::vector<std::thread> boost;
    struct boost_guard { std::atomic<bool> &on; std::vector<std::thread> &th; ~boost_guard() { on.store(false); for (auto &t : th) if (t.joinable()) t.join(); } } boost_stop{boost_on, boost};
    { const char *e = getenv("ANCHOR_BOOST_THREADS"); int nb = e ? atoi(e) : 0; if (nb > 16) nb = 16;
      for (int i = 0; i < nb; i++) boost.emplace_back([&boost_on] { while (boost_on.load(std::memory_order_relaxed)) { for (int k = 0; k < 256; k++) __asm__ __volatile__("" ::: "memory"); } });
      if (nb > 0) outf("ENGINE boost: %d spinning threads keep the clocks up", nb); }
    if (mtp_k > 0) outf("ENGINE MTP draft: %s (k=%d, p_min=%.2f, rollback depth %u)", mtp ? "on" : "UNAVAILABLE (no nextn layer or symbols); decoding plainly", mtp_k, mtp_pmin, llama_n_rs_seq(ctx));
    /* one persistent CPU pool: without it every scheduler split respawns the threads (REPORT.md 12) */
    void *cpu_h = dlopen(cpu_so.c_str(), RTLD_NOW);
    tp_new_fn tp_new = cpu_h ? (tp_new_fn)dlsym(cpu_h, "ggml_threadpool_new") : nullptr;
    if (tp_new) { ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads); ggml_threadpool *tp = tp_new(&tpp); llama_attach_threadpool(ctx, tp, tp); }
    outf("ENGINE context ready, %d threads, persistent pool=%s", n_threads, tp_new ? "yes" : "no");

    /* The shared-prefix KV (prefix-kv.h): SHIELDED_PREFIX_KV names the file
     * the owner streamed into the encrypted store, SHIELDED_PREFIX_FILE the
     * prefix text, SHIELDED_PREFIX_KV_PK the platform's prefix key. The
     * files may still be arriving over vsock: wait for them, verify the
     * sidecar against this model (the calib digest) and the exact text,
     * load the sequence, and prefill only the user's part. The prompt the
     * owner sent is the user's part; the prefix goes in front of it here. */
    int n_loaded = 0;
    std::string full_prompt = prompt;
    if (const char *kv = getenv("SHIELDED_PREFIX_KV"); kv && *kv) {
        const char *pkh = getenv("SHIELDED_PREFIX_KV_PK"), *pf = getenv("SHIELDED_PREFIX_FILE");
        uint8_t pk[32], digest[32];
        if (!pkh || !sh_pads_hex2bin(pkh, pk, 32) || !pf) { outf("ENGINE prefix KV: missing SHIELDED_PREFIX_KV_PK / SHIELDED_PREFIX_FILE; refusing"); return 2; }
        std::string side = std::string(kv) + ".sig";
        for (int waited = 0; waited < 1200 && !(access(kv, R_OK) == 0 && access(side.c_str(), R_OK) == 0 && access(pf, R_OK) == 0); waited++) usleep(100000);
        std::string prefix, cal;
        auto slurp = [](const char *p, std::string &o) { FILE *f = fopen(p, "rb"); if (!f) return false; char b[65536]; size_t k; while ((k = fread(b, 1, sizeof b, f)) > 0) o.append(b, k); fclose(f); return true; };
        if (!slurp(pf, prefix) || !slurp(calib_path, cal)) { outf("ENGINE prefix KV: files never arrived (%s)", pf); return 2; }
        { uint8_t h[64]; crypto_hash(h, (const uint8_t *)cal.data(), cal.size()); memcpy(digest, h, 32); }
        char err[256]; uint64_t ntok = 0;
        if (sh_prefix_kv_verify(kv, pk, digest, prefix.data(), prefix.size(), &ntok, err, sizeof err)) { outf("ENGINE prefix KV REFUSED: %s", err); return 2; }
        std::vector<llama_token> loaded(ntok + 16); size_t got = 0;
        if (!llama_state_seq_load_file(ctx, kv, 0, loaded.data(), loaded.size(), &got) || got != ntok) { outf("ENGINE prefix KV load failed (%zu of %llu tokens)", got, (unsigned long long)ntok); return 2; }
        n_loaded = (int)got;
        full_prompt = prefix + prompt;
        outf("ENGINE prefix KV: %d tokens loaded and verified (%s)", n_loaded, kv);
    }
    const char *rest = full_prompt.c_str() + (n_loaded ? full_prompt.size() - strlen(prompt) : 0);
    std::vector<llama_token> toks(strlen(rest) + 16);
    int n = llama_tokenize(vocab, rest, (int)strlen(rest), toks.data(), (int)toks.size(), n_loaded == 0, n_loaded != 0);
    if (n < 0) { outf("ENGINE tokenize failed"); return 2; }
    toks.resize(n);
    const long t_pp0 = ggml_time_us();
    if (n > 0) {
        int rc;
        if (mtp) {   /* the head mirrors the prompt: it needs the target's nextn row at every prompt position */
            llama_batch pb = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; i++) { pb.token[i] = toks[i]; pb.pos[i] = n_loaded + i; pb.n_seq_id[i] = 1; pb.seq_id[i][0] = 0; pb.logits[i] = 1; }
            pb.n_tokens = n;
            rc = llama_decode(ctx, pb);
            llama_batch_free(pb);
        } else rc = llama_decode(ctx, llama_batch_get_one(toks.data(), n));
        if (rc) { outf("ENGINE prefill failed (%d tokens)", n); dump_err(); return 2; }
    }
    const long t_pp1 = ggml_time_us();
    { uint64_t o = 0, l = 0, m = 0, v = 0; if (stats) stats(&o, &l, &m, &v);
      outf("ENGINE prefill %d tokens in %.0f ms: %llu nodes offloaded, %llu local, %.2f GMAC, verify_fail %llu (first graph = weights to the worker + pool warm-up)",
           n, (t_pp1 - t_pp0) / 1e3, (unsigned long long)o, (unsigned long long)l, m / 1e9, (unsigned long long)v);
      if (o == 0) outf("ENGINE WARNING: nothing offloaded; the CPU backend repacked the weights, or the calibration did not match this model"); }

    std::string out; llama_token cur = 0; int n_gen = 0;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto argmax = [&](const float *logits) { int best = 0; float bv = logits[0]; for (int t = 1; t < n_vocab; t++) if (logits[t] > bv) { bv = logits[t]; best = t; } return (llama_token)best; };
    /* false = end of generation (eog or the budget) */
    auto emit = [&](llama_token t) { if (llama_vocab_is_eog(vocab, t)) return false;
        char piece[256]; int pn = llama_token_to_piece(vocab, t, piece, sizeof piece, 0, true);
        if (pn > 0) { out.append(piece, pn); outf("TOKEN %.*s", pn, piece); }
        return ++n_gen < n_predict; };
    int n_past = n_loaded + n, mtp_rounds = 0, mtp_drafted = 0, mtp_accepted = 0;
    double t_draft = 0, t_verify = 0, t_observe = 0;   /* where a round's wall time goes (us) */
    const long t_tg0 = ggml_time_us();
    cur = argmax(llama_get_logits_ith(ctx, -1));
    if (mtp && n > 0) { if (anchor_mtp_harvest(mtp, ctx, n) || anchor_mtp_observe(mtp, n_loaded, toks.data(), n)) { outf("ENGINE MTP: the head could not observe the prompt; decoding plainly"); anchor_mtp_free(mtp); mtp = nullptr; } }
    bool go = n_predict > 0 && emit(cur);
    while (go) {
        if (!mtp) {
            if (llama_decode(ctx, llama_batch_get_one(&cur, 1))) { outf("ENGINE decode failed"); dump_err(); break; }
            n_past++; cur = argmax(llama_get_logits_ith(ctx, -1)); go = emit(cur); continue;
        }
        int32_t d[8]; int k = mtp_k; if (k > n_predict - n_gen - 1) k = n_predict - n_gen - 1;
        const long t_r0 = ggml_time_us();
        const int nd = k > 0 ? anchor_mtp_draft(mtp, cur, n_past, k, mtp_pmin, d) : 0;
        const long t_r1 = ggml_time_us();
        llama_token rows[9]; rows[0] = cur; for (int i = 0; i < nd; i++) rows[i + 1] = d[i];
        llama_batch vb = llama_batch_init(nd + 1, 0, 1);
        for (int i = 0; i <= nd; i++) { vb.token[i] = rows[i]; vb.pos[i] = n_past + i; vb.n_seq_id[i] = 1; vb.seq_id[i][0] = 0; vb.logits[i] = 1; }
        vb.n_tokens = nd + 1;
        const int rc = llama_decode(ctx, vb); llama_batch_free(vb);
        const long t_r2 = ggml_time_us();
        if (rc) { outf("ENGINE decode failed (verify of %d rows)", nd + 1); dump_err(); break; }
        int acc = 0; while (acc < nd && argmax(llama_get_logits_ith(ctx, acc)) == d[acc]) acc++;
        const llama_token bonus = argmax(llama_get_logits_ith(ctx, acc));
        if (acc < nd && !llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + acc + 1, -1)) { outf("ENGINE MTP: rollback of %d rows REFUSED (n_rs_seq %u)", nd - acc, llama_n_rs_seq(ctx)); break; }
        if (anchor_mtp_harvest(mtp, ctx, acc + 1) || anchor_mtp_observe(mtp, n_past, rows, acc + 1)) { outf("ENGINE MTP: observe failed; decoding plainly from here"); anchor_mtp_free(mtp); mtp = nullptr; }
        n_past += acc + 1; mtp_rounds++; mtp_drafted += nd; mtp_accepted += acc;
        t_draft += t_r1 - t_r0; t_verify += t_r2 - t_r1; t_observe += ggml_time_us() - t_r2;
        go = true; for (int i = 0; i < acc && go; i++) go = emit(d[i]);
        if (go) go = emit(bonus);
        cur = bonus;
    }
    const long t_tg1 = ggml_time_us();
    if (mtp_rounds) outf("ENGINE MTP: %d rounds, %d drafted, %d accepted (%.2f tokens per round, %.0f%% of drafts); per round: draft %.0f ms, verify %.0f ms, rollback+observe %.0f ms",
                         mtp_rounds, mtp_drafted, mtp_accepted, (double)(mtp_accepted + mtp_rounds) / mtp_rounds, mtp_drafted ? 100.0 * mtp_accepted / mtp_drafted : 0.0,
                         t_draft / 1e3 / mtp_rounds, t_verify / 1e3 / mtp_rounds, t_observe / 1e3 / mtp_rounds);
    if (mtp) anchor_mtp_free(mtp);
    uint64_t off = 0, loc = 0, macs = 0, vf = 0;
    if (stats) stats(&off, &loc, &macs, &vf);
    outf("{\"engine\":\"avf-pvm\",\"prompt_tokens\":%d,\"generated\":%d,\"completion\":\"%s\",\"prefill_ms\":%.0f,"
         "\"decode_ms_per_tok\":%.1f,\"offloaded_nodes\":%llu,\"local_nodes\":%llu,\"gmac\":%.2f,\"verify_fail\":%llu,\"threads\":%d}",
         n, n_gen, out.c_str(), (t_pp1 - t_pp0) / 1e3, n_gen ? (t_tg1 - t_tg0) / 1e3 / n_gen : 0.0,
         (unsigned long long)off, (unsigned long long)loc, macs / 1e9, (unsigned long long)vf, n_threads);
    if (pads && pads_used) { uint64_t pu = 0, pm = 0; pads_used(&pu, &pm); pads_receipt(pads, pu, (uint64_t)n + (uint64_t)n_gen); }
    /* the backend's profile lines (SHIELDED_PROFILE=1: exchange counts, mask/wire/unmask
     * time, pad waits) live in stderr; hand the owner the summary so a run explains itself */
    if (err_path[0]) {
        FILE *f = fopen(err_path, "r");
        if (f) {
            char line[1024]; int warned = 0;
            while (fgets(line, sizeof line, f)) {
                const bool summary = strstr(line, "[shielded] profile: exchanges") || strstr(line, "[shielded] widths:");
                const bool warning = strstr(line, "[shielded]") && (strstr(line, "offload failed") || strstr(line, "refus") || strstr(line, "contend") || strstr(line, "unavailable") || strstr(line, "cannot") || strstr(line, "REJECT") || strstr(line, "stopped") || strstr(line, "link:"));
                if (!summary && !(warning && warned < 12)) continue;
                if (warning) warned++;
                line[strcspn(line, "\n")] = 0; outf("ENGINE %s", line);
            }
            fclose(f);
        }
    }
    llama_free(ctx); llama_model_free(model);
    return 0;
}
