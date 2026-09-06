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
static void quiet_log(enum ggml_log_level, const char *, void *) {}   /* llama's load chatter stays off the control channel */

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
    llama_log_set(quiet_log, nullptr);
    setenv("SHIELDED_HOST", "adopted-fd", 1); setenv("SHIELDED_PORT", "0", 1);
    setenv("SHIELDED_CALIB", calib_path, 1);
    setenv("SHIELDED_PROFILE", "1", 1);

    std::string cpu_so = std::string(lib_dir) + "/libggml-cpu.so", sh_so = std::string(lib_dir) + "/libggml-shielded.so";
    if (!ggml_backend_load(cpu_so.c_str())) { outf("ENGINE cpu backend failed to load"); return 2; }
    ggml_backend_reg_t r = ggml_backend_load(sh_so.c_str());
    if (!r) { outf("ENGINE shielded backend failed to load"); return 2; }
    void *sh_h = dlopen(sh_so.c_str(), RTLD_NOW);
    stats_fn stats = sh_h ? (stats_fn)dlsym(sh_h, "ggml_backend_shielded_stats") : nullptr;
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
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    char path[64]; snprintf(path, sizeof path, "/proc/self/fd/%d", model_fd);
    const long t_load0 = ggml_time_us();
    llama_model *model = llama_model_load_from_file(path, mp);
    if (!model) { outf("ENGINE model load failed from %s", path); return 2; }
    outf("ENGINE model loaded in %.1f s", (ggml_time_us() - t_load0) / 1e6);
    const llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { outf("ENGINE context failed"); return 2; }
    /* one persistent CPU pool: without it every scheduler split respawns the threads (REPORT.md 12) */
    void *cpu_h = dlopen(cpu_so.c_str(), RTLD_NOW);
    tp_new_fn tp_new = cpu_h ? (tp_new_fn)dlsym(cpu_h, "ggml_threadpool_new") : nullptr;
    if (tp_new) { ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads); ggml_threadpool *tp = tp_new(&tpp); llama_attach_threadpool(ctx, tp, tp); }
    outf("ENGINE context ready, %d threads, persistent pool=%s", n_threads, tp_new ? "yes" : "no");

    std::vector<llama_token> toks(512);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), (int)toks.size(), true, false);
    if (n < 0) { outf("ENGINE tokenize failed"); return 2; }
    toks.resize(n);
    const long t_pp0 = ggml_time_us();
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n))) { outf("ENGINE prefill failed"); return 2; }
    const long t_pp1 = ggml_time_us();
    { uint64_t o = 0, l = 0, m = 0, v = 0; if (stats) stats(&o, &l, &m, &v);
      outf("ENGINE prefill %d tokens in %.0f ms: %llu nodes offloaded, %llu local, %.2f GMAC, verify_fail %llu (first graph = weights to the worker + pool warm-up)",
           n, (t_pp1 - t_pp0) / 1e3, (unsigned long long)o, (unsigned long long)l, m / 1e9, (unsigned long long)v);
      if (o == 0) outf("ENGINE WARNING: nothing offloaded; the CPU backend repacked the weights, or the calibration did not match this model"); }

    std::string out; llama_token cur = 0; int n_gen = 0;
    const long t_tg0 = ggml_time_us();
    for (int i = 0; i < n_predict; i++) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        int best = 0; float bv = logits[0];
        for (int t = 1; t < n_vocab; t++) if (logits[t] > bv) { bv = logits[t]; best = t; }
        cur = best;
        if (llama_vocab_is_eog(vocab, cur)) break;
        char piece[256]; int pn = llama_token_to_piece(vocab, cur, piece, sizeof piece, 0, true);
        if (pn > 0) { out.append(piece, pn); outf("TOKEN %.*s", pn, piece); }
        n_gen++;
        if (llama_decode(ctx, llama_batch_get_one(&cur, 1))) { outf("ENGINE decode failed"); break; }
    }
    const long t_tg1 = ggml_time_us();
    uint64_t off = 0, loc = 0, macs = 0, vf = 0;
    if (stats) stats(&off, &loc, &macs, &vf);
    outf("{\"engine\":\"avf-pvm\",\"prompt_tokens\":%d,\"generated\":%d,\"completion\":\"%s\",\"prefill_ms\":%.0f,"
         "\"decode_ms_per_tok\":%.1f,\"offloaded_nodes\":%llu,\"local_nodes\":%llu,\"gmac\":%.2f,\"verify_fail\":%llu,\"threads\":%d}",
         n, n_gen, out.c_str(), (t_pp1 - t_pp0) / 1e3, n_gen ? (t_tg1 - t_tg0) / 1e3 / n_gen : 0.0,
         (unsigned long long)off, (unsigned long long)loc, macs / 1e9, (unsigned long long)vf, n_threads);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
