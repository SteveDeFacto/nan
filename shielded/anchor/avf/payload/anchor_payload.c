/*
 * anchor_payload -- the Shielded anchor's trusted half, running INSIDE a
 * protected VM on the phone, driving a REAL GPU worker.
 *
 * This is the property the whole search was for: the pad key, the pads,
 * u = r.W, the Freivalds secrets, the plaintext activation and every unmasked
 * product live in memory that pKVM has unmapped from the host -- the phone's
 * owner, with root, cannot read it.
 *
 * The VM is non-debuggable in production, so it has no console: everything
 * it says goes to its owner over vsock, and everything it needs arrives the
 * same way. Two listeners, both accepted from the host app's connectVsock():
 *
 *   7777  control   host -> guest:  CHAL <64 hex>            attestation challenge
 *                                   WORKER bridge|local      where the GEMMs go
 *                                   SHAPE K N nodes iters xmax   (repeatable; xmax 0 = auto)
 *                                   RUN
 *                   guest -> host:  ATTEST/CERT/SIG lines, one JSON line per shape, END
 *   7778  worker    one connection per shape; the host bridges it to a TCP
 *                   shielded worker. Only ciphertext frames cross it, which is
 *                   the phone topology's socket rule made concrete.
 *
 * The per-shape flow is harness/split-harness.c verbatim, fixture and all, so
 * a pVM run against the same worker must reproduce the x86 and S21+ digests
 * in REPORT.md section 3 bit for bit (invariant 6).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <poll.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <dirent.h>
#include <linux/vm_sockets.h>
#include <time.h>
#include <unistd.h>
#include <android/log.h>

#include "vm_payload.h"
#include "third_party/tweetnacl.h"
#include "anchor-core.h"
#include "shielded-field.h"
#include "shielded-simd.h"
#include "shielded-wire.h"
#include "worker-client.h"
#include "fixture.h"

#define TAG "anchor-pvm"
#define CTRL_PORT   7777

/* The transport key: an Ed25519 pair minted INSIDE the VM at every boot, the
 * identity the relay pins this tunnel to (keyFp = sha256 of its SPKI) and the
 * thing the attested key vouches for by signing (SPKI || nonce). The secret
 * half never leaves the VM. TweetNaCl (public domain) does the arithmetic;
 * randombytes() below is the guest's getrandom. */
static unsigned char g_tpk[32], g_tsk[64];
void randombytes(unsigned char *p, unsigned long long n) {
    while (n) { ssize_t r = getrandom(p, (size_t)n, 0); if (r <= 0) abort(); p += r; n -= (unsigned long long)r; }
}
static const uint8_t ED25519_SPKI_PREFIX[12] = { 0x30,0x2a,0x30,0x05,0x06,0x03,0x2b,0x65,0x70,0x03,0x21,0x00 };
#define WORKER_PORT 7778
#define MODEL_PORT  7779
#define PADS_PORT   7780     /* owner -> guest: dealt-pad shipments into the bank dir (PADS <name> <bytes>\n, bytes) */
#define ECHO_PORT   7780
#define MAX_SHAPES  16

/* ---- the mouth: every line to stdout (debug VMs), logcat, and the control vsock ---- */
static int g_ctl = -1;
static void outf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void outf(const char *fmt, ...) {
    char line[4096]; va_list ap; va_start(ap, fmt); int n = vsnprintf(line, sizeof line - 1, fmt, ap); va_end(ap);
    if (n < 0) return; if ((size_t)n > sizeof line - 2) n = sizeof line - 2;
    line[n] = '\n'; line[n + 1] = 0;
    fputs(line, stdout); fflush(stdout);
    __android_log_print(ANDROID_LOG_INFO, TAG, "%.*s", n, line);
    if (g_ctl >= 0) { const char *p = line; size_t left = (size_t)n + 1; while (left) { ssize_t w = write(g_ctl, p, left); if (w <= 0) { close(g_ctl); g_ctl = -1; break; } p += w; left -= (size_t)w; } }
}
#define OUT(...) outf(__VA_ARGS__)

static void hexline(const char *label, const uint8_t *p, size_t n) {
    const size_t CH = 512;
    OUT("%s bytes=%zu chunks=%zu", label, n, (n + CH - 1) / CH);
    for (size_t off = 0; off < n; off += CH) {
        size_t m = n - off < CH ? n - off : CH; char s[CH * 2 + 1];
        for (size_t i = 0; i < m; i++) sprintf(s + 2 * i, "%02x", p[off + i]);
        s[2 * m] = 0; OUT("%s[%zu] %s", label, off / CH, s);
    }
}

/* ---- vsock listeners: bound before notifyPayloadReady, accepted when the owner arrives ---- */
static int vs_bind(unsigned port) {
    int ls = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (ls < 0) return -1;
    struct sockaddr_vm sa = { .svm_family = AF_VSOCK, .svm_port = port, .svm_cid = VMADDR_CID_ANY };
    if (bind(ls, (struct sockaddr *)&sa, sizeof sa) != 0 || listen(ls, 4) != 0) { close(ls); return -1; }
    return ls;
}
static int vs_accept(int ls, int grace_ms) {
    if (ls < 0) return -1;
    struct pollfd pf = { .fd = ls, .events = POLLIN };
    if (poll(&pf, 1, grace_ms) <= 0) return -1;
    return accept(ls, NULL, NULL);
}
static int read_line(int fd, char *buf, size_t cap) {
    size_t n = 0;
    while (n + 1 < cap) { char c; ssize_t r = read(fd, &c, 1); if (r <= 0) return -1; if (c == '\n') break; buf[n++] = c; }
    buf[n] = 0; return (int)n;
}

/* ---- attestation: the certificate a verifier will check, bound to the owner's challenge ---- */
static size_t unhex(const char *hex, uint8_t *out, size_t cap) {
    size_t n = 0;
    for (; n < cap && hex[2 * n] && hex[2 * n + 1]; n++) { unsigned v; if (sscanf(hex + 2 * n, "%2x", &v) != 1) break; out[n] = (uint8_t)v; }
    return n;
}
/* Request the certificate over `hex` (32 bytes) and sign `bound_hex` with the
 * attested key: the relay's binding is challenge = sha256(SPKI || nonce) and
 * signature over (SPKI || nonce). Ends with "ATTEST end" whatever happened. */
static void attest(const char *hex, const char *bound_hex) {
    uint8_t ch[32] = {0}; unhex(hex, ch, 32);
    uint8_t bound[1024]; size_t blen = unhex(bound_hex, bound, sizeof bound);
    if (!blen) { memcpy(bound, ch, 32); blen = 32; }
    AVmAttestationResult *res = NULL;
    AVmAttestationStatus st = AVmPayload_requestAttestation(ch, sizeof ch, &res);
    OUT("ATTEST status=%s code=%d", AVmAttestationStatus_toString(st), (int)st);
    if (st == ATTESTATION_OK && res) {
        size_t n = AVmAttestationResult_getCertificateCount(res);
        OUT("ATTEST certs=%zu", n);
        for (size_t i = 0; i < n; i++) {
            size_t sz = AVmAttestationResult_getCertificateAt(res, i, NULL, 0);
            uint8_t *c = malloc(sz); if (!c) continue;
            AVmAttestationResult_getCertificateAt(res, i, c, sz);
            char label[24]; snprintf(label, sizeof label, "CERT%zu", i); hexline(label, c, sz); free(c);
        }
        size_t ssz = AVmAttestationResult_sign(res, bound, blen, NULL, 0);
        uint8_t *sig = malloc(ssz);
        if (sig) { AVmAttestationResult_sign(res, bound, blen, sig, ssz); hexline("SIG", sig, ssz); free(sig); }
        AVmAttestationResult_free(res);
    }
    OUT("ATTEST end");
}

/* ---- the untrusted half: a real worker over the bridge, or the in-guest stand-in ---- */
typedef struct {
    int bridge;
    wc_client wc;
    /* local stand-in */
    const int8_t *const *W; int64_t K; const int64_t *N; int n;
    int64_t *xm; uint8_t *reply; size_t rlen;
} wk;

static int wk_exchange(wk *w, const int8_t *planes, const uint8_t **reply, size_t *len, int *ywidth) {
    if (w->bridge) { int rc = wc_exchange(&w->wc, planes, 1, reply, len); *ywidth = w->wc.ywidth; return rc == SH_OK; }
    const int8_t *p0 = planes, *p1 = planes + w->K, *p2 = planes + 2 * w->K;
    for (int64_t k = 0; k < w->K; k++) w->xm[k] = sh_crt(p0[k], p1[k], p2[k]);
    size_t off = 0;
    for (int nd = 0; nd < w->n; nd++)
        for (int64_t j = 0; j < w->N[nd]; j++) {
            const int8_t *row = w->W[nd] + j * w->K; int64_t acc = 0;
            for (int64_t k = 0; k < w->K; k++) acc += w->xm[k] * row[k];
            int32_t b = (int32_t)sh_balanced(acc); memcpy(w->reply + off, &b, 4); off += 4;
        }
    *reply = w->reply; *len = w->rlen; *ywidth = 4; return 1;
}

static int rng_os(void *buf, size_t n) {
    uint8_t *p = buf;
    while (n) { ssize_t r = getrandom(p, n, 0); if (r < 0) return -1; p += r; n -= (size_t)r; }
    return 0;
}
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(double *v, int n) { qsort(v, (size_t)n, sizeof *v, cmp_d); return v[n / 2]; }

/* The refill kernel, generic vs SDOT, on THE SAME thread back to back, so the
 * comparison is the kernel and not which core the scheduler handed the VM's
 * vCPU this second (the /foreground cpuset mixes A510s and A715s). Interleaved
 * rounds, best of each; the outputs must agree byte for byte. */
static void bench_refill(int64_t K, int64_t N) {
    int8_t *W = malloc((size_t)K * N); uint8_t *planes = malloc((size_t)3 * K);
    int32_t *ug = malloc((size_t)N * 4), *un = malloc((size_t)N * 4), *acc = malloc((size_t)12 * N * 4);
    if (!W || !planes || !ug || !un || !acc) return;
    uint32_t s = 0x9e3779b9u;
    for (int64_t i = 0; i < K * N; i++) { s = s * 1103515245u + 12345u; W[i] = (int8_t)((int)((s >> 8) % 239) - 119); }
    for (int64_t i = 0; i < 3 * K; i++) { s = s * 1103515245u + 12345u; planes[i] = (uint8_t)((s >> 8) % 251); }
    double bg = 1e18, bn = 1e18;
    for (int round = 0; round < 5; round++) {
        double t0 = now_us(); sh_simd_generic_refill(planes, 1, W, K, N, ug, N, acc); double t1 = now_us();
        sh_simd_neon_refill(planes, 1, W, K, N, un, N, acc); double t2 = now_us();
        if (t1 - t0 < bg) bg = t1 - t0; if (t2 - t1 < bn) bn = t2 - t1;
    }
    OUT("{\"bench\":\"refill\",\"K\":%" PRId64 ",\"N\":%" PRId64 ",\"generic_us\":%.1f,\"neon_sdot_us\":%.1f,\"speedup\":%.2f,\"agree\":%s,\"gmac_s\":{\"generic\":%.2f,\"neon\":%.2f}}",
        K, N, bg, bn, bg / bn, memcmp(ug, un, (size_t)N * 4) == 0 ? "true" : "false",
        12.0 * K * N / bg / 1e3, 12.0 * K * N / bn / 1e3);   /* the kernel dots 3 planes x 4 rows per weight row */
    free(W); free(planes); free(ug); free(un); free(acc);
}

/* ---- ENGINE mode: the whole inference engine in this VM (PLAN.md phase 3, steps 5-6) ----
 * The owner streams the public model over vsock 7779 into a file of ours (a memfd, or /data
 * when the VM has encrypted storage), bridges the worker on 7778, and this dlopens the
 * libraries from the APK (RTLD_GLOBAL, dependency order) and hands everything to
 * libengine.so's engine_main. Nothing here touches a secret: the model is public, the
 * worker sees ciphertext, and the calibration is public data under the attested codeHash. */
static int read_exact(int fd, void *buf, size_t n) {
    uint8_t *p = buf; while (n) { ssize_t r = read(fd, p, n); if (r <= 0) return -1; p += r; n -= (size_t)r; } return 0;
}
/* Where the model lives: the VM's encrypted storage when the owner attached some
 * (persistent per VM instance, so the stream happens once), else a memfd
 * (which this payload domain is denied on the phone, measured EACCES), else
 * /data. `*existing` says a same-sized file was already there. */
/* The cache key: the owner's claimed sha256 of the model, kept in a sidecar next
 * to the file once a stream completed. The model is public data, so a wrong
 * claim costs output quality, never a secret; the sidecar is what makes a
 * same-sized different model stream again instead of being mistaken for cached. */
static char g_model_sha[80] = "";
static void sidecar_path(char *out, size_t cap, const char *es) { snprintf(out, cap, "%s/model.gguf.sha256", es); }
static int model_file(uint64_t bytes, int *existing) {
    *existing = 0;
    const char *es = AVmPayload_getEncryptedStoragePath();
    if (es) {
        char path[512], side[512]; snprintf(path, sizeof path, "%s/model.gguf", es); sidecar_path(side, sizeof side, es);
        struct stat st; char have[80] = "";
        { FILE *f = fopen(side, "r"); if (f) { if (!fgets(have, sizeof have, f)) have[0] = 0; fclose(f); have[strcspn(have, "\n")] = 0; } }
        if (stat(path, &st) == 0 && (uint64_t)st.st_size == bytes && g_model_sha[0] && !strcmp(have, g_model_sha)) {
            int fd = open(path, O_RDONLY); if (fd >= 0) { *existing = 1; return fd; }
        }
        unlink(side);                                             /* whatever is there is not what the owner is offering */
        int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd >= 0 && ftruncate(fd, (off_t)bytes) == 0) return fd;
        OUT("ENGINE encrypted storage %s: %s", path, strerror(errno));
        if (fd >= 0) close(fd);
    }
    int fd = memfd_create("model", 0);
    if (fd >= 0 && ftruncate(fd, (off_t)bytes) == 0) return fd;
    OUT("ENGINE memfd: fd=%d ftruncate errno=%d (%s)", fd, errno, strerror(errno));
    if (fd >= 0) close(fd);
    fd = open("/data/anchor-model.gguf", O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd >= 0 && ftruncate(fd, (off_t)bytes) == 0) return fd;
    if (fd >= 0) close(fd);
    return -1;
}
/* The stream: 8-byte length from the owner, one byte back ('K' = keep, I have
 * it; 'S' = send), then the bytes. */
static int receive_model(int ls_model, uint64_t bytes, int *out_fd) {
    int c = vs_accept(ls_model, 60000);
    if (c < 0) { OUT("ENGINE no model stream from the owner"); return -1; }
    uint64_t hdr = 0; if (read_exact(c, &hdr, 8) != 0 || hdr != bytes) { OUT("ENGINE model stream header %" PRIu64 " != %" PRIu64, hdr, bytes); close(c); return -1; }
    int existing = 0, fd = model_file(bytes, &existing);
    if (fd < 0) { OUT("ENGINE nowhere to put %" PRIu64 " bytes of model (encrypted storage, memfd and /data all refused)", bytes); close(c); return -1; }
    if (existing) { (void)!write(c, "K", 1); close(c); OUT("ENGINE model %" PRIu64 " MiB already in the VM's encrypted storage", bytes >> 20); *out_fd = fd; return 0; }
    (void)!write(c, "S", 1);
    static uint8_t buf[1 << 20]; uint64_t got = 0, mark = 0; double t0 = now_us();
    while (got < bytes) {
        size_t want = bytes - got < sizeof buf ? (size_t)(bytes - got) : sizeof buf;
        ssize_t r = read(c, buf, want); if (r <= 0) { OUT("ENGINE model stream ended at %" PRIu64, got); close(c); return -1; }
        if (write(fd, buf, (size_t)r) != r) { OUT("ENGINE model write failed at %" PRIu64 ": %s", got, strerror(errno)); close(c); return -1; }
        got += (uint64_t)r;
        if (got - mark >= (200u << 20)) { mark = got; OUT("ENGINE model %" PRIu64 " MiB received", got >> 20); }
    }
    close(c); fsync(fd);
    if (g_model_sha[0] && AVmPayload_getEncryptedStoragePath()) {    /* remember what this file is, for next time */
        char side[512]; sidecar_path(side, sizeof side, AVmPayload_getEncryptedStoragePath());
        FILE *f = fopen(side, "w"); if (f) { fprintf(f, "%s\n", g_model_sha); fclose(f); }
    }
    OUT("ENGINE model %" PRIu64 " MiB received in %.1f s", got >> 20, (now_us() - t0) / 1e6);
    *out_fd = fd; return 0;
}
/* ---- dealt pads (shielded/dealer/PLAN.md, anchor_pads.h) -------------------
 * The VM mints an X25519 pad key at boot and announces it (PADKEY) right after
 * the transport key; the owner presents it to the relay with the attestation.
 * The platform's seed comes back boxed to that key (PADSEED); requests the
 * owner relays are signed here with the transport key (PADSIGN); windows the
 * engine asks for go out as PADWIN and come back verified against the ledger
 * key (PADLEDGER). Shipments stream in on PADS_PORT into the bank directory. */
#include "anchor_pads.h"
#include "shielded-pads.h"
static uint8_t g_ppk[32], g_psk[32], g_ledger_pk[32], g_seed[32], g_seed_id[16];
static char g_seed_id_hex[33] = "", g_pad_name[64] = "", g_pads_dir[512] = "";
static int g_have_ledger = 0, g_have_seed = 0;
static void pads_dir(char *out, size_t cap) {
    const char *es = AVmPayload_getEncryptedStoragePath();
    if (es) snprintf(out, cap, "%s/pads", es); else snprintf(out, cap, "/data/anchor-pads");
    mkdir(out, 0700);
}
/* Shipments are named <seed_id>-<index0>-<count>.pads. Files of another
 * seed (an older key or epoch) can never open here and are dropped at
 * PADSEED. Spent shipments are the engine's call (SHIELDED_PAD_PRUNE: it
 * knows the lowest live cursor; a window edge alone is NOT safe, a lagging
 * group may still read below it). `below` stays for that engine-side use. */
static int pads_prune(const char *keep_seed, unsigned long long below) {
    if (!g_pads_dir[0]) return 0;
    DIR *d = opendir(g_pads_dir); if (!d) return 0;
    int n = 0; struct dirent *e;
    while ((e = readdir(d))) {
        size_t len = strlen(e->d_name);
        if (len < 6 || strcmp(e->d_name + len - 5, ".pads")) continue;
        char sid[33] = ""; unsigned long long i0 = 0, cnt = 0;
        int drop = 0;
        if (sscanf(e->d_name, "%32[0-9a-f]-%llu-%llu.pads", sid, &i0, &cnt) != 3) drop = 1;
        else if (keep_seed && strcmp(sid, keep_seed)) drop = 1;
        else if (below && i0 + cnt <= below) drop = 1;
        if (!drop) continue;
        char path[700]; snprintf(path, sizeof path, "%s/%s", g_pads_dir, e->d_name);
        if (unlink(path) == 0) n++;
    }
    closedir(d);
    return n;
}
/* One shipment per connection: "PADS <name> <bytes>\n"; the VM answers one
 * byte, 'H' (have it already, same size: nothing more is sent) or 'G' (go),
 * then the bytes follow, written tmp-then-rename so the engine's reader
 * never sees a partial file. */
static void *pads_receiver(void *arg) {
    int ls = (int)(intptr_t)arg;
    for (;;) {
        int c = vs_accept(ls, 3600000);
        if (c < 0) continue;
        char hdr[256]; size_t n = 0;
        while (n + 1 < sizeof hdr) { char ch; if (read(c, &ch, 1) != 1) { n = 0; break; } if (ch == '\n') break; hdr[n++] = ch; }
        hdr[n] = 0;
        char name[128] = ""; unsigned long long bytes = 0;
        if (n == 0 || sscanf(hdr, "PADS %127s %llu", name, &bytes) != 2 || strchr(name, '/') || strstr(name, "..")) { close(c); continue; }
        char tmp[700], fin[700]; snprintf(tmp, sizeof tmp, "%s/.%s.tmp", g_pads_dir, name); snprintf(fin, sizeof fin, "%s/%s", g_pads_dir, name);
        struct stat st;
        if (stat(fin, &st) == 0 && (unsigned long long)st.st_size == bytes) { (void)!write(c, "H", 1); close(c); continue; }
        (void)!write(c, "G", 1);
        int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        unsigned long long got = 0; static char buf[1 << 16];
        while (fd >= 0 && got < bytes) {
            size_t want = bytes - got < sizeof buf ? (size_t)(bytes - got) : sizeof buf;
            ssize_t r = read(c, buf, want); if (r <= 0) break;
            if (write(fd, buf, (size_t)r) != r) break;
            got += (unsigned long long)r;
        }
        if (fd >= 0) { fsync(fd); close(fd); }
        if (got == bytes && rename(tmp, fin) == 0) { (void)!write(c, "K", 1); OUT("PADS %s %llu bytes", name, got); }
        else { unlink(tmp); (void)!write(c, "E", 1); OUT("PADS %s FAILED at %llu of %llu", name, got, bytes); }
        close(c);
    }
    return NULL;
}

typedef int (*engine_main_fn)(int, int, int, const char *, const char *, const char *, int, int, const anchor_pads *);
static void run_engine(int ls_wk, int ls_model, int ls_pads, const char *prompt, int n_predict, int threads, uint64_t model_bytes, int with_pads) {
    const char *apk = AVmPayload_getApkContentsPath();
    char lib_dir[512], calib[512]; snprintf(lib_dir, sizeof lib_dir, "%s/lib/arm64-v8a", apk); snprintf(calib, sizeof calib, "%s/assets/model.calib", apk);
    int worker_fd = vs_accept(ls_wk, 60000);
    if (worker_fd < 0) { OUT("ENGINE no worker bridge from the owner"); return; }
    int model_fd = -1;
    if (receive_model(ls_model, model_bytes, &model_fd) != 0) { close(worker_fd); return; }
    static const char *libs[] = { "libc++_shared.so", "libggml-base.so", "libggml.so", "libggml-cpu.so", "libllama.so", "libengine.so" };
    void *h = NULL;
    for (unsigned i = 0; i < sizeof libs / sizeof *libs; i++) {
        char path[600]; snprintf(path, sizeof path, "%s/%s", lib_dir, libs[i]);
        h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (!h) { OUT("ENGINE dlopen %s: %s", libs[i], dlerror()); close(worker_fd); close(model_fd); return; }
    }
    engine_main_fn em = (engine_main_fn)dlsym(h, "engine_main");
    if (!em) { OUT("ENGINE libengine.so has no engine_main"); return; }
    anchor_pads pads = { g_tsk, g_ledger_pk, g_pad_name, g_seed_id_hex };
    const anchor_pads *pp = NULL;
    if (with_pads) {
        if (!g_have_seed || !g_have_ledger) { OUT("ENGINE pads requested but no seed/ledger from the owner; refusing to mint for myself"); close(worker_fd); close(model_fd); return; }
        pads_dir(g_pads_dir, sizeof g_pads_dir);
        char hs[65], hid[33], hsk[65];
        sh_pads_bin2hex(g_seed, 32, hs); sh_pads_bin2hex(g_seed_id, 16, hid); sh_pads_bin2hex(g_psk, 32, hsk);
        setenv("SHIELDED_PAD_SOURCE", g_pads_dir, 1); setenv("SHIELDED_PAD_SEED", hs, 1); setenv("SHIELDED_PAD_SEED_ID", hid, 1); setenv("SHIELDED_PAD_SK", hsk, 1);
        setenv("SHIELDED_PAD_PRUNE", "1", 1);      /* the encrypted store's copy is ours: spent shipments go */
        setenv("SHIELDED_PAD_CHECK", "1", 0);        /* the pVM checks every imported pad against the weights: a wrong dealer is refused before use */
        /* Pin the model: only shipments the dealer minted for THIS calibration
         * (SHA-512/256 of the calib file, what shielded-dealer records) are used. */
        { FILE *cf = fopen(calib, "rb"); if (cf) { static uint8_t cb[1 << 20]; size_t n = fread(cb, 1, sizeof cb, cf); fclose(cf);
            uint8_t dg[64]; crypto_hash(dg, cb, n); char dh[65]; sh_pads_bin2hex(dg, 32, dh); setenv("SHIELDED_PAD_MODEL_DIGEST", dh, 1); } }
        memset(hs, 0, sizeof hs); memset(hsk, 0, sizeof hsk);
        pthread_t th; pthread_create(&th, NULL, pads_receiver, (void *)(intptr_t)ls_pads); pthread_detach(th);
        pp = &pads;
        OUT("ENGINE dealt pads on: bank %s, seed %s", g_pads_dir, g_seed_id_hex);
    }
    OUT("ENGINE libraries loaded from %s; starting", lib_dir);
    int rc = em(g_ctl, worker_fd, model_fd, lib_dir, calib, prompt, n_predict, threads, pp);
    OUT("ENGINE exit %d", rc);
    close(model_fd);
}

/* split-harness.c's main, as a function: same fixture, same order of draws, same digest */
static void run_shape(int64_t K, int64_t N, int n_nodes, int iters, int xmax, int bridge_fd) {
    if (xmax <= 0) { double s_ = 900.0 * sqrt(896.0 / (double)K); xmax = (int)(s_ < 1 ? 1 : s_); }
    fx_rng g = { FX_SEED };
    int8_t *w[AN_MAX_NODES] = { 0 };
    for (int i = 0; i < n_nodes; i++) if (!(w[i] = fx_weight(&g, K, N))) { OUT("{\"K\":%" PRId64 ",\"error\":\"oom\"}", K); return; }
    int64_t *x = malloc((size_t)K * 8); int8_t *planes = malloc((size_t)3 * K), *planes2 = malloc((size_t)3 * K);
    int64_t Ks[AN_MAX_NODES], Ns[AN_MAX_NODES];
    for (int i = 0; i < n_nodes; i++) { Ks[i] = K; Ns[i] = N; }
    const size_t footprint = an_footprint(n_nodes, Ks, Ns);

    an_ctx *a = an_create(rng_os);
    if (!a || !x || !planes || !planes2) { OUT("{\"K\":%" PRId64 ",\"error\":\"oom\"}", K); return; }
    for (int i = 0; i < n_nodes; i++) an_add_weight(a, w[i], K, N);
    double t0 = now_us();
    if (an_prepare(a) != AN_OK) { OUT("{\"K\":%" PRId64 ",\"error\":\"prepare\"}", K); return; }
    double prepare_us = now_us() - t0;

    wk W; memset(&W, 0, sizeof W);
    W.bridge = bridge_fd >= 0;
    if (W.bridge) {
        for (int i = 0; i < n_nodes; i++) if (wc_add(&W.wc, K, N) < 0) { OUT("{\"K\":%" PRId64 ",\"error\":\"wc_add %s\"}", K, W.wc.err); return; }
        sh_pipe *pipe = sh_pipe_open_fd(bridge_fd);
        if (!pipe || wc_install(&W.wc, pipe, (const int8_t *const *)w, 0) != SH_OK) { OUT("{\"K\":%" PRId64 ",\"error\":\"install %s\"}", K, W.wc.err); return; }
    } else {
        W.W = (const int8_t *const *)w; W.K = K; W.N = Ns; W.n = n_nodes;
        W.xm = malloc((size_t)K * 8); W.rlen = (size_t)n_nodes * N * 4; W.reply = malloc(W.rlen);
    }

    double *tp = malloc(iters * 8), *tm = malloc(iters * 8), *tw = malloc(iters * 8), *tf = malloc(iters * 8);
    int exact = 1, verified = 1, lie_rejected = 0, pads_distinct = 0, ywidth = 0;
    int64_t peak = 0; uint64_t digest = 1469598103934665603ull;
    const uint8_t *reply; size_t rlen;

    fx_activation(&g, K, x, xmax);
    for (int r = 0; r < 2; r++) {
        if (an_pad_gen(a) != AN_OK || an_mask(a, x, r ? planes2 : planes) != AN_OK) return;
        if (!wk_exchange(&W, r ? planes2 : planes, &reply, &rlen, &ywidth)) { OUT("{\"K\":%" PRId64 ",\"error\":\"exchange %s\"}", K, W.wc.err); return; }
        if (an_finish(a, reply, rlen, ywidth) != AN_OK) verified = 0;
    }
    pads_distinct = memcmp(planes, planes2, (size_t)3 * K) != 0;

    if (an_pad_gen(a) == AN_OK && an_mask(a, x, planes) == AN_OK && wk_exchange(&W, planes, &reply, &rlen, &ywidth)) {
        uint8_t *evil = malloc(rlen);
        if (evil) { memcpy(evil, reply, rlen); evil[rlen / 2] ^= 1; lie_rejected = an_finish(a, evil, rlen, ywidth) == AN_ERR_VERIFY; free(evil); }
    }

    int done = 0;
    for (int it = 0; it < iters; it++) {
        fx_activation(&g, K, x, xmax);
        double a0 = now_us(); if (an_pad_gen(a) != AN_OK) break;
        double a1 = now_us(); if (an_mask(a, x, planes) != AN_OK) break;
        double a2 = now_us(); if (!wk_exchange(&W, planes, &reply, &rlen, &ywidth)) { OUT("{\"K\":%" PRId64 ",\"error\":\"exchange %s\"}", K, W.wc.err); break; }
        double a3 = now_us(); int rc = an_finish(a, reply, rlen, ywidth);
        double a4 = now_us();
        if (rc != AN_OK) { verified = 0; break; }
        if (an_check_local(a) != AN_OK) { exact = 0; break; }
        for (int nd = 0; nd < n_nodes; nd++) { digest ^= an_y_digest(a, nd); digest *= 1099511628211ull; }
        { int64_t pk = an_peak_abs_y(a); if (pk > peak) peak = pk; }
        tp[it] = a1 - a0; tm[it] = a2 - a1; tw[it] = a3 - a2; tf[it] = a4 - a3; done++;
    }
    uint64_t pads = 0, ex = 0, vf = 0; an_stats(a, &pads, &ex, &vf);
    if (W.bridge) wc_close(&W.wc);
    const int pass = exact && verified && lie_rejected && pads_distinct && done == iters;
    OUT("{\"rung\":\"%s\",\"K\":%" PRId64 ",\"N\":%" PRId64 ",\"nodes\":%d,\"iters\":%d,\"done\":%d,\"xmax\":%d,\"ywidth\":%d,"
        "\"exact\":%s,\"verified\":%s,\"lie_rejected\":%s,\"pads_distinct\":%s,"
        "\"footprint_kb\":%zu,\"prepare_us\":%.0f,\"peak_abs_y\":%" PRId64 ",\"y_digest\":\"%016" PRIx64 "\","
        "\"median_us\":{\"pad\":%.1f,\"mask\":%.1f,\"worker\":%.1f,\"finish\":%.1f},"
        "\"pads_issued\":%" PRIu64 ",\"verify_fail\":%" PRIu64 ",\"PASS\":%s}",
        W.bridge ? "avf-pvm-gpu" : "avf-pvm-local", K, N, n_nodes, iters, done, xmax, ywidth,
        exact?"true":"false", verified?"true":"false", lie_rejected?"true":"false", pads_distinct?"true":"false",
        footprint / 1024, prepare_us, peak, digest,
        done?median(tp,done):0, done?median(tm,done):0, done?median(tw,done):0, done?median(tf,done):0,
        pads, vf, pass?"true":"false");
    an_destroy(a);
    for (int i = 0; i < n_nodes; i++) free(w[i]);
    free(x); free(planes); free(planes2); free(tp); free(tm); free(tw); free(tf); free(W.xm); free(W.reply);
}

int AVmPayload_main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int ls_ctl = vs_bind(CTRL_PORT), ls_wk = vs_bind(WORKER_PORT), ls_model = vs_bind(MODEL_PORT), ls_pads = vs_bind(PADS_PORT);
    crypto_sign_keypair(g_tpk, g_tsk);
    crypto_box_keypair(g_ppk, g_psk);                 /* the pad key: the platform's seed is boxed to it */
    AVmPayload_notifyPayloadReady();
    g_ctl = vs_accept(ls_ctl, 20000);
    {   /* the first thing the owner hears is the transport key it will present to the relay */
        uint8_t spki[44]; memcpy(spki, ED25519_SPKI_PREFIX, 12); memcpy(spki + 12, g_tpk, 32);
        char hx[89]; for (int i = 0; i < 44; i++) sprintf(hx + 2 * i, "%02x", spki[i]); hx[88] = 0;
        OUT("SPKI %s", hx);
        char pk[65]; sh_pads_bin2hex(g_ppk, 32, pk);
        OUT("PADKEY %s", pk);
    }
    OUT("ANCHOR start in pVM apk=%s control=%s", AVmPayload_getApkContentsPath(), g_ctl >= 0 ? "owner-connected" : "none");
    {
        FILE *f = fopen("/proc/cpuinfo", "r"); char line[1024]; char feats[1024] = "?";
        if (f) { while (fgets(line, sizeof line, f)) if (!strncmp(line, "Features", 8)) { strncpy(feats, line + 10, sizeof feats - 1); break; } fclose(f); }
        feats[strcspn(feats, "\n")] = 0;
        OUT("ANCHOR cpu nproc=%ld features=%s", sysconf(_SC_NPROCESSORS_ONLN), feats);
    }

    /* the owner's instructions; without an owner (a vm-tool run) the built-in local self-test */
    int bridge = 0, n_shapes = 0; int64_t SK[MAX_SHAPES], SN[MAX_SHAPES]; int Snode[MAX_SHAPES], Siter[MAX_SHAPES], Sx[MAX_SHAPES];
    int engine = 0, eng_n = 8, eng_threads = 4; uint64_t eng_model = 0; static char eng_prompt[2048] = "The capital of France is";
    int echo = 0, with_pads = 0;
    if (g_ctl >= 0) {
        char l[2400]; static char bound[2100] = "";
        while (read_line(g_ctl, l, sizeof l) >= 0) {
            if (!strncmp(l, "BOUND ", 6)) { strncpy(bound, l + 6, sizeof bound - 1); bound[sizeof bound - 1] = 0; }
            else if (!strncmp(l, "CHAL ", 5)) attest(l + 5, bound);
            else if (!strncmp(l, "PADLEDGER ", 10)) {  /* the relay's ledger key: windows are verified against it */
                g_have_ledger = sh_pads_hex2bin(l + 10, g_ledger_pk, 32);
                OUT("PADLEDGER %s", g_have_ledger ? "ok" : "fail");
            }
            else if (!strncmp(l, "PADSEED ", 8)) {     /* PADSEED <name> <seed_id> <epoch> <epk> <nonce> <box> */
                char name[64] = "", sid[33] = "", epk_h[65] = "", nonce_h[25] = "", box_h[97] = ""; unsigned epoch = 0;
                uint8_t epk[32], nonce[12], box[48];
                if (sscanf(l + 8, "%63s %32s %u %64s %24s %96s", name, sid, &epoch, epk_h, nonce_h, box_h) == 6 &&
                    sh_pads_hex2bin(sid, g_seed_id, 16) && sh_pads_hex2bin(epk_h, epk, 32) && sh_pads_hex2bin(nonce_h, nonce, 12) && sh_pads_hex2bin(box_h, box, 48) &&
                    sh_pads_seed_open(epk, nonce, box, 48, g_psk, g_ppk, g_seed) == 0) {
                    strncpy(g_pad_name, name, sizeof g_pad_name - 1); strncpy(g_seed_id_hex, sid, 32); g_have_seed = 1;
                    if (!g_pads_dir[0]) pads_dir(g_pads_dir, sizeof g_pads_dir);
                    int dropped = pads_prune(sid, 0);
                    OUT("PADSEED ok %s", sid);
                    if (dropped) OUT("PADS dropped %d shipment(s) of other seeds", dropped);
                } else { g_have_seed = 0; OUT("PADSEED fail"); }
            }
            else if (!strncmp(l, "PADSIGN ", 8)) {     /* PADSIGN <kind> <nonce> [fields...] -> PADSIG <hex> */
                char *save = NULL, *kind = strtok_r(l + 8, " ", &save), *nonce = kind ? strtok_r(NULL, " ", &save) : NULL;
                const char *fields[8]; size_t nf = 0; char *f;
                while (nonce && nf < 8 && (f = strtok_r(NULL, " ", &save))) fields[nf++] = f;
                if (kind && nonce) { uint8_t sig[64]; char hs[129]; sh_pads_request_sign(g_tsk, kind, fields, nf, nonce, sig); sh_pads_bin2hex(sig, 64, hs); OUT("PADSIG %s", hs); }
                else OUT("PADSIG fail");
            }
            else if (!strncmp(l, "WORKER ", 7)) bridge = !strcmp(l + 7, "bridge");
            else if (!strncmp(l, "ENGINE ", 7)) {          /* ENGINE model_bytes=N n=N threads=N prompt=<hex> */
                engine = 1; char *q;
                if ((q = strstr(l, "model_bytes="))) eng_model = strtoull(q + 12, NULL, 10);
                if ((q = strstr(l, "model_sha256="))) { strncpy(g_model_sha, q + 13, 64); g_model_sha[64] = 0; }
                if ((q = strstr(l, " n="))) eng_n = atoi(q + 3);
                if ((q = strstr(l, "threads="))) eng_threads = atoi(q + 8);
                if ((q = strstr(l, "prompt="))) { size_t k = unhex(q + 7, (uint8_t *)eng_prompt, sizeof eng_prompt - 1); eng_prompt[k] = 0; }
                with_pads = strstr(l, " pads=1") != NULL;
            }
            else if (!strncmp(l, "SHAPE ", 6) && n_shapes < MAX_SHAPES) {
                long long k, n; int nd, it, xm;
                if (sscanf(l + 6, "%lld %lld %d %d %d", &k, &n, &nd, &it, &xm) == 5) { SK[n_shapes] = k; SN[n_shapes] = n; Snode[n_shapes] = nd; Siter[n_shapes] = it; Sx[n_shapes] = xm; n_shapes++; }
            }
            else if (!strcmp(l, "ECHO")) echo = 1;
            else if (!strcmp(l, "RUN")) break;
        }
    }
    if (echo) {   /* the vsock round trip itself, app <-> guest, nothing else in the loop */
        int ls = vs_bind(ECHO_PORT); int c = vs_accept(ls, 20000);
        OUT("ECHO %s", c >= 0 ? "connected" : "no peer");
        if (c >= 0) { static uint8_t b[65536]; ssize_t r; while ((r = read(c, b, sizeof b)) > 0) { if (write(c, b, (size_t)r) != r) break; } close(c); }
        if (ls >= 0) close(ls);
        OUT("END");
        if (g_ctl >= 0) { shutdown(g_ctl, SHUT_WR); close(g_ctl); }
        sleep(1); return 0;
    }
    if (engine) {
        OUT("ANCHOR engine mode: model %" PRIu64 " bytes, %d tokens, %d threads", eng_model, eng_n, eng_threads);
        run_engine(ls_wk, ls_model, ls_pads, eng_prompt, eng_n, eng_threads, eng_model, with_pads);
        OUT("END");
        if (ls_model >= 0) close(ls_model); if (ls_wk >= 0) close(ls_wk); if (ls_ctl >= 0) close(ls_ctl);
        if (g_ctl >= 0) { shutdown(g_ctl, SHUT_WR); close(g_ctl); }
        sleep(1); return 0;
    }
    if (n_shapes == 0) { SK[0]=256; SN[0]=256; Snode[0]=1; Siter[0]=30; Sx[0]=0; SK[1]=896; SN[1]=896; Snode[1]=1; Siter[1]=30; Sx[1]=0; SK[2]=896; SN[2]=4864; Snode[2]=2; Siter[2]=12; Sx[2]=0; n_shapes = 3; }
    OUT("ANCHOR worker=%s shapes=%d", bridge ? "bridge" : "local", n_shapes);
    bench_refill(896, 896); bench_refill(896, 4864);

    for (int s = 0; s < n_shapes; s++) {
        int fd = -1;
        if (bridge) { fd = vs_accept(ls_wk, 20000); if (fd < 0) { OUT("{\"K\":%" PRId64 ",\"error\":\"no worker bridge\"}", SK[s]); continue; } }
        run_shape(SK[s], SN[s], Snode[s], Siter[s], Sx[s], fd);
    }
    OUT("END");
    if (ls_model >= 0) close(ls_model);
    if (ls_wk >= 0) close(ls_wk);
    if (ls_ctl >= 0) close(ls_ctl);
    if (g_ctl >= 0) { shutdown(g_ctl, SHUT_WR); close(g_ctl); }
    sleep(1);
    return 0;
}
