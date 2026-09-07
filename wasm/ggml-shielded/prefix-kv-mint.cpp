/* prefix-kv-mint: the platform's shared-prefix service, offline.
 *
 *   GGML_CPU_SO=libggml-cpu.so prefix-kv-mint model.gguf --prefix-file P --out kv.bin \
 *       --calib model.calib --key <Ed25519 sk, 128 hex, or a file holding "sk <hex>">
 *
 * Prefills the prefix in the clear (it is public), saves the sequence state
 * with llama_state_seq_save_file, and signs the sidecar (prefix-kv.h). Prints
 * the public key a consumer pins (SHIELDED_PREFIX_KV_PK). No shielded backend
 * is loaded: nothing here is private. */
#include "llama.h"
#include "ggml-backend.h"
#include "prefix-kv.h"
#include "shielded-pads.h"
extern "C" {
#include "tweetnacl.h"
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_file(const char *path, std::string &out) {
    FILE *f = fopen(path, "rb"); if (!f) return false;
    char buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f); return true;
}
static bool file_digest(const char *path, uint8_t out[32]) {   /* the dealer's model label: SHA-512 of the calib, first 32 bytes */
    std::string s; if (!read_file(path, s)) return false;
    uint8_t h[64]; crypto_hash(h, (const uint8_t *)s.data(), s.size()); memcpy(out, h, 32); return true;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : nullptr, *prefix_file = nullptr, *out = nullptr, *calib = nullptr, *key = nullptr;
    int threads = 8;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--prefix-file")) prefix_file = argv[i + 1];
        else if (!strcmp(argv[i], "--out")) out = argv[i + 1];
        else if (!strcmp(argv[i], "--calib")) calib = argv[i + 1];
        else if (!strcmp(argv[i], "--key")) key = argv[i + 1];
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!model_path || !prefix_file || !out || !calib || !key) {
        fprintf(stderr, "usage: GGML_CPU_SO=.. prefix-kv-mint model.gguf --prefix-file P --out kv --calib C --key <sk hex | file> [--threads N]\n");
        return 2;
    }
    std::string prefix; if (!read_file(prefix_file, prefix)) { fprintf(stderr, "cannot read %s\n", prefix_file); return 2; }
    uint8_t digest[32]; if (!file_digest(calib, digest)) { fprintf(stderr, "cannot read calib %s\n", calib); return 2; }
    uint8_t sk[64];
    {
        std::string k = key;
        if (k.size() != 128) { std::string s; if (!read_file(key, s)) { fprintf(stderr, "cannot read key %s\n", key); return 2; }
            const size_t at = s.find("sk "); if (at == std::string::npos || s.size() < at + 3 + 128) { fprintf(stderr, "key file needs an 'sk <128 hex>' line\n"); return 2; }
            k = s.substr(at + 3, 128); }
        if (!sh_pads_hex2bin(k.c_str(), sk, 64)) { fprintf(stderr, "bad key hex\n"); return 2; }
    }
    if (const char *cpu_so = getenv("GGML_CPU_SO")) { if (!ggml_backend_load(cpu_so)) { fprintf(stderr, "cpu backend failed to load\n"); return 2; } }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prefix.size() + 16);
    int n = llama_tokenize(vocab, prefix.c_str(), (int)prefix.size(), toks.data(), (int)toks.size(), true, true);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 2; }
    toks.resize(n);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(n + 256); cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 2; }
    for (int at = 0; at < n; at += 512) {
        const int b = n - at < 512 ? n - at : 512;
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + at, b))) { fprintf(stderr, "prefill failed at %d\n", at); return 2; }
    }
    const size_t wrote = llama_state_seq_save_file(ctx, out, 0, toks.data(), (size_t)n);
    if (!wrote) { fprintf(stderr, "state save failed\n"); return 2; }
    char err[256];
    if (sh_prefix_kv_sign(out, digest, prefix.data(), prefix.size(), (uint64_t)n, sk, err, sizeof err)) { fprintf(stderr, "sign failed: %s\n", err); return 2; }
    uint8_t pk[32]; memcpy(pk, sk + 32, 32); char pkh[65]; sh_pads_bin2hex(pk, 32, pkh);
    char dh[65]; sh_pads_bin2hex(digest, 32, dh);
    printf("prefix-kv: %s: %d tokens, %zu bytes, model %.16s..., signed; pin SHIELDED_PREFIX_KV_PK=%s\n", out, n, wrote, dh, pkh);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
