/* shielded-dealer: mint a .pads shipment for a model (shielded/dealer/PLAN.md).
 *
 *   SHIELDED_SO=libggml-shielded.so GGML_CPU_SO=libggml-cpu.so SHIELDED_CALIB=model.calib \
 *   shielded-dealer model.gguf --out seed-0-64.pads --seed <64 hex> --seed-id <32 hex> \
 *                  --pk <consumer X25519 public key, 64 hex> [--index0 0] [--count 64]
 *
 * Loads the model exactly as the engine does (the same registration, grouping
 * and field encoding), against a worker that is never contacted, and mints
 * through the backend's ggml_backend_shielded_mint. The model digest recorded
 * in the shipment is SHA-512/256 of the calibration file. Prints the shipment
 * path and the group table so a consumer can be checked against it. */
#include "ggml-backend.h"
#include "llama.h"
#include "shielded-pads.h"
extern "C" {
#include "tweetnacl.h"
}
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef int (*mint_fn)(const char *, const char *, const char *, uint64_t, uint64_t, const char *, const char *);

static bool file_digest(const char *path, uint8_t out[32]) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    std::vector<unsigned char> buf;
    unsigned char chunk[1 << 16];
    size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
    fclose(f);
    unsigned char h[64];
    crypto_hash(h, buf.data(), buf.size());
    memcpy(out, h, 32);
    return true;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : nullptr;
    const char *out = nullptr, *seed = nullptr, *seed_id = nullptr, *pk = nullptr, *ranges = nullptr, *worker = nullptr;
    uint64_t index0 = 0, count = 64;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--out")) out = argv[i + 1];
        else if (!strcmp(argv[i], "--ranges")) ranges = argv[i + 1];   /* i0:count,i0:count,...; --out is then a template with {index0} and {count} */
        else if (!strcmp(argv[i], "--seed")) seed = argv[i + 1];
        else if (!strcmp(argv[i], "--seed-id")) seed_id = argv[i + 1];
        else if (!strcmp(argv[i], "--pk")) pk = argv[i + 1];
        else if (!strcmp(argv[i], "--index0")) index0 = strtoull(argv[i + 1], nullptr, 10);
        else if (!strcmp(argv[i], "--count")) count = strtoull(argv[i + 1], nullptr, 10);
        else if (!strcmp(argv[i], "--worker")) worker = argv[i + 1];   /* host:port of the DEALER'S OWN worker: r goes to it unmasked, u = r.W comes back at GPU speed */
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    const char *backend = getenv("SHIELDED_SO"), *calib = getenv("SHIELDED_CALIB");
    if (!model_path || !out || !seed || !seed_id || !pk || !backend || !calib) {
        fprintf(stderr, "usage: SHIELDED_SO=.. GGML_CPU_SO=.. SHIELDED_CALIB=.. shielded-dealer model.gguf --out F --seed H64 --seed-id H32 --pk H64 [--index0 N] [--count N] | [--ranges i0:n,i0:n --out template{index0}{count}]\n");
        return 2;
    }
    /* The weights register against a link that is never connected: a port
     * nothing listens on, and the whole card budget so every site registers. */
    if (worker) {
        /* Minting through a worker the dealer owns: zero pads, so the wire
         * carries r itself and the product IS u. Never point this at an
         * operator's worker - it would learn every mask it later unmasks. */
        std::string w = worker; const size_t c = w.rfind(':');
        if (c == std::string::npos) { fprintf(stderr, "--worker needs host:port\n"); return 2; }
        setenv("SHIELDED_HOST", w.substr(0, c).c_str(), 1);
        setenv("SHIELDED_PORT", w.substr(c + 1).c_str(), 1);
        setenv("SHIELDED_ZERO_PADS", "1", 1);
        setenv("SHIELDED_PAD_CHECK", "1", 1);      /* builds the mod-M check vectors the mint verifies the worker with */
    } else {
        setenv("SHIELDED_HOST", "127.0.0.1", 0);
        setenv("SHIELDED_PORT", "1", 0);
        setenv("SHIELDED_RESERVE_BYTES", "1099511627776", 0);   /* dead link: the whole "card" so every site registers */
    }
    setenv("SHIELDED_WARM_MS", "0", 0);
    if (const char *cpu_so = getenv("GGML_CPU_SO")) {
        if (!ggml_backend_load(cpu_so)) { fprintf(stderr, "cpu backend failed to load\n"); return 2; }
    }
    ggml_backend_reg_t r = ggml_backend_load(backend);
    if (!r) { fprintf(stderr, "shielded backend failed to load\n"); return 2; }
    void *h = dlopen(backend, RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen(backend, RTLD_NOW);
    mint_fn mint = h ? (mint_fn)dlsym(h, worker ? "ggml_backend_shielded_mint_worker" : "ggml_backend_shielded_mint") : nullptr;
    if (!mint) { fprintf(stderr, "ggml_backend_shielded_mint%s not exported by %s\n", worker ? "_worker" : "", backend); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 8; cp.n_threads_batch = 8;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 2; }
    /* Registration happens when the FIRST graph is planned (sh_plan places
     * every pending calibrated site and registers it with the link), not at
     * context creation; so decode one token. The connect to the dead port
     * fails and that token is computed in the clear, which is fine here. */
    {
        const llama_vocab *vocab = llama_model_get_vocab(model);
        std::vector<llama_token> toks(16);
        const char *probe = "Hello";
        int n = llama_tokenize(vocab, probe, (int)strlen(probe), toks.data(), (int)toks.size(), true, false);
        if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 2; }
        toks.resize(n);
        llama_batch batch = llama_batch_get_one(toks.data(), n);
        if (llama_decode(ctx, batch)) { fprintf(stderr, "probe decode failed\n"); return 2; }
    }

    uint8_t digest[32]; char digest_hex[65];
    if (!file_digest(calib, digest)) { fprintf(stderr, "cannot read calib %s\n", calib); return 2; }
    sh_pads_bin2hex(digest, 32, digest_hex);
    /* One model load, any number of shipments: the loop that keeps a bank ahead
     * of the ledger mints every missing range in one process. */
    std::vector<std::pair<uint64_t, uint64_t>> plan;
    if (ranges) {
        std::string r = ranges;
        for (size_t at = 0; at < r.size();) {
            size_t comma = r.find(',', at); if (comma == std::string::npos) comma = r.size();
            const std::string one = r.substr(at, comma - at); at = comma + 1;
            const size_t colon = one.find(':');
            if (colon == std::string::npos) { fprintf(stderr, "bad range %s\n", one.c_str()); return 2; }
            plan.emplace_back(strtoull(one.substr(0, colon).c_str(), nullptr, 10), strtoull(one.substr(colon + 1).c_str(), nullptr, 10));
        }
    } else plan.emplace_back(index0, count);
    for (auto &pr : plan) {
        std::string path = out;
        auto sub = [&](const char *key, uint64_t v) { for (size_t k; (k = path.find(key)) != std::string::npos;) path.replace(k, strlen(key), std::to_string(v)); };
        sub("{index0}", pr.first); sub("{count}", pr.second);
        const int rc = mint(seed, seed_id, digest_hex, pr.first, pr.second, pk, path.c_str());
        if (rc != 0) { fprintf(stderr, "mint failed: %d\n", rc); return 1; }
        printf("minted %s: indices [%llu, %llu), model digest %s\n", path.c_str(), (unsigned long long)pr.first,
               (unsigned long long)(pr.first + pr.second), digest_hex);
        fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
