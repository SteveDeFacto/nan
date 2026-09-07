#include "prefix-kv.h"
#include "shielded-pads.h"
#include "tweetnacl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int file_sha512(const char *path, uint8_t out[64]) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    /* TweetNaCl's crypto_hash is one-shot: read the whole file (a KV artifact
     * is tens to hundreds of MB; fine for a load that happens once). */
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return -1; }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)n ? (size_t)n : 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return -1; }
    fclose(f);
    crypto_hash(out, buf, (unsigned long long)n);
    free(buf);
    return 0;
}

static int body_lines(char *out, size_t cap, const uint8_t model_digest[32], const uint8_t prefix_h[64], uint64_t n_tokens, const uint8_t file_h[64]) {
    char md[65], ph[129], fh[129];
    sh_pads_bin2hex(model_digest, 32, md); sh_pads_bin2hex(prefix_h, 64, ph); sh_pads_bin2hex(file_h, 64, fh);
    const int n = snprintf(out, cap, "enclave-prefix-kv-v1\nmodel %s\nprefix-sha512 %s\ntokens %llu\nfile-sha512 %s\n", md, ph, (unsigned long long)n_tokens, fh);
    return n > 0 && (size_t)n < cap ? n : -1;
}

int sh_prefix_kv_sign(const char *kv_path, const uint8_t model_digest[32], const char *prefix, size_t prefix_len,
                      uint64_t n_tokens, const uint8_t sk[64], char *err, size_t err_cap) {
    uint8_t fh[64], ph[64];
    if (file_sha512(kv_path, fh)) { snprintf(err, err_cap, "cannot hash %s", kv_path); return -1; }
    crypto_hash(ph, (const uint8_t *)prefix, (unsigned long long)prefix_len);
    char body[512];
    const int n = body_lines(body, sizeof body, model_digest, ph, n_tokens, fh);
    if (n < 0) { snprintf(err, err_cap, "sidecar body too long"); return -1; }
    uint8_t sm[64 + 512]; unsigned long long smlen = 0;
    crypto_sign(sm, &smlen, (const uint8_t *)body, (unsigned long long)n, sk);
    char sig_hex[129]; sh_pads_bin2hex(sm, 64, sig_hex);
    char path[1024]; snprintf(path, sizeof path, "%s.sig", kv_path);
    FILE *f = fopen(path, "w");
    if (!f) { snprintf(err, err_cap, "cannot write %s", path); return -1; }
    fputs(body, f); fprintf(f, "sig %s\n", sig_hex);
    fclose(f);
    return 0;
}

int sh_prefix_kv_verify(const char *kv_path, const uint8_t pk[32], const uint8_t model_digest[32], const char *prefix, size_t prefix_len,
                        uint64_t *n_tokens_out, char *err, size_t err_cap) {
    char path[1024]; snprintf(path, sizeof path, "%s.sig", kv_path);
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(err, err_cap, "no sidecar %s", path); return -1; }
    char side[1024]; size_t got = fread(side, 1, sizeof side - 1, f); fclose(f); side[got] = 0;
    /* split: body = everything up to the "sig " line */
    char *sigline = strstr(side, "\nsig ");
    if (!sigline) { snprintf(err, err_cap, "sidecar has no signature"); return -1; }
    const size_t body_len = (size_t)(sigline + 1 - side);
    char sig_hex[129] = ""; uint8_t sig[64];
    if (sscanf(sigline + 5, "%128s", sig_hex) != 1 || strlen(sig_hex) != 128 || !sh_pads_hex2bin(sig_hex, sig, 64)) { snprintf(err, err_cap, "malformed signature"); return -1; }
    /* the signature first: nothing below is believed until it holds */
    uint8_t sm[64 + 1024], m[64 + 1024]; unsigned long long mlen = 0;
    memcpy(sm, sig, 64); memcpy(sm + 64, side, body_len);
    if (crypto_sign_open(m, &mlen, sm, 64 + (unsigned long long)body_len, pk) != 0 || mlen != body_len) { snprintf(err, err_cap, "signature REJECTED (not the platform's prefix key)"); return -1; }
    /* then the body must name this model, this prefix and this file */
    char md[65] = "", ph[129] = "", fh[129] = ""; unsigned long long ntok = 0;
    if (strncmp(side, "enclave-prefix-kv-v1\n", 21) ||
        sscanf(side, "enclave-prefix-kv-v1\nmodel %64s\nprefix-sha512 %128s\ntokens %llu\nfile-sha512 %128s\n", md, ph, &ntok, fh) != 4) {
        snprintf(err, err_cap, "malformed sidecar"); return -1;
    }
    uint8_t want_md[32], want_ph[64], want_fh[64]; char hex[129];
    sh_pads_bin2hex(model_digest, 32, hex);
    if (strcmp(hex, md)) { snprintf(err, err_cap, "signed for another model (%.*s...)", 16, md); return -1; }
    crypto_hash(want_ph, (const uint8_t *)prefix, (unsigned long long)prefix_len); sh_pads_bin2hex(want_ph, 64, hex);
    if (strcmp(hex, ph)) { snprintf(err, err_cap, "signed for another prefix text"); return -1; }
    if (file_sha512(kv_path, want_fh)) { snprintf(err, err_cap, "cannot hash %s", kv_path); return -1; }
    sh_pads_bin2hex(want_fh, 64, hex);
    if (strcmp(hex, fh)) { snprintf(err, err_cap, "the KV file does not match its signature (tampered or truncated)"); return -1; }
    (void)want_md;
    if (n_tokens_out) *n_tokens_out = ntok;
    return 0;
}
