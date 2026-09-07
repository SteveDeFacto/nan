/* Test helper for the Node<->C seed/window interop (test/pads-interop.test.mjs):
 *   pads-unbox seed <pad_sk hex> <pad_pk hex> <epk hex> <nonce hex> <box hex>   -> "seed <hex>" or "fail"
 *   pads-unbox window <ledger_pk hex> <seed_id> <lo> <hi> <iat> <sig hex>       -> "ok" or "fail"
 *   pads-unbox keypair                                                          -> "pk <hex>\nsk <hex>" (Ed25519 transport key)
 *   pads-unbox sign <sk hex 128> <kind> <nonce hex> [fields...]                  -> "sig <hex>" */
#include "shielded-pads.h"
#include "tweetnacl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc >= 7 && !strcmp(argv[1], "seed")) {
        uint8_t sk[32], pk[32], epk[32], nonce[12], box[48], seed[32]; char h[65];
        if (!sh_pads_hex2bin(argv[2], sk, 32) || !sh_pads_hex2bin(argv[3], pk, 32) || !sh_pads_hex2bin(argv[4], epk, 32) ||
            !sh_pads_hex2bin(argv[5], nonce, 12) || !sh_pads_hex2bin(argv[6], box, 48)) { puts("fail"); return 1; }
        if (sh_pads_seed_open(epk, nonce, box, 48, sk, pk, seed) != 0) { puts("fail"); return 1; }
        sh_pads_bin2hex(seed, 32, h); printf("seed %s\n", h); return 0;
    }
    if (argc >= 8 && !strcmp(argv[1], "window")) {
        uint8_t pk[32], sig[64];
        if (!sh_pads_hex2bin(argv[2], pk, 32) || !sh_pads_hex2bin(argv[7], sig, 64)) { puts("fail"); return 1; }
        const bool ok = sh_pads_window_verify(pk, argv[3], strtoull(argv[4], NULL, 10), strtoull(argv[5], NULL, 10), strtoull(argv[6], NULL, 10), sig);
        puts(ok ? "ok" : "fail"); return ok ? 0 : 1;
    }
    if (argc >= 7 && !strcmp(argv[1], "dump")) {
        /* dump <sk hex> <dir> <seed_id hex> <index0> <count>: one line per (index, group)
         * with a FNV-1a of the opened u, so two banks minted for one seed can be diffed. */
        uint8_t sk[32], sid[16];
        if (!sh_pads_hex2bin(argv[2], sk, 32) || !sh_pads_hex2bin(argv[4], sid, 16)) { puts("fail"); return 1; }
        int err = 0;
        sh_pads_reader *r = sh_pads_reader_open(argv[3], sid, sk, &err);
        if (!r) { printf("fail open %d\n", err); return 1; }
        sh_pads_group groups[4096];
        const uint32_t ng = sh_pads_reader_groups(r, groups, 4096);
        if (!ng) { puts("fail: no shipment of that seed opens with this key in that directory"); return 1; }
        if (sh_pads_reader_bind(r, groups, ng) != 0) { puts("fail bind"); return 1; }
        uint64_t umax = 0; for (uint32_t g = 0; g < ng; g++) if (groups[g].u_len > umax) umax = groups[g].u_len;
        int32_t *u = (int32_t *)malloc(umax * sizeof *u);
        const unsigned long long i0 = strtoull(argv[5], NULL, 10), cnt = strtoull(argv[6], NULL, 10);
        for (unsigned long long i = i0; i < i0 + cnt; i++)
            for (uint32_t g = 0; g < ng; g++) {
                const int rc = sh_pads_reader_cell(r, g, i, u);
                if (rc != 0) { printf("%llu %u ERR %d\n", i, g, rc); continue; }
                uint64_t hsh = 1469598103934665603ull;
                for (uint64_t j = 0; j < groups[g].u_len; j++) { hsh ^= (uint32_t)u[j]; hsh *= 1099511628211ull; }
                printf("%llu %u %s %016llx\n", i, g, groups[g].name, (unsigned long long)hsh);
            }
        free(u); sh_pads_reader_close(r); return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "keypair")) {
        uint8_t pk[32], sk[64]; char h[129];
        crypto_sign_keypair(pk, sk);
        sh_pads_bin2hex(pk, 32, h); printf("pk %s\n", h);
        sh_pads_bin2hex(sk, 64, h); printf("sk %s\n", h); return 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "sign")) {
        uint8_t sk[64], sig[64]; char h[129];
        if (!sh_pads_hex2bin(argv[2], sk, 64)) { puts("fail"); return 1; }
        sh_pads_request_sign(sk, argv[3], (const char *const *)(argv + 5), (size_t)(argc - 5), argv[4], sig);
        sh_pads_bin2hex(sig, 64, h); printf("sig %s\n", h); return 0;
    }
    fprintf(stderr, "usage: pads-unbox seed|window|keypair|sign ...\n"); return 2;
}
