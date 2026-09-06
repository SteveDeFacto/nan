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
