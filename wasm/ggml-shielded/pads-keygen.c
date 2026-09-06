/* Print a fresh X25519 pad keypair as hex: "pk <64 hex>\nsk <64 hex>". The
 * consumer keeps sk (a pVM mints its own in the payload); the dealer gets pk. */
#include "shielded-pads.h"
#include "tweetnacl.h"
#include <stdio.h>
int main(void) {
    unsigned char pk[32], sk[32]; char h[65];
    crypto_box_keypair(pk, sk);
    sh_pads_bin2hex(pk, 32, h); printf("pk %s\n", h);
    sh_pads_bin2hex(sk, 32, h); printf("sk %s\n", h);
    return 0;
}
