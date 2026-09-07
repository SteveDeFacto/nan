/* prefix-kv-selftest: the signed sidecar of a shared-prefix KV artifact
 * (prefix-kv.h) opens only for the platform's key, this model, this exact
 * prefix text and this exact file; anything else is refused before a byte
 * of state would be loaded. Runs from `make all`; shielded-cbackend.test.mjs
 * asserts the line it prints. */
#include "prefix-kv.h"
#include "shielded-pads.h"
#include "tweetnacl.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char dir[] = "/tmp/prefix-kv-selftest-XXXXXX";
    assert(mkdtemp(dir));
    char kv[600]; snprintf(kv, sizeof kv, "%s/prefix.kv", dir);
    /* a stand-in artifact: the sidecar binds the file's bytes, not its meaning */
    { FILE *f = fopen(kv, "wb"); assert(f); for (int i = 0; i < 100000; i++) fputc((i * 7) & 0xff, f); fclose(f); }
    uint8_t pk[32], sk[64], pk2[32], sk2[64];
    crypto_sign_keypair(pk, sk); crypto_sign_keypair(pk2, sk2);
    uint8_t model[32]; for (int i = 0; i < 32; i++) model[i] = (uint8_t)(i * 3);
    const char *prefix = "You are a terse assistant.\nUser:";
    char err[256]; uint64_t n = 0;

    assert(sh_prefix_kv_sign(kv, model, prefix, strlen(prefix), 25, sk, err, sizeof err) == 0);
    assert(sh_prefix_kv_verify(kv, pk, model, prefix, strlen(prefix), &n, err, sizeof err) == 0 && n == 25);

    /* wrong key */
    assert(sh_prefix_kv_verify(kv, pk2, model, prefix, strlen(prefix), &n, err, sizeof err) != 0 && strstr(err, "REJECTED"));
    /* another model */
    uint8_t model2[32]; memcpy(model2, model, 32); model2[0] ^= 1;
    assert(sh_prefix_kv_verify(kv, pk, model2, prefix, strlen(prefix), &n, err, sizeof err) != 0 && strstr(err, "another model"));
    /* another prefix text (one byte) */
    assert(sh_prefix_kv_verify(kv, pk, model, "You are a terse assistant.\nUser:!", strlen(prefix) + 1, &n, err, sizeof err) != 0 && strstr(err, "another prefix"));
    /* the file changes under its signature */
    { FILE *f = fopen(kv, "r+b"); assert(f); fseek(f, 50000, SEEK_SET); int c = fgetc(f); fseek(f, 50000, SEEK_SET); fputc(c ^ 1, f); fclose(f); }
    assert(sh_prefix_kv_verify(kv, pk, model, prefix, strlen(prefix), &n, err, sizeof err) != 0 && strstr(err, "does not match"));
    { FILE *f = fopen(kv, "r+b"); assert(f); fseek(f, 50000, SEEK_SET); int c = fgetc(f); fseek(f, 50000, SEEK_SET); fputc(c ^ 1, f); fclose(f); }
    assert(sh_prefix_kv_verify(kv, pk, model, prefix, strlen(prefix), &n, err, sizeof err) == 0);
    /* the sidecar's token count changes under its signature */
    { char side[700]; snprintf(side, sizeof side, "%s.sig", kv); FILE *f = fopen(side, "r+b"); assert(f);
      char buf[1024]; size_t got = fread(buf, 1, sizeof buf - 1, f); buf[got] = 0; char *t = strstr(buf, "tokens 25"); assert(t); t[7] = '2'; t[8] = '6';
      rewind(f); fwrite(buf, 1, got, f); fclose(f); }
    assert(sh_prefix_kv_verify(kv, pk, model, prefix, strlen(prefix), &n, err, sizeof err) != 0 && strstr(err, "REJECTED"));
    /* no sidecar at all */
    { char side[700]; snprintf(side, sizeof side, "%s.sig", kv); unlink(side); }
    assert(sh_prefix_kv_verify(kv, pk, model, prefix, strlen(prefix), &n, err, sizeof err) != 0 && strstr(err, "no sidecar"));

    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); (void)!system(cmd);
    printf("prefix-kv-selftest: ok\n");
    return 0;
}
