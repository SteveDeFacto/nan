/* Dealt pads, end to end on synthetic weights and no worker:
 *   mint a shipment -> read every cell back -> compare with an exact int64
 *   oracle of r.W where r comes from the shared derivation; a flipped byte is
 *   refused; an index past the shipment is EXHAUST; the ledger window
 *   advances before use and survives a reopen; the link's own dealt import
 *   (env-configured, the refill threads' path) yields the same pads.
 *
 *   make dealt-selftest && ./dealt-selftest
 */
#include "shielded-tee.h"
#include "shielded-pads.h"
#include "shielded-field.h"
#include "tweetnacl.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint32_t st = 99;
/* A window provider standing in for the pVM's owner-app path: hands out
 * consecutive windows of `want` and counts the calls. */
int provider(void *ctx, uint64_t want, uint64_t *lo, uint64_t *hi) {
    struct { int *calls; uint64_t *next; } *c = ctx;
    (*c->calls)++;
    *lo = *c->next; *hi = *c->next + want; *c->next = *hi;
    return 0;
}
static uint32_t nxt(void) { st = st * 1664525U + 1013904223U; return st; }

static sh_link *build(int8_t *wa, int8_t *wb, int8_t *wc, int64_t KA, int64_t NA, int64_t NB, int64_t KC, int64_t NC) {
    int err = 0;
    sh_link *l = sh_link_open("127.0.0.1", 1, false, &err);
    assert(l);
    const int a = sh_link_add_weight(l, "blk.0.attn_q.weight", wa, KA, NA, 8, -1); assert(a >= 0);
    const int b = sh_link_add_weight(l, "blk.0.attn_k.weight", wb, KA, NB, 8, a);  assert(b >= 0);
    const int c = sh_link_add_weight(l, "blk.0.ffn_down.weight", wc, KC, NC, 8, -1); assert(c >= 0);
    return l;
}

int main(void) {
    const int64_t KA = 256, NA = 48, NB = 16, KC = 192, NC = 40;
    int8_t *wa = malloc(KA * NA), *wb = malloc(KA * NB), *wc = malloc(KC * NC);
    for (int64_t i = 0; i < KA * NA; i++) wa[i] = (int8_t)((int)(nxt() % 239) - 119);
    for (int64_t i = 0; i < KA * NB; i++) wb[i] = (int8_t)((int)(nxt() % 239) - 119);
    for (int64_t i = 0; i < KC * NC; i++) wc[i] = (int8_t)((int)(nxt() % 239) - 119);

    char dir[] = "/tmp/dealt-selftest-XXXXXX";
    assert(mkdtemp(dir));
    char ship[600], ledger[600];
    snprintf(ship, sizeof ship, "%s/seed-0-20.pads", dir);
    snprintf(ledger, sizeof ledger, "%s/ledger", dir);

    uint8_t seed[32], seed_id[16], digest[32], pk[32], sk[32];
    for (int i = 0; i < 32; i++) { seed[i] = (uint8_t)nxt(); digest[i] = (uint8_t)(i * 7); }
    for (int i = 0; i < 16; i++) seed_id[i] = (uint8_t)(200 + i);
    crypto_box_keypair(pk, sk);

    /* --- mint --- */
    sh_link *dealer = build(wa, wb, wc, KA, NA, NB, KC, NC);
    const uint64_t COUNT = 20;
    int rc = sh_link_mint_shipment(dealer, seed, seed_id, digest, 0, COUNT, pk, ship);
    assert(rc == SH_OK);
    sh_pads_group table[4];
    const int ng = sh_link_group_table(dealer, table, 4);
    assert(ng == 2 && table[0].u_len == (uint64_t)(NA + NB) && table[1].u_len == (uint64_t)NC);

    /* --- read back, oracle --- */
    int err = 0;
    sh_pads_reader *rd = sh_pads_reader_open(dir, seed_id, sk, &err);
    assert(rd && sh_pads_reader_bind(rd, table, (uint32_t)ng) == SH_OK);
    assert(sh_pads_reader_extent(rd) == COUNT);
    int32_t *r = malloc(KA * sizeof *r), *u = malloc((NA + NB) * sizeof *u);
    for (uint64_t idx = 0; idx < COUNT; idx++) {
        for (int g = 0; g < ng; g++) {
            const int64_t K = g == 0 ? KA : KC;
            sh_pad_r(seed, (uint32_t)g, idx, K, r);
            for (int64_t k = 0; k < K; k++) assert(r[k] >= 0 && r[k] < SH_M_MOD);
            assert(sh_pads_reader_cell(rd, (uint32_t)g, idx, u) == SH_OK);
            /* group 0 = attn_q rows then attn_k rows; group 1 = ffn_down */
            const int64_t ulen = (int64_t)table[g].u_len;
            for (int64_t j = 0; j < ulen; j++) {
                const int8_t *w; int64_t row;
                if (g == 0) { if (j < NA) { w = wa; row = j; } else { w = wb; row = j - NA; } }
                else { w = wc; row = j; }
                int64_t exact = 0;
                for (int64_t k = 0; k < K; k++) exact += (int64_t)r[k] * w[row * K + k];
                if (u[j] != sh_balanced(exact)) { fprintf(stderr, "MISMATCH g%d idx%llu j%lld: %d vs %lld\n", g, (unsigned long long)idx, (long long)j, u[j], (long long)sh_balanced(exact)); return 1; }
            }
        }
    }
    /* determinism of the derivation across calls */
    { int32_t *r2 = malloc(KA * sizeof *r2); sh_pad_r(seed, 0, 7, KA, r); sh_pad_r(seed, 0, 7, KA, r2); assert(!memcmp(r, r2, KA * sizeof *r)); sh_pad_r(seed, 1, 7, KA, r2); assert(memcmp(r, r2, KA * sizeof *r)); free(r2); }
    assert(sh_pads_reader_cell(rd, 0, COUNT, u) == SH_ERR_EXHAUST);

    /* --- tamper: one byte inside a cell, then restored --- */
    {
        int fd = open(ship, O_RDWR); assert(fd >= 0);
        sh_pads_hdr h; assert(pread(fd, &h, sizeof h, 0) == (ssize_t)sizeof h);
        const off_t off = (off_t)(h.data_off + 3 * h.row_bytes + 16 + 5);   /* index 3, group 0, a data byte */
        uint8_t b; assert(pread(fd, &b, 1, off) == 1);
        uint8_t t = b ^ 0x5A; assert(pwrite(fd, &t, 1, off) == 1);
        assert(sh_pads_reader_cell(rd, 0, 3, u) == SH_ERR_VERIFY);
        assert(sh_pads_reader_cell(rd, 1, 3, u) == SH_OK);                 /* other cells unaffected */
        assert(pwrite(fd, &b, 1, off) == 1); close(fd);
        assert(sh_pads_reader_cell(rd, 0, 3, u) == SH_OK);
    }
    /* --- wrong key: nothing opens --- */
    {
        uint8_t pk2[32], sk2[32]; crypto_box_keypair(pk2, sk2);
        int e2 = 0; sh_pads_reader *bad = sh_pads_reader_open(dir, seed_id, sk2, &e2);
        assert(bad && sh_pads_reader_extent(bad) == 0);                     /* the shipment did not verify: not listed */
        sh_pads_reader_close(bad);
    }
    sh_pads_reader_close(rd);

    /* --- ledger window: reserve-before-use, persistent --- */
    { uint64_t lo, hi;
      assert(sh_pads_window_reserve(ledger, 8, &lo, &hi) == SH_OK && lo == 0 && hi == 8);
      assert(sh_pads_window_reserve(ledger, 8, &lo, &hi) == SH_OK && lo == 8 && hi == 16);
      FILE *f = fopen(ledger, "r"); unsigned long long mark = 0; assert(f && fscanf(f, "%llu", &mark) == 1 && mark == 16); fclose(f); }
    /* a fresh ledger for the link below */
    snprintf(ledger, sizeof ledger, "%s/ledger2", dir);

    /* --- the link's own dealt path (env) --- */
    {
        char hs[65], hid[33], hsk[65];
        sh_pads_bin2hex(seed, 32, hs); sh_pads_bin2hex(seed_id, 16, hid); sh_pads_bin2hex(sk, 32, hsk);
        setenv("SHIELDED_PAD_SOURCE", dir, 1); setenv("SHIELDED_PAD_SEED", hs, 1); setenv("SHIELDED_PAD_SEED_ID", hid, 1);
        setenv("SHIELDED_PAD_SK", hsk, 1); setenv("SHIELDED_PAD_LEDGER", ledger, 1); setenv("SHIELDED_PAD_WINDOW", "8", 1);
        setenv("SHIELDED_PAD_WAIT_MS", "100", 1);
        sh_link *c = build(wa, wb, wc, KA, NA, NB, KC, NC);
        int32_t *R = calloc((size_t)2 * 3 * KA, sizeof *R), *U = calloc((size_t)2 * 3 * (NA + NB), sizeof *U);
        const int got = sh_link_dealt_selftest(c, 3, R, U);
        assert(got == 6);
        /* group 0 rows 0..2 then group 1 rows 0..2: each group starts at done*Kmax / done*ulen_max and packs its rows at its own K / u_len */
        const int64_t Kmax = KA, umax = NA + NB;
        for (int g = 0; g < 2; g++) for (int i = 0; i < 3; i++) {
            const int64_t K = g == 0 ? KA : KC, ulen = g == 0 ? NA + NB : NC;
            sh_pad_r(seed, (uint32_t)g, (uint64_t)i, K, r);
            if (memcmp(R + (size_t)(g * 3) * Kmax + (size_t)i * K, r, (size_t)K * sizeof *r)) {
                const int32_t *got = R + (size_t)(g * 3) * Kmax + (size_t)i * K;
                fprintf(stderr, "r mismatch g%d i%d: got %d %d %d %d want %d %d %d %d\n", g, i, got[0], got[1], got[2], got[3], r[0], r[1], r[2], r[3]);
                /* is it another index of the same group? */
                int32_t *probe = malloc((size_t)K * sizeof *probe);
                for (uint64_t x = 0; x < 20; x++) { sh_pad_r(seed, (uint32_t)g, x, K, probe); if (!memcmp(probe, got, (size_t)K * sizeof *probe)) fprintf(stderr, "  = index %llu\n", (unsigned long long)x); }
                for (int gg = 0; gg < 2; gg++) { sh_pad_r(seed, (uint32_t)gg, (uint64_t)i, K, probe); if (!memcmp(probe, got, (size_t)K * sizeof *probe)) fprintf(stderr, "  = group %d\n", gg); }
                free(probe); return 1;
            }
            for (int64_t j = 0; j < ulen; j++) {
                const int8_t *w; int64_t row;
                if (g == 0) { if (j < NA) { w = wa; row = j; } else { w = wb; row = j - NA; } } else { w = wc; row = j; }
                int64_t exact = 0; for (int64_t k = 0; k < K; k++) exact += (int64_t)r[k] * w[row * K + k];
                assert(U[(size_t)(g * 3) * umax + (size_t)i * ulen + j] == sh_balanced(exact));
            }
        }
        /* the window was reserved before use: the mark is already 8 */
        FILE *f = fopen(ledger, "r"); unsigned long long mark = 0; assert(f && fscanf(f, "%llu", &mark) == 1 && mark == 8); fclose(f);
        /* past the shipment: exhaust, never mint */
        assert(sh_link_dealt_selftest(c, 3, R, U) == 6);                     /* rows 3..5 */
        setenv("SHIELDED_PAD_WINDOW", "64", 1);
        int32_t *R2 = calloc((size_t)2 * 30 * KA, sizeof *R2), *U2 = calloc((size_t)2 * 30 * (NA + NB), sizeof *U2);
        assert(sh_link_dealt_selftest(c, 30, R2, U2) == SH_ERR_EXHAUST);    /* window of 8 cannot hold 30 */
        free(R); free(U); free(R2); free(U2);
        sh_link_close(c);
        /* a window provider replaces the ledger file: the pVM's path */
        unsetenv("SHIELDED_PAD_LEDGER");
        setenv("SHIELDED_PAD_WINDOW", "8", 1);
        {
            static uint64_t next_lo = 0;
            sh_link *pv = build(wa, wb, wc, KA, NA, NB, KC, NC);
            int calls = 0;
            struct { int *calls; uint64_t *next; } pctx = { &calls, &next_lo };
            sh_link_set_window_provider(pv, provider, &pctx);
            int32_t *R3 = calloc((size_t)2 * 3 * KA, sizeof *R3), *U3 = calloc((size_t)2 * 3 * (NA + NB), sizeof *U3);
            assert(sh_link_dealt_selftest(pv, 3, R3, U3) == 6 && calls == 1 && next_lo == 8);
            free(R3); free(U3); sh_link_close(pv);
            /* and with neither a ledger nor a provider, the link refuses to start */
            sh_link *none = build(wa, wb, wc, KA, NA, NB, KC, NC);
            assert(sh_link_dealt_selftest(none, 3, R, U) == SH_ERR_RANGE);
            sh_link_close(none);
        }
        setenv("SHIELDED_PAD_LEDGER", ledger, 1);
        /* the pad check (SHIELDED_PAD_CHECK): the good shipment passes; a shipment
         * minted for ANOTHER seed opens (same consumer key) but fails u = r.W */
        {
            uint8_t seed2[32]; for (int i = 0; i < 32; i++) seed2[i] = (uint8_t)(seed[i] ^ 0xA5);
            char dir2[] = "/tmp/dealt-selftest2-XXXXXX"; assert(mkdtemp(dir2));
            char ship2[600], led2[600]; snprintf(ship2, sizeof ship2, "%s/seed-0-20.pads", dir2); snprintf(led2, sizeof led2, "%s/ledger", dir2);
            assert(sh_link_mint_shipment(dealer, seed2, seed_id, digest, 0, COUNT, pk, ship2) == SH_OK);   /* same seed_id on the label, wrong r inside */
            setenv("SHIELDED_PAD_CHECK", "1", 1);
            char led3[600]; snprintf(led3, sizeof led3, "%s/ledger3", dir);
            setenv("SHIELDED_PAD_LEDGER", led3, 1);                   /* the good bank, a fresh ledger: rows 0..2 */
            sh_link *ok = build(wa, wb, wc, KA, NA, NB, KC, NC);
            int32_t *R4 = calloc((size_t)2 * 3 * KA, sizeof *R4), *U4 = calloc((size_t)2 * 3 * (NA + NB), sizeof *U4);
            { const int got = sh_link_dealt_selftest(ok, 3, R4, U4); if (got != 6) { fprintf(stderr, "pad check on the good shipment: rc %d err '%s'\n", got, sh_link_last_error(ok)); return 1; } }
            sh_link_close(ok);
            setenv("SHIELDED_PAD_SOURCE", dir2, 1); setenv("SHIELDED_PAD_LEDGER", led2, 1);
            sh_link *bad = build(wa, wb, wc, KA, NA, NB, KC, NC);
            assert(sh_link_dealt_selftest(bad, 3, R4, U4) == SH_ERR_VERIFY);   /* refused at import: u != r.W for our seed */
            assert(strstr(sh_link_last_error(bad), "FAILED the check"));
            sh_link_close(bad);
            free(R4); free(U4);
            unsetenv("SHIELDED_PAD_CHECK"); setenv("SHIELDED_PAD_SOURCE", dir, 1); setenv("SHIELDED_PAD_LEDGER", ledger, 1);
            char rm2[700]; snprintf(rm2, sizeof rm2, "rm -rf %s", dir2); (void)!system(rm2);
        }
        /* a partial env refuses to open */
        unsetenv("SHIELDED_PAD_SK");
        int e3 = 0; assert(sh_link_open("127.0.0.1", 1, false, &e3) == NULL && e3 == SH_ERR_RANGE);
        unsetenv("SHIELDED_PAD_SOURCE");
    }
    /* --- the same shipment minted THROUGH A WORKER (SHIELDED_WORKER=host:port
     * offered by the harness): zero pads carry r itself, the mint checks the
     * worker's product mod M, and every cell must equal the in-process mint. */
    const char *wk = getenv("SHIELDED_WORKER");
    if (wk && *wk) {
        const char *colon = strrchr(wk, ':'); assert(colon);
        char host[128]; snprintf(host, sizeof host, "%.*s", (int)(colon - wk), wk);
        setenv("SHIELDED_ZERO_PADS", "1", 1); setenv("SHIELDED_PAD_CHECK", "1", 1);
        int e4 = 0; sh_link *wl = sh_link_open(host, atoi(colon + 1), true, &e4); assert(wl);
        const int a2 = sh_link_add_weight(wl, "blk.0.attn_q.weight", wa, KA, NA, 8, -1); assert(a2 >= 0);
        const int b2 = sh_link_add_weight(wl, "blk.0.attn_k.weight", wb, KA, NB, 8, a2); assert(b2 >= 0);
        const int c2 = sh_link_add_weight(wl, "blk.0.ffn_down.weight", wc, KC, NC, 8, -1); assert(c2 >= 0);
        const int src = sh_link_start(wl);
        if (src != SH_OK) { fprintf(stderr, "worker link did not start: %s\n", sh_link_last_error(wl)); assert(src == SH_OK); }
        char dir3[] = "/tmp/dealt-selftest-worker-XXXXXX"; assert(mkdtemp(dir3));
        char ship3[700]; snprintf(ship3, sizeof ship3, "%s/seed-0-20.pads", dir3);
        const int mrc = sh_link_mint_shipment_worker(wl, seed, seed_id, digest, 0, COUNT, pk, ship3);
        if (mrc != SH_OK) { fprintf(stderr, "worker mint failed: %s\n", sh_link_last_error(wl)); assert(mrc == SH_OK); }
        int e5 = 0; sh_pads_reader *rd3 = sh_pads_reader_open(dir3, seed_id, sk, &e5);
        assert(rd3 && sh_pads_reader_bind(rd3, table, (uint32_t)ng) == SH_OK && sh_pads_reader_extent(rd3) == COUNT);
        int e6 = 0; sh_pads_reader *rd0 = sh_pads_reader_open(dir, seed_id, sk, &e6);      /* the in-process shipment, reopened */
        assert(rd0 && sh_pads_reader_bind(rd0, table, (uint32_t)ng) == SH_OK);
        int32_t *u3 = malloc((NA + NB) * sizeof *u3);
        size_t cells = 0;
        for (uint64_t idx = 0; idx < COUNT; idx++)
            for (int g = 0; g < ng; g++) {
                assert(sh_pads_reader_cell(rd0, (uint32_t)g, idx, u) == SH_OK);
                assert(sh_pads_reader_cell(rd3, (uint32_t)g, idx, u3) == SH_OK);
                if (memcmp(u, u3, table[g].u_len * sizeof *u)) { fprintf(stderr, "worker mint differs at g%d idx%llu\n", g, (unsigned long long)idx); assert(0); }
                cells++;
            }
        free(u3); sh_pads_reader_close(rd3); sh_pads_reader_close(rd0); sh_link_close(wl);
        unsetenv("SHIELDED_ZERO_PADS"); unsetenv("SHIELDED_PAD_CHECK");
        char rm3[700]; snprintf(rm3, sizeof rm3, "rm -rf %s", dir3); (void)!system(rm3);
        printf("dealt-selftest: worker mint via %s identical on %zu cells\n", wk, cells);
    }
    sh_link_close(dealer);
    free(r); free(u); free(wa); free(wb); free(wc);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); (void)!system(cmd);
    printf("dealt-selftest: ok\n");
    return 0;
}
