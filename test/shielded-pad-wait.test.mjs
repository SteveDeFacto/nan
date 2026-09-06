import test from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

test("waiting for in-flight pads preserves consumption and times out without taking unpublished slots", () => {
  const dir = mkdtempSync(join(tmpdir(), "shielded-pad-wait-"));
  const source = fileURLToPath(new URL("../wasm/ggml-shielded/shielded-tee.c", import.meta.url));
  try {
    writeFileSync(join(dir, "test.c"), `
#include ${JSON.stringify(source)}
#include <assert.h>

typedef struct { sh_link *l; sh_group *g; int n; bool stop; } publication;
static void *publish(void *arg) {
    publication *p = arg;
    usleep(10000);
    pthread_mutex_lock(&p->l->pool_mu);
    p->g->count += p->n;
    p->g->generating -= p->n;
    if (p->stop) p->l->stop = true;
    pthread_cond_broadcast(&p->l->pool_filled);
    pthread_mutex_unlock(&p->l->pool_mu);
    return NULL;
}
int main(void) {
    sh_link l = {0}; sh_group g = {0}; int slots[4];
    pthread_mutex_init(&l.pool_mu, NULL);
    pthread_cond_init(&l.need_refill, NULL);
    pthread_condattr_t a; pthread_condattr_init(&a);
    assert(pthread_condattr_setclock(&a, CLOCK_MONOTONIC) == 0);
    pthread_cond_init(&l.pool_filled, &a); pthread_condattr_destroy(&a);
    g.depth = 8; g.head = 6; g.generating = 4;

    // Disabled by default: reservations are never mistaken for ready pads.
    assert(take_pads(&l, &g, 4, slots) == 0);
    assert(g.head == 6 && g.generating == 4 && g.held == 0);
    assert(l.pad_wait_ms == 0 && l.pads_waited == 0);

    // Wait for a publication across the ring boundary. Each slot is consumed
    // once and stays held until the request explicitly releases it.
    l.pad_wait_us = 1000000; // generous scheduling margin for this test only
    publication p = {&l, &g, 4, false}; pthread_t producer;
    assert(pthread_create(&producer, NULL, publish, &p) == 0);
    assert(take_pads(&l, &g, 4, slots) == 4);
    pthread_join(producer, NULL);
    assert(slots[0] == 6 && slots[1] == 7 && slots[2] == 0 && slots[3] == 1);
    assert(g.head == 2 && g.count == 0 && g.generating == 0 && g.held == 4);
    assert(l.pads_waited == 4 && l.pad_wait_ms > 0);
    assert(group_deficit(&g) == 4);
    assert(take_pads(&l, &g, 4, slots) == 0); // cannot take held pads again
    release_pads(&l, &g, 4);
    assert(g.held == 0 && group_deficit(&g) == 8);

    // There are not enough reserved pads to satisfy this batch: do not wait
    // for work that no producer has begun.
    double waited = l.pad_wait_ms;
    g.count = 1; g.generating = 1;
    assert(take_pads(&l, &g, 4, slots) == 1 && slots[0] == 2);
    assert(l.pad_wait_ms == waited && g.generating == 1);
    release_pads(&l, &g, 1);

    // A producer that never finishes cannot block the request indefinitely.
    // Only the already-ready prefix may leave the ring on timeout.
    g.count = 1; g.generating = 3; l.pad_wait_us = 2000;
    double start = now_ms();
    assert(take_pads(&l, &g, 4, slots) == 1 && slots[0] == 3);
    double elapsed = now_ms() - start;
    assert(elapsed >= 1 && elapsed < 500);
    assert(g.head == 4 && g.generating == 3 && g.held == 1);
    release_pads(&l, &g, 1);

    // Publishing those same reservations LATER does not recycle the
    // consumed prefix or lose the reserved slots after a timeout.
    g.count = 3; g.generating = 0;
    assert(take_pads(&l, &g, 3, slots) == 3);
    assert(slots[0] == 4 && slots[1] == 5 && slots[2] == 6);
    release_pads(&l, &g, 3);
    assert(g.head == 7 && g.held == 0 && g.count == 0);

    // Shutdown wakes a waiting request without treating unfinished pads as
    // usable or reusing any prior request's pad.
    g.generating = 4; l.pad_wait_us = 1000000;
    p = (publication){&l, &g, 0, true};
    assert(pthread_create(&producer, NULL, publish, &p) == 0);
    assert(take_pads(&l, &g, 4, slots) == 0);
    pthread_join(producer, NULL);
    assert(g.head == 7 && g.generating == 4 && g.held == 0);
    pthread_cond_destroy(&l.pool_filled); pthread_cond_destroy(&l.need_refill);
    pthread_mutex_destroy(&l.pool_mu);
}
`);
    // The test directly exercises the production ring operations. Discard
    // unrelated transport/SIMD sections so no worker or model is required.
    execFileSync("cc", ["-std=c11", "-O1", "-Wall", "-Wextra", "-ffunction-sections", "-fdata-sections",
      join(dir, "test.c"), "-Wl,--gc-sections", "-lpthread", "-o", join(dir, "test")], { timeout: 30_000 });
    execFileSync(join(dir, "test"), { timeout: 5_000 });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
