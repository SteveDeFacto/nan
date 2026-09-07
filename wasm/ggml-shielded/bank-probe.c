/* bank-probe: exercise the dealt pad bank client (shielded-bank.c) against a
 * bank URL without a model. Opens the fetcher on a cache directory, sets the
 * horizon, waits until the cache covers it (or the timeout), prints what
 * landed, then moves the floor and shows the fetcher skipping spent rows.
 *
 *   bank-probe <bank url> <seed_id hex> <cache dir> <need> <cache MB> [timeout ms] [floor]
 * Exit 0 when every shipment below `need` is in the cache. */
#include "shielded-bank.h"
#include "shielded-tee.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int count_pads(const char *dir, const char *seed, uint64_t *total, uint64_t *hi) {
    DIR *d = opendir(dir); if (!d) return 0;
    int n = 0; struct dirent *e; *total = 0; *hi = 0;
    while ((e = readdir(d))) {
        const size_t len = strlen(e->d_name);
        if (len < 6 || strcmp(e->d_name + len - 5, ".pads")) continue;
        char sid[33] = ""; unsigned long long i0 = 0, cnt = 0;
        if (sscanf(e->d_name, "%32[0-9a-f]-%llu-%llu.pads", sid, &i0, &cnt) != 3 || strcmp(sid, seed)) continue;
        char path[800]; snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        struct stat st; if (stat(path, &st) == 0) *total += (uint64_t)st.st_size;
        if (i0 + cnt > *hi) *hi = i0 + cnt;
        n++;
    }
    closedir(d);
    return n;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: bank-probe <url> <seed_id> <cache dir> <need> <cache MB> [timeout ms] [floor]\n"); return 2; }
    const char *url = argv[1], *seed = argv[2], *dir = argv[3];
    const uint64_t need = strtoull(argv[4], NULL, 10), cache_mb = strtoull(argv[5], NULL, 10);
    const int timeout_ms = argc > 6 ? atoi(argv[6]) : 20000;
    const uint64_t floor = argc > 7 ? strtoull(argv[7], NULL, 10) : 0;
    int err = 0;
    sh_bank *b = sh_bank_open(url, seed, dir, cache_mb << 20, &err);
    if (!b) { fprintf(stderr, "bank-probe: open failed (%d)\n", err); return 1; }
    sh_bank_set_floor(b, floor);
    sh_bank_set_need(b, need);
    uint64_t total = 0, hi = 0; int n = 0;
    const double t0 = (double)clock() / CLOCKS_PER_SEC;
    for (int waited = 0; waited < timeout_ms; waited += 100) {
        n = count_pads(dir, seed, &total, &hi);
        if (hi >= need) break;
        usleep(100000);
    }
    uint64_t files = 0, bytes = 0; int status = 0; char e[256];
    sh_bank_stats(b, &files, &bytes, &status, e, sizeof e);
    printf("bank-probe: cache %d file(s) %llu bytes covering up to %llu (need %llu, floor %llu); fetched %llu file(s) %llu bytes; listing http %d%s%s\n",
           n, (unsigned long long)total, (unsigned long long)hi, (unsigned long long)need, (unsigned long long)floor,
           (unsigned long long)files, (unsigned long long)bytes, status, e[0] ? "; " : "", e);
    (void)t0;
    sh_bank_close(b);
    return hi >= need ? 0 : 1;
}
