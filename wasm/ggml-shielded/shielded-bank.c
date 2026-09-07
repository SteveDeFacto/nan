#define _GNU_SOURCE
#include "shielded-bank.h"
#include "shielded-http.h"
#include "shielded-tee.h"
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define BANK_MAX_ENTRIES 4096
#define BANK_TIMEOUT_MS  20000      /* connect / each read; a 27B shipment streams at LAN speed well inside this per read */
#define BANK_POLL_MS     1000

typedef struct { char name[160]; uint64_t bytes, index0, count; } bank_entry;

struct sh_bank {
    char url[1024], seed_id[33], dir[512];
    uint64_t cache_max, floor, need;
    pthread_t th; pthread_mutex_t mu; pthread_cond_t cv;
    bool stop, running;
    uint64_t fetched_files, fetched_bytes; int last_status;
    char err[256];
};

/* The listing is the platform's own shape; the fields we need are flat
 * inside each object, so a scan for the keys between braces is enough. */
static const char *json_num(const char *obj, const char *end, const char *key, uint64_t *out) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = obj;
    while ((p = strstr(p, pat)) && p < end) {
        p += strlen(pat);
        while (p < end && (*p == ' ' || *p == '"')) p++;
        char *e = NULL; unsigned long long v = strtoull(p, &e, 10);
        if (e == p) return NULL;
        *out = v; return e;
    }
    return NULL;
}
static int parse_listing(const char *json, bank_entry *out, int cap) {
    int n = 0;
    const char *p = strstr(json, "\"shipments\"");
    if (!p) return 0;
    while (n < cap && (p = strchr(p, '{'))) {
        const char *end = strchr(p, '}');
        if (!end) break;
        const char *nm = strstr(p, "\"name\":\"");
        bank_entry e; memset(&e, 0, sizeof e);
        if (nm && nm < end) {
            nm += 8; const char *q = strchr(nm, '"');
            if (q && q < end && (size_t)(q - nm) < sizeof e.name) memcpy(e.name, nm, (size_t)(q - nm));
        }
        if (e.name[0] && !strchr(e.name, '/') && !strstr(e.name, "..")
            && json_num(p, end, "bytes", &e.bytes) && json_num(p, end, "index0", &e.index0) && json_num(p, end, "count", &e.count))
            out[n++] = e;
        p = end + 1;
    }
    return n;
}
static int by_index0(const void *a, const void *b) {
    const bank_entry *x = (const bank_entry *)a, *y = (const bank_entry *)b;
    return x->index0 < y->index0 ? -1 : x->index0 > y->index0;
}

static uint64_t cache_bytes(const sh_bank *b) {
    DIR *d = opendir(b->dir); if (!d) return 0;
    uint64_t total = 0; struct dirent *e;
    const size_t sl = strlen(b->seed_id);
    while ((e = readdir(d))) {
        const size_t n = strlen(e->d_name);
        if (n < sl + 6 || strncmp(e->d_name, b->seed_id, sl) || e->d_name[sl] != '-' || strcmp(e->d_name + n - 5, ".pads")) continue;
        char path[800]; snprintf(path, sizeof path, "%s/%s", b->dir, e->d_name);
        struct stat st; if (stat(path, &st) == 0) total += (uint64_t)st.st_size;
    }
    closedir(d);
    return total;
}

static void *bank_main(void *arg) {
    sh_bank *b = (sh_bank *)arg;
    bank_entry *list = (bank_entry *)calloc(BANK_MAX_ENTRIES, sizeof *list);
    if (!list) return NULL;
    while (!b->stop) {
        pthread_mutex_lock(&b->mu);
        const uint64_t floor = b->floor, need = b->need;
        pthread_mutex_unlock(&b->mu);

        char url[1400]; snprintf(url, sizeof url, "%s?seed_id=%s", b->url, b->seed_id);
        uint8_t *body = NULL; size_t blen = 0; int status = 0;
        int n = 0;
        if (sh_http_get(url, BANK_TIMEOUT_MS, &body, &blen, &status) == 0 && status == 200) n = parse_listing((const char *)body, list, BANK_MAX_ENTRIES);
        else snprintf(b->err, sizeof b->err, "bank listing %s: %s%d", b->url, status ? "http " : "unreachable ", status);
        free(body);
        pthread_mutex_lock(&b->mu); b->last_status = status; pthread_mutex_unlock(&b->mu);
        qsort(list, (size_t)n, sizeof *list, by_index0);

        uint64_t have = cache_bytes(b);
        for (int i = 0; i < n && !b->stop; i++) {
            const bank_entry *e = &list[i];
            if (e->index0 + e->count <= floor) continue;                      /* spent */
            const bool wanted = e->index0 < need;                             /* covers the engine's horizon: fetch regardless of budget */
            if (!wanted && have + e->bytes > b->cache_max) break;             /* ahead of the horizon: only inside the budget */
            char fin[800], tmp[800];
            snprintf(fin, sizeof fin, "%s/%s", b->dir, e->name);
            snprintf(tmp, sizeof tmp, "%s/.%s.part", b->dir, e->name);
            struct stat st;
            if (stat(fin, &st) == 0 && (uint64_t)st.st_size == e->bytes) continue;
            char furl[1400]; snprintf(furl, sizeof furl, "%s/%s/%s", b->url, b->seed_id, e->name);
            uint64_t got = 0;
            if (sh_http_download(furl, tmp, BANK_TIMEOUT_MS, &got) == 0 && got == e->bytes && rename(tmp, fin) == 0) {
                pthread_mutex_lock(&b->mu); b->fetched_files++; b->fetched_bytes += got; pthread_mutex_unlock(&b->mu);
                have += got;
            } else {
                unlink(tmp);
                snprintf(b->err, sizeof b->err, "bank fetch %s failed (%llu of %llu bytes)", e->name, (unsigned long long)got, (unsigned long long)e->bytes);
                break;                                                         /* retry next pass; keep index order */
            }
        }

        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += (long)BANK_POLL_MS * 1000000L; while (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_mutex_lock(&b->mu);
        if (!b->stop && b->need == need && b->floor == floor) pthread_cond_timedwait(&b->cv, &b->mu, &ts);
        pthread_mutex_unlock(&b->mu);
    }
    free(list);
    return NULL;
}

sh_bank *sh_bank_open(const char *url, const char *seed_id_hex, const char *dir, uint64_t cache_max, int *err) {
    if (err) *err = SH_OK;
    if (!url || strncmp(url, "http://", 7) || !seed_id_hex || strlen(seed_id_hex) != 32 || !dir || !*dir) { if (err) *err = SH_ERR_RANGE; return NULL; }
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) { if (err) *err = SH_ERR_IO; return NULL; }
    sh_bank *b = (sh_bank *)calloc(1, sizeof *b);
    if (!b) { if (err) *err = SH_ERR_NOMEM; return NULL; }
    snprintf(b->url, sizeof b->url, "%s", url);
    { size_t n = strlen(b->url); while (n && b->url[n - 1] == '/') b->url[--n] = 0; }
    snprintf(b->seed_id, sizeof b->seed_id, "%s", seed_id_hex);
    snprintf(b->dir, sizeof b->dir, "%s", dir);
    b->cache_max = cache_max;
    pthread_mutex_init(&b->mu, NULL); pthread_cond_init(&b->cv, NULL);
    if (pthread_create(&b->th, NULL, bank_main, b) != 0) {
        pthread_mutex_destroy(&b->mu); pthread_cond_destroy(&b->cv); free(b);
        if (err) *err = SH_ERR_IO; return NULL;
    }
    b->running = true;
    return b;
}
void sh_bank_set_floor(sh_bank *b, uint64_t floor) {
    if (!b) return;
    pthread_mutex_lock(&b->mu); if (floor > b->floor) b->floor = floor; pthread_cond_signal(&b->cv); pthread_mutex_unlock(&b->mu);
}
void sh_bank_set_need(sh_bank *b, uint64_t need) {
    if (!b) return;
    pthread_mutex_lock(&b->mu); if (need > b->need) b->need = need; pthread_cond_signal(&b->cv); pthread_mutex_unlock(&b->mu);
}
void sh_bank_stats(const sh_bank *b, uint64_t *files, uint64_t *bytes, int *last_status, char *err, size_t err_cap) {
    if (!b) {
        if (files) *files = 0;
        if (bytes) *bytes = 0;
        if (last_status) *last_status = 0;
        if (err && err_cap) err[0] = 0;
        return;
    }
    pthread_mutex_lock((pthread_mutex_t *)&b->mu);
    if (files) *files = b->fetched_files;
    if (bytes) *bytes = b->fetched_bytes;
    if (last_status) *last_status = b->last_status;
    if (err && err_cap) snprintf(err, err_cap, "%s", b->err);
    pthread_mutex_unlock((pthread_mutex_t *)&b->mu);
}
void sh_bank_close(sh_bank *b) {
    if (!b) return;
    pthread_mutex_lock(&b->mu); b->stop = true; pthread_cond_broadcast(&b->cv); pthread_mutex_unlock(&b->mu);
    if (b->running) pthread_join(b->th, NULL);
    pthread_mutex_destroy(&b->mu); pthread_cond_destroy(&b->cv);
    free(b);
}
