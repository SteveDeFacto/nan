/* Dealt pads: see shielded-pads.h and shielded/dealer/PLAN.md. */
#define _GNU_SOURCE
#include "shielded-pads.h"
#include "shielded-field.h"
#include "shielded-tee.h"
#include "tweetnacl.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

/* The engine's ChaCha20 block (shielded-tee.c), exported for the derivation. */
void sh_chacha20_block(const uint32_t key[8], uint64_t counter, uint32_t out[16]);

/* TweetNaCl's entropy hook: the OS, as everywhere else in the trusted half. */
void randombytes(unsigned char *p, unsigned long long n) {
    while (n) {
        ssize_t r = getrandom(p, n, 0);
        if (r < 0) { if (errno == EINTR) continue; abort(); }
        p += r; n -= (unsigned long long)r;
    }
}

/* ---- r derivation ------------------------------------------------------ */
void sh_pad_r(const uint8_t seed[32], uint32_t group, uint64_t index, int64_t K, int32_t *r_out) {
    uint32_t key[8];
    for (int i = 0; i < 8; i++)
        key[i] = (uint32_t)seed[4 * i] | ((uint32_t)seed[4 * i + 1] << 8) |
                 ((uint32_t)seed[4 * i + 2] << 16) | ((uint32_t)seed[4 * i + 3] << 24);
    uint64_t ctr = ((uint64_t)group << 48) | ((index & 0xFFFFFFull) << 24);
    uint32_t blk[16];
    int64_t produced = 0;
    while (produced < K) {
        sh_chacha20_block(key, ctr++, blk);
        for (int i = 0; i + 1 < 16 && produced < K; i += 2) {
            const uint64_t v = ((uint64_t)blk[i + 1] << 32) | blk[i];
            r_out[produced++] = (int32_t)(v % (uint64_t)SH_M_MOD);
        }
    }
}

/* ---- helpers ------------------------------------------------------------ */
static void put_u64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put_u32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }

static void cell_nonce(uint8_t n[24], const uint8_t seed_id[16], uint64_t index, uint32_t group) {
    memcpy(n, seed_id, 8); put_u64(n + 8, index); put_u32(n + 16, group); memset(n + 20, 0, 4);
}
static void key_nonce(uint8_t n[24], const uint8_t seed_id[16]) { memcpy(n, seed_id, 16); memset(n + 16, 0xFF, 8); }
static void hdr_nonce(uint8_t n[24], const uint8_t seed_id[16]) { memcpy(n, seed_id, 16); memset(n + 16, 0xEE, 8); }

static uint64_t cell_bytes(uint64_t u_len) { return SH_PADS_CELL_TAG + 3 * u_len; }

/* Serialise the fixed header + table, with key_box/hdr_box zeroed, and hash it. */
static void header_hash(const sh_pads_hdr *h, const sh_pads_group *groups, uint32_t n, uint8_t out[64]) {
    sh_pads_hdr c = *h;
    memset(c.key_box, 0, sizeof c.key_box);
    memset(c.hdr_box, 0, sizeof c.hdr_box);
    const size_t len = sizeof c + (size_t)n * sizeof(sh_pads_group);
    uint8_t *buf = (uint8_t *)malloc(len);
    if (!buf) { memset(out, 0, 64); return; }
    memcpy(buf, &c, sizeof c);
    memcpy(buf + sizeof c, groups, (size_t)n * sizeof(sh_pads_group));
    crypto_hash(out, buf, len);
    free(buf);
}

static int secretbox_seal(uint8_t *out /* 16 + n */, const uint8_t *msg, size_t n, const uint8_t nonce[24], const uint8_t key[32]) {
    uint8_t *m = (uint8_t *)calloc(n + 32, 1), *c = (uint8_t *)malloc(n + 32);
    if (!m || !c) { free(m); free(c); return SH_ERR_NOMEM; }
    memcpy(m + 32, msg, n);
    crypto_secretbox(c, m, n + 32, nonce, key);
    memcpy(out, c + 16, n + 16);
    free(m); free(c);
    return SH_OK;
}
static int secretbox_open(uint8_t *msg, const uint8_t *box /* 16 + n */, size_t n, const uint8_t nonce[24], const uint8_t key[32]) {
    uint8_t *c = (uint8_t *)calloc(n + 32, 1), *m = (uint8_t *)malloc(n + 32);
    if (!c || !m) { free(c); free(m); return SH_ERR_NOMEM; }
    memcpy(c + 16, box, n + 16);
    const int rc = crypto_secretbox_open(m, c, n + 32, nonce, key);
    if (rc == 0) memcpy(msg, m + 32, n);
    free(c); free(m);
    return rc == 0 ? SH_OK : SH_ERR_VERIFY;
}

/* ---- writer ------------------------------------------------------------- */
struct sh_pads_writer {
    int fd;
    sh_pads_hdr hdr;
    sh_pads_group *groups;
    uint64_t *group_off;               /* within a row */
    uint8_t key[32];
    uint8_t *cell, *plain;             /* scratch: largest cell */
    size_t cell_cap;
};

sh_pads_writer *sh_pads_writer_open(const char *path, const uint8_t model_digest[32],
                                    const uint8_t seed_id[16], const sh_pads_group *groups, uint32_t n_groups,
                                    uint64_t index0, uint64_t index_count, const uint8_t consumer_pk[32], int *err) {
    *err = SH_OK;
    if (!n_groups || !index_count) { *err = SH_ERR_RANGE; return NULL; }
    sh_pads_writer *w = (sh_pads_writer *)calloc(1, sizeof *w);
    if (!w) { *err = SH_ERR_NOMEM; return NULL; }
    w->fd = -1;
    w->groups = (sh_pads_group *)calloc(n_groups, sizeof *w->groups);
    w->group_off = (uint64_t *)calloc(n_groups, sizeof *w->group_off);
    if (!w->groups || !w->group_off) { *err = SH_ERR_NOMEM; goto fail; }
    memcpy(w->groups, groups, (size_t)n_groups * sizeof *groups);
    uint64_t row = 0, biggest = 0;
    for (uint32_t g = 0; g < n_groups; g++) {
        w->group_off[g] = row;
        const uint64_t cb = cell_bytes(groups[g].u_len);
        row += cb;
        if (cb > biggest) biggest = cb;
    }
    w->cell_cap = (size_t)biggest;
    w->cell = (uint8_t *)malloc(w->cell_cap);
    w->plain = (uint8_t *)malloc(w->cell_cap);
    if (!w->cell || !w->plain) { *err = SH_ERR_NOMEM; goto fail; }

    sh_pads_hdr *h = &w->hdr;
    memset(h, 0, sizeof *h);
    memcpy(h->magic, SH_PADS_MAGIC, 8);
    h->version = SH_PADS_VERSION;
    h->group_count = n_groups;
    memcpy(h->model_digest, model_digest, 32);
    memcpy(h->seed_id, seed_id, 16);
    h->index0 = index0; h->index_count = index_count;
    h->data_off = sizeof *h + (uint64_t)n_groups * sizeof(sh_pads_group);
    h->data_off = (h->data_off + 4095) & ~(uint64_t)4095;
    h->row_bytes = row;

    /* Shipment key, boxed to the consumer from an ephemeral pair. */
    randombytes(w->key, 32);
    uint8_t epk[32], esk[32], shared[32], nonce[24];
    crypto_box_keypair(epk, esk);
    crypto_box_beforenm(shared, consumer_pk, esk);
    memcpy(h->dealer_pk, epk, 32);
    key_nonce(nonce, seed_id);
    if (secretbox_seal(h->key_box, w->key, 32, nonce, shared) != SH_OK) { *err = SH_ERR_NOMEM; goto fail; }
    memset(esk, 0, sizeof esk); memset(shared, 0, sizeof shared);

    uint8_t digest[64];
    header_hash(h, w->groups, n_groups, digest);
    hdr_nonce(nonce, seed_id);
    if (secretbox_seal(h->hdr_box, digest, 64, nonce, w->key) != SH_OK) { *err = SH_ERR_NOMEM; goto fail; }

    w->fd = open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (w->fd < 0) { *err = SH_ERR_IO; goto fail; }
    if (pwrite(w->fd, h, sizeof *h, 0) != (ssize_t)sizeof *h ||
        pwrite(w->fd, w->groups, (size_t)n_groups * sizeof *w->groups, sizeof *h) != (ssize_t)((size_t)n_groups * sizeof *w->groups)) {
        *err = SH_ERR_IO; goto fail;
    }
    if (ftruncate(w->fd, (off_t)(h->data_off + h->row_bytes * index_count)) != 0) { *err = SH_ERR_IO; goto fail; }
    return w;
fail:
    sh_pads_writer_close(w);
    return NULL;
}

int sh_pads_writer_cell(sh_pads_writer *w, uint64_t index, uint32_t group, const int32_t *u) {
    if (!w || group >= w->hdr.group_count) return SH_ERR_RANGE;
    if (index < w->hdr.index0 || index >= w->hdr.index0 + w->hdr.index_count) return SH_ERR_RANGE;
    const uint64_t u_len = w->groups[group].u_len;
    uint8_t *p = w->plain;
    for (uint64_t j = 0; j < u_len; j++) {
        const int64_t v = (int64_t)u[j] + SH_HALF_M;
        if (v < 0 || v >= SH_M_MOD) return SH_ERR_RANGE;
        p[3 * j] = (uint8_t)v; p[3 * j + 1] = (uint8_t)(v >> 8); p[3 * j + 2] = (uint8_t)(v >> 16);
    }
    uint8_t nonce[24];
    cell_nonce(nonce, w->hdr.seed_id, index, group);
    int rc = secretbox_seal(w->cell, p, (size_t)(3 * u_len), nonce, w->key);
    if (rc != SH_OK) return rc;
    const uint64_t off = w->hdr.data_off + (index - w->hdr.index0) * w->hdr.row_bytes + w->group_off[group];
    const size_t n = (size_t)cell_bytes(u_len);
    if (pwrite(w->fd, w->cell, n, (off_t)off) != (ssize_t)n) return SH_ERR_IO;
    return SH_OK;
}

int sh_pads_writer_close(sh_pads_writer *w) {
    if (!w) return SH_OK;
    int rc = SH_OK;
    if (w->fd >= 0) { if (fsync(w->fd) != 0) rc = SH_ERR_IO; close(w->fd); }
    memset(w->key, 0, sizeof w->key);
    free(w->groups); free(w->group_off); free(w->cell); free(w->plain); free(w);
    return rc;
}

/* ---- reader ------------------------------------------------------------- */
typedef struct {
    char path[512];
    int fd;
    sh_pads_hdr hdr;
    sh_pads_group *groups;
    uint64_t *group_off;
    uint8_t key[32];
    int *ordinal_of;                    /* consumer group -> this shipment's group, or -1 */
} sh_pads_file;

struct sh_pads_reader {
    char dir[512];
    uint8_t seed_id[16];
    uint8_t sk[32];
    sh_pads_file *files; size_t n_files, cap_files;
    sh_pads_group *bound; uint32_t n_bound;
    uint8_t *cell, *plain; size_t cell_cap;
};

static void file_close(sh_pads_file *f) {
    if (f->fd >= 0) close(f->fd);
    free(f->groups); free(f->group_off); free(f->ordinal_of);
    memset(f->key, 0, sizeof f->key);
    memset(f, 0, sizeof *f); f->fd = -1;
}

static bool reader_has(const sh_pads_reader *r, const char *path) {
    for (size_t i = 0; i < r->n_files; i++) if (!strcmp(r->files[i].path, path)) return true;
    return false;
}

/* Open one shipment: header, key unwrap, header check, group table. */
static int file_open(sh_pads_reader *r, const char *path, sh_pads_file *f) {
    memset(f, 0, sizeof *f); f->fd = -1;
    snprintf(f->path, sizeof f->path, "%s", path);
    f->fd = open(path, O_RDONLY | O_CLOEXEC);
    if (f->fd < 0) return SH_ERR_IO;
    if (pread(f->fd, &f->hdr, sizeof f->hdr, 0) != (ssize_t)sizeof f->hdr) { file_close(f); return SH_ERR_IO; }
    sh_pads_hdr *h = &f->hdr;
    if (memcmp(h->magic, SH_PADS_MAGIC, 8) || h->version != SH_PADS_VERSION || !h->group_count || h->group_count > 65535 ||
        memcmp(h->seed_id, r->seed_id, 16)) { file_close(f); return SH_ERR_VERIFY; }
    f->groups = (sh_pads_group *)calloc(h->group_count, sizeof *f->groups);
    f->group_off = (uint64_t *)calloc(h->group_count, sizeof *f->group_off);
    if (!f->groups || !f->group_off) { file_close(f); return SH_ERR_NOMEM; }
    const size_t tb = (size_t)h->group_count * sizeof(sh_pads_group);
    if (pread(f->fd, f->groups, tb, sizeof *h) != (ssize_t)tb) { file_close(f); return SH_ERR_IO; }

    uint8_t shared[32], nonce[24];
    crypto_box_beforenm(shared, h->dealer_pk, r->sk);
    key_nonce(nonce, h->seed_id);
    int rc = secretbox_open(f->key, h->key_box, 32, nonce, shared);
    memset(shared, 0, sizeof shared);
    if (rc != SH_OK) { file_close(f); return SH_ERR_VERIFY; }

    uint8_t want[64], got[64];
    header_hash(h, f->groups, h->group_count, want);
    hdr_nonce(nonce, h->seed_id);
    if (secretbox_open(got, h->hdr_box, 64, nonce, f->key) != SH_OK || memcmp(want, got, 64)) { file_close(f); return SH_ERR_VERIFY; }

    uint64_t row = 0;
    for (uint32_t g = 0; g < h->group_count; g++) {
        f->group_off[g] = row;
        row += cell_bytes(f->groups[g].u_len);
    }
    if (row != h->row_bytes) { file_close(f); return SH_ERR_VERIFY; }
    return SH_OK;
}

static int file_bind(sh_pads_reader *r, sh_pads_file *f) {
    free(f->ordinal_of);
    f->ordinal_of = (int *)malloc((size_t)r->n_bound * sizeof(int));
    if (!f->ordinal_of) return SH_ERR_NOMEM;
    for (uint32_t b = 0; b < r->n_bound; b++) {
        f->ordinal_of[b] = -1;
        for (uint32_t g = 0; g < f->hdr.group_count; g++) {
            const sh_pads_group *sg = &f->groups[g], *bg = &r->bound[b];
            if (!strncmp(sg->name, bg->name, SH_PADS_NAME_MAX) && sg->K == bg->K && sg->u_len == bg->u_len) { f->ordinal_of[b] = (int)g; break; }
        }
        if (f->ordinal_of[b] < 0) return SH_ERR_VERIFY;   /* a shipment that lacks a group is not usable */
    }
    return SH_OK;
}

static int reader_scan(sh_pads_reader *r) {
    DIR *d = opendir(r->dir);
    if (!d) return SH_ERR_IO;
    struct dirent *e;
    int added = 0;
    while ((e = readdir(d))) {
        const size_t n = strlen(e->d_name);
        if (n < 6 || strcmp(e->d_name + n - 5, ".pads")) continue;
        char path[512];
        snprintf(path, sizeof path, "%s/%s", r->dir, e->d_name);
        if (reader_has(r, path)) continue;
        if (r->n_files == r->cap_files) {
            const size_t cap = r->cap_files ? r->cap_files * 2 : 8;
            sh_pads_file *nf = (sh_pads_file *)realloc(r->files, cap * sizeof *nf);
            if (!nf) { closedir(d); return SH_ERR_NOMEM; }
            r->files = nf; r->cap_files = cap;
        }
        sh_pads_file *f = &r->files[r->n_files];
        if (file_open(r, path, f) != SH_OK) continue;          /* not ours, or damaged: skipped, never trusted */
        if (r->n_bound && file_bind(r, f) != SH_OK) { file_close(f); continue; }
        r->n_files++; added++;
    }
    closedir(d);
    return added;
}

sh_pads_reader *sh_pads_reader_open(const char *dir, const uint8_t seed_id[16], const uint8_t consumer_sk[32], int *err) {
    *err = SH_OK;
    sh_pads_reader *r = (sh_pads_reader *)calloc(1, sizeof *r);
    if (!r) { *err = SH_ERR_NOMEM; return NULL; }
    snprintf(r->dir, sizeof r->dir, "%s", dir);
    memcpy(r->seed_id, seed_id, 16);
    memcpy(r->sk, consumer_sk, 32);
    const int rc = reader_scan(r);
    if (rc < 0) { *err = rc; sh_pads_reader_close(r); return NULL; }
    return r;
}

int sh_pads_reader_bind(sh_pads_reader *r, const sh_pads_group *groups, uint32_t n_groups) {
    if (!r || !n_groups) return SH_ERR_RANGE;
    free(r->bound);
    r->bound = (sh_pads_group *)calloc(n_groups, sizeof *groups);
    if (!r->bound) return SH_ERR_NOMEM;
    memcpy(r->bound, groups, (size_t)n_groups * sizeof *groups);
    r->n_bound = n_groups;
    uint64_t biggest = 0;
    for (uint32_t g = 0; g < n_groups; g++) if (cell_bytes(groups[g].u_len) > biggest) biggest = cell_bytes(groups[g].u_len);
    free(r->cell); free(r->plain);
    r->cell_cap = (size_t)biggest;
    r->cell = (uint8_t *)malloc(r->cell_cap);
    r->plain = (uint8_t *)malloc(r->cell_cap);
    if (!r->cell || !r->plain) return SH_ERR_NOMEM;
    /* Re-bind what is already open; drop shipments that do not fit. */
    size_t kept = 0;
    for (size_t i = 0; i < r->n_files; i++) {
        if (file_bind(r, &r->files[i]) == SH_OK) { if (kept != i) r->files[kept] = r->files[i]; kept++; }
        else file_close(&r->files[i]);
    }
    r->n_files = kept;
    return SH_OK;
}

static sh_pads_file *reader_find(sh_pads_reader *r, uint64_t index) {
    for (size_t i = 0; i < r->n_files; i++) {
        sh_pads_file *f = &r->files[i];
        if (index >= f->hdr.index0 && index < f->hdr.index0 + f->hdr.index_count) return f;
    }
    return NULL;
}

int sh_pads_reader_cell(sh_pads_reader *r, uint32_t group, uint64_t index, int32_t *u_out) {
    if (!r || !r->n_bound || group >= r->n_bound) return SH_ERR_RANGE;
    sh_pads_file *f = reader_find(r, index);
    if (!f) { reader_scan(r); f = reader_find(r, index); }
    if (!f) return SH_ERR_EXHAUST;
    const int g = f->ordinal_of[group];
    const uint64_t u_len = f->groups[g].u_len;
    const size_t n = (size_t)cell_bytes(u_len);
    const uint64_t off = f->hdr.data_off + (index - f->hdr.index0) * f->hdr.row_bytes + f->group_off[g];
    if (pread(f->fd, r->cell, n, (off_t)off) != (ssize_t)n) return SH_ERR_IO;
    uint8_t nonce[24];
    cell_nonce(nonce, f->hdr.seed_id, index, (uint32_t)g);
    int rc = secretbox_open(r->plain, r->cell, (size_t)(3 * u_len), nonce, f->key);
    if (rc != SH_OK) return rc;
    for (uint64_t j = 0; j < u_len; j++) {
        const int64_t v = (int64_t)r->plain[3 * j] | ((int64_t)r->plain[3 * j + 1] << 8) | ((int64_t)r->plain[3 * j + 2] << 16);
        if (v >= SH_M_MOD) return SH_ERR_VERIFY;
        u_out[j] = (int32_t)(v - SH_HALF_M);
    }
    return SH_OK;
}

uint64_t sh_pads_reader_extent(const sh_pads_reader *r) {
    uint64_t hi = 0;
    for (size_t i = 0; r && i < r->n_files; i++) {
        const uint64_t e = r->files[i].hdr.index0 + r->files[i].hdr.index_count;
        if (e > hi) hi = e;
    }
    return hi;
}

void sh_pads_reader_close(sh_pads_reader *r) {
    if (!r) return;
    for (size_t i = 0; i < r->n_files; i++) file_close(&r->files[i]);
    free(r->files); free(r->bound); free(r->cell); free(r->plain);
    memset(r->sk, 0, sizeof r->sk);
    free(r);
}

/* ---- ledger window (P1: a local file) ----------------------------------- */
int sh_pads_window_reserve(const char *ledger_path, uint64_t want, uint64_t *lo, uint64_t *hi) {
    if (!ledger_path || !want) return SH_ERR_RANGE;
    int fd = open(ledger_path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return SH_ERR_IO;
    char buf[32] = {0};
    ssize_t n = pread(fd, buf, sizeof buf - 1, 0);
    uint64_t mark = 0;
    if (n > 0) mark = strtoull(buf, NULL, 10);
    const uint64_t next = mark + want;
    const int len = snprintf(buf, sizeof buf, "%llu\n", (unsigned long long)next);
    /* Advance the mark BEFORE handing out the window (reserve-before-use). */
    if (pwrite(fd, buf, (size_t)len, 0) != len || ftruncate(fd, len) != 0 || fsync(fd) != 0) { close(fd); return SH_ERR_IO; }
    close(fd);
    *lo = mark; *hi = next;
    return SH_OK;
}

/* ---- hex --------------------------------------------------------------- */
bool sh_pads_hex2bin(const char *hex, uint8_t *out, size_t n) {
    if (!hex || strlen(hex) != 2 * n) return false;
    for (size_t i = 0; i < n; i++) {
        unsigned v = 0;
        for (int k = 0; k < 2; k++) {
            const char c = hex[2 * i + k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else return false;
        }
        out[i] = (uint8_t)v;
    }
    return true;
}
void sh_pads_bin2hex(const uint8_t *in, size_t n, char *out) {
    static const char *d = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) { out[2 * i] = d[in[i] >> 4]; out[2 * i + 1] = d[in[i] & 15]; }
    out[2 * n] = 0;
}
