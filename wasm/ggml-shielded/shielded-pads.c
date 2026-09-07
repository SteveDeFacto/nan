/* Dealt pads: see shielded-pads.h and shielded/dealer/PLAN.md. */
#define _GNU_SOURCE
#include "shielded-pads.h"
#include "shielded-field.h"
#include "shielded-tee.h"
#include "tweetnacl.h"
#include "poly1305-donna.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

/* ChaCha20 block with a 64-bit counter and a zero nonce: the mask bank's stream
 * (shielded-tee.c) and the dealt-pad derivation draw from this one definition. */
#define B_ROTL32(v, c) (((v) << (c)) | ((v) >> (32 - (c))))
#define B_QR(a, b, c, d) (a += b, d ^= a, d = B_ROTL32(d, 16), c += d, b ^= c, b = B_ROTL32(b, 12), a += b, d ^= a, d = B_ROTL32(d, 8), c += d, b ^= c, b = B_ROTL32(b, 7))
void sh_chacha20_block(const uint32_t key[8], uint64_t counter, uint32_t out[16]) {
    static const uint32_t C[4] = { 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574 };
    uint32_t s[16];
    s[0] = C[0]; s[1] = C[1]; s[2] = C[2]; s[3] = C[3];
    for (int i = 0; i < 8; i++) s[4 + i] = key[i];
    s[12] = (uint32_t)counter; s[13] = (uint32_t)(counter >> 32);
    s[14] = 0; s[15] = 0;
    uint32_t x[16]; memcpy(x, s, sizeof x);
    for (int i = 0; i < 10; i++) {
        B_QR(x[0], x[4], x[ 8], x[12]); B_QR(x[1], x[5], x[ 9], x[13]);
        B_QR(x[2], x[6], x[10], x[14]); B_QR(x[3], x[7], x[11], x[15]);
        B_QR(x[0], x[5], x[10], x[15]); B_QR(x[1], x[6], x[11], x[12]);
        B_QR(x[2], x[7], x[ 8], x[13]); B_QR(x[3], x[4], x[ 9], x[14]);
    }
    for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
}

/* TweetNaCl's entropy hook: the OS, as everywhere else in the trusted half.
 * Weak, so a host that already defines one (the pVM payload) keeps its own. */
__attribute__((weak)) void randombytes(unsigned char *p, unsigned long long n) {
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

/* Cells and the header are ChaCha20-Poly1305 (RFC 8439) under the shipment
 * key: the shipment key is fresh per file, so the 96-bit nonce only has to be
 * unique within it - (index, group) is, and the header takes an all-0xEE
 * nonce no cell can have (group 0xEEEEEEEE is far past the 65,535 cap). The
 * key wrap stays crypto_box (32 bytes; speed is irrelevant there). */
static void cell_nonce(uint8_t n[12], uint64_t index, uint32_t group) { put_u64(n, index); put_u32(n + 8, group); }
static void hdr_nonce(uint8_t n[12]) { memset(n, 0xEE, 12); }
static void key_nonce(uint8_t n[24], const uint8_t seed_id[16]) { memcpy(n, seed_id, 16); memset(n + 16, 0xFF, 8); }

static uint64_t cell_bytes(uint64_t u_len) { return SH_PADS_CELL_TAG + 3 * u_len; }
static void chacha_ietf_block(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12], uint8_t out[64]);
static int  aead_seal(const uint8_t key[32], const uint8_t nonce[12], const uint8_t *msg, size_t n, uint8_t *out /* n + 16 */);
static int  aead_open(const uint8_t key[32], const uint8_t nonce[12], const uint8_t *box /* n + 16 */, size_t n, uint8_t *out);

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

    uint8_t digest[64], hn[12];
    header_hash(h, w->groups, n_groups, digest);
    hdr_nonce(hn);
    if (aead_seal(w->key, hn, digest, 64, h->hdr_box) != SH_OK) { *err = SH_ERR_NOMEM; goto fail; }

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
    uint8_t nonce[12];
    cell_nonce(nonce, index, group);
    int rc = aead_seal(w->key, nonce, p, (size_t)(3 * u_len), w->cell);
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

/* Several importer threads read cells concurrently: the file list is guarded
 * by `mu` (scans mutate it), cell scratch is per call, and pread on a shared
 * descriptor is safe. */
struct sh_pads_reader {
    char dir[512];
    uint8_t seed_id[16];
    uint8_t sk[32];
    pthread_mutex_t mu;
    bool have_digest; uint8_t model_digest[32];   /* when set, shipments for another model are not ours */
    sh_pads_file *files; size_t n_files, cap_files;
    sh_pads_group *bound; uint32_t n_bound;
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
        memcmp(h->seed_id, r->seed_id, 16) || (r->have_digest && memcmp(h->model_digest, r->model_digest, 32))) { file_close(f); return SH_ERR_VERIFY; }
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

    uint8_t want[64], got[64], hn[12];
    header_hash(h, f->groups, h->group_count, want);
    hdr_nonce(hn);
    if (aead_open(f->key, hn, h->hdr_box, 64, got) != SH_OK || memcmp(want, got, 64)) { file_close(f); return SH_ERR_VERIFY; }

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
    pthread_mutex_init(&r->mu, NULL);
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
    /* Resolve the cell under the lock (scans mutate the list), then read and
     * open it outside: the file entry only ever grows the list, never moves. */
    pthread_mutex_lock(&r->mu);
    sh_pads_file *f = reader_find(r, index);
    if (!f) { reader_scan(r); f = reader_find(r, index); }
    int fd = -1, g = -1; uint64_t u_len = 0, off = 0; uint8_t key[32], seed_id[16];
    if (f) {
        g = f->ordinal_of[group];
        u_len = f->groups[g].u_len;
        off = f->hdr.data_off + (index - f->hdr.index0) * f->hdr.row_bytes + f->group_off[g];
        fd = f->fd;
        memcpy(key, f->key, 32); memcpy(seed_id, f->hdr.seed_id, 16);
    }
    pthread_mutex_unlock(&r->mu);
    if (!f) return SH_ERR_EXHAUST;
    const size_t n = (size_t)cell_bytes(u_len);
    uint8_t *cell = (uint8_t *)malloc(n), *plain = (uint8_t *)malloc((size_t)(3 * u_len));
    int rc = SH_OK;
    if (!cell || !plain) rc = SH_ERR_NOMEM;
    else if (pread(fd, cell, n, (off_t)off) != (ssize_t)n) rc = SH_ERR_IO;
    else {
        uint8_t nonce[12];
        cell_nonce(nonce, index, (uint32_t)g);
        (void)seed_id;
        rc = aead_open(key, nonce, cell, (size_t)(3 * u_len), plain);
        for (uint64_t j = 0; rc == SH_OK && j < u_len; j++) {
            const int64_t v = (int64_t)plain[3 * j] | ((int64_t)plain[3 * j + 1] << 8) | ((int64_t)plain[3 * j + 2] << 16);
            if (v >= SH_M_MOD) { rc = SH_ERR_VERIFY; break; }
            u_out[j] = (int32_t)(v - SH_HALF_M);
        }
    }
    memset(key, 0, sizeof key);
    free(cell); free(plain);
    return rc;
}

uint32_t sh_pads_reader_groups(const sh_pads_reader *r, sh_pads_group *out, uint32_t cap) {
    if (!r || !r->n_files) return 0;
    pthread_mutex_lock((pthread_mutex_t *)&r->mu);
    const sh_pads_file *f = &r->files[0];
    uint32_t n = f->hdr.group_count < cap ? f->hdr.group_count : cap;
    if (out) memcpy(out, f->groups, (size_t)n * sizeof *out);
    pthread_mutex_unlock((pthread_mutex_t *)&r->mu);
    return n;
}

int sh_pads_reader_prune_below(sh_pads_reader *r, uint64_t floor, bool unlink_files) {
    if (!r || !floor) return 0;
    int dropped = 0;
    pthread_mutex_lock(&r->mu);
    for (size_t i = 0; i < r->n_files; ) {
        sh_pads_file *f = &r->files[i];
        if (f->hdr.index0 + f->hdr.index_count > floor) { i++; continue; }
        char path[512]; snprintf(path, sizeof path, "%s", f->path);
        file_close(f);
        if (unlink_files) unlink(path);
        r->n_files--;
        if (i < r->n_files) { r->files[i] = r->files[r->n_files]; memset(&r->files[r->n_files], 0, sizeof r->files[0]); r->files[r->n_files].fd = -1; }
        dropped++;
    }
    pthread_mutex_unlock(&r->mu);
    return dropped;
}

uint64_t sh_pads_reader_extent(const sh_pads_reader *r) {
    uint64_t hi = 0;
    if (!r) return 0;
    pthread_mutex_lock((pthread_mutex_t *)&r->mu);
    for (size_t i = 0; i < r->n_files; i++) {
        const uint64_t e = r->files[i].hdr.index0 + r->files[i].hdr.index_count;
        if (e > hi) hi = e;
    }
    pthread_mutex_unlock((pthread_mutex_t *)&r->mu);
    return hi;
}

void sh_pads_reader_close(sh_pads_reader *r) {
    if (!r) return;
    for (size_t i = 0; i < r->n_files; i++) file_close(&r->files[i]);
    free(r->files); free(r->bound);
    pthread_mutex_destroy(&r->mu);
    memset(r->sk, 0, sizeof r->sk);
    free(r);
}

void sh_pads_reader_require_digest(sh_pads_reader *r, const uint8_t model_digest[32]) {
    if (!r) return;
    pthread_mutex_lock(&r->mu);
    memcpy(r->model_digest, model_digest, 32); r->have_digest = true;
    /* Drop anything already opened for another model. */
    size_t kept = 0;
    for (size_t i = 0; i < r->n_files; i++) {
        if (!memcmp(r->files[i].hdr.model_digest, model_digest, 32)) { if (kept != i) r->files[kept] = r->files[i]; kept++; }
        else file_close(&r->files[i]);
    }
    r->n_files = kept;
    pthread_mutex_unlock(&r->mu);
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


/* ---- HMAC/HKDF-SHA512 over TweetNaCl's crypto_hash --------------------- */
static void hmac_sha512(const uint8_t *key, size_t klen, const uint8_t *m1, size_t n1, const uint8_t *m2, size_t n2, uint8_t out[64]) {
    uint8_t k[128] = {0}, kh[64];
    if (klen > 128) { crypto_hash(kh, key, klen); memcpy(k, kh, 64); } else memcpy(k, key, klen);
    uint8_t *inner = (uint8_t *)malloc(128 + n1 + n2), ih[64], outer[128 + 64];
    if (!inner) { memset(out, 0, 64); return; }
    for (int i = 0; i < 128; i++) inner[i] = k[i] ^ 0x36;
    memcpy(inner + 128, m1, n1); memcpy(inner + 128 + n1, m2, n2);
    crypto_hash(ih, inner, 128 + n1 + n2);
    for (int i = 0; i < 128; i++) outer[i] = k[i] ^ 0x5c;
    memcpy(outer + 128, ih, 64);
    crypto_hash(out, outer, 128 + 64);
    free(inner);
}
/* HKDF-SHA512 (RFC 5869), one or two blocks: what the relay's hkdfSync does. */
static void hkdf_sha512(const uint8_t *ikm, size_t ikm_len, const uint8_t *salt, size_t salt_len,
                        const char *info, uint8_t *okm, size_t okm_len) {
    uint8_t prk[64], t[64]; size_t tl = 0, done = 0; uint8_t ctr = 1;
    hmac_sha512(salt, salt_len, ikm, ikm_len, NULL, 0, prk);
    while (done < okm_len) {
        uint8_t *buf = (uint8_t *)malloc(tl + strlen(info) + 1);
        memcpy(buf, t, tl); memcpy(buf + tl, info, strlen(info)); buf[tl + strlen(info)] = ctr;
        hmac_sha512(prk, 64, buf, tl + strlen(info) + 1, NULL, 0, t);
        free(buf);
        const size_t n = okm_len - done < 64 ? okm_len - done : 64;
        memcpy(okm + done, t, n); done += n; tl = 64; ctr++;
    }
    memset(prk, 0, sizeof prk); memset(t, 0, sizeof t);
}

/* ---- ChaCha20 (RFC 8439: 32-bit counter, 96-bit nonce) + Poly1305 -------- */
#define P_ROTL(v, c) (((v) << (c)) | ((v) >> (32 - (c))))
#define P_QR(a, b, c, d) (a += b, d ^= a, d = P_ROTL(d, 16), c += d, b ^= c, b = P_ROTL(b, 12), a += b, d ^= a, d = P_ROTL(d, 8), c += d, b ^= c, b = P_ROTL(b, 7))
static uint32_t ld32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void chacha_ietf_block(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12], uint8_t out[64]) {
    static const uint32_t C[4] = { 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574 };
    uint32_t s[16], x[16];
    for (int i = 0; i < 4; i++) s[i] = C[i];
    for (int i = 0; i < 8; i++) s[4 + i] = ld32(key + 4 * i);
    s[12] = counter; s[13] = ld32(nonce); s[14] = ld32(nonce + 4); s[15] = ld32(nonce + 8);
    memcpy(x, s, sizeof x);
    for (int i = 0; i < 10; i++) {
        P_QR(x[0], x[4], x[8], x[12]); P_QR(x[1], x[5], x[9], x[13]); P_QR(x[2], x[6], x[10], x[14]); P_QR(x[3], x[7], x[11], x[15]);
        P_QR(x[0], x[5], x[10], x[15]); P_QR(x[1], x[6], x[11], x[12]); P_QR(x[2], x[7], x[8], x[13]); P_QR(x[3], x[4], x[9], x[14]);
    }
    for (int i = 0; i < 16; i++) { const uint32_t v = x[i] + s[i]; out[4 * i] = (uint8_t)v; out[4 * i + 1] = (uint8_t)(v >> 8); out[4 * i + 2] = (uint8_t)(v >> 16); out[4 * i + 3] = (uint8_t)(v >> 24); }
}
/* RFC 8439 AEAD with no AAD, Poly1305 by poly1305-donna (incremental, so the
 * tag runs over the ciphertext without a padded copy): tag over ct || pad16 ||
 * le64(0) || le64(ct_len) under the one-time key from block 0, data from
 * blocks 1.. Layout on the wire: 16-byte tag, then the ciphertext. */
static void aead_tag(const uint8_t otk[32], const uint8_t *ct, size_t n, uint8_t tag[16]) {
    static const uint8_t zeros[16] = {0};
    poly1305_context st;
    poly1305_init(&st, otk);
    poly1305_update(&st, ct, n);
    if (n % 16) poly1305_update(&st, zeros, 16 - n % 16);
    uint8_t lens[16] = {0};
    for (int i = 0; i < 8; i++) lens[8 + i] = (uint8_t)((uint64_t)n >> (8 * i));
    poly1305_update(&st, lens, 16);
    poly1305_finish(&st, tag);
}
static void aead_stream(const uint8_t key[32], const uint8_t nonce[12], const uint8_t *in, size_t n, uint8_t *out) {
    uint8_t blk[64];
    for (size_t off = 0, ctr = 1; off < n; off += 64, ctr++) {
        chacha_ietf_block(key, (uint32_t)ctr, nonce, blk);
        const size_t take = n - off < 64 ? n - off : 64;
        for (size_t i = 0; i < take; i++) out[off + i] = in[off + i] ^ blk[i];
    }
    memset(blk, 0, sizeof blk);
}
static int aead_seal(const uint8_t key[32], const uint8_t nonce[12], const uint8_t *msg, size_t n, uint8_t *out) {
    uint8_t blk[64], otk[32];
    chacha_ietf_block(key, 0, nonce, blk); memcpy(otk, blk, 32);
    aead_stream(key, nonce, msg, n, out + 16);
    aead_tag(otk, out + 16, n, out);
    memset(otk, 0, sizeof otk);
    return SH_OK;
}
static int aead_open(const uint8_t key[32], const uint8_t nonce[12], const uint8_t *box, size_t n, uint8_t *out) {
    uint8_t blk[64], otk[32], tag[16];
    chacha_ietf_block(key, 0, nonce, blk); memcpy(otk, blk, 32);
    aead_tag(otk, box + 16, n, tag);
    memset(otk, 0, sizeof otk);
    if (!poly1305_verify(tag, box)) return SH_ERR_VERIFY;
    aead_stream(key, nonce, box + 16, n, out);
    return SH_OK;
}

int sh_pads_seed_open(const uint8_t epk[32], const uint8_t nonce[12], const uint8_t *box, size_t box_len,
                      const uint8_t pad_sk[32], const uint8_t pad_pk[32], uint8_t seed_out[32]) {
    if (box_len != 32 + 16) return SH_ERR_RANGE;
    uint8_t shared[32], salt[64], key[32];
    if (crypto_scalarmult(shared, pad_sk, epk) != 0) return SH_ERR_VERIFY;
    memcpy(salt, epk, 32); memcpy(salt + 32, pad_pk, 32);
    hkdf_sha512(shared, 32, salt, 64, "enclave-pads-seed-box", key, 32);
    uint8_t laid[48]; memcpy(laid, box + 32, 16); memcpy(laid + 16, box, 32);   /* Node's ct||tag -> our tag||ct */
    const int rc = aead_open(key, nonce, laid, 32, seed_out);
    memset(shared, 0, sizeof shared); memset(key, 0, sizeof key);
    return rc;
}

bool sh_pads_window_verify(const uint8_t ledger_pk[32], const char *seed_id_hex, uint64_t lo, uint64_t hi, uint64_t iat, const uint8_t sig[64]) {
    char msg[256];
    const int n = snprintf(msg, sizeof msg, "enclave-pads-window\n%s\n%llu\n%llu\n%llu", seed_id_hex,
                           (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)iat);
    if (n <= 0 || n >= (int)sizeof msg) return false;
    uint8_t sm[64 + 256], m[64 + 256]; unsigned long long mlen = 0;
    memcpy(sm, sig, 64); memcpy(sm + 64, msg, (size_t)n);
    return crypto_sign_open(m, &mlen, sm, 64 + (unsigned long long)n, ledger_pk) == 0 && mlen == (unsigned long long)n;
}

void sh_pads_request_sign(const uint8_t transport_sk[64], const char *kind, const char *const *fields, size_t n_fields, const char *nonce_hex, uint8_t sig_out[64]) {
    char msg[1024]; size_t at = 0;
    at += (size_t)snprintf(msg + at, sizeof msg - at, "enclave-pads-%s", kind);
    for (size_t i = 0; i < n_fields && at < sizeof msg; i++) at += (size_t)snprintf(msg + at, sizeof msg - at, "\n%s", fields[i]);
    if (at < sizeof msg) at += (size_t)snprintf(msg + at, sizeof msg - at, "\n%s", nonce_hex);
    if (at >= sizeof msg) at = sizeof msg - 1;
    uint8_t *sm = (uint8_t *)malloc(64 + at); unsigned long long smlen = 0;
    crypto_sign(sm, &smlen, (const uint8_t *)msg, at, transport_sk);
    memcpy(sig_out, sm, 64);
    free(sm);
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
