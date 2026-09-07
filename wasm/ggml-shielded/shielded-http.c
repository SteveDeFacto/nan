#define _GNU_SOURCE
#include "shielded-http.h"
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

typedef struct { char host[256]; char port[8]; char path[1200]; } sh_url;

static char g_bearer[256] = "";
void sh_http_set_bearer(const char *token) {
    if (token && *token) snprintf(g_bearer, sizeof g_bearer, "%s", token); else g_bearer[0] = 0;
}

static int url_parse(const char *url, sh_url *u) {
    if (!url || strncmp(url, "http://", 7)) return -1;
    const char *p = url + 7, *slash = strchr(p, '/');
    const size_t hl = slash ? (size_t)(slash - p) : strlen(p);
    if (hl == 0 || hl >= sizeof u->host) return -1;
    char hp[256]; memcpy(hp, p, hl); hp[hl] = 0;
    char *colon = strrchr(hp, ':');
    if (colon && !strchr(colon, ']')) { *colon = 0; snprintf(u->port, sizeof u->port, "%.7s", colon + 1); }
    else snprintf(u->port, sizeof u->port, "80");
    if (hp[0] == '[') { size_t n = strlen(hp); if (n >= 2 && hp[n - 1] == ']') { memmove(hp, hp + 1, n - 2); hp[n - 2] = 0; } }
    snprintf(u->host, sizeof u->host, "%s", hp);
    snprintf(u->path, sizeof u->path, "%s", slash && *slash ? slash : "/");
    return 0;
}

static int tcp_connect(const sh_url *u, int timeout_ms) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints); hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(u->host, u->port, &hints, &res) != 0) return -1;
    int fd = -1;
    for (struct addrinfo *a = res; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv; tv.tv_sec = timeout_ms / 1000; tv.tv_usec = (timeout_ms % 1000) * 1000;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static int write_all(int fd, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    while (n) { ssize_t w = write(fd, b, n); if (w <= 0) { if (w < 0 && errno == EINTR) continue; return -1; } b += w; n -= (size_t)w; }
    return 0;
}

/* Buffered reader over the socket. */
typedef struct { int fd; uint8_t buf[1 << 16]; size_t len, pos; int eof; } rd_t;
static int rd_fill(rd_t *r) {
    if (r->eof) return 0;
    if (r->pos) { memmove(r->buf, r->buf + r->pos, r->len - r->pos); r->len -= r->pos; r->pos = 0; }
    if (r->len == sizeof r->buf) return -1;                /* a line longer than the buffer */
    ssize_t n = read(r->fd, r->buf + r->len, sizeof r->buf - r->len);
    if (n < 0) { if (errno == EINTR) return rd_fill(r); return -1; }
    if (n == 0) { r->eof = 1; return 0; }
    r->len += (size_t)n;
    return 1;
}
/* One CRLF-terminated line into `out` (without the CRLF); -1 on error/EOF. */
static int rd_line(rd_t *r, char *out, size_t cap) {
    for (;;) {
        for (size_t i = r->pos; i + 1 < r->len; i++) {
            if (r->buf[i] == '\r' && r->buf[i + 1] == '\n') {
                size_t n = i - r->pos; if (n >= cap) n = cap - 1;
                memcpy(out, r->buf + r->pos, n); out[n] = 0;
                r->pos = i + 2;
                return 0;
            }
        }
        int f = rd_fill(r);
        if (f < 0 || (f == 0 && r->eof)) return -1;
    }
}
/* Hand exactly n bytes to the sink (n may be huge; streamed). */
static int rd_take(rd_t *r, uint64_t n, sh_http_sink sink, void *ctx) {
    while (n) {
        if (r->pos == r->len) { int f = rd_fill(r); if (f < 0 || r->eof) return -1; }
        size_t have = r->len - r->pos; if (have > n) have = (size_t)n;
        if (sink && sink(ctx, r->buf + r->pos, have)) return -1;
        r->pos += have; n -= have;
    }
    return 0;
}
/* Everything until EOF (no length given). */
static int rd_rest(rd_t *r, sh_http_sink sink, void *ctx) {
    for (;;) {
        if (r->pos < r->len) { if (sink && sink(ctx, r->buf + r->pos, r->len - r->pos)) return -1; r->pos = r->len; }
        int f = rd_fill(r); if (f < 0) return -1; if (r->eof && r->pos == r->len) return 0;
    }
}

int sh_http_request(const char *method, const char *url, const char *content_type,
                    const void *body, size_t body_len, int timeout_ms,
                    sh_http_sink sink, void *ctx, int *status) {
    sh_url u;
    if (status) *status = 0;
    if (url_parse(url, &u)) return -1;
    int fd = tcp_connect(&u, timeout_ms);
    if (fd < 0) return -1;
    char hdr[2048];
    int hn = snprintf(hdr, sizeof hdr,
                      "%s %s HTTP/1.1\r\nHost: %s:%s\r\nUser-Agent: enclave-shielded/1\r\nAccept: */*\r\nConnection: close\r\n%s%s%s%s%s%s"
                      "Content-Length: %zu\r\n\r\n",
                      method, u.path, u.host, u.port,
                      content_type ? "Content-Type: " : "", content_type ? content_type : "", content_type ? "\r\n" : "",
                      g_bearer[0] ? "Authorization: Bearer " : "", g_bearer[0] ? g_bearer : "", g_bearer[0] ? "\r\n" : "",
                      body ? body_len : (size_t)0);
    if (hn <= 0 || (size_t)hn >= sizeof hdr || write_all(fd, hdr, (size_t)hn) || (body && body_len && write_all(fd, body, body_len))) { close(fd); return -1; }

    rd_t *r = (rd_t *)calloc(1, sizeof *r);
    if (!r) { close(fd); return -1; }
    r->fd = fd;
    char line[4096];
    int rc = -1;
    if (rd_line(r, line, sizeof line) == 0 && !strncmp(line, "HTTP/1.", 7)) {
        int st = atoi(line + 9);
        if (status) *status = st;
        long long clen = -1; int chunked = 0;
        for (;;) {
            if (rd_line(r, line, sizeof line)) goto out;
            if (!line[0]) break;
            if (!strncasecmp(line, "Content-Length:", 15)) clen = atoll(line + 15);
            else if (!strncasecmp(line, "Transfer-Encoding:", 18) && strcasestr(line + 18, "chunked")) chunked = 1;
        }
        if (chunked) {
            for (;;) {
                if (rd_line(r, line, sizeof line)) goto out;
                unsigned long long n = strtoull(line, NULL, 16);
                if (n == 0) { rc = 0; break; }                  /* trailers, if any, are ignored */
                if (rd_take(r, n, sink, ctx)) goto out;
                if (rd_line(r, line, sizeof line)) goto out;    /* the CRLF after the chunk */
            }
        } else if (clen >= 0) {
            rc = rd_take(r, (uint64_t)clen, sink, ctx);
        } else {
            rc = rd_rest(r, sink, ctx);
        }
    }
out:
    free(r);
    close(fd);
    return rc;
}

typedef struct { uint8_t *p; size_t n, cap; } mem_t;
static int mem_sink(void *ctx, const uint8_t *p, size_t n) {
    mem_t *m = (mem_t *)ctx;
    if (m->n + n + 1 > m->cap) {
        size_t cap = m->cap ? m->cap : 4096;
        while (cap < m->n + n + 1) cap *= 2;
        if (cap > (64u << 20)) return -1;                       /* a listing or a window, never a shipment */
        uint8_t *np = (uint8_t *)realloc(m->p, cap); if (!np) return -1;
        m->p = np; m->cap = cap;
    }
    memcpy(m->p + m->n, p, n); m->n += n; m->p[m->n] = 0;
    return 0;
}

static int to_mem(const char *method, const char *url, const char *json, int timeout_ms, uint8_t **out, size_t *out_len, int *status) {
    mem_t m; memset(&m, 0, sizeof m);
    if (mem_sink(&m, (const uint8_t *)"", 0)) return -1;
    int rc = sh_http_request(method, url, json ? "application/json" : NULL, json, json ? strlen(json) : 0, timeout_ms, mem_sink, &m, status);
    if (rc) { free(m.p); *out = NULL; *out_len = 0; return -1; }
    *out = m.p; *out_len = m.n;
    return 0;
}
int sh_http_get(const char *url, int timeout_ms, uint8_t **out, size_t *out_len, int *status) {
    return to_mem("GET", url, NULL, timeout_ms, out, out_len, status);
}
int sh_http_post_json(const char *url, const char *json, int timeout_ms, uint8_t **out, size_t *out_len, int *status) {
    return to_mem("POST", url, json, timeout_ms, out, out_len, status);
}

typedef struct { int fd; uint64_t n; } file_t;
static int file_sink(void *ctx, const uint8_t *p, size_t n) {
    file_t *f = (file_t *)ctx;
    if (write_all(f->fd, p, n)) return -1;
    f->n += n;
    return 0;
}
int sh_http_download(const char *url, const char *path, int timeout_ms, uint64_t *bytes) {
    file_t f; f.n = 0;
    f.fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (f.fd < 0) return -1;
    int status = 0;
    int rc = sh_http_request("GET", url, NULL, NULL, 0, timeout_ms, file_sink, &f, &status);
    if (rc == 0 && status == 200 && fsync(f.fd) == 0) { close(f.fd); if (bytes) *bytes = f.n; return 0; }
    close(f.fd); unlink(path);
    return -1;
}
