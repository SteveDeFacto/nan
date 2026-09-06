#define _GNU_SOURCE
#include "shielded-wire.h"

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>   /* after sys/socket.h: needs sa_family_t (aarch64 glibc) */
#include <sys/uio.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SH_CPU_RELAX() _mm_pause()
#define SH_SFENCE()    _mm_sfence()
#elif defined(__aarch64__)
#define SH_CPU_RELAX() __asm__ __volatile__("yield")
#define SH_SFENCE()    __asm__ __volatile__("dmb ishst" ::: "memory")
#else
#define SH_CPU_RELAX() ((void)0)
#define SH_SFENCE()    __sync_synchronize()
#endif

/* A single frame is capped well below the point where a bad length header could
 * make us allocate the machine. Mirrors wire.py's MAX_FRAME. */
#define SH_MAX_FRAME ((size_t)256 << 20)
#define SH_HDR 9

struct sh_pipe {
    int  fd;
    char err[256];
    /* The reply buffer: grows to the largest reply ever seen, never shrinks,
     * and is reused by every exchange. Before this each decode exchange paid a
     * malloc and a free of the reply (608 KB for one lm_head row of the 0.5B)
     * on the request path. */
    uint8_t *rbuf;
    size_t   rcap;
    /* The shared-memory ring, when SHIELDED_SHM named one and the worker
     * granted it (see shielded-wire.h). `map` is the whole file/BAR; `ring`
     * the one ring this connection owns. `seq` is ours: monotonic, never
     * reused, so a stale reply from a worker restarted onto the same file can
     * never match. `misses` counts consecutive ring failures; at 3 the ring is
     * abandoned for the socket. */
    uint8_t *map;    size_t map_len;
    uint8_t *ring;
    uint64_t seq;
    int      misses;
};

/* Frames per exchange that fit the stack-resident iovec/header arrays. Every
 * production exchange is one frame; the pipelined batches of the older
 * protocol were three. Beyond this the arrays are malloc'd. */
#define SH_STACK_FRAMES 16

static int reply_reserve(sh_pipe *p, size_t want) {
    if (p->rcap >= want) return SH_OK;
    size_t cap = p->rcap ? p->rcap : 4096;
    while (cap < want) cap *= 2;
    uint8_t *nb = (uint8_t *)realloc(p->rbuf, cap);
    if (!nb) return SH_ERR_NOMEM;
    p->rbuf = nb; p->rcap = cap;
    return SH_OK;
}

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static uint64_t get_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

size_t sh_pack_hello(void *dst, uint32_t major, uint64_t reserve_bytes) {
    uint8_t *p = (uint8_t *)dst;
    put_u32(p, major);
    if (reserve_bytes == 0) return 4;      /* the pre-1.3 frame, byte for byte */
    put_u64(p + 4, reserve_bytes);
    return 12;
}
size_t sh_pack_alloc(void *dst, uint64_t size, const char *role) {
    uint8_t *p = (uint8_t *)dst; size_t n = strlen(role);
    put_u64(p, size); put_u32(p + 8, (uint32_t)n); memcpy(p + 12, role, n);
    return 12 + n;
}
size_t sh_pack_region(void *dst, uint64_t bid, uint64_t offset, uint64_t nbytes) {
    uint8_t *p = (uint8_t *)dst;
    put_u64(p, bid); put_u64(p + 8, offset); put_u64(p + 16, nbytes);
    return 24;
}
size_t sh_pack_set_tensor_header(void *dst, uint64_t bid, uint64_t offset, uint64_t nbytes) {
    return sh_pack_region(dst, bid, offset, nbytes);
}
size_t sh_pack_recompute(void *dst, uint32_t node, uint32_t m) {
    uint8_t *p = (uint8_t *)dst; put_u32(p, node); put_u32(p + 4, m); return 8;
}
size_t sh_pack_field_gemm(void *dst, uint32_t n_nodes, uint32_t m, const int *nodes) {
    uint8_t *p = (uint8_t *)dst; put_u32(p, n_nodes); put_u32(p + 4, m);
    for (uint32_t i = 0; i < n_nodes; i++) put_u32(p + 8 + 4 * i, (uint32_t)nodes[i]);
    return 8 + 4 * (size_t)n_nodes;
}

const char *sh_pipe_last_error(const sh_pipe *p) { return p ? p->err : ""; }

sh_pipe *sh_pipe_open(const char *host, int port, int *err) {
    if (err) *err = SH_OK;
    /* "vsock" or "vsock:<cid>": AF_VSOCK to the host (CID 2 unless told
     * otherwise). A vsock round trip is a small fraction of slirp's, and at
     * ~50 exchanges per token that fraction is most of the decode budget. */
    if (!strncmp(host, "vsock", 5)) {
        int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
        if (fd < 0) { if (err) *err = SH_ERR_IO; return NULL; }
        struct sockaddr_vm vm; memset(&vm, 0, sizeof vm);
        vm.svm_family = AF_VSOCK;
        vm.svm_cid = host[5] == ':' ? (unsigned)atoi(host + 6) : VMADDR_CID_HOST;
        vm.svm_port = (unsigned)port;
        if (connect(fd, (struct sockaddr *)&vm, sizeof vm) != 0) { close(fd); if (err) *err = SH_ERR_IO; return NULL; }
        sh_pipe *p = (sh_pipe *)calloc(1, sizeof *p);
        if (!p) { close(fd); if (err) *err = SH_ERR_NOMEM; return NULL; }
        p->fd = fd;
        return p;
    }
    char portstr[16]; snprintf(portstr, sizeof portstr, "%d", port);
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &res) != 0) { if (err) *err = SH_ERR_IO; return NULL; }
    int fd = -1;
    for (struct addrinfo *a = res; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) { if (err) *err = SH_ERR_IO; return NULL; }
    /* Mandatory, not an optimisation: without it Nagle holds the small SET_TENSOR
     * frame waiting for an ACK that the pipelined GET_TENSOR is itself waiting on,
     * and the exchange stalls for a full delayed-ACK timer. */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    sh_pipe *p = (sh_pipe *)calloc(1, sizeof *p);
    if (!p) { close(fd); if (err) *err = SH_ERR_NOMEM; return NULL; }
    p->fd = fd;
    return p;
}

void sh_pipe_close(sh_pipe *p) {
    if (!p) return;
    if (p->map) munmap(p->map, p->map_len);
    if (p->fd >= 0) close(p->fd);
    free(p->rbuf);
    free(p);
}

/* Not every libc exposes IOV_MAX; POSIX guarantees at least 16. */
#ifndef IOV_MAX
#define IOV_MAX 1024
#endif

static int write_all(int fd, struct iovec *iov, int iovcnt) {
    while (iovcnt > 0) {
        ssize_t n = writev(fd, iov, iovcnt > IOV_MAX ? IOV_MAX : iovcnt);
        if (n < 0) { if (errno == EINTR) continue; return SH_ERR_IO; }
        while (iovcnt > 0 && (size_t)n >= iov->iov_len) { n -= (ssize_t)iov->iov_len; iov++; iovcnt--; }
        if (iovcnt > 0 && n > 0) {
            iov->iov_base = (char *)iov->iov_base + n;
            iov->iov_len -= (size_t)n;
        }
    }
    return SH_OK;
}

/* How long a reply read spins before it blocks, in microseconds.
 *
 * Inside a CVM a blocking read is a vCPU halt, and getting the thread back is
 * an interrupt injected into an SEV-SNP guest plus the VM exits around it --
 * tens of microseconds each way, paid 49 times per token, and the term that
 * the host-loopback measurements never see. The reply to a decode exchange
 * arrives ~50-150 us after the request went out, so a bounded spin on a
 * non-blocking receive catches almost every reply with the vCPU still
 * running; a long one (lm_head, a cold worker) falls through to the blocking
 * read after the budget. Costs one core for the spin, which this thread was
 * spending in the halt anyway.
 *
 * OFF by default: it is an in-guest experiment, not a measured win. On the
 * host it is a wash over TCP (-2 us) and a LOSS over vsock loopback (+10 to
 * +24 us: the spinner starves the loopback transport's kernel worker on its
 * own CPU), and the guest, where it should pay, is exactly where it has not
 * been measured. Set SHIELDED_SPIN_US in the tenant's environment to try it. */
static int spin_us(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("SHIELDED_SPIN_US"); v = (e && *e) ? atoi(e) : 0; if (v < 0) v = 0; }
    return v;
}

static int read_all(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    const int budget = spin_us();
    if (budget > 0) {
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int spins = 0; n; spins++) {
            ssize_t r = recv(fd, p, n, MSG_DONTWAIT);
            if (r > 0) { p += r; n -= (size_t)r; continue; }
            if (r == 0) return SH_ERR_IO;
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return SH_ERR_IO;
            if ((spins & 63) == 63) {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                if ((t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_nsec - t0.tv_nsec) / 1000L > budget) break;
            }
        }
    }
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return SH_ERR_IO; }
        if (r == 0) return SH_ERR_IO;             /* peer closed mid-frame */
        p += r; n -= (size_t)r;
    }
    return SH_OK;
}

void sh_reply_free(sh_reply *r) {
    /* The bytes live in the pipe's buffer; only the view is cleared. */
    if (!r) return;
    r->data = NULL; r->len = 0;
}

int sh_pipe_exchange(sh_pipe *p, const sh_frame *frames, size_t n, sh_reply *out) {
    if (!p || p->fd < 0) return SH_ERR_IO;
    memset(out, 0, n * sizeof *out);

    /* One writev for every frame: 3 iovecs per frame (header, payload, payload2).
     * On the stack: a decode exchange is one frame, and two mallocs per
     * exchange are measurable against a ~60 us round trip. */
    struct iovec  iov_stack[3 * SH_STACK_FRAMES];
    uint8_t       hdr_stack[SH_HDR * SH_STACK_FRAMES];
    struct iovec *iov  = iov_stack;
    uint8_t      *hdrs = hdr_stack;
    if (n > SH_STACK_FRAMES) {
        iov  = (struct iovec *)malloc(3 * n * sizeof *iov);
        hdrs = (uint8_t *)malloc(n * SH_HDR);
        if (!iov || !hdrs) { free(iov); free(hdrs); return SH_ERR_NOMEM; }
    }
    int iovcnt = 0;
    for (size_t i = 0; i < n; i++) {
        size_t total = frames[i].len + frames[i].len2;
        uint8_t *h = hdrs + i * SH_HDR;
        h[0] = frames[i].cmd;
        put_u64(h + 1, total);
        iov[iovcnt].iov_base = h; iov[iovcnt].iov_len = SH_HDR; iovcnt++;
        if (frames[i].len) {
            iov[iovcnt].iov_base = (void *)frames[i].payload;
            iov[iovcnt].iov_len  = frames[i].len; iovcnt++;
        }
        if (frames[i].len2) {
            iov[iovcnt].iov_base = (void *)frames[i].payload2;
            iov[iovcnt].iov_len  = frames[i].len2; iovcnt++;
        }
    }
    int rc = write_all(p->fd, iov, iovcnt);
    if (n > SH_STACK_FRAMES) { free(iov); free(hdrs); }
    if (rc != SH_OK) { snprintf(p->err, sizeof p->err, "write failed: %s", strerror(errno)); return rc; }

    /* Replies land back to back in the pipe's buffer. Growing it can move the
     * earlier ones, so pointers are assigned once the whole batch is in. */
    size_t used = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t h[SH_HDR];
        if ((rc = read_all(p->fd, h, SH_HDR)) != SH_OK) {
            snprintf(p->err, sizeof p->err, "short response header at frame %zu", i);
            goto fail;
        }
        uint64_t size = get_u64(h + 1);
        if (size > SH_MAX_FRAME) {
            snprintf(p->err, sizeof p->err, "response frame %llu exceeds cap", (unsigned long long)size);
            rc = SH_ERR_PROTO; goto fail;
        }
        if (size && (rc = reply_reserve(p, used + (size_t)size)) != SH_OK) goto fail;
        out[i].status = h[0];
        out[i].len = (size_t)size;
        if (size && (rc = read_all(p->fd, p->rbuf + used, (size_t)size)) != SH_OK) {
            snprintf(p->err, sizeof p->err, "short response body at frame %zu", i);
            goto fail;
        }
        if (out[i].status != 0) {
            /* A violation is always the last frame: the worker closes after it.
             * Surface the reason verbatim -- it names the node and the op, which
             * is the difference between a five-minute fix and an afternoon. */
            int m = (int)(out[i].len < sizeof p->err - 1 ? out[i].len : sizeof p->err - 1);
            memcpy(p->err, p->rbuf + used, (size_t)m); p->err[m] = 0;
            rc = SH_ERR_VIOLATION; goto fail;
        }
        used += (size_t)size;
    }
    used = 0;
    for (size_t i = 0; i < n; i++) { out[i].data = out[i].len ? p->rbuf + used : NULL; used += out[i].len; }
    return SH_OK;
fail:
    for (size_t i = 0; i < n; i++) sh_reply_free(&out[i]);
    return rc;
}

int sh_pipe_call(sh_pipe *p, uint8_t cmd, const void *payload, size_t len, sh_reply *out) {
    sh_frame f = { cmd, payload, len, NULL, 0 };
    return sh_pipe_exchange(p, &f, 1, out);
}

/* --- the shared-memory ring ------------------------------------------------
 * Why: in the CVM the vhost-vsock exchange costs 152 us, the socket itself
 * ~10 and the card ~28 (REPORT.md 13.13); the rest is VM exits and interrupts
 * that a polled ring never raises. Measured in an SEV-SNP guest on an ivshmem
 * BAR mapped write-back: 0.97 us of transport for the 0.5B gate|up exchange
 * (scratchpad/shm-ring/DESIGN.md). Every byte read from the ring is bounded
 * by OUR constants and compared against OUR expectation before use. */
static inline uint64_t ld_acq(const uint8_t *at) { return __atomic_load_n((const uint64_t *)at, __ATOMIC_ACQUIRE); }
static inline void st_rel(uint8_t *at, uint64_t v) {
    /* The guest's mapping of the BAR may be write-combining (sysfs
     * resource2_wc): WC stores are weakly ordered, so the payload is fenced
     * before the sequence number is published. Free on a write-back mapping. */
    SH_SFENCE();
    __atomic_store_n((uint64_t *)at, v, __ATOMIC_RELEASE);
}

int sh_pipe_ring_live(const sh_pipe *p) { return p && p->ring != NULL; }

static void ring_drop(sh_pipe *p) {
    if (p->map) munmap(p->map, p->map_len);
    p->map = NULL; p->ring = NULL;
}

int sh_pipe_shm_attach(sh_pipe *p, const char *path, int index, size_t bytes) {
    if (!p || p->fd < 0 || !path || !*path) return SH_ERR_IO;
    if (index < 0 || (size_t)index >= SH_RING_MAX_FILE / SH_RING_BYTES) {
        snprintf(p->err, sizeof p->err, "shm ring index %d out of range", index); return SH_ERR_IO;
    }
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) { snprintf(p->err, sizeof p->err, "shm open %s: %s", path, strerror(errno)); return SH_ERR_IO; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); snprintf(p->err, sizeof p->err, "shm fstat: %s", strerror(errno)); return SH_ERR_IO; }
    size_t len = st.st_size > 0 ? (size_t)st.st_size : bytes;
    if (len > SH_RING_MAX_FILE) len = SH_RING_MAX_FILE;
    len = (len / SH_RING_BYTES) * SH_RING_BYTES;
    /* The mapping is THE bound: every ring address is derived from the index
     * and the compiled constants, never from anything the peer writes. */
    if (len < (size_t)(index + 1) * SH_RING_BYTES) {
        close(fd);
        snprintf(p->err, sizeof p->err, "shm %s holds %zu ring(s); ring %d asked", path, len / SH_RING_BYTES, index);
        return SH_ERR_IO;
    }
    void *m = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { snprintf(p->err, sizeof p->err, "shm mmap %s: %s", path, strerror(errno)); return SH_ERR_IO; }

    uint8_t pay[4]; put_u32(pay, (uint32_t)index);
    sh_reply rep;
    int rc = sh_pipe_call(p, SH_CMD_SHM_ATTACH, pay, 4, &rep);
    if (rc != SH_OK) { munmap(m, len); return rc; }
    if (rep.len != 25) {
        munmap(m, len); sh_reply_free(&rep);
        snprintf(p->err, sizeof p->err, "SHM_ATTACH reply is %zu bytes, expected 25", rep.len);
        return SH_ERR_PROTO;
    }
    const int granted = rep.data[0];
    const uint64_t rb = get_u64(rep.data + 1), rq = get_u64(rep.data + 9), rp = get_u64(rep.data + 17);
    sh_reply_free(&rep);
    if (!granted) { munmap(m, len); snprintf(p->err, sizeof p->err, "worker did not grant shm ring %d", index); return SH_ERR_IO; }
    /* The geometry is CHECKED against our constants, never adopted. */
    if (rb != SH_RING_BYTES || rq != SH_RING_REQ_CAP || rp != SH_RING_REP_CAP) {
        munmap(m, len);
        snprintf(p->err, sizeof p->err, "worker ring geometry %llu/%llu/%llu differs from %zu/%zu/%zu",
                 (unsigned long long)rb, (unsigned long long)rq, (unsigned long long)rp,
                 SH_RING_BYTES, SH_RING_REQ_CAP, SH_RING_REP_CAP);
        return SH_ERR_PROTO;
    }
    p->map = (uint8_t *)m; p->map_len = len;
    p->ring = p->map + (size_t)index * SH_RING_BYTES;
    /* Sequence base from the clock: a worker restarted onto the same file
     * with an old reply still in the slot cannot match a fresh link's seq. */
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    p->seq = (((uint64_t)ts.tv_sec << 24) | ((uint64_t)ts.tv_nsec >> 6)) & 0x3fffffffffffffffULL;
    p->misses = 0;
    return SH_OK;
}

int sh_pipe_shm_attach_available(sh_pipe *p, const char *path, size_t bytes, int *index) {
    const size_t bound = bytes && bytes < SH_RING_MAX_FILE ? bytes : SH_RING_MAX_FILE;
    int rc = SH_ERR_IO;
    if (index) *index = -1;
    for (size_t i = 0; i < bound / SH_RING_BYTES; i++) {
        rc = sh_pipe_shm_attach(p, path, (int)i, bytes);
        if (rc == SH_OK) { if (index) *index = (int)i; return SH_OK; }
        // Occupied/unavailable rings can fall back. An invalid peer reply
        // cannot be retried into a seemingly trustworthy connection.
        if (rc != SH_ERR_IO) return rc;
    }
    return rc;
}

/* Longest the TEE spins for a ring reply before it sends the same frame on
 * the socket. The 4B lm_head at m=8 is ~1 ms on the card; 3 ms leaves room
 * for a card shared with another tenant without stalling the token forever. */
static int ring_spin_us(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("SHIELDED_SHM_SPIN_US"); v = (e && *e) ? atoi(e) : 3000; if (v < 1) v = 1; }
    return v;
}

int sh_pipe_ring_exchange(sh_pipe *p, const sh_frame *f, size_t want, sh_reply *out) {
    if (!p || !p->ring) return SH_ERR_IO;
    memset(out, 0, sizeof *out);
    const size_t total = f->len + f->len2;
    /* Both directions must fit by OUR arithmetic before a byte moves. */
    if (f->cmd != SH_CMD_FIELD_GEMM || total > SH_RING_REQ_CAP || want > SH_RING_REP_CAP || want == 0) return SH_ERR_IO;
    int rc = reply_reserve(p, want);
    if (rc != SH_OK) return rc;
    uint8_t *r = p->ring;
    /* What is written here is exactly the socket frame: header and payload. */
    if (f->len)  memcpy(r + SH_RING_OFF_RQP, f->payload, f->len);
    if (f->len2) memcpy(r + SH_RING_OFF_RQP + f->len, f->payload2, f->len2);
    r[SH_RING_OFF_RQH] = f->cmd;
    put_u64(r + SH_RING_OFF_RQH + 1, total);
    const uint64_t seq = ++p->seq;
    st_rel(r + SH_RING_OFF_REQ, seq);

    const int budget = ring_spin_us();
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int spins = 0;; spins++) {
        if (ld_acq(r + SH_RING_OFF_REP) == seq) break;
        SH_CPU_RELAX();
        if ((spins & 255) == 255) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if ((t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_nsec - t0.tv_nsec) / 1000L > budget) {
                if (++p->misses >= 3) ring_drop(p);
                snprintf(p->err, sizeof p->err, "shm ring: no reply within %d us", budget);
                return SH_ERR_IO;     /* the caller sends the same frame on the socket */
            }
        }
    }
    /* The header is validated against what WE expect before the payload is
     * read. A wrong length is a refusal, never "use the peer's length". */
    const uint8_t status = r[SH_RING_OFF_RPH];
    const uint64_t len = get_u64(r + SH_RING_OFF_RPH + 1);
    if (status != 0) {
        /* A violation on the ring is a violation: surface the (bounded)
         * reason and let the link take the connection down, as on the socket. */
        size_t m = len < sizeof p->err - 1 ? (size_t)len : sizeof p->err - 1;
        memcpy(p->err, r + SH_RING_OFF_RPP, m); p->err[m] = 0;
        return SH_ERR_VIOLATION;
    }
    if (len != want) {
        if (++p->misses >= 3) ring_drop(p);
        snprintf(p->err, sizeof p->err, "shm ring: reply of %llu bytes, expected %zu", (unsigned long long)len, want);
        return SH_ERR_IO;
    }
    /* Out of the shared page and into OUR buffer: unmask and Freivalds run on
     * this copy, so nothing the host does to the ring afterwards matters. */
    memcpy(p->rbuf, r + SH_RING_OFF_RPP, want);
    p->misses = 0;
    out->status = 0; out->data = p->rbuf; out->len = want;
    return SH_OK;
}
