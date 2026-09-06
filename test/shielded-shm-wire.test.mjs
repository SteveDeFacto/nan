import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

test('ring discovery skips occupied slots, stays within its BAR and stops on invalid peer replies', () => {
  const dir = mkdtempSync(join(tmpdir(), 'shielded-shm-wire-'));
  const source = fileURLToPath(new URL('../wasm/ggml-shielded/shielded-wire.c', import.meta.url));
  try {
    writeFileSync(join(dir, 'test.c'), `
#include ${JSON.stringify(source)}
#include <assert.h>
#include <pthread.h>

typedef struct { int fd, mode, seen; } peer;
static void *serve(void *arg) {
    peer *p = arg;
    const int requests = p->mode < 2 ? 2 : 1;
    for (int i = 0; i < requests; i++) {
        uint8_t req[13]; assert(read_all(p->fd, req, sizeof req) == SH_OK);
        assert(req[0] == SH_CMD_SHM_ATTACH && get_u64(req + 1) == 4);
        assert(req[9] == i && req[10] == 0 && req[11] == 0 && req[12] == 0);
        p->seen++;
        uint8_t reply[35] = {0};
        const size_t size = p->mode == 3 ? 26 : 25;
        reply[0] = p->mode == 4 ? 1 : 0;
        put_u64(reply + 1, size);
        reply[9] = p->mode == 1 ? 0 : (p->mode >= 2 || i == 1);
        put_u64(reply + 10, SH_RING_BYTES + (p->mode == 2));
        put_u64(reply + 18, SH_RING_REQ_CAP);
        put_u64(reply + 26, SH_RING_REP_CAP);
        size_t offset = 0;
        while (offset < size + 9) {
            ssize_t n = write(p->fd, reply + offset, size + 9 - offset);
            assert(n > 0); offset += (size_t)n;
        }
    }
    close(p->fd);
    return NULL;
}
static void run_case(int mode, const char *file) {
    int sockets[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    sh_pipe *p = calloc(1, sizeof *p); assert(p); p->fd = sockets[0];
    peer worker = {sockets[1], mode, 0}; pthread_t thread;
    assert(pthread_create(&thread, NULL, serve, &worker) == 0);
    int index = -1;
    int rc = sh_pipe_shm_attach_available(p, file, 2 * SH_RING_BYTES, &index);
    pthread_join(thread, NULL);
    if (mode == 0) {
        assert(rc == SH_OK && index == 1 && worker.seen == 2);
        assert(sh_pipe_ring_live(p) && p->ring == p->map + SH_RING_BYTES);
    } else if (mode == 1) {
        assert(rc == SH_ERR_IO && index == -1 && worker.seen == 2);
        assert(!sh_pipe_ring_live(p));
    } else {
        assert(rc == (mode == 4 ? SH_ERR_VIOLATION : SH_ERR_PROTO));
        assert(index == -1 && worker.seen == 1 && !sh_pipe_ring_live(p));
    }
    sh_pipe_close(p);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    int fd = open(argv[1], O_CREAT | O_TRUNC | O_RDWR, 0600); assert(fd >= 0);
    assert(ftruncate(fd, 2 * SH_RING_BYTES) == 0); close(fd);
    for (int mode = 0; mode < 5; mode++) run_case(mode, argv[1]);
    unlink(argv[1]);
}
`);
    execFileSync('cc', ['-std=c11', '-O1', '-Wall', '-Wextra', '-ffunction-sections', '-fdata-sections',
      join(dir, 'test.c'), '-Wl,--gc-sections', '-lpthread', '-o', join(dir, 'test')], { timeout: 30_000 });
    execFileSync(join(dir, 'test'), [join(dir, 'bar')], { timeout: 5_000 });
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
