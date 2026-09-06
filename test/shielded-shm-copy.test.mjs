import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

test('optional streaming ring copy preserves exact lengths, alignments and guard boundaries', {
  skip: process.arch !== 'x64',
}, t => {
  const dir = mkdtempSync(join(tmpdir(), 'shielded-shm-copy-'));
  const source = fileURLToPath(new URL('../wasm/ggml-shielded/shielded-wire.c', import.meta.url));
  try {
    writeFileSync(join(dir, 'test.c'), `
#include ${JSON.stringify(source)}
#include <assert.h>
int main(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("sse4.1")) return 77;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    uint8_t *m = mmap(NULL, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(m != MAP_FAILED && mprotect(m + page, page, PROT_READ | PROT_WRITE) == 0);
    uint8_t *dest = malloc(page + 32); assert(dest);
    const size_t sizes[] = {0,1,2,3,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025};
    for (int round = 0; round < 3; round++) {
        for (size_t k = 0; k < page; k++) m[page + k] = (uint8_t)(k * 71 + round * 37);
        for (unsigned i = 0; i < sizeof sizes / sizeof *sizes; i++) {
            const size_t n = sizes[i];
            for (size_t gap = 0; gap < 16; gap++) for (size_t off = 0; off < 16; off++) {
                uint8_t *src = m + 2 * page - gap - n;
                memset(dest, 0xa5, page + 32);
                ring_stream_copy(dest + off, src, n);
                assert(memcmp(dest + off, src, n) == 0);
                for (size_t k = 0; k < off; k++) assert(dest[k] == 0xa5);
                for (size_t k = off + n; k < page + 32; k++) assert(dest[k] == 0xa5);
            }
        }
        // The maximum page-bound copy touches both guards if it overruns.
        ring_stream_copy(dest, m + page, page);
        assert(memcmp(dest, m + page, page) == 0);
    }
    free(dest); munmap(m, 3 * page);
    return 0;
}
`);
    execFileSync('cc', ['-std=c11', '-O2', '-Wall', '-Wextra', '-ffunction-sections', '-fdata-sections',
      join(dir, 'test.c'), '-Wl,--gc-sections', '-o', join(dir, 'test')], { timeout: 30_000 });
    const result = spawnSync(join(dir, 'test'), { encoding: 'utf8', timeout: 10_000 });
    if (result.status === 77) { t.skip('SSE4.1 hardware unavailable; optimized object compiled'); return; }
    assert.equal(result.status, 0, result.stderr || String(result.error));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
