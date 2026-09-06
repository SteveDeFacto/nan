import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('OpenMP wall profiling preserves local-library work and excludes inactive/nested regions', {
  skip: process.platform !== 'linux',
}, () => {
  const dir = mkdtempSync(join(tmpdir(), 'shielded-omp-profile-'));
  const source = new URL('../wasm/ggml-shielded/shielded-omp-profile.c', import.meta.url).pathname;
  try {
    const library = join(dir, 'profile.so');
    execFileSync('cc', ['-std=c11', '-O2', '-D_FORTIFY_SOURCE=2', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared', source,
      '-ldl', '-lpthread', '-o', library]);
    writeFileSync(join(dir, 'task.c'), `
#define _DEFAULT_SOURCE
#include <omp.h>
#include <unistd.h>
int run(int nested) {
    int sum = 0;
    omp_set_dynamic(0);
    omp_set_max_active_levels(3);
    #pragma omp parallel num_threads(4) reduction(+:sum)
    {
        const int id = omp_get_thread_num();
        int inner = 1;
        if (nested) {
            inner = 0;
            #pragma omp parallel num_threads(2) reduction(+:inner)
            {
                int factor = 1;
                if (nested > 1) {
                    factor = 0;
                    #pragma omp parallel num_threads(2) reduction(+:factor)
                    {
                        factor += 1;
                        #pragma omp barrier
                    }
                }
                inner += factor;
                #pragma omp barrier
            }
        }
        sum += (id + 1) * inner;
        if (id == 0) usleep(2000);
        #pragma omp barrier
    }
    return sum;
}
`);
    execFileSync('cc', ['-O2', '-fPIC', '-shared', '-fopenmp', join(dir, 'task.c'), '-o', join(dir, 'task.so')]);
    writeFileSync(join(dir, 'test.c'), `
#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
static int active;
int ProfilingIsEnabledForAllThreads(void) { return active; }
int main(int argc, char **argv) {
    assert(argc == 2);
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    assert(lib);
    int (*run)(int) = (int (*)(int))dlsym(lib, "run");
    void (*snapshot)(uint64_t *) = (void (*)(uint64_t *))dlsym(RTLD_DEFAULT, "enclave_omp_profile_snapshot");
    void (*snapshot_v2)(uint64_t *) = (void (*)(uint64_t *))dlsym(RTLD_DEFAULT, "enclave_omp_profile_snapshot_v2");
    assert(run && snapshot && snapshot_v2);
    uint64_t values[3];
    uint64_t extended[5];
    assert(run(0) == 10);
    snapshot(values); assert(values[0] == 0);
    snapshot_v2(extended); assert(extended[3] == 0 && extended[4] == 0);
    active = 1;
    for (int i = 0; i < 10; i++) assert(run(0) == 10);
    snapshot(values);
    assert(values[0] == 10 && values[2] >= 20000000 && values[1] >= values[2]);
    snapshot_v2(extended);
    for (int i = 0; i < 3; i++) assert(extended[i] == values[i]);
    assert(extended[3] == 10 && extended[4] > 0 && extended[4] <= extended[2]);
    assert(run(1) == 20);
    snapshot(values); assert(values[0] == 11 && values[1] >= values[2]);
    snapshot_v2(extended);
    assert(extended[3] == 12 && extended[4] <= extended[2]);
    assert(run(2) == 40);
    snapshot(values); assert(values[0] == 12 && values[1] >= values[2]);
    snapshot_v2(extended);
    assert(extended[3] == 15 && extended[4] <= extended[2]);
    uint64_t saved[3] = { values[0], values[1], values[2] };
    uint64_t saved_barriers[2] = { extended[3], extended[4] };
    active = 0;
    assert(run(2) == 40);
    snapshot(values);
    for (int i = 0; i < 3; i++) assert(saved[i] == values[i]);
    snapshot_v2(extended);
    assert(extended[3] == saved_barriers[0] && extended[4] == saved_barriers[1]);
    return 0;
}
`);
    execFileSync('cc', ['-O2', '-rdynamic', join(dir, 'test.c'), '-ldl', '-o', join(dir, 'test')]);
    assert.doesNotThrow(() => execFileSync(join(dir, 'test'), [join(dir, 'task.so')], {
      env: { ...process.env, LD_PRELOAD: library }, timeout: 10000,
    }));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
