/* Opt-in companion to the bounded CPU sampler. Timings cover complete outer
 * GOMP_parallel calls and the calling thread's callback, including its waits.
 * Their difference is region entry/exit time, not all synchronization time.
 * No tensor data is read. Disabled captures pass the original call through. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

extern int ProfilingIsEnabledForAllThreads(void) __attribute__((weak));
typedef void (*parallel_fn)(void (*)(void *), void *, unsigned, unsigned);
static parallel_fn real_parallel;
static void (*real_barrier)(void);
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t totals_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t totals[5]; /* regions, region ns, caller ns, caller barriers, barrier ns */
static _Thread_local unsigned depth;

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static void resolve_parallel(void) {
    /* ggml may load libgomp in a local dlopen scope, outside RTLD_NEXT.
     * Looking up the ABI in the library itself also avoids our interposer. */
    void *lib = dlopen("libgomp.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib) {
        real_parallel = (parallel_fn)dlsym(lib, "GOMP_parallel");
        real_barrier = (void (*)(void))dlsym(lib, "GOMP_barrier");
    }
    if (!real_parallel || !real_barrier) {
        static const char message[] = "OpenMP profile: libgomp ABI unavailable\n";
        const ssize_t written = write(STDERR_FILENO, message, sizeof message - 1);
        (void)written;
        _exit(127);
    }
}

struct invocation {
    void (*fn)(void *);
    void *data;
    pthread_t caller;
    uint64_t caller_ns;
    uint64_t caller_barriers;
    uint64_t caller_barrier_ns;
};
static _Thread_local struct invocation *profiled_caller;

/* Count the caller's time inside public barriers, not the sum across all
 * workers. This includes waiting for slower workers and scheduler wakeups;
 * it is not necessarily removable synchronization overhead. */
void GOMP_barrier(void) {
    pthread_once(&resolve_once, resolve_parallel);
    struct invocation *call = profiled_caller;
    if (!call) {
        real_barrier();
        return;
    }
    const uint64_t start = now_ns();
    real_barrier();
    call->caller_barriers++;
    call->caller_barrier_ns += now_ns() - start;
}

static void invoke(void *arg) {
    struct invocation *call = arg;
    const int caller = pthread_equal(pthread_self(), call->caller);
    struct invocation *previous = profiled_caller;
    profiled_caller = caller ? call : NULL;
    const uint64_t start = caller ? now_ns() : 0;
    depth++;
    call->fn(call->data);
    depth--;
    if (caller) call->caller_ns = now_ns() - start;
    profiled_caller = previous;
}

struct nested_invocation {
    void (*fn)(void *);
    void *data;
};

/* Propagate nesting to newly created inner-team threads as well as the
 * original caller, so deeper nested teams do not become extra outer regions. */
static void invoke_nested(void *arg) {
    struct nested_invocation *call = arg;
    depth++;
    call->fn(call->data);
    depth--;
}

void GOMP_parallel(void (*fn)(void *), void *data, unsigned threads, unsigned flags) {
    pthread_once(&resolve_once, resolve_parallel);
    if (depth) {
        struct nested_invocation call = {fn, data};
        real_parallel(invoke_nested, &call, threads, flags);
        return;
    }
    if (!ProfilingIsEnabledForAllThreads || !ProfilingIsEnabledForAllThreads()) {
        real_parallel(fn, data, threads, flags);
        return;
    }
    struct invocation call = {fn, data, pthread_self(), 0, 0, 0};
    depth++;
    const uint64_t start = now_ns();
    real_parallel(invoke, &call, threads, flags);
    const uint64_t elapsed = now_ns() - start;
    depth--;
    pthread_mutex_lock(&totals_mu);
    totals[0]++;
    totals[1] += elapsed;
    totals[2] += call.caller_ns;
    totals[3] += call.caller_barriers;
    totals[4] += call.caller_barrier_ns;
    pthread_mutex_unlock(&totals_mu);
}

/* Process-wide sums can exceed elapsed capture time if callers overlap.
 * A region active at capture stop is included when it finishes. */
void enclave_omp_profile_snapshot(uint64_t out[3]) {
    pthread_mutex_lock(&totals_mu);
    for (int i = 0; i < 3; i++) out[i] = totals[i];
    pthread_mutex_unlock(&totals_mu);
}

/* Keep the original three-value ABI intact for older native backends. */
void enclave_omp_profile_snapshot_v2(uint64_t out[5]) {
    pthread_mutex_lock(&totals_mu);
    for (int i = 0; i < 5; i++) out[i] = totals[i];
    pthread_mutex_unlock(&totals_mu);
}
