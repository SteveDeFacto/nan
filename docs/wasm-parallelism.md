# Multi-core Wasm on the platform: what works, what doesn't, and why

Written 2026-08-07 after building against the engine rather than reading about
it. Every claim below was produced by running something; the commands are
reproducible from `tools/parallelism-probe/`.

Short version, as of the final 2026-08-07 revision: **shared-everything-threads
now really spawn OS threads and really run in parallel from inside a
component. Measured 27.9x on 32 threads (16 physical cores), on the engine we build.**
`thread.spawn-ref` and `thread.spawn-indirect` return live thread ids, guest
`user` CPU time scales linearly with thread count while `real` stays flat, and
the whole thing is reachable from a component — the boundary that carries our
egress rules, the loopback wall and the WASI capability model.

The rest of this file is the history of getting there, kept because it records
which walls are real and which only looked real. Read
"[SET spawn is real](#set-spawn-is-real-2026-08-07-final)" first if you only
want the current state.

## The three threading models, kept distinct

They get conflated constantly, so:

| model | what it gives | status here |
|---|---|---|
| **cooperative threads** (wasip3 🧵) | `pthread`/`std::thread` interleaved on one core — concurrency, thread-shaped code ports | **SHIPPED** 2026-08-07, see docs/wasip3-threads.md |
| **wasi-threads** (p1) | real OS threads, shared linear memory, true parallelism | deleted upstream (b4b23fe583), **REBUILT here** in `src/commands/run.rs`; core modules only |
| **shared-everything-threads** (SET) | true parallelism, reachable from a component | **BUILT, MEASURED, AND `clang -pthread` TARGETS IT** — 27.9x; full guest toolchain shipped 2026-08-07, see "The guest toolchain exists now" below. Not in the Dockerfile chain yet (review gate) |

## UPDATE 2026-08-07 (later the same day): three of those layers turned out to
## be buildable, and I built them. The wall is somewhere else.

An earlier revision of this file said SET "cannot be built from here". That was
wrong about layers 2-4 and I have since built them. Corrected status:

- **Layer 2 (CLI wiring) — BUILT.** See below; the flag was a silent no-op.
- **Layer 3 (shared types) — BUILT for functions.** `WasmSubType::{is,as,unwrap}_func`
  treated `shared` as "not a func", so every SET intrinsic panicked in
  `unwrap_func` the moment its trampoline compiled. `shared` on a FUNC type
  changes neither signature, calling convention nor ABI, so those accessors now
  see through it; `type_registry`'s GC-layout assertions were likewise narrowed
  (a shared func has no GC layout). The GC accessors still assert — there
  `shared` really does change allocation and barriers.
- **Layer 4 (intrinsics) — ONE IS WORKING.** `thread.available_parallelism`
  runs end-to-end through all seven plumbing sites (wasmparser → translate →
  inline → dfg → info → cranelift trampoline → runtime libcall):

      $ wasmtime run -W threads,shared-everything-threads,component-model-threading \
          --invoke 'run()' tools/parallelism-probe/set-available-parallelism.wat
      32
      $ ENCLAVE_AVAILABLE_PARALLELISM=8 ...   ->   8

  It answers from the TENANT's slice, not the node's core count — a guest
  sizing a pool from 32 while holding a 0.25 share would just build 32 threads
  to fight over 8 cores' worth of cgroup weight.

  All of this is in `wasm/wasmtime-set-threads.patch.wip`, deliberately NOT in
  the Dockerfile chain.

- **Layer 4 (intrinsics) — ALL THREE NOW IMPLEMENTED.** `thread.spawn-ref`,
  `thread.spawn-indirect` and `thread.available_parallelism` are real
  trampolines now; previously the two spawn intrinsics `bail!`-ed at
  TRANSLATION time, which meant a SET component could not even be loaded.
  A complete SET guest — shared memory, shared func types, a concrete
  `(ref null $start)`, guest atomics, `thread.spawn-ref` — now loads and runs
  (`tools/parallelism-probe/set-spawn-fallback.wat`):

      $ wasmtime run -W threads,shared-everything-threads,component-model-threading,shared-memory \
          --invoke 'run()' set-spawn-fallback.wat
      32007          # 32 cores * 1000 + 7 units of work completed
      $ ENCLAVE_AVAILABLE_PARALLELISM=4 ...
      4007

  Spawn returns the ABI's documented failure (-1), the guest takes its
  sequential fallback, and the work still happens through shared-memory
  atomics. That is the honest answer on this engine and it is a very
  different situation from "the component is rejected": SET guests are
  loadable, runnable and forward-compatible — the day the engine can really
  spawn, the same binaries get parallelism with no rebuild.

  Getting there also needed two more fixes worth naming: concrete references
  to shared func types had to be interned (otherwise the trampoline's
  `(ref null $start)` panicked with "no entry found for key") and the
  `NegativeTwo` host-result arm had to be implemented in the component
  trampoline — it was a `todo!()`, and it is the only sentinel that lets a
  libcall hand `-1` back to the guest instead of trapping on it.

### What looked like the wall: `thread.spawn-*` needs a thread-safe Store

"Shared everything" means a spawned thread runs in the **same instance** —
same memory, same tables, same globals. In wasmtime, entering guest code
requires `&mut Store`, so it is exclusive *by construction*. This is not a
feature flag; the borrow checker rejects it outright:

    error[E0499]: cannot borrow `store` as mutable more than once at a time

The conclusion drawn at the time was that the remaining work meant making
wasmtime's execution model thread-safe — store bookkeeping, fuel, epoch
interruption, trap handling, stack limits and GC roots all concurrent — the
rearchitecture upstream has not started.

**That conclusion was wrong, and the mistake is worth naming**: it treated
"the spawned thread must share the Store" as part of the requirement. It is
not. The requirement is that the thread shares *instance state* — memory,
tables, globals. The `Store` is where *execution* state lives (stack limits,
last-wasm entry/exit SP+FP, epoch deadline, fuel), and execution state is
exactly the thing each thread must NOT share. Sharing the store is not the
goal; it is the bug the compile error was already warning about.

That reframing is what unlocked the next section.

## SET spawn is real (2026-08-07, final)

`thread.spawn-ref` and `thread.spawn-indirect` now spawn actual OS threads
that run actual guest code in parallel, from inside a component. Measured on
the 32-core workstation, `tools/parallelism-probe/set-spawn-parallel.wat`:

**Constant work PER THREAD** — the shape that cannot be faked. Each thread
runs 900M LCG iterations; `real` stays flat while `user` climbs linearly,
which is only possible if the threads are on different cores:

| threads | real | user | cores busy |
|---|---|---|---|
| 1  | 0.878s | 0.867s | 1.0 |
| 2  | 0.878s | 1.747s | 2.0 |
| 4  | 0.879s | 3.495s | 4.0 |
| 8  | 0.879s | 6.975s | 7.9 |
| 16 | 0.884s | 14.020s | 15.9 |
| 32 | 0.997s | 31.121s | **31.2** |

**Constant TOTAL work** — 14.4 billion iterations split N ways, i.e. the
speedup a real workload sees:

| threads | real | speedup |
|---|---|---|
| 1  | 13.966s | 1.0x |
| 2  | 6.988s  | 2.0x |
| 4  | 3.498s  | 4.0x |
| 8  | 1.752s  | 8.0x |
| 16 | 0.884s  | 15.8x |
| 32 | 0.501s  | **27.9x** |

Best of three runs each, on an otherwise idle box, using a wasmtime built from
`wasm/wasmtime-set-threads.patch.wip` applied to a **fresh checkout** — not the
working tree it was developed in. Measure on a quiet machine: an earlier pass
taken while a compile was running read 21.6x at n=32 purely from CPU
contention.

Linear to 16 (the AMD EPYC 9115's physical core count), then 27.9x at 32
logical cores. Getting ~1.8x out of SMT is more than a pure-ALU loop usually
sees, and the reason is the benchmark's inner loop: an LCG carries a
loop-carried multiply dependency, so each thread spends most cycles waiting on
multiply latency and leaves plenty of issue slots for its sibling. A
throughput-bound or memory-bound guest should expect the flatter curve past
16. `thread.spawn-indirect`
through a shared funcref table is verified functionally by
`set-spawn-indirect.wat`.

Reproduce:

```sh
wasmtime run -W threads,shared-everything-threads,component-model-threading,shared-memory \
  --invoke 'run(16, 900000000)' tools/parallelism-probe/set-spawn-parallel.wat
```

### How: per-thread execution views over shared instance state

The design follows from the reframing above. Each spawned thread gets its
**own `Store`** — so `&mut Store` keeps enforcing execution exclusivity at
every one of those ~320 call sites, unchanged, and the compiler goes on
proving the invariant rather than us asserting it. What the thread shares is
the *instance state*:

- The worker instantiates the **same module** against the **same
  already-resolved import records** the primary was instantiated with
  (captured on the spawning thread, where `&mut StoreOpaque` is legally held).
- Its defined shared memories are then **re-pointed at the primary's
  `SharedMemory`**. That is an `Arc<SharedMemoryInner>` whose
  `VMMemoryDefinition` every instance points at directly, with growth behind a
  lock — wasmtime's own existing mechanism for sharing one memory across
  stores. Linear memory is therefore *physically* the same memory, not a copy.
- Plain (numeric/vector) globals are snapshotted from the primary at spawn.
- The start function is invoked through the **view's own funcref**, so every
  instruction the thread executes enters through the view's vmctx and the
  worker store's `VMStoreContext`.

Traps, backtraces, signal handling and epochs all work unmodified, because
"N threads, N stores, one engine" is already-supported wasmtime usage: signal
handlers are process-global, `sigaltstack` is per-thread and lazily installed
on first wasm entry, and the `CallThreadState` chain the handler consults is
per-thread TLS.

The stub that became real is `thread_spawn` in
`crates/wasmtime/src/runtime/vm/component/libcalls.rs` (now split into
`thread_spawn_ref` / `thread_spawn_indirect`, since the two intrinsics carry
genuinely different payloads and the previous single payload-less trampoline
could not tell them apart); the machinery is
`crates/wasmtime/src/runtime/vm/component/set_threads.rs` (new, ~330 lines
with the design comment).

**Cost note:** each spawn is a full component instantiation. Measured at
roughly 200µs per spawn/join cycle on this box (8000 cycles in 1.6s), so it is
a thread-pool-at-startup mechanism, not something to call per work item.

### Sharedness, stated exactly

Not "shared everything" in the full proposal's sense, and the difference is
enforced rather than hoped for:

- **memory: physically shared.** The thing SET guests actually coordinate
  through.
- **tables: per-worker**, materialized from the same initializers. Cranelift
  rejects `table.atomic.*` upstream ("not yet implemented"), so no guest this
  engine can compile is *able* to express cross-thread table mutation.
- **globals: per-worker, at their INITIAL values.** Nothing is copied from the
  spawner. Under SET an unshared global is per-THREAD state, canonically
  `__stack_pointer`, and inheriting the spawner's value would start a worker on
  the spawner's C shadow stack with both pushing into the same region of shared
  memory. `global.atomic.*` does not compile, and a module with a MUTABLE
  `shared` global is refused at spawn rather than silently diverging. (An
  earlier revision snapshotted plain `shared` globals into a worker; that code
  is gone along with the design that needed it.)
- **WASI: per-worker.** Component resource handles are per-instance, so a
  handle cached in shared memory names nothing in another thread; the libc
  keeps its descriptor table, stdio streams and preopens in TLS to match, and
  an fd carries the identity of the thread that owns it, so a cross-thread fd
  FAILS with `EBADF` instead of aliasing a different file of the same number.

Thread CREATION is rate-limited as well as capped, because those are different
quantities: the cap bounds how many threads exist at once, and a guest that
spawns and immediately exits never approaches it while still costing the node a
thread creation per iteration. See `max_spawn_rate` in `set_threads.rs`.

Spawn returns the ABI's `-1` ("spawn failed", which guests are required to
handle) with a rate-limited stderr diagnostic for any module shape whose worker
would not faithfully share state: a non-shared defined memory, NO shared memory
at all, an imported table/global/memory/tag, a mutable shared global, segmented
memory initialization, a table with no declared maximum, or GC use. Guests take
their sequential fallback. This is the fail-closed posture the attestation
requires: a shape we cannot represent honestly gets refused, never approximated.

### Stopping a worker: three paths, because each alone has a hole

A SET worker is an OS thread running guest code. Making it *stoppable* is the
part that took three review rounds to get right, and the reason is that a
thread can be in exactly one of three places, none of which the other two
reach:

1. **Running compiled code.** Stopped at an EPOCH CHECK. The worker's store
   takes a small epoch deadline whose callback reads the group's stop flag and
   either traps or re-arms. Note what this is NOT: a run-time budget. An
   earlier revision gave workers a deadline of `ENCLAVE_SET_EPOCH_TICKS` (600)
   and trapped on it, which killed healthy long-running workers AND never fired
   for the runaway one — because `wasmtime --wasm timeout` bumps the engine
   epoch exactly ONCE, enough for a deadline of 1 and nothing larger. Measured
   both ways with `-W timeout=2s`: the default hung at exit 124, and
   `ENCLAVE_SET_EPOCH_TICKS=1` exited at 2.0s.
2. **Parked in `memory.atomic.wait`** (every SET mutex and every
   `pthread_join`). Stopped by the PARKING SPOT polling the same flag on a
   bounded quantum. The flag now lives on the shared memory's parking spot, not
   only in a worker thread-local, so the thread that most needs waking — a MAIN
   thread parked in `pthread_join` after a worker exited — is reached too.
3. **Blocked in a HOST call.** Reaches no epoch check and is in no parking
   spot. Stopped by the embedder DROPPING the worker's guest future, which is
   wasmtime's own cancellation path (`FiberFuture::drop` disposes the fiber).
   Measured before this existed: 12 s of guest-controlled host block against a
   1 s embedder timeout, at essentially zero CPU.

**Epoch interruption is therefore a requirement, not a tuning knob.** Without
`Config::epoch_interruption` the compiled code carries no epoch checks at all,
and without a PERIODIC ticker nothing advances the epoch. The wasmtime CLI
turns both on automatically whenever `-W shared-everything-threads` is passed
(10 ms period); any other embedder must, and `thread.spawn` says so on stderr
the first time it is asked on an engine that has not.

The measured cost of that instrumentation, on the constant-total-work
benchmark: ~0% at 1 thread, a few percent at 16, and ~12% at 32 (where SMT
siblings compete for issue slots and the extra loop-backedge check is no longer
free). 15.8x became **14.9x** and 27.9x became **24.9x**. That is the price of a
worker that can be stopped, and the earlier numbers were measured on an engine
where it could not be.

### The soundness invariants, and how each is held

A worker holds **no** pointers into the store that spawned it: its component is
a refcounted handle, its shared memories are `Arc`s, and its own store owns
everything else — `SetWorkerRequest` is `Send` without a single `unsafe impl`,
and the compiler proves it. Two things follow:

1. **Teardown does not have to block, and must not.** `Drop for Store<T>` asks
   the group to stop, drives all three stop paths, waits on the group's live
   count with a bound (`ENCLAVE_SET_JOIN_TIMEOUT_MS`, default 2s), and DETACHES
   a straggler with a loud diagnostic rather than wedging the process. The old
   unbounded join was inherited from the revision where workers really did
   borrow the primary's import records; keeping it afterwards meant any
   non-returning worker wedged teardown forever and pinned a core — and an
   ORDINARY program reaches that state, by detaching a compute thread and
   returning from `main` (`tools/parallelism-probe/worker-spin-teardown.c`).
   A detached worker keeps its process-wide cap slot until it really exits, so
   leaking them costs the ability to spawn rather than accumulating silently.
2. **No re-entry.** A worker must never call back into the primary store.
   Component intrinsics and canon-lowered host imports all funnel through
   `ComponentInstance::enter_host_from_wasm`, which now compares the
   component's `VMStoreContext` pointer against the one this thread is
   currently executing. On a mismatch it records a trap **on the worker's own
   thread** and returns the ABI's unwind sentinel, so the worker unwinds
   through its own store and dies alone with a clear message:

       set-thread-1: error while executing at wasm backtrace:
           0: 0x6d - m!worker
       Caused by:
           a shared-everything-threads worker cannot call back into the
           component that spawned it (nested `thread.spawn-*`, ...)

   This matters because there is one realistic way to hit it: **nested
   spawn**, a worker spawning another thread — perfectly reasonable guest code
   (thread pools do it). The first version of this guard aborted the process,
   which was wrong: legitimate guest code should get a trap, not take the host
   down. Verified by `tools/parallelism-probe/set-nested-spawn.wat`.
   Aborting is retained only for the residual case where the libcall's return
   type has no unwind sentinel (a libcall that structurally cannot trap, none
   of which should be reachable this way), because then there is nothing
   truthful to hand back.

   Making this trap needed one new piece of machinery:
   `HostResult::unwind_sentinel()`, so a caller can signal a trap *without
   holding the store* — which is the whole point, since the guard fires
   exactly when touching that store would be the race.

`unsafe impl Sync for Store` appears nowhere. That was the constraint and it
held: the borrow checker still refuses two threads on one store, and it still
would if this code were wrong.

### Evidence beyond "it went fast"

- **ThreadSanitizer**: `-Zsanitizer=thread` build with `-Zbuild-std`, all
  four SET probes clean (spawn-parallel; spawn-indirect; a stress probe doing
  800 spawn/join cycles with contended atomics plus cross-thread
  `memory.grow`; and nested-spawn, which must trap rather than race). The toolchain was verified to actually report races by
  running a known-racy program through it first — a clean TSan run means
  nothing until you have proved the instrumentation is live.
  **Honest scope**: TSan instruments the *host runtime*, not JIT-compiled
  guest code, so it validates instance creation, store lifetimes and the
  `SharedMemory` growth lock — not guest-level accesses.
- **Soak**: 8000 spawn/join cycles (500 rounds x 16 threads) with a contended
  atomic accumulator and cross-thread `memory.grow`, all completions
  accounted for, no hangs.
- **Test suite**: 187 `wasmtime` unit tests and 238 component-model/threads
  integration tests pass.
- **Fuzzing**: `component_api` (2048 execs, 54.8k coverage points, no crashes)
  and `instantiate`. Modest exec counts — component generation is slow — so
  read this as a smoke test, not a campaign. `instantiate` does find a crash — it reproduces
  **identically on the SET-free baseline**, so it is pre-existing: the fuzz
  harness at `crates/fuzzing/src/generators/config.rs:470` unwraps a
  `to_store()` that legitimately fails when a generated config's GC heap
  exceeds the pooling allocator's memory limit. Worth reporting upstream;
  not ours. (Always re-run a fuzz crash against the unpatched baseline before
  believing it is yours — and note the fuzz targets need
  `git submodule update --init`, plus `--no-default-features` unless you have
  OCaml installed.)
- Regression-checked: `thread.available_parallelism` still answers 32 (and 8
  under `ENCLAVE_AVAILABLE_PARALLELISM=8`), plain core modules unaffected.
- **Adversarial review, 2026-08-07** — four independent readers plus a
  hostile-guest probe. It found real UB (a worker could re-enter the primary
  store through an imported memory's libcall, because the *core*
  `enter_host_from_wasm` was unguarded) and four other critical defects. All
  fixed; see `docs/HANDOFF-set-threads.md` for the list and for what is
  known-and-accepted rather than fixed. The lesson worth keeping: the bug that
  mattered was found by writing a hostile guest and by fresh readers, not by
  re-reading code I had just written.

### Engine changes beyond the spawn path

Getting a valid SET guest to *load* needed four more fixes past what the
earlier session had landed:

- Two further `assert!(!ty.composite_type.shared)` sites in
  `module_types.rs`'s `UnpackedIndex::Module` arm (the earlier session fixed
  the `Id` arm). A shared funcref table's element type reaches both.
- `(shared func)` as an ABSTRACT heap type was `wasm_unsupported!`. A funcref
  is a pointer whether or not it is shared, so it now maps to `WasmHeapType::Func`.
  Only FUNC gets this — shared extern and shared GC heap types really do change
  allocation and barriers, and stay unsupported.
- The `shared` bit on globals was parsed and silently dropped; it is now
  carried on `environ::Global` so the spawn guard can see it.
- `has_guest_start` on `environ::Module`, because `startup` conflates a guest
  `(start)` section with wasmtime-synthesized initialization, and the view
  must run the latter while refusing the former.

### One real bug fixed on the way

The earlier session's `NegativeTwo` trampoline arm was paired with a
`Result<u32>` libcall. `u32`'s unwind sentinel is `u64::MAX` (i.e. -1), but
the trampoline tests for -2 — so a host-side trap would have been passed
through to the guest as a *plausible-looking* "spawn failed" while leaving the
recorded unwind pending in TLS. Unreachable while the body was `Ok(-1)` on
every path; live the moment spawn does real work. Fixed with a
`ThreadSpawnResult` newtype whose sentinel is `u64::MAX - 1`, matching the
trampoline.

## The guest toolchain exists now (2026-08-07): `clang -pthread` targets SET

The wall that was "no SET guest toolchain in any language" is down. A C or C++
program built with `-pthread` now produces a component whose threads run on
real cores, with no hand-written WAT. Three pieces, all in the repo:

1. **wasi-libc SET thread model** (`wasm/wasi-libc-set-threads.patch`): a new
   `ENABLE_SET_THREADS` flavor built for `wasm32-wasip2`. It is the
   wasi-threads (posix) pthread implementation — real threads, shared memory,
   `memory.atomic` futexes — with two swaps for the shared-everything world:

   - **Spawn is not a host import.** wasi-threads calls the host's
     `wasi.thread-spawn`; the component model has no such import. Instead
     `pthread_create` calls through a VOLATILE function pointer
     (`__enclave_set_spawn_fp`) that defaults to an in-module stub returning
     -1. The componentizer (below) patches the real `thread.spawn-indirect`
     builtin into that pointer's table slot. A component that skips
     componentization still links and runs — single-threaded, because the stub
     answers -1 and `pthread_create` reports EAGAIN.
   - **Tids are libc-assigned, not host-assigned.** The SET spawn ABI carries a
     single i32 context, so the host cannot hand the child a tid. The libc
     assigns it before spawning and passes it in `start_args` (read by
     `wasi_set_thread_start.s`), using the engine's spawn return only as a
     success/failure signal.

2. **set-componentize** (`wasm/set-componentize/`): after `wasm-component-ld`
   links the core module into a component, this APPENDS the two canonical
   builtins no linker can express — `thread.spawn-indirect` over the module's
   `__indirect_function_table`, and `thread.available_parallelism` — plus a
   static fixup module that `table.set`s them into the libc's spawn/ap slots.
   It only appends, so no index inside the linked module moves.

3. **The blessed image** `wasm/Dockerfile.wasipsetc-build`: wasi-sdk 34-rc.2
   clang 23 + the SET-flavored libc + set-componentize, wrapped so
   `docker run … app.c -o app.wasm` yields a wired component directly. Its
   in-image smoke test asserts the output is a layer-1 component carrying the
   `[set-spawn-indirect]` marker the publish path stamps `set: true` from.

Measured end-to-end (a real `pthread_create`/`pthread_join` C program through
the whole chain onto the SET engine): `user` CPU scales linearly with thread
count while `real` stays flat — 15.8x on 16 cores, the same curve the
hand-written WAT probe showed.

### The three engine changes the toolchain forced

Real toolchain output is not hand-tuned WAT, and three engine assumptions had
to give (all in `wasm/wasmtime-set-threads.patch.wip`, and a small
`wasm/wasmparser-set-relax.patch` fork):

- **The validator accepts a PLAIN (unshared) spawn type and a plain funcref
  table.** `clang` emits an ordinary `(func (param i32))` start function and an
  ordinary `__indirect_function_table`. Upstream wasmparser requires the spawn
  type and table to be `shared`, because in a one-instance SET implementation a
  plain start function would race on unshared state. This engine gives every
  worker its own execution VIEW, so unshared state is per-thread by
  construction — the relaxation is sound *here* and nowhere the view model
  doesn't hold. The synthesized spawn import's sharedness now follows the spawn
  type so a plain-typed guest links against a plain-typed import.
- **The canonical ABI accepts a SHARED memory.** A threaded component's only
  memory is shared; refusing shared-memory canonical lowerings would mean no
  threaded component could call or export anything. Every canon trampoline is
  main-thread-only (a worker hitting one traps via the cross-thread entry
  guard), and wasmtime reserves shared memories to their declared max up front,
  so the base a trampoline copies through never moves under concurrent growth.
- **Worker views stub every import instead of copying the primary's.** The
  earlier design copied the primary store's resolved import records and
  allow-listed component trampolines — which carried raw cross-store pointers
  into the worker and, worse, broke on real wit-component output (whose shim
  pattern makes the main module import plain functions of sibling instances).
  Now `run_view` satisfies every function import with a stub host function in
  the WORKER's own store that traps ("host APIs and sibling core instances are
  main-thread-only"). `SpawnPayload` consequently holds NO raw pointers into
  the primary and is `Send` without a single `unsafe impl`. A guest start
  section is likewise now ALLOWED and runs once per view — `wasm-ld
  --shared-memory` emits `__wasm_init_memory` behind an atomic once-flag in
  shared memory for exactly this instantiation-per-thread pattern.

### 2026-08-10: the fork has a SECOND consumer — the upload gateway

Forking the validator forked it for everyone who validates, and the publish
path was still on stock. The first real SET app (risc-box) could not be
uploaded at all:

```
upload rejected: wasm validation failed:
error: mismatch in the shared flag for memories (at offset 0x30b8d2)
```

That is the `cabi_memory_at` relaxation above, seen from the other side: the
gateway's Tier 2 check (`scripts/ipfs-add-gateway.py`, `WASM_TOOLS`) ran an
upstream `wasm-tools`, whose `cabi_memory_at` compares the component's shared
canonical-ABI memory against a hardcoded `shared: false`. Nothing was wrong with
the component — the engine launches it.

The general shape: **Tier 2 is only meaningful as a preview of the engine, so it
has to be the same validator.** Stricter than the engine = false refusals at
publish; looser = pins that will never launch. `scripts/build-wasm-tools-set.sh`
builds `wasm-tools` at the tag whose `crates/wasmparser` IS the engine's pinned
crate (it diffs the git tag against the sha-pinned crates.io tarball and aborts
if they ever diverge), applies the same `wasmparser-set-relax.patch`, and emits
a static musl binary; `scripts/deploy-wasm-tools-set.sh` installs it and points
`WASM_TOOLS` at it. Both must be re-run whenever the patch or
`WASMPARSER_VERSION` in `wasm/Dockerfile.wasmtime` moves.

Worth generalising past wasm-tools: an engine fork silently creates a
consistency obligation everywhere else the same parser runs. Grep for the other
copies before shipping the next one.

## The 2026-08-07 adversarial review — and why the engine is NOT fleet-ready

Four independent adversarial reviewers (fresh context, each told to REFUTE, plus
hostile guests) went at the toolchain-forced engine changes. This is the same
process that found real UB last time, and it earned its keep again: **one
CRITICAL, two HIGH, one MEDIUM.** Two are fixed; two are the reason the engine
stays out of the Dockerfile chain. The lesson holds — a change that needs this
many fixes on first serious read must soak, and nothing on the fleet waits on
it. Repro files were under a scratch dir; the mechanisms are recorded here.

**FIXED — teardown deadlock (HIGH).** A SET worker parked in
`memory.atomic.wait` (the primitive under every SET mutex) is invisible to
epoch interruption, and `Store::drop` joined workers unconditionally — so a
worker parked forever hung teardown forever, reachable by ordinary guest code (a
lock handoff whose releaser traps before it notifies), and under `wasmtime
serve` it leaked a tokio worker per request and pinned the process-global thread
slot. The comment claiming epochs rescued this was simply false. Fixed the way
the review's own key insight pointed: because a worker now holds NO pointers
into the primary store (stubbed imports, `Arc<SharedMemory>`, by-value globals),
the blocking join is not required for safety. Teardown now sets a per-worker
cancellation flag (`parking_spot`) and unparks each worker; a parked futex wakes
on a bounded poll, raises an interrupt trap, and unwinds out of its own store,
freeing its slot. Non-worker threads never install the flag, so their wait path
is byte-identical. Verified: the two hang repros now exit cleanly.

**FIXED — active-data re-init clobber (MEDIUM).** For a module with SEGMENTED
memory initialization (active data segments — offset via `global.get`,
extended-const arithmetic, or a >16 MiB sparse span), wasmtime's synthesized
startup re-applies the segments via a compiled `memory.init` on every spawn,
into the now-swapped-in SHARED memory, clobbering data the primary and live
siblings already wrote. The once-flag reasoning only covers `wasm-ld`'s PASSIVE
segments; active segments were an unguarded path. Fixed fail-closed: spawn now
refuses a module whose `memory_initialization.is_segmented()`. `wasm-ld
--shared-memory` output is not segmented, so the real toolchain is unaffected
(verified: `scale.wasm` still threads); a hand-written segmented guest takes the
sequential fallback instead of silent corruption. (A hand-written guest `(start)`
that scribbles shared memory without a once-flag still re-runs per view — that
one is in-sandbox, self-inflicted, and the same contract wasi-threads imposes;
left as a documented residual.)

### 2026-08-08: both blockers fixed — one component instance per thread

The two blockers really were one problem, and the fix is a single idea: **a
spawned SET thread gets a full instantiation of the whole component in its own
`Store`, and shares nothing with its spawner except linear memory.**

*Execution.* `SetWorkerRequest::run_async` (in `set_threads.rs`) instantiates
the same `Component` in the worker's store, with the spawning core instance's
defined memories re-pointed at the primary's `SharedMemory` before that
instance's startup function runs (`SetViewPlan`, threaded through the component
`Instantiator`). The start function is then named component-relatively, as
`(RuntimeInstanceIndex, FuncIndex)`, so it resolves in any instantiation, and
entered with the same `enter_guest_sync_call` bookkeeping the primary uses for a
core `start`. Every direct call Cranelift baked in now lands on code whose vmctx
is the worker's own, which is what makes it sound. Building the store needs the
embedder's `T`, so the embedder registers a `SetWorkerHost`
(`Store::set_set_worker_host`); the CLI does this in both `run` and `serve`, and
spawn REFUSES with the ABI's `-1` when no host is registered rather than
half-building a thread.

*WASI.* Because each thread has its own component instance, it has its own
resource tables — and component resource handles are per-instance, so a handle
cached in shared memory names nothing in another thread. The libc follows:
the descriptor table, the lazily-cached stdio streams and the preopen table are
now `_Thread_local` (`__wasilibc_thread_state`), and each thread lazily rebuilds
its own from its own instance's imports. Threads share linear memory, which is
what pthreads needs; they do NOT share a descriptor table, which is the same
honest limitation the wasi-threads (p1) path documents.

*Dying threads.* A thread that traps skips the entire libc epilogue, so joiners
used to park forever on `detach_state` and then on `__thread_list_lock` — in
practice hanging the process, since the main thread is usually the one in
`pthread_join`. A new optional guest export `__enclave_set_thread_died` runs the
parts of that epilogue other threads depend on, and the engine calls it on the
worker's own store after a trap. `pthread_join` on a trapped worker now returns
`PTHREAD_CANCELED` instead of hanging.

*Canonical ABI.* The host no longer forms a Rust slice over guest memory that
could be `shared`. `component/guest_memory.rs` introduces `GuestMemory` /
`GuestMemoryMut`: for a shared memory, bytes are copied out through
`read_volatile` into host-owned buffers and validated **on the copy**, so a
guest cannot invalidate a `str` the host has already checked; for every other
component the path is the same zero-copy borrow as before (`Cow::Borrowed`), by
design, because that is the path that has to stay fast. Sharedness is recorded
where `extract_memory` already distinguishes the two cases. Two paths that
cannot be made copy-safe fail closed instead of racing: the async/streams
machinery (`options_memory_unshared`), and fused-adapter string transcoders,
which are refused at instantiation via a new compile-time `transcoder_memories`
list. Neither is reachable from a `clang -pthread` component.

#### ThreadSanitizer: read the reports before believing them

The first TSan run on the new engine reported data races — all of them inside
the fiber machinery (`Fiber::new` racing `Fiber::resume`), on the same address,
between two different worker threads. It is a false positive, and the way that
was established is worth keeping, because the same trap will be laid again:

1. **Separate reuse from concurrency.** `run(200, 1, 2000)` runs 200 workers
   ONE AT A TIME — no two ever run concurrently, so no data race is possible —
   and reported 12 races. `run(1, 8, 2000)` runs 8 workers concurrently with no
   stack reuse and reported 0. The reports track *address reuse*, not
   concurrency.
2. **Find the mechanism.** Fiber stacks are `mmap`ed and freed by
   `rustix::mm::munmap` (`crates/fiber/src/unix.rs`). On Linux rustix defaults
   to its `linux_raw` backend — direct syscalls, which TSan cannot intercept —
   so TSan never clears its shadow memory for a freed stack. The next `mmap`
   handing back the same address looks like a race with the thread that died.
3. **Prove it.** Rebuilding with `--cfg rustix_use_libc` so `munmap` goes
   through libc and TSan can see it: **every report disappears.**

```sh
CARGO_TARGET_DIR=target-tsan RUSTFLAGS="-Zsanitizer=thread --cfg rustix_use_libc" \
  cargo +nightly build -Zbuild-std --target x86_64-unknown-linux-gnu --release -p wasmtime-cli
```

With that binary: `run(50, 8, 2000)` = 400, zero races; `set-spawn-indirect`,
`set-spawn-parallel`, `set-worker-import`, `set-nested-spawn` and
`set-cabi-race` all zero races; and the R4 harness at 20k lifts against 30.9M
guest flips is clean too. **Do not run TSan on this engine without
`--cfg rustix_use_libc`** — the results are noise, and the temptation to wave
them away as "just fibers" is exactly how a real report would get missed.

Measured after the change: `set-spawn-parallel.wat` 14.08s → 0.90s at 16
threads (**15.6x**, `user` 14.1s against `real` 0.90s), soak
`run(500,16,2000)` = 8000, 187 wasmtime unit tests, 648 enclave tests. Worker
`printf`/`fflush`, `clock_gettime` and `socket()` all work; a trapping worker no
longer hangs its joiner. The R4 race is now a deterministic harness
(`tools/parallelism-probe/set-cabi-race.wat` + `cabi_race.rs`): a worker flips a
string between valid and invalid utf8 while the host lifts it 200k times, and
the harness checks whether the host was handed a borrow into shared memory. It
reports zero borrows and zero invalid `str`s — and, run against a deliberately
reverted borrowing ABI, it reports 90 borrows and catches an actually-invalid
`str`, which is what proves the harness detects rather than merely passing.

### 2026-08-08 (round 3): stoppable workers, and the RAM gate that did not bind

The four-reviewer pass on the round-2 design found **1 CRITICAL + 15 HIGH**.
Every one of them is fixed here; the ones worth carrying forward as knowledge:

* **A guest could abort the host process.** `wasmtime run` built a worker's
  embedder context by hand, with only the wasip1 ctx and the store limits set,
  while the worker instantiated against the PRIMARY's linker — whose accessors
  `unwrap()` `wasi_nn_wit` / `wasi_http` / `wasi_config` / `wasi_keyvalue` /
  `wasi_tls`. A worker touching any of them panicked, and `worker_main` turned
  a panic into `std::process::abort()`. Both halves are fixed: `build_host` is
  now the ONLY place a CLI `Host` is built, for the primary and every worker
  alike, and a panic on a worker kills that worker exactly like a trap does.
  The second half is regression-tested in
  `tests/all/component_model/set_threads.rs` — a test whose failure mode is
  that the test binary dies rather than reporting.
* **Teardown could not stop a worker, and an ordinary program hit it.**
  Rewritten around a `SetThreadGroup` and three stop paths; see "Stopping a
  worker" above for why one path is never enough. `Store::drop` now bounds its
  wait and detaches, which is sound because a worker holds nothing of the
  store that spawned it.
* **`-W max-memory-size` did not bind shared memory at all.** Upstream's
  `SharedMemory::grow` passes no `ResourceLimiter`, so growth escaped it —
  ~124 MB reached under a 16 MiB cap, even single-threaded. On this platform
  that flag IS the tenant's purchased RAM ceiling and the SET toolchain links
  every guest with `--max-memory=1073741824`, so a SET tenant could have grown
  to 1 GiB whatever it bought. The limiter is now consulted on every shared
  grow, before the write lock (it may await) with the approved size re-checked
  under it, so a racing pair cannot exceed the largest approved size. Verified:
  `worker-mem-grow.c` stops at exactly 16777216 bytes on both threads under
  `-W max-memory-size=16777216`. **This was the open question for the platform
  embedder; it is now closed, and it was a real hole.**
* **A worker's `exit()` wedged the component.** `proc_exit` on a worker
  unwinds only that worker's store, so the atexit handlers ran (poisoning every
  FILE lock via `__stdio_exit`), the status vanished, and the main thread hung
  in `pthread_join`. The engine now carries the status in the group and stops
  the rest of it; the embedder recognises its own exit error through a new
  `SetWorkerHost::exit_status`, because `wasmtime` cannot name
  `wasmtime_wasi::I32Exit`. Note that the *status* is capped by wasip2:
  `wasi:cli/exit` carries success/failure, not a code.
* **Cross-thread fds ALIASED rather than failing.** Both threads' tables
  allocated the lowest free index while musl's `FILE` objects stayed shared, so
  a worker's fd 4 and main's fd 4 were different files with the same name.
  Reproduced by writing a worker's buffered secret into main's file with every
  call returning success. An fd now carries its owner's namespace slot
  (`worker-fd-alias.c`: worker gets 4194308, main gets 4, cross-thread write
  gives `EBADF`).
* **A worker trapping inside `printf` wedged stdio for every thread.** musl
  registers a FILE on `stdio_locks` only from the EXPLICIT locking API, never
  from the internal `FLOCK` path that `printf` takes, so the orphan sweep
  walked a list the FILE was never on. `__lockfile`/`__unlockfile` now register
  and unregister under SET (and `flockfile` correspondingly does not, or the
  list becomes a cycle).
* **The fused-adapter refusal was unreachable AND mis-placed.** It ran after
  the initializer loop — but core `start` sections run inside that loop — and
  it could not fire anyway, because FACT emits a shared adapter memory with
  `maximum: None`, which is invalid wasm, so such a component panicked in
  `.expect("invalid adapter module generated")` first: a guest-controlled panic
  inside `Component::new`. It is now a clean compile-time refusal in
  `partition_adapter_modules`, covering EVERY adapter that needs memory rather
  than only string transcoders.
* **`wasmtime serve` could not run a SET guest at all**, and no reviewer found
  it — verification did. `serve` defaults to the POOLING allocator, which
  cannot allocate a `shared` memory, so every SET component failed to load with
  "memory is shared which is not supported in the pooling allocator". `run` was
  unaffected, which is why every probe missed it. The default is now off when
  `-W shared-everything-threads` is on. End-to-end serving of a SET http guest
  is still unproven: the blessed toolchain image builds `wasi:cli` commands and
  carries no C `wasi:http` binding generator, so no such guest exists yet.
* Also: per-store limits and fuel are per-WORKER (documented, and `--wasm
  fuel=N` no longer hangs — `run` used to give workers zero, so they trapped
  before reaching the libc epilogue that releases a joiner); the death hook
  gets a protected epoch budget, without which it trapped at its own entry on
  exactly the path that needed it; the refusal diagnostic is time-rate-limited
  per site instead of `Once` per process; per-thread libc state is reclaimed at
  thread exit; and `SetViewPlan::install`'s preconditions are real checks.

### 2026-08-08 (round 4): the cap counted the wrong thing, and other lessons

A fourth adversarial pass on the round-3 design found 3 HIGH plus a
resource-exhaustion escape. Every round so far has found real defects, which is
the single most important fact about this project. The generalisable lessons:

**A cap on how many exist is not a cap on how many are made.** The live-thread
cap increments on spawn and decrements on exit, so `worker(){ spawn(worker);
return; }` keeps the live count at 1-2 forever while creating threads as fast as
the kernel allows: 35,187 create+exit pairs in 2 s, 2.6 s of CPU, and
`ENCLAVE_MAX_SET_THREADS=4` changed nothing. Whenever a resource is "capped",
ask which of *stock* and *flow* the cap measures.

The token bucket this round added (`ENCLAVE_MAX_SET_SPAWN_RATE`) is, four
rounds later, **off by default** — see round 8 below. Both implementations of it
made things worse, and the cgroup was already charging the tenant for the
creation cost.

**A recovery window is a renewable resource unless you say otherwise.** The
guest's thread-death hook necessarily runs with every stop path disarmed. It had
a 10-second budget, and it runs on the worker's own store — which has the worker
host and the thread group installed — so a hook that spawns a thread that traps
gets another hook, forever. The budget is now 0.2 s AND spawn refuses once the
group is stopping.

**A repro that exercises the wrong mechanism is worse than none.**
`worker-block-teardown.c` blocks on a timer, which is asynchronous and therefore
cancellable — so it "proved" a stop path that did not work for the blocking
FILESYSTEM calls the platform actually runs, because `allow_blocking_current
_thread` (on whenever `--wasm timeout` is absent, i.e. always on-fleet) runs
those inline on the fiber where nothing can cancel them.

**Recycling an identity re-creates the aliasing you removed.** Round 3 tagged
each fd with its owning thread. The tag came from a recycled slot bitmap, and
recycling is deterministic, so a dead thread's fd became valid again for the
next worker — naming a different file. Monotonic ids, always.

**And: a fix can make a latent bug reachable.** Giving the death hook a
protected epoch budget is what made it possible for the hook to run at all when
the trap happened before `wasi_set_thread_start` had installed TLS — at which
point `__pthread_self()` is the MAIN thread's `struct pthread`, and the epilogue
zeroes its tid (breaking `__tl_lock` and every FILE lock) and can drive
`threads_minus_1` to zero, which sets `libc.need_locks = -1` and turns every
lock in the process into a no-op.

The full per-finding list, with the repro for each, is in
`wasm/SET-REVIEW-HISTORY.md`.

**One claim retracted:** nested spawn does NOT trap, and never should have been
documented as doing so. A worker runs its own whole component instantiation, so
`thread.spawn-*` from it enters through the worker's own vmctx and the
cross-thread guard correctly sees a match. Nested spawn is supported — the
design installs the host and the group on a worker's store precisely to make it
work. The guard's real job is a worker reaching the PRIMARY's store through an
imported memory's `memory.grow`/futex, which spawn also refuses outright.

### 2026-08-08: the CRITICAL blocker's real cause is vmctx type confusion

The symptom below (`call stack exhausted` on a worker's first import) was
diagnosed on 2026-08-08 and the cause is not what the text predicted. It is not
stacks, and not a missing async/fiber context — driving the worker's entry
through `call_async` changes nothing. **Stubbing a view's imports does not work
at all for a component's core modules.**

Cranelift devirtualizes a call to a statically-known import into a DIRECT call
to the callee's compiled body, while still loading the callee vmctx out of the
import slot (`crates/cranelift/src/func_environ.rs`, the
`KnownFunc::FuncKey(FuncKey::DefinedWasmFunction(..))` arm — "The import is
always satisfied with the given defined Wasm function, so do a direct call to
that function!"). Substituting an import at instantiation time cannot redirect
a call address that was baked in at compile time. So a SET worker runs the
PRIMARY's compiled code with the stub's `VMArrayCallHostFuncContext` as its
vmctx: wasm executing against a vmctx of an entirely different type, reading
memory bases, globals and `stack_limit` out of a host-function allocation.
`call stack exhausted` is simply what that produces when the callee's prologue
reads a `stack_limit` that is not one.

Proven, not inferred — `tools/parallelism-probe/set-worker-import-foreign.wat`:
the view's import slot really does hold the stub (vmctx magic `ACHF`), the stub
closure is never entered, and the `i32.div_s` by zero written into the
*primary's* imported function traps on the worker thread. With an empty
imported function the same probe returns a clean result, which is why every
earlier probe missed this: they had no imports, and a zero-stack leaf callee
skips the prologue check that turns the confusion into a visible trap.

This raises the severity. The engine is not merely "sound for pure-compute
workers"; a worker in any component whose core module imports a sibling's
function — which is everything `wasm-component-ld` emits — performs wild
accesses through a mistyped vmctx. Import stubbing has to be abandoned rather
than repaired: the fix is a full component instantiation per worker, in the
worker's own store, so that every statically-known direct call lands on code
whose vmctx is the worker's own.

**BLOCKER — worker threads cannot make component/WASI calls (CRITICAL).** A
worker execution view can recurse thousands of frames in pure wasm, but its
FIRST canon-lowered import call (stdout, clock, sockets — any WASI) traps `call
stack exhausted`. So a worker that does I/O fails; only the main thread can.
Every real threaded program does I/O on workers, so today's toolchain is sound
only for **pure-compute workers** (parallel number-crunching — exactly what the
benchmarks are, which is why this hid). The libc compounds it: its thread
epilogue (release `__thread_list_lock`, set `detach_state`) runs only on the
normal return path, so a trapped worker used to hang a joining sibling or vanish
with its result unwritten — the teardown-cancellation fix above turns the hang
into a clean exit, but the worker's I/O is still lost. The real fix is engine
work: set up each spawned view's component-call execution context (the
async/fiber + reentrance state a core→component transition needs) so worker→host
calls don't spuriously exhaust. Until then, do not advertise real-core threads
for apps whose workers touch WASI.

**BLOCKER — shared canonical-ABI memory is a host-TCB data race (HIGH).** To let
a threaded component (whose only memory is `shared`) use the canonical ABI at
all, the validator was relaxed to accept a shared cabi memory (R4 in the
wasmparser fork). But the host's canon lift/lower borrows guest memory as Rust
`&[u8]`/`&mut [u8]` and validates-then-copies — so a hostile guest that races a
worker's writes against a main-thread canon lift produces an invalid Rust
`String` (a violated library invariant = UB) and a genuine data race in the host
TCB. "Base never moves" (true — shared memories are reserved to max) rules out
OOB/UAF but NOT the race on the contents, which was the crux the original
justification missed. Upstream refuses shared cabi memory for exactly this
reason. The correct fix is a copy-safe canonical ABI for shared memory (eagerly
copy the accessed bytes out through atomic/volatile reads into host-owned
buffers and validate the copy, per wasmtime's own `Memory::data` doctrine) —
a substantial, ~25-site change to the canonical lift/lower machinery that is its
own measured-TCB project. R4 is kept in the out-of-chain `.wip` because without
it the toolchain's components cannot do ANY canonical I/O (even main-thread
`printf`), so it is required for the local/benchmark demonstration and is
sound for the trusted guests that use it there — but it MUST NOT enter the fleet
chain until the copy-safe path exists. This blocker and the worker-import one
above both point at the same missing piece: a fully correct component execution
model for shared-memory threads.

### What is still NOT done

- **Not in the Dockerfile patch chain until a fresh adversarial pass clears.**
  `wasm/wasmtime-set-threads.patch.wip` stays out of `wasm/Dockerfile.wasmtime`.
  The blockers from rounds 1-3 are fixed and the verification bar is met; the
  engine enters the measured TCB only after a four-reviewer pass on THIS design,
  which has caught real UB every single time it has been run. The platform `set`
  capability plumbing IS wired and tested (probe → `[set-spawn-indirect]` marker
  → publish stamp → claim gate → per-tenant `-W` flag → fleet-AND, generalising
  the coop `coopThreads` shape; tests in `test/wasm-set.test.mjs`), and inert
  until the repin, exactly like the coop capability was.
- **Per-store limits are per-WORKER, and that is a multiplication.**
  `max-instances`, `max-table-elements`, `max-resources` and `--wasm fuel` are
  enforced on each worker's own store, so a group of N workers may use up to
  (1 + N) times the configured amount, bounded by the live-thread cap. Linear
  MEMORY is the exception and the one that matters on this platform: the shared
  memory every SET thread actually works in is bound once, by
  `-W max-memory-size`, from whichever thread grows it.
- **The libc leaks a trapped thread's descriptor table.** A thread that exits
  normally reclaims it; a thread that TRAPS releases only its fd-namespace slot,
  because dropping a resource handle is a component call and a thread that has
  just trapped should not be making more. Bounded by the live-thread cap.
- **Rust is still gated on the same LLVM-23 event as coop threads.** The SET
  libc is C-buildable today (wasi-sdk 34's clang 23); `rustc` still emits
  LLVM-22 codegen, so `std::thread` over SET waits on the retirement event
  documented in `Dockerfile.wasip3-build`.
- Cross-instance spawn, mutable shared globals, and shared tables are refused
  rather than implemented. Each needs its own design; none is needed by the
  workloads that motivated this.
- `set-spawn-fallback.wat` now really spawns, so its counter races by design
  (it was written when spawn could only fail). It is kept as a load-and-run
  probe; `set-spawn-parallel.wat` is the one that joins properly.

## The layer map (measured, not assumed)

1. **Spec / validator — EXISTS.** `wasm-tools validate --features
   shared-everything-threads` accepts shared globals, shared composite types,
   and a component carrying shared memory. The encoding is real.

2. **wasmtime CLI wiring — WAS MISSING, now fixed here.** `-W
   shared-everything-threads` was *parsed and then never applied to the engine
   Config*: it is absent from the `handle_conditionally_compiled!` table in
   `crates/cli-flags/src/lib.rs`, and the only caller of the setter was the
   wast test runner. The flag was a silent no-op. Five-line fix in
   `wasm/wasmtime-set-cli-flag.patch.notinchain`. With it applied, shared
   globals and shared function types **compile**.

   That patch is deliberately **NOT in the Dockerfile patch chain.** It
   enables an engine feature that cannot be completed (layer 3), so on the
   fleet it would buy nothing and only widen TCB behaviour. It is kept
   applied-and-tested for the day SET is real, and is worth sending upstream.

3. **Cranelift codegen — MISSING.** 31 SET operators return
   `wasm_unsupported!("shared-everything-threads operators are not yet
   implemented")`. 22 are GC struct/array atomics (irrelevant to pthreads);
   the ones a threaded C program would actually need are the global and table
   atomics (`global.atomic.*`, `table.atomic.*`).

4. **Component spawn intrinsics — MISSING.** `thread.spawn-ref`,
   `thread.spawn-indirect` and `thread.available-parallelism` are parsed by
   wasmparser and then hit `bail!("unsupported intrinsic")` in
   `crates/environ/src/component/translate.rs`. This is the layer that would
   let a *component* start a thread at all.

5. **Guest toolchain — MISSING ENTIRELY.** wasi-libc has exactly one p3
   threading model, `ENABLE_COOP_THREADS` (the cooperative one we shipped).
   There is no SET thread model, so even a perfect engine would have no
   compiler emitting programs that use it. Building one means porting musl's
   pthreads onto SET primitives — the same scale of work the coop-threads
   directory took upstream.

Upstream has not started: the only SET commits in wasmtime's history are
"threads: add feature flags" (#10206, #10569), the tracking issue (#9466) has
no linked PRs, and the entire SET test suite is a single 3-line `.wast` that
asserts an empty module parses.

## WORKING: real parallel pthreads, measured 11.3x (2026-08-07)

Not a harness this time — a real C program using `pthread_create`/`pthread_join`,
compiled by clang, running on our patched wasmtime and using many cores. The
host side of wasi-threads is REBUILT in `src/commands/run.rs` (upstream deleted
it in b4b23fe583). Source: `tools/parallelism-probe/pthread-scaling.c`.

| threads | guest wall | real | user | cores busy |
|---|---|---|---|---|
| 1  | 219ms | 0.239s | 0.256s | 1.1 |
| 2  | 220ms | 0.227s | 0.444s | 2.0 |
| 4  | 219ms | 0.226s | 0.881s | 3.9 |
| 8  | 231ms | 0.238s | 1.781s | 7.5 |
| 16 | 309ms | 0.316s | 3.928s | **12.4** |

16 threads x 900M iterations = 14.4 BILLION iterations in 309ms. Sequential
would be ~3.5s: **11.3x**. `user` climbing to 3.9s while `real` stays ~0.3s is
the part that cannot be faked — those are real cores.

Build the guest with imported+exported shared memory, which the WASI p1 host
also requires so it can reach guest memory:

    clang --target=wasm32-wasip1-threads -O2 -pthread \
      -Wl,--import-memory,--shared-memory,--export-memory,--max-memory=67108864 \
      -o app.wasm app.c
    wasmtime run -W threads,shared-memory app.wasm

Two things upstream's version could not do, solved here:

- **No clonable WASI ctx.** wasi-threads needed `T: Clone`, which is why
  `wasi-common` was deleted in the SAME commit — the modern `wasmtime_wasi` ctx
  is not `Clone`. Cloning was never the actual requirement: each thread just
  needs *a* context, so we build a FRESH one per thread (inheriting stdio, so
  `printf` from a worker still reaches the terminal). Honest consequence:
  threads share LINEAR MEMORY (what pthreads needs) but NOT a file-descriptor
  table. Compute-parallel work is unaffected; opening an fd on one thread and
  reading it on another is not supported.
- **Async engine.** The CLI configures async, so the sync `instantiate`/`call`
  entrypoints deadlock in a spawned thread. Threads drive the async ones on
  their own tokio context, and the main thread returning `process::exit`s (the
  wasi-threads rule that the main thread ending ends them all) — without that
  the CLI blocks forever on workers parked in a futex.

**Scope, stated plainly:** this is core modules (wasip1-threads), not
components. It is real parallelism available today; it costs the component
boundary that carries egress, the loopback wall and the WASI capability model.
That trade is a product decision, not a technical one, and this file exists so
it can be made with numbers instead of guesses.

## The earlier harness: 7.8x, and how



The machinery for real parallelism is still in the engine — only the
wasi-threads *crate* was deleted. `SharedMemory`, `-W threads`, `-W
shared-memory` and guest atomics are all intact. The trick wasi-threads used
sidesteps wasmtime's core constraint (a `Store` is not `Sync`, so two OS
threads can never execute in *one* store):

> **one shared linear memory, one `Store` per OS thread.** The host creates a
> `SharedMemory`, every per-thread instance *imports* it, and each OS thread
> instantiates into its own `Store`. What is shared is the memory, not the
> store.

`tools/parallelism-probe/main.rs` rebuilds exactly that on wasmtime 49.
Identical total work (4.8e9 iterations), same binary, same machine:

```
1 thread  (sequential): wall=1167ms
8 threads (parallel)  : wall=149ms      -> 7.8x
```

and `guest_atomic_counter=8` confirms the guest's own
`i32.atomic.rmw.add` against the shared memory was correct across all eight
OS threads. This is genuine multi-core wasm with working shared-memory
synchronisation, on the engine we ship.

**Why our apps still cannot use it:** the mechanism is core-module shaped. Our
platform requires *components* (the classifier refuses core modules at
publish, and it is the component boundary that carries the whole security
model — egress, the loopback wall, WASI capabilities). A component can carry
shared memory (verified: it validates and compiles), but nothing in the
component model can *start a thread* until layer 4 exists. Bridging that gap
without the SET intrinsics would mean inventing spawn semantics ahead of an
unratified spec, inside a measured TCB — the one place in this system where
guessing is unacceptable.

## What shipped alongside this investigation

**CPU fair-share is now on by default**, proportional to purchased `cpuShare`
(`_cpu_weight_for`, `wasm/wasm_manager.py`; tests in
`test/wasm-cpu-weight.test.mjs`). A tenant holding 0.25 of a node gets
cpu.weight 2500 of 10000, so under contention it gets what it paid for.

This is the prerequisite for every parallelism story, and it is deliberately
the *weight* and not the cap: a weight never throttles anyone (cgroup-v2 only
consults it when CPU is contended, so apps still burst to idle cores), while a
hard cap throttles even an idle node and stays opt-in behind
`WASM_CPU_MAX_PCT`. Turning fair-share on only *after* parallel guests exist
would be closing the door behind the horse. `WASM_CPU_WEIGHT=0` restores the
old unweighted behaviour.

## What to do next (SET is real; the gate is now the toolchain)

The engine half is done and measured. The remaining sequence:

1. **Guest toolchain.** No wasi-libc SET thread model exists, so guests are
   hand-written WAT today. Porting musl's pthreads onto SET primitives is the
   real next project — comparable in scale to upstream's coop-threads
   directory. Until it lands, the audience for SET is hand-written or
   compiler-generated-by-us wasm, not `clang -pthread`.
2. **Review before the Dockerfile chain.** `wasmtime-set-threads.patch.wip`
   stays out of `wasm/Dockerfile.wasmtime` until the concurrency design in
   `set_threads.rs` has been reviewed by someone who did not write it. It is
   entering a measured TCB; the TSan and soak evidence above is necessary,
   not sufficient. Three adversarial rounds have now been run and every one
   found real UB, including a case where the previous round's fix was silently
   ineffective. Treat a clean round as the bar, not as a formality.
3. **Platform plumbing.** Already designed and shipped for coop threads, and
   it generalises directly: compile-probe → byte-marker sniff → publish stamp
   → claim gate → per-tenant engine flag → fleet-AND. Add a `set` capability
   beside `coopThreads` rather than inventing a new shape. Note the
   compile-probe must use `set-spawn-parallel.wat`, not a help-text grep — the
   feature flags lie.
4. **Revisit the hard cap.** Real parallelism is exactly the case where an
   operator may genuinely want `cpu.max` and not just the weight: one tenant
   spawning `available_parallelism` threads on an idle node is fine, but the
   contended case now has real teeth. `thread.available_parallelism` already
   reports the tenant's purchased slice, which is the first line of defence.
5. **Upstream.** The CLI-flag fix and the shared-func-type accessor fixes are
   independently useful and worth sending to bytecodealliance/wasmtime#9466.
   The execution-view design is worth *proposing* there, but it is ours to
   defend: it deliberately implements less than the full SET proposal (see
   "Sharedness, stated exactly") and trades completeness for provable
   soundness.

## Per-deployment inference compute threads

`nnThreads` in the deployment configuration sets the ggml compute-thread count
for that instance. The manager caps it at the instance's purchased CPU share
(rounding up to a whole vCPU). Omitting it preserves the engine default. For
example, `"nnThreads": 4` on a 53% share of a 16-vCPU node uses four compute
threads; requesting 16 is capped at 9.

This controls the CPU operations surrounding GPU inference. The shielded
backend's background masking workers have a separate bounded pool, so reducing
the compute pool can reduce contention while the GPUs perform the matrix
operations. Benchmark the setting with the same model, context and prompt. It
does not allocate additional CPUs, alter the model, or reduce the context.

`nnThreadsBatch` sets the thread count for batch compute (prefill: any
ubatch of 32 or more tokens) separately, under the same cap. The shielded
tier pulls the two apart: decode wants few compute threads, because its
masking refill threads are the binding resource and every compute thread past
four measured as contention, while prefill runs in the enclave on cores by
policy and scales with them. Omitting it keeps prefill on the `nnThreads`
count, so existing configurations do not change.

`nnShieldedPadWaitUs` optionally waits up to 0–50000 microseconds for an
already-reserved privacy-mask batch before generating missing masks on the
request thread. It defaults to zero. Waiting does not add workers or reuse
masks; a timeout consumes only ready masks and generates the remainder as
before. Compare TTFT, generation throughput, and the shielded profile's
`waited`/`wait` counters before enabling it for a workload.

Three more keys size the masking pool itself. Every pad is one row of
`u = r·W` over a weight group, and the request path draws one pad per group
per token row: speculative decode verifies and drafts about three rows per
accepted token, so on a 27B even a single chat consumes most of the refill
capacity, and the refill threads, not the compute pool, are the binding
resource. `nnShieldedRefillBatch` (1–64, engine default 4) is how many rows
one refill job computes; past four rows the engine's row-blocked kernel
streams the group's weights once per job instead of once per four rows, which
raises the aggregate refill rate but makes a drained group wait for a longer
job. `nnShieldedPoolDepth` (16–4096, default four times the widest batch the
graph presents, at least 16) is the pad ring per group; it must hold at least
two refill jobs, and each 32 rows of depth cost up to about 1 GiB of enclave memory
on a 27B. `nnShieldedRefillThreads` (1–64, default half the vCPUs) is the
refill thread total, divided over the card links. Measure all three with the
same model, prompt and context: the wins are workload-shaped.
