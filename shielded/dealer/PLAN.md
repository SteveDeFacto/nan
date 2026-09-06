# Dealt pads: the Shielded tier with the pad work off the trusted half

Steven's brief (2026-09-06) is the specification; this file maps it onto the
code that exists, fixes the formats and indices, and orders the work. Nothing
here changes the masking construction (SECURITY.md sections 2-5 still hold);
what changes is WHO computes `u = r·W` and WHERE the plaintext half runs.

## Why

Measured today on metal0 (27B Q8, two V100s): the trusted half's pad refill
is the binding resource for everything. One chat draws ~3 pad rows per
token through speculation and keeps the refill threads 85% busy at 11.5
tok/s; every batch shape tops out at the refill rate (~20 token-rows/s);
the cards sit at 2-4% busy; and prefill, which turned out to ride the same
pads (one per group per prompt token), runs at exactly the refill rate too.
The row-blocked kernel moved that ceiling from DRAM-bound to compute-bound
and no further: the arithmetic is 3 planes x the model's MACs per row, on
CPU cores, inside the boundary. A dealer moves those MACs to any GPU that
never sees `x + r`, and the trusted half keeps vector-sized work plus
attention.

## Parties, mapped to code

| Party | Role in the brief | What exists | What is new |
|---|---|---|---|
| Platform CVM | model registry, dealer, consumption ledger, shared-prefix service, verifier, billing | model volumes + calib (`metal/shielded-overlay/calib`), relay `avf-verify.mjs`, registry/badges | `shielded-dealer` (mints, encrypts, ships), the ledger endpoint, the prefix service |
| Operator | GPU worker, NVMe bank, transport, phone holder | `shielded/worker-cuda`, the anchor owner app (`shielded/anchor/avf/host`) with its vsock bridges | the bank + prefetch bridge (vsock 7779), pad storage on the GPU box |
| Pixel pVM | the trusted half | the whole ggml-shielded engine in Microdroid (anchor phase 3), transport key + attestation (phase 4), Freivalds `s~` per node | an external PAD SOURCE for the ring, PRF-derived `r` from a dealt seed, the ledger window, refusal on exhaustion, receipts |
| User | attests, TLS into the pVM | session/TLS in the pVM (phase 1) | unchanged |

## The pad, exactly

The engine's ring (`shielded-tee.c`, `sh_group`) holds `depth` slots of
`(r, u)` per group: `r` is `K` values in `[0, M)`, `u` is `u_len` balanced
values in `(-M/2, M/2)` covering every node of the group, `M = 14,457,349`
(`SH_M_MOD`, ~2^23.8). Today `generate()` fills a slot from a ChaCha20 mask
bank keyed by a random per-process key plus the row-blocked refill; the
request path masks with `r` in place and unmasks with `u`.

Dealt pads keep the slot, change the fill:

- **Identity.** `(seed_id, group, index)`. `seed_id` names a 32-byte seed the
  dealer derived for one pVM (`HKDF(master, pvm_id, epoch)`); `group` is the
  engine's group ordinal under a fixed registration order (recorded in the
  shipment header so both sides agree); `index` counts pads per group.
- **`r` is derived, never shipped.** `r = ChaCha20(seed, counter)` with
  `counter = (group << 48) | (index << 24) | block`, reduced to `[0, M)`
  exactly as `maskbank_issue` does today (uint64 draws, ~2^-40 bias). Both
  the dealer and the pVM compute it; 65,535 groups, 16M pads per group per
  seed, 2^24 blocks per pad.
- **`u` is shipped.** 3 bytes per column, offset binary `u + M/2` in
  `[0, M)`; 12.6 MB per token-row on the 27B (`sum u_len` = 4.2M columns),
  1.7 MB on the 0.8B. Incompressible by construction.
- **One use.** The slot is consumed the moment the request takes it (rule 1
  in the ring's comment). Reuse across boots is what the ledger prevents.

## Shipment format (`.pads`)

Row-major by index so a bank can prefetch prefixes and a pVM can start on the
first rows:

```
header (little-endian, 4 KiB-aligned, sealed):
  magic "ENCLPAD1", version u32, model_digest[32] (calib digest),
  seed_id[16], group_count u32, index0 u64, index_count u64,
  groups[]: { group u32, K u32, u_len u64, name[64] }
  then per index i in [index0, index0+index_count):
    per group g in registration order: u[g][i] as 3*u_len bytes
```

Encryption: per CELL (one index, one group) `crypto_secretbox` (XSalsa20-
Poly1305, TweetNaCl, vendored beside the engine as it already is in the pVM
payload) under a shipment key `k`, nonce = `seed_id[0:8] || index || group`;
`k` boxed to the consumer's X25519 pad key from an ephemeral dealer pair
(`crypto_box`); the header and group table are authenticated by a boxed
SHA-512 under `k`. Cells are independent because groups consume at
different rates (the vocabulary projection is asked for far fewer rows than
the layers), so each group keeps its own cursor and opens only its own
cells. The operator's bank sees ciphertext and sizes only; prefetch
granularity is one cell.

Implemented (P1, engine side): `wasm/ggml-shielded/shielded-pads.{h,c}`
(format, derivation, writer, reader, local ledger window),
`shielded-tee.c` dealt mode (`SHIELDED_PAD_*` env, refill threads import
instead of minting, the request path never mints, reserve-before-use
window, `sh_link_mint_shipment`, `sh_link_group_table`), and
`dealt-selftest.c` (mint -> oracle -> tamper -> wrong key -> exhaustion ->
ledger -> link import), run by `test/shielded-cbackend.test.mjs`.

**P1 end to end, 2026-09-06 (CPU only, no GPU touched):** `shielded-dealer`
minted 64 rows for Qwen3.5-0.8B (123 MB, 1.9 MB per row) to a consumer key
from `pads-keygen`; `shielded-run` in dealt mode against `worker.py
--device cpu` (the reference worker's exact float64 path) offloaded all
2363 nodes, used 2033 pads, minted none, verified every product (0
failures), advanced the ledger to 64 before use, and produced the exact
text of the self-minting run. Two things the run taught: importers are
threads, so the reader keeps per-call scratch and a lock around its file
list; and a ring that is short at an exchange must WAIT for the importers
(`dealt_wait`), not fail, since a dealt engine has nothing to mint with.

## Ledger and single use across reboots

pVM state is ephemeral, so the ledger is the only durable record. The unsafe
order (consume, then report) lets a crash replay indices. The order is
therefore:

1. pVM asks the ledger to RESERVE `[mark, mark + W)` for `(seed_id)`; the
   ledger advances `mark` to `mark + W` and answers with the window.
2. pVM consumes only inside its window; it asks for the next window before
   the current one runs out.
3. On any restart the pVM starts at the ledger's current `mark`. Whatever was
   left unconsumed in the last window is burned (W = 64 rows ~ 0.8 GB of
   pads on the 27B, worth minutes of dealer time, nothing else).

The ledger is the "high-water mark the operator cannot roll back": it lives
with the platform, signed responses, and the pVM refuses to run without a
fresh window. Per-group cursors inside a window are the pVM's own.

## Integrity

Unchanged from SECURITY.md section 5: every product is checked with the
Freivalds vector `s~` that never leaves the trusted half (`st32 = W·s~` per
node, `y·s == x·(W s~)`). That check already covers a wrong `u`, because `u`
is subtracted before it runs; the dealer is not trusted for correctness. A
separate pad check (`u·s~ == r·(W s~)`, O(K+N) per pad) is cheap and tells a
bad dealer from a bad worker; it is optional and off the hot path.

## Throughput, the only accounting that matters

Per token-row: 12.6 MB of `u` in, 8.9 MB of masked planes out to the worker,
~13 MB of products back (24-bit replies), plus 240 dependent round trips.
So: the dealer's egress and the operator's link set sustained rows/s
(1 Gbps ~ 10 rows/s of the 27B, ~90 of the 4B, ~500 of the 0.8B); the bank's
NVMe sets the burst; USB and the exchange chain set per-token latency; pVM
memory sets context. Speculation costs rows, so with dealt pads plain decode
is the bandwidth-efficient mode. Prefill needs a pad per prompt token, so the
bank must be sized for the largest prompt burst, not the decode rate.

## Phases

### P1. Format, dealer tool, engine pad source (CPU-testable end to end) - DONE 2026-09-06
- `wasm/ggml-shielded/shielded-dealer.cpp`: loads a GGUF + calib the way
  `shielded-calib`/`shielded-run` do, registers the groups through the same
  `sh_link_add_weight` path (identical ordering, `K`, `u_len`), derives `r`
  from a given seed, mints `u` with the row-blocked refill, writes a `.pads`
  shipment for an index range, encrypted to a given X25519 public key.
- `shielded-tee.c`: `SHIELDED_PAD_SOURCE=<dir>` + `SHIELDED_PAD_SEED` +
  `SHIELDED_PAD_KEY`: the refill threads read rows from shipments instead of
  minting; `generate()` on the request path is refused (a dealt engine never
  mints; a dry pool stalls, rule: fail closed); the ring accounting, take,
  hold and release are untouched.
- Test: `shielded/e2e.py` style run on the 0.8B with `worker.py` (CPU):
  dealt engine vs self-minting engine produce identical tokens, and a
  tampered `u` row fails the Freivalds check.

### P2. Ledger window, attestation-gated release, pad key in the binding - BUILT 2026-09-06 (device test pending)
- relay (`relay/pads.mjs`, routes in api-relay.js): `/v1/pads/key` (the
  Ed25519 ledger key), `/v1/pads/seed` (the pVM's seed, derived
  HKDF(master, keyFp, epoch), boxed X25519 -> HKDF-SHA512 -> ChaCha20-Poly1305
  to the pad key the tunnel bound), `/v1/pads/reserve` (reserve-before-use
  window, signed), `/v1/pads/ledger` (the mark). Requests are signed by the
  attested transport key; `tunnel.js` keeps spki + padKey per tunnel.
- trusted half (`shielded-pads.c`): opens the seed box, verifies windows,
  signs requests; `sh_link_set_window_provider` replaces the ledger file.
  `test/pads-ledger.test.mjs`, `test/pads-interop.test.mjs` (Node <-> C).
- pVM (`shielded/anchor/avf`): the payload mints an X25519 pad key at boot
  (PADKEY after SPKI), opens PADSEED, signs PADSIGN, receives shipments on
  vsock 7780 into the bank dir (tmp-then-rename), and runs the engine with
  `SHIELDED_PAD_*` set and an `anchor_pads` context; `engine.cpp` installs a
  window provider that writes PADWIN on the control socket and verifies the
  reply; the owner app (`PadsClient.java`) presents padKey with the
  attestation, bootstraps the seed after ACCEPTED, relays PADWIN, and streams
  `.pads` files from `--es pads <dir>`. Compile-checked (arm64 payload links,
  Java builds); no device attached to run it. AOSP's `vm_payload.h`/map are
  vendored in `avfref/` so the build is reproducible.
- Still to do here: the pad check (`u.s~ == r.(W s~)`), the shipment digest
  pinned from the calib in the pVM, and the seed epoch rotation.

### P3. Operator bank and prefetch
- Owner app: pad bridge on vsock 7779; NVMe store on the GPU box's agent;
  prefetch depth from link rate and the largest expected prompt.

### P4. Shared-prefix KV, receipts, billing
- Prefix service signs KV for (model, public prefix); pVM loads instead of
  prefilling; pVM-signed usage receipts feed operator payout.

Each phase lands behind flags; the self-minting engine stays the default
until P2 and P3 exist, so nothing in production moves until then.
