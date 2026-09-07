# Shielded inference — masked GGML offload to untrusted GPUs

Status: DESIGN (2026-08-14), **RUNNING END TO END (2026-08-25)**, nothing on the fleet.

The oracle in `shielded/reference/` still exists to prove the constructions. Since 2026-08-25
there is also a working implementation: `shielded/worker.py` (the untrusted GPU half),
`shielded/tee.py` + `shielded/model.py` (the trusted half), and `metal/guest/shielded.mjs`
(the CVM's client). A real GGUF model generates real tokens with every linear op masked and
executed on an untrusted RTX 3070, and the output is bit-identical to the same model run
entirely in-TEE. Results and the two design changes it forced are in `shielded/REPORT.md`
§10. `model.py` is not an engine -- it is the specification and the equivalence reference the
C++ ELL backend must reproduce bit for bit. This document is the synthesis of the required
reading (TwinShield arXiv:2507.03278, KV-Shield arXiv:2409.04040, permutation equivariance
arXiv:2304.07735, Slalom arXiv:1806.03287, Amulet arXiv:2512.07495, and the ggml-rpc source)
against this repo's actual plumbing. It exists so that implementation starts from settled
decisions, not from the papers. Where a paper is silent (all of them are silent on
autoregressive decode), the gap is named and our design fills it explicitly.

Goal: one masked-offload engine serving five interfaces — chat (LLM), STT, TTS, image
generation, vision (VLM) — where user inputs, activations, KV cache, and sampling state are
never plaintext outside the SEV-SNP CVM. The GPU and its host are fully untrusted. Weights
are public (open-weight catalog models only); weight secrecy is a non-goal everywhere.

This is the successor to the "Layer 4" caveat in `worker/worker.py` and
`supervisor.js` (freed VRAM is not zeroed; residual-data scrubbing unimplemented): in the
shielded tier that caveat dissolves, because nothing that ever reaches VRAM is plaintext.
Ghost data in VRAM is ciphertext by construction, not by scrubbing.

## What this does and does not provide

Provides, against a malicious host operator with root on the GPU box, full PCIe/DMA
visibility, VRAM read access, and the ability to replace the GPU-side runtime:

- Confidentiality of prompts, input/output audio, input/output images, voice-cloning
  reference audio, activations, KV cache, logits, and sampling state. Every tensor that
  leaves the CVM is protected by an additive one-time pad over a prime field (Slalom
  construction; TwinShield construction for v2 attention offload). No bare permutation is
  ever the sole protection — a permutation of a known matrix is reversible by column
  matching, and our weights are public (see "Why permutation is plumbing, not protection").
- Integrity: every offloaded product is verified (preprocessed Freivalds), and a request
  aborts before any output token/sample/pixel leaves the CVM if a check fails. A cheating
  GPU can deny service; it cannot make us emit a wrong or attacker-influenced result.

Does NOT provide:

- Weight secrecy (non-goal; the constructions require the TEE to know W, and public weights
  are what make `r·W` precomputation possible at all).
- Shape/timing secrecy beyond declared buckets: padded request shapes, context-length
  buckets, audio-duration buckets, image resolution/step buckets, and coarse timing are
  public. The graph topology sent to the worker reveals the model architecture — which is
  public anyway (curated catalog).
- Availability: the GPU host can stall, refuse, or corrupt (we detect and abort — the
  fallback is CPU-in-TEE or another box, not toleration).
- Anything that requires trusting the GPU driver, host kernel, or operator restraint.
  Per the brief, any such design is rejected by definition.

## The one architectural decision

All five interfaces run on GGML-family runtimes (llama.cpp incl. mtmd for chat/vision/TTS,
whisper.cpp for STT, stable-diffusion.cpp for image gen). We therefore build exactly one
component with two halves:

- **TEE side** (inside the CVM, linked into the ELL engine): a GGML backend + graph
  executor that classifies ops, executes everything nonlinear/sensitive on CPU, and
  offloads masked linear ops to the remote worker. The split mechanism is
  `ggml_backend_sched` — it already partitions a graph across a priority-ordered backend
  list, inserts the cross-boundary copies, and supports per-node pinning via
  `ggml_backend_sched_set_tensor_backend`. Backend order: `[shielded-remote (prio 0),
  trusted-CPU (last, as required by sched)]`. The mask/unmask interposer wraps the only two
  commands that ever carry activation bytes: `SET_TENSOR` outbound, `GET_TENSOR` inbound.
- **GPU side** (`shielded/`, new top-level dir): a stateless worker derived from the
  ggml-rpc server, hardened (below). It holds public weights resident, accepts masked
  tensors, runs a fixed vetted op set, returns masked results. It never sees sampling,
  logits-as-plaintext, nonlinear ops on secret data, or any unmasked activation —
  including transiently.

All five frontends inherit confidentiality from this backend; per-interface work is
pre/post-processing placement. Existing catalog guests (llm-chat etc.) keep their wasi-nn
WIT contract unchanged — the shielded tier is a launch-time engine configuration, not an
app-visible API.

Critical fact from the ggml-rpc source read: **ggml-rpc as shipped is whole-graph** — the
server executes the entire cgraph including softmax/norms/rope on its local backend, and
`GRAPH_COMPUTE` will run any op an attacker serializes (see GHSA-j8rj-fmpv-wcxw). It is a
transport template only. The trust split comes from the scheduler in the TEE; the worker's
compute surface must be reduced to an installed, vetted graph (§ worker protocol).

## Numeric foundation: the field is not optional

Additive one-time masking has no exact group structure over floats: `(X+R)·W − R·W ≠ X·W`
in fp arithmetic once R is large, and "uniform" masks don't exist over IEEE floats. Slalom's
load-bearing move — inherited by TwinShield — is fixed-point embedding into a prime field:

- Field **Z_p, p = 2^24 − 3**; values quantized as `round(2^l · x)` with **l = 8**
  fractional bits (biases at 2^2l). p < 2^24 so field elements and their products remain
  exactly representable in fp significands.
- Masks uniform over Z_p ⇒ per-tensor one-time-pad, information-theoretic hiding (PRG-seeded
  in practice ⇒ computational). Mask reuse leaks differences; masks are strictly one-shot.
- Everything offloaded is exact integer arithmetic ⇒ after unmasking, results are
  **bit-deterministic** and equal to a TEE-computed fixed-point reference exactly. The
  equivalence deliverable becomes a hard equality check in the field domain, plus a
  documented tolerance versus the fp16 GPU baseline (quantization-class noise: Slalom
  measured <0.5% accuracy cost; TwinShield +0.21 ppl on LLaMA-7B at these parameters —
  comparable to a GGUF q8 step. Our catalog already serves q4/q8 quants).

Measured in `shielded/reference/` (see "Reference oracle" below), correcting an
assumption this document previously carried:

- **Width is not the risk.** The matmul accumulator sits at scale 2^2l and was expected
  to grow with the inner dimension. It does not: standard 1/sqrt(d) init is
  variance-preserving, so a normalised activation yields an O(1) output at any width.
  Measured bit requirement is flat at ~18.7 bits from d=64 through d=14336, leaving
  ~5 bits of headroom under p/2. (p, l) = (2^24−3, 8) therefore holds at production width.
- **Magnitude is the risk.** LLMs carry massive-activation channels. Measured overflow
  appears around 10^4× outlier channels (27.6 bits needed vs 23 available); 10^3× still
  fits at 22.8 bits. Known outliers run 10^2–10^3×, so we sit inside the envelope with
  roughly one bit to spare. Mitigation is mandatory and cheap: a **per-tensor magnitude
  guard that fails closed** before encode (never silently wrap — a wrap decodes to noise
  and would corrupt output with no error signal), plus per-tensor rescaling in the GGUF
  per-block-scale style.
- **Escape hatch, proven exact: RNS.** Two coprime ~24-bit primes with CRT recombination
  gives 48 bits of dynamic range while keeping both limbs fp64-friendly, so the kernel
  plan below is untouched. Verified bit-exact at d=4096 in the oracle. This is
  arithmetic, not cryptography — the one-time-pad argument is per-channel identical to
  Slalom's, so it adds no security assumption. Carry it for outlier-heavy models.

### GPU kernel: measured, and the plan changed

Measured on an RTX 3070 (`shielded/bench/field_gemm_bench.py`), every rung verified
bit-exact against an int64 reference before being timed. **The original limb plan is
superseded**: RNS over byte-sized primes needs one int8 tensor-core GEMM *per prime*
instead of N² limb cross-products, and measures ~10× faster.

| rung | prefill (M=512–2048) | notes |
|---|---|---|
| **RNS-3, int8 TC** | **2.6–3.6× fp16** | 3 primes, ~23.8 bits of range. The design point. |
| RNS-4, int8 TC | 3.4–4.9× fp16 | 31.6 bits; the outlier-safe fallback. |
| limb-int8 (old plan) | 25–33× fp16 | 16 cross-product GEMMs. Abandoned. |
| fp64-RNS | 200–800× fp16 | Exact and simple, but 1/64 fp64 rate on consumer parts. |

Two hard constraints fell out of the measurement rather than the design:

- **int8 tensor cores refuse M ≤ 16** (`torch._int_mm` requires M > 16), so batch-1 decode
  cannot use them at all and needs a bespoke kernel.
- **fp16 cannot hold the accumulator**: it saturates at 65504 and represents integers
  exactly only to 2048. fp64-RNS with byte primes is exact with no chunking at all
  (products ≤ 15625, K=14336 accumulation ~2.2e8, far inside 2^53); fp32 would need 5-bit
  primes and 5–6 channels.

At decode the GEMM is bandwidth-bound, so the cost is **bytes per weight**, not FLOPs.
Measured read bandwidth 322–387 GB/s; RNS-3 at 3 B/weight projects to **1.5× fp16 but 6× a
q4_K baseline** — and q4_K is what the fleet actually serves. That is the honest decode
denominator, and it sits at the kill line rather than comfortably inside it. Batching
amortises the weight read, which is why the criterion is stated at batch ≥ 4.

Weights are converted once at worker start: GGUF → dequant → fixed-point field
representation (per-limb planes for the kernel in use). This inflates weight VRAM vs q4/q8;
budget ~4 bytes/param (fp64 path) down to ~3 (limb paths). Public weights, so resident
plaintext-field form is fine.

## Constructions (cited only, no homebrew)

### Linear offload: Slalom additive OTP

Per offloaded linear op `y = x·W` (matmul, conv-as-im2col, embeddings-as-matmul where ever
actually used):

- Offline: sample `r` from AES-CTR PRG; precompute unblinding factor `u = r·W`
  (for convs: `u = Conv(r, W)`). Store `(seed-index, u)` AES-GCM-encrypted in untrusted
  DRAM/disk (Slalom's trick — the bank does not consume CVM RAM).
- Online: send `x + r mod p`; GPU returns `(x+r)·W`; TEE computes `y = (x+r)·W − u`.
  TEE online cost is O(|x|+|y|) additions. One-shot masks, strictly.

**The precompute economics are the honest core of this design.** `u = r·W` costs the same
MACs as the offloaded product itself, so at 100% duty cycle sustained throughput is capped
by TEE mask-refill rate, not by the GPU. Three things rescue this:

1. Mask generation is a *batched, offline* GEMM (`R_bank · W` for thousands of future
   masks in one streaming pass) — an order of magnitude more MAC-efficient on CPU than
   latency-bound single-token decode, and embarrassingly parallel across idle CVM cores.
2. Banks convert off-peak CPU into peak GPU throughput (store GBs of `u` encrypted on the
   untrusted side).
3. The arithmetic is int8-shaped (byte-sized RNS residues, int32 accumulate), which is
   exactly what AVX-512 VNNI accelerates — and `torch._int_mm` reaches it through
   FBGEMM/oneDNN on CPU today, verified exact for byte primes. That path is roughly an
   order of magnitude faster than stock fp64 BLAS, and it is the difference between an 8B
   model being servable and not.

Measured numbers, per-model ceilings, and the core-count scaling live in
`shielded/REPORT.md`; treat that file as authoritative over any figure quoted here.

Mask banking is per-(model, layer, shape-bucket), and decode masks for step t+1 are staged
during the GPU compute of step t so mask handling never sits on the token critical path.

### Integrity: preprocessed Freivalds, per token, before anything streams

Naive Freivalds (`y·s =? x·(W·s)`) is useless at batch 1: computing `W·s` costs the same as
the matvec being checked. Slalom's preprocessed variant fixes exactly this: TEE keeps a
secret `s̃ = W·s` per weight matrix (computed once, reusable), and the online check is
`y·s =? x·s̃` — **O(|x|+|y|) multiplications per check even at batch 1**. Soundness with
Slalom's parameters (`s` entries from |S| ≈ 2^20, k=2 repetitions) is ~2^-40 per check; one
secret `s` is reusable across layers/steps/requests with only linear (union-bound) soundness
decay — resample periodically. Checks run on recovered plaintext in the field domain, so
they are exact equalities, no tolerance games.

Policy: verify every offloaded product for a token before that token is sampled/streamed.
Verification is cheap enough to be synchronous per token; there is no
"stream now, verify later" window. Any failure aborts the request (fail closed), logs a
worker-integrity incident, and quarantines the box from the shielded tier.

TwinShield's U-Verify (embedded hash row, ~33% cheaper than naive Freivalds) is noted but
not adopted for v1: preprocessed Freivalds is simpler, batch-1-optimal, and doesn't couple
verification to the masking layout.

### v2 attention offload: TwinShield, with a decode-shaped caveat we found

TwinShield's OutAttnMult offloads `Q·K^T` (both operands secret) by stacking masked blocks
`[Q+R_Q ; a·R_Q]` (secret row perm λ1) against `[K^T+R_K | b·R_K]` (secret col perm λ2);
one 2m×2p GPU matmul (4× FLOPs of the m×p product) returns four blocks from which the TEE
recovers `Q·K^T` with cheap block algebra. OutSoftMax offloads only the exponentials
(`x−r` out, `e^(x−r)` back, rescale by precomputed `e^r`, normalize in TEE). Softmax·V
reuses the OutAttnMult shape.

Our analysis (to be adversarially reviewed before v2 lands — this is TCB work): the
construction's hiding degrades with row count. The GPU sees `a·R_Q` rows raw, so Q is
confined to a (candidate-rows × 2^24-scalar) family rather than an OTP-uniform set; the
paper's own security accounting is `log(d·(2m)!)` bits, which at **decode (m=1) collapses
to ~25 bits**. Additionally, real activations have strong priors, so small candidate
families are dangerous. Consequences baked into the plan:

- v2 offloads attention for **prefill and batched contexts only** (m large: prefill QK^T
  at 1k–8k rows is also exactly where CPU attention hurts most).
- **Decode attention stays in the TEE permanently.** At batch 1 this costs
  ~2·n_layer·n_ctx·d MACs/token on CPU (with flash-attn CPU path + q8 KV) and is the main
  reason the chat kill-criterion is measured at batch ≥ 4.
- Cross-request row batching to inflate m for decode is noted as a possible v3, not
  designed here.
- TwinShield's OutSoftMax has an unresolved numeric hole (real-valued `e^x` vs field
  masks; overflow of `e^(x−r)` for unbounded r) — any adoption must bound mask ranges and
  therefore downgrade the hiding claim from perfect to statistical, documented as such in
  SECURITY.md, or stay TEE-side. Softmax remains in-TEE until that analysis is written.

Per the brief: TwinShield construction only for activation-activation products, no
homebrew variants. The caveats above restrict *where* we use it, they do not modify it.

### Why permutation is plumbing, not protection

KV-Shield protects the KV cache by feature-permuting `W_{q,k,v}` columns (`W·Π`) inside a
TrustZone TEE and inverse-permuting attention output. Against public weights this is void:
the untrusted side holds `W·Π` and public `W`; columns of a trained weight matrix are
pairwise distinct, so hashing columns recovers Π completely in ~O(d) — no search. The
permutation-equivariance paper (2304.07735) supplies what *is* useful: per-op commutation
rules (softmax/elementwise/norms commute freely; linear layers need conjugated weights;
multi-head restricts feature perms to within-head ⊗ head-swap, 768! → 12!·64!), which is
precisely the bookkeeping the masked executor needs to push transforms through a ggml
graph, plus the head-blocked permutation constraint KV-Shield itself missed. We build the
KV-Shield-style permuted pipeline once, as Phase 1 plumbing validation (tensor reordering,
inverse-perm bookkeeping, correctness harness: bit-exact when the permuted axis is not a
contraction axis, ulp-tolerance when it is) — then never rely on it for security.

Also inherited from that read: RoPE consumes token positions inside every layer (token
permutation needs permuted position ids), causal masks must be conjugated, and none of the
published theorems cover incremental KV decode. The correctness harness covers these cases
explicitly.

## Op placement (v1)

| Op class | Where | Why |
|---|---|---|
| QKV / O / FFN up,gate,down / lm_head matmuls | GPU, masked | Slalom OTP + Freivalds; bulk of FLOPs |
| Conv1d/conv2d (UNet, whisper front convs if offloaded) | GPU, masked (Phase 4/5) | Slalom conv masking, `u = Conv(r,W)` |
| Embedding lookup | TEE | it's a gather keyed by the secret token id; as-matmul would cost n_vocab·d |
| QK^T, attn·V | TEE at decode, PERMANENTLY; GPU masked for prefill only (v2, TwinShield) | activation×activation; TwinShield is recoverable at m≤4, proven in the oracle |
| softmax, RMSNorm/LayerNorm, SiLU/GELU, RoPE, residual adds | TEE | nonlinear / cheap / position-secret |
| Sampling, scheduler math, noise schedules | TEE | secret state, trivially cheap |
| KV cache | TEE RAM (plaintext inside CVM), permanently | GQA + q8 make it affordable; see "Decode and the KV cache" |
| Logits | GPU produces masked, TEE unmasks | lm_head is just another masked matmul; sampling never leaves |
| VAE decode, vocoders, text encoders ≤2B, mel spectrograms, phonemization, image pre/post | TEE CPU | small-model rule: CPU-in-TEE is strictly stronger; take it whenever the budget allows |

Small-model rule concretized (≤ ~2B params q8 defaults to full CPU-in-TEE):

- Whisper large-v3 encoder+decoder: 1.55B → **CPU-in-TEE first**; ship CPU-only if RTF
  ≤ 0.5 at target concurrency and skip GPU offload for STT entirely (expected outcome).
- TTS (both candidates, below): well under the line → CPU-in-TEE.
- SD VAE (~50–100M), CLIP-L/G (~0.1–0.7B): CPU-in-TEE.
- T5-xxl (4.7B, Flux/SD3 text encoder) is the one pre/post component over the line:
  masked-offload it, or serve SD3-medium's no-T5 degraded mode where acceptable. Decide
  per catalog model in Phase 3.

## Decode and the KV cache (the part no paper wrote down)

Every source paper stops at a single forward pass. TwinShield's benchmarks are perplexity
and encoder workloads; Slalom is CNNs; Amulet is classification and full-sequence scoring;
KV-Shield names the cache but assumes secret weights. None of them state where a KV cache
lives, how masks compose across decode steps, or what a cache does to integrity. That gap
is the reason this section exists, and the short version is: **the natural constructions
are provably inadequate at decode, so the answer is not a cleverer mask — it is a placement
rule plus the architectural facts that make it affordable.**

### Why decode attention cannot be offloaded (proven, not asserted)

The two attention products are `s = q·K^T` and `o = p·V`. Both are activation×activation,
so Slalom does not apply (its whole leverage is that W is public and fixed, making `r·W`
precomputable). The cited option is TwinShield's OutAttnMult. It fails here, and the
failure is structural:

At decode m=1. The GPU receives two rows — `u = q + R_q` and `v = a·R_q` — in secret order.
Whichever the order, `q = u − c·v` for a single unknown scalar `c = a^-1`. **q is therefore
confined to a line in Z_p^d.** The paper's security accounting (log(d·(2m)!) bits) counts a
brute-force space, but no brute force is needed: real activations are small while `u − c·v`
for wrong `c` is uniform over the field, so enumerating plausible values of *one* coordinate
pins `c` to a few thousand candidates and any second coordinate filters to a unique answer.

The oracle implements this attack and it recovers q exactly. It also recovers **m=4**, which
matters because 4 is a real GQA group size — the obvious rescue of batching a decode step's
query heads against their shared KV head does not work. Measured search space (pairings ×
plausible-scalar candidates): m=1 → 14 bits, m=4 → 24 bits, m=8 → 42, m=16 → 86,
m=512 → 4907. **Safe regime is prefill/batched only, at m in the hundreds.** Decode
attention stays in the TEE permanently; this is not a v1 simplification to revisit.

Inventing a construction to close the gap is out of scope by instruction ("no custom
cryptography beyond the cited constructions") and would be reckless in a TCB regardless.

### What we do instead, and why it is not a compromise

Keep attention and the KV cache in the TEE; offload every linear. That still offloads the
large majority of decode FLOPs under a sound, cited construction. Measured on real geometry
(`capacity()` in the oracle), attention is **6.7% of decode MACs at 2k context and 22% at
8k** for Llama-3-8B — so masked offload of the linears alone covers 93% / 78% of the work.

The affordability of TEE attention rests on an architectural fact, not on optimism: decode
attention is **bandwidth-bound, not compute-bound**, and GQA cuts the bandwidth by the group
ratio. Per-token KV streamed at 8k context, q8: Llama-3-8B (GQA 4:1) 537 MB → ~9 ms at
60 GB/s; Llama-2-7B (MHA 1:1) 2.1 GB → ~36 ms. That is the whole policy:

- **GQA is a shielded-tier catalog requirement.** MHA models are admitted only at short
  context buckets, or not at all. This is a concrete, measured admission rule, not taste.
- Context buckets {2k, 8k} for GQA models; 32k is where GQA hits the same wall MHA hits at
  8k (attention becomes 53% of MACs, ~36 ms/token streaming) and needs its own decision.
- q8 KV is a **capacity lever with a throughput cost**, not a free win. Measured on this
  box: CPU decode at 8k drops 36% for the 8B (10.48 -> 6.72 t/s) and 45% for the 1.5B,
  against a 47% KV memory saving; on GPU the cost is only 3-4%. Since shielded KV lives in
  TEE RAM on the CPU path, that is the expensive side. Spend it for concurrency/context
  reach, not for speed.

### The cache changes the integrity rules

K and V arrive from an **offloaded** matmul and then persist for the rest of the session.
This is qualitatively different from a corrupted activation: a bad activation costs one
token, a **bad cache entry poisons every future token that attends to it**. So:

> KV-producing matmuls are verified strictly, per step, before insertion. No deferral, no
> batching of the check across steps.

Other matmuls could in principle defer verification to amortise; KV projections may not.
The oracle exercises this: a worker that tampers only with the key projection is caught and
the request aborts with nothing inserted. No source paper states this rule because no source
paper has a cache.

### Mask lifecycle across steps

One-time means one-time, and decode consumes masks continuously. Per-token mask
consumption is ~3.7 MB of unblinding factors for Llama-3-8B, ~11.4 MB for a Qwen3-32B-class
model (measured in the capacity model). The bank is per-(model, shape-bucket), shared across
sessions, with a strict monotonic issuance counter.

> **Bank exhaustion stalls the request. It never wraps.** A wraparound is not a slowdown, it
> is OTP reuse across two activations, which hands the adversary their difference. The
> oracle asserts both halves (no reuse, exhaustion raises).

Masks for step t+1 are staged during the GPU compute of step t, so mask handling never sits
on the token critical path.

### What actually binds throughput

Three ceilings, and the honest answer is that they are all in the same order of magnitude:

- **TEE serial** (bandwidth-bound): KV streaming for attention, ~9 ms/token at 8k GQA →
  ~110 tok/s.
- **TEE background** (throughput-bound): mask refill costs `u = r·W`, i.e. *exactly* the
  linear MACs the GPU performs, multiplied by the number of RNS channels. **Now measured
  (`shielded/bench/refill_bench.py`) and it is worse than this document previously
  estimated.** On 16 physical EPYC 9115 cores the best *verified-exact* GEMM is fp64 at
  159 G-MAC/s, which sustains only ~7 tok/s for an 8B model at RNS-3. bf16 is faster but
  the exactness probe says it is **not** exact, so that rate is unusable. The path that
  rescues it is int8 via `torch._int_mm` (FBGEMM/oneDNN, AVX-512 VNNI), verified exact for
  byte primes and roughly an order of magnitude faster than fp64 — so refill is a solved
  *engineering* problem, not an open research one, but it must actually be built into the
  TEE side. Current figures in `shielded/REPORT.md`.
- **GPU**: linear MACs in field arithmetic with limb inflation.

**Refill is the binding constraint.** The GPU leg costs 2.6-3.6x fp16 and TEE attention
costs ~9 ms/token at 8k, but refill sets the sustained ceiling. Two things have to be true,
and neither is optional:

1. **An int8/VNNI GEMM on the TEE side.** This is the single highest-priority engineering
   dependency of the whole tier, ahead of the GPU kernel, which is already fast enough.
   The primitive exists and is verified exact (`torch._int_mm` via FBGEMM/oneDNN); what does
   not exist is its integration into the executor.
2. **A wide CVM.** Refill parallelises cleanly, so a 64-128 core fleet box is worth 4-8x the
   16-core development machine. Ceilings are reported per-physical-core in
   `shielded/REPORT.md` precisely so they extrapolate.

Model-size policy follows directly: 1.5B-class models are comfortable today, 8B is viable
with VNNI or a wide CVM, and 32B-class models are out of reach on anything resembling this
hardware. That is a catalog decision the measurement makes for us.

### The per-token loop

Per token, v1, with installed per-segment graphs on the worker:

1. TEE: embed token (lookup), RMSNorm.
2. Per layer: mask x → `SET_TENSOR` + doorbell (fire-and-forget) → blocking `GET_TENSOR`
   of masked {Q,K,V} (one fused round trip — QKV share the input upload); TEE: unmask,
   verify, RoPE, append KV, attention core, norm; masked round trip for O-proj; TEE:
   residual, norm; masked round trip for up+gate (shared input); TEE: activation; masked
   round trip for down; TEE: residual. **4 blocking round trips per layer.**
3. lm_head masked round trip; TEE: unmask, verify all checks for this token, sample,
   stream.

Budget at 7B/32 layers: ~128 blocking RTs/token; at 50–150µs CVM↔host loopback RTT that is
a 6–19ms/token transport floor plus ~5–10MB/token of masked activation traffic (fits
loopback/virtio comfortably). MEASURED 2026-08-26 (`shielded/REPORT.md` §11): the engine
backend does ~2 exchanges per layer (gate+up share one, down is one; the attention
projections stay on the CPU below a size floor), one frame each way, and the CVM reaches
the worker over AF_VSOCK rather than slirp — Qwen2.5-0.5B decodes at ~100 tok/s on
metal0's deployed app, 154 tok/s on the host loopback. Refill is off the critical path
entirely (a background-filled pad pool), which this section's budget assumed and the
first engine implementation did not do. Ceiling ≈ 40–100 tok/s before GPU compute — against the
kill criterion (≤5× vs baseline at batch ≥4) this is tight but credible, and batching
amortizes RTs across concurrent requests. Amulet's two-RT-per-request discipline is the
bar we hold prefill and image-gen to; decode is structurally per-token and the brief
accepts that. Every RT rides a pre-installed graph (`GRAPH_RECOMPUTE`-style doorbell, no
topology resend) — which requires the TEE-side op classifier to be **deterministic across
steps**, because `ggml_backend_sched` reallocates (and would force full graph resends) if
per-node backend assignment wobbles between steps.

Prefill: one masked round trip per linear op over the whole m×d prompt matrix (m = padded
bucket length) — Freivalds batching makes verification ~1 mult/element; this is the
friendly regime, as is diffusion (per-step batched denoiser, seconds-tolerant users).

## Dealt pads: the mask work off the trusted half

Everything above has the trusted half mint its own pads: `u = r·W` for every
row it will ever mask, three residue planes of the model's MACs, on cores,
inside the boundary. Measured on a 27B (2026-09-06) that is the binding
resource for every shape of load: one chat keeps the refill threads 85%
busy through speculation, prefill rides the same pads, and the accelerators
sit at a few percent. The dealt-pad construction moves that work to a party
that never sees `x + r`, and keeps the masking construction exactly as it is.

**Parties.** The *dealer* holds a seed the trusted half also holds, derives
`r` from it, mints `u = r·W` in batch on any hardware, and ships `u`
encrypted to the trusted half's pad key. The *accelerator* still sees only
`x + r`. The *trusted half* derives the same `r`, masks, unmasks with the
shipped `u`, verifies every product as before (SECURITY.md section 5), and
never mints. The *platform* keeps a consumption ledger. Privacy now rests
on the dealer and the accelerator not colluding (the dealer knows `r`, the
accelerator knows `x + r`, neither knows both); integrity rests on the same
Freivalds check as before plus an import-time check of each pad.

**The pad.** `r = ChaCha20(seed, (group << 48) | (index << 24) | block)`
reduced to `[0, M)` exactly as the mask bank draws; `u` ships as 3 bytes per
column (`M ~ 2^23.8`), 12.6 MB per token row on the 27B, 1.9 MB on the
0.8B, incompressible. A shipment (`.pads`) is row-major by index with one
ChaCha20-Poly1305 cell per `(index, group)` under a per-shipment key boxed
to the consumer's X25519 pad key; the operator's bank sees ciphertext and
sizes.

**Single use across reboots.** The consumer reserves a window `[mark,
mark + W)` from the platform's ledger and the mark advances before the
window is signed and returned; a restart resumes at the mark. The ledger
is not on the operator's box, so it cannot be rolled back.

**Integrity.** The product check is unchanged and already covers a wrong
`u`. With `SHIELDED_PAD_CHECK` the consumer also checks each imported cell,
`(u·s) == (r·(W s)) mod M` with a per-node random `s` and one weight pass at
registration, so a wrong dealer is refused at import and named, before any
use, instead of surfacing later as a worker failure.

**What binds throughput now.** The dealer's egress and the consumer's link
(rows per second), the bank (burst), the exchange chain (latency), and
memory (context). Speculation costs rows, so plain decode is the
bandwidth-efficient mode; prefill needs a pad per prompt token, so the bank
is sized for the largest prompt burst.

**Status.** `shielded/dealer/PLAN.md` is the design and log. Built:
`shielded-pads.{h,c}` (format, derivation, reader, writer, ledger window),
dealt mode in `shielded-tee.c` (`SHIELDED_PAD_*`), `shielded-dealer` and
`shielded/dealer/dealer-loop.py` (mint ahead of the mark, push to the
platform), `relay/pads.mjs` (seeds, ledger, shipment store; `/v1/pads/*`),
the Pixel anchor's pad key, seed, windows and bank. Verified: the 0.8B on
CPU end to end (exact text, no local minting), and twice on a Pixel 8 Pro's
protected VM against a workstation hub (2363 nodes offloaded, 0 local, 0
verification failures, the same text). For a CVM tenant the manager keys are
`nnShieldedPad{Source,Seed,SeedId,Sk,Ledger,ModelDigest,Window,Check}`,
seed and sk as deployment secrets; the bank transport for a metal box is
not built yet.

### The CVM tier as a consumer: bank over HTTP, windows from an agent

A metal box's trusted half has no phone app to stream shipments and relay
windows, so the engine does both itself over plain HTTP (no TLS in the
enclave; shipments are ciphertext to the pad key and windows are signed, so
the transport carries no trust):

- `nnShieldedPadSource: "http://<bank>/v1/pads/shipments"` - a bank that
  speaks the platform store's shape (the relay itself, or the operator's NVMe
  box on the LAN). The engine fetches shipments into `nnShieldedPadCache`
  (default `/tmp/shielded-pads-<seed_id>`) in index order: everything covering
  the current ledger window plus one ahead regardless of budget, then up to
  `nnShieldedPadCacheMb` (default 2048) further. Spent shipments (wholly below
  the lowest live group cursor, never a window edge alone) are unlinked
  (`nnShieldedPadPrune`, default on for a bank cache, off for a directory).
- `nnShieldedPadWindowUrl: "http://127.0.0.1:<port>/window"` with
  `nnShieldedPadLedgerPk` (64 hex): the engine POSTs `{want, seed_id}` to a
  loopback agent and verifies the answer's Ed25519 signature against the
  pinned platform ledger key, exactly as the pVM verifies windows relayed by
  its owner app. `shielded/dealer/window-agent.mjs` is that agent: `--relay`
  mode signs `/v1/pads/reserve` requests with the tunnel's transport key,
  `--ledger` mode is a self-signed local ledger for tests and single boxes.
  The two knobs are refused separately: an agent without a pinned key could
  sign its own windows. On a metal box the guest agent (`metal/guest/agent.mjs`)
  is that agent: it mints an X25519 pad key in-CVM (published in the RAD
  document, recorded by the relay at attach), fetches and opens this box's seed
  after a successful attach, leaves `{seed, seed_id, sk, ledger_pk, window_url}`
  root-only at `/run/enclave/pads/bootstrap.json`, and serves
  `POST /pads/window` on its loopback RAD port with a reserve signed by the
  P-256 transport key. The wasm manager fills a deployment whose
  `nnShieldedPadSource` is a bank and that names no keys from that file, so a
  tenant config carries only the bank URL; `nnShieldedPadSource: "platform"`
  means the platform's own store through the agent's loopback proxy
  (`GET /pads/shipments`, the store's two shapes only, so the engine never
  speaks TLS). With no bootstrap file yet the manager drops the dealt env
  entirely rather than open a half-configured link. The model digest pin
  comes from the calibration file (SHA-512, first 32 bytes, the dealer's own
  label) when the config names none (`test/metal-agent-pads.test.mjs`).

**The dealer daemon.** `GET /v1/pads/consumers` lists every attached tunnel
that offered a pad key with its seed id, ledger mark and whether it has asked
for its seed; `shielded/dealer/dealer-loop.py --all --master <hex> --push
[--worker host:port]` keeps each issued consumer's bank ahead of its mark
every pass, pruning spent shipments from the store. A consumer that never
asked for its seed gets nothing minted.

**The agent's per-boot token.** Tenants share the loopback namespace, so the
guest agent mints a token per boot, hands it to the manager in the bootstrap
file, and refuses a window or a receipt without it (`401`); the manager passes
it to the engine as `SHIELDED_PAD_AGENT_TOKEN`, sent as a bearer on those
POSTs. With dealt pads configured and no way to run them (no window, no
shipment, no worker) the engine fails the request rather than computing the
model in the clear, exactly as it does on pad exhaustion.

**Usage receipts from the CVM tier.** With a window agent configured the
engine also reports usage: after every reserved window (outside the pool lock)
and once more when the link closes, it POSTs the delta of cells consumed and
rows reserved since its last receipt to the agent's `/receipt` sibling of the
window URL; the guest agent signs it as this box (`enclave-pads-receipt`,
P-256) and relays it to `POST /v1/pads/receipt`, so the platform's per-seed
totals add up to exactly what the engine drew (2,033 pads / 56 rows on the
x86 proof, 7 receipts). A missing route is logged once and never blocks a
window. The pVM signs its own receipt on the control channel instead.

**Minting at GPU speed.** `shielded-dealer --worker host:port` mints through a
worker the dealer owns: with `SHIELDED_ZERO_PADS=1` every ring pad is zero, so
the "masked" planes carry the seed's mask row r itself and the exchange returns
r·W mod M, which is u. The dealer checks each row against the worker with the
same identity a consumer uses for a shipment, (u·s) ≡ (r·(W s)) mod M
(`SHIELDED_PAD_CHECK=1` builds the vectors), because the field-range product
check cannot hold for a 24-bit input. Never point it at an operator's worker:
that worker would learn every mask it later unmasks. The dealer never sees a
tenant's x either way. On the 0.8B, 16 rows through the CPU reference worker
are byte-identical to the in-process mint on all 1,552 cells
(`dealt-selftest` with `SHIELDED_WORKER=`, in `test/shielded-cbackend.test.mjs`);
`dealer-loop.py --worker` passes the address through. For the 27B this turns
minting from a 30 GB weight stream per row on CPU cores into a batched GEMV on
whatever GPU the platform dealer owns.

Proved on x86 against the local hub: four 16-row shipments in the store,
8-row windows from the agent, 16 tokens: 2363 nodes offloaded, 0 local, 0
missed, pad check clean, text identical to self-minting; shipments dropped at
floors 20 and 36 and the cache never held more than its 64 MB budget
(`test/pads-bank-client.test.mjs` pins the fetcher and the agent without a
model).

## Worker protocol (hardened ggml-rpc derivative)

Transport: TCP loopback CVM↔host (repo has no vsock anywhere; house pattern is TCP +
derived-token auth). Framing follows the nn-arbiter lesson: compact, byte-pinned frames
with raw-bytes tests (`wasm_manager.py:1917` — wire format is not style). Auth: bearer
derived HMAC-style from `SECRET` per the `X-Vmmgr-Token` pattern — this gates GPU
consumption; it is *not* a security boundary for confidentiality (masks are). Nothing
secret crosses the link by design, so no TLS in v1; revisit if the worker ever moves off-
box.

Command surface vs stock ggml-rpc (proto 5.0.0):

| Command | Fate | Note |
|---|---|---|
| HELLO | keep | version + capability pinning (op count static_assert stays); 1.3: `u32 major [u64 reserve_bytes]`, the tenant's VRAM reservation, refused if the budget or the driver cannot give it; the reply names `vram_budget`, `vram_reserved`, `vram_reserve`, `vram_free` |
| ALLOC_BUFFER / FREE_BUFFER / GET_ALIGNMENT / GET_MAX_SIZE / BUFFER_GET_BASE / GET_DEVICE_MEMORY / DEVICE_COUNT | keep | allocation plane |
| SET_TENSOR | keep | the only inbound data path: field-form weights at load; masked activations at run |
| GET_TENSOR | restrict | readable only from declared output tensors of installed graphs; stock allows arbitrary region reads of any live buffer |
| GRAPH_COMPUTE | replace | becomes GRAPH_INSTALL: graph accepted only if every node's op ∈ allowlist {field-GEMM custom op, view/reshape/permute-meta, cpy} — the stock command is an arbitrary-op execution primitive |
| GRAPH_RECOMPUTE | keep | the per-step doorbell; the only compute trigger after install |
| SET_TENSOR_HASH + cache_dir | delete | persists request bytes to disk; nothing may persist |
| COPY_TENSOR / MEMSET_TENSOR / BUFFER_CLEAR | delete | server-side mutation primitives we don't need |
| buffer=0/data≠0 deserialize path | fatal | keep the create_node guard; any unvalidated buffer reference kills the connection |

Statelessness: per-request worker state is masked activations in VRAM only; weights (public,
field form) are the only long-lived residents. Crash/restart of the worker loses nothing
secret and the TEE simply re-verifies on reconnect (weights re-uploaded or digest-checked).

The worker is not part of the measurement and runs no TEE — it can be replaced wholesale by
the operator, which is exactly the point: its honesty is enforced by Freivalds, not by
attestation. On fleet boxes it must coexist with existing tenants: it takes an MPS slice
and (when co-resident with the wasi-nn arbiter) an arbiter-client turn per compute quantum,
same discipline as `worker/worker.py` children. v1 runs it single-tenant on a dedicated
commodity box.

## Repo integration

- `shielded/` (new top-level): the worker. Dockerfile with digest-pinned CUDA base (TCB
  comment per `worker/Dockerfile`), README, systemd units for the off-fleet/self-hosted
  mode (precedent: `metal/`). Registered in `scripts/release.sh` (CONTEXT/ORDER) and the
  `deploy.yml` detect case, container block in flavor configs when it ships to fleet boxes.
- TEE-side executor: patch stack work in `wasm/` — a new `wasmtime-nn-ggml-shield.patch`
  (or an extension of the ELL shim in `wasm/llama-shim/`), added to `Dockerfile.wasmtime`'s
  ordered apply AND `.github/workflows/wasmtime-patch-check.yml`. Unlike every other patch
  in that stack, this one **fails closed** (a masking/verification fault must never fall
  back to plaintext offload or silent CPU divergence) — that inversion gets stated loudly
  in the patch header.
- Engine changes ride the ELL cascade: `llamacpp-toolchain.yml` → new `mm30+` tarball
  (never reuse a tag) → `ELL_URL/ELL_SHA256` repin in `Dockerfile.wasmtime` →
  `toolchain.yml` → `WASMTIME_IMAGE` repin in `Dockerfile.wasm`. Each repin is a
  measurement event. Dev loop before any repin: local llama.cpp/sd.cpp builds (the
  `wasm/sd-shim` symlink precedent).
- whisper.cpp is not in the tree today; it enters via `llamacpp-toolchain.yml` as a third
  pinned clone (it shares ggml), with a thin shim if the Rust FFI needs one.
- Orchestration: attach-a-shielded-worker rides the existing `/vms` contract as optional
  fields on `POST /vms` (supervisor `launchSpecFrom` → wasm-manager record), not a new
  endpoint. New fleet flavor `enclaves/gpu-shielded/` for boxes where the GPU is NOT
  passed through into the CVM (the existing gpu flavor's passthrough topology is the
  opposite trust shape).
- Tests: `test/shielded-*.test.mjs`, pure functions + `*_SELFTEST` seams; wire-format
  assertions on raw bytes; the field-GEMM and mask/unmask cores get exhaustive
  small-dimension exact tests against a reference big-int implementation.
- Docs: this file is the design; `SECURITY.md` (final deliverable) carries the per-op
  leakage arguments and per-interface residual-leakage sections.

## Reference oracle

`shielded/reference/shielded_ref.py` is the executable form of everything above, driven by
`test/shielded-reference.test.mjs` (`python3 shielded/reference/shielded_ref.py --verbose`
for the human-readable dump). It is a correctness and security oracle, not a performance
model: toy dimensions, pure numpy, no CUDA. It exists so the engine has something to be
validated against, and so the security argument can be re-run instead of re-read.

What it establishes today:

- Fixed-point field round-trip within one quantum; exact field matmul; the int64 bound.
- Slalom offload recovers **bit-exactly**, and the worker's transcript never contains a
  plaintext input.
- Mask bank: no reuse, and exhaustion raises rather than wrapping.
- Preprocessed Freivalds catches a **single-element** lie in 64/64 trials with no false
  positives, at 40 bits of soundness per check.
- TwinShield recovery at m ∈ {1, 2, 4}; search-space table out to m=512.
- A 3-layer GQA decoder generating 12 tokens with tiered placement produces output
  **identical** to the same arithmetic run entirely in-TEE, across 352 boundary crossings.
- Leakage assertions against the adversary transcript: uniformity (chi-square 54.4 vs 117
  threshold) and pooled correlation 0.0007 against a 3-sigma null of 0.019. Per-tensor
  correlations are reported with their null bound so small-sample noise is not misread as
  a leak — a trap this suite hit and now guards against.
- KV poisoning caught before insertion.
- Field scaling vs width and outlier magnitude; RNS exactness at d=4096.

## Per-interface notes

**Chat.** GGUF models, streaming decode as above. Output equivalence: bit-exact in field
domain vs TEE reference; documented tolerance vs fp16 baseline. KV q8 in TEE RAM; context
buckets {1k, 2k, 4k, 8k} padded.

**STT.** Log-mel in TEE (trivial). **MEASURED and settled: whisper large-v3 q8_0 runs
CPU-in-TEE at RTF 0.168 single-stream (3x margin) and passes at 3 concurrent streams per
16-core CVM.** STT therefore never touches the GPU, and its accelerator-side leakage surface
is nothing at all rather than bucketed. Config: q8_0 (2x f16's throughput at equal quality
here), -t 16 per stream, cap 3 streams, ~2.7 GiB RSS each; never -t == nproc. Duration
buckets still apply to the relay/timing surface. Caveats: bare metal (no SEV-SNP memory
encryption overhead) and only 16% margin at N=3.

**TTS.** Pick: **Pocket TTS (kyutai) via llama.cpp's native mtmd path**, with
**Qwen3-TTS-12Hz-1.7B as the quality/multilingual fallback** — same code path. Evaluation
(2026-08): llama.cpp's old OuteTTS demo is gone; TTS is now a first-class mtmd audio-out
capability (Qwen3-TTS merged 2026-05, Pocket TTS 2026-08-07 with the SEANet
transposed-conv→GEMM optimization that halves CPU decode cost — exactly our constraint).
bark.cpp is dormant (last push 2024-11); TTS.cpp (mmwillet) has the best model coverage but
requires a forked ggml patch stack and has no Linux/CUDA story — disqualified for a pinned
fleet engine. Pocket TTS is ~0.2–0.25B total (acoustic + Mimi/SEANet decoder) and
Qwen3-TTS is 1.7B + ~0.4B code2wav — both under the small-model rule ⇒ **TTS is
CPU-in-TEE by default**, streaming, TTFA target <1s measured in Phase 5. Voice-cloning
reference audio is secret input, same handling as prompts (it only ever exists in-TEE).
Note: the current ELL pin (llama.cpp ddd4ec14, mm29) predates Pocket TTS — TTS integration
requires an engine bump, i.e. a measurement event; schedule it with the Phase-4 mm30 repin,
not before.

**Image generation.** DiT first (SD3/Flux class): the denoiser is a transformer and reuses
the masking path unchanged; per-step masked offload, steps batch well, users tolerate
seconds — friendliest interface to RT overhead, held to the two-RT-per-step discipline.
UNet (SDXL) second via conv masking (Slalom lineage; Amulet's conv coverage is
weight-hiding, wrong direction — bar only). Text encoders per small-model rule (T5-xxl
exception above). Noise schedule + final sampling in TEE; VAE decode in TEE. Leakage:
resolution buckets {512, 768, 1024}, step-count buckets {≤4 (turbo), 20, 28, 50}.

**Vision.** Preprocess (resize/patchify/normalize) in TEE; ViT encoder via the masked
backend (additive masks — permutation equivariance covers ViT correctness, but public
weights forbid bare permutation as protection); projector is a linear offload; decoder
shares chat. The input image and everything derived from it is secret.

## Phases

Status as of 2026-08-14. "Modelled" means arithmetic on measured primitives, not an
end-to-end run: no engine exists yet, so no phase is closed in the shipping sense.

0. **Baselines** — PARTIAL. GPU field-GEMM ladder and CPU refill rate measured on an
   RTX 3070 + EPYC 9115 (`shielded/bench/`). Chat and STT engine baselines were run
   separately; see `shielded/REPORT.md`. Still missing: a stock ggml-rpc remote-GPU run to
   isolate transport cost, and anything on real fleet hardware. **Transport is no longer
   modelled**: a masked exchange over the host<->guest loopback measures 0.56 ms warm
   (REPORT.md §10.1).
1. **Masked backend v1** — **BUILT AND PROVEN, in reference form.** Constructions proven in
   the oracle; worker admission rules implemented and enforced over a real socket
   (`shielded/worker.py`, `test/shielded-gpu.test.mjs`); kernel choice settled by measurement
   (RNS-3 int8); the TEE side, the wire protocol, the mask bank, the refill and the
   verifier all exist and run (`shielded/tee.py`). A real model runs end to end and matches
   an in-TEE reference exactly. NOT built: the C++/CUDA fleet worker, and the sched-pinned
   executor inside wasmtime -- `shielded/model.py` is its specification, not its
   replacement. The KV-Shield permuted pipeline was analysed and deliberately not built --
   it is void against public weights, so it would validate plumbing while teaching a wrong
   habit.
2. **whisper.cpp / sd.cpp (DiT)** — PARTIAL. Conv masking and a ViT block are oracle-proven;
   the STT CPU-in-TEE feasibility measurement is in `shielded/REPORT.md`. sd.cpp itself is
   untouched.
3. **TwinShield v2** — ANALYSED + BOUNDED. Prefill offload implemented and exact at m=64
   and m=256; decode offload proven unsafe and permanently excluded. Softmax offload
   remains refused pending the real-vs-field numeric analysis. The adversarial review this
   phase was gated on has effectively happened, and its outcome was to shrink the phase.
4. **TTS + SDXL conv** — PARTIAL. TTS pick made (Pocket TTS via mtmd) and blocked on an
   mm30 engine bump; the conv masking path it shares with SDXL is oracle-proven.
5. **Final report** — `shielded/REPORT.md` (§10 is the end-to-end revision), plus
   `shielded/SECURITY.md` for the per-op leakage argument and per-interface residual leakage.
6. **Self-hosted exposure** — **DONE for metal.** `metal/` can put a card on the untrusted
   host and let the CVM use it: `shieldedWorker` in the box config launches the worker on
   127.0.0.1, the guest reaches it at 10.0.2.2, and `metal/guest/shielded-probe.mjs` runs one
   real masked GEMM at boot and asserts exactness, verification, lie rejection and denylist
   enforcement before the box advertises the path. Proven on a live SEV-SNP guest
   (REPORT.md §10.5). Capacity (protocol 1.3, REPORT.md §13.14.2): `shieldedWorker.vramGb`
   is the part of the card dedicated to Enclave and is the card the fleet sees; the worker
   holds none of it at start-up, a tenant reserves its share at HELLO (the manager exports
   `SHIELDED_RESERVE_BYTES`, the same bytes the share was sized from) and gets it back to the
   driver at disconnect, and the box advertises `min(budget - reserved, driver free)` plus
   `vramReservedGb`. A HELLO the card cannot honour is refused and the tenant computes in
   its enclave until it can.

Kill criteria (from the brief, unchanged): chat/vision >5× at batch ≥4 after optimization;
image gen >3× per-image wall clock at batch ≥4; STT/TTS failing realtime on both CPU-in-TEE
and masked-GPU paths; any design requiring GPU-driver/host-kernel/operator trust is dead on
arrival. On any kill: stop and write up why, with measurements.

## Open risks, ranked

0. **Mask refill on the TEE side is the tier's throughput ceiling.** It costs one TEE MAC
   per GPU MAC and cannot be offloaded (a GPU computing `r·W` learns the pad). The fast
   exact primitive exists (int8/VNNI, verified) and is now integrated and exactness-probed
   (`tee.refill`, one stacked GEMM over the three residue planes). Figures:
   `shielded/REPORT.md`.


1. **Field-GEMM throughput on commodity GPUs** — the whole tier's economics. int8-limb
   kernels are the load-bearing bet; Phase 1 measures before more is built on it.
2. **Decode against a q4_K baseline sits at the kill line** — RNS-3 is 1.5x fp16 but 6x
   q4_K on weight bandwidth, and q4_K is what the fleet serves. Batching amortises it;
   batch-1 decode does not. Needs the bespoke small-M kernel (int8 TC refuses M<=16).
3. ~~**Field magnitude guard**~~ — **RESOLVED 2026-08-25, and the margin was not there.**
   At the design's fixed `l = 8`, `ffn_down` on a real model reaches 1.81× M/2 and WRAPS
   (REPORT.md §10.2). Two changes, both implemented: detection is now Freivalds over the
   integers modulo an unrelated prime, which catches a wrap and a lying worker in the same
   two dot products (a mod-M check cannot catch a wrap at all — the wrapped value is
   congruent); and prevention is **outlier splitting**, where the TEE keeps the top-k
   activation channels and computes their contribution in int64. `k = 4` takes ffn_down from
   1.81× to 0.12×, at 0.08% of that site's multiplies. The activation exponent becomes a
   per-site public constant calibrated offline (`shielded/calibrate.py`), never adapted per
   request — an adaptive one would leak activation magnitude.
4. **TwinShield v2 OutSoftMax numerics** — the real-valued exponential versus field masks
   is unresolved in the paper; softmax stays in-TEE until analysed. v2's attention half is
   now settled as prefill-only by measurement rather than by review.
5. **CPU attention at 32k context** — resolved for 2k/8k GQA by the capacity model (~9 ms/
   token at 8k); 32k is where GQA models hit ~36 ms/token and 53% of MACs, and needs its
   own bucket decision before any 32k shielded offering.
5b. ~~**Field-form weights inflate VRAM ~5x**~~ — **CLOSED.** Weights stay in native q8_0
   and the residues are derived in-kernel; verified exact, and measured end to end at 501 MiB
   resident for a 0.5B model. The same identity applies on the TEE side: a byte-limited weight
   satisfies `w mod q_i == w` for every prime, so one int8 plane serves all three channels
   there too.
6. **T5-xxl** breaks the tidy "encoders in TEE" story for Flux-class models.
