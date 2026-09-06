# Enclave Metal: self-hosted TEE enclave server

Metal runs the exact same enclave stack as the hosted fleet (supervisor +
wasm-manager, the same digest-pinned images) inside a **self-launched
confidential VM** on hardware you own. No external controlplane, no vendor
shim, no hosted vault: every service the managed fleet gets from its provider
is replaced by an auditable first-party equivalent in this directory.

It targets **any TEE, CPU and/or GPU**: AMD SEV-SNP today (QEMU
`sev-snp-guest`), Intel TDX by swapping the launch object (`tdx-guest`, same
image, same agent, because the guest attestation driver is the configfs-tsm
abstraction, which serves both), and NVIDIA confidential computing by GPU
passthrough into the CVM (the GPU flavor's worker/MPS containers ride along
unchanged; CC attestation still comes from NVML inside the guest).

**See also:** [PROTOCOL.md](PROTOCOL.md), the permissionless protocol for
anyone to sell hosting on enclave.host anonymously (attestation-gated attach,
chain-escrowed runner payout to the seller's own wallet, keyless relay).
[HANDOFF.md](HANDOFF.md) covers what is live and the operator-gated
production steps.

```
host (untrusted)                       guest CVM (trusted, measured)
┌──────────────────────────┐   ┌───────────────────────────────────────┐
│ systemd: enclave-metal   │   │ /init (busybox, in the launch digest) │
│   └─ QEMU -object        │   │   ├─ wasm-manager  (chroot, :8091)    │
│      sev-snp-guest       │──▶│   ├─ supervisor    (chroot, :8080)    │
│      kernel+initrd+      │   │   └─ metal-agent   (:8443)            │
│      cmdline measured    │   │       ├─ in-CVM TLS front             │
│                          │   │       ├─ RAD: /.well-known/           │
│  (no disk, RAM only)     │   │       │   enclave-attestation         │
└──────────────────────────┘   │       │   (configfs-tsm SNP report,   │
                               │       │    report_data = TLS pubkey)  │
        outbound WSS only      │       └─ fleet-tunnel client ─────────┼──▶ api-relay
        (CGNAT-safe)           └───────────────────────────────────────┘   (enclave.host)
```

## Staying current (auto-update)

Tinfoil enclaves are PUSHED: merge to main, CI cuts a release, `tinfoil-cli`
moves each running enclave. A metal box usually has no inbound path at all
(CGNAT), so it PULLS the same artifact on a timer instead:

    cp metal/systemd/enclave-metal-update.* ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now enclave-metal-update.timer

    node metal/update.mjs --check     # what it would do, changes nothing
    node metal/update.mjs --force     # ignore the idle policy (still health-gated)

It tracks the newest published release for this box's flavor (`vX.Y.Z-cpu` for a
CPU box) — the same artifact the CPU fleet moves to — and builds it in a
throwaway git worktree at that tag, never from the working tree this box runs
out of. That tree is also somebody's desk, and an enclave must not attest to
half-finished work.

**It stages and rolls back.** The new image is built beside the live one and
swapped in; if the box does not answer `/v1/health` with a fresh watcher inside
`healthGraceSec`, the previous image goes back and a halt marker
(`metal/.update-halted`) stops further updates until a human clears it. This is
not hypothetical: on 2026-07-27 a manager build reached metal0 whose wasmtime
did not know a flag that build passes unconditionally, every tenant died at
spawn, and the box handed back the lease it had just resumed. An unattended
updater without this gate would have done that at 3am and left it there.

**It updates as soon as a release exists.** A restart relaunches every tenant
and re-issues every app-zone certificate — the guest is initramfs-only, so
nothing survives, and that is intended. Until 2026-09-04 the default waited
for the box to be idle, because those issuances were rationed: Let's Encrypt
allows 5 duplicates per 168h per name. App-zone names now go to the platform
certificate service instead (`CERTS_API`, derived from `relayUrl`), which
orders from ZeroSSL under the platform EAB and keeps Let's Encrypt behind it,
paced centrally — so a restart no longer spends anything scarce, and a box
should not sit on an old release for a reason that expired.

A box that would rather wait for a quiet moment sets
`autoUpdate: { "onlyWhenIdle": true }`; `maxDeferSec` (6h) then caps the wait
so a permanently busy box still takes fixes.

Note the measurement changes with the image, by design. While the relay's
`METAL_ALLOWED_MEASUREMENTS` allowlist is empty (token-only attach) that costs
nothing; once it is in use, the new release's measurement has to be published
before boxes move to it, exactly as for any other release.

## Why the trust story is *stronger* than the hosted fleet

A managed TEE provider measures the container config and images at its own
controlplane, but without a guest RTMR-extend a launched image's digest is
never folded into the hardware registers: the quote proves only that *some* CVM
of theirs is running, and a transparency log ties the config to the repo.
Metal's guest is a **single initramfs**: kernel, initrd
(which *contains* the supervisor and wasm-manager bytes, extracted from the
same digest-pinned images the fleet runs) and cmdline are all covered by the
SEV-SNP **launch digest** via direct-boot measured OVMF (`kernel-hashes=on`).
There is no unmeasured byte in the TCB: the quote alone proves the exact
supervisor code, with no transparency-log detour needed (the Sigstore record
remains as provenance, not as a trust root).

## Components

| file | role |
|---|---|
| `build-image.mjs` | **unprivileged** guest image build, reproducible when both `--supervisor`/`--wasm` are pinned by `@sha256:` (the manifest records `reproducible`): pulls the OCI images straight from ghcr (no docker), a pinned Arch kernel package, busybox; emits `dist/` (kernel, initramfs, cmdline) + `dist/manifest.json` with every input digest and the expected SNP launch digest |
| `oci-pull.mjs` | dockerless OCI puller (anon token, digest-verified blobs, whiteout-aware extraction) |
| `guest/init` | PID 1 in the CVM: mounts, virtio + TSM modules, DHCP, per-boot secret minting, starts the three services, restarts them, reboots on wedge |
| `guest/agent.mjs` | the metal-agent, the whole ingress side in-CVM: mints the in-CVM TLS key, gets the SNP report over configfs-tsm with `report_data[0:32] = sha256(TLS pubkey SPKI)`, serves the RAD, fronts the supervisor with TLS, and maintains the outbound fleet tunnel |
| `enclave-metal.mjs` | host-side launcher: builds QEMU argv (`dev` \| `snp` \| `tdx` mode), spawns, watches, restarts; serial console to the journal |
| `systemd/enclave-metal.service` | user service (`systemctl --user`), `Restart=always` |
| `verify.mjs` | first-party attestation verification: SNP report signature chain VCEK → ASK → ARK fetched **directly from AMD KDS** (`kdsintf.amd.com`), launch-digest comparison against `manifest.json`, TLS-key binding check. No third-party endpoints. Used by the CLI (`enclave attest` learns the `metal` RAD formats) |
| `volumes.mjs` | builds the attested read-only **model volumes** (ext4 + appended dm-verity hash tree, reproducible: same model tree in → same root hash out); needs no root |
| `guest/mverity.c` | static in-guest dm-verity setup over the raw device-mapper ioctls (no cryptsetup in a slim measured image) |
| `config.example.json` | host config: mode, cpus/ram, enclave name, relay URL, key paths, attached volumes |

## Model volumes (the self-hosted Modelwrap)

Large read-only weights — GGUF/ONNX LLMs, diffusion checkpoints, RAG corpora —
reach tenants the way Tinfoil's Modelwrap delivers them, without Tinfoil. Each
volume is **one host file**: an ext4 image of the model tree with a dm-verity
hash tree appended. The launcher attaches it as a read-only virtio-blk disk; the
guest brings dm-verity up **itself** and mounts it read-only, so every block a
tenant reads is hash-checked inside the CVM and a host that flips a byte gets an
I/O error instead of serving different weights.

```sh
sudo mkdir -p /vm/enclave-volumes && sudo chown $USER /vm/enclave-volumes   # once
node metal/volumes.mjs build qwen2.5-0.5b-gguf --src ~/models/qwen2.5-0.5b-gguf
node metal/volumes.mjs list
# metal/config.json:  "volumes": ["qwen2.5-0.5b-gguf"]   (or "*" for the whole store)
systemctl --user restart enclave-metal
```

**The volume set is signed by the CPU.** The launcher launches the VM with
`sha256(volume table)` in **HOST_DATA** (`MRCONFIGID` on TDX), which the hardware
stamps into every attestation report; the guest reads HOST_DATA back out of its
own report and refuses to mount **anything** whose table doesn't hash to it. So a
buyer can tell *from an attestation alone* which weights an enclave is serving —
the property Modelwrap gets by putting its verity root on the measured cmdline —
and `metal/verify.mjs` checks it for you (`HOST_DATA binds this exact
model-volume set`, then one line per volume with its root hash).

HOST_DATA rather than the cmdline is deliberate: it binds host-supplied config to
the quote *without* entering the launch measurement, so attaching a model does
not invalidate the release measurement that `dist/manifest.json` pins and the
relay's `METAL_ALLOWED_MEASUREMENTS` allowlists. Identity of the **code** and
identity of the **data** stay separable, which is what lets an anonymous seller
carry their own models and still attach permissionlessly (PROTOCOL.md gate 1).

Volume images are reproducible (fixed fs UUID, hash seed, `SOURCE_DATE_EPOCH`,
name-derived verity salt), so anyone holding the same model files can rebuild
the image and check the root hash the enclave attests to is the model they think
it is. `--gguf <file>` picks one quantization out of a multi-file tree, `--sd`
marks a volume that preloads through the stable-diffusion.cpp backend rather
than ggml; the guest passes both through to the wasm-manager as `MODEL_VOLUMES`
/ `MODEL_VOLUMES_SD`, which is how deployments then attach them by name
(console volume picker, or `volumes` in the deployment's config CID).

## Shielded GPU: sell the card without putting it in the enclave

A metal box can offer GPU work with the card **outside** the confidential VM, outside the
launch measurement, and outside the trust boundary entirely. This is the shielded tier
(`docs/shielded-inference.md`, `shielded/README.md`): the guest sends the GPU public weights
and one-time-padded activations, and verifies every product it gets back. The GPU's operator
is assumed hostile, so the GPU does not have to be trusted, attested, or passed through.

That is a different trust shape from the GPU flavor, which passes a card INTO the CVM and
relies on NVIDIA confidential computing. Shielded needs neither CC-capable hardware nor
passthrough, so it works with a consumer card — the reference measurements are on an RTX 3070.

```json
"shieldedWorker": { "port": 9500, "vramGb": 6.5 }
```

For multiple cards, use `shieldedWorkers` instead of `shieldedWorker`. Give each
worker a distinct GPU UUID (from `nvidia-smi -L`) and TCP/vsock port. Keep the array
order stable: it defines card IDs, and a live lease stays bound to its card.

```json
"shieldedWorkers": [
  { "device": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "port": 9500, "vramGb": 6.5, "computeShare": 0.5, "priceUsdHr": 0.05 },
  { "device": "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "port": 9501, "vramGb": 31, "priceUsdHr": 0.05 },
  { "device": "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc", "port": 9502, "vramGb": 31, "priceUsdHr": 0.05 }
]
```

Each worker receives its own `CUDA_VISIBLE_DEVICES`, probe, VRAM reservations,
and inference fairness queue. Volta/V100 (`sm_70`) is included in the CUDA worker's
default build. Rebuild the worker and update the measured guest image before
activating multiple cards. TCP/vsock are supported; the shared-memory ring is
currently single-worker only.

To sell all cards as **one GPU pool**, add:

```json
"shieldedPool": {
  "mode": "layers",
  "pricing": { "tflopUsdHr": 0.001, "vramGiBUsdHr": 0.0045 }
}
```

The GPU tag then shows combined **dedicated** VRAM and rated dense FP16 compute.
A 10% GPU share reserves 10% of each worker's VRAM budget and compute, with a
single fairness queue for pooled tenants. CPU admission remains independent.
The pool hourly price is `TFLOPS × tflopUsdHr + GiB × vramGiBUsdHr`, rounded once
to micro-USDC/second for the existing registry. It does not fluctuate with free
memory or measured throughput. With a 20.4 TFLOPS / 6.5 GiB RTX slice and
101.7 / 31 plus 113 / 31 V100 slices, the pool is 235.1 TFLOPS / 68.5 GiB at
$0.5436/hour. Per-worker `priceUsdHr` is ignored in pool mode.

The measured shielded backend distributes calibrated Q8_0 linear weights by
layer across the reserved cards. It keeps a whole layer together when it fits;
otherwise it places activation groups across cards and leaves overflow on CPU.
Members sharing an activation (q/k/v, gate/up) always use the same worker and
pad. Nonlinear operations, plaintext activations, KV, and verification stay
inside the enclave. Unsupported or uncalibrated weights remain on CPU.
Each worker link has its own reservation, transport and reconnection state;
refill threads share one process-wide budget. Aggregate rated TFLOPS describe
capacity, not the throughput of a single sequential inference.

All configured workers must pass their boot proof before the pool opens. A
failed worker withdraws **new pooled shares** until it recovers; existing tenants
fall back to CPU for that worker's weights while other links continue. Totals
and pricing remain stable. A restored pre-pool lease retains its original card
until resize/reclaim replaces its reservation. The manager probes the built
backend's pool capability before accepting pooled launches.

Without `shieldedPool`, `shieldedWorkers` retains the legacy independent-card
allocator and per-card sizing floor. Use equal `priceUsdHr` values in that mode,
since the registry has one GPU price per node.

`vramGb` is the part of the card dedicated to Enclave: the fleet sees a card of exactly
that size, and the worker refuses to start if the card is smaller. It is held only as
tenants reserve it: the worker takes no device memory for it at start-up, each tenant
reserves its share when it connects (protocol 1.3, refused if the card cannot give it, in
which case the tenant computes in its enclave and retries), and what a tenant reserved goes
back to the driver when it disconnects. The box advertises `budget - reserved`, capped by
what the driver says is free, so the number the fleet sees is what can still be sold.

With that set, `enclave-metal.mjs` starts the worker on `127.0.0.1:<port>` and supervises
it, and hands the guest the address over fw_cfg. It prefers the C++/CUDA worker when it
has been built (`make -C shielded/worker-cuda`; needs a CUDA toolkit and clang++ on the
host, and is what makes decode fast) and falls back to `shielded/worker.py` (needs
`python3` with torch and triton). The guest needs nothing but the measured image.

The guest reaches the worker two ways. Over slirp at `10.0.2.2:<port>`, which is what the
boot probe uses and what a guest without a vsock driver keeps using; and over
**AF_VSOCK** (host CID 2, same port), which the launcher enables by attaching
`vhost-vsock-pci` whenever `/dev/vhost-vsock` exists and the CUDA worker is in use. The
image carries the vsock modules and the engine backend tries vsock first, then TCP, so
there is nothing to configure; `"shieldedWorker": { "vsock": false }` turns it off. A
vsock round trip is a fraction of slirp's, and at ~50 masked exchanges per decoded token
that is the difference between ~100 tok/s and a few tens.

**Transport tuning, from config.** In the CVM the tier is transport-bound: a vhost-vsock
exchange costs ~152 us against 46 on the host loopback and ~10 for the socket itself, and a
decoded token makes ~49 of them (`shielded/REPORT.md` 13.13). Most of the difference is two
VM exits and two wakeups per exchange, the guest's from HLT via an injected interrupt. Two
knobs address the guest half, both host config, neither measured:

```json
"shieldedWorker": { "port": 9500, "vramGb": 6.5,
                    "tenantEnv": { "SHIELDED_SPIN_US": "150" },
                    "workerEnv": { "SHIELDED_WORKER_SPIN_US": "100" } },
"guest": { "haltpoll": true }
```

- `guest.haltpoll` (default **on**) loads `cpuidle-haltpoll` in the guest at boot: an idle
  vCPU polls for a bounded, self-tuning window (200 us ceiling, kernel defaults) before it
  halts, so a reply that lands inside the window finds the vCPU running and costs no
  interrupt injection. The cost is that window: every idle vCPU may burn its host core for
  up to 200 us after each piece of work before halting, and the window shrinks back to zero
  when wakeups stop landing in it. `false` restores plain HLT; an object
  `{ "ns", "growStart", "grow", "shrink", "allowShrink" }` sets the governor's parameters
  (nanoseconds). The whole policy rides fw_cfg as one string that PID 1 validates
  name-by-name before writing sysfs; the measured image carries only the driver.
- `shieldedWorker.tenantEnv` is `SHIELDED_*` environment for the tenant's engine, applied
  after the manager's own defaults so config wins. Only `SHIELDED_*` names survive: the guest
  drops everything else when it writes the verdict file, the supervisor drops it again, and
  the wasm-manager drops it a third time, because an operator may tune the backend the tenant
  already talks to but must not be able to put arbitrary environment into a tenant.
  `SHIELDED_SPIN_US` is the wire layer's bounded non-blocking poll before it blocks on a reply.
- `shieldedWorker.workerEnv` is plain environment for the host-side worker process.
  `SHIELDED_WORKER_SPIN_US` is the same poll on the worker's side, for the next request.

None of this changes what crosses the boundary: masked residue planes, public weights and
their field products, as before. It changes only whether a vCPU that is waiting for those
bytes halts or spins, and the host could always see when the guest went idle.

**Optional shared-memory transport.** A CUDA worker can also set
`"shm": { "path": "/dev/shm/enclave-shielded-card0", "mib": 32 }`.
Each worker needs a distinct absolute backing-file path; sizes are powers of
two from 8 through 64 MiB. The launcher attaches each file as an ivshmem BAR.
The guest validates the PCI device and BAR size, then exposes only that BAR at
a fixed path inside the native engine's chroot. WASI preopens are unchanged.
Pooled engines receive a separate mapping for each card and claim a free ring
per link. An absent mapping, occupied ring, or ring timeout retains the socket
path; malformed peer replies still fail verification. The message contents,
one-time pads and Freivalds checks are unchanged. This is off by default and
requires a launcher restart; benchmark the actual workload before enabling it.
Mapping paths and sizes come from the guest's device discovery, not `tenantEnv`.

**A dead worker never takes the box down.** It is restarted with backoff, and the enclave
keeps serving CPU work meanwhile. What tells you the path is healthy is the boot probe:

```
[gsup] shielded GPU OK: NVIDIA GeForce RTX 3070 at 10.0.2.2:9500 — exact=true
       verified=true lie_rejected=true denylist=true corr=-0.053 chi2=74.8
       rt=0.563ms warm (327ms cold, kernel compile)
```

`metal/guest/shielded-probe.mjs` runs one real masked field GEMM at boot and asserts four
things against the bytes that actually crossed: the unmasked product is exact; Freivalds
accepts the honest result and rejects a single-element lie; the worker refuses a denylisted op
on the wire; and the transcript is uncorrelated with the secret and uniform over the field.
It exits, it is not a service, and a failure is logged rather than fatal.

**The TEE-side backend is compiled from source during the image build**, not copied in. It
holds the one-time pads, the Freivalds secret and every plaintext activation, so it is the one
binary in the image whose provenance matters most — and as a committed `.so` it had two bad
properties: a reviewer approving a change to it saw only `Bin 76040 -> 145344 bytes`, and
anyone who compromised the workstation or toolchain that produced it could put code inside the
measurement without touching this repo's source. `build-image.mjs` now compiles
`wasm/ggml-shielded` against vendored ggml headers (`vendor/ggml`, text, reviewable) and the
`libggml-base.so` inside the digest-pinned engine image, records every source hash and the
toolchain in the manifest under `shieldedBuild`, and **fails the build** rather than falling
back — including when the vendored headers disagree with the engine the image ships, which is
the moment an engine repin would otherwise introduce silent UB. `metal/shielded-overlay/`
therefore carries calibration data only. The remaining gap is honest and recorded: the
compiler is *recorded*, not pinned, so two builders on different toolchains get different
bytes and different measurements.

**Why the worker's address is not measured, and does not need to be.** It arrives over fw_cfg,
which the host controls. A host that redirects it to a worker it wrote itself gains nothing:
the pad never crosses the boundary, and Freivalds rejects any product that is not the real
one. The worst it can do is refuse to answer — denial of service, which this design explicitly
does not promise to prevent. The GPU's address is ordinary configuration, not a trust anchor,
and that is exactly why the GPU can sit outside the enclave at all.

## RAD format

The metal-agent serves `/.well-known/enclave-attestation` (and the supervisor
finds it via the `RAD_URL` env override; on the hosted fleet that env is
unset and its own loopback attestation path is used, unchanged):

```json
{ "format": "sev-snp-guest-metal-v1", "body": "<base64 SNP report>",
  "certs": { "vcek": "...", "chain": "..." },
  "manifest": { "kernel": "sha256:...", "initrd": "sha256:...", "cmdline": "..." } }
```

Dev mode (TEE disabled in BIOS, plain KVM) serves `format:
"dev-unattested-metal-v1"`, which verifiers and the site MUST render as
**UNATTESTED (dev)**; it exists so the whole pipeline can be exercised before
the SEV BIOS toggle, and becomes `sev-snp-guest-metal-v1` with no other change
once the hardware is enabled. TDX guests serve `tdx-guest-metal-v1` from the
same configfs-tsm code path.

## Reachability: the fleet tunnel (CGNAT-safe)

Every existing relay→enclave path dials INTO the enclave's public endpoint; a
self-hosted box behind CGNAT has none. Metal inverts the transport: the
metal-agent dials **out** to `wss://api.enclave.host/v1/fleet-tunnel` and holds
a multiplexed channel; the api-relay routes that enclave's `/v1/*`, `/x/*` and
app-zone traffic over the tunnel instead of `proxyTo`. Identity: the agent
holds an ed25519 tunnel key minted in-guest at first boot; acceptance is an
explicit per-enclave allowlist **in the relay code** (public keys are public, so
committed in-repo, deployed by the normal relay CI; no new on-box secrets).
The SNI relay reaches tunnel enclaves by dialing the api-relay hub
(`wss://api.enclave.host/t/<name>/x/...`) so raw-TLS passthrough (in-CVM
termination, TLS-ALPN-01 ACME) keeps working without the fleet SECRET.

A metal enclave with a public IP (a colo box) skips the tunnel: set
`PUBLIC_URL`, the agent fronts 443 directly, the relay dials in as it does for
the hosted fleet.

## Certificate issuance: who holds what

Every app-zone certificate (`<label>.app.enclave.host`) is minted for a private
key that is generated **inside the CVM** and never leaves it. What moved is the
CA account: until now each box carried the fleet's ZeroSSL EAB pair (Tinfoil
secrets on the hosted fleet, `metal/config.json` -> fw_cfg on metal) and ran
its own ACME client. That put the platform's CA credential in an
operator-readable file on every seller box, and left nobody in a position to
pace the fleet against the CA rate limits (Let's Encrypt's 50 certificates per
registered domain per week is shared by every seller).

| holds | where | what it can do |
|---|---|---|
| the private key | the CVM (tmpfs) | nothing leaves; the CSR is built in-guest |
| the CA account (ZeroSSL EAB, Let's Encrypt account) | the API relay, `/etc/nan-relay/api-relay.env` (`ACME_EAB_KID`/`ACME_EAB_HMAC`), accounts persisted encrypted in the relay data dir | order a certificate for a CSR it was handed; it signs nothing and terminates no app TLS |
| `CERTS_KEY` = HMAC-SHA256(fleet `SECRET`, `"enclave certs v1"`) | the relay env (derived key only, never the raw `SECRET`) | verify that a request came from a fleet box |
| the operator key (`registryKey`) | the CVM | sign `enclave-certs-issue:<name>:<endpoint>:<ts>`; the relay checks it against the on-chain lease holder for the name's deployment |
| `DNS_TXT_KEY` | the relay (it already has it for the dns-01 push) | answer `_acme-challenge` for names in the platform zones only |

The service issues for `<label>.app.enclave.host` (and the TCP zone if
configured) and for a **verified custom domain** attached to a deployment, only
when the requesting enclave holds that deployment's live lease, and caches by
`(name, SPKI)` so a restart with the same key costs no issuance. The in-enclave
CA slots remain the fallback for both; without an EAB pair of its own a metal
box's fallback is Let's Encrypt alone, whose 5-per-week duplicate cap a day of
restarts can spend — the service's ZeroSSL account has no such cap.

**Config on a metal box.** `certsApi` (default: the API relay origin derived
from `relayUrl`, i.e. `https://api.enclave.host`) is all a first-party box
needs. Remove `acmeEabKid`/`acmeEabHmac` from `config.json`: the launcher no
longer forwards them unless `acmeBringYourOwn: true` is also set (a seller
minting from their own free ZeroSSL account), and logs a warning when it drops
them. The fw_cfg file is 0600 since eed7f2fd because it carried them; with
them gone it carries the registry key and the fleet secret only.

**Rollout order** (each step is safe with the previous ones in place; do not
reorder, the per-box EAB pair is the fallback until the last step):

1. **Deploy the relay** with the service configured, on nan
   (`/etc/nan-relay/api-relay.env`):
   - `CERTS_KEY` derived from the fleet `SECRET` (the raw secret stays off the
     relay; this is the same derived-key pattern as the dns-relay TXT key):
     ```sh
     node -e 'console.log(require("node:crypto").createHmac("sha256", process.env.SECRET).update("enclave certs v1").digest("hex"))'
     ```
     run wherever the fleet `SECRET` is in the environment (the Tinfoil secret
     value / `fleetSecret` in `metal/config.json`; they are the same string).
   - `ACME_EAB_KID` / `ACME_EAB_HMAC`: the platform ZeroSSL pair, moved here
     from `metal/config.json` / the Tinfoil secrets.
   - `DNS_API` / `DNS_TXT_KEY`: already present (the dns-01 push).
   Until this is done the route answers `503 certs_disabled` and every box
   falls back to its in-guest client, so nothing changes yet.
2. **Release the supervisor** (push to main): boxes that pick it up try
   `CERTS_API` first and fall back to their own ACME slots on 503/unreachable.
3. **Boxes restart on their schedule** (the metal auto-update timer takes the
   release when it appears, unless the box opted into `onlyWhenIdle`; the
   hosted fleet repoints on release). Watch the relay log for `certs: issued`
   lines per flavor.
4. **Remove the per-box EAB pair**: delete `acmeEabKid`/`acmeEabHmac` from every
   first-party `metal/config.json`, and drop `ACME_EAB_KID`/`ACME_EAB_HMAC` from
   `enclaves/*/tinfoil-config.yml` (Tinfoil binds secrets at container
   creation, so that is a rebind + relaunch, not a hot change). From here the
   only copy of the CA credential is on the relay.

## What degrades without the fleet secrets (and stays off by default)

- fleet deployment-secrets fetch (`SECRET`-HMAC with the relay): off
- in-enclave ACME DNS-01 push (`SECRET`-HMAC with dns-relay): off; the platform certificate service (`certsApi`, authorized by the operator key's proof-of-lease, no fleet secret needed) or TLS-ALPN-01 via the tunnel replaces it
- dedicated-IP egress/ingress (`EGRESS_RELAY_TOKEN`): off
- on-chain registry + claim loop + earning: off until the seller sets
  `registryKey` (a funded operator EOA; a few dollars of Base ETH for gas) and
  `payoutAddress` in `metal/config.json`. With them set, the guest supervisor
  registers, claims funded deployments, and the rev-7 `EnclaveDeployments`
  ledger pays the runner share from escrow, auto-swept to `payoutAddress`
  (see PROTOCOL.md, "How payout works")
- running **your own** apps here for free: off until that wallet is published
  on-chain with `enclave host declare-payout` (one transaction, from the wallet
  itself). A rev-12 ledger then charges nothing for deployments that wallet
  owns — no balance needed, none burned — while a paid app's publisher fee is
  untouched. `metal/config.json`'s `payoutAddress` alone does NOT do this: it
  only tells the supervisor where to sweep, and the chain cannot see it
- `SECRET`/`ADMIN_TOKEN` are minted **in-guest per boot**, so the host operator
  cannot read them (stronger than vault injection)

A dev-mode metal enclave additionally keeps `CLAIM_ENABLED=0` and is excluded
from the fleet capability-AND so it can neither claim paid work nor degrade
fleet-wide feature flags.

## Quick start (this repo, any Linux host with /dev/kvm)

```sh
curl -fsSL https://get.enclave.host | sh   # the enclave CLI, if you have not got it

enclave host init          # scaffold metal/config.json + mint this box's key
enclave host build         # the measured guest image (unprivileged; the first
                           # run pulls the pinned images, so give it a while)
enclave host run           # boot it in the foreground, ctrl-c to stop
enclave host check         # is the guest answering, is the quote real hardware
enclave host install       # or run it under systemd: enabled at boot, survives
                           # logout, pointed at THIS checkout
```

Each of those maps onto a script in this directory if you would rather drive it
yourself (`build-image.mjs`, `enclave-metal.mjs`, `systemd/`); the CLI just
fills in the paths and checks the answers. To sell hosting, keep going with
`enclave host fund` and `enclave host status` (see PROTOCOL.md).

SNP mode needs: BIOS `SMEE`/`SEV-SNP` enabled + SNP RMP coverage, kernel
`kvm_amd sev_snp=Y`, `/dev/sev` present; then set `"mode": "snp"` in the
config. TDX mode: `"mode": "tdx"` on a TDX host.
