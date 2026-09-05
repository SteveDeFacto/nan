#!/usr/bin/env node
// enclave-metal — the host-side launcher. Boots the measured guest image as a
// confidential VM (SEV-SNP or TDX) — or plain KVM in dev mode — wires the serial
// console to stdout/journal, and restarts a wedged guest. Designed to run under
// systemd (system service, root: a VMM needs /dev/kvm + /dev/sev).
//
//   node metal/enclave-metal.mjs --config metal/config.json
//
// config.json (see config.example.json):
//   { "mode":"snp|tdx|dev", "name":"metal0", "cpus":8, "memMiB":8192,
//     "publicUrl":"https://metal0.enclave.host",
//     "relayUrl":"wss://api.enclave.host/v1/fleet-tunnel", "tunnelToken":"…",
//     "hostfwd":[{"host":18080,"guest":8080}], "ovmf":"…", "qemu":"…", "dist":"…" }
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import net from 'node:net';
import tls from 'node:tls';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
function arg(name, dflt) { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt; }
const cfgPath = arg('config', path.join(HERE, 'config.json'));
const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));

let stopping = false;      // hoisted: the shielded-worker supervisor reads it too

const MODE = cfg.mode || 'snp';
const NAME = cfg.name || 'metal0';
const CPUS = String(cfg.cpus || 8);
const MEM = String(cfg.memMiB || 8192);
const DIST = cfg.dist || path.join(HERE, 'dist');
const OVMF = cfg.ovmf || '/usr/share/edk2/x64/OVMF.4m.fd';
const QEMU = cfg.qemu || 'qemu-system-x86_64';
const SEV_DEVICE = cfg.sevDevice || '/dev/sev';
const KERNEL = path.join(DIST, 'vmlinuz');
const INITRD = path.join(DIST, 'initramfs.cpio.gz');
for (const f of [KERNEL, INITRD]) if (!fs.existsSync(f)) { console.error(`missing ${f}; run: node metal/build-image.mjs`); process.exit(1); }

// --- attested model volumes ---------------------------------------------------
// Each entry in cfg.volumes (names, or "*" for the whole store) is one file
// built by metal/volumes.mjs: an ext4 image of the model tree with a dm-verity
// hash tree appended. We attach it as a read-only virtio-blk disk and hand the
// guest its verity parameters through fw_cfg; the guest brings dm-verity up
// itself, so the host (this process) is never trusted for the CONTENT.
//
// What makes it attested rather than merely hashed: the digest of the whole
// volume table is launched into the CPU's HOST_DATA field, which the hardware
// signs into every attestation report this VM ever produces. Add, drop, or swap
// a model and the quote says so — the property Tinfoil's Modelwrap gets by
// putting its dm-verity root on the measured cmdline. HOST_DATA rather than the
// cmdline on purpose: it is host-supplied config bound to the quote WITHOUT
// entering the launch measurement, so the release measurement (what allowlists
// and dist/manifest.json pin) stays stable while the model set stays provable.
// The guest reads HOST_DATA back out of its own report and refuses to mount a
// table that doesn't hash to it.
const VOL_STORE = cfg.volumeStore || '/vm/enclave-volumes';
function loadVolumes() {
  const want = cfg.volumes === '*'
    // dot-directories are staging, not volumes: model source trees get parked
    // next to the images they were built from, and "*" must not try to attach one
    ? (fs.existsSync(VOL_STORE) ? fs.readdirSync(VOL_STORE, { withFileTypes: true })
        .filter((e) => e.isDirectory() && !e.name.startsWith('.')).map((e) => e.name).sort() : [])
    : (Array.isArray(cfg.volumes) ? cfg.volumes : []);
  const out = [];
  for (const name of want) {
    const dir = path.join(VOL_STORE, name);
    let v; try { v = JSON.parse(fs.readFileSync(path.join(dir, 'volume.json'), 'utf8')); }
    catch (e) { console.error(`[enclave-metal] volume ${name}: no volume.json in ${dir} (${e.code || e.message}); SKIPPED`); continue; }
    const img = path.join(dir, v.image || 'volume.img');
    if (!fs.existsSync(img)) { console.error(`[enclave-metal] volume ${name}: missing ${img}; SKIPPED`); continue; }
    out.push({
      name: v.name || name, image: img, bytes: v.bytes || 0,
      alg: v.verity.alg || 'sha256', root: v.verity.root, salt: v.verity.salt,
      dataBlockSize: v.verity.dataBlockSize, hashBlockSize: v.verity.hashBlockSize,
      dataBlocks: v.verity.dataBlocks, hashStartBlock: v.verity.hashStartBlock,
      sd: !!v.sd, gguf: v.gguf || '',
    });
  }
  return out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}
const VOLUMES = loadVolumes();
// virtio-blk serials are how the guest maps a disk to its table entry (device
// enumeration order is not a contract). 20 bytes max, so index them.
VOLUMES.forEach((v, i) => { v.serial = `mvol${i}`; });
// canonical volume-set digest — the SAME construction in metal/guest/gsup.mjs
// (which enforces it) and metal/verify.mjs (which checks it against a quote).
const volLine = (v) => [v.name, v.alg, v.root, v.salt, v.dataBlockSize, v.hashBlockSize,
  v.dataBlocks, v.hashStartBlock, v.sd ? 1 : 0, v.gguf || ''].join('|');
const VOL_DIGEST = VOLUMES.length
  ? createHash('sha256').update(VOLUMES.map(volLine).sort().join('\n') + '\n').digest('hex') : '';

// --- app-zone certificates: who holds what -----------------------------------
// The platform certificate service (relay/certs.js, docs/custom-domains.md
// "Certificate issuance: who holds what") is the DEFAULT. The guest keeps the
// private key and builds the CSR inside the CVM; the relay holds the CA
// account (ZeroSSL EAB, Let's Encrypt fallback) and paces issuance for the
// whole fleet. So a first-party box no longer needs -- and should not carry --
// the ZeroSSL EAB pair in config.json: before this, every box held the fleet's
// CA credentials in an operator-readable file (eed7f2fd made that file 0600,
// which is a bandage, not a fix).
//
// The per-box EAB pair is the EXCEPTION, for a seller who runs their own free
// ZeroSSL account and wants the in-guest ACME client to mint directly: BOTH
// keys set AND acmeBringYourOwn: true. Keys without the flag are dropped with
// a warning rather than forwarded, so a config.json that predates this change
// stops shipping the credential the moment the launcher is updated.
//
// certsApi = the service origin; default = the API relay the box already
// dials for its tunnel (wss://api.enclave.host/... -> https://api.enclave.host),
// so a config that names one relay cannot silently ask another for certs.
function certsCfg(cfg, log = (...a) => console.log(...a)) {
  const kid = String(cfg.acmeEabKid || ''), hmac = String(cfg.acmeEabHmac || '');
  const byo = cfg.acmeBringYourOwn === true;
  if (byo && kid && hmac) {
    log('[enclave-metal] certificates: bring-your-own ZeroSSL EAB forwarded (acmeBringYourOwn); the guest still asks the platform certificate service first for its own-zone names and falls back to this pair, then Let\'s Encrypt, in the guest');
    return { acmeEabKid: kid, acmeEabHmac: hmac, certsApi: certsApiOf(cfg) };
  }
  if (byo) log('[enclave-metal] certificates: acmeBringYourOwn is set but acmeEabKid/acmeEabHmac are not BOTH set; ignoring it');
  else if (kid || hmac) log('[enclave-metal] certificates: acmeEabKid/acmeEabHmac in config.json are NOT forwarded (acmeBringYourOwn is not true) - remove them; first-party boxes use the platform certificate service');
  const certsApi = certsApiOf(cfg);
  if (certsApi) log(`[enclave-metal] certificates: platform certificate service at ${certsApi} (the key stays in the CVM; the relay holds the CA account)`);
  else log('[enclave-metal] certificates: no certsApi and no relayUrl to derive one from; app-zone TLS stays off');
  return { certsApi };
}
function certsApiOf(cfg) {
  if (cfg.certsApi) return String(cfg.certsApi).replace(/\/+$/, '');
  if (!cfg.relayUrl) return '';
  try { const u = new URL(cfg.relayUrl); return `https://${u.host}`; } catch { return ''; }
}
const CERTS = certsCfg(cfg);

// The kernel cmdline is MEASURED (kernel-hashes=on), so it carries only what is
// part of the enclave's identity: the mode. Deployment-specific runtime config
// (name, public URL, relay, tunnel token) is delivered out-of-band via QEMU
// fw_cfg — NOT measured — so a given image has ONE stable launch measurement
// regardless of which relay/token it uses, and no secret ever enters the quote.
const cmdline = [
  'console=ttyS0', 'root=/dev/ram0', 'rootfstype=ramfs', 'quiet',
  `metal.mode=${MODE}`,
].join(' ');
const runtimeCfg = { name: NAME, mode: MODE, publicUrl: cfg.publicUrl || '', relayUrl: cfg.relayUrl || '', tunnelToken: cfg.tunnelToken || '',
  // seller earning (metal/PROTOCOL.md Phase C): the operator EOA key that
  // registers/claims/earns on-chain (needs a little Base ETH for gas), and the
  // wallet the supervisor auto-sweeps accrued USDC earnings to. Both optional:
  // without registryKey the enclave serves via the tunnel but neither claims
  // nor earns. Delivered via fw_cfg like the tunnel token — out-of-band, never
  // in the launch measurement or the quote.
  registryKey: cfg.registryKey || '', payoutAddress: cfg.payoutAddress || '',
  // how app-zone certificates get minted: see certsCfg() above. Rides fw_cfg
  // like the keys: out-of-band, unmeasured.
  ...CERTS,
  // NOMINAL node RAM (fleet parity): the Tinfoil flavors advertise their baked
  // size (64/512 GB), not the guest kernel's MemTotal, so metal advertises the
  // size the host gives the VM the same way. gsup caps it at the measured
  // total + a small boot haircut so a config typo (or a dishonest seller)
  // can't advertise RAM the VM doesn't have.
  nodeRamGb: Math.round(Number(MEM) / 1024),
  // what this operator CHARGES, in USD per hour for a FULL node / FULL card
  // (see gsup: converted to the ledger's per-second 6dp basis). The GPU ask is
  // only meaningful on a GPU enclave; gsup drops it otherwise.
  priceCpuUsdHr: cfg.priceCpuUsdHr != null ? Number(cfg.priceCpuUsdHr) : null,
  priceGpuUsdHr: cfg.priceGpuUsdHr != null ? Number(cfg.priceGpuUsdHr) : null,
  // optional FLEET secret (first-party boxes only): with it the guest joins
  // the fleet's deployment-secrets plane (the relay's fetch auth derives from
  // it); without it the guest mints its own SECRET per boot and truthfully
  // advertises secrets-incapable. Anonymous sellers leave this unset.
  fleetSecret: cfg.fleetSecret || '',
  // ggml engine tuning for the wasm-manager's tenants. The Tinfoil flavors set
  // these in their compose env; metal had no channel for them at all, so every
  // metal box ran the ENGINE defaults - and the one that matters is
  // ENCLAVE_GGML_MAX_SESSIONS, whose default is 1. One inference slot per
  // graph means a single chat turn's own internal passes queue behind its main
  // answer, and the app sits in "[sessions_busy] - waiting for a free slot"
  // until its 5-minute budget runs out (2026-08-31, eyesoff-ai on metal0).
  // Raising it is FREE: sessions are sequence handles into ONE shared KV pool
  // sized by N_CTX, so a slot costs no pre-allocated memory of its own.
  // gsup allowlists which keys may pass; a typo'd or dangerous name is dropped
  // there rather than reaching the engine.
  ggml: (cfg.ggml && typeof cfg.ggml === 'object') ? cfg.ggml : {},
  // dm-verity parameters for the attached model volumes. Unmeasured on its own
  // — the guest hashes this table and refuses to mount unless it matches the
  // measured metal.vols digest above, so this is a delivery channel, not a
  // trusted one. The host path never crosses: only the serial the disk carries.
  volumes: VOLUMES.map(({ image, ...v }) => v) };

// Optional egress helper. QEMU user-net (slirp) NATs outbound for a normal host,
// but some sandboxed/dev hosts block slirp's EXTERNAL sockets while still routing
// the guest→host (10.0.2.2) path. When cfg.egressHelper is set, we run a tiny
// host-side pipe the guest can reach at 10.0.2.2:<port> and which adds TLS to the
// real relay — so the guest speaks plaintext ws to the helper and the TLS leg to
// the relay is end-to-end from the helper. Only needed in such environments; a
// normal seller box dials the relay directly (leave egressHelper unset).
if (cfg.egressHelper && cfg.relayUrl) {
  const ru = new URL(cfg.relayUrl);
  const port = cfg.egressHelper.port || 9443;
  const targetHost = ru.hostname, targetPort = Number(ru.port) || 443;
  net.createServer((gsock) => {
    const up = tls.connect({ host: targetHost, port: targetPort, servername: targetHost }, () => { gsock.pipe(up); up.pipe(gsock); });
    // The WAN leg can die without a FIN (ISP drop, NAT timeout) and this dumb
    // pipe would then hold both legs open forever — the guest's tunnel ws
    // reads as connected while the relay hub has long detached the box.
    // Kernel keepalive turns that silence into an error kill() can cascade to
    // the guest, whose agent then redials. The loopback leg cannot die
    // silently, so only the upstream needs it.
    up.setKeepAlive(true, 30_000);
    const kill = () => { gsock.destroy(); up.destroy(); };
    up.on('error', kill); gsock.on('error', kill); up.on('close', kill); gsock.on('close', kill);
  }).listen(port, '127.0.0.1', () => console.log(`[enclave-metal] egress helper 127.0.0.1:${port} → ${targetHost}:${targetPort} (guest reaches it at 10.0.2.2:${port})`));
  // guest dials the helper in plaintext ws, but keeps the real relay host for the Host header + SNI identity
  runtimeCfg.relayUrl = `ws://10.0.2.2:${port}${ru.pathname}`;
  runtimeCfg.relayHost = targetHost;
}

// Optional shielded GPU worker. The card stays on the HOST, outside the enclave
// and outside the measurement, and the guest reaches it at 10.0.2.2:<port> over
// the same slirp path the egress helper uses. That is the whole point of the
// shielded tier: docs/shielded-inference.md assumes the GPU's operator is hostile
// and gives it only public weights and one-time-padded activations, so a box can
// sell GPU work without the GPU ever entering the TCB.
//
// The endpoint rides fw_cfg, which is NOT covered by the launch measurement, and
// that is correct rather than sloppy. A host that redirects this to a worker it
// wrote gains nothing -- the pad never crosses, and Freivalds rejects any product
// that is not the real one. The worst it can do is refuse to answer, and
// availability is explicitly not something this design promises.
const shieldedChildren = new Set();
const shieldedWorkers = cfg.shieldedWorkers ?? (cfg.shieldedWorker ? [cfg.shieldedWorker] : []);
if (!Array.isArray(shieldedWorkers) || shieldedWorkers.length > 16)
  throw new Error('shieldedWorkers must be an array of at most 16 workers');
const workerPorts = new Set(), workerVsockPorts = new Set(), workerDevices = new Set();
for (const [id, sw] of shieldedWorkers.entries()) {
  if (!sw || typeof sw !== 'object' || Array.isArray(sw))
    throw new Error('Each shielded worker must be an object');
  const port = Number(sw.port || 9500);
  if (!Number.isInteger(port) || port < 1 || port > 65535 || workerPorts.has(port))
    throw new Error('shieldedWorkers need distinct valid TCP ports');
  workerPorts.add(port);
  if (sw.vsock !== false) {
    const vsockPort = Number(sw.vsockPort || port);
    if (!Number.isInteger(vsockPort) || vsockPort < 1 || vsockPort > 65535 || workerVsockPorts.has(vsockPort))
      throw new Error('shieldedWorkers need distinct valid vsock ports');
    workerVsockPorts.add(vsockPort);
  }
  if (shieldedWorkers.length > 1 && !/^GPU-[a-f0-9-]{36}$/i.test(sw.device || ''))
    throw new Error('Every worker in a multi-GPU pool needs an explicit GPU UUID');
  if (sw.device && workerDevices.has(sw.device.toLowerCase()))
    throw new Error('A GPU UUID must not be advertised by multiple workers');
  if (sw.device) workerDevices.add(sw.device.toLowerCase());
  if (shieldedWorkers.length > 1 && sw.shm)
    throw new Error('Multi-GPU workers currently use TCP/vsock, not a shared-memory ring');
}
// Pool pricing is computed from each probed dedicated slice, never physical
// VRAM or momentary benchmark speed. The existing registry price buys one pool.
if (cfg.shieldedPool !== undefined) {
  const p = cfg.shieldedPool;
  if (!p || p.mode !== 'layers' || shieldedWorkers.length < 1)
    throw new Error('shieldedPool needs mode "layers" and configured workers');
  const pricing = p.pricing || {};
  for (const key of ['tflopUsdHr', 'vramGiBUsdHr'])
    if (!Number.isFinite(pricing[key]) || pricing[key] < 0 || pricing[key] > 100)
      throw new Error(`shieldedPool.pricing.${key} must be a nonnegative hourly resource rate`);
  if (!(pricing.tflopUsdHr + pricing.vramGiBUsdHr > 0))
    throw new Error('shieldedPool pricing must be positive');
  runtimeCfg.shieldedPool = { mode: 'layers', pricing,
    cardIds: shieldedWorkers.map((_, id) => id) };
}
runtimeCfg.shieldedWorkers = [];
for (const [id, sw] of shieldedWorkers.entries()) {
  const port = sw.port || 9500;
  // The C++/CUDA worker when it has been built (make -C shielded/worker-cuda),
  // the Python reference otherwise. Same protocol, same admission rules; the
  // difference is ~80 us against ~350 us per exchange, which at ~50 exchanges
  // per decoded token is the difference between 150 tok/s and 30.
  const cudaWorker = sw.binary || path.join(HERE, '..', 'shielded', 'worker-cuda', 'shielded-worker');
  const useCuda = sw.python === undefined && sw.script === undefined && fs.existsSync(cudaWorker);
  // vsock: the guest reaches the host at CID 2 without slirp in the path. Needs
  // /dev/vhost-vsock on the host and the vsock modules in the guest image; the
  // guest probe checks the second and tells tenants only when both hold.
  const vsockOn = sw.vsock !== false && fs.existsSync('/dev/vhost-vsock');
  const vsockPort = vsockOn ? (sw.vsockPort || port) : 0;
  const guestCid = sw.guestCid || 3;
  const python = sw.python || 'python3';
  const script = sw.script || path.join(HERE, '..', 'shielded', 'worker.py');
  const swArgs = useCuda ? ['--host', '127.0.0.1', '--port', String(port)]
                         : [script, '--host', '127.0.0.1', '--port', String(port)];
  if (sw.vramGb) swArgs.push('--vram-gb', String(sw.vramGb));
  if (useCuda && vsockPort) swArgs.push('--vsock-port', String(vsockPort));
  // The shared-memory ring, OFF unless configured:
  //   "shm": { "path": "/dev/shm/enclave-shielded-ring", "mib": 32 }
  // A file this launcher creates and sizes, mapped by the worker (--shm) and
  // attached to the CVM as the BAR of an ivshmem-plain device (baseArgs), so
  // the guest and the worker poll the same cache lines instead of paying a VM
  // exit and an interrupt per direction per exchange: in the CVM that is 152
  // us of the 10 ms token on vhost-vsock, against ~1 us on the ring
  // (shielded/REPORT.md 13.13, scratchpad/shm-ring/DESIGN.md). The pages
  // carry what the socket carries -- one-time-padded planes and public
  // headers -- so sharing them with the host gives it nothing new; the guest
  // treats the ring as a hostile peer exactly like the socket. The device is
  // not in the launch measurement (device topology never is); the guest
  // learns the ring exists from fw_cfg and ignores it when absent. 8 MiB per
  // ring, one ring per tenant link; 32 MiB = 4 links.
  const shm = useCuda && sw.shm && sw.shm.path ? { path: String(sw.shm.path), mib: Math.max(8, Number(sw.shm.mib) || 32) } : null;
  if (shm) {
    // sized here, never by the worker or the guest; power of two for the BAR
    shm.mib = 2 ** Math.ceil(Math.log2(shm.mib));
    const fd = fs.openSync(shm.path, 'w'); fs.ftruncateSync(fd, shm.mib * 1048576); fs.closeSync(fd);
    swArgs.push('--shm', shm.path);
  }
  const swExe = useCuda ? cudaWorker : python;
  // Host-side worker knobs as config, so a tuning pass never needs a code
  // change or a hand-edited unit file. Every entry lands verbatim in the
  // spawned worker's environment; the one that exists today is
  // SHIELDED_WORKER_SPIN_US (worker.cu: bounded MSG_DONTWAIT poll before the
  // blocking read of the next frame, default 0 = off). Names are checked to be
  // environment-variable-shaped so a typo fails at launch, not as a silently
  // unset knob.
  const workerEnv = envMap(sw.workerEnv, 'shieldedWorker.workerEnv');
  // computeShare: the fraction of the card's COMPUTE this box sells (VRAM has
  // vramGb). Enforced by pointing the worker at the host MPS daemon
  // (metal/host-mps.sh + enclave-mps.user.service) with the matching SM cap —
  // the same CUDA_MPS_ACTIVE_THREAD_PERCENTAGE mechanism the hosted fleet
  // uses, except here it caps the ONE worker (the platform's whole footprint
  // against this desktop's other GPU users) rather than a tenant. Fail-open by
  // MPS's own design: if the daemon is down the worker attaches directly,
  // uncapped — the box keeps serving, exactly like a dead worker never takes
  // the box down. An explicit workerEnv entry wins over both defaults.
  const computeShare = Number(sw.computeShare) > 0 && Number(sw.computeShare) < 1 ? Number(sw.computeShare) : 0;
  const mpsEnv = computeShare ? {
    CUDA_MPS_PIPE_DIRECTORY: `/run/user/${process.getuid()}/enclave-mps`,
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: String(Math.max(1, Math.round(computeShare * 100))),
  } : {};
  const swEnv = { ...process.env, ...mpsEnv, ...workerEnv, ...(sw.device ? { CUDA_VISIBLE_DEVICES: sw.device } : {}) };
  if (computeShare)
    console.log(`[enclave-metal] shielded worker capped to ${Math.round(computeShare * 100)}% of the card's SMs (MPS)`);
  if (Object.keys(workerEnv).length)
    console.log(`[enclave-metal] shielded worker env: ${Object.entries(workerEnv).map(([k, v]) => `${k}=${v}`).join(' ')}`);
  let swRestarts = 0;
  const startWorker = () => {
    const shieldedChild = spawn(swExe, swArgs, { stdio: ['ignore', 'inherit', 'inherit'], env: swEnv });
    shieldedChildren.add(shieldedChild);
    shieldedChild.on('exit', (code, sig) => {
      shieldedChildren.delete(shieldedChild);
      if (stopping) return;
      swRestarts++;
      const delay = Math.min(2000 * swRestarts, 30000);
      // A dead worker must NEVER take the box down with it. The enclave keeps
      // serving; it just has no GPU to offload to, and the shielded flavor's
      // health probe is what tells the fleet so.
      console.error(`[enclave-metal] shielded worker exited code=${code} sig=${sig}; restart in ${delay}ms (#${swRestarts})`);
      setTimeout(startWorker, delay);
    });
    shieldedChild.on('error', (e) => console.error(`[enclave-metal] shielded worker spawn error: ${e.message}`));
  };
  startWorker();
  console.log(`[enclave-metal] shielded worker (${useCuda ? 'cuda' : 'python'}) on 127.0.0.1:${port} `
    + `(guest reaches it at 10.0.2.2:${port}${useCuda && vsockPort ? `, vsock CID 2 port ${vsockPort}` : ''})`);
  // Guest-side knobs for the TENANT, carried to the guest over fw_cfg and from
  // there into the verdict file, the supervisor's provision request and the
  // wasm-manager's tenant environment. Only SHIELDED_* names survive that trip
  // (the manager drops everything else, and so does this end): the host may
  // tune the backend it already talks to, not inject arbitrary environment
  // into a tenant. Today's knob is SHIELDED_SPIN_US (shielded-wire.c: bounded
  // MSG_DONTWAIT poll before the blocking read of a reply, default 0). None of
  // this touches what crosses the boundary -- it decides whether the tenant's
  // vCPU halts or spins while it waits for bytes the host was always going to
  // send it.
  const tenantEnv = envMap(sw.tenantEnv, 'shieldedWorker.tenantEnv', /^SHIELDED_/);
  const guestWorker = { id, host: '10.0.2.2', port,
    ...(sw.device ? { deviceUuid: sw.device } : {}),
    ...(useCuda && vsockPort ? { vsockPort, guestCid } : {}),
    ...(Object.keys(tenantEnv).length ? { tenantEnv } : {}),
    // the ring's size in MiB, so the guest can bound its mapping of the BAR
    // (it maps the ivshmem device 1af4:1110 it finds, never a host address)
    ...(shm ? { shmMib: shm.mib } : {}),
    // the box's ask for a WHOLE shielded card, USD/hour. Config, not a probe
    // result, but it rides with the endpoint so the guest sees one object.
    ...(Number(sw.priceUsdHr) > 0 ? { priceUsdHr: Number(sw.priceUsdHr) } : {}),
    // the compute fraction the MPS cap above enforces, so the guest's verdict
    // (and from it the supervisor's pool) advertises only what is for sale
    ...(computeShare ? { computeShare } : {}) };
  runtimeCfg.shieldedWorkers.push(guestWorker);
}
// Older guest images see only the original card until their update lands.
runtimeCfg.shieldedWorker = runtimeCfg.shieldedWorkers[0];

// A string->string map from config, validated to be environment-shaped
// (NAME=value, printable values, bounded) and optionally filtered to a name
// prefix. Anything else is a config error at launch rather than a surprise
// inside a tenant.
function envMap(src, where, prefix = null) {
  const out = {};
  if (src == null) return out;
  if (typeof src !== 'object' || Array.isArray(src)) throw new Error(`${where} must be an object of NAME: "value"`);
  for (const [k, v] of Object.entries(src)) {
    if (!/^[A-Z_][A-Z0-9_]{0,63}$/.test(k)) throw new Error(`${where}: "${k}" is not an environment variable name`);
    if (prefix && !prefix.test(k)) throw new Error(`${where}: "${k}" is not allowed here (only ${prefix.source.replace(/^\^/, '')}* names)`);
    if (typeof v !== 'string' && typeof v !== 'number') throw new Error(`${where}: ${k} must be a string`);
    const val = String(v);
    if (val.length > 256 || /[^\x20-\x7e]/.test(val)) throw new Error(`${where}: ${k} must be printable ASCII, at most 256 bytes`);
    out[k] = val;
  }
  return out;
}

// Guest idle policy: cpuidle-haltpoll. An idle SNP vCPU halts, and waking it
// for the shielded worker's reply costs a VM exit plus an injected interrupt
// on every one of the ~49 exchanges a decoded token makes -- in the CVM that
// exchange is 152 us against 46 us on the host's own loopback and ~10 us for
// the socket itself (shielded/REPORT.md 13.13). haltpoll makes an idle vCPU
// poll for a bounded, self-tuning window before it halts, so a reply that
// lands inside the window finds the vCPU running. The COST is that window:
// an idle vCPU burns its host core for up to guest_halt_poll_ns (200 us by
// default) after every piece of work before it halts, so a box with many
// tenants and few cores pays for it in host CPU time. The window only ever
// grows while wakeups keep landing inside it and shrinks back to zero when
// they stop, which is why the default is on.
//
// The whole decision rides fw_cfg as one string, NOT the kernel cmdline: the
// cmdline is measured, and a poll window is a tuning parameter, not part of
// the enclave's identity. The guest (metal/guest/init) loads the driver, which
// is in the image whether or not it is used, and applies the parameters only
// after checking each one is a known name with a numeric value -- the host
// wrote this string and the guest trusts none of it by default.
//
//   "guest": { "haltpoll": true | false | { "ns": 200000, "growStart": 50000,
//                                          "grow": 2, "shrink": 2, "allowShrink": true } }
const haltpollFwCfg = (() => {
  const g = cfg.guest && cfg.guest.haltpoll;
  if (g === false) return 'off';
  const o = (g && typeof g === 'object') ? g : {};
  const num = (k, dflt, max) => {
    if (o[k] == null) return dflt;
    const n = Number(o[k]);
    if (!Number.isInteger(n) || n < 0 || n > max) throw new Error(`guest.haltpoll.${k} must be an integer in [0, ${max}]`);
    return n;
  };
  return ['on',
    `guest_halt_poll_ns=${num('ns', 200000, 10_000_000)}`,             // ceiling of the window, ns
    `guest_halt_poll_grow_start=${num('growStart', 50000, 10_000_000)}`, // first window after a hit
    `guest_halt_poll_grow=${num('grow', 2, 1000)}`,                     // window *= grow on a hit
    `guest_halt_poll_shrink=${num('shrink', 2, 1000)}`,                 // window /= shrink on a miss (0 = to zero)
    `guest_halt_poll_allow_shrink=${o.allowShrink === false ? 0 : 1}`,
  ].join(' ');
})();

// The fw_cfg file carries the fleet secret, the registry key and (bring-your-
// own boxes only) the ACME EAB pair. It is OURS, not the world's: mode 0600, and the files of launchers
// that are gone (a crash, a SIGKILL) are swept here rather than left in /tmp
// forever -- 2026-08-27: three of them from earlier in the week, readable by
// any local user. QEMU reads the file once at start-up.
const fwCfgPath = path.join(os.tmpdir(), `metal-fwcfg-${process.pid}.json`);
for (const f of fs.readdirSync(os.tmpdir()).filter((n) => /^metal-fwcfg-\d+\.json$/.test(n))) {
  const pid = Number(n(f));
  if (pid === process.pid) continue;
  try { process.kill(pid, 0); continue; } catch {}   // still alive: leave it
  try { fs.unlinkSync(path.join(os.tmpdir(), f)); } catch {}
}
function n(f) { return f.replace(/^metal-fwcfg-/, '').replace(/\.json$/, ''); }
fs.writeFileSync(fwCfgPath, JSON.stringify(runtimeCfg), { mode: 0o600 });

function baseArgs() {
  const a = [
    '-machine', MODE === 'dev' ? 'q35,accel=kvm' : 'q35,accel=kvm,confidential-guest-support=cx0,memory-backend=ram0',
    '-cpu', 'host', '-smp', CPUS, '-m', MEM,
    '-nographic', '-no-reboot',
    '-kernel', KERNEL, '-initrd', INITRD, '-append', cmdline,
    // deployment config, out-of-band (not measured): the guest reads it from
    // /sys/firmware/qemu_fw_cfg/by_name/opt/org.enclave.metal/raw
    '-fw_cfg', `name=opt/org.enclave.metal,file=${fwCfgPath}`,
    // the guest's idle policy (see haltpoll above): read by PID 1 before any
    // service starts, hence its own plain-string item rather than a JSON field
    '-fw_cfg', `name=opt/org.enclave.metal/haltpoll,string=${haltpollFwCfg}`,
    // outbound-only user networking (slirp NAT); the enclave dials OUT to the
    // relay, so no inbound is needed. hostfwd exposes loopback ports for testing.
    // Offloads are DISABLED: in a confidential guest, memory is encrypted and
    // DMA goes through bounce buffers, so the host cannot fix up checksums/GSO
    // for offloaded packets — with them on, external TCP SYNs are silently
    // dropped and every outbound connection times out.
    '-netdev', netdev(),
    '-device', 'virtio-net-pci,netdev=net0,csum=off,gso=off,guest_csum=off,'
      + 'host_tso4=off,host_tso6=off,guest_tso4=off,guest_tso6=off,guest_ecn=off,'
      + 'host_ufo=off,guest_ufo=off',
    '-serial', 'mon:stdio',
  ];
  // The shielded worker's vsock. Host CID 2 is implicit; the guest's own CID is
  // whatever the config says (3 by default) and is never used for anything.
  if (runtimeCfg.shieldedWorkers.some(sw => sw.vsockPort))
    a.push('-device', `vhost-vsock-pci,guest-cid=${runtimeCfg.shieldedWorkers.find(sw => sw.vsockPort).guestCid || 3}`);
  // The shielded worker's shared-memory ring: the worker's --shm file becomes
  // BAR2 of an ivshmem-plain device. Verified to boot and poll exit-free
  // under SEV-SNP with this OVMF and kernel (2026-08-26, throwaway VM).
  if (runtimeCfg.shieldedWorker?.shmMib && shieldedWorkers[0]?.shm?.path) {
    a.push('-object', `memory-backend-file,id=shring,share=on,mem-path=${shieldedWorkers[0].shm.path},size=${runtimeCfg.shieldedWorker.shmMib}M`);
    a.push('-device', 'ivshmem-plain,memdev=shring');
  }
  // model volumes: one read-only virtio-blk disk each. cache=none keeps the
  // host page cache out of it — the guest caches what it reads (verified), and
  // a second copy of a 60 GB model in host RAM only steals memory from the CVM.
  // The guest never trusts these bytes: dm-verity checks every block.
  for (const v of VOLUMES) {
    a.push('-drive', `file=${v.image},if=none,id=${v.serial},format=raw,readonly=on,cache=none,aio=threads`);
    a.push('-device', `virtio-blk-pci,drive=${v.serial},serial=${v.serial}`);
  }
  if (MODE === 'dev') return a;
  // confidential VM: private guest memory via memfd + the TEE launch object
  a.push('-object', `memory-backend-memfd,id=ram0,size=${MEM}M,share=true,prealloc=false`);
  a.push('-bios', OVMF);
  // The attached model volumes' set digest, bound into every attestation report
  // this VM produces: HOST_DATA (32 bytes) on SEV-SNP, MRCONFIGID (48, zero-
  // padded) on TDX. Not part of the launch measurement — deliberately, see
  // "attested model volumes" above.
  const volB64 = (bytes) => VOL_DIGEST
    ? Buffer.concat([Buffer.from(VOL_DIGEST, 'hex'), Buffer.alloc(bytes - 32)]).toString('base64') : '';
  if (MODE === 'tdx') {
    // NOTE: the guest does NOT self-check MRCONFIGID today (it enforces
    // HOST_DATA on SNP only), so on TDX the binding is verifiable by a remote
    // party but not fail-closed inside the guest. No TDX hardware to test on.
    a.push('-object', `tdx-guest,id=cx0${VOL_DIGEST ? `,mrconfigid=${volB64(48)}` : ''}`);
    a.push('-machine', 'q35,accel=kvm,confidential-guest-support=cx0,memory-backend=ram0,kernel-irqchip=split');
  } else {
    // SEV-SNP with measured kernel hashes → kernel+initrd+cmdline in the launch digest.
    //
    // GUEST POLICY is deliberately left at QEMU's default (0x30000: reserved
    // bit 17 set, SMT allowed, DEBUG clear) rather than spelled out. Know what
    // rides on that: the policy is NOT part of the launch measurement, so a
    // guest booted with DEBUG set produces a byte-identical measurement while
    // the host can read and write its memory freely. Both verifiers now refuse
    // that — relay/snp-verify.mjs (tunnel attach) and metal/verify.mjs (what a
    // buyer runs) — so if a future QEMU ever changed this default the failure
    // is loud at attach time rather than a silently transparent box. Spelling
    // `policy=` out here would be belt-and-braces; it is left alone only
    // because a boot-line change to a serving box wants a real boot to test.
    a.push('-object', `sev-snp-guest,id=cx0,cbitpos=51,reduced-phys-bits=1,kernel-hashes=on,sev-device=${SEV_DEVICE}`
      + (VOL_DIGEST ? `,host-data=${volB64(32)}` : ''));
  }
  return a;
}
function netdev() {
  const fwds = (cfg.hostfwd || []).map((f) => `,hostfwd=tcp:127.0.0.1:${f.host}-:${f.guest}`).join('');
  return `user,id=net0${fwds}`;
}

let child = null, restarts = 0;
function launch() {
  const args = baseArgs();
  console.log(`[enclave-metal] launching ${NAME} mode=${MODE} ${CPUS}vcpu/${MEM}MiB`);
  if (VOLUMES.length) {
    console.log(`[enclave-metal] model volumes (${VOLUMES.length}, set digest ${VOL_DIGEST.slice(0, 16)}… — MEASURED):`);
    for (const v of VOLUMES)
      console.log(`[enclave-metal]   ${v.name.padEnd(26)} ${(v.bytes / 1e9).toFixed(2).padStart(7)} GB  verity ${v.root.slice(0, 24)}…`);
  }
  console.log(`[enclave-metal] cmdline: ${cmdline}`);
  console.log(`[enclave-metal] guest haltpoll: ${haltpollFwCfg}`);
  child = spawn(QEMU, args, { stdio: ['ignore', 'inherit', 'inherit'] });
  child.on('exit', (code, sig) => {
    if (stopping) return;
    restarts++;
    const delay = Math.min(2000 * restarts, 15000);
    console.error(`[enclave-metal] guest exited code=${code} sig=${sig}; relaunch in ${delay}ms (#${restarts})`);
    setTimeout(launch, delay);
  });
  child.on('error', (e) => console.error(`[enclave-metal] spawn error: ${e.message}`));
}
for (const s of ['SIGTERM', 'SIGINT']) process.on(s, () => {
  stopping = true;
  for (const c of [child, ...shieldedChildren]) if (c) { try { c.kill('SIGTERM'); } catch {} }
  setTimeout(() => process.exit(0), 2000);
});
launch();
