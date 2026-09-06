#!/usr/bin/env node
// build-image.mjs — reproducible, UNPRIVILEGED build of the metal guest image.
//
// Produces a measured SEV-SNP guest from the exact same digest-pinned container
// images the hosted fleet runs. No docker, no root: the OCI images are pulled
// and extracted by oci-pull.mjs, the GPU payload is stripped for the CPU flavor,
// the supervisor image becomes the guest root, the wasm-manager image becomes a
// chroot under /opt/roots/wasm, and the whole thing is packed into a single
// initramfs whose kernel+initrd+cmdline are folded into the launch measurement
// (kernel-hashes=on). Everything that runs in the TCB is therefore covered by
// the hardware measurement — no unmeasured byte, no transparency-log detour.
//
//   node metal/build-image.mjs [--supervisor <ref>] [--wasm <ref>] [--kernel <path>]
//
// Outputs under metal/dist/: vmlinuz, initramfs.cpio.gz, cmdline, manifest.json
import { execFileSync, execSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BUILD = path.join(HERE, 'build');
// --out lets the auto-updater build BESIDE the live image instead of over it:
// a staged build that fails leaves the serving image untouched (metal/update.mjs).
const DIST = path.resolve(arg('out', path.join(HERE, 'dist')));
// --release stamps which published tag this image was built from, so the
// updater can tell "already current" from "never checked" without guessing.
const RELEASE = arg('release', '');
const ROOT = path.join(BUILD, 'root');                       // the guest / (initramfs)
const sh = (cmd, args, opts = {}) => execFileSync(cmd, args, { stdio: ['ignore', 'inherit', 'inherit'], ...opts });
const out = (cmd, args) => execFileSync(cmd, args, { maxBuffer: 1 << 30 }).toString();
const sha256File = (p) => { const h = createHash('sha256'); h.update(fs.readFileSync(p)); return h.digest('hex'); };

function arg(name, dflt) { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt; }

const KERNEL = arg('kernel', '/boot/vmlinuz-linux');
// The modules must match the kernel we PACK, not the one this host happens to
// be RUNNING. On a rolling distro those diverge the moment a kernel package
// lands and the box has not rebooted: /boot/vmlinuz-linux is already the new
// kernel while `uname -r` still names the old one, whose /usr/lib/modules tree
// the upgrade deleted. Taking the version from `uname -r` then points MODROOT
// at nothing, every module "skips", and the initramfs ships with none —
// including virtio_net, so the guest boots with NO NETWORK: no relay tunnel,
// no egress, nothing answering health. That is not hypothetical: it happened
// on metal0 on 2026-09-04 (host running 7.1.6, /usr/lib/modules holding only
// 7.2.2) and only the updater's health gate and rollback saved the box.
//
// Distros keep the kernel image beside its own modules (Arch and Fedora both
// ship /usr/lib/modules/<ver>/vmlinuz), so the honest answer is: whichever
// module tree holds a byte-identical copy of the kernel we are about to pack.
function kverOfKernel(kernelPath) {
  let want;
  try { want = fs.readFileSync(kernelPath); } catch { return null; }
  let dirs = [];
  try { dirs = fs.readdirSync('/usr/lib/modules'); } catch { return null; }
  for (const ver of dirs) {
    try {
      const cand = path.join('/usr/lib/modules', ver, 'vmlinuz');
      if (fs.existsSync(cand) && Buffer.compare(want, fs.readFileSync(cand)) === 0) return ver;
    } catch { /* unreadable tree: try the next */ }
  }
  return null;
}
const KVER = arg('kver', kverOfKernel(KERNEL) || os.release());
const MODROOT = arg('modroot', `/usr/lib/modules/${KVER}`);

// Refuse a kernel/module mismatch before anything is built. Individual modules
// may legitimately be absent (a kernel with them built in), but a MODROOT that
// does not exist, or one holding a different kernel than the one being packed,
// means the module set cannot be right — and the guest that comes out of it
// fails in a way no test on this host would see.
if (!fs.existsSync(MODROOT))
  throw new Error(`no module tree at ${MODROOT} for the kernel being packed (${KERNEL}). `
    + `This host is running ${os.release()}; if that differs from the packed kernel, its modules were `
    + `removed by an upgrade. Reboot onto the packed kernel, or pass --kver/--modroot explicitly.`);
{
  const beside = path.join(MODROOT, 'vmlinuz');
  if (fs.existsSync(beside) && Buffer.compare(fs.readFileSync(KERNEL), fs.readFileSync(beside)) !== 0)
    throw new Error(`${MODROOT} holds a DIFFERENT kernel than ${KERNEL} — packing one kernel with `
      + `another's modules. Pass --kernel and --kver that belong together.`);
}
// a cheap seam for operators and for the test: resolve and exit, build nothing
if (process.argv.includes('--print-kver')) {
  console.log(JSON.stringify({ kernel: KERNEL, kver: KVER, modroot: MODROOT }));
  process.exit(0);
}

// IMAGE DEFAULTS. Getting these wrong is not a build detail: they are the two
// images that BECOME the measured enclave, so a default that drifts changes
// what the box attests to without anyone choosing it.
//
// supervisor -> the RELEASE'S OWN image (`enclave-supervisor:<release tag>`).
//   NOT the digest in enclaves/<flavor>/tinfoil-config.yml: the release
//   workflow rewrites that line inside its own checkout and commits back only
//   the wasm-manager repins, so the committed supervisor pin is a placeholder
//   last touched 2026-07-07 whose image ghcr has since garbage-collected — it
//   404s. And NOT `:latest`, which is the wrong flavor for a CPU build and
//   moves under you: on 2026-08-09 metal0 was built from `:latest` and
//   attested to a supervisor no release describes (that tag moved again within
//   the hour). A version tag is immutable in practice and names exactly the
//   code the fleet runs for that release.
// wasm-manager -> the flavor config's pin, which CI DOES repin and commit, so
//   it is current and specific.
const FLAVOR_YML = path.join(HERE, '..', 'enclaves', 'cpu', 'tinfoil-config.yml');
const pinnedRef = (containerName) => {
  try {
    const lines = fs.readFileSync(FLAVOR_YML, 'utf8').split('\n');
    let inContainer = false;
    for (const line of lines) {
      const n = line.match(/^\s*-\s*name:\s*"?([\w-]+)"?/);
      if (n) { inContainer = n[1] === containerName; continue; }
      if (!inContainer) continue;
      const m = line.match(/^\s*image:\s*"?([^"\s]+)"?/);
      if (m) return m[1];
    }
  } catch { /* no config (a bare checkout): fall through to the tag */ }
  return null;
};
// Any TAG is resolved to the digest it names right now, BEFORE the pull, so
// the manifest records exactly what went into the measurement and a rebuild is
// byte-identical. Without this a `--release` build is reproducible only by
// luck: metal/update.mjs runs unattended on a timer, and every run would
// re-resolve the tag and quietly produce a different, unpinnable image.
// oci-pull --resolve hashes the manifest body itself, so this pins to bytes we
// checked. A failure here is not fatal — the pull that follows reports the
// real error, and a dev pointing at a local registry should not be blocked.
const resolvePinned = (ref) => {
  if (/@sha256:[0-9a-f]{64}$/.test(ref)) return ref;
  try {
    const out = execFileSyncCapture('node', [path.join(HERE, 'oci-pull.mjs'), ref, '--resolve']);
    const line = out.trim().split('\n').pop().trim();
    if (/^\S+@sha256:[0-9a-f]{64}$/.test(line)) { console.log(`[build] pinned ${ref} -> ${line.split('@')[1]}`); return line; }
  } catch { /* fall through */ }
  console.error(`[build] WARNING: could not resolve ${ref} to a digest; the build will not be reproducible`);
  return ref;
};
const SUPERVISOR_REF = resolvePinned(arg('supervisor', RELEASE
  ? `ghcr.io/enclavehost/enclave-supervisor:${RELEASE}`
  : 'ghcr.io/enclavehost/enclave-supervisor:latest'));
// KEEP IN STEP WITH THE SUPERVISOR. These are two independently-tagged images
// that share a loopback control plane, and its token derivation changed in
// c1b7352c (raw fleet SECRET → HMAC(SECRET, "enclave vmmgr v1")). Pair a
// post-c1b7352c supervisor with an older manager and control auth fails
// SILENTLY in the only direction that looks healthy: /health falls back to its
// unauthenticated liveness subset, so the enclave keeps answering while
// advertising no volumes, no capacity and no nn probe.
const WASM_REF = resolvePinned(arg('wasm', pinnedRef('wasm-manager') || 'ghcr.io/enclavehost/enclave-wasm-manager:040ab777'));

console.log(`[build] kernel=${KERNEL} kver=${KVER}`);
console.log(`[build] supervisor=${SUPERVISOR_REF}`);
console.log(`[build] wasm=${WASM_REF}`);

fs.rmSync(BUILD, { recursive: true, force: true });
fs.mkdirSync(ROOT, { recursive: true });
fs.mkdirSync(DIST, { recursive: true });

// --- 1. pull the two images (digests captured for the manifest) --------------
// The manifest is the record of WHAT WAS BUILT — the thing a third party
// reproduces against. An unparseable digest line must stop the build, not be
// written down as 'unknown'.
function pull(ref, dest) {
  const r = execFileSyncCapture('node', [path.join(HERE, 'oci-pull.mjs'), ref, dest]);
  const line = r.trim().split('\n').pop().trim();
  if (!/^sha256:[0-9a-f]{64}$/.test(line))
    throw new Error(`oci-pull did not report a digest for ${ref} (got ${JSON.stringify(line.slice(0, 80))})`);
  return line;
}
const isPinned = (ref) => /@sha256:[0-9a-f]{64}$/.test(ref);
function execFileSyncCapture(cmd, args) {
  const r = spawnSync(cmd, args, { stdio: ['ignore', 'inherit', 'pipe'], maxBuffer: 1 << 28 });
  if (r.status !== 0) { process.stderr.write(r.stderr || ''); throw new Error(`${cmd} exited ${r.status}`); }
  return (r.stderr || Buffer.alloc(0)).toString();
}
const WASM_ROOT = path.join(ROOT, 'opt/roots/wasm');
fs.mkdirSync(path.dirname(WASM_ROOT), { recursive: true });
console.log('[build] pulling supervisor (guest root)…');
const supDigest = pull(SUPERVISOR_REF, ROOT);
console.log('[build] pulling wasm-manager (chroot)…');
const wasmDigest = pull(WASM_REF, WASM_ROOT);
// A tag was resolved, not pinned: say so once, loudly, and hand over the exact
// command that IS reproducible. A measurement built from a moving tag must
// never be curated into METAL_ALLOWED_MEASUREMENTS as if it were.
for (const [what, ref, dgst] of [['supervisor', SUPERVISOR_REF, supDigest], ['wasm-manager', WASM_REF, wasmDigest]]) {
  if (isPinned(ref)) continue;
  const repo = ref.replace(/[@:][^/@:]+$/, '');
  console.error(`[build] WARNING: ${what} was pulled by TAG (${ref}) — this build is NOT reproducible.`);
  console.error(`[build]          for a reproducible build, rebuild with:  --${what === 'supervisor' ? 'supervisor' : 'wasm'} ${repo}@${dgst}`);
}

// --- 2. strip the GPU payload from the wasm chroot (CPU flavor) --------------
console.log('[build] stripping GPU payload from wasm chroot…');
const stripGlobs = [
  'usr/local/cuda-12.6', 'usr/local/cuda',
  'usr/lib/x86_64-linux-gnu/libcudnn*', 'usr/lib/x86_64-linux-gnu/libnccl*',
  'usr/lib/x86_64-linux-gnu/libcublas*', 'usr/lib/x86_64-linux-gnu/libcusparse*',
  'usr/lib/x86_64-linux-gnu/libnpp*', 'usr/lib/x86_64-linux-gnu/libcufft*',
  'usr/lib/x86_64-linux-gnu/libcurand*', 'usr/lib/x86_64-linux-gnu/libnvrtc*',
  'usr/local/lib/libggml-cuda.so', 'usr/local/lib/libonnxruntime_providers_cuda.so',
  'usr/local/lib/libonnxruntime_providers_tensorrt.so',
];
let stripped = 0;
for (const g of stripGlobs) {
  for (const f of globSync(path.join(WASM_ROOT, g))) {
    const before = duBytes(f); fs.rmSync(f, { recursive: true, force: true }); stripped += before;
  }
}
console.log(`[build] stripped ${(stripped / 1e9).toFixed(2)} GB of GPU libraries`);

// --- 3. flavor env (non-secret) from the CPU tinfoil-config ------------------
console.log('[build] extracting CPU flavor env…');
fs.mkdirSync(path.join(ROOT, 'opt/metal'), { recursive: true });
const flavorEnv = parseFlavorEnv(path.join(HERE, '..', 'enclaves', 'cpu', 'tinfoil-config.yml'), 'supervisor');
delete flavorEnv.PORT; delete flavorEnv.PUBLIC_URL;          // metal sets these
fs.writeFileSync(path.join(ROOT, 'opt/metal/flavor-env.json'), JSON.stringify(flavorEnv, null, 2));
// wasm image env (for the chroot) from its OCI config
const wasmCfg = JSON.parse(fs.readFileSync(path.join(WASM_ROOT, '.oci-config.json'), 'utf8'));
const wasmEnv = Object.fromEntries((wasmCfg.config.Env || []).map((e) => { const i = e.indexOf('='); return [e.slice(0, i), e.slice(i + 1)]; }));
fs.writeFileSync(path.join(ROOT, 'opt/metal/wasm-env.json'), JSON.stringify(wasmEnv, null, 2));

// --- 4. metal guest files ----------------------------------------------------
console.log('[build] installing metal guest files…');
const md = path.join(ROOT, 'opt/metal'); fs.mkdirSync(md, { recursive: true });
// shielded.mjs + shielded-probe.mjs are the CVM's client for an untrusted GPU on
// the host. They go INSIDE the measurement like everything else in /opt/metal --
// the guest half of the shielded tier is trusted code and must be attested. The
// worker it talks to is not, and is not shipped here: it runs on the host, is
// assumed hostile, and its address arrives unmeasured over fw_cfg.
for (const f of ['gsup.mjs', 'agent.mjs', 'shielded.mjs', 'shielded-probe.mjs', 'shielded-shm.mjs'])
  fs.copyFileSync(path.join(HERE, 'guest', f), path.join(md, f));
// init (PID1), with the kernel version substituted in
let init = fs.readFileSync(path.join(HERE, 'guest', 'init'), 'utf8').replaceAll('__KVER__', KVER);
fs.writeFileSync(path.join(ROOT, 'init'), init, { mode: 0o755 });
fs.chmodSync(path.join(ROOT, 'init'), 0o755);
// CA trust store — the slim base image ships none, so any TLS the guest does
// (the fleet tunnel's wss, the supervisor's Base RPC, ACME) needs a bundle. Copy
// the host's; it is public data (not a secret) and does not affect the measurement
// story (it is just root certs).
try {
  const caSrc = fs.realpathSync('/etc/ssl/certs/ca-certificates.crt');
  const caDst = path.join(ROOT, 'etc/ssl/certs/ca-certificates.crt');
  fs.mkdirSync(path.dirname(caDst), { recursive: true });
  fs.copyFileSync(caSrc, caDst);
  console.log(`[build] installed CA bundle (${fs.readFileSync(caDst, 'utf8').match(/BEGIN CERT/g)?.length || 0} roots)`);
} catch (e) { console.log(`[build] (no host CA bundle: ${e.message})`); }

// compile the static helpers (no kmod/iproute2/cryptsetup in the slim base image)
console.log('[build] compiling netup + minsmod + mverity…');
sh('gcc', ['-static', '-Os', '-o', path.join(md, 'netup'), path.join(HERE, 'guest', 'netup.c')]);
sh('gcc', ['-static', '-Os', '-o', path.join(md, 'minsmod'), path.join(HERE, 'guest', 'minsmod.c')]);
sh('gcc', ['-static', '-Os', '-o', path.join(md, 'mverity'), path.join(HERE, 'guest', 'mverity.c')]);

// --- 4b. the shielded engine backend, if this box has one --------------------
// The ggml backend that offloads masked linear ops to an untrusted GPU is
// TEE-SIDE code: it holds the one-time pads, the Freivalds secret, and every
// plaintext activation. So it belongs INSIDE the measurement, exactly like
// /opt/metal above -- bind-mounting it in at run time would put the trusted half
// of the tier outside the thing that attests to it, which is the whole point of
// the tier. The WORKER stays outside and unmeasured, because it is assumed
// hostile and its honesty is enforced by verification rather than attestation.
//
// THE BACKEND IS COMPILED HERE, FROM SOURCE, and this is a supply-chain
// property rather than a convenience. It used to ship as a prebuilt
// libggml-shielded.so committed to the repo, which gave the single most
// security-critical binary in the image two bad properties: a reviewer
// approving a change to it saw only `Bin 76040 -> 145344 bytes`, so no human
// ever read what went in; and anyone who compromised the workstation or the
// toolchain that produced it could put code holding the pads, the Freivalds
// secret and every plaintext activation INSIDE the measurement without ever
// touching this repo's source. Deriving the bytes here from reviewable C, with
// ggml's headers vendored (wasm/ggml-shielded/vendor/ggml) and its library
// taken from the digest-pinned engine image, removes both.
//
// It FAILS THE BUILD rather than falling back. A missing compiler or a header
// that disagrees with the pinned engine must not silently produce an image that
// serves a shielded card with stale or mismatched code -- and the old fallback,
// "copy whatever .so is lying in the overlay directory", is exactly the thing
// being removed.
//
// The overlay directory now carries DATA only: one <volume>.calib per model
// from wasm/ggml-shielded/shielded-calib. Those are public per-site constants (activation
// exponent + outlier channel indices) derived from public weights, they are
// text, and they are hashed into the manifest like everything else.
const SHIELDED_SRC = path.resolve(arg('shielded', path.join(HERE, 'shielded-overlay')));
const SHIELDED_CODE = path.resolve(arg('shielded-src', path.join(HERE, '..', 'wasm', 'ggml-shielded')));
const shieldedFiles = [];
let shieldedBuild = null;

// Compile wasm/ggml-shielded into the wasm chroot. Returns the build record for
// the manifest, or throws with a reason a human can act on.
function buildShieldedBackend(dstRoot) {
  const vendorDir = path.join(SHIELDED_CODE, 'vendor', 'ggml');
  const libDir = path.join(WASM_ROOT, 'usr/local/lib');
  if (!fs.existsSync(path.join(vendorDir, 'ggml-backend-impl.h')))
    throw new Error(`shielded: no vendored ggml headers at ${vendorDir}`);

  // The headers are vendored but the library comes from the pinned engine
  // image, so the two can drift apart at exactly one moment: an engine repin.
  // A header/library mismatch is undefined behaviour rather than an error, so
  // check it by name here instead of discovering it as a corrupt tenant later.
  const want = fs.readFileSync(path.join(vendorDir, 'VERSION'), 'utf8').trim();
  const soname = fs.existsSync(libDir)
    ? fs.readdirSync(libDir).find((f) => /^libggml-base\.so\.\d+\.\d+\.\d+$/.test(f)) : null;
  if (!soname)
    throw new Error(`shielded: the engine image ships no libggml-base.so.X.Y.Z under ${libDir}`);
  const have = soname.replace('libggml-base.so.', '');
  if (have !== want)
    throw new Error(`shielded: vendored ggml headers are ${want} but the pinned engine image ships ${have}. `
                  + `Refresh wasm/ggml-shielded/vendor/ggml (see its README) and rebuild.`);

  const objDir = path.join(BUILD, 'shielded-obj');
  fs.rmSync(objDir, { recursive: true, force: true });
  fs.mkdirSync(objDir, { recursive: true });
  const inc = ['-I' + vendorDir, '-I' + SHIELDED_CODE];
  const base = ['-O2', '-Wall', '-Wextra', '-fPIC'];
  // Flags mirror wasm/ggml-shielded/Makefile. Two are load-bearing rather than
  // taste: -ffp-contract=off on the field encoder, because the worker runs the
  // same source and an FMA would round differently on one side, making the
  // unmasking subtraction return noise; and the SIMD file compiled TWICE, since
  // the .so must load on any x86-64 (a SIGILL inside the engine is not a
  // degraded mode) and picks its kernels at run time after checking the two
  // builds agree.
  const units = [
    { src: 'shielded-field.c', obj: 'shielded-field.o', cc: 'cc',  flags: [...base, '-ffp-contract=off'] },
    { src: 'shielded-wire.c',  obj: 'shielded-wire.o',  cc: 'cc',  flags: base },
    { src: 'shielded-tee.c',   obj: 'shielded-tee.o',   cc: 'cc',  flags: base },
    { src: 'shielded-simd.c',  obj: 'shielded-simd-avx512.o', cc: 'cc',
      flags: [...base, '-O3', '-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl', '-mavx512vnni', '-DSH_SIMD_AVX512'] },
    { src: 'shielded-simd.c',  obj: 'shielded-simd-generic.o', cc: 'cc', flags: [...base, '-O3'] },
    { src: 'ggml-shielded.cpp', obj: 'ggml-shielded.o', cc: 'c++',
      flags: [...base, '-std=c++17', '-DGGML_BACKEND_DL', '-DGGML_BACKEND_SHARED'] },
  ];
  const sources = [];
  for (const u of units) {
    const src = path.join(SHIELDED_CODE, u.src);
    if (!fs.existsSync(src)) throw new Error(`shielded: missing source ${src}`);
    sh(u.cc, [...u.flags, ...inc, '-c', src, '-o', path.join(objDir, u.obj)]);
    if (!sources.some((x) => x.name === u.src))
      sources.push({ name: u.src, sha256: sha256File(src) });
  }
  for (const h of fs.readdirSync(SHIELDED_CODE).filter((f) => /\.h$/.test(f)).sort())
    sources.push({ name: h, sha256: sha256File(path.join(SHIELDED_CODE, h)) });

  // rpath is the IN-CHROOT path, not the builder's. The prebuilt binary this
  // replaces carried a RUNPATH into the scratch directory it happened to be
  // compiled in, which resolved only because the engine had already loaded ggml.
  fs.mkdirSync(dstRoot, { recursive: true });
  const so = path.join(dstRoot, 'libggml-shielded.so');
  sh('c++', ['-shared', '-o', so, ...units.map((u) => path.join(objDir, u.obj)),
             '-L' + libDir, '-lggml-base', '-lpthread', '-lm',
             '-Wl,-rpath,/usr/local/lib']);
  fs.chmodSync(so, 0o755);
  const ver = (c) => { try { return out(c, ['--version']).split('\n')[0].trim(); } catch { return 'unknown'; } };
  return { ggml: have, cc: ver('cc'), cxx: ver('c++'), sources,
           vendoredHeaders: fs.readdirSync(vendorDir).filter((f) => /\.h$/.test(f)).sort()
             .map((f) => ({ name: f, sha256: sha256File(path.join(vendorDir, f)) })) };
}

if (fs.existsSync(SHIELDED_SRC)) {
  console.log('[build] compiling the shielded engine backend from source…');
  // INTO THE WASM CHROOT, not the guest root. The manager and every tenant's
  // wasmtime run chrooted into /opt/roots/wasm, so that is the filesystem the
  // paths in _shielded_tenant_env resolve against -- GGML_BACKEND_PATH and
  // SHIELDED_CALIB are read by the tenant process, not by the guest's PID 1.
  // Installed at the guest root instead, the files are present, hashed into the
  // manifest, and invisible to the only code that opens them: the tenant simply
  // logs "NO calibration", claims nothing, and runs every matmul in the enclave
  // -- which looks like a working deployment that has quietly stopped using the
  // card it is paying for.
  const dstRoot = path.join(WASM_ROOT, 'opt/enclave/shielded');
  const record = (to, r) => shieldedFiles.push({
    path: `/opt/roots/wasm/opt/enclave/shielded/${r}`,
    tenantPath: `/opt/enclave/shielded/${r}`,
    bytes: fs.statSync(to).size, sha256: sha256File(to),
  });

  shieldedBuild = buildShieldedBackend(dstRoot);          // throws rather than shipping stale code
  record(path.join(dstRoot, 'libggml-shielded.so'), 'libggml-shielded.so');
  console.log(`[build]   built from source against ggml ${shieldedBuild.ggml} (${shieldedBuild.cc})`);

  // DATA ONLY from the overlay directory. A .so found here is ignored and said
  // out loud: it is the artifact this build step exists to stop shipping, and
  // silently preferring either one would make "which backend is in the image"
  // unanswerable from the source.
  const walk = (src, rel = '') => {
    for (const ent of fs.readdirSync(src, { withFileTypes: true }).sort((a, b) => a.name < b.name ? -1 : 1)) {
      const from = path.join(src, ent.name), r = rel ? `${rel}/${ent.name}` : ent.name;
      if (ent.isDirectory()) { walk(from, r); continue; }
      if (!ent.isFile()) continue;
      if (/\.so$/.test(r)) {
        console.log(`[build]   (ignoring prebuilt ${r}; the backend is compiled from source)`);
        continue;
      }
      const to = path.join(dstRoot, r);
      fs.mkdirSync(path.dirname(to), { recursive: true });
      fs.copyFileSync(from, to);
      record(to, r);
    }
  };
  walk(SHIELDED_SRC);
  for (const f of shieldedFiles)
    console.log(`[build]   ${f.tenantPath} (in the wasm chroot)  ${(f.bytes / 1e6).toFixed(2)} MB  sha256:${f.sha256.slice(0, 16)}…`);
} else {
  console.log('[build] (no shielded overlay at ' + SHIELDED_SRC + '; this image serves no shielded GPU)');
}

// --- 5. kernel modules (decompress the exact set we insmod, keep the tree) ---
console.log('[build] collecting kernel modules…');
const wantModules = [
  'kernel/drivers/firmware/qemu_fw_cfg.ko',              // read deployment config (name/relay/token) out-of-band
  'kernel/net/core/failover.ko',
  'kernel/drivers/net/net_failover.ko',
  'kernel/drivers/net/virtio_net.ko',
  'kernel/drivers/virt/coco/guest/tsm_report.ko',
  'kernel/drivers/virt/coco/sev-guest/sev-guest.ko',
  'kernel/drivers/virt/coco/tdx-guest/tdx-guest.ko',
  // attested model volumes: dm-verity over a read-only virtio-blk disk. Load
  // order matters and so does completeness — insmod resolves no dependencies
  // itself, and a missing one fails as a bare "No such file or directory" from
  // finit_module (unresolved symbols), not as anything that names the module.
  // dm-verity needs dm-mod (which also registers the /dev/mapper/control node
  // mverity opens), dm-bufio, and reed_solomon (built in for the optional FEC
  // path). virtio_blk and ext4 are built into this kernel, so neither needs a
  // module.
  'kernel/drivers/md/dm-mod.ko',
  'kernel/drivers/md/dm-bufio.ko',
  'kernel/lib/reed_solomon/reed_solomon.ko',
  'kernel/drivers/md/dm-verity.ko',
  // AF_VSOCK to the host, for the shielded worker: vsock core, then the virtio
  // transport's common half, then the transport itself. Harmless on a box that
  // attaches no vhost-vsock device -- the socket family exists and nothing
  // listens; the guest probe checks /dev/vsock before telling tenants.
  'kernel/net/vmw_vsock/vsock.ko',
  'kernel/net/vmw_vsock/vmw_vsock_virtio_transport_common.ko',
  'kernel/net/vmw_vsock/vmw_vsock_virtio_transport.ko',
];
// Shipped in the image but NOT in modules.list: init loads these itself, with
// parameters and behind host config. cpuidle-haltpoll (CONFIG_HALTPOLL_CPUIDLE=m
// in the pinned Arch kernel; its governor is built in, CONFIG_CPU_IDLE_GOV_
// HALTPOLL=y; no module dependencies) lets an idle vCPU poll for a bounded
// window before it halts, so the shielded worker's reply lands on a running
// vCPU instead of costing a VM exit and an injected interrupt per exchange.
// The window is a tuning parameter, so it rides fw_cfg (unmeasured), not the
// cmdline; only the driver's presence is measured. See metal/guest/init.
const optionalModules = [
  'kernel/drivers/cpuidle/cpuidle-haltpoll.ko',
];
const modList = [];
for (const rel of [...wantModules, ...optionalModules]) {
  const src = path.join(MODROOT, rel + '.zst');
  const plain = path.join(MODROOT, rel);
  const dstRel = rel;                                       // keep same layout in the initramfs
  const dst = path.join(ROOT, 'lib/modules', KVER, dstRel);
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  if (fs.existsSync(src)) { fs.writeFileSync(dst, execFileSync('zstd', ['-dc', src], { maxBuffer: 1 << 28 })); modList.push(dstRel); }
  else if (fs.existsSync(plain)) { fs.copyFileSync(plain, dst); modList.push(dstRel); }
  else console.log(`[build]   (skip missing ${rel})`);
}
fs.writeFileSync(path.join(md, 'modules.list'),
  modList.filter((m) => !optionalModules.includes(m)).join('\n') + '\n');
console.log(`[build] modules: ${modList.map((m) => path.basename(m)).join(', ')}`);

// --- 6. copy the kernel + record hashes --------------------------------------
fs.copyFileSync(KERNEL, path.join(DIST, 'vmlinuz'));
const kernelSha = sha256File(path.join(DIST, 'vmlinuz'));

// --- 7. manifest (written into the image AND to dist; the in-image copy lets
// the agent report exactly what it was built from) ---------------------------
// The MEASURED cmdline carries only the mode; everything else is fw_cfg (unmeasured).
const cmdlineTemplate = 'console=ttyS0 root=/dev/ram0 rootfstype=ramfs quiet metal.mode=${MODE}';
const manifest = {
  builtWith: 'metal/build-image.mjs',
  flavor: 'cpu',
  ...(RELEASE ? { release: RELEASE } : {}),   // the published tag this image was built from (metal/update.mjs reads it)
  images: { supervisor: { ref: SUPERVISOR_REF, digest: supDigest, pinned: isPinned(SUPERVISOR_REF) },
            wasmManager: { ref: WASM_REF, digest: wasmDigest, pinned: isPinned(WASM_REF) } },
  // Whether THIS build is reproducible, stated rather than assumed. A tag ref
  // resolves to whatever the registry serves today, so the same command a week
  // later yields a different launch measurement — which is precisely the claim
  // the Metal release allowlist rests on ("anyone can rebuild the release and
  // reproduce each measurement", metal/PROTOCOL.md). Only a build whose every
  // input is digest-pinned can carry that claim, so it is recorded per build
  // instead of asserted in prose.
  // A prebuilt binary inside the measurement cannot carry this claim either: the
  // shielded backend is the most security-critical code in the image, and if it
  // were copied in rather than compiled from the source recorded below, nobody
  // could rebuild this measurement and check it. `shieldedBuild` is non-null
  // exactly when it was compiled here.
  reproducible: isPinned(SUPERVISOR_REF) && isPinned(WASM_REF)
                && (shieldedFiles.length === 0 || shieldedBuild !== null),
  kernel: { path: KERNEL, kver: KVER, sha256: kernelSha },
  modules: modList,
  // Every shielded file baked in, by hash. Empty on a box with no shielded card.
  // In the manifest rather than only in the build log because these bytes are
  // inside the launch measurement and a reader should be able to enumerate them.
  shielded: shieldedFiles,
  // How the backend in `shielded` came to exist: the ggml it was compiled
  // against, the toolchain that did it, and the hash of every source file that
  // went in. Two builders comparing measurements can see WHICH input differed
  // instead of only that the answer did. The toolchain is recorded rather than
  // pinned, and that is the honest remaining gap: same source and same ggml on
  // a different compiler yields different bytes and therefore a different
  // measurement. Pinning it needs the toolchain in a digest-pinned image, which
  // is a larger change than this one.
  ...(shieldedBuild ? { shieldedBuild } : {}),
  cmdlineTemplate,
  // How attached model volumes are bound to the hardware. The launcher puts the
  // digest of the volume table in HOST_DATA (SEV-SNP) / MRCONFIGID (TDX), which
  // the CPU signs into every report — NOT in the launch measurement, so this
  // measurement stays valid whatever models the box carries. Recompute the
  // digest from the RAD's volume table alone: one line per volume, sorted,
  // newline-separated, with a trailing newline.
  volumeBinding: {
    field: 'HOST_DATA (SNP report offset 0xC0, 32 bytes); MRCONFIGID on TDX',
    canonicalLine: 'name|alg|root|salt|dataBlockSize|hashBlockSize|dataBlocks|hashStartBlock|sd(1/0)|gguf',
    digest: 'sha256(lines.sort().join("\\n") + "\\n")',
    note: 'Volume images are built by metal/volumes.mjs and are themselves reproducible: '
        + 'same source tree in, same verity root hash out. The guest reads HOST_DATA back '
        + 'from its own report and refuses to mount a table that does not hash to it.',
  },
  note: 'The SEV-SNP launch measurement is a function of (OVMF, this kernel, the '
      + 'initramfs, and the final cmdline). Recompute it independently with '
      + 'sev-snp-measure using the sha256 fields here; metal/verify.mjs checks a '
      + 'live report against the pinned launch measurement.',
};
fs.writeFileSync(path.join(md, 'manifest.json'), JSON.stringify(manifest, null, 2));

// --- 8. pack the initramfs (newc cpio + gzip), unprivileged + REPRODUCIBLE ---
// The whole trust story rests on a third party rebuilding this image and getting
// the SAME launch measurement, so the initramfs must be byte-identical across
// builds: fixed mtimes, sorted entry order, and gzip -n (no timestamp/name in
// the gzip header). The kernel applies root:root ownership at unpack regardless
// of on-disk uid, so our uid doesn't enter the image.
console.log('[build] packing initramfs (reproducible)…');
const initrd = path.join(DIST, 'initramfs.cpio.gz');
execSync('find . -exec touch --no-dereference -d @0 {} +', { cwd: ROOT, stdio: 'inherit', shell: '/bin/bash' });
execSync(`cd ${JSON.stringify(ROOT)} && find . -print0 | LC_ALL=C sort -z | `
       + `cpio --null -o -H newc --reproducible --quiet | gzip -n -9 > ${JSON.stringify(initrd)}`,
  { stdio: ['ignore', 'inherit', 'inherit'], shell: '/bin/bash' });
const initrdSha = sha256File(initrd);
manifest.initramfs = { sha256: initrdSha, bytes: fs.statSync(initrd).size };
fs.writeFileSync(path.join(DIST, 'manifest.json'), JSON.stringify(manifest, null, 2));
// refresh the in-image manifest too (so its recorded initrd hash matches) — but
// that would change the initrd, so we deliberately keep the in-image manifest
// WITHOUT the initrd hash (it can't contain its own hash); dist/manifest.json is
// the authoritative one for verification.

fs.writeFileSync(path.join(DIST, 'cmdline'), cmdlineTemplate + '\n');

// --- 9. expected SNP launch measurement (reproducible reference) -------------
// Compute the launch digest the same way any third party can, for the default
// runtime cmdline. It is a function of (OVMF, kernel, initrd, cmdline, vcpus,
// cpu sig); the cmdline carries the deployment's public_url/relay/token, so the
// reference is per (vcpus × cmdline). metal/verify.mjs recomputes and compares.
try {
  const cpu = readCpuSig();
  const OVMF = arg('ovmf', '/usr/share/edk2/x64/OVMF.4m.fd');
  const vcpuList = (arg('measure-vcpus', '4,8,16')).split(',').map((s) => parseInt(s, 10)).filter(Boolean);
  const concreteCmdline = cmdlineTemplate
    .replace('${MODE}', 'snp').replace('${NAME}', 'metal0')
    .replace('${PUBLIC_URL}', '').replace('${RELAY}', '').replace('${TOKEN}', '');
  const measures = {};
  for (const vcpus of vcpuList) {
    const m = spawnSync(process.env.HOME + '/.local/bin/sev-snp-measure',
      ['--mode', 'snp', '--vcpus', String(vcpus),
        '--vcpu-family', String(cpu.family), '--vcpu-model', String(cpu.model), '--vcpu-stepping', String(cpu.stepping),
        '--vmm-type', 'QEMU', '--ovmf', OVMF,
        '--kernel', path.join(DIST, 'vmlinuz'), '--initrd', initrd,
        '--append', concreteCmdline, '--output-format', 'hex'],
      { encoding: 'utf8' });
    if (m.status === 0 && /^[0-9a-f]{96}$/.test(m.stdout.trim())) measures[vcpus] = m.stdout.trim();
  }
  if (Object.keys(measures).length) {
    manifest.expectedMeasurement = {
      note: 'SEV-SNP launch digest for the DEFAULT cmdline (empty public_url/relay/token), '
          + 'per vcpu count. A non-empty cmdline changes this — recompute with the values in dist/cmdline.',
      ovmf: OVMF, ovmfSha256: sha256File(OVMF), cpuSig: cpu, vmmType: 'QEMU',
      cmdline: concreteCmdline, byVcpus: measures,
    };
    fs.writeFileSync(path.join(DIST, 'manifest.json'), JSON.stringify(manifest, null, 2));
    console.log(`[build] expected measurement (4 vcpu): ${measures[4] ? measures[4].slice(0, 24) + '…' : '(n/a)'}`);
  }
} catch (e) { console.log(`[build] (sev-snp-measure unavailable: ${e.message}; measurement left unpinned)`); }

console.log(`\n[build] DONE`);
console.log(`[build]   vmlinuz            ${(fs.statSync(path.join(DIST, 'vmlinuz')).size / 1e6).toFixed(1)} MB  sha256:${kernelSha.slice(0, 16)}…`);
console.log(`[build]   initramfs.cpio.gz  ${(manifest.initramfs.bytes / 1e6).toFixed(1)} MB  sha256:${initrdSha.slice(0, 16)}…`);
console.log(`[build]   manifest.json      metal/dist/manifest.json`);

// ---- helpers ----------------------------------------------------------------
function globSync(pattern) {
  const dir = path.dirname(pattern), base = path.basename(pattern);
  if (!base.includes('*')) return fs.existsSync(pattern) ? [pattern] : [];
  if (!fs.existsSync(dir)) return [];
  const re = new RegExp('^' + base.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*') + '$');
  return fs.readdirSync(dir).filter((f) => re.test(f)).map((f) => path.join(dir, f));
}
function duBytes(p) { try { return parseInt(out('du', ['-sb', p]).split('\t')[0], 10) || 0; } catch { return 0; } }
function readCpuSig() {
  // family/model/stepping the VMSA is measured with (the launch digest depends
  // on it). Defaults are this box's EPYC (fam 26 / model 2 / stepping 1).
  const info = fs.readFileSync('/proc/cpuinfo', 'utf8');
  const g = (re, d) => { const m = info.match(re); return m ? parseInt(m[1], 10) : d; };
  return {
    family: parseInt(arg('cpu-family', String(g(/cpu family\s*:\s*(\d+)/, 26))), 10),
    model: parseInt(arg('cpu-model', String(g(/model\s*:\s*(\d+)/, 2))), 10),
    stepping: parseInt(arg('cpu-stepping', String(g(/stepping\s*:\s*(\d+)/, 1))), 10),
  };
}
function parseFlavorEnv(ymlPath, containerName) {
  // minimal parser for the `env:` list under the named container in a
  // tinfoil-config.yml (lines like `      - KEY: "value"`), non-secret only.
  const lines = fs.readFileSync(ymlPath, 'utf8').split('\n');
  const env = {}; let inContainer = false, inEnv = false;
  for (const line of lines) {
    if (/^\s*-\s*name:\s*"?([\w-]+)"?/.test(line)) { inContainer = RegExp.$1 === containerName; inEnv = false; continue; }
    if (!inContainer) continue;
    if (/^\s{4}env:\s*$/.test(line)) { inEnv = true; continue; }
    if (/^\s{4}\w/.test(line) && !/^\s{4}env:/.test(line)) inEnv = false;   // left env: block
    if (!inEnv) continue;
    const m = line.match(/^\s*-\s*([A-Z0-9_]+):\s*(.*)$/);
    if (m) {
      let v = m[2].trim();
      if (v.startsWith('"')) { const q = v.match(/^"((?:[^"\\]|\\.)*)"/); v = q ? q[1] : v.slice(1); }   // quoted: take up to closing quote
      else v = v.replace(/\s+#.*$/, '').trim();                                                          // unquoted: strip inline # comment
      env[m[1]] = v;
    }
  }
  return env;
}
