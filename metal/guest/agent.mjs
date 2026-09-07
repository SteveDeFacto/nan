#!/usr/bin/env node
// metal-agent — the in-CVM service that replaces everything the Tinfoil shim
// did for us, with no Tinfoil dependency:
//
//   1. Attestation. Generates the enclave's transport keypair inside the CVM,
//      asks the CPU for a hardware attestation report over configfs-tsm with
//      report_data[0:32] = sha256(transport pubkey SPKI), and serves the Remote
//      Attestation Document at http://127.0.0.1:<RAD_PORT>/.well-known/
//      enclave-attestation. The supervisor is pointed here via ATTESTATION_URL,
//      so it relays + parses this document exactly as it did the shim's.
//
//   2. Reachability. Behind CGNAT there is no inbound, so the agent dials OUT to
//      the relay and holds a fleet tunnel; the relay forwards the enclave's
//      public HTTP surface (/v1/*, /availability, /x/*) back over it. A colo box
//      with a public IP can skip this (set metal.public_url and front 443).
//
// No secret in this file leaves the CVM; the transport key is minted per boot.
import http from 'node:http';
import net from 'node:net';
import { createHash, createHmac, createDecipheriv, createPublicKey, diffieHellman, generateKeyPairSync, hkdfSync, randomBytes, sign } from 'node:crypto';
import { createRequire } from 'node:module';
import fs from 'node:fs';

// borrow ws from the supervisor image (/app); fall back to normal resolution so
// the agent also runs outside the CVM (harnesses, dev boxes)
const WebSocket = (() => {
  for (const base of ['/app/package.json', import.meta.url]) {
    try { return createRequire(base)('ws'); } catch {}
  }
  throw new Error("cannot resolve 'ws' (looked in /app and next to this file)");
})();

const MODE      = process.env.METAL_MODE || 'snp';
const NAME      = process.env.METAL_NAME || 'metal0';
const RAD_PORT  = parseInt(process.env.METAL_RAD_PORT || '8443', 10);
const SUP_URL   = process.env.METAL_SUP_URL || 'http://127.0.0.1:8080';
// the ONE destination anything arriving over the tunnel may reach, resolved
// once at boot so no frame can move it (see forward() / stream())
const SUP_HOST  = (() => { try { return new URL(SUP_URL).hostname; } catch { return '127.0.0.1'; } })();
const SUP_PORT  = (() => { try { return Number(new URL(SUP_URL).port) || 80; } catch { return 8080; } })();
const SUP_HOST_URL = `http://${SUP_HOST}:${SUP_PORT}`;
const RELAY_URL = process.env.METAL_RELAY_URL || '';
const RELAY_HOST = process.env.METAL_RELAY_HOST || '';    // Host/SNI when dialing via an egress helper
const TOKEN     = process.env.METAL_TUNNEL_TOKEN || '';
// Gate for the supervisor's scoped attach-signer (see below). Derived, never the
// raw fleet SECRET on the wire, and the untrusted relay never holds it.
const OPSIGN_TOKEN = process.env.SECRET
  ? createHmac('sha256', process.env.SECRET).update('enclave opsign v1').digest('hex') : '';
const log = (...a) => console.log('[agent]', ...a);

// --- transport identity (minted in-CVM) -------------------------------------
const { publicKey, privateKey } = generateKeyPairSync('ec', { namedCurve: 'prime256v1' });
const spkiDer = publicKey.export({ type: 'spki', format: 'der' });
const keyFp = createHash('sha256').update(spkiDer).digest();         // 32 bytes
log(`transport key fingerprint ${keyFp.toString('hex').slice(0, 16)}…`);

// --- dealt pads: the pad key, minted in-CVM like the transport key ----------
// shielded/dealer/PLAN.md (P4a). The X25519 public half rides in the RAD
// document (the relay records it at attach); the platform boxes this box's
// pad seed to it. After a successful attach the agent fetches the seed, opens
// it here, and leaves {seed, seed_id, sk, ledger_pk, window_url} for the wasm
// manager in METAL_PADS_DIR/bootstrap.json (root-only) - never on a socket,
// since tenants share the loopback namespace. Ledger windows are relayed on
// the RAD port (POST /pads/window): a signed reserve by this transport key,
// answered with the platform ledger's own signature, which the engine pins.
const PADS_DIR = process.env.METAL_PADS_DIR || '/run/enclave/pads';
const padPair = generateKeyPairSync('x25519');
const padPkHex = Buffer.from(padPair.publicKey.export({ type: 'spki', format: 'der' })).subarray(-32).toString('hex');
const padSkHex = Buffer.from(padPair.privateKey.export({ type: 'pkcs8', format: 'der' })).subarray(-32).toString('hex');
function relayHttpBase() {
  const u = new URL(RELAY_URL);
  return (u.protocol === 'wss:' ? 'https://' : 'http://') + u.host;
}
async function padsCall(path, body) {
  const r = await fetch(relayHttpBase() + path, {
    method: body ? 'POST' : 'GET',
    headers: { 'content-type': 'application/json', ...(RELAY_HOST ? { host: RELAY_HOST } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  return { status: r.status, body: await r.json().catch(() => ({})) };
}
// The canonical request line set relay/pads.mjs verifies (signedMessage):
// "enclave-pads-<kind>\n<name>\n<fields...>\n<nonce>", ECDSA P-256 over sha256.
function padsSign(kind, fields) {
  const nonce = randomBytes(16).toString('hex');
  const msg = ['enclave-pads-' + kind, NAME, ...fields.map(String), nonce].join('\n');
  return { nonce, sig: sign('sha256', Buffer.from(msg), privateKey).toString('hex') };
}
async function padsWindow(want, seed_id) {
  const { nonce, sig } = padsSign('reserve', [seed_id, want]);
  return padsCall('/v1/pads/reserve', { name: NAME, seed_id, want, nonce, sig });
}
// X25519(epk, pad key) -> HKDF-SHA512(shared, salt = epk || pad pk, "enclave-pads-seed-box") -> ChaCha20-Poly1305, tag last.
function openSeedBox({ epk, nonce, box }) {
  const peer = createPublicKey({ key: Buffer.concat([Buffer.from('302a300506032b656e032100', 'hex'), Buffer.from(epk, 'hex')]), format: 'der', type: 'spki' });
  const shared = diffieHellman({ privateKey: padPair.privateKey, publicKey: peer });
  const key = Buffer.from(hkdfSync('sha512', shared, Buffer.concat([Buffer.from(epk, 'hex'), Buffer.from(padPkHex, 'hex')]), 'enclave-pads-seed-box', 32));
  const ct = Buffer.from(box, 'hex');
  const d = createDecipheriv('chacha20-poly1305', key, Buffer.from(nonce, 'hex'), { authTagLength: 16 });
  d.setAuthTag(ct.subarray(-16));
  return Buffer.concat([d.update(ct.subarray(0, -16)), d.final()]);
}
let padsBootstrapped = false;
async function padsBootstrap() {
  if (padsBootstrapped) return;
  const key = await padsCall('/v1/pads/key');
  if (key.status !== 200 || !/^[0-9a-f]{64}$/.test(String(key.body.key || ''))) { log(`pads: the platform has no pad ledger (http ${key.status}); dealt pads off`); return; }
  const { nonce, sig } = padsSign('seed', []);
  const r = await padsCall('/v1/pads/seed', { name: NAME, nonce, sig });
  if (r.status !== 200) { log(`pads: seed refused (http ${r.status} ${r.body.error || ''})`); return; }
  const seed = openSeedBox(r.body);
  if (seed.length !== 32) throw new Error('seed box opened to the wrong size');
  fs.mkdirSync(PADS_DIR, { recursive: true, mode: 0o700 });
  const out = { name: NAME, seed_id: r.body.seed_id, epoch: r.body.epoch, seed: seed.toString('hex'), sk: padSkHex,
                ledger_pk: key.body.key, window_url: `http://127.0.0.1:${RAD_PORT}/pads/window` };
  const file = `${PADS_DIR}/bootstrap.json`, tmp = file + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(out), { mode: 0o600 });
  fs.renameSync(tmp, file);
  padsBootstrapped = true;
  log(`pads: seed ${r.body.seed_id} (epoch ${r.body.epoch}) installed for the wasm manager at ${file}`);
}

// --- hardware attestation via configfs-tsm ----------------------------------
const TSM = '/sys/kernel/config/tsm/report';
function tsmReport(reportData64) {
  // one entry per call; the kernel serialises access and bumps `generation`
  const dir = `${TSM}/metal-${process.pid}`;
  fs.mkdirSync(dir);
  try {
    fs.writeFileSync(`${dir}/inblob`, reportData64);                 // 64 bytes
    const outblob = fs.readFileSync(`${dir}/outblob`);               // raw report
    let auxblob = null, provider = null;
    try { auxblob = fs.readFileSync(`${dir}/auxblob`); } catch {}    // cert chain (VCEK…)
    try { provider = fs.readFileSync(`${dir}/provider`, 'utf8').trim(); } catch {}
    return { outblob, auxblob, provider };
  } finally { try { fs.rmdirSync(dir); } catch {} }
}

// Build a RAD whose report_data binds an arbitrary 32-byte prefix (the default
// is the transport-key fingerprint; the fleet tunnel's attestation attach asks
// for sha256(transportKey || challenge) so the quote is FRESH per attach).
function radWithReportData(first32) {
  const reportData = Buffer.alloc(64);
  first32.copy(reportData, 0);
  let doc;
  if (MODE === 'dev') {
    // the same clearly-labeled UNATTESTED shape buildRad() serves in dev mode,
    // with the challenge binding in the report-data slot (nothing here proves
    // hardware; the relay only accepts it behind its own dev gate)
    const body = Buffer.alloc(0x4a0);
    reportData.copy(body, 0x50);
    doc = { format: 'dev-unattested-metal-v1', body: body.toString('base64') };
  } else {
    const { outblob, auxblob, provider } = tsmReport(reportData);
    const fmt = (provider || '').includes('tdx') || MODE === 'tdx' ? 'tdx-guest-metal-v1' : 'sev-snp-guest-metal-v1';
    doc = { format: fmt, body: outblob.toString('base64') };
    if (auxblob) doc.certs = auxblob.toString('base64');
  }
  doc.transportKey = spkiDer.toString('base64'); doc.transportKeyFp = keyFp.toString('hex'); doc.name = NAME;
  doc.padKey = padPkHex;                                                  // the relay records it at attach (dealt pads)
  try { doc.manifest = JSON.parse(fs.readFileSync('/opt/metal/manifest.json', 'utf8')); } catch {}
  // Attached model volumes, as the guest actually set them up (gsup writes this
  // after checking the table against the measured metal.vols digest). It is
  // published UNSIGNED on purpose: a verifier recomputes the digest from these
  // entries, folds it into the cmdline, and recomputes the launch measurement —
  // if anything here were altered the measurement would stop matching the
  // quote. So the hardware ends up vouching for which models are mounted.
  try { doc.volumes = JSON.parse(fs.readFileSync('/run/metal-volumes.json', 'utf8')); } catch {}
  return doc;
}
// Quote bound to a relay challenge: report_data = sha256(transportKey SPKI || nonce).
function radForChallenge(nonceB64) {
  const nonce = Buffer.from(nonceB64, 'base64');
  const bind = createHash('sha256').update(Buffer.concat([spkiDer, nonce])).digest();
  return radWithReportData(bind);
}

let radCache = null, radAt = 0;
function buildRad() {
  if (radCache && Date.now() - radAt < 10000) return radCache;
  const reportData = Buffer.alloc(64);
  keyFp.copy(reportData, 0);
  let doc;
  if (MODE === 'dev') {
    // Clearly-labeled UNATTESTED document for pre-SNP bring-up. Same shape, but
    // the format name tells every verifier this proves nothing about hardware.
    const body = Buffer.alloc(0x4a0);
    reportData.copy(body, 0x50);                                     // keep the key-binding surface
    doc = { format: 'dev-unattested-metal-v1', body: body.toString('base64') };
  } else {
    const { outblob, auxblob, provider } = tsmReport(reportData);
    const fmt = (provider || '').includes('tdx') || MODE === 'tdx'
      ? 'tdx-guest-metal-v1' : 'sev-snp-guest-metal-v1';
    doc = { format: fmt, body: outblob.toString('base64') };
    if (auxblob) doc.certs = auxblob.toString('base64');
  }
  doc.transportKey = spkiDer.toString('base64');
  doc.transportKeyFp = keyFp.toString('hex');
  doc.padKey = padPkHex;
  doc.name = NAME;
  try { doc.manifest = JSON.parse(fs.readFileSync('/opt/metal/manifest.json', 'utf8')); } catch {}
  // Attached model volumes, as the guest actually set them up (gsup writes this
  // after checking the table against the measured metal.vols digest). It is
  // published UNSIGNED on purpose: a verifier recomputes the digest from these
  // entries, folds it into the cmdline, and recomputes the launch measurement —
  // if anything here were altered the measurement would stop matching the
  // quote. So the hardware ends up vouching for which models are mounted.
  try { doc.volumes = JSON.parse(fs.readFileSync('/run/metal-volumes.json', 'utf8')); } catch {}
  radCache = doc; radAt = Date.now();
  return doc;
}

// --- RAD endpoint (loopback; the supervisor's ATTESTATION_URL points here) ---
http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/pads/window') {
    // a ledger window for a tenant engine: signed here, answered by the platform
    let raw = '';
    req.on('data', (c) => { raw += c; if (raw.length > 4096) req.destroy(); });
    req.on('end', async () => {
      let body; try { body = JSON.parse(raw || '{}'); } catch { body = {}; }
      const want = Number(body.want), seed_id = String(body.seed_id || '');
      const reply = (status, b) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(b)); };
      if (!Number.isInteger(want) || want < 1 || want > 4096 || !/^[0-9a-f]{32}$/.test(seed_id)) return reply(400, { error: 'bad_request' });
      try { const r = await padsWindow(want, seed_id); reply(r.status, r.body); }
      catch (e) { reply(502, { error: 'relay_unreachable', message: String(e && e.message || e) }); }
    });
    return;
  }
  if (req.url.startsWith('/.well-known/enclave-attestation') || req.url === '/health') {
    try {
      const body = req.url === '/health' ? { ok: true } : buildRad();
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify(body));
    } catch (e) {
      res.writeHead(503, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: String(e && e.message || e) }));
    }
  } else { res.writeHead(404); res.end(); }
}).listen(RAD_PORT, '127.0.0.1', () => log(`RAD endpoint on 127.0.0.1:${RAD_PORT} (mode=${MODE})`));

// warm the report so the first /v1/attestation is instant and to fail loudly if
// the silicon can't attest in a non-dev build.
try { const d = buildRad(); log(`attestation ready: format=${d.format}`); }
catch (e) { log(`WARNING: attestation unavailable: ${e.message}`); }

// --- fleet tunnel: forward the enclave's public HTTP surface over an outbound
// wss the relay accepts. Framed request/response (Phase 1 is buffered; SSE
// streaming rides a chunked-frame upgrade). ------------------------------------
// The relay is a ROUTER, not a trusted party — that is the whole premise of the
// tunnel. `new URL(frame.path, SUP_URL)` honored an ABSOLUTE path frame, so a
// relay that sent `http://169.254.169.254/…` (or any host on this box's
// network) got the agent to dial it from INSIDE the CVM: a full SSRF primitive
// handed to the one component the design says not to trust. The raw-stream path
// below always pinned the supervisor's port for exactly this reason; the
// buffered path now does the same. Only the path and query come off the wire.
function forward(frame, send) {
  const reply = (status, text) => send({ t: 'res', id: frame.id, status, headers: {}, body: Buffer.from(text).toString('base64') });
  const p = String(frame.path || '');
  if (!p.startsWith('/') || p.startsWith('//')) return reply(400, 'bad request target');
  let u; try { u = new URL(SUP_HOST_URL + p); } catch { return reply(400, 'bad request target'); }
  const body = frame.body ? Buffer.from(frame.body, 'base64') : undefined;
  const req = http.request({ host: SUP_HOST, port: SUP_PORT, path: u.pathname + u.search,
                             method: frame.method || 'GET', headers: frame.headers || {} }, (r) => {
    const chunks = [];
    r.on('data', (c) => chunks.push(c));
    r.on('end', () => send({
      t: 'res', id: frame.id, status: r.statusCode,
      headers: r.headers, body: Buffer.concat(chunks).toString('base64'),
    }));
  });
  req.on('error', (e) => send({ t: 'res', id: frame.id, status: 502, headers: {}, body: Buffer.from(String(e.message)).toString('base64') }));
  if (body) req.write(body);
  req.end();
}

// --- raw streams (Phase D): the hub splices a client's WebSocket UPGRADE into
// a plain TCP connection to the supervisor, which answers the handshake itself
// — this carries the /x/<id>/tls and /x/<id>/https bridges (TLS terminating in
// THIS CVM) out through the tunnel. Frames: s+ open · s= ack · sd data · sx
// close. The only reachable target is the supervisor's own port — the tunnel
// must never become a generic proxy into the guest (SUP_HOST/SUP_PORT above).
const MAX_GUEST_STREAMS = 128, GUEST_STREAM_IDLE_MS = 15 * 60_000;
const streams = new Map();                        // sid -> net.Socket
function stream(f, send) {
  if (f.t === 's+') {
    if (streams.size >= MAX_GUEST_STREAMS) return send({ t: 's=', sid: f.sid, ok: false, err: 'stream cap' });
    const sock = net.connect({ host: SUP_HOST, port: SUP_PORT });
    streams.set(f.sid, sock);
    sock.setTimeout(GUEST_STREAM_IDLE_MS, () => sock.destroy());
    sock.on('connect', () => send({ t: 's=', sid: f.sid, ok: true }));
    sock.on('data', (c) => send({ t: 'sd', sid: f.sid, d: c.toString('base64') }));
    const bye = () => { if (streams.delete(f.sid)) send({ t: 'sx', sid: f.sid }); };
    sock.on('error', (e) => { if (streams.has(f.sid) && !sock.remotePort) { streams.delete(f.sid); send({ t: 's=', sid: f.sid, ok: false, err: e.code || 'connect failed' }); } });
    sock.on('close', bye);
    return;
  }
  const sock = streams.get(f.sid);
  if (!sock) return;
  if (f.t === 'sd') { try { sock.write(Buffer.from(f.d || '', 'base64')); } catch {} }
  else if (f.t === 'sx') { streams.delete(f.sid); sock.destroy(); }
}

// PROVE WHO THIS BOX IS, not just what it runs. A quote proves the IMAGE and the
// transport key is minted per boot, so on the attest path a name could be taken
// by any box running the same published release while its real owner was down -
// and with it the routing for that name's on-chain id. The registry OPERATOR key
// is the identity that survives a reboot, and it lives in the supervisor, so ask
// it to sign this attach's challenge. Best effort: a box with no registry key
// (it does not sell) simply sends nothing, and unregistered names stay
// first-come, exactly as before.
function attachSignature(nonce) {
  return new Promise((resolve) => {
    if (!OPSIGN_TOKEN) return resolve('');
    const body = JSON.stringify({ name: NAME, nonce });
    const req = http.request({ host: SUP_HOST, port: SUP_PORT, path: '/v1/internal/tunnel-attach-sig',
                               method: 'POST', timeout: 10000,
                               headers: { 'content-type': 'application/json',
                                          'content-length': Buffer.byteLength(body),
                                          'x-opsign-token': OPSIGN_TOKEN } }, (r) => {
      let buf = '';
      r.on('data', (c) => (buf += c));
      r.on('end', () => { try { resolve(JSON.parse(buf).signature || ''); } catch { resolve(''); } });
    });
    req.on('error', () => resolve(''));
    req.on('timeout', () => { req.destroy(); resolve(''); });
    req.write(body); req.end();
  });
}

// One-shot connectivity probe so failures are diagnosable from the journal:
// separate DNS from TCP-reachability, and IPv4 from IPv6.
async function probe() {
  const net = await import('node:net');
  const dns = await import('node:dns');
  let host = 'api.enclave.host';
  try { host = new URL(RELAY_URL).hostname; } catch {}
  try {
    const a = await dns.promises.resolve4(host).catch(() => []);
    const aaaa = await dns.promises.resolve6(host).catch(() => []);
    log(`probe: ${host} -> A=[${a.join(',')}] AAAA=[${aaaa.length ? aaaa.join(',') : 'none'}]`);
    for (const ip of a.slice(0, 1)) {
      await new Promise((res) => {
        const s = net.connect({ host: ip, port: 443, family: 4 });
        const t = setTimeout(() => { s.destroy(); log(`probe: TCP ${ip}:443 (v4) TIMEOUT`); res(); }, 6000);
        s.on('connect', () => { clearTimeout(t); log(`probe: TCP ${ip}:443 (v4) OK`); s.destroy(); res(); });
        s.on('error', (e) => { clearTimeout(t); log(`probe: TCP ${ip}:443 (v4) ${e.code}`); res(); });
      });
    }
  } catch (e) { log(`probe error: ${e.message}`); }
}

function connectTunnel() {
  if (!RELAY_URL) { log('no relay configured; tunnel disabled (RAD-only mode)'); return; }
  probe();
  let ws, alive = false;
  const dial = () => {
    log(`tunnel dialing ${RELAY_URL}`);
    // force IPv4: QEMU user-net (slirp) NATs IPv4 only, but api.enclave.host has
    // an AAAA record, so an unpinned dial hangs on the IPv6 attempt. RELAY_HOST,
    // when set (egress-helper mode), carries the real relay hostname for the Host
    // header even though the socket target is the helper.
    const headers = { 'x-metal-name': NAME, 'x-metal-token': TOKEN };
    if (!TOKEN) headers['x-metal-attest'] = '1';   // no token → prove identity by SEV-SNP quote (permissionless)
    if (RELAY_HOST) headers.host = RELAY_HOST;
    ws = new WebSocket(RELAY_URL, { headers, family: 4 });
    const send = (o) => { try { ws.send(JSON.stringify(o)); } catch {} };
    // Liveness, mirrored from the hub's side of the same contract: the hub
    // pings every 30s and culls a tunnel silent for 90s — but a half-open
    // socket (a WAN drop eats the FIN, NAT timeout) never fires 'close' HERE,
    // so this agent would hold a dead ws forever while the hub has already
    // detached the name and the box has silently left the fleet listing.
    // Any frame refreshes the clock; 90s without one means the path is gone:
    // terminate, and the existing 'close' path redials. The clock also starts
    // at dial time, so a connect that hangs is bounded by the same rule.
    let lastFrame = Date.now();
    const liveness = setInterval(() => {
      if (Date.now() - lastFrame > 90_000) {
        log('tunnel silent for 90s — assuming half-open, redialing');
        try { ws.terminate(); } catch {}
      }
    }, 15_000);
    ws.on('open', () => {
      alive = true; lastFrame = Date.now();
      send({ t: 'hello', name: NAME, mode: MODE, token: TOKEN, publicUrl: process.env.METAL_PUBLIC_URL || '', transportKeyFp: keyFp.toString('hex') });
      log('tunnel open');
    });
    ws.on('message', (data) => {
      lastFrame = Date.now();
      let f; try { f = JSON.parse(data); } catch { return; }
      if (f.t === 'req') forward(f, send);
      else if (f.t === 's+' || f.t === 'sd' || f.t === 'sx') stream(f, send);
      else if (f.t === 'ping') send({ t: 'pong' });
      else if (f.t === 'challenge') {         // permissionless attach: answer with a fresh quote over the nonce
        // ...and, when this box sells, a signature from the registry operator
        // key that OWNS this name on chain (attachSignature). The quote says
        // what runs; that signature says who it is.
        (async () => {
          try {
            const rad = radForChallenge(f.nonce);
            const operatorSig = await attachSignature(f.nonce);
            send({ t: 'attest', rad, ...(operatorSig ? { operatorSig } : {}) });
            log(`sent attestation for tunnel challenge${operatorSig ? ' + operator signature' : ''}`);
          } catch (e) { log(`attest failed: ${e.message}`); }
        })();
      } else if (f.t === 'attest-result') {
        log(`attest ${f.ok ? 'ACCEPTED' + (f.measurement ? ' meas=' + String(f.measurement).slice(0, 16) + '…' : '') : 'REJECTED: ' + f.reason}`);
        if (f.ok) padsBootstrap().catch((e) => log(`pads: bootstrap failed: ${e.message}`));
      }
    });
    ws.on('unexpected-response', (_req, res) => { log(`tunnel handshake rejected: HTTP ${res.statusCode}`); try { ws.terminate(); } catch {} });
    ws.on('close', () => { clearInterval(liveness); if (alive) log('tunnel closed'); alive = false; for (const s of [...streams.values()]) s.destroy(); streams.clear(); setTimeout(dial, 2000); });
    ws.on('error', (e) => { log(`tunnel error: ${e.code || ''} ${e.message || String(e)}`); try { ws.terminate(); } catch {} });
  };
  dial();
}
connectTunnel();
