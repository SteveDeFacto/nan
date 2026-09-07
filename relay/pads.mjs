// relay/pads.mjs — the platform half of dealt pads (shielded/dealer/PLAN.md):
// per-pVM seeds, the consumption LEDGER, and signed reserve-before-use windows.
//
// A phone-anchored trusted half (an attested pVM on the fleet tunnel) never
// mints its own masks: a dealer mints u = r.W for it and the pVM derives r from
// a seed only it and the platform know. Two things must live with the platform
// for that to be safe across reboots, and both are here:
//
//   the SEED for a pVM      derived, not stored: HKDF(master, keyFp, epoch). The
//                            dealer derives the same seed to mint; the pVM gets
//                            it once per boot, boxed to its X25519 pad key.
//   the LEDGER              the high-water mark of pad indices handed to each
//                            seed. A pVM asks for a window [mark, mark+W) and
//                            the mark advances BEFORE the answer leaves, so a
//                            pVM that reboots resumes past everything it could
//                            have used. The operator cannot roll it back: it is
//                            not on his box.
//
// Requests are authenticated by the pVM's attested Ed25519 transport key (the
// SPKI the tunnel bound at attach); windows are signed by the relay's Ed25519
// ledger key (GET /v1/pads/key) so the pVM refuses a window from anyone else.
//
// Endpoints (wired in api-relay.js):
//   GET  /v1/pads/key                       { key: <hex 32-byte Ed25519 public key> }
//   POST /v1/pads/seed     { name, nonce, sig }
//                          -> { seed_id, epoch, epk, nonce, box }  the pVM's seed,
//                             X25519(epk, padKey) -> HKDF-SHA512 -> ChaCha20-Poly1305
//   POST /v1/pads/reserve  { name, seed_id, want, nonce, sig }
//   POST /v1/pads/receipt  { name, seed_id, pads, tokens, nonce, sig }
//     the pVM's signed usage at the end of a run (pads = cells consumed,
//     tokens = prompt + generated); the platform bills and pays on these
//   GET  /v1/pads/receipts?seed_id=  { seed_id, pads, tokens, runs, last[] }
//   GET  /v1/pads/consumers          { consumers: [{ name, keyFp, padKey, seed_id, epoch, mark, issued }] }
//     every attached tunnel that offered a pad key: what a dealer daemon
//     serves (public keys and marks only; the seed is derived from keyFp by
//     whoever holds the master)
//                          -> { seed_id, lo, hi, iat, sig }
//   sig (requests) = Ed25519 over the canonical line set in signedMessage();
//   sig (windows)  = Ed25519 over windowMessage().
//
// State: <dir>/pads-ledger.json, written tmp-then-rename like every other relay
// store; the master seed and the ledger key are minted once and kept there.
import { createHash, createPrivateKey, createPublicKey, createCipheriv, diffieHellman, generateKeyPairSync,
         hkdfSync, randomBytes, sign as edSign, verify as edVerify } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

/* ---- the HTTP surface, shared by api-relay.js and the local hub ----------
 * Returns true when the request was one of ours (answered), false otherwise.
 * `json(res, status, body)` and `readBody(req, max) -> Buffer` are the host's. */
export function padsRouter({ ledger, store, dealerToken, json, readBody }) {
  return async function handle(req, res, url) {
    const p = url.pathname;
    if (!p.startsWith("/v1/pads/")) return false;
    if (!ledger || !store) { json(res, 503, { error: "pads_disabled", message: "no data dir for the pads ledger" }); return true; }
    if (p === "/v1/pads/key" && req.method === "GET") { json(res, 200, { key: ledger.key(), epoch: PADS_EPOCH }); return true; }
    if (p === "/v1/pads/pvm" && req.method === "GET") {
      const r = ledger.pvm(url.searchParams.get("name") || "");
      json(res, r ? 200 : 404, r || { error: "unknown_tunnel" }); return true;
    }
    if (p === "/v1/pads/ledger" && req.method === "GET") {
      const m = ledger.mark(url.searchParams.get("seed_id") || "");
      json(res, m ? 200 : 404, m || { error: "unknown_seed" }); return true;
    }
    if (p === "/v1/pads/consumers" && req.method === "GET") { json(res, 200, { consumers: ledger.consumers() }); return true; }
    if (p === "/v1/pads/receipts" && req.method === "GET") {
      const r = ledger.receipts(url.searchParams.get("seed_id") || "");
      json(res, r ? 200 : 404, r || { error: "unknown_seed" }); return true;
    }
    if ((p === "/v1/pads/seed" || p === "/v1/pads/reserve" || p === "/v1/pads/receipt") && req.method === "POST") {
      let body;
      try { body = JSON.parse((await readBody(req, 8192)).toString("utf8") || "{}"); }
      catch (e) { json(res, e.message === "body too large" ? 413 : 400, { error: "bad_json", message: e.message }); return true; }
      const r = p === "/v1/pads/seed" ? ledger.seed(body || {}) : p === "/v1/pads/reserve" ? ledger.reserve(body || {}) : ledger.receipt(body || {});
      json(res, r.status, r.body); return true;
    }
    if (p === "/v1/pads/shipments" && req.method === "GET") {
      const seed_id = url.searchParams.get("seed_id") || "";
      json(res, 200, { seed_id, shipments: store.list(seed_id) }); return true;
    }
    const ship = p.match(/^\/v1\/pads\/shipments\/([0-9a-f]{32})\/([A-Za-z0-9._-]{1,120})$/);
    if (ship) {
      if (req.method === "GET") {
        const f = store.file(ship[1], ship[2]);
        if (!f) { json(res, 404, { error: "no_such_shipment" }); return true; }
        const st = fs.statSync(f);
        res.writeHead(200, { "content-type": "application/octet-stream", "content-length": st.size, "cache-control": "private, max-age=3600" });
        fs.createReadStream(f).pipe(res); return true;
      }
      const auth = String(req.headers.authorization || "");
      if (!dealerToken || auth !== "Bearer " + dealerToken) { json(res, 403, { error: "dealer_only", message: "PUT/DELETE need the dealer's bearer" }); return true; }
      const plan = store.plan(ship[1], ship[2]);
      if (!plan) { json(res, 400, { error: "bad_name", message: "<seed_id>-<index0>-<count>.pads, for that seed" }); return true; }
      if (req.method === "DELETE") { json(res, store.remove(ship[1], ship[2]) ? 200 : 404, { removed: ship[2] }); return true; }
      if (req.method === "PUT") {
        const want = String(url.searchParams.get("sha256") || "").toLowerCase();
        if (!/^[0-9a-f]{64}$/.test(want)) { json(res, 400, { error: "need_sha256" }); return true; }
        const hash = createHash("sha256");
        const out = fs.createWriteStream(plan.tmp);
        let bytes = 0;
        req.on("data", (ch) => { hash.update(ch); bytes += ch.length; });
        req.pipe(out);
        out.on("finish", () => {
          const got = hash.digest("hex");
          if (got !== want) { try { fs.unlinkSync(plan.tmp); } catch {} return json(res, 409, { error: "sha256_mismatch", got, want }); }
          try { fs.renameSync(plan.tmp, plan.final); } catch (e) { return json(res, 500, { error: "store", message: e.message }); }
          json(res, 200, { stored: ship[2], bytes, sha256: got });
        });
        out.on("error", (e) => json(res, 500, { error: "store", message: e.message }));
        return true;
      }
      json(res, 405, { error: "method" }); return true;
    }
    json(res, 404, { error: "not_found" }); return true;
  };
}

/* ---- the shipment store --------------------------------------------------
 * The dealer PUTs finished shipments here and the phone prefetches them into
 * its own storage (the platform is the bank; an operator NVMe cache can front
 * it later). Files are ciphertext to a pad key, so GET is public; PUT/DELETE
 * take the dealer's bearer. Names are `<seed_id>-<index0>-<count>.pads`. */
const SHIP_NAME = /^([0-9a-f]{32})-(\d+)-(\d+)\.pads$/;
export function createShipmentStore({ dir }) {
  const root = path.join(dir, "pads-shipments");
  const seedDir = (seed_id) => path.join(root, seed_id);
  const okSeed = (s) => /^[0-9a-f]{32}$/.test(String(s || ""));
  return {
    root,
    /* Where an upload lands (tmp) and its final path; the name must carry the seed it is for. */
    plan(seed_id, name) {
      const m = SHIP_NAME.exec(String(name || ""));
      if (!okSeed(seed_id) || !m || m[1] !== seed_id) return null;
      fs.mkdirSync(seedDir(seed_id), { recursive: true });
      return { tmp: path.join(seedDir(seed_id), "." + name + ".part"), final: path.join(seedDir(seed_id), name), index0: Number(m[2]), count: Number(m[3]) };
    },
    list(seed_id) {
      if (!okSeed(seed_id)) return [];
      let names = [];
      try { names = fs.readdirSync(seedDir(seed_id)); } catch { return []; }
      return names.filter((n) => SHIP_NAME.test(n)).map((n) => {
        const m = SHIP_NAME.exec(n), st = fs.statSync(path.join(seedDir(seed_id), n));
        return { name: n, bytes: st.size, index0: Number(m[2]), count: Number(m[3]) };
      }).sort((a, b) => a.index0 - b.index0);
    },
    file(seed_id, name) {
      const p = this.plan(seed_id, name);
      return p && fs.existsSync(p.final) ? p.final : null;
    },
    remove(seed_id, name) {
      const p = this.plan(seed_id, name);
      if (!p) return false;
      try { fs.unlinkSync(p.final); return true; } catch { return false; }
    },
  };
}

export const PADS_EPOCH = Number(process.env.PADS_EPOCH || 1);   // bump (env) to re-key every pVM's seed; old shipments become foreign
export const MAX_WINDOW = 4096;
const RECEIPT_MEMORY = 64;     // per seed, the most recent receipts kept verbatim (totals are cumulative)
const NONCE_MEMORY = 256;                        // recent request nonces kept per seed (replay guard)

export function signedMessage(kind, fields) {
  return ["enclave-pads-" + kind, ...fields.map(String)].join("\n");
}
export function windowMessage(seed_id, lo, hi, iat) {
  return signedMessage("window", [seed_id, lo, hi, iat]);
}

/* The seed a (keyFp, epoch) gets, and its public name. Both sides of the deal
 * derive these: the relay to answer /v1/pads/seed, the dealer to mint. */
export function deriveSeed(master, keyFp, epoch = PADS_EPOCH) {
  const seed = Buffer.from(hkdfSync("sha512", master, Buffer.from(keyFp, "hex"), `enclave-pads-seed:${epoch}`, 32));
  const seed_id = Buffer.from(hkdfSync("sha512", master, Buffer.from(keyFp, "hex"), `enclave-pads-seed-id:${epoch}`, 16)).toString("hex");
  return { seed, seed_id };
}

/* Box 32 bytes to an X25519 public key: ephemeral X25519, HKDF-SHA512 over the
 * shared secret with both public keys as salt, ChaCha20-Poly1305. The C side
 * (the pVM) opens it with TweetNaCl's scalarmult + its own ChaCha20 + Poly1305. */
export function boxToPadKey(padKeyHex, plain) {
  const padKey = Buffer.from(padKeyHex, "hex");
  if (padKey.length !== 32) throw new Error("pad key must be 32 bytes");
  const eph = generateKeyPairSync("x25519");
  const epk = Buffer.from(eph.publicKey.export({ type: "spki", format: "der" })).subarray(-32);
  const peer = createPublicKey({ key: Buffer.concat([Buffer.from("302a300506032b656e032100", "hex"), padKey]), format: "der", type: "spki" });
  const shared = diffieHellman({ privateKey: eph.privateKey, publicKey: peer });
  const key = Buffer.from(hkdfSync("sha512", shared, Buffer.concat([epk, padKey]), "enclave-pads-seed-box", 32));
  const nonce = randomBytes(12);
  const c = createCipheriv("chacha20-poly1305", key, nonce, { authTagLength: 16 });
  const ct = Buffer.concat([c.update(plain), c.final(), c.getAuthTag()]);
  return { epk: epk.toString("hex"), nonce: nonce.toString("hex"), box: ct.toString("hex") };
}

function rawEd25519Public(keyObject) {
  return Buffer.from(keyObject.export({ type: "spki", format: "der" })).subarray(-32);
}

export function createPadsLedger({ dir, hub, log = console.log, masterSeed = null }) {
  const file = path.join(dir, "pads-ledger.json");
  let state = { master: null, ledgerKey: null, seeds: {} };
  try { state = { ...state, ...JSON.parse(fs.readFileSync(file, "utf8")) }; } catch {}
  if (masterSeed) state.master = Buffer.from(masterSeed, "hex").toString("hex");
  if (!state.master || state.master.length !== 64) state.master = randomBytes(32).toString("hex");
  if (!state.ledgerKey) {
    const kp = generateKeyPairSync("ed25519");
    state.ledgerKey = kp.privateKey.export({ type: "pkcs8", format: "pem" });
  }
  const priv = createPrivateKey(state.ledgerKey);
  const pub = createPublicKey(priv);
  const master = Buffer.from(state.master, "hex");
  const save = () => {
    fs.mkdirSync(dir, { recursive: true });
    const tmp = file + ".tmp";
    fs.writeFileSync(tmp, JSON.stringify(state, null, 1));
    fs.renameSync(tmp, file);
  };
  save();
  log(`[pads] ledger ${file}: ${Object.keys(state.seeds).length} seed(s), key ${rawEd25519Public(pub).toString("hex").slice(0, 16)}…`);

  /* The attested caller: the tunnel record for `name` must be a bound
   * attestation with a transport SPKI, and `sig` must be that key's
   * signature over the canonical message. */
  function callerOf(name, kind, fields, nonce, sig) {
    const t = hub && hub.info ? hub.info(name) : null;
    if (!t || !t.spki) return { error: "unknown_tunnel", message: "no attested tunnel by that name" };
    if (!/^[0-9a-f]{32,128}$/.test(String(nonce || ""))) return { error: "bad_nonce", message: "nonce must be 16..64 bytes hex" };
    let key;
    try { key = createPublicKey({ key: Buffer.from(t.spki, "base64"), format: "der", type: "spki" }); }
    catch { return { error: "bad_key", message: "the tunnel's transport key is not a public key" }; }
    let ok = false;
    // a pVM signs with Ed25519 (TweetNaCl in the payload); a metal box's agent
    // with its P-256 transport key (ECDSA over sha256) - the key on record decides
    const alg = key.asymmetricKeyType === "ed25519" ? null : "sha256";
    try { ok = edVerify(alg, Buffer.from(signedMessage(kind, [name, ...fields, nonce])), key, Buffer.from(String(sig || ""), "hex")); } catch {}
    if (!ok) return { error: "bad_signature", message: "the request is not signed by this tunnel's attested key" };
    return { tunnel: t };
  }

  function seedRecord(seed_id) {
    return state.seeds[seed_id] || null;
  }

  return {
    key: () => rawEd25519Public(pub).toString("hex"),
    /* The dealer's view, for minting: the seed and its id for an attested pVM. */
    seedFor: (keyFp, epoch = PADS_EPOCH) => deriveSeed(master, keyFp, epoch),

    /* POST /v1/pads/seed */
    seed({ name, nonce, sig }) {
      const c = callerOf(name, "seed", [], nonce, sig);
      if (c.error) return { status: 403, body: c };
      const t = c.tunnel;
      if (!t.padKey) return { status: 409, body: { error: "no_pad_key", message: "the tunnel attached without a pad key" } };
      const { seed, seed_id } = deriveSeed(master, t.keyFp, PADS_EPOCH);
      const rec = state.seeds[seed_id] || (state.seeds[seed_id] = { name, keyFp: t.keyFp, epoch: PADS_EPOCH, mark: 0, updated: 0, nonces: [] });
      rec.name = name;
      save();
      const boxed = boxToPadKey(t.padKey, seed);
      return { status: 200, body: { seed_id, epoch: PADS_EPOCH, ...boxed } };
    },

    /* POST /v1/pads/reserve: advance the mark FIRST, then sign the window. */
    reserve({ name, seed_id, want, nonce, sig }) {
      want = Number(want);
      if (!Number.isInteger(want) || want < 1 || want > MAX_WINDOW) return { status: 400, body: { error: "bad_window", message: `want must be 1..${MAX_WINDOW}` } };
      if (!/^[0-9a-f]{32}$/.test(String(seed_id || ""))) return { status: 400, body: { error: "bad_seed_id" } };
      const c = callerOf(name, "reserve", [seed_id, want], nonce, sig);
      if (c.error) return { status: 403, body: c };
      const rec = seedRecord(seed_id);
      if (!rec || rec.keyFp !== c.tunnel.keyFp) return { status: 403, body: { error: "not_your_seed", message: "this seed was not issued to this tunnel's key" } };
      if (rec.nonces.includes(nonce)) return { status: 409, body: { error: "replay", message: "nonce already used" } };
      rec.nonces.push(nonce); if (rec.nonces.length > NONCE_MEMORY) rec.nonces.splice(0, rec.nonces.length - NONCE_MEMORY);
      const lo = rec.mark, hi = rec.mark + want, iat = Math.floor(Date.now() / 1000);
      rec.mark = hi; rec.updated = iat;
      save();                                                  // durable BEFORE the window is handed out
      const wsig = edSign(null, Buffer.from(windowMessage(seed_id, lo, hi, iat)), priv).toString("hex");
      return { status: 200, body: { seed_id, lo, hi, iat, sig: wsig } };
    },

    /* POST /v1/pads/receipt: the pVM's word on what a run consumed. Signed by
     * the same transport key as the windows, so neither the owner app nor an
     * operator can inflate it; totals are what billing and the operator payout
     * read. Nonces share the seed's replay memory with reserve. */
    receipt({ name, seed_id, pads, tokens, nonce, sig }) {
      pads = Number(pads); tokens = Number(tokens);
      if (!/^[0-9a-f]{32}$/.test(String(seed_id || ""))) return { status: 400, body: { error: "bad_seed_id" } };
      if (!Number.isSafeInteger(pads) || pads < 0 || !Number.isSafeInteger(tokens) || tokens < 0) return { status: 400, body: { error: "bad_receipt", message: "pads and tokens must be non-negative integers" } };
      const c = callerOf(name, "receipt", [seed_id, pads, tokens], nonce, sig);
      if (c.error) return { status: 403, body: c };
      const rec = seedRecord(seed_id);
      if (!rec || rec.keyFp !== c.tunnel.keyFp) return { status: 403, body: { error: "not_your_seed", message: "this seed was not issued to this tunnel's key" } };
      if (rec.nonces.includes(nonce)) return { status: 409, body: { error: "replay", message: "nonce already used" } };
      rec.nonces.push(nonce); if (rec.nonces.length > NONCE_MEMORY) rec.nonces.splice(0, rec.nonces.length - NONCE_MEMORY);
      const iat = Math.floor(Date.now() / 1000);
      const u = rec.usage || (rec.usage = { pads: 0, tokens: 0, runs: 0, last: [] });
      u.pads += pads; u.tokens += tokens; u.runs += 1;
      u.last.push({ pads, tokens, iat, nonce }); if (u.last.length > RECEIPT_MEMORY) u.last.splice(0, u.last.length - RECEIPT_MEMORY);
      save();
      return { status: 200, body: { seed_id, pads: u.pads, tokens: u.tokens, runs: u.runs, iat } };
    },

    /* GET /v1/pads/receipts?seed_id= (billing and the operator payout read the totals). */
    receipts(seed_id) {
      const rec = seedRecord(seed_id);
      if (!rec) return null;
      const u = rec.usage || { pads: 0, tokens: 0, runs: 0, last: [] };
      return { seed_id, name: rec.name, pads: u.pads, tokens: u.tokens, runs: u.runs, last: u.last };
    },

    /* GET /v1/pads/pvm?name=: what a dealer needs to mint for an attached pVM -
     * public keys only (the seed itself is derived from keyFp by whoever holds
     * the master), plus the seed id and the mark so a bank can be planned. */
    pvm(name) {
      const t = hub && hub.info ? hub.info(name) : null;
      if (!t || !t.keyFp) return null;
      const { seed_id } = deriveSeed(master, t.keyFp, PADS_EPOCH);
      const rec = seedRecord(seed_id);
      return { name, keyFp: t.keyFp, padKey: t.padKey || "", seed_id, epoch: PADS_EPOCH, mark: rec ? rec.mark : 0, issued: !!rec };
    },

    /* GET /v1/pads/consumers: every attached tunnel with a pad key, for the
     * dealer daemon that keeps all of them ahead of their marks. */
    consumers() {
      const names = hub && hub.origins ? hub.origins().map((o) => o.name) : [];
      const out = [];
      for (const name of names) {
        const t = hub.info(name);
        if (!t || !t.keyFp || !t.padKey) continue;
        const { seed_id } = deriveSeed(master, t.keyFp, PADS_EPOCH);
        const rec = seedRecord(seed_id);
        out.push({ name, keyFp: t.keyFp, padKey: t.padKey, seed_id, epoch: PADS_EPOCH, mark: rec ? rec.mark : 0, issued: !!rec });
      }
      return out;
    },

    /* GET /v1/pads/ledger?seed_id= (operators and the dealer read the mark). */
    mark(seed_id) {
      const rec = seedRecord(seed_id);
      return rec ? { seed_id, mark: rec.mark, updated: rec.updated, name: rec.name } : null;
    },
  };
}
