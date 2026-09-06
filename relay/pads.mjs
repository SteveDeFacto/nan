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

export const PADS_EPOCH = 1;                     // bump to re-key every pVM's seed
export const MAX_WINDOW = 4096;
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
    try { ok = edVerify(null, Buffer.from(signedMessage(kind, [name, ...fields, nonce])), key, Buffer.from(String(sig || ""), "hex")); } catch {}
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

    /* GET /v1/pads/ledger?seed_id= (operators and the dealer read the mark). */
    mark(seed_id) {
      const rec = seedRecord(seed_id);
      return rec ? { seed_id, mark: rec.mark, updated: rec.updated, name: rec.name } : null;
    },
  };
}
