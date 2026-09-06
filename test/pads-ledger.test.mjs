// The platform half of dealt pads (relay/pads.mjs): seeds derive the same way
// for the relay and the dealer, a pVM's seed reaches only its X25519 pad key,
// ledger windows are handed out reserve-before-use and signed, requests must
// carry the attested tunnel key's signature, and nothing replays.
//
//   run: node --test test/pads-ledger.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createDecipheriv, createHash, createPrivateKey, createPublicKey, diffieHellman, generateKeyPairSync, hkdfSync,
         randomBytes, sign, verify } from "node:crypto";
import { createPadsLedger, deriveSeed, signedMessage, windowMessage, PADS_EPOCH, MAX_WINDOW } from "../relay/pads.mjs";

function pvm() {
  const ed = generateKeyPairSync("ed25519");
  const x = generateKeyPairSync("x25519");
  const spkiDer = Buffer.from(ed.publicKey.export({ type: "spki", format: "der" }));
  return {
    ed, x,
    spki: spkiDer.toString("base64"),
    keyFp: createHash("sha256").update(spkiDer).digest("hex"),
    padKey: Buffer.from(x.publicKey.export({ type: "spki", format: "der" })).subarray(-32).toString("hex"),
    signReq(kind, name, fields, nonce) {
      return sign(null, Buffer.from(signedMessage(kind, [name, ...fields, nonce])), ed.privateKey).toString("hex");
    },
    unbox({ epk, nonce, box }) {
      const peer = createPublicKey({ key: Buffer.concat([Buffer.from("302a300506032b656e032100", "hex"), Buffer.from(epk, "hex")]), format: "der", type: "spki" });
      const shared = diffieHellman({ privateKey: x.privateKey, publicKey: peer });
      const key = Buffer.from(hkdfSync("sha512", shared, Buffer.concat([Buffer.from(epk, "hex"), Buffer.from(this.padKey, "hex")]), "enclave-pads-seed-box", 32));
      const ct = Buffer.from(box, "hex");
      const d = createDecipheriv("chacha20-poly1305", key, Buffer.from(nonce, "hex"), { authTagLength: 16 });
      d.setAuthTag(ct.subarray(-16));
      return Buffer.concat([d.update(ct.subarray(0, -16)), d.final()]);
    },
  };
}

function hubWith(tunnels) {
  return { info: (name) => tunnels[name] ? { name, mode: "avf", ...tunnels[name] } : null };
}

test("seed: derived per (keyFp, epoch), boxed only to the pad key, same for the dealer", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pads-"));
  const a = pvm(), stranger = pvm();
  const L = createPadsLedger({ dir, hub: hubWith({ phone1: { keyFp: a.keyFp, spki: a.spki, padKey: a.padKey } }), log: () => {} });
  const nonce = randomBytes(16).toString("hex");
  const r = L.seed({ name: "phone1", nonce, sig: a.signReq("seed", "phone1", [], nonce) });
  assert.equal(r.status, 200);
  const seed = a.unbox(r.body);
  assert.equal(seed.length, 32);
  assert.equal(r.body.epoch, PADS_EPOCH);
  const dealer = L.seedFor(a.keyFp);
  assert.deepEqual(dealer.seed, seed, "the dealer derives the seed the pVM received");
  assert.equal(dealer.seed_id, r.body.seed_id);
  assert.throws(() => stranger.unbox(r.body), "another pad key cannot open the box");
  // a request signed by another key is refused
  const n2 = randomBytes(16).toString("hex");
  assert.equal(L.seed({ name: "phone1", nonce: n2, sig: stranger.signReq("seed", "phone1", [], n2) }).status, 403);
  assert.equal(L.seed({ name: "nobody", nonce: n2, sig: a.signReq("seed", "nobody", [], n2) }).status, 403);
  // the master survives a restart, so the seed does too
  const L2 = createPadsLedger({ dir, hub: hubWith({}), log: () => {} });
  assert.deepEqual(L2.seedFor(a.keyFp).seed, seed);
  assert.equal(L2.key(), L.key());
});

test("reserve: windows advance the durable mark before they are signed; replay and foreign seeds refused", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pads-"));
  const a = pvm(), b = pvm();
  const hub = hubWith({ phone1: { keyFp: a.keyFp, spki: a.spki, padKey: a.padKey }, phone2: { keyFp: b.keyFp, spki: b.spki, padKey: b.padKey } });
  let L = createPadsLedger({ dir, hub, log: () => {} });
  const n0 = randomBytes(16).toString("hex");
  const { seed_id } = L.seed({ name: "phone1", nonce: n0, sig: a.signReq("seed", "phone1", [], n0) }).body;
  const pubKey = createPublicKey({ key: Buffer.concat([Buffer.from("302a300506032b6570032100", "hex"), Buffer.from(L.key(), "hex")]), format: "der", type: "spki" });
  const req = (who, name, want, nonce) => ({ name, seed_id, want, nonce, sig: who.signReq("reserve", name, [seed_id, want], nonce) });
  const n1 = randomBytes(16).toString("hex");
  const w1 = L.reserve(req(a, "phone1", 64, n1));
  assert.equal(w1.status, 200);
  assert.equal(w1.body.lo, 0); assert.equal(w1.body.hi, 64);
  assert.ok(verify(null, Buffer.from(windowMessage(seed_id, 0, 64, w1.body.iat)), pubKey, Buffer.from(w1.body.sig, "hex")), "window signed by the ledger key");
  assert.equal(L.mark(seed_id).mark, 64, "the mark moved before the window left");
  // replayed nonce
  assert.equal(L.reserve(req(a, "phone1", 64, n1)).status, 409);
  // a restart resumes from the durable mark
  L = createPadsLedger({ dir, hub, log: () => {} });
  const n2 = randomBytes(16).toString("hex");
  const w2 = L.reserve(req(a, "phone1", 8, n2));
  assert.equal(w2.body.lo, 64); assert.equal(w2.body.hi, 72);
  // another attested pVM cannot draw on this seed
  const n3 = randomBytes(16).toString("hex");
  assert.equal(L.reserve(req(b, "phone2", 8, n3)).status, 403);
  // bounds and a wrong signature
  assert.equal(L.reserve(req(a, "phone1", MAX_WINDOW + 1, randomBytes(16).toString("hex"))).status, 400);
  const n4 = randomBytes(16).toString("hex");
  assert.equal(L.reserve({ ...req(a, "phone1", 8, n4), sig: "00" }).status, 403);
  assert.equal(L.mark(seed_id).mark, 72);
  // the dealer's view: public identity + mark, never the seed
  const v = L.pvm("phone1");
  assert.equal(v.seed_id, seed_id); assert.equal(v.padKey, a.padKey); assert.equal(v.mark, 72); assert.ok(!("seed" in v));
  assert.equal(L.pvm("nobody"), null);
});
