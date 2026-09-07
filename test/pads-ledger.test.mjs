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

test("receipt: signed usage accrues to the seed; replay, foreign seeds and bad counts refused", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pads-"));
  const a = pvm(), b = pvm();
  const hub = hubWith({ phone1: { keyFp: a.keyFp, spki: a.spki, padKey: a.padKey }, phone2: { keyFp: b.keyFp, spki: b.spki, padKey: b.padKey } });
  const L = createPadsLedger({ dir, hub, log: () => {} });
  const n0 = randomBytes(16).toString("hex");
  const { seed_id } = L.seed({ name: "phone1", nonce: n0, sig: a.signReq("seed", "phone1", [], n0) }).body;
  const req = (who, name, pads, tokens, nonce) => ({ name, seed_id, pads, tokens, nonce, sig: who.signReq("receipt", name, [seed_id, pads, tokens], nonce) });
  const n1 = randomBytes(16).toString("hex");
  const r1 = L.receipt(req(a, "phone1", 2033, 61, n1));
  assert.equal(r1.status, 200, JSON.stringify(r1));
  assert.deepEqual([r1.body.pads, r1.body.tokens, r1.body.runs], [2033, 61, 1]);
  assert.equal(L.receipt(req(a, "phone1", 2033, 61, n1)).status, 409);                 // replay
  const r2 = L.receipt(req(a, "phone1", 100, 7, randomBytes(16).toString("hex")));
  assert.deepEqual([r2.body.pads, r2.body.tokens, r2.body.runs], [2133, 68, 2]);
  assert.equal(L.receipt(req(b, "phone2", 5, 5, randomBytes(16).toString("hex"))).status, 403); // not its seed
  assert.equal(L.receipt({ ...req(a, "phone1", 5, 5, randomBytes(16).toString("hex")), pads: -1 }).status, 400);
  assert.equal(L.receipt({ ...req(a, "phone1", 5, 5, randomBytes(16).toString("hex")), sig: "00" }).status, 403);
  const tot = L.receipts(seed_id);
  assert.deepEqual([tot.pads, tot.tokens, tot.runs, tot.last.length], [2133, 68, 2, 2]);
  assert.equal(L.receipts("0".repeat(32)), null);
});

test("a metal box's P-256 transport key signs requests too (ECDSA over sha256); the key on record decides", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pads-"));
  const ec = generateKeyPairSync("ec", { namedCurve: "prime256v1" });
  const x = generateKeyPairSync("x25519");
  const spkiDer = Buffer.from(ec.publicKey.export({ type: "spki", format: "der" }));
  const box = { keyFp: createHash("sha256").update(spkiDer).digest("hex"), spki: spkiDer.toString("base64"),
                padKey: Buffer.from(x.publicKey.export({ type: "spki", format: "der" })).subarray(-32).toString("hex") };
  const signReq = (kind, name, fields, nonce) => sign("sha256", Buffer.from(signedMessage(kind, [name, ...fields, nonce])), ec.privateKey).toString("hex");
  const L = createPadsLedger({ dir, hub: hubWith({ metal0: box }), log: () => {} });
  const n0 = randomBytes(16).toString("hex");
  const s = L.seed({ name: "metal0", nonce: n0, sig: signReq("seed", "metal0", [], n0) });
  assert.equal(s.status, 200, JSON.stringify(s));
  const n1 = randomBytes(16).toString("hex");
  const w = L.reserve({ name: "metal0", seed_id: s.body.seed_id, want: 16, nonce: n1, sig: signReq("reserve", "metal0", [s.body.seed_id, 16], n1) });
  assert.equal(w.status, 200, JSON.stringify(w));
  assert.deepEqual([w.body.lo, w.body.hi], [0, 16]);
  // an Ed25519 signature under a P-256 record, or a P-256 one under an Ed25519 record, is just a bad signature
  const ed = pvm();
  assert.equal(L.reserve({ name: "metal0", seed_id: s.body.seed_id, want: 16, nonce: randomBytes(16).toString("hex"), sig: ed.signReq("reserve", "metal0", [s.body.seed_id, 16], "00") }).status, 403);
});

test("consumers: every attached tunnel with a pad key, with its seed id and mark, for the dealer daemon", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pads-"));
  const a = pvm(), b = pvm();
  const recs = { phone1: { keyFp: a.keyFp, spki: a.spki, padKey: a.padKey }, phone2: { keyFp: b.keyFp, spki: b.spki, padKey: b.padKey }, plain: { keyFp: "ab".repeat(32), spki: a.spki, padKey: "" } };
  const hub = { info: (name) => recs[name] ? { name, mode: "avf", ...recs[name] } : null, origins: () => Object.keys(recs).map((name) => ({ name })) };
  const L = createPadsLedger({ dir, hub, log: () => {}, masterSeed: "22".repeat(32) });
  let c = L.consumers();
  assert.deepEqual(c.map((x) => x.name), ["phone1", "phone2"]);              // no pad key = not a consumer
  assert.ok(c.every((x) => !x.issued && x.mark === 0 && /^[0-9a-f]{32}$/.test(x.seed_id)));
  const n0 = randomBytes(16).toString("hex");
  const { seed_id } = L.seed({ name: "phone1", nonce: n0, sig: a.signReq("seed", "phone1", [], n0) }).body;
  const n1 = randomBytes(16).toString("hex");
  L.reserve({ name: "phone1", seed_id, want: 24, nonce: n1, sig: a.signReq("reserve", "phone1", [seed_id, 24], n1) });
  c = L.consumers();
  assert.deepEqual(c.find((x) => x.name === "phone1"), { name: "phone1", keyFp: a.keyFp, padKey: a.padKey, seed_id, epoch: 1, mark: 24, issued: true });
  assert.equal(c.find((x) => x.name === "phone2").issued, false);
});
