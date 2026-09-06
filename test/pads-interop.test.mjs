// Node <-> C interop for dealt pads: the relay boxes a seed (relay/pads.mjs)
// and signs windows; the trusted half opens the box and verifies the window
// with the C in wasm/ggml-shielded/shielded-pads.c; the C signs a request the
// relay accepts. Built through the C backend's Makefile (pads-unbox).
//
//   run: node --test test/pads-interop.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash, generateKeyPairSync, randomBytes } from "node:crypto";
import { fileURLToPath } from "node:url";
import { boxToPadKey, createPadsLedger } from "../relay/pads.mjs";

const repo = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const dir = path.join(repo, "wasm", "ggml-shielded");
const tool = path.join(dir, "pads-unbox");
function built() {
  const r = spawnSync("make", ["-s", "pads-unbox"], { cwd: dir, encoding: "utf8", timeout: 300_000 });
  return r.status === 0 && fs.existsSync(tool);
}
const run = (...args) => execFileSync(tool, args, { encoding: "utf8", timeout: 60_000 }).trim();

test("a seed boxed by the relay opens in the trusted half's C, and only untampered", (t) => {
  if (!built()) return t.skip("no toolchain for the C backend");
  const x = generateKeyPairSync("x25519");
  const sk = Buffer.from(x.privateKey.export({ type: "pkcs8", format: "der" })).subarray(-32).toString("hex");
  const pk = Buffer.from(x.publicKey.export({ type: "spki", format: "der" })).subarray(-32).toString("hex");
  const seed = randomBytes(32);
  const b = boxToPadKey(pk, seed);
  assert.equal(run("seed", sk, pk, b.epk, b.nonce, b.box), "seed " + seed.toString("hex"));
  const bad = Buffer.from(b.box, "hex"); bad[5] ^= 1;
  assert.equal(spawnSync(tool, ["seed", sk, pk, b.epk, b.nonce, bad.toString("hex")], { encoding: "utf8" }).stdout.trim(), "fail");
});

test("a request signed by the C transport key is accepted; the signed window verifies in C", (t) => {
  if (!built()) return t.skip("no toolchain for the C backend");
  const kp = Object.fromEntries(run("keypair").split("\n").map((l) => l.split(" ")));
  const spkiDer = Buffer.concat([Buffer.from("302a300506032b6570032100", "hex"), Buffer.from(kp.pk, "hex")]);
  const keyFp = createHash("sha256").update(spkiDer).digest("hex");
  const x = generateKeyPairSync("x25519");
  const padKey = Buffer.from(x.publicKey.export({ type: "spki", format: "der" })).subarray(-32).toString("hex");
  const hub = { info: (name) => name === "pixel" ? { name, mode: "avf", keyFp, spki: spkiDer.toString("base64"), padKey } : null };
  const L = createPadsLedger({ dir: fs.mkdtempSync(path.join(os.tmpdir(), "pads-")), hub, log: () => {} });
  const n0 = randomBytes(16).toString("hex");
  const sig0 = run("sign", kp.sk, "seed", n0, "pixel").split(" ")[1];
  const s = L.seed({ name: "pixel", nonce: n0, sig: sig0 });
  assert.equal(s.status, 200, JSON.stringify(s.body));
  const n1 = randomBytes(16).toString("hex");
  const sig1 = run("sign", kp.sk, "reserve", n1, "pixel", s.body.seed_id, "64").split(" ")[1];
  const w = L.reserve({ name: "pixel", seed_id: s.body.seed_id, want: 64, nonce: n1, sig: sig1 });
  assert.equal(w.status, 200, JSON.stringify(w.body));
  assert.equal(run("window", L.key(), s.body.seed_id, w.body.lo, w.body.hi, w.body.iat, w.body.sig), "ok");
  assert.equal(spawnSync(tool, ["window", L.key(), s.body.seed_id, String(w.body.lo), String(w.body.hi + 1), String(w.body.iat), w.body.sig], { encoding: "utf8" }).stdout.trim(), "fail");
});
