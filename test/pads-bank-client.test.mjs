// The CVM-tier pad bank client and the window agent (shielded/dealer/PLAN.md
// P4a): the trusted half fetches shipments over plain HTTP from a bank that
// speaks the platform store's shape, and takes signed ledger windows from a
// loopback agent. Both are exercised here without a model: a stub bank serves
// a listing and fake shipment bytes, bank-probe (C) drives shielded-bank.c,
// and window-agent.mjs answers windows whose signatures verify against the
// key it publishes.
//   run: node --test test/pads-bank-client.test.mjs   (builds bank-probe via make)
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";
import { createPublicKey, verify } from "node:crypto";
import { execFile, spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { windowMessage } from "../relay/pads.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const gg = path.join(root, "wasm", "ggml-shielded");
const seed = "0123456789abcdef0123456789abcdef";

function stubBank(shipments) {
  // shipments: [{index0, count, bytes}] -> served as <seed>-<i0>-<n>.pads of that many bytes
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://x");
    if (url.pathname === "/v1/pads/shipments") {
      const list = shipments.map((s) => ({ name: `${seed}-${s.index0}-${s.count}.pads`, bytes: s.bytes, index0: s.index0, count: s.count }));
      const body = JSON.stringify({ seed_id: seed, shipments: list });
      res.writeHead(200, { "content-type": "application/json" });   // no content-length: chunked, like Node does by default
      res.end(body); return;
    }
    const m = url.pathname.match(/^\/v1\/pads\/shipments\/([0-9a-f]{32})\/(.+\.pads)$/);
    const s = m && shipments.find((x) => `${seed}-${x.index0}-${x.count}.pads` === m[2]);
    if (!s) { res.writeHead(404); res.end("no"); return; }
    server.hits.push(m[2]);
    res.writeHead(200, { "content-type": "application/octet-stream", "content-length": s.bytes });
    res.end(Buffer.alloc(s.bytes, s.index0 & 0xff));
  });
  server.hits = [];
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

test("bank client: fetches in index order up to the horizon, honours the cache budget and the floor", async (t) => {
  const mk = spawnSync("make", ["-s", "-C", gg, "bank-probe"], { encoding: "utf8", timeout: 600_000 });
  assert.equal(mk.status, 0, mk.stderr);
  const MB = 1 << 20;
  const server = await stubBank([{ index0: 0, count: 16, bytes: 2 * MB }, { index0: 16, count: 16, bytes: 2 * MB }, { index0: 32, count: 16, bytes: 2 * MB }, { index0: 48, count: 16, bytes: 2 * MB }]);
  t.after(() => server.close());
  const url = `http://127.0.0.1:${server.address().port}/v1/pads/shipments`;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "bank-"));
  const probe = (args) => new Promise((resolve) => execFile(path.join(gg, "bank-probe"), args, { encoding: "utf8", timeout: 60_000 }, (err, stdout, stderr) => resolve({ code: err ? err.code : 0, stdout, stderr })));

  // need 32 with a 5 MB budget: the two shipments under the horizon come regardless, a third only if it fits (it does not: 4+2 > 5)
  let r = await probe([url, seed, dir, "32", "5", "20000"]);
  assert.equal(r.code, 0, r.stdout + r.stderr);
  assert.deepEqual(fs.readdirSync(dir).filter((f) => f.endsWith(".pads")).sort(), [`${seed}-0-16.pads`, `${seed}-16-16.pads`]);
  assert.equal(fs.statSync(path.join(dir, `${seed}-0-16.pads`)).size, 2 * MB);
  assert.deepEqual(server.hits, [`${seed}-0-16.pads`, `${seed}-16-16.pads`]);

  // floor 20: the first shipment is spent (never refetched even when deleted); need 64 with a big budget fetches the rest in order
  fs.unlinkSync(path.join(dir, `${seed}-0-16.pads`));
  server.hits.length = 0;
  r = await probe([url, seed, dir, "64", "64", "20000", "20"]);
  assert.equal(r.code, 0, r.stdout + r.stderr);
  assert.deepEqual(server.hits, [`${seed}-32-16.pads`, `${seed}-48-16.pads`]);
  assert.deepEqual(fs.readdirSync(dir).filter((f) => f.endsWith(".pads")).sort(), [`${seed}-16-16.pads`, `${seed}-32-16.pads`, `${seed}-48-16.pads`]);
  assert.match(r.stdout, /listing http 200/);

  // an unreachable bank: the probe times out with nothing, and says so
  r = await probe(["http://127.0.0.1:1/v1/pads/shipments", seed, fs.mkdtempSync(path.join(os.tmpdir(), "bank-")), "16", "64", "1500"]);
  assert.equal(r.code, 1);
  assert.match(r.stdout, /unreachable/);
});

test("window agent (local mode): reserve-before-use windows signed by the key it publishes; bad requests refused", async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agent-"));
  const port = 20000 + Math.floor(Math.random() * 20000);
  const agent = spawn(process.execPath, [path.join(root, "shielded", "dealer", "window-agent.mjs"), "--port", String(port), "--ledger", path.join(dir, "ledger"), "--key", path.join(dir, "agent.pem")], { stdio: ["ignore", "pipe", "pipe"] });
  t.after(() => agent.kill());
  await new Promise((resolve, reject) => { agent.stdout.on("data", (d) => { if (String(d).includes("ledger key")) resolve(); }); agent.on("exit", (c) => reject(new Error("agent exited " + c))); });
  const base = `http://127.0.0.1:${port}`;
  const { key } = await fetch(base + "/key").then((r) => r.json());
  assert.match(key, /^[0-9a-f]{64}$/);
  const pub = createPublicKey({ key: Buffer.concat([Buffer.from("302a300506032b6570032100", "hex"), Buffer.from(key, "hex")]), format: "der", type: "spki" });
  const post = (body) => fetch(base + "/window", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then(async (r) => ({ status: r.status, body: await r.json() }));

  const w1 = await post({ want: 8, seed_id: seed });
  assert.equal(w1.status, 200);
  assert.deepEqual([w1.body.lo, w1.body.hi], [0, 8]);
  assert.ok(verify(null, Buffer.from(windowMessage(seed, w1.body.lo, w1.body.hi, w1.body.iat)), pub, Buffer.from(w1.body.sig, "hex")));
  assert.equal(fs.readFileSync(path.join(dir, "ledger"), "utf8").trim(), "8");        // the mark moved before the answer
  const w2 = await post({ want: 64, seed_id: seed });
  assert.deepEqual([w2.body.lo, w2.body.hi], [8, 72]);
  assert.ok(verify(null, Buffer.from(windowMessage(seed, 8, 72, w2.body.iat)), pub, Buffer.from(w2.body.sig, "hex")));
  assert.equal((await post({ want: 0, seed_id: seed })).status, 400);
  assert.equal((await post({ want: 5000, seed_id: seed })).status, 400);
  assert.equal((await post({ want: 8, seed_id: "nope" })).status, 400);
  // the key persists: a restarted agent signs with the same key
  agent.kill();
  const again = spawn(process.execPath, [path.join(root, "shielded", "dealer", "window-agent.mjs"), "--port", String(port), "--ledger", path.join(dir, "ledger"), "--key", path.join(dir, "agent.pem")], { stdio: ["ignore", "pipe", "pipe"] });
  t.after(() => again.kill());
  await new Promise((resolve) => again.stdout.on("data", (d) => { if (String(d).includes("ledger key")) resolve(); }));
  assert.equal((await fetch(base + "/key").then((r) => r.json())).key, key);
  const w3 = await post({ want: 8, seed_id: seed });
  assert.deepEqual([w3.body.lo, w3.body.hi], [72, 80]);
});
