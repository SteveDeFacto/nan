// metal/guest/agent.mjs as a dealt-pads consumer (shielded/dealer/PLAN.md P4a):
// the pad key rides in the RAD document, the seed is fetched and opened after a
// successful attach and left for the wasm manager in METAL_PADS_DIR, and the
// loopback /pads/window relays a reserve signed with the box's P-256 transport
// key that the platform ledger accepts. A fake relay here runs the real ledger
// (relay/pads.mjs) behind the tunnel's challenge/attest exchange.
//   run: node --test test/metal-agent-pads.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { createHash, createPublicKey, verify } from "node:crypto";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { WebSocketServer } from "ws";
import { createPadsLedger, windowMessage } from "../relay/pads.mjs";

const AGENT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "metal", "guest", "agent.mjs");
const freePort = () => new Promise((r) => { const s = net.createServer().listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => r(p)); }); });
const readJson = (req) => new Promise((resolve) => { let raw = ""; req.on("data", (c) => (raw += c)); req.on("end", () => { try { resolve(JSON.parse(raw || "{}")); } catch { resolve({}); } }); });
let bearer = "";
const postJson = (url, body, auth = true) => fetch(url, { method: "POST", headers: { "content-type": "application/json", ...(auth && bearer ? { authorization: "Bearer " + bearer } : {}) }, body: JSON.stringify(body) }).then(async (r) => ({ status: r.status, body: await r.json() }));

test("agent: pad key in the RAD, seed bootstrapped after attach, /pads/window relays a reserve the ledger signs", async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agent-pads-"));
  const tunnels = new Map();
  const ledger = createPadsLedger({ dir: path.join(dir, "ledger"), hub: { info: (name) => tunnels.get(name) || null }, log: () => {}, masterSeed: "11".repeat(32) });

  // the fake relay: the tunnel's websocket (challenge -> attest -> attest-result) and the pads HTTP routes
  const relaySrv = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    const json = (status, body) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(body)); };
    if (req.method === "GET" && url.pathname === "/v1/pads/key") return json(200, { key: ledger.key(), epoch: 1 });
    if (req.method === "POST" && url.pathname === "/v1/pads/seed") { const r = ledger.seed(await readJson(req)); return json(r.status, r.body); }
    if (req.method === "POST" && url.pathname === "/v1/pads/reserve") { const r = ledger.reserve(await readJson(req)); return json(r.status, r.body); }
    if (req.method === "POST" && url.pathname === "/v1/pads/receipt") { const r = ledger.receipt(await readJson(req)); return json(r.status, r.body); }
    if (req.method === "GET" && url.pathname === "/v1/pads/shipments") return json(200, { seed_id: url.searchParams.get("seed_id"), shipments: [{ name: `${url.searchParams.get("seed_id")}-0-16.pads`, bytes: 3 * 1024 * 1024 + 7, index0: 0, count: 16 }] });
    if (req.method === "GET" && /^\/v1\/pads\/shipments\/[0-9a-f]{32}\/[0-9a-f]{32}-0-16\.pads$/.test(url.pathname)) { const b = Buffer.alloc(3 * 1024 * 1024 + 7, 0x5a); res.writeHead(200, { "content-type": "application/octet-stream", "content-length": b.length }); return res.end(b); }
    json(404, { error: "not_found" });
  });
  const relayPort = await new Promise((r) => relaySrv.listen(0, "127.0.0.1", () => r(relaySrv.address().port)));
  const wss = new WebSocketServer({ server: relaySrv });
  t.after(() => { try { wss.close(); } catch {} try { relaySrv.close(); } catch {} });
  let rad = null;
  const attached = new Promise((resolve) => wss.on("connection", (sock) => {
    sock.on("message", (d) => {
      let f; try { f = JSON.parse(d); } catch { return; }
      if (f.t === "hello") sock.send(JSON.stringify({ t: "challenge", nonce: Buffer.alloc(32, 7).toString("base64") }));
      if (f.t === "attest") {
        rad = f.rad;
        const spki = Buffer.from(rad.transportKey, "base64");
        tunnels.set("padbox", { name: "padbox", mode: "snp", keyFp: createHash("sha256").update(spki).digest("hex"), spki: rad.transportKey, padKey: rad.padKey });
        sock.send(JSON.stringify({ t: "attest-result", ok: true, measurement: "test" }));
        resolve();
      }
    });
  }));

  const radPort = await freePort();
  const padsDir = path.join(dir, "pads");
  const proc = spawn(process.execPath, [AGENT], {
    env: { ...process.env, METAL_MODE: "dev", METAL_NAME: "padbox", METAL_SUP_URL: "http://127.0.0.1:1",
           METAL_RELAY_URL: `ws://127.0.0.1:${relayPort}/v1/fleet-tunnel`, METAL_TUNNEL_TOKEN: "t",
           METAL_RAD_PORT: String(radPort), METAL_PADS_DIR: padsDir },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let log = "";
  proc.stdout.on("data", (d) => (log += d)); proc.stderr.on("data", (d) => (log += d));
  t.after(() => { try { proc.kill("SIGKILL"); } catch {} });
  await Promise.race([attached, new Promise((_, rej) => setTimeout(() => rej(new Error("agent never attested:\n" + log.slice(-1500))), 10000))]);

  // the RAD document carries the X25519 pad key the relay recorded
  assert.match(rad.padKey, /^[0-9a-f]{64}$/);
  assert.equal(tunnels.get("padbox").padKey, rad.padKey);

  // the bootstrap file appears, root-only, with the seed the ledger derived for this key
  const file = path.join(padsDir, "bootstrap.json");
  for (let i = 0; i < 100 && !fs.existsSync(file); i++) await new Promise((r) => setTimeout(r, 100));
  assert.ok(fs.existsSync(file), "no bootstrap file:\n" + log.slice(-1500));
  assert.equal(fs.statSync(file).mode & 0o777, 0o600);
  const boot = JSON.parse(fs.readFileSync(file, "utf8"));
  const expect = ledger.seedFor(tunnels.get("padbox").keyFp);
  assert.equal(boot.seed_id, expect.seed_id);
  assert.equal(boot.seed, expect.seed.toString("hex"));
  assert.equal(boot.ledger_pk, ledger.key());
  assert.equal(boot.window_url, `http://127.0.0.1:${radPort}/pads/window`);
  assert.match(boot.sk, /^[0-9a-f]{64}$/);
  assert.match(boot.token, /^[0-9a-f]{64}$/);
  bearer = boot.token;
  // without the per-boot token, windows and receipts are nobody's
  assert.equal((await postJson(boot.window_url, { want: 8, seed_id: boot.seed_id }, false)).status, 401);
  assert.equal((await postJson(`http://127.0.0.1:${radPort}/pads/receipt`, { seed_id: boot.seed_id, pads: 1, tokens: 1 }, false)).status, 401);
  assert.equal(ledger.mark(boot.seed_id).mark, 0);
  // the pad secret in the file is the private half of the key in the RAD
  const pk = createPublicKey({ key: Buffer.concat([Buffer.from("302e020100300506032b656e04220420", "hex"), Buffer.from(boot.sk, "hex")]), format: "der", type: "pkcs8" });
  assert.equal(Buffer.from(pk.export({ type: "spki", format: "der" })).subarray(-32).toString("hex"), rad.padKey);

  // a window through the agent: the ledger's mark moves first, the answer carries the ledger's signature
  const w = await postJson(boot.window_url, { want: 8, seed_id: boot.seed_id });
  assert.equal(w.status, 200, JSON.stringify(w));
  assert.deepEqual([w.body.lo, w.body.hi], [0, 8]);
  const ledgerPub = createPublicKey({ key: Buffer.concat([Buffer.from("302a300506032b6570032100", "hex"), Buffer.from(ledger.key(), "hex")]), format: "der", type: "spki" });
  assert.ok(verify(null, Buffer.from(windowMessage(boot.seed_id, 0, 8, w.body.iat)), ledgerPub, Buffer.from(w.body.sig, "hex")));
  assert.equal(ledger.mark(boot.seed_id).mark, 8);
  const w2 = await postJson(boot.window_url, { want: 64, seed_id: boot.seed_id });
  assert.deepEqual([w2.body.lo, w2.body.hi], [8, 72]);
  // malformed asks never reach the relay
  assert.equal((await postJson(boot.window_url, { want: 0, seed_id: boot.seed_id })).status, 400);
  assert.equal((await postJson(boot.window_url, { want: 8, seed_id: "nope" })).status, 400);
  // another box's seed is refused by the ledger (signed by this key, wrong keyFp)
  const foreign = await postJson(boot.window_url, { want: 8, seed_id: "0".repeat(32) });
  assert.equal(foreign.status, 403);

  // usage receipts through the agent: signed as this box, accrued by the ledger per seed
  const rc1 = await postJson(`http://127.0.0.1:${radPort}/pads/receipt`, { seed_id: boot.seed_id, pads: 970, tokens: 10 });
  assert.equal(rc1.status, 200, JSON.stringify(rc1));
  const rc2 = await postJson(`http://127.0.0.1:${radPort}/pads/receipt`, { seed_id: boot.seed_id, pads: 30, tokens: 1 });
  assert.deepEqual([rc2.body.pads, rc2.body.tokens, rc2.body.runs], [1000, 11, 2]);
  assert.deepEqual([ledger.receipts(boot.seed_id).pads, ledger.receipts(boot.seed_id).tokens], [1000, 11]);
  assert.equal((await postJson(`http://127.0.0.1:${radPort}/pads/receipt`, { seed_id: boot.seed_id, pads: -1, tokens: 1 })).status, 400);

  // the bank proxy: the platform store's listing and bytes, verbatim, through the loopback (no TLS in the engine)
  assert.equal(boot.bank_url, `http://127.0.0.1:${radPort}/pads/shipments`);
  const list = await fetch(`${boot.bank_url}?seed_id=${boot.seed_id}`).then((r) => r.json());
  assert.equal(list.shipments[0].name, `${boot.seed_id}-0-16.pads`);
  const ship = await fetch(`${boot.bank_url}/${boot.seed_id}/${boot.seed_id}-0-16.pads`);
  assert.equal(ship.status, 200);
  const bytes = Buffer.from(await ship.arrayBuffer());
  assert.equal(bytes.length, 3 * 1024 * 1024 + 7);
  assert.ok(bytes.every((b) => b === 0x5a));
  assert.equal((await fetch(`${boot.bank_url}/${boot.seed_id}/evil.pads`)).status, 400);      // only the store's two shapes pass
  assert.equal((await fetch(`${boot.bank_url}/../../v1/pads/key`)).status, 404);              // (the client normalises the traversal away; the agent has no such route)
  assert.equal((await fetch(`${boot.bank_url}?seed_id=nope`)).status, 400);
});
