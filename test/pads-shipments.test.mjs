// The platform's shipment store (relay/pads.mjs createShipmentStore) and the
// dealer loop's --push against a stub that answers like the relay's routes.
//
//   run: node --test test/pads-shipments.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";
import { createHash, randomBytes } from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
const execFileP = promisify(execFile);
import { fileURLToPath } from "node:url";
import { createShipmentStore, deriveSeed } from "../relay/pads.mjs";

const repo = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const loop = path.join(repo, "shielded", "dealer", "dealer-loop.py");

test("the store only accepts a seed's own well-named shipments and lists them in index order", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ship-"));
  const store = createShipmentStore({ dir });
  const seed = randomBytes(16).toString("hex"), other = randomBytes(16).toString("hex");
  assert.equal(store.plan(seed, `${other}-0-64.pads`), null, "another seed's name");
  assert.equal(store.plan(seed, "../x.pads"), null);
  const p = store.plan(seed, `${seed}-64-64.pads`); fs.writeFileSync(p.final, "b");
  const q = store.plan(seed, `${seed}-0-64.pads`); fs.writeFileSync(q.final, "a");
  assert.deepEqual(store.list(seed).map((s) => [s.name, s.bytes, s.index0, s.count]), [[`${seed}-0-64.pads`, 1, 0, 64], [`${seed}-64-64.pads`, 1, 64, 64]]);
  assert.equal(store.file(seed, `${seed}-0-64.pads`), q.final);
  assert.ok(store.remove(seed, `${seed}-0-64.pads`)); assert.equal(store.list(seed).length, 1);
});

test("dealer --push streams new shipments with their sha256 and deletes spent ones", async () => {
  const bank = fs.mkdtempSync(path.join(os.tmpdir(), "bank-"));
  const seed = randomBytes(16).toString("hex");
  // a "spent" shipment below the mark, and a fake dealer that writes what --ranges asks for
  fs.writeFileSync(path.join(bank, `${seed}-0-64.pads`), "spent");
  const fakeDealer = path.join(repo, "test", "fixtures", "fake-dealer.sh");   // writes 1000 random bytes per range
  const seen = { put: [], del: [] };
  const srv = http.createServer((req, res) => {
    const u = new URL(req.url, "http://x");
    if (req.method === "PUT") {
      const h = createHash("sha256"); let n = 0;
      req.on("data", (c) => { h.update(c); n += c.length; });
      req.on("end", () => { const got = h.digest("hex"); seen.put.push({ path: u.pathname, ok: got === u.searchParams.get("sha256"), n, auth: req.headers.authorization });
        res.setHeader("content-type", "application/json"); res.end(JSON.stringify({ stored: true, bytes: n, sha256: got })); });
      return;
    }
    if (req.method === "DELETE") { seen.del.push(u.pathname); res.statusCode = 200; return res.end("{}"); }
    res.statusCode = 404; res.end("{}");
  });
  await new Promise((r) => srv.listen(0, "127.0.0.1", r));
  const base = `http://127.0.0.1:${srv.address().port}`;
  // async: the stub above lives in THIS process, so the loop must keep turning while python runs
  const { stdout: out } = await execFileP("python3", [loop, "--once", "--push", "--relay", base, "--seed", "00".repeat(32), "--seed-id", seed, "--pk", "11".repeat(32),
    "--mark", "70", "--ahead", "128", "--chunk", "64", "--model", "m.gguf", "--calib", "c.calib", "--out", bank],
    { encoding: "utf8", env: { ...process.env, DEALER: fakeDealer, PADS_DEALER_TOKEN: "tok" }, timeout: 60_000 });
  srv.closeAllConnections(); srv.close(); srv.unref();
  assert.match(out, /pruned .*-0-64\.pads/);
  assert.deepEqual(seen.del, [`/v1/pads/shipments/${seed}/${seed}-0-64.pads`]);
  assert.deepEqual(seen.put.map((p) => p.path).sort(), [`/v1/pads/shipments/${seed}/${seed}-128-64.pads`, `/v1/pads/shipments/${seed}/${seed}-192-64.pads`, `/v1/pads/shipments/${seed}/${seed}-64-64.pads`].sort());
  assert.ok(seen.put.every((p) => p.ok && p.n === 1000 && p.auth === "Bearer tok"), "every upload carried its sha256 and the dealer bearer");
});

test("dealer --all serves every consumer the relay lists that has asked for its seed", async () => {
  const bank = fs.mkdtempSync(path.join(os.tmpdir(), "bank-"));
  const fakeDealer = path.join(repo, "test", "fixtures", "fake-dealer.sh");
  const master = "33".repeat(32);
  const issued = deriveSeed(Buffer.from(master, "hex"), "aa".repeat(32)), idle = deriveSeed(Buffer.from(master, "hex"), "bb".repeat(32));   // the ledger holds the master as bytes
  const seen = { put: [] };
  const srv = http.createServer((req, res) => {
    const u = new URL(req.url, "http://x");
    if (req.method === "GET" && u.pathname === "/v1/pads/consumers") {
      res.setHeader("content-type", "application/json");
      return res.end(JSON.stringify({ consumers: [
        { name: "box1", keyFp: "aa".repeat(32), padKey: "11".repeat(32), seed_id: issued.seed_id, epoch: 1, mark: 10, issued: true },
        { name: "box2", keyFp: "bb".repeat(32), padKey: "22".repeat(32), seed_id: idle.seed_id, epoch: 1, mark: 0, issued: false },
      ] }));
    }
    if (req.method === "PUT") { let n = 0; req.on("data", (c) => (n += c.length)); req.on("end", () => { seen.put.push(u.pathname); res.setHeader("content-type", "application/json"); res.end(JSON.stringify({ stored: true, bytes: n, sha256: "x" })); }); return; }
    res.statusCode = 404; res.end("{}");
  });
  await new Promise((r) => srv.listen(0, "127.0.0.1", r));
  const base = `http://127.0.0.1:${srv.address().port}`;
  const { stdout: out } = await execFileP("python3", [loop, "--once", "--all", "--push", "--relay", base, "--master", master,
    "--ahead", "64", "--chunk", "32", "--model", "m.gguf", "--calib", "c.calib", "--out", bank],
    { encoding: "utf8", env: { ...process.env, DEALER: fakeDealer, PADS_DEALER_TOKEN: "tok" }, timeout: 60_000 });
  srv.closeAllConnections(); srv.close(); srv.unref();
  assert.match(out, new RegExp(`consumer box1 \\(${issued.seed_id}, mark 10\\)`));
  assert.doesNotMatch(out, /consumer box2/);                                          // never asked for a seed: nothing to mint yet
  assert.deepEqual(seen.put.sort(), [`/v1/pads/shipments/${issued.seed_id}/${issued.seed_id}-0-32.pads`, `/v1/pads/shipments/${issued.seed_id}/${issued.seed_id}-32-32.pads`, `/v1/pads/shipments/${issued.seed_id}/${issued.seed_id}-64-32.pads`]);
  assert.deepEqual(fs.readdirSync(bank).filter((f) => f.endsWith(".pads")).sort(), [`${issued.seed_id}-0-32.pads`, `${issued.seed_id}-32-32.pads`, `${issued.seed_id}-64-32.pads`]);
});
