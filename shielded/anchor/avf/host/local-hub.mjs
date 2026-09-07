// local-hub.mjs -- a fleet tunnel hub on this box, for driving the phone's
// relay attach end to end before it meets api.enclave.host.
//
//   node host/local-hub.mjs [--port 8787] [--code-hash hex] [--authority hex] [--root-pin hex]
//   adb reverse tcp:8787 tcp:8787
//   ... am start-foreground-service ... --es relay ws://127.0.0.1:8787/v1/fleet-tunnel --es name pixel
//
// With no pins given it uses placeholders, so a real chain is REFUSED at the
// code-hash step (after the root and signature passed): that refusal, and the
// reasons before it, are what this run is for.
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createTunnelHub } from "../../../../relay/tunnel.js";
import { createPadsLedger, createShipmentStore, padsRouter } from "../../../../relay/pads.mjs";
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const port = +arg("--port", 8787);
const avf = { codeHashes: [arg("--code-hash", "00".repeat(32))], authorityHashes: [arg("--authority", "00".repeat(64))] };
if (arg("--root-pin", null)) avf.rootPins = [arg("--root-pin")];
// --dev-unattested: bind a phone that cannot attest (Pixel 8 Pro) on its transport
// key alone; needs ENCLAVE_DEV_UNATTESTED=1 in the environment as well (tunnel.js).
const devUnattested = process.argv.includes("--dev-unattested");
const hub = createTunnelHub({ allow: [], attest: { avf, devUnattested }, onChange: (why, name) => {
  console.log(`[hub] ${why} ${name}:`, JSON.stringify(hub.origins().find((o) => o.name === name) || null));
  if (why === "attach" || why === "hello") hub.fetchJson(`tunnel://${name}`, "/availability").then((r) => console.log(`[hub] ${name} /availability ->`, JSON.stringify(r).slice(0, 300))).catch((e) => console.log(`[hub] ${name} /availability failed: ${e.message}`));
} });
// Dealt pads: the same ledger, seeds, windows and shipment store api-relay serves
// (relay/pads.mjs), on this hub, so the phone's PadsClient and the dealer loop run
// against it. State under out/local-hub-data; PADS_MASTER_SEED / PADS_DEALER_TOKEN
// as on the relay.
const dataDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "out", "local-hub-data");
const ledger = createPadsLedger({ dir: dataDir, hub, masterSeed: process.env.PADS_MASTER_SEED || null, log: (m) => console.log(m) });
const store = createShipmentStore({ dir: dataDir });
const json = (res, status, body) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(body)); };
const readBody = (req, max = 262144) => new Promise((resolve, reject) => {
  const chunks = []; let n = 0;
  req.on("data", (ch) => { n += ch.length; if (n > max) { req.destroy(); reject(new Error("body too large")); } else chunks.push(ch); });
  req.on("end", () => resolve(Buffer.concat(chunks))); req.on("error", reject);
});
const pads = padsRouter({ ledger, store, dealerToken: (process.env.PADS_DEALER_TOKEN || "").trim(), json, readBody });
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (await pads(req, res, url)) return;
  res.end("local hub\n");
});
server.on("upgrade", (req, socket, head) => hub.handleUpgrade(req, socket, head));
server.listen(port, "127.0.0.1", () => console.log(`[hub] listening ws://127.0.0.1:${port}/v1/fleet-tunnel  avf policy: code=${avf.codeHashes[0].slice(0, 16)}… authority=${avf.authorityHashes[0].slice(0, 16)}…${avf.rootPins ? " root pinned to " + avf.rootPins[0].slice(0, 16) + "…" : " (Google roots)"}`));
