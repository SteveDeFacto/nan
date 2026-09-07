#!/usr/bin/env node
// window-agent.mjs -- the ledger-window agent a CVM-tier engine talks to
// (SHIELDED_PAD_WINDOW_URL). The engine POSTs {want, seed_id} and gets back
// {lo, hi, iat, sig}, which it verifies against SHIELDED_PAD_LEDGER_PK, so the
// agent can never widen a window on its own. Two modes:
//
//   --ledger FILE --key FILE     local: this process IS the ledger (tests, a
//                                single-box deployment). Reserve-before-use on
//                                the file (the same decimal mark the C ledger
//                                keeps), signed with the agent's own Ed25519
//                                key; GET /key prints the public half.
//   --relay URL --name N --sign-key FILE
//                                platform: relay POST /v1/pads/reserve as the
//                                attested tunnel `name`, signing the request
//                                with the tunnel's transport key (PKCS8 PEM);
//                                the platform's ledger signs the window.
//
//   node shielded/dealer/window-agent.mjs --port 9701 --ledger /tmp/ledger --key /tmp/agent.pem
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { createPrivateKey, createPublicKey, generateKeyPairSync, randomBytes, sign } from "node:crypto";
import { fileURLToPath } from "node:url";
import { MAX_WINDOW, signedMessage, windowMessage } from "../../relay/pads.mjs";

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => a.startsWith("--") ? [a.slice(2), all[i + 1] && !all[i + 1].startsWith("--") ? all[i + 1] : "1"] : []).filter(Boolean));
const port = Number(args.port || 9701);

function loadOrMakeKey(file) {
  if (file && fs.existsSync(file)) return createPrivateKey(fs.readFileSync(file, "utf8"));
  const { privateKey } = generateKeyPairSync("ed25519");
  if (file) fs.writeFileSync(file, privateKey.export({ type: "pkcs8", format: "pem" }), { mode: 0o600 });
  return privateKey;
}
const rawPub = (priv) => Buffer.from(createPublicKey(priv).export({ type: "spki", format: "der" })).subarray(-32).toString("hex");

let reserve;   // async ({ want, seed_id }) => { status, body }
let keyHex = "";
if (args.relay) {
  if (!args.name || !args["sign-key"]) { console.error("relay mode needs --name and --sign-key"); process.exit(2); }
  const sk = createPrivateKey(fs.readFileSync(args["sign-key"], "utf8"));
  const base = String(args.relay).replace(/\/+$/, "");
  const keyRes = await fetch(base + "/v1/pads/key").then((r) => r.json());
  keyHex = keyRes.key;
  reserve = async ({ want, seed_id }) => {
    const nonce = randomBytes(16).toString("hex");
    const sig = sign(null, Buffer.from(signedMessage("reserve", [args.name, seed_id, want, nonce])), sk).toString("hex");
    const r = await fetch(base + "/v1/pads/reserve", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: args.name, seed_id, want, nonce, sig }) });
    return { status: r.status, body: await r.json() };
  };
} else {
  const ledger = args.ledger || path.join(process.cwd(), "pads-ledger");
  const priv = loadOrMakeKey(args.key);
  keyHex = rawPub(priv);
  reserve = async ({ want, seed_id }) => {
    let mark = 0;
    try { mark = Number(fs.readFileSync(ledger, "utf8").trim()) || 0; } catch {}
    const lo = mark, hi = mark + want, iat = Math.floor(Date.now() / 1000);
    const tmp = ledger + ".tmp";
    fs.writeFileSync(tmp, String(hi) + "\n");
    const fd = fs.openSync(tmp, "r"); fs.fsyncSync(fd); fs.closeSync(fd);
    fs.renameSync(tmp, ledger);                       // durable BEFORE the window is signed
    const sig = sign(null, Buffer.from(windowMessage(seed_id, lo, hi, iat)), priv).toString("hex");
    return { status: 200, body: { seed_id, lo, hi, iat, sig } };
  };
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  const json = (status, body) => { const b = JSON.stringify(body); res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(b) }); res.end(b); };
  if (req.method === "GET" && url.pathname === "/key") return json(200, { key: keyHex });
  if (req.method === "POST" && url.pathname === "/window") {
    let raw = ""; for await (const c of req) { raw += c; if (raw.length > 4096) { return json(413, { error: "too_big" }); } }
    let body; try { body = JSON.parse(raw || "{}"); } catch { return json(400, { error: "bad_json" }); }
    const want = Number(body.want);
    if (!Number.isInteger(want) || want < 1 || want > MAX_WINDOW) return json(400, { error: "bad_window" });
    if (!/^[0-9a-f]{32}$/.test(String(body.seed_id || ""))) return json(400, { error: "bad_seed_id" });
    try { const r = await reserve({ want, seed_id: body.seed_id }); return json(r.status, r.body); }
    catch (e) { return json(502, { error: "reserve_failed", message: String(e && e.message || e) }); }
  }
  json(404, { error: "not_found" });
});
server.listen(port, "127.0.0.1", () => {
  console.log(`[window-agent] ${args.relay ? "relay " + args.relay + " as " + args.name : "local ledger " + (args.ledger || "pads-ledger")} on http://127.0.0.1:${port}; ledger key ${keyHex}`);
});
