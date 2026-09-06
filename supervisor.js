// Enclave supervisor - the WHOLE service, running INSIDE the Tinfoil enclave behind
// the shim (the single ingress). It is the measured/attested image: the same
// published code that checks a user's signature, gates on escrow, mints the
// session token, launches the per-use container, and proxies the data path.
//
// There is no external tier. Browser -> shim -> here, for BOTH control and data:
//   control:  /v1/*        (SIWE login, deployments, account, attestation)
//   data:     /x/:id/*     (verify session token + ownership, proxy to the
//                           spawned container; fly.io used to do nothing here -
//                           now nothing external touches a prompt at all)
//
// One token type: the session JWT the browser gets at login is reused as the
// capability on the data path. It is ES256-signed by a key MINTED IN-ENCLAVE at
// boot (see initSessionKey) — the operator, who provisions the fleet SECRET,
// cannot forge one because the private half never leaves this CVM. SECRET now
// only backs the manager control-token and the DNS-push HMAC — it never signs
// or verifies a session token.
//
// >>> The ONLY thing left to implement for your CVM is spawn/stop/measure below.

import express from "express";
import cors from "cors";
import http from "node:http";
import https from "node:https";
import net from "node:net";
import tls from "node:tls";
import dgram from "node:dgram";
import { createHash, createHmac, randomBytes, generateKeyPairSync, createPublicKey, createPrivateKey, sign as cryptoSign, verify as cryptoVerify, timingSafeEqual, X509Certificate } from "node:crypto";
import { gunzipSync } from "node:zlib";
import { execFileSync, execFile } from "node:child_process";
import { promisify } from "node:util";
const pexec = promisify(execFile);
import { mkdtempSync, readFileSync, rmSync, mkdirSync, writeFileSync, renameSync, existsSync, chmodSync, statfsSync, readdirSync, unlinkSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { WebSocketServer, createWebSocketStream } from "ws";
import { verifyMessage, createPublicClient, createWalletClient, http as viemHttp, fallback as viemFallback, getAddress, keccak256, toHex, stringToBytes, parseEventLogs, encodeAbiParameters } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { base } from "viem/chains";
import { SignJWT, jwtVerify } from "jose";
import { Verifier, assembleAttestationBundle } from "@tinfoilsh/verifier";
// dedicated-IP egress: the outbound half of the per-deployment address (see egress.js)
import { createEgress } from "./egress.js";
// contract addresses: LIVE BINDINGS owned by addressbook.js — seeded from the
// baked env, overridden from the on-chain EnclaveAddressBook when
// ADDRESS_BOOK_ADDRESS is set, and re-polled so contract redeploys reach a
// RUNNING enclave without a new release.
import { initAddressBook, REGISTRY_ADDRESS, DEPLOYMENTS_ADDRESS, APP_CATALOG_ADDRESS,
         FORWARDER_ADDRESS, PROOF_OF_TIME_ADDRESS } from "./addressbook.js";

// Process-wide crash guards. This is Express 4: a rejected async route (or any
// stray background rejection) would otherwise take the whole process down —
// killing EVERY tenant app hosted on this CVM, not just the one request. Log in
// the house style and KEEP RUNNING; per-request failures are already answered by
// the wrap() adapter + error middleware (see the app below). We deliberately do
// NOT exit: a single bad request or a transient library throw must never evict
// the fleet of apps this supervisor is fronting.
process.on("unhandledRejection", (reason) => {
  console.error(`[fatal-guard] unhandledRejection (kept running): ${reason && (reason.stack || reason.message || reason)}`);
});
process.on("uncaughtException", (err) => {
  console.error(`[fatal-guard] uncaughtException (kept running): ${err && (err.stack || err.message || err)}`);
});

// Resolve the book BEFORE anything below derives state from the addresses
// (top-level await; a no-op without ADDRESS_BOOK_ADDRESS, baked env on failure).
await initAddressBook();

// ----------------------------------------------------------------------------
// config
// ----------------------------------------------------------------------------
const PORT           = parseInt(process.env.PORT || "8080", 10);
const SECRET         = new TextEncoder().encode(need("SECRET")); // signs + verifies the session/capability token
const PUBLIC_URL     = (process.env.PUBLIC_URL || "").replace(/\/+$/, ""); // own shim URL; else derived per-request
const SIWE_DOMAIN    = process.env.SIWE_DOMAIN || "enclave.host";
const SIWE_URI       = process.env.SIWE_URI || "https://enclave.host";
const CHAIN_ID       = parseInt(process.env.CHAIN_ID || "8453", 10);
const CORS_ORIGINS   = (process.env.CORS_ORIGINS || "https://enclave.host").split(",").map(s => s.trim()).filter(Boolean);

// ---- session signing key (in-enclave, asymmetric) --------------------------
// The session JWT proves "you hold wallet X" for private deployments, logs, and
// owner-only endpoints. It USED to be HS256 over the fleet-wide SECRET — but that
// makes the MINTING key equal to the VERIFYING key equal to a value the operator
// provisions, so the operator could mint a token for ANY wallet and skip the
// signature check that login enforces. Now the token is ES256, signed by an
// EC P-256 private key MINTED IN-ENCLAVE at boot (like the TLS-bridge key): the
// operator never sees the private half, so they cannot forge a session. The
// public half is published (/v1/session-jwks, and inside /v1/attestation) so
// anyone can verify a token — and confirm the operator did not mint it — holding
// no secret. Persisted to its OWN tmpfs (never host disk) so a container restart
// within a CVM boot keeps sessions valid; a full relaunch mints a fresh key, at
// which point clients re-attest + re-login anyway (the shim TLS pin also rotates).
const SESSION_KEY_DIR = process.env.SESSION_KEY_DIR || "/mnt/ramdisk/enclave-session";
let SESSION_PRIV = null, SESSION_PUB = null, SESSION_JWK = null, SESSION_KID = "";

function initSessionKey() {
  const keyPath = join(SESSION_KEY_DIR, "session-ec-p256.pkcs8.pem");
  let privObj = null;
  try { privObj = createPrivateKey(readFileSync(keyPath, "utf8")); } catch {}   // reuse across a container restart
  if (!privObj) {
    const { privateKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
    privObj = privateKey;
    try {
      mkdirSync(SESSION_KEY_DIR, { recursive: true });
      writeFileSync(keyPath, privateKey.export({ type: "pkcs8", format: "pem" }), { mode: 0o600 });
    } catch (e) { console.error("[session] tmpfs persist failed — key is in-memory only this boot:", e.message); }
    console.log("[session] minted in-enclave ES256 session-signing key");
  }
  SESSION_PRIV = privObj;
  SESSION_PUB  = createPublicKey(privObj);
  const j = SESSION_PUB.export({ format: "jwk" });                 // { kty:'EC', crv:'P-256', x, y }
  SESSION_KID = jwkThumbprint(j);                                  // RFC 7638; stable per key, unique per enclave
  SESSION_JWK = { kty: j.kty, crv: j.crv, x: j.x, y: j.y, kid: SESSION_KID, alg: "ES256", use: "sig" };
}

// ---- proof-of-time signing key (in-enclave, secp256k1) ---------------------
// The key that signs EnclaveProofOfTime checkpoints: "deployment X was running
// here through time T, anchored to block N". Minted IN-ENCLAVE at boot for
// exactly the same reason as the session key above — and here the reason is
// money. The operator EOA (REGISTRY_PRIVATE_KEY) is provisioned from OUTSIDE
// the CVM: the seller creates it and hands it in (Tinfoil secret, or metal
// fw_cfg), so a signature from THAT key proves nothing about what this box is
// running. This key the operator has never seen, so a checkpoint signed with it
// can only have been produced by the measured supervisor inside this CVM, which
// signs only for tenants it has just probed as alive.
//
// secp256k1, not the session key's P-256: the ledger verifies with ecrecover,
// which is a precompile on every EVM chain, so no dependency on RIP-7212 being
// enabled. Same tmpfs as the session key (never host disk): it survives a
// container restart within one CVM boot, and a full relaunch mints a fresh one
// that the enclave republishes to the registry (setProofKey) at boot. Rotation
// is safe mid-lease — the ledger checks each proof against the CURRENTLY
// registered key, and proofs already accepted are immutable.
let PROOF_ACCOUNT = null;                      // viem account; PROOF_ACCOUNT.address is what we register

function initProofKey() {
  const keyPath = join(SESSION_KEY_DIR, "proof-secp256k1.hex");
  let hex = "";
  try { hex = readFileSync(keyPath, "utf8").trim(); } catch {}   // reuse across a container restart
  if (!/^0x[0-9a-f]{64}$/i.test(hex)) {
    hex = "0x" + randomBytes(32).toString("hex");
    try {
      mkdirSync(SESSION_KEY_DIR, { recursive: true });
      writeFileSync(keyPath, hex, { mode: 0o600 });
    } catch (e) { console.error("[proof] tmpfs persist failed — key is in-memory only this boot:", e.message); }
    console.log("[proof] minted in-enclave secp256k1 proof-of-time signing key");
  }
  PROOF_ACCOUNT = privateKeyToAccount(hex);
  console.log(`[proof] proof-of-time signer ${PROOF_ACCOUNT.address}`);
}

// Mint the session token: ES256 over the in-enclave key. `iss`/`kid` = our key
// thumbprint, so a verifier can tell OUR tokens from another enclave's.
async function mintSession(subject, expiresAt) {
  return new SignJWT({}).setProtectedHeader({ alg: "ES256", kid: SESSION_KID })
    .setIssuer(SESSION_KID).setSubject(subject)
    .setExpirationTime(expiresAt.getTime() / 1000 | 0).sign(SESSION_PRIV);
}

// Verify a session token -> checksummed address, or null. ES256 ONLY, verified
// against our in-enclave PUBLIC key, with `algorithms` pinned so no other alg
// (e.g. an HS256 token an attacker tries to have verified against the EC key as
// an HMAC secret — the classic alg-confusion) is ever accepted. A token whose
// kid is a DIFFERENT enclave's fails closed here → the client re-runs SIWE
// against whichever enclave serves it (pin-to-issuer). On the current
// single-enclave fleet that never triggers. (Transparent fleet roaming via
// attestation-anchored peer JWKS is the documented follow-on — see docs/session-auth.md.)
async function verifySessionToken(token) {
  if (!token || typeof token !== "string" || !SESSION_PUB) return null;
  let hdr;
  try { hdr = JSON.parse(Buffer.from(token.split(".")[0] || "", "base64url").toString("utf8")); } catch { return null; }
  if (!hdr || hdr.alg !== "ES256" || hdr.kid !== SESSION_KID) return null;
  try {
    const { payload } = await jwtVerify(token, SESSION_PUB, { algorithms: ["ES256"], issuer: SESSION_KID });
    // A CONTROL-PLANE session carries no audience. App-origin cookie tokens
    // (mintAppToken) are signed by this same key and would otherwise verify
    // here identically — and that token lives in a cookie on a TENANT origin,
    // reachable by any app bug that can make the browser issue a same-origin
    // request. Refusing every token that names an audience keeps the blast
    // radius of a leaked app cookie at "that one app", instead of handing over
    // deployment listing, logs, secrets and every other `authed` route.
    if (payload.aud !== undefined) return null;
    return getAddress(payload.sub);
  }
  catch { return null; }
}

// ---- app-origin session (private deployments in a browser) -----------------
// A browser's top-level navigation cannot carry `Authorization: Bearer`, so a
// private deployment was unreachable by clicking a link even AS ITS OWNER — the
// gate below answered a bare 401 JSON blob. The fix is a second carriage (a
// cookie on the app origin) for the SAME owner check, never a second rule.
//
// It is deliberately NOT the control-plane session token:
//   • `aud` binds it to ONE deployment, so it opens nothing else on this box;
//   • the TTL is short next to SESSION_TTL's 7 days;
//   • verifySessionToken above refuses it outright.
// The cookie is HttpOnly (app JS can never read it) and stripped back off the
// request before we proxy (see the /x/:id handler), so the tenant never sees it.
const APP_COOKIE  = "enclave_app";
const APP_TTL_SEC = 12 * 3600;
const appAud = (id) => "app:" + id;

async function mintAppToken(subject, id) {
  return new SignJWT({}).setProtectedHeader({ alg: "ES256", kid: SESSION_KID })
    .setIssuer(SESSION_KID).setSubject(subject).setAudience(appAud(id))
    .setExpirationTime((Date.now() / 1000 | 0) + APP_TTL_SEC).sign(SESSION_PRIV);
}

// Verify an app cookie for THIS deployment -> checksummed address, or null.
// `audience` is passed to jwtVerify rather than compared afterwards so the
// check cannot be skipped by a payload shape we did not anticipate; a token
// minted for another deployment on this same enclave fails closed here.
async function verifyAppToken(token, id) {
  if (!token || typeof token !== "string" || !SESSION_PUB || !id) return null;
  let hdr;
  try { hdr = JSON.parse(Buffer.from(token.split(".")[0] || "", "base64url").toString("utf8")); } catch { return null; }
  if (!hdr || hdr.alg !== "ES256" || hdr.kid !== SESSION_KID) return null;
  try {
    const { payload } = await jwtVerify(token, SESSION_PUB,
      { algorithms: ["ES256"], issuer: SESSION_KID, audience: appAud(id) });
    return getAddress(payload.sub);
  } catch { return null; }
}

// Minimal RFC 6265 cookie read, returning EVERY value sent under `name`.
// Values we mint are base64url JWTs, so no quoted-string or percent-decoding
// case can arise; anything unparseable reads as absent and the caller falls
// through to the login bounce.
//
// All of them, not the first, because a browser will happily send several
// pairs of the same name and the server cannot tell which came from where.
// `app.enclave.host` is not on the Public Suffix List, so a hostile tenant at
// hostile.app.enclave.host can set `enclave_app=junk; Domain=app.enclave.host`
// and have it delivered to a VICTIM's app origin alongside the real host-only
// cookie. RFC 6265 §5.4 sorts by longer path then earlier creation — both of
// which the attacker chooses — so reading only the first pair let a stranger
// lock an owner out of their own paid app indefinitely (each re-login lands
// behind the planted pair, and the loop-breaker then reports a hard refusal).
// No token of theirs can ever VERIFY for someone else's deployment — the
// audience forbids it — so this was always denial of access, never access.
function cookieVals(req, name) {
  const raw = req.headers.cookie;
  if (!raw) return [];
  const out = [];
  for (const part of String(raw).split(";")) {
    const eq = part.indexOf("=");
    if (eq < 0) continue;
    if (part.slice(0, eq).trim() === name) out.push(part.slice(eq + 1).trim());
  }
  return out;
}
// This enclave's on-chain identity is bound to its OWN attested shim-cert SAN
// (see registerFromShimCert), never to the request Host/x-forwarded-host header —
// a spoofable value — so no caller can make the enclave advertise a bogus origin.
// --- pay-per-deploy (no custody): users pay the EnclavePay forwarder; the supervisor
//     WATCHES it for Paid events and converts each payment to runtime. No held
//     balance, no escrow contract, no key in the enclave that can move funds.
//     (FORWARDER_ADDRESS is a live binding from ./addressbook.js)
const USDC_ADDRESS       = process.env.USDC_ADDRESS || "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"; // USDC on Base
// --- app approval: EnclaveAppCatalog (read-only) is the deploy gate for ALL apps
//     (the image ships no deployable apps of its own). Only the catalog's owner
//     (the EOA that deployed it) can approve/reject a version, by signing a
//     setApproval transaction; an ipfs://<cid> deploy is refused until its
//     version is Approved. Empty = nothing can deploy at all (fail closed).
//     (APP_CATALOG_ADDRESS is a live binding from ./addressbook.js)
const PAYMENT_WINDOW_SEC = parseInt(process.env.PAYMENT_WINDOW_SEC || "600", 10); // unpaid awaiting_payment TTL
// Cap concurrent UNPAID (awaiting_payment) reservations per owner. Each one holds
// hardware for the whole PAYMENT_WINDOW_SEC before any money lands, so without a
// bound one wallet could reserve the node's capacity for free. Paid/running
// deployments are never counted. 0 disables the cap (previous behavior).
const MAX_UNPAID_PER_OWNER = parseInt(process.env.MAX_UNPAID_PER_OWNER || "3", 10);
const GRACE_SEC          = parseInt(process.env.GRACE_SEC || "90", 10);           // post-expiry grace before teardown
const PAY_POLL_SEC       = parseInt(process.env.PAY_POLL_SEC || "12", 10);        // Base log poll interval
// --- fair billing: funded runtime is a BALANCE, not a wall-clock deadline -----
// remainingMs drains only on ticks where the platform is actually serving:
// supervisor up, backend manager healthy, payment watcher fresh, app instance
// alive. Any outage FREEZES every clock; it resumes on the first healthy tick.
// State is persisted so a supervisor restart freezes (never forfeits) the clock.
const BILL_TICK_SEC      = parseInt(process.env.BILL_TICK_SEC || "15", 10);       // billing/reaper cadence
const WATCHER_STALE_SEC  = Math.max(60, 5 * PAY_POLL_SEC);                        // watcher silence that freezes billing
const STATE_FILE         = process.env.STATE_FILE || "/var/lib/enclave/state.json";   // mount a volume here to survive restarts
// manual-billing / pilot: boot deployments WITHOUT waiting for an on-chain payment.
//   AUTO_PROVISION=1            -> every deploy provisions immediately (closed pilot).
//   ADMIN_TOKEN set            -> operator can provision one deployment on demand via
//                                 POST /v1/admin/deployments/:id/provision (x-admin-token).
//   AUTO_PROVISION_HOURS > 0   -> optional safety expiry; 0 = runs until deleted.
const AUTO_PROVISION       = /^(1|true|on)$/i.test(process.env.AUTO_PROVISION || "");
const AUTO_PROVISION_HOURS = parseFloat(process.env.AUTO_PROVISION_HOURS || "0");
const ADMIN_TOKEN          = process.env.ADMIN_TOKEN || "";
const BASE_RPC       = process.env.BASE_RPC || "https://mainnet.base.org";
// Multi-provider RPC pool: one throttled provider must never fail-close the
// claim path — the catalog-approval gate and hint evaluation read the chain,
// and when the single RPC rate-limits, every claim declines "Could not verify
// this app's approval" while capacity sits free (observed fleet-wide
// 2026-07-17 on mainnet.base.org). BASE_RPC stays FIRST so an explicit
// override is authoritative; the rest are fallbacks (mirrors cli/enclave.mjs
// and the relay's publicnode lesson).
const RPC_POOL = [...new Set([BASE_RPC,
  "https://base-rpc.publicnode.com", "https://base.drpc.org",
  "https://1rpc.io/base", "https://mainnet.base.org"])];
const rpcTransport = () => viemFallback(RPC_POOL.map((u) => viemHttp(u, { retryCount: 2, retryDelay: 500 })));
const SESSION_TTL    = parseInt(process.env.SESSION_TTL || "604800", 10); // 7d: SIWE is lazy now (only logs/attestation/private data need it) - make the one signature rare
const DEFAULT_IMAGE  = process.env.DEFAULT_IMAGE || "debian:bookworm-slim"; // any stock image
// --- worker launch: tenants run as the manager's wasmtime/CUDA processes ------
const MPS_PIPE_DIR   = process.env.CUDA_MPS_PIPE_DIRECTORY || "/tmp/nvidia-mps";
const ENABLE_MPS     = !/^(0|false|off)$/i.test(process.env.ENABLE_MPS || "1"); // MPS enforces BOTH the SM cap and the VRAM cap (validated under CC)
const SPAWN_TIMEOUT_MS = parseInt(process.env.SPAWN_TIMEOUT_MS || "300000", 10); // includes image pull / wasm fetch (prefetched claims hit the cache, this is headroom)
const WORKER_MEM      = process.env.WORKER_MEM || "16g";                // host-RAM cap per worker (not GPU)
const WORKER_PIDS     = process.env.WORKER_PIDS || "512";
// ---- worker MANAGER (Layer 2/3) --------------------------------------------
// The GPU container runs a manager that forks one MPS-capped CHILD PROCESS per
// tenant. The supervisor routes deploys/submissions HERE instead of creating
// containers itself (Tinfoil forbids runtime container creation). Reachable over
// the enclave-local network; default loopback.
const WORKER_MGR_URL  = (process.env.WORKER_MGR_URL || "http://127.0.0.1:8090").replace(/\/+$/, "");
// Opt-in bearer for the GPU worker manager's control plane (worker/worker.py
// WORKER_TOKEN). Unset = no header (worker runs its loopback-only, tokenless
// default); set the SAME value in both envs to require auth on /tenants,/run,etc.
const WORKER_TOKEN    = process.env.WORKER_TOKEN || "";
// provisioning backend: "worker" = GPU PTX submission (default), "vm" = tenant-app
// hosting via the app manager on VMMGR_URL (the wasm-manager runs each app as a
// `wasmtime serve` process). The "vm"/VMMGR_URL names are legacy, kept for config compat.
const PROVISION_BACKEND = (process.env.PROVISION_BACKEND || "worker").toLowerCase();
const VMMGR_URL = (process.env.VMMGR_URL || "http://127.0.0.1:8091").replace(/\/+$/, "");
function mgrReq(method, path, body, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const u = new URL(WORKER_MGR_URL + path);
    const data = body != null ? Buffer.from(JSON.stringify(body)) : null;
    const r = http.request(
      { host: u.hostname, port: u.port || 80, path: u.pathname + u.search, method, timeout: timeoutMs,
        headers: { "Content-Type": "application/json", ...(data ? { "Content-Length": data.length } : {}),
                   ...(WORKER_TOKEN ? { "Authorization": `Bearer ${WORKER_TOKEN}` } : {}) } },
      (res) => { let buf = ""; res.on("data", (c) => (buf += c));
                 res.on("end", () => { let j; try { j = JSON.parse(buf || "{}"); } catch { j = { raw: buf }; }
                                       resolve({ status: res.statusCode || 0, body: j }); }); });
    r.on("error", reject); r.on("timeout", () => r.destroy(new Error("manager timeout")));
    if (data) r.write(data); r.end();
  });
}
async function mgrHealth(timeoutMs = 3000) {
  const r = await mgrReq("GET", "/health", null, timeoutMs);
  if (r.status !== 200) throw new Error(`manager /health ${r.status}`);
  // the worker holds the card this container can't see: adopt its probed VRAM
  if (r.body && r.body.gpuVramSource === "nvidia-smi") adoptCardVram(r.body.gpuVramGb, "worker");
  return r.body;
}

// --- app manager client ("vm" backend on VMMGR_URL; the wasm-manager) --------
// The manager's control API is loopback-reachable by TENANTS too (guests hold
// outbound HTTP), so it enforces a shared control token when configured: both
// containers derive it from the same SECRET and the manager rejects control
// calls without it. VMMGR_TOKEN overrides if the two ever need to differ.
//
// DERIVED, never the raw SECRET (matches SECRETS_FETCH_KEY / DNS_TXT_KEY
// below). Unlike those two - which are only ever HMAC keys - this one is SENT,
// as a bearer header on every launch/kill/control call. Sending the master that
// the fleet's other credentials are derived from meant one observed header
// yielded the whole keyring; sending a leaf yields only this control plane.
// The manager's rollout-window acceptance of the raw value is GONE (2026-07-27),
// so this derivation is the only thing that opens its control plane: a manager
// image older than c1b7352c would now fail every control call (its /health
// silently drops to the unauthenticated subset — no volumes, no capacity, no nn
// probe), which is why metal/build-image.mjs pins the two images in step.
const VMMGR_TOKEN = process.env.VMMGR_TOKEN
  || (SECRET.length ? createHmac("sha256", SECRET).update("enclave vmmgr v1").digest("hex") : "");
function vmReq(method, path, body, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const u = new URL(VMMGR_URL + path);
    const data = body != null ? Buffer.from(JSON.stringify(body)) : null;
    const r = http.request(
      { host: u.hostname, port: u.port || 80, path: u.pathname + u.search, method, timeout: timeoutMs,
        headers: { "Content-Type": "application/json", ...(VMMGR_TOKEN ? { "X-Vmmgr-Token": VMMGR_TOKEN } : {}),
                   ...(data ? { "Content-Length": data.length } : {}) } },
      (res) => { let buf = ""; res.on("data", (c) => (buf += c));
                 res.on("end", () => { let j; try { j = JSON.parse(buf || "{}"); } catch { j = { raw: buf }; }
                                       resolve({ status: res.statusCode || 0, body: j }); }); });
    r.on("error", reject); r.on("timeout", () => r.destroy(new Error("vmmanager timeout")));
    if (data) r.write(data); r.end();
  });
}
async function vmHealth(timeoutMs = 3000) {
  const r = await vmReq("GET", "/health", null, timeoutMs);
  if (r.status !== 200) throw new Error(`vmmanager /health ${r.status}`);
  // the wasm-manager holds the card this container can't see: adopt its probed VRAM
  if (r.body && r.body.gpuVramSource === "nvidia-smi") adoptCardVram(r.body.gpuVramGb, "manager");
  return r.body;
}

// ---- on-chain discovery: self-register in EnclaveRegistry (no trusted gateway) --
// On boot the enclave publishes itself (endpoint + attestation repo) to the
// registry contract on Base, then heartbeats. Callers read the registry from
// any RPC and connect DIRECTLY, verifying attestation themselves. Entirely
// opt-in: if REGISTRY_ENABLED isn't set, the enclave just doesn't advertise.
const REGISTRY_ENABLED  = /^(1|true|on)$/i.test(process.env.REGISTRY_ENABLED || "");
// (REGISTRY_ADDRESS is a live binding from ./addressbook.js)
const REGISTRY_PK       = process.env.REGISTRY_PRIVATE_KEY || "";        // operator key (enclave secret); needs a little Base ETH for gas
const ENCLAVE_REPO      = process.env.ENCLAVE_REPO || "";                // e.g. "EnclaveHost/enclave" - what callers attest against; MUST match GitHub's canonical casing (Sigstore compares it verbatim)
const ENCLAVE_MEASUREMENT = process.env.ENCLAVE_MEASUREMENT || ("0x" + "0".repeat(64)); // optional cross-check
const HEARTBEAT_SEC     = parseInt(process.env.REGISTRY_HEARTBEAT_SEC || "900", 10);
// The endpoint we advertise is NOT configured — it is derived from the request
// (originOf: the exact hostname the caller reached us at, which is the attested
// one and the only thing a verifier can use). Static config is validated once;
// the endpoint arrives per-request. PUBLIC_URL, if set, pins it (eager register).
const REGISTRY_READY    = REGISTRY_ENABLED && !!(REGISTRY_ADDRESS && REGISTRY_PK && ENCLAVE_REPO);
if (REGISTRY_ENABLED && !REGISTRY_READY)
  console.warn("[registry] REGISTRY_ENABLED but REGISTRY_ADDRESS/REGISTRY_PRIVATE_KEY/ENCLAVE_REPO incomplete — not advertising");
// Registry schema 2 added the two per-machine prices to register() and the
// entry itself (the ledger reads them when we claim); schema 3 appended the
// PROOF KEY, the in-CVM signer whose checkpoints earn us our lease seconds.
// Older registries are still out there during a migration, so we sniff and fall
// back — on a schema-1 registry pricing stays the old global list price, and on
// anything below 3 there is nowhere to publish a proof key (a rev-9 ledger
// refuses to sell us work in that state, which is the correct fail-closed).
const REGISTRY_ABI = [
  { type: "function", name: "register", stateMutability: "nonpayable",
    inputs: [{ name: "endpoint", type: "string" }, { name: "repo", type: "string" }, { name: "measurement", type: "bytes32" },
             { name: "cpuPricePerSec6", type: "uint64" }, { name: "gpuPricePerSec6", type: "uint64" },
             { name: "proofKey", type: "address" }],
    outputs: [{ name: "id", type: "bytes32" }] },
  { type: "function", name: "setProofKey", stateMutability: "nonpayable",
    inputs: [{ name: "id", type: "bytes32" }, { name: "proofKey", type: "address" }], outputs: [] },
  { type: "function", name: "setPrices", stateMutability: "nonpayable",
    inputs: [{ name: "id", type: "bytes32" }, { name: "cpuPricePerSec6", type: "uint64" }, { name: "gpuPricePerSec6", type: "uint64" }],
    outputs: [] },
  { type: "function", name: "heartbeat", stateMutability: "nonpayable",
    inputs: [{ name: "id", type: "bytes32" }], outputs: [] },
];
const REGISTRY_ABI_V1 = [
  { type: "function", name: "register", stateMutability: "nonpayable",
    inputs: [{ name: "endpoint", type: "string" }, { name: "repo", type: "string" }, { name: "measurement", type: "bytes32" }],
    outputs: [{ name: "id", type: "bytes32" }] },
];
const REGISTRY_ABI_V2 = [
  { type: "function", name: "register", stateMutability: "nonpayable",
    inputs: [{ name: "endpoint", type: "string" }, { name: "repo", type: "string" }, { name: "measurement", type: "bytes32" },
             { name: "cpuPricePerSec6", type: "uint64" }, { name: "gpuPricePerSec6", type: "uint64" }],
    outputs: [{ name: "id", type: "bytes32" }] },
];
// 4 = the entry also carries the seller's declared PAYOUT WALLET (what makes a
// deployment free on its owner's own box); 3 = proof key; 2 = prices only;
// 1 = none of it (the getter reverts there)
let _registryRev = 0;
let _registryRevOf = null;        // WHICH registry _registryRev was sniffed against
async function registryRev() {
  // Cache per REGISTRY_ADDRESS: the book repoints it live, and a rev sniffed
  // against the old contract picks the wrong register() ABI on the new one.
  const on = (REGISTRY_ADDRESS || "").toLowerCase();
  if (_registryRev && _registryRevOf === on) return _registryRev;
  try {
    const rev = Number(await chainClient.readContract({ address: getAddress(REGISTRY_ADDRESS),
      abi: [{ type: "function", name: "registrySchema", stateMutability: "view", inputs: [], outputs: [{ type: "uint256" }] }],
      functionName: "registrySchema" })) || 1;
    _registryRev = rev; _registryRevOf = on;
    return rev;
  } catch (e) {
    // ONLY a revert means "no such getter, so this is a rev-1 registry". Any
    // other failure is the RPC, and caching it as rev 1 is unrecoverable: the
    // box then calls the 3-arg register() forever, which does not exist on a
    // schema-2/3 registry, so every attempt reverts and no retry can heal it.
    // Cost a live box its registration entirely (metal0, 2026-07-28: sniffed
    // one second after boot, RPC not warm, wrong ABI from then on).
    if (/revert/i.test(e?.shortMessage || e?.message || "")) {
      _registryRev = 1; _registryRevOf = on;
      return 1;
    }
    console.warn(`[registry] schema sniff failed (not caching): ${e.shortMessage || e.message}`);
    throw e;                       // callers already treat a failed register as retryable
  }
}

// Register THIS enclave under `endpoint` (the hostname a caller reached us at),
// then heartbeat. Fires at most once (guarded); a transient failure resets the
// guard so a later request retries. Never fatal — a failed advertisement must
// not take down the enclave.
let _registered = false;
let _registeredOn = null;         // WHICH registry _registered refers to (see below)
let _enclaveId = null;            // our EnclaveRegistry id (keccak256 of the advertised endpoint); claim gating needs it
let _advertisedEndpoint = null;   // the endpoint we registered under; adopted deployments build their URL from it
let _certSan = null;              // our own attested shim-cert SAN hostname — the ONLY name we self-register / advertise
/// Has the address book moved the registry out from under our registration?
/// Every guard that would otherwise short-circuit on `_registered` has to ask
/// this too, or the box latches onto a registry nobody reads any more.
function registryRepointed() {
  return _registered && !!_registeredOn && _registeredOn !== (REGISTRY_ADDRESS || "").toLowerCase();
}
async function registerOnChain(endpoint) {
  endpoint = (endpoint || "").replace(/\/+$/, "");
  // REGISTRY_ADDRESS is a LIVE binding: the address-book poller repoints it in a
  // running process, which is the entire point of the book. The once-per-process
  // guard therefore has to be keyed on WHICH registry we registered with, not on
  // "have we ever registered". Keyed on the boolean alone, a registry redeploy
  // stranded the whole fleet: every box kept _registered = true and never
  // registered on the new contract, while the new ledger (which reads that
  // contract) rejected every claim with "not operator" — an unregistered id
  // reads back as a zero-filled entry whose operator is 0x0. Nothing self-healed
  // and nothing said why; the boxes looked healthy and simply never claimed.
  if (registryRepointed()) {
    console.warn(`[registry] address book repointed the registry ${_registeredOn} -> ${(REGISTRY_ADDRESS || "").toLowerCase()}; re-registering`);
    _registered = false;
    _enclaveId = null;                                      // the old id does not exist on the new registry
  }
  if (!REGISTRY_READY || _registered || !endpoint) return;
  _registered = true;                                       // claim before await so a request burst registers once
  _registeredOn = (REGISTRY_ADDRESS || "").toLowerCase();
  try {
    // register/heartbeat go through the shared operator-tx queue (sendOperatorTx,
    // defined with the claim loop): the SAME EOA signs registry and ledger txs,
    // and public RPCs cap EIP-7702-delegated accounts at one in-flight tx — so
    // every tx from this key is serialized through confirmation, never raced.
    const id = keccak256(stringToBytes(endpoint));
    // schema 2: the entry carries our PRICE, and the ledger charges tenants
    // out of it. Re-registering re-states it, so a price change lands with the
    // next boot; setPrices (below) covers a change without one.
    const rev = await registryRev();
    const proofKey = (PROOF_ACCOUNT && PROOF_ACCOUNT.address) || "0x0000000000000000000000000000000000000000";
    const hash = rev >= 3
      ? await sendOperatorTx(REGISTRY_ADDRESS, REGISTRY_ABI, "register",
          [endpoint, ENCLAVE_REPO, ENCLAVE_MEASUREMENT, BigInt(SELL_CPU_PRICE6), BigInt(SELL_GPU_PRICE6), proofKey])
      : rev >= 2
      ? await sendOperatorTx(REGISTRY_ADDRESS, REGISTRY_ABI_V2, "register",
          [endpoint, ENCLAVE_REPO, ENCLAVE_MEASUREMENT, BigInt(SELL_CPU_PRICE6), BigInt(SELL_GPU_PRICE6)])
      : await sendOperatorTx(REGISTRY_ADDRESS, REGISTRY_ABI_V1, "register",
          [endpoint, ENCLAVE_REPO, ENCLAVE_MEASUREMENT]);
    _enclaveId = id; _advertisedEndpoint = endpoint;        // unlocks the claim loop (portable deployments)
    console.log(`[registry] registered ${endpoint} repo=${ENCLAVE_REPO} id=${id} tx=${hash}`
      + (rev >= 2 ? ` price=${SELL_CPU_PRICE6}/sec node${IS_GPU ? ` + ${SELL_GPU_PRICE6}/sec card` : ""}` : " (schema 1: unpriced)")
      + (rev >= 3 ? ` proofKey=${proofKey}` : " (pre-schema-3 registry: no proof key published)"));
    // heartbeat loop - refresh liveness so readers don't treat us as down,
    // and keep the published price in step with our configuration (an operator
    // re-pricing by env restart must not have to re-register; a mismatch would
    // also mean we quote one price and charge another)
    setInterval(async () => {
      try {
        const h = await sendOperatorTx(REGISTRY_ADDRESS, REGISTRY_ABI, "heartbeat", [id]);
        console.log(`[registry] heartbeat tx=${h}`);
      } catch (e) { console.warn(`[registry] heartbeat failed: ${e.shortMessage || e.message}`); }
      // an IDLE box gets no requests, so the heartbeat is its only chance to
      // notice the book repointed the registry - without this it heartbeats
      // forever into a contract the ledger no longer reads
      if (registryRepointed()) return void registerFromShimCert().catch(() => {});
      await syncRegisteredPrice(id).catch(() => {});
      await syncRegisteredProofKey(id).catch(() => {});
      await syncDeclaredPayoutWallet(id).catch(() => {});
    }, Math.max(60, HEARTBEAT_SEC) * 1000).unref();
    syncRegisteredPrice(id).catch(() => {});
    syncRegisteredProofKey(id).catch(() => {});
    syncDeclaredPayoutWallet(id).catch(() => {});
  } catch (e) {
    _registered = false; _registeredOn = null;              // let a later request retry
    console.warn(`[registry] self-registration failed: ${e.shortMessage || e.message}`);
  }
}

// Publish this box's configured price if the chain doesn't already carry it.
// One eth_call per heartbeat, a tx only on an actual mismatch — which is the
// operator changing SELL_*_PRICE6, or a registry that predates our entry. On a
// schema-1 registry there is nowhere to put it, so this is a no-op.
const REGISTRY_GET_ABI = [{ type: "function", name: "get", stateMutability: "view",
  inputs: [{ type: "bytes32" }], outputs: [{ type: "tuple", components: [
    { name: "endpoint", type: "string" }, { name: "repo", type: "string" }, { name: "measurement", type: "bytes32" },
    { name: "operator", type: "address" }, { name: "registeredAt", type: "uint64" }, { name: "lastSeen", type: "uint64" },
    { name: "active", type: "bool" }, { name: "cpuPricePerSec6", type: "uint64" }, { name: "gpuPricePerSec6", type: "uint64" },
    { name: "proofKey", type: "address" }, { name: "payoutWallet", type: "address" },
    { name: "caps", type: "uint64" }, { name: "region", type: "string" }] }] }];
// The same shape truncated to each older schema: a registry that predates a
// field decodes SHORT and viem throws on the tuple, so every read must ask for
// exactly what the deployed registry answers or it would spam failures on an
// older contract mid-migration. 13 fields = schema 5 (caps + region), 11 =
// schema 4 (payout wallet), 10 = schema 3 (proof key), 9 = schema 2 (prices
// only). This is the FOURTH copy of the registry entry shape — the other three
// (scripts/enclave-discover.mjs, relay/fleet.mjs, relay/api-relay.js) decode
// getPage and are pinned byte-identical by test/registry-schema.test.mjs; this
// one decodes get() and is pinned by its own assertion there. A copy that lags
// a schema bump is a cutover outage in whichever service lagged.
const REGISTRY_GET_ABI_AT = (n) => [{ type: "function", name: "get", stateMutability: "view",
  inputs: [{ type: "bytes32" }], outputs: [{ type: "tuple", components:
    REGISTRY_GET_ABI[0].outputs[0].components.slice(0, n) }] }];
const REGISTRY_GET_ABI_V4 = REGISTRY_GET_ABI_AT(11);
const REGISTRY_GET_ABI_V3 = REGISTRY_GET_ABI_AT(10);
const REGISTRY_GET_ABI_V2 = REGISTRY_GET_ABI_AT(9);
async function registryEntry(id) {
  const rev = await registryRev();
  return chainClient.readContract({ address: getAddress(REGISTRY_ADDRESS),
    abi: rev >= 5 ? REGISTRY_GET_ABI : rev >= 4 ? REGISTRY_GET_ABI_V4
       : rev >= 3 ? REGISTRY_GET_ABI_V3 : REGISTRY_GET_ABI_V2,
    functionName: "get", args: [id] });
}

async function syncRegisteredPrice(id) {
  if (await registryRev() < 2) return;
  let e;
  try { e = await registryEntry(id); }
  catch { return; }                                         // transient read: try again next heartbeat
  if (Number(e.cpuPricePerSec6) === SELL_CPU_PRICE6 && Number(e.gpuPricePerSec6) === SELL_GPU_PRICE6) return;
  const h = await sendOperatorTx(REGISTRY_ADDRESS, REGISTRY_ABI, "setPrices",
    [id, BigInt(SELL_CPU_PRICE6), BigInt(SELL_GPU_PRICE6)]);
  console.log(`[registry] price published: ${SELL_CPU_PRICE6}/sec node, ${SELL_GPU_PRICE6}/sec card `
    + `(was ${e.cpuPricePerSec6}/${e.gpuPricePerSec6}) tx=${h}`);
}

// Keep the registered proof key equal to the key this CVM actually holds. The
// key is memory-only, so a CVM relaunch mints a new one and the entry MUST
// follow it: a stale entry means every checkpoint we sign is rejected, the
// ledger's meter stops at our last proof, and we serve for free. One eth_call
// per heartbeat, a tx only on an actual mismatch (a relaunch, or a first boot
// against a freshly migrated registry).
//
// It also keeps us HONEST in the direction that matters: whatever we publish
// here is what /v1/attestation serves, so the two can always be compared by
// anyone who cares to.
async function syncRegisteredProofKey(id) {
  if (!PROOF_ACCOUNT) return;
  if (await registryRev() < 3) return;                      // nowhere to put it yet
  let e;
  try { e = await registryEntry(id); }
  catch { return; }                                         // transient read: try again next heartbeat
  const want = getAddress(PROOF_ACCOUNT.address);
  if (e.proofKey && getAddress(e.proofKey) === want) return;
  const h = await sendOperatorTx(REGISTRY_ADDRESS, REGISTRY_ABI, "setProofKey", [id, want]);
  console.log(`[proof] proof key published: ${want} (was ${e.proofKey || "unset"}) tx=${h}`);
}

// ---- the seller's declared payout wallet (registry schema 4) ---------------
// A rev-12 ledger charges NOTHING for a deployment whose owner is this box's
// declared payout wallet: a seller's own app on a seller's own box is free.
// This box cannot declare it. That is the point — setPayoutWallet records
// msg.sender, so only the wallet itself can consent, and no operator can drag a
// stranger's deployment into a free tier their rate cap could not evict. All we
// do here is MIRROR what the chain says, because the claim loop has to price
// exactly the way claim() will, and tell the operator when the two views of
// "their wallet" disagree.
let _declaredPayout = null;         // the on-chain declaration, checksummed (null = none)
let _payoutNagged = null;           // last state we complained about, so this logs on change only
async function syncDeclaredPayoutWallet(id) {
  if (await registryRev() < 4) return;                      // nowhere to declare one yet
  let e;
  try { e = await registryEntry(id); }
  catch { return; }                                         // transient read: try again next heartbeat
  const ZERO = "0x0000000000000000000000000000000000000000";
  const declared = e.payoutWallet && e.payoutWallet !== ZERO ? getAddress(e.payoutWallet) : null;
  if (declared !== _declaredPayout) {
    _declaredPayout = declared;
    console.log(declared
      ? `[registry] payout wallet declared on-chain: ${declared} — its own deployments run free here`
      : `[registry] no payout wallet declared on-chain: every deployment here is charged`);
  }
  // The wallet the supervisor SWEEPS to is config (PAYOUT_ADDRESS); the wallet
  // the ledger prices against is the declaration. They are usually the same
  // address and the seller expects them to be, so say so when they aren't —
  // silently charging a seller to run their own app is exactly the confusion
  // this feature exists to remove.
  if (!PAYOUT_ADDRESS) return;
  const want = getAddress(PAYOUT_ADDRESS);
  const state = declared || "none";
  if (state === _payoutNagged || declared === want) { _payoutNagged = state; return; }
  _payoutNagged = state;
  console.warn(declared
    ? `[registry] earnings sweep to ${want} but the on-chain payout wallet is ${declared}: `
      + `deployments owned by ${want} are CHARGED here. Send EnclaveRegistry.setPayoutWallet(${id}) from ${want} to change that.`
    : `[registry] earnings sweep to ${want}, but no payout wallet is declared on-chain — so this box charges `
      + `even its own owner. One transaction fixes it: EnclaveRegistry.setPayoutWallet(${id}) sent FROM ${want} `
      + `(the operator key cannot send it; that is deliberate).`);
}

// Boot-time hostname discovery: the shim terminates TLS inside this CVM, and
// its certificate names this enclave's public ingress (<name>.containers.
// tinfoil.dev) in the SANs — the same cert whose key the attestation quote
// binds, so it is stronger provenance for our own endpoint than the Host
// header of whoever happens to call first. Reading it over loopback lets a
// fresh enclave advertise without config and without waiting for external
// traffic — which may never come: discovery needs the registry entry, and the
// entry needed traffic (the lazy middleware alone deadlocks on this).
function shimCertHostname(port = 443, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const s = tls.connect({ host: "127.0.0.1", port, rejectUnauthorized: false }, () => {
      try {
        const sans = (s.getPeerX509Certificate()?.subjectAltName || "")
          .split(",").map((e) => e.trim()).filter((e) => e.startsWith("DNS:")).map((e) => e.slice(4));
        // the public ingress name; *.hpke/*.hatt.tinfoil.sh SANs are Tinfoil-internal
        resolve(sans.find((n) => /\.containers\.tinfoil\.dev$/i.test(n)) || null);
      } catch (e) { reject(e); } finally { s.destroy(); }
    });
    s.setTimeout(timeoutMs, () => s.destroy(new Error("shim cert read timeout")));
    s.on("error", reject);
  });
}
// Self-register ONLY our own attested shim-cert SAN — never the request Host.
// Caches the SAN once discovered so originOf() and the lazy trigger reuse it.
async function registerFromShimCert() {
  if (!REGISTRY_READY || (_registered && !registryRepointed())) return;
  const name = _certSan || (_certSan = await shimCertHostname());
  if (name) await registerOnChain("https://" + name);
}
// Register eagerly from the shim cert, retrying with backoff: early in boot the
// shim may present a placeholder cert with no public SAN (ACME still running),
// and a register tx can fail transiently. The lazy per-request path (below) also
// kicks this, and the loop ends the moment either wins (_registered).
async function advertiseFromShimCert() {
  for (let delaySec = 5; REGISTRY_READY && !_registered; delaySec = Math.min(delaySec * 2, 300)) {
    try {
      await registerFromShimCert();
      if (!_registered && !_certSan) console.log("[registry] shim cert has no public SAN yet — retrying");
    } catch (e) { console.warn(`[registry] shim cert read failed (${e.message}) — retrying`); }
    if (!_registered) await new Promise((r) => setTimeout(r, delaySec * 1000).unref());
  }
}
function need(n){ const v = process.env[n]; if(!v){ console.error("FATAL: missing env", n); process.exit(1);} return v; }

// ============================================================================
// in-enclave ACME (RFC 8555) - PURE HALF: crypto/DER/JOSE helpers, no network,
// no state. Browsers reaching <label>.APP_CERT_DOMAIN should terminate TLS
// INSIDE this CVM, not at the relay's Caddy - which means the enclave itself
// must hold a CA-signed cert for each app subdomain. So the supervisor speaks
// ACME directly: ZeroSSL by default (its External Account Binding means one
// EAB credential pair works forever, with no per-boot account approval), the
// dns-01 challenge (the enclave serves no port 80, and the TXT record is
// pushed through the platform DNS daemon), and a hand-built PKCS#10 CSR
// (Node can mint keys but not CSRs; the ~90 lines of DER below are the whole
// gap, same spirit as the hand-rolled ABI/DER encodings elsewhere).
//
// CVMs have no disk. Certs live in memory and, when ACME_STORE_DIR names a
// MEMORY-BACKED tmpfs (the Tinfoil ramdisk that survives container restarts
// within one CVM boot), in that tmpfs too - so a release repoint, which
// restarts every container on the fleet, does not re-issue every name from
// scratch (2026-08-27: ZeroSSL hung for hours, Let's Encrypt's 5-per-name
// weekly duplicate limit was spent by the day's restarts, and kryptos served
// NO certificate for any of its names). A key never touches host-backed disk:
// the store refuses any directory that is not tmpfs (acmeStoreGuard). A full
// CVM relaunch still starts empty, as before.
//
// The runtime half (account, orders, issuance queue, SNI contexts) lives next
// to the TLS bridge below. These helpers sit up here, before ANY boot side
// effect, so ACME_SELFTEST can exercise them and exit: this monolith exports
// nothing, so `ACME_SELFTEST=csr node supervisor.js` IS the test seam
// (test/acme.test.mjs validates the outputs with openssl and jose).
// ----------------------------------------------------------------------------
// Feature is OFF unless at least one CA slot is complete (everything below
// no-ops gracefully when disabled). Issuance speaks to an ORDERED LIST of CAs
// - redundancy added after the 2026-07-18 blackout, when ZeroSSL/Sectigo
// replaced their entire ACME plane with a maintenance page (status pages
// green) and every app cert on the platform stopped renewing. Slot 1 keeps
// the classic env names (ACME_DIRECTORY, defaulting to ZeroSSL, plus its EAB
// pair); slots 2/3 are ACME_DIRECTORY_2 / ACME_EAB_KID_2 / ACME_EAB_HMAC_2
// (and _3). A fallback slot opts in by naming its directory; its EAB pair is
// optional AS A PAIR (Let's Encrypt needs none), but half a pair is a config
// error and skips the slot. acmeIssue walks the list in order, cooling off
// slots that fail at the infrastructure level.
//
// SLOT 0, ahead of every in-enclave CA, is the PLATFORM CERTIFICATE SERVICE
// (CERTS_API, e.g. https://api.enclave.host - the relay-owned
// POST /v1/certs/issue). It serves ONLY names in our own zones:
// <label>.APP_CERT_DOMAIN, and <label>.TCP_CERT_DOMAIN when that zone is
// configured. For those the enclave mints the key and the PKCS#10 CSR exactly
// as it does for a CA (buildCsr below), but POSTs the CSR to the relay instead
// of running the ACME dance itself: the relay holds the CA accounts (ZeroSSL
// with the platform EAB pair first, Let's Encrypt behind it), answers dns-01,
// paces the rate limits that every seller shares, and hands back the chain.
// The private key never leaves this CVM - the service sees a CSR and returns
// a certificate, nothing else (relay/certs.js is the other half). Why: until
// now the ZeroSSL EAB pair reached EVERY box as a fleet secret; the service
// takes it off the boxes, and a single place that sees every order is the
// only place that can pace Let's Encrypt's 50-per-registered-domain week.
//
// Slot order: platform service (own zones only) -> ACME_DIRECTORY (ZeroSSL,
// EAB) -> ACME_DIRECTORY_2 -> ACME_DIRECTORY_3. A customer's domain is not in
// our zones, so it never touches slot 0 and runs the in-enclave CAs unchanged.
// A service REFUSAL (4xx) is name-level: the in-enclave slots run as if the
// service did not exist, so nothing regresses while it rolls out. A 5xx /
// network error / non-JSON reply cools the service slot off like a CA. A 202
// (order in flight, or the CAs are cooling off behind the service) is
// neither: the name waits retryAfterSec and asks again with the SAME key, and
// no in-enclave CA is spent on it meanwhile. "Not registered yet" (boot, the
// registry tx still unconfirmed) is a DEFERRAL too, not a refusal: the pump
// waits 15-30 s and the in-enclave CAs are not walked on every held name at
// every restart. The key minted for a name stays in _platformPending across
// a timeout / 5xx as well as a 202, so whatever the service is still doing
// for that (name, SPKI) is what the retry finds.
//
// Who signs what (relay/certs.js is the other half; both bind the KEY):
//   opSig = personal_sign("enclave-certs-issue:<name>:<endpoint>:<spkiHash>:<ts>")
//           by this box's registry operator -- REQUIRED, the authorization
//   sig   = HMAC-SHA256(hex-decode(CERTS_KEY), "<name>:<endpoint>:<spkiHash>:<ts>")
//           -- sent ONLY by a box holding the real fleet SECRET (below); a sig
//           that is sent must verify, so a box that can't must send none.
const APP_CERT_DOMAIN = (process.env.APP_CERT_DOMAIN || "").trim().replace(/^\*?\./, "").replace(/\.$/, "").toLowerCase(); // e.g. "app.enclave.host"
const TCP_CERT_DOMAIN = (process.env.TCP_CERT_DOMAIN || "").trim().replace(/^\*?\./, "").replace(/\.$/, "").toLowerCase(); // optional second platform zone
const DNS_API         = (process.env.DNS_API || "").trim().replace(/\/+$/, "");  // platform DNS daemon's TXT push API
const CERTS_API       = (process.env.CERTS_API || "").trim().replace(/\/+$/, ""); // platform certificate service (relay); "" = slot absent
// The service's fleet factor: HMAC(SECRET, "enclave certs v1"), the same
// derived-key pattern as DNS_TXT_KEY and SECRETS_FETCH_KEY - the relay env
// holds only this hex (CERTS_KEY=), never the fleet SECRET. The relay keys
// its HMAC with the DECODED 32 bytes (relay/certs.js issueSig: Buffer.from(
// keyHex, "hex"), the secrets.js fetchSig convention), and so does certsSig
// below; the hex string's own bytes would sign a different tuple and every
// request would be 401 bad_sig.
// Only a box whose SECRET IS the fleet secret can derive a CERTS_KEY the
// relay knows. On metal, gsup.mjs mints SECRET = FLEET_SECRET || random and
// says which with FLEET_SECRET_PRESENT (0 = a per-boot random secret: a
// seller box, or a first-party metal box without cfg.fleetSecret); the Tinfoil
// image passes the fleet SECRET straight through and sets no flag, so unset
// means present. The relay refuses a sig that fails (fail closed, never
// silently ignored), so a box without the fleet secret must SEND NONE and
// let its operator signature + the lease authorize it: CERTS_KEY is "" here
// and the request goes out opSig-only.
const FLEET_SECRET_PRESENT = SECRET.length > 0 && !/^(0|false|off)$/i.test(process.env.FLEET_SECRET_PRESENT || "1");
const CERTS_KEY       = FLEET_SECRET_PRESENT ? createHmac("sha256", SECRET).update("enclave certs v1").digest("hex") : "";
// Slot 0 itself: shaped like a CA slot (host + downUntil) so the walker treats
// it as one; .platform routes issuance to the service client.
const ACME_PLATFORM   = CERTS_API ? { host: "platform", api: CERTS_API, platform: true, downUntil: 0 } : null;
// Account contact, sent on every newAccount when set. Google Trust Services
// REJECTS contactless registrations ("Accounts must have at least one
// contact", found 2026-07-19 by scripts/acme-eab-check.mjs); ZeroSSL and
// Let's Encrypt accept-and-ignore a contact, so sending it everywhere is
// safe. A bare address gets the mailto: prefix; no CA email-verifies it.
const _contactRaw  = (process.env.ACME_CONTACT || "").trim();
const ACME_CONTACT = _contactRaw && (_contactRaw.includes(":") ? _contactRaw : `mailto:${_contactRaw}`);
const ACME_CAS = [];   // ordered CA slots: { directory, host, eabKid, eabHmac, dir, account, nonce, downUntil }
for (const suf of ["", "_2", "_3"]) {
  const directory = (process.env[`ACME_DIRECTORY${suf}`] || (suf ? "" : "https://acme.zerossl.com/v2/DV90")).trim().replace(/\/+$/, "");
  const eabKid    = (process.env[`ACME_EAB_KID${suf}`]  || "").trim();
  const eabHmac   = (process.env[`ACME_EAB_HMAC${suf}`] || "").trim();
  if (!!eabKid !== !!eabHmac) { console.warn(`[acme] ACME_EAB_KID${suf}/ACME_EAB_HMAC${suf}: half an EAB pair - slot skipped`); continue; }
  if (suf ? !directory : !eabKid) {                           // slot 1's default directory alone is not an opt-in
    if (suf && eabKid) console.warn(`[acme] ACME_EAB_*${suf} set but ACME_DIRECTORY${suf} missing - slot skipped`);
    continue;
  }
  let host; try { host = new URL(directory).host; } catch { console.warn(`[acme] ACME_DIRECTORY${suf}: bad URL - slot skipped`); continue; }
  ACME_CAS.push({ directory, host, eabKid, eabHmac, dir: null, account: null, nonce: null, downUntil: 0 });
}
// On with the in-enclave CAs (a slot + the DNS daemon) OR the platform service
// alone: a box that holds no EAB pair and no DNS_TXT_KEY - the end state once
// the service is everywhere - still certifies its app-zone names through it.
const ACME_ENABLED = !!(APP_CERT_DOMAIN && ((ACME_CAS.length && DNS_API) || ACME_PLATFORM));
// Is this a name the platform service may issue for? Exactly one label under
// one of OUR zones - the relay refuses anything else (an apex, a deeper
// name, a stranger's hostname) with a 403 that never reaches a CA, so asking
// would only spend a round trip. A VERIFIED custom domain is the other kind
// of name the service issues for (acmeSlotsFor, at runtime: the relay holds
// it to its domain record and to the lease) - not known in this pure half,
// so the self-test seam sees only the zone rule.
const CERT_ZONES = [APP_CERT_DOMAIN, TCP_CERT_DOMAIN].filter(Boolean);
function platformCertName(name) {
  const n = String(name || "").toLowerCase();
  return CERT_ZONES.some((z) => n.endsWith(`.${z}`) && /^[a-z0-9-]+$/.test(n.slice(0, -(z.length + 1))));
}

// Which context an in-enclave TLS termination point serves for a client's SNI
// (pure - the runtime sniSelect below the TLS bridge feeds it live contexts;
// ACME_SELFTEST=sni drives it with stand-ins). App-zone names are
// CA-or-nothing: <label>.APP_CERT_DOMAIN promises validating clients a real CA
// cert, and the self-signed bridge pair is name-valid for it (same wildcard
// CN) — serving that placeholder while issuance is pending would invite
// click-through and -k/noverify sessions that no client can authenticate and
// any on-path box could equally terminate, so sensitive app traffic could
// leak. Those names get their CA cert or a refused handshake, never a
// stand-in. Everything else (no SNI, the bare domain, the legacy tcp zone)
// keeps the self-signed bridge pair: that pair is verified by fingerprint over
// the attested origin (/v1/tls-bridge), not by name, and refusing it would
// break the pin flow.
// A CUSTOMER's own domain (`managed`) follows the same rule as an app-zone
// name, and for a sharper reason: the bridge pair is a wildcard for OUR zone,
// so on their hostname it is not merely unauthenticatable but plainly invalid.
// Serving it would put a certificate error on the customer's brand and teach
// their users to click through one. No held cert, no handshake.
function sniDecide(servername, heldCtx, bridgeCtx, managed = false) {
  if (heldCtx) return { use: "acme", ctx: heldCtx };
  if (managed) return { use: "refuse" };
  if (APP_CERT_DOMAIN && String(servername || "").toLowerCase().endsWith(`.${APP_CERT_DOMAIN}`)) return { use: "refuse" };
  return { use: "bridge", ctx: bridgeCtx || undefined };
}

const statSyncMode = (p) => { try { return (statSync(p).mode & 0o777).toString(8); } catch { return null; } };   // self-test seam only
// base64url without padding - the encoding EVERYTHING in JOSE/ACME speaks.
const b64u     = (b) => Buffer.from(b).toString("base64url");
const b64uJson = (o) => b64u(JSON.stringify(o));

// RFC 7638 JWK thumbprint: sha256 over the canonical JSON of the REQUIRED
// members only, keys in lexicographic order - for an EC key that is exactly
// {"crv","kty","x","y"}, no whitespace, nothing else. Building the string by
// hand (not JSON.stringify of the object) is the point: member order in the
// source object must not matter. (Cross-checked against jose in the tests.)
function jwkThumbprint(jwk) {
  return b64u(createHash("sha256").update(`{"crv":"${jwk.crv}","kty":"${jwk.kty}","x":"${jwk.x}","y":"${jwk.y}"}`).digest());
}

// dns-01 proof: keyAuthorization = token "." thumbprint; the TXT value the CA
// looks for is base64url(sha256(keyAuthorization)) (RFC 8555 §8.4).
const dns01TxtValue = (token, thumbprint) => b64u(createHash("sha256").update(`${token}.${thumbprint}`).digest());

// One flat-format JWS, ES256 (all ACME envelope signatures). The signature is
// raw R||S (ieee-p1363), NOT the DER that ECDSA usually emits - JOSE's one
// deviation. payload === null -> "" (POST-as-GET, RFC 8555 §6.3).
function jwsSignEs256(protectedHeader, payload, privateKey) {
  const prot = b64uJson(protectedHeader);
  const body = payload === null ? "" : b64uJson(payload);
  const sig  = cryptoSign("sha256", Buffer.from(`${prot}.${body}`), { key: privateKey, dsaEncoding: "ieee-p1363" });
  return { protected: prot, payload: body, signature: b64u(sig) };
}

// External Account Binding (RFC 8555 §7.3.4): an INNER JWS proving we hold the
// CA-issued EAB credential. HS256 with the base64url-DECODED HMAC key; the
// payload is our ACME account's public JWK; url = the newAccount URL. It rides
// inside the newAccount payload, not the envelope.
function eabJws(kid, hmacB64u, accountJwk, newAccountUrl) {
  const prot    = b64uJson({ alg: "HS256", kid, url: newAccountUrl });
  const payload = b64uJson(accountJwk);
  const sig     = createHmac("sha256", Buffer.from(hmacB64u, "base64url")).update(`${prot}.${payload}`).digest();
  return { protected: prot, payload, signature: b64u(sig) };
}

// ---- minimal DER writer + PKCS#10 CSR builder ------------------------------
// Just enough ASN.1 to emit one CSR: TLV with long-form lengths, OIDs, and the
// handful of universal types a CertificationRequest touches. Everything is a
// Buffer in, Buffer out; structures compose by concatenation.
function derLen(n) {
  if (n < 0x80) return Buffer.from([n]);
  const b = []; for (let x = n; x > 0; x >>>= 8) b.unshift(x & 0xff);
  return Buffer.from([0x80 | b.length, ...b]);
}
const derTlv   = (tag, ...body) => { const b = Buffer.concat(body); return Buffer.concat([Buffer.from([tag]), derLen(b.length), b]); };
const derSeq   = (...p) => derTlv(0x30, ...p);
const derSet   = (...p) => derTlv(0x31, ...p);
const derInt0  = ()     => derTlv(0x02, Buffer.from([0]));            // INTEGER 0 (the only integer a CSR needs: version)
const derOctet = (b)    => derTlv(0x04, b);
const derBits  = (b)    => derTlv(0x03, Buffer.from([0]), b);          // BIT STRING, 0 unused bits
const derUtf8  = (s)    => derTlv(0x0c, Buffer.from(s, "utf8"));
const derCtx   = (n, constructed, ...body) => derTlv((constructed ? 0xa0 : 0x80) | n, ...body);
function derOid(oid) {
  const a = oid.split(".").map(Number), body = [40 * a[0] + a[1]];
  for (const v of a.slice(2)) {
    const enc = [v & 0x7f];
    for (let x = Math.floor(v / 128); x > 0; x = Math.floor(x / 128)) enc.unshift((x & 0x7f) | 0x80);
    body.push(...enc);
  }
  return derTlv(0x06, Buffer.from(body));
}
const pemWrap = (label, der) =>
  `-----BEGIN ${label}-----\n${der.toString("base64").match(/.{1,64}/g).join("\n")}\n-----END ${label}-----\n`;

// Build a CSR for ONE dns name: fresh P-256 pair, subject CN=name (cosmetic -
// CAs read the SAN), and an extensionRequest attribute carrying subjectAltName
// with that single dNSName. Signed ecdsa-with-SHA256; crypto.sign with
// dsaEncoding "der" already emits the DER ECDSA-Sig-Value the BIT STRING wants.
function buildCsr(name) {
  const { publicKey, privateKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
  const spki = publicKey.export({ type: "spki", format: "der" });        // already a full DER SubjectPublicKeyInfo
  const san  = derSeq(derCtx(2, false, Buffer.from(name, "ascii")));     // GeneralNames: [2] dNSName (context-primitive IA5 bytes)
  const ext  = derSeq(derOid("2.5.29.17"), derOctet(san));               // Extension: id-ce-subjectAltName, extnValue OCTET STRING
  const attr = derSeq(derOid("1.2.840.113549.1.9.14"), derSet(derSeq(ext))); // pkcs-9-at-extensionRequest { SET { Extensions } }
  const cri  = derSeq(                                                   // CertificationRequestInfo
    derInt0(),                                                           //   version 0
    derSeq(derSet(derSeq(derOid("2.5.4.3"), derUtf8(name)))),            //   subject: CN=name
    spki,                                                                //   subjectPKInfo
    derCtx(0, true, attr));                                              //   attributes [0] IMPLICIT SET OF Attribute
  const sig  = cryptoSign("sha256", cri, { key: privateKey, dsaEncoding: "der" });
  const csr  = derSeq(cri, derSeq(derOid("1.2.840.10045.4.3.2")), derBits(sig)); // + ecdsa-with-SHA256 (params absent per RFC 5758)
  return { csrDer: csr, csrPem: pemWrap("CERTIFICATE REQUEST", csr),
           keyPem: privateKey.export({ type: "pkcs8", format: "pem" }) };
}

// ---- issuance slot walking + the platform service client (pure: every
//      network and signing dependency is injected, so ACME_SELFTEST=platform
//      can drive the real client against a mock service and stub CAs) --------
// Errors that indict the SLOT rather than the name carry .caLevel: the walker
// cools that slot off and fails over. A .deferMs error is the platform's 202:
// neither the slot nor the name is at fault, the answer is "ask again later".
const caErr    = (msg) => Object.assign(new Error(msg), { caLevel: true });
const deferErr = (msg, deferMs) => Object.assign(new Error(msg), { deferMs });
// EVERY issuance network call is bounded. node's fetch has no default timeout,
// and the whole failover design reads an ERROR to cool a slot off and try the
// next one — so a CA that accepts the connection and never answers (an
// overloaded LB, a black-holing middlebox: the shape the 2026-07-18 outage
// took hours before the endpoint died outright) would hang the issuance
// forever and the failover would never fire. acmePoll's own 90s deadline
// can't save it either: that deadline is only checked between posts. An
// AbortError arrives at the existing catch sites, which already mark it
// caLevel — a timeout indicts the CA, exactly like a refused connection.
// (dohQuery in this same file has always been bounded; ACME was the sibling
// that wasn't.)
// 45 s, not 20: ZeroSSL answers in 0.2 s or in 25 s (2026-08-27, hours at
// a time); at 20 s every call timed out, the slot cooled off and the walk
// fell to Let's Encrypt, whose duplicate limit the day's restarts had spent.
const ACME_HTTP_MS = parseInt(process.env.ACME_HTTP_TIMEOUT_MS || "45000", 10);
const acmeFetch = (url, init = {}, ms = ACME_HTTP_MS) =>
  fetch(url, { ...init, signal: AbortSignal.timeout(ms) });
// The platform service's POST gets its OWN bound, longer than a CA call's: the
// relay holds the request open for up to CERTS_SYNC_WAIT_MS (relay/certs.js,
// default 8000 ms) while the order runs, then answers 202 retryAfterSec if it
// is still going. COUPLING: CERTS_HTTP_MS must exceed the relay's sync wait
// by a comfortable margin (30 s vs 8 s), or every real order is aborted here
// as a caErr, the slot cools, and the in-enclave CAs double-issue the name
// while the relay's order completes for a key nobody keeps. Change one, look
// at the other.
const CERTS_HTTP_MS = parseInt(process.env.CERTS_HTTP_TIMEOUT_MS || "30000", 10);

// The platform service's request signature, two factors (the secrets-fetch
// rule, relay/secrets.js says why): the fleet HMAC proves "a holder of the
// fleet key", the operator signature proves "THIS endpoint" - without it any
// fleet member could name another box's endpoint and be issued a certificate
// for a deployment that box holds the lease on. personal_sign (EIP-191), so
// the operator key's signature can never be replayed as a transaction.
// BOTH bind the KEY: spkiHash = sha256 hex of the CSR key's DER
// SubjectPublicKeyInfo. The relay recomputes it from the CSR it parsed, so a
// captured request cannot be replayed with somebody else's CSR (the
// signatures are single-use on the relay as well; this closes the other half).
//   sig   = HMAC-SHA256(hex-decode(CERTS_KEY), "<name>:<endpoint>:<spkiHash>:<ts>") hex
//   opSig = personal_sign("enclave-certs-issue:<name>:<endpoint>:<spkiHash>:<ts>")
// Pinned byte-for-byte against relay/certs.js issueSig / issueMessage by
// test/acme.test.mjs.
const certsSigTuple  = (name, endpoint, spkiHash, ts) => `${name}:${endpoint}:${spkiHash}:${ts}`;
const certsOpSigText = (name, endpoint, spkiHash, ts) => `enclave-certs-issue:${name}:${endpoint}:${spkiHash}:${ts}`;
const certsSig = (name, endpoint, spkiHash, ts, key = CERTS_KEY) =>
  createHmac("sha256", Buffer.from(key, "hex")).update(certsSigTuple(name, endpoint, spkiHash, ts)).digest("hex");
const spkiHashOf = (keyPem) =>
  createHash("sha256").update(createPublicKey(keyPem).export({ type: "spki", format: "der" })).digest("hex");
// A key minted for an order is KEPT and re-presented on the retry -- after a
// 202, and equally after a timeout or a 5xx, since the relay may well be
// finishing that order regardless of how our side of the connection ended:
// the service caches by (name, SPKI) and resumes the in-flight order for the
// CSR it holds, so a fresh key per attempt would miss the cache and spend a
// second issuance on the same name. Stored BEFORE the POST for that reason.
// Bounded: an hour, then re-mint. A refusal (4xx) or an install drops it.
// Across a container RESTART the same cache does the work: an installed cert
// is persisted with its key (ACME_STORE_DIR), so the restored name is never
// re-asked at all - and if it is (past renewAt), the relay's (name, SPKI)
// cache still cannot hit, since a renewal mints a fresh key; that is
// intended, a renewal is a new issuance. Nothing else changes here.
const PLATFORM_PENDING_MS = 3600_000;
// How long a box waits when it is asked for a certificate before its registry
// entry (and so its endpoint) exists: a deferral, not a failure, and short --
// registration lands 10-60 s after boot.
const PLATFORM_UNREGISTERED_DEFER_MS = () => 15_000 + Math.floor(Math.random() * 15_000);
// ...but BOUNDED: a box whose registration never lands (operator key out of
// gas, registry unreachable) must still get certificates from its own CAs,
// so after this many deferrals per name the platform slot fails name-level
// and the walk continues. Reset once the endpoint is known.
const PLATFORM_UNREGISTERED_MAX_DEFERS = 4;
const _unregisteredDefers = new Map();             // name -> deferrals so far
// Ask the platform service for a certificate for `name`: mint the key + CSR
// in-CVM, POST the CSR with both signatures, install the returned chain.
//   200 -> the issued record (keyPem ours, certPem theirs)
//   202 -> .deferMs error (retry at retryAfterSec, same key)
//   4xx -> plain error: name-level, the in-enclave slots take over
//   5xx / network / non-JSON -> .caLevel error: the slot cools off
// deps: endpoint (our registered origin), keyHex (CERTS_KEY), signOp (async
// text -> EIP-191 signature, or null when this box has no operator key),
// pending (Map name -> { csrPem, keyPem, at }).
async function acmeIssueViaPlatform(slot, name, { endpoint, keyHex = CERTS_KEY, signOp = null, pending = new Map() }) {
  // Not registered yet (boot: the register tx is in the operator queue, the
  // endpoint is adopted when it confirms) is nobody's fault and over in
  // seconds: defer, so the in-enclave CAs are not walked for every held name
  // on every restart. No key is minted for it either.
  if (!endpoint) {
    const n = (_unregisteredDefers.get(name) || 0) + 1;
    _unregisteredDefers.set(name, n);
    if (n <= PLATFORM_UNREGISTERED_MAX_DEFERS) throw deferErr(`platform: this enclave has not registered its endpoint yet`, PLATFORM_UNREGISTERED_DEFER_MS());
    throw new Error(`platform: endpoint still unregistered after ${n - 1} deferrals; the in-enclave CAs take this round`);
  }
  _unregisteredDefers.delete(name);
  // The operator signature is the authorization (the relay checks it against
  // the registry entry and the live lease); the fleet HMAC is an extra factor
  // only a box holding the fleet SECRET can add (CERTS_KEY is "" otherwise).
  // A box with neither has nothing the service would accept: say so before
  // the round trip.
  if (!signOp && !keyHex) throw new Error(`platform: this box has neither an operator key nor the fleet secret to sign with`);
  let held = pending.get(name);
  if (!held || Date.now() - held.at > PLATFORM_PENDING_MS) {
    const { csrPem, keyPem } = buildCsr(name);
    held = { csrPem, keyPem, spkiHash: spkiHashOf(keyPem), at: Date.now() };
  }
  const ts  = Math.floor(Date.now() / 1000);
  const sig = keyHex ? certsSig(name, endpoint, held.spkiHash, ts, keyHex) : "";
  let opSig = "";
  if (signOp) {
    try { opSig = await signOp(certsOpSigText(name, endpoint, held.spkiHash, ts)); }
    catch (e) { console.warn(`[acme] platform: operator co-sign failed (${e.message})${sig ? "; HMAC only" : ""}`); }
  }
  if (!opSig && !sig) throw new Error(`platform: could not sign the request for ${name}`);
  // The key is on record BEFORE the POST: a timeout or a 5xx on our side says
  // nothing about whether the relay's order is running, and the next ask has
  // to present the same SPKI to find it (see PLATFORM_PENDING_MS).
  pending.set(name, held);
  let r;
  try {
    r = await acmeFetch(`${slot.api}/v1/certs/issue`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, csr: held.csrPem, endpoint, ts, ...(sig ? { sig } : {}), ...(opSig ? { opSig } : {}) }) }, CERTS_HTTP_MS);
  } catch (e) { throw caErr(`platform: ${e.message}`); }                     // pending kept: same key next time
  const isJson = /json/.test(r.headers.get("content-type") || "");
  const data = isJson ? await r.json().catch(() => null) : await r.text();
  if (r.status === 202) {
    const sec = Math.max(5, Math.min(3600, Number(data?.retryAfterSec) || 60));
    throw deferErr(`platform: order for ${name} in flight, retry in ${sec}s`, sec * 1000);
  }
  if (r.status >= 500 || !isJson || !data)
    throw caErr(`platform: HTTP ${r.status} ${isJson ? String(data?.error || "") : String(data).slice(0, 120)}`.trim());
  if (r.status >= 400) {
    pending.delete(name);                                     // a refusal ends this order; a retry starts fresh
    throw new Error(`platform refused ${name}: ${r.status} ${data.error || "?"} ${data.message || ""}`.trim());
  }
  // The reply is trusted only as far as it checks out: the leaf must be for
  // OUR key and OUR name, or the service (or whatever answered as it) is
  // broken and cools off; the in-enclave CAs then issue instead.
  const certPem = String(data.certPem || "");
  let leaf;
  try { leaf = new X509Certificate(certPem); } catch (e) { throw caErr(`platform: reply carries no certificate (${e.message})`); }
  if (!leaf.checkPrivateKey(createPrivateKey(held.keyPem))) throw caErr(`platform: certificate for ${name} is not for our key`);
  if (!leaf.checkHost(name)) throw caErr(`platform: certificate is not for ${name} (${leaf.subject.replace(/\n/g, " ")})`);
  pending.delete(name);
  const nb = new Date(leaf.validFrom).getTime(), na = new Date(leaf.validTo).getTime();
  return { keyPem: held.keyPem, certPem, expiresAt: na,
           renewAt: nb + Math.round((na - nb) * 2 / 3),         // renew past 2/3 of lifetime (same rule as acmeIssueVia)
           ctx: tls.createSecureContext({ key: held.keyPem, cert: certPem }),
           issuer: `platform (${String(data.ca || "?")})`, cached: !!data.cached };
}
// Issue via the first slot that can: walk `slots` in order. A slot that fails
// at the infrastructure level (.caLevel: directory/nonce/account trouble,
// 5xx, network errors, HTML where problem+json belongs, validation that never
// completes) cools off for cooldownMs so the strictly-serial pump doesn't
// burn its 90s timeouts on that slot for every queued name; any success
// clears the latch. Name-level rejections (authz became invalid, the platform
// refused) also move on to the next slot - a second CA may validate what the
// first refused - but indict only the name. Every slot cooling off = try them
// all anyway: a stale latch must never be the reason issuance stops entirely.
// A deferral (.deferMs, the platform's 202) ends the round at once: the order
// is in flight behind the service, and spending an in-enclave CA on the same
// name meanwhile would double-issue it.
// A slot the CA has RATE-LIMITED for this name (rateLimitedUntil(ca) > now,
// the per-name per-CA memory above) is not walked at all - unlike a cool-off,
// a rate limit is a stated date, and asking earlier only spends the CA's
// goodwill; every slot rate-limited = fail the round with the earliest date
// (.allRateLimited), which the retry plan honours as the name's next attempt.
async function acmeWalkSlots(name, { slots, issueVia, cooldownMs, rateLimitedUntil = () => 0, onRateLimited = null }) {
  if (!slots.length) throw new Error(`no issuance slot can serve ${name} (the platform service covers our own zones only)`);
  const now = Date.now();
  const allLimited = (t) => { const untils = slots.map((ca) => rateLimitedUntil(ca)).filter((u) => u > t); return untils.length === slots.length ? Math.min(...untils) : 0; };
  const limitedUntil = allLimited(now);
  if (limitedUntil) throw Object.assign(rateLimitErr(`every slot is rate-limited for ${name} until ${new Date(limitedUntil).toISOString()}`, limitedUntil), { allRateLimited: true });
  const usable = slots.filter((ca) => !(rateLimitedUntil(ca) > now));
  const live = usable.filter((ca) => !(ca.downUntil > now));
  let lastErr = null;
  const cooledNow = [];                      // slots this round put on cool-off
  let networkProven = false;                 // a later slot got far enough to refuse the NAME
  for (const ca of (live.length ? live : usable)) {
    try {
      const issued = await issueVia(ca, name);
      ca.downUntil = 0;
      return { ...issued, caHost: ca.host, issuer: issued.issuer || ca.host };
    } catch (e) {
      lastErr = e;
      if (e.deferMs) throw e;
      if (e.rateLimitedUntil && onRateLimited) onRateLimited(ca, e.rateLimitedUntil);
      if (e.caLevel) {
        ca.downUntil = Date.now() + cooldownMs;
        cooledNow.push(ca);
        console.warn(`[acme] ${ca.host}: ${e.message} - cooling this ${ca.platform ? "slot" : "CA"} off ${Math.round(cooldownMs / 60_000)}m`);
      } else {
        networkProven = true;
      }
    }
  }
  // Second chance. A slot that timed out THIS round while a later slot reached
  // its account and order endpoints fine (and then refused only the name -- a
  // rate limit, say) most likely hit a transient: egress not yet up at boot,
  // a slow nonce. One immediate retry of that slot is cheap and is the
  // difference between a certificate now and one after the cool-off.
  if (networkProven && cooledNow.length) {
    for (const ca of cooledNow) {
      try {
        const issued = await issueVia(ca, name);
        ca.downUntil = 0;
        console.log(`[acme] ${ca.host}: second chance succeeded for ${name}`);
        return { ...issued, caHost: ca.host, issuer: issued.issuer || ca.host };
      } catch (e) {
        lastErr = e;
        if (e.deferMs) throw e;
        if (e.caLevel) ca.downUntil = Date.now() + cooldownMs;
      }
    }
  }
  // the round ended on a rate limit and every slot now carries one: the
  // name's next attempt is the EARLIEST of them, not the last CA's
  const nowLimited = lastErr?.rateLimitedUntil ? allLimited(Date.now()) : 0;
  if (nowLimited) { lastErr.rateLimitedUntil = nowLimited; lastErr.allRateLimited = true; }
  throw lastErr;
}
// What a failed round means for the NAME's next attempt (the pump's policy,
// pure so the seam can pin it). Three cases:
//  - deferred (platform 202): retry exactly when the service said, and the
//    failure count does NOT move - nobody refused the name;
//  - a slot is cooling off: retry the moment it is back, not a per-name
//    backoff on top of the next 10-minute tick (2026-08-26: ZeroSSL timed
//    out, Let's Encrypt was at its weekly limit for the name, and a running
//    app served no certificate for 17 minutes -- 10 of them waiting on the
//    name's own backoff after both CAs were already usable again);
//  - name-level rejection by every slot: the doubling backoff (5 min, capped
//    at 1h) - that one is the name's fault.
//  - rate-limited by EVERY slot that could serve the name (.allRateLimited):
//    retry when the earliest CA said to, and the failure count does not move
//    either - the CA named the date, a doubling backoff on top is just wrong.
//    (One CA rate-limited and another refusing the name is the ordinary
//    backoff: the limited CA is simply not walked until its date.)
function acmeRetryPlan(e, prevFailures, coolingUntil, now = Date.now()) {
  if (e?.deferMs) return { failures: prevFailures, nextAt: now + e.deferMs, why: "deferred" };
  if (e?.allRateLimited && e.rateLimitedUntil > now)
    return { failures: prevFailures, nextAt: Math.min(e.rateLimitedUntil, now + ACME_RATE_LIMIT_CAP_MS), why: "ratelimited" };
  const failures = prevFailures + 1;
  const cooling = coolingUntil.filter((t) => t > now);
  const backoff = cooling.length ? Math.max(1000, Math.min(...cooling) - now + 1000)
                                 : Math.min(3600_000, 300_000 * 2 ** (failures - 1));
  return { failures, nextAt: now + backoff, why: cooling.length ? "cooling" : "backoff" };
}

// ---- the certificate store (ACME_STORE_DIR) --------------------------------
// What survives a container restart: every issued { key, cert } as
// <sha256(name)>.json, the ACME account per CA (accounts.json: a restart that
// registers afresh each time also runs into Let's Encrypt's 10 new
// registrations per IP per 3 h), and the per-name per-CA rate-limit
// timestamps (ratelimits.json: a CA that said "retry after <date>" must not
// be asked again before then just because we forgot). The rule is the
// session key's (SESSION_KEY_DIR) and the TLS bridge's (TLS_BRIDGE_DIR):
// MEMORY-BACKED ONLY, never host disk. Enforced, not assumed: acmeStoreGuard
// accepts only a directory whose filesystem statfs reports as tmpfs (f_type
// 0x01021994) -- a PATH tells nothing: inside the Tinfoil supervisor
// container /mnt/ramdisk is the container's own overlay, not the CVM's
// ramdisk; the ramdisk reaches the container only as the /var/lib/enclave
// bind (enclaves/*/tinfoil-config.yml `volumes:`), and a bind of a tmpfs
// statfs's as tmpfs. Anything else is refused - logged once, certs stay
// memory-only - unless
// ACME_STORE_ALLOW_DISK=1 says so explicitly (metal/dev: the metal guest is
// initramfs-only, so "disk" there is RAM anyway). Empty ACME_STORE_DIR
// disables the store. Files are 0600 in a 0700 dir, written tmp+rename.
const ACME_STORE_DIR = process.env.ACME_STORE_DIR === undefined ? "/var/lib/enclave/acme" : process.env.ACME_STORE_DIR.trim();
const ACME_STORE_ALLOW_DISK = /^(1|true|yes)$/i.test(process.env.ACME_STORE_ALLOW_DISK || "");
const TMPFS_MAGIC = 0x01021994;                     // linux/magic.h TMPFS_MAGIC
const ACME_RATE_LIMIT_CAP_MS = 7 * 86400e3;         // no CA limit we meet lasts longer than a week
function acmeStoreGuard(dir, { allowDisk = false, statfs = (d) => statfsSync(d).type } = {}) {
  if (!dir) return { ok: false, why: "disabled" };
  const norm = String(dir).replace(/\/+$/, "") || "/";
  if (allowDisk) return { ok: true, why: "ACME_STORE_ALLOW_DISK=1" };
  let type;
  try { type = statfs(norm); }
  catch { try { type = statfs(dirname(norm)); } catch (e) { return { ok: false, why: `cannot statfs ${norm}: ${e.message}` }; } }
  if (Number(type) === TMPFS_MAGIC) return { ok: true, why: "tmpfs" };
  return { ok: false, why: `filesystem type 0x${Number(type).toString(16)} is not tmpfs (0x${TMPFS_MAGIC.toString(16)})` };
}
// Open the store, or null (refused / disabled / unusable), logging why once.
// Every method swallows its own I/O errors: the store is a cache, and a
// failure to persist must never fail an issuance.
function acmeStoreOpen(dir, { allowDisk = ACME_STORE_ALLOW_DISK, statfs, log = console } = {}) {
  const g = acmeStoreGuard(dir, { allowDisk, ...(statfs ? { statfs } : {}) });
  if (!g.ok) {
    if (g.why === "disabled") log.log("[acme] ACME_STORE_DIR is empty - certificates live in memory only, re-issued on every restart");
    else log.warn(`[acme] REFUSING ACME_STORE_DIR=${dir}: ${g.why}. Private keys never touch host-backed disk; point it at a tmpfs (the Tinfoil ramdisk) or set ACME_STORE_ALLOW_DISK=1 if this is a metal/dev box whose "disk" is RAM. Certificates stay memory-only this boot.`);
    return null;
  }
  try { mkdirSync(dir, { recursive: true, mode: 0o700 }); chmodSync(dir, 0o700); }
  catch (e) { log.warn(`[acme] ACME_STORE_DIR=${dir} unusable (${e.message}) - certificates stay memory-only this boot`); return null; }
  const certFile   = (name) => join(dir, `${createHash("sha256").update(String(name).toLowerCase()).digest("hex")}.json`);
  const isCertFile = (f) => /^[0-9a-f]{64}\.json$/.test(f);
  const ACCOUNTS = join(dir, "accounts.json"), RATELIMITS = join(dir, "ratelimits.json");
  const readJson = (file) => { try { return JSON.parse(readFileSync(file, "utf8")); } catch { return null; } };
  const writeAtomic = (file, obj) => {
    const tmp = `${file}.${process.pid}.${randomBytes(4).toString("hex")}.tmp`;
    try { writeFileSync(tmp, JSON.stringify(obj), { mode: 0o600 }); renameSync(tmp, file); }
    catch (e) { log.warn(`[acme] store write ${file}: ${e.message}`); try { unlinkSync(tmp); } catch {} }
  };
  const rm = (file) => { try { unlinkSync(file); } catch {} };
  const eachCert = (fn) => { let files = []; try { files = readdirSync(dir).filter(isCertFile); } catch {} for (const f of files) fn(join(dir, f), readJson(join(dir, f))); };
  return {
    dir,
    putCert(name, rec) {
      const { keyPem, certPem, expiresAt, renewAt, issuer = "", cached = false } = rec;
      writeAtomic(certFile(name), { v: 1, name: String(name).toLowerCase(), keyPem, certPem, expiresAt, renewAt, issuer, cached, storedAt: Date.now() });
    },
    delCert(name) { rm(certFile(name)); },
    // every usable record, ctx rebuilt; expired, corrupt and un-parseable ones
    // are deleted on the way (a restored cert has to complete a handshake, so
    // createSecureContext is the parse that counts)
    loadCerts(now = Date.now()) {
      const out = [];
      eachCert((file, rec) => {
        if (!rec || rec.v !== 1 || !rec.name || !rec.keyPem || !rec.certPem || !(rec.expiresAt > now)) return rm(file);
        let ctx; try { ctx = tls.createSecureContext({ key: rec.keyPem, cert: rec.certPem }); } catch { return rm(file); }
        out.push({ name: rec.name, keyPem: rec.keyPem, certPem: rec.certPem, expiresAt: rec.expiresAt, renewAt: rec.renewAt || 0,
                   issuer: rec.issuer || "?", cached: !!rec.cached, ctx, restored: true });
      });
      return out;
    },
    certNames() { const n = []; eachCert((_, rec) => { if (rec?.name) n.push(rec.name); }); return n; },
    prune(keep) { let n = 0; eachCert((file, rec) => { if (!rec?.name || !keep.has(rec.name)) { rm(file); n++; } }); return n; },
    // one account per CA directory URL: { kid, keyPem (pkcs8) }
    loadAccount(directory) { const a = readJson(ACCOUNTS)?.[directory]; return a?.kid && a?.keyPem ? a : null; },
    saveAccount(directory, acct) { writeAtomic(ACCOUNTS, { ...(readJson(ACCOUNTS) || {}), [directory]: { kid: acct.kid, keyPem: acct.keyPem, savedAt: Date.now() } }); },
    forgetAccount(directory) { const all = readJson(ACCOUNTS) || {}; delete all[directory]; writeAtomic(ACCOUNTS, all); },
    // "<name>|<ca host>" -> ms timestamp; past entries fall away on load and save
    loadRateLimits(now = Date.now()) {
      const m = new Map();
      for (const [k, v] of Object.entries(readJson(RATELIMITS) || {})) if (Number(v) > now) m.set(k, Number(v));
      return m;
    },
    saveRateLimits(map, now = Date.now()) {
      const o = {}; for (const [k, v] of map) if (v > now) o[k] = v;
      writeAtomic(RATELIMITS, o);
    },
  };
}
// The one store handle, opened lazily (so the refusal is logged once, at the
// first use) - a `let` up here, not a const in the runtime half, because the
// hoisted account/issuance functions reach it from the self-test seam too.
let _acmeStore;
function acmeStore() { if (_acmeStore === undefined) _acmeStore = acmeStoreOpen(ACME_STORE_DIR); return _acmeStore; }

// ---- rate-limit honesty ----------------------------------------------------
// urn:ietf:params:acme:error:rateLimited says WHEN to come back: Let's Encrypt
// puts "retry after 2026-09-03T10:04:00Z" in the detail and/or a Retry-After
// header (seconds or an HTTP-date). Take the later of what was said, an hour
// when nothing was, never more than a week - and remember it per (name, CA),
// so that name skips that CA until then while the other CAs are still tried,
// and a restart (the store) does not forget it. The pump's 5-minute doubling
// stays for every other failure.
const acmeRateLimits = new Map();                   // "<name>|<ca host>" -> nextAt ms
const rateLimitKey = (name, host) => `${String(name).toLowerCase()}|${host}`;
function acmeRetryAfterAt(detail, header, now = Date.now()) {
  let at = 0;
  const m = /retry after (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2}))/i.exec(String(detail || ""));
  if (m) { const t = Date.parse(m[1]); if (t > at) at = t; }
  const h = String(header || "").trim();
  if (h) { const t = /^\d+$/.test(h) ? now + Number(h) * 1000 : Date.parse(h); if (Number.isFinite(t) && t > at) at = t; }
  if (!(at > now)) at = now + 3600_000;
  return Math.min(at, now + ACME_RATE_LIMIT_CAP_MS);
}
const rateLimitErr = (msg, until) => Object.assign(new Error(msg), { rateLimitedUntil: until });
function acmeRateLimitSet(name, host, until, store = acmeStore()) {
  acmeRateLimits.set(rateLimitKey(name, host), until);
  store?.saveRateLimits(acmeRateLimits);
  console.warn(`[acme] ${host}: rate-limited for ${name} until ${new Date(until).toISOString()} - that CA is skipped for this name till then`);
}
const acmeRateLimitedUntil = (name, host, now = Date.now()) => { const t = acmeRateLimits.get(rateLimitKey(name, host)) || 0; return t > now ? t : 0; };

// ---- self-test seam ---------------------------------------------------------
// ACME_SELFTEST=csr|cas|sni|vectors prints the helpers' outputs as one JSON line
// and exits BEFORE any boot side effect (nothing above this point opens a
// socket or touches state). Driven by test/acme.test.mjs; also handy in a CVM
// shell. Never active in production - the var appears in no env file.
if (process.env.ACME_SELFTEST) {
  if (process.env.ACME_SELFTEST === "csr") {
    const name = process.env.ACME_SELFTEST_NAME || "test.app.enclave.host";
    const { csrPem, keyPem } = buildCsr(name);
    console.log(JSON.stringify({ name, csrPem, keyPem }));
  } else if (process.env.ACME_SELFTEST === "cas") {
    // the parsed CA slot list, secrets reduced to presence bits
    console.log(JSON.stringify({ enabled: ACME_ENABLED, platform: !!ACME_PLATFORM,
      cas: ACME_CAS.map(({ directory, host, eabKid }) => ({ directory, host, eab: !!eabKid })) }));
  } else if (process.env.ACME_SELFTEST === "platform") {
    // The platform certificate service client, end to end against a MOCK
    // service (CERTS_API points at the test's server) with stub in-enclave CA
    // slots: ACME_SELFTEST_PLATFORM = { name, endpoint, opKey?, cooldownMs?,
    // rounds?, cas: [{ host, outcome: "ok"|"caErr"|"nameErr" }] }. Each round
    // walks the slots once (the platform slot first when the name is in our
    // zones) and reports what happened; the pending-key map is shared across
    // rounds so a 202-then-200 sequence proves the key is re-presented. Also
    // prints the signed tuple / opSig text and the retry plan for a deferral,
    // so the relay side's expectations can be checked against these strings.
    // A stub CA may also answer "rateLimited" (with retryAfterDetail /
    // retryAfterHeader, parsed by the real acmeRetryAfterAt) - the walker's
    // per-(name, CA) memory is exercised through the real acmeRateLimitSet /
    // acmeRateLimitedUntil pair, persisted to ACME_STORE_DIR when that is set
    // (acmeRestore loads it first, exactly as boot does). With
    // `restoreFirst`, the restored certs decide (as acmeReconcile does, by
    // renewAt) whether a round is walked at all.
    const c = JSON.parse(process.env.ACME_SELFTEST_PLATFORM || "{}");
    const name = c.name, endpoint = c.endpoint, ts = 1_700_000_000, RA_NOW = Date.parse("2026-08-28T00:00:00Z");
    const heldCerts = new Map();
    const restoredCount = c.restoreFirst ? acmeRestore(heldCerts) : 0;
    const signOp = c.opKey ? (text) => privateKeyToAccount(c.opKey).signMessage({ message: text }) : null;
    const pending = new Map();
    // the fixed-vector key: its SPKI hash goes into the printed tuple, its PEM
    // is printed so the test can recompute the hash independently
    const vecKey = buildCsr(name).keyPem, spkiHash = spkiHashOf(vecKey);
    const cas = (c.cas || []).map((x) => ({ host: x.host, outcome: x.outcome, downUntil: 0 }));
    const rounds = [];
    for (let i = 0; i < (c.rounds || 1); i++) {
      const tried = [];
      const slots = [...(ACME_PLATFORM && platformCertName(name) ? [ACME_PLATFORM] : []), ...cas];
      const issueVia = async (slot, n) => {
        tried.push(slot.host);
        if (slot.platform) return acmeIssueViaPlatform(slot, n, { endpoint, signOp, pending });
        if (slot.outcome === "caErr")   throw caErr(`${slot.host} stub: down`);
        if (slot.outcome === "nameErr") throw new Error(`${slot.host} stub: refused ${n}`);
        if (slot.outcome === "rateLimited") {
          const x = (c.cas || []).find((y) => y.host === slot.host) || {};
          throw rateLimitErr(`ACME 429 at ${slot.host}: urn:ietf:params:acme:error:rateLimited ${x.retryAfterDetail || ""}`,
                             acmeRetryAfterAt(x.retryAfterDetail, x.retryAfterHeader));
        }
        const { keyPem } = buildCsr(n);
        return { keyPem, certPem: "stub", expiresAt: 0, renewAt: 0, ctx: null };
      };
      let out;
      if (c.restoreFirst && heldCerts.get(name)?.renewAt > Date.now()) {          // acmeReconcile's "held and still fresh"
        rounds.push({ outcome: "held", issuer: heldCerts.get(name).issuer, tried, cooled: [], pendingHeld: pending.has(name), pendingSpki: "" });
        continue;
      }
      try {
        const r = await acmeWalkSlots(name, { slots, issueVia, cooldownMs: c.cooldownMs || 120_000,
          rateLimitedUntil: (ca) => acmeRateLimitedUntil(name, ca.host),
          onRateLimited: (ca, until) => acmeRateLimitSet(name, ca.host, until) });
        out = { outcome: "issued", issuer: r.issuer, caHost: r.caHost, cached: !!r.cached, expiresAt: r.expiresAt, renewAt: r.renewAt,
                certPem: r.certPem, ctxOk: !!r.ctx, keyHeld: !!r.keyPem };
      } catch (e) {
        const plan = acmeRetryPlan(e, 0, slots.map((s) => s.downUntil), 1_000_000);
        const livePlan = acmeRetryPlan(e, 0, slots.map((s) => s.downUntil));          // the same at the real clock (rate-limit dates are absolute)
        out = { outcome: e.deferMs ? "deferred" : "failed", error: e.message, deferMs: e.deferMs || 0, caLevel: !!e.caLevel, plan, livePlan,
                rateLimitedUntil: e.rateLimitedUntil || 0, allRateLimited: !!e.allRateLimited };
      }
      rounds.push({ ...out, tried, cooled: slots.filter((s) => s.downUntil > Date.now()).map((s) => s.host),
                    rateLimited: Object.fromEntries(slots.map((s) => [s.host, acmeRateLimitedUntil(name, s.host)]).filter(([, t]) => t)),
                    pendingHeld: pending.has(name), pendingSpki: pending.get(name)?.spkiHash || "" });
    }
    console.log(JSON.stringify({
      inZone: platformCertName(name), zones: CERT_ZONES,
      fleetSecretPresent: FLEET_SECRET_PRESENT, certsHttpMs: CERTS_HTTP_MS,
      vecKeyPem: vecKey, spkiHash,
      tuple: certsSigTuple(name, endpoint, spkiHash, ts), opSigText: certsOpSigText(name, endpoint, spkiHash, ts),
      certsKey: CERTS_KEY, sig: CERTS_KEY ? certsSig(name, endpoint, spkiHash, ts) : "",
      opSig: signOp ? await signOp(certsOpSigText(name, endpoint, spkiHash, ts)) : "",
      // the pump's policy for the three failure shapes, at a fixed clock
      plans: { deferred: acmeRetryPlan(deferErr("x", 45_000), 3, [], 1_000_000),
               cooling:  acmeRetryPlan(new Error("x"), 3, [1_000_000 + 30_000], 1_000_000),
               backoff:  acmeRetryPlan(new Error("x"), 1, [], 1_000_000),
               ratelimited: acmeRetryPlan(Object.assign(rateLimitErr("x", 1_000_000 + 86400e3), { allRateLimited: true }), 3, [], 1_000_000),
               ratelimitedCapped: acmeRetryPlan(Object.assign(rateLimitErr("x", 1_000_000 + 30 * 86400e3), { allRateLimited: true }), 3, [], 1_000_000) },
      // retry-after parsing at a fixed clock: detail alone, header (seconds /
      // HTTP-date) alone, both (the later wins), neither (an hour), a cap -
      // the clock is 2026-08-28T00:00Z so the fixture's date is inside the week
      retryAfter: { detail: acmeRetryAfterAt("too many certificates (5) already issued for this exact set of identifiers in the last 168h0m0s, retry after 2026-09-03T10:04:00Z: see https://letsencrypt.org/docs/rate-limits/", null, RA_NOW),
                    headerSec: acmeRetryAfterAt("", "3600", RA_NOW),
                    headerDate: acmeRetryAfterAt("", "Thu, 03 Sep 2026 10:04:00 GMT", RA_NOW),
                    both: acmeRetryAfterAt("retry after 2026-09-03T10:04:00Z", "60", RA_NOW),
                    neither: acmeRetryAfterAt("too many requests", "", RA_NOW),
                    capped: acmeRetryAfterAt("retry after 2099-01-01T00:00:00Z", null, RA_NOW) },
      restoredCount, held: [...heldCerts.keys()],
      rounds }));
  } else if (process.env.ACME_SELFTEST === "store") {
    // The tmpfs store, driven on a directory the test owns:
    // ACME_SELFTEST_STORE = { dir, allowDisk?, fsType? (fakes statfs; absent =
    // the real one), now?, certs: [{ name, keyPem, certPem, expiresAt,
    // renewAt, issuer }], keep?: [names], account?: { directory, kid },
    // rateLimits?: { "<name>|<host>": at } }. Reports the guard's verdict, what
    // a fresh load restores (expired ones dropped), what a prune keeps, and the
    // account / rate-limit round trips - plus every log line the store wrote.
    const c = JSON.parse(process.env.ACME_SELFTEST_STORE || "{}");
    const logs = [];
    const log = { log: (m) => logs.push(m), warn: (m) => logs.push(m) };
    const statfs = c.fsType !== undefined ? () => c.fsType : undefined;
    const guard = acmeStoreGuard(c.dir, { allowDisk: !!c.allowDisk, ...(statfs ? { statfs } : {}) });
    const store = c.guardOnly ? null : acmeStoreOpen(c.dir, { allowDisk: !!c.allowDisk, statfs, log });
    const now = c.now || Date.now();
    const out = { guard, opened: !!store, logs };
    if (store) {
      for (const rec of c.certs || []) store.putCert(rec.name, rec);
      out.filesWritten = readdirSync(store.dir).sort();
      out.modes = Object.fromEntries(readdirSync(store.dir).map((f) => [f, (statSyncMode(join(store.dir, f)))]));
      out.dirMode = statSyncMode(store.dir);
      const restored = store.loadCerts(now);
      out.restored = restored.map((r) => ({ name: r.name, expiresAt: r.expiresAt, renewAt: r.renewAt, issuer: r.issuer, ctxOk: !!r.ctx, keyHeld: !!r.keyPem, restoredFlag: !!r.restored }));
      out.filesAfterLoad = readdirSync(store.dir).filter((f) => /^[0-9a-f]{64}\.json$/.test(f)).length;
      // the boot path proper: acmeRestore fills a certs map from the store
      const certs = new Map();
      out.restoreCount = acmeRestore(certs, now, store);
      out.restoredNames = [...certs.keys()].sort();
      if (c.keep) { out.pruned = store.prune(new Set(c.keep)); out.namesAfterPrune = store.certNames().sort(); }
      if (c.account) {
        const { privateKey, publicKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
        store.saveAccount(c.account.directory, { kid: c.account.kid, keyPem: privateKey.export({ type: "pkcs8", format: "pem" }) });
        const back = store.loadAccount(c.account.directory);
        out.account = { kid: back?.kid, keyMatches: !!back && createPublicKey(createPrivateKey(back.keyPem)).export({ format: "jwk" }).x === publicKey.export({ format: "jwk" }).x,
                        other: store.loadAccount("https://nowhere.invalid/directory") };
        store.forgetAccount(c.account.directory);
        out.account.afterForget = store.loadAccount(c.account.directory);
      }
      if (c.rateLimits) {
        store.saveRateLimits(new Map(Object.entries(c.rateLimits)), now);
        out.rateLimits = Object.fromEntries(store.loadRateLimits(now));
      }
    }
    console.log(JSON.stringify(out));
  } else if (process.env.ACME_SELFTEST === "account") {
    // The account half against a MOCK ACME directory (ACME_DIRECTORY_2 = the
    // test's server; slot 2 needs no EAB) with the store at ACME_STORE_DIR:
    // reports the kid and whether it was restored from the store or
    // registered this run - the test's mock counts newAccount POSTs.
    const ca = ACME_CAS[0];
    let out;
    try { const a = await acmeAccount(ca); out = { kid: a.kid, restored: !!a.restored, thumbprint: a.thumbprint }; }
    catch (e) { out = { error: e.message, caLevel: !!e.caLevel }; }
    out.stored = acmeStore()?.loadAccount(ca.directory)?.kid || null;
    console.log(JSON.stringify(out));
  } else if (process.env.ACME_SELFTEST === "sni") {
    // the SNI decision table (APP_CERT_DOMAIN from env): "acme" = the held CA
    // cert, "bridge" = the self-signed pair, "refuse" = fail closed.
    const ctx = {};                     // truthy stand-in: the decision only routes contexts
    const use = (name, held, managed) => sniDecide(name, held, ctx, managed).use;
    console.log(JSON.stringify({
      domain:     APP_CERT_DOMAIN,
      held:       use("a.app.enclave.host", ctx),
      appNoCert:  use("b.app.enclave.host", null),
      subSub:     use("x.b.app.enclave.host", null),
      caseFold:   use("B.APP.ENCLAVE.HOST", null),
      bareDomain: use("app.enclave.host", null),
      legacyTcp:  use("b.tcp.enclave.host", null),
      noSni:      use(undefined, null),
      // a customer's own domain: attached-and-verified but no cert held yet is
      // a REFUSAL, never the wildcard bridge pair (which is invalid for it)
      customNoCert: use("shop.example.com", null, true),
      customHeld:   use("shop.example.com", ctx, true),
      // …and a name we manage nothing for stays on the pin flow, unchanged
      unknownName:  use("shop.example.com", null, false),
    }));
  } else {
    // RFC 7515 Appendix A.3's P-256 key: the fixed vector the tests compare
    // against an independent RFC 7638 implementation (jose).
    const vec = { kty: "EC", crv: "P-256", x: "f83OJ3D2xF1Bg8vub9tLe1gHMzV76e8Tus9uPHvRVEU", y: "x_FEzRu9m36HLN_tue659LNpXW6pCyStikYjKIWI5a0" };
    const { publicKey, privateKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
    const ownJwk = publicKey.export({ format: "jwk" });
    const jws = jwsSignEs256({ alg: "ES256", nonce: "nonce", url: "https://ca/x" }, { hello: 1 }, privateKey);
    console.log(JSON.stringify({
      thumbprint: jwkThumbprint(vec),
      thumbprintScrambled: jwkThumbprint({ y: vec.y, x: vec.x, kty: vec.kty, crv: vec.crv, extra: "ignored" }),
      ownThumbprintStable: jwkThumbprint(ownJwk) === jwkThumbprint({ ...ownJwk }),
      b64uRoundtrip: Buffer.from(b64u(Buffer.from([0, 251, 255, 62, 63])), "base64url").equals(Buffer.from([0, 251, 255, 62, 63])),
      b64uNoPad: !/[=+/]/.test(b64u(randomBytes(33))),
      jwsVerifies: cryptoVerify("sha256", Buffer.from(`${jws.protected}.${jws.payload}`),
                                { key: publicKey, dsaEncoding: "ieee-p1363" }, Buffer.from(jws.signature, "base64url")),
      dns01: dns01TxtValue("token", jwkThumbprint(vec)),
      eab: eabJws("kid1", b64u(Buffer.from("secret")), vec, "https://ca/newAccount"),
    }));
  }
  process.exit(0);
}

// ---- reachability watchdog — pure half --------------------------------------
// The 2026-07-11 kryptos failure: a CVM whose public DNS record vanished (the
// whole front went with it) but whose OUTBOUND still worked kept claiming and
// renewing on-chain work for six hours — tenants paid for apps nobody could
// reach. Detection keys on ONE precise signal: the advertised hostname
// disappearing from public DNS, affirmed by EVERY configured DoH resolver,
// REACH_DNS_STRIKES checks in a row. DNS-over-HTTPS because it needs no
// hairpin route (a self-request through our own front might) and no trust in
// the CVM's local resolver; a resolver outage reads as SERVFAIL/timeout ->
// "error" -> the strike count HOLDS, so third-party trouble never trips it.
// Any positive answer resets everything. The impure half (DoH fetch, the
// trip/abandon actions) lives with the claim loop; these helpers sit up here,
// before any boot side effect, for the REACH_SELFTEST seam below.

// The hostname worth watching in an advertised endpoint, or null when DNS has
// nothing to lose: IP literals, localhost, mDNS names, single labels (dev
// setups) resolve outside public DNS or not at all.
function reachHostname(endpoint) {
  let host; try { host = new URL(endpoint).hostname; } catch { return null; }
  host = host.replace(/^\[|\]$/g, "").toLowerCase();          // URL keeps IPv6 brackets
  if (net.isIP(host) || !host.includes(".") || host.endsWith(".local")) return null;
  return host;
}

// One DoH JSON body -> "resolves" | "gone" | "error". "gone" only when the
// resolver AFFIRMED the absence: NXDOMAIN (Status 3), or NOERROR with an empty
// answer section. Any record of any type (a CNAME counts: the zone still knows
// the name) is proof of life. Anything else — SERVFAIL, REFUSED, junk — is
// "error" and must never advance the trip counter.
function dohVerdict(body) {
  if (!body || typeof body.Status !== "number") return "error";
  if (body.Status === 0 && Array.isArray(body.Answer) && body.Answer.length) return "resolves";
  if (body.Status === 0 || body.Status === 3) return "gone";
  return "error";
}

// Fold one round of per-resolver verdicts into the next watchdog state. Pure —
// the caller owns fetching and the trip side effects. ANY resolver seeing the
// name = healthy (full reset, clears a trip); EVERY resolver affirming "gone"
// = one strike, tripping at `strikes`; a mixed or errored round holds still.
function reachStep(state, verdicts, strikes) {
  if (!verdicts.length || verdicts.some((v) => v === "resolves")) return { strikes: 0, tripped: false };
  if (verdicts.every((v) => v === "gone")) {
    const n = state.strikes + 1;
    return { strikes: n, tripped: state.tripped || n >= strikes };
  }
  return { strikes: state.strikes, tripped: state.tripped };
}

// REACH_SELFTEST='{"hosts":[...],"bodies":[...],"steps":[{state,verdicts,strikes}]}'
// prints each helper mapped over its inputs as one JSON line and exits — same
// contract as ACME_SELFTEST above (test/reach.test.mjs drives it).
if (process.env.REACH_SELFTEST) {
  const cases = JSON.parse(process.env.REACH_SELFTEST);
  console.log(JSON.stringify({
    hosts: (cases.hosts || []).map(reachHostname),
    verdicts: (cases.bodies || []).map(dohVerdict),
    steps: (cases.steps || []).map((c) => reachStep(c.state, c.verdicts, c.strikes)),
  }));
  process.exit(0);
}

// ---- claim-sweep ordering: own leases outrank new work ----------------------
// Split a ledger pass into leases THIS enclave already holds but is not
// locally serving (a previous life: an update reboot wipes local state while
// leases live on) vs everything else. Resumes must run FIRST, and while any
// own lease is still unresumed the sweep takes NO new work: the lease holder
// already paid for the slice, and a new claim admitted first can consume the
// very capacity the resume needs ("no free capacity") — observed live
// 2026-07-17: a fresh 49% GPU claim displaced an orphaned 49% tenant, which
// then sat dark on a live, still-billing lease. The hold is bounded by the
// lease itself (<= leaseSec): an unresumable lease lapses and re-queues.
function sweepPartition(ledger, enclaveId, nowMs, isLocallyServing) {
  const own = [], rest = [];
  for (const d of ledger) {
    // A stopped deployment may still have an unexpired orphan lease. It
    // cannot resume, so it must not hold new claims until that lease expires.
    const ours = d.active !== false && Number(d.leaseUntil) * 1000 > nowMs && d.runner === enclaveId
      && !isLocallyServing(d.id);
    (ours ? own : rest).push(d);
  }
  return { own, rest };
}

// SWEEP_SELFTEST='{"enclaveId":"0x…","nowMs":…,"ledger":[…],"serving":["id",…]}'
// prints the partition's id lists as one JSON line and exits — same contract
// as the seams above (test/claim-sweep.test.mjs drives it).
if (process.env.SWEEP_SELFTEST) {
  const c = JSON.parse(process.env.SWEEP_SELFTEST);
  const serving = new Set(c.serving || []);
  const { own, rest } = sweepPartition(c.ledger || [], c.enclaveId, c.nowMs, (id) => serving.has(id));
  console.log(JSON.stringify({ own: own.map((d) => d.id), rest: rest.map((d) => d.id) }));
  process.exit(0);
}

// ---- ledger move: the address book retired the contract a record came from --
// Pure core of ledgerMoveSweep (the claim loop's teardown pass below): what to
// do with ONE local record given the CURRENT deployments address. "stamp" = a
// record persisted by an older release with no _ledger field — the current
// ledger is the only one it can be from; "teardown" = the ledger it was
// claimed on was retired by a book repoint, so its lease is void; "skip" =
// nothing to do (not on-chain, terminal, or already on the current ledger).
function ledgerMoveVerdict(rec, currentAddr) {
  if (!currentAddr || !rec._onchain || !["running", "claimed"].includes(rec.status)) return "skip";
  if (!rec._ledger) return "stamp";
  return rec._ledger.toLowerCase() === currentAddr.toLowerCase() ? "skip" : "teardown";
}

// LEDGER_MOVE_SELFTEST='{"current":"0x…","records":[{id,_onchain,status,_ledger},…]}'
// prints each record's verdict as one JSON line and exits — same contract as
// the seams above (test/ledger-move.test.mjs drives it).
if (process.env.LEDGER_MOVE_SELFTEST) {
  const c = JSON.parse(process.env.LEDGER_MOVE_SELFTEST);
  console.log(JSON.stringify((c.records || []).map((r) => ({ id: r.id, verdict: ledgerMoveVerdict(r, c.current) }))));
  process.exit(0);
}

// ---- approval verdict: the deployability decision for a catalog version ----
// PURE (the chain reads live in gateAppReference below; this is the decision
// they feed). Returns null (deployable) or the refusal message. `forPrivate`
// — the deployment this reference runs under is PRIVATE (owner-token-gated
// data path) — is the publisher dev-mode loop ("devDeploy" in /availability):
// it relaxes ONLY the pending state, so a version awaiting approval can be
// tested on the real fleet off the public data path. The STANDING refusals
// (rejected / yanked / app delisted) hold regardless of visibility. Default
// is the strict gate (fail closed): a caller that doesn't know the
// deployment's visibility gets approved-only.
function approvalVerdict({ active, yanked, approval, slug, version }, forPrivate) {
  if (!active)                  return "This app is delisted from the catalog.";
  if (yanked)                   return `${slug}:${version} was yanked by its publisher.`;
  if (Number(approval) === 2)   return `${slug}:${version} was rejected by the catalog owner.`;
  if (Number(approval) !== 1 && forPrivate !== true)
    return `${slug}:${version} is awaiting catalog-owner approval; until then it can only be deployed PRIVATE (owner-only data path) for testing.`;
  return null;
}

// APPROVAL_SELFTEST='{"cases":[{active,yanked,approval,slug,version,forPrivate},…]}'
// prints each case's verdict (null = deployable) as one JSON line and exits —
// same contract as the seams above (test/approval-gate.test.mjs drives it).
if (process.env.APPROVAL_SELFTEST) {
  const c = JSON.parse(process.env.APPROVAL_SELFTEST);
  console.log(JSON.stringify((c.cases || []).map((x) => approvalVerdict(x, x.forPrivate))));
  process.exit(0);
}

// ---- yank teardown: a publisher's yank terminates running deployments ------
// PURE (yankSweep below does the chain reads). Which record ids must be torn
// down, given the set of catalog:// references whose version reads back
// yanked. A yank is the publisher's "this version must not run" — unlike
// delisting (app.active=false), which only blocks NEW deploys at the gate
// while existing deployments serve out, a yank reaches into live work: every
// running/claimed deployment on that exact version is terminated. Only those
// two statuses are in scope — terminal records hold no resources — and only
// a POSITIVE yanked read gets a record on the plan: an unreachable catalog
// keeps everything serving (RPC noise must never tear down paid work).
function yankTeardownPlan(records, yankedRefs) {
  return records
    .filter((r) => (r.status === "running" || r.status === "claimed") && yankedRefs.has(r.ref))
    .map((r) => r.id);
}

// YANK_SELFTEST='{"records":[{id,status,ref},…],"yankedRefs":["catalog://…/0",…]}'
// prints the teardown plan (record ids) as one JSON line and exits — same
// contract as the seams above (test/yank-teardown.test.mjs drives it).
if (process.env.YANK_SELFTEST) {
  const c = JSON.parse(process.env.YANK_SELFTEST);
  console.log(JSON.stringify(yankTeardownPlan(c.records || [], new Set(c.yankedRefs || []))));
  process.exit(0);
}

// ---- owner version change (setAppRef): does the serving record switch? ------
// Pure core of the audit's in-place upgrade: a RUNNING record whose ledger row
// carries a different appRef restarts onto the new record (the lease and the
// balance carry — that is the whole point of setAppRef). Every other state
// leaves the new ref to the normal paths: a terminal record is re-claimed with
// the fresh ref by the sweep, a mid-provision one is caught next audit pass.
function needsVersionSwitch(status, localRef, chainRef) {
  return status === "running" && !!chainRef && !!localRef && chainRef !== localRef;
}

// ---- owner share resize (setShares, ledger rev 6): does the record re-slice? -
// Pure core of the audit's in-place resize: a RUNNING record whose ledger row
// carries different bought shares must re-gate and restart on a re-sliced
// allocation — the rate already changed on-chain, so serving the old size
// would bill the user for capacity they don't get (or the platform for
// capacity it doesn't sell). "stamp" = a record persisted by a pre-resize
// release with no _shares field: adopt what it is CURRENTLY serving
// (reconstructed from its quantized resources) without a restart; the next
// pass compares that stamp against the ledger, so a resize landing in the
// rollout window is still caught one pass later. Non-running states leave the
// new shares to the normal claim paths, exactly like needsVersionSwitch.
function shareResizeVerdict(status, localShares, chainGpuMilli, chainCpuMilli) {
  if (status !== "running") return "skip";
  if (!localShares) return "stamp";
  return Number(localShares.gpuMilli) === Number(chainGpuMilli)
      && Number(localShares.cpuMilli) === Number(chainCpuMilli) ? "skip" : "resize";
}

// SWITCH_SELFTEST='{"switch":[{status,localRef,chainRef}],"backoff":[{entry,nowMs,appRef}],
//                   "resize":[{status,localShares,gpuMilli,cpuMilli}],
//                   "decline":[{entry,nowMs}],"probe":[{entry,nowMs}]}'
// prints each helper mapped over its inputs as one JSON line and exits — same
// contract as the seams above (test/version-switch.test.mjs drives it).
// (Function declarations hoist, so provisionBackoffHolds resolves from here.)
if (process.env.SWITCH_SELFTEST) {
  const c = JSON.parse(process.env.SWITCH_SELFTEST);
  console.log(JSON.stringify({
    switch: (c.switch || []).map((x) => needsVersionSwitch(x.status, x.localRef, x.chainRef)),
    backoff: (c.backoff || []).map((x) => provisionBackoffHolds(x.entry, x.nowMs, x.appRef)),
    resize: (c.resize || []).map((x) => shareResizeVerdict(x.status, x.localShares, x.gpuMilli, x.cpuMilli)),
    decline: (c.decline || []).map((x) => provisionDeclineReason(x.entry, x.nowMs)),
    probe: (c.probe || []).map((x) => prefetchProbeDue(x.entry, x.nowMs)),
  }));
  process.exit(0);
}

// ---- deployment options envelope (create()'s configCid field, repurposed) ---
// A deployer-supplied config CID stays refused (a CID names bytes nobody
// reviewed), but the create() field itself is the one deploy-time string a
// deployment carries on-chain, so it carries a strict JSON envelope of
// PER-DEPLOYMENT settings — interpreted by the supervisor, whitelisted,
// fail-closed: an option this build doesn't recognize REFUSES the claim rather
// than shrugging — silently ignoring one would serve traffic the owner
// believes is filtered (or run an app on a config the owner believes was
// replaced). On-chain (not local state) so it survives the update reboots
// that wipe local records: every claim AND resume re-parses it.
//
// Two namespaces:
//   `waf` — an in-enclave HTTP guard applied at the /x/:id proxy (the single
//   chokepoint both the relay path and the in-enclave TLS bridge flow
//   through). Deliberately no content inspection: everything here is
//   enforceable without buffering bodies, so streaming and WebSockets keep
//   working.
//   `config` — the deployer's app-config override: an inline JSON OBJECT that
//   replaces the picked version's config as THIS deployment's ENCLAVE_CONFIG
//   (volumes key included). The catalog version's config stays the
//   approval-covered default every other deployment gets; the override rides
//   the deployment record only, and a version switch (setAppRef) keeps it.
//   `configCid` — the same override, split the way catalog rev 7 splits a
//   VERSION's config: the bulk lives at a pinned CID and `config` above becomes
//   the inline ROUTING MANIFEST. It exists because this envelope shares one
//   4096-byte ledger field with waf/network, so an app whose config is larger
//   than that had no override available at all.
//     The BARE-CID form of this field stays retired (see the refusal below):
//   that one was the entire config reference, resolved before there was a
//   fail-closed schema to check it against. Here the CID is one validated key
//   of the envelope, and the bytes are accepted at launch only because they
//   re-hash to it — wasm_manager._resolve_config_cid, the same untrusted-
//   gateway path a rev-7 version's config already rides — so the indirection
//   costs integrity nothing. What it costs is a FETCH, and that is what the
//   manifest is for: `volumes` is the one key this runner reads off a
//   deployment's config before launch (volumeGate picks the box on it), so
//   hoisting it keeps the claim and resume gates exactly as I/O-free as they
//   are today. wasi/threads/set/mem64 are read off the VERSION's config and
//   never an override's — they describe the binary, which no override changes.
const DEP_OPTIONS_MAX_BYTES = 4096;
const WAF_KEYS = ["rps", "burst", "maxConcurrent", "maxBodyMb", "methods", "pathBlock", "blockScanners", "uaBlock"];
// The only keys a deployment's inline `config` may carry once `configCid` holds
// the real one: what THIS runner reads off a deployment's config to PLACE it.
// Deliberately shorter than the catalog's ROUTING_KEYS — wasi/threads/set/mem64
// describe the binary (read off the version record, and no override can change
// which bytes the guest is), `gpuOptional` has its own `gpu` namespace here, and
// `_media` is refused outright below.
const DEP_MANIFEST_KEYS = ["volumes"];
// A bare CID, no ipfs:// prefix — mirrors wasm_manager._resolve_config_cid's own
// check, so a CID this parser accepts is one that path can still refuse to trust
// (it verifies the bytes) but can never fail to PARSE.
const DEP_CONFIG_CID_RE = /^[A-Za-z0-9]{10,100}$/;
// blockScanners preset: root-anchored prefixes of the paths bulk scanners
// probe on every host they meet. Prefix-matched on the DECODED, lowercased
// path so %2e%65nv doesn't slip past; deliberately short and boring — an app
// that legitimately serves one of these opts out by not ticking the preset.
const WAF_SCANNER_PATHS = [
  "/.env", "/.git", "/.svn", "/.aws", "/.ssh", "/.htaccess", "/.htpasswd",
  "/.ds_store", "/.vscode", "/.idea", "/wp-admin", "/wp-login.php", "/wp-includes",
  "/wp-content", "/xmlrpc.php", "/phpmyadmin", "/phpinfo", "/cgi-bin",
  "/vendor/phpunit", "/server-status", "/actuator", "/web.config", "/appsettings.json",
  "/id_rsa", "/backup.sql", "/dump.sql", "/config.php",
];
// `gpuMilli` (when the caller has the ledger record) enables the one
// cross-field rule the envelope has: `gpu.optional` is meaningful ONLY on a
// deployment that bought GPU share. Without a slice there is no card to make
// optional, and accepting the flag anyway would tell an owner their CPU-only
// deployment was "preferring" hardware it can never be given.
// A version config's `gpuOptional: true` — the publisher declaring that this
// app's GPU specs describe what it WOULD use, not what it needs to start. An
// unparseable config declares nothing (fail closed: the specs stay required).
function gpuOptionalOfConfig(cfg) {
  try { return JSON.parse(String(cfg || "{}") || "{}").gpuOptional === true; } catch { return false; }
}
// The version's declared wasi world contract ("0.2" | "0.3"), stamped into the
// config by the publish path from the component's own export section. Claim
// ROUTING only — the manager re-classifies the actual bytes at launch, so a
// config that lies fails there with a readable error. Undeclared/unparseable
// means "0.2": every version published before the key existed is wasip2, and
// fail-open-to-p2 is safe both ways (a p2-capable box serves it; an undeclared
// p3 version launch-fails only on boxes that could never serve it anyway).
function wasiOfConfig(cfg) {
  try { return JSON.parse(String(cfg || "{}") || "{}").wasi === "0.3" ? "0.3" : "0.2"; } catch { return "0.2"; }
}
// Cooperative threads (🧵), same doctrine one key over: `threads: true` is
// stamped by the publish path when the binary carries the coop-runtime's
// [thread- imports. Routing only — the manager re-sniffs the bytes at launch.
// Undeclared means "no threads": every pre-threads version, and fail-open in
// that direction is safe (any box serves a non-threaded guest).
function threadsOfConfig(cfg) {
  try { return JSON.parse(String(cfg || "{}") || "{}").threads === true; } catch { return false; }
}
// Shared-everything threads (SET ⚡), same doctrine one key over: `set: true`
// is stamped by the publish path when the binary carries the componentizer's
// [set-spawn-indirect] import. Routing only — the manager re-sniffs the bytes
// at launch. Undeclared = no SET (fail-open direction, any box serves a
// non-SET guest). Independent of `threads`: a guest is coop OR SET, not both.
function setOfConfig(cfg) {
  try { return JSON.parse(String(cfg || "{}") || "{}").set === true; } catch { return false; }
}
// wasm64 (memory64 — the >4 GiB guests), same doctrine one key over:
// `mem64: true` is stamped by the publish path when the module's memory
// section carries the 64-bit flag. Routing only — the manager re-sniffs the
// bytes at launch. Undeclared = wasm32 (fail-open direction: any box serves
// a wasm32 guest).
function mem64OfConfig(cfg) {
  try { return JSON.parse(String(cfg || "{}") || "{}").mem64 === true; } catch { return false; }
}
function parseDepOptions(raw, gpuMilli) {
  const s = String(raw || "").trim();
  if (!s) return {};
  if (s.length > DEP_OPTIONS_MAX_BYTES) throw new Error(`options exceed ${DEP_OPTIONS_MAX_BYTES} bytes`);
  if (!s.startsWith("{") && !s.startsWith("["))
    throw new Error("configCid is retired: a CID names bytes nobody validated — this field may only carry a deployment-options JSON envelope like {\"waf\":{…},\"config\":{…}} (config = an inline app-config override for this deployment); recreate the deployment without a config reference");
  let o; try { o = JSON.parse(s); } catch (e) { throw new Error("options envelope is not valid JSON: " + e.message); }
  if (!o || Array.isArray(o) || typeof o !== "object") throw new Error("options envelope must be a JSON object");
  const unknown = Object.keys(o).filter((k) => k !== "waf" && k !== "config" && k !== "configCid" && k !== "gpu" && k !== "network");
  if (unknown.length) throw new Error(`unknown option namespace ${JSON.stringify(unknown[0])} (this runner knows: waf, config, configCid, gpu, network)`);
  const opts = {};
  if ("network" in o) {
    // WHICH RELAY carries this deployment's traffic. Unlike every other
    // namespace here, nothing in this CVM acts on it: the choice is consumed at
    // the DNS layer, which answers <label>.app.enclave.host with the chosen
    // relay's address instead of the zone-wide default. It is validated here
    // anyway, and refused rather than ignored, for the reason the whole
    // envelope is fail-closed — an owner who typo'd their relay would otherwise
    // be told nothing and quietly keep the default, which is exactly the class
    // of silence this field exists to avoid.
    //
    // A NAME, not an address: the relay set moves (a box is replaced, an
    // address changes) and a deployment should follow the relay it chose rather
    // than pin the machine it happened to be on. Resolution name -> address is
    // the fleet's job, not this record's.
    const n = o.network;
    if (!n || Array.isArray(n) || typeof n !== "object")
      throw new Error("network must be a JSON object like {\"relay\":\"us-west\"}");
    const badN = Object.keys(n).filter((k) => k !== "relay");
    if (badN.length) throw new Error(`unknown network option ${JSON.stringify(badN[0])} (this runner knows: relay)`);
    if ("relay" in n) {
      const r = n.relay;
      // "" / null is a deliberate, expressible choice: back to the zone default.
      if (r === null || r === "") opts.relay = "";
      else if (typeof r !== "string" || !/^[a-z0-9][a-z0-9-]{0,62}$/.test(r))
        throw new Error("network.relay must be a relay name: lowercase letters, digits and dashes, up to 63 characters (or \"\" for the fleet default)");
      else opts.relay = r;
    }
  }
  if ("gpu" in o) {
    // The card requirement, softened. `optional: true` says the deployment
    // PREFERS a GPU enclave (it bought a slice and pays for one where a card
    // exists) but will run on a CPU-only enclave rather than wait — where the
    // ledger charges only the cpu half, because a CPU enclave posts no GPU
    // price. `false` is the default and the old behaviour: GPU-only.
    const g = o.gpu;
    if (!g || Array.isArray(g) || typeof g !== "object") throw new Error("gpu must be a JSON object like {\"optional\":true}");
    const badG = Object.keys(g).filter((k) => k !== "optional");
    if (badG.length) throw new Error(`unknown gpu option ${JSON.stringify(badG[0])} (this runner knows: optional)`);
    if ("optional" in g) {
      if (typeof g.optional !== "boolean") throw new Error("gpu.optional must be true or false");
      // Only an option when a GPU share was actually bought. Refused, not
      // ignored: an owner who sets it on a CPU-only deployment has
      // misunderstood what it does, and silence would leave them believing
      // their app is chasing a card.
      if (g.optional && gpuMilli != null && Number(gpuMilli) <= 0)
        throw new Error("gpu.optional applies only to a deployment that bought GPU share (this one is 0% GPU, so it already runs anywhere - there is no card requirement to make optional)");
      opts.gpuOptional = g.optional;
    }
  }
  // BEFORE `config`: it decides what that field means (real config vs manifest)
  if ("configCid" in o) {
    const cid = o.configCid;
    if (typeof cid !== "string" || !DEP_CONFIG_CID_RE.test(cid))
      throw new Error("configCid must be a bare IPFS CID (10-100 alphanumeric characters, no ipfs:// prefix) naming this deployment's pinned ENCLAVE_CONFIG");
    opts.configCid = cid;
  }
  if ("config" in o) {
    // the app-config override: shape-checked only (object, no reserved keys) —
    // the CONTENT is the app's own contract with its deployer, exactly like a
    // version's config is the app's contract with its publisher
    const c = o.config;
    if (!c || Array.isArray(c) || typeof c !== "object")
      throw new Error("config must be a JSON object — it replaces the version's config as this deployment's ENCLAVE_CONFIG (use {} for an explicitly empty config)");
    if ("_media" in c)
      throw new Error("config._media is reserved for the catalog's store media and never reaches an app — remove it from the override");
    // With configCid set this field is NOT the app's config — it is the routing
    // manifest, and the manager takes the CID and ignores it. Anything else here
    // would be a key the owner believes their app receives and never does.
    // Refused rather than trimmed, for the reason the whole envelope is
    // fail-closed: that silence is exactly what this field exists to avoid.
    if (opts.configCid) {
      const extra = Object.keys(c).filter((k) => !DEP_MANIFEST_KEYS.includes(k));
      if (extra.length) throw new Error(
        `with configCid set, config is the routing manifest and may only carry ${DEP_MANIFEST_KEYS.join("/")} — `
        + `move ${extra.join(", ")} into the pinned config (the guest receives that document, not this one)`);
    }
    opts.config = c;
  }
  if (!("waf" in o)) return opts;
  const w = o.waf;
  if (!w || Array.isArray(w) || typeof w !== "object") throw new Error("waf must be a JSON object");
  const bad = Object.keys(w).filter((k) => !WAF_KEYS.includes(k));
  if (bad.length) throw new Error(`unknown waf option ${JSON.stringify(bad[0])} (this runner knows: ${WAF_KEYS.join(", ")})`);
  const out = {};
  const num = (k, min, max, int) => {
    if (w[k] == null) return null;
    const v = Number(w[k]);
    if (!Number.isFinite(v) || v < min || v > max || (int && !Number.isInteger(v)))
      throw new Error(`waf.${k} must be ${int ? "an integer" : "a number"} in [${min}, ${max}]`);
    return v;
  };
  const rps = num("rps", 0.1, 10000);           if (rps != null) out.rps = rps;
  const burst = num("burst", 1, 100000, true);
  if (burst != null) { if (rps == null) throw new Error("waf.burst needs waf.rps"); out.burst = burst; }
  else if (rps != null) out.burst = Math.max(5, Math.ceil(rps * 4));   // default: ~4s of headroom
  const conc = num("maxConcurrent", 1, 10000, true); if (conc != null) out.maxConcurrent = conc;
  const body = num("maxBodyMb", 0.001, 1024);        if (body != null) out.maxBodyMb = body;
  const strs = (k, max, maxLen, check, what) => {
    if (w[k] == null) return null;
    if (!Array.isArray(w[k]) || w[k].length < 1 || w[k].length > max) throw new Error(`waf.${k} must be an array of 1..${max} ${what}`);
    return w[k].map((x) => {
      if (typeof x !== "string" || !x.trim() || x.length > maxLen || !check(x.trim())) throw new Error(`waf.${k} entry ${JSON.stringify(x)} is not ${what}`);
      return x.trim();
    });
  };
  const methods = strs("methods", 10, 10, (x) => /^[A-Za-z]{3,10}$/.test(x), "an HTTP method name");
  if (methods) out.methods = [...new Set(methods.map((m) => m.toUpperCase()))];
  const paths = strs("pathBlock", 64, 200, (x) => x.startsWith("/"), "a path prefix starting with /");
  if (paths) out.pathBlock = [...new Set(paths.map((p) => p.toLowerCase()))];
  // a 1-2 char UA needle would match nearly every agent string — refuse it
  const uas = strs("uaBlock", 32, 100, (x) => x.length >= 3, "a User-Agent substring of 3+ chars");
  if (uas) out.uaBlock = [...new Set(uas.map((u) => u.toLowerCase()))];
  if (w.blockScanners != null) {
    if (typeof w.blockScanners !== "boolean") throw new Error("waf.blockScanners must be a boolean");
    if (w.blockScanners) out.blockScanners = true;
  }
  if (!Object.keys(out).length) throw new Error("waf enables nothing: set at least one of " + WAF_KEYS.join(", "));
  opts.waf = out;
  return opts;
}

// What an options envelope contributes to a record's CONFIG fields — the two
// that decide what the guest receives as ENCLAVE_CONFIG, and where it comes
// from. `g` is the resolved version (the fallback when the envelope overrides
// nothing). One function because three paths must agree byte-for-byte — first
// claim/adopt, a version switch (setAppRef), and an owner's live setConfig edit
// — and any disagreement between them serves one config while the owner's
// record claims another.
//   Both fields are rewritten in BOTH directions on purpose. An override must
// clear the version's appConfigCid or the manager would fetch that and silently
// ignore what the owner signed; dropping an override must restore it, or a
// rev-7 version would hand its guest the routing manifest as its config.
//   With `configCid` the inline object is the MANIFEST: it stays on the record
// (the owner's view, and volumeGate's input) while the manager takes the CID and
// ignores it — the exact pairing a rev-7 VERSION already produces, which is why
// nothing downstream of here needs to know which of the two set it.
function overrideConfigFields(o, g) {
  if (!("config" in o) && !("configCid" in o))
    return { config: (g && g.config) || "", appConfigCid: (g && g.configCid) || "", configOverride: false };
  return { config: "config" in o ? JSON.stringify(o.config) : "",
           appConfigCid: o.configCid || "",
           configOverride: true };
}

// ---- owner envelope change (setConfig): does the serving record re-apply? ---
// Pure core of the audit's envelope watch. The options envelope is mutable
// on-chain (EnclaveDeployments.setConfig) but was only ever read at claim -
// this verdict is what makes an owner's edit reach a LIVE deployment. Verdicts:
//   "skip"    - not applicable, or nothing changed
//   "stamp"   - a record from before this watch: adopt the current value
//               WITHOUT a restart (rollout must not restart the fleet)
//   "waf"     - only the waf namespace changed: swap rec.waf live, no restart
//   "restart" - the config namespace changed: relaunch on the new
//               ENCLAVE_CONFIG (the setAppRef restart-in-place rule)
//   "error"   - the new envelope doesn't parse under THIS build's rules: keep
//               the old config serving and surface why (the claim gate's
//               fail-closed refusal can't apply to what already runs)
function envelopeEditVerdict(rec, chainCid) {
  if (!rec._onchain || rec.status !== "running") return "skip";
  const cur = String(chainCid || "");
  if (rec._envelope == null) return "stamp";
  if (rec._envelope === cur) return "skip";
  let oldO = {}, newO;
  try { oldO = parseDepOptions(rec._envelope); } catch {}    // a stale-unparsable stamp reads as "no options"
  try { newO = parseDepOptions(cur); } catch { return "error"; }
  // "config absent" (version default) and "config: {}" (explicitly empty) are
  // different owner intents - null vs "{}" keeps them distinct here too. The CID
  // rides the same key: repointing it at a different pinned document is a config
  // change even when the inline manifest is byte-identical, and swapping an
  // inline override for a CID one is too.
  const cfg = (o) => ("config" in o || "configCid" in o)
    ? JSON.stringify([o.configCid || "", "config" in o ? o.config : null]) : null;
  return cfg(newO) === cfg(oldO) ? "waf" : "restart";
}

// CFG_EDIT_SELFTEST='{"records":[{"rec":{…},"chainCid":"…"},…]}' prints each
// record's verdict as one JSON line and exits - same contract as the seams
// above (test/config-edit.test.mjs drives it).
if (process.env.CFG_EDIT_SELFTEST) {
  const c = JSON.parse(process.env.CFG_EDIT_SELFTEST);
  console.log(JSON.stringify((c.records || []).map((r) => ({ verdict: envelopeEditVerdict(r.rec || {}, r.chainCid) }))));
  process.exit(0);
}

// Is this app-relative URL blocked by the deployment's path rules? Decoded
// (percent-encoding must not dodge a prefix), lowercased, query stripped.
function wafPathBlocked(w, url) {
  let p = String(url || "/").split("?")[0];
  try { p = decodeURIComponent(p); } catch {}                 // undecodable %-junk: match the raw bytes
  p = ("/" + p.replace(/^\/+/, "")).toLowerCase();
  if (w.blockScanners && WAF_SCANNER_PATHS.some((x) => p.startsWith(x))) return true;
  return (w.pathBlock || []).some((x) => p.startsWith(x));
}

// WAF_SELFTEST='{"parse":["…", …],"paths":[{"waf":{…},"url":"…"}, …]}' prints
// each helper mapped over its inputs as one JSON line and exits — same
// contract as the seams above (test/waf.test.mjs drives it).
if (process.env.WAF_SELFTEST) {
  const c = JSON.parse(process.env.WAF_SELFTEST);
  console.log(JSON.stringify({
    // an entry may be a bare envelope string, or {raw, gpuMilli} to drive the
    // one cross-field rule (gpu.optional needs a bought GPU share)
    parse: (c.parse || []).map((x) => {
      const raw = (x && typeof x === "object") ? x.raw : x;
      const gpuMilli = (x && typeof x === "object") ? x.gpuMilli : undefined;
      try { return { ok: parseDepOptions(raw, gpuMilli) }; } catch (e) { return { err: e.message }; }
    }),
    paths: (c.paths || []).map((x) => wafPathBlocked(x.waf, x.url)),
  }));
  process.exit(0);
}

// TENANT_HEADERS_SELFTEST='[{…upstream headers…}, …]' prints what the /x/:id
// proxy would actually send for each, same seam contract as WAF_SELFTEST.
// (tenantHeaders is a hoisted function declaration, so it is reachable here.)
if (process.env.TENANT_HEADERS_SELFTEST) {
  console.log(JSON.stringify(JSON.parse(process.env.TENANT_HEADERS_SELFTEST).map((h) => tenantHeaders(h))));
  process.exit(0);
}

// APPAUTH_SELFTEST='{"ids":["0x…","0x…"],"addrs":["0x…"],"cookies":[…],"reqs":[…]}'
// prints what the private-app browser path would decide, same seam contract as
// WAF_SELFTEST (test/private-app-auth.test.mjs drives it).
//
// The property that matters most here is NEGATIVE and cannot be seen by reading
// one function: the app cookie and the control-plane session are signed by the
// SAME in-enclave key, so only the audience keeps a token that lives on a
// tenant origin from opening /v1/deployments, logs and secrets. That crossing
// is what this seam exists to pin.
if (process.env.APPAUTH_SELFTEST) {
  const c = JSON.parse(process.env.APPAUTH_SELFTEST);
  initSessionKey();
  const [idA, idB] = c.ids || [];
  const addr = (c.addrs || [])[0];
  const appTok  = await mintAppToken(addr, idA);
  const sessTok = await mintSession(addr, new Date(Date.now() + 3600e3));
  console.log(JSON.stringify({
    // the app token opens ITS deployment, and nothing else
    appForOwnId:   await verifyAppToken(appTok, idA),
    appForOtherId: await verifyAppToken(appTok, idB),
    // …and is not a control-plane session, though the same key signed it
    appAsSession:  await verifySessionToken(appTok),
    // …while a real session still is one, and is not an app cookie
    sessAsSession: await verifySessionToken(sessTok),
    sessAsApp:     await verifyAppToken(sessTok, idA),
    cookies: (c.cookies || []).map((raw) => ({ read: cookieVals({ headers: { cookie: raw } }, APP_COOKIE),
                                               kept: stripAppCookie(raw) })),
    reqs: (c.reqs || []).map((r) => wantsHtml({ method: r.method || "GET", headers: r.headers || {} })),
    // What the hand-off route DECIDES, driven with stand-ins: which paths it
    // claims (an unclaimed one falls through to the owner gate), and what it
    // actually sets. `tok`/`sess` in a case substitute the real tokens above.
    routes: await Promise.all((c.routes || []).map(async (r) => {
      // Checksummed, as a live record is (view() builds owner via getAddress):
      // both sides of the ownership compare must be in the same case.
      const rec = { id: r.id || idA, owner: getAddress(r.owner || addr), public: !!r.public };
      const auth = r.auth === "tok" ? "Bearer " + appTok
                 : r.auth === "sess" ? "Bearer " + sessTok
                 : r.auth;
      const out = { status: 0, cookie: "", body: "" };
      let done; const settled = new Promise((s) => { done = s; });
      const res = {
        headersSent: false,
        setHeader(k, v) { if (/^set-cookie$/i.test(k)) out.cookie = v; },
        status(s) { out.status = s; return res; },
        json(o) { out.body = JSON.stringify(o); res.headersSent = true; done(); return res; },
        end(b) { if (b) out.body = String(b).slice(0, 120); res.headersSent = true; done(); return res; },
      };
      const req = { method: r.method || "GET", url: r.url || "/",
                    headers: auth ? { authorization: auth } : {} };
      const handled = appSessionRoute(rec, req, res);
      if (handled) await Promise.race([settled, new Promise((s) => setTimeout(s, 250))]);
      // Report the cookie's NAME, ATTRIBUTES and whether it carries a value —
      // never the token itself, so a failing assertion cannot print a credential.
      const [pair, ...attrs] = out.cookie.split(";").map((s) => s.trim());
      const name = pair ? pair.slice(0, pair.indexOf("=")) : "";
      return { handled, status: out.status, name, attrs,
               set: !out.cookie ? "" : /(^|;)\s*Max-Age=0\s*(;|$)/.test(out.cookie) ? "cleared"
                    : pair.slice(pair.indexOf("=") + 1) ? "set" : "empty" };
    })),
  }));
  process.exit(0);
}

// CLIENT_IP_SELFTEST='[{"headers":{…},"remoteAddress":"…","stamped":"…"}, …]'
// prints the key the WAF and the hint limiter would bucket each request under.
// Same seam contract as WAF_SELFTEST; clientIpOf is a hoisted declaration, so it
// is reachable from here.
if (process.env.CLIENT_IP_SELFTEST) {
  console.log(JSON.stringify(JSON.parse(process.env.CLIENT_IP_SELFTEST).map((c) =>
    clientIpOf({ headers: c.headers || {},
                 socket: { remoteAddress: c.remoteAddress, ...(c.stamped ? { _clientIp: c.stamped } : {}) } }))));
  process.exit(0);
}

// The pure half of the ledger schema-sniff machinery (depsAbi below). A REAL
// pre-deploymentsSchema ledger has code at its address, so probing the unknown
// selector REVERTS; "returned no data"/"0x" means the RPC saw NO CODE there —
// with a multi-provider pool that's a lagging or throttled member answering
// for a freshly migrated ledger, never proof of a rev-1 contract. Caching
// rev 1 on it poisoned every get/getPage decode until reboot (observed live
// 2026-07-17 on kryptos: claim/resume/hint all 502'd IntegerOutOfRange while
// the box sat 99.6% free and its tenants' leases lapsed).
function sniffCachePolicy(errMsg) {
  return /revert/i.test(errMsg || "") ? "cache-rev1" : "retry";
}
// Does a get/getPage failure mean OUR cached component list is wrong for the
// bytes the ledger returned? Misaligned tuple decodes surface as viem's
// integer-range / bounds / data-size / boolean errors, never as transport
// errors — those keep the shape and bubble as chain_unreachable.
function shapeDecodeError(errMsg) {
  return /safe integer range|out[- ]of[- ]bounds|data size|not a valid boolean/i.test(errMsg || "");
}

// SNIFF_SELFTEST='{"probeErrors":["…"],"decodeErrors":["…"]}' prints each
// message's classification as one JSON line and exits — same contract as the
// seams above (test/ledger-sniff.test.mjs drives it).
if (process.env.SNIFF_SELFTEST) {
  const c = JSON.parse(process.env.SNIFF_SELFTEST);
  console.log(JSON.stringify({
    probe: (c.probeErrors || []).map(sniffCachePolicy),
    decode: (c.decodeErrors || []).map(shapeDecodeError),
  }));
  process.exit(0);
}

// ---- resource model: EXACT RESOURCES -> TWO CALCULATED SHARES ---------------
// Apps specify EXACT resources on four axes: vramGb + gpuTflops of one GPU card
// (both 0 = CPU-only app) and memMb + cpuGflops of the node. CPU compute is
// measured in GFLOPS, not TFLOPS: a whole 16-vCPU node peaks around ONE
// TFLOPS (vs 989 for the card), so TFLOPS-grained CPU asks round to "all of
// it or nothing" - GFLOPS is the honest grain. From those the
// platform CALCULATES two normalized shares — the allocation/routing/billing
// unit — by dividing the app's spec by the server's spec, taking the LARGER of
// the memory- and compute-derived share per pool (both axes are occupied
// together, so the bigger one is what's really consumed), rounded UP to the
// whole-percent grain so a share is never worth less than what was asked for:
//   gpuShare (0..1) — max(vramGb / CARD_VRAM_GB, gpuTflops / CARD_TFLOPS).
//                     The MPS compute % and the VRAM cap both follow it.
//   cpuShare (0..1) — max(memMb / node RAM, cpuGflops / NODE_GFLOPS); the
//                     node's vCPUs come along; the wasm guest is capped at memMb.
// The two are INDEPENDENT (ledger rev 13 dropped the old gpuShare >= cpuShare
// rule): different pools, reserved separately, priced additively. A CPU-heavy
// app may hold most of a node beside a sliver of a card, and a card-heavy one
// the reverse. The leftovers are a feature: a
// tenant taking a whole card + 10% of the node's RAM leaves 90% of the CPU/RAM,
// which GPU enclaves rent to CPU-only apps (CPU-only enclaves get first claim;
// see the claim loop). CC disables MIG, so a card is ONE trust domain sliced in
// SOFTWARE: isolation comes from the process boundary, not the slice size.
let GPU_COUNT         = parseInt(process.env.GPU_COUNT || "1", 10);     // cards in this enclave; 0 = CPU-only enclave
// GPU work (gpuShare > 0) runs ONLY on GPU-enabled enclaves. CPU-only work runs
// on CPU-only enclaves first, and on GPU enclaves out of leftover cpu pool.
//
// NOT const, and not boot-time, because a SHIELDED card arrives late: the box
// only learns it has one when the guest's boot probe finishes a real masked GEMM,
// which happens after this process is already serving. adoptShieldedCard() below
// is what flips it, and it is the whole reason these four are `let`.
let IS_GPU            = GPU_COUNT > 0;
const NODE_VCPUS      = parseInt(process.env.NODE_VCPUS || "16", 10);   // node size, for CPU pricing/readouts
const NODE_RAM_GB     = parseInt(process.env.NODE_RAM_GB || "64", 10);
const NODE_GFLOPS     = parseFloat(process.env.NODE_GFLOPS || "")       // CPU compute per node in GFLOPS (16 vCPU ≈ 1000)
                     || parseFloat(process.env.NODE_TFLOPS || "1") * 1000; // legacy env name (was TFLOPS-denominated)
let CARD_VRAM_GB      = parseFloat(process.env.GPU_VRAM_GB || "141");   // usable VRAM per card (fallback until the card itself is probed - see adoptCardVram)
let CARD_VRAM_SRC     = process.env.GPU_VRAM_GB ? "env" : "default";
let CARD_TFLOPS       = parseFloat(process.env.GPU_TFLOPS || "989");    // GPU compute per card (H200 FP16 dense)
// --- this enclave's PRICE (ledger rev 8; metal/PROTOCOL.md) ------------------
// What renting THIS machine costs, in USDC 6dp per second for a FULL node
// (vCPU+RAM) and a FULL card (GPU+VRAM). It is not a floor or a hint: it is the
// price, published in our EnclaveRegistry entry, and the ledger charges a
// tenant exactly this fraction of it — shares/1000 — when we claim their work.
// A deployment whose owner set a lower ceiling than we charge is simply not
// ours to take (see rateCapRefusal); a buyer picks whoever is cheap enough.
// Defaults are the hosted fleet's long-standing $3.00/hr node and $6.00/hr
// card, so an enclave that says nothing prices exactly as it always has. A
// CPU-only enclave has no card to sell (0).
const SELL_CPU_PRICE6 = Math.max(1, Math.round(parseFloat(process.env.SELL_CPU_PRICE6 || "") || 834));
let SELL_GPU_PRICE6   = IS_GPU ? Math.max(0, Math.round(parseFloat(process.env.SELL_GPU_PRICE6 || "") || 1667)) : 0;
// Whether the operator STATED that price or inherited the default. Only the
// pre-rev-8 floor cares: on those ledgers the price is global and a record may
// have been created at an older, lower one, so applying our default as a floor
// would refuse live tenants the fleet has always served. Explicit = a seller
// who means it (metal boxes); default = behave exactly as before rev 8.
const PRICE_IS_EXPLICIT = !!(process.env.SELL_CPU_PRICE6 || process.env.SELL_GPU_PRICE6);
// USDC/sec as floats, for the human-facing quotes (/v1/pricing, HTTP deploys)
const CPU_RATE        = SELL_CPU_PRICE6 / 1e6;   // the WHOLE node's vCPU+RAM
const FULL_RATE       = SELL_GPU_PRICE6 / 1e6;   // a WHOLE card
const CTX_OVERHEAD_GB = parseFloat(process.env.CTX_OVERHEAD_GB || "0.5"); // per-worker context cost, reserved on top of the cap
let SM_TOTAL          = parseInt(process.env.SM_TOTAL || "132", 10);   // SMs per card (H200=132); for reporting granted SMs
const MIN_COMPUTE_PCT = parseInt(process.env.MIN_COMPUTE_PCT || "1", 10); // floor; CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is an integer 1..100
// Request rounding for the share-derived VRAM cap. The grain must stay small
// relative to the SMALLEST card this ships to: at "1", a 1% tenant on a 6.5 GB
// shielded card ceil'd to a whole GB and booked 15% of the card's advertised
// VRAM (metal0, 2026-09-01). Claims re-normalize on adopt, so a changed grain
// re-books existing tenants at the next restart/re-claim, not mid-lease.
const GRANULARITY_GB  = parseFloat(process.env.VRAM_GRANULARITY_GB || "0.1");

const round1 = (x) => Math.round(x * 10) / 10;
const round3 = (x) => Math.round(x * 1000) / 1000;
// compute is dialed by an INTEGER percent (the MPS cap grain) - quantize any
// requested share to whole percent, floored at MIN_COMPUTE_PCT. This is the true
// allocatable unit; there is no finer control and no 1/7 floor.
const quantizePct = (share) => Math.min(100, Math.max(MIN_COMPUTE_PCT, Math.round(share * 100)));
// app specs -> MINIMUM shares: divide the app's exact spec by THIS server's
// spec, take the LARGER of the memory- and compute-derived share per pool,
// round UP to the percent grain — the minimum share is never worth less than
// the resources the app declared it needs.
//
// NOT clamped to 100, deliberately, and this is the whole difference between
// sizing and allocating. A 50 GB app on a 6.5 GB card needs 7.69 CARDS; the old
// Math.min(100, …) reported that as "1 card" and destroyed the only fact the
// placer needed. Two ways it lied: a box would report `needs gpuShare 1` when
// no share it can sell is enough, and a deployment dialled at the full 1.0
// passed the floor check while still being 7.7x short. A ratio above 1 is a
// legitimate answer meaning "more than this box has" — quantizePct above keeps
// the 1..100 clamp because the MPS thread percentage genuinely is per-card and
// integral, and that is an ALLOCATION grain, not a measure of need.
const pctCeil = (x) => Math.max(MIN_COMPUTE_PCT, Math.ceil(x * 100 - 1e-9));
const gpuShareOf = (vramGb, gpuTflops = 0) => (vramGb > 0 || gpuTflops > 0)
  ? pctCeil(Math.max(vramGb / CARD_VRAM_GB, gpuTflops / CARD_TFLOPS)) / 100 : 0;
const cpuShareOf = (memMb, cpuGflops = 0) =>
  pctCeil(Math.max(memMb / (NODE_RAM_GB * 1024), cpuGflops / NODE_GFLOPS)) / 100;
// An app's catalog specs -> the minimum shares a deployment must buy here.
// Zero-guarded: axes the app didn't declare add no minimum. Each axis floors on
// its own hardware and nothing else.
// `min.gpuOptional` (the publisher's word, from the version config) turns the
// declared GPU axes from a REQUIREMENT into a preference: the app runs without
// a card, and would use one if given it. The floor then stops forcing a GPU
// dial, so the deployment may buy 0% GPU and any enclave can serve it. It does
// NOT stop the specs meaning something - they still size the slice a deployer
// who wants the card should buy, and the console recommends exactly that.
// Which is why `gpuShare` and `gpuFloor` are now SEPARATE returns: the old code
// collapsed the requirement to 0 the moment the publisher said "optional", so a
// box asked to serve a 50 GB app on a 6.5 GB card saw no GPU requirement at all
// and accepted the work as though the card had never mattered. The requirement
// is always computed; `gpuOptional` decides what may be done about it, and the
// caller - not this function - makes that call.
//
// `opts.volGb` is the size of the model volumes the app's effective config
// names, and it corrects the GPU floor ONLY. On a card the weights really are
// resident in the tenant's own VRAM slice, so a declared `vramMb` under the
// volume it names is an under-declaration and the volume is the truth.
//
// It deliberately does NOT touch the CPU floor, though the symmetry is
// tempting. On cores the weights are mmap'd page cache the kernel reclaims,
// and the platform charges them to the NODE, not the share -
// wasm_manager._nn_budgets: "charging 20 GB of reclaimable page cache against
// a share was wrong twice over: it refused models a node has ample room for,
// and it measured the wrong bytes." A 4% tenant may legitimately map a 17 GB
// GGUF. Adding volGb here would re-introduce exactly that, and would refuse a
// deployment the box then serves perfectly well.
// Each axis floors on its OWN hardware: the GPU minimum from the card, the CPU
// minimum from the node. Before rev 13 the GPU floor was additionally lifted to
// the CPU floor, purely to keep the derived minimums legal under the ledger's
// gpuMilli >= cpuMilli rule — it was never a statement about what the app
// needs. site/js/core/pricing.js:minPctsOf mirrors this EXACTLY; a console
// floor below this one sells a deployment no runner will claim.
function minSharesOf(min, opts = {}) {
  const volGb = Math.max(0, Number(opts.volGb) || 0);
  const memMb = min.memMb || 0, cpuGflops = min.cpuGflops || 0;
  const cpu = (memMb || cpuGflops) ? cpuShareOf(memMb, cpuGflops) : 0;
  // What the same work costs on cores, where the weights land in node RAM
  // beside the guest's own linear memory instead of on a card.
  // The volumes CORRECT a declared VRAM figure, they never create one. A
  // publisher who declared no card is asking for cores, and folding the model
  // size into a GPU ask there would invent a requirement nobody stated (and
  // make every CPU-only LLM app look card-bound).
  const vramGb = (min.vramMb || 0) > 0 ? Math.max(min.vramMb / 1024, volGb) : 0;
  const gpu = (vramGb > 0 || min.gpuGflops)
    ? gpuShareOf(vramGb, (min.gpuGflops || 0) / 1000) : 0;
  return {
    gpuShare: gpu,                              // the TRUE ask, in whole cards of THIS box (may exceed 1)
    gpuFloor: min.gpuOptional ? 0 : gpu,        // what a dial must actually meet (optional = none)
    cpuShare: cpu,
    gpuOptional: !!min.gpuOptional,
  };
}

// Can this box serve a GPU-dialled deployment's CARD ask, and if not, may the
// work run on cores instead? Pure, so the claim gate and the SIZING_SELFTEST
// seam share one implementation rather than two that drift.
//
// Two ways the ask goes unmet: no card at all, or a card too small for the app
// whatever share is bought. allocGpu best-fits a SINGLE card and cannot span
// one tenant across two (CC gives one trust domain per device), so a
// requirement above 1.0 is unservable here however free the pools are. Saying
// so is the whole point of the unclamped ratio: before it a 50 GB app on a
// 6.5 GB card reported "needs 1 card", the box agreed it had one, and the
// tenant died at weight-load.
function gpuRouting(mins, { declaresGpu = false, envelopeOptional = false } = {}) {
  // "Too small for the app" is the second unmet reason (the first being no card
  // at all) — but ONLY for a LOCAL card, where the weights are resident in the
  // tenant's own VRAM slice and a model that exceeds it dies at weight-load
  // (ggml_abort, fatal). A SHIELDED card is the opposite: offload is per-matmul
  // over the masked protocol, the reservation is an offload BUDGET not weight
  // residency, and a model larger than the card is merely SLOWER (more terms
  // stay in the enclave), never fatal. So it serves any size — the dial claims
  // as much of the card as it buys and the backend offloads the subset that
  // fits. Refusing it for being "too small" is exactly what dropped a 27B onto
  // cores while a calibrated 6.5 GB shielded card sat idle (metal0, 2026-08-31).
  // The ratio is still computed (it is what the console recommends buying); on a
  // shielded card it just stops being a reason to decline the work.
  const unmet = !IS_GPU ? "this enclave has no card"
    : (_shieldedAdopted || mins.gpuShare <= 1 + 1e-9) ? null
    : `the app needs ${round1(mins.gpuShare)}x this box's whole card `
        + `(${round1(CARD_VRAM_GB)} GB / ${round1(CARD_TFLOPS)} TFLOPS)`;
  if (!unmet) return { onGpu: true, unmet: null, refusal: null };
  // Either flag opens the fallback; neither one can waive a HARD requirement.
  // The envelope is the OWNER's dial and may waive their own preference; only
  // the PUBLISHER may say a declared card need is soft, or a version that
  // needs 128 GB of VRAM "falls back" to CPU and thrashes forever.
  if (!(mins.gpuOptional || (envelopeOptional && !declaresGpu)))
    return { onGpu: false, unmet,
             refusal: `GPU work this enclave cannot serve: ${unmet}`
                    + (declaresGpu ? " (the version declares the card required)" : "") };
  return { onGpu: false, unmet, refusal: null };
}

// per-card free pools (vram + compute). With CC on there is exactly one whole
// device per card - no MIG instances to enumerate.
const gpuCards = Array.from({ length: GPU_COUNT }, (_, i) => ({ id: i, uuid: null, vramFree: CARD_VRAM_GB, computeFree: 1 }));

// The card outranks config: GPU_VRAM_GB is only the boot fallback. The real
// memory.total arrives from whichever probe can reach the card - our own
// nvidia-smi discovery where this process can see the GPU, or the manager's
// boot probe via /health on Tinfoil (the supervisor container has neither).
// Rebase the free pools by the delta so reservations made before adoption
// (loadState restores, early claims) stay accounted.
function adoptCardVram(gb, source) {
  if (!IS_GPU || !(gb > 0)) return;
  if (Math.abs(gb - CARD_VRAM_GB) < 0.05) { CARD_VRAM_SRC = source; return; }
  const delta = gb - CARD_VRAM_GB;
  for (const c of gpuCards) {
    c.vramFree += delta;
    if (c.vramFree < 0) {
      console.warn(`[gpu] card ${c.id}: live reservations exceed probed ${gb} GB by ${(-c.vramFree).toFixed(1)} GB - clamping (frees as tenants release)`);
      c.vramFree = 0;
    }
  }
  console.log(`[gpu] card VRAM ${CARD_VRAM_GB} GB (${CARD_VRAM_SRC}) -> ${gb} GB (${source})`);
  CARD_VRAM_GB = gb; CARD_VRAM_SRC = source;
}

// The node's vCPU+RAM pool — EVERY enclave has one. On a CPU-only enclave it is
// the only pool; on a GPU enclave every GPU deployment's cpuShare draws from it
// too, and whatever is left over is rentable by CPU-only apps. The wasm-manager
// admits by the same share, so the two allocators agree. A CPU-only handle
// lives in rec._gpu as { cpu: true, share } so every reserve/release/persist
// call site is shared with the GPU path.
const cpuPool = { shareFree: 1 };
function allocCpu(share) {
  if (cpuPool.shareFree < share - 1e-9) return null;
  cpuPool.shareFree -= share;
  return { cpu: true, share };
}
const maxFreeCpu = () => Math.max(0, Math.min(1, cpuPool.shareFree));
// CPU requests use the same whole-percent grain as GPU compute; priced at the
// share of the whole-node rate.
const normalizeCpuReq = (share) => { const pct = quantizePct(share); return { cpu: true, gpuShare: 0, cpuShare: pct / 100, share: pct / 100, pct }; };

// price = both shares, additively: the GPU slice at the whole-card rate plus
// the CPU slice at the whole-node rate (mirrors EnclaveDeployments' rate formula).
const rateFor = (gpuShare, cpuShare) => FULL_RATE * gpuShare + CPU_RATE * cpuShare;
// normalize a GPU request: quantize both shares to the integer-percent grain
// (the MPS cap grain — the true allocatable unit) and derive the VRAM cap from
// the GPU share (rounded UP to granularity — the tenant gets the round-up for
// free). The two shares are INDEPENDENT: they come from different pools (this
// card's VRAM+compute, the node's vCPU+RAM) and allocGpu reserves them
// separately. This used to clamp cpuShare down to gpuShare to mirror the
// pre-rev-13 ledger rule; a rev-13 record may legitimately buy most of a node
// beside a sliver of a card, and clamping would have silently starved it of
// the CPU it paid for.
function normalizeGpuReq(gpuShare, cpuShare) {
  const gpct = quantizePct(gpuShare);
  const cpct = quantizePct(cpuShare);
  const v = _shieldedPool ? (gpct / 100) * CARD_VRAM_GB
    : Math.ceil((gpct / 100) * CARD_VRAM_GB / GRANULARITY_GB) * GRANULARITY_GB;
  return { gpuShare: gpct / 100, cpuShare: cpct / 100, vramGb: v,
           computeShare: gpct / 100, computePct: gpct };
}
// CTX_OVERHEAD_GB is a PER-WORKER CUDA context cost, and a shielded card has no
// per-TENANT worker to charge it to. A shielded tenant runs inside the CVM with
// CUDA_VISIBLE_DEVICES="" and opens no context at all; the single worker process
// on the untrusted host holds one context for the whole box, however many
// tenants are offloading to it. Charging it per tenant books it TWICE -- once on
// top of the tenant's own slice, and again as the next tenant's entry cost in
// maxFreeGpuShare -- which on a 6.5 GB card makes any share above ~84.6%
// advertise a completely full card. Measured on metal0 2026-08-26: an 85% lease
// booked the entire 6.5 GB budget while the card actually held 736 MiB.
//
// The share-proportional VRAM reservation itself is KEPT. It over-reserves for
// this tier too (a shielded tenant's real VRAM is its model's encoded weights,
// not a fraction of the card), but it is the only thing bounding how many models
// land on one card, and erring high there costs advertised capacity rather than
// a tenant that fails to launch.
// Set one-way by adoptShieldedCard() below. Declared HERE rather than beside it
// because the pool math reads it and the POOL_SELFTEST seam evaluates during
// module init, which a later `let` would put in its temporal dead zone.
let _shieldedAdopted = false;
let _shieldedPool = null;
const ctxOverheadGb = () => (_shieldedAdopted ? 0 : CTX_OVERHEAD_GB);

// reserve an arbitrary slice on a single card (best-fit on VRAM) PLUS the
// deployment's cpuShare from the node pool — both or neither. VRAM overhead is
// reserved on top of the cap so the sum of live workers never exceeds physical.
const cardVram = c => c.vramTotal ?? CARD_VRAM_GB;
const cardFreeVram = c => c.shielded
  ? (c.available ? Math.min(c.vramFree, c.proof?.vramFreeGb || 0) : 0)
  : c.vramFree;
function allocGpu(vramGb, computeShare, cpuShare) {
  if (_shieldedPool) {
    if (!(computeShare > 0) || cpuPool.shareFree < cpuShare - 1e-9) return null;
    const shares = _shieldedPool.cardIds.map(id => ({ cardId: id, computeShare,
      vramGb: computeShare * cardVram(gpuCards[id]),
      _needV: computeShare * cardVram(gpuCards[id]) }));
    if (shares.some(h => cardFreeVram(gpuCards[h.cardId]) < h._needV - 1e-9 ||
        gpuCards[h.cardId].computeFree < computeShare - 1e-9)) return null;
    for (const h of shares) {
      gpuCards[h.cardId].vramFree -= h._needV;
      gpuCards[h.cardId].computeFree -= computeShare;
    }
    cpuPool.shareFree -= cpuShare;
    return { pooled: true, cardId: shares[0].cardId, cards: shares,
      vramGb: shares.reduce((sum, h) => sum + h.vramGb, 0), computeShare, cpuShare };
  }
  const need = c => c.shielded
    ? Math.max(vramGb, Math.ceil(computeShare * cardVram(c) / GRANULARITY_GB) * GRANULARITY_GB)
    : vramGb + ctxOverheadGb();
  if (cpuPool.shareFree < cpuShare - 1e-9) return null;
  const fit = gpuCards
    .filter(c => cardFreeVram(c) >= need(c) - 1e-9 && c.computeFree >= computeShare - 1e-9)
    .sort((a, b) => (cardFreeVram(a) - need(a)) - (cardFreeVram(b) - need(b)));
  const card = fit[0];
  if (!card) return null;
  const needV = need(card);
  if (card.shielded) vramGb = needV;
  card.vramFree -= needV; card.computeFree -= computeShare;
  cpuPool.shareFree -= cpuShare;
  return { cardId: card.id, vramGb, computeShare, cpuShare, _needV: needV };
}
function releaseGpu(h) {
  if (!h) return;
  if (h.cpu) { cpuPool.shareFree = Math.min(1, cpuPool.shareFree + h.share); return; }
  cpuPool.shareFree = Math.min(1, cpuPool.shareFree + (h.cpuShare || 0));
  for (const slice of h.cards || [h]) {
    const card = gpuCards[slice.cardId]; if (!card) continue;
    card.vramFree = Math.min(cardVram(card), card.vramFree + slice._needV);
    card.computeFree = Math.min(1, card.computeFree + slice.computeShare);
  }
}
// largest slice a single card can still take (VRAM net of overhead; compute share)
const maxFreeVram    = () => _shieldedPool ? maxFreeGpuShare() * CARD_VRAM_GB
  : Math.max(0, ...gpuCards.map(c => cardFreeVram(c) - ctxOverheadGb()));
const maxFreeCompute = () => _shieldedPool ? Math.min(...gpuCards.map(c => c.computeFree))
  : Math.max(0, ...gpuCards.map(c => c.computeFree));
// largest GPU share a single card can still take (vram + compute must fit together)
const maxFreeGpuShare = () => !IS_GPU ? 0 : Math.max(0,
  (_shieldedPool ? Math.min : Math.max)(...gpuCards.map(c =>
    Math.min(c.computeFree, (cardFreeVram(c) - ctxOverheadGb()) / cardVram(c) || 0))));

const _applyGpu = (text) => {
  let got = 0; const totals = [];
  for (const line of text.trim().split("\n")) {
    const [idx, uuid, memMiB] = line.split(",").map(s => s.trim());
    const i = parseInt(idx, 10);
    if (gpuCards[i] && /^GPU-/.test(uuid || "")) {
      gpuCards[i].uuid = uuid; got++;
      const totalGb = parseFloat(memMiB) / 1024;
      if (totalGb > 0) { totals.push(totalGb); console.log(`[gpu] card ${i} ${uuid} (${totalGb.toFixed(0)}GB)`); }
    }
  }
  if (totals.length) adoptCardVram(round1(Math.min(...totals)), "nvidia-smi");
  return got;
};
const GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"];

// Discover card UUIDs (so GPU shares can be pinned) via local nvidia-smi when
// this process can see the card. The supervisor container has no nvidia-smi and
// the card lives in the worker/wasm-manager container, so on the CVM this is a
// best-effort no-op — card VRAM/UUIDs arrive from the manager's /health probe.
async function discoverGpus() {
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) return 0;
  try { const { stdout } = await pexec("nvidia-smi", GPU_QUERY.slice(1), { timeout: 8000 });
        return _applyGpu(stdout); } catch { return 0; }
}

// Lazily ensure UUIDs are known before a GPU spawn - covers a boot where the
// card wasn't visible to nvidia-smi yet (discovery re-runs on first spawn).
let _gpuDiscovering = null;
async function ensureGpuUuids() {
  if (gpuCards.every(c => c.uuid)) return true;
  if (!_gpuDiscovering) _gpuDiscovering = discoverGpus()
    .finally(() => { _gpuDiscovering = null; });
  await _gpuDiscovering;
  return gpuCards.some(c => c.uuid);
}

async function initGpu() {
  // Best-effort at boot; if nvidia-smi can't see the card here, card VRAM/UUIDs
  // arrive from the manager's /health probe and ensureGpuUuids() retries on the
  // first spawn. Never blocks boot.
  const got = await discoverGpus();
  if (got < GPU_COUNT) console.warn(`[gpu] boot discovery ${got}/${GPU_COUNT} - will retry on first spawn`);
}

async function initMps() {
  // Start the MPS control daemon ONCE at boot. Workers join it as clients (sharing
  // MPS_PIPE_DIR) and the driver enforces, per client, BOTH the SM cap
  // (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) and the VRAM cap (CUDA_MPS_PINNED_DEVICE_MEM_LIMIT)
  // - confirmed enforced under CC via %smid. Without MPS, compute-share is unenforced
  // and we fall back to admission control + watchdog (workers still run).
  if (!ENABLE_MPS) { console.warn("[mps] disabled by env - compute-share will NOT be enforced"); return; }
  try {
    execFileSync("mkdir", ["-p", MPS_PIPE_DIR]);
    // already running? control daemon answers on the pipe dir.
    try { execFileSync("nvidia-cuda-mps-control", ["get_server_list"],
            { env: { ...process.env, CUDA_MPS_PIPE_DIRECTORY: MPS_PIPE_DIR }, stdio: "ignore" });
          console.log("[mps] daemon already running"); return; } catch {}
    execFileSync("nvidia-cuda-mps-control", ["-d"],
      { env: { ...process.env, CUDA_MPS_PIPE_DIRECTORY: MPS_PIPE_DIR } });
    console.log(`[mps] control daemon started (pipe ${MPS_PIPE_DIR})`);
  } catch (e) {
    console.warn("[mps] could not start daemon - compute caps unenforced:", e.message);
  }
}

// Public RPCs rate-limit per IP and the claim loop's bursts run into it
// (observed live 2026-07-05: "over rate limit" from mainnet.base.org killed
// whole claim passes). Longer exponential retry absorbs a burst cap; the
// per-tick call budget is kept low by deriving post-tx state from receipts.
const chainClient = createPublicClient({ chain: base, transport: rpcTransport() });

// ----------------------------------------------------------------------------
// state (in-process; this service is the single enclave instance)
// ----------------------------------------------------------------------------
const nonces     = new Map(); // nonce -> { address, exp }
const NONCE_MAX  = parseInt(process.env.NONCE_MAX || "10000", 10);   // hard cap (LRU/FIFO evict) alongside the TTL sweep, so a flood of /v1/auth/nonce can't grow this map unbounded between sweeps
const deployments = new Map(); // id -> record (incl. local container handle)
setInterval(() => { const t = Date.now(); for (const [n,v] of nonces) if (v.exp < t) nonces.delete(n); }, 60_000).unref?.();
// CSPRNG, not Math.random. This mints the SIWE nonce, and a nonce exists to be
// UNPREDICTABLE (EIP-4361 says so outright). /v1/auth/nonce is unauthenticated,
// so anyone can pull as many samples as they like from this process — and V8's
// Math.random is xorshift128+, whose state is recoverable from a handful of
// outputs. Eight base-36 chars off one draw was ~41 bits at best and zero once
// the state is solved; a predicted nonce lets an attacker prepare the exact
// message the enclave will issue for a wallet before it asks for one. Twelve
// random bytes, hex, from the same source the session key comes from. (The
// relay's store.js rid() has always done this; only this copy drifted.)
const rid = (p) => p + randomBytes(12).toString("hex");
// Constant-time string compare for secret/token checks (guards length first, as
// timingSafeEqual throws on unequal-length buffers).
function safeEqStr(a, b) {
  const ba = Buffer.from(String(a ?? ""), "utf8"), bb = Buffer.from(String(b ?? ""), "utf8");
  return ba.length === bb.length && timingSafeEqual(ba, bb);
}

// ---- state persistence (fair billing across restarts) -----------------------
// Everything billing-critical (deployments, payment cursor, dedup set) is written
// to STATE_FILE so a supervisor restart FREEZES clocks instead of forfeiting them:
// on boot the downtime gap is measured via savedAt and never charged, unpaid
// reservation windows shift by the gap, and the payment watcher resumes scanning
// from the block it last finished - payments made during the outage are credited.
// Without a writable STATE_FILE this degrades to freezing within one process only.
let _stateDirty = false, _statePersistable = true;
function serializeState() {
  const recs = [...deployments.values()].map(r => {
    const o = { ...r };
    for (const k of ["_payTimer", "_respawning", "_renewing", "_restarting"]) delete o[k];   // live handles only
    return o;
  });
  return JSON.stringify({
    savedAt: Date.now(),
    payFromBlock: _payFromBlock == null ? null : _payFromBlock.toString(),
    seenLogs: [..._seenLogs].map(([k, b]) => [k, b.toString()]),
    deployments: recs,
  });
}
function saveStateNow() {
  if (!_statePersistable) return;
  try {
    const tmp = STATE_FILE + ".tmp";
    writeFileSync(tmp, serializeState());
    renameSync(tmp, STATE_FILE);                 // atomic: a crash never leaves a torn file
    _stateDirty = false;
  } catch (e) { console.warn(`[state] save failed: ${e.message}`); }
}
function saveStateSoon() { _stateDirty = true; }
function initStatePersistence() {
  try { mkdirSync(dirname(STATE_FILE), { recursive: true }); }
  catch (e) {
    _statePersistable = false;
    console.warn(`[state] ${STATE_FILE} unavailable (${e.message}) - clocks freeze only within this process`);
    return;
  }
  const t = setInterval(() => { if (_stateDirty) saveStateNow(); }, 2000);
  if (t.unref) t.unref();
  // flush on shutdown so savedAt marks the true start of the outage. On-chain
  // leases are released first (bounded wait): a clean shutdown refunds the
  // unused lease tail and reopens the queue immediately; instant no-op when
  // this enclave holds no claims.
  for (const sig of ["SIGTERM", "SIGINT"]) process.on(sig, () => {
    saveStateNow();
    releaseClaimsOnShutdown().finally(() => process.exit(0));
  });
}
// Statuses that hold NO resources: every flip into one of these releases the
// record's slice in the same synchronous block (all sites verified 2026-07-18),
// which is what makes them safe markers for the restore guard and the pool
// reconciler below. Also exactly the set a claim may re-adopt over -
// CLAIM_TERMINAL (defined near the claim path) aliases this.
const TERMINAL_STATUSES = new Set(["expired", "failed", "terminated", "stopping"]);

function loadState() {
  if (!_statePersistable || !existsSync(STATE_FILE)) return;
  let s; try { s = JSON.parse(readFileSync(STATE_FILE, "utf8")); }
  catch (e) { console.warn(`[state] unreadable (${e.message}) - starting fresh`); return; }
  const gapMs = Math.max(0, Date.now() - (s.savedAt || Date.now()));
  if (s.payFromBlock != null) _payFromBlock = BigInt(s.payFromBlock);   // watcher resumes where it stopped
  for (const [k, b] of s.seenLogs || []) _seenLogs.set(k, BigInt(b));
  let running = 0, waiting = 0;
  for (const r of s.deployments || []) {
    r._payTimer = null; r._respawning = false; r._restarting = false;
    // legacy terminal status: "stopping" was set AFTER teardown completed and
    // nothing ever finalized it, so restored records sat "stopping" forever
    if (r.status === "stopping") r.status = "terminated";
    // a record persisted MID-PROVISION (claimed/provisioning/…) has no
    // instance to resume and nothing downstream ever finalizes it - restored
    // as-is it would hold its slice forever AND block its own id's re-claim
    // ("already serving it here"). Expire it: expired holds nothing, and the
    // sweep legitimately re-adopts an expired id while its lease is still ours.
    if (!TERMINAL_STATUSES.has(r.status) && r.status !== "running" && r.status !== "awaiting_payment") {
      r.status = "expired"; r.error = r.error || "interrupted mid-provision by a restart";
    }
    deployments.set(r.id, r);
    if (r.payRef) payRefIndex.set(r.payRef.toLowerCase(), r.id);
    // terminal records never hold resources (see TERMINAL_STATUSES) - a
    // persisted handle on one is crash drift; drop it instead of re-reserving
    // a slice nothing would ever release (the 2026-07-18 kryptos leak)
    if (r._gpu && TERMINAL_STATUSES.has(r.status)) r._gpu = null;
    if (r._gpu) {                       // re-reserve the slices this deployment still holds
      if (r._gpu.cpu) cpuPool.shareFree = Math.max(0, cpuPool.shareFree - r._gpu.share);
      else {
        cpuPool.shareFree = Math.max(0, cpuPool.shareFree - (r._gpu.cpuShare || 0));
        for (const h of r._gpu.cards || [r._gpu]) {
          const card = gpuCards[h.cardId];
          if (card) { card.vramFree -= h._needV; card.computeFree -= h.computeShare; }
        }
      }
    }
    if (r.status === "running") {
      // FREEZE the outage: the gap between savedAt and now is never charged. The
      // first healthy tick verifies the instance (respawning it if the backend
      // lost it) and resumes the clock.
      r._lastTickAt = Date.now();
      r.paused = true; r.pauseReason = "restart_recovery";
      r._respawnAt = 0; r._respawnBackoffMs = 0;
      running++;
    } else if (r.status === "awaiting_payment") {
      r.payDeadline = (r.payDeadline || Date.now()) + gapMs;   // reservation window frozen too
      armPayTimer(r);
      waiting++;
    }
  }
  if (deployments.size) console.log(`[state] restored ${deployments.size} deployment(s) `
    + `(${running} running, ${waiting} awaiting payment) after ${Math.round(gapMs / 1000)}s down; clocks were frozen`);
}

// ---- pool reconciliation: the RECORDS are the truth, the pools are a cache --
// Every alloc/release pair should keep gpuCards/cpuPool exactly in step with
// the deployments map, but one missed release - a crash between paths, a bug
// like the 2026-07-18 kryptos leak (~27 GB of the card held by dead work) -
// shrinks sellable capacity FOREVER: the claim gauntlet, /availability and the
// deploy page all read these pools, and nothing ever put the slice back. So
// rebuild free from first principles on a cadence: total minus what
// non-terminal records actually hold. Safe because every allocGpu/allocCpu ->
// deployments.set attach is synchronous (no await between), so a pass can
// never see a half-attached slice; a terminal record still holding a handle is
// itself drift (terminal flips release synchronously) and is dropped here.
// Corrects BOTH directions: leaked holds (under-free) and double releases
// (over-free, which would oversell the card).
function reconcilePools() {
  const held = { cpu: 0, cards: gpuCards.map(() => ({ vram: 0, compute: 0 })) };
  for (const r of deployments.values()) {
    const h = r._gpu;
    if (!h) continue;
    if (TERMINAL_STATUSES.has(r.status)) { r._gpu = null; saveStateSoon(); continue; }
    if (h.cpu) { held.cpu += h.share; continue; }
    held.cpu += h.cpuShare || 0;
    for (const slice of h.cards || [h]) {
      const c = held.cards[slice.cardId];
      if (c) { c.vram += (slice._needV != null ? slice._needV : slice.vramGb + ctxOverheadGb()); c.compute += slice.computeShare; }
    }
  }
  const fixed = [];
  const apply = (obj, key, want, label, fmt) => {
    if (Math.abs(obj[key] - want) <= 1e-6) return;
    fixed.push(`${label} ${fmt(obj[key])}->${fmt(want)}`);
    obj[key] = want;
  };
  apply(cpuPool, "shareFree", Math.max(0, Math.min(1, 1 - held.cpu)), "cpu share", round3);
  gpuCards.forEach((card, i) => {
    apply(card, "vramFree", Math.max(0, Math.min(cardVram(card), cardVram(card) - held.cards[i].vram)), `card${i} vramGb`, round1);
    apply(card, "computeFree", Math.max(0, Math.min(1, 1 - held.cards[i].compute)), `card${i} compute`, round3);
  });
  if (fixed.length) console.warn(`[pool] reconciled drift - dead reservations reclaimed (${fixed.join(", ")})`);
  return fixed;
}
function startPoolReconciler() {
  reconcilePools();          // boot pass: restored drift dies before the first claim/availability answer
  const t = setInterval(reconcilePools, 60_000);
  if (t.unref) t.unref();
}

// POOL_SELFTEST='{"cardVramGb":140.4,"cards":[{"vramFree":86.4,"computeFree":0.64}],
//   "cpuShareFree":0.8,"records":[{"id":"a","status":"running","_gpu":{…}}]}'
// applies the scenario, runs reconcilePools TWICE (the second must be a
// no-op), prints one JSON line and exits - same contract as the seams above
// (test/pool-reconcile.test.mjs drives it).
if (process.env.POOL_SELFTEST) {
  const c = JSON.parse(process.env.POOL_SELFTEST);
  if (c.cardVramGb > 0) CARD_VRAM_GB = c.cardVramGb;
  if (c.shielded) { _shieldedAdopted = true; IS_GPU = true; }
  gpuCards.length = 0;
  (c.cards || []).forEach((k, i) => gpuCards.push({ id: i, uuid: null, vramFree: k.vramFree, computeFree: k.computeFree }));
  if (c.cpuShareFree != null) cpuPool.shareFree = c.cpuShareFree;
  for (const r of c.records || []) deployments.set(r.id, r);
  const fixed = reconcilePools(), fixedAgain = reconcilePools();
  console.log(JSON.stringify({ fixed, fixedAgain,
    cpuShareFree: round3(cpuPool.shareFree),
    cards: gpuCards.map((k) => ({ vramFree: round1(k.vramFree), computeFree: round3(k.computeFree) })),
    maxFreeGpuShare: round3(maxFreeGpuShare()),
    dropped: [...deployments.values()].filter((r) => !r._gpu).map((r) => r.id) }));
  process.exit(0);
}

// SIZING_SELFTEST='{"cardVramGb":6.5,"cardTflops":42.7,"isGpu":true,"volGb":25.94,
//   "min":{"vramMb":51200,"gpuGflops":320000,"memMb":4096,"cpuGflops":10,"gpuOptional":true},
//   "gpuMilli":360,"cpuMilli":120,"envelopeOptional":false}'
// runs the REAL minSharesOf + gpuRouting for one app on one box's hardware and
// prints the verdict as one JSON line, then exits — same contract as the seams
// above (test/sizing.test.mjs drives it). This exists because the console
// mirrors this math in site/js/core/pricing.js and a mirror that drifts is
// exactly how a deployment gets sold that no runner will claim: the parity test
// compares the two implementations, not two copies of one.
if (process.env.SIZING_SELFTEST) {
  const c = JSON.parse(process.env.SIZING_SELFTEST);
  if (c.cardVramGb > 0) CARD_VRAM_GB = c.cardVramGb;
  if (c.cardTflops > 0) CARD_TFLOPS = c.cardTflops;
  if (c.isGpu != null) IS_GPU = !!c.isGpu;
  // A shielded card is a runtime adoption, not a boot constant: the seam must be
  // able to set it so the parity test can prove the "serves any size" branch.
  if (c.shielded) { _shieldedAdopted = true; IS_GPU = true; }
  const mins = minSharesOf(c.min || {}, { volGb: c.volGb || 0 });
  const gpuShare = Number(c.gpuMilli || 0) / 1000, cpuShare = Number(c.cpuMilli || 0) / 1000;
  const declaresGpu = (c.min?.vramMb || 0) > 0 || (c.min?.gpuGflops || 0) > 0;
  const r = gpuShare > 0
    ? gpuRouting(mins, { declaresGpu, envelopeOptional: !!c.envelopeOptional })
    : { onGpu: false, unmet: null, refusal: null };
  const onGpu = gpuShare > 0 && r.onGpu;
  const needGpu = onGpu ? mins.gpuFloor : 0;
  const needCpu = mins.cpuShare;
  const below = !r.refusal && (gpuShare < needGpu - 1e-9 || cpuShare < needCpu - 1e-9);
  console.log(JSON.stringify({
    gpuShare: round3(mins.gpuShare), gpuFloor: round3(mins.gpuFloor),
    cpuShare: round3(mins.cpuShare),
    gpuOptional: mins.gpuOptional,
    onGpu, asCpuFallback: gpuShare > 0 && !r.onGpu && !r.refusal,
    unmet: r.unmet, refusal: r.refusal,
    needGpu: round3(needGpu), needCpu: round3(needCpu), below,
  }));
  process.exit(0);
}

// ---- instance reconciliation: the RECORDS are the truth, the BACKEND's
//      instances are a cache ------------------------------------------------
// One layer below reconcilePools. That sweep fixes OUR accounting of what our
// own records hold; it cannot see an instance the backend is still running for
// a record we no longer have. Every teardown path ends in stopContainer, whose
// contract is explicitly best-effort over HTTP - and twelve of its thirteen
// callers discard the boolean, because they MUST proceed regardless (a lease
// that ended has to be released on-chain whether or not the local stop
// confirmed). So an unconfirmed stop left the process alive, holding its whole
// slice, with nothing on either side able to see it: the supervisor had
// dropped the record, and the manager has no owner-liveness check of its own.
//
// Live proof (2026-08-03, kryptos): an owner's setActive(false) released the
// lease on-chain and dropped the record, but the DELETE never landed. The
// wasm-manager kept the tenant "running" - 4% of the share pool and 17.6 GB of
// resident ggml weights, 35% of the node's sellable RAM - against a deployment
// that no longer existed. Nothing reclaimed it, and nothing could: the state is
// in-memory, so only a manager restart would have cleared it. Same class as the
// 2026-07-25 leak quoted in stopContainer's own comment.
//
// The fix is structural rather than per-call-site: reconcile the backend's
// instance list against our records on a cadence, and tear down anything we do
// not own. That also covers the cases a call-site fix cannot - a supervisor
// crash between stop and release, a restart that expired a mid-provision
// record, a manager that outlived us entirely.
//   The listing itself is load-bearing beyond the plan it feeds: the manager
// re-checks each record's process on every /vms read (_public -> _refresh_status)
// and its RAM ledger counts only starting/running, but nothing READ on a
// cadence - so a tenant that died on its own kept its bytes committed until
// something happened to ask. Polling this every 60s keeps that ledger honest
// too, which is why the sweep runs even when it reaps nothing.
const INSTANCE_REAP_MIN_AGE_SEC = Math.max(30, parseInt(process.env.INSTANCE_REAP_MIN_AGE_SEC || "120", 10));

// PURE core. Which backend instances are orphans, given our records.
// Ownership is keyed on the DEPLOYMENT ID, which both backends carry: the vm
// manager mints its own random vid ("app_1a2b3c4d5") and echoes the deployment
// id as `name`, while the worker manager keys tenants by the deployment id
// itself. Keying on the id (not the vid) is what closes the provisioning race:
// a record enters `deployments` at reservation, long before spawnContainer
// returns the vid we would otherwise match on, so an in-flight launch is
// always recognised as owned.
//   `minAgeSec` is belt-and-braces on top of that: an instance younger than it
// is never reaped, so even a backend that ignores `name`, or a launch path
// that somehow registers late, gets a grace window instead of a kill. An
// instance with no usable createdAt is reaped on age grounds alone being
// unknowable - it is old enough to have predated the field.
//   Terminal records own nothing (TERMINAL_STATUSES: every flip into one
// releases the slice synchronously), so an instance still running for one is
// exactly the leak this sweep exists to catch.
function orphanInstancePlan(instances, records, { nowMs, minAgeSec }) {
  const owned = new Set();
  for (const r of records || []) {
    if (r && r.id != null && !TERMINAL_STATUSES.has(r.status)) owned.add(String(r.id));
  }
  const plan = [];
  for (const v of instances || []) {
    if (!v) continue;
    const vmId = String(v.id ?? "");
    const key = String(v.name || v.id || "");
    if (!vmId || !key || owned.has(key)) continue;
    const created = Number(v.createdAt) * 1000;
    if (Number.isFinite(created) && created > 0 && nowMs - created < minAgeSec * 1000) continue;
    plan.push({ vmId, id: key });
  }
  return plan;
}

// INSTANCE_SELFTEST='{"instances":[{"id":"app_1","name":"0xabc","createdAt":1}],
//   "records":[{"id":"0xabc","status":"running"}],"nowMs":…,"minAgeSec":120}'
// prints the reap plan as one JSON line and exits - same contract as the seams
// above (test/instance-reconcile.test.mjs drives it).
if (process.env.INSTANCE_SELFTEST) {
  const c = JSON.parse(process.env.INSTANCE_SELFTEST);
  console.log(JSON.stringify(orphanInstancePlan(c.instances || [], c.records || [], {
    nowMs: c.nowMs != null ? c.nowMs : Date.now(),
    minAgeSec: c.minAgeSec != null ? c.minAgeSec : INSTANCE_REAP_MIN_AGE_SEC })));
  process.exit(0);
}

// List what the backend is actually running. Returns null - meaning "no
// answer", never "nothing runs" - when the manager cannot be reached or
// answers a shape we don't recognise. A failed listing MUST NOT read as an
// empty fleet: that would turn one unreachable manager into a sweep that tears
// down every live tenant on the box.
async function listBackendInstances() {
  if (PROVISION_BACKEND === "vm") {
    // 30s, not the 10s this shipped with: GET /vms serves the whole list under
    // the manager's _lock (and _public re-stats every encrypted volume), so it
    // queues behind any launch/teardown/audit holding it. A timeout here is not
    // a slow answer, it is a SKIPPED PASS - the sweep must be patient enough
    // that contention never reads as "nothing to reconcile".
    const r = await vmReq("GET", "/vms", null, 30_000);
    return r.status === 200 && Array.isArray(r.body && r.body.vms) ? r.body.vms : null;
  }
  const r = await mgrReq("GET", "/tenants", null, 30_000);
  return r.status === 200 && Array.isArray(r.body && r.body.tenants) ? r.body.tenants : null;
}

// ---- the heartbeat half: vouch for what we still own ------------------------
// The sweep above is a PUSH - we decide a tenant should die and send a DELETE -
// so it fails open on every link: a lost DELETE, an unreachable manager, a
// supervisor that crashes between stopping and releasing. That is how the same
// tenant stranded twice. The manager therefore runs a dead-man lease and we
// hold the switch: name the tenants we still own, every pass, and anything we
// stop naming the manager reaps on its own. We never have to successfully
// instruct a teardown - we only have to stop vouching, which is also exactly
// what a crashed supervisor does for free.
//
// WHICH records vouch is the whole safety question. Non-terminal is necessary
// but not sufficient: a record the chain no longer backs would otherwise be
// vouched for forever, which is the one shape the sweep also cannot catch (it
// sees an owned instance). On a box that claims on-chain work, an OFF-LEDGER
// record vouches for nothing - there is no such thing as legitimate off-ledger
// tenancy here. AUTO_PROVISION is the deliberate exception: a pilot/manual
// box has no chain to appeal to, so local records are the only truth it has.
//   Note what is deliberately NOT required: a live on-chain lease read. The
// audit already keeps serving through an unreadable ledger ("the lease is
// prepaid"), and making the heartbeat hostage to RPC availability would let an
// RPC outage reap paid, running work. Chain-BACKED is the test, not chain-live.
function vouchesForLease(rec) {
  if (TERMINAL_STATUSES.has(rec.status)) return false;
  return rec._onchain ? true : AUTO_PROVISION === true;
}
// VOUCH_SELFTEST='{"records":[{"id":"0xabc","status":"running","_onchain":true},…]}'
// prints the ids that would be vouched for as one JSON line and exits - same
// contract as the seams above (test/tenant-lease.test.mjs drives it).
if (process.env.VOUCH_SELFTEST) {
  const c = JSON.parse(process.env.VOUCH_SELFTEST);
  console.log(JSON.stringify((c.records || []).filter(vouchesForLease).map((r) => String(r.id))));
  process.exit(0);
}
let _lastVouch = { at: null, ok: null, vouched: 0, reason: "not yet run" };
async function vouchTenants() {
  if (PROVISION_BACKEND !== "vm") return;   // worker backend has no lease route
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) return;
  const ids = [...deployments.values()].filter(vouchesForLease).map((r) => String(r.id));
  try {
    const r = await vmReq("POST", "/vms/lease", { ids }, 15_000);
    const ok = r.status === 200;
    _lastVouch = { at: new Date().toISOString(), ok, vouched: ids.length,
                   reason: ok ? "ok" : `HTTP ${r.status}`,
                   ...(Array.isArray(r.body && r.body.unvouched) && r.body.unvouched.length
                       ? { unvouched: r.body.unvouched } : {}) };
    if (ok && r.body && r.body.unvouched && r.body.unvouched.length) {
      console.warn(`[lease] ${r.body.unvouched.length} tenant(s) we do not own are running on the `
                 + `backend and will be reaped when their lease lapses: `
                 + r.body.unvouched.map((u) => `${u.id} (${u.expiresIn}s)`).join(", "));
    }
  } catch (e) {
    // A failed heartbeat is SAFE by construction: the manager only enforces
    // while it is hearing from us, so silence suspends the lease rather than
    // triggering it. Recorded because a heartbeat that never lands means the
    // dead-man switch is not actually armed, which must be visible.
    _lastVouch = { at: new Date().toISOString(), ok: false, vouched: ids.length, reason: e.message };
    console.warn(`[lease] heartbeat failed (${e.message}) - the manager suspends enforcement while silent`);
  }
}

// Last pass's outcome, surfaced on /availability. A reconciler with no way to
// report itself is one you can only debug with shell access to the box: the
// 2026-08-03 recurrence cost hours precisely because "the sweep ran and found
// nothing" and "the sweep never completed a pass" look identical from outside.
// Anything that can delete a tenant must be able to say what it did.
let _instanceSweep = { at: null, ok: null, reason: "not yet run", seen: 0, orphans: 0, reaped: 0 };
function instanceSweepStatus() { return { ..._instanceSweep }; }

async function reconcileInstances() {
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) return [];
  const stamp = (o) => { _instanceSweep = { at: new Date().toISOString(), ...o }; };
  // Vouch FIRST, and unconditionally. The heartbeat is what keeps legitimate
  // tenants alive, so it must never be skipped by an early return below - a
  // listing failure that also suppressed vouching would turn one unreadable
  // GET into every tenant on the box lapsing.
  await vouchTenants();
  let instances;
  try {
    instances = await listBackendInstances();
  } catch (e) {
    console.warn(`[instance] backend listing failed (${e.message}) - skipping this pass`);
    stamp({ ok: false, reason: `listing failed: ${e.message}`, seen: 0, orphans: 0, reaped: 0 });
    return [];
  }
  if (instances === null) {
    console.warn("[instance] backend listing unavailable - skipping this pass");
    stamp({ ok: false, reason: "listing unavailable (non-200 or unexpected shape)", seen: 0, orphans: 0, reaped: 0 });
    return [];
  }
  const plan = orphanInstancePlan(instances, [...deployments.values()],
                                  { nowMs: Date.now(), minAgeSec: INSTANCE_REAP_MIN_AGE_SEC });
  const reaped = [];
  for (const o of plan) {
    const path = PROVISION_BACKEND === "vm"
      ? `/vms/${encodeURIComponent(o.vmId)}` : `/tenants/${encodeURIComponent(o.vmId)}`;
    try {
      const r = PROVISION_BACKEND === "vm"
        ? await vmReq("DELETE", path, null, 30_000) : await mgrReq("DELETE", path, null, 30_000);
      if (r.status === 200 || r.status === 404) {
        reaped.push(o.id);
        console.warn(`[instance] reaped orphan ${o.id} (vm=${o.vmId}) - no record owns it; `
                   + "its slice and any resident model weights are back in the pool");
      } else {
        console.warn(`[instance] orphan ${o.id} (vm=${o.vmId}): DELETE HTTP ${r.status} - retrying next pass`);
      }
    } catch (e) {
      console.warn(`[instance] orphan ${o.id} (vm=${o.vmId}): ${e.message} - retrying next pass`);
    }
  }
  // A pass that found orphans it could not delete is NOT ok - that is the leak
  // still leaking, and it must not read as a clean sweep.
  stamp({ ok: reaped.length === plan.length, seen: instances.length,
          orphans: plan.length, reaped: reaped.length,
          reason: plan.length === 0 ? "clean"
                : reaped.length === plan.length ? `reaped ${reaped.length}`
                : `${plan.length - reaped.length} orphan(s) survived DELETE`,
          // WHICH instances the backend is running and who we think owns them:
          // the single fact that would have ended the 2026-08-03 recurrence in
          // one request instead of an evening of inference.
          held: instances.slice(0, 20).map((v) => ({
            id: String(v.name || v.id || ""), vm: String(v.id ?? ""), status: v.status || null,
            owned: !plan.some((o) => o.vmId === String(v.id ?? "")) })) });
  return reaped;
}

// Debounced kick, so an unconfirmed stop converges in seconds rather than
// waiting out the cadence. The pass itself is idempotent, so a coalesced burst
// of failures costs one sweep.
let _instanceSweepTimer = null;
function reconcileInstancesSoon(delayMs = 5_000) {
  if (_instanceSweepTimer) return;
  _instanceSweepTimer = setTimeout(() => {
    _instanceSweepTimer = null;
    reconcileInstances().catch((e) => console.warn(`[instance] sweep failed: ${e.message}`));
  }, delayMs);
  if (_instanceSweepTimer.unref) _instanceSweepTimer.unref();
}

// INSTANCE_SWEEP_SELFTEST='{"records":[{"id":"0xabc","status":"running"},…]}'
// runs ONE REAL reconcileInstances() pass against whatever VMMGR_URL/MGR_URL
// point at (a stub manager in the tests), prints the reaped ids as one JSON
// line and exits. Unlike INSTANCE_SELFTEST above - which drives the pure
// planner - this exercises the part that actually issues DELETEs, including
// the fail-closed rule that an unreadable listing reaps NOTHING.
// Top-level await, so the rest of the module never evaluates - the seam must
// not boot a server or bind a port, exactly like the synchronous seams above.
if (process.env.INSTANCE_SWEEP_SELFTEST) {
  const c = JSON.parse(process.env.INSTANCE_SWEEP_SELFTEST);
  for (const r of c.records || []) deployments.set(r.id, r);
  try {
    const reaped = await reconcileInstances();
    console.log(JSON.stringify({ reaped, sweep: instanceSweepStatus() }));
    process.exit(0);
  } catch (e) { console.log(JSON.stringify({ error: e.message })); process.exit(1); }
}

function startInstanceReconciler() {
  // The boot pass is DELAYED, unlike the pool reconciler's. That one reads our
  // own restored records; this one can delete another process's work, and at
  // t=0 loadState has run but nothing has re-registered or re-claimed yet. The
  // delay also keeps the min-age window meaningful across a supervisor restart.
  const first = setTimeout(() => {
    reconcileInstances().catch((e) => console.warn(`[instance] boot sweep failed: ${e.message}`));
  }, Math.max(30_000, INSTANCE_REAP_MIN_AGE_SEC * 1000 / 2));
  if (first.unref) first.unref();
  const t = setInterval(() => {
    reconcileInstances().catch((e) => console.warn(`[instance] sweep failed: ${e.message}`));
  }, 60_000);
  if (t.unref) t.unref();
}

// ============================================================================
// >>> IMPLEMENT THESE for your CVM launch mechanism (e.g. the app manager on
//     VMMGR_URL). Contract: one ingress port, no sibling reach.
//     Tinfoil exposes no guest RTMR-extend, so a launched image's digest cannot
//     be folded into the hardware measurements; /attestation reports exactly
//     that (per-app `coverage` in getMeasurements) instead of implying it.
// ============================================================================
// ============================================================================
// WORKER LAUNCH - one container per tenant. The process boundary is the ONLY
// thing giving memory isolation + fault containment + VRAM scrub-on-exit at once
// (all empirically confirmed). Compute + VRAM are capped by MPS, also confirmed
// enforced under CC. Never co-locate two tenants in one process.
//   Image digests are NOT RTMR-extended (no guest extend interface) - the
//   attestation endpoint reports that coverage gap explicitly, never fakes it.
// ============================================================================
// resolve the pinned image ref: prefer name@sha256:digest when a digest is given
function pinnedRef(image) {
  const ref = (image?.reference || DEFAULT_IMAGE).trim();
  const dig = (image?.digest || "").trim();
  if (ref.includes("@")) return ref;                              // already digest-pinned
  if (/^sha256:[0-9a-f]{64}$/i.test(dig)) return `${ref.replace(/:[^/:]+$/, "")}@${dig}`;
  return ref;                                                     // tag-only (pin verification is the attestation step)
}
function toBytes(s) {
  const m = /^(\d+)\s*([gmk]?)b?$/i.exec(String(s).trim());
  if (!m) return 0;
  const n = +m[1], u = m[2].toLowerCase();
  return u === "g" ? n*1073741824 : u === "m" ? n*1048576 : u === "k" ? n*1024 : n;
}
// EVERY input a launch hands the guest, derived in ONE place, because there are
// two launch sites (provisionTenant and respawnTenant) and they must not
// disagree about what a launch is.
//
// They disagreed for three weeks. Per-deployment secrets (and later the
// deployment's own hostnames) were wired into provisionTenant only;
// respawnTenant predates both and kept its own literal spawn argument object,
// so it relaunched tenants with their $NAME config placeholders LITERAL and no
// ENCLAVE_HOSTS. Both paths recover a died app or a restarted manager — the
// billing ticker every 15s, the claim loop's crash recovery every 60s — so
// they RACE, and which one gets there decides whether the app comes back with
// its secrets. That is why it read as intermittent and unreproducible.
//
// Nothing said so, either: the relay logs a fetch, and a respawn never asked
// for one, so the evidence read as "the enclave never even requested its
// secrets" and pointed a two-day investigation at egress. A respawn IS a
// launch. Same inputs.
//
// Pure on purpose (the caller does the awaiting): the seam below pins the
// contract, and `launchSpec` is the only thing that builds one.
function launchSpecFrom(rec, sec, hosts) {
  return { deploymentId: rec.id,
    cardId: rec._gpu?.cardId ?? rec.resources?.cardId ?? 0,
    gpuVramGb: rec._gpu?.vramGb,
    gpuCardsHeld: rec._gpu?.cards,
    gpuShare: rec.resources.gpuShare || 0, cpuShare: rec.resources.cpuShare,
    image: { reference: rec.appWasm || (rec.image && rec.image.reference) },
    appPort: rec.network.port, ports: rec.firewall,
    config: rec.config || "", configCid: rec.appConfigCid || "",
    secrets: sec && Object.keys(sec.env).length ? sec.env : null,
    hosts: hosts || [] };
}

// LAUNCH_SPEC_SELFTEST='{"cases":[{"rec":{…},"sec":{"rev":1,"env":{…}},"hosts":[…]},…]}'
// prints each spec as one JSON line with the secret NAMES in place of the map
// (values never travel through a log or a test fixture) and exits — same seam
// contract as CFG_EDIT_SELFTEST et al.
if (process.env.LAUNCH_SPEC_SELFTEST) {
  const c = JSON.parse(process.env.LAUNCH_SPEC_SELFTEST);
  console.log(JSON.stringify((c.cases || []).map(({ rec, sec, hosts }) => {
    const spec = launchSpecFrom(rec, sec, hosts);
    return { ...spec, secrets: spec.secrets ? Object.keys(spec.secrets).sort() : null };
  })));
  process.exit(0);
}
async function spawnContainer({ deploymentId, gpuShare, cpuShare, cardId, gpuVramGb, gpuCardsHeld, image, appPort, ports, config, configCid, secrets, hosts }) {
  // Two backends. "vm": hand the app reference to the app manager on VMMGR_URL
  // (the wasm-manager runs it as a `wasmtime serve` process; cpuShare is its
  // admission unit and sets the guest memory cap — cpuShare × node RAM;
  // gpuShare buys the wasi-nn GPU interface: the manager launches the tenant
  // with `-S nn` and MPS-caps its process at gpuShare SM% / gpuShare × VRAM.
  // The compute the shares grant is passed too - GPU in TFLOPS, CPU in
  // GFLOPS - so the manager can enforce catalog compute minimums).
  // "worker": fork an MPS-capped CUDA child PROCESS (GPU PTX submission);
  // gpuShare sets the MPS cap.
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) {
    console.log(`[mock] ${PROVISION_BACKEND} tenant ${deploymentId}`);
    return { internalPort: 0 };
  }

  if (PROVISION_BACKEND === "vm") {
    const ref = image && image.reference;
    if (!ref) throw new Error("VM backend requires an image reference.");
    const c = (cpuShare != null) ? cpuShare : 0.05, g = gpuShare || 0;
    // A shielded card is reached over the masked-offload protocol, not through a
    // local CUDA device, so the manager must NOT try to MPS-cap this tenant: there
    // is no device to cap and no MPS pipe to join. It gets the worker's address
    // instead, and the tenant's engine loads the shielded ggml backend. Sending
    // the flag unconditionally is safe -- a manager that predates it ignores an
    // unknown field, and on such a manager gpuShare would simply fail to launch,
    // which is the correct outcome rather than a silently unshielded tenant.
    const shieldedCard = g > 0 ? shieldedCapacity(_shieldedPool ? null : cardId ?? 0) : null;
    const shieldedSpec = g > 0 && _shieldedAdopted ? shieldedLaunchSpec(cardId, g, gpuVramGb, gpuCardsHeld) : null;
    const r = await vmReq("POST", "/vms",
      { image: ref, cpuShare: c, gpuShare: g,
        ...(shieldedSpec ? { shielded: shieldedSpec } : {}),
        gpuTflops: round1(g * (shieldedCard?.cardTflops || CARD_TFLOPS)), cpuGflops: Math.round(c * NODE_GFLOPS),
        // cpuTflops: legacy field for managers pinned before the GFLOPS switch
        cpuTflops: round3(c * NODE_GFLOPS / 1000),
        appPort: appPort || 8080, name: deploymentId, ports: ports || [],
        // the approved version's config JSON, verbatim from the catalog record
        // (already validated by the publish path; the manager re-parses and
        // passes it to the tenant as ENCLAVE_CONFIG; empty = app defaults)
        config: config || "",
        // rev-7 large config: the manager fetches this CID and accepts the
        // bytes only because they re-hash to it. When set, `config` above is
        // the routing manifest, not what the guest receives. A manager that
        // predates the field ignores it — which is why the claim path gates on
        // the fleet advertising config_cid before taking such a deployment.
        configCid: configCid || "",
        // dedicated-IP egress: a per-deployment SOCKS URL the manager forwards
        // verbatim as the guest's ENCLAVE_EGRESS (empty when egress is off). The
        // token in it is minted from the enclave SECRET, so the manager never
        // needs the secret and the value never touches a log line.
        egress: egress ? egress.envFor(deploymentId) : "",
        // relay-staged owner secrets (fetchDepSecrets): guest-only --env vars,
        // validated by the manager; like config/egress, never in a log line
        ...(secrets ? { secrets } : {}),
        // every hostname this deployment answers on (its own subdomain + any
        // verified custom domain) -> the guest's ENCLAVE_HOSTS. A manager that
        // predates the field ignores it, so the app simply doesn't get the
        // list — nothing else changes.
        ...(hosts && hosts.length ? { hosts: hosts.join(",") } : {}) }, SPAWN_TIMEOUT_MS);
    if (r.status !== 201)
      throw new Error(`vmmanager: ${r.body.error || r.body.message || r.status}`);
    console.log(`[spawn-vm] ${deploymentId} image=${ref} cpuShare=${c} gpuShare=${g}`
              + (shieldedCard ? ` shielded=${shieldedCard.endpoint}` : "")
              + ` vm=${r.body.id} hostPort=${r.body.hostPort} status=${r.body.status}`);
    // The VM boots asynchronously; the data path 502s until its server is up.
    // status carries the manager's state.
    return { internalPort: r.body.hostPort || 0, vmId: r.body.id, hostPort: r.body.hostPort,
             portMap: r.body.portMap || {}, status: r.body.status };   // logical "tcp:5432" -> actual loopback bind
  }

  // worker backend (GPU)
  if (!(gpuShare > 0)) throw new Error("The worker backend serves GPU deployments only (gpuShare > 0 required).");
  const g = Math.min(1, Math.max(MIN_COMPUTE_PCT / 100, gpuShare));
  const r = await mgrReq("POST", "/tenants", { id: deploymentId, gpuShare: g }, SPAWN_TIMEOUT_MS);
  if (r.status !== 201 || r.body.status !== "running")
    throw new Error(`worker manager: ${r.body.error || r.body.status || r.status} `
                  + `(sm_granted=${r.body.sm_granted ?? "?"})`);
  console.log(`[spawn] tenant=${deploymentId} gpuShare=${g.toFixed(3)} `
            + `sm_granted=${r.body.sm_granted} device=${r.body.device}`);
  return { internalPort: 0, smGranted: r.body.sm_granted };
}

// Tear down the tenant's instance. Returns true when the instance is
// verifiably gone (deleted now, or no longer known); false when the teardown
// could not be confirmed. A caller about to re-provision onto the same
// capacity MUST treat false as "the old process may still hold its VRAM/RAM"
// and defer - an unconfirmed stop followed by a fresh launch leaves TWO
// processes on one slice (live 2026-07-25: an upgrade's leaked 27b tenant
// kept ~30 GiB of weights+KV pinned and the replacement failed context
// allocation against memory it could not see).
//   Callers that cannot defer - a lease that ended must be released on-chain
// whether or not the local stop confirmed - are covered by the instance
// reconciler instead: every false here schedules a sweep, which finds the
// instance unowned once the record goes terminal and tears it down for real.
// So an unconfirmed stop is now a LATENCY, never a permanent leak.
function stopUnconfirmed(rec, why) {
  console.warn(`[stop] ${rec.id}: UNCONFIRMED (${why}) - the instance may still hold its slice; `
             + "handing off to the instance reconciler");
  reconcileInstancesSoon();
  return false;
}
async function stopContainer(rec) {
  if (PROVISION_BACKEND === "vm") {
    if (!rec._vmId) return true;               // nothing was ever provisioned
    try {
      const r = await vmReq("DELETE", `/vms/${encodeURIComponent(rec._vmId)}`);
      if (r.status === 200 || r.status === 404) return true;
      return stopUnconfirmed(rec, `HTTP ${r.status}`);
    } catch (e) { return stopUnconfirmed(rec, e.message); }
  }
  // Tear down the tenant's MPS-capped child. The manager terminates the process,
  // which returns its context/VRAM to the driver and releases the share. NOTE:
  // freed VRAM is not zeroed here - residual-data scrubbing is Layer 4.
  try {
    const r = await mgrReq("DELETE", `/tenants/${encodeURIComponent(rec.id)}`);
    if (r.status === 200 || r.status === 404) return true;
    return stopUnconfirmed(rec, `HTTP ${r.status}`);
  } catch (e) { return stopUnconfirmed(rec, e.message); }
}
// What app ran, as an attestation-visible identity. For ipfs://<cid> the CID IS a
// content hash the (attested) wasm-manager verified the bytes against before running,
// so reporting it here is honest: "the enclave ran exactly this CID."
function appMeasurement(rec) {
  const ref = (rec.image && rec.image.reference) || null;
  const m = /^ipfs:\/\/([^/?#]+)/.exec(ref || "");
  return m ? { kind: "ipfs", reference: ref, cid: m[1], verifiedAgainstCid: true,
               coverage: "Bytes were verified against this CID inside the enclave by the attested "
                       + "wasm-manager before launch. The CID itself is NOT in a hardware register." }
           : { kind: "catalog", reference: ref,
               coverage: "Baked into the attested enclave image, so it is covered by the enclave "
                       + "measurement registers below." };
}
// ---- REAL ATTESTATION -------------------------------------------------------
// The Tinfoil shim generates the enclave TLS key, obtains a CPU attestation
// report (AMD SEV-SNP on today's fleet; Intel TDX flows through the same path)
// whose report_data[0:32] = sha256(TLS pubkey, SPKI DER), and serves the signed
// Remote Attestation Document at /.well-known/tinfoil-attestation. We RELAY that
// document verbatim and PARSE the quote so the registers are inspectable - but
// we never assert trust on the client's behalf: the party being verified cannot
// vouch for itself. What we DO publish is verification.selfCheck - this enclave
// running the same five checks a client would (via @tinfoilsh/verifier) and
// reporting the outcome as a clearly-labeled diagnostic, so a healthy deployment
// reads as a wall of passes instead of a bare "verified: false". Clients
// reproduce it with tinfoil-cli / @tinfoilsh/verifier against the Sigstore-
// signed measurements on ENCLAVE_REPO's releases, over their OWN connection.
const RAD_PATH        = "/.well-known/tinfoil-attestation";
const ATTESTATION_URL = process.env.ATTESTATION_URL || "";                 // explicit RAD URL override
const RAD_CACHE_MS    = parseInt(process.env.RAD_CACHE_MS || "300000", 10); // convenience-copy staleness bound
const sha256Hex = (b) => createHash("sha256").update(b).digest("hex");

// GET url, tolerate the shim's cert (the RAD is SELF-verifying: the quote binds
// the TLS key, so transport trust adds nothing), and capture the peer key so we
// can report the fingerprint exactly as Tinfoil's verifier computes it.
function fetchRad(url, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = (u.protocol === "https:" ? https : http).request(u,
      { method: "GET", timeout: timeoutMs, rejectUnauthorized: false }, (res) => {
        // grab the peer key NOW - the socket detaches from res once the body ends
        let liveTlsKeyFP = null;
        try { const x = res.socket.getPeerX509Certificate?.();
              if (x) liveTlsKeyFP = sha256Hex(x.publicKey.export({ type: "spki", format: "der" })); } catch {}
        let buf = ""; res.on("data", (c) => (buf += c));
        res.on("end", () => {
          if (res.statusCode !== 200) return reject(new Error(`${u.host}${u.pathname}: HTTP ${res.statusCode}`));
          let doc; try { doc = JSON.parse(buf); } catch { return reject(new Error(`${u.host}: not JSON`)); }
          if (typeof doc.format !== "string" || typeof doc.body !== "string")
            return reject(new Error(`${u.host}: not a Tinfoil attestation document`));
          resolve({ doc, liveTlsKeyFP, url });
        });
      });
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error(`${u.host}: timeout`)));
    req.end();
  });
}
let _radCache = null;                 // { doc, liveTlsKeyFP, url, at }
let _radInflight = null;
async function fetchEnclaveRad(origin) {
  if (_radCache && Date.now() - _radCache.at < RAD_CACHE_MS) return _radCache;
  if (_radInflight) return _radInflight;
  // The shim terminates TLS inside this CVM, so loopback is the trusted source.
  // We deliberately DO NOT fall back to the public origin (origin + RAD_PATH): that
  // hairpin leaves the CVM and re-enters through the untrusted ingress with
  // rejectUnauthorized off, so a MITM on that path could answer it. Loopback (the
  // shim, in-CVM) is ALWAYS available here, so there is genuinely no case with "no
  // loopback option" — we fail closed rather than trust the hairpin. ATTESTATION_URL
  // remains the explicit override for a shim that binds elsewhere.
  void origin;
  const candidates = ATTESTATION_URL ? [ATTESTATION_URL]
    : ["https://127.0.0.1" + RAD_PATH, "http://127.0.0.1" + RAD_PATH];
  _radInflight = (async () => {
    let lastErr = null;
    for (const url of candidates) {
      try { const r = await fetchRad(url); _radCache = { ...r, at: Date.now() }; return _radCache; }
      catch (e) { lastErr = e; }
    }
    throw new Error(`attestation document unreachable (${lastErr?.message || "no candidates"})`);
  })();
  try { return await _radInflight; } finally { _radInflight = null; }
}

// Parse a raw Intel TDX quote (DCAP QuoteV4/V5, in case a CVM lands on TDX
// hardware). Offsets are the fixed TD-report layout; report_data is what binds
// the TLS key. Returns null on anything odd -
// the verbatim document is still returned, clients parse it themselves anyway.
function parseTdxQuote(q) {
  try {
    if (q.length < 48 + 584) return null;
    const version = q.readUInt16LE(0), teeType = q.readUInt32LE(4);
    if (teeType !== 0x81) return null;                                  // TDX
    let body;
    if (version === 4) body = q.subarray(48, 48 + 584);
    else if (version === 5) {
      const bodyType = q.readUInt16LE(48);
      if (bodyType !== 2 && bodyType !== 3) return null;                // TD 1.0 / 1.5
      body = q.subarray(54);
    } else return null;
    if (body.length < 584) return null;
    const hx = (o, n) => body.subarray(o, o + n).toString("hex");
    return { quoteVersion: version,
             mrSeam: hx(16, 48), mrTd: hx(136, 48),
             rtmr0: hx(328, 48), rtmr1: hx(376, 48), rtmr2: hx(424, 48), rtmr3: hx(472, 48),
             reportData: hx(520, 64) };
  } catch { return null; }
}
// AMD SEV-SNP report (today's fleet): fixed offsets too.
function parseSnpReport(r) {
  try {
    if (r.length < 0x90 + 48) return null;
    return { measurement: r.subarray(0x90, 0x90 + 48).toString("hex"),
             reportData:  r.subarray(0x50, 0x50 + 64).toString("hex") };
  } catch { return null; }
}
function parseRad(doc) {
  let raw = Buffer.from(doc.body, "base64");
  if (raw[0] === 0x1f && raw[1] === 0x8b) raw = gunzipSync(raw);         // predicate v2 bodies are gzipped
  const fmt = doc.format || "";
  if (fmt.includes("tdx-guest")) {
    const p = parseTdxQuote(raw);
    return { technology: "intel-tdx", quote: raw.toString("base64"),
             measurements: p && { mrTd: p.mrTd, rtmr0: p.rtmr0, rtmr1: p.rtmr1, rtmr2: p.rtmr2, rtmr3: p.rtmr3 },
             reportData: p?.reportData ?? null, quoteVersion: p?.quoteVersion };
  }
  if (fmt.includes("sev-snp-guest")) {
    const p = parseSnpReport(raw);
    return { technology: "amd-sev-snp", quote: raw.toString("base64"),
             measurements: p && { measurement: p.measurement }, reportData: p?.reportData ?? null };
  }
  return { technology: fmt, quote: raw.toString("base64"), measurements: null, reportData: null };
}

// The CPU-TEE technology as DETECTED from this enclave's own attestation
// document - never asserted from config or hardcoded (the fleet has moved
// silicon before, and a hardcoded value in every deployment record is exactly
// the kind of unverifiable claim this platform exists to avoid). null until
// the first RAD fetch lands; a miss kicks a background fetch so the next
// caller reads the real answer.
let _vmTech = null;
function vmTech() {
  if (_vmTech) return _vmTech;
  if (_radCache?.doc) {
    try { _vmTech = parseRad(_radCache.doc).technology || null; } catch {}
  } else fetchEnclaveRad().catch(() => {});
  return _vmTech;
}

// GPU evidence comes from the worker manager (the container that holds the card):
// NVML's conf-compute attestation report, signed by the GPU, over OUR nonce.
async function fetchGpuEvidence(nonceHex, timeoutMs = 30000) {
  const r = await mgrReq("GET", `/attestation?nonce=${nonceHex}`, null, timeoutMs);
  if (r.status !== 200 || r.body.available === false)
    throw new Error(r.body.error || `worker manager /attestation ${r.status}`);
  return r.body;
}
// Shared evidence for UNAUTHENTICATED callers: NVML report generation is not
// free, so anonymous requests share one short-lived report over a self-chosen
// nonce (freshness for them comes from the TLS-bound CPU quote instead). An
// owner session is what buys a fresh report over a caller-chosen nonce.
let _gpuEvCache = null;                       // { ev?, err?, at } - failures cached briefly too
async function cachedGpuEvidence() {
  const ttl = _gpuEvCache?.err ? 10_000 : 60_000;
  if (!_gpuEvCache || Date.now() - _gpuEvCache.at > ttl) {
    try { _gpuEvCache = { ev: await fetchGpuEvidence(randomBytes(32).toString("hex"), 15000), at: Date.now() }; }
    catch (e) { _gpuEvCache = { err: e, at: Date.now() }; }
  }
  if (_gpuEvCache.err) throw _gpuEvCache.err;
  return _gpuEvCache.ev;
}

// ---- SELF-CHECK (diagnostic, not trust) --------------------------------------
// The enclave runs the exact five-step client verification against ITSELF
// (@tinfoilsh/verifier: SNP report -> AMD root, Sigstore release provenance,
// measurement comparison, cert binding) and reports the outcome. Labeled a
// self-check because self-vouching carries no trust - its value is (a) honest
// green-by-default optics and (b) catching config drift (wrong repo casing,
// stale release, broken egress) before a customer's verifier does. Fetches its
// own PUBLIC origin (hairpin) plus Tinfoil's GitHub/KDS proxies; if any of that
// is unreachable from inside, it degrades to "unavailable", never an error.
const SELF_CHECK_TTL_MS  = parseInt(process.env.SELF_CHECK_TTL_MS || "300000", 10); // re-check cadence after a pass (non-pass retries after 30s)
const SELF_CHECK_WAIT_MS = parseInt(process.env.SELF_CHECK_WAIT_MS || "8000", 10);  // max time one request waits on a fresh run
const SELF_CHECK_NOTE = "Run by the enclave itself as a diagnostic: it proves this deployment is configured "
                      + "to verify, not that you should trust it. Reproduce it on your side with `cli`, `npm`, "
                      + "or `browser` - trust ends at YOUR verifier, never at this field.";
// Flavor-aware verification. The stock Verifier compares an enclave against the
// repo's single "latest" GitHub release, which carries our GPU flavor — a CPU
// (or gpu8) enclave measures differently and would always fail
// compareMeasurements. So: fast-path latest, and ONLY on a mismatch probe the
// same version's sibling-flavor tags (vX.Y.Z-cpu / -gpu8), verifying against
// whichever release the enclave's own measurement matches. Security is
// unchanged — every candidate's provenance is still Sigstore-verified inside
// verifyBundle; we only widen WHICH signed release is the reference. The
// github-proxy the enclave reaches whitelists only /releases/latest,
// /releases/download/<tag>/tinfoil.hash and /attestations/<digest>, so we probe
// known tags rather than enumerate releases. Mirrored in site/js/core/verify.js.
const GITHUB_PROXY = "https://github-proxy.tinfoil.sh";
async function verifyMatchingRelease(host, repo) {
  const base = await assembleAttestationBundle(host, repo);   // enclave attestation + latest's digest/sigstore
  const attempt = async (digest, sigstoreBundle) => {
    const v = new Verifier({ configRepo: repo });
    try { await v.verifyBundle({ ...base, digest, sigstoreBundle }); } catch { /* step failure recorded on the doc */ }
    return v.getVerificationDocument();
  };
  const latest = await attempt(base.digest, base.sigstoreBundle);
  if (latest?.securityVerified) return latest;                // the common case: this node runs the latest (GPU) release
  // Bounded, like every other outbound call here: this runs inside the
  // self-check that /v1/attestation's consumers read, and node's fetch has no
  // default timeout — a github-proxy that accepts and stalls would pin the
  // shared in-flight self-check promise (and every request awaiting it) rather
  // than answering "unavailable".
  const ghFetch = (u) => fetch(u, { signal: AbortSignal.timeout(15000) });
  let latestTag;
  try { latestTag = (await (await ghFetch(`${GITHUB_PROXY}/repos/${repo}/releases/latest`)).json())?.tag_name; } catch { /* offline */ }
  for (const suffix of (latestTag ? ["-cpu", "-gpu8"] : [])) {
    const tag = latestTag + suffix;
    let digest, sigstoreBundle;
    try {
      const hr = await ghFetch(`${GITHUB_PROXY}/${repo}/releases/download/${tag}/tinfoil.hash`);
      if (!hr.ok) continue;
      digest = (await hr.text()).trim();
      sigstoreBundle = (await (await ghFetch(`${GITHUB_PROXY}/repos/${repo}/attestations/sha256:${digest}`)).json())?.attestations?.[0]?.bundle;
    } catch { continue; }
    if (!digest || !sigstoreBundle) continue;
    const doc = await attempt(digest, sigstoreBundle);
    if (doc?.securityVerified) return doc;                    // matched this flavor's signed release
  }
  return latest;   // nothing matched: the latest-comparison doc carries the mismatch detail
}

let _selfCheck = null;                  // { data, at }
let _selfCheckRun = null;               // in-flight run (shared across concurrent requests)
async function runSelfCheck(origin) {
  if (!ENCLAVE_REPO) return { result: "unavailable", error: "ENCLAVE_REPO not configured" };
  if (!origin)       return { result: "unavailable", error: "public origin not known yet (no external request seen)" };
  let doc, failure = null;
  try { doc = await verifyMatchingRelease(new URL(origin).hostname, ENCLAVE_REPO); }
  catch (e) { failure = e; }
  if (!doc) return { result: "unavailable", error: failure?.message || "verifier produced no document" };
  const word = (s) => !s || s.status === "pending" ? "skipped" : s.status === "success" ? "pass" : "fail";
  const steps = {};
  for (const k of ["fetchDigest", "verifyEnclave", "verifyCode", "compareMeasurements", "verifyCertificate"]) {
    steps[k] = word(doc.steps?.[k]);
    if (doc.steps?.[k]?.error) steps[k] += `: ${doc.steps[k].error}`;
  }
  return { result: doc.securityVerified ? "pass" : "fail",
           ...(failure ? { error: failure.message } : {}),
           steps,
           release: doc.releaseDigest ? `sha256:${doc.releaseDigest}` : null,
           measurement: doc.enclaveFingerprint || null };
}
async function getSelfCheck(origin) {
  const ttl = _selfCheck?.data?.result === "pass" ? SELF_CHECK_TTL_MS : Math.min(SELF_CHECK_TTL_MS, 30_000);
  if (_selfCheck && Date.now() - _selfCheck.at < ttl) return _selfCheck.data;
  if (!_selfCheckRun)
    _selfCheckRun = runSelfCheck(origin)
      .catch((e) => ({ result: "unavailable", error: e.message }))
      .then((r) => { const data = { result: r.result, ...r, checkedAt: new Date().toISOString(), note: SELF_CHECK_NOTE };
                     _selfCheck = { data, at: Date.now() }; _selfCheckRun = null; return data; });
  // don't hold the attestation response hostage to a slow first run
  const done = await Promise.race([_selfCheckRun,
    new Promise((res) => setTimeout(res, SELF_CHECK_WAIT_MS).unref())]);
  return done || { result: "pending",
                   detail: "self-check still running - request this endpoint again in a few seconds",
                   note: SELF_CHECK_NOTE };
}

async function getMeasurements(rec, { origin = PUBLIC_URL, nonce, freshGpu = true } = {}) {
  let enclaveHost = null; try { enclaveHost = origin ? new URL(origin).host : null; } catch {}
  const out = {
    // No server-asserted "verified" boolean: the machine being verified cannot
    // vouch for itself, and a hardcoded `false` reads like an outage. Instead,
    // selfCheck reports this enclave running the same checks a client would
    // (labeled diagnostic), and the pointers beside it reproduce the result
    // client-side in seconds - where it actually carries trust.
    verification: {
      selfCheck: await getSelfCheck(origin),
      how: "Fetch " + RAD_PATH + " from this origin over your OWN TLS connection, verify the quote "
         + "against the Intel/AMD root of trust, compare the registers to the Sigstore-signed "
         + "measurements on the release page of `repo` (exact casing - Sigstore compares it verbatim), "
         + "and check that reportData[0:32] equals sha256 of the TLS public key (SPKI DER) your "
         + "connection presents. Tinfoil's verifier does all of this for you:",
      cli: enclaveHost && ENCLAVE_REPO
         ? `tinfoil attestation verify -e ${enclaveHost} -r ${ENCLAVE_REPO}` : null,  // github.com/tinfoilsh/tinfoil-cli
      npm: "@tinfoilsh/verifier",  // Node + browsers: await new Verifier({ serverURL, configRepo: repo }).verify()
      browser: "https://enclave.host/#attest",
      repo: ENCLAVE_REPO || null,
      attestationEndpoint: (origin || "") + RAD_PATH,
    },
    tlsKeyFingerprint: null,
    app: rec ? appMeasurement(rec) : null,
    vm: null,
    gpu: null,
  };
  try {
    const { doc, liveTlsKeyFP, url } = await fetchEnclaveRad(origin);
    const parsed = parseRad(doc);
    const attestedTlsFP = parsed.reportData ? parsed.reportData.slice(0, 64) : null;
    out.tlsKeyFingerprint = attestedTlsFP ? `sha256:${attestedTlsFP}` : null;
    out.enclave = {
      attestationDocument: doc,               // verbatim Tinfoil RAD - feed to Tinfoil's verifier (tinfoil-cli / @tinfoilsh/verifier)
      fetchedFrom: url, fetchedAt: new Date(_radCache.at).toISOString(),
      // fingerprint of the key the shim ACTUALLY presented when we fetched; equals
      // the quote-bound one unless the shim rotated its key mid-cache-window.
      observedTlsKeyFingerprint: liveTlsKeyFP,
    };
    out.vm = { technology: parsed.technology, quote: parsed.quote, quoteVersion: parsed.quoteVersion,
               measurements: parsed.measurements, reportData: parsed.reportData };
  } catch (e) {
    out.enclave = { available: false, error: e.message,
                    note: "Fetch " + RAD_PATH + " from this origin yourself - the shim serves it directly." };
  }
  // GPU evidence only when this deployment actually holds a card slice (a
  // CPU-only app placed on a GPU enclave holds none — no card fields for it).
  if (IS_GPU && rec?._gpu && !rec._gpu.cpu) {
    const n = nonce || randomBytes(32).toString("hex");
    try {
      const ev = freshGpu ? await fetchGpuEvidence(n) : await cachedGpuEvidence();
      out.gpu = { technology: "nvidia-cc", ccMode: ev.ccMode ?? null, nonce: ev.nonce || n,
                  driverVersion: ev.driverVersion ?? null,
                  // first card's material at the top level (single-card enclaves); all cards in gpus[]
                  report: ev.gpus?.[0]?.attestationReport_b64 ?? null,
                  certChain: ev.gpus?.[0]?.attestationCertChain_b64 ?? null,
                  gpus: ev.gpus || [],
                  gpuShare: rec.resources.gpuShare,
                  vramCapGb: round1((rec.resources.gpuShare || 0) * CARD_VRAM_GB),
                  computeShare: rec.resources.gpuShare,
                  verify: "Check the report + cert chain with NVIDIA NRAS or nvtrust's local_gpu_verifier; "
                        + "confirm it signs YOUR nonce. The whole card is one CC trust domain - the VRAM/compute "
                        + "split is enforced by the attested supervisor+MPS, not by the hardware report." };
    } catch (e) {
      out.gpu = { technology: "nvidia-cc", available: false, error: e.message };
    }
  }
  return out;
}
// ============================================================================

const app = express();
// Express advertises itself in `X-Powered-By` on EVERY response. This process
// answers on the public data path (api-relay proxies /v1 and /x straight
// through, so the header reaches the internet verbatim — it is observable today
// on api.enclave.host), and it names the server framework of code running
// INSIDE the CVM. That is a free hint about which CVE list to try against a box
// whose whole pitch is that you cannot see in. Nothing reads it.
app.disable("x-powered-by");
// Express 4 does not catch a REJECTED async route handler — it becomes an
// unhandledRejection (the process-level guard above logs it, but the request
// would hang and, pre-guard, the process died). Forward async rejections to the
// error middleware (registered after the routes) instead. We install it once, as
// a thin shim over the route-registration methods, so EVERY route handler
// (present and future) is covered and none can be forgotten. Only real handlers
// are wrapped (arity < 4); 4-arg error middleware is passed through untouched.
// app.use() is intentionally NOT shimmed — the streaming proxies mounted with it
// (/x/:id, the platform-model proxy) manage their own lifecycle.
const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
for (const m of ["get", "post", "patch", "put", "delete"]) {
  const orig = app[m].bind(app);
  app[m] = (path, ...handlers) => orig(path, ...handlers.map((h) => (typeof h === "function" && h.length < 4 ? wrap(h) : h)));
}
app.use(cors({
  origin: CORS_ORIGINS.includes("*") ? true : CORS_ORIGINS,
  methods: ["GET","POST","PATCH","DELETE","OPTIONS"],
  allowedHeaders: ["Authorization","Content-Type"],
  maxAge: 86400,
}));

// Transport + sniffing headers on OUR OWN responses. The apex vhost sets these
// for enclave.host, but api.enclave.host is a different host and was serving
// neither: a client that reaches the API directly (the CLI, the MCP server, any
// browser that has not first seen the apex and taken its includeSubDomains pin)
// had no HSTS pin at all, and JSON answers went out sniffable. Set here rather
// than in Caddy because this is the process that KNOWS which responses are ours
// — and because the vhost lives outside this repo, so a header that matters is
// better owned by the code it protects.
//
// /x is excluded on purpose: those bytes are the TENANT's app, and nosniff can
// change how an app's own content renders. Their default headers are
// tenantHeaders()' business, not this one's. (Express merges what is set here
// with what the proxy passes to writeHead, so the exclusion has to be explicit.)
app.use((req, res, next) => {
  if (!req.path.startsWith("/x/")) {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Strict-Transport-Security", "max-age=31536000; includeSubDomains");
  }
  next();
});

// A hoisted declaration, not a const arrow: the selftest seams call handlers
// that fail() from far earlier in the module, where a module-level const is
// still in its temporal dead zone — and the ReferenceError surfaces as a
// handler that silently never responds, not as an error. Same reason
// tenantHeaders is written this way.
function fail(res, status, code, message) { return res.status(status).json({ code, message }); }
// Prefer our attested SAN once known; fall back to the request Host only during
// early boot before the shim cert is read (client verifies attestation over its
// own TLS, so a reflected Host there is not trusted for identity).
const originOf = (req) => PUBLIC_URL || (_certSan ? `https://${_certSan}` : `https://${req.headers["x-forwarded-host"] || req.headers.host}`);

// Kick self-registration on any request in case the boot loop is mid-backoff. We
// register ONLY our attested shim-cert SAN (registerFromShimCert), never the
// request Host, so a spoofed Host can never make us advertise a bogus origin.
app.use((req, _res, next) => {
  if (REGISTRY_READY && !_registered) registerFromShimCert().catch(() => {});
  next();
});

async function addrFromAuth(req) {
  const m = (req.headers.authorization || "").match(/^Bearer\s+(.+)$/i);
  return m ? verifySessionToken(m[1]) : null;
}

// ---------------------------------------------------------------------------
// DATA PATH - registered BEFORE express.json() so the body streams untouched.
// Same token, same origin as control; supervisor checks ownership, then proxies.
// ---------------------------------------------------------------------------
// Firewall config validation. Mirrors the wasm-manager's rules so a bad spec fails
// fast at create (422) instead of at provision: entries are "http" (default serve
// mode) | "http:N" | "tcp:N" | "udp:N"; N in 1..49999 (logical labels; <1024 always
// remapped), excluding infra ports (8080 supervisor, 8091 manager). The ceiling
// matches what the public relay binds (RELAY_PORTS=1-49999).
const FW_MIN = 1, FW_MAX = 49999, FW_RESERVED = new Set([1080, 8080, 8090, 8091]);   // infra ports: 1080 egress SOCKS, 8080 supervisor, 8090 GPU worker, 8091 wasm-manager (logical labels; <1024 is always remapped to an unprivileged actual by the manager)
function parseFirewall(fw) {
  const raw = (fw && Array.isArray(fw.ports)) ? fw.ports : [];
  if (raw.length > 8) throw new Error("firewall.ports: at most 8 entries.");
  const out = [];
  for (const e of raw) {
    const s = String(e).trim().toLowerCase();
    if (!s || s === "http") continue;                       // default serve mode marker
    const m = /^(http|tcp|udp):(\d{1,5})$/.exec(s);
    if (!m) throw new Error(`firewall.ports: bad entry "${e}" (use http[:N] | tcp:N | udp:N).`);
    const p = +m[2];
    if (p < FW_MIN || p > FW_MAX || FW_RESERVED.has(p))
      throw new Error(`firewall.ports: port ${p} not allowed (${FW_MIN}-${FW_MAX}, excluding ${[...FW_RESERVED].join("/")}).`);
    if (!out.includes(m[1] + ":" + p)) out.push(m[1] + ":" + p);
  }
  if (out.filter((x) => x.startsWith("http:")).length > 1) throw new Error("firewall.ports: only one http:N entry.");
  return out;                                               // [] = classic wasi:http serve mode
}
const fwTcpPorts = (rec) => (rec.firewall || []).filter((x) => x.startsWith("tcp:")).map((x) => +x.slice(4));
const fwUdpPorts = (rec) => (rec.firewall || []).filter((x) => x.startsWith("udp:")).map((x) => +x.slice(4));

// ---------------------------------------------------------------------------
// Per-deployment addressing — each deployment gets its OWN IPv6 out of the
// relay box's routed /64, and the relays route by destination IP. This is the
// deployment's dedicated address: the udp-relay serves its udp:N ports there,
// and the tcp6-relay serves its tcp:N ports there (at the LOGICAL port, no SNI,
// no remapping — clients use the port the app declared). The address is
// DETERMINISTIC from the deployment id (sha256 → low 64 host bits), so the
// supervisor and every relay derive the identical value with no shared state.
// DEP_ADDR_PREFIX (or the legacy UDP_ADDR_PREFIX) is the relay box's routed /64
// (e.g. "2a01:4f9:c013:9b52::/64", the live fleet's); unset = dedicated addressing off (the
// /x/:id/(tcp|udp) bridges still work for direct callers, but no address is
// advertised). See relay/README.md.
const DEP_ADDR_PREFIX = (process.env.DEP_ADDR_PREFIX || process.env.UDP_ADDR_PREFIX || "").trim();
function v6ToBig(s) {                                       // parse an IPv6 (incl. "::") to a 128-bit BigInt
  const [head, tail] = s.split("::");
  const hi = head ? head.split(":").filter(Boolean) : [];
  const lo = tail ? tail.split(":").filter(Boolean) : [];
  const mid = Array(8 - hi.length - lo.length).fill("0");
  const groups = s.includes("::") ? [...hi, ...mid, ...lo] : s.split(":");
  if (groups.length !== 8) throw new Error(`bad IPv6 "${s}"`);
  return groups.reduce((a, g) => (a << 16n) | BigInt(parseInt(g || "0", 16)), 0n);
}
function bigToV6(n) {                                       // 128-bit BigInt → compressed IPv6 string
  const g = [];
  for (let i = 0; i < 8; i++) g[i] = Number((n >> BigInt((7 - i) * 16)) & 0xffffn);  // g[0] = most significant group
  let best = { i: -1, len: 0 }, cur = { i: -1, len: 0 };    // longest zero-run for "::"
  g.forEach((v, i) => {
    if (v === 0) { if (cur.i < 0) cur = { i, len: 0 }; cur.len++; if (cur.len > best.len) best = { ...cur }; }
    else cur = { i: -1, len: 0 };
  });
  const hex = g.map((v) => v.toString(16));
  if (best.len > 1) { hex.splice(best.i, best.len, ""); if (best.i === 0) hex.unshift(""); if (best.i + best.len === 8) hex.push(""); }
  return hex.join(":").replace(/:{3,}/, "::");
}
// Deterministic host part: sha256(id) low 64 bits, kept clear of the low range
// so it never lands on the box's own ::1 / infrastructure addresses.
function depAddrFor(id) {
  if (!DEP_ADDR_PREFIX) return null;
  const [prefix] = DEP_ADDR_PREFIX.split("/");
  const net128 = v6ToBig(prefix) & (~0n << 64n);            // zero the low 64 (host) bits
  let host = BigInt("0x" + createHash("sha256").update(id).digest("hex").slice(0, 16)) & ((1n << 64n) - 1n);
  if (host < 0x10000n) host += 0x10000n;                    // reserve the low range for infra
  return bigToV6(net128 | host);
}
// public deployments exposing udp ports, with their address + logical ports —
// the udp-relay reads this to know what to bind and where to route.
const udpMap = () => [...deployments.values()]
  .filter((r) => r.public && r.status === "running" && fwUdpPorts(r).length)
  .map((r) => ({ id: r.id, address: depAddrFor(r.id), ports: fwUdpPorts(r) }));
// public deployments with tcp OR udp ports, each with its dedicated address and
// per-protocol logical ports — the tcp6-relay (tcp) and udp-relay (udp) poll
// this to bind [address]:port and route into /x/:id/(tcp|udp)/:port.
const netMap = () => [...deployments.values()]
  .filter((r) => r.public && r.status === "running" && (fwTcpPorts(r).length || fwUdpPorts(r).length))
  .map((r) => ({ id: r.id, address: depAddrFor(r.id), tcp: fwTcpPorts(r), udp: fwUdpPorts(r) }));

// --- dedicated-IP EGRESS (the outbound half of depAddrFor) ------------------
// A deployment's OUTBOUND connections leave from its own IPv6, mirroring the
// inbound tcp6/udp relays. Guests opt in via ENCLAVE_EGRESS (a per-deployment SOCKS
// URL); the enclave front is here (egress.js), the source-binding dialer is
// relay/egress-relay.js. Enabled only when dedicated addressing is on AND a
// shared relay token is configured (EGRESS_RELAY_TOKEN — proves the control/
// data channels are the real relay, not a random client hitting the shim).
const EGRESS_RELAY_TOKEN = (process.env.EGRESS_RELAY_TOKEN || "").trim();
const EGRESS_SOCKS_PORT  = parseInt(process.env.EGRESS_SOCKS_PORT || "1080", 10);
// The relay name the DEFAULT egress relay attaches as (it owns DEP_ADDR_PREFIX's
// /64 and source-binds). A deployment with no network.relay choice, or one whose
// chosen relay isn't attached, egresses through it. Must match that relay's
// RELAY_NAME. Unset preserves the pre-multi-relay default ("default").
const EGRESS_DEFAULT_RELAY = (process.env.EGRESS_DEFAULT_RELAY || "").trim();
// The relay that owns DEP_ADDR_PREFIX's /64 and can source-bind a dedicated IP.
// Defaults to the default relay (today they're the same box). Set this when the
// default egress relay is a NEARBY plain relay but dedicated-IP egress should
// still be reachable on the /64 owner by explicit network.relay choice.
const EGRESS_DEDICATED_RELAY = (process.env.EGRESS_DEDICATED_RELAY || "").trim();
// A deployment's chosen relay (network.relay), so its OUTBOUND follows the same
// relay as its inbound. "" / unset / unparseable => the default relay.
function egressRelayFor(id) {
  const r = deployments.get(id);
  if (!r) return null;
  try { return parseDepOptions(r._envelope, r.gpuMilli).relay || null; }
  catch { return null; }
}
const egress = (DEP_ADDR_PREFIX && EGRESS_RELAY_TOKEN)
  ? createEgress({
      secret: SECRET, socksPort: EGRESS_SOCKS_PORT, relayToken: EGRESS_RELAY_TOKEN,
      sourceAddrFor: depAddrFor, relayFor: egressRelayFor,
      defaultRelay: EGRESS_DEFAULT_RELAY, dedicatedRelay: EGRESS_DEDICATED_RELAY,
      // "claimed" is mid-provision on THIS enclave: the app process starts
      // (and may dial out — its very first S3 fetch) fractionally before the
      // provision path flips the record to "running", and the egress token in
      // its env is enclave-minted either way. Excluding claimed made every
      // boot-time connect lose the race and get "credential rejected" with no
      // retry (risc-box fetches its kernel at exec and died on it; lazy
      // dialers like net-probe never noticed).
      isKnown: (id) => { const r = deployments.get(id); return !!r && (r.status === "running" || r.status === "claimed"); },
      log: (m) => console.log(m),
    })
  : null;

// On-chain ids are bytes32, and a full 64-hex id exceeds DNS's 63-char label
// limit - app subdomains carry a hex PREFIX of the id instead, resolved here
// (unique match only; the canonical label is the FIRST 8 CHARS = 32 bits,
// any longer prefix works too). Shared by the HTTP data path and the
// The requester's IP as this enclave can best know it. On the /x/:id/https
// bridge the inner requests ride a decrypted stream with no socket address, so
// the upgrade handler stamps the IP it saw onto the socket; otherwise it comes
// from x-forwarded-for.
//
// The LAST entry, not the first. X-Forwarded-For is written by the CLIENT and
// APPENDED to by each proxy, so `X-Forwarded-For: 1.2.3.4` sent from a browser
// arrives here as "1.2.3.4, <real client>" — the first entry is whatever the
// sender typed. This key is not only abuse damping any more: it is the bucket
// for the DEPLOYER-BOUGHT WAF (the options envelope's per-IP rate limit and
// concurrency cap), a control a tenant pays for and points at attackers. Keyed
// on the first hop, a flood that varies one header never shares a bucket and the
// limit does nothing. api-relay forwards headers verbatim and Caddy appends its
// peer, so on the relay path — the path tenants are actually exposed on — the
// last entry IS the client and cannot be chosen by them.
//
// A DIRECT caller (dialing the enclave's own hostname) can still forge it if
// nothing in front appends, which is no worse than before: a forger lands in
// someone else's bucket or the shared one. Same reading as api-relay's
// clientIp (2fc10a34) and the add-gateway's.
function clientIpOf(req) {
  if (req.socket && req.socket._clientIp) return req.socket._clientIp;
  const xs = String(req.headers["x-forwarded-for"] || "").split(",").map((x) => x.trim()).filter(Boolean);
  return xs[xs.length - 1] || req.socket?.remoteAddress || "?";
}

// ---- per-deployment WAF (the options envelope's `waf` namespace) ------------
// Enforced HERE, at the one proxy every HTTP request to a tenant crosses —
// in-enclave, so it holds on the relay path, the in-enclave TLS bridge, and
// direct-to-origin callers alike, and no operator box has to see app traffic
// to provide it. State is per-deployment and in-memory only: buckets refill
// from wall clock, so a reboot merely forgives a burst.
const _wafStates = new Map();   // dep id -> { buckets: Map(ip -> {tokens, at}), active: Map(ip -> n) }
setInterval(() => {
  const cut = Date.now() - 600_000;
  for (const [id, st] of _wafStates) {
    if (!deployments.has(id)) { _wafStates.delete(id); continue; }
    for (const [ip, b] of st.buckets) if (b.at < cut) st.buckets.delete(ip);
  }
}, 300_000).unref?.();
// Run rec.waf against this request. True = proceed; false = the response has
// been written. Order: cheap static rules first, counters last (a blocked
// path must not consume rate tokens - the point of pathBlock is that junk
// stays free).
function wafGate(rec, req, res) {
  const w = rec.waf;
  if (!w) return true;
  if (w.methods && !w.methods.includes(req.method)) {
    fail(res, 405, "waf_method", `This deployment's protection rules allow only: ${w.methods.join(", ")}.`);
    return false;
  }
  if ((w.blockScanners || w.pathBlock) && wafPathBlocked(w, req.url)) {
    fail(res, 403, "waf_path", "Blocked by this deployment's protection rules.");
    return false;
  }
  if (w.uaBlock) {
    const ua = String(req.headers["user-agent"] || "").toLowerCase();
    if (w.uaBlock.some((s) => ua.includes(s))) {
      fail(res, 403, "waf_agent", "Blocked by this deployment's protection rules.");
      return false;
    }
  }
  // Content-Length fast reject; chunked/lying bodies are caught by the
  // counted stream at the proxy pipe (the other half of maxBodyMb).
  const cl = Number(req.headers["content-length"]);
  if (w.maxBodyMb && Number.isFinite(cl) && cl > w.maxBodyMb * 1048576) {
    fail(res, 413, "waf_body", `Request body exceeds this deployment's ${w.maxBodyMb} MB limit.`);
    return false;
  }
  const ip = clientIpOf(req);
  let st = _wafStates.get(rec.id);
  if (!st) { st = { buckets: new Map(), active: new Map() }; _wafStates.set(rec.id, st); }
  if (w.maxConcurrent) {
    const n = st.active.get(ip) || 0;
    if (n >= w.maxConcurrent) {
      res.setHeader("Retry-After", "1");
      fail(res, 429, "waf_busy", `Too many concurrent requests from your address (limit ${w.maxConcurrent}).`);
      return false;
    }
    st.active.set(ip, n + 1);
    res.once("close", () => {
      const m = (st.active.get(ip) || 1) - 1;
      m > 0 ? st.active.set(ip, m) : st.active.delete(ip);
    });
  }
  if (w.rps) {   // same token-bucket math as hintRateOk, dialed by the deployer
    const now = Date.now();
    let b = st.buckets.get(ip);
    if (!b) { b = { tokens: w.burst, at: now }; st.buckets.set(ip, b); }
    b.tokens = Math.min(w.burst, b.tokens + ((now - b.at) / 1000) * w.rps);
    b.at = now;
    if (b.tokens < 1) {
      res.setHeader("Retry-After", String(Math.max(1, Math.ceil((1 - b.tokens) / w.rps))));
      fail(res, 429, "waf_rate_limited", `Rate limit: this deployment allows ${w.rps} requests/sec per address (burst ${w.burst}).`);
      return false;
    }
    b.tokens -= 1;
  }
  return true;
}

// /x/:id/https upgrade path (browser TLS terminated in-enclave).
function depByIdOrPrefix(id) {
  let rec = deployments.get(id);
  if (!rec && /^0x[0-9a-f]{8,64}$/.test(id)) {
    const hits = [...deployments.keys()].filter(k => k.startsWith(id));
    if (hits.length === 1) rec = deployments.get(hits[0]);
  }
  return rec || null;
}

// ---- default response headers for TENANT traffic ---------------------------
// Every tenant app is served at <label>.app.enclave.host, and a passkey's rpId
// is `enclave.host`. WebAuthn lets ANY origin under a registrable domain
// exercise that domain's credentials, so a hostile app can call
// navigator.credentials.get({ rpId: "enclave.host" }) and the browser's prompt
// names "enclave.host" — the legitimate brand — on a page the user does not
// control. rpId cannot be narrowed to fix this: it must be a suffix of the
// site's own domain, and the site is at the APEX (see cb68e88e).
//
// Both verifiers already refuse such an assertion by origin — the credit vault
// on-chain (cb68e88e) and the relay's auth.js (expectedOrigin) — so this is
// defence in depth, not the fix. What it adds is (1) the browser refusing the
// call outright, before a human is asked to tap anything, and (2) coverage for
// the NEXT verifier someone writes, which is the one that will forget.
//
// OPT-OUT, deliberately not a new options-envelope namespace: an app that
// genuinely wants WebAuthn (legitimately, with its OWN subdomain as rpId) sets
// its own Permissions-Policy and we do not touch it. That keeps a standard HTTP
// mechanism as the escape hatch instead of inventing config surface, and means
// this cannot permanently break an app whose author knows what they want.
function tenantHeaders(upstream) {
  // Declared INSIDE the function on purpose. This is a hoisted function
  // declaration and the selftest seam calls it from far earlier in the module;
  // a module-level `const` here would still be in its temporal dead zone at
  // that point and throw ReferenceError. (Found by the test, not by reading.)
  const POLICY = "publickey-credentials-get=(), publickey-credentials-create=()";
  // Hop-by-hop headers are per-leg (RFC 9110 7.6.1) and were being copied
  // from the tenant leg onto the client leg verbatim. The tenant leg is
  // deliberately keep-alive:false, so every tenant response says
  // `connection: close` — and forwarding that told Node to close the CLIENT
  // connection after every response, which put a fresh ~1.4-2s in-enclave
  // TLS handshake (client → relay → tunnel → this bridge) under every
  // request an app-zone client made. Measured 2026-09-01 on eyesoff/metal0:
  // tls ≈ 1.3-1.7s of a ≈ 2s total per request, keep-alive never engaging,
  // for either curl or a browser. Forwarded `transfer-encoding` was wrong
  // the same way: pipe() writes the DECODED body, so clients only parsed
  // these responses because connection-close framing let them read to EOF.
  // Strip the whole hop-by-hop set and let Node frame the client leg itself.
  // (Declared inside the function for the same hoisting/TDZ reason as POLICY.)
  const HOP = ["connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"];
  const h = {};
  for (const [k, v] of Object.entries(upstream)) {
    if (!HOP.includes(String(k).toLowerCase())) h[k] = v;
  }
  // Case-INSENSITIVE check, though Node lower-cases incoming header names and
  // this could just test "permissions-policy". Enforcing it here makes the
  // invariant true by construction instead of by an assumption about the
  // runtime: if that ever stopped holding, an app shipping `Permissions-Policy:`
  // would emit BOTH its value and ours, and a duplicated header is resolved by
  // the browser in a way neither of us chose.
  if (!Object.keys(h).some((k) => k.toLowerCase() === "permissions-policy"))
    h["permissions-policy"] = POLICY;
  return h;
}

// ---- private-app browser access -------------------------------------------
// Who is calling a private deployment? The bearer is the machine path (CLI,
// fetch, MCP) and stays FIRST so nothing about it changes. The cookie is the
// browser path, and is accepted only for THIS deployment (audience-bound).
async function addrForApp(req, rec) {
  const bearer = await addrFromAuth(req);
  if (bearer) return bearer;
  // Try EVERY pair sent under our name: a planted duplicate (see cookieVals)
  // must not shadow the real one. Bounded by how many a browser will send, and
  // each miss is one cheap ES256 verify.
  for (const tok of cookieVals(req, APP_COOKIE)) {
    const addr = await verifyAppToken(tok, rec.id);
    if (addr) return addr;
  }
  return null;
}

// Is this a browser NAVIGATION (as opposed to an API call)? Only navigations
// get HTML back; everything else keeps the JSON error it already parses, so no
// existing client can be broken by this. `sec-fetch-dest` is the precise
// signal where it exists; the Accept sniff covers browsers that omit it.
//
// A hoisted declaration, not a const arrow: the APPAUTH_SELFTEST seam calls it
// from far earlier in the module, where a module-level const is still in its
// temporal dead zone (the same trap tenantHeaders documents above).
function wantsHtml(req) {
  return req.method === "GET" &&
    (req.headers["sec-fetch-dest"] === "document" ||
     (!req.headers["sec-fetch-dest"] && /\btext\/html\b/i.test(req.headers.accept || "")));
}

// Drop OUR cookie pairs from a Cookie header, preserving the app's own. ALL of
// them, matching cookieVals — a planted duplicate must not survive into the
// tenant either. Returns the header UNCHANGED when we set none of it, so a
// public deployment's request bytes are exactly what they always were.
function stripAppCookie(raw) {
  if (!raw) return "";
  const s = String(raw);
  if (!s.includes(APP_COOKIE)) return s;
  return s.split(";")
    .filter((p) => { const eq = p.indexOf("="); return eq < 0 || p.slice(0, eq).trim() !== APP_COOKIE; })
    .map((p) => p.trim()).filter(Boolean).join("; ");
}

// A self-contained page with NO external references, served under a nonce CSP
// that forbids everything else. It runs at a TENANT origin, so it deliberately
// carries no wallet code and asks for no signature: the only wallet prompt in
// this whole flow happens on enclave.host, an origin the tenant cannot serve.
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Embed an untrusted value as a JS string literal inside an inline <script>.
// JSON.stringify handles quotes, backslashes and control characters; the "<"
// rewrite handles the HTML tokenizer, which does not care that it is looking at
// a string: "<!--<script" inside one flips it into script-data-double-escaped
// state and it then swallows our own </script>, blanking the page. < is a
// legal escape INSIDE a string literal (it would not be as an operator, which
// is why this belongs here and not over the whole script).
const jsStr = (v) => JSON.stringify(String(v)).replace(/</g, "\\u003c");

function htmlPage(res, status, title, body, script) {
  const nonce = randomBytes(16).toString("base64");
  const js = script ? `<script nonce="${nonce}">${String(script).replace(/<\//g, "<\\/")}</script>` : "";
  res.status(status);
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Content-Security-Policy",
    `default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}'; ` +
    `connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`);
  res.end(`<!doctype html><html lang="en"><head><meta charset="utf-8">` +
    `<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">` +
    `<title>${title}</title><style nonce="${nonce}">` +
    `:root{color-scheme:light dark}body{margin:0;min-height:100vh;display:grid;place-items:center;` +
    `font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;` +
    `background:#0b0d10;color:#e6e8eb}main{max-width:32rem;padding:2rem;text-align:center}` +
    `h1{font-size:1.15rem;font-weight:600;margin:0 0 .5rem}p{margin:0 0 1rem;color:#9aa4b2}` +
    `a{color:#e6e8eb}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em}` +
    `@media (prefers-color-scheme:light){body{background:#fff;color:#11151a}p{color:#5a6572}a{color:#11151a}}` +
    `</style></head><body><main>${body}</main>${js}</body></html>`);
}

// The 401 a browser gets instead of a JSON blob: bounce to enclave.host, which
// holds the wallet code, and come back with a token. `p` is the path the user
// actually asked for, computed in the page (see the `root` subtraction) so it
// survives both access modes — the app subdomain, where the relay rewrites
// /x/<id> away, and a direct https://<enclave>/x/<id>/… hit, where it does not.
function loginBounce(rec, req, res, wrongAddr) {
  const authorize = `${SIWE_URI}/authorize?d=${encodeURIComponent(rec.id)}`;
  // `root` is the app's base path as the BROWSER sees it, recovered by
  // subtracting the sub-path we were asked for. It is "" behind the app
  // subdomain (the relay rewrites /x/<id> away) and "/x/<id>" on a direct
  // enclave hit; the two must not be confused, and trimming a path segment
  // instead would land on "/a/b" for a request to "/a/b/c".
  const rootJs =
    `var SUB=${jsStr(req.url || "/")};` +
    `var here=location.pathname+location.search;` +
    `var root=here.length>=SUB.length&&here.slice(here.length-SUB.length)===SUB?here.slice(0,here.length-SUB.length):"";`;
  // A WRONG wallet is a dead end, not a step: bouncing would send the user to a
  // page that can only tell them the same thing, and back again. Say it here
  // and make the next move a deliberate click. 403, matching the JSON path.
  if (wrongAddr)
    return htmlPage(res, 403, "Wrong wallet &middot; Enclave",
      `<h1>Wrong wallet</h1><p>This app is private. You are signed in as <code>${esc(wrongAddr)}</code>, ` +
      `which does not own it. Switch wallets, then sign in again.</p>` +
      `<p><a href="${esc(authorize)}">Sign in with a different wallet</a></p>`,
      // Drop the stale cookie, or every later navigation repeats this page.
      rootJs + `fetch(root+"/__enclave/signout",{method:"POST"}).catch(function(){});`);

  htmlPage(res, 401, "Sign in &middot; Enclave",
    `<h1>This app is private</h1><p>Sign in with the wallet that owns it. Redirecting&hellip;</p>` +
    `<p><a id="go" href="${esc(authorize)}">Continue to enclave.host</a></p>`,
    rootJs +
    `var u=${jsStr(authorize)}+"#p="+encodeURIComponent(here.slice(root.length)||"/");` +
    `document.getElementById("go").href=u;` +
    // Loop breaker. If we just came BACK from a successful hand-off and are
    // still being refused, redirecting again spins forever — a JS bounce gets
    // no "too many redirects" from the browser. Stop and say so instead.
    `var t=0;try{t=+sessionStorage.getItem("enclave_az")||0;sessionStorage.removeItem("enclave_az");}catch(e){}` +
    `if(Date.now()-t<15000){document.querySelector("p").textContent=` +
    `"You signed in, but this enclave still refuses the app. Its owner may have changed, or the deployment moved. Try again from your dashboard.";}` +
    `else location.replace(u);`);
}

// The hand-off, served at the app origin. GET renders the page that lifts the
// token out of the fragment (never the query — a fragment reaches no server log
// and no Referer); POST carries it back in a header and is what actually sets
// the cookie. Returns true when it handled the request.
function appSessionRoute(rec, req, res) {
  const path = (req.url || "").split("?")[0].replace(/\/+$/, "") || "/";
  if (path === "/__enclave/signout") {
    if (req.method !== "POST") return false;
    res.setHeader("Set-Cookie", `${APP_COOKIE}=; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=0`);
    res.status(204).end();
    return true;
  }
  if (path !== "/__enclave/session") return false;

  if (req.method === "POST") {
    // Token arrives as a bearer so this needs no body parser — the data path is
    // mounted ahead of express.json() on purpose (bodies stream untouched).
    (async () => {
      const m = (req.headers.authorization || "").match(/^Bearer\s+(.+)$/i);
      // Shape-check BEFORE this value can reach a response header. Node would
      // throw on a CR/LF in setHeader, and a doctored token should fail the
      // signature anyway — but base64 decoding is lenient about junk
      // characters, so "verified" is a thin thing to rest a header-injection
      // argument on. A compact JWS is three base64url segments and nothing else.
      const tok = m && /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(m[1]) ? m[1] : "";
      const addr = tok ? await verifyAppToken(tok, rec.id) : null;
      // Re-check ownership at REDEMPTION, not just at mint: a deployment can
      // change hands inside the token's lifetime, and the old owner's token
      // must stop opening it the moment it does.
      if (!addr || rec.owner !== addr) return fail(res, 401, "unauthorized", "Invalid or expired sign-in.");
      // Host-only (no Domain=) so it never reaches enclave.host or a sibling
      // subdomain. Path=/ covers the direct /x/<id>/… mode too; a sibling
      // deployment on this box rejects it anyway (the audience is bound to one
      // id) and the tenant never sees it (stripAppCookie on the proxy leg).
      //
      // STRICT, not Lax. A cookie is ambient where a bearer was not, so Lax
      // would have handed every site on the internet the ability to navigate an
      // owner's browser into an authenticated GET on their private app — and
      // the tenant cannot defend itself, because we strip the cookie before
      // proxying and leave it no signal to gate on. Strict costs nothing here:
      // the dashboard's "open" link points at /authorize, so the only cross-site
      // navigation in the flow is enclave.host -> /__enclave/session, which
      // carries the token in a fragment and needs no cookie; the POST below and
      // the redirect after it are both same-origin. Address-bar hits and
      // bookmarks are same-site and unaffected.
      res.setHeader("Set-Cookie",
        `${APP_COOKIE}=${tok}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=${APP_TTL_SEC}`);
      res.status(204).end();
    })().catch(() => { if (!res.headersSent) fail(res, 500, "error", "Sign-in failed."); });
    return true;
  }
  if (req.method !== "GET") return false;

  htmlPage(res, 200, "Signing in &middot; Enclave",
    `<h1>Signing you in&hellip;</h1><p id="m">One moment.</p>`,
    `var h=new URLSearchParams(location.hash.slice(1));` +
    `var t=h.get("t")||"",p=h.get("p")||"/";` +
    // Resolve and compare ORIGINS rather than pattern-match the string. A
    // leading-slash regex looks sufficient and is not: the URL parser folds
    // backslashes into slashes, so "/\\evil.com" passes /^\/(?!\/)/ and then
    // resolves to https://evil.com/. Whatever the parser will actually DO with
    // this value is the only thing worth asking.
    `try{var _u=new URL(p,location.href);p=_u.origin===location.origin?_u.pathname+_u.search:"/";}catch(e){p="/";}` +
    `var root=location.pathname.replace(/\\/__enclave\\/session\\/?$/,"");` +
    `function bad(x){document.getElementById("m").textContent=x;}` +
    `if(!t)bad("This sign-in link is incomplete. Start again from enclave.host.");else{` +
    `fetch(location.pathname,{method:"POST",headers:{Authorization:"Bearer "+t},cache:"no-store"})` +
    `.then(function(r){if(!r.ok)throw 0;` +
    // Mark the hand-off so a bounce arriving right back here is recognised as a
    // LOOP rather than redirected onward (see loginBounce's loop breaker).
    `try{sessionStorage.setItem("enclave_az",String(Date.now()));}catch(e){}` +
    // Scrub the token out of history BEFORE leaving, so Back cannot resurrect it.
    `history.replaceState(null,"",root+p);location.replace(root+p);})` +
    `.catch(function(){bad("That sign-in expired. Start again from enclave.host.");});}`);
  return true;
}

// How long the TENANT leg may move NOTHING before we cut it. An inactivity
// bound, not a total cap: this path carries SSE and long completions that are
// legitimately open for hours, and they re-arm this on every byte (risc-box
// heartbeats its SSE every 15 s precisely so silence stays diagnostic).
//
// What it catches is the tenant that ACCEPTS the socket and then never answers.
// With no timeout at all — which is what this proxy had — such a request hung
// the client FOREVER: no headers, no error, no log, nothing on the wire to tell
// a stalled app from a dead network. That invisibility is what made a stalled
// risc-box read as a Moonlight UDP fault for hours (2026-08-24). Matches the
// 180 s idle allowance internalAppServer already keeps.
const TENANT_IDLE_MS = parseInt(process.env.TENANT_IDLE_MS || "180000", 10);

// The tenant leg gets its OWN connection, never a pooled one.
//
// This path carries two things that must not share a socket pool: ordinary
// bounded requests, and SSE streams that are abandoned mid-flight every time a
// viewer closes a tab or a Moonlight session ends. With Node's default
// globalAgent (keepAlive, 256 free sockets) an abandoned stream leaves the
// pool holding a socket that is not at a clean request/response boundary; the
// next request is handed that socket, writes onto it, and never sees response
// headers. The caller cannot tell that from a hung app -- gs-bridge reports it
// as EAGAIN, the browser as a stream that goes silent forever.
//
// That is not hypothetical: gs-bridge's own source records it as "a single
// unrelated request to the same deployment silences an open /display stream
// permanently, with the socket still ESTABLISHED and its receive queue empty"
// (the SSE wedge, 2026-08-16), and it is why reconnecting to a RISC Box
// deployment used to fail until it was left alone for minutes.
//
// Pooling buys almost nothing here anyway: the tenant is on 127.0.0.1, where a
// fresh connect costs microseconds. Correctness is worth far more than that.
const tenantAgent = new http.Agent({ keepAlive: false, maxSockets: Infinity });

// Hop-by-hop headers are the proxy's own business and must never be forwarded
// (RFC 9110 7.6.1). Forwarding a client's `Connection: keep-alive` onto a leg
// we deliberately do not keep alive is exactly the kind of contradiction that
// leaves a socket in a state neither end agrees about.
const HOP_BY_HOP = ["connection", "keep-alive", "proxy-authenticate",
                    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"];

app.use("/x/:id", async (req, res) => {
  const rec = depByIdOrPrefix(req.params.id);
  if (!rec) return fail(res, 404, "not_found", "Unknown deployment.");
  // Ownership probe: the relay (and the TLS-issuance gate) asks HEAD /x/<id>
  // to learn which enclave serves an id. That is OUR knowledge, not the
  // app's - proxying it into the tenant made the answer depend on the app's
  // router treating HEAD / as a route (llm-chat 404'd it, so the relay
  // thought nobody owned the id and refused to mint the subdomain cert).
  // Answer bare-root HEADs here; HEAD on a real subpath still proxies.
  if (req.method === "HEAD" && (req.url === "/" || req.url === "")) {
    res.writeHead(204); return res.end();
  }
  // Deployer-enabled WAF (the options envelope): before auth so a flood on a
  // private deployment can't grind token verification either. The relay's
  // bare-root HEAD probe above stays exempt - that's infrastructure, not
  // app traffic.
  if (!wafGate(rec, req, res)) return;
  // The session hand-off, served BY US at the app origin. Scoped to private
  // deployments alone: a public app keeps its entire path space, so this can
  // never shadow a route a tenant already serves.
  if (!rec.public && appSessionRoute(rec, req, res)) return;
  // Public deployments serve anyone (websites/APIs). Private ones require the owner's
  // token (checked before status so a private deployment's state isn't leaked).
  if (!rec.public) {
    const addr = await addrForApp(req, rec);
    // A browser cannot put a bearer on a top-level navigation, so answering
    // JSON here is what made private apps unreachable by clicking a link. Send
    // navigations to the sign-in bounce; every other caller (CLI, fetch, API)
    // keeps the machine-readable 401 it already parses.
    if (!addr) return wantsHtml(req) ? loginBounce(rec, req, res)
                                     : fail(res, 401, "unauthorized", "Missing or invalid token.");
    if (rec.owner !== addr) return wantsHtml(req) ? loginBounce(rec, req, res, addr)
                                                  : fail(res, 403, "forbidden", "Not your deployment.");
  }
  if (rec.status !== "running") return fail(res, 409, "not_running", `Deployment is ${rec.status}.`);

  // vm backend: proxy to the app's loopback port on the app manager, on shared
  // localhost. worker backend: /x/:id/<sub> -> /tenants/:id/<sub>.
  const sub = req.url.replace(/^\/+/, "");
  let target;
  if (PROVISION_BACKEND === "vm") {
    if (!rec._vmHostPort) {
      return (rec.firewall && rec.firewall.length && !rec.firewall.some((x) => x.startsWith("http")))
        ? fail(res, 502, "no_http", "This app exposes raw TCP/UDP ports, not HTTP. Reach declared TCP ports via the WebSocket bridge at /x/:id/tcp/:port (e.g. websocat).")
        : fail(res, 502, "vm_not_ready", "The VM has no forwarded port yet.");
    }
    target = new URL(`http://127.0.0.1:${rec._vmHostPort}/${sub}`);
  } else {
    // /tenants/<id>/ is a TRUST BOUNDARY, not just a path prefix: above it live
    // the worker manager's own control endpoints (create/kill a tenant, the GPU
    // attestation, every other tenant's record). The remainder of the request
    // path is attacker-chosen on any PUBLIC deployment — no session is required
    // to reach here — and WHATWG URL resolves dot segments, `%2e%2e` included,
    // so `/x/<id>/../../tenants` would land squarely on that control plane. The
    // manager's own bearer gate is what refuses it today; this is the wall that
    // does not depend on a second component being configured correctly.
    const prefix = `/tenants/${encodeURIComponent(rec.id)}/`;
    target = new URL(`${WORKER_MGR_URL}${prefix}${sub}`);
    if (!target.pathname.startsWith(prefix))
      return fail(res, 400, "bad_path", "The path escapes this deployment's namespace.");
  }
  const headers = { ...req.headers, host: target.host };
  delete headers.authorization; // the Enclave token stays at the supervisor; the worker never sees it
  // Same invariant for the app-origin cookie, which rides an ORDINARY header
  // the browser attaches to every request: without this the tenant would read
  // its owner's session token straight out of `Cookie` and could replay it
  // against this deployment for the rest of its 12h life. Only OUR pair is
  // dropped — an app's own cookies are its business.
  const kept = stripAppCookie(headers.cookie);
  if (kept) headers.cookie = kept; else delete headers.cookie;
  for (const h of HOP_BY_HOP) delete headers[h];
  let timedOut = false;
  const up = http.request(
    { host: target.hostname, port: target.port || 80, method: req.method,
      path: target.pathname + target.search, headers, agent: tenantAgent },
    (upRes) => {
      if (res.destroyed) return up.destroy();
      res.writeHead(upRes.statusCode || 502, tenantHeaders(upRes.headers));
      upRes.pipe(res);
      // The mirror of the res 'close' rule below: a tenant leg that dies
      // mid-response must take the CLIENT leg down WITH it. pipe() forwards
      // only a CLEAN end ('end' -> res.end()); an ABORTED upstream — the app
      // reaped the connection, the tenant restarted, the socket reset — emits
      // 'close' with `complete` false, the pipe simply stops, and the client
      // socket sat ESTABLISHED and silent forever: no FIN, no error, so an
      // EventSource could not tell it from a still screen and never redialed.
      // That silence WAS the SSE wedge (live 2026-08-16, risc-box /display).
      // destroy(), not end(): an abnormal upstream end must stay abnormal on
      // the wire, or a truncated body reads as a complete one.
      upRes.on("error", () => res.destroy());
      upRes.on("close", () => { if (!upRes.complete && !res.destroyed) res.destroy(); });
    });
  // The tenant leg's inactivity bound — see TENANT_IDLE_MS. Fires only when the
  // socket has moved nothing for that long, so a live stream re-arms it forever
  // and a silent one does not. LOG it: a stall that leaves no trace is the bug
  // this whole timeout exists to make visible.
  up.setTimeout(TENANT_IDLE_MS, () => {
    timedOut = true;
    console.warn(`[proxy] ${rec.id} tenant idle ${TENANT_IDLE_MS}ms `
      + `${res.headersSent ? "mid-response" : "before headers"} ${req.method} ${req.url} - cutting`);
    // Same honesty rule as the abort paths below: once headers are on the wire,
    // tearing the connection down is the only truthful signal left.
    if (res.headersSent) res.destroy();
    else fail(res, 504, "tenant_timeout", "The app accepted the connection but did not respond.");
    up.destroy();
  });
  // Mid-response request errors follow the same rule: with headers already on
  // the wire the only honest signal left is tearing the connection down.
  up.on("error", (e) => {
    if (timedOut) return;  // the timeout above already answered; do not truncate it
    if (res.headersSent) return res.destroy();
    res.writeHead(502); res.end("upstream error: " + e.message);
  });
  // A client that dies mid-response must take the tenant leg down WITH it.
  // pipe() only stops the FLOW when its destination closes - it never destroys
  // the source - so an abandoned stream left `up` open: the paused pipe
  // backpressured into the tenant's socket until the app sat parked in a
  // blocking write that can neither finish nor fail. For an inference app
  // that parked thread holds a session slot out of a small fixed pool, so a
  // handful of closed tabs wedged the deployment into [sessions_busy] until a
  // human restarted it (live 2026-08-08). Destroying `up` closes the tenant
  // connection, the app's next write errors, and it releases what it holds.
  // `writableEnded` distinguishes the client hanging up from a response that
  // simply finished ('close' fires for both).
  res.on("close", () => { if (!res.writableEnded) up.destroy(); });
  req.on("error", () => up.destroy());
  // maxBodyMb, the counted half: Content-Length was checked in wafGate, but a
  // chunked (or lying) body only shows its size on the wire. Counting rides
  // alongside pipe's own data listener; on overflow kill both directions.
  if (rec.waf && rec.waf.maxBodyMb) {
    const cap = rec.waf.maxBodyMb * 1048576;
    let seen = 0;
    req.on("data", (c) => {
      if ((seen += c.length) <= cap) return;
      up.destroy();
      if (!res.headersSent) fail(res, 413, "waf_body", `Request body exceeds this deployment's ${rec.waf.maxBodyMb} MB limit.`);
      req.destroy();
    });
  }
  req.pipe(up);
});

app.use(express.json({ limit: "256kb" }));

async function authed(req, res, next) {
  const addr = await addrFromAuth(req);
  if (!addr) return fail(res, 401, "unauthorized", "Missing or invalid session.");
  req.address = addr; next();
}

// ============================================================================
// system
// ============================================================================
app.get("/v1/health", (_req, res) => res.json({ status: "ok", deployments: deployments.size,
  // watcher freshness is billing-critical: while it's stale, funded clocks are frozen
  watcher: FORWARDER_ADDRESS ? { lastPollOkAt: _lastPollOkAt ? new Date(_lastPollOkAt).toISOString() : null,
                                 fresh: (Date.now() - _lastPollOkAt) < WATCHER_STALE_SEC * 1000 } : null,
  // reachability watchdog: "unreachable" = our advertised hostname is gone from
  // public DNS — claiming paused, held work released (see the claim loop)
  reach: (CLAIM_READY && REACH_DNS_STRIKES) ? { state: _reach.tripped ? "unreachable" : "ok",
    strikes: _reach.strikes, host: _reach.host,
    checkedAt: _reach.checkedAt ? new Date(_reach.checkedAt).toISOString() : null } : null,
  // runner earnings (rev-7 ledgers): what this enclave's operator EOA has
  // accrued on the deployments contract and where the sweep sends it
  earnings: (CLAIM_READY && PAYOUT_ADDRESS) ? {
    payoutAddress: PAYOUT_ADDRESS,
    accruedUsdc: _earn.earned6 == null ? null : Number(_earn.earned6) / 1e6,
    withdrawnUsdc: Number(_earn.withdrawnTotal6) / 1e6,
    checkedAt: _earn.checkedAt ? new Date(_earn.checkedAt).toISOString() : null } : null,
  // proof of time (rev-9 ledgers): whether this box is currently EARNING the
  // lease seconds it holds. `lastRoundAt` going stale, or a rising `rejected`,
  // is the operator-facing signal that income has stopped — watch it, not the
  // payout report. Reported whenever claiming is on, even before a prover is
  // bound, precisely so a missing prover is visible.
  proofOfTime: CLAIM_READY ? {
    ready: PROOF_READY(),
    prover: PROOF_OF_TIME_ADDRESS || null,
    signer: PROOF_ACCOUNT ? PROOF_ACCOUNT.address : null,
    intervalSec: PROOF_INTERVAL_SEC,
    lastRoundAt: _proof.at ? new Date(_proof.at).toISOString() : null,
    proved: _proof.proved, rejected: _proof.rejected, lastError: _proof.lastError } : null,
  // The anti-sybil gate's state on THIS box. `ok: false` is the honest reason a
  // seller's enclave has gone quiet: the ledger asks a bond it is not willing
  // (CLAIM_BOND_MAX6) or not able to post, so it stops claiming rather than
  // spend the operator's float. null = claiming is off here entirely.
  claimBond: CLAIM_READY ? { ok: _bond.ok, max6: CLAIM_BOND_MAX6.toString(),
    checkedAt: _bond.at ? new Date(_bond.at).toISOString() : null, why: _bond.why } : null }));
app.get("/v1/version", (_req, res) => res.json({ service: "enclave-supervisor/0.1.0", contract: "enclave-openapi/1.0.0", chainId: CHAIN_ID }));

app.get("/v1/pricing", async (_req, res) => {
  // One model on every flavor: apps specify EXACT resources, the two billing
  // shares are CALCULATED from them. A CPU-only enclave simply has no card to
  // sell (vramGb must be 0 here).
  // Deploy coordinates ride along: deployments are created and funded on the
  // EnclaveDeployments ledger (see POST /v1/deployments for the method shapes),
  // and the console needs the contract, the USDC EIP-712 domain, and an
  // ETH/USD quote for fundEth estimates - all public, all cache-friendly.
  const [ethUsd8, usdcDomain] = await Promise.all([
    ethUsdPrice8().catch(() => null), refreshUsdcDomain().catch(() => null)]);
  const base = {
    assets: ["ETH","USDC"], gpu: IS_GPU,
    deploymentsContract: DEPLOYMENTS_ADDRESS || null, chainId: CHAIN_ID,
    usdc: USDC_ADDRESS, usdcDomain,
    ethUsd: ethUsd8 ? (Number(ethUsd8) / 1e8).toFixed(2) : null,
    model: "Deployments buy TWO shares: gpuShare (0..1 of ONE GPU card — VRAM and compute move together; 0 = CPU-only app) and cpuShare (0..1 of the node's vCPU+RAM). Apps declare their exact specs in the catalog — VRAM GB + GPU TFLOPS of a card, RAM MB + CPU GFLOPS of the node; those specs divided by this server's spec (the LARGER of the memory and compute axes per pool, rounded up to the whole percent) are the MINIMUM shares a deployment may buy. A GPU app's gpuShare must be >= its cpuShare. Billed per second, additively.",
    pricing: "The prices below are THIS enclave's, published in its EnclaveRegistry entry — every enclave sets its own, and a deployment is charged its shares of whichever one claims it. Set maxRatePerHourUsdc (create's maxRate6) to cap that: an enclave costing more than the cap cannot take the work, which is also what bounds where a deployment fails over to when its host dies. Change it any time with setMaxRate.",
    node: { vcpus: NODE_VCPUS, ramGb: NODE_RAM_GB, gflops: NODE_GFLOPS,
            wholeNodePerSecondUsdc: CPU_RATE.toFixed(7), wholeNodePerHourUsdc: (CPU_RATE * 3600).toFixed(2) },
    computeGranularity: { unit: "percent", step: 1, minPercent: MIN_COMPUTE_PCT },
    formula: "ratePerSecondUsdc = gpuShare × wholeCardPerSecond + cpuShare × wholeNodePerSecond; minGpuShare = ceilPct(max(vramGb / cardVramGb, gpuTflops / cardTflops)); minCpuShare = ceilPct(max(memMb / nodeRam, cpuGflops / nodeGflops))",
    billingIncrementSeconds: 1,
  };
  const example = (g, c) => {
    const r = rateFor(g, c);
    return { gpuShare: g, cpuShare: c,
             ...(g > 0 ? { vramGb: round1(g * CARD_VRAM_GB), gpuTflops: round1(g * CARD_TFLOPS) } : {}),
             ramGb: round1(c * NODE_RAM_GB), vcpus: round1(c * NODE_VCPUS), cpuGflops: Math.round(c * NODE_GFLOPS),
             ratePerSecondUsdc: r.toFixed(7), ratePerHourUsdc: (r * 3600).toFixed(2) };
  };
  if (!IS_GPU) return res.json({
    ...base,
    note: "CPU-only enclave: gpuShare is not served here (set it to 0); GPU apps run on GPU enclaves.",
    examples: [0.05, 0.1, 0.25, 1].map(c => example(0, c)),
  });
  res.json({
    ...base,
    card: { vramGb: CARD_VRAM_GB, tflops: CARD_TFLOPS, count: GPU_COUNT, sms: SM_TOTAL,
            wholeCardPerSecondUsdc: FULL_RATE.toFixed(7), wholeCardPerHourUsdc: (FULL_RATE * 3600).toFixed(2) },
    vramGranularityGb: GRANULARITY_GB,
    examples: [[1, 0.1], [0.5, 0.1], [0.25, 0.05], [0.05, 0.05]].map(([g, c]) => example(g, c)),
  });
});

// Fast-path claim: a freshly funded on-chain deployment shouldn't wait out
// the sweep cadence (up to CLAIM_POLL_SEC + jitter + the CPU-first grace).
// Unauthenticated on purpose - a hint is just "look at this id now"; every
// fact is re-read from the chain and the claim tx is gated exactly like the
// sweep, so the worst a bogus hint costs is a few RPC reads.
const _hintBusy = new Set();
// Per-source-IP token bucket for the (deliberately unauthenticated) claim-hint:
// a hint for an id we don't already track triggers on-chain reads against the
// shared RPC, so we bound how fast one source can drive those. Cheap local-cache
// hits (already-serving/evaluating ids) never consume a token — only hints that
// would reach the chain do. A rate-limited hint is non-fatal: the deployment is
// already on-chain, so the normal claim sweep still picks it up within
// CLAIM_POLL_SEC. Keyed on x-forwarded-for (the shim's client IP) when present,
// else on the socket peer — behind a shim that doesn't forward the client IP this
// degrades to ONE shared bucket, which still bounds RPC amplification (just not
// per-IP). CLAIM_HINT_BURST=0 disables the limit; tune the pair for your traffic.
const CLAIM_HINT_BURST = parseInt(process.env.CLAIM_HINT_BURST || "20", 10);   // bucket size (allowed burst)
const CLAIM_HINT_RPS   = parseFloat(process.env.CLAIM_HINT_RPS || "2");        // sustained refill (tokens/sec)
const _hintBuckets = new Map();   // ip -> { tokens, at }
setInterval(() => { const cut = Date.now() - 600_000; for (const [ip, b] of _hintBuckets) if (b.at < cut) _hintBuckets.delete(ip); }, 300_000).unref?.();
function hintRateOk(req) {
  if (CLAIM_HINT_BURST <= 0) return true;
  const ip = clientIpOf(req);
  const now = Date.now();
  let b = _hintBuckets.get(ip);
  if (!b) { b = { tokens: CLAIM_HINT_BURST, at: now }; _hintBuckets.set(ip, b); }
  b.tokens = Math.min(CLAIM_HINT_BURST, b.tokens + ((now - b.at) / 1000) * CLAIM_HINT_RPS);
  b.at = now;
  if (b.tokens < 1) return false;
  b.tokens -= 1;
  return true;
}
app.post("/v1/claim-hint", async (req, res) => {
  const id = String((req.body && req.body.id) || "").toLowerCase().trim();
  // force:true = a resume-class control (console Resume, `enclave resume`):
  // the ONLY hint allowed to override this box's provision-failure cooldown.
  // Most hints are automatic (the relay's post-funding nudge, the console's
  // queued-row why-probe every 30s) - if those overrode it, a crash-looping
  // app would be re-claimed the moment it failed, forever, instead of
  // resting in "failed" between escalating retries.
  const force = !!(req.body && req.body.force === true);
  if (!/^0x[0-9a-f]{64}$/.test(id))
    return fail(res, 422, "invalid_spec", "id must be the bytes32 deployment id (0x + 64 hex chars).");
  if (!CLAIM_READY || !_enclaveId)
    return fail(res, 503, "not_claiming", "This enclave is not claiming on-chain deployments right now.");
  const ex = deployments.get(id);
  if (ex && !CLAIM_TERMINAL.has(ex.status)) return res.json({ accepted: true, status: ex.status });
  if (_hintBusy.has(id)) return res.json({ accepted: true, status: "evaluating" });
  // Beyond here every path does on-chain reads — rate-limit the source first.
  if (!hintRateOk(req)) return fail(res, 429, "rate_limited", "Too many new-id claim hints from your source; retry shortly.");
  _hintBusy.add(id);
  try {
    const d = await readOnchainDeployment(id);
    if (!d || !Number(d.createdAt)) return fail(res, 404, "not_found", "No such deployment on the ledger.");
    // Preflight the CONTRACT's own gating synchronously (simulate the exact
    // claim tx) so structural failures - a stale registry pointer, an expired
    // entry, a lease race - surface HERE with the revert reason, instead of
    // "accepted: true" followed by a silent background failure. This is how
    // the 2026-07-05 wrong-registry-pointer bug should have been caught.
    // SKIP it when we already hold the live lease: no claim tx will be sent
    // (that's the resume path) and simulating one just reverts "leased",
    // wedging the only route back to a lease we own but lost the record of.
    const resuming = Number(d.leaseUntil) * 1000 > Date.now() && d.runner === _enclaveId;
    if (!resuming) {
      try {
        await chainClient.simulateContract({ address: getAddress(DEPLOYMENTS_ADDRESS), abi: CLAIM_TX_ABI,
          functionName: "claim", args: [id, _enclaveId], account: claimSigner().account });
      } catch (e) {
        return res.json({ accepted: false, reason: "claim would revert on-chain: " + (e.shortMessage || e.message) });
      }
    }
    const reason = await considerClaim(d, { hinted: true, forced: force, background: true });
    if (reason) return res.json({ accepted: false, reason });
    res.json({ accepted: true, status: "claiming" });
  } catch (e) {
    fail(res, 502, "chain_unreachable", "Could not evaluate the hint: " + (e.shortMessage || e.message));
  } finally { _hintBusy.delete(id); }
});

// --- relay services: the NETWORK this box carries, beside the compute it sells --
// Any host may also relay; the two are independent, and a box with no capacity
// at all is simply one that ONLY relays (which is what the console renders as a
// "relay" row — the badge is read from having no resources, not from this
// block). What differs between relays is which of the fleet's network services
// they actually offer, and that is a per-BOX fact about which daemons run
// alongside this supervisor and how the machine is wired — a routed /64, a
// bindable port range, a public address that CGNAT boxes can dial in to. None
// of it is discoverable from in here, so it is declared, and broadcast on
// /availability like every other capability the fleet ANDs or routes on.
// Absent entirely = this box carries nothing for anyone but itself.
const RELAY_SERVICES = (() => {
  const on = (k) => /^(1|true|yes|on)$/i.test((process.env[k] || "").trim());
  const svc = {
    sni:       on("RELAY_SNI"),        // app-zone 443 passthrough (relay/relay.js) — the data path every app subdomain crosses
    tcp:       on("RELAY_TCP"),        // per-deployment dedicated-IPv6 raw TCP (relay/tcp6-relay.js)
    udp:       on("RELAY_UDP"),        // per-deployment dedicated-IPv6 datagrams (relay/udp-relay.js)
    egress:    on("RELAY_EGRESS"),     // per-deployment outbound source address (relay/egress-relay.js)
    tunnelHub: on("RELAY_TUNNEL_HUB"), // accepts reverse tunnels — the only way onto the network for a seller behind CGNAT
  };
  const region = (process.env.RELAY_REGION || "").trim();
  const prefix = (process.env.RELAY_V6_PREFIX || "").trim();   // the routed /64 the dedicated-IP features hand out of
  const ports  = (process.env.RELAY_PORTS || "").trim();       // the public port range this box will bind, as configured
  // What the app zone answers with for a deployment that CHOOSES this box as
  // its relay ({"network":{"relay":"…"}}). Without it the box still relays for
  // the fleet default, but it cannot be picked: there is no address to point a
  // name at. Declare the v6 half only if the passthrough listener really binds
  // one — a chosen relay answers only from its own addresses, so an unbacked
  // AAAA here is a black hole for every v6-preferring client.
  const addr   = (process.env.RELAY_PUBLIC_ADDRESS  || "").trim();
  const addr6  = (process.env.RELAY_PUBLIC_ADDRESS6 || "").trim();
  if (!Object.values(svc).some(Boolean)) return null;          // declares no service = not a relay, say nothing
  return { ...svc, ...(addr ? { address: addr } : {}), ...(addr6 ? { address6: addr6 } : {}),
           ...(region ? { region } : {}), ...(prefix ? { v6Prefix: prefix } : {}),
           ...(ports ? { ports } : {}) };
})();
if (RELAY_SERVICES)
  console.log(`[relay-services] carrying ${Object.entries(RELAY_SERVICES)
    .filter(([, v]) => v === true).map(([k]) => k).join(", ")}`
    + (RELAY_SERVICES.region ? ` · ${RELAY_SERVICES.region}` : ""));

// ---- shielded GPU: a card this enclave uses but does NOT trust ---------------
//
// A metal box can offer GPU work with the card on the untrusted HOST, outside the
// CVM and outside the launch measurement, reached over a masked-offload protocol
// that sends it public weights and one-time pads and verifies everything it sends
// back (docs/shielded-inference.md). The guest's boot probe writes this file after
// one real masked GEMM comes back exact, verified, and with the op denylist
// enforced on the wire; it deletes the file when the probe does not pass.
//
// IT REPORTS `gpu: true`, and this reverses an earlier decision worth recording.
// The first version kept shielded capacity in its own namespace so that a router
// placing ordinary GPU work could not land on a box whose card sits outside the
// enclave. The effect was a pool nothing could buy: every GPU consumer in the
// stack -- the router, the ledger's gpuMilli, the console -- asks `gpu`, so a
// card that answered only to a new question was unreachable, and the box
// advertised a price for hardware no deployment could rent.
//
// So a shielded card is a card. It is metered, priced and sold as one, and the
// `shielded` block below still says WHERE it is, which is the part a buyer
// actually needs: the badge, the attestation and this block all keep saying that
// the silicon is on an untrusted host and the tenant reaches it through masked
// offload. What changed is that saying so no longer costs the card its market.
// A shielded card becomes THIS ENCLAVE'S card the first time the probe passes.
// One-way on purpose: a verdict that later disappears stops us advertising FREE
// capacity (availability reads the live verdict every time), but it must not
// retract a card that live leases are already running on -- that would strand
// tenants mid-lease over a probe that may simply be restarting.
// (`_shieldedAdopted` is declared with the card pool above, because the pool
// math reads it and the POOL_SELFTEST seam runs before this point.)
function adoptShieldedCards(values) {
  if (!values.length) return;
  _shieldedAdopted = true;
  IS_GPU = true;
  CARD_VRAM_SRC = "shielded-probe";
  for (const card of gpuCards) if (card.shielded) card.available = false;
  for (const v of values) {
    while (gpuCards.length <= v.id)
      gpuCards.push({ id: gpuCards.length, uuid: null, vramFree: 0, computeFree: 1,
                      vramTotal: 0, shielded: true, available: false });
    const card = gpuCards[v.id];
    const budget = v.vramBudgetGb > 0 ? v.vramBudgetGb : v.vramGb;
    const old = card.vramTotal ?? CARD_VRAM_GB;
    card.vramFree = Math.max(0, card.vramFree + budget - old);
    card.vramTotal = budget;
    card.shielded = true;
    card.available = true;
    card.proof = v;
    card.uuid = v.deviceUuid || card.uuid;
  }
  GPU_COUNT = _shieldedPool ? 1 : gpuCards.length;
  // Legacy consumers have one sizing unit. Use a conservative per-card floor;
  // each actual allocation and launch uses the selected card's own budget.
  const known = gpuCards.filter(c => c.shielded && c.vramTotal > 0);
  const combine = xs => _shieldedPool ? xs.reduce((a, b) => a + b, 0) : Math.min(...xs);
  CARD_VRAM_GB = combine(known.map(c => c.vramTotal));
  const rated = known.map(c => c.proof?.cardTflops || (c.proof?.gmacPerSec || 0) * 2 / 1000).filter(v => v > 0);
  if (rated.length) CARD_TFLOPS = round1(combine(rated));
  const sms = known.map(c => c.proof?.smCount || 0).filter(v => v > 0);
  if (sms.length) SM_TOTAL = combine(sms);
  const prices = known.map(c => c.proof?.pricePerSec6 || 0).filter(v => v > 0);
  if (_shieldedPool) {
    const rate = _shieldedPool.pricing;
    SELL_GPU_PRICE6 = Math.round((CARD_TFLOPS * rate.tflopUsdHr + CARD_VRAM_GB * rate.vramGiBUsdHr) * 1e6 / 3600);
  } else if (prices.length) SELL_GPU_PRICE6 = Math.min(...prices);
  // Reconstruct every sibling's holds after discovery (restored arrays may
  // predate the probe that creates card 1..N in this process).
  if (_shieldedPool) reconcilePools();
}

const SHIELDED_VERDICT = process.env.SHIELDED_VERDICT || "/run/shielded-gpu.json";
let _shieldedCache = { at: 0, cards: [] };
function shieldedCapacity(cardId = null) {
  const now = Date.now();
  if (now - _shieldedCache.at >= 10_000) {
    const cards = [], ids = new Set(), endpoints = new Set(), devices = new Set();
    let pool = null;
    try {
      const raw = JSON.parse(readFileSync(SHIELDED_VERDICT, "utf8"));
      const rp = raw?.pool;
      if (rp?.mode === "layers" && Array.isArray(rp.cardIds) && rp.cardIds.length > 0 && rp.cardIds.length <= 16 &&
          rp.cardIds.every((id, i) => id === i) &&
          [rp.pricing?.tflopUsdHr, rp.pricing?.vramGiBUsdHr].every(v => Number.isFinite(v) && v >= 0 && v <= 100) &&
          rp.pricing.tflopUsdHr + rp.pricing.vramGiBUsdHr > 0) pool = rp;
      // A pool cannot silently degrade into separately priced cards.
      if (rp && !pool) throw new Error("invalid shielded pool");
      for (const v of Array.isArray(raw?.cards) ? raw.cards : [raw]) {
        const id = v?.id === undefined ? 0 : Number(v.id);
        if (!v || !v.exact || !v.verified || !v.lie_rejected || !v.denylist_refused || !(v.vram_total_gb > 0)) continue;
        if (!Number.isInteger(id) || id < 0 || id >= 16 || ids.has(id) || !v.endpoint || endpoints.has(v.endpoint)) continue;
        if (v.deviceUuid && devices.has(String(v.deviceUuid).toLowerCase())) continue;
        if (pool && (!Number.isFinite(v.card_tflops) || v.card_tflops <= 0 || !(v.vram_budget_gb > 0))) continue;
        ids.add(id); endpoints.add(v.endpoint); if (v.deviceUuid) devices.add(String(v.deviceUuid).toLowerCase());
        cards.push({
        id,
        deviceUuid: String(v.deviceUuid || ""),
        card: String(v.name || "gpu").slice(0, 64),
        vramGb: Number(v.vram_total_gb) || 0,
        // vram_free_gb is what the card can still be SOLD for: the budget minus
        // what live tenants have reserved at HELLO, capped by the driver's own
        // free figure (gsup computes it; protocol 1.3). Before 1.3 the worker
        // held its whole budget at start-up and this read 0.43 GB of 6.5 with
        // one tenant on an otherwise idle card -- the hold hid itself from
        // every free figure (shielded/REPORT.md 13.14.2).
        vramFreeGb: Number(v.vram_free_gb) || 0,
        vramBudgetGb: Number(v.vram_budget_gb) || 0,
        vramReservedGb: Number(v.vram_reserved_gb) || 0,   // held by live tenants, per their HELLOs
        gmacPerSec: Number(v.field_gmac_per_s) || 0,
        cardTflops: Number(v.card_tflops) || 0,
        smCount: Number(v.sm_count) || 0,
        // The fraction of the card's COMPUTE this box sells (operator config,
        // enforced host-side by MPS on the worker). VRAM has its own budget;
        // this caps the pool's computeFree so the ledger never sells SM time
        // the worker cannot legally burn. Absent = the whole card, as before.
        computeShare: Math.min(1, Number(v.compute_share) > 0 ? Number(v.compute_share) : 1),
        capability: String(v.capability || "").slice(0, 8),
        roundTripMs: Number(v.round_trip_ms) || 0,
        verifiedAt: String(v.at || ""),
        endpoint: String(v.endpoint || ""),
        vsockPort: Number(v.vsockPort) || 0,
        ...shieldedShmSpec(v, id),
        pricePerSec6: Number(v.pricePerSec6) || 0,
        // host-configured tenant knobs (SHIELDED_* only; the guest wrote the
        // file and the manager filters again, this is the middle of three)
        tenantEnv: Object.fromEntries(Object.entries((v.tenantEnv && typeof v.tenantEnv === "object") ? v.tenantEnv : {})
          .filter(([k, val]) => /^SHIELDED_[A-Z0-9_]{0,63}$/.test(k) && typeof val === "string" && val.length <= 256)
          .slice(0, 32)),
      });
      }
    } catch {} // An absent or malformed verdict withdraws capacity, not leases.
    for (const card of gpuCards) if (card.shielded) card.available = false;
    if (pool && !_shieldedPool) {
      // Do not publish or admit a fraction of a pool while its siblings are
      // still booting. Once adopted, retain the total through probe failures.
      if (cards.length !== pool.cardIds.length || !pool.cardIds.every(id => cards.some(c => c.id === id))) cards.length = 0;
      else _shieldedPool = pool;
    }
    if (_shieldedPool && (!pool || JSON.stringify(pool.cardIds) !== JSON.stringify(_shieldedPool.cardIds))) cards.length = 0;
    if (_shieldedPool) {
      for (let i = cards.length - 1; i >= 0; i--) if (!_shieldedPool.cardIds.includes(cards[i].id)) cards.splice(i, 1);
    }
    adoptShieldedCards(cards);
    _shieldedCache = { at: now, cards };
  }
  return cardId == null ? (_shieldedPool ? shieldedPoolCapacity() : (_shieldedCache.cards[0] || null))
    : (_shieldedCache.cards.find(c => c.id === Number(cardId)) || null);
}

function shieldedShmSpec(v, cardId) {
  const bytes = v?.shmBytes;
  return v?.shmPath === `/dev/enclave-shielded-shm/card-${cardId}` &&
    Number.isInteger(bytes) && bytes >= 8 * 1048576 && bytes <= 64 * 1048576 && !(bytes & (bytes - 1))
    ? { shmPath: v.shmPath, shmBytes: bytes } : {};
}

function shieldedPoolCapacity() {
  const cards = _shieldedPool.cardIds.map(id => gpuCards[id]?.proof).filter(Boolean);
  if (cards.length !== _shieldedPool.cardIds.length) return null;
  const sum = key => cards.reduce((a, c) => a + (Number(c[key]) || 0), 0);
  return { pooled: true, mode: "layers", card: "GPU", cardCount: cards.length,
    vramGb: CARD_VRAM_GB, vramBudgetGb: CARD_VRAM_GB,
    vramFreeGb: maxFreeGpuShare() * CARD_VRAM_GB,
    vramReservedGb: sum("vramReservedGb"), cardTflops: CARD_TFLOPS,
    smCount: SM_TOTAL, pricePerSec6: SELL_GPU_PRICE6,
    pricing: _shieldedPool.pricing, endpoint: cards[0].endpoint,
    available: _shieldedPool.cardIds.every(id => gpuCards[id].available),
  };
}

function shieldedLaunchSpec(cardId, gpuShare, vramGb, heldCards) {
  // A restored legacy handle still reserves one card. Keep its route until a
  // resize/reclaim replaces the handle; never grant unbooked sibling VRAM.
  if (_shieldedPool && (heldCards || vramGb == null)) {
    const holds = heldCards || _shieldedPool.cardIds.map(id => ({ cardId: id, vramGb: gpuShare * cardVram(gpuCards[id]) }));
    const workers = holds.map(h => {
      const card = shieldedCapacity(h.cardId);
      if (!card) throw new Error(`Shielded GPU ${h.cardId} is unavailable; refusing a partial pool launch`);
      return { cardId: card.id, deviceUuid: card.deviceUuid, endpoint: card.endpoint,
        vsockPort: card.vsockPort || 0, vramGb: h.vramGb, ...shieldedShmSpec(card, card.id) };
    });
    return { pooled: true, mode: "layers", endpoint: workers[0].endpoint, workers,
      vramGb: workers.reduce((a, w) => a + w.vramGb, 0),
      tenantEnv: gpuCards[workers[0].cardId].proof.tenantEnv || {} };
  }
  const card = shieldedCapacity(cardId ?? 0);
  if (!card) throw new Error(`Shielded GPU ${cardId ?? 0} is unavailable; refusing to route its lease to another card`);
  return {
    endpoint: card.endpoint, cardId: card.id,
    vramGb: vramGb ?? round1(gpuShare * (card.vramBudgetGb || card.vramGb)),
    ...(card.vsockPort ? { vsockPort: card.vsockPort } : {}),
    ...shieldedShmSpec(card, card.id),
    ...(Object.keys(card.tenantEnv || {}).length ? { tenantEnv: card.tenantEnv } : {}),
  };
}

// Exercise the production allocator and launch routing against a changing
// verdict without starting services or touching a real device.
if (process.env.SHIELDED_POOL_SELFTEST) {
  const actions = JSON.parse(process.env.SHIELDED_POOL_SELFTEST), handles = new Map(), results = [];
  shieldedCapacity();
  for (const action of actions) {
    if (action.verdict) {
      writeFileSync(SHIELDED_VERDICT, JSON.stringify(action.verdict));
      _shieldedCache.at = 0; shieldedCapacity();
    }
    let handle, route, error;
    if (action.alloc) {
      const q = normalizeGpuReq(action.alloc.gpu, action.alloc.cpu);
      handle = allocGpu(q.vramGb, q.computeShare, q.cpuShare);
      if (handle) { handles.set(action.alloc.name, handle); deployments.set(action.alloc.name, { status: 'running', _gpu: handle }); }
    }
    if (action.launch) {
      const h = handles.get(action.launch);
      try { route = shieldedLaunchSpec(h.cardId, h.computeShare, h.vramGb, h.cards); }
      catch (e) { error = e.message; }
    }
    if (action.release) { releaseGpu(handles.get(action.release)); handles.delete(action.release); deployments.delete(action.release); }
    if (action.reconcile) reconcilePools();
    results.push({ handle, route, error, free: maxFreeGpuShare(), cpu: maxFreeCpu(),
      capacity: shieldedCapacity(), pricePerSec6: SELL_GPU_PRICE6, sizingUnits: GPU_COUNT,
      cards: gpuCards.map(c => ({ id: c.id, total: cardVram(c), free: c.vramFree, available: c.available })) });
  }
  console.log(JSON.stringify(results)); process.exit(0);
}

// SHIELDED_SELFTEST=1 with SHIELDED_VERDICT pointing at a verdict file: parse
// it exactly as /availability would, print one JSON line with the parsed block
// and the gpu-pool figure it yields, and exit -- same contract as the seams
// above (test/shielded-capacity.test.mjs drives it).
if (process.env.SHIELDED_SELFTEST) {
  const sh = shieldedCapacity();
  const gpuFree = !sh ? 0 : maxFreeGpuShare();
  console.log(JSON.stringify({ shielded: sh, cardVramGb: _shieldedAdopted ? CARD_VRAM_GB : 0,
                               cards: _shieldedCache.cards,
                               cardTflops: _shieldedAdopted ? CARD_TFLOPS : 0,
                               smTotal: _shieldedAdopted ? SM_TOTAL : 0, gpuShareFree: round3(gpuFree),
                               vramFreeGb: _shieldedAdopted ? round1(gpuFree * CARD_VRAM_GB) : 0 }));
  process.exit(0);
}

app.get("/availability", async (_req, res) => {
  // FIRST, before anything reads IS_GPU: a shielded card is adopted by this call,
  // and the object below captures IS_GPU as it builds. Without this the first
  // /availability after boot would publish a CPU-only box and the relay would
  // cache exactly the answer this whole change exists to stop it giving.
  shieldedCapacity();
  // Every enclave reports BOTH pools: gpuShareFree (the largest slice one card
  // can still take; 0 on a CPU-only enclave) and cpuShareFree (the node's
  // leftover vCPU+RAM share — on a GPU enclave that leftover is rentable by
  // CPU-only apps). The cpu pool prefers the vm backend's live accounting (the
  // wasm-manager admits by cpuShare); the gpu pool prefers the worker backend's
  // accounting when that backend holds the card, else our own card allocator.
  // Callers route on the pair: GPU work needs gpuShareFree, CPU-only work fits
  // wherever cpuShareFree is big enough. `maxShare` is a DEPRECATED alias of
  // the flavor's primary pool, kept one release for old routers.
  const shape = (cpuFree, gpuFree, source, note) => ({
    gpu: IS_GPU, type: IS_GPU ? "gpu" : "cpu",
    // The CPU-TEE technology this box DETECTED from its own attestation document
    // ("amd-sev-snp" / "intel-tdx"; a metal dev box reports its unattested
    // format; null until the first document read lands). Never asserted from
    // config or from the flavor: the fleet row badges "TEE CPU" from this, and a
    // box with some other root of trust (a phone-anchored host, one day) must
    // not inherit a green pill it did not prove. See vmTech().
    teeCpu: vmTech(),
    gpuShareFree: round3(gpuFree), cpuShareFree: round3(cpuFree),
    usedGpuShare: IS_GPU ? round3(1 - gpuFree) : 0, usedCpuShare: round3(1 - cpuFree),
    maxShare: round3(IS_GPU ? gpuFree : cpuFree),
    vcpusFree: round1(cpuFree * NODE_VCPUS), ramGbFree: round1(cpuFree * NODE_RAM_GB),
    cpuGflopsFree: Math.round(cpuFree * NODE_GFLOPS),
    nodeVcpus: NODE_VCPUS, nodeRamGb: NODE_RAM_GB, nodeGflops: NODE_GFLOPS,
    smFree: IS_GPU ? Math.round(gpuFree * SM_TOTAL) : 0, smTotal: IS_GPU ? SM_TOTAL : 0,
    vramFreeGb: IS_GPU ? round1(gpuFree * CARD_VRAM_GB) : 0,
    gpuTflopsFree: IS_GPU ? round1(gpuFree * CARD_TFLOPS) : 0,
    cardVramGb: IS_GPU ? CARD_VRAM_GB : 0, cardTflops: IS_GPU ? CARD_TFLOPS : 0, cards: GPU_COUNT,
    ...(_shieldedPool ? { gpuPool: "layers", physicalGpuCount: gpuCards.length } : {}),
    ...(IS_GPU ? { cardVramSource: CARD_VRAM_SRC } : {}),   // "nvidia-smi"/"manager"/"worker" = probed hardware; "env"/"default" = config fallback
    ...(() => {
      const sh = shieldedCapacity();
      if (!sh) return {};
      // askShieldedPricePerSec6 sits at the TOP level beside askGpu/askCpu, not
      // inside the block, because that is where every price consumer already
      // looks. The block carries what the card IS; the ask is what it costs.
      return { shielded: sh,
               shieldedCards: _shieldedCache.cards.map(v => ({ ...v,
                 gpuShareFree: round3(Math.max(0, Math.min(gpuCards[v.id].computeFree,
                   cardFreeVram(gpuCards[v.id]) / cardVram(gpuCards[v.id])))) })),
               ...(sh.pricePerSec6 > 0 ? { askShieldedPricePerSec6: sh.pricePerSec6 } : {}) };
    })(),   // a card on the UNTRUSTED host, reached by masked offload; NOT `gpu` — see shieldedCapacity()
    ...(RELAY_SERVICES ? { relay: RELAY_SERVICES } : {}),   // network this box carries for the fleet; see the block above
    networkOptions: true,   // this build accepts the envelope's `network` namespace (per-deployment relay choice). SAME FLEET-AND RULE as waf/config/gpuOptional and for the sharpest reason: the envelope is fail-closed, so a deployment carrying {"network":…} that lands on a runner which predates this is REFUSED OUTRIGHT, not degraded. The console must keep the Network tab hidden until every live runner reports true
    waf: true,   // this build accepts + enforces the deployment-options envelope (waf); the relay ANDs this across the fleet and the console shows the Protection controls only then
    configOverride: true,   // this build accepts the envelope's `config` namespace (per-deployment app-config override); same fleet-AND rule — the console unlocks the App config box only when every live runner honors it
    configEdit: true,   // this build's audit re-applies an owner's setConfig to LIVE deployments (waf live-swapped, config = restart in place); without it an edit only lands at the next re-claim — same fleet-AND rule
    shareResize: true,   // this build's audit re-slices a LIVE deployment on an owner's setShares (rev-6 ledgers); without it the billing would change while the served slice silently didn't — same fleet-AND rule, clients refuse the tx against an older fleet
    gpuOptional: true,   // this build understands {"gpu":{"optional":true}}: a GPU-dialled deployment may fall back to a CPU-only enclave instead of queueing. Same fleet-AND rule — against an older fleet the envelope would be REFUSED outright (unknown namespace), so the console must not offer the control until every live runner knows it
    secrets: SECRETS_CAPABLE,   // this build pulls relay-staged per-deployment secrets into the guest env at every launch; fleet-AND'd with the relay's own secretsEnabled() before clients see it. The fetch authenticates with a key DERIVED FROM THE FLEET SECRET, so a box running its own minted SECRET (a metal enclave without cfg.fleetSecret) sets SECRETS_CAPABLE=0 and reports false — honest, and the fleet-AND then hides the feature rather than stranding secret-bearing deploys on it
    secretsInConfig: SECRETS_CAPABLE,   // this build also resolves $NAME/${NAME} placeholders in config STRING values from those secrets at launch (wasm-manager _subst_secrets); same fleet-AND rule
    customDomains: !!DOMAINS_API,   // this build pulls a deployment's owner-attached hostnames, mints their certificates in-CVM (dns-01 through the delegated challenge alias) and hands the guest ENCLAVE_HOSTS; fleet-AND'd with the relay's domainsEnabled() before clients see it — a lease landing on a runner without it would leave the customer's own domain refusing handshakes with nothing on the dashboard to explain why
    devDeploy: true,   // this build's approval gate admits PENDING catalog versions for PRIVATE deployments (publisher dev-mode testing); public deploys of pending versions stay refused — same fleet-AND rule, clients only offer the option when every live runner honors it
    rateCap: true,   // this build honors per-deployment rate caps (ledger rev 8): it prices claims off its own registry entry and treats a cap-blocked renew as "stop at lease end", not an error to retry — same fleet-AND rule, so clients only offer cap edits when every live runner would behave
    // FREE SELF-HOSTING (ledger rev 12): this build prices a claim at ZERO when
    // its registry-declared payout wallet owns the deployment, so it will take
    // such work with an EMPTY BALANCE. Advertised per box, not fleet-AND'd by
    // meaning: it is a property of one seller's own box, and the only thing an
    // older runner does wrong is decline its owner's unfunded deployment as
    // "out of funded time" — the tenant sees Queued rather than a wrong charge.
    // Clients read it to explain that, and `payoutWallet` says WHOSE.
    selfHostFree: true,
    payoutWallet: _declaredPayout,   // null until the wallet declares itself on-chain
    // PROVES its service (ledger rev 9): it mints an in-CVM proof key, publishes
    // it to the registry, and posts block-anchored checkpoints for every tenant
    // it is serving — including a final one at teardown so a partial period
    // still pays. Advertised per box and AND-ed across the fleet by the relay,
    // because "every host here is held to account for the hours it bills" is
    // only true if it is true of ALL of them. A box reports false when it has no
    // prover bound or no key, which is exactly when it would be serving for free.
    proofOfTime: PROOF_READY(),
    // THE PRICE of renting this machine: USDC 6dp per second for a FULL node /
    // FULL card, the same numbers this enclave publishes in its registry entry
    // and the ledger charges tenants out of (shares/1000 of each). Clients rank
    // boxes on it, quote "from $X/hr" off the cheapest, and set a deployment's
    // rate cap against it. The gpu ask is omitted on a CPU-only enclave — there
    // is no card to sell.
    askCpuPricePerSec6: SELL_CPU_PRICE6,
    ...(IS_GPU && SELL_GPU_PRICE6 > 0 ? { askGpuPricePerSec6: SELL_GPU_PRICE6 } : {}),
    claimEnabled: CLAIM_READY && !!_enclaveId,   // whether this enclave CLAIMS ledger work RIGHT NOW: configured for it AND its on-chain registration landed (_enclaveId is only set by a successful register tx — a staged seller with an unfunded gas EOA truthfully reports false until the first register confirms). The relay sizes app minimums, fleet capacity and the deploy target list over CLAIMING enclaves only
    source, ...(note ? { note } : {}), updatedAt: new Date().toISOString(),
  });
  try {
    const h = PROVISION_BACKEND === "vm" ? await vmHealth() : await mgrHealth();
    const c = h.capacity || {};
    const cpuFree = PROVISION_BACKEND === "vm" ? (c.cpuShareFree ?? c.maxShare ?? maxFreeCpu()) : maxFreeCpu();
    // vm backend: fold the wasm-manager's VRAM-reservation ledger (what the
    // device can still physically hold, from the process owner) into our card
    // allocator's plan - a physical-vs-planned divergence (leaked process,
    // out-of-band device users) then surfaces as reduced advertised capacity
    // instead of a claim that fails at provision time.
    // A SHIELDED card is not in the manager's VRAM ledger -- the manager owns no
    // CUDA device here, so its gpuShareFree is 0 and folding it in would
    // advertise nothing. The honest source is the probe's own reading of the
    // card: the budget minus what live tenants reserved at HELLO, capped by the
    // driver's free figure on the untrusted host -- so it accounts for the
    // tenants this box placed AND for whatever else that host is doing with the
    // card (on a desktop, an X server; a game, 2026-08-26).
    // A verdict that has gone away means the path is not provable right now:
    // advertise no free capacity, without retracting the card itself.
    const shNow = shieldedCapacity();
    const gpuFree = !IS_GPU || (_shieldedPool && h.shieldedPool !== true) ? 0
      : shNow ? maxFreeGpuShare()
      : _shieldedAdopted ? 0
      : PROVISION_BACKEND === "vm" ? Math.min(maxFreeGpuShare(), c.gpuShareFree ?? Infinity)
      : (c.gpuShareFree ?? c.maxShare ?? maxFreeGpuShare());
    // wasi-nn readiness rides along (vm backend): `nn` says whether GPU
    // deployments can launch; `nnProbe` carries the boot probe's diagnosis,
    // making a broken GPU path visible from outside without operator access.
    const nn = PROVISION_BACKEND === "vm" && h.nn !== undefined ? { nn: h.nn, nnProbe: h.nnProbe } : {};
    // Same idea for the cross-tenant loopback wall: the manager reports false when
    // its wasmtime predates wasmtime-loopback.patch, and a box running tenants
    // without that wall should be sayable from outside rather than only in a log
    // nobody can read. Reported ONLY when it is missing — an absent key means the
    // box either enforces it or is too old to have an opinion.
    const lbw = PROVISION_BACKEND === "vm" && h.loopbackWall === false ? { loopbackWall: false } : {};
    // WASIp3 serving capability: the manager probed its own wasmtime for
    // `-S p3` (and honors the operator's WASM_P3 switch). Reported as a real
    // boolean either way — the relay ANDs `p3 === true` across the claiming
    // fleet before clients rely on it, and this box's own claim gate refuses
    // wasi-0.3 versions when it is false. Absent = a manager too old to have
    // an opinion, which the AND treats as false (correct: it predates p3).
    const p3 = PROVISION_BACKEND === "vm" && h.p3 !== undefined ? { p3: h.p3 === true } : {};
    // cooperative threads: same shape, same AND semantics at the relay
    const cth = PROVISION_BACKEND === "vm" && h.coopThreads !== undefined ? { coopThreads: h.coopThreads === true } : {};
    // shared-everything threads (SET): same shape, same AND semantics
    const setc = PROVISION_BACKEND === "vm" && h.set !== undefined ? { set: h.set === true } : {};
    // wasm64 (memory64) core modules — the >4 GiB guests: same shape, same
    // AND semantics. Absent = a manager too old to have an opinion = false.
    const m64 = PROVISION_BACKEND === "vm" && h.mem64 !== undefined ? { mem64: h.mem64 === true } : {};
    // catalog rev-7 large configs: the manager fetches a version's configCid and
    // accepts the bytes only because they re-hash to it, then delivers past the
    // argv ceiling by file. Same shape and AND semantics as p3 — absent means a
    // manager too old to have an opinion, which the AND correctly reads as
    // false. `configMaxBytes` is the spam ceiling it will actually honor, so
    // the publish UI can size its own check off the fleet rather than guess.
    const ccid = PROVISION_BACKEND === "vm" && h.configCid !== undefined
      ? { configCid: h.configCid === true,
          ...(h.configMaxBytes ? { configMaxBytes: Number(h.configMaxBytes) } : {}) } : {};
    // attached model volumes this enclave carries (Modelwrap): the console and
    // clients read this to know which volumes a deployment here can mount.
    const vols = PROVISION_BACKEND === "vm" && Array.isArray(h.volumes) ? { volumes: h.volumes } : {};
    // The instance reconciler's last pass. Published because the ledger fields
    // below tell you capacity is missing but never WHY, and the sweep that is
    // supposed to reclaim it had no voice at all: a box leaking a tenant and a
    // box with nothing to reclaim were indistinguishable from outside.
    // ...and the dead-man switch's own state, both halves: whether we are still
    // vouching (ours) and whether the manager considers the lease armed and
    // enforcing (its). A switch nobody can see is a switch nobody knows is off.
    const sweep = { instanceSweep: instanceSweepStatus(), tenantVouch: { ..._lastVouch },
                    ...(c.tenantLease ? { tenantLease: c.tenantLease } : {}) };
    // RAM-reservation ledger passthrough (vm backend with accounting on): the
    // binding constraint behind cpuShareFree when it is tighter than the share
    // pool - lets consumers show/gate on exact MB headroom, not just the folded
    // fraction. sharePoolFree is the raw share ledger for comparison.
    const ram = PROVISION_BACKEND === "vm" && c.ramBudgetMb
      ? { ramBudgetMb: c.ramBudgetMb, ramCommittedMb: c.ramCommittedMb, ramFreeMb: c.ramFreeMb,
          // how much of `committed` is model weights held resident rather than
          // tenant reservations — the term that makes an idle-looking box read
          // 85% used, so consumers can SAY why instead of leaving it a mystery
          ...(c.ramNnResidentMb ? { ramNnResidentMb: c.ramNnResidentMb } : {}),
          ...(c.sharePoolFree !== undefined ? { sharePoolFree: c.sharePoolFree } : {}) } : {};
    // VRAM-reservation ledger passthrough (vm backend with accounting on):
    // the physical constraint behind gpuShareFree when it is tighter than the
    // card allocator's plan - same contract as the RAM ledger above. Beside
    // the arithmetic rides the DEVICE's own count (manager nvidia-smi) and
    // the divergence between the two - a positive divergence is memory the
    // card holds that no ledger sold, which is exactly the 2026-08-18
    // incident (~104 GiB orphaned by an in-place update, every ledger blind
    // to it), so it must be visible from outside without a shell.
    const vram = PROVISION_BACKEND === "vm" && c.vramBudgetGb
      ? { vramBudgetGb: c.vramBudgetGb, vramCommittedGb: c.vramCommittedGb, vramLedgerFreeGb: c.vramFreeGb,
          ...(c.vramDevFreeGb != null ? { vramDevFreeGb: c.vramDevFreeGb, vramDevTotalGb: c.vramDevTotalGb,
                                          vramDivergenceGb: c.vramDivergenceGb } : {}) } : {};
    // The envelope's `configCid` namespace (a per-deployment config override
    // SPLIT the way catalog rev 7 splits a version's: bulk at a pinned CID,
    // `config` demoted to the routing manifest). It needs BOTH halves — this
    // build to parse the namespace, and the MANAGER to fetch and hash-verify the
    // pinned body — so it is derived from ccid rather than declared, and stays
    // false on a box whose manager is too old or unreachable. A box that
    // understood the namespace but could not resolve a CID would take the claim
    // and then fail every launch on the fetch, which is worse than refusing it.
    // Same fleet-AND rule as the rest of the envelope, and for the sharp reason
    // the whole thing is fail-closed: a deployment carrying {"configCid":…} that
    // lands on a runner without it is REFUSED OUTRIGHT, not degraded.
    const ccidOv = { configCidOverride: ccid.configCid === true };
    return res.json({ ...shape(cpuFree, gpuFree, PROVISION_BACKEND === "vm" ? "vmmanager" : "worker"), ...nn, ...lbw, ...p3, ...cth, ...setc, ...m64, ...ccid, ...ccidOv, ...vols, ...ram, ...vram, ...sweep });
  } catch (e) {
    return res.json(shape(maxFreeCpu(), maxFreeGpuShare(), "fallback",
      `${PROVISION_BACKEND === "vm" ? "wasm" : "worker"} manager unreachable`));
  }
});

// External proof that MPS caps are live: each running tenant's granted SM count
// (sanitized - no tenant ids). A 25% tenant should report ~33 of 132 SMs.
//
// HISTORY WARNING (2026-08-18): this endpoint asks the WORKER manager, which
// serves only the PTX-submission path. On a vm-backend box every LLM tenant
// lives in the WASM manager instead, so the worker's "capacity" here read
// {vramFreeGb: 140.4, tenants: []} while the card physically had ~35 GB free
// and eleven instances were live - and the MCP `gpu_capacity` tool repeated
// that emptiness as the box's truth. The response now carries the serving
// backend's ledger AND the device-measured count beside the worker view, so
// no caller can mistake one container's empty ledger for an empty card.
app.get("/v1/gpu", async (_req, res) => {
  shieldedCapacity();
  if (_shieldedAdopted) return res.json({
    ok: true, role: "shielded", ...(_shieldedPool ? { pool: shieldedPoolCapacity() } : {}), cards: gpuCards.filter(c => c.shielded).map(c => ({
      ...c.proof, id: c.id, available: !!c.available,
      vramBudgetGb: cardVram(c), vramLedgerFreeGb: round3(c.vramFree),
      vramFreeGb: round3(Math.max(0, cardFreeVram(c))),
      gpuShareFree: round3(Math.max(0, Math.min(c.computeFree, cardFreeVram(c) / cardVram(c) || 0))),
    })),
  });
  if (!IS_GPU) return fail(res, 404, "no_gpu", "This is a CPU-only enclave: no GPU is attached.");
  try {
    const h = await mgrHealth(5000);
    const out = {
      ok: true, role: h.role, mpsActive: !!h.mps_pipe, capacity: h.capacity, smTotal: SM_TOTAL,
      tenants: (h.tenants || []).map((t) => ({ pct: t.pct, status: t.status, smGranted: t.sm_granted })),
    };
    if (PROVISION_BACKEND === "vm") {
      const vc = await vmHealth(5000).then((v) => v.capacity || {}).catch(() => null);
      if (vc) out.vm = {
        apps: vc.apps, usedGpuShare: round3(1 - (vc.gpuShareFree ?? 1)),
        vramBudgetGb: vc.vramBudgetGb, vramCommittedGb: vc.vramCommittedGb, vramLedgerFreeGb: vc.vramFreeGb,
        ...(vc.vramDevFreeGb != null ? { vramDevFreeGb: vc.vramDevFreeGb, vramDevTotalGb: vc.vramDevTotalGb,
                                         vramDivergenceGb: vc.vramDivergenceGb } : {}),
      };
    }
    res.json(out);
  } catch (e) {
    res.status(503).json({ ok: false, error: `worker manager unreachable: ${e.message}` });
  }
});

// ============================================================================
// auth (SIWE)
// ============================================================================
app.get("/v1/auth/nonce", (req, res) => {
  let address; try { address = getAddress(String(req.query.address || "")); }
  catch { return fail(res, 422, "invalid_address", "Provide a valid ?address."); }
  const nonce = rid("");
  const issuedAt = new Date(), expirationTime = new Date(issuedAt.getTime() + 10 * 60_000);
  nonces.set(nonce, { address, exp: expirationTime.getTime() });
  // Bound the map: the TTL sweep runs every 60s, but a burst of nonce requests
  // between sweeps could grow it without limit. Map keeps insertion order, so the
  // oldest entries evict first (FIFO ≈ LRU; they'd expire soonest anyway).
  while (nonces.size > NONCE_MAX) { const k = nonces.keys().next().value; if (k === undefined) break; nonces.delete(k); }
  const statement = "Sign in to Enclave. This signature is free and will not move funds.";
  const message =
    `${SIWE_DOMAIN} wants you to sign in with your Ethereum account:\n${address}\n\n${statement}\n\n` +
    `URI: ${SIWE_URI}\nVersion: 1\nChain ID: ${CHAIN_ID}\nNonce: ${nonce}\n` +
    `Issued At: ${issuedAt.toISOString()}\nExpiration Time: ${expirationTime.toISOString()}`;
  res.json({ address, message, nonce, statement, domain: SIWE_DOMAIN, uri: SIWE_URI, version: "1",
             chainId: CHAIN_ID, issuedAt: issuedAt.toISOString(), expirationTime: expirationTime.toISOString() });
});

app.post("/v1/auth/login", async (req, res) => {
  const { message, signature } = req.body || {};
  if (!message || !signature) return fail(res, 422, "invalid_request", "message and signature are required.");
  const nm = message.match(/\nNonce: (\S+)\n/), am = message.match(/^(0x[0-9a-fA-F]{40})$/m);
  if (!nm || !am) return fail(res, 422, "invalid_message", "Malformed SIWE message.");
  // Bind the signed message to THIS enclave's SIWE parameters. /v1/auth/nonce
  // issues the exact message (domain/uri/chainId/expiration all ours) and the
  // client signs it verbatim (site/js/core/wallet.js: it uses the server's
  // `message` as-is; its buildSiwe fallback also resolves to these same values on
  // enclave.host). So a signature over a message that names a DIFFERENT domain/uri/
  // chain — or is already past its Expiration Time — is not a login here. We assert
  // ONLY fields the message actually carries (absent field => not asserted), so a
  // legitimate login is never locked out on a format we didn't emit.
  const dmatch = message.match(/^(.+?) wants you to sign in with your Ethereum account:/);
  const umatch = message.match(/^URI: (\S+)$/m);
  const cmatch = message.match(/^Chain ID: (\d+)$/m);
  const ematch = message.match(/^Expiration Time: (\S+)$/m);
  if (dmatch && dmatch[1] !== SIWE_DOMAIN) return fail(res, 401, "bad_domain", "SIWE message domain does not match this enclave.");
  if (umatch && umatch[1] !== SIWE_URI)    return fail(res, 401, "bad_uri", "SIWE message URI does not match this enclave.");
  if (cmatch && Number(cmatch[1]) !== CHAIN_ID) return fail(res, 401, "bad_chain", "SIWE message chain does not match this enclave.");
  if (ematch) { const t = Date.parse(ematch[1]); if (Number.isFinite(t) && t <= Date.now()) return fail(res, 401, "expired", "SIWE message has expired."); }
  const nonce = nm[1], claimed = getAddress(am[1]), rec = nonces.get(nonce);
  if (!rec || rec.exp < Date.now()) { nonces.delete(nonce); return fail(res, 401, "bad_nonce", "Unknown or expired nonce."); }
  if (getAddress(rec.address) !== claimed) return fail(res, 401, "address_mismatch", "Address does not match nonce.");
  let ok = false; try { ok = await verifyMessage({ address: claimed, message, signature }); } catch {}
  if (!ok) return fail(res, 401, "bad_signature", "Signature verification failed.");
  nonces.delete(nonce);
  const expiresAt = new Date(Date.now() + SESSION_TTL * 1000);
  const token = await mintSession(claimed, expiresAt);
  res.json({ token, tokenType: "Bearer", address: claimed, expiresAt: expiresAt.toISOString() });
});

// ============================================================================
// payments (pay-per-deploy) - the supervisor WATCHES the EnclavePay forwarder on
// Base for Paid events and converts each payment into runtime. No held balance.
// (outbound Base RPC required - confirm the CVM egress allows BASE_RPC.)
// ============================================================================
const PAY_EVENT = { type: "event", name: "Paid", inputs: [
  { name: "deploymentId", type: "bytes32", indexed: true },
  { name: "payer",        type: "address", indexed: true },
  { name: "amount",       type: "uint256", indexed: false } ] };
const PAY_ETH_EVENT = { type: "event", name: "PaidEth", inputs: [
  { name: "deploymentId", type: "bytes32", indexed: true },
  { name: "payer",        type: "address", indexed: true },
  { name: "amountWei",    type: "uint256", indexed: false } ] };

const payRefIndex = new Map();   // payRef (hex, lowercase) -> deployment id

// USDC (6dp) funded at `rate` USDC/sec buys this many seconds of runtime.
const usdcToSeconds = (amountRaw, rate) => (Number(amountRaw) / 1e6) / (rate || 1);

// --- ETH payments: priced via the Chainlink ETH/USD feed on Base -------------
// Feed address verified on-chain (description() == "ETH / USD", decimals() == 8).
// On another chain (e.g. Base Sepolia) set ETH_USD_FEED to that chain's feed.
const ETH_USD_FEED = process.env.ETH_USD_FEED || "0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70";
const ETH_FEED_MAX_AGE_SEC = parseInt(process.env.ETH_FEED_MAX_AGE_SEC || "21600", 10); // 6h (feed heartbeat ~20min)
const FEED_ABI = [{ type: "function", name: "latestRoundData", stateMutability: "view", inputs: [],
  outputs: [{ type: "uint80" }, { type: "int256" }, { type: "uint256" }, { type: "uint256" }, { type: "uint80" }] }];

// wei (1e18) * price (8dp) -> USDC-equivalent (6dp):  / 1e20
const weiToUsd6 = (wei, price8) => (wei * price8) / 10n ** 20n;

let _ethUsd = { price8: null, at: 0 };            // cached oracle read (for instructions + conversion)
async function ethUsdPrice8() {
  if (_ethUsd.price8 && (Date.now() - _ethUsd.at) < 60_000) return _ethUsd.price8;
  const [, answer, , updatedAt] = await chainClient.readContract({
    address: getAddress(ETH_USD_FEED), abi: FEED_ABI, functionName: "latestRoundData" });
  const ageSec = Math.floor(Date.now() / 1000) - Number(updatedAt);
  if (answer <= 0n) throw new Error(`ETH/USD feed returned ${answer}`);
  if (ageSec > ETH_FEED_MAX_AGE_SEC) throw new Error(`ETH/USD feed stale (${ageSec}s old)`);
  _ethUsd = { price8: answer, at: Date.now() };
  return answer;
}

// ETH payments retry if the oracle read fails: a payment must never be lost to a
// flaky RPC. Queue drains at the top of every poll tick.
const _pendingEth = [];
async function onPaidEth(payRefHex, payer, wei) {
  try {
    const price8 = await ethUsdPrice8();
    const usd6 = weiToUsd6(wei, price8);
    console.log(`[pay] eth ${wei} wei @ $${(Number(price8) / 1e8).toFixed(2)} -> ${(Number(usd6) / 1e6).toFixed(2)} USDC-equiv (${payRefHex})`);
    await onPaid(payRefHex, payer, usd6);
  } catch (e) {
    _pendingEth.push({ payRefHex, payer, wei });
    console.warn(`[pay] eth payment queued for retry (${e.shortMessage || e.message})`);
  }
}

// --- USDC EIP-712 domain: payers sign an EIP-3009 ReceiveWithAuthorization ---
// against the TOKEN's own domain, so instructions must carry its exact fields.
// name()/version() differ per deployment (mainnet USDC: "USD Coin"; Base Sepolia
// testnet USDC: "USDC"), so read them from the token once and cache forever.
const ERC20_META_ABI = [
  { type: "function", name: "name",    stateMutability: "view", inputs: [], outputs: [{ type: "string" }] },
  { type: "function", name: "version", stateMutability: "view", inputs: [], outputs: [{ type: "string" }] },
];
let _usdcDomain = null;
async function refreshUsdcDomain() {
  if (_usdcDomain) return _usdcDomain;
  const addr = getAddress(USDC_ADDRESS);
  const [name, version] = await Promise.all([
    chainClient.readContract({ address: addr, abi: ERC20_META_ABI, functionName: "name" }),
    chainClient.readContract({ address: addr, abi: ERC20_META_ABI, functionName: "version" }).catch(() => "2"), // FiatTokenV2+ is "2"
  ]);
  _usdcDomain = { name, version, chainId: CHAIN_ID, verifyingContract: addr };
  console.log(`[pay] USDC EIP-712 domain: name="${name}" version="${version}"`);
  return _usdcDomain;
}

function paymentInstructions(rec) {
  return {
    chainId: CHAIN_ID, asset: "USDC", assets: ["USDC", "ETH"], usdc: USDC_ADDRESS,
    forwarder: FORWARDER_ADDRESS || null,
    deploymentRef: rec.payRef,                       // bytes32 to pass to payWithAuthorization() / payEth()
    ratePerSecondUsdc: (rec.rate || 0).toFixed(7),
    method: "payWithAuthorization(bytes32 deploymentId, address from, uint256 value, uint256 validAfter, uint256 validBefore, bytes32 nonce, bytes signature)",
    payEthMethod: "payEth(bytes32 deploymentId) payable",
    usdcDomain: _usdcDomain,                         // EIP-712 domain to sign ReceiveWithAuthorization against; null until the first token read
    ethUsd: _ethUsd.price8 ? (Number(_ethUsd.price8) / 1e8).toFixed(2) : null,   // cached Chainlink read; null until first refresh
    note: "USDC (EIP-3009, no approve): sign a USDC ReceiveWithAuthorization (EIP-712, to = forwarder, "
        + "nonce = first 16 bytes of deploymentRef + 16 random bytes), then anyone submits payWithAuthorization; "
        + "amount(6dp)/rate = seconds. "
        + "ETH: payEth(deploymentRef) with msg.value; credited as USDC-equivalent at the live Chainlink ETH/USD rate.",
  };
}

app.get("/v1/account", authed, (req, res) => {
  const mine = [...deployments.values()].filter(d => d.owner === req.address);
  res.json({
    address: req.address, chainId: CHAIN_ID,
    payment: { forwarder: FORWARDER_ADDRESS || null, usdc: USDC_ADDRESS, asset: "USDC", assets: ["USDC", "ETH"] },
    deployments: {
      running: mine.filter(d => d.status === "running").length,
      awaitingPayment: mine.filter(d => d.status === "awaiting_payment").length,
      total: mine.length,
      totalTimeRemainingSec: mine.reduce((s, d) => s + (timeRemainingSec(d) || 0), 0),
    },
  });
});

// ============================================================================
// deployments
// ============================================================================
// remainingMs === null means unlimited (auto-provision pilot); otherwise it only
// drains on healthy billing ticks, so it IS the truth even mid-outage.
// On-chain deployments: remainingMs mirrors only the CURRENT lease (minutes),
// while the rest of the funded runtime sits in the ledger balance - report
// lease tail + balance/rate or the console shows "12m left" on a 2-day fund.
const timeRemainingSec = (rec) => {
  if (rec.remainingMs == null) return null;
  const lease = Math.max(0, Math.round(rec.remainingMs / 1000));
  if (!rec._onchain) return lease;
  // Rate 0 on a chain record means the rev-12 waiver: this box hosts its own
  // payout wallet's app for nothing. There is no funded time to run down and
  // its lease renews at zero cost, so a countdown would be a fiction that
  // reads "28m left" forever. null is the API's existing "unlimited".
  // Gated on this box having a declaration at all, so that on a fleet where the
  // waiver cannot exist a rate we simply failed to read keeps reporting the
  // lease tail exactly as it always did, rather than claiming "unlimited".
  if (!(rec.rate > 0)) return _declaredPayout ? null : lease;
  return lease + Math.max(0, Math.round((rec._balance6 || 0) / (rec.rate * 1e6)));
};
const spentOf = (rec) => (((rec.consumedMs || 0) / 1000) * (rec.rate || 0)).toFixed(2);
// EXPLICIT allowlist of record fields exposed to the owner (was a delete-denylist).
// Allowlist-shaped so a NEW internal field added to a record never leaks by
// default — it has to be added here on purpose. This is the exact set the
// denylist previously let through (creation + claim + provision + failure paths);
// the computed fields below (rate/spent/paid/time/expires, payment, onchain,
// network) are layered on top just as before.
const VIEW_FIELDS = ["id", "owner", "status", "public", "firewall", "image", "command",
  "app", "appWasm", "config", "resources", "network", "attestation", "region",
  "createdAt", "startedAt", "paused", "pauseReason", "payDeadline", "digest",
  "payRef", "paidUsdc", "portMap", "error", "waf", "configOverride", "versionChange",
  "configChange",  // an envelope edit (setConfig) this runner deferred/refused - the owner's evidence, versionChange's twin
  "appConfigCid",  // where the config above ACTUALLY lives when `config` is only the routing manifest: a version's own configCid (catalog rev 7), or — when configOverride is set beside it — the DEPLOYER's, from the options envelope's configCid namespace. Public on purpose either way: otherwise an owner reading the record would see the manifest and think it was their app's config
  "secretsRev"];   // which relay secrets snapshot the running instance was launched with (names/values never leave the guest)
const view = (rec) => {
  const o = {};
  for (const k of VIEW_FIELDS) if (k in rec) o[k] = rec[k];
  // vmTechnology reflects what this enclave's OWN attestation document says
  // today, not what the record stored at create time (records persisted by
  // older builds carry a hardcoded guess; detection self-heals them).
  if (o.attestation) o.attestation = { ...o.attestation, vmTechnology: vmTech() ?? o.attestation.vmTechnology ?? null };
  o.ratePerSecondUsdc = (rec.rate || 0).toFixed(7);
  o.spentUsdc = spentOf(rec);
  o.paidUsdc = ((rec.paidUsdc || 0) / 1e6).toFixed(2);
  o.timeRemainingSec = timeRemainingSec(rec);
  // an ESTIMATE only: the balance drains solely while service is healthy, so a
  // frozen (paused) deployment has no meaningful wall-clock expiry.
  o.expiresAt = (rec.remainingMs != null && rec.status === "running" && !rec.paused)
    ? new Date(Date.now() + Math.max(0, timeRemainingSec(rec)) * 1000).toISOString() : null;
  o.payment = rec._onchain ? onchainPaymentInstructions(rec) : paymentInstructions(rec);
  // claimed-from-chain deployments surface their ledger identity + current lease
  if (rec._onchain) o.onchain = { contract: DEPLOYMENTS_ADDRESS, id: rec.id,
    leaseUntil: rec._leaseUntil ? new Date(rec._leaseUntil * 1000).toISOString() : null };
  // Proof of time, from the runner's side: how far THIS enclave has proven it
  // served, and when. A tenant reads it to see that the box hosting their app is
  // being held to account for the hours it bills — and can check it against the
  // chain (EnclaveDeployments.provenUntil) rather than taking our word for it.
  if (rec._onchain && rec._provenUntil) o.onchain.proofOfTime = {
    prover: PROOF_OF_TIME_ADDRESS || null,
    provenUntil: new Date(rec._provenUntil * 1000).toISOString(),
    lastProofAt: rec._provenAt ? new Date(rec._provenAt).toISOString() : null,
    verify: "EnclaveDeployments.provenUntil(id) is authoritative; the runner meter never pays past it." };
  // Dedicated per-deployment IPv6: declared tcp/udp ports are reachable at
  // [address]:<logical port> (tcp via the tcp6-relay, udp via the udp-relay).
  // Surface it so the dashboard/clients get a ready-to-use endpoint at the
  // real port the app declared, e.g. [addr]:5432, [addr]:443, [addr]:53.
  // With dedicated-IP egress on, the SAME address is also every deployment's
  // outbound identity - so it's surfaced even with no inbound ports declared
  // (network.egress marks the outbound half so clients can label it).
  const tcpPorts = fwTcpPorts(rec), udpPorts = fwUdpPorts(rec);
  const depAddr = depAddrFor(rec.id);
  if (depAddr && (tcpPorts.length || udpPorts.length || egress)) {
    o.network = { ...o.network, address: depAddr };
    if (egress) o.network.egress = true;
    if (tcpPorts.length) o.network.tcp = { address: depAddr, ports: tcpPorts };
    if (udpPorts.length) o.network.udp = { address: depAddr, ports: udpPorts };
  }
  return o;
};

// Arm (or re-arm after a restart) the unpaid-reservation timer from payDeadline.
// If the payment watcher is blind when the deadline hits, the reservation is
// frozen too: a payment may already sit in the unscanned window, so expiry
// defers until the watcher has caught back up to the chain tip.
function armPayTimer(rec) {
  if (rec._payTimer) clearTimeout(rec._payTimer);
  rec._payTimer = setTimeout(() => {
    if (rec.status !== "awaiting_payment") return;
    if (FORWARDER_ADDRESS && (Date.now() - _lastPollOkAt) > WATCHER_STALE_SEC * 1000) {
      rec.payDeadline = Date.now() + 60_000;
      armPayTimer(rec); saveStateSoon();
      return;
    }
    if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
    rec.status = "expired"; rec.error = "unpaid";
    console.log(`[pay] ${rec.id} reservation released (unpaid after payment window)`);
    saveStateSoon();
  }, Math.max(0, rec.payDeadline - Date.now()));
  if (rec._payTimer.unref) rec._payTimer.unref();
}

// ---- app approval (vm backend): catalog-gated deploys -----------------------
// A deployment references the CATALOG RECORD of the version it runs:
//   catalog://<appId>/<versionIndex>
// (Steven, 2026-07-09.) The wasm CID is a content address, NOT an app identity:
// two versions may share bytes and differ entirely in approved config, and the
// config alone changes behavior. So the record — never the deployer — is the
// authority for everything the owner's approval covered: the wasm CID (now
// just a fetch address), the config (ENCLAVE_CONFIG + volume mounts), the
// ports, and the resource minimums. Version rows are append-only and
// immutable, so the reference resolves to the same artifact forever; only the
// deployability flags (approval / yanked / app active) are live, and they are
// exactly what gets re-checked here on every claim, respawn and resume.
const CATALOG_ABI = [
  { type: "function", name: "getApp", stateMutability: "view",
    inputs: [{ name: "appId", type: "bytes32" }],
    outputs: [{ type: "tuple", components: [
      { name: "appId", type: "bytes32" }, { name: "publisher", type: "address" },
      { name: "slug", type: "string" }, { name: "name", type: "string" },
      { name: "description", type: "string" }, { name: "versionCount", type: "uint32" },
      { name: "createdAt", type: "uint64" }, { name: "updatedAt", type: "uint64" },
      { name: "active", type: "bool" },
    ] }] },
  { type: "function", name: "getVersion", stateMutability: "view",
    inputs: [{ name: "appId", type: "bytes32" }, { name: "index", type: "uint256" }],
    outputs: [{ type: "tuple", components: [
      { name: "cid", type: "string" }, { name: "version", type: "string" },
      { name: "vramMb", type: "uint32" }, { name: "gpuGflops", type: "uint32" },
      { name: "memMb", type: "uint32" }, { name: "cpuGflops", type: "uint32" },
      { name: "createdAt", type: "uint64" }, { name: "verified", type: "bool" },
      { name: "yanked", type: "bool" }, { name: "ports", type: "string" },
      { name: "approval", type: "uint8" },  // 0 pending | 1 approved | 2 rejected
      { name: "config", type: "string" },
    ] }] },
  // rev-5 surface (side mapping, so the tuples above decode on every rev):
  // only CALLED when the catalog sniffs >= 5
  { type: "function", name: "versionFee", stateMutability: "view",
    inputs: [{ name: "appId", type: "bytes32" }, { name: "index", type: "uint256" }],
    outputs: [{ type: "uint256" }] },
  // rev-7 surface (side mapping too): the CID holding a large config. Empty
  // means the inline `config` IS the app's config, exactly as on every earlier
  // rev; set means the inline field is the routing manifest and the FETCHED
  // content is what the guest gets. Only CALLED when the catalog sniffs >= 7.
  { type: "function", name: "versionConfigCid", stateMutability: "view",
    inputs: [{ name: "appId", type: "bytes32" }, { name: "index", type: "uint256" }],
    outputs: [{ type: "string" }] },
  { type: "function", name: "catalogSchema", stateMutability: "view", inputs: [], outputs: [{ type: "uint256" }] },
];
// Which feature surface the catalog at APP_CATALOG_ADDRESS speaks. Same live
// re-sniff-per-address + poisoning care as the deployments-ledger sniff below
// (depsAbi): a transient RPC failure must never cache an old rev — only a
// definitive revert proves a pre-catalogSchema contract. null = unknown this
// round; callers that need the answer fail closed and retry next pass.
let _catRev = { addr: null, rev: null };
async function catSchemaRev() {
  if (!APP_CATALOG_ADDRESS) return null;
  if (_catRev.addr === APP_CATALOG_ADDRESS && _catRev.rev != null) return _catRev.rev;
  try {
    const rev = Number(await chainClient.readContract({ address: getAddress(APP_CATALOG_ADDRESS),
      abi: CATALOG_ABI, functionName: "catalogSchema" }));
    _catRev = { addr: APP_CATALOG_ADDRESS, rev };
    console.log(`[approval] catalog ${APP_CATALOG_ADDRESS} schema rev ${rev}`);
  } catch (e) {
    if (sniffCachePolicy(e.shortMessage || e.message || "") === "cache-rev1") {
      _catRev = { addr: APP_CATALOG_ADDRESS, rev: 2 };   // pre-catalogSchema catalog
      console.log(`[approval] catalog ${APP_CATALOG_ADDRESS} schema rev 2 (pre-catalogSchema contract)`);
    } else {
      return null;
    }
  }
  return _catRev.rev;
}
const CATALOG_REF_RE = /^catalog:\/\/(0x[0-9a-fA-F]{64})\/(\d{1,9})$/;
const ZERO32 = "0x" + "0".repeat(64);
// Gate a vm-backend app reference on catalog approval. Returns
// { ref, wasmRef, config, ports, app: {appId,index,slug,version,publisher}, min }
// or { error }. An RPC failure REJECTS the deploy (fail closed): this is the
// enforcement point, so an outage must not waive it. The image ships NO
// deployable apps (nn-demo.wasm inside it is solely the boot probe's fixture,
// launched by the manager itself, never through this API), so approved catalog
// records are the only deploy surface — anything else is refused here, which
// also keeps bare paths under the manager's APPS_DIR (e.g. a cached
// ipfs-<cid>.wasm) from dodging the approval check.
const NO_MIN = { vramMb: 0, gpuGflops: 0, memMb: 0, cpuGflops: 0 };
async function gateAppReference(reference, opts = {}) {
  const deny = (status, code, msg) => ({ error: { status, code, msg } });
  const ref = String(reference || "").trim();
  const m = CATALOG_REF_RE.exec(ref);
  if (!m) {
    // ipfs:// and bare-CID references are RETIRED: a CID can belong to several
    // versions with different approved configs, so it cannot name what to run.
    return deny(422, "invalid_spec", "image.reference must be catalog://<appId>/<versionIndex> — the on-chain record of a catalog version. CID references are retired (a CID names bytes, not a version); redeploy from the console or CLI.");
  }
  if (!APP_CATALOG_ADDRESS)
    return deny(503, "approval_unavailable", "Catalog apps are disabled on this enclave: APP_CATALOG_ADDRESS is not configured, so approval cannot be verified.");
  const [appId, index] = [m[1], Number(m[2])];
  const [ar, vr] = await Promise.allSettled([
    chainClient.readContract({ address: getAddress(APP_CATALOG_ADDRESS),
      abi: CATALOG_ABI, functionName: "getApp", args: [appId] }),
    chainClient.readContract({ address: getAddress(APP_CATALOG_ADDRESS),
      abi: CATALOG_ABI, functionName: "getVersion", args: [appId, BigInt(index)] }),
  ]);
  if (ar.status === "rejected") {
    console.warn(`[approval] getApp(${appId}) failed: ${ar.reason?.shortMessage || ar.reason?.message}`);
    return deny(503, "catalog_unreachable", "Could not verify this app's approval against the on-chain catalog; try again shortly.");
  }
  const a = ar.value;
  if (!a || a.appId === ZERO32) return deny(403, "not_approved", "This appId is not in the app catalog.");
  if (index >= Number(a.versionCount)) return deny(403, "not_approved", `App '${a.slug}' has no version index ${index} (it has ${a.versionCount}).`);
  if (vr.status === "rejected") {   // index exists, so this is RPC trouble, not a bad ref
    console.warn(`[approval] getVersion(${appId}, ${index}) failed: ${vr.reason?.shortMessage || vr.reason?.message}`);
    return deny(503, "catalog_unreachable", "Could not verify this app's approval against the on-chain catalog; try again shortly.");
  }
  const v = vr.value;
  const verdict = approvalVerdict({ active: a.active, yanked: v.yanked, approval: v.approval,
                                    slug: a.slug, version: v.version }, opts.forPrivate);
  if (verdict) return deny(403, "not_approved", verdict);
  // The version's publisher fee is part of what approval covered (like config
  // and ports). Resolved here so every consumer of the gate carries it; the
  // fee-vs-ledger comparison lives in feeGate (claim/resume/upgrade paths).
  // Same fail-closed posture as the rest of the gate: not knowing the fee is
  // not the same as the fee being zero.
  let feePerSec6 = 0n;
  const catRev = await catSchemaRev();
  if (catRev == null)
    return deny(503, "catalog_unreachable", "Could not determine the catalog's schema revision; try again shortly.");
  if (catRev >= 5) {
    try {
      feePerSec6 = await chainClient.readContract({ address: getAddress(APP_CATALOG_ADDRESS),
        abi: CATALOG_ABI, functionName: "versionFee", args: [appId, BigInt(index)] });
    } catch (e) {
      console.warn(`[approval] versionFee(${appId}, ${index}) failed: ${e.shortMessage || e.message}`);
      return deny(503, "catalog_unreachable", "Could not verify this app's publisher fee against the on-chain catalog; try again shortly.");
    }
  }
  // rev-7 large configs: the version may keep its ENCLAVE_CONFIG at a CID
  // instead of inline. Read it here so every consumer of the gate carries it,
  // same fail-closed posture as the fee — not knowing whether a config CID
  // exists is NOT the same as there being none, because launching without it
  // would serve the app with its routing manifest as its config.
  let configCid = "";
  if (catRev >= 7) {
    try {
      configCid = await chainClient.readContract({ address: getAddress(APP_CATALOG_ADDRESS),
        abi: CATALOG_ABI, functionName: "versionConfigCid", args: [appId, BigInt(index)] }) || "";
    } catch (e) {
      console.warn(`[approval] versionConfigCid(${appId}, ${index}) failed: ${e.shortMessage || e.message}`);
      return deny(503, "catalog_unreachable", "Could not read this app's config reference from the on-chain catalog; try again shortly.");
    }
  }
  return { ref, wasmRef: "ipfs://" + v.cid, config: v.config || "", configCid, ports: v.ports || "", feePerSec6,
           pending: Number(v.approval) !== 1,   // true only on the forPrivate dev-mode path
           app: { appId, index, slug: a.slug, version: v.version, publisher: a.publisher },
           min: { vramMb: Number(v.vramMb) || 0, gpuGflops: Number(v.gpuGflops) || 0,
                  memMb: Number(v.memMb) || 0, cpuGflops: Number(v.cpuGflops) || 0,
                  // The publisher's own word on whether the card is required,
                  // read from the VERSION config - the same place `volumes`
                  // lives, and immutable per version like everything else in
                  // it. On-chain the axes are just numbers; only the app knows
                  // whether it degrades to CPU or cannot start without a card.
                  gpuOptional: gpuOptionalOfConfig(v.config) } };
}

// A paid app is servable only if the DEPLOYMENT snapshotted the version's
// publisher fee (and the right payee) at create — the funding splits pay the
// publisher out of that snapshot, so serving an under-declared record would
// mean the publisher never sees a cent of it. Fail closed, exactly like the
// approval gate: the ledger not answering is not a waiver. Free apps
// (feePerSec6 == 0) never reach the chain here. A deployment that overpays
// (snapshot above the version's ask, e.g. repointed at a cheaper release)
// stays servable — the publisher just keeps the larger cut it signed up for.
// Returns null (servable) or { why, transient }.
const FEE_OF_ABI = [{ type: "function", name: "feeOf", stateMutability: "view",
  inputs: [{ name: "id", type: "bytes32" }],
  outputs: [{ name: "recipient", type: "address" }, { name: "feePerSec6", type: "uint256" }] }];
// What THIS enclave charges, per second, for the shares a deployment bought:
// our published per-machine price scaled by gpuMilli/cpuMilli, ceil'd exactly
// as EnclaveDeployments._hostRate does. The publisher fee is the deployment's,
// not ours, and rides on top.
const hostRate6 = (gpuMilli, cpuMilli) =>
  Math.ceil((SELL_GPU_PRICE6 * (Number(gpuMilli) || 0) + SELL_CPU_PRICE6 * (Number(cpuMilli) || 0)) / 1000);
const usdPerHour = (perSec6) => "$" + (perSec6 * 3600 / 1e6).toFixed(2) + "/hr";

// FREE SELF-HOSTING (ledger rev 12 + registry schema 4): the ledger waives this
// box's entire charge when our on-chain DECLARED payout wallet owns the
// deployment — the seller's own app on the seller's own box. The claim loop has
// to price exactly the way claim() will, in both directions: price it high and
// we refuse funded-looking work the chain would give us for nothing; price it
// low on an older ledger and we take work we are then charged for. So this is
// gated on BOTH schema revs and on the declaration the chain actually carries
// (never on the configured PAYOUT_ADDRESS — that is this box's private opinion,
// and the ledger has never heard of it).
async function hostChargeWaived(d) {
  if (!_declaredPayout || !d || !d.owner) return false;
  try {
    if (await registryRev() < 4) return false;
    if ((await depsAbi()).rev < 12) return false;
  } catch { return false; }                             // unreadable schema: assume we charge
  try { return getAddress(d.owner) === _declaredPayout; } catch { return false; }
}
const CAP_OF_ABI = [{ type: "function", name: "capOf", stateMutability: "view",
  inputs: [{ name: "id", type: "bytes32" }], outputs: [{ name: "maxRate6", type: "uint256" }] }];

// The buyer's ceiling (ledger rev 8): a deployment states the most it will pay
// per second, and an enclave whose price for those shares exceeds it may not
// claim — the chain enforces the same check, so taking the work anyway would
// just burn a reverted tx. This is what bounds automatic failover: when a host
// dies, its tenants land on whichever enclave both FITS them and is cheap
// enough, and stay queued rather than get silently re-priced upward.
//
// Fail closed on an unreadable cap: a claim we can't price is not ours.
// Returns null (acceptable) or a refusal string for the sweep log.
async function rateCapRefusal(d, g) {
  const { rev } = await depsAbi();
  if (rev < 8) return priceFloorRefusal(d);             // pre-rev-8 ledger: the old ask-as-floor rule
  let cap6;
  try {
    cap6 = Number(await chainClient.readContract({ address: getAddress(DEPLOYMENTS_ADDRESS),
      abi: CAP_OF_ABI, functionName: "capOf", args: [d.id] }));
  } catch {
    return "could not read the deployment's rate cap from the ledger; try again shortly";
  }
  const mine6 = (await hostChargeWaived(d)) ? 0 : hostRate6(d.gpuMilli, d.cpuMilli);
  return capVerdict({ mine6, fee6: Number((g && g.feePerSec6) || 0),
                      cap6, balance6: Number(d.balance6) });
}

// The decision itself, pure so it can be checked without a chain: what WE
// charge for these shares plus the deployment's own publisher fee, against the
// owner's ceiling (0 = uncapped — only reachable for grandfathered imports)
// and against what its balance can actually buy at our price. The contract
// enforces the same two rules, so a refusal here just saves a reverted tx.
// Returns null (acceptable) or a refusal string.
function capVerdict({ mine6, fee6 = 0, cap6 = 0, balance6 = 0 }) {
  const total6 = mine6 + fee6;                          // exactly what claim() would snapshot
  if (cap6 > 0 && total6 > cap6)
    return `this enclave charges ${usdPerHour(total6)} for those shares`
         + (fee6 ? ` (incl. ${usdPerHour(fee6)} publisher fee)` : "")
         + `, above the owner's rate cap of ${usdPerHour(cap6)}`;
  if (balance6 < total6)
    return `out of funded time at this enclave's price (${usdPerHour(total6)}`
         + (mine6 === 0 && fee6 > 0 ? ", the publisher's fee alone - hosting here is free" : "")
         + `) - fund it and retry`;
  return null;
}

// PRICE_SELFTEST='[{"gpuMilli":…,"cpuMilli":…,"fee6":…,"cap6":…,"balance6":…},…]'
// prints, per case, what THIS enclave charges for those shares and whether it
// would claim the work — the buyer's rate cap decided without a chain. Same
// seam contract as SWEEP_SELFTEST (test/rate-cap.test.mjs drives it, with the
// enclave's own price set through SELL_CPU_PRICE6 / SELL_GPU_PRICE6).
if (process.env.PRICE_SELFTEST) {
  const out = JSON.parse(process.env.PRICE_SELFTEST).map((c) => {
    // `free: true` = the rev-12 waiver (our declared payout wallet owns it)
    const mine6 = c.free ? 0 : hostRate6(c.gpuMilli || 0, c.cpuMilli || 0);
    return { mine6, refusal: capVerdict({ mine6, fee6: c.fee6 || 0, cap6: c.cap6 || 0, balance6: c.balance6 || 0 }) };
  });
  console.log(JSON.stringify(out));
  process.exit(0);
}

// PRE-REV-8 seller price floor: refuse work that pays less than THIS operator
// asks for the shares it would occupy. On those ledgers the price is global and
// the deployment's snapshotted `rate` covers the shares PLUS the version's
// publisher fee, and the fee is not the runner's revenue — so the comparison is
// against rate minus that fee (read from the ledger like feeGate does; a fee
// that can't be read is treated as 0, which only ever makes this gate STRICTER,
// never laxer, so an RPC blip can't trick the box into taking underpriced work).
// Returns null (acceptable) or a refusal string for the sweep log.
async function priceFloorRefusal(d) {
  if (!PRICE_IS_EXPLICIT) return null;                  // hosted fleet on an old ledger: list price, no floor
  const gpuMilli = Number(d.gpuMilli) || 0, cpuMilli = Number(d.cpuMilli) || 0;
  const ask6 = hostRate6(gpuMilli, cpuMilli);
  if (ask6 <= 0) return null;                           // nothing priced for these shares
  let fee6 = 0;
  try {
    const { rev } = await depsAbi();
    if (rev >= 4) {
      const [, f] = await chainClient.readContract({ address: getAddress(DEPLOYMENTS_ADDRESS),
        abi: FEE_OF_ABI, functionName: "feeOf", args: [d.id] });
      fee6 = Number(f) || 0;
    }
  } catch { /* unreadable fee -> 0 -> stricter comparison */ }
  const share6 = Math.max(0, Number(d.rate) - fee6);    // what the shares themselves pay, per second
  if (share6 >= ask6) return null;
  return `pays ${usdPerHour(share6)} for these shares, below this operator's ask of ${usdPerHour(ask6)}`;
}

async function feeGate(id, g) {
  if (!(g.feePerSec6 > 0n)) return null;
  const { rev } = await depsAbi();
  if (rev < 4)
    return { why: "this app charges a publisher fee, which this ledger predates (deploymentsSchema < 4)", transient: false };
  let recipient, fee6;
  try {
    [recipient, fee6] = await chainClient.readContract({ address: getAddress(DEPLOYMENTS_ADDRESS),
      abi: FEE_OF_ABI, functionName: "feeOf", args: [id] });
  } catch (e) {
    return { why: "could not verify the deployment's publisher-fee snapshot against the ledger; try again shortly", transient: true };
  }
  if (fee6 < g.feePerSec6 || String(recipient).toLowerCase() !== String(g.app.publisher).toLowerCase())
    return { why: `the deployment under-declares the app's publisher fee (snapshot ${fee6}/sec to ${recipient}; `
                + `${g.app.slug}:${g.app.version} asks ${g.feePerSec6}/sec to its publisher ${g.app.publisher}); redeploy from the console or CLI`,
             transient: false };
  return null;
}

app.post("/v1/deployments", authed, async (req, res) => {
  const b = req.body || {};
  // RETIRED on the wasm backend (Steven, 2026-07-05): this path held the spec
  // and the funded clock in enclave-local state, which died with the CVM on
  // every update. Deployments are created ON-CHAIN instead (EnclaveDeployments):
  // create() from the owner's wallet, fund with fundWithAuthorization (EIP-3009
  // USDC) or fundEth, and any enclave claims, serves, renews - and re-claims
  // after this one is updated or dies. The ledger IS the deployment; an
  // enclave is just its current runner.
  if (PROVISION_BACKEND === "vm") {
    return res.status(410).json({
      code: "deploy_on_chain",
      message: "Deployments are created on-chain, not through this endpoint: send create() to the "
             + "EnclaveDeployments contract from your wallet (you own the record), fund it via "
             + "fundWithAuthorization (EIP-3009 USDC, nonce prefixed with the id's first 16 bytes) or "
             + "fundEth, then POST /v1/claim-hint {id} to start it immediately. The deploy console at "
             + "the site does all of this for you. On-chain deployments survive enclave updates: the "
             + "ledger holds the spec and balance, and runners hold expiring leases.",
      onchain: {
        contract: DEPLOYMENTS_ADDRESS || null, chainId: CHAIN_ID, usdc: USDC_ADDRESS,
        createMethod: "create(string appRef, uint16 gpuMilli, uint16 cpuMilli, uint32 appPort, string ports, bool isPublic, "
                    + ((await depsAbi()).rev >= 2 ? "" : "string sshPubKey, ")
                    + "string configCid) returns (bytes32 id) — appRef is catalog://<appId>/<versionIndex> (runners refuse CID refs: a CID names bytes, not a version); configCid carries either \"\" or the deployment-options envelope {\"waf\":{…},\"config\":{…}} (waf = per-IP request filter; config = an app-config override replacing the version's config as THIS deployment's ENCLAVE_CONFIG — without it the version's approved record applies; ports/appPort ride along informational)",
        fundMethod: "fundWithAuthorization(bytes32 id, address from, uint256 value, uint256 validAfter, uint256 validBefore, bytes32 nonce, bytes signature)",
        fundEthMethod: "fundEth(bytes32 id) payable",
        setAppRefMethod: "setAppRef(bytes32 id, string appRef) — owner-only version change (ledgers with deploymentsSchema() >= 3): "
                       + "repoint a funded deployment at another approved catalog version; the current runner restarts it in place, "
                       + "balance and lease carry over — no second buy-in to upgrade",
        hint: "POST /v1/claim-hint {\"id\": \"0x…\"}",
      },
    });
  }
  // Per-owner cap on UNPAID reservations: an awaiting_payment deployment holds
  // hardware for the whole PAYMENT_WINDOW_SEC before any payment lands, so an
  // unbounded caller could reserve the node's capacity for free. Paid/running
  // deployments are never counted; MAX_UNPAID_PER_OWNER=0 disables the cap.
  if (MAX_UNPAID_PER_OWNER > 0) {
    const unpaid = [...deployments.values()].filter((d) => d.owner === req.address && d.status === "awaiting_payment").length;
    if (unpaid >= MAX_UNPAID_PER_OWNER)
      return fail(res, 429, "too_many_unpaid",
        `You have ${unpaid} reservation(s) awaiting payment (max ${MAX_UNPAID_PER_OWNER}). Pay for or cancel one before reserving more.`);
  }
  let image = (b.image && b.image.reference) ? b.image : { reference: DEFAULT_IMAGE };
  // Approval gate (vm backend runs catalog apps): only catalog://<appId>/<idx>
  // records the catalog owner APPROVED may deploy. Checked before any
  // reservation so a refused app never holds capacity or a payment window. The
  // gate also returns the version's exact declared resources — they become the
  // request defaults and the floor a request may not undercut.
  let appMin = { ...NO_MIN };
  let appCfg = "";                       // the version's config, for volume-aware sizing
  // Public endpoint: anyone can reach the app's data path (hosting a website/API).
  // Private (default): only the owner's SIWE token can. Management stays owner-only
  // either way. Confidentiality is unchanged — the TEE still hides the app from the
  // operator; "public" only governs who may send it requests. Computed BEFORE the
  // approval gate: a private deployment may run a still-pending version (dev mode).
  const isPublic = b.public === true || b.public === "true";
  if (PROVISION_BACKEND === "vm") {
    const g = await gateAppReference(image.reference, { forPrivate: !isPublic });
    if (g.error) return fail(res, g.error.status, g.error.code, g.error.msg);
    // A paid app can only run as an on-chain deployment: the publisher's cut
    // is carved out of ledger fundings, and this direct path has none.
    if (g.feePerSec6 > 0n)
      return fail(res, 403, "publisher_fee_unpayable",
        "This app charges a publisher fee, so it must be deployed on-chain (create + fund on the deployments ledger); this direct deploy path cannot pay the publisher.");
    image = { ...image, reference: g.ref };
    appMin = g.min;
    appCfg = g.config || "";
  }
  const appPort = Number(b.port) || 8080;
  // Firewall: the app's per-version ports config from the catalog ("http" | "http:N"
  // | "tcp:N" | "udp:N"). The wasm-manager grants wasi:sockets and enforces that the
  // app binds ONLY these (bind audit kills violators). Declared TCP ports are reached
  // through the one attested origin as a WebSocket bridge at /x/:id/tcp/:port.
  let firewall;
  try { firewall = parseFirewall(b.firewall); }
  catch (e) { return fail(res, 422, "invalid_spec", e.message); }

  // resource request: TWO SHARES, nothing else. resources.gpuShare (0..1 of one
  // GPU card: VRAM + compute together; 0 = CPU-only app) and resources.cpuShare
  // (0..1 of the node's vCPU+RAM). The app's exact specs in the catalog set the
  // MINIMUM shares (spec / this server's spec, the larger of the memory and
  // compute axes, rounded up to the percent grain) — a request below either
  // minimum is refused. The two shares are otherwise independent: a GPU app may
  // buy more node than card (rev-13 ledgers; earlier ones refused it). Routing:
  // GPU work needs a GPU enclave; CPU-only work runs on either flavor — a GPU
  // enclave serves it from LEFTOVER cpu pool.
  const r0 = b.resources || {};
  if (r0.share != null || r0.computeShare != null || r0.vramGb != null || r0.memMb != null
      || r0.gpuTflops != null || r0.cpuTflops != null || r0.gpuGflops != null || r0.cpuGflops != null)
    return fail(res, 422, "invalid_spec", "Deployments buy SHARES: request resources.gpuShare (0..1 of one GPU card; 0 = CPU-only) and resources.cpuShare (0..1 of the node). Exact resources (vramGb/memMb/compute) are declared by the app in the catalog and only set the minimum shares.");
  const appVols = volumesInConfig(appCfg);
  const mins = minSharesOf(appMin, { volGb: appVols.length
    ? volumeGb(appVols, await vmHealth().catch(() => null)) : 0 });
  // An ask this box's card cannot cover at ANY share is its own refusal, and it
  // has to be said before the 0..1 validation below turns 7.69 into an
  // unhelpful "must be in [0, 1]". This is the sizing ratio doing its job: the
  // number IS the answer to "how much bigger a card would this need".
  if (mins.gpuShare > 1 + 1e-9 && !mins.gpuOptional)
    return fail(res, 409, "no_capacity", `This app needs ${round1(mins.gpuShare)}x this enclave's whole card `
      + `(${round1(CARD_VRAM_GB)} GB / ${round1(CARD_TFLOPS)} TFLOPS); deploy it on a larger card.`);
  const gpuShare0 = r0.gpuShare != null ? Number(r0.gpuShare) : mins.gpuFloor;
  if (!(gpuShare0 >= 0 && gpuShare0 <= 1))
    return fail(res, 422, "invalid_spec", "resources.gpuShare must be in [0, 1].");
  if (gpuShare0 > 0 && !IS_GPU)
    return fail(res, 422, "invalid_spec", "This is a CPU-only enclave: GPU shares are not served here. Set resources.gpuShare to 0 (CPU-only), or deploy to a GPU enclave.");
  const needCpu0 = mins.cpuShare;
  const cpuShare0 = r0.cpuShare != null ? Number(r0.cpuShare)
    : Math.max(needCpu0, gpuShare0 > 0 ? Math.min(0.05, gpuShare0) : 0.05);
  if (!(cpuShare0 > 0 && cpuShare0 <= 1))
    return fail(res, 422, "invalid_spec", "resources.cpuShare must be in (0, 1].");
  if (gpuShare0 < mins.gpuFloor - 1e-9 || cpuShare0 < needCpu0 - 1e-9)
    return fail(res, 422, "invalid_spec", `Below this app's minimum shares: its declared specs need at least gpuShare ${round3(mins.gpuFloor)} and cpuShare ${round3(needCpu0)} on this hardware.`);

  let slice, gpu, rate;
  if (!(gpuShare0 > 0)) {
    slice = normalizeCpuReq(cpuShare0);
    gpu = allocCpu(slice.cpuShare);
    if (!gpu) return fail(res, 409, "no_capacity",
      `Requested ${slice.pct}% of the node's CPU/RAM but only ${Math.round(maxFreeCpu() * 100)}% is free.`);
    rate = rateFor(0, slice.cpuShare);
  } else {
    // reserve an arbitrary GPU slice + its CPU slice; the worker isn't spawned until payment lands
    slice = normalizeGpuReq(gpuShare0, cpuShare0);
    if (slice.gpuShare > maxFreeGpuShare() + 1e-9)
      return fail(res, 422, "invalid_spec", `requested gpuShare ${round3(slice.gpuShare)} exceeds the largest free slice of a single card (${round3(maxFreeGpuShare())} = ${round1(maxFreeGpuShare() * CARD_VRAM_GB)} GB / ${round1(maxFreeGpuShare() * CARD_TFLOPS)} TFLOPS).`);
    gpu = allocGpu(slice.vramGb, slice.computeShare, slice.cpuShare);
    if (!gpu) return fail(res, 409, "no_capacity",
      `No capacity for gpuShare ${round3(slice.gpuShare)} + cpuShare ${round3(slice.cpuShare)} (free: ${round3(maxFreeGpuShare())} of a card, ${round3(maxFreeCpu())} of the node).`);
    rate = rateFor(slice.gpuShare, slice.cpuShare);
  }

  const id = rid("dep_");
  const payRef = keccak256(stringToBytes(id));          // the bytes32 to pass to EnclavePay.payWithAuthorization()
  const rec = {
    id, owner: req.address, status: "awaiting_payment", public: isPublic, firewall,
    image, command: b.command || [],
    // the two shares bought (the app's catalog specs only set the minimums)
    resources: gpu.cpu
      ? { gpuShare: 0, cpuShare: slice.cpuShare }
      : { gpuShare: slice.gpuShare, cpuShare: slice.cpuShare, cardId: gpu.cardId },
    network: { port: appPort, protocol: "https", endpoint: `${originOf(req)}/x/${id}` },
    attestation: { available: true, vmTechnology: vmTech(), gpuTechnology: IS_GPU ? "nvidia-cc" : null, href: `/v1/deployments/${id}/attestation` },
    region: "tinfoil", createdAt: new Date().toISOString(), startedAt: null,
    // fair-billing clock: a funded BALANCE (null = unlimited pilot) drained only
    // on healthy ticks - see startBillingTicker. paused surfaces a frozen clock.
    remainingMs: null, consumedMs: 0, paused: false, pauseReason: null, _lastTickAt: 0,
    payDeadline: Date.now() + PAYMENT_WINDOW_SEC * 1000,
    digest: image.digest || null, rate, payRef, paidUsdc: 0,
    _gpu: gpu, _gpuSpec: gpu.cpu ? null : { cardId: gpu.cardId, cardUuid: gpuCards[gpu.cardId]?.uuid || null, vramCapGb: gpu.vramGb, computeShare: gpu.computeShare },
    _port: 0, _payTimer: null,
  };
  deployments.set(id, rec);
  payRefIndex.set(payRef.toLowerCase(), id);
  armPayTimer(rec);            // release the reservation if unpaid by payDeadline
  saveStateSoon();

  // AUTO_PROVISION: boot now without an on-chain payment (manual billing / pilot).
  if (AUTO_PROVISION) {
    if (!(await forceProvision(rec)))
      return fail(res, 502, "provision_failed", rec.error || "provisioning failed");
    console.log(`[auto-provision] ${id} booted without payment; `
              + `remaining=${rec.remainingMs != null ? Math.round(rec.remainingMs / 1000) + "s" : "unlimited"}`);
  }

  const out = view(rec);                                  // includes payment instructions
  res.status(201).json(out);
});

// ---- per-deployment secrets (relay-stored, guest-env injected) ---------------
// The owner stages env-var-shaped private values (S3 keys, API tokens) on the
// api-relay — POST /v1/secrets/:id, relay/secrets.js documents the whole trust
// model — and every provision pulls the current snapshot here to hand the
// manager as guest --env vars. The fetch authenticates with a key DERIVED from
// the fleet SECRET (the dns-txt pattern: the relay env holds only the derived
// key) and names our endpoint; the relay releases a deployment's secrets only
// to its live on-chain lease holder. Failures never block a launch: the app
// starts on the last snapshot this process fetched, else without secrets, and
// the log says which — a relay outage must not take provisioning down with it.
// SECRETS_API="" disables the pull (values themselves never touch STATE_FILE).
const SECRETS_API = (process.env.SECRETS_API ?? "https://api.enclave.host").trim().replace(/\/+$/, "");
const SECRETS_FETCH_KEY = createHmac("sha256", SECRET).update("enclave secrets v1").digest("hex");
const _secretsCache = new Map();                 // dep id -> { rev, env } (RAM only)
// Skips are announced ONCE per reason. A skip here is indistinguishable from a
// relay refusal downstream — both end with the app running and its $NAME
// placeholders unresolved — but only this side knows the request was never
// sent. Staying quiet about it cost a full investigation: every gate on the
// relay and on chain checked out (secrets stored, capability true, correct
// operator key, live lease held), which is exactly the evidence you get when
// nobody asked.
const _secretsSkipSaid = new Set();
function _secretsSkip(why) {
  if (_secretsSkipSaid.has(why)) return;
  _secretsSkipSaid.add(why);
  console.error(`[secrets] NOT fetching per-deployment secrets: ${why}. `
    + "Deployments whose config uses $NAME placeholders will launch with them UNRESOLVED.");
}
async function fetchDepSecrets(id) {
  if (!SECRETS_API || !_advertisedEndpoint || /^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) {
    if (!SECRETS_API) _secretsSkip("SECRETS_API is empty");
    else if (!_advertisedEndpoint) _secretsSkip("this enclave has not registered its endpoint yet");
    return _secretsCache.get(id) || null;
  }
  const idL = String(id).toLowerCase();
  let last = "";
  for (let attempt = 0; attempt < 3; attempt++) {
    if (attempt) await sleepMs(attempt === 1 ? 1000 : 3000);
    try {
      const ts = Math.floor(Date.now() / 1000);
      const sig = createHmac("sha256",
          createHmac("sha256", Buffer.from(SECRETS_FETCH_KEY, "hex")).update("fetch-auth v1").digest())
        .update(`${idL}:${_advertisedEndpoint}:${ts}`).digest("hex");
      // SECOND FACTOR, and the one that is actually OURS. The HMAC above is the
      // fleet-wide derived key: it proves "a holder of the fleet key" and not
      // "this endpoint", so on its own any fleet member could name another
      // member's endpoint and be handed that deployment's secrets (relay/
      // secrets.js says so at the line). The operator key that REGISTERED this
      // endpoint is per-enclave, and the relay can check it against the registry
      // — so sign the same tuple with it. personal_sign, so this signature can
      // never be replayed as a transaction by the key that also sends claims.
      let opSig = "";
      if (REGISTRY_PK) {
        try { opSig = await claimSigner().account.signMessage({
          message: `enclave-secrets-fetch:${idL}:${_advertisedEndpoint}:${ts}` }); }
        catch { /* unsigned: the relay decides whether its policy still allows it */ }
      }
      const r = await fetch(`${SECRETS_API}/v1/secrets/fetch`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ id: idL, endpoint: _advertisedEndpoint, ts, sig, ...(opSig ? { opSig } : {}) }),
        signal: AbortSignal.timeout(5000) });
      if (r.ok) {
        const b = await r.json();
        const out = { rev: Number(b.rev) || 0, env: b.env && typeof b.env === "object" ? b.env : {} };
        _secretsCache.set(id, out);
        return out;
      }
      // 503 = relay without the feature, 404 = record not on its ledger view:
      // authoritative "nothing to inject", not worth retrying this launch
      if (r.status === 503 || r.status === 404) return _secretsCache.get(id) || null;
      last = `HTTP ${r.status}`;                 // 409 right after our claim tx = ledger lag; retry
    } catch (e) { last = e.message; }
  }
  const cached = _secretsCache.get(id);
  console.warn(`[secrets] ${idL} fetch failed (${last}); launching ${cached ? `on the cached rev ${cached.rev} snapshot` : "without secrets"}`);
  return cached || null;
}

// --- custom domains ---------------------------------------------------------
// The hostnames an owner attached to a deployment THIS enclave runs
// (relay/domains.js holds the records and proves their DNS). We pull them for
// three reasons: to mint their certificates, to know which names our TLS may
// answer for, and to tell the guest what it is allowed to consider its own.
//
// Same fetch shape and same authority as the secrets pull — the fleet HMAC
// and/or this enclave's registry operator key, scoped by the on-chain lease —
// because it is the same question: what does the box running this deployment
// get to know about it? The reply is a list of names, not a secret, but the
// authorization matters anyway: a box that could read another deployment's
// domains could mint certificates for a customer's hostname.
//
// The response also carries the ACME challenge alias to push TXT records to
// (see acmeChallengeName), and the request carries an issuance REPORT, which
// is the only way a customer ever learns that a CA refused their domain.
const DOMAINS_API = (process.env.DOMAINS_API ?? SECRETS_API).trim().replace(/\/+$/, "");
const _depDomains  = new Map();      // dep id -> string[] hostnames we may serve
const _domainOwner = new Map();      // hostname -> dep id (reverse index: ACME + SNI + the Host check)
// Until the first refresh has answered, "no deployment claims this custom
// name" is not yet a fact - it is a gap in local state. A RESTORED custom
// cert (ACME_STORE_DIR) is present at the very first reconcile, before the
// relay has been asked, and must not be dropped for that. No DOMAINS_API =
// there are no custom names to learn, so the answer is known at once.
let _domainsKnown = !DOMAINS_API;
const _certReports = new Map();      // hostname -> { ok, ca, error } queued for the next fetch

function reindexDomains() {
  _domainOwner.clear();
  for (const [id, hosts] of _depDomains) for (const h of hosts) _domainOwner.set(h, id);
}
// Is this a customer hostname we manage certificates for? (sniDecide's third
// case, and the Host check on the HTTPS bridge.)
const customDomainOwner = (host) => _domainOwner.get(String(host || "").toLowerCase().split(":")[0].replace(/\.+$/, "")) || null;

async function fetchDepDomains(id) {
  if (!DOMAINS_API || !_advertisedEndpoint || /^(1|true|on)$/i.test(process.env.MOCK_SPAWN || ""))
    return _depDomains.get(id) || [];
  const idL = String(id).toLowerCase();
  try {
    const ts = Math.floor(Date.now() / 1000);
    const sig = createHmac("sha256",
        createHmac("sha256", Buffer.from(SECRETS_FETCH_KEY, "hex")).update("domains-fetch v1").digest())
      .update(`${idL}:${_advertisedEndpoint}:${ts}`).digest("hex");
    let opSig = "";
    if (REGISTRY_PK) {
      try { opSig = await claimSigner().account.signMessage({
        message: `enclave-domains-fetch:${idL}:${_advertisedEndpoint}:${ts}` }); }
      catch { /* the relay decides whether its policy still allows the HMAC alone */ }
    }
    // report only on the names this deployment owns, and clear them as they go:
    // a report that fails to send is retried by the next tick, not lost.
    const mine = _depDomains.get(idL) || [];
    const report = [];
    for (const h of mine) { const r = _certReports.get(h); if (r) report.push({ hostname: h, ...r }); }
    const r = await fetch(`${DOMAINS_API}/v1/domains/fetch`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: idL, endpoint: _advertisedEndpoint, ts, sig,
                             ...(opSig ? { opSig } : {}), ...(report.length ? { report } : {}) }),
      signal: AbortSignal.timeout(5000) });
    if (!r.ok) {
      // 503 (relay without the feature) and 404 (not on its ledger view) are
      // authoritative "no custom domains", not an outage worth logging every tick.
      if (r.status === 503 || r.status === 404) { _depDomains.set(idL, []); reindexDomains(); return []; }
      throw new Error(`HTTP ${r.status}`);
    }
    const b = await r.json();
    for (const item of report) _certReports.delete(item.hostname);        // delivered
    const hosts = (Array.isArray(b.domains) ? b.domains : [])
      .map((h) => String(h).toLowerCase().replace(/\.+$/, ""))
      .filter((h) => /^[a-z0-9.-]{1,253}$/.test(h));
    const before = (_depDomains.get(idL) || []).join(",");
    _depDomains.set(idL, hosts);
    reindexDomains();
    if (before !== hosts.join(",")) console.log(`[domains] ${idL.slice(0, 10)}… serves ${hosts.length ? hosts.join(", ") : "no custom domains"}`);
    return hosts;
  } catch (e) {
    // KEEP the last known list. A relay blip must not withdraw a live
    // customer's certificate or stop the app answering on their name.
    console.warn(`[domains] ${idL.slice(0, 10)}… fetch failed (${e.message}); keeping the last known list`);
    return _depDomains.get(idL) || [];
  }
}

// Refresh every deployment we are running, then let ACME pick up anything new.
// Cheap (one small POST per deployment per cycle) and it is what makes an
// attach reach a live app without a restart.
async function refreshCustomDomains() {
  if (!DOMAINS_API) return;
  const live = [...deployments.values()].filter((r) => r._onchain && r.status === "running");
  for (const rec of live) await fetchDepDomains(rec.id);
  for (const id of [...(_depDomains.keys())]) if (!live.some((r) => r.id.toLowerCase() === id)) _depDomains.delete(id);
  reindexDomains();
  _domainsKnown = true;
  acmeReconcileSoon();
}

// Every hostname a deployment answers on: its own subdomain first (the
// canonical one), then the customer's. This is what the guest gets as
// ENCLAVE_HOSTS and what the Host check on the bridge accepts.
function hostsFor(rec) {
  const names = [];
  if (APP_CERT_DOMAIN && (servesHttp(rec) || fwTcpPorts(rec).length)) names.push(appCertName(rec.id));
  for (const h of _depDomains.get(String(rec.id).toLowerCase()) || []) names.push(h);
  return names;
}

// The relay-held half of a launch: the owner's secrets and the deployment's
// own hostnames, both pulled FRESH. A launch is when the guest learns them, so
// `enclave restart` after a secrets change applies it, and an app that was
// down while a hostname was attached comes back knowing about it. Failures
// never block the launch (fetchDepSecrets falls back to its snapshot and says
// so) — a relay outage must not take provisioning down with it.
async function launchSpec(rec) {
  const relayHeld = PROVISION_BACKEND === "vm" && rec._onchain;
  const sec = relayHeld ? await fetchDepSecrets(rec.id) : null;
  if (sec) { if (sec.rev > 0) rec.secretsRev = sec.rev; else delete rec.secretsRev; }
  if (relayHeld) await fetchDepDomains(rec.id).catch(() => {});
  return launchSpecFrom(rec, sec, hostsFor(rec));
}

// Spawn the tenant's MPS-capped worker process (called once, on first payment).
async function provisionTenant(rec) {
  try {
    const sp = await spawnContainer(await launchSpec(rec));
    rec._port = sp.internalPort;
    if (sp.vmId) { rec._vmId = sp.vmId; rec._vmHostPort = sp.hostPort; }
    if (sp.portMap) rec.portMap = sp.portMap;   // logical -> actual (public: clients see their mapping)
    if (!rec.startedAt) rec.startedAt = Date.now();
    rec.status = "running"; rec.paused = false; rec.pauseReason = null; rec._lastTickAt = Date.now();
    acmeReconcileSoon();   // public+http deployments earn a browser cert for <label>.APP_CERT_DOMAIN (no-op unless ACME is configured)
    return true;
  } catch (e) {
    rec.status = "failed"; rec.error = e.message;
    if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
    console.error(`[provision] ${rec.id} failed: ${e.message}`);
    return false;
  }
}

// Provision a deployment WITHOUT a payment (auto-provision / admin). Clears the
// unpaid-reservation timer and sets the optional safety expiry.
async function forceProvision(rec) {
  if (rec._payTimer) { clearTimeout(rec._payTimer); rec._payTimer = null; }
  const ok = await provisionTenant(rec);
  if (ok) rec.remainingMs = AUTO_PROVISION_HOURS > 0 ? AUTO_PROVISION_HOURS * 3600 * 1000 : null; // null = unlimited
  saveStateSoon();
  return ok;
}

// A Paid event landed: provision on first payment, extend expiry on top-ups.
async function onPaid(payRefHex, payer, amountRaw) {
  const id = payRefIndex.get(String(payRefHex).toLowerCase());
  if (!id) { console.warn(`[pay] payment for unknown ref ${payRefHex} (${amountRaw})`); return; }
  const rec = deployments.get(id); if (!rec) return;
  const seconds = usdcToSeconds(amountRaw, rec.rate);
  rec.paidUsdc = (rec.paidUsdc || 0) + Number(amountRaw);
  if (rec.status === "awaiting_payment") {
    if (rec._payTimer) { clearTimeout(rec._payTimer); rec._payTimer = null; }
    if (!(await provisionTenant(rec))) { saveStateSoon(); return; }  // failed provisioning surfaces in the record
    rec.remainingMs = seconds * 1000;
    console.log(`[pay] ${id} funded ${(Number(amountRaw)/1e6).toFixed(2)} USDC -> +${Math.round(seconds)}s, provisioned`);
  } else if (rec.status === "running") {
    // top-up adds to the balance; a grace overrun (negative balance) is forgiven.
    // remainingMs === null (unlimited pilot) stays unlimited.
    if (rec.remainingMs != null) rec.remainingMs = Math.max(0, rec.remainingMs) + seconds * 1000;
    console.log(`[pay] ${id} top-up ${(Number(amountRaw)/1e6).toFixed(2)} USDC -> +${Math.round(seconds)}s (${timeRemainingSec(rec) ?? "unlimited"}s left)`);
  } else {
    console.warn(`[pay] ${id} payment ${(Number(amountRaw)/1e6).toFixed(2)} USDC but status=${rec.status}; ignored (no refunds in pay-per-deploy)`);
  }
  saveStateSoon();
}

// Watch the forwarder for Paid events (poll getLogs; robust on public RPC).
// Robust log watch on a public RPC. Two failure modes the naive "scan to tip,
// advance past it" loop hits, and how we kill both:
//  (1) MISSED LOGS -> lost payments. mainnet.base.org is load-balanced; getBlockNumber()
//      and getLogs() can hit different nodes at different heights, so a log in the newest
//      blocks can be absent from the response — and advancing past it drops the payment
//      forever. Fix: only finalize up to tip - PAY_CONFIRMATIONS, and re-scan a trailing
//      overlap every poll so a momentarily-behind node gets a second look.
//  (2) DOUBLE-CREDIT. Re-scanning would re-run onPaid (a top-up) for the same event.
//      Fix: dedup on txHash:logIndex — each payment log is handled exactly once, which
//      also makes a mid-poll RPC failure safe to retry.
const PAY_CONFIRMATIONS = parseInt(process.env.PAY_CONFIRMATIONS || "3", 10);    // blocks of lag before trusting a log
const PAY_RESCAN_BLOCKS = parseInt(process.env.PAY_RESCAN_BLOCKS || "20", 10);   // trailing overlap re-scanned each poll (~40s on Base)
const PAY_CHUNK_BLOCKS  = BigInt(process.env.PAY_CHUNK_BLOCKS || "4000");        // max getLogs range per call (public-RPC safe)
const PAY_MAX_CATCHUP   = BigInt(process.env.PAY_MAX_CATCHUP_BLOCKS || "200000"); // ~4.6 days of Base blocks
const _seenLogs = new Map();   // "txHash:logIndex" -> blockNumber (pruned as the window advances)
let _payFromBlock = null;      // persisted: after downtime the watcher resumes HERE, not at the tip
let _lastPollOkAt = 0;         // freshness signal: billing + reservation expiry freeze while the watcher is blind
let _polling = false;
async function pollPayments() {
  if (!FORWARDER_ADDRESS || _polling) return;   // no overlap: catch-up after downtime can outlast one poll interval
  _polling = true;
  try {
    // instructions need the token's EIP-712 domain; retry here until the first read lands
    if (!_usdcDomain) refreshUsdcDomain().catch(() => {});
    // retry ETH payments that missed an oracle read (never lose a payment)
    if (_pendingEth.length) {
      const q = _pendingEth.splice(0);
      for (const p of q) await onPaidEth(p.payRefHex, p.payer, p.wei);
    }
    const tip = await chainClient.getBlockNumber();
    const safe = tip - BigInt(PAY_CONFIRMATIONS);                    // don't finalize logs newer than this
    if (safe < 0n) return;
    if (_payFromBlock == null) _payFromBlock = safe + 1n;            // first EVER run: start at the (confirmed) tip
    if (safe < _payFromBlock) { _lastPollOkAt = Date.now(); return; } // no new confirmed blocks yet
    if (safe - _payFromBlock > PAY_MAX_CATCHUP) {                    // bound a very long outage
      console.warn(`[pay] catch-up clamped: ${safe - _payFromBlock} blocks behind, scanning last ${PAY_MAX_CATCHUP}`);
      _payFromBlock = safe - PAY_MAX_CATCHUP;
    }
    // Walk the window in chunks: after an outage it can span hours of blocks, and
    // public RPCs reject or silently truncate huge ranges. _lastPollOkAt stays
    // stale until fully caught up, so clocks stay frozen while payments made
    // during the outage are still being credited.
    while (_payFromBlock <= safe) {
      const from = _payFromBlock > BigInt(PAY_RESCAN_BLOCKS) ? _payFromBlock - BigInt(PAY_RESCAN_BLOCKS) : 0n;
      const to = (safe - from) > PAY_CHUNK_BLOCKS ? from + PAY_CHUNK_BLOCKS : safe;
      for (const [k, b] of _seenLogs) if (b < from) _seenLogs.delete(k);   // prune dedup set below the window
      for (const [evt, isEth] of [[PAY_EVENT, false], [PAY_ETH_EVENT, true]]) {
        const logs = await chainClient.getLogs({ address: getAddress(FORWARDER_ADDRESS),
          event: evt, fromBlock: from, toBlock: to });
        for (const lg of logs) {
          const key = `${lg.transactionHash}:${lg.logIndex}`;
          if (_seenLogs.has(key)) continue;                          // exactly-once, even across re-scans / partial failures
          _seenLogs.set(key, lg.blockNumber);
          const a = lg.args || {};
          if (isEth) await onPaidEth(a.deploymentId, a.payer, a.amountWei);
          else       await onPaid(a.deploymentId, a.payer, a.amount);
        }
      }
      _payFromBlock = to + 1n;
      saveStateSoon();                                               // persist the cursor: a restart resumes, not skips
    }
    _lastPollOkAt = Date.now();
  } catch (e) { console.warn(`[pay] poll error: ${e.shortMessage || e.message}`); }
  finally { _polling = false; }
}
function startPaymentWatcher() {
  if (!FORWARDER_ADDRESS) { console.warn("[pay] FORWARDER_ADDRESS unset - payments disabled (deployments will sit awaiting_payment)"); return; }
  console.log(`[pay] watching ${FORWARDER_ADDRESS} for Paid + PaidEth events every ${PAY_POLL_SEC}s (ETH priced via ${ETH_USD_FEED})`);
  const t = setInterval(pollPayments, PAY_POLL_SEC * 1000); if (t.unref) t.unref();
  pollPayments();
  // prime the USDC EIP-712 domain so payment instructions can carry it (retries in pollPayments)
  refreshUsdcDomain().catch((e) => console.warn(`[pay] USDC domain read: ${e.shortMessage || e.message}`));
  // prime + keep the ETH/USD cache warm so payment instructions can quote ethUsd
  ethUsdPrice8().catch((e) => console.warn(`[pay] ETH/USD feed: ${e.shortMessage || e.message}`));
  const p = setInterval(() => ethUsdPrice8().catch(() => {}), 300_000); if (p.unref) p.unref();
}

// ---- fair-billing ticker (replaces the wall-clock reaper) -------------------
// Drains each running deployment's remainingMs by REAL elapsed time, but ONLY
// while the platform is serving: backend manager healthy, payment watcher fresh,
// app instance actually alive. Otherwise the deployment is marked paused and its
// clock FREEZES; it resumes on the first healthy tick. Supervisor downtime
// freezes too: ticks simply don't happen while we're down, and loadState()
// resets _lastTickAt on boot so the gap is never charged.
const isMock = () => /^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "");
async function backendHealthy() {
  if (isMock()) return true;
  try { PROVISION_BACKEND === "vm" ? await vmHealth() : await mgrHealth(); return true; }
  catch { return false; }
}
// vm backend: is this deployment's app instance still alive in the manager?
// (the wasm-manager runs apps as subprocesses; a manager restart loses them)
async function instanceAlive(rec) {
  if (isMock() || PROVISION_BACKEND !== "vm") return true;  // worker backend: manager health is the best signal we have
  if (!rec._vmId) return false;
  const r = await vmReq("GET", `/vms/${encodeURIComponent(rec._vmId)}`, null, 5000).catch(() => null);
  if (!r) return false;                        // manager unreachable mid-tick: freeze, don't respawn
  // the manager reports crashed processes as status "failed" (with the exit
  // signal in .error) - an existing record is NOT the same as a live app
  return r.status === 200 && r.body && r.body.status === "running";
}
function pauseRec(rec, reason) {
  if (!rec.paused || rec.pauseReason !== reason) {
    console.warn(`[bill] ${rec.id} clock FROZEN (${reason})`);
    rec.paused = true; rec.pauseReason = reason; saveStateSoon();
  }
  rec._lastTickAt = Date.now();                // downtime is never charged
}
function resumeRec(rec) {
  if (rec.paused) {
    console.log(`[bill] ${rec.id} clock resumed (${timeRemainingSec(rec) ?? "unlimited"}s left)`);
    rec.paused = false; rec.pauseReason = null;
    rec._lastTickAt = Date.now();              // bill from the moment of resume, not the frozen span
    saveStateSoon();
  }
}
// Respawn a still-funded app whose instance vanished (e.g. the manager container
// restarted). Never marks the record failed - the user keeps their frozen
// balance; retries back off so a broken image can't hammer the manager.
// Launches through launchSpec like every other path: this one used to build its
// own argument object, which is how it came to relaunch tenants without their
// secrets or hostnames.
async function respawnTenant(rec) {
  if (rec._respawning || Date.now() < (rec._respawnAt || 0)) return false;
  rec._respawning = true;
  try {
    const sp = await spawnContainer(await launchSpec(rec));
    rec._port = sp.internalPort;
    if (sp.vmId) { rec._vmId = sp.vmId; rec._vmHostPort = sp.hostPort; }
    if (sp.portMap) rec.portMap = sp.portMap;
    rec._respawnAt = 0; rec._respawnBackoffMs = 0;
    console.log(`[bill] ${rec.id} instance respawned after outage`);
    return true;
  } catch (e) {
    rec._respawnBackoffMs = Math.min((rec._respawnBackoffMs || 15000) * 2, 300000);
    rec._respawnAt = Date.now() + rec._respawnBackoffMs;
    console.warn(`[bill] ${rec.id} respawn failed (${e.message}); retry in ${Math.round(rec._respawnBackoffMs / 1000)}s`);
    return false;
  } finally { rec._respawning = false; }
}
function startBillingTicker() {
  const t = setInterval(async () => {
    const now = Date.now();
    const healthy = await backendHealthy();
    const watcherOk = !FORWARDER_ADDRESS || (now - _lastPollOkAt) < WATCHER_STALE_SEC * 1000;
    for (const rec of deployments.values()) {
      if (rec.status !== "running") continue;
      // On-chain (claimed) deployments: the lease is prepaid wall-clock and the
      // chain doesn't stop while we're unhealthy, so freeze/pause semantics
      // don't apply. remainingMs mirrors the lease; the claim loop extends it
      // by renewing, and an unrenewable lease runs out right here through the
      // normal reaper (the deployment then goes back on the open queue).
      if (rec._onchain) {
        if (rec.paused) { rec.paused = false; rec.pauseReason = null; }   // restart recovery: pause is meaningless
        if (healthy && !(await instanceAlive(rec))) await respawnTenant(rec);
        const elapsed = Math.min(Math.max(0, now - (rec._lastTickAt || now)), 2 * BILL_TICK_SEC * 1000);
        rec._lastTickAt = now;
        rec.consumedMs = (rec.consumedMs || 0) + elapsed;
        rec.remainingMs = rec._leaseUntil * 1000 - now;
        saveStateSoon();
        if (rec.remainingMs < -GRACE_SEC * 1000) {
          console.log(`[reaper] ${rec.id} lease over (not renewed) -> teardown`);
          // Settle what we served in this last, partial period. There is no
          // release on this path (the lease ended on its own), so without a
          // final proof the tail from our last checkpoint to the lease end is
          // never credited — and nothing else would ever come back for it.
          // The checkpoint clamps at leaseUntil, so this can only ever claim
          // time the tenant actually paid for.
          await proveFinalPeriod(rec, "lease expired").catch(() => {});
          try { await stopContainer(rec); } catch {}
          if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
          rec.status = "expired";           // claim sweep may re-adopt it if funded again
        }
        continue;
      }
      if (!healthy)   { pauseRec(rec, "backend_down");  continue; }
      if (!watcherOk) { pauseRec(rec, "watcher_stale"); continue; }
      if (!(await instanceAlive(rec))) {
        pauseRec(rec, "instance_missing");
        if (await respawnTenant(rec)) resumeRec(rec);
        continue;
      }
      resumeRec(rec);
      if (rec.remainingMs == null) { rec._lastTickAt = now; continue; }   // unlimited (pilot)
      // clamp so an event-loop stall or clock jump can't overcharge one tick
      const elapsed = Math.min(Math.max(0, now - (rec._lastTickAt || now)), 2 * BILL_TICK_SEC * 1000);
      rec._lastTickAt = now;
      rec.remainingMs -= elapsed;
      rec.consumedMs = (rec.consumedMs || 0) + elapsed;
      saveStateSoon();
      if (rec.remainingMs < -GRACE_SEC * 1000) {
        console.log(`[reaper] ${rec.id} out of funded time -> teardown`);
        try { await stopContainer(rec); } catch {}
        if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
        rec.status = "expired";
      }
    }
    if (_stateDirty) saveStateNow();
  }, BILL_TICK_SEC * 1000);
  if (t.unref) t.unref();
}

app.get("/v1/deployments", authed, (req, res) =>
  res.json({ data: [...deployments.values()].filter(d => d.owner === req.address).map(view), cursor: null }));

app.get("/v1/deployments/:id", authed, (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  res.json(view(rec));
});

// Trade a control-plane session for an app-origin one, so the owner can open a
// PRIVATE deployment in a browser (a navigation cannot carry a bearer). The
// caller already proved the wallet by SIWE; this only narrows what it holds.
//
// The returned token is useless anywhere else: `aud` pins it to this one
// deployment and verifySessionToken refuses any token that names an audience,
// so it opens no control-plane route even on the enclave that minted it. That
// is the whole point — it is handed to a page on a TENANT origin.
app.post("/v1/deployments/:id/app-token", authed, async (req, res) => {
  const rec = depByIdOrPrefix(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  if (rec.public) return fail(res, 400, "not_private", "This deployment is public — open it directly.");
  const token = await mintAppToken(req.address, rec.id);
  res.json({ token, tokenType: "Cookie", deployment: rec.id,
             expiresAt: new Date(Date.now() + APP_TTL_SEC * 1000).toISOString() });
});

// Operator-only: provision an awaiting_payment deployment WITHOUT a payment.
// Gated by ADMIN_TOKEN (x-admin-token header); returns 404 if the token is unset
// or wrong, so the endpoint is invisible without it. Use for manually-billed deploys.
app.post("/v1/admin/deployments/:id/provision", async (req, res) => {
  if (!ADMIN_TOKEN || !safeEqStr(req.headers["x-admin-token"], ADMIN_TOKEN))
    return fail(res, 404, "not_found", "Not found.");
  const rec = deployments.get(req.params.id);
  if (!rec) return fail(res, 404, "not_found", "No such deployment.");
  if (rec.status !== "awaiting_payment")
    return fail(res, 409, "not_provisionable", `Deployment is ${rec.status}.`);
  if (!(await forceProvision(rec)))
    return fail(res, 502, "provision_failed", rec.error || "provisioning failed");
  console.log(`[admin] ${rec.id} provisioned by operator (no payment)`);
  res.json(view(rec));
});

// Operator-initiated graceful release — the consolidation lever: stop the app
// here, refund the unused lease tail on-chain (releaseLease), and let another
// enclave's sweep re-claim the deployment. The tenant experiences a restart
// (same as any lease migration); nothing is charged for the moved tail. The
// evacuation set stops THIS enclave from re-claiming what it just dropped —
// without it the source's own sweep wins the race and the move never happens.
const _evacuated = new Map();                 // id -> until (ms)
const EVAC_HOLDOFF_MS = 15 * 60_000;
// An OWNER-initiated move reuses the same set with a much shorter fuse: long
// enough for the destination to win the race it is being handed, short enough
// that a move nobody accepts falls back here instead of leaving the app dark.
const MOVE_HOLDOFF_MS = Math.max(30_000, parseInt(process.env.MOVE_HOLDOFF_SEC || "120", 10) * 1000);
app.post("/v1/admin/deployments/:id/release", async (req, res) => {
  if (!ADMIN_TOKEN || !safeEqStr(req.headers["x-admin-token"], ADMIN_TOKEN))
    return fail(res, 404, "not_found", "Not found.");
  const rec = deployments.get(req.params.id);
  if (!rec) return fail(res, 404, "not_found", "No such deployment.");
  if (!rec._onchain || !["running", "claimed"].includes(rec.status))
    return fail(res, 409, "not_releasable", `Deployment is ${rec.status}.`);
  try { await stopContainer(rec); } catch {}
  if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
  rec.status = "expired";
  rec.error = "released by the operator (fleet consolidation); it re-queues and another enclave picks it up";
  _evacuated.set(rec.id, Date.now() + EVAC_HOLDOFF_MS);
  proveAndRelease(rec, "operator release (consolidation)").catch(() => {});
  saveStateSoon();
  console.log(`[admin] ${rec.id} released by operator (consolidation)`);
  res.json(view(rec));
});

// The attribution table for a VRAM incident, one authenticated GET away: the
// manager's device-measured memory, the per-PID compute-app list, the
// reservation ledger and each tenant's pin. On 2026-08-18 ~104 GiB of device
// memory orphaned by an in-place update had to be reconstructed from share
// arithmetic and boot-log fragments because a Tinfoil CVM offers no shell;
// this is the lever that makes the next such hunt a single request. Gated
// like the other admin levers (404 when the token is unset).
app.get("/v1/admin/gpu", async (req, res) => {
  if (!ADMIN_TOKEN || !safeEqStr(req.headers["x-admin-token"], ADMIN_TOKEN))
    return fail(res, 404, "not_found", "Not found.");
  if (!IS_GPU) return fail(res, 404, "no_gpu", "This is a CPU-only enclave: no GPU is attached.");
  if (PROVISION_BACKEND !== "vm") return fail(res, 404, "no_vm_backend", "No wasm-manager on this backend.");
  try {
    const r = await vmReq("GET", "/gpu", null, 15000);
    if (r.status !== 200) return fail(res, 502, "manager_error", `wasm-manager /gpu ${r.status}`);
    res.json(r.body);
  } catch (e) {
    fail(res, 502, "manager_unreachable", e.message);
  }
});

// Order an MPS bounce: the reclaim lever for device memory a dead tenant
// generation's MPS servers still hold (the 2026-08-18 residue took a full CVM
// restart only because nothing could order this). The manager drops the order
// on the shared pipe-dir volume; the mps-control daemon consumes it within
// seconds, with its own cooldown against back-to-back orders. EVERY live GPU
// tenant's CUDA context dies with the bounce and this supervisor respawns
// them - an incident action, gated like the other admin levers.
app.post("/v1/admin/gpu/bounce-mps", async (req, res) => {
  if (!ADMIN_TOKEN || !safeEqStr(req.headers["x-admin-token"], ADMIN_TOKEN))
    return fail(res, 404, "not_found", "Not found.");
  if (!IS_GPU) return fail(res, 404, "no_gpu", "This is a CPU-only enclave: no GPU is attached.");
  if (PROVISION_BACKEND !== "vm") return fail(res, 404, "no_vm_backend", "No wasm-manager on this backend.");
  try {
    const r = await vmReq("POST", "/gpu/bounce-mps",
      { reason: (req.body && req.body.reason) || "admin api" }, 15000);
    console.log(`[admin] MPS bounce ordered (${r.status})`);
    res.status(r.status).json(r.body);
  } catch (e) {
    fail(res, 502, "manager_unreachable", e.message);
  }
});

app.delete("/v1/deployments/:id", authed, async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  if (rec._payTimer) { clearTimeout(rec._payTimer); rec._payTimer = null; }
  // An unpaid reservation never ran: cancel = REMOVE it, so the deploy page
  // doesn't show a ghost (nothing ran, nothing paid — no history worth keeping).
  // A payment broadcast anyway after this lands uncredited at payout, exactly
  // like paying after the reservation window expires.
  if (rec.status === "awaiting_payment") {
    if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
    deployments.delete(rec.id);
    if (rec.payRef) payRefIndex.delete(rec.payRef.toLowerCase());
    saveStateSoon();
    return res.json({ id: rec.id, status: "canceled",
                      note: "Reservation released; nothing was charged." });
  }
  try { await stopContainer(rec); } catch {}
  if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
  // stopContainer was awaited: the instance is gone, so this is the final state
  rec.status = "terminated";
  saveStateSoon();
  if (rec._onchain) {
    // hand the lease back (refunds the unused tail to the on-chain balance).
    // While the deployment stays active+funded on-chain, ANY enclave — this one
    // included — may legitimately re-claim it; a permanent stop is the owner's
    // setActive(false) transaction, not a local delete.
    //
    // ?evacuate=1 — the owner is MOVING off this box, not stopping. Re-claiming
    // is normally correct (a released lease is open work and we may be the best
    // home for it), but here it defeats the whole request: this enclave still
    // has the app staged, so its own sweep re-claims within seconds and the
    // move silently never happens. Same holdoff the operator consolidation
    // path uses, and for the same stated reason — just far SHORTER, because
    // these two failures are not symmetrical: consolidation wants the box
    // drained, while a move that no other enclave accepts should fall back
    // here rather than leave the app dark for a quarter of an hour.
    const evacuate = /^(1|true|yes)$/i.test(String(req.query.evacuate || ""));
    if (evacuate) _evacuated.set(rec.id, Date.now() + MOVE_HOLDOFF_MS);
    proveAndRelease(rec, evacuate ? "owner move" : "owner delete").catch(() => {});
    return res.json({ id: rec.id, status: "terminated",
               ranSeconds: Math.round((rec.consumedMs || 0) / 1000),
               ...(evacuate ? { standDownSec: Math.round(MOVE_HOLDOFF_MS / 1000) } : {}),
               note: "On-chain deployment: lease released (unused lease time refunded to its balance). It stays "
                   + "claimable by any enclave while active and funded — call setActive(false) on EnclaveDeployments "
                   + "to stop it for good."
                   + (evacuate ? ` This enclave stands down from re-claiming it for ${Math.round(MOVE_HOLDOFF_MS / 1000)}s so another can take it; if none does, it becomes claimable here again.` : "") });
  }
  res.json({ id: rec.id, status: "terminated",
             paidUsdc: ((rec.paidUsdc || 0) / 1e6).toFixed(2),
             ranSeconds: Math.round((rec.consumedMs || 0) / 1000),
             note: "Pay-per-deploy: no balance is held, so unused funded time is forfeit on early stop." });
});


// Top-up instructions. An on-chain deployment funds the EnclaveDeployments
// ledger - the contract, not this box, meters its runtime, so EnclavePay
// instructions here would take the user's money without crediting balance6.
// Legacy pre-on-chain records still get the forwarder instructions.
app.post("/v1/deployments/:id/topup", authed, (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  if (!["running", "awaiting_payment"].includes(rec.status))
    return fail(res, 409, "not_toppable", `Deployment is ${rec.status}.`);
  if (rec._onchain) return res.json({
    id: rec.id, status: rec.status, timeRemainingSec: timeRemainingSec(rec),
    funding: {
      contract: DEPLOYMENTS_ADDRESS || null, chainId: CHAIN_ID, usdc: USDC_ADDRESS,
      ratePerSecondUsdc: (rec.rate || 0).toFixed(7),
      fundMethod: "fundWithAuthorization(bytes32 id, address from, uint256 value, uint256 validAfter, uint256 validBefore, bytes32 nonce, bytes signature)",
      fundEthMethod: "fundEth(bytes32 id) payable",
      usdcDomain: _usdcDomain,
      ethUsd: _ethUsd.price8 ? (Number(_ethUsd.price8) / 1e8).toFixed(2) : null,
      note: "On-chain deployment: fund the EnclaveDeployments contract, NOT the legacy forwarder (a forwarder "
          + "payment cannot credit an on-chain balance). USDC: sign a ReceiveWithAuthorization (EIP-712, to = the "
          + "contract, nonce = first 16 bytes of the id + 16 random bytes), then anyone submits fundWithAuthorization. "
          + "ETH: fundEth(id) with msg.value, credited at the contract's Chainlink ETH/USD read. Each payment adds "
          + "amount(6dp)/rate seconds to the on-chain balance.",
    } });
  res.json({ id: rec.id, status: rec.status, timeRemainingSec: timeRemainingSec(rec), payment: paymentInstructions(rec) });
});

// Optional ?nonce=<64 hex chars>: freshness challenge folded into the GPU report
// (the CPU quote needs none - it binds the long-lived TLS key, and freshness
// comes from fetching the RAD over your own connection).
function attestNonce(req, res) {
  const n = req.query.nonce;
  if (n == null) return randomBytes(32).toString("hex");
  if (!/^[0-9a-fA-F]{64}$/.test(n)) { fail(res, 422, "bad_nonce", "nonce must be 32 bytes of hex."); return null; }
  return n.toLowerCase();
}
// PUBLIC: attestation exists for the app's counterparties — its users decide
// whether to send data, and they are exactly the people WITHOUT an owner
// session (everything here is public anyway: the deployment row is on-chain,
// the enclave measurements are served unauthenticated at /v1/attestation on
// this same origin). An owner session buys one upgrade: the GPU report is
// regenerated FRESH over the caller-chosen ?nonce; anonymous callers share the
// cached report (NVML generation is not free) and get freshness from the
// TLS-bound CPU quote fetched over their own connection.
app.get("/v1/deployments/:id/attestation", async (req, res) => {
  // Prefix-resolved, like /x/:id. GET /v1/deployments/<prefix> answers for a
  // hex prefix (the relay falls back to the LEDGER when no enclave claims it,
  // and the CLI/site pass prefixes everywhere), but attestation can only come
  // from the enclave itself - there is no ledger fallback to paper over an
  // exact-match miss, so the same prefix 404'd here while the base route
  // worked. Unique match only, 8+ hex, same rule the app subdomains use.
  const rec = depByIdOrPrefix(req.params.id);
  if (!rec) return fail(res, 404, "not_found", "No such deployment.");
  const isOwner = (await addrFromAuth(req)) === rec.owner;
  let nonce = null;
  if (isOwner) { nonce = attestNonce(req, res); if (nonce == null) return; }
  // A non-owner ?nonce is NOT an error - it's just not honored (fresh NVML
  // generation is the owner's perk). The returned gpu.nonce then won't match
  // the caller's, which is exactly how a verifier should read "freshness
  // unproven" - a 401 here would lock the whole document away from the
  // keyed-but-not-owner users attestation exists for.
  try { res.json({ deploymentId: rec.id, generatedAt: new Date().toISOString(),
                   ...(await getMeasurements(rec, { origin: originOf(req), nonce, freshGpu: isOwner })),
                   guideUrl: "https://enclave.host/#attest" }); }
  catch (e) { fail(res, 502, "attestation_error", e.message); }
});

// The RAW attestation document, PUBLIC and verbatim — the artifact every
// verifier actually consumes (tinfoil-cli, @tinfoilsh/verifier, the site's
// in-browser check all fetch exactly this path and parse {format, body}).
//
// On Tinfoil the SHIM serves this path in front of us, so this route is dead
// code there — the shim wins and nothing changes. It exists for METAL, where
// there is no shim: the document lives on a private loopback port
// (ATTESTATION_URL, e.g. http://127.0.0.1:8443/.well-known/enclave-attestation)
// and nothing re-served it publicly. So a metal box advertised
// verification.attestationEndpoint pointing at a path that 404'd — its own
// selfCheck read "unavailable", and no client could ever verify it (found
// 2026-07-28: every metal0-hosted app failed to verify in the browser).
//
// Serving it verbatim gives nothing away: the quote is self-verifying and
// already public by design — it is what a stranger is SUPPOSED to fetch before
// trusting this enclave with a byte. We serve the cached copy (RAD_CACHE_MS)
// rather than re-fetching per request, for the same reason the shim does:
// quote generation is not free and this endpoint is unauthenticated.
app.get(RAD_PATH, async (req, res) => {
  try { res.json((await fetchEnclaveRad(originOf(req))).doc); }
  catch (e) { fail(res, 503, "attestation_unavailable", e.message); }
});

// Enclave-level attestation, PUBLIC: verify the enclave before logging in or
// sending a byte. GPU evidence is included from the shared cache (refreshed
// with a self-chosen nonce) so an unauthenticated caller can't spam NVML report
// generation; an owner nonce on the per-deployment endpoint buys a fresh challenge.
app.get("/v1/attestation", async (req, res) => {
  const out = await getMeasurements(null, { origin: originOf(req) });
  // Bind the session-verification key to the attestation: a client that trusts
  // this document can trust this key, and thus verify that a session token was
  // ES256-signed in-enclave (not HMAC-minted by the operator). Full key at /v1/session-jwks.
  out.sessionKey = SESSION_JWK
    ? { kid: SESSION_KID, alg: "ES256", jwks: "/v1/session-jwks", keySource: "in-enclave",
        note: "Session JWTs are ES256-signed by this in-enclave key; the operator cannot mint one." }
    : null;
  // Bind the PROOF-OF-TIME key the same way, and for a sharper reason: this is
  // how the "the operator registered a key it holds outside the CVM" lie gets
  // caught. EnclaveRegistry.get(<our id>).proofKey is a CLAIM; the address below
  // is what actually signs, served over the attested origin. Anyone — a tenant,
  // a watcher, a competing seller — can compare the two, and a mismatch is
  // public evidence against a host that is billing for service it cannot prove.
  out.proofKey = PROOF_ACCOUNT
    ? { address: PROOF_ACCOUNT.address, curve: "secp256k1", keySource: "in-enclave",
        registry: { enclaveId: _enclaveId, field: "proofKey" },
        verify: "Compare with EnclaveRegistry.get(enclaveId).proofKey — they MUST match.",
        note: "Signs EnclaveProofOfTime checkpoints. Minted in this CVM; the operator has never held it. "
            + "Rotates on CVM relaunch (memory-only), and the enclave republishes it with setProofKey at boot." }
    : null;
  if (!IS_GPU)                                 // CPU-only enclave: no card, no NVML evidence to fetch
    return res.json({ generatedAt: new Date().toISOString(), ...out, guideUrl: "https://enclave.host/#attest" });
  try {
    const ev = await cachedGpuEvidence();
    out.gpu = { technology: "nvidia-cc", ccMode: ev.ccMode ?? null, nonce: ev.nonce,
                driverVersion: ev.driverVersion ?? null, generatedAt: new Date(_gpuEvCache.at).toISOString(),
                report: ev.gpus?.[0]?.attestationReport_b64 ?? null,
                certChain: ev.gpus?.[0]?.attestationCertChain_b64 ?? null, gpus: ev.gpus || [] };
  } catch (e) { out.gpu = { technology: "nvidia-cc", available: false, error: e.message }; }
  res.json({ generatedAt: new Date().toISOString(), ...out, guideUrl: "https://enclave.host/#attest" });
});

// TLS-bridge cert binding, PUBLIC: closes the attestation gap on the relay
// path (relay/README.md). A relay-path session terminates against a cert
// minted in-enclave at boot (initTlsBridge) — self-signed, so CA validation
// says nothing about it. Publishing the cert + fingerprints OVER THE ATTESTED
// ORIGIN is what binds it: verify /v1/attestation, read the expected
// fingerprint from the same origin, then require exactly that cert when
// connecting to <dep-id>.tcp.<domain>:<port>. The private key never left the
// CVM, so nothing outside the enclave — a MITM relay, the operator — can
// present a cert that passes the pin.
app.get("/v1/tls-bridge", (_req, res) => {
  if (!TLS_BRIDGE_INFO) return res.json({ enabled: false });
  res.json({ enabled: true, ...TLS_BRIDGE_INFO,
             verify: "Verify this origin's /v1/attestation first; then require the served cert on "
                   + "<dep-id>.tcp.<domain> connections to match fingerprint256 (or pin spkiPinSha256, "
                   + "or use `certificate` as your sole trust root - it is self-signed, minted in-enclave)." });
});

// PUBLIC session-verification key set (JWKS, RFC 7517). The session JWT is
// ES256-signed by a key minted in-enclave at boot; this is its PUBLIC half,
// served over the attested origin. A client/relay/peer enclave can verify a
// token — and confirm the operator did NOT mint it — while holding no secret.
// The private key never left this CVM.
app.get("/v1/session-jwks", (_req, res) => {
  res.set("cache-control", "public, max-age=300");
  res.json({ keys: SESSION_JWK ? [SESSION_JWK] : [] });
});

// ---- prove this box's on-chain identity to the fleet hub --------------------
// A tunnel name is a routing key, and a quote proves the IMAGE, never which box
// it is; the metal transport key is minted per boot, so neither survives a
// reboot as an identity. The one thing that does is the REGISTRY OPERATOR key —
// the key that registered `<relay>/t/<name>` and claims work under it. So the
// hub asks an attaching box to sign its attach challenge with that key, and
// this is where the guest agent gets that signature (the key lives here, not in
// the agent).
//
// DELIBERATELY NOT A SIGNING ORACLE. The message is BUILT here from a validated
// name + nonce, never taken from the caller, and personal_sign's EIP-191 prefix
// means what comes out can never be replayed as a transaction — which matters
// because this same key sends claim/renew. The gate is a token derived from the
// fleet SECRET (the agent has it; the untrusted relay does not, and every relay
// frame reaches this process through the agent's forwarder).
const OPSIGN_KEY = SECRET.length ? createHmac("sha256", SECRET).update("enclave opsign v1").digest("hex") : "";
app.post("/v1/internal/tunnel-attach-sig", async (req, res) => {
  if (!OPSIGN_KEY || !safeEqStr(req.headers["x-opsign-token"], OPSIGN_KEY))
    return fail(res, 404, "not_found", "Not found.");
  if (!REGISTRY_PK) return fail(res, 409, "no_operator_key", "This enclave holds no registry key.");
  const name = String((req.body && req.body.name) || "").trim();
  const nonce = String((req.body && req.body.nonce) || "").trim();
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) return fail(res, 422, "bad_name", "name must be a plain label.");
  if (!/^[A-Za-z0-9+/=]{1,128}$/.test(nonce)) return fail(res, 422, "bad_nonce", "nonce must be base64.");
  try {
    const signature = await claimSigner().account.signMessage({
      message: `enclave-tunnel-attach:${name}:${nonce}` });
    res.json({ signature, operator: claimSigner().account.address });
  } catch (e) { fail(res, 502, "sign_failed", e.shortMessage || e.message); }
});

// UDP routing map, PUBLIC: the udp-relay (relay/udp-relay.js) polls this to learn
// which per-deployment IPv6 to bind and which logical ports to route into the
// /x/:id/udp/:port bridge. Only public+running deployments with udp ports; the
// addresses are the deterministic ones the relay also derives from the id.
app.get("/v1/udp-map", (_req, res) =>
  res.json({ enabled: !!DEP_ADDR_PREFIX, prefix: DEP_ADDR_PREFIX || null, deployments: udpMap() }));

// Dedicated-IP routing map, PUBLIC: the tcp6-relay (and udp-relay) poll this to
// learn each public+running deployment's dedicated IPv6 and its per-protocol
// logical ports, then bind [address]:port and route into /x/:id/(tcp|udp)/:port.
// Same deterministic addresses the relays also derive from the id.
app.get("/v1/net-map", (_req, res) =>
  res.json({ enabled: !!DEP_ADDR_PREFIX, prefix: DEP_ADDR_PREFIX || null, deployments: netMap() }));

// Tail the worker's stdout/stderr (owner only). ?tail=N (default 200, max 2000).
app.get("/v1/deployments/:id/logs", authed, async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) return res.type("text/plain").send("[mock] no real worker; logs unavailable\n");
  const tail = String(Math.min(2000, Math.max(1, parseInt(req.query.tail, 10) || 200)));
  // vm backend: the wasm-manager keeps each tenant's stdout+stderr - the
  // app's stage markers and, crucially, its last words when it dies (panic,
  // CUDA/ORT abort). Without this, a crashed wasm app is undebuggable by
  // its owner.
  if (PROVISION_BACKEND === "vm") {
    if (!rec._vmId) return fail(res, 409, "no_instance", "No app instance (not provisioned here yet).");
    try {
      const r = await vmReq("GET", `/vms/${encodeURIComponent(rec._vmId)}/logs?tail=${tail}`, null, 15000);
      if (r.status !== 200) return fail(res, 502, "logs_error", (r.body && (r.body.error || r.body.message)) || `HTTP ${r.status}`);
      const b = r.body || {};
      const head = `# status=${b.status || rec.status}${b.exited ? ` exited(code=${b.exitCode})` : ""}${b.error ? ` error=${b.error}` : ""}\n`;
      return res.type("text/plain").send(head + (b.lines || []).join("\n") + "\n");
    } catch (e) { return fail(res, 502, "logs_error", (e.message || "").toString().slice(0, 300)); }
  }
  // worker (GPU PTX) backend: jobs are request/response, no per-tenant log stream.
  return fail(res, 501, "logs_unavailable", "Log retrieval is only available for wasm (vm) deployments.");
});

// Owner restart: stop the app instance and relaunch it in place — same
// version, same lease, same balance (app state is ephemeral by design). The
// remedy for a wedged instance the crash detector can't see: the process is
// up and answering, it just can't do its job — e.g. a tenant that booted
// before its model volume finished mounting and so can never load the model
// (the wasi-nn graph registry seals at process start; seen live 2026-07-18,
// qwen3.5-9b). Same core as the death-relaunch path; _restarting keeps the
// audit tick's claimed-branch from double-provisioning mid-restart.
app.post("/v1/deployments/:id/restart", authed, async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  if (PROVISION_BACKEND !== "vm")
    return fail(res, 501, "restart_unavailable", "Restart is only available for wasm (vm) deployments.");
  if (rec._restarting) return fail(res, 409, "restart_in_progress", "A restart is already in progress.");
  if (rec.status !== "running")
    return fail(res, 409, "not_running", `Only a running deployment restarts in place (status: ${rec.status}). `
      + "Queued/failed work relaunches on its own; suspended work resumes with setActive(true).");
  rec._restarting = true;
  try {
    console.log(`[restart] ${rec.id} owner-requested restart`);
    try { if (rec._vmId) await vmReq("DELETE", `/vms/${encodeURIComponent(rec._vmId)}`, null, 15000).catch(() => {}); } catch {}
    rec._deaths = 0;                 // an owner restart earns a fresh crash budget (the version-switch rule)
    rec.status = "claimed";          // the provision path's input state
    if (!(await provisionTenant(rec))) {
      // same contract as every failed provision: the record keeps the reason,
      // the lease goes back refunded so the fleet (this node included) retries
      noteProvisionFailure(rec.id, rec.image && rec.image.reference);
      proveAndRelease(rec, "owner restart provision failed").catch(() => {});
      saveStateSoon();
      return fail(res, 502, "restart_failed",
        rec.error || "Relaunch failed — the lease was handed back; the fleet retries.");
    }
    saveStateSoon();
    res.json({ id: rec.id, status: rec.status,
               note: "Restarted in place — same version and balance; app state is ephemeral." });
  } finally { rec._restarting = false; }
});

app.use((_req, res) => fail(res, 404, "not_found", "No such route."));
// Final error middleware (4-arg): a rejected async handler was forwarded here by
// wrap() (installed at the top of the app). Never crash — log and return a clean
// 500, and never double-send if the handler already began a response.
app.use((err, req, res, _next) => {
  console.error(`[error] ${req.method} ${req.originalUrl}: ${err && (err.stack || err.message || err)}`);
  if (res.headersSent) { try { res.end(); } catch {} return; }
  fail(res, 500, "internal_error", "Internal error.");
});
if (IS_GPU) { await initGpu(); await initMps(); }        // CPU-only enclave: no cards to discover, no MPS
else if (PROVISION_BACKEND !== "vm")
  console.warn("[cpu] GPU_COUNT=0 but PROVISION_BACKEND!=vm — a CPU enclave has no GPU worker; deploys will fail");
// restore persisted deployments/payment cursor BEFORE serving traffic or polling:
// the downtime gap is frozen (never charged) and reservations shift by the gap.
initStatePersistence();
loadState();

// ---------------------------------------------------------------------------
// WebSocket upgrades - the TCP/UDP/TLS bridges below ride the one attested
// origin (no second external port). Gate: session JWT (Authorization header
// or ?token= for browsers/websocat) + ownership where the route demands it.
// ---------------------------------------------------------------------------
const server = http.createServer(app);
const wss = new WebSocketServer({ noServer: true });

async function authUpgrade(req) {
  let token = null;
  const h = (req.headers.authorization || "").match(/^Bearer\s+(.+)$/i);
  if (h) token = h[1];
  else { try { token = new URL(req.url, "http://x").searchParams.get("token"); } catch {} }
  if (!token) return null;
  return verifySessionToken(token);
}

// --- platform-terminated TLS for app TCP ports (/x/:id/tls/:port) -----------
// The public relay (relay/relay.js, on any untrusted box) forwards a client's
// raw TLS bytes into this bridge; the session terminates HERE, inside the
// attested enclave. The key pair is MINTED IN-ENCLAVE at boot — never
// provisioned as a secret, so no operator, ACME account, or
// secret store ever holds it and the relay stays a dumb ciphertext pipe. The
// cert is self-signed for *.<TLS_BRIDGE_DOMAIN>; clients bind it to the
// enclave via the fingerprints published over the attested origin at
// /v1/tls-bridge (CA validation never proved enclave residency anyway — see
// relay/README.md). Stock clients that don't validate certs (psql
// sslmode=require, irssi --tls) connect unchanged; validating clients pin or
// use the published PEM as their trust root. The pair persists in the TLS-bridge
// dir (tmpfs, see tlsBridgeDir), so the fingerprint is stable across supervisor
// restarts within one CVM boot; a full relaunch (tmpfs wiped) mints a fresh key —
// re-read the pin from the attested origin. TLS_BRIDGE_DOMAIN unset = the /tls/
// path answers 503; /tcp/ is unchanged.
const TLS_BRIDGE_DOMAIN = (process.env.TLS_BRIDGE_DOMAIN || "").trim().replace(/^\*\./, "").replace(/\.$/, "");
let TLS_BRIDGE_CTX = null, TLS_BRIDGE_INFO = null;
// The TLS-bridge PRIVATE KEY is minted in-enclave and MUST live on memory-backed
// storage (tmpfs) — it must never touch host-persisted disk. STATE_FILE, by
// contrast, MAY be pointed at a host-backed volume for billing persistence (see
// its comment), so we keep the TLS key on its OWN path, independent of STATE_FILE.
// TLS_BRIDGE_DIR overrides; otherwise use the ramdisk the gpu/cpu configs mount,
// falling back (with a loud warning) to the STATE_FILE dir only when no ramdisk is
// present. Per-boot rotation is preserved: a fresh CVM boot has an empty tmpfs.
function tlsBridgeDir() {
  const explicit = (process.env.TLS_BRIDGE_DIR || "").trim();
  if (explicit) return explicit;
  if (existsSync("/mnt/ramdisk")) return "/mnt/ramdisk/enclave-tls";
  const fallback = join(dirname(STATE_FILE), "tls-bridge");
  console.warn(`[tls-bridge] no /mnt/ramdisk and TLS_BRIDGE_DIR unset — minting the in-enclave TLS key under ${fallback} (the STATE_FILE dir). This dir MUST be memory-backed (tmpfs): if STATE_FILE is on host-backed storage the private key would leak to the host. Set TLS_BRIDGE_DIR to a tmpfs path.`);
  return fallback;
}
function initTlsBridge() {
  if (!TLS_BRIDGE_DOMAIN) return;
  if (!/^[a-z0-9][a-z0-9.-]*$/i.test(TLS_BRIDGE_DOMAIN))
    return console.error(`[tls-bridge] TLS_BRIDGE_DOMAIN ${JSON.stringify(TLS_BRIDGE_DOMAIN)} is not a hostname - /tls/ bridge disabled`);
  const dir = tlsBridgeDir();                    // OWN tmpfs path, independent of STATE_FILE (the key must never hit host disk)
  const certPath = join(dir, "cert.pem"), keyPath = join(dir, "key.pem");
  try {
    let certPem = null, keyPem = null;
    try {   // reuse the persisted pair unless it's near expiry or the domain changed
      const c = readFileSync(certPath, "utf8"), k = readFileSync(keyPath, "utf8");
      const x = new X509Certificate(c);
      if (new Date(x.validTo).getTime() - Date.now() > 30 * 86400e3
          && (x.subjectAltName || "").split(", ").includes(`DNS:*.${TLS_BRIDGE_DOMAIN}`)) { certPem = c; keyPem = k; }
    } catch {}
    if (!certPem) {
      mkdirSync(dir, { recursive: true });
      // 10y self-signed EC P-256: expiry is a formality — trust comes from the
      // attested-origin pin, and a CVM relaunch mints a fresh pair long before.
      execFileSync("openssl", ["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256",
        "-keyout", keyPath, "-out", certPath, "-days", "3650", "-nodes",
        "-subj", `/CN=*.${TLS_BRIDGE_DOMAIN}`,
        "-addext", `subjectAltName=DNS:*.${TLS_BRIDGE_DOMAIN},DNS:${TLS_BRIDGE_DOMAIN}`]);
      chmodSync(keyPath, 0o600);
      certPem = readFileSync(certPath, "utf8"); keyPem = readFileSync(keyPath, "utf8");
      console.log(`[tls-bridge] minted in-enclave key + self-signed cert for *.${TLS_BRIDGE_DOMAIN}`);
    }
    TLS_BRIDGE_CTX = tls.createSecureContext({ cert: certPem, key: keyPem });
    const x = new X509Certificate(certPem);
    TLS_BRIDGE_INFO = {
      subject: x.subject, subjectAltName: x.subjectAltName || null,
      validFrom: x.validFrom, validTo: x.validTo,
      fingerprint256: x.fingerprint256,                               // SHA-256 of the leaf DER (openssl x509 -fingerprint -sha256)
      spkiPinSha256: createHash("sha256")                             // HPKP-style public-key pin (curl --pinnedpubkey)
        .update(x.publicKey.export({ type: "spki", format: "der" })).digest("base64"),
      certificate: x.toString(),
      selfSigned: true, keySource: "in-enclave",                      // the key never existed outside this CVM
    };
  } catch (e) { console.error("[tls-bridge] in-enclave cert mint failed (openssl missing?) - /tls/ bridge disabled:", e.message); }
}
initSessionKey();
initProofKey();
initTlsBridge();
if (TLS_BRIDGE_CTX) console.log(`[tls-bridge] in-enclave TLS termination enabled (/x/:id/tls/:port) · ${TLS_BRIDGE_INFO.fingerprint256}`);

// ============================================================================
// in-enclave ACME - RUNTIME HALF (the pure helpers live up top, next to the
// self-test seam). One account per CA, one cert per public HTTP app at
// <label>.APP_CERT_DOMAIN, held in memory: { keyPem, certPem, ctx } - and
// mirrored into the tmpfs store (ACME_STORE_DIR, pure half) so a container
// restart restores them before any issuance runs (acmeRestore). The
// SNI hook below slots these contexts into the SAME TLS bridge that serves the
// self-signed pair, so a browser hitting /x/:id/https (or a validating client
// on /tls/) gets a CA-signed cert whose key never left this CVM.
// ============================================================================
const acmeCerts = new Map();   // name -> { keyPem, certPem, ctx, expiresAt, renewAt, issuer, cached }
const acmeRetry = new Map();   // name -> { failures, nextAt } (per-name backoff)
const acmeQueue = [];          // names awaiting issuance, FIFO, deduped
let _acmePumping = false;
const sleepMs = (ms) => new Promise((r) => { const t = setTimeout(r, ms); if (t.unref) t.unref(); });

// The relay's canonical app label (MUST mirror relay/api-relay.js depFromHost):
// on-chain 0x ids -> the first 8 hex chars (32 bits; collisions are fantasy);
// a retired-era dep_ id -> the id minus its redundant dep_ prefix.
const appCertLabel = (id) => { const s = String(id).toLowerCase(); return s.startsWith("0x") ? s.slice(2, 10) : s.replace(/^dep[-_]/, ""); };
const appCertName  = (id) => `${appCertLabel(id)}.${APP_CERT_DOMAIN}`;
// "serves HTTP" = empty firewall (classic wasi:http serve mode) or an explicit
// http:N entry; tcp/udp-only apps get no browser subdomain cert.
const servesHttp   = (rec) => { const fw = rec.firewall || []; return fw.length === 0 || fw.some((x) => String(x).startsWith("http")); };
const desiredCertNames = (rec) => {
  const names = [];
  // ONE hostname per deployment: on <label>.APP_CERT_DOMAIN, port 443 is the
  // HTTP surface (unless the tenant declared tcp:443 — their socket wins) and
  // every other DECLARED tcp port is the tenant's socket, all behind the same
  // in-enclave /tls/ terminator — so any http-serving OR tcp-declaring
  // deployment needs the app-zone cert. The tcp.<domain> zone is SUNSET: no
  // per-name certs are issued under it anymore; TLS_BRIDGE_DOMAIN now only
  // names the self-signed fallback pair and gates the /tls/ path.
  if (servesHttp(rec) || fwTcpPorts(rec).length) names.push(appCertName(rec.id));
  // …plus every custom domain the owner attached and the relay verified. These
  // are ordinary single-name certs like the one above, issued by the same
  // slots (the platform service first, acmeSlotsFor); what differs is only
  // WHERE the dns-01 TXT record goes (acmeChallengeName), because we cannot
  // write in the customer's zone — they delegate a name in ours to us.
  for (const h of _depDomains.get(String(rec.id).toLowerCase()) || []) names.push(h);
  return names;
};

// --- ACME protocol plumbing (network; every entry point is ACME_ENABLED-gated
//     via startAcme/acmeReconcileSoon, so none of this runs unconfigured).
//     Every helper takes ONE slot from ACME_CAS and keeps its state - cached
//     directory, account, nonce - ON that slot; accounts and nonces never mix
//     across CAs. Errors that indict the CA rather than the name carry
//     .caLevel: acmeIssue reads it to cool the slot off and fail over.
//     caErr, the bounded acmeFetch and the slot walker live up in the pure
//     half (next to buildCsr) so the platform-service client shares them. ---
async function acmeDir(ca) {
  if (!ca.dir) {
    let r; try { r = await acmeFetch(ca.directory); } catch (e) { throw caErr(`directory: ${e.message}`); }
    if (!r.ok) throw caErr(`directory fetch ${r.status}`);
    ca.dir = await r.json().catch(() => { throw caErr("directory is not JSON (an outage page?)"); });
  }
  return ca.dir;
}
async function takeNonce(ca) {
  if (ca.nonce) { const n = ca.nonce; ca.nonce = null; return n; }
  let r; try { r = await acmeFetch((await acmeDir(ca)).newNonce, { method: "HEAD" }); }
  catch (e) { throw e.caLevel ? e : caErr(`newNonce: ${e.message}`); }
  const n = r.headers.get("replay-nonce");
  if (!n) throw caErr("newNonce returned no replay-nonce");
  return n;
}
// Signed POST (the only verb ACME knows): ES256 JWS envelope, jwk before the
// account exists / kid after, fresh-nonce retry ONCE on badNonce (RFC 8555
// §6.5 - a stale cached nonce is routine, not an error). payload null =
// POST-as-GET. Returns { status, headers, data } with data json-or-text.
// 5xx, network failures, and non-JSON error bodies (a real ACME error is
// always problem+json; HTML means an outage page) are .caLevel.
async function acmePost(ca, url, payload, { useJwk = false } = {}) {
  for (let attempt = 0; ; attempt++) {
    const nonce = await takeNonce(ca);
    const prot  = { alg: "ES256", nonce, url, ...(useJwk ? { jwk: ca.account.jwk } : { kid: ca.account.kid }) };
    let r; try {
      r = await acmeFetch(url, { method: "POST", headers: { "content-type": "application/jose+json" },
                                 body: JSON.stringify(jwsSignEs256(prot, payload, ca.account.key)) });
    } catch (e) { throw caErr(`POST ${url}: ${e.message}`); }
    ca.nonce = r.headers.get("replay-nonce") || ca.nonce;       // every reply carries the next nonce
    const isJson = /json/.test(r.headers.get("content-type") || "");
    const data = isJson ? await r.json().catch(() => null) : await r.text();
    if (r.status >= 400) {
      if (attempt === 0 && data && /badNonce/.test(data.type || "")) continue;
      const e = new Error(`ACME ${r.status} at ${url}: ${isJson ? `${data?.type || "?"} ${data?.detail || ""}`.trim() : String(data).slice(0, 200)}`);
      e.status = r.status;
      if (r.status >= 500 || !isJson) e.caLevel = true;
      const type = String(data?.type || "");
      // the CA named a date: carry it, the walker remembers it per (name, CA)
      if (/:rateLimited$/.test(type)) e.rateLimitedUntil = acmeRetryAfterAt(data?.detail, r.headers.get("retry-after"));
      // a RESTORED account the CA no longer knows (deactivated, or the CA
      // reset): forget it everywhere and cool the slot; the next round
      // registers afresh instead of failing every name on this slot forever
      if (/:accountDoesNotExist$/.test(type) && ca.account?.kid) {
        console.warn(`[acme] ${ca.host}: account ${ca.account.kid} is gone (${type}) - forgetting it, re-registering next round`);
        ca.account = null; if (!ca.platform) acmeStore()?.forgetAccount(ca.directory);
        e.caLevel = true;
      }
      throw e;
    }
    return { status: r.status, headers: r.headers, data };
  }
}
// One account per CA, kept in memory and in the tmpfs store (accounts.json,
// keyed by directory URL) so a container restart reuses it: EAB makes ZeroSSL
// re-registration free, but Let's Encrypt counts new registrations per IP (10
// per 3 h) and a fleet restart is a burst of them. The EAB inner JWS binds our
// key to the CA-issued credential; CAs that need no EAB (an eab-less fallback
// slot) just skip the binding. The Location header is the kid all later JWS
// use. Failing to establish an account is always CA-level: nothing issues
// without one.
async function acmeAccount(ca) {
  if (ca.account?.kid) return ca.account;
  const dir = await acmeDir(ca);
  const saved = ca.platform ? null : acmeStore()?.loadAccount(ca.directory);
  if (saved) {
    try {
      const key = createPrivateKey(saved.keyPem);
      const j = createPublicKey(key).export({ format: "jwk" });
      ca.account = { key, jwk: { crv: j.crv, kty: j.kty, x: j.x, y: j.y }, thumbprint: jwkThumbprint(j), kid: saved.kid, restored: true };
      console.log(`[acme] ${ca.host}: account restored from ACME_STORE_DIR (${saved.kid})`);
      return ca.account;
    } catch (e) { console.warn(`[acme] ${ca.host}: stored account unusable (${e.message}) - registering afresh`); acmeStore()?.forgetAccount(ca.directory); }
  }
  // a slot whose CA demands EAB but that carries none (secrets not set yet)
  // can never register - say so precisely instead of POSTing a doomed request
  if (dir.meta?.externalAccountRequired && !ca.eabKid)
    throw caErr("CA requires External Account Binding but this slot has no EAB pair (secrets unset?)");
  const { publicKey, privateKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
  const j = publicKey.export({ format: "jwk" });
  ca.account = { key: privateKey, jwk: { crv: j.crv, kty: j.kty, x: j.x, y: j.y }, thumbprint: jwkThumbprint(j), kid: null };
  try {
    const r = await acmePost(ca, dir.newAccount,
      { termsOfServiceAgreed: true,
        ...(ACME_CONTACT ? { contact: [ACME_CONTACT] } : {}),
        ...(ca.eabKid ? { externalAccountBinding: eabJws(ca.eabKid, ca.eabHmac, ca.account.jwk, dir.newAccount) } : {}) },
      { useJwk: true });
    ca.account.kid = r.headers.get("location");
    if (!ca.account.kid) throw new Error("newAccount returned no Location (account kid)");
    console.log(`[acme] account registered at ${ca.account.kid}`);
    if (!ca.platform) acmeStore()?.saveAccount(ca.directory, { kid: ca.account.kid, keyPem: privateKey.export({ type: "pkcs8", format: "pem" }) });
  } catch (e) { ca.account = null; e.caLevel = true; throw e; }
  return ca.account;
}
// TXT push/cleanup through the platform DNS daemon. The body HMAC uses a
// DERIVED key, never the raw SECRET: SECRET mints session JWTs, and the DNS
// daemon lives on the relay box, which by design holds no platform secrets —
// it gets only HMAC(SECRET, "enclave dns-txt v1"), which authorizes TXT
// pushes and nothing else. The daemon's env SECRET= is that derived hex:
//   node -e 'console.log(require("node:crypto").createHmac("sha256",
//     process.argv[1]).update("enclave dns-txt v1").digest("hex"))' "$SECRET"
const DNS_TXT_KEY = SECRET ? createHmac("sha256", SECRET).update("enclave dns-txt v1").digest("hex") : "";
// WHERE a name's dns-01 TXT record goes. For our own app zone that is simply
// _acme-challenge.<name>, the record we serve ourselves. For a CUSTOMER's
// domain we have no write access to their zone at all — so they publish a
// permanent CNAME from _acme-challenge.<their name> into ours, and we push the
// record at the far end of it. CAs follow that CNAME as a matter of course; it
// is the standard delegation and it survives renewals with no further action
// from the customer.
//
// The alias is the deployment's OWN challenge name, not a per-domain one, and
// that is load-bearing: dns-relay authorizes an operator-signed push at
// _acme-challenge.<label>.<zone> only when <label> is a hex prefix of a
// deployment that pusher holds the live lease for. Reusing it means a
// self-hosted box — which holds no fleet secret — can mint its tenants' custom
// certificates with exactly the authority it already has. Concurrent orders on
// one alias are fine: the challenge store keeps a SET of values per name and a
// DELETE removes one value, not the name.
function acmeChallengeName(name) {
  const id = customDomainOwner(name);
  return id ? `_acme-challenge.${appCertName(id)}` : `_acme-challenge.${name}`;
}

// The on-ledger deployment a cert name belongs to (its label is the id's hex
// prefix) — the operator-signed push (below) binds the challenge to its lease.
function certNameDeployment(txtName) {
  const host = String(txtName).replace(/^_acme-challenge\./, "");
  for (const r of deployments.values())
    if (/^0x[0-9a-f]{64}$/i.test(r.id || "") && appCertName(r.id) === host) return r.id.toLowerCase();
  return null;
}
// TXT push auth rides two headers and the daemon takes whichever verifies:
//  x-relay-sig     fleet HMAC (derived key) — first-party boxes, any name.
//  x-operator-sig  EIP-191 signature by THIS box's operator key over the same
//                  body — a SELLER box holds no fleet secret (its HMAC is
//                  noise to the daemon), but it can PROVE it holds the
//                  deployment's on-chain lease; the daemon checks the ledger's
//                  runnerOperator. Per-deployment authority, no shared secret —
//                  this is what lets a permissionless metal box mint real certs.
async function dnsTxt(method, name, value) {
  const depId = certNameDeployment(name);
  const body = JSON.stringify({ name, value, ttlSec: 300, ts: Math.floor(Date.now() / 1000),
                                ...(depId ? { deploymentId: depId } : {}) });
  const headers = { "content-type": "application/json" };
  if (DNS_TXT_KEY) headers["x-relay-sig"] = createHmac("sha256", DNS_TXT_KEY).update(body).digest("hex");
  if (depId && REGISTRY_PK) {
    try { headers["x-operator-sig"] = await claimSigner().account.signMessage({ message: body }); }
    catch (e) { console.warn(`[acme] operator co-sign failed (${e.message}); HMAC only`); }
  }
  // bounded like the ACME calls: the challenge daemon is another network hop
  // that can accept and stall, and an unanswered TXT push holds the whole
  // issuance (and, on removal, leaves the record behind)
  const r = await acmeFetch(`${DNS_API}/v1/txt`, { method, headers, body });
  if (!r.ok) throw new Error(`DNS_API ${method} ${name}: HTTP ${r.status}`);
}
// Poll an authz/order URL (POST-as-GET) until ok/bad/timeout, gentle backoff.
// A definitive "became invalid" indicts the name; a poll that TIMES OUT
// indicts the CA (validation/finalization that never completes was exactly
// the 2026-07-18 failure mode, hours before the endpoint died outright).
async function acmePoll(ca, url, what, isOk, isBad, timeoutMs = 90_000) {
  const t0 = Date.now();
  for (let delay = 2000; ; delay = Math.min(Math.round(delay * 1.5), 10_000)) {
    const { data } = await acmePost(ca, url, null);
    if (isOk(data)) return data;
    if (isBad(data)) {
      const errs = data.error || (data.challenges || []).map((c) => c.error).filter(Boolean);
      throw new Error(`${what} became ${data.status}: ${JSON.stringify(errs).slice(0, 300)}`);
    }
    if (Date.now() - t0 > timeoutMs) throw caErr(`${what} still ${data.status} after ${Math.round(timeoutMs / 1000)}s`);
    await sleepMs(delay);
  }
}
// The full dns-01 dance for one name against ONE CA: order -> TXT ->
// challenge -> CSR -> finalize -> download. TXT is deleted win or lose.
// The download is retried on 404 for up to ~30 s: the certificate URL can
// lag the order's `valid` by seconds at Let's Encrypt ("malformed ::
// Certificate not found", replication between its datacenters), and giving
// up throws this key away - the next round orders afresh, which on
// 2026-09-01 spent eyesoff.ai's last duplicate token of the week on a
// certificate nobody collected. (relay/certs.js keeps the ORDER instead; the
// CVM has no store for that, so it waits here.)
const ACME_CERT_FETCH_TRIES = 10, ACME_CERT_FETCH_GAP_MS = 3000;
async function acmeIssueVia(ca, name) {
  const acct = await acmeAccount(ca);
  const dir  = await acmeDir(ca);
  const order = await acmePost(ca, dir.newOrder, { identifiers: [{ type: "dns", value: name }] });
  const orderUrl = order.headers.get("location");
  const authzUrl = order.data.authorizations[0];
  const authz = await acmePost(ca, authzUrl, null);
  // CAs reuse fresh authorizations across orders (renewals often land inside
  // the reuse window): an already-valid authz means no TXT dance at all.
  let txtName = null, txtValue = null;
  if (authz.data.status !== "valid") {
    const chal = (authz.data.challenges || []).find((c) => c.type === "dns-01");
    if (!chal) throw new Error(`no dns-01 challenge offered for ${name}`);
    txtName  = acmeChallengeName(name);
    txtValue = dns01TxtValue(chal.token, acct.thumbprint);
    await dnsTxt("POST", txtName, txtValue);
  }
  try {
    if (txtName) {
      const chal = authz.data.challenges.find((c) => c.type === "dns-01");
      await sleepMs(5000);                                    // let the DNS daemon start answering before the CA looks
      await acmePost(ca, chal.url, {});                       // {} = "I'm ready" (RFC 8555 §7.5.1)
      await acmePoll(ca, authzUrl, `authz for ${name}`, (a) => a.status === "valid",
                     (a) => ["invalid", "revoked", "deactivated", "expired"].includes(a.status));
    }
    const { csrDer, keyPem } = buildCsr(name);
    await acmePost(ca, order.data.finalize, { csr: b64u(csrDer) });
    const done = await acmePoll(ca, orderUrl, `order for ${name}`, (o) => o.status === "valid" && o.certificate,
                                (o) => o.status === "invalid");
    let cert;
    for (let i = 1; ; i++) {
      try { cert = await acmePost(ca, done.certificate, null); break; }   // POST-as-GET; body = PEM chain
      catch (e) {
        if (e.status !== 404 || i >= ACME_CERT_FETCH_TRIES) throw e;
        console.warn(`[acme] ${ca.host}: certificate for ${name} not downloadable yet (${e.message}) - retrying (${i}/${ACME_CERT_FETCH_TRIES})`);
        await sleepMs(ACME_CERT_FETCH_GAP_MS);
      }
    }
    const certPem = String(cert.data);
    const leaf = new X509Certificate(certPem);                // parses the first (leaf) cert of the chain
    const nb = new Date(leaf.validFrom).getTime(), na = new Date(leaf.validTo).getTime();
    return { keyPem, certPem, expiresAt: na,
             renewAt: nb + Math.round((na - nb) * 2 / 3),     // renew past 2/3 of lifetime
             ctx: tls.createSecureContext({ key: keyPem, cert: certPem }) };
  } finally {                                                 // cleanup is best-effort: a leftover TXT is cosmetic
    if (txtName) dnsTxt("DELETE", txtName, txtValue).catch((e) => console.warn(`[acme] TXT cleanup failed for ${txtName}: ${e.message}`));
  }
}
// Issue via the first slot that can (acmeWalkSlots, up in the pure half, is
// the walk itself): the platform service first for a name in our own zones
// or a custom domain attached to a deployment here, then ACME_CAS in order. Two minutes of cool-off, not ten. The cool-off
// bounds how long a dead slot can hold the serial pump's timeouts; two
// minutes bounds it well enough, and ten was the dark window on 2026-08-26:
// ZeroSSL's first call after boot hung (egress was still coming up), Let's
// Encrypt was at its weekly limit for the name, and a running app served no
// certificate until ZeroSSL's cool-off ended.
const ACME_CA_COOLDOWN_MS = parseInt(process.env.ACME_CA_COOLDOWN_MS || "120000", 10);
const _platformPending = new Map();        // name -> { csrPem, keyPem, at }: the key behind a deferred (202) order
// The platform slot serves our own zones AND every custom domain the relay
// listed for a deployment on this box (customDomainOwner): the relay holds
// the name to its verified record and to the on-chain lease, and answers
// the dns-01 at the delegated alias itself. Until 2026-09-02 a custom domain
// went straight to the in-enclave CAs, which on a box without a ZeroSSL pair
// of its own meant Let's Encrypt alone - and its 5/week duplicate cap, spent
// by a day of restarts (eyesoff.ai, 2026-09-01). The in-enclave CAs stay as
// the fallback for both kinds of name.
const acmeSlotsFor = (name) => [...(ACME_PLATFORM && (platformCertName(name) || customDomainOwner(name)) ? [ACME_PLATFORM] : []), ...ACME_CAS];
async function acmeIssue(name) {
  return acmeWalkSlots(name, {
    slots: acmeSlotsFor(name), cooldownMs: ACME_CA_COOLDOWN_MS,
    rateLimitedUntil: (ca) => acmeRateLimitedUntil(name, ca.host),
    onRateLimited: (ca, until) => acmeRateLimitSet(name, ca.host, until),
    issueVia: (slot, n) => slot.platform
      // endpoint = the origin we registered under (PUBLIC_URL before the
      // registry answers): the service checks the lease's runner against it,
      // and the operator key that registered it co-signs, as for a secrets fetch
      ? acmeIssueViaPlatform(slot, n, { endpoint: _advertisedEndpoint || PUBLIC_URL, pending: _platformPending,
          signOp: REGISTRY_PK ? (text) => claimSigner().account.signMessage({ message: text }) : null })
      : acmeIssueVia(slot, n) });
}

// --- coverage + lifecycle ----------------------------------------------------
// Desired set = every public+running deployment that serves HTTP. Reconcile
// diffs desired-vs-held (missing, or past renewAt) into the queue; the pump
// drains it strictly serially with 2s spacing (CA politeness) and per-name
// exponential backoff on failure (5 min doubling, capped at 1h). Reconcile
// runs at boot, every 10 min, and is poked whenever a deployment flips to
// running (provisionTenant).
function acmeReconcile() {
  if (!ACME_ENABLED) return;
  const now = Date.now();
  const desired = new Set();
  for (const r of deployments.values()) {
    // "claimed" counts, not just "running". The certificate depends on nothing
    // the app provides — dns-01 proves control of the NAME, and the name is a
    // function of the deployment id — so ordering it can overlap provisioning
    // instead of queueing behind it. That matters most exactly when the wait is
    // worst: a model-volume tenant only reports "running" once its weights are
    // loaded, so a 32 GB model on a card meant minutes of provisioning followed
    // by a cold ~25s issuance the user watched as "site can't be reached".
    // Cost of being early: a claim that then fails to provision has spent one
    // issuance on a name it will not serve. Bounded (two CAs, and a failed
    // claim is rare) and far cheaper than serialising every move behind a load.
    // PRIVATE deployments get a name too, since they became browser-reachable:
    // the certificate proves control of a NAME, never a right to the content
    // behind it, and the owner gate on /x/:id is what withholds that. Without a
    // cert here sniSelect fails the handshake closed and the owner cannot reach
    // their own app in a browser at all. No new disclosure: the label is 8 hex
    // of an id that is already public on-chain, and HEAD /x/<id> has always
    // answered 204 to anyone.
    if (r.status !== "running" && r.status !== "claimed") continue;
    for (const name of desiredCertNames(r)) {
      if (acmeCerts.get(name)?.renewAt > now) continue;       // held and still fresh
      if (acmeRetry.get(name)?.nextAt > now)  continue;       // failing; wait out the backoff
      if (!acmeQueue.includes(name)) acmeQueue.push(name);
      desired.add(name);
    }
  }
  // A DETACHED custom domain forgets its key material here. Routing already
  // stopped at the relay, so a held cert for a name nobody claims is not
  // reachable — but "delete" has to mean the enclave stops holding the key too,
  // not "stops holding it at the next reboot".
  //
  // Scoped to custom names on purpose: app-zone certs are NOT pruned, because a
  // deployment that is briefly absent from `deployments` (a restart, a
  // re-claim) would otherwise throw away a perfectly good certificate and spend
  // an issuance re-minting it. A custom name has a positive statement behind it
  // — the relay listed it — so its absence is information; an app-zone name's
  // absence is just a gap in local state.
  const isAppZone = (name) => APP_CERT_DOMAIN && name.endsWith(`.${APP_CERT_DOMAIN}`);
  for (const name of [...acmeCerts.keys()]) {
    if (desired.has(name)) continue;
    if (isAppZone(name)) continue;
    if (_domainOwner.has(name)) continue;                     // still attached, just not running right now
    if (!_domainsKnown) continue;                             // restored before the relay answered: not yet a verdict
    acmeCerts.delete(name);
    acmeRetry.delete(name);
    acmeStore()?.delCert(name);
    console.log(`[acme] dropped ${name} — no deployment here claims it`);
  }
  // The STORE is pruned harder than memory: a file is kept only for a name
  // some deployment on this box still owns (desired, or an app-zone name
  // whose deployment is at least on the books, or an attached custom domain).
  // The memory copy of an app-zone cert for a briefly-absent deployment stays
  // (above); its file does not survive that absence into the next restart -
  // the cost is one re-issuance in a rare case, against never holding a key
  // for a name that left this box.
  const store = acmeStore();
  if (store) {
    const keep = new Set(desired);
    for (const r of deployments.values()) for (const n of desiredCertNames(r)) keep.add(n);
    for (const n of _domainOwner.keys()) keep.add(n);
    if (!_domainsKnown) for (const n of store.certNames()) if (!isAppZone(n)) keep.add(n);
    const pruned = store.prune(keep);
    if (pruned) console.log(`[acme] pruned ${pruned} stored certificate(s) no deployment here claims`);
  }
  if (acmeQueue.length) acmePump();
}
let _acmeSoonTimer = null;
function acmeReconcileSoon() {                                // the status->running hook (cheap, debounced)
  if (!ACME_ENABLED || _acmeSoonTimer) return;
  _acmeSoonTimer = setTimeout(() => { _acmeSoonTimer = null; acmeReconcile(); }, 1000);
  if (_acmeSoonTimer.unref) _acmeSoonTimer.unref();
}
// A reconcile at a KNOWN time -- when a failed name's retry falls due -- so
// the retry does not also wait for the 10-minute interval to come round. One
// timer, kept at the earliest pending due time.
let _acmeAtTimer = null, _acmeAtWhen = 0;
function acmeReconcileAt(when) {
  if (!ACME_ENABLED) return;
  if (_acmeAtTimer && _acmeAtWhen <= when) return;
  if (_acmeAtTimer) clearTimeout(_acmeAtTimer);
  _acmeAtWhen = when;
  _acmeAtTimer = setTimeout(() => { _acmeAtTimer = null; _acmeAtWhen = 0; acmeReconcile(); }, Math.max(0, when - Date.now()));
  if (_acmeAtTimer.unref) _acmeAtTimer.unref();
}
async function acmePump() {
  if (_acmePumping) return;
  _acmePumping = true;
  try {
    while (acmeQueue.length) {
      const name = acmeQueue.shift();
      if (acmeCerts.get(name)?.renewAt > Date.now()) continue;  // became fresh while queued (double-enqueue race)
      try {
        const issued = await acmeIssue(name);
        acmeCerts.set(name, issued);
        acmeRetry.delete(name);
        acmeStore()?.putCert(name, issued);                   // replaces the previous record for the name
        // A customer's domain: tell the relay, which is the only path by which
        // the person who owns that name learns their certificate exists.
        if (customDomainOwner(name)) _certReports.set(name, { ok: true, ca: issued.issuer });
        console.log(`[acme] issued ${name} via ${issued.issuer}${issued.cached ? " [cached]" : ""} (expires ${new Date(issued.expiresAt).toISOString()})`);
      } catch (e) {
        // Three kinds of failure, three kinds of wait (acmeRetryPlan says
        // which and why): a platform deferral retries when the service said
        // and counts no failure; a cooling slot retries the moment it is back;
        // name-level rejection everywhere gets the doubling backoff.
        const prev = acmeRetry.get(name)?.failures || 0;
        const { failures, nextAt, why } = acmeRetryPlan(e, prev, acmeSlotsFor(name).map((ca) => ca.downUntil));
        acmeRetry.set(name, { failures, nextAt });
        acmeReconcileAt(nextAt);
        const inSec = Math.round((nextAt - Date.now()) / 1000);
        if (why === "deferred") { console.log(`[acme] deferred ${name}: ${e.message} (asking again in ${inSec}s)`); await sleepMs(2000); continue; }
        // …and the same on failure. "Your domain has no certificate and here is
        // the CA's reason" is the single most useful thing this feature can say.
        if (customDomainOwner(name)) _certReports.set(name, { ok: false, error: `${e.message} (attempt ${failures})` });
        console.error(`[acme] failed ${name}: ${e.message} (retry #${failures} in ${inSec}s${why === "cooling" ? ", when a cooling slot is back" : ""})`);
      }
      await sleepMs(2000);
    }
  } finally { _acmePumping = false; }
}
// Boot restore from the tmpfs store: every unexpired cert (ctx rebuilt) goes
// into acmeCerts, and the per-name per-CA rate-limit dates come back, so the
// first reconcile sees the names as held and asks no CA for them. Pure
// enough to share with the self-test seam (certs = the map to fill).
function acmeRestore(certs = acmeCerts, now = Date.now(), store = acmeStore()) {
  if (!store) return 0;
  let n = 0;
  for (const rec of store.loadCerts(now)) { if (!(certs.get(rec.name)?.expiresAt > rec.expiresAt)) { certs.set(rec.name, rec); n++; } }
  for (const [k, v] of store.loadRateLimits(now)) if (!(acmeRateLimits.get(k) > v)) acmeRateLimits.set(k, v);
  console.log(`[acme] restored ${n} certificate(s) from ACME_STORE_DIR (${store.dir})${acmeRateLimits.size ? `, ${acmeRateLimits.size} rate-limit date(s)` : ""}`);
  return n;
}
function startAcme() {                                        // called at the bottom, with the other boot starters
  if (!ACME_ENABLED) {
    if (ACME_CAS.length || APP_CERT_DOMAIN || DNS_API || process.env.ACME_EAB_KID || process.env.ACME_EAB_HMAC)
      console.warn("[acme] partially configured - needs a complete CA slot (ACME_EAB_KID+ACME_EAB_HMAC, and/or ACME_DIRECTORY_2 with an optional EAB_2 pair) plus APP_CERT_DOMAIN and DNS_API; app-subdomain TLS stays off");
    return;
  }
  acmeRestore();                                              // what the last container life held, BEFORE any issuance
  acmeReconcile();                                            // boot coverage (loadState already ran)
  const t = setInterval(acmeReconcile, 600_000);              // renewals + anything the running-hook missed
  if (t.unref) t.unref();
  const order = [...(ACME_PLATFORM ? [`platform service ${CERTS_API} (own zones + attached custom domains)`] : []), ...ACME_CAS.map((c) => c.host)];
  console.log(`[acme] in-enclave issuance on: <label>.${APP_CERT_DOMAIN} via ${order.join(" -> ")}${ACME_CAS.length ? ` (dns-01 through ${DNS_API})` : ""}`);
}

// --- SNI selection -----------------------------------------------------------
// One lookup shared by every in-enclave TLS termination point. The rule and
// its rationale live in sniDecide (pure, up with the ACME helpers): a held CA
// cert wins, an app-zone name WITHOUT one is refused outright rather than
// served the self-signed placeholder, anything else keeps the pin-verified
// bridge pair. An expired held cert counts as absent — it too fails
// validation, so serving it would reopen the same leak.
const acmeCtxFor = (servername) => {
  const held = acmeCerts.get(String(servername || "").toLowerCase());
  return held && held.expiresAt > Date.now() ? held.ctx : null;
};
const sniSelect = (servername, cb) => {
  const d = sniDecide(servername, acmeCtxFor(servername), TLS_BRIDGE_CTX, !!customDomainOwner(servername));
  if (d.use === "refuse")
    return cb(new Error(`no CA cert held for ${servername} - refusing the handshake instead of serving the self-signed placeholder`));
  cb(null, d.ctx);
};

// --- /x/:id/https - browser HTTPS terminated in-enclave -----------------------
// The passthrough relay forwards a browser's raw TLS bytes here (same WS
// transport as /tls/); we unwrap them with the deployment's ACME cert and feed
// the plaintext into the express app THROUGH a real (non-listening) http.Server
// - so keep-alive, chunked bodies and pipelining all ride Node's own HTTP
// parser, zero hand-rolled parsing. The handler pins every inner request to the
// deployment resolved AT UPGRADE TIME by prefixing its /x/<fullId>; because the
// prefix is ALWAYS applied, a smuggled inner "/x/other/..." merely becomes
// "/x/<id>/x/other/..." - a subpath inside the same deployment, harmless.
const internalAppServer = http.createServer((req, res) => {
  const fullId = req.socket._appDepId;
  if (!fullId) { res.writeHead(500); return res.end(); }      // unreachable: only our emit('connection') feeds this server
  // A Host that belongs to ANOTHER deployment is refused, never served. The
  // relay routes by SNI and the certificate has to match, so getting here with
  // someone else's Host takes a client that deliberately mismatched the two —
  // but "which app answers" must never come down to a header we don't control.
  // 421 is the precise answer (RFC 7540 §9.1.2: this connection is not
  // authoritative for that host) and it is what tells a reusing client to open
  // a fresh connection rather than retry into the same wrong place.
  //
  // Deliberately narrow: only a host we KNOW belongs elsewhere is refused.
  // Anything unrecognized (an IP, a bare origin, a name the fetch hasn't
  // learned yet) still passes, because a stale domain list must never take a
  // running app off the air.
  const owner = customDomainOwner(req.headers.host);
  if (owner && owner !== String(fullId).toLowerCase()) {
    res.writeHead(421, { "Content-Type": "text/plain", "Cache-Control": "no-store" });
    return res.end("Misdirected Request\n");
  }
  req.url = `/x/${fullId}${req.url.startsWith("/") ? "" : "/"}${req.url}`;
  app(req, res);                                              // the express app is a plain (req,res) function
});
internalAppServer.keepAliveTimeout = 180_000;                 // match the data path's idle allowance
internalAppServer.headersTimeout   = 185_000;                 // must exceed keepAliveTimeout (Node's slowloris guard)
internalAppServer.requestTimeout   = 0;                       // streaming request/response bodies can be long-lived
function wsHttpsBridge(req, socket, head, fullId) {
  wss.handleUpgrade(req, socket, head, (ws) => {
    const wsStream = createWebSocketStream(ws);
    const tlsSock  = new tls.TLSSocket(wsStream, { isServer: true, secureContext: TLS_BRIDGE_CTX || undefined, SNICallback: sniSelect });
    tlsSock._appDepId = fullId;                               // the internal server reads this to pin req.url
    // inner requests ride the decrypted stream (no socket address, and the
    // client's headers are TLS-sealed past the relay) — carry the client IP
    // the UPGRADE saw so the WAF's per-IP buckets keep working on this path
    tlsSock._clientIp  = clientIpOf(req);
    const close = () => { try { ws.close(); } catch {} try { tlsSock.destroy(); } catch {} };
    wsStream.on("error", close); wsStream.on("close", close); tlsSock.on("error", close);
    internalAppServer.emit("connection", tlsSock);            // Node parses HTTP off the decrypted stream
  });
}

// bridge a WebSocket to a local TCP port, binary frames both ways
function wsTcpBridge(req, socket, head, port) {
  wss.handleUpgrade(req, socket, head, (ws) => {
    const tcp = net.connect(port, "127.0.0.1");
    const close = () => { try { ws.close(); } catch {} try { tcp.destroy(); } catch {} };
    tcp.on("connect", () => {
      ws.on("message", (d) => tcp.write(d));
      tcp.on("data", (d) => { if (ws.readyState === ws.OPEN) ws.send(d); });
    });
    ws.on("close", close); ws.on("error", close);
    tcp.on("close", close); tcp.on("error", close);
  });
}

// like wsTcpBridge, but the frames carry the CLIENT's TLS session: unwrap it
// here (key never leaves the enclave) and pipe cleartext to the app's loopback
// port. The app still speaks plain TCP; TLS is platform dressing on top.
function wsTlsBridge(req, socket, head, port) {
  wss.handleUpgrade(req, socket, head, (ws) => {
    const wsStream = createWebSocketStream(ws);
    // SNI naming a managed ACME cert gets THAT cert (CA-signed, browser-green);
    // app-zone SNI without one is refused (sniSelect fails closed); no SNI or
    // other names keep the pin-verified self-signed bridge pair.
    const tlsSock  = new tls.TLSSocket(wsStream, { isServer: true, secureContext: TLS_BRIDGE_CTX, SNICallback: sniSelect });
    const tcp = net.connect(port, "127.0.0.1");
    const close = () => { try { ws.close(); } catch {} try { tlsSock.destroy(); } catch {} try { tcp.destroy(); } catch {} };
    tlsSock.pipe(tcp); tcp.pipe(tlsSock);
    for (const s of [wsStream, tlsSock, tcp]) { s.on("error", close); s.on("close", close); }
  });
}

// UDP bridge: one WebSocket carries one client's datagram flow. WebSocket
// messages are already framed, so 1 binary message == 1 datagram, both ways —
// no length prefixing. Each flow gets its own loopback dgram socket toward the
// app's actual port, so replies fan back to the right client. UDP has no close,
// so an idle timer tears the flow down.
const UDP_IDLE_MS = parseInt(process.env.UDP_IDLE_MS || "120000", 10);
function wsUdpBridge(req, socket, head, port) {
  wss.handleUpgrade(req, socket, head, (ws) => {
    const udp = dgram.createSocket("udp4");
    let timer = null;
    const close = () => { clearTimeout(timer); try { ws.close(); } catch {} try { udp.close(); } catch {} };
    const bump = () => { clearTimeout(timer); timer = setTimeout(close, UDP_IDLE_MS); };
    ws.on("message", (d, isBinary) => { if (isBinary) { udp.send(d, port, "127.0.0.1"); bump(); } });
    udp.on("message", (d) => { if (ws.readyState === ws.OPEN) { ws.send(d); bump(); } });
    ws.on("close", close); ws.on("error", close); udp.on("error", close);
    bump();
  });
}

server.on("upgrade", async (req, socket, head) => {
  const deny = (line) => { socket.write(`HTTP/1.1 ${line}\r\n\r\n`); socket.destroy(); };

  // dedicated-IP egress: the relay's control channel (/v1/egress-control) and
  // per-connection data streams (/x/egress/<cid>). Both are relay-token gated
  // inside handleUpgrade; it returns true once it owns the path.
  if (egress && egress.handleUpgrade(req, socket, head)) return;

  // ---- app HTTPS: /x/:id/https — the browser's TLS, terminated IN-ENCLAVE ----
  // The passthrough relay tunnels the raw TLS bytes of <label>.APP_CERT_DOMAIN
  // sessions here. Prefix ids resolve like the HTTP path (the subdomain label
  // IS a prefix).
  //
  // This once refused every private deployment outright, on the reasoning that
  // they "keep the token-gated relay-terminated path, so nothing is lost by the
  // 403". Nothing was lost for curl; everything was lost for a browser, because
  // appEndpoint hands out the app-subdomain form and a top-level navigation
  // cannot carry a bearer — so an owner's own private app was simply
  // unreachable from a browser. The refusal was also redundant: this bridge
  // terminates TLS in-enclave and feeds the decrypted request into the SAME
  // express app (internalAppServer rewrites req.url to /x/<id>/…), so the WAF
  // and the owner gate on /x/:id judge it exactly as they judge every other
  // request. Access is decided there, once, not twice.
  const hx = (req.url || "").match(/^\/x\/([^/?]+)\/https(?:\?|$)/);
  if (hx) {
    const rec = depByIdOrPrefix(hx[1]);
    if (!rec)                                   return deny("404 Not Found");
    if (rec.status !== "running")               return deny("409 Conflict");
    if (!TLS_BRIDGE_CTX && !acmeCerts.size)     return deny("503 Service Unavailable"); // no context could complete a handshake
    return wsHttpsBridge(req, socket, head, rec.id);
  }

  // ---- app TCP ports: /x/:id/(tcp|tls)/:port — the declared-firewall data path ----
  // Auth follows the deployment's `public` flag (like the HTTP path). Two gates
  // before bridging: the port must be DECLARED in the firewall config, and the
  // manager must confirm THIS app actually bound it (boundPorts) — otherwise a
  // tenant could declare-but-not-bind a port and bridge into a sibling's socket.
  // /tls/ is the same data path except the supervisor terminates the client's
  // TLS in-enclave first (see wsTlsBridge) — it's what the public relay targets.
  // A declared "tcp:N" serves both flavors; there is no separate tls: entry.
  const t = (req.url || "").match(/^\/x\/([^/?]+)\/(tcp|tls)\/(\d{1,5})(?:\?|$)/);
  if (t) {
    const rec = deployments.get(t[1]), mode = t[2], port = +t[3]; // `port` is the LOGICAL port (the app's advertised one)
    if (!rec)                                 return deny("404 Not Found");
    if (mode === "tls" && !TLS_BRIDGE_CTX)    return deny("503 Service Unavailable");
    if (!fwTcpPorts(rec).includes(port))      return deny("404 Not Found");
    if (!rec.public) {
      const addr = await authUpgrade(req);
      if (!addr)              return deny("401 Unauthorized");
      if (rec.owner !== addr) return deny("403 Forbidden");
    }
    if (rec.status !== "running" || !rec._vmId) return deny("409 Conflict");
    // resolve logical -> actual bind for THIS deployment (two tenants can both be
    // "the 5432 app"; each has its own actual port), then confirm the app bound it.
    const actual = (rec.portMap && rec.portMap["tcp:" + port]) || port;
    const vr = await vmReq("GET", `/vms/${encodeURIComponent(rec._vmId)}`).catch(() => null);
    const bound = (vr && vr.body && vr.body.boundPorts) || [];
    if (!bound.includes(actual)) return deny("409 Conflict");   // app hasn't bound it (yet)
    return (mode === "tls" ? wsTlsBridge : wsTcpBridge)(req, socket, head, actual);
  }

  // ---- app UDP ports: /x/:id/udp/:port — datagrams tunneled over the WS ----
  // Same gates as the tcp path (declared + bound), but bridged as datagrams.
  // The udp-relay routes here by the deployment's per-tenant IPv6, so it only
  // reaches public deployments; private udp is not exposed in v1.
  const u = (req.url || "").match(/^\/x\/([^/?]+)\/udp\/(\d{1,5})(?:\?|$)/);
  if (u) {
    const rec = deployments.get(u[1]), port = +u[2];
    if (!rec)                            return deny("404 Not Found");
    if (!fwUdpPorts(rec).includes(port)) return deny("404 Not Found");
    if (!rec.public) {
      const addr = await authUpgrade(req);
      if (!addr)              return deny("401 Unauthorized");
      if (rec.owner !== addr) return deny("403 Forbidden");
    }
    if (rec.status !== "running" || !rec._vmId) return deny("409 Conflict");
    const actual = (rec.portMap && rec.portMap["udp:" + port]) || port;
    const vr = await vmReq("GET", `/vms/${encodeURIComponent(rec._vmId)}`).catch(() => null);
    const bound = (vr && vr.body && vr.body.boundPorts) || [];
    if (!bound.includes(actual)) return deny("409 Conflict");
    return wsUdpBridge(req, socket, head, actual);
  }

  socket.destroy();
});

// ============================================================================
// portable deployments — the EnclaveDeployments claim loop (see contracts/DEPLOYMENTS.md)
// ============================================================================
// Deployments created on-chain are work items on a queue: this enclave CLAIMS
// one (burning a bounded lease from its funded balance), serves it through the
// exact same provisioning path as HTTP deploys, RENEWs while healthy, and
// RELEASEs on graceful teardown (refunding the unused tail). If we die
// silently, the lease expires on its own and any other enclave picks the
// deployment up — at-most-one-runner is enforced by the contract, not by us.
// Signing uses REGISTRY_PRIVATE_KEY: claims are gated to the operator of our
// registry entry, so advertising (registerOnChain) is a hard prerequisite.
// (DEPLOYMENTS_ADDRESS is a live binding from ./addressbook.js)
const CLAIM_ENABLED    = /^(1|true|on)$/i.test(process.env.CLAIM_ENABLED || "");
// Deployment-secrets capability (availability flags). TWO ways to authenticate
// the fetch, and a box needs only one:
//   - the fleet HMAC, derived from the shared SECRET (the hosted fleet), or
//   - this enclave's OWN registry key, signing the same tuple; the relay checks
//     it against the endpoint's EnclaveRegistry entry and then against the
//     ledger's lease holder.
// The second is the stronger claim and the only one a self-hosted seller can
// make: the fleet SECRET also derives the DNS-TXT key, which authorizes an
// _acme-challenge push for ANY name in the app zone — a certificate for every
// deployment on the platform — and on a metal box it would live in an
// operator-readable file outside the CVM. So a box holding a registry key is
// secrets-capable even with no fleet secret at all; SECRETS_CAPABLE=0 remains
// the explicit opt-out for a box that has neither.
// (Kept env-driven so an operator can still switch it off explicitly; the
// metal guest computes it from whichever credential that box actually holds.)
// ...AND the pull has to be configured at all. Without `&& !!SECRETS_API` this
// flag is a claim about the BUILD while the fleet reads it as a claim about the
// BOX: a box with SECRETS_API="" advertises `secrets: true`, the relay's
// fleet-AND keeps the feature on, the console offers it, and the
// `/v1/secrets/exists` gate — which exists precisely so a secret-bearing
// deployment is "never claimed by a box that would launch it without its env"
// — is defeated by the box's own answer. The deployment then runs, forever,
// with unresolved placeholders and nothing anywhere saying why.
const SECRETS_CAPABLE  = !/^(0|false|off)$/i.test(process.env.SECRETS_CAPABLE || "1") && !!SECRETS_API;
const CLAIM_POLL_SEC   = parseInt(process.env.CLAIM_POLL_SEC || "60", 10);    // sweep + audit + renew cadence
const RENEW_MARGIN_SEC = parseInt(process.env.RENEW_MARGIN_SEC || "600", 10); // renew when less lease than this remains (early renewal is FREE: the contract extends FROM leaseUntil, so a wide margin only buys more attempts)
const CLAIM_MAX_PER_SWEEP = parseInt(process.env.CLAIM_MAX_PER_SWEEP || "3", 10); // new adoptions kicked off per pass (resumes uncapped)
// CPU-only work prefers CPU-only enclaves: a GPU enclave waits this long after
// a CPU-only deployment becomes claimable (created, or its last lease expired)
// before bidding, so CPU enclaves get first claim and GPU leftovers stay a
// fallback rather than the default home.
const CPU_CLAIM_GRACE_SEC = parseInt(process.env.CPU_CLAIM_GRACE_SEC || "120", 10);
const CLAIM_PAGE = 100;
const CLAIM_READY = CLAIM_ENABLED && !!(DEPLOYMENTS_ADDRESS && REGISTRY_READY && PROVISION_BACKEND === "vm");

// ---- reachability watchdog — impure half ------------------------------------
// (verdict logic + rationale sit with the REACH_SELFTEST seam up top.) Runs as
// the claim tick's first stage: while the advertised hostname is affirmed gone
// from public DNS, this enclave stops claiming, stops renewing, and hands back
// everything it holds so a REACHABLE enclave re-claims it within a sweep. A
// positive resolve clears the trip and the sweep takes work again by itself.
const REACH_DNS_STRIKES = parseInt(process.env.REACH_DNS_STRIKES || "5", 10);   // consecutive "gone" rounds to trip; 0 disables
const REACH_DOH_RESOLVERS = (process.env.REACH_DOH_RESOLVERS
  || "https://cloudflare-dns.com/dns-query,https://dns.google/resolve")
  .split(",").map((s) => s.trim()).filter(Boolean);
const _reach = { strikes: 0, tripped: false, checkedAt: null, host: null };

async function dohQuery(resolver, host, type) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 5000);
  try {
    const r = await fetch(`${resolver}?name=${encodeURIComponent(host)}&type=${type}`,
      { headers: { accept: "application/dns-json" }, signal: ctrl.signal });
    if (!r.ok) return "error";
    return dohVerdict(await r.json());
  } catch { return "error"; }
  finally { clearTimeout(t); }
}

// A resolver affirms "gone" only when BOTH address families are absent; one
// live record of either kind proves public DNS still knows the name.
async function resolverVerdict(resolver, host) {
  const [a, aaaa] = await Promise.all([dohQuery(resolver, host, "A"), dohQuery(resolver, host, "AAAA")]);
  if (a === "resolves" || aaaa === "resolves") return "resolves";
  if (a === "gone" && aaaa === "gone") return "gone";
  return "error";
}

async function reachTick() {
  if (!REACH_DNS_STRIKES || !_advertisedEndpoint) return;
  const host = reachHostname(_advertisedEndpoint);
  if (!host) return;
  const verdicts = await Promise.all(REACH_DOH_RESOLVERS.map((r) => resolverVerdict(r, host)));
  const was = _reach.tripped;
  Object.assign(_reach, reachStep(_reach, verdicts, REACH_DNS_STRIKES), { checkedAt: Date.now(), host });
  if (_reach.tripped && !was) {
    console.warn(`[reach] ${host} is GONE from public DNS (${REACH_DNS_STRIKES} consecutive rounds, all resolvers agree): `
               + `unreachable by name — releasing on-chain work and pausing claims`);
    await abandonClaims("runner unreachable: its advertised endpoint vanished from public DNS");
  } else if (!_reach.tripped && was) {
    console.log(`[reach] ${host} resolves again — resuming claims`);
  } else if (!_reach.tripped && _reach.strikes) {
    console.warn(`[reach] ${host}: public DNS affirms no records (strike ${_reach.strikes}/${REACH_DNS_STRIKES})`);
  }
}

// Hand back EVERYTHING held on-chain — the same teardown the audit applies to
// a lost lease. The work re-queues the moment the release lands; keeping an
// app alive behind a dead front only burns its owner's balance. "expired" is
// CLAIM_TERMINAL, so once DNS returns the sweep may re-claim it right here.
async function abandonClaims(why) {
  for (const rec of [...deployments.values()]) {
    if (!rec._onchain || !["running", "claimed"].includes(rec.status)) continue;
    try { await stopContainer(rec); } catch {}
    if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
    rec.status = "expired"; rec.error = why;      // the owner's evidence (console polls the record)
    proveAndRelease(rec, why).catch(() => {});
  }
  saveStateSoon();
}

// mirrors EnclaveDeployments.Deployment (field order must match the struct
// exactly; schema rev 2)
const DEPLOYMENT_COMPONENTS = [
  { name: "id", type: "bytes32" }, { name: "owner", type: "address" },
  { name: "appRef", type: "string" }, { name: "ports", type: "string" },
  { name: "configCid", type: "string" },
  { name: "gpuMilli", type: "uint16" }, { name: "cpuMilli", type: "uint16" },
  { name: "appPort", type: "uint32" },
  { name: "isPublic", type: "bool" }, { name: "active", type: "bool" },
  { name: "createdAt", type: "uint64" },
  { name: "rate", type: "uint256" }, { name: "balance6", type: "uint256" }, { name: "spent6", type: "uint256" },
  { name: "runner", type: "bytes32" }, { name: "runnerOperator", type: "address" }, { name: "leaseUntil", type: "uint64" },
];
// rev-1 ledgers carry a removed sshPubKey string after ports (decoded, ignored)
const DEPLOYMENT_COMPONENTS_V1 = [
  ...DEPLOYMENT_COMPONENTS.slice(0, 4), { name: "sshPubKey", type: "string" }, ...DEPLOYMENT_COMPONENTS.slice(4),
];
const depsAbiFor = (components) => [
  { type: "function", name: "claim", stateMutability: "nonpayable",
    inputs: [{ name: "id", type: "bytes32" }, { name: "enclaveId", type: "bytes32" }], outputs: [] },
  { type: "function", name: "renew", stateMutability: "nonpayable",
    inputs: [{ name: "id", type: "bytes32" }], outputs: [] },
  { type: "function", name: "release", stateMutability: "nonpayable",
    inputs: [{ name: "id", type: "bytes32" }], outputs: [] },
  { type: "function", name: "claimable", stateMutability: "view",
    inputs: [{ name: "id", type: "bytes32" }], outputs: [{ type: "bool" }] },
  // "could THIS enclave claim it", at the rate WE would charge — see the
  // openForUs() note by tryClaim for why claimable() alone is the wrong
  // question on a free-hosted deployment. rev >= 8; older ledgers revert.
  { type: "function", name: "claimableBy", stateMutability: "view",
    inputs: [{ name: "id", type: "bytes32" }, { name: "enclaveId", type: "bytes32" }],
    outputs: [{ type: "bool" }] },
  { type: "function", name: "get", stateMutability: "view",
    inputs: [{ name: "id", type: "bytes32" }],
    outputs: [{ type: "tuple", components }] },
  { type: "function", name: "getPage", stateMutability: "view",
    inputs: [{ name: "start", type: "uint256" }, { name: "n", type: "uint256" }],
    outputs: [{ type: "tuple[]", components }] },
];
// Which struct shape the ledger at DEPLOYMENTS_ADDRESS speaks. The address is
// a LIVE binding (the address-book poll repoints it mid-flight on a contract
// migration), so the sniff is cached PER ADDRESS and re-runs whenever the
// address changes. A boot-once sniff kept the rev-1 ABI after a live repoint
// to a rev-2 ledger (observed 2026-07-13, minutes after the migration cutover:
// every get/getPage misdecoded and claim-hints 502'd). Only get/getPage decode
// depends on the shape - claim/renew/release/claimable use CLAIM_TX_ABI below.
let _depShape = { addr: null, rev: 1, abi: depsAbiFor(DEPLOYMENT_COMPONENTS_V1) };
async function depsAbi() {
  if (!DEPLOYMENTS_ADDRESS || _depShape.addr === DEPLOYMENTS_ADDRESS) return _depShape;
  try {
    const rev = Number(await chainClient.readContract({ address: getAddress(DEPLOYMENTS_ADDRESS),
      abi: [{ type: "function", name: "deploymentsSchema", stateMutability: "view", inputs: [], outputs: [{ type: "uint256" }] }],
      functionName: "deploymentsSchema" }));
    _depShape = { addr: DEPLOYMENTS_ADDRESS, rev,
                  abi: depsAbiFor(rev >= 2 ? DEPLOYMENT_COMPONENTS : DEPLOYMENT_COMPONENTS_V1) };
    console.log(`[claim] ledger ${DEPLOYMENTS_ADDRESS} struct schema rev ${rev}`);
  } catch (e) {
    if (sniffCachePolicy(e.shortMessage || e.message || "") === "cache-rev1") {
      _depShape = { addr: DEPLOYMENTS_ADDRESS, rev: 1, abi: depsAbiFor(DEPLOYMENT_COMPONENTS_V1) };
      console.log(`[claim] ledger ${DEPLOYMENTS_ADDRESS} struct schema rev 1 (pre-deploymentsSchema contract)`);
    }
    // anything else — transport trouble AND "returned no data" (an RPC that
    // sees no code at the address yet) — don't cache: serve the last known
    // shape this round and re-sniff on the next call
  }
  return _depShape;
}
// tx surface (inputs only, never decodes a Deployment tuple) - shape-independent
const CLAIM_TX_ABI = depsAbiFor(DEPLOYMENT_COMPONENTS);
// Claim/renew receipts carry the post-tx lease in their event - read it from
// THERE, never from a follow-up eth_call: the public RPC's load balancer can
// serve pre-tx state for a minute after confirmation (and rate-limit the read
// outright), and both failure modes made the loop double-renew leases and
// abandon its own freshly-claimed work (observed live 2026-07-05).
const DEPLOYMENT_EVENTS = [
  { type: "event", name: "Claimed", inputs: [
    { name: "id", type: "bytes32", indexed: true }, { name: "enclaveId", type: "bytes32", indexed: true },
    { name: "operator", type: "address", indexed: true }, { name: "leaseUntil", type: "uint64" }, { name: "burned6", type: "uint256" }] },
  { type: "event", name: "Renewed", inputs: [
    { name: "id", type: "bytes32", indexed: true }, { name: "enclaveId", type: "bytes32", indexed: true },
    { name: "leaseUntil", type: "uint64" }, { name: "burned6", type: "uint256" }] },
];
function leaseFromReceipt(rcpt, eventName, id) {
  try {
    const logs = parseEventLogs({ abi: DEPLOYMENT_EVENTS, logs: rcpt.logs, eventName, strict: false });
    const hit = logs.find(l => (l.args.id || "").toLowerCase() === id.toLowerCase());
    return hit ? Number(hit.args.leaseUntil) : null;
  } catch { return null; }
}

// One shared signer (the registry operator EOA) and ONE queue for every tx it
// signs — registry register/heartbeat and ledger claim/renew/release alike.
// The queue serializes through CONFIRMATION, not just send order: public RPCs
// cap EIP-7702-delegated EOAs at a single in-flight tx ("in-flight transaction
// limit reached for delegated accounts"), and even a plain EOA avoids account-
// nonce races this way. A dropped tx can't wedge the queue (receipt wait is
// bounded and failures are swallowed — the caller still sees its own error).
let _claimAccount = null, _claimWallet = null, _txChain = Promise.resolve();
function claimSigner() {
  if (!_claimWallet) {
    _claimAccount = privateKeyToAccount(REGISTRY_PK.startsWith("0x") ? REGISTRY_PK : `0x${REGISTRY_PK}`);
    _claimWallet  = createWalletClient({ account: _claimAccount, chain: base, transport: rpcTransport() });
  }
  return { account: _claimAccount, wallet: _claimWallet };
}
// Bare-metal receipt wait: poll eth_getTransactionReceipt every 2s. viem's
// waitForTransactionReceipt spent the FULL 120s timeout on every tx here even
// though each one mined within seconds — the whole night of 2026-07-19 the
// operator queue ticked at exactly one tx per ~122s on BOTH the public pool
// and a dedicated Alchemy endpoint (metronomic Claimed/Renewed spacing in the
// ledger), which capped fleet tx throughput below ~30 apps' renewal demand.
// Whatever its block-subscription/replacement machinery was doing, the
// simplest primitive can't do it: a mined receipt returns on the next poll.
async function awaitReceipt(hash, timeoutMs = 90_000) {
  const t0 = Date.now();
  for (;;) {
    const r = await chainClient.getTransactionReceipt({ hash }).catch(() => null);
    if (r) return r;
    if (Date.now() - t0 > timeoutMs) throw new Error(`no receipt for ${hash} after ${Math.round(timeoutMs / 1000)}s`);
    await sleepMs(2000);
  }
}
function sendOperatorTx(address, abi, functionName, args) {
  const p = _txChain.then(() => claimSigner().wallet.writeContract({
    address: getAddress(address), abi, functionName, args }));
  const rcptP = p.then((hash) => awaitReceipt(hash));
  _txChain = rcptP.then(() => {}, () => {});   // keep the queue alive across failures
  p.receipt = rcptP;   // callers that need the outcome share the queue's own
  return p;            // receipt wait instead of polling for it a second time
}
const sendClaimTx = (functionName, args) => sendOperatorTx(DEPLOYMENTS_ADDRESS, CLAIM_TX_ABI, functionName, args);

// ---- claim bond (anti-sybil, EnclaveDeployments rev 7) ----------------------
// The ledger can require an operator to lock USDC before it may claim. That
// gate is the only on-chain cost of a SYBIL claim: registration is open and the
// price and proof key in a registry entry are both self-declared, so without a
// bond any address can take the lease on a funded deployment, serve nothing and
// collect the runner escrow. Nothing here posted a bond until 2026-07-29, which
// meant turning the gate on would have made every claim in the fleet revert
// "bond required" - an outage, not a defence. So the switch was unflippable.
//
// This makes it flippable: we read what the ledger asks for and top the bond up
// to it before claiming. While claimBond6 is 0 (the deploy default) every line
// below is inert and no USDC moves, so shipping this changes nothing until the
// owner actually raises the bond - which is the point. CLAIM_BOND_MAX6 is the
// operator's own ceiling on that: the ledger's owner sets the ask, and this box
// refuses to lock more than its operator agreed to, going quiet instead of
// silently spending a seller's float.
const CLAIM_BOND_MAX6 = BigInt(process.env.CLAIM_BOND_MAX6 || "0");
const BOND_ABI = [
  { type: "function", name: "claimBond6", stateMutability: "view", inputs: [], outputs: [{ type: "uint256" }] },
  { type: "function", name: "bondOf", stateMutability: "view", inputs: [{ name: "operator", type: "address" }],
    outputs: [{ name: "amount6", type: "uint256" }, { name: "exitAt", type: "uint64" }] },
  { type: "function", name: "postBond", stateMutability: "nonpayable",
    inputs: [{ name: "amount6", type: "uint256" }], outputs: [] },
];
const ERC20_APPROVE_ABI = [
  { type: "function", name: "approve", stateMutability: "nonpayable",
    inputs: [{ name: "spender", type: "address" }, { name: "amount", type: "uint256" }], outputs: [{ type: "bool" }] },
  { type: "function", name: "allowance", stateMutability: "view",
    inputs: [{ name: "owner", type: "address" }, { name: "spender", type: "address" }], outputs: [{ type: "uint256" }] },
];
let _bond = { ok: null, at: 0, why: null };
// Are we bonded enough to claim? Cached briefly - this runs ahead of the sweep,
// not per deployment. Returns true when the ledger asks for nothing.
async function claimBondReady() {
  if (!CLAIM_READY) return false;
  if (_bond.ok !== null && Date.now() - _bond.at < 300_000) return _bond.ok;
  const dep = getAddress(DEPLOYMENTS_ADDRESS);
  try {
    const need = await chainClient.readContract({ address: dep, abi: BOND_ABI, functionName: "claimBond6" });
    if (need === 0n) { _bond = { ok: true, at: Date.now(), why: null }; return true; }
    const me = claimSigner().account.address;
    const [have, exitAt] = await chainClient.readContract({ address: dep, abi: BOND_ABI, functionName: "bondOf", args: [me] });
    // a pending exit stops authorizing claims even when the amount is there;
    // posting again re-commits and clears it, so the top-up path covers both
    if (have >= need && exitAt === 0n) { _bond = { ok: true, at: Date.now(), why: null }; return true; }
    const short = have >= need ? 0n : need - have;
    if (CLAIM_BOND_MAX6 === 0n || need > CLAIM_BOND_MAX6) {
      _bond = { ok: false, at: Date.now(),
        why: `ledger asks a ${need} claim bond; CLAIM_BOND_MAX6 is ${CLAIM_BOND_MAX6} - not claiming` };
      console.warn(`[bond] ${_bond.why}`);
      return false;
    }
    // approve exactly what we are about to lock, then lock it
    const usdc = getAddress(USDC_ADDRESS);
    const allowance = await chainClient.readContract({ address: usdc, abi: ERC20_APPROVE_ABI,
      functionName: "allowance", args: [me, dep] });
    const post = short > 0n ? short : 1n;   // re-commit clears a pending exit
    if (allowance < post) {
      const ap = sendOperatorTx(usdc, ERC20_APPROVE_ABI, "approve", [dep, post]);
      await ap; await ap.receipt;
    }
    const tx = sendOperatorTx(dep, BOND_ABI, "postBond", [post]);
    await tx;
    const rcpt = await tx.receipt;
    if (rcpt.status !== "success") throw new Error("postBond reverted");
    console.log(`[bond] posted ${post} USDC6; bonded for claiming on ${dep}`);
    _bond = { ok: true, at: Date.now(), why: null };
    return true;
  } catch (e) {
    _bond = { ok: false, at: Date.now(), why: e.shortMessage || e.message };
    console.warn(`[bond] cannot satisfy the claim bond (${_bond.why}); not claiming this pass`);
    return false;
  }
}
// get/getPage through the sniffed shape, with one self-heal retry: a decode
// error means the cached shape is wrong for this ledger however it got there
// (a poisoned sniff, a mid-flight migration) — drop the cache, re-sniff, and
// re-read once instead of staying wedged until the next reboot or repoint.
async function readLedgerContract(functionName, args) {
  const read = async () => chainClient.readContract({
    address: getAddress(DEPLOYMENTS_ADDRESS), abi: (await depsAbi()).abi, functionName, args });
  try { return await read(); }
  catch (e) {
    if (!shapeDecodeError(e.shortMessage || e.message)) throw e;
    console.warn(`[claim] ledger ${functionName} misdecoded with the cached rev-${_depShape.rev} shape — re-sniffing (${e.shortMessage || e.message})`);
    _depShape = { ..._depShape, addr: null };
    return read();
  }
}
const readOnchainDeployment = (id) => readLedgerContract("get", [id]);

// local rec states that no longer hold the lease — safe to re-adopt over
// "stopping" is the pre-terminated legacy name, kept so records persisted by an
// older supervisor still count as terminal after an upgrade.
const CLAIM_TERMINAL = TERMINAL_STATUSES;   // claim/resume may re-adopt exactly the statuses that hold no resources

// ids that failed provisioning here — exponential claim cooldown (see
// considerClaim). In-memory on purpose: a reboot is a fresh chance.
// stage/why/cid ride along so the decline names the cause (a bare "backing
// off" hid a gateway-wide outage for hours — the 2026-08-20 catalog-wasm
// incident) and so prefetch-stage holds can clear themselves (see
// prefetchProbeDue).
const _provisionBackoff = new Map();          // id -> { n, until, ref, stage, why, cid, probedAt, probeClears }
function noteProvisionFailure(id, ref, stage, why, cid) {
  const prev = _provisionBackoff.get(id);
  const n = (prev?.n || 0) + 1;
  const coolMs = Math.min(60 * 60_000, 5 * 60_000 * 2 ** (n - 1));   // 5m, 10m, 20m … cap 1h
  _provisionBackoff.set(id, { n, until: Date.now() + coolMs, ref: ref ?? prev?.ref ?? null,
    stage: stage ?? prev?.stage ?? null, why: why ? String(why).slice(0, 200) : prev?.why ?? null,
    cid: cid ?? prev?.cid ?? null, probedAt: prev?.probedAt || 0, probeClears: prev?.probeClears || 0 });
  return coolMs;
}
// A cooldown binds to the appRef that failed: the owner repointing the
// deployment at another version (setAppRef) is a fresh chance, not the same
// doomed work item on a timer — without this an upgrade shipped to FIX a
// broken version would still sit out up to an hour of backoff here.
function provisionBackoffHolds(entry, nowMs, appRef) {
  if (!entry || nowMs >= entry.until) return false;
  return !entry.ref || entry.ref === appRef;
}
// The decline a held id answers to claim-hints. The legacy prefix stays
// verbatim (humans and notes match on it); the parenthetical names the stage
// and the actual error, so a queued row's why-probe line reads as a diagnosis
// instead of a shrug. Keyword classifiers (WHY_TERMINAL in the console) key
// on structural words this string only carries if the underlying error does.
function provisionDeclineReason(entry, nowMs) {
  const min = Math.max(1, Math.round((entry.until - nowMs) / 60_000));
  const detail = entry.why ? ` (${entry.stage || "provision"}: ${entry.why})` : "";
  return `provisioning failed here recently${detail}; backing off ~${min}min`;
}
// A prefetch failure never burned a claim tx (prefetch runs BEFORE the claim,
// and its cooldown exists to save bandwidth, not the chain), so its hold may
// clear EARLY once the gateway serves the CID again — that is what turns a
// repaired gateway back into a working fleet without anyone clicking Resume.
// Post-claim ("provision") holds never probe: those are the crash-loop guard.
// probedAt gates the network probe to one per id per minute across the sweep
// and every hint source. probeClears caps how often a byte-serving CID may
// clear its own hold: a CID that answers a ranged byte but keeps failing the
// FULL verified prefetch (truncated DAG, size cap) would otherwise clear and
// re-download forever — after 3 cleared-then-failed rounds the plain ladder
// rules again, and only an owner's forced Resume retries early. A repaired
// gateway needs exactly one clear.
function prefetchProbeDue(entry, nowMs) {
  return entry.stage === "prefetch" && !!entry.cid && (entry.probeClears || 0) < 3
      && nowMs - (entry.probedAt || 0) >= 60_000;
}
// One cheap ranged read against the SAME gateway the manager prefetches from
// (same env, same default — wasm_manager.py's IPFS_GATEWAY). 200/206 = the
// bytes are addressable again; the full prefetch+verify still runs on the
// claim path, so a lying gateway only buys itself another verified failure.
// A network-level failure (down, unreachable) marks the gateway bad for a
// minute — one probe may stall its 10s, but a queue of held ids must not each
// pay that stall on every sweep pass.
const IPFS_GATEWAY = (process.env.IPFS_GATEWAY || "https://ipfs.enclave.host").replace(/\/+$/, "");
let _gatewayDownUntil = 0;
async function gatewayServes(cid) {
  if (Date.now() < _gatewayDownUntil) return false;
  try {
    const r = await fetch(`${IPFS_GATEWAY}/ipfs/${cid}`, { headers: { range: "bytes=0-0" },
                          signal: AbortSignal.timeout(10_000) });
    try { await r.body?.cancel(); } catch {}
    return r.status === 200 || r.status === 206;
  } catch { _gatewayDownUntil = Date.now() + 60_000; return false; }
}

// Release with retries, in the background. A failed release strands the lease
// until it expires (~leaseSec of dead air for the user) — observed live when
// the release tx right behind a confirmed claim bounced off the public RPC.
async function releaseLease(id, why) {
  for (let i = 0; i < 4; i++) {
    try {
      const sent = sendClaimTx("release", [id]);
      await sent;
      const rcpt = await sent.receipt;
      if (rcpt.status !== "success") throw new Error("release tx reverted");
      console.log(`[claim] released ${id} (${why})`);
      return true;
    } catch (e) {
      console.warn(`[claim] release ${id} (${why}) attempt ${i + 1}/4 failed: ${e.shortMessage || e.message}`);
      await new Promise(r => setTimeout(r, 15_000 * (i + 1)));
    }
  }
  console.warn(`[claim] release ${id} (${why}) gave up; the lease expires on its own`);
  return false;
}

// Renew every adopted lease that's inside the margin. A failed renew is not
// fatal: "unfunded" means the balance is empty (the reaper will tear down when
// the lease runs out — "processed until there is no more time left"), anything
// else retries after a per-record backoff.
//
// A lease that ALREADY LAPSED while the app kept serving is a different verb:
// the contract's renew() requires a live lease ("lease expired" revert), and
// the sweep refuses such records too ("already serving it here") — so without
// this branch a lapsed-but-running app live-locks as on-chain QUEUED forever,
// serving unbilled, while its reverting renew burns a queue slot every pass
// (observed fleet-wide 2026-07-19 after the renewal-starvation cascade:
// Running 27→16 while Queued grew). RE-CLAIM it instead — claim(id,
// enclaveId) re-acquires the lease in place, no re-provisioning, the app
// never blips. If another enclave won it meanwhile, the claim reverts and the
// audit's displacement handling takes over.
async function renewLeases() {
  // Unreachable: let stragglers lapse instead of paying to extend them. (A
  // hint-claim provisioning in the background when the watchdog tripped can
  // finish AFTER abandonClaims swept — this catches that record too; the
  // lease runs out within one quantum and the reaper tears it down.)
  if (_reach.tripped) return;
  for (const rec of deployments.values()) {
    if (!rec._onchain || rec.status !== "running" || rec._renewing) continue;
    if (rec._leaseUntil * 1000 - Date.now() > RENEW_MARGIN_SEC * 1000) continue;
    if ((rec._renewBackoffUntil || 0) > Date.now()) continue;   // a reverting tx must not re-fire every pass
    const lapsed = rec._leaseUntil * 1000 <= Date.now();
    rec._renewing = true;
    try {
      const sent = lapsed ? sendClaimTx("claim", [rec.id, _enclaveId]) : sendClaimTx("renew", [rec.id]);
      await sent;
      const rcpt = await sent.receipt;
      if (rcpt.status !== "success") throw new Error(`${lapsed ? "re-claim" : "renew"} tx reverted`);
      // the Renewed/Claimed event IS the new lease - a follow-up read can be
      // stale or rate-limited, and a missed update here renews AGAIN next
      // tick, burning an extra quantum of the user's money every cycle
      // (observed live)
      const until = leaseFromReceipt(rcpt, lapsed ? "Claimed" : "Renewed", rec.id);
      if (until == null) throw new Error(`${lapsed ? "re-claim" : "renew"} receipt carried no lease event`);
      // the renewal moved one quantum from balance into the lease; mirror that
      // locally so lease+balance (the reported time left) doesn't jump between
      // audit refreshes
      rec._balance6 = Math.max(0, (rec._balance6 || 0) - Math.round(Math.max(0, until - rec._leaseUntil) * rec.rate * 1e6));
      rec._leaseUntil = until;
      rec.remainingMs = rec._leaseUntil * 1000 - Date.now();
      rec._renewBackoffUntil = 0;
      rec.rateCapBlocked = null;      // time bought: whatever the cap was, it isn't blocking now
      console.log(`[claim] ${rec.id} lease ${lapsed ? "RE-CLAIMED (had lapsed)" : "renewed"} until ${new Date(rec._leaseUntil * 1000).toISOString()}`);
      saveStateSoon();
    } catch (e) {
      const msg = e.shortMessage || e.message || "";
      // The owner dropped the rate cap under what this deployment costs here:
      // the chain refuses to sell more time, and no retry will change that.
      // Serve out the lease they already paid for, tell them why on the
      // record, and stop re-sending a tx that can only revert. (Raising the
      // cap again clears rec.rateCapBlocked on the next audit pass, and the
      // sweep re-claims once the lease lapses.)
      if (/over rate cap/i.test(msg)) {
        rec.rateCapBlocked = `the owner's rate cap is below this enclave's price for these shares, so the lease cannot be `
          + `extended; the app stops when the current lease ends (${new Date(rec._leaseUntil * 1000).toISOString()}). `
          + `Raise the cap or move to a cheaper enclave.`;
        rec._renewBackoffUntil = Date.now() + 300_000;
        console.warn(`[claim] ${rec.id} renew refused by the ledger: over the owner's rate cap; not retrying this quantum`);
        saveStateSoon();
        continue;
      }
      // 60s, NOT minutes: the renewal window is finite and a transient RPC
      // failure must not eat the rest of it (a 5-min backoff once equalled
      // the entire pre-2026-07-19 window - one hiccup guaranteed the lapse)
      rec._renewBackoffUntil = Date.now() + 60_000;
      console.warn(`[claim] ${lapsed ? "re-claim" : "renew"} ${rec.id} failed (${msg}); `
                 + `lease ${lapsed ? "lapsed" : "expires"} ${new Date(rec._leaseUntil * 1000).toISOString()}; backing off 60s`);
    } finally { rec._renewing = false; }
  }
}

// Owner repointed a SERVING deployment at another catalog version (setAppRef)
// and/or re-bought its shares (setShares, ledger rev 6 — the two ride one
// multicall when a new version needs different resources): upgrade IN PLACE.
// Same lease, same balance — the artifact and/or the slice change: re-gate the
// new record exactly like a claim (catalog approval + minimum shares against
// the row's CURRENT shares, fail closed), prefetch the new wasm BEFORE
// stopping the old instance (downtime ≈ one relaunch), swap the held slice
// when the shares changed, then relaunch through the normal provisioning
// path. A refused or unreachable gate keeps the OLD version serving — an
// upgrade tx racing an approval flip must never leave the user dark; the
// refusal is logged on change, surfaced on the record (rec.versionChange, the
// console polls it), and retried every audit pass. EXCEPT when the shares
// changed and the refusal is structural (not transient): the billing already
// moved on-chain, so serving the old size would charge the user for capacity
// they don't get (or serve capacity the platform no longer sells) — evict
// instead: stop, release the slice AND the lease (tail refunded), and let an
// enclave that fits the new record claim it. Same eviction when the resized
// slice cannot be allocated here.
async function switchTenantVersion(rec, d) {
  const to = d.appRef;
  const resize = shareResizeVerdict(rec.status, rec._shares, d.gpuMilli, d.cpuMilli) === "resize";
  const evict = async (why) => {
    console.warn(`[claim] ${rec.id} resized shares cannot be served here: ${why}; releasing so a fitting enclave claims it`);
    try { await stopContainer(rec); } catch {}
    if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
    rec.status = "expired";          // CLAIM_TERMINAL: any enclave (this one included) may re-adopt at the new size
    rec.error = "share resize: " + why;
    delete rec.versionChange;
    proveAndRelease(rec, "share resize: " + why).catch(() => {});
    saveStateSoon();
  };
  const refuse = (why, transient) => {
    if (resize && !transient) return evict(why);
    if (rec.versionChange?.error !== why)
      console.warn(`[claim] ${rec.id} version change to ${to} ${transient ? "deferred" : "refused"}: ${why}`);
    rec.versionChange = { to, error: why };
    saveStateSoon();
  };
  // a resize to GPU work can never be served by a CPU-only enclave, whatever
  // the catalog says — no point consulting it. UNLESS the owner marked the
  // card optional: they said they would rather keep running on cores than
  // queue for hardware, and evicting them into the queue is the one thing
  // that flag exists to prevent.
  if (resize && Number(d.gpuMilli) > 0 && !IS_GPU) {
    let optional = false;
    try { optional = parseDepOptions(d.configCid, d.gpuMilli).gpuOptional === true; } catch { optional = false; }
    if (!optional) return evict("GPU work on a CPU-only enclave");
  }
  let g;
  try { g = await gateAppReference(to, { forPrivate: !d.isPublic }); }
  catch (e) { g = { error: { status: 503, msg: e.shortMessage || e.message } }; }
  if (g.error) return refuse(g.error.msg, g.error.status === 503);
  // the version must fit the shares the row NOW carries (bought at create, or
  // re-bought by the owner's latest resize), sized against where this tenant
  // is ACTUALLY running: a CPU-fallback tenant holds no card, so the new
  // version's weights would land in node RAM and the volume-inclusive floor is
  // the one it has to clear.
  const onGpu = Number(rec.resources?.gpuShare || 0) > 0;
  const mins = minSharesOf(g.min,
    { volGb: volumeGb(neededVolumes(d, g), await vmHealth().catch(() => null)) });
  const gpuShare = Number(d.gpuMilli) / 1000, cpuShare = Number(d.cpuMilli) / 1000;
  const needGpu = onGpu ? mins.gpuFloor : 0;
  const needCpu = mins.cpuShare;
  if (gpuShare < needGpu - 1e-9 || cpuShare < needCpu - 1e-9)
    return refuse(`the new version needs more than this deployment's shares on this hardware `
                + `(needs gpuShare ${round3(needGpu)} / cpuShare ${round3(needCpu)}`
                + (onGpu ? "" : ", serving on cores") + ")");
  let firewall;
  try { firewall = parseFirewall({ ports: g.ports ? String(g.ports).split(",") : [] }); }
  catch (e) { return refuse("the new version's port spec is not servable here: " + e.message); }
  // the fee snapshot is as immutable as the shares: a repoint at a version
  // asking MORE than the deployment snapshotted can never pay the publisher,
  // so it is refused the same way (the old version keeps serving)
  const feeWhy = await feeGate(rec.id, g);
  if (feeWhy) return refuse(feeWhy.why, feeWhy.transient);
  // fetch + verify + cache the new bytes while the old version keeps serving
  if (PROVISION_BACKEND === "vm" && /^ipfs:\/\//.test(g.wasmRef)) {
    try {
      const r = await vmReq("POST", "/prefetch", { image: g.wasmRef }, 300_000);
      if (r.status !== 200) throw new Error((r.body && (r.body.error || r.body.message)) || `HTTP ${r.status}`);
    } catch (e) { return refuse("could not fetch the new version's wasm: " + e.message, true); }
  }
  const versionChanged = (rec.image && rec.image.reference) !== g.ref;
  console.log(`[claim] ${rec.id} owner changed ${versionChanged && resize ? "version + shares" : resize ? "shares" : "version"}: `
            + `${rec.image && rec.image.reference} -> ${to} (${g.app.slug}:${g.app.version})`
            + `${resize ? ` gpuMilli ${rec._shares?.gpuMilli ?? "?"} -> ${Number(d.gpuMilli)}, cpuMilli ${rec._shares?.cpuMilli ?? "?"} -> ${Number(d.cpuMilli)}` : ""}; restarting in place`);
  // an unconfirmed stop must DEFER the switch: provisioning over a possibly
  // still-live instance doubles up on the slice (the old process keeps its
  // VRAM/RAM while routing follows the new record) - the old version keeps
  // serving and the next audit pass retries
  if (!(await stopContainer(rec)))
    return refuse("the old instance could not be verifiably stopped; it keeps serving until teardown succeeds", true);
  if (resize) {
    // swap the held slice for one at the row's new size. Synchronous release +
    // realloc (no await between): the freed capacity of the OLD slice counts
    // toward fitting the new one, which is what lets in-place grows work at
    // all. If the new size doesn't fit this box, evict — the lease hands back
    // refunded and a box with room (or a fresh capacity window here) claims it.
    const slice = gpuShare > 0 ? normalizeGpuReq(gpuShare, cpuShare) : normalizeCpuReq(cpuShare);
    if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
    const next = slice.cpu ? allocCpu(slice.cpuShare) : allocGpu(slice.vramGb, slice.computeShare, slice.cpuShare);
    if (!next) return evict(`no free capacity for gpuShare ${round3(slice.gpuShare || 0)} / cpuShare ${round3(slice.cpuShare)} here right now`);
    rec._gpu = next;
    rec.resources = slice.cpu
      ? { gpuShare: 0, cpuShare: slice.cpuShare }
      : { gpuShare: slice.gpuShare, cpuShare: slice.cpuShare, cardId: next.cardId };
    rec._gpuSpec = slice.cpu ? null : { cardId: next.cardId, cardUuid: gpuCards[next.cardId]?.uuid || null,
                                        vramCapGb: next.vramGb, computeShare: next.computeShare };
    rec._shares = { gpuMilli: Number(d.gpuMilli), cpuMilli: Number(d.cpuMilli) };
    rec.rate = Number(d.rate) / 1e6;   // the resize recalculated it on-chain
  }
  const httpFw = firewall.find((x) => x.startsWith("http:"));
  rec.image = { reference: g.ref };
  rec.app = g.app; rec.appWasm = g.wasmRef;
  // the deployer's config override rides the DEPLOYMENT (the on-chain
  // envelope), not the version — a version switch keeps it; only an
  // override-free deployment follows the new version's config. The envelope
  // was validated at claim; a stale-unparsable one degrades like adopt's.
  try {
    const f = overrideConfigFields(parseDepOptions(d.configCid, d.gpuMilli), g);
    rec.config = f.config;
    rec.appConfigCid = f.appConfigCid;
    if (f.configOverride) rec.configOverride = true; else delete rec.configOverride;
    rec._envelope = String(d.configCid || "");   // the envelope watch must not re-restart for what this switch just applied
  } catch { rec.config = g.config || ""; rec.appConfigCid = g.configCid || ""; }
  rec.firewall = firewall;
  rec.network.port = httpFw ? +httpFw.slice(5) : 8080;
  delete rec.portMap;              // the new version's spawn recomputes its own mapping
  rec._deaths = 0;                 // new code earns a fresh crash budget
  delete rec.versionChange;
  rec.status = "claimed";          // the provision path's input state
  if (await provisionTenant(rec)) {
    console.log(`[claim] ${rec.id} now serving ${g.app.slug}:${g.app.version} (${g.ref})`);
  } else {
    // same contract as every failed provision: keep the failed record as the
    // owner's evidence, hand the lease back refunded, back off THIS version
    // here (the backoff clears if the owner switches the version again)
    noteProvisionFailure(rec.id, to);
    proveAndRelease(rec, "version change provision failed").catch(() => {});
  }
  saveStateSoon();
}

// Apply an owner's on-chain envelope rewrite (setConfig) to a SERVING record -
// the audit calls this on envelopeEditVerdict's waf/restart/error verdicts.
// WAF swaps live (the /x gate reads rec.waf per request); a config-namespace
// change relaunches the app on the new ENCLAVE_CONFIG exactly like a version
// switch (same lease, balance and endpoint; app state is ephemeral by design).
// When the override is REMOVED the record falls back to the VERSION's
// approval-covered config, resolved fresh from the catalog - a transient
// catalog failure defers to the next pass rather than launching on the wrong
// config. rec.configChange mirrors versionChange: the owner's evidence when an
// edit can't apply (the console/CLI poll the record).
async function applyEnvelopeEdit(rec, d, verdict) {
  const cur = String(d.configCid || "");
  const refuse = (why) => {
    if (rec.configChange?.error !== why)
      console.warn(`[claim] ${rec.id} config change deferred/refused: ${why}`);
    rec.configChange = { error: why };
    saveStateSoon();
  };
  if (verdict === "error") {
    let why = "unparseable envelope";
    try { parseDepOptions(cur); } catch (e) { why = e.message; }
    return refuse(why);
  }
  const o = parseDepOptions(cur);                  // waf/restart: the verdict already parsed it clean
  if (o.waf) rec.waf = o.waf; else delete rec.waf;
  if (verdict === "waf") {
    rec._envelope = cur;
    delete rec.configChange;
    console.log(`[claim] ${rec.id} owner updated the waf envelope; applied live`);
    saveStateSoon();
    return;
  }
  let g = null;
  if (!("config" in o) && !("configCid" in o)) {
    // override removed: back to the version's own config — which on a rev-7
    // version means restoring its CID, not just its (manifest-only) inline field
    try { g = await gateAppReference(d.appRef, { forPrivate: !d.isPublic }); }
    catch (e) { g = { error: { msg: e.shortMessage || e.message } }; }
    if (g.error) return refuse("couldn't resolve the version's config to fall back to: " + g.error.msg);
  }
  const f = overrideConfigFields(o, g);
  console.log(`[claim] ${rec.id} owner changed the deployment config on-chain; restarting in place`);
  try { await stopContainer(rec); } catch {}
  rec.config = f.config;
  rec.appConfigCid = f.appConfigCid;
  if (f.configOverride) rec.configOverride = true; else delete rec.configOverride;
  rec._envelope = cur;
  delete rec.configChange;
  rec._deaths = 0;                 // an owner-initiated relaunch earns a fresh crash budget (the version-switch rule)
  rec.status = "claimed";          // the provision path's input state
  if (!(await provisionTenant(rec))) {
    noteProvisionFailure(rec.id, rec.image && rec.image.reference);
    proveAndRelease(rec, "config change provision failed").catch(() => {});
  }
  saveStateSoon();
}

// Split-brain guard + owner-stop watcher + crash recovery. The chain is the
// source of truth: if we no longer hold the lease, stop serving (the new runner
// is attested identically and app state is ephemeral by design); if the owner
// deactivated, tear down AND release so the tail refunds; if we crashed between
// claim and provision (status "claimed"), finish the job or hand it back.
// The address book can repoint `deployments` at a NEW contract while records
// claimed on the OLD one are still serving (an owner-side wipe/migration: the
// abandoned ledger's leases are void and its deployments invisible to the
// console). The audit below can't catch this — old ids are simply ABSENT from
// the new ledger, and absence there is deliberately read as an RPC anomaly
// (keep serving, the lease is prepaid). Records therefore carry the ledger
// they were claimed on (rec._ledger, stamped at adopt) and this sweep — run
// ahead of renewals on BOTH claim-loop clocks — tears down whatever a retired
// ledger left behind. Local teardown only: renew/release would target the NEW
// contract (unknown id -> revert), and the old one is dead by governance
// decision (see the addressbook.js trust note). Teardown order matches the
// audit's owner-stop branch: stop, release the slice, THEN the terminal flip
// (TERMINAL_STATUSES' hold-no-resources invariant).
let _ledgerMoveBusy = false;
async function ledgerMoveSweep() {
  if (_ledgerMoveBusy) return;
  _ledgerMoveBusy = true;
  try {
    for (const rec of [...deployments.values()]) {
      const verdict = ledgerMoveVerdict(rec, DEPLOYMENTS_ADDRESS);
      if (verdict === "stamp") { rec._ledger = DEPLOYMENTS_ADDRESS; saveStateSoon(); continue; }
      if (verdict !== "teardown") continue;
      console.log(`[claim] ${rec.id}: its ledger ${rec._ledger} was retired (book says ${DEPLOYMENTS_ADDRESS}) -> teardown`);
      try { await stopContainer(rec); } catch {}
      if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
      rec.status = "terminated";
      rec.error = `deployments ledger moved to ${DEPLOYMENTS_ADDRESS}; the lease on ${rec._ledger} is void`;
      saveStateSoon();
    }
  } finally { _ledgerMoveBusy = false; }
}

// Publisher-yank enforcement (see yankTeardownPlan for the policy). Runs on
// its own slow clock: yank state changes rarely, and each pass costs one
// getVersion read per DISTINCT version among live records, not per record.
// Teardown order matches the audit's owner-stop branch (stop, release the
// slice, terminal flip), then the lease is released so the unused tail
// refunds to the deployment's balance — which stays with the owner, who can
// setAppRef it to another version to revive the deployment (the claim gate
// refuses yanked versions, so nobody re-claims it as-is).
const YANK_SWEEP_SEC = parseInt(process.env.YANK_SWEEP_SEC || "300", 10);
let _yankBusy = false;
async function yankSweep() {
  if (_yankBusy || !APP_CATALOG_ADDRESS) return;
  _yankBusy = true;
  try {
    const live = [...deployments.values()].filter((r) =>
      r._onchain && (r.status === "running" || r.status === "claimed")
      && CATALOG_REF_RE.test(String(r.image && r.image.reference || "")));
    if (!live.length) return;
    const distinct = new Map();                    // ref string -> {appId, index}
    for (const rec of live) {
      const ref = String(rec.image.reference);
      const m = CATALOG_REF_RE.exec(ref);
      distinct.set(ref, { appId: m[1], index: Number(m[2]) });
    }
    const yankedRefs = new Set();
    for (const [ref, { appId, index }] of distinct) {
      try {
        const v = await chainClient.readContract({ address: getAddress(APP_CATALOG_ADDRESS),
          abi: CATALOG_ABI, functionName: "getVersion", args: [appId, BigInt(index)] });
        if (v && v.yanked === true) yankedRefs.add(ref);
      } catch (e) {
        console.warn(`[yank] getVersion(${ref}) failed (keeping it serving): ${e.shortMessage || e.message}`);
      }
    }
    if (!yankedRefs.size) return;
    const plan = new Set(yankTeardownPlan(
      live.map((r) => ({ id: r.id, status: r.status, ref: String(r.image.reference) })), yankedRefs));
    for (const rec of live) {
      if (!plan.has(rec.id)) continue;
      console.log(`[yank] ${rec.id}: its version was yanked by the publisher -> teardown + release`);
      try { await stopContainer(rec); } catch {}
      if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
      rec.status = "terminated";
      rec.error = "the app version this deployment runs was yanked by its publisher; "
                + "switch the deployment to another version (upgrade) to revive it";
      proveAndRelease(rec, "version yanked by publisher").catch(() => {});
      saveStateSoon();
    }
  } finally { _yankBusy = false; }
}

async function auditClaims(ledgerById) {
  const me = claimSigner().account.address.toLowerCase();
  for (const rec of [...deployments.values()]) {
    if (!rec._onchain || !["running", "claimed"].includes(rec.status)) continue;
    // an owner restart is mid-flight (stop -> claimed -> provision): the
    // claimed-branch below would double-provision it, and the death check
    // would count the deliberate stop as a crash. The restart route owns it.
    if (rec._restarting) continue;
    // the tick already paged the whole ledger once - one read serves the audit
    // AND the sweep (per-record re-reads were what blew the RPC rate budget)
    const d = ledgerById.get(rec.id.toLowerCase());
    if (!d) continue;                         // not in the page (RPC anomaly): keep serving, the lease is prepaid
    rec.paidUsdc = Number(d.spent6 + d.balance6);
    rec._balance6 = Number(d.balance6);          // funded-runtime display: balance beyond the current lease
    rec.rate = Number(d.rate) / 1e6;             // a setShares resize recalculates it on-chain; keep the mirror honest
    // transferDeployment (rev 11): rec.owner is otherwise a claim-time snapshot
    // (adopt), and it keys every owner gate on this box — the private data
    // path, logs, delete/restart, top-up. Mirror it here so a transfer moves
    // those gates to the new wallet within one audit pass instead of only on
    // the next re-claim. Same checksummed form adopt stamps.
    const chainOwner = getAddress(d.owner);
    if (rec.owner !== chainOwner) {
      console.log(`[claim] ${rec.id} owner transferred on-chain ${rec.owner} -> ${chainOwner}`);
      rec.owner = chainOwner;
      saveStateSoon();
    }
    // OWNERSHIP is keyed on the ENCLAVE ID (d.runner === _enclaveId), matching the
    // sweep (considerClaim) and the resume path — NOT on runnerOperator. On a
    // SHARED gas key several enclaves sign as the same operator EOA but have
    // distinct enclave ids; keying on the operator made each of them think it owned
    // the OTHER's live-leased deployments (split-brain double-serve/double-renew).
    // runnerOperator stays available below only as a lagging-RPC fallback signal,
    // never as the sole ownership test.
    const mine = d.runner === _enclaveId
              && Number(d.leaseUntil) * 1000 > Date.now();
    if (!d.active) {
      console.log(`[claim] ${rec.id} stopped by owner on-chain -> teardown + release`);
      try { await stopContainer(rec); } catch {}
      if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
      rec.status = "terminated";
      if (mine) proveAndRelease(rec, "owner setActive(false)").catch(() => {});
      saveStateSoon();
    } else if (!mine) {
      // Re-acquire-in-place gate for a LAPSED lease: prefer our own enclave id
      // (matches `mine`); keep the shared-operator match ONLY as an additional
      // fallback for a lagging RPC node that hasn't yet surfaced our fresh claim.
      // This is safe because the branch below is further gated on !leaseLive AND a
      // locally-running tenant, so it can't double-serve another enclave's LIVE lease.
      const opMine = d.runner === _enclaveId || (!!_enclaveId && (d.runnerOperator || "").toLowerCase() === me);
      const leaseLive = Number(d.leaseUntil) * 1000 > Date.now();
      // OUR lease that lapsed (a missed renew, or our own fresh claim not yet
      // visible on a lagging node) around a still-healthy tenant: re-acquire
      // in place. Tearing down a serving app to re-claim it seconds later
      // helps nobody and burns the user's lease.
      if (opMine && !leaseLive && rec.status === "running") {
        try {
          const sent = sendClaimTx("claim", [rec.id, _enclaveId]);
          await sent;
          const rcpt = await sent.receipt;
          const until = rcpt.status === "success" ? leaseFromReceipt(rcpt, "Claimed", rec.id) : null;
          if (until != null) {
            rec._leaseUntil = until;
            rec.remainingMs = rec._leaseUntil * 1000 - Date.now();
            rec._loseStrikes = 0;
            console.log(`[claim] ${rec.id} re-acquired our lapsed lease in place`);
            saveStateSoon();
            continue;
          }
        } catch (e) { /* someone else won it - fall through to the strikes */ }
      }
      // Public RPC nodes can serve STALE state right after a confirmation
      // (observed live: an audit pass read the pre-claim lease one minute
      // after our own claim and tore down a freshly provisioned tenant).
      // One read never kills a serving tenant: it takes two consecutive
      // audit passes agreeing that the lease is lost.
      rec._loseStrikes = (rec._loseStrikes || 0) + 1;
      if (rec._loseStrikes < 2) {
        console.log(`[claim] ${rec.id} lease looks lost (strike 1/2; chain says runner=${d.runnerOperator}, leaseUntil=${d.leaseUntil}); re-checking next pass`);
        continue;
      }
      console.log(`[claim] ${rec.id} lease lost -> teardown (chain says runner=${d.runnerOperator})`);
      try { await stopContainer(rec); } catch {}
      if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
      rec.status = "expired";                 // sweep may legitimately re-claim it later
      saveStateSoon();
    } else if (rec.status === "claimed") {    // crashed after claim, before provision
      rec._loseStrikes = 0;
      if (!(await provisionTenant(rec))) {
        // keep the "failed" record as the owner's evidence (see adopt())
        noteProvisionFailure(rec.id, rec.image && rec.image.reference);
        proveAndRelease(rec, "provision failed after crash recovery").catch(() => {});
        saveStateSoon();
      }
    } else {
      rec._loseStrikes = 0;                   // healthy pass: chain agrees the lease is ours
      // Owner repointed the deployment at another catalog version (setAppRef)
      // and/or re-bought its shares (setShares): restart it in place onto the
      // new record — lease and balance carry. A pre-resize-release record
      // first stamps what it is serving (from its quantized resources), so
      // the NEXT pass can tell an owner resize from the missing field.
      const sv = shareResizeVerdict(rec.status, rec._shares, d.gpuMilli, d.cpuMilli);
      if (sv === "stamp") {
        rec._shares = { gpuMilli: Math.round((rec.resources?.gpuShare || 0) * 1000),
                        cpuMilli: Math.round((rec.resources?.cpuShare || 0) * 1000) };
        saveStateSoon();
      }
      if (needsVersionSwitch(rec.status, rec.image && rec.image.reference, d.appRef) || sv === "resize") {
        await switchTenantVersion(rec, d);
        continue;
      }
      // a stale refusal must not outlive its cause (the owner switched back,
      // or the version's approval landed and the switch above succeeded)
      if (rec.versionChange) { delete rec.versionChange; saveStateSoon(); }
      // Owner rewrote the deployment-options envelope (setConfig): re-apply it
      // in place - the mutable half of the record the claim gate only reads
      // once. See envelopeEditVerdict/applyEnvelopeEdit for the rules.
      const ev = envelopeEditVerdict(rec, d.configCid);
      if (ev === "stamp") { rec._envelope = String(d.configCid || ""); saveStateSoon(); }
      else if (ev === "skip") { if (rec.configChange) { delete rec.configChange; saveStateSoon(); } }
      else {
        await applyEnvelopeEdit(rec, d, ev);
        if (ev === "restart") continue;          // just provisioned; the alive-check below would race it
      }
      // Crash recovery for a DIED app instance (fatal signal, OOM-kill): the
      // lease is ours and paid, the wasm is cached - relaunch. Bounded: an
      // app that keeps dying (crash-on-first-request) gets handed back after
      // 3 deaths instead of flapping forever on the owner's dime.
      if (rec.status === "running" && !(await instanceAlive(rec))) {
        rec._deaths = (rec._deaths || 0) + 1;
        if (rec._deaths > 3) {
          console.warn(`[claim] ${rec.id} app died ${rec._deaths}x; giving it back`);
          rec.status = "failed"; rec.error = rec.error || "app process kept dying";
          if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
          noteProvisionFailure(rec.id, rec.image && rec.image.reference);
          proveAndRelease(rec, "app kept dying").catch(() => {});
          saveStateSoon();
          continue;
        }
        console.warn(`[claim] ${rec.id} app instance died; relaunching (death ${rec._deaths}/3)`);
        try { if (rec._vmId) await vmReq("DELETE", `/vms/${encodeURIComponent(rec._vmId)}`, null, 15000).catch(() => {}); } catch {}
        rec.status = "claimed";               // provision path's input state
        if (!(await provisionTenant(rec))) {
          noteProvisionFailure(rec.id, rec.image && rec.image.reference);
          proveAndRelease(rec, "relaunch after app death failed").catch(() => {});
        }
        saveStateSoon();
      }
    }
  }
}

// One paged read of the whole ledger per tick, shared by the audit and the
// sweep - the per-stage reads it replaces were enough burst to trip the public
// RPC's per-IP rate limit, which killed the tail of every pass (the sweep)
// while the head (renewals) kept working: new deployments sat unclaimed for
// hours with all gauntlet conditions green (observed live 2026-07-05).
async function fetchLedger() {
  const all = [];
  for (let start = 0n; ; start += BigInt(CLAIM_PAGE)) {
    const page = await readLedgerContract("getPage", [start, BigInt(CLAIM_PAGE)]);
    all.push(...page);
    if (page.length < CLAIM_PAGE) break;
  }
  return all;
}

// Sweep the ledger for claimable work this enclave can actually serve: funded,
// unleased, fits our free capacity, passes the same catalog-approval gate as
// HTTP deploys (fail closed). Checked BEFORE claiming so we never burn a
// user's lease on something we can't run. Decline reasons are LOGGED on
// change: a silent decline loop is indistinguishable from a dead sweep from
// outside the enclave, and it took chain forensics to tell them apart once.
const _sweepDeclines = new Map();             // id -> last logged reason
let _resumeHoldLogged = "";                   // last logged hold reason (log on change)
async function claimSweep(ledger) {
  if (!(await backendHealthy())) return;      // don't take work we'd immediately fail
  const sweepOne = async (d, opts) => {
    let reason;
    try { reason = await considerClaim(d, opts); }
    catch (e) { reason = "error: " + (e.shortMessage || e.message); }   // one bad item must not end the pass
    if (!reason) { _sweepDeclines.delete(d.id); return null; }          // a claim was attempted
    if (reason !== _sweepDeclines.get(d.id) && !reason.startsWith("already serving")) {
      console.log(`[claim] sweep skips ${d.id}: ${reason}`);
      _sweepDeclines.set(d.id, reason);
    }
    return reason;
  };
  // Own live leases resume FIRST, and an unresumed one holds all new claims
  // this pass — its owner already paid for the slice (see sweepPartition).
  const { own, rest } = sweepPartition(ledger, _enclaveId, Date.now(), (id) => {
    const ex = deployments.get(id);
    return !!ex && !CLAIM_TERMINAL.has(ex.status);
  });
  let hold = null;
  for (const d of own) { const r = await sweepOne(d); if (r) hold ??= `${d.id}: ${r}`; }
  if (hold) {
    if (hold !== _resumeHoldLogged) {
      console.log(`[claim] holding new claims: own lease not yet resumed (${hold})`);
      _resumeHoldLogged = hold;
    }
    return;
  }
  _resumeHoldLogged = "";
  // Renewal liveness OUTRANKS new adoptions (2026-07-19: a 60-claim storm,
  // each awaited through claim tx + provisioning, starved the renew stage for
  // minutes at a time and funded leases lapsed fleet-wide in creation order).
  // Three guards, none touching the resume-first path above:
  //  - any running lease inside HALF the renew margin = the tx queue belongs
  //    to renewals; take no new work this pass,
  //  - new adoptions run BACKGROUND (the hint path's mode) so an IPFS fetch
  //    never blocks the next renew stage,
  //  - at most CLAIM_MAX_PER_SWEEP adoptions start per pass, so a large
  //    claimable backlog drains gently instead of stampeding the queue.
  const now = Date.now();
  const pressed = [...deployments.values()].some((r) =>
    r._onchain && r.status === "running" && r._leaseUntil * 1000 - now < RENEW_MARGIN_SEC * 500);
  if (pressed) return;
  let started = 0;
  for (const d of rest) {
    if (started >= CLAIM_MAX_PER_SWEEP) break;
    const r = await sweepOne(d, { background: true });
    if (!r) started++;
  }
}

// One deployment through the full claim gauntlet. Returns a reason string
// when we pass (shared by the sweep, which drops it, and POST /v1/claim-hint,
// which surfaces it to the deployer); null/undefined = a claim was attempted.
// `hinted` skips the CPU-first grace and the anti-stampede jitter: a hint is
// the deploying user asking THIS enclave to start their work now. `background`
// fires the claim without awaiting it (claim tx + provision can take tens of
// seconds - an IPFS fetch of a 100MB+ app is part of it - and a hint response
// must not hang that long; the deployer watches the ledger for the runner).
// true / false / null(unreachable). 404/503 = the relay has no record / no
// secrets feature — authoritatively nothing staged, false.
async function depHasSecrets(id){
  if (!SECRETS_API) return false;
  try {
    const r = await fetch(`${SECRETS_API}/v1/secrets/exists`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: String(id).toLowerCase() }), signal: AbortSignal.timeout(5000) });
    if (r.status === 404 || r.status === 503) return false;
    if (!r.ok) return null;
    const b = await r.json();
    return b.exists === true;
  } catch { return null; }
}

const volumesInConfig = (cfgStr) => {
  try { const c = JSON.parse(cfgStr || "{}"); if (Array.isArray(c.volumes)) return c.volumes.map(String); } catch {}
  return [];
};

// The model volumes the app will actually mount: the version's config, or the
// deployment's override where it has one. Pure — no I/O, so both the volume
// gate and the sizing math can ask.
function neededVolumes(d, g){
  let cfgStr = g.config || "";
  try {
    const o = parseDepOptions(d.configCid, d.gpuMilli);  // strict; considerClaim already accepted it
    // the override REPLACES the version's config — its volumes included. With
    // configCid the body sits at a CID this gate deliberately does NOT fetch
    // (placing a deployment must stay I/O-free), so the inline manifest is the
    // whole declaration: no manifest means the override declares NO volumes,
    // never the version's. Falling back there would gate the box on a config
    // that is not the one about to run.
    if (o && (o.config || o.configCid)) cfgStr = o.config ? JSON.stringify(o.config) : "{}";
  } catch { /* unreachable: parsed earlier in considerClaim */ }
  return volumesInConfig(cfgStr);
}

// GB of weights those volumes carry HERE, from the manager's own advertisement
// (the same rows /availability publishes). Volumes this box doesn't have count
// 0 — volumeGate refuses that deployment anyway, and guessing a size for a
// volume we cannot see would size the claim off a number nothing verified.
function volumeGb(need, h){
  if (!need.length) return 0;
  const bytes = new Map((Array.isArray(h && h.volumes) ? h.volumes : [])
    .filter((v) => v && v.name).map((v) => [String(v.name), Number(v.bytes) || 0]));
  return need.reduce((t, n) => t + (bytes.get(n) || 0), 0) / 1e9;
}

async function volumeGate(d, g, health){
  const need = neededVolumes(d, g);
  if (!need.length) return null;
  const h = health !== undefined ? health : await vmHealth().catch(() => null);
  if (!h) return "app manager unreachable (volume check)";
  const have = new Set((Array.isArray(h.volumes) ? h.volumes : []).map((x) => String((x && x.name) || x)));
  const missing = need.filter((n) => !have.has(n));
  return missing.length ? "app needs model volume(s) this enclave doesn't carry: " + missing.join(", ") : null;
}

async function considerClaim(d, { hinted = false, forced = false, background = false } = {}) {
  const ex = deployments.get(d.id);
  if (ex && !CLAIM_TERMINAL.has(ex.status)) return "already serving it here (status " + ex.status + ")";
  // Unreachable enclaves take no work — resumes included: re-provisioning an
  // app behind a dead front burns the owner's lease for service nobody gets.
  if (_reach.tripped) return "this enclave's advertised endpoint is gone from public DNS (unreachable); not claiming";
  if (!d.active) return "deployment is deactivated (owner setActive(false))";
  // A live lease held by OUR OWN enclaveId with no local record = a previous
  // life of this enclave (an update reboot wipes local state and cannot
  // release on-chain). RESUME it instead of leaving the app dark until the
  // lease lapses: we already own the lease, so no claim tx is needed (the
  // contract would refuse one anyway) - adopt + provision directly. This is
  // what makes enclave updates near-seamless for tenants.
  const leaseLive = Number(d.leaseUntil) * 1000 > Date.now();
  const resume = leaseLive && d.runner === _enclaveId;
  if (leaseLive && !resume) return "another enclave holds a live lease";
  // The record's own `rate` is the WORST case (its ceiling, until a host prices
  // it), so this is only a cheap pre-filter; rateCapRefusal below decides
  // exactly, at our price. A deployment we host for free has no funding to run
  // out of, so it must skip this or it would sit queued on the one box that
  // would run it for nothing.
  if (!resume && d.balance6 < d.rate && !(await hostChargeWaived(d)))
    return "out of funded time - fund it and retry";
  // configCid as a CID is retired: the field carries only the deployment-
  // options envelope (waf, config, …) — parsed strictly; anything this build
  // doesn't recognize is refused, never ignored: silently dropping an option
  // would serve traffic the owner believes is filtered, or run the app on a
  // config the owner believes was overridden.
  try { parseDepOptions(d.configCid, d.gpuMilli); }
  catch (e) { return "deployment options refused: " + e.message; }
  // Routing: the deployment bought two shares. GPU work (gpuMilli > 0)
  // runs ONLY on GPU enclaves and must fit a card AND the node's cpu pool.
  // CPU-only work runs on CPU enclaves immediately; a GPU enclave bids on
  // it only after CPU_CLAIM_GRACE_SEC (CPU enclaves get first claim) and
  // only out of LEFTOVER cpu pool.
  // Back off ids that just failed provisioning HERE: without this a broken
  // app (or a transient local fault) claims / fails / releases in a loop.
  // A FORCED hint overrides the timer: it is the owner's explicit Resume
  // click asking for one real retry NOW, and on a small fleet the
  // ex-runner's cooldown can be the only box able to take the id back - a
  // Resume that answers "backing off" for up to an hour reads as broken.
  // `hinted` alone is NOT enough: hints also arrive automatically (the
  // relay's post-funding nudge, the console's queued-row why-probe every
  // 30s), and when those overrode the cooldown a crash-looping app
  // flip-flopped running<->queued forever instead of resting in "failed".
  // Override the timer only, never the record: the entry stays, so if this
  // forced retry fails too noteProvisionFailure keeps escalating the SWEEP's
  // cooldown, and a forced-hint loop can't turn a doomed id into a fast
  // claim/fail/release cycle - each forced retry still walks the full
  // prefetch+provision gauntlet and the per-source hint rate limit bounds
  // how often anyone can ask (the race-loser gas here is the cents the
  // fan-out design already accepts).
  const pf = _provisionBackoff.get(d.id);
  if (provisionBackoffHolds(pf, Date.now(), d.appRef)) {
    // Prefetch-stage holds self-clear when the gateway serves the CID again.
    // until:0 (not delete) keeps the failure count: a CID that answers a
    // ranged byte but still fails the full verified prefetch re-notes and
    // climbs the same ladder instead of resetting to 5min forever.
    let cleared = false;
    if (!forced && prefetchProbeDue(pf, Date.now())) {
      pf.probedAt = Date.now();
      if (await gatewayServes(pf.cid)) {
        pf.until = 0; pf.probeClears = (pf.probeClears || 0) + 1; cleared = true;
        console.log(`[claim] ${d.id} prefetch backoff cleared: gateway serves ${pf.cid} again (after ${pf.n} failure${pf.n === 1 ? "" : "s"})`);
      }
    }
    if (!cleared) {
      if (!forced) return provisionDeclineReason(pf, Date.now());
      console.log(`[claim] ${d.id} provision backoff (failure ${pf.n}, ${Math.max(1, Math.round((pf.until - Date.now()) / 60000))}min left) overridden by owner resume`);
    }
  }
  const ev = _evacuated.get(d.id);
  if (ev && Date.now() < ev) return "evacuated from here for consolidation; leaving it for another enclave";
  // `gpu.optional` on the envelope: the deployment bought a card slice and
  // PREFERS one, but would rather run on cores than sit in the queue. A
  // CPU-only enclave may then take it, and the ledger charges only the cpu
  // half by itself - _hostRate multiplies gpuMilli by THIS enclave's posted
  // gpu price, which is zero on a box with no card. Honoured only when the
  // app's own version declares no hard GPU need: an envelope may waive the
  // OWNER's dial, never the publisher's stated requirement, or a version that
  // needs 128 GB of VRAM "falls back" to CPU and thrashes forever.
  let gpuOptional = false;
  try { gpuOptional = parseDepOptions(d.configCid, d.gpuMilli).gpuOptional === true; } catch { gpuOptional = false; }
  // The PUBLISHER's flag counts too, and on its own. A version that declares
  // its GPU axes desired-not-required says the app runs without a card - which
  // is exactly the licence to serve it here. Requiring the deployment envelope
  // as well stranded the obvious case: a soft-GPU app that bought a slice for a
  // GPU box could never move back to a CPU one ("GPU work on a CPU-only
  // enclave"), even though its publisher had already said cores are fine.
  const gpuShare = Number(d.gpuMilli) / 1000, cpuShare = Number(d.cpuMilli) / 1000;
  // The VERSION now comes before the routing decision, because what the app
  // needs is what decides whether this box can serve the card ask at all. It
  // also removes the second gateAppReference the CPU-only branch used to make.
  const g = await gateAppReference(d.appRef, { forPrivate: !d.isPublic });
  if (g.error) return "app not deployable: " + g.error.msg;   // unapproved/unknown record (or catalog unreachable: fail closed)
  // One /health serves the sizing, the GPU readiness check and volumeGate
  // below. Fetched only when one of them will actually want it, so a plain CPU
  // claim on a volume-less app costs no extra round trip. null = unreachable,
  // which every consumer already treats as fail-closed; undefined = never
  // asked, and volumeGate refetches if it turns out to need one.
  const wantVols = neededVolumes(d, g);
  const health = (wantVols.length || gpuShare > 0) ? await vmHealth().catch(() => null) : undefined;
  const mins = minSharesOf(g.min, { volGb: wantVols.length ? volumeGb(wantVols, health) : 0 });
  // The PUBLISHER's own declaration, not the volume-corrected figure: this is
  // "did the version state a card requirement", which is what decides whether
  // the OWNER's envelope flag is allowed to waive it.
  const declaresGpu = (g.min.vramMb || 0) > 0 || (g.min.gpuGflops || 0) > 0;
  let slice;
  let asCpuFallback = false;
  if (gpuShare > 0) {
    const r = gpuRouting(mins, { declaresGpu, envelopeOptional: gpuOptional });
    if (r.refusal) return r.refusal;
    if (!r.onGpu) {
      asCpuFallback = true;
      console.log(`[claim] ${d.id}: GPU requirement unmet here - ${r.unmet}; serving on cores instead`);
    }
  }
  if (gpuShare > 0 && !asCpuFallback) {
    // Don't claim GPU work the manager would 503: right after a boot the CUDA
    // readiness probe is still running, and a claim during that window burns
    // the user's lease on a doomed provision (observed live 2026-07-05:
    // claim -> 503 warming up -> failed release -> lease stranded 30 min).
    const h = health;
    if (!h) return "app manager unreachable";
    // A SHIELDED card has no CUDA readiness probe to pass and never will: there
    // is no local device for the manager to warm up, and nnProbe sits at "off"
    // forever on this box. Its readiness signal is the boot probe's verdict --
    // one real masked GEMM, exact, verified, with a lie rejected and the op
    // denylist enforced on the wire -- and the supervisor already refuses to
    // advertise the card at all without it (shieldedCapacity). Gating a shielded
    // claim on the CUDA probe instead would mean a box that advertises a card,
    // prices it, and then declines every deployment that tries to buy one.
    const shReady = shieldedCapacity();
    if (!shReady && h.nnProbe && h.nnProbe.state && h.nnProbe.state !== "ok")
      return "GPU interface not ready (CUDA readiness probe: " + h.nnProbe.state + ")";
    if (shReady && !shReady.endpoint)
      return "shielded GPU has no worker endpoint";
    if (_shieldedPool && h.shieldedPool !== true)
      return "shielded model-layer backend is not ready";
    slice = normalizeGpuReq(gpuShare, cpuShare);
    if (slice.vramGb > maxFreeVram() + 1e-9 || slice.cpuShare > maxFreeCpu() + 1e-9)
      return "no free capacity for those shares here right now";
    // The card's own count outranks the share arithmetic when it says LESS.
    // Every check above is a ledger: it knows what was handed out, never what
    // the device actually holds - and on 2026-08-18 ~104 GiB orphaned by an
    // in-place update was invisible to all of them, so a 51 GB spec passed
    // every fit check, loaded its weights and SIGABRT'd at first generation,
    // burning the owner's lease on a claim/crash loop. The manager reports
    // nvidia-smi memory.free in its capacity; a claim that cannot physically
    // fit is refused HERE, before a lease is burned. Absent field (older
    // manager, probe failure) skips the check - the ledgers above still hold.
    const devFreeGb = Number(h && h.capacity && h.capacity.vramDevFreeGb);
    if (Number.isFinite(devFreeGb) && slice.vramGb + CTX_OVERHEAD_GB > devFreeGb + 1e-9) {
      console.warn(`[claim] ${d.id}: needs ${slice.vramGb} GB VRAM but the device physically has `
        + `${devFreeGb} GB free (ledger free ${h.capacity.vramFreeGb ?? "?"} GB, divergence `
        + `${h.capacity.vramDivergenceGb ?? "?"} GB) - refusing the claim`);
      return "the card physically lacks the VRAM for those shares right now (device-measured; the share ledger disagrees, which is itself the incident signature)";
    }
  } else {
    // asCpuFallback lands here too: a GPU-dialled deployment served on cores
    // takes exactly the CPU path - the cpu slice it bought, no card, and the
    // manager sees gpuShare 0, so wasi-nn comes from the ggml CPU backend.
    if (IS_GPU && !hinted && !resume) {
      const claimableSince = Math.max(Number(d.createdAt), Number(d.leaseUntil));
      if (Date.now() < (claimableSince + CPU_CLAIM_GRACE_SEC) * 1000) return "cpu-first grace";
    }
    slice = normalizeCpuReq(cpuShare);
    if (slice.cpuShare > maxFreeCpu() + 1e-9) return "no free CPU capacity here right now";
  }
  // the app's catalog specs set its MINIMUM shares on our hardware, gating
  // claims exactly like HTTP deploys: a deployment that bought less than
  // the app needs is nobody's work item. The CPU floor is the same either way:
  // the weights are charged to the node, never to the share (see minSharesOf).
  const onGpu = gpuShare > 0 && !asCpuFallback;
  const needGpu = onGpu ? mins.gpuFloor : 0;
  const needCpu = mins.cpuShare;
  if (gpuShare < needGpu - 1e-9 || cpuShare < needCpu - 1e-9)
    return `below the app's minimum shares on this hardware (needs gpuShare ${round3(needGpu)} / cpuShare ${round3(needCpu)}`
         + (asCpuFallback ? ", serving on cores" : "") + ")";
  // WASIp3 is a RUNTIME capability, gated like the card and the volumes: the
  // version's config declares `wasi: "0.3"` (stamped from the binary by the
  // publish path) and a box whose wasmtime cannot serve p3 could only
  // claim-fail-release in a loop. The manager is the authority — it probed
  // its own binary (`p3` on /health) — and unreachable means fail closed,
  // exactly like the catalog. Undeclared versions are wasip2 (all of them,
  // before the key existed); the manager still re-classifies the actual
  // bytes at launch, so this gate is routing, not trust.
  if (wasiOfConfig(g.config) === "0.3") {
    const p3h = await vmHealth().catch(() => null);
    if (!p3h) return "app targets WASIp3 and the app manager cannot be asked (unreachable)";
    if (p3h.p3 !== true) return "app targets WASIp3 and this box's runtime does not serve it";
  }
  // A rev-7 CID config is gated the same way, and it is the case where failing
  // OPEN is worst: an older manager ignores the configCid field entirely and
  // launches the app with the inline field — which on this path is the routing
  // manifest, not the config. The app would come up "healthy" serving the
  // wrong configuration, with nothing in any log to say so. Refuse instead and
  // let a capable box take it.
  if (g.configCid) {
    const ch = await vmHealth().catch(() => null);
    if (!ch) return "app keeps its config at a CID and the app manager cannot be asked (unreachable)";
    if (ch.configCid !== true) return "app keeps its config at a CID and this box's manager cannot fetch it";
  }
  // Cooperative threads (🧵): gated exactly like p3 — the manager probed its
  // own engine (coopThreads on /health: the thread.new-indirect compile
  // probe), and a box that can't serve them could only claim-fail-release.
  if (threadsOfConfig(g.config)) {
    const th = await vmHealth().catch(() => null);
    if (!th) return "app uses cooperative threads and the app manager cannot be asked (unreachable)";
    if (th.coopThreads !== true) return "app uses cooperative threads and this box's runtime does not serve them";
  }
  // Shared-everything threads (⚡): gated exactly like coop threads, on the
  // manager's own thread.spawn-indirect compile probe (`set` on /health).
  if (setOfConfig(g.config)) {
    const sh = await vmHealth().catch(() => null);
    if (!sh) return "app uses shared-everything threads and the app manager cannot be asked (unreachable)";
    if (sh.set !== true) return "app uses shared-everything threads and this box's runtime does not serve them";
  }
  // wasm64 (memory64): gated exactly the same, on the manager's own flagless
  // memory64 compile probe (`mem64` on /health). A box whose engine cannot
  // parse a 64-bit memory could only claim-fail-release in a loop.
  if (mem64OfConfig(g.config)) {
    const mh = await vmHealth().catch(() => null);
    if (!mh) return "app is a wasm64 (memory64) module and the app manager cannot be asked (unreachable)";
    if (mh.mem64 !== true) return "app is a wasm64 (memory64) module and this box's runtime does not serve it";
  }
  // The firewall is the VERSION's declared ports — part of what approval
  // covered. The deployment's own ports field is ignored (create() still
  // carries it for the ledger's benefit; the record is the authority).
  let firewall;
  try { firewall = parseFirewall({ ports: g.ports ? String(g.ports).split(",") : [] }); }
  catch (e) { return "the version's port spec is not servable here: " + e.message; }
  // a paid app's fee snapshot must cover the version's ask (fail closed)
  const feeWhy = await feeGate(d.id, g);
  if (feeWhy) return feeWhy.why;
  // Price: what WE charge for these shares against what the owner agreed to
  // pay. Needs the version's fee (it rides on top of our price), so it runs
  // after the fee gate. A resume skips it — that lease is already bought and
  // paid for, and walking away from it would strand a tenant we owe service.
  if (!resume) {
    const priceWhy = await rateCapRefusal(d, g);
    if (priceWhy) return priceWhy;
  }
  // Model volumes are PER-BOX hardware, like the card: the config the app
  // will actually run with (the version's, or the deployment's override)
  // names what it must mount, and a box that doesn't carry a named volume
  // could only claim-fail-release in a loop. Fail closed here; the boxes
  // that carry the volume claim instead. (Mattered from the first
  // heterogeneous fleet: a metal box carries no Modelwrap volumes.)
  const volWhy = await volumeGate(d, g, health);
  if (volWhy) return volWhy;
  // Staged secrets are injected at launch via a FLEET-secret-derived auth this
  // box may not hold (SECRETS_CAPABLE=0: a metal enclave running its own
  // minted SECRET). The fetch fails SOFT — the app would launch WITHOUT its
  // env, silently broken — so a secret-bearing deployment is not our work
  // item: probe the relay and fail closed, exactly like volumes/fees/approval.
  if (!SECRETS_CAPABLE) {
    const has = await depHasSecrets(d.id);
    if (has !== false) return has === true
      ? "deployment has staged secrets and this enclave lacks the fleet secrets key"
      : "cannot verify the deployment has no staged secrets (relay probe unreachable)";
  }
  if (background) {
    tryClaim(d, g, firewall, slice, { hinted, resume })
      .catch(e => console.warn(`[claim] hinted claim ${d.id} failed: ${e.shortMessage || e.message}`));
    return null;
  }
  await tryClaim(d, g, firewall, slice, { hinted, resume });
  return null;
}

/* "Is this still open TO US?" — the last check before spending gas, and it has
   to be asked at OUR price, not at the record's.

   claimable(id) is `_open(id) && balance6 >= rate`, where `rate` is whatever
   the record currently stores. That is the right question for a stranger and
   the WRONG one for a deployment this box hosts for free: _hostRate returns 0
   when the enclave's payoutWallet is the deployment's owner, so we would charge
   nothing and the balance is irrelevant — but the stored rate is whatever the
   last host (or an import) left behind, and `0 >= 481` is false. The claim
   would have succeeded: claim() overwrites d.rate with OUR price before
   _burnLease ever runs. We just never sent it.

   It cost three free self-hosted apps a silent outage on 2026-08-11, after a
   ledger migration seeded their rate from their cap and turned a 0 into a 481.
   considerClaim already knew better (hostChargeWaived); this call didn't.

   claimableBy(id, enclaveId) asks the ledger the question we actually mean:
   it prices the record with rateFor(id, enclaveId) — our price — and checks the
   balance against THAT. Pre-rev-8 ledgers have no such function; they also have
   no per-enclave pricing, so claimable() is exactly right there. Reverts and
   unreachable RPCs fall back to it rather than blocking the claim. */
async function openForUs(id) {
  try {
    return await chainClient.readContract({ address: getAddress(DEPLOYMENTS_ADDRESS),
      abi: CLAIM_TX_ABI, functionName: "claimableBy", args: [id, _enclaveId] });
  } catch (e) {
    return await chainClient.readContract({ address: getAddress(DEPLOYMENTS_ADDRESS),
      abi: CLAIM_TX_ABI, functionName: "claimable", args: [id] });
  }
}

// Jitter de-syncs enclaves that saw the same queue state; the claimable()
// re-check catches a claim that landed during the wait without paying for a
// reverted tx. Losing the race anyway costs one reverted tx (cents on Base).
async function tryClaim(d, g, firewall, slice, { hinted = false, resume = false } = {}) {
  // Ahead of the prefetch, not after it: if the ledger wants a bond we cannot
  // post, there is no point pulling a 100MB image we may never launch. A RESUME
  // holds its lease already and the contract re-checks nothing on renew or
  // release, so winding an existing tenant down is never gated on the bond.
  if (!resume && !(await claimBondReady())) return;
  if (!hinted && !resume) await new Promise(r => setTimeout(r, Math.random() * 5000));
  // Fetch + verify + cache the app BEFORE burning a lease: the launch's own
  // fetch then hits the manager's local cache instead of racing a 100MB+
  // IPFS transfer against the spawn window, and an unfetchable CID costs the
  // user nothing (no claim ever happens).
  if (PROVISION_BACKEND === "vm" && /^ipfs:\/\//.test(g.wasmRef)) {
    try {
      const r = await vmReq("POST", "/prefetch", { image: g.wasmRef }, 300_000);
      if (r.status !== 200) throw new Error((r.body && (r.body.error || r.body.message)) || `HTTP ${r.status}`);
      if (r.body && r.body.seconds > 1) console.log(`[claim] ${d.id} prefetched ${r.body.bytes} bytes in ${r.body.seconds}s`);
    } catch (e) {
      const coolMs = noteProvisionFailure(d.id, d.appRef, "prefetch", e.message, g.wasmRef.slice("ipfs://".length));
      console.warn(`[claim] ${d.id} prefetch failed (${e.message}); not claiming, backing off ${Math.round(coolMs / 60000)}min`);
      return;
    }
  }
  if (resume) {
    // we already HOLD this lease (a previous life of this enclave claimed
    // it; the reboot wiped local state, not the chain) - no claim tx, just
    // pick the work back up
    console.log(`[claim] ${d.id} resuming our own live lease after a restart`);
    await adopt(d, g, firewall, slice);
    return;
  }
  const open = await openForUs(d.id);
  // Silence here reads as "queued forever": every gate above this point returns
  // a REASON that reaches the console's why-probe, but a false here used to
  // just return — no record, no backoff, no log — so the hint kept cheerfully
  // answering "claiming" while nothing ever happened. Say it out loud.
  if (!open) { console.log(`[claim] ${d.id} no longer open to us (claimableBy=false); skipping`); return; }
  let rcpt;
  try {
    const sent = sendClaimTx("claim", [d.id, _enclaveId]);
    await sent;
    rcpt = await sent.receipt;
  } catch (e) { console.log(`[claim] ${d.id} claim tx failed (${e.shortMessage || e.message})`); return; }
  if (rcpt.status !== "success") { console.log(`[claim] ${d.id} lost the race`); return; }
  // The receipt's Claimed event is the proof we won AND the new lease bounds;
  // re-reading the ledger here once handed a stale/rate-limited answer and the
  // loop walked away from its own paid lease (tenant dark for a full quantum).
  const until = leaseFromReceipt(rcpt, "Claimed", d.id);
  if (until == null) {   // success receipt without our event: should be impossible - refund rather than strand
    console.warn(`[claim] ${d.id} claim confirmed but no Claimed event found; releasing`);
    releaseLease(d.id, "claim receipt unreadable").catch(() => {});   // never provisioned: nothing to prove
    return;
  }
  await adopt({ ...d, leaseUntil: BigInt(until), runner: _enclaveId,
                runnerOperator: claimSigner().account.address }, g, firewall, slice);
}

// On-chain record -> local rec, then the SAME provisioning path as HTTP deploys.
// rec.id IS the on-chain id, so the data path (/x/:id, tcp bridge, udp address)
// and clients resolving id -> runner -> endpoint from chain state need no
// mapping. rec.owner is the on-chain owner address — SIWE tokens already carry
// an address, so owner-only routes (status, delete) work unchanged.
async function adopt(d, g, firewall, slice) {
  const left = deployments.get(d.id);
  if (left) {                                               // terminal leftover from an earlier lease/life
    if (left._gpu) {                                        // a leftover still holding its slice is restore/crash
      releaseGpu(left._gpu); left._gpu = null;              // drift (terminal flips release synchronously) - reclaim
      console.warn(`[claim] ${d.id}: leftover record still held its slice - released before re-adopt`);
    }
    deployments.delete(d.id);
  }
  const gpu = slice.cpu ? allocCpu(slice.cpuShare) : allocGpu(slice.vramGb, slice.computeShare, slice.cpuShare);
  if (!gpu) {                                                // capacity vanished since the sweep checked
    releaseLease(d.id, "capacity vanished").catch(() => {}); // hand it back with the lease refunded (never served: no proof)
    return;
  }
  // the version's declared http:N entry is the app port; the record decides
  // (create()'s appPort field, like its ports field, is not consulted)
  const httpFw = firewall.find((x) => x.startsWith("http:"));
  const appPort = httpFw ? +httpFw.slice(5) : 8080;
  const rec = {
    id: d.id, owner: getAddress(d.owner), status: "claimed", public: d.isPublic, firewall,
    // image.reference is the CATALOG RECORD (the deployment's identity — the
    // dashboard shows app.slug:app.version from it); the wasm CID in appWasm
    // is only the manager's fetch address
    image: { reference: g.ref }, command: [],
    app: g.app, appWasm: g.wasmRef, config: g.config || "",
    // rev-7: when the version keeps its config at a CID, `config` above is only
    // the routing manifest and this is where the real one lives (the manager
    // fetches and hash-checks it). Empty on every inline version.
    ...(g.configCid ? { appConfigCid: g.configCid } : {}),
    // deployer's envelope options: considerClaim validated this exact string
    // before any adopt path (claim or resume), so the catch is unreachable —
    // kept so a hypothetical stale record degrades to the version's config and
    // no waf, not a crash. An override REPLACES the version's config wherever
    // that lives (the spread lands after config:/appConfigCid: above on
    // purpose — overrideConfigFields rewrites BOTH); configOverride marks the
    // record so the owner can see their override is what's serving.
    ...(() => { try {
      const o = parseDepOptions(d.configCid, d.gpuMilli);
      const f = overrideConfigFields(o, g);
      return { ...(o.waf ? { waf: o.waf } : {}),
               ...(f.configOverride ? { config: f.config, appConfigCid: f.appConfigCid, configOverride: true } : {}) };
    } catch { return {}; } })(),
    // the two shares the deployment bought on-chain — _shares keeps the exact
    // ledger millis so the audit can tell an owner resize (setShares) from
    // the quantized slice actually held
    resources: slice.cpu
      ? { gpuShare: 0, cpuShare: slice.cpuShare }
      : { gpuShare: slice.gpuShare, cpuShare: slice.cpuShare, cardId: gpu.cardId },
    _shares: { gpuMilli: Number(d.gpuMilli), cpuMilli: Number(d.cpuMilli) },
    network: { port: appPort, protocol: "https", endpoint: `${_advertisedEndpoint}/x/${d.id}` },
    attestation: { available: true, vmTechnology: vmTech(), gpuTechnology: IS_GPU ? "nvidia-cc" : null, href: `/v1/deployments/${d.id}/attestation` },
    region: "tinfoil", createdAt: new Date(Number(d.createdAt) * 1000).toISOString(), startedAt: null,
    // the local clock only mirrors the CURRENT lease; the chain holds the rest
    remainingMs: Number(d.leaseUntil) * 1000 - Date.now(), consumedMs: 0,
    paused: false, pauseReason: null, _lastTickAt: Date.now(),
    rate: Number(d.rate) / 1e6, paidUsdc: Number(d.spent6 + d.balance6),
    // on a fresh claim this page read predates the claim tx, so it still counts
    // the quantum the claim just burned - the next audit pass (~CLAIM_POLL_SEC)
    // corrects it; the resume path is exact
    _balance6: Number(d.balance6),
    // which envelope string this instance runs on: the audit's envelope watch
    // (envelopeEditVerdict) compares it against the ledger to catch setConfig
    _envelope: String(d.configCid || ""),
    // which ledger this lease lives on: a book repoint retires the old
    // contract and ledgerMoveSweep tears down what it left behind
    _onchain: true, _ledger: DEPLOYMENTS_ADDRESS, _leaseUntil: Number(d.leaseUntil), _renewing: false,
    _gpu: gpu, _gpuSpec: gpu.cpu ? null : { cardId: gpu.cardId, cardUuid: gpuCards[gpu.cardId]?.uuid || null, vramCapGb: gpu.vramGb, computeShare: gpu.computeShare },
    _port: 0, _payTimer: null,
  };
  deployments.set(rec.id, rec); saveStateSoon();
  // Start the certificate NOW, in parallel with the launch below. On a move the
  // destination has never served this hostname, so it always needs a fresh one;
  // doing it here hides the whole issuance behind provisioning.
  acmeReconcileSoon();
  if (await provisionTenant(rec)) {
    // a real success retires the ladder: the next unrelated failure starts at 5min
    _provisionBackoff.delete(rec.id);
    console.log(`[claim] ${rec.id} adopted: app=${g.app.slug}:${g.app.version} (${g.ref}) gpuShare=${round3(slice.gpuShare || 0)} cpuShare=${round3(slice.cpuShare)} `
              + `lease until ${new Date(rec._leaseUntil * 1000).toISOString()}`);
  } else {
    // launch failed (bad wasm, manager 503, spawn timeout, ...): hand the
    // lease back refunded and back off this id here. KEEP the failed record -
    // provisionTenant stamped status "failed" + rec.error, and that record is
    // the owner's only evidence of WHY (the console polls it). "failed" is in
    // CLAIM_TERMINAL, so any enclave (this one included) may still re-adopt.
    const coolMs = noteProvisionFailure(rec.id, rec.image && rec.image.reference, "provision", rec.error);
    console.warn(`[claim] provision failed for ${rec.id} (${rec.error || "?"}); backing off ${Math.round(coolMs / 60000)}min here`);
    releaseLease(rec.id, "provision failed").catch(() => {});   // never served here: no proof, we earn nothing
  }
  saveStateSoon();
}

// Graceful shutdown: release every held lease (refunds the unused tail and
// reopens the queue immediately) with a hard 10s cap so a dead RPC can't hang
// the exit. GPU handles are freed so a restart doesn't re-reserve ghosts.
let _shutdownReleased = false;
async function releaseClaimsOnShutdown() {
  if (_shutdownReleased) return; _shutdownReleased = true;
  const mine = [...deployments.values()].filter(r => r._onchain && ["running", "claimed"].includes(r.status)
    // a record from a retired ledger has nothing to release: the tx would
    // target the CURRENT contract, where its id does not exist
    && (!r._ledger || r._ledger.toLowerCase() === DEPLOYMENTS_ADDRESS.toLowerCase()));
  if (!mine.length || !CLAIM_READY) return;
  console.log(`[claim] shutdown: releasing ${mine.length} lease(s)`);
  await Promise.race([
    Promise.allSettled(mine.map(async (rec) => {
      rec.status = "terminated";     // this enclave's instance dies with the CVM; the on-chain deployment stays claimable
      if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
      // Prove the partial period, THEN release — release clears the watermark,
      // so the other order silently donates this shift's tail. Inside the same
      // 10s cap as the releases: a clean shutdown must stay fast, and losing
      // the proof costs only us.
      await proveFinalPeriod(rec, "shutdown").catch(() => {});
      await sendClaimTx("release", [rec.id]);
    })),
    new Promise(r => setTimeout(r, 10_000)),
  ]);
  saveStateNow();
}

// ---- runner earnings (rev-7 ledgers): sweep credits to the payout wallet ----
// The ledger pays THIS enclave's operator EOA for lease time it holds
// (contracts/EnclaveDeployments.sol rev 7: renew/release advance the credit
// meter; the money sits escrowed on the contract until withdrawn). This tick
// sweeps the accrued balance to PAYOUT_ADDRESS once it clears a minimum, so
// gas never eats a meaningful slice. Off unless PAYOUT_ADDRESS is set: the
// hosted fleet points it at the platform cold wallet, a metal seller at their
// own wallet (metal/config.json payoutAddress) — the operator EOA that signs
// lives in the CVM either way, so only this configured address can be paid.
const PAYOUT_ADDRESS    = (process.env.PAYOUT_ADDRESS || "").trim();
const EARNINGS_MIN_USDC = parseFloat(process.env.EARNINGS_MIN_USDC || "5");        // withdraw when >= this many USDC
const EARNINGS_CHECK_SEC = parseInt(process.env.EARNINGS_CHECK_SEC || "3600", 10); // how often to look
const EARN_ABI = [
  { type: "function", name: "earned6", stateMutability: "view",
    inputs: [{ name: "operator", type: "address" }], outputs: [{ type: "uint256" }] },
  { type: "function", name: "withdrawEarnings", stateMutability: "nonpayable",
    inputs: [{ name: "to", type: "address" }], outputs: [] },
];
// ============================================================================
// proof of time (rev-9 ledgers) — earning the lease seconds we hold
// ============================================================================
// Through rev 8 a held lease paid by itself. From rev 9 it does not: the ledger
// credits min(now, leaseUntil, provenUntil), and provenUntil only moves when
// EnclaveProofOfTime accepts a CHECKPOINT signed by this CVM's in-enclave proof
// key (PROOF_ACCOUNT) and anchored to a recent block hash. If this loop stops,
// our income stops within one proof window — which is the entire point, and why
// it is written to be the most conservative loop in this file:
//
//   - we prove a deployment ONLY after confirming its app is actually up, and
//     "up" here means a real TCP connect to the port the tenant serves on, not
//     just "the manager still has a record" (a SIGFPE'd app once served
//     ECONNREFUSED for an hour while its lease kept being renewed — see
//     _refreshStatus in wasm/wasm_manager.py). A proof we cannot stand behind is
//     one we do not sign;
//   - we anchor to the PREVIOUS block, the newest hash that exists on-chain, so
//     the proof has the full ~256-block window to land;
//   - we prove every PROOF_INTERVAL_SEC, comfortably inside the contract's
//     window, so one failed tx never costs a second of income: the next
//     checkpoint's window still reaches back over the gap;
//   - and we batch through checkpointMany, which is tolerant by design — one
//     deployment whose lease just lapsed must not cost us the proofs for every
//     other tenant on this box.
const PROOF_INTERVAL_SEC = parseInt(process.env.PROOF_INTERVAL_SEC || "300", 10);   // 5 min, vs the contract's 15 min window
const PROOF_MAX_PER_TX   = parseInt(process.env.PROOF_MAX_PER_TX || "25", 10);      // keep one batch inside the block gas limit
const PROOF_PROBE_MS     = parseInt(process.env.PROOF_PROBE_MS || "2000", 10);      // per-app liveness probe timeout

const PROOF_ABI = [
  { type: "function", name: "checkpoint", stateMutability: "nonpayable", inputs: [
    { name: "id", type: "bytes32" }, { name: "enclaveId", type: "bytes32" }, { name: "upto", type: "uint64" },
    { name: "anchorBlock", type: "uint64" }, { name: "anchorHash", type: "bytes32" }, { name: "sig", type: "bytes" }],
    outputs: [] },
  { type: "function", name: "checkpointMany", stateMutability: "nonpayable", inputs: [
    { name: "cps", type: "tuple[]", components: [
      { name: "id", type: "bytes32" }, { name: "enclaveId", type: "bytes32" }, { name: "upto", type: "uint64" },
      { name: "anchorBlock", type: "uint64" }, { name: "anchorHash", type: "bytes32" }, { name: "sig", type: "bytes" }] }],
    outputs: [{ type: "bool[]" }] },
  { type: "function", name: "proofWindowSec", stateMutability: "view", inputs: [], outputs: [{ type: "uint64" }] },
  { type: "function", name: "unprovenSec", stateMutability: "view",
    inputs: [{ type: "bytes32" }], outputs: [{ type: "uint64" }] },
];
// EIP-712 typed data, mirroring EnclaveProofOfTime.PROOF_TYPEHASH exactly. The
// domain binds chainId + the prover's address, so a proof cannot be replayed on
// a fork or against a different prover.
const PROOF_712_TYPES = {
  ProofOfTime: [
    { name: "id", type: "bytes32" }, { name: "enclaveId", type: "bytes32" }, { name: "operator", type: "address" },
    { name: "upto", type: "uint64" }, { name: "anchorBlock", type: "uint64" }, { name: "anchorHash", type: "bytes32" },
  ],
};

const PROOF_REJECT_ABI = [{ type: "event", name: "CheckpointRejected", inputs: [
  { name: "id", type: "bytes32", indexed: true }, { name: "reason", type: "string" }] }];

// Proving needs everything claiming needs, PLUS a bound prover contract to prove
// to and an in-CVM key to prove with. Missing either is not an error on a rev-8
// ledger (nothing to prove) — but on a rev-9 one it means we serve for free, so
// startClaimLoop says so loudly.
const PROOF_READY = () => CLAIM_READY && !!PROOF_OF_TIME_ADDRESS && !!PROOF_ACCOUNT && !!_enclaveId;

let _proof = { at: 0, proved: 0, rejected: 0, lastError: null };

// Is this deployment's app REALLY serving right now? instanceAlive only asks the
// manager whether the process exists; this also opens a socket to the port the
// tenant's traffic goes to, which is the closest thing this supervisor can
// honestly assert about the app on a tenant's behalf. Undetermined (no forwarded
// port yet, raw-TCP-only app with nothing on the HTTP port) falls back to the
// manager's answer rather than refusing to pay ourselves for real work.
async function tenantServing(rec) {
  if (!(await instanceAlive(rec))) return false;
  const port = rec._vmHostPort;
  if (!port) return true;                                  // nothing to probe yet; the manager's word stands
  return await new Promise((resolve) => {
    const sock = net.connect({ host: "127.0.0.1", port }, () => { sock.destroy(); resolve(true); });
    sock.setTimeout(PROOF_PROBE_MS, () => { sock.destroy(); resolve(false); });
    sock.on("error", () => { sock.destroy(); resolve(false); });
  });
}

// Sign one checkpoint with the in-enclave proof key. `upto` is capped at the
// lease end: the contract clamps there anyway, and asking for more would just
// make our own logs lie about what we proved.
async function signCheckpoint(rec, upto, anchorBlock, anchorHash) {
  const sig = await PROOF_ACCOUNT.signTypedData({
    domain: { name: "EnclaveProofOfTime", version: "1", chainId: base.id,
              verifyingContract: getAddress(PROOF_OF_TIME_ADDRESS) },
    types: PROOF_712_TYPES,
    primaryType: "ProofOfTime",
    message: { id: rec.id, enclaveId: _enclaveId, operator: claimSigner().account.address,
               upto: BigInt(upto), anchorBlock: BigInt(anchorBlock), anchorHash },
  });
  return { id: rec.id, enclaveId: _enclaveId, upto: BigInt(upto),
           anchorBlock: BigInt(anchorBlock), anchorHash, sig };
}

// Build a checkpoint for every live lease whose app we can confirm is serving.
// Returns the batch plus the records it covers, so callers can report per-record.
async function buildCheckpoints(recs) {
  const head = await chainClient.getBlock({ blockTag: "latest" });
  // anchor on the PARENT: blockhash(latest) is 0 inside the very next block's
  // execution on some clients, and the parent is guaranteed to be in range
  const anchorBlock = Number(head.number) - 1;
  const anchorHash = head.parentHash;
  const now = Math.floor(Date.now() / 1000);
  const batch = [], covered = [];
  for (const rec of recs) {
    if (batch.length >= PROOF_MAX_PER_TX) break;
    if (!(await tenantServing(rec))) {
      console.warn(`[proof] ${rec.id} not serving — NOT signing a proof for it`);
      continue;
    }
    const upto = Math.min(now, rec._leaseUntil || now);
    if (upto <= 0) continue;
    batch.push(await signCheckpoint(rec, upto, anchorBlock, anchorHash));
    covered.push(rec);
  }
  return { batch, covered };
}

// Every live lease, oldest proof first, in as many batches as it takes.
//
// The ordering is not cosmetic and the loop is not optional. A box can serve
// well over PROOF_MAX_PER_TX tenants (the RAM-budget ceiling is ~60), and
// `deployments` iterates in INSERTION order — so a single capped batch off the
// front would prove the same first 25 every round and the remaining tenants
// would never be proven at all, earning this box nothing for them, silently,
// forever. Sorting by staleness makes coverage rotate on its own AND puts
// whoever is closest to losing income first; the loop then finishes the round.
async function proveAllLeases(recs, why) {
  const queue = [...recs].sort((a, b) => (a._provenAt || 0) - (b._provenAt || 0));
  let landed = 0, sent = 0;
  // one batch per PROOF_MAX_PER_TX, plus a hard stop so a pathological record
  // count can never turn one round into an unbounded tx storm
  const maxBatches = Math.ceil(queue.length / Math.max(1, PROOF_MAX_PER_TX)) + 1;
  let rest = queue;
  while (rest.length && sent < maxBatches) {
    const before = rest.length;
    landed += await proveLeases(rest, why);
    sent++;
    // drop whatever this batch covered (proveLeases stamps _provenAt) and keep
    // going; if a pass covers nothing (nothing serving), stop rather than spin
    rest = rest.filter(r => (r._provenAt || 0) < _proof.at);
    if (rest.length === before) break;
  }
  return landed;
}

// Post proofs for everything we are serving. `why` only shapes the log line.
async function proveLeases(recs, why) {
  if (!PROOF_READY() || !recs.length) return 0;
  const { batch, covered } = await buildCheckpoints(recs);
  if (!batch.length) return 0;
  const fn = batch.length === 1 ? "checkpoint" : "checkpointMany";
  const args = batch.length === 1
    ? [batch[0].id, batch[0].enclaveId, batch[0].upto, batch[0].anchorBlock, batch[0].anchorHash, batch[0].sig]
    : [batch];
  const sent = sendOperatorTx(PROOF_OF_TIME_ADDRESS, PROOF_ABI, fn, args);
  await sent;
  const rcpt = await sent.receipt;
  if (rcpt.status !== "success") throw new Error(`${fn} tx reverted`);
  // checkpointMany swallows per-item failures on purpose (see the contract):
  // surface them here so a seller sees WHY a proof was refused instead of a
  // silently short payout weeks later.
  const rejects = parseEventLogs({ abi: PROOF_REJECT_ABI, logs: rcpt.logs, eventName: "CheckpointRejected", strict: false });
  const refused = new Set();
  for (const r of rejects) {
    refused.add(String(r.args.id).toLowerCase());
    console.warn(`[proof] rejected ${r.args.id}: ${r.args.reason}`);
  }
  const landed = batch.length - refused.size;
  _proof.proved += landed; _proof.rejected += refused.size; _proof.lastError = null;
  // Stamp ONLY what actually landed. A rejected proof left marked as proven
  // would report a provenUntil to the tenant that is not on-chain, and would
  // make the rotation in proveAllLeases treat this record as done — so the
  // retry would wait a whole round instead of happening in this one.
  const byId = new Map(batch.map(b => [String(b.id).toLowerCase(), b]));
  for (const rec of covered) {
    const key = String(rec.id).toLowerCase();
    if (refused.has(key)) continue;
    rec._provenUntil = Number(byId.get(key)?.upto || 0);
    rec._provenAt = Date.now();
  }
  console.log(`[proof] ${landed}/${batch.length} checkpoint(s) landed (${why})`);
  saveStateSoon();
  return landed;
}

// The steady clock. Everything we hold a lease on, every PROOF_INTERVAL_SEC.
async function proofTick() {
  if (!PROOF_READY()) return;
  if (Date.now() - _proof.at < PROOF_INTERVAL_SEC * 1000) return;
  _proof.at = Date.now();
  const mine = [...deployments.values()].filter(r =>
    r._onchain && r.status === "running" && r._leaseUntil * 1000 > Date.now()
    && (!r._ledger || r._ledger.toLowerCase() === DEPLOYMENTS_ADDRESS.toLowerCase()));
  if (!mine.length) return;
  try { await proveAllLeases(mine, "steady"); }
  catch (e) {
    // Not fatal and not retried harder: the next tick's window reaches back
    // over this gap, so a transient RPC failure costs nothing as long as it
    // does not persist past the contract's proofWindowSec.
    _proof.lastError = e.shortMessage || e.message;
    console.warn(`[proof] checkpoint round failed (${_proof.lastError}); the next round covers the gap`);
  }
}

// A deployment is ending HERE and now: crash, owner stop, eviction, drain,
// shutdown. Prove the fraction of the period we actually served BEFORE the
// lease goes back, because release() clears the watermark and a proof after it
// has no lease left to settle against. This is what makes a partial hour pay.
//
// Best-effort by design: if it fails we still release (the tenant getting their
// unserved tail back matters more than our last few minutes of income), and the
// only cost is ours.
async function proveFinalPeriod(rec, why) {
  if (!PROOF_READY() || !rec._onchain) return;
  if (rec._ledger && rec._ledger.toLowerCase() !== DEPLOYMENTS_ADDRESS.toLowerCase()) return;
  if (!rec._leaseUntil) return;
  // NEVER sign a proof for a record that never ran here. startedAt is stamped
  // only after spawnContainer succeeds (see provisionTenant), so this is the
  // honest line between "we served and are settling up" and "we claimed and
  // never provisioned" — the latter releases with nothing proven, which is
  // exactly right: the tenant gets the whole lease back and we earn nothing.
  if (!rec.startedAt) return;
  try {
    // Deliberately NOT gated on tenantServing: the app is already gone by the
    // time most teardown paths reach here. What we are proving is the time up
    // to now, which the steady loop was probing all along — and the contract
    // still caps it at one window past our last checkpoint, so this can only
    // settle service the loop was already vouching for.
    const head = await chainClient.getBlock({ blockTag: "latest" });
    const upto = Math.min(Math.floor(Date.now() / 1000), rec._leaseUntil);
    const cp = await signCheckpoint(rec, upto, Number(head.number) - 1, head.parentHash);
    const sent = sendOperatorTx(PROOF_OF_TIME_ADDRESS, PROOF_ABI, "checkpoint",
      [cp.id, cp.enclaveId, cp.upto, cp.anchorBlock, cp.anchorHash, cp.sig]);
    await sent;
    const rcpt = await sent.receipt;
    if (rcpt.status !== "success") throw new Error("final checkpoint reverted");
    console.log(`[proof] ${rec.id} final period settled through ${new Date(upto * 1000).toISOString()} (${why})`);
  } catch (e) {
    console.warn(`[proof] ${rec.id} final checkpoint failed (${why}): ${e.shortMessage || e.message}; `
               + `the unproven tail of this period is forfeit`);
  }
}

// Prove, then release, in that order — the one ordering the ledger cares about.
async function proveAndRelease(rec, why) {
  await proveFinalPeriod(rec, why);
  return releaseLease(rec.id, why);
}

let _earn = { checkedAt: 0, earned6: null, withdrawnTotal6: 0n };
async function payoutTick() {
  if (!PAYOUT_ADDRESS || !CLAIM_READY) return;
  if (Date.now() - _earn.checkedAt < EARNINGS_CHECK_SEC * 1000) return;
  _earn.checkedAt = Date.now();
  if ((await depsAbi()).rev < 7) return;                 // pre-payout ledger: nothing to sweep
  // Settle the hour before reading it. Two things the steady loops don't cover:
  //   - a checkpoint proves service, but the ledger only CREDITS it on a call
  //     that runs the meter, and a lease that lapsed without being re-claimed or
  //     released never gets one. settle() is the permissionless collector for
  //     exactly that final quantum (contracts/EnclaveDeployments.sol settle()),
  //     and nothing in this file used to call it — so a dead deployment's last
  //     proven period sat uncredited until some other enclave claimed the id.
  //   - proving one more time right here means the hour we are about to withdraw
  //     includes the minutes since the last steady round, rather than trailing
  //     it by up to PROOF_INTERVAL_SEC.
  await settleLapsedLeases();
  const earned = await chainClient.readContract({ address: getAddress(DEPLOYMENTS_ADDRESS),
    abi: EARN_ABI, functionName: "earned6", args: [claimSigner().account.address] });
  _earn.earned6 = earned;
  if (Number(earned) / 1e6 < EARNINGS_MIN_USDC) return;
  const to = getAddress(PAYOUT_ADDRESS);                 // throws on a mistyped address before any tx
  console.log(`[earn] withdrawing ${(Number(earned) / 1e6).toFixed(2)} USDC of runner earnings -> ${to}`);
  const rcpt = await sendOperatorTx(DEPLOYMENTS_ADDRESS, EARN_ABI, "withdrawEarnings", [to]).receipt;
  if (rcpt.status !== "success") { console.warn(`[earn] withdrawEarnings reverted`); return; }
  _earn.withdrawnTotal6 += BigInt(earned);
  _earn.earned6 = 0n;
  console.log(`[earn] runner earnings withdrawn (${(Number(_earn.withdrawnTotal6) / 1e6).toFixed(2)} USDC lifetime)`);
}

// Run the ledger's meter over leases that ended while we held them, so the last
// proven period is credited within the hour it happened instead of waiting for
// another enclave to claim the id. Permissionless and idempotent on-chain — it
// only ever moves money already owed from escrow into our balance.
async function settleLapsedLeases() {
  const stale = [...deployments.values()].filter(r =>
    r._onchain && r._leaseUntil && r._leaseUntil * 1000 <= Date.now()
    && ["expired", "failed", "terminated"].includes(r.status)
    && !r._settled
    && (!r._ledger || r._ledger.toLowerCase() === DEPLOYMENTS_ADDRESS.toLowerCase()));
  for (const rec of stale.slice(0, 10)) {                // bounded: this rides the hourly tick
    try {
      const sent = sendOperatorTx(DEPLOYMENTS_ADDRESS, SETTLE_ABI, "settle", [rec.id]);
      await sent;
      if ((await sent.receipt).status !== "success") throw new Error("settle reverted");
      rec._settled = true;                               // once is enough; the meter is monotonic
      console.log(`[earn] settled the final period of ${rec.id}`);
      saveStateSoon();
    } catch (e) {
      console.warn(`[earn] settle ${rec.id} failed: ${e.shortMessage || e.message}`);
    }
  }
}
const SETTLE_ABI = [{ type: "function", name: "settle", stateMutability: "nonpayable",
  inputs: [{ name: "id", type: "bytes32" }], outputs: [] }];

let _claimBusy = false;
function startClaimLoop() {
  if (!CLAIM_ENABLED) return;
  if (!CLAIM_READY) {
    console.warn("[claim] CLAIM_ENABLED but not claimable: needs DEPLOYMENTS_ADDRESS, registry advertising "
               + "(REGISTRY_ENABLED/ADDRESS/PRIVATE_KEY/ENCLAVE_REPO), and PROVISION_BACKEND=vm — not claiming");
    return;
  }
  // Each stage runs in its own catch: renewals, the audit and the sweep are
  // independent duties, and a throw in an early stage starving the later ones
  // is exactly how the sweep silently died for hours (rate-limited RPC call
  // in the audit -> shared catch -> claimSweep never ran, renews fine).
  const stage = async (name, fn) => {
    try { await fn(); }
    catch (e) { console.warn(`[claim] ${name} failed: ${e.shortMessage || e.message}`); }
  };
  // Renewals ride their OWN clock, decoupled from pass duration: a pass can
  // stall for many minutes inside audit/sweep (dozens of queued ids × catalog
  // + ledger reads over a rate-limited public RPC), and a renewal window that
  // opens mid-pass would otherwise wait the pass out (2026-07-19: ~10-15 min
  // passes vs a 5-min window — freshly-claimed leases lapsed on a 30-min
  // treadmill, 3 renewals landing in 45 min). The in-pass stage below stays
  // for back-to-back coverage; both entries share the per-record _renewing /
  // _renewBackoffUntil guards, so a double-fire is safe and cheap.
  const rt = setInterval(async () => {
    if (!_enclaveId) return;
    await stage("ledger-move", ledgerMoveSweep);  // a repointed book voids leases: never renew on a retired ledger
    await stage("renew", renewLeases);
    // Proofs ride the renewal clock, not the slower full pass: from rev 9 a
    // lapse here costs real income (unproven time is never paid), and the full
    // pass can stall for minutes inside audit/sweep — the same reason renewals
    // were moved off it. Self-throttled to PROOF_INTERVAL_SEC.
    await stage("proof", proofTick);
    await stage("earnings", payoutTick);          // self-throttled to EARNINGS_CHECK_SEC; no-op unless PAYOUT_ADDRESS
  }, 60_000);
  if (rt.unref) rt.unref();
  // Publisher-yank enforcement on its own slow clock (yank state changes
  // rarely; one catalog read per distinct running version per pass).
  const yt = setInterval(async () => {
    if (!_enclaveId) return;
    await stage("yank", yankSweep);
  }, YANK_SWEEP_SEC * 1000);
  if (yt.unref) yt.unref();
  const t = setInterval(async () => {
    if (_claimBusy || !_enclaveId) return;   // not advertised yet, or a slow pass is still running
    _claimBusy = true;
    try {
      await stage("ledger-move", ledgerMoveSweep);
      await stage("reach", reachTick);       // first: renew/sweep below consult the verdict
      await stage("renew", renewLeases);
      let ledger = null;
      try { ledger = await fetchLedger(); }
      catch (e) { console.warn(`[claim] ledger read failed: ${e.shortMessage || e.message}`); }
      if (ledger) {
        const byId = new Map(ledger.map(d => [String(d.id).toLowerCase(), d]));
        await stage("audit", () => auditClaims(byId));
        await stage("sweep", () => claimSweep(ledger));
      }
    } finally { _claimBusy = false; }
  }, CLAIM_POLL_SEC * 1000);
  if (t.unref) t.unref();
  console.log(`[claim] loop on: ${DEPLOYMENTS_ADDRESS} every ${CLAIM_POLL_SEC}s (renew margin ${RENEW_MARGIN_SEC}s)`);
  // Say plainly whether we can earn. On a rev-9 ledger past its cutover, a box
  // that cannot prove is a box working for free — that must never be something
  // an operator discovers from a payout report.
  (async () => {
    try {
      const { rev } = await depsAbi();
      if (rev < 9) return console.log(`[proof] ledger rev ${rev}: proof of time not required (held time still pays)`);
      if (!PROOF_OF_TIME_ADDRESS)
        return console.warn("[proof] rev-9 ledger but NO prover address (address-book key `proofOfTime`, or "
                          + "PROOF_OF_TIME_ADDRESS): we cannot prove service and will earn NOTHING after the cutover");
      const required = await chainClient.readContract({ address: getAddress(DEPLOYMENTS_ADDRESS),
        abi: [{ type: "function", name: "proofRequired", stateMutability: "view", inputs: [], outputs: [{ type: "bool" }] }],
        functionName: "proofRequired" });
      console.log(`[proof] on: ${PROOF_OF_TIME_ADDRESS} every ${PROOF_INTERVAL_SEC}s, signer ${PROOF_ACCOUNT.address}`
        + (required ? " — proven-time metering is LIVE" : " — still in grace (held time pays for now)"));
    } catch (e) { console.warn(`[proof] readiness check failed: ${e.shortMessage || e.message}`); }
  })();
}

// Funding instructions for a claimed deployment: top-ups go to the ledger
// contract (credited on-chain), NOT to EnclavePay — same EIP-3009 shape, different
// receiver, and the nonce binds to the on-chain id.
function onchainPaymentInstructions(rec) {
  return {
    chainId: CHAIN_ID, asset: "USDC", assets: ["USDC", "ETH"], usdc: USDC_ADDRESS,
    contract: DEPLOYMENTS_ADDRESS || null,
    deploymentRef: rec.id,                    // the bytes32 id to pass to fundWithAuthorization() / fundEth()
    ratePerSecondUsdc: (rec.rate || 0).toFixed(7),
    method: "fundWithAuthorization(bytes32 id, address from, uint256 value, uint256 validAfter, uint256 validBefore, bytes32 nonce, bytes signature)",
    payEthMethod: "fundEth(bytes32 id) payable",
    usdcDomain: _usdcDomain,
    ethUsd: _ethUsd.price8 ? (Number(_ethUsd.price8) / 1e8).toFixed(2) : null,
    note: "On-chain deployment: fund EnclaveDeployments directly. USDC (EIP-3009, no approve): sign a USDC "
        + "ReceiveWithAuthorization (EIP-712, to = the EnclaveDeployments contract, nonce = first 16 bytes of the "
        + "deployment id + 16 random bytes), then anyone submits fundWithAuthorization; amount(6dp)/rate = seconds. "
        + "ETH: fundEth(id) with msg.value; credited on-chain at the live Chainlink ETH/USD rate.",
  };
}

server.listen(PORT, () => console.log(`enclave supervisor on :${PORT} · ${IS_GPU
  ? `${GPU_COUNT}×GPU @ ${CARD_VRAM_GB}GB (arbitrary split)`
  : `CPU-only enclave (${NODE_VCPUS} vCPU / ${NODE_RAM_GB}GB, node-share split)`}`));
// warm the CPU-TEE detection (shim loopback) so the first deployment record
// created after boot already reports the real silicon, not null
fetchEnclaveRad().then(() => console.log(`[attest] CPU TEE detected: ${vmTech()}`)).catch(() => {});
if (egress) { egress.start(); console.log(`[egress] dedicated-IP egress on (SOCKS 127.0.0.1:${EGRESS_SOCKS_PORT}); awaiting relay control channel`); }

// advertise this enclave on-chain (opt-in, non-blocking, never fatal)
// If the origin is pinned (PUBLIC_URL), advertise eagerly at boot; otherwise
// discover our public hostname from the shim's loopback TLS cert and register
// eagerly, with the first-external-request middleware above as the fallback.
if (PUBLIC_URL) registerOnChain(PUBLIC_URL);
else if (REGISTRY_READY) advertiseFromShimCert();

// pay-per-deploy: watch the forwarder for payments + fair-billing ticker (drains
// funded time only while healthy; freezes through outages; reaps at -grace)
startPaymentWatcher();
startBillingTicker();
startPoolReconciler();

// ...and the same doctrine one layer down: tear down backend instances no
// record owns, so an unconfirmed stop can never strand a tenant's slice (and
// its resident model weights) against a deployment that no longer exists.
startInstanceReconciler();

// portable deployments: claim/renew/release on-chain leases (opt-in; see
// contracts/DEPLOYMENTS.md). Requires registry advertising + DEPLOYMENTS_ADDRESS.
startClaimLoop();

// in-enclave ACME: issue + renew per-app browser certs for <label>.APP_CERT_DOMAIN
// (opt-in; a warning-then-no-op unless EAB + domain + DNS API are all configured).
startAcme();

// custom domains: pull each running deployment's verified hostnames, report
// issuance back, and nudge ACME when the list changes. Runs even with ACME off
// — the list also feeds the guest's ENCLAVE_HOSTS and the bridge's Host check.
// One minute is fast enough that an attach reaches a live app while the
// customer is still looking at the dashboard, and slow enough to be free.
if (DOMAINS_API) {
  const dt = setInterval(() => refreshCustomDomains().catch((e) => console.warn("[domains] refresh:", e.message)), 60_000);
  if (dt.unref) dt.unref();
  refreshCustomDomains().catch(() => {});
  console.log(`[domains] custom hostnames via ${DOMAINS_API}`);
}
