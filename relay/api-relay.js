// Enclave API relay — discovery + placement front door for the fleet. UNTRUSTED
// as a router (it can misroute, not impersonate: enclaves are attested on
// their own origins), but on the /v1 gateway path it IS a TLS terminator and
// sees control-plane traffic — accepted trade for giving browsers one origin.
//
// It reads EnclaveRegistry on Base for live enclaves (slow-moving truth: who
// exists), polls each one's public /availability (fast-moving truth: free
// capacity), and routes each request by what it IS. A deployment lives on ONE
// enclave, sessions are stateless JWTs (ES256, signed by each enclave's
// in-enclave key — a login is currently pinned to the enclave that issued it;
// see docs/session-auth.md), and only CREATION is a placement decision:
//
//   POST /v1/deployments        -> pick() by the body's resources.{gpuShare,cpuShare}
//                                  (CPU-only work -> CPU enclaves first; GPU work
//                                  -> a GPU enclave with both pools free)
//   GET  /v1/deployments        -> fan out to every live enclave, merge the lists,
//                                  then MERGE THE LEDGER: every EnclaveDeployments
//                                  record the wallet owns appears, hosted or not
//                                  (queued/stopped/unfunded work is real work) —
//                                  this endpoint answers even with ZERO enclaves
//   GET  /v1/deployments/:id    -> the owning enclave when one is live, else the
//                                  ledger record (same zero-enclave guarantee)
//   /v1/deployments/:id/*, /x/:id* and app subdomains
//                               -> the enclave that OWNS the deployment (probed
//                                  once, cached)
//   /availability               -> FLEET aggregate (best card slice + best node
//                                  pool across enclaves; what deploy dials want)
//   /v1/auth/*, everything else -> one sticky enclave (nonces are per-enclave
//                                  state; GPU enclave preferred, it serves the
//                                  full API surface)
//   GET /route                  -> JSON answer { endpoint, repo } for clients
//                                  that want to hit the enclave directly
//
// Config (env):
//   REGISTRY_ADDRESS   required*   EnclaveRegistry on Base (chain 8453)
//   BASE_RPC           optional    RPC url (default https://mainnet.base.org)
//   ENCLAVES           required*   *instead of the registry: static comma list
//                                  of enclave origins (pilot / local dev)
//   API_RELAY_PORT     optional    listen port (default 8100)
//   API_RELAY_BIND     optional    bind address. DEFAULT = all interfaces (kept
//                                  so a directly-exposed relay isn't broken); set
//                                  API_RELAY_BIND=127.0.0.1 whenever a local Caddy
//                                  fronts :8100 (the production `nan` box does) so
//                                  the port is never reachable except via the proxy.
//   TRUSTED_PROXY      optional    "1"/on (default) trusts Caddy's x-forwarded-host
//                                  /x-forwarded-for; set 0/off/none when the relay
//                                  is directly internet-exposed so clients can't
//                                  spoof routing/source via those headers.
//   INTERNAL_TOKEN     optional    shared secret for /internal/* (the on-demand
//                                  TLS ask). Unset, those routes admit a
//                                  loopback caller that carries NO forwarding
//                                  headers — which is Caddy's own `ask` request
//                                  but not anything Caddy proxies. Set it (plus
//                                  `header_up X-Internal-Token` on the ask
//                                  handler in the Caddyfile) to replace that
//                                  heuristic with a real credential.
//   TRUSTED_OPERATORS  optional    comma-separated lowercased EnclaveRegistry
//                                  operator addresses; when set, on-chain discovery
//                                  is filtered to these (closes B1/B2/B3). Unset =
//                                  follow every registered enclave (+ loud warning).
//   CORS_ORIGINS       optional    comma-separated allowed browser origins
//                                  (default https://enclave.host,https://www.enclave.host)
//   SSO_SIGNER_KEY     optional    secp256k1 private key (0x-hex) for minting
//                                  Sign-in-with-Enclave (EST1) tokens - a
//                                  DEDICATED key, never one that signs
//                                  transactions (relay/sso.js). Unset = the
//                                  /v1/sso/* endpoints answer 503. Prefer
//                                  SSO_SIGNER_KEY_FILE (systemd LoadCredential,
//                                  see enclave-relay-agent.service) over the
//                                  EnvironmentFile for the same reason the
//                                  operator key does.
//   FANOUT_MAX_INFLIGHT optional   global cap on concurrent upstream fan-out (256)
//   AVAIL_POLL_SEC     optional    availability poll cadence (default 10)
//   REGISTRY_POLL_SEC  optional    registry re-read cadence (default 300)
//   STALE_AFTER_SEC    optional    drop enclaves silent on-chain > this (3600)
//   MCP_DOMAIN         optional    Host(s) served as the MCP endpoint (default
//                                  mcp.enclave.host); /mcp on the API host
//                                  serves it too. See mcp.js.

import http from "node:http";
import https from "node:https";
import net from "node:net";
import tls from "node:tls";
import fs from "node:fs";
import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { readCappedText, MAX_BODY_BYTES, installProcessGuards } from "./fleet.mjs";
import { isBlockedHost } from "./net-guard.mjs";
import { isMcpHost, handleMcp } from "./mcp.js";
import { handleAccount, initAccounts } from "./auth.js";
import { handleSso, initSso } from "./sso.js";
import { handleBilling, initBilling } from "./billing.js";
import { handleSecrets, initSecrets, secretsEnabled, startSecretsSweep } from "./secrets.js";
import { handleDomains, initDomains, domainsEnabled, startDomainSweep, domainDeployment, tlsAskAllowed } from "./domains.js";
import { handleCerts, initCerts } from "./certs.js";
import { createTunnelHub } from "./tunnel.js";
import { createPadsLedger, PADS_EPOCH } from "./pads.mjs";
import { dataDir } from "./store.js";
import { boxOrigin, boxLabelOfHost } from "./boxhost.js";
installProcessGuards("api-relay");

// --- fleet tunnel: self-hosted (CGNAT) enclaves dial IN and are routed to over
// the tunnel instead of being dialed. The allowlist is a committed set of
// {name, tokenSha256} (only the hash is public; the token stays off-repo) plus
// an optional env of raw name:token pairs the relay hashes itself. Attach auth
// is routing-only; clients still verify each enclave's attestation end-to-end.
const DEFAULT_METAL_ALLOW = [
  { name: "metal0", tokenSha256: "3b28c8d9564f47b1f5031e519c8f6e7bbfaca99e41884b2740e7958e83acec81" },
  // us-west attaches by ON-CHAIN OWNERSHIP instead (TUNNEL_OPERATOR_ATTACH):
  // it signs the hub's challenge with the operator key that registered its
  // endpoint, so nothing about it is hardcoded here. Its token entry was
  // removed deliberately and must not come back — a name on this list is
  // RESERVED against every other path, so re-adding it would silently disable
  // the very mechanism it now uses.
];
const ENV_METAL_ALLOW = (process.env.METAL_TUNNEL_TOKENS || "").split(",").map((s) => s.trim()).filter(Boolean)
  .map((pair) => { const i = pair.indexOf(":"); const name = pair.slice(0, i), token = pair.slice(i + 1);
                   return name && token ? { name, tokenSha256: createHash("sha256").update(token).digest("hex") } : null; })
  .filter(Boolean);
// Permissionless attach (metal/PROTOCOL.md): with a curated allowlist of published
// Metal release measurements, ANY enclave that proves by fresh SEV-SNP quote that
// it runs one of them attaches — no token, no operator vetting. Off by default
// (empty allowlist → token-only); enable by setting METAL_ALLOWED_MEASUREMENTS on
// the box. VCEK enforcement is ON unless explicitly disabled (only disable for a
// lab box whose part has no KDS-published VCEK).
const METAL_ALLOWED_MEASUREMENTS = (process.env.METAL_ALLOWED_MEASUREMENTS || "").split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
const METAL_REQUIRE_VCEK = process.env.METAL_REQUIRE_VCEK !== "0";
// Phone-anchored hosts (shielded/anchor/PLAN.md): the anchor APK builds admitted
// (codeHash = the APK's v4 Merkle root) and the APK signing certificate(s) that
// may sign them (authorityHash = sha512 of the certificate). Both empty by
// default -> no AVF attach. The verifier pins Google's roots itself.
const METAL_AVF_CODE_HASHES = (process.env.METAL_AVF_CODE_HASHES || "").split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
const METAL_AVF_AUTHORITY_HASHES = (process.env.METAL_AVF_AUTHORITY_HASHES || "").split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
const AVF_ATTEST = METAL_AVF_CODE_HASHES.length && METAL_AVF_AUTHORITY_HASHES.length ? { codeHashes: METAL_AVF_CODE_HASHES, authorityHashes: METAL_AVF_AUTHORITY_HASHES } : null;
// The origin a CGNAT seller registers itself under: `<origin>/t/<name>` is the
// URL its on-chain entry carries, and keccak of it is the runner id its leases
// record. Configurable because a relay can be reached under more than one name;
// unset = the hosted default.
const TUNNEL_ORIGIN = (process.env.TUNNEL_PUBLIC_ORIGIN || "https://api.enclave.host").replace(/\/+$/, "");
// WHO OWNS A TUNNEL NAME on chain, for the hub's attest gate (see tunnel.js).
// null = nobody has registered `<origin>/t/<name>`, so the name is still
// first-come; an address = only that operator's key may attach under it.
// Inactive entries are treated as unowned: a deregistered seller has given the
// name up. Errors propagate — the hub decides how to fail, and it fails closed
// against an owner it has already seen.
const DEFAULT_TRUSTED_OPERATORS = ["0x390e2e0e0bc34b7f428f1e31c9b6770d5028ecc1"]; // canonical Enclave fleet operator
const _rawOperators = (process.env.TRUSTED_OPERATORS ?? "").trim();
const OPERATORS_UNRESTRICTED = /^(\*|any|all)$/i.test(_rawOperators);
const TRUSTED_OPERATORS = OPERATORS_UNRESTRICTED ? []
  : (_rawOperators
      ? _rawOperators.toLowerCase().split(",").map((s) => s.trim()).filter(Boolean)
      : DEFAULT_TRUSTED_OPERATORS.slice());

async function tunnelNameOwner(name) {
  if (!REGISTRY_ADDRESS) return null;
  const c = await chain();
  // A minted box name (boxhost.js) registers under its OWN host, so its id is
  // computable from the name alone — which is the whole reason the derived
  // label doubles as the attach name. Legacy names keep the path form.
  const id = await endpointId(boxOrigin(name) || `${TUNNEL_ORIGIN}/t/${name}`);
  const e = await c.readContract({ address: REGISTRY_ADDRESS, abi: GET_ABI, functionName: "get", args: [id] });
  const op = String(e?.operator || "");
  return e?.active && !/^0x0{40}$/i.test(op) ? op.toLowerCase() : null;
}
// Dealt pads ledger: created on first use so the data dir and the hub exist.
let padsLedgerInstance;
function padsLedger() {
  if (padsLedgerInstance === undefined) {
    const d = dataDir();
    padsLedgerInstance = d ? createPadsLedger({ dir: d, hub: tunnelHub, masterSeed: process.env.PADS_MASTER_SEED || null }) : null;
    if (!padsLedgerInstance) console.log("[pads] no data dir: the pads ledger is disabled");
  }
  return padsLedgerInstance;
}
const tunnelHub = createTunnelHub({
  allow: [...DEFAULT_METAL_ALLOW, ...ENV_METAL_ALLOW],
  attest: METAL_ALLOWED_MEASUREMENTS.length || AVF_ATTEST ? { allowedMeasurements: METAL_ALLOWED_MEASUREMENTS, requireVcek: METAL_REQUIRE_VCEK, ...(AVF_ATTEST ? { avf: AVF_ATTEST } : {}) } : null,
  operatorFor: tunnelNameOwner,
  // TUNNEL_OPERATOR_ATTACH=1 — let a box prove its tunnel name with the operator
  // key that registered its endpoint on chain, instead of a token whose hash
  // someone committed here. It is how a RELAY attaches: not a TEE by design, so
  // it has no measurement to quote, but it does have an on-chain identity.
  // OFF by default because a tunnel row bypasses the dial-time operator
  // allowlist, so this lets anyone who registers /t/<name> appear in the fleet
  // listing under that name — a deliberate widening, behind a deliberate switch.
  operatorAttach: /^(1|true|yes|on)$/i.test((process.env.TUNNEL_OPERATOR_ATTACH || "").trim()),
  // and the owner still has to clear the same operator bar the dial path uses —
  // a tunnel row bypasses that filter, so it is enforced at attach instead
  trustedOperators: TRUSTED_OPERATORS, operatorsUnrestricted: OPERATORS_UNRESTRICTED,
  // when an enclave attaches/detaches, refresh discovery + availability now so it
  // enters/leaves `live` immediately rather than on the slow (5 min) registry poll
  onChange: () => { pollRegistry().then(pollAvailability).catch(() => {}); },
});

let   REGISTRY_ADDRESS  = (process.env.REGISTRY_ADDRESS || "").trim();   // env fallback; the address book (below) overrides
let   DEPLOYMENTS_ADDRESS = (process.env.DEPLOYMENTS_ADDRESS || "").trim(); // EnclaveDeployments ledger; book overrides too
const ADDRESS_BOOK      = (process.env.ADDRESS_BOOK_ADDRESS || "").trim();
const BASE_RPC          = process.env.BASE_RPC || "https://mainnet.base.org";
const STATIC_ENCLAVES   = (process.env.ENCLAVES || "").split(",").map((s) => s.trim().replace(/\/+$/, "")).filter(Boolean);
const PORT              = parseInt(process.env.API_RELAY_PORT || "8100", 10);
const AVAIL_POLL_SEC    = parseInt(process.env.AVAIL_POLL_SEC || "10", 10);
const REGISTRY_POLL_SEC = parseInt(process.env.REGISTRY_POLL_SEC || "300", 10);
const STALE_AFTER_SEC   = parseInt(process.env.STALE_AFTER_SEC || "3600", 10);
// Per-deployment app subdomains: <dep-id>.<APP_DOMAIN> maps to the enclave's
// /x/<id> data path, so each app is its OWN origin (isolated from the frontend
// and from other tenants). Host uses a hyphen ("dep-abc") since "_" is invalid
// in a hostname; we map it back to the canonical "dep_abc". Empty = disabled.
// Comma-separated: during a domain cutover both the new and the old suffix
// route (e.g. "app.enclave.host,app.nan.host"); the first entry is primary.
const APP_DOMAINS       = (process.env.APP_DOMAIN || "").toLowerCase().split(",")
  .map(s => s.trim().replace(/^\.+|\.+$/g, "")).filter(Boolean);

if (!REGISTRY_ADDRESS && !ADDRESS_BOOK && !STATIC_ENCLAVES.length) {
  console.error("fatal: set ADDRESS_BOOK_ADDRESS or REGISTRY_ADDRESS (on-chain discovery) or ENCLAVES (static list)");
  process.exit(1);
}

// --- hardening config ----------------------------------------------------------
// SECURITY (B1/B2/B3): the on-chain registry is permissionless — anyone can
// register an endpoint. TRUSTED_OPERATORS (comma-separated, lowercased Enclave-
// Registry operator addresses) restricts on-chain discovery to vetted operators,
// so session tokens and /x data-path traffic only ever reach them.
//
// FAIL CLOSED: this single control sits behind three trust boundaries (token
// harvest, egress-token leak, subdomain hijack), so an UNSET var must never
// silently reopen them on a rebuilt or fresh box. When TRUSTED_OPERATORS is
// unset/empty we fall back to the BAKED canonical operator set below (not "trust
// everyone"). Running a genuinely unrestricted relay is still possible but only
// as an explicit, auditable opt-in: TRUSTED_OPERATORS=* (or "any"/"all").
const isHttpsEndpoint = (ep) => { try { return new URL(ep).protocol === "https:"; } catch { return false; } };
let _warnedUnauth = false;
function warnIfUnauthenticated() {
  if (STATIC_ENCLAVES.length || !OPERATORS_UNRESTRICTED || _warnedUnauth) return;
  _warnedUnauth = true;
  console.error("[api-relay] WARNING: TRUSTED_OPERATORS=* — routing tokens/traffic to EVERY endpoint in the " +
    "permissionless EnclaveRegistry (no operator allowlist), by explicit configuration. Unset it to restrict to the vetted operator set.");
}
// CORS (fix 5): allowlist instead of reflecting any Origin with credentials.
const CORS_ORIGINS = (process.env.CORS_ORIGINS || "https://enclave.host,https://www.enclave.host")
  .split(",").map((s) => s.trim()).filter(Boolean);
// Trusted-proxy switch (fix 6): Caddy fronts the relay in production and sets
// x-forwarded-host / x-forwarded-for. Default trusts those (current behavior).
// Set TRUSTED_PROXY to an off value (0/false/off/no/none) when the relay is
// directly internet-exposed, so a client can't spoof routing via x-forwarded-*.
const TRUSTED_PROXY = !/^(0|false|off|no|none)$/i.test((process.env.TRUSTED_PROXY ?? "1").trim());
const routingHost = (req) =>
  (TRUSTED_PROXY && req.headers["x-forwarded-host"]) || req.headers.host;
const clientIp = (req) => {
  if (TRUSTED_PROXY) {
    // The LAST entry, not the first. X-Forwarded-For is a client-writable
    // header that the proxy APPENDS its peer address to (Caddy's default), so a
    // request arriving as `X-Forwarded-For: 1.2.3.4` reaches us as
    // `1.2.3.4, <real client>` - the first entry is whatever the sender typed.
    // Keying rate limits on it let anyone mint a fresh bucket per request by
    // varying a header, which is not a small thing here: this key guards the
    // ACME on-demand-TLS miss limiter (burning the CA's rate limit is a
    // platform-wide outage), the passkey/SIWE attempt limits, and the paid
    // featured-view dedupe. Taking the last entry is correct whether the proxy
    // appends or replaces; it assumes exactly ONE trusted hop, which is this
    // deployment (Caddy in front). Directly internet-exposed => TRUSTED_PROXY=0,
    // and then the socket address below is the only thing anyone can trust.
    const xs = String(req.headers["x-forwarded-for"] || "").split(",").map((x) => x.trim()).filter(Boolean);
    if (xs.length) return xs[xs.length - 1];
  }
  return req.socket?.remoteAddress || "unknown";
};

// --- featured-slot view metering ------------------------------------------------
// The site beacons one POST /v1/featured-view {app} per paid featured
// impression; we count at most ONE view per client per app per UTC day (marks
// are salted hashes, reset daily - no IP ever persists). Lifetime totals are
// MONOTONIC and snapshot to FEATURED_VIEWS_FILE every minute, so a relay
// restart can only ever UNDER-count (the EnclaveFeatured contract's stated
// bias: the meter may under-charge an advertiser, never over-charge). The
// admin console reads GET /v1/featured-views and settles the delta vs the
// CampaignSettled event sum on-chain.
const FEAT_FILE = process.env.FEATURED_VIEWS_FILE || "featured-views.json";
const FEAT_MAX_APPS = 500, FEAT_MAX_MARKS = 200000;       // memory bounds, not billing limits
const featViews = (() => { try { return JSON.parse(fs.readFileSync(FEAT_FILE, "utf8")) || {}; } catch { return {}; } })();
let featDirty = false, featDay = "", featMarks = new Set();
const featSalt = randomBytes(16).toString("hex");         // per-boot; a restart forgives the day's dedup, never the totals
setInterval(() => { if (featDirty) { featDirty = false; fs.writeFile(FEAT_FILE, JSON.stringify(featViews), () => {}); } }, 60000).unref();
function featCount(req, app) {
  if (!/^0x[0-9a-f]{64}$/.test(app)) return false;
  const day = new Date().toISOString().slice(0, 10);
  if (day !== featDay) { featDay = day; featMarks = new Set(); }
  const mark = createHash("sha256").update(featSalt + "|" + day + "|" + clientIp(req) + "|" + app).digest("base64");
  if (featMarks.has(mark) || featMarks.size >= FEAT_MAX_MARKS) return false;
  if (featViews[app] == null && Object.keys(featViews).length >= FEAT_MAX_APPS) return false;
  featMarks.add(mark);
  featViews[app] = (featViews[app] || 0) + 1;
  featDirty = true;
  return true;
};
const isLoopback = (req) => {
  const a = req.socket?.remoteAddress || "";
  return /^127\./.test(a) || a === "::1" || a === "::ffff:127.0.0.1" || a.startsWith("::ffff:127.");
};
// "Loopback" is NOT "internal" on a box with a proxy in front of it, and this
// relay always has one. Every request Caddy forwards arrives from 127.0.0.1, so
// a socket-address check alone admits the entire internet — verified against
// production 2026-07-30, where a public curl of /internal/tls-ask was answered
// (200 for a real deployment subdomain, 400 for junk: an unauthenticated
// existence oracle, and with custom domains a "which hostnames does this
// platform serve" oracle).
//
// What actually distinguishes the real caller: Caddy GENERATES its on-demand
// `ask` request itself, so it carries no forwarding headers, while anything it
// proxies on a client's behalf always does — clientIp() above depends on
// exactly that being true. Requiring their absence closes the door with no
// Caddyfile change. INTERNAL_TOKEN is the stronger form when someone can edit
// the Caddyfile: a shared header the internet cannot guess, checked instead of
// the heuristic.
const FORWARDED_HEADERS = ["x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
                           "x-forwarded-port", "x-real-ip", "forwarded", "via"];
const INTERNAL_TOKEN = (process.env.INTERNAL_TOKEN || "").trim();
const isInternalCall = (req) => {
  if (!isLoopback(req)) return false;
  if (INTERNAL_TOKEN) {
    const got = String(req.headers["x-internal-token"] || "");
    return got.length === INTERNAL_TOKEN.length
      && timingSafeEqual(Buffer.from(got), Buffer.from(INTERNAL_TOKEN));
  }
  return FORWARDED_HEADERS.every((h) => req.headers[h] == null);
};
// In-memory token-bucket rate limiter (fix 2), per source key. Generous by
// design — it only sheds the abusive miss/fan-out traffic, not normal browsing.
function makeRateLimiter({ capacity, refillPerSec }) {
  const buckets = new Map();
  setInterval(() => { const now = Date.now(); for (const [k, b] of buckets) if (now - b.at > 300_000) buckets.delete(k); }, 60_000).unref?.();
  return (key) => {
    const now = Date.now();
    let b = buckets.get(key);
    if (!b) { b = { tokens: capacity, at: now }; buckets.set(key, b); }
    b.tokens = Math.min(capacity, b.tokens + ((now - b.at) / 1000) * refillPerSec);
    b.at = now;
    if (b.tokens < 1) return false;
    b.tokens -= 1; return true;
  };
}
const rlMiss = makeRateLimiter({ capacity: 60, refillPerSec: 10 });   // /x + app-subdomain owner misses
const rlHint = makeRateLimiter({ capacity: 20, refillPerSec: 2 });    // /v1/claim-hint fan-out
// Signed-upload authorization (/v1/apps/upload-token): per-wallet token-mint cap.
// Generous burst, ~30/hr steady — the gateway enforces the real BYTE budget.
const rlUpload = makeRateLimiter({ capacity: 30, refillPerSec: 30 / 3600 });
// Dedicated secret shared ONLY with the wasm add-gateway on this box (NOT the
// fleet SECRET). Empty = signed uploads unavailable (503). See ipfs-add-gateway.py.
const UPLOAD_KEY = process.env.UPLOAD_KEY || "";
// Global cap on concurrent upstream fan-out requests (fix 2): bounds the
// amplification of one inbound request into N enclave requests.
const FANOUT_MAX = parseInt(process.env.FANOUT_MAX_INFLIGHT || "256", 10);
let fanoutInflight = 0;
const fanoutReserve = (n) => { if (fanoutInflight + n > FANOUT_MAX) return false; fanoutInflight += n; return true; };
const fanoutRelease = (n) => { fanoutInflight = Math.max(0, fanoutInflight - n); };

// --- registry read (mirrors scripts/enclave-discover.mjs) ------------------------
const ENCLAVE_TUPLE_V1 = [
  { name: "endpoint", type: "string" }, { name: "repo", type: "string" },
  { name: "measurement", type: "bytes32" }, { name: "operator", type: "address" },
  { name: "registeredAt", type: "uint64" }, { name: "lastSeen", type: "uint64" },
  { name: "active", type: "bool" }];
const ABI = [
  { type: "function", name: "count", stateMutability: "view", inputs: [],
    outputs: [{ type: "uint256" }] },
  { type: "function", name: "getPage", stateMutability: "view",
    inputs: [{ name: "start", type: "uint256" }, { name: "n", type: "uint256" }],
    outputs: [{ type: "tuple[]", components: ENCLAVE_TUPLE_V1 }] },
];
// Registry schema 2 APPENDED the operator's per-machine prices to the entry.
// Schema 3 appended the PROOF KEY, schema 4 the seller's declared PAYOUT WALLET
// (the wallet whose own deployments a rev-12 ledger hosts here for free — set
// only by that wallet, so a 0x0 here is a box that charges everyone).
// Appended fields keep every earlier field's
// offset, so a short decode drops the tail rather than misreading it — but the
// shape is still sniffed per address (cached; the address book can repoint us
// mid-flight) so callers SEE the newer fields. Schema-1 registries have no
// getter at all: it reverts. THIS BLOCK IS PINNED IDENTICAL across
// scripts/enclave-discover.mjs, relay/fleet.mjs and relay/api-relay.js by
// test/registry-schema.test.mjs — a drift is a cutover outage in whichever
// service lagged.
const ENCLAVE_TUPLE = [...ENCLAVE_TUPLE_V1,
  { name: "cpuPricePerSec6", type: "uint64" }, { name: "gpuPricePerSec6", type: "uint64" },
  { name: "proofKey", type: "address" }, { name: "payoutWallet", type: "address" },
  { name: "caps", type: "uint64" }, { name: "region", type: "string" }];
const ENCLAVE_TUPLE_V4 = ENCLAVE_TUPLE.slice(0, 11);  // schema 4: payout wallet, no caps
const ENCLAVE_TUPLE_V3 = ENCLAVE_TUPLE.slice(0, 10);  // schema 3: proof key, no payout wallet
const ENCLAVE_TUPLE_V2 = ENCLAVE_TUPLE.slice(0, 9);   // schema 2: priced, no proof key
const abiForRev = (rev) => [ABI[0], { ...ABI[1],
  outputs: [{ type: "tuple[]", components:
    rev >= 5 ? ENCLAVE_TUPLE : rev >= 4 ? ENCLAVE_TUPLE_V4 : rev >= 3 ? ENCLAVE_TUPLE_V3
             : rev >= 2 ? ENCLAVE_TUPLE_V2 : ENCLAVE_TUPLE_V1 }] }];
// Schema 5 capability bits (EnclaveRegistry CAP_*): what a registered box DOES,
// now that being listed no longer implies running code. caps === 0 is every row
// written before schema 5 and every one of them IS an enclave, so 0 MUST read as
// CAP_HOST — a reader that takes it for "no capabilities" empties the fleet the
// day the new registry deploys.
const CAP_HOST = 1n, CAP_APP_SNI = 2n, CAP_TCP_PORTS = 4n, CAP_UDP = 8n, CAP_TUNNEL_HUB = 16n;
const CAP_RELAY_ANY = CAP_APP_SNI | CAP_TCP_PORTS | CAP_UDP | CAP_TUNNEL_HUB;
const capsOf = (e) => { try { return BigInt(e.caps ?? 0); } catch { return 0n; } };
const isHostRow = (e) => { const c = capsOf(e); return c === 0n || (c & CAP_HOST) !== 0n; };
const isRelayRow = (e) => (capsOf(e) & CAP_RELAY_ANY) !== 0n;
const SCHEMA_ABI = [{ type: "function", name: "registrySchema", stateMutability: "view",
  inputs: [], outputs: [{ type: "uint256" }] }];
// Single-entry read (tunnelNameOwner). The v1 tuple is a PREFIX of the v2 one
// and `operator` sits inside it, so this decode is correct on both schemas —
// no sniff needed for the one field it reads.
const GET_ABI = [{ type: "function", name: "get", stateMutability: "view",
  inputs: [{ type: "bytes32" }], outputs: [{ type: "tuple", components: ENCLAVE_TUPLE_V1 }] }];
// Which entry shape the registry at `addr` speaks: 2 = priced entries (the
// getter exists), 1 = the original seven fields (it reverts there). Cached per
// address, because the address book can repoint us at a new registry mid-run.
const _regRev = new Map();
async function registryAbi(client, addr) {
  const key = String(addr).toLowerCase();
  if (!_regRev.has(key)) {
    const rev = await client.readContract({ address: addr, abi: SCHEMA_ABI, functionName: "registrySchema" })
      .then(Number).catch(() => 1);
    _regRev.set(key, rev);
  }
  return abiForRev(_regRev.get(key));
}

let _client = null;
async function chain() {
  if (!_client) {
    const { createPublicClient, http: viemHttp } = await import("viem");
    const { base } = await import("viem/chains");
    _client = createPublicClient({ chain: base, transport: viemHttp(BASE_RPC) });
  }
  return _client;
}
// "registry" as ascii-right-padded bytes32 (the EnclaveAddressBook key)
const BOOK_ABI = [{ type: "function", name: "addr", stateMutability: "view",
  inputs: [{ type: "bytes32" }], outputs: [{ type: "address" }] }];
const BOOK_KEY_REGISTRY = "0x7265676973747279000000000000000000000000000000000000000000000000";
// An enclave's registry id is keccak256(bytes(endpoint)) — the contract's own
// derivation (EnclaveRegistry.register), which is also what EnclaveDeployments
// records as a lease's `runner`. Carrying the id on every registry row lets
// ledger rows be matched against the live fleet (see ledgerStatus).
let _hashEndpoint = null;
async function endpointId(endpoint) {
  if (!_hashEndpoint) {
    const { keccak256, stringToBytes } = await import("viem");
    _hashEndpoint = (s) => keccak256(stringToBytes(s));
  }
  return _hashEndpoint(endpoint);
}
// Seller declarations out of the registry, keyed by the SAME endpoint id the
// ledger uses (keccak of the registered URL, which is what a claim records as
// `runner`). Rebuilt every discovery pass from every ACTIVE, non-stale entry —
// deliberately BEFORE the TRUSTED_OPERATORS allowlist below.
//
// That allowlist is a DIAL-SAFETY gate: it decides whose registered endpoint
// this relay is willing to route traffic to (B2, alongside the https and SSRF
// filters). Reading what a box DECLARED about itself is not a routing decision,
// and a tunnel box is authenticated at attach time and already being served.
// Gating the declaration on the allowlist would leave free self-hosting broken
// for exactly the sellers the tunnel exists for: a third-party CGNAT box has no
// reason to be on a first-party operator list, so its owner's own deployments
// would be priced at its posted ask and read as unfundable forever. The chain is
// the authority either way — EnclaveDeployments._hostRate prices a claim off
// this very entry, whatever this relay believes.
let declaredById = new Map();

async function readRegistry() {
  // tunnel enclaves are locally-authenticated at attach time, so they bypass the
  // dial-based discovery filters and are simply appended to whatever the static
  // list / on-chain registry yields. A discovery failure (e.g. a transient RPC
  // outage) must NOT drop attached tunnels — surface it only when there are none.
  let base = [];
  try { base = await discoverRegistry(); }
  catch (e) { if (!tunnelHub.count()) throw e; console.error(`[api-relay] discovery failed, serving ${tunnelHub.count()} tunnel enclave(s) only: ${e.message}`); }
  // A tunnel box that SELLS registers on-chain under its public relay-routed
  // URL, and every lease it claims records keccak(that URL) as `runner`. Stamp
  // the tunnel row with that id so runner matching — ledgerStatus's "running",
  // runnerEndpointOf's routing, enclaveNameOf's label — recognizes the box
  // (the synthetic tunnel:<name> id matched nothing and left hosted rows stuck
  // on "claimed"). Never-registered dev tunnels keep the synthetic id. A
  // discovered twin of the same id is dropped: the tunnel is the working route
  // (its https endpoint would just dial back through this relay).
  const tuns0 = await Promise.all(tunnelHub.origins().map(async (o) =>
    o.publicUrl ? { ...o, id: await endpointId(o.publicUrl) } : o));
  const tids = new Set(tuns0.map((o) => String(o.id).toLowerCase()));
  // A tunnel row is built at attach time from what the box said about itself,
  // so it carries no on-chain facts at all - and the discovered twin that does
  // is dropped just below. `payoutWallet`, `caps` and `region` are SELLER
  // DECLARATIONS living in the registry entry and nowhere else, so they are
  // read back from declaredById (the pre-allowlist snapshot above, keyed by the
  // same endpoint id) - payoutWallet especially, since the contract writes it
  // only from the wallet itself (setPayoutWallet) and that is precisely what
  // makes it unforgeable. The box repeats it in its own /availability, but
  // every consumer deliberately refuses that copy: a box must not be able to
  // quote itself free.
  // Losing it is not cosmetic. EnclaveDeployments._hostRate returns 0 when the
  // declared payout wallet owns the deployment, so a self-hosted record's
  // correct state is an EMPTY balance - and with payoutWallet missing, both
  // clients price the box at its posted ask and refuse a transaction the ledger
  // accepts ("the remaining balance can't fund even one second at the new
  // rate"). Seen 2026-08-31 on eyesoff-ai/metal0: rate 0 on-chain, resize
  // refused by the console AND the CLI, which share this one field.
  // The tunnel keeps its ROUTING fields (endpoint/id/name/publicUrl/mode): its
  // https twin would only dial back through this relay.
  const tuns = tuns0.map((o) => {
    const c = declaredById.get(String(o.id).toLowerCase());
    return c ? { ...o, payoutWallet: c.payoutWallet,
                 ...(o.caps === undefined ? { caps: c.caps } : {}),
                 ...(o.region === undefined ? { region: c.region } : {}) } : o;
  });
  return [...base.filter((e) => !e.id || !tids.has(String(e.id).toLowerCase())), ...tuns];
}
// Human-facing label for an https endpoint: its first hostname label
// ("https://kryptos.enclave.host/..." -> "kryptos"). Tunnel rows carry their
// tunnel name instead (tunnel.js origins()); endpoints stay the routing keys.
function endpointName(endpoint) {
  try { return new URL(endpoint).hostname.split(".")[0] || null; } catch { return null; }
}
async function discoverRegistry() {
  if (STATIC_ENCLAVES.length) {
    declaredById = new Map();     // a static list is a dev seam: no chain, nothing declared
    return Promise.all(STATIC_ENCLAVES.map(async (endpoint) =>
      ({ endpoint, id: await endpointId(endpoint), name: endpointName(endpoint), repo: null, lastSeen: null })));
  }
  const c = await chain();
  // resolve the registry from the on-chain address book each cycle, so a
  // registry redeploy reaches this box with one owner tx (no env edits)
  if (ADDRESS_BOOK) {
    try {
      const a = await c.readContract({ address: ADDRESS_BOOK, abi: BOOK_ABI, functionName: "addr", args: [BOOK_KEY_REGISTRY] });
      if (a && !/^0x0{40}$/i.test(a) && a.toLowerCase() !== REGISTRY_ADDRESS.toLowerCase()) {
        console.log(`[api-relay] address book: registry ${REGISTRY_ADDRESS || "(unset)"} -> ${a}`);
        REGISTRY_ADDRESS = a;
      }
    } catch (e) { /* keep the current registry; next poll retries */ }
  }
  if (!REGISTRY_ADDRESS) throw new Error("no registry address (book unresolved and REGISTRY_ADDRESS unset)");
  const abi = await registryAbi(c, REGISTRY_ADDRESS);
  const total = Number(await c.readContract({ address: REGISTRY_ADDRESS, abi, functionName: "count" }));
  const out = [];
  for (let start = 0; start < total; start += 50)
    out.push(...await c.readContract({ address: REGISTRY_ADDRESS, abi,
      functionName: "getPage", args: [BigInt(start), 50n] }));
  const now = Math.floor(Date.now() / 1000);
  warnIfUnauthenticated();
  const fresh = out.filter((e) => e.active && now - Number(e.lastSeen) <= STALE_AFTER_SEC);
  // Snapshot every live entry's declarations before the dial-safety filters —
  // see declaredById above. Deregistered and stale rows are excluded with the
  // rest: a box that stopped saying it is here declares nothing.
  declaredById = new Map(await Promise.all(fresh.map(async (e) =>
    [String(await endpointId(String(e.endpoint || "").replace(/\/+$/, ""))).toLowerCase(),
     { payoutWallet: e.payoutWallet || null, caps: Number(capsOf(e)), region: e.region || null }])));
  return Promise.all(fresh
    // B2: only vetted operators (baked default, or the env allowlist). Pass-all
    // ONLY under the explicit TRUSTED_OPERATORS=* opt-in — never by omission.
    .filter((e) => OPERATORS_UNRESTRICTED || TRUSTED_OPERATORS.includes(String(e.operator || "").toLowerCase()))
    .map((e) => ({ e, endpoint: e.endpoint.replace(/\/+$/, "") }))
    // B1/B3: never route to a non-https discovered endpoint (real enclaves are https)
    .filter(({ endpoint }) => { const ok = isHttpsEndpoint(endpoint); if (!ok) console.error(`[api-relay] skipping non-https registry endpoint: ${endpoint}`); return ok; })
    // SSRF: the registry is permissionless — drop any endpoint whose host is a
    // literal private/loopback/link-local IP (or localhost) so an attacker can't
    // register https://127.0.0.1/ or https://169.254.169.254/ and make this relay
    // dial its own localhost / cloud metadata. (Real enclaves are public domains.)
    .filter(({ endpoint }) => { let h; try { h = new URL(endpoint).hostname; } catch { return false; } const ok = !isBlockedHost(h); if (!ok) console.error(`[api-relay] skipping non-global registry endpoint: ${endpoint}`); return ok; })
    // payoutWallet (schema 4) rides along because ledgerStatus needs it: it is
    // what tells a self-hosted row from an unfunded one. Undefined on an older
    // registry, which reads as "nobody hosts anything free" — the safe default.
    // caps/region (schema 5) ride along the same way, for whenever that registry
    // deploys. Nothing routes on them today: which box RELAYS is read from its
    // /availability (hasNoResources + the `relay` service block), so the feature
    // needs no contract and no migration to work.
    .map(async ({ e, endpoint }) =>
      ({ endpoint, id: await endpointId(endpoint), name: endpointName(endpoint), repo: e.repo,
         lastSeen: Number(e.lastSeen), payoutWallet: e.payoutWallet || null,
         caps: Number(capsOf(e)), region: e.region || null })));
}

// --- EnclaveDeployments ledger (the source of truth for a wallet's work) --------
// The fleet only reports deployments it currently HOSTS; created/funded/stopped
// records live on-chain regardless, so the list/get endpoints read the ledger
// too. Resolved from the address book like the registry; paged eth_calls (no
// log scans - public RPCs cap those), cached briefly.
const BOOK_KEY_DEPLOYMENTS = "0x6465706c6f796d656e7473" + "0".repeat(42);   // "deployments" ascii right-padded
const DEP_TUPLE = [   // Deployment struct, schema rev 2
  { name: "id", type: "bytes32" }, { name: "owner", type: "address" },
  { name: "appRef", type: "string" }, { name: "ports", type: "string" },
  { name: "configCid", type: "string" },
  { name: "gpuMilli", type: "uint16" }, { name: "cpuMilli", type: "uint16" },
  { name: "appPort", type: "uint32" }, { name: "isPublic", type: "bool" },
  { name: "active", type: "bool" }, { name: "createdAt", type: "uint64" },
  { name: "rate", type: "uint256" }, { name: "balance6", type: "uint256" },
  { name: "spent6", type: "uint256" }, { name: "runner", type: "bytes32" },
  { name: "runnerOperator", type: "address" }, { name: "leaseUntil", type: "uint64" },
];
// rev-1 ledgers carry a removed sshPubKey string after ports (decoded, unused)
const DEP_TUPLE_V1 = [...DEP_TUPLE.slice(0, 4), { name: "sshPubKey", type: "string" }, ...DEP_TUPLE.slice(4)];
const depAbiFor = (components) => [
  { type: "function", name: "count", stateMutability: "view", inputs: [], outputs: [{ type: "uint256" }] },
  { type: "function", name: "getPage", stateMutability: "view",
    inputs: [{ name: "start", type: "uint256" }, { name: "n", type: "uint256" }],
    outputs: [{ type: "tuple[]", components }] },
];
// Which shape the ledger at DEPLOYMENTS_ADDRESS speaks: deploymentsSchema()
// reverts on rev-1 contracts (that IS the answer); cached per address so an
// address-book repoint re-sniffs. Transport errors don't cache - this round
// reads rev 1 and the next call retries the sniff.
let _depShape = { addr: null, abi: depAbiFor(DEP_TUPLE_V1) };
async function depAbi(c) {
  if (_depShape.addr === DEPLOYMENTS_ADDRESS) return _depShape.abi;
  try {
    const rev = Number(await c.readContract({ address: DEPLOYMENTS_ADDRESS,
      abi: [{ type: "function", name: "deploymentsSchema", stateMutability: "view", inputs: [], outputs: [{ type: "uint256" }] }],
      functionName: "deploymentsSchema" }));
    _depShape = { addr: DEPLOYMENTS_ADDRESS, abi: depAbiFor(rev >= 2 ? DEP_TUPLE : DEP_TUPLE_V1) };
  } catch (e) {
    // Only a genuine REVERT proves a rev-1 contract (it has code but not the
    // selector). "returned no data" means THIS provider sees no code at the
    // address — a lagging/throttled pool member during a migration — and
    // viem's wrapper names (ContractFunction*) cover both, so classify by
    // message. Caching rev 1 on a transient wedged the supervisor's whole
    // claim path 2026-07-17 (supervisor.js sniffCachePolicy).
    if (/revert/i.test(e?.shortMessage || e?.message || "")) _depShape = { addr: DEPLOYMENTS_ADDRESS, abi: depAbiFor(DEP_TUPLE_V1) };
    else return depAbiFor(DEP_TUPLE_V1);
  }
  return _depShape.abi;
}
// a misaligned tuple decode (wrong cached shape for this ledger) — drop the
// sniff cache so the next tick re-sniffs instead of staying wedged
const shapeDecodeError = (e) => /safe integer range|out[- ]of[- ]bounds|data size|not a valid boolean/i.test(e?.shortMessage || e?.message || "");
async function resolveDeployments() {
  if (!ADDRESS_BOOK) return;
  try {
    const c = await chain();
    const a = await c.readContract({ address: ADDRESS_BOOK, abi: BOOK_ABI, functionName: "addr", args: [BOOK_KEY_DEPLOYMENTS] });
    if (a && !/^0x0{40}$/i.test(a) && a.toLowerCase() !== DEPLOYMENTS_ADDRESS.toLowerCase()) {
      console.log(`[api-relay] address book: deployments ${DEPLOYMENTS_ADDRESS || "(unset)"} -> ${a}`);
      DEPLOYMENTS_ADDRESS = a;
    }
  } catch (e) { /* keep the current address; next poll retries */ }
}
const LEDGER_TTL_MS = 10_000;
let _ledger = { rows: [], at: 0, inflight: null };
async function ledgerRows() {
  if (!DEPLOYMENTS_ADDRESS) return _ledger.rows;
  if (Date.now() - _ledger.at < LEDGER_TTL_MS) return _ledger.rows;
  if (_ledger.inflight) return _ledger.inflight;
  _ledger.inflight = (async () => {
    try {
      const c = await chain();
      const abi = await depAbi(c);
      const total = Number(await c.readContract({ address: DEPLOYMENTS_ADDRESS, abi, functionName: "count" }));
      const rows = [];
      for (let start = 0; start < total; start += 50)
        rows.push(...await c.readContract({ address: DEPLOYMENTS_ADDRESS, abi,
          functionName: "getPage", args: [BigInt(start), 50n] }));
      _ledger.rows = rows; _ledger.at = Date.now();
      return rows;
    } catch (e) {
      if (shapeDecodeError(e)) _depShape = { addr: null, abi: _depShape.abi };
      throw e;
    } finally { _ledger.inflight = null; }
  })();
  return _ledger.inflight;
}
// A ledger record's status, synthesized WITHOUT asking any enclave (the
// tokenless list is built purely from these): "running" = a live lease whose
// runner is a live, answering enclave — matched by registry id, keccak256 of
// the endpoint (the moment right after a claim, while the runner still
// provisions, reads as running here; the signed-in view carries the enclave's
// finer-grained truth); "claimed" = a lease is live but its runner isn't
// answering (enclave down/restarting); "queued" = funded work awaiting a claim
// (incl. expired leases - claimable); they resume by themselves, nothing needs
// the owner. "unfunded" = the balance can't buy one second (drained mid-run or
// funded below the rate): no enclave will claim it until the owner tops up —
// the boundary mirrors the contract's claimable() (balance6 >= rate) and the
// supervisor's own sweep gate, so "queued" always means "will start by itself".
// "free" (ledger rev 12) sits outside all of that: an answering enclave has
// declared this owner as its payout wallet, so the deployment's price THERE is
// zero and it never needs money at all.
const ZERO32 = /^0x0+$/;
const runnerIsLive = (runner) => {
  runner = String(runner).toLowerCase();
  return live.some((e) => e.id && e.id.toLowerCase() === runner);
};
// Human label for a lease's runner: registry id -> the enclave's name
// ("kryptos", "metal0"). The full registry beats `live` here — a briefly
// unanswering box should still be NAMED on the rows it holds, not blanked.
function enclaveNameOf(runner) {
  runner = String(runner || "").toLowerCase();
  if (!runner || ZERO32.test(runner)) return null;
  const hit = [...registry, ...live].find((e) => e.id && e.id.toLowerCase() === runner);
  return hit ? (hit.name || endpointName(hit.endpoint)) : null;
}
// Is some ANSWERING enclave hosting this owner for free — i.e. has it declared
// this very wallet as its payout wallet (registry schema 4)? Then the ledger
// charges it nothing there, and the two money-shaped statuses below would both
// libel it: "awaiting_payment" before its first claim (it will never have a
// balance) and "unfunded" between leases (nothing is owed). `live` rather than
// the full registry, because the promise "queued means it will start by itself"
// is only true of a box that is actually answering.
const hostedFree = (owner) => {
  owner = String(owner || "").toLowerCase();
  if (!owner || ZERO32.test(owner)) return false;
  return live.some((e) => e.payoutWallet && String(e.payoutWallet).toLowerCase() === owner);
};
function ledgerStatus(d) {
  if (!d.active) return "stopped";
  const free = hostedFree(d.owner);
  if (!free && !(d.balance6 > 0n || d.spent6 > 0n)) return "awaiting_payment";
  if (!ZERO32.test(d.runner) && Number(d.leaseUntil) * 1000 > Date.now())
    return runnerIsLive(d.runner) ? "running" : "claimed";
  return free || d.balance6 >= d.rate ? "queued" : "unfunded";
}
// Dedicated per-deployment IPv6, synthesized from PUBLIC data (mirrors the
// supervisor's depAddrFor exactly: sha256(id) low 64 host bits into the routed
// /64, low range reserved for infra). The tokenless dashboard reads ledger
// rows, so without this no signed-out view ever shows a deployment's address.
// Only rows the inbound relays actually serve get one here: public + running +
// declared tcp/udp ports (the tcp6/udp relays' own netMap gate). Egress-only
// addresses stay the enclave view's call - it alone knows egress is enabled.
// DEP_ADDR_PREFIX = the relay box's routed /64 (same env as the supervisor).
const DEP_ADDR_PREFIX = (process.env.DEP_ADDR_PREFIX || "").trim();
function v6ToBig(s) {
  const [head, tail] = s.split("::");
  const hi = head ? head.split(":").filter(Boolean) : [];
  const lo = tail ? tail.split(":").filter(Boolean) : [];
  const mid = Array(8 - hi.length - lo.length).fill("0");
  const groups = s.includes("::") ? [...hi, ...mid, ...lo] : s.split(":");
  if (groups.length !== 8) throw new Error(`bad IPv6 "${s}"`);
  return groups.reduce((a, g) => (a << 16n) | BigInt(parseInt(g || "0", 16)), 0n);
}
function bigToV6(n) {
  const g = [];
  for (let i = 0; i < 8; i++) g[i] = Number((n >> BigInt((7 - i) * 16)) & 0xffffn);
  let best = { i: -1, len: 0 }, cur = { i: -1, len: 0 };
  g.forEach((v, i) => {
    if (v === 0) { if (cur.i < 0) cur = { i, len: 0 }; cur.len++; if (cur.len > best.len) best = { ...cur }; }
    else cur = { i: -1, len: 0 };
  });
  const hex = g.map((v) => v.toString(16));
  if (best.len > 1) { hex.splice(best.i, best.len, ""); if (best.i === 0) hex.unshift(""); if (best.i + best.len === 8) hex.push(""); }
  return hex.join(":").replace(/:{3,}/, "::");
}
function depAddrFor(id) {
  if (!DEP_ADDR_PREFIX) return null;
  const [prefix] = DEP_ADDR_PREFIX.split("/");
  const net128 = v6ToBig(prefix) & (~0n << 64n);
  let host = BigInt("0x" + createHash("sha256").update(id).digest("hex").slice(0, 16)) & ((1n << 64n) - 1n);
  if (host < 0x10000n) host += 0x10000n;
  return bigToV6(net128 | host);
}
// the ledger row's declared ports ("http:8000,tcp:7777,udp:53"): only tcp:/udp:
// entries live on the dedicated address (http rides the gateway origin)
const rowPorts = (d, proto) => String(d.ports || "").split(",")
  .map((s) => s.trim()).filter((s) => s.startsWith(proto + ":"))
  .map((s) => +s.slice(proto.length + 1)).filter((p) => Number.isInteger(p) && p > 0);
function ledgerNetwork(d, status) {
  if (status !== "running" || !d.isPublic) return null;
  const address = depAddrFor(d.id); if (!address) return null;
  const tcp = rowPorts(d, "tcp"), udp = rowPorts(d, "udp");
  if (!tcp.length && !udp.length) return null;
  const net = { address };
  if (tcp.length) net.tcp = { address, ports: tcp };
  if (udp.length) net.udp = { address, ports: udp };
  return net;
}
// Shape a ledger record like the enclaves' own rows (supervisor view()), so
// dashboards/CLIs treat both alike. `ledger: true` marks the synthesis - logs
// and attestation exist only once a runner hosts it.
function ledgerView(d) {
  const rate6 = Number(d.rate);                               // per-second price, 6dp USDC
  // remaining runtime = the live lease's prepaid tail + what the balance still
  // buys (mirrors the supervisor's own view()). Balance alone reads ~0 the
  // moment a renew burns it into the lease - the owner still has minutes left.
  const leaseTail = Math.max(0, Number(d.leaseUntil) - Math.floor(Date.now() / 1000));
  const status = ledgerStatus(d);
  const network = ledgerNetwork(d, status);
  // name the serving box while a lease is live (running or claimed) — the
  // dashboard row answers "where is this app?" without another lookup
  const enclave = (status === "running" || status === "claimed") ? enclaveNameOf(d.runner) : null;
  // Before its first claim a free deployment still carries its CEILING as the
  // rate (no host has priced it yet), so the row would quote a price it will
  // never pay. Say so explicitly rather than leave clients to infer it.
  const free = hostedFree(d.owner);
  return {
    id: d.id, owner: d.owner.toLowerCase(), status, public: d.isPublic,
    ...(free ? { hostedFree: true } : {}),
    ...(enclave ? { enclave } : {}),
    ...(network ? { network } : {}),
    image: { reference: d.appRef },
    resources: { gpuShare: Number(d.gpuMilli) / 1000, cpuShare: Number(d.cpuMilli) / 1000 },
    createdAt: new Date(Number(d.createdAt) * 1000).toISOString(),
    ratePerSecondUsdc: (rate6 / 1e6).toFixed(7),
    spentUsdc: (Number(d.spent6) / 1e6).toFixed(2),
    paidUsdc: ((Number(d.balance6) + Number(d.spent6)) / 1e6).toFixed(2),
    timeRemainingSec: rate6 > 0 && !free ? leaseTail + Math.floor(Number(d.balance6) / rate6) : null,
    onchain: { contract: DEPLOYMENTS_ADDRESS, id: d.id,
               leaseUntil: Number(d.leaseUntil) ? new Date(Number(d.leaseUntil) * 1000).toISOString() : null },
    ledger: true,
  };
}
// The wallet the session token names. The relay can't VERIFY the fleet's JWTs
// (that would mean holding the enclave SECRET here, and the relay is untrusted
// by design) - and it doesn't need to: every field a ledger row carries is
// public on-chain data; the token only picks WHICH owner's public records to
// return. Enclaves keep verifying it for everything they serve.
function tokenAddress(auth) {
  // SECURITY INVARIANT (fix 11): this decodes the JWT WITHOUT verifying its
  // signature (the relay holds no key — it can't verify an ES256 token and
  // deliberately never held the old HS256 SECRET either), so the returned
  // address is UNTRUSTED. It may ONLY be used to scope which
  // wallet's PUBLIC on-chain ledger rows to return — never to authorize an
  // action or release private data. Anything sensitive stays enclave-verified.
  const m = /^Bearer\s+(.+)$/.exec(auth || ""); if (!m) return null;
  try {
    const p = JSON.parse(Buffer.from(m[1].split(".")[1], "base64url").toString());
    if (p.exp && p.exp * 1000 <= Date.now()) return null;
    return (typeof p.sub === "string" && /^0x[0-9a-fA-F]{40}$/.test(p.sub)) ? p.sub.toLowerCase() : null;
  } catch { return null; }
}

// --- availability polling -----------------------------------------------------
async function fetchJson(url, timeoutMs = 4000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try { const r = await fetch(url, { signal: ctrl.signal }); return r.ok ? JSON.parse(await readCappedText(r)) : null; }
  catch { return null; } finally { clearTimeout(t); }
}

let registry = [];                 // [{endpoint, id, repo, lastSeen}] (id = the registry's keccak256(endpoint))
let live = [];                     // registry ∩ answering, each + {availability, checkedAt}
let updatedAt = null;

async function pollRegistry() {
  try { registry = await readRegistry(); }
  catch (e) { console.error("[api-relay] registry read failed:", e.message); }
}
// Bounded /availability poll. The registry is permissionless, so an attacker can
// inflate the row count; a naive Promise.all(registry.map(...)) would open one
// concurrent socket PER row every cycle (unbounded fan-out / self-driving SSRF).
// A fixed worker pool caps in-flight probes regardless of registry size.
const AVAIL_POLL_CONCURRENCY = parseInt(process.env.AVAIL_POLL_CONCURRENCY || "32", 10);

// A RELAY is a host that carries network but sells no compute. There is no
// separate kind of box and no separate registration: any enclave may also relay
// (it says which services on /availability), and one with NO RESOURCES AT ALL is
// simply a box that only relays. That is the whole rule — the console badges it
// from this, and nothing else has to agree on a definition.
//
// It is also a safety rule, not just presentation. servingEnclaves() decides the
// fleet-minimum spec* fields and every fleet-AND capability flag, so a box
// advertising zero vCPUs inside that set would collapse the minima and turn the
// feature flags false fleet-wide. That is the metal0 sizing incident exactly,
// and a resourceless relay reproduces it unless it is excluded here.
// DECLARED zero, never merely absent. An enclave that omits these fields (an
// older build, a minimal stub) must read as a host with unknown size, not as a
// box with nothing — the failure mode of the looser test is that real capacity
// silently leaves the serving set and the fleet shrinks with nothing logged.
// Note `maxShare` is deliberately not consulted: it reports what is FREE, so a
// fully-booked enclave publishes 0 there while still being a host.
const hasNoResources = (a) =>
  !!a && a.gpu !== true && a.nodeVcpus === 0 && a.nodeRamGb === 0;
async function pollAvailability() {
  const src = registry, rows = new Array(src.length);
  let i = 0;
  const worker = async () => {
    for (;;) {
      const idx = i++;
      if (idx >= src.length) return;
      const e = src[idx];
      const a = e.tunnel ? await tunnelHub.fetchJson(e.endpoint, "/availability").catch(() => null)
                         : await fetchJson(`${e.endpoint}/availability`);
      rows[idx] = a ? { ...e, availability: a, relay: hasNoResources(a),
                        checkedAt: new Date().toISOString() } : null;
    }
  };
  await Promise.all(Array.from({ length: Math.min(AVAIL_POLL_CONCURRENCY, src.length || 1) }, worker));
  live = rows.filter(Boolean);
  updatedAt = new Date().toISOString();
}

// ---- the relay roster, and which deployment chose which relay --------------
//
// Two questions with one answer, because they have to agree. The console's
// Network tab asks "which relays can this deployment pick"; DNS asks "what
// address does <label>.app.enclave.host answer with". Both are computed here
// from the same two facts — a relay's own /availability block, and the
// deployment's on-chain options envelope — so every name the picker offers is a
// name the zone can actually resolve, and a name it can't resolve is never
// offered.
//
// A relay is named by its fleet row: an endpoint's first hostname label, or its
// tunnel name. Those names must be unambiguous, so two relays answering to the
// same name are BOTH dropped — the same rule zone 1 already applies to an
// ambiguous id prefix, and for the same reason: a choice nobody can resolve is
// worse than no choice at all.
//
// Membership here is "declares an address it relays on", NOT the `relay` badge.
// The badge means a box sells no compute, which is a presentation rule; ANY
// host may also carry network, and one that does is a legitimate choice — an
// app placed on the same box as its relay is the shortest inbound path there
// is. The badge rides along as `relayOnly` for surfaces that want to say which
// kind of box it is.
// LIVE ONLY, deliberately, and not a last-good memo. The temptation is to
// remember a relay's address across a missed poll so a blip doesn't move
// traffic; the arithmetic says otherwise, because the two errors are not the
// same size. Forgetting a relay that is actually fine costs one DNS TTL of
// traffic on the DEFAULT relay — slower, never down. Remembering one that is
// actually gone points every app that chose it at a black hole for as long as
// the memory lasts — down, not slower. A latency feature must not be able to
// cause an outage, so the doubt resolves toward the default every time.
const RELAY_NAME_RE = /^[a-z0-9][a-z0-9-]{0,62}$/;   // byte-for-byte the supervisor's envelope rule
const RELAY_SERVICES = ["sni", "tcp", "udp", "egress", "tunnelHub"];
const fleetName = (e) => String(e.name || endpointName(e.endpoint) || "").toLowerCase();
function relayRowOf(e) {
  const r = e.availability?.relay;
  if (!r || typeof r !== "object" || Array.isArray(r)) return null;
  const name = fleetName(e);
  if (!RELAY_NAME_RE.test(name)) return null;        // unnameable in an envelope = unpickable
  // A malformed or missing address leaves the row listed with address:null
  // rather than dropping it: the box is still a fleet member worth seeing, and
  // an addressless relay is simply one nothing can be pointed at (relayLabels
  // requires an address, so no deployment can end up aimed at nowhere).
  const addr  = String(r.address  || "").trim();
  const addr6 = String(r.address6 || "").trim();
  return {
    name, endpoint: e.endpoint, relayOnly: e.relay === true,
    address:  net.isIPv4(addr)  ? addr  : null,
    address6: net.isIPv6(addr6) ? addr6 : null,
    region: typeof r.region === "string" ? r.region.slice(0, 64) : null,
    ports:  typeof r.ports  === "string" ? r.ports.slice(0, 64)  : null,
    v6Prefix: typeof r.v6Prefix === "string" ? r.v6Prefix.slice(0, 64) : null,
    services: Object.fromEntries(RELAY_SERVICES.map((k) => [k, r[k] === true])),
  };
}
function relayRoster() {
  const seen = new Map(), dup = new Set();
  for (const e of live) {
    const row = relayRowOf(e);                       // any box that declares one, badge or not
    if (!row) continue;
    if (seen.has(row.name)) { dup.add(row.name); continue; }
    seen.set(row.name, row);
  }
  for (const n of dup) seen.delete(n);
  return [...seen.values()].sort((a, b) => a.name.localeCompare(b.name));
}
// The subdomain label for a deployment id — the first 8 hex chars, the same
// rule the site's appLabel and the enclaves' prefix resolution use.
const appLabelOf = (id) => String(id).slice(2, 10).toLowerCase();
// The `network.relay` a deployment's envelope names, or null. Deliberately
// forgiving where the supervisor is strict: this reads a record that is already
// on-chain, and a malformed one has already been refused by whatever tried to
// run it — there is nothing left to fail closed about, and throwing here would
// take the whole zone's answers down with one bad record.
function relayChoiceOf(configCid) {
  const s = String(configCid || "").trim();
  if (!s.startsWith("{")) return null;
  let o; try { o = JSON.parse(s); } catch { return null; }
  const n = o && o.network;
  if (!n || typeof n !== "object" || Array.isArray(n)) return null;
  const r = String(n.relay || "").trim().toLowerCase();
  return RELAY_NAME_RE.test(r) ? r : null;
}
// label -> the address its chosen relay answers on. A choice that names a relay
// the fleet no longer has, or one that doesn't splice SNI, is simply ABSENT
// here: the zone default carries it, which keeps the app reachable instead of
// pointing its name at a box that cannot serve it. The preference is a
// preference; reachability wins.
// RELAY_DEFAULT_LABEL: the relay every deployment WITHOUT an explicit
// network.relay choice resolves to. Unset preserves today's behavior (the
// zone wildcard, i.e. whichever box serves DNS) — which is measurably wrong
// whenever that box is far from the fleet: a request through a far relay
// pays the relay<->enclave distance per round trip, measured at +280ms/req
// from nan-relay (Finland) to kryptos against 58ms via the us-west relay
// beside it. The whole fleet is one region today, so one default label is
// the honest fix; when enclaves span regions this becomes a per-enclave map
// keyed by the deployment's holder.
const RELAY_DEFAULT_LABEL = (process.env.RELAY_DEFAULT_LABEL || "").trim();

function relayLabels(rows, roster) {
  const by = new Map(roster.filter((r) => r.services.sni && (r.address || r.address6)).map((r) => [r.name, r]));
  const out = {}, clash = new Set();
  for (const d of rows) {
    if (!/^0x[0-9a-f]{64}$/i.test(String(d.id || ""))) continue;
    const label = appLabelOf(d.id);
    // Two deployments whose ids share the first 8 hex chars share one app-zone
    // NAME, so there is no answer that serves both. That name is already
    // ambiguous with or without this feature; what must not happen is one of
    // them silently deciding where the other's traffic goes. Neither gets an
    // override — the zone default carries the name, exactly as it does today.
    if (label in out || clash.has(label)) { delete out[label]; clash.add(label); continue; }
    const r = by.get(relayChoiceOf(d.configCid)) || by.get(RELAY_DEFAULT_LABEL);
    if (!r) { out[label] = null; continue; }         // placeholder: claims the label, answers nothing
    out[label] = { relay: r.name,
      ...(r.address ? { a: r.address } : {}), ...(r.address6 ? { aaaa: r.address6 } : {}) };
  }
  for (const [k, v] of Object.entries(out)) if (v === null) delete out[k];
  return out;
}

// Share-based routing — same rule as enclave-discover.mjs. Deployments buy two
// shares, so callers route on the shares they intend to buy (the app's specs
// only set the MINIMUM shares — compute those from /availability's
// cardVramGb/cardTflops/nodeRamGb/nodeGflops if you're sizing from specs).
// GPU work (gpuShare > 0) needs a GPU enclave whose free card slice AND cpu
// pool both fit. CPU-only work prefers CPU-only enclaves; GPU enclaves are the
// FALLBACK, serving it out of leftover cpu pool (a tenant buying a whole card
// + 10% of the node leaves 90% rentable). maxShare = deprecated fallback for
// old enclaves.
const gpuFreeOf = (a) => a.gpuShareFree ?? (a.gpu ? a.maxShare ?? 0 : 0);
const cpuFreeOf = (a) => a.cpuShareFree ?? (a.gpu ? 0 : a.maxShare ?? 0);
function pick(want = {}) {
  const { gpuShare = 0, cpuShare = 0 } = want;
  const pool = servingEnclaves();   // only boxes that CLAIM can be routed to (a tunnel demo box has no dialable endpoint and takes no work)
  if (gpuShare > 0) {
    return pool
      .filter((e) => e.availability.gpu && gpuFreeOf(e.availability) >= gpuShare
                                        && cpuFreeOf(e.availability) >= cpuShare)
      .sort((a, b) => gpuFreeOf(b.availability) - gpuFreeOf(a.availability))[0] || null;
  }
  const fits = pool.filter((e) => cpuFreeOf(e.availability) >= cpuShare);
  const byCpuFree = (a, b) => cpuFreeOf(b.availability) - cpuFreeOf(a.availability);
  return fits.filter((e) => !e.availability.gpu).sort(byCpuFree)[0]
      || fits.filter((e) => e.availability.gpu).sort(byCpuFree)[0]
      || null;
}

// --- http ----------------------------------------------------------------------
// CORS (fix 5): the browser page (https://enclave.host) talks only to this relay,
// so WE answer preflight and stamp CORS on every response. Origin is matched
// against an ALLOWLIST (CORS_ORIGINS env, else enclave.host + www) rather than
// reflected — credentials (Authorization) are only granted to allowlisted
// origins, so a hostile page can't ride a signed-in user's session. "*" in the
// list serves the wildcard (without credentials, which browsers forbid anyway).
const corsAllowed = (origin) => !!origin && (CORS_ORIGINS.includes("*") || CORS_ORIGINS.includes(origin));
const cors = (req) => {
  const origin = req.headers.origin;
  const h = {
    "Vary": "Origin",
    "Access-Control-Allow-Methods": "GET,POST,PATCH,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": req.headers["access-control-request-headers"] || "Authorization,Content-Type",
    "Access-Control-Max-Age": "600",
  };
  if (corsAllowed(origin)) {
    h["Access-Control-Allow-Origin"] = origin;
    h["Access-Control-Allow-Credentials"] = "true";
  } else if (CORS_ORIGINS.includes("*")) {
    h["Access-Control-Allow-Origin"] = "*";                    // wildcard, no credentials
  }
  return h;
};
// Headers on the relay's OWN answers. api.enclave.host is not the apex, so it
// never inherited the site vhost's HSTS/nosniff: a caller that goes straight to
// the API (the CLI, the MCP server, a browser that has not first seen
// enclave.host and taken its includeSubDomains pin) had no pin at all. Only OUR
// responses — a proxied tenant response is the app's to shape (proxyTo passes
// its headers through), and nosniff can change how an app's own bytes render.
const OWN_HEADERS = { "X-Content-Type-Options": "nosniff",
                      "Strict-Transport-Security": "max-age=31536000; includeSubDomains" };
const json = (res, code, body, req) => {
  res.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store", ...OWN_HEADERS,
                        ...(req ? cors(req) : { "Access-Control-Allow-Origin": "*" }) });
  res.end(JSON.stringify(body));
};

// Reverse-proxy this request to `origin` (an enclave). Streams method+headers+
// body through and pipes the response back, swapping the enclave's CORS for
// ours. This is the API-gateway path: the relay terminates TLS, so it sees the
// control-plane token/body in plaintext (accepted trade for a single origin).
// Attestation fetched this way is informational — real verification stays
// client-side-direct via Tinfoil SecureClient.
// Reverse-proxy `req` to `enclaveOrigin + path`. `setCors`: on the api.enclave.host
// control-plane paths WE own CORS (swap the enclave's for ours); on an app
// subdomain the app is its own origin, so pass its headers through untouched.
function proxyTo(origin, req, res, { path = req.url, setCors = true, idleMs = 30000 } = {}) {
  if (tunnelHub.isTunnel(origin)) return proxyViaTunnel(origin, req, res, { path, setCors });
  const target = new URL(origin.replace(/\/+$/, "") + path);
  const headers = { ...req.headers, host: target.host };
  delete headers["accept-encoding"];                          // let the enclave send identity; simpler passthrough
  const lib = target.protocol === "https:" ? https : http;
  const up = lib.request(
    { hostname: target.hostname, port: target.port || (target.protocol === "https:" ? 443 : 80),
      path: target.pathname + target.search, method: req.method, headers, timeout: idleMs },
    (upRes) => {
      if (res.destroyed) return up.destroy();
      const out = {};
      for (const [k, v] of Object.entries(upRes.headers)) {
        if (/^connection$|^transfer-encoding$/i.test(k)) continue;
        if (setCors && /^access-control-/i.test(k)) continue;
        out[k] = v;
      }
      if (setCors) Object.assign(out, cors(req));
      res.writeHead(upRes.statusCode || 502, out);
      upRes.pipe(res);
      // An enclave leg that dies mid-response must take the client leg down
      // with it — pipe() forwards only a clean 'end', and an aborted upstream
      // otherwise leaves the client socket open and silent until the idle
      // timeout (the supervisor's copy of this proxy had no timeout at all,
      // and the silence was the SSE wedge; same fix there, 2026-08-16).
      // destroy(), not end(): truncation must stay visible on the wire.
      upRes.on("error", () => res.destroy());
      upRes.on("close", () => { if (!upRes.complete && !res.destroyed) res.destroy(); });
    });
  up.on("timeout", () => up.destroy(new Error("upstream timeout")));
  up.on("error", (e) => { if (res.headersSent) return res.destroy();
                          res.writeHead(502, { "Content-Type": "application/json", ...(setCors ? cors(req) : {}) });
                          res.end(JSON.stringify({ error: "upstream_error", message: e.message })); });
  // A client that dies mid-stream must take the enclave leg down with it:
  // pipe() stops the flow on destination close but never destroys its source,
  // so an abandoned SSE stream held `up` open and backpressured into the
  // enclave until the app sat parked in a write that could neither finish nor
  // fail - for llm-chat that parked write pins an inference session slot, and
  // a handful of them wedged the deployment into [sessions_busy] (live
  // 2026-08-08). The idle timeout above reaps SOME of these after idleMs, but
  // only once bytes stop moving; closing eagerly frees the slot in seconds.
  // `writableEnded` tells a hangup from a response that simply finished.
  res.on("close", () => { if (!res.writableEnded) up.destroy(); });
  req.on("error", () => up.destroy());
  req.pipe(up);
}

// Reverse-proxy `req` to a tunnel enclave (CGNAT self-hosted). Buffers the body
// (Phase-1 buffered request/response; the streaming/WS upgrade path over tunnels
// is a follow-on), forwards method+path+headers over the tunnel, writes the
// framed response back with our CORS on control-plane paths.
function proxyViaTunnel(origin, req, res, { path = req.url, setCors = true }) {
  const chunks = []; let size = 0;
  req.on("data", (c) => { size += c.length; if (size > 8 * 1024 * 1024) req.destroy(); else chunks.push(c); });
  req.on("end", async () => {
    try {
      const headers = { ...req.headers }; delete headers["accept-encoding"];
      const r = await tunnelHub.request(origin, { method: req.method, path, headers, body: chunks.length ? Buffer.concat(chunks) : null });
      const out = {};
      for (const [k, v] of Object.entries(r.headers || {})) {
        if (/^connection$|^transfer-encoding$|^content-length$/i.test(k)) continue;
        if (setCors && /^access-control-/i.test(k)) continue;
        out[k] = v;
      }
      if (setCors) Object.assign(out, cors(req));
      res.writeHead(r.status || 502, out);
      res.end(r.body);
    } catch (e) {
      if (!res.headersSent) res.writeHead(502, { "Content-Type": "application/json", ...(setCors ? cors(req) : {}) });
      res.end(JSON.stringify({ error: "tunnel_error", message: e.message }));
    }
  });
  req.on("error", () => { try { res.destroy(); } catch {} });
}

const proxied = (p) => p.startsWith("/v1/") || p === "/availability" || p === "/x" || p.startsWith("/x/");

// --- fleet-aware gateway helpers ----------------------------------------------
// Sticky enclave for non-deployment-scoped calls (auth nonces are per-enclave
// state, so /v1/auth/* must land on one box consistently). A GPU enclave is
// preferred because it serves the full API surface (/v1/gpu, card pricing).
const sticky = () =>
     live.filter((e) => e.availability.gpu).sort((a, b) => a.endpoint.localeCompare(b.endpoint))[0]
  || live.slice().sort((a, b) => a.endpoint.localeCompare(b.endpoint))[0] || null;

// Which enclave owns a deployment id — probed once, cached. Two probes:
// /x/:id (unauth; 404 = not here) covers the data path, and the /v1 record
// itself (with the caller's token; 200 = here) covers control-plane calls even
// after the instance is gone (a terminated record still exists on its enclave).
//
// ORDER MATTERS: the LEDGER is consulted before this cache, never after. The
// cache exists to skip the fan-out PROBE, which is expensive; it was never
// meant to outrank on-chain truth, and outranking it is a bug with a 5-minute
// blast radius. A Move rewrites the runner in one transaction, so the instant
// it lands every cached entry for that id names the box that just gave it up —
// and that box answers the control plane "No such deployment." until the entry
// ages out (found 2026-07-28, right after a metal0 -> kryptos -> metal0 move).
// Reading the ledger first costs nothing a request didn't already pay:
// ledgerRows() is itself cached (LEDGER_TTL_MS), so this is an in-memory scan,
// and runnerEndpointOf declines anything the chain can't answer for (unleased,
// ambiguous prefix, non-ledger dep_ id) — which is exactly when the cache and
// then the probe should get their turn.
const OWNER = new Map();                                     // dep id -> { endpoint, at }
const OWNER_TTL_MS = 5 * 60_000;
const OWNER_NEG = new Map();                                 // dep id -> at (miss, short-lived; fix 2)
const OWNER_NEG_TTL_MS = 10_000;
const ownerCached = (id) => {
  const hit = OWNER.get(id);
  return (hit && Date.now() - hit.at < OWNER_TTL_MS && live.some((e) => e.endpoint === hit.endpoint))
    ? hit.endpoint : null;
};
const ownerNegRecent = (id) => { const at = OWNER_NEG.get(id); return at != null && Date.now() - at < OWNER_NEG_TTL_MS; };
const ownerLearn = (id, endpoint) => { if (id && endpoint) { OWNER.set(id, { endpoint, at: Date.now() }); OWNER_NEG.delete(id); } };
async function probe(url, init) {
  // tunnel endpoints can't be fetch()ed — round-trip the hub instead. Callers
  // only read .status, which is all the hub reply carries anyway.
  if (tunnelHub.isTunnel(url)) {
    try {
      const s = String(url), cut = s.indexOf("/", "tunnel://".length);
      const origin = cut < 0 ? s : s.slice(0, cut), path = cut < 0 ? "/" : s.slice(cut);
      return await tunnelHub.request(origin, { method: (init && init.method) || "GET", path, headers: (init && init.headers) || {} });
    } catch { return null; }
  }
  const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), 4000);
  try { return await fetch(url, { ...init, signal: ctrl.signal }); }
  catch { return null; } finally { clearTimeout(t); }
}
// SECURITY (fix 1c / B3): prefer the deployment's ON-CHAIN runner (the enclave
// that actually claimed the lease) over "first endpoint answering non-404", so
// a hostile enclave can't hijack another tenant's /x traffic by answering for
// its id. Returns the runner's endpoint only when it's a known, in-fleet enclave
// (which is already https-/operator-filtered); null (=> probe fallback) on any
// uncertainty, so a valid deployment never becomes unroutable.
async function runnerEndpointOf(id) {
  const h = String(id).toLowerCase();
  if (!/^0x[0-9a-f]{8,64}$/.test(h)) return null;           // dep_/non-onchain ids: probe path
  let rows; try { rows = await ledgerRows(); } catch { return null; }
  const hits = rows.filter((d) => String(d.id).toLowerCase().startsWith(h));
  if (hits.length !== 1) return null;                       // unknown/ambiguous -> fall back
  const d = hits[0];
  if (ZERO32.test(String(d.runner)) || Number(d.leaseUntil) * 1000 <= Date.now()) return null;
  const runner = String(d.runner).toLowerCase();
  const e = live.find((x) => x.id && x.id.toLowerCase() === runner);
  return e ? e.endpoint : null;
}
// SECURITY: an app subdomain is a deployment id PREFIX — canonically the first
// 8 hex chars, which is 32 bits. Ids are keccak256(creator, nonce), so a
// collision is not a birthday accident to wave off: an attacker grinds
// candidate creator addresses OFFLINE until one's next id shares a victim's
// prefix (seconds of hashing, no gas), then creates that one deployment. Both
// records then answer to <prefix>.<APP_DOMAIN>. runnerEndpointOf already
// declines an ambiguous prefix, but declining used to mean "fall back to the
// fan-out probe", and the probe takes the FIRST enclave that answers — so the
// ground twin could win the victim's subdomain and be cached as its owner for
// five minutes. A prefix the LEDGER says names two deployments names neither:
// refuse it outright rather than let a race pick.
async function prefixAmbiguous(id) {
  const h = String(id).toLowerCase();
  if (!/^0x[0-9a-f]{8,63}$/.test(h)) return false;          // full id or non-ledger shape: nothing to confuse
  let rows; try { rows = await ledgerRows(); } catch { return false; }
  return rows.filter((d) => String(d.id).toLowerCase().startsWith(h)).length > 1;
}
async function xOwnerOf(id) {                                // data-path resolve (no auth needed)
  const byRunner = await runnerEndpointOf(id);              // fix 1c: on-chain claimer wins
  if (byRunner) { ownerLearn(id, byRunner); return byRunner; }
  const hit = ownerCached(id); if (hit) return hit;
  if (await prefixAmbiguous(id)) return null;
  if (ownerNegRecent(id)) return null;                      // recent miss: don't re-fan-out (fix 2)
  if (!fanoutReserve(live.length)) return null;             // global fan-out cap (fix 2)
  let ep = null;
  try {
    const found = await Promise.all(live.map(async (e) =>
      (r => r && r.status !== 404 ? e.endpoint : null)(await probe(`${e.endpoint}/x/${encodeURIComponent(id)}`, { method: "HEAD" }))));
    ep = found.find(Boolean) || null;
  } finally { fanoutRelease(live.length); }
  if (ep) ownerLearn(id, ep); else OWNER_NEG.set(id, Date.now());
  return ep;
}
async function v1OwnerOf(id, auth) {                         // control-plane probe (caller's token)
  const byRunner = await runnerEndpointOf(id);              // fix 1c: on-chain claimer wins
  if (byRunner) { ownerLearn(id, byRunner); return byRunner; }
  const hit = ownerCached(id); if (hit) return hit;
  if (await prefixAmbiguous(id)) return null;               // a prefix naming two deployments names neither
  let ep = null;
  if (fanoutReserve(live.length)) {
    try {
      const found = await Promise.all(live.map(async (e) => {
        const r = await probe(`${e.endpoint}/v1/deployments/${encodeURIComponent(id)}`,
                              { headers: auth ? { Authorization: auth, Accept: "application/json" } : { Accept: "application/json" } });
        return r && r.status === 200 ? e.endpoint : null;
      }));
      ep = found.find(Boolean) || null;
    } finally { fanoutRelease(live.length); }
  }
  if (ep) ownerLearn(id, ep);
  return ep || xOwnerOf(id);                                 // fall back to the data-path probe (e.g. expired token)
}

function readBody(req, max = 262144) {
  return new Promise((resolve, reject) => {
    const chunks = []; let n = 0;
    req.on("data", (ch) => { n += ch.length; if (n > max) { req.destroy(); reject(new Error("body too large")); } else chunks.push(ch); });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}
// Buffered forward (vs proxyTo's streaming): used where the relay needs to SEE
// the body or the response — placement reads the create request's shares, and
// the create/list responses teach the owner cache.
async function forward(origin, req, body, path = req.url) {
  const headers = {};
  for (const [k, v] of Object.entries(req.headers))
    if (!/^(host|connection|content-length|transfer-encoding|accept-encoding)$/i.test(k)) headers[k] = v;
  if (tunnelHub.isTunnel(origin)) {          // buffered hub round-trip, same shape + cap as the fetch leg
    const r = await tunnelHub.request(origin, { method: req.method, path, headers, body: body && body.length ? body : undefined });
    if (r.body && r.body.length > MAX_BODY_BYTES) throw new Error("response body too large");
    const ct = Object.entries(r.headers || {}).find(([k]) => k.toLowerCase() === "content-type");
    return { status: r.status, contentType: ct ? ct[1] : null, text: (r.body || Buffer.alloc(0)).toString("utf8") };
  }
  const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), 30000);
  try {
    const r = await fetch(origin.replace(/\/+$/, "") + path,
      { method: req.method, headers, body: body && body.length ? body : undefined, signal: ctrl.signal });
    return { status: r.status, contentType: r.headers.get("content-type"), text: await readCappedText(r) };
  } finally { clearTimeout(t); }
}
function sendForwarded(res, r, req) {
  // a buffered CONTROL-plane answer (placement, list merge): ours to stamp, same
  // as json() — never the /x tenant path, which streams through proxyTo
  res.writeHead(r.status, { "Cache-Control": "no-store", ...(r.contentType ? { "Content-Type": r.contentType } : {}), ...OWN_HEADERS, ...cors(req) });
  res.end(r.text);
}

// Fleet /availability: the deploy-dial view. Best single-card slice across GPU
// enclaves + best node pool across ALL enclaves (they can be different boxes —
// that is the point of the two-pool model). gpuEnclaveCpuShareFree is the cpu
// pool on the best GPU enclave: a GPU deployment's cpuShare must fit THERE.
//
// spec* fields are for SIZING app specs into minimum shares, and they are the
// fleet-wide MINIMA of the hardware numbers the runners themselves divide by
// in their claim gate. The plain cardVramGb/nodeRamGb describe the BEST box
// (capacity view) — a dial floor computed on the biggest card under-sells on
// every smaller one, and a deployment below a runner's minimum is unclaimable
// there forever (created shares are immutable). Sizing against the minima
// keeps a bought share valid on EVERY live enclave.
// The subset of live enclaves that actually CLAIM ledger work — the only ones
// whose hardware can be BOUGHT. Sizing minima, capacity views, capability ANDs
// and routing all compute over this set: a present-but-not-claiming box (the
// metal demo enclave, CLAIM_ENABLED=0) still shows in the listing and answers
// over its tunnel, but it must never set the fleet's sizing floor (its 3 GB
// node made every CPU app's minimum share balloon ~17x, observed 2026-07-25)
// nor inflate the buyable-capacity totals. `claimEnabled` is explicit from
// newer supervisors; hosted (non-tunnel) enclaves predate the field and have
// always claimed, tunnel boxes count only when they SAY they claim (Phase C
// sellers with registryKey set report true).
function servingEnclaves() {
  // A relay is never in this set. It says so itself (claimEnabled:false), but
  // the row is checked here too: this function decides the fleet-minimum spec*
  // fields and every fleet-AND capability flag, so one box that reports no vCPUs
  // slipping in is a fleet-wide sizing and feature outage. Belt and braces on
  // the one filter that has already failed that way once.
  return live.filter((e) => !e.relay && (e.availability?.claimEnabled === true
    || (e.availability?.claimEnabled == null && !e.tunnel)));
}
function aggregateAvailability() {
  const serving = servingEnclaves();
  const gpus = serving.filter((e) => e.availability.gpu);
  const g = gpus.slice()
    .sort((a, b) => gpuFreeOf(b.availability) - gpuFreeOf(a.availability))[0]?.availability || null;
  const c = serving.slice()
    .sort((a, b) => cpuFreeOf(b.availability) - cpuFreeOf(a.availability))[0]?.availability || null;
  const minOf = (rows, field) => rows.reduce((m, e) => {
    const v = Number(e.availability?.[field]);
    return Number.isFinite(v) && v > 0 ? (m > 0 ? Math.min(m, v) : v) : m;
  }, 0);
  return {
    aggregate: true, enclaves: live.length, gpu: !!g, type: g ? "gpu" : "cpu",
    gpuShareFree: g ? gpuFreeOf(g) : 0, cpuShareFree: c ? cpuFreeOf(c) : 0,
    gpuEnclaveCpuShareFree: g ? cpuFreeOf(g) : 0,
    maxShare: g ? gpuFreeOf(g) : (c ? cpuFreeOf(c) : 0),     // deprecated alias, same rule as the enclaves'
    vramFreeGb: g ? g.vramFreeGb ?? 0 : 0, gpuTflopsFree: g ? g.gpuTflopsFree ?? 0 : 0,
    smFree: g ? g.smFree ?? 0 : 0, smTotal: g ? g.smTotal ?? 0 : 0,
    cardVramGb: g ? g.cardVramGb ?? 0 : 0, cardTflops: g ? g.cardTflops ?? 0 : 0, cards: g ? g.cards ?? 0 : 0,
    vcpusFree: c ? c.vcpusFree ?? 0 : 0, ramGbFree: c ? c.ramGbFree ?? 0 : 0, cpuGflopsFree: c ? c.cpuGflopsFree ?? 0 : 0,
    nodeVcpus: c ? c.nodeVcpus ?? 0 : 0, nodeRamGb: c ? c.nodeRamGb ?? 0 : 0, nodeGflops: c ? c.nodeGflops ?? 0 : 0,
    specCardVramGb: minOf(gpus, "cardVramGb"), specCardTflops: minOf(gpus, "cardTflops"),
    specNodeVcpus: minOf(serving, "nodeVcpus"), specNodeRamGb: minOf(serving, "nodeRamGb"), specNodeGflops: minOf(serving, "nodeGflops"),
    // WHY the cpu pool is small, carried up from the enclave whose numbers we
    // just quoted. The aggregate reported the folded cpuShareFree and dropped
    // every term behind it, so a box reading 65% free with one resident model
    // was unexplainable without per-enclave access: ramNnResidentMb is usually
    // the whole answer, and sharePoolFree says how much is actually SOLD.
    // instanceSweep is the reclaimer's own last word - a stuck sweep is the
    // difference between "in use" and "leaking", and it must be sayable here.
    ...(c && c.ramBudgetMb ? { ramBudgetMb: c.ramBudgetMb, ramCommittedMb: c.ramCommittedMb,
                               ramFreeMb: c.ramFreeMb, sharePoolFree: c.sharePoolFree,
                               ...(c.ramNnResidentMb ? { ramNnResidentMb: c.ramNnResidentMb } : {}) } : {}),
    ...(c && c.instanceSweep ? { instanceSweep: c.instanceSweep } : {}),
    ...(c && c.tenantVouch ? { tenantVouch: c.tenantVouch } : {}),
    ...(c && c.tenantLease ? { tenantLease: c.tenantLease } : {}),
    // deployment-options capability (per-IP rate limit / WAF): true only when
    // EVERY live enclave enforces the envelope — any runner may claim any
    // deployment, so a mixed fleet would strand protected deploys on old
    // runners ("configCid retired" refusal). Same fleet-minimum rule as spec*.
    waf: serving.length > 0 && serving.every((e) => e.availability?.waf === true),
    // envelope `config` namespace (per-deployment app-config override): same
    // fleet-AND — a mixed fleet would strand an overridden deploy on a runner
    // that refuses the namespace, so the console only unlocks the box on true
    configOverride: serving.length > 0 && serving.every((e) => e.availability?.configOverride === true),
    // envelope `configCid` namespace: the SAME override, split the way catalog
    // rev 7 splits a version's config — bulk at a pinned CID, the inline field
    // demoted to the routing manifest (volumes). It exists because the whole
    // envelope shares one 4096-byte ledger field, so an app whose config is
    // larger than that has no override without it. Same fleet-AND, and it is
    // strictly narrower than configOverride: a runner that knows the split
    // necessarily knows the inline form, never the reverse.
    configCidOverride: serving.length > 0 && serving.every((e) => e.availability?.configCidOverride === true),
    // setConfig reaches LIVE deployments (audit envelope watch: waf swaps in
    // place, a config change restarts the app on the new value): fleet-AND -
    // on false an edit still lands on-chain but only applies at re-claim
    configEdit: serving.length > 0 && serving.every((e) => e.availability?.configEdit === true),
    // setShares reaches LIVE deployments (audit share watch: re-slice +
    // restart in place, or hand the lease to a box that fits): fleet-AND —
    // on false a resize tx would change the BILLING while the served slice
    // silently didn't, so clients refuse to send it against an older fleet
    shareResize: serving.length > 0 && serving.every((e) => e.availability?.shareResize === true),
    // {"gpu":{"optional":true}} — a GPU-dialled deployment may fall back to a
    // CPU-only enclave rather than queue for a card: fleet-AND, and strictly so.
    // A runner that predates the namespace REFUSES the whole envelope as
    // unknown (deliberately - options are never silently dropped), which would
    // strand the deployment unclaimable on that box. So the console must not
    // offer the control until every live runner knows the word.
    gpuOptional: serving.length > 0 && serving.every((e) => e.availability?.gpuOptional === true),
    // per-deployment relay choice (the envelope's `network` namespace). Nothing
    // in a CVM acts on it — DNS does — but the ENVELOPE is fail-closed, so a
    // deployment carrying {"network":…} that lands on a runner predating it is
    // refused outright. Same fleet-AND rule, same reason: the console must keep
    // the Network tab hidden until every live runner knows the word.
    networkOptions: serving.length > 0 && serving.every((e) => e.availability?.networkOptions === true),
    // per-deployment secrets (relay-stored, injected as guest env by the lease
    // holder): needs BOTH this relay configured (SECRETS_KEY + data dir) and a
    // fleet-AND of runners that fetch+inject — a mixed fleet would run the same
    // app with secrets on one runner and without them after a lease migration
    secrets: secretsEnabled() && serving.length > 0 && serving.every((e) => e.availability?.secrets === true),
    // $NAME placeholders in config strings resolving from those secrets at
    // launch — a build refinement on top of `secrets`, same fleet-AND
    secretsInConfig: secretsEnabled() && serving.length > 0 && serving.every((e) => e.availability?.secretsInConfig === true),
    // customer-owned hostnames (relay/domains.js): needs this relay configured
    // AND a fleet-AND of runners that fetch the names and mint their
    // certificates. Strictly AND-ed: a lease migrating to a runner that doesn't
    // know the feature would leave the customer's own domain refusing
    // handshakes with nothing on the dashboard to explain it, so the console
    // only offers the section when every live runner can honour it.
    customDomains: domainsEnabled() && serving.length > 0 && serving.every((e) => e.availability?.customDomains === true),
    // publisher dev-mode: runners admit PENDING catalog versions for PRIVATE
    // deployments (public deploys of pending versions stay refused). Fleet-AND —
    // on false a pending-version deploy would sit Queued forever on old runners,
    // so clients only offer the option when every live runner honors it
    devDeploy: serving.length > 0 && serving.every((e) => e.availability?.devDeploy === true),
    // per-deployment rate caps (ledger rev 8): runners price claims off their
    // own registry entry and treat a cap-blocked renew as "stop at lease end".
    // Fleet-AND — against an older runner a lowered cap would just look like a
    // stuck renewal, so clients only offer cap edits when every live runner
    // handles it
    rateCap: serving.length > 0 && serving.every((e) => e.availability?.rateCap === true),
    // Every live runner proves the time it bills for (ledger rev 9): it signs
    // block-anchored checkpoints from a key minted inside its own CVM, and the
    // ledger pays it only for service it proved. AND-ed, not OR-ed: a buyer can
    // only be told "the hosts here are held to account" if every host is.
    proofOfTime: serving.length > 0 && serving.every((e) => e.availability?.proofOfTime === true),
    // WASIp3 (component-model async) serving: each runner probes its own
    // wasmtime for `-S p3` and reports per box; a version publishes `wasi:
    // "0.3"` in its config and only p3-capable boxes claim it. Fleet-AND for
    // the same reason as devDeploy — on false a p3 deploy could sit Queued
    // until a capable box has room, so clients warn (and the console only
    // offers p3 publishes cleanly) when every claiming runner serves it.
    // Per-box truth stays visible in the target list for the canary flow:
    // deploying pinned to a p3-capable box is legitimate while the AND is
    // still false.
    p3: serving.length > 0 && serving.every((e) => e.availability?.p3 === true),
    // Cooperative threads (🧵), one capability over: `threads: true` versions
    // route only to boxes whose engine passed the thread.new-indirect compile
    // probe (coopThreads on each runner's availability). Same fleet-AND
    // reasoning as p3, same per-box canary escape hatch.
    coopThreads: serving.length > 0 && serving.every((e) => e.availability?.coopThreads === true),
    // Shared-everything threads (⚡), one more capability over: `set: true`
    // versions route only to boxes whose engine passed the thread.spawn-indirect
    // compile probe (`set` on each runner's availability). Same fleet-AND and
    // per-box canary escape hatch as p3/coopThreads.
    set: serving.length > 0 && serving.every((e) => e.availability?.set === true),
    // wasm64 (memory64) core modules — the >4 GiB guests: `mem64: true`
    // versions route only to boxes whose engine passed the flagless memory64
    // compile probe (`mem64` on each runner's availability). Same fleet-AND
    // and per-box canary escape hatch as the rest.
    mem64: serving.length > 0 && serving.every((e) => e.availability?.mem64 === true),
    // Catalog rev-7 large configs, same fleet-AND and same per-box canary
    // escape hatch: a version whose config lives at a CID routes only to boxes
    // that fetch and hash-verify it. A box without it REFUSES the claim rather
    // than serving the routing manifest as the config — correct, but it means
    // an un-rolled-out fleet leaves such a deployment Queued with its funding
    // tied up, so clients must be able to see this before they create one.
    configCid: serving.length > 0 && serving.every((e) => e.availability?.configCid === true),
    // The smallest config the whole fleet will accept — a config over this
    // publishes fine and then fails every launch, so the publish UI sizes its
    // own check off the fleet rather than a hardcoded guess.
    configMaxBytes: serving.length > 0
      ? Math.min(...serving.map((e) => Number(e.availability?.configMaxBytes) || 0)) : 0,
    // the CHEAPEST posted price across the claiming fleet, USDC 6dp/sec for a
    // whole node / whole card. Each enclave sets its own (registry entry), so
    // "what does this cost" is a fleet-minimum question now, not a contract
    // constant. Clients quote "from $X/hr" off these and default a new
    // deployment's rate cap to the box it actually picked.
    ...cheapestAsk(serving, gpus),
    // attached model volumes across the fleet (Modelwrap), deduped by name -
    // each carries `enclaves`: which endpoints can mount it (placement matters,
    // a volume only lives where its enclave declares it)
    volumes: fleetVolumes(),
    source: "api-relay", updatedAt,
  };
}

// The floor price the fleet can serve at: min over the CLAIMING enclaves of
// each axis, taken independently (the cheapest node and the cheapest card need
// not be the same box — a buyer picks per deployment, and rankEnclavesFor
// shows them the real per-box numbers). Omitted entirely when no live enclave
// posts a price: an old fleet has none, and a made-up number would quote a
// price nobody charges.
function cheapestAsk(serving, gpus) {
  const min = (rows, key) => {
    const vals = rows.map((e) => Number(e.availability?.[key])).filter((v) => Number.isFinite(v) && v > 0);
    return vals.length ? Math.min(...vals) : null;
  };
  const cpu = min(serving, "askCpuPricePerSec6"), gpu = min(gpus, "askGpuPricePerSec6");
  return { ...(cpu ? { cheapestCpuPricePerSec6: cpu } : {}), ...(gpu ? { cheapestGpuPricePerSec6: gpu } : {}) };
}

// Union of every live enclave's advertised model volumes, keyed by name, each
// annotated with the endpoints that carry it.
const MAX_VOLUMES_PER_ENCLAVE = 256;                        // guard a hostile /availability (fix 8)
function fleetVolumes() {
  const byName = new Map();
  for (const e of live) {
    const vols = e.availability?.volumes;
    for (const v of (Array.isArray(vols) ? vols.slice(0, MAX_VOLUMES_PER_ENCLAVE) : [])) {
      if (!v || !v.name) continue;
      const cur = byName.get(v.name) || { name: v.name, bytes: v.bytes || 0, onnx: !!v.onnx, gguf: !!v.gguf, sd: !!v.sd, endpoints: [] };
      cur.bytes = Math.max(cur.bytes, v.bytes || 0);
      cur.onnx = cur.onnx || !!v.onnx;
      cur.gguf = cur.gguf || !!v.gguf;
      cur.sd = cur.sd || !!v.sd;
      if (!cur.endpoints.includes(e.endpoint)) cur.endpoints.push(e.endpoint);
      byName.set(v.name, cur);
    }
  }
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
}

const DEP_PATH_RE = /^\/v1\/deployments\/([A-Za-z0-9_-]+)(?:\/|$)/;
const X_PATH_RE   = /^\/x\/([A-Za-z0-9_-]+)(?:\/|$)/;

async function gateway(u, req, res) {
  const p = u.pathname;

  // Ledger-backed reads FIRST: the wallet's list and bare record reads answer
  // from EnclaveDeployments even with zero live enclaves - on-chain work is
  // real whether or not anything currently hosts it.
  if (p === "/v1/deployments" && req.method === "GET") return listDeployments(u, req, res);
  const bare = p.match(/^\/v1\/deployments\/([A-Za-z0-9_-]+)$/);
  if (bare && req.method === "GET") return getDeployment(bare[1], u, req, res);

  // The relay roster + the per-deployment choices made against it. Sits with
  // the ledger reads ABOVE the fleet-down guard deliberately: this is DNS's
  // input, and the app zone must not lose its per-deployment answers because
  // the enclaves blinked. `relays: []` with zero live relays is the honest
  // answer — every name falls back to the zone default, which is where it was.
  if (p === "/v1/relays" && req.method === "GET") {
    const relays = relayRoster();
    let rows;
    try { rows = await ledgerRows(); }
    catch (e) {
      // 503, NOT 200-with-empty-labels: an empty map is indistinguishable from
      // "nobody chose a relay", and a consumer that believed it would move every
      // deployment back to the default relay on one bad RPC read. The error
      // makes the caller keep its last good map instead.
      return json(res, 503, { error: "ledger_unavailable", relays, updatedAt,
        message: "Could not read the deployments ledger just now; the relay roster is current but the per-deployment choices are not." }, req);
    }
    return json(res, 200, { updatedAt, relays, labels: relayLabels(rows, relays) }, req);
  }

  if (!live.length) {
    // fleet-down answers that tell the truth about WHAT is down: the API
    // front door (this relay) is healthy - only enclave-served things are out
    if (p === "/v1/health")
      return json(res, 200, { ok: true, enclaves: 0, of: registry.length, gateway: "api-relay",
        note: "API relay up; no live enclaves right now - funded deployments queue on the ledger and are claimed when one returns.", updatedAt }, req);
    if (p.startsWith("/v1/auth/"))
      return json(res, 503, { error: "auth_unavailable",
        message: "Sign-in needs a live enclave (SIWE nonces and session tokens are enclave-issued; this relay deliberately can't mint them) and none is up right now. Everything wallet-signed still works without a session: deploying, funding, top-ups, terminate, and your deployment list.", updatedAt }, req);
    return json(res, 503, { error: "no_capacity", message: "No live enclaves.", updatedAt }, req);
  }
  if (p === "/availability") return json(res, 200, aggregateAvailability(), req);

  const dep = p.match(DEP_PATH_RE), x = p.match(X_PATH_RE);
  if (dep || x) {
    const id = (dep || x)[1];
    // rate-limit only the misses (the fan-out probe); cached routes stay fast (fix 2)
    if (!ownerCached(id) && !rlMiss(clientIp(req)))
      return json(res, 429, { error: "rate_limited", message: "Too many deployment lookups; retry shortly.", updatedAt }, req);
    const owner = dep ? await v1OwnerOf(id, req.headers.authorization) : await xOwnerOf(id);
    if (!owner) return json(res, 404, { error: "not_found", message: `No live enclave has ${id}.`, updatedAt }, req);
    // Tenant data path: generous idle window. A model-serving app's first
    // request can sit silent for the length of a session init (e.g. wasi-nn
    // loading a 100MB+ model onto the GPU under CC); 30s cut those off and
    // the abandoned sync load wedged the tenant's runtime threads.
    return proxyTo(owner, req, res, { idleMs: 180000 });
  }

  if (p === "/v1/apps/upload-token" && req.method === "POST") {
    // Authorize a wasm pin: the publisher signs `enclave-upload:<sha256hex>:<expiry>`
    // with their wallet; we recover the address (viem), rate-limit per wallet, and
    // mint an HMAC token the add-gateway verifies before it pins (the gateway does
    // NO EC crypto and never sees the fleet secret). Closes the open-pin storage DoS.
    if (!UPLOAD_KEY) return json(res, 503, { error: "upload_disabled", message: "Signed uploads are not configured here." }, req);
    let raw; try { raw = await readBody(req, 8192); } catch (e) { return json(res, 413, { error: "too_large", message: e.message }, req); }
    let b; try { b = JSON.parse(raw.toString() || "{}"); } catch { return json(res, 400, { error: "bad_json", message: "Body must be JSON." }, req); }
    const hash = String(b.hash || "").toLowerCase().replace(/^0x/, "");
    const expiry = parseInt(b.expiry, 10);
    const signature = String(b.signature || "");
    const now = Math.floor(Date.now() / 1000);
    if (!/^[0-9a-f]{64}$/.test(hash)) return json(res, 422, { error: "bad_hash", message: "hash must be the 32-byte sha256 hex of the upload." }, req);
    if (!Number.isFinite(expiry) || expiry < now || expiry > now + 600) return json(res, 422, { error: "bad_expiry", message: "expiry must be a unix time within the next 10 minutes." }, req);
    if (!/^0x[0-9a-fA-F]{130}$/.test(signature)) return json(res, 422, { error: "bad_sig", message: "signature must be a 65-byte personal_sign hex." }, req);
    let address;
    try {
      const { recoverMessageAddress } = await import("viem");
      address = (await recoverMessageAddress({ message: `enclave-upload:${hash}:${expiry}`, signature })).toLowerCase();
    } catch (e) { return json(res, 400, { error: "bad_sig", message: "Could not recover the signer: " + (e.shortMessage || e.message) }, req); }
    if (!rlUpload(address)) return json(res, 429, { error: "rate_limited", message: "Too many upload authorizations from this wallet; retry later." }, req);
    const token = createHmac("sha256", UPLOAD_KEY).update(`${address}:${hash}:${expiry}`).digest("hex");
    return json(res, 200, { token, address, expiry }, req);
  }

  // Dealt pads (relay/pads.mjs, shielded/dealer/PLAN.md): the ledger key, a
  // pVM's seed, and reserve-before-use windows, each authenticated by the
  // attested tunnel's own transport key rather than an account session.
  if (p.startsWith("/v1/pads/")) {
    const L = padsLedger();
    if (!L) return json(res, 503, { error: "pads_disabled", message: "no data dir for the pads ledger" }, req);
    if (p === "/v1/pads/key" && req.method === "GET") return json(res, 200, { key: L.key(), epoch: PADS_EPOCH }, req);
    if (p === "/v1/pads/ledger" && req.method === "GET") {
      const m = L.mark(u.searchParams.get("seed_id") || "");
      return m ? json(res, 200, m, req) : json(res, 404, { error: "unknown_seed" }, req);
    }
    if ((p === "/v1/pads/seed" || p === "/v1/pads/reserve") && req.method === "POST") {
      let body;
      try { body = JSON.parse((await readBody(req, 8192)).toString("utf8") || "{}"); }
      catch (e) { return json(res, e.message === "body too large" ? 413 : 400, { error: "bad_json", message: e.message }, req); }
      const r = p === "/v1/pads/seed" ? L.seed(body || {}) : L.reserve(body || {});
      return json(res, r.status, r.body, req);
    }
    return json(res, 404, { error: "not_found" }, req);
  }

  if (p === "/v1/claim-hint" && req.method === "POST") {
    // Fan the hint to the CLAIMING enclaves (a non-claiming box can only
    // decline): CPU-only enclaves take CPU work immediately, GPU enclaves
    // skip their CPU-first grace when hinted, and the EnclaveDeployments
    // contract referees any race (the loser's claim tx reverts; gas is
    // cents). Enclaves answer fast - the actual claim runs in their
    // background; deployers watch the ledger for the runner.
    // The body may carry {enclave: "<name>"} - the deploy console's target
    // pick: the hint then goes ONLY to that box, giving it first crack (the
    // 60s sweeps referee everything else). An unknown name falls back to the
    // full fan-out - a hint must never strand a funded deploy.
    // unauthenticated fan-out amplifier: per-source rate limit + global in-flight
    // cap (fix 2), and the response body from each enclave is size-capped (fix 8).
    if (!rlHint(clientIp(req)))
      return json(res, 429, { error: "rate_limited", message: "Too many claim hints; retry shortly." }, req);
    let body; try { body = await readBody(req); } catch (e) { return json(res, 413, { error: "too_large", message: e.message }, req); }
    let prefer = "";
    try { prefer = String(JSON.parse(body.toString() || "{}").enclave || "").trim().toLowerCase(); } catch {}
    const serving = servingEnclaves();
    const match = prefer ? serving.filter((e) =>
      String(e.name || "").toLowerCase() === prefer || String(e.endpoint || "").toLowerCase() === prefer) : [];
    const pool = match.length ? match : serving;
    if (!fanoutReserve(pool.length))
      return json(res, 503, { accepted: false, reason: "Relay busy (fan-out cap); the sweep will still pick the deployment up." }, req);
    let results;
    try {
      results = await Promise.all(pool.map(async (e) => {
        try {
          // tunnel-attached enclaves (Phase C sellers) have no dialable
          // endpoint - the hint rides the hub, like every relay->tunnel call
          if (tunnelHub.isTunnel(e.endpoint)) {
            const r = await tunnelHub.request(e.endpoint, { method: "POST", path: "/v1/claim-hint",
              headers: { "content-type": "application/json" }, body });
            return JSON.parse(r.body.toString("utf8").slice(0, 4096));
          }
          const r = await fetch(e.endpoint + "/v1/claim-hint",
            { method: "POST", headers: { "content-type": "application/json" },
              body, signal: AbortSignal.timeout(15_000) });
          return JSON.parse(await readCappedText(r));
        } catch { return null; }
      }));
    } finally { fanoutRelease(pool.length); }
    const best = results.find(r => r && r.accepted) || results.find(Boolean)
              || { accepted: false, reason: "No live enclave answered the hint; the sweep will still pick the deployment up." };
    return json(res, 200, best, req);
  }

  if (p === "/v1/deployments" && req.method === "POST") {    // placement: the ONE routing decision
    let body; try { body = await readBody(req); } catch (e) { return json(res, 413, { error: "too_large", message: e.message }, req); }
    let want = {};
    try { const r = JSON.parse(body.toString() || "{}").resources || {};
          want = { gpuShare: Number(r.gpuShare) || 0, cpuShare: Number(r.cpuShare) || 0 }; } catch {}
    const c = pick(want);
    if (!c) return json(res, 409, { error: "no_capacity",
      message: `No live enclave has gpuShare >= ${want.gpuShare} and cpuShare >= ${want.cpuShare} free.`, updatedAt }, req);
    const r = await forward(c.endpoint, req, body).catch((e) => ({ status: 502, contentType: "application/json",
      text: JSON.stringify({ error: "upstream_error", message: e.message }) }));
    if (r.status === 201) { try { ownerLearn(JSON.parse(r.text).id, c.endpoint); } catch {} }
    return sendForwarded(res, r, req);
  }

  // Auth is enclave-scoped state on BOTH halves: the SIWE nonce, and the
  // session token itself — each enclave signs with its own in-enclave ES256
  // key and verifies ONLY its own kid (supervisor verifySessionToken), on
  // purpose, so that no box can mint a session another box will honor. That
  // makes "which enclave signed me in" load-bearing: a session from the sticky
  // box is rejected by every other one, and every owner-authenticated call on
  // a deployment hosted elsewhere 401s "Missing or invalid session" (found
  // 2026-07-27 — Restart/logs/attestation/Move on a metal0-hosted deployment
  // all failed this way). So let the CLIENT say which enclave should mint it:
  // ?enclave=<name> pins the whole SIWE round trip to that box. Unknown name
  // falls back to sticky rather than failing — a pin is an optimization, and
  // signing in against the wrong box is recoverable while not signing in isn't.
  const pin = String(u.searchParams.get("enclave") || "").trim().toLowerCase();
  const pinned = pin && p.startsWith("/v1/auth/")
    ? live.find((e) => String(e.name || "").toLowerCase() === pin
                    || String(e.endpoint || "").toLowerCase() === pin) : null;
  const c = pinned || sticky();                              // auth, pricing, version, attestation, ...
  return proxyTo(c.endpoint, req, res);
}

// The address to scope PUBLIC ledger reads by: a session token's sub when one
// rides the request, else an explicit ?owner= (connected-wallet-only clients -
// everything a ledger row carries is public on-chain data, so naming an owner
// is scoping, not authentication; enclaves still verify tokens for their part).
const ownerScope = (u, req) =>
  tokenAddress(req.headers.authorization)
  || ((o) => /^0x[0-9a-fA-F]{40}$/.test(o) ? o.toLowerCase() : null)(u.searchParams.get("owner") || "");

// One wallet, one list: fan out to the live fleet (hosted rows carry live
// status/network - only for token holders; enclaves verify), then merge
// in the LEDGER's rows for the wallet - every on-chain deployment appears
// whether or not an enclave hosts it right now.
async function listDeployments(u, req, res) {
  const auth = req.headers.authorization;
  const addr = ownerScope(u, req);
  // no token = no enclave view (they'd all 401); the ledger alone answers
  const rs = auth ? await Promise.all(live.map((e) =>
    forward(e.endpoint, req, null).then((r) => ({ e, r })).catch(() => null))) : [];
  const answered = rs.filter(Boolean);
  const oks = answered.filter((x) => x.r.status === 200);
  // the fleet REFUSING a presented token is real (expired/garbage session):
  // surface it rather than mask it with public ledger rows
  if (auth && answered.length && !oks.length && answered.every((x) => x.r.status === 401))
    return sendForwarded(res, answered[0].r, req);
  if (!addr && !oks.length)
    return json(res, 401, { error: "unauthorized", message: "Pass ?owner=0x… (or a session token) to say whose deployments to list." }, req);
  const data = [], seen = new Set();
  // registry-id -> the ids that enclave's 200 list carried. The same token
  // scoped both the fan-out and the ledger loop below, so for THIS owner an
  // answering runner whose list LACKS a leased id is definitive: the lease
  // outlived the local record (enclave restart/update wiped state, or the
  // resume found no capacity) and nothing actually serves the app.
  const hostedByRunner = new Map();
  // id -> the index in `data` of the row we are showing for it, so a second
  // enclave claiming the same deployment REPLACES rather than duplicates.
  const at = new Map();
  // Releasing a deployment does not delete the ex-runner's local record, so
  // after a Move BOTH boxes answer for the same id — the new host with a live
  // row, the old one with its terminated copy. Pushing both produced a phantom
  // duplicate that shadowed the real row: it offered Resume (terminated reads
  // as resumable), and Resume then found the ledger already active, skipped the
  // setActive tx, and returned having done nothing. That is precisely "resume
  // never asks for a signature and never resumes" — the app was running the
  // whole time (found 2026-07-28 on a deployment moved metal0 -> kryptos ->
  // metal0). The ledger's runner is the tiebreak, exactly as it is for routing.
  //
  // Deduping the two hosted rows is NOT enough on its own, because usually
  // only ONE of them is in `oks` at all: sessions are per-enclave (each box
  // signs with its own in-enclave key and verifies only its own kid), and the
  // fan-out presents the caller's token to every box, so the box that did not
  // mint it answers 401 and drops out. Sign in on the old host and it is the
  // ONLY answering enclave — its terminated copy is the sole hosted row, it
  // suppresses the ledger row via `seen`, and the deployment reads TERMINATED
  // on a box that released it while the real host serves happily elsewhere.
  // So a hosted row from a box the chain says is not the runner is not just
  // outranked, it is DISCARDED: the ledger row (which knows the true runner,
  // resources and lease) takes its place.
  const runnerOf = new Map();
  try {
    for (const d of await ledgerRows())
      runnerOf.set(String(d.id).toLowerCase(),
                   { runner: String(d.runner).toLowerCase(), leaseLive: Number(d.leaseUntil) * 1000 > Date.now() });
  } catch {}
  // Narrow on purpose, and only for a row that declares the deployment DEAD.
  // A live-looking row from a box the chain does not yet name is usually just
  // lag — an enclave serves from the moment it claims, while ledgerRows() is
  // up to LEDGER_TTL_MS behind the claim tx — and showing it is harmless and
  // self-correcting. A terminated/stopped/expired row from a box the chain
  // says is not the runner is the stale-copy case: it cannot become true
  // again, and it is the one that misreports a running app as ended.
  const TERMINAL = /^(terminated|stopped|expired)$/;
  const contradicted = (row, e) => {
    if (!TERMINAL.test(String(row.status || ""))) return false;
    const want = runnerOf.get(String(row.id).toLowerCase());
    return !!(want && want.leaseLive && !ZERO32.test(want.runner)
              && e.id && String(e.id).toLowerCase() !== want.runner);
  };
  const wins = (row, e) => {
    const want = runnerOf.get(String(row.id).toLowerCase());
    if (want && want.runner && e.id) return String(e.id).toLowerCase() === want.runner;   // the chain names the host
    return !/^(terminated|stopped|expired)$/.test(String(row.status || ""));   // else prefer a live row
  };
  for (const { e, r } of oks) {
    const ids = new Set();
    try {
      for (const it of JSON.parse(r.text).data || []) {
        if (it.enclave == null) it.enclave = e.name || endpointName(e.endpoint);
        const key = String(it.id).toLowerCase();
        if (contradicted(it, e)) continue;    // stale copy on an ex-runner: let the ledger row answer
        const prev = at.get(key);
        if (prev == null) { at.set(key, data.length); data.push(it); }
        else if (wins(it, e)) data[prev] = it;
        seen.add(key); ids.add(key);
        // only the winning row's host may be cached as the owner: learning the
        // ex-runner here is what sent control-plane calls to the box that had
        // already dropped the record ("No such deployment.")
        if (data[at.get(key)] === it) ownerLearn(it.id, e.endpoint);
      }
    } catch {}
    if (e.id) hostedByRunner.set(String(e.id).toLowerCase(), ids);
  }
  const tokenOwner = tokenAddress(auth);
  if (addr) {
    try {
      for (const d of await ledgerRows()) {
        if (d.owner.toLowerCase() !== addr || seen.has(d.id.toLowerCase())) continue;
        const view = ledgerView(d);
        // ledgerStatus says "running" for lease-live + runner-alive — but a
        // runner that answered this owner's list WITHOUT the id is not serving
        // it. Show "claimed" (+ stranded) instead of a lie the owner pays for
        // (observed live 2026-07-17: a displaced tenant read RUNNING for 30
        // minutes while its app was dark).
        if (view.status === "running" && tokenOwner && addr === tokenOwner) {
          const hosted = hostedByRunner.get(String(d.runner).toLowerCase());
          if (hosted && !hosted.has(d.id.toLowerCase())) { view.status = "claimed"; view.stranded = true; }
        }
        data.push(view);
      }
    } catch (e) { console.error("[api-relay] ledger read failed:", e.message); }
  }
  return json(res, 200, { data, cursor: null }, req);
}

// Bare record read: for token holders the owning enclave has the live view
// (status transitions, network) - prefer it; tokenless reads (and any id
// no live enclave hosts) answer from the ledger, so watchers keep working
// across enclave restarts, for still-queued work, and with no session at all.
//
// The enclave leg is BUFFERED (forward), not streamed, because the box's
// answer is not always the truth about the DEPLOYMENT, and two of its answers
// must not reach the caller:
//   - 404: the box verified the token but holds no local record under the id
//     (state wiped by a release restart, a claim not yet re-adopted, a prefix
//     its exact-id GET can't resolve) while the on-chain record still exists.
//     Forwarding it told signed-in owners their own running app didn't exist,
//     while the dashboard - which merges ledger rows - showed it fine (found
//     2026-08-19, /authorize on a private app).
//   - 401: sessions are per-enclave, so a token minted by any OTHER box reads
//     as invalid here. That is a fact about the session, not the deployment.
// Both fall through to the ledger, exactly like a tokenless read: these are
// public on-chain rows (see listDeployments), so the fallback serves nothing
// a bare ?owner= read wouldn't.
// A 200 leaves stamped with the serving box's fleet name: supervisors don't
// know their own registry name, so the list fan-out adds `enclave` to every
// row - and this read has to match, because /authorize signs in to
// dep.enclave, and a running app without it reads as down.
async function getDeployment(id, u, req, res) {
  const auth = req.headers.authorization;
  if (live.length && auth) {
    const owner = await v1OwnerOf(id, auth);
    if (owner) {
      let r = null;
      try { r = await forward(owner, req, null); } catch {}   // box died mid-read: the ledger still answers
      if (r && r.status !== 404 && r.status !== 401) {
        if (r.status === 200) try {
          const row = JSON.parse(r.text);
          if (row && row.id && row.enclave == null) {
            const e = live.find((x) => x.endpoint === owner);
            if (e) { row.enclave = e.name || endpointName(e.endpoint); r.text = JSON.stringify(row); }
          }
        } catch {}
        return sendForwarded(res, r, req);
      }
    }
  }
  const addr = ownerScope(u, req);
  let rows;
  try { rows = await ledgerRows(); }
  catch (e) { return json(res, 502, { error: "ledger_error", message: e.message, updatedAt }, req); }
  const want = id.toLowerCase();
  // full ids and unique prefixes both resolve (the CLI passes prefixes); an
  // ?owner=/token scope disambiguates, but isn't required - records are public
  const hits = rows.filter((d) => (!addr || d.owner.toLowerCase() === addr) && d.id.toLowerCase().startsWith(want));
  if (hits.length !== 1) {
    // Say WHICH kind of missing: a scope that filtered out an existing record
    // is a wrong-wallet story, and sharing one message with "no such id" made
    // the sign-in page tell owners to switch wallets for records a box had
    // merely lost. Naming the split leaks nothing - the unscoped read answers.
    const others = !hits.length && addr && rows.some((d) => d.id.toLowerCase().startsWith(want));
    return json(res, 404, { error: "not_found",
      message: hits.length ? `${id} is ambiguous (${hits.length} deployments match).`
             : others     ? `${id} belongs to a different wallet than ${addr}. Switch wallets and try again.`
                          : `No live enclave has ${id}, and the ledger has no deployment under it.`, updatedAt }, req);
  }
  return json(res, 200, ledgerView(hits[0]), req);
}

// <label>.<APP_DOMAIN> -> canonical dep_<label>, or null if not an app subdomain.
// The subdomain drops the "dep_" (redundant in this namespace): "abc123" ->
// "dep_abc123". A legacy "dep-abc123" is still accepted.
function depFromHost(host) {
  host = (host || "").toLowerCase().split(":")[0];
  const dom = APP_DOMAINS.find(d => host.endsWith("." + d));
  if (!dom) return null;
  const label = host.slice(0, -(dom.length + 1)).replace(/^dep[-_]/, "");   // strip a legacy prefix if present
  // On-chain (EnclaveDeployments) ids are bytes32; a full 64-hex id exceeds DNS's
  // 63-char label limit, so their subdomain is a hex PREFIX of the id - the
  // canonical label is the FIRST 8 CHARS (32 bits; collisions are fantasy),
  // and any longer prefix keeps working. Enclaves resolve the prefix to the
  // unique matching deployment. (A retired-era dep_ label that happened to be
  // pure hex could shadow here; those deployments no longer exist.)
  const hex = label.startsWith("0x") ? label.slice(2) : label;
  if (/^[0-9a-f]{8,64}$/.test(hex)) return "0x" + hex;
  const id = "dep_" + label;
  return /^dep_[a-z0-9]+$/.test(id) ? id : null;
}

// Does a deployment exist on the fleet? (the /x owner probe: some enclave
// answers non-404 for it.) Gates on-demand TLS issuance so nobody can burn the
// CA rate limit with random <junk>.<APP_DOMAIN> names; the owner cache keeps
// repeat lookups cheap.
const deploymentExists = async (id) => !!(await xOwnerOf(id));

// Shared helpers handed to the account/billing modules (auth.js, billing.js):
// they reuse the relay's CORS, raw-body reader and cached ledger reader
// without circular imports. deploymentsAddress is a thunk because the address
// book live-updates the binding.
const relayCtx = { json, cors, clientIp, readBody, ledgerRows, ledgerView,
                   deploymentsAddress: () => DEPLOYMENTS_ADDRESS,
                   // billing.js quotes at the fleet's cheapest posted price
                   // (rev-8 ledgers carry none of their own)
                   fleetAsk: () => cheapestAsk(servingEnclaves(), servingEnclaves().filter((e) => e.availability?.gpu === true)),
                   // secrets.js: match a fetch's claimed endpoint to a lease's
                   // on-chain runner id, and drop the ledger cache when a row
                   // must be newer than the 10s TTL (just-claimed/just-created)
                   endpointIdOf: endpointId, ledgerExpire: () => { _ledger.at = 0; },
                   // ...and WHOSE key that endpoint is, so a secrets fetch can be
                   // held to the operator that registered it rather than to the
                   // fleet-wide derived key alone (relay/secrets.js). Same read
                   // as the tunnel's name-ownership gate: an inactive or absent
                   // entry is "nobody", and errors propagate to the caller's own
                   // fail-closed decision.
                   operatorOfEndpoint: async (endpoint) => {
                     if (!REGISTRY_ADDRESS) return null;
                     const c = await chain();
                     const e = await c.readContract({ address: REGISTRY_ADDRESS, abi: GET_ABI,
                       functionName: "get", args: [await endpointId(endpoint)] });
                     const op = String(e?.operator || "");
                     return e?.active && !/^0x0{40}$/i.test(op) ? op.toLowerCase() : null;
                   } };

// Every inbound request runs inside one guard, for two reasons.
//
// The request TARGET first. Node hands an absolute-form target (`GET
// http://elsewhere/y HTTP/1.1`, legal for proxies) to req.url verbatim, and
// this relay routes on the PARSED pathname while forwarding req.url — two
// readings of one target, which is the shape request smuggling is made of, and
// which concatenates into a nonsense upstream host besides. Origin-form only.
// The pathname is then read with the target APPENDED to a base rather than
// resolved against one, so a `//host/path` target cannot be read as an
// authority by the router and as a path by the forwarder.
//
// Then the throw. A synchronous throw in a Node request listener is an
// uncaughtException, and installProcessGuards turns that into exit(1): one
// malformed request would dark the fleet's whole front door until systemd
// restarted it. Answer 500 and stay up.
const server = http.createServer((req, res) => {
  if (!String(req.url || "").startsWith("/")) {
    res.writeHead(400, { "Content-Type": "application/json", "Cache-Control": "no-store" });
    return res.end(JSON.stringify({ error: "bad_request_target",
      message: "The request target must be origin-form (/path), not absolute-form or *." }));
  }
  try { return handleRequest(req, res); }
  catch (e) {
    console.error("[api-relay] request handler threw:", (e && e.stack) || e);
    try { if (!res.headersSent) res.writeHead(500, { "Content-Type": "application/json", "Cache-Control": "no-store" }); } catch {}
    try { res.end(JSON.stringify({ error: "internal_error", message: "The relay could not handle this request." })); } catch {}
  }
});

function handleRequest(req, res) {
  const u = new URL("http://x" + req.url);

  // On-demand TLS gate: Caddy asks before minting a cert for <host>. Allow only
  // real deployment subdomains so random <junk>.<APP_DOMAIN> can't burn the CA
  // rate limit. Rate-limited on the miss (fix 2) so it can't be driven as a
  // fan-out probe, and restricted to genuinely internal callers — see
  // isInternalCall for why "loopback" alone was not that, and was answering the
  // public internet until 2026-07-30.
  if (u.pathname === "/internal/tls-ask") {
    if (!isInternalCall(req)) { res.writeHead(403); return res.end("forbidden"); }
    const asked = (u.searchParams.get("domain") || "").toLowerCase().replace(/\.+$/, "").split(":")[0];
    // A CUSTOMER's own hostname (relay/domains.js): authorized only while its
    // record says verified/active, answered from memory so this stays a
    // handshake-time-safe lookup, and never for a name in a zone we own — that
    // check lives in domains.js and is applied here even though the add
    // endpoint already refused such a name, because this gate is the last thing
    // between a request and a certificate.
    if (tlsAskAllowed(asked)) { res.writeHead(200); return res.end(); }
    const id = depFromHost(asked);
    if (!id) { res.writeHead(400); return res.end("bad domain"); }
    if (!ownerCached(id) && !rlMiss(clientIp(req))) { res.writeHead(429); return res.end("rate limited"); }
    return deploymentExists(id).then((ok) => { res.writeHead(ok ? 200 : 404); res.end(); });
  }

  // App subdomain: <dep-id>.<APP_DOMAIN> is the deployment's OWN origin. Route it
  // to the OWNING enclave's /x/<id> data path, passing the app's own headers
  // through (it's a distinct origin, so the gateway doesn't impose CORS).
  //
  // A verified CUSTOM domain resolves to the same deployment and takes exactly
  // this path. The SNI relay is the normal route for those names, so reaching
  // here means something terminated TLS in front of us (a customer's CDN, or a
  // Caddy that fronts this relay) — the routing answer must be the same either
  // way, and an unknown hostname must reach no app at all rather than the
  // first one that happens to answer.
  const depHost = depFromHost(routingHost(req))             // x-forwarded-host only when TRUSTED_PROXY (fix 6)
               || domainDeployment(routingHost(req));
  if (depHost) {
    if (!ownerCached(depHost) && !rlMiss(clientIp(req))) return json(res, 429, { error: "rate_limited", message: "Too many lookups; retry shortly." });
    return xOwnerOf(depHost).then((owner) => {
      if (!owner) return json(res, 404, { error: "not_found", message: "No live enclave has " + depHost + "." });
      const rest = req.url === "/" ? "/" : req.url;           // preserve path+query under /x/<id>
      // same generous idle window as the /x data path (see gateway()): app
      // subdomains ARE the data path, and long-silent first bytes are real
      proxyTo(owner, req, res, { path: "/x/" + depHost + rest, setCors: false, idleMs: 180000 });
    });
  }

  // MCP endpoint (mcp.enclave.host, or /mcp on the API host): the coding-agent
  // front door. Checked AFTER the app-subdomain branch so a tenant app's own
  // /mcp path is never shadowed; handles its own CORS/OPTIONS (any origin, no
  // credentials — tokens ride per-call params, not cookies). See mcp.js.
  if (isMcpHost(routingHost(req)) || u.pathname === "/mcp" || u.pathname === "/mcp/")
    return handleMcp(req, res, u).catch((e) =>
      json(res, 500, { error: "mcp_error", message: e.message }, req));

  if (req.method === "OPTIONS") { res.writeHead(204, cors(req)); return res.end(); }   // preflight for any path

  if (u.pathname === "/health")
    return json(res, 200, { ok: true, enclaves: live.length, of: registry.length, updatedAt }, req);

  if (u.pathname === "/enclaves") {
    // rows list EVERY live enclave (presentation + tunnel health); the
    // aggregate totals count only the CLAIMING subset — capacity nobody can
    // buy (a present-but-not-claiming tunnel box) must not inflate them
    const serving = servingEnclaves();
    const servingSet = new Set(serving);
    // every row carries an explicit `serving` verdict so display surfaces
    // (fleet panel, deploy target lists) can hide what cannot take work
    // without re-deriving the rule client-side
    const rows = live.map((e) => ({ ...e, serving: servingSet.has(e) }));
    const agg = {
      enclaves: live.length, serving: serving.length,
      totalGpuShareFree: Math.round(serving.reduce((s, e) => s + gpuFreeOf(e.availability), 0) * 1000) / 1000,
      totalCpuShareFree: Math.round(serving.reduce((s, e) => s + cpuFreeOf(e.availability), 0) * 1000) / 1000,
      totalVramFreeGb: Math.round(serving.reduce((s, e) => s + (e.availability.vramFreeGb || 0), 0) * 10) / 10,
    };
    return json(res, 200, { updatedAt, aggregate: agg, enclaves: rows }, req);
  }

  if (u.pathname === "/route") {
    // ?gpuShare=&cpuShare= — the two shares the deployment intends to buy
    // (0..1). gpuShare 0 = CPU-only (CPU enclaves preferred, GPU leftovers as
    // fallback). Legacy ?share= is read as gpuShare.
    const want = {
      gpuShare: parseFloat(u.searchParams.get("gpuShare") ?? u.searchParams.get("share") ?? "0") || 0,
      cpuShare: parseFloat(u.searchParams.get("cpuShare") || "0") || 0,
    };
    const c = pick(want);
    if (!c) return json(res, 503, { error: "no_capacity",
      message: `No live enclave has gpuShare >= ${want.gpuShare} and cpuShare >= ${want.cpuShare} free.`, updatedAt }, req);
    return json(res, 200, { endpoint: c.endpoint, repo: c.repo, availability: c.availability, updatedAt,
                            note: "Verify attestation at the endpoint (Tinfoil SecureClient + repo) before sending anything." }, req);
  }

  // Featured-slot view metering (owned by the relay, never proxied): the
  // beacon POST counts a deduped impression; the GET serves lifetime totals.
  if (u.pathname === "/v1/featured-view" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => { body += c; if (body.length > 512) req.destroy(); });
    req.on("end", () => {
      let app = "";
      try { app = String(JSON.parse(body).app || "").toLowerCase(); } catch {}
      if (!/^0x[0-9a-f]{64}$/.test(app)) return json(res, 400, { error: "bad_app", message: "body must be {app:<bytes32 appId>}" }, req);
      featCount(req, app);                     // dedup result deliberately unreadable from outside
      json(res, 200, { ok: true }, req);
    });
    return;
  }
  if (u.pathname === "/v1/featured-views")
    return json(res, 200, { updatedAt, views: featViews }, req);

  // Relay-owned account + billing endpoints (auth.js / billing.js): passkey +
  // SIWE account sessions, orders, Stripe webhook, USDC payment status. They
  // answer with ZERO live enclaves (billing must not depend on the fleet) and
  // are never proxied. /v1/auth/* below stays enclave-proxied - the enclave
  // session system is a separate trust domain and is untouched. The BARE
  // /v1/account path is NOT ours: that is the supervisor's wallet-scoped
  // summary (enclave session), and matching it here 404'd `enclave account`.
  if (u.pathname.startsWith("/v1/account/"))
    return handleAccount(req, res, u, relayCtx).catch((e) =>
      json(res, 500, { error: "account_error", message: e.message }, req));
  // Sign in with Enclave (sso.js): mints audience-bound EST1 tokens for
  // tenant apps against the relay ACCOUNT session. Relay-owned like accounts:
  // answers with zero live enclaves.
  if (u.pathname === "/v1/sso" || u.pathname.startsWith("/v1/sso/"))
    return handleSso(req, res, u, relayCtx).catch((e) =>
      json(res, 500, { error: "sso_error", message: e.message }, req));
  if (u.pathname === "/v1/billing" || u.pathname.startsWith("/v1/billing/"))
    return handleBilling(req, res, u, relayCtx).catch((e) =>
      json(res, 500, { error: "billing_error", message: e.message }, req));
  // Per-deployment secrets (secrets.js): relay-OWNED state like accounts/
  // billing — never proxied, answers with zero live enclaves (owners stage
  // secrets between create and the first claim; the fleet being down must not
  // block that).
  if (u.pathname === "/v1/secrets" || u.pathname.startsWith("/v1/secrets/"))
    return handleSecrets(req, res, u, relayCtx).catch((e) =>
      json(res, 500, { error: "secrets_error", message: e.message }, req));
  // Custom domains (domains.js): relay-owned like secrets — an owner attaches a
  // hostname long before the deployment that will serve it is claimed, so this
  // must answer with zero live enclaves too.
  if (u.pathname === "/v1/domains" || u.pathname.startsWith("/v1/domains/"))
    return handleDomains(req, res, u, relayCtx).catch((e) =>
      json(res, 500, { error: "domains_error", message: e.message }, req));
  // Platform certificates (certs.js): a lease-holding enclave trades a CSR
  // for a CA cert on <label>.APP_ZONE — the CA account and EAB pair live
  // here, the private key stays in the CVM. Relay-owned, answers with zero
  // live enclaves like the two above.
  if (u.pathname === "/v1/certs" || u.pathname.startsWith("/v1/certs/"))
    return handleCerts(req, res, u, relayCtx).catch((e) =>
      json(res, 500, { error: "certs_error", message: e.message }, req));

  // A box reached by its OWN name: e<hex>.<BOX_ZONE> is a whole host, so the
  // request arrives with that Host and an ordinary path. Forward the path
  // untouched over that box's tunnel.
  //
  // This is what makes a tunnel box verifiable. Every attestation verifier
  // takes a bare HOST and builds /.well-known/tinfoil-attestation from it, so
  // the /t/<name> path form below gets truncated to the relay's own hostname
  // and ends up verifying the RELAY (found 2026-07-28: every metal0-hosted app
  // failed in-browser this way). With a real host the URL survives, and once
  // the zone is served with SNI passthrough the client's TLS terminates at the
  // BOX — so the quote's reportData binds the certificate the client actually
  // saw, which no relay-terminated path can ever do.
  const boxName = boxLabelOfHost(routingHost(req));
  if (boxName) {
    const origin = `tunnel://${boxName}`;
    if (!tunnelHub.origins().some((o) => o.endpoint === origin))
      return json(res, 404, { error: "no_tunnel", message: `No enclave is attached for ${routingHost(req)}.` }, req);
    return proxyTo(origin, req, res, { path: u.pathname + (u.search || ""), setCors: true });
  }

  // Reach a SPECIFIC tunnel enclave through the relay: /t/<name>/<rest> forwards
  // <rest> over that enclave's tunnel. This is how a CGNAT self-hosted enclave
  // (no public endpoint) is reachable + independently verifiable — e.g.
  // /t/metal0/v1/attestation returns its live SEV-SNP attestation, which the
  // client verifies end-to-end (the relay only shuttles bytes).
  const tm = u.pathname.match(/^\/t\/([A-Za-z0-9_-]+)(\/.*|)$/);
  if (tm) {
    const origin = `tunnel://${tm[1]}`;
    if (!tunnelHub.origins().some((o) => o.endpoint === origin))
      return json(res, 404, { error: "no_tunnel", message: `No tunnel enclave named ${tm[1]} is attached.` }, req);
    return proxyTo(origin, req, res, { path: (tm[2] || "/") + (u.search || ""), setCors: true });
  }

  // API gateway: fleet-aware routing (see the header) — placement on create,
  // owner affinity on deployment-scoped calls, fan-out merge on list, fleet
  // aggregate on /availability, sticky enclave for the rest.
  if (proxied(u.pathname))
    return gateway(u, req, res).catch((e) =>
      json(res, 502, { error: "gateway_error", message: e.message, updatedAt }, req));

  json(res, 404, { error: "not_found", routes: ["/health", "/enclaves", "/v1/relays", "/route?gpuShare=0.25&cpuShare=0.05", "/v1/* /x/* /availability (fleet-routed to the enclaves)"] }, req);
}

// WebSocket upgrades. Node hands Upgrade requests to an 'upgrade' listener, not
// the request handler — without one the relay silently ate the enclaves' WS
// surfaces (the /x/:id/tcp/:port raw-TCP bridge, any app's own websockets) and
// bridge clients had to bypass the gateway for the enclave origin. Routing
// mirrors the request path: an app subdomain maps onto the owner's /x/<id>
// data path, a gateway /x/<id>/... URL passes through verbatim. The relay
// forwards the handshake bytes untouched and splices sockets after it — it
// never speaks WS itself, so anything the enclave upgrades to just works.
const UPGRADE_IDLE_MS = 180000;                              // match the /x data path's window
// The supervisor's WS bridges look deployments up by EXACT id (deployments.get),
// unlike its HTTP /x path which resolves hex prefixes — so a subdomain label
// (8-hex prefix) must be canonicalized to the full ledger id before proxying.
// Falls back to the given id when the ledger can't answer or the prefix is
// ambiguous; full-id URLs then still work exactly as before.
async function fullDepId(id) {
  if (!/^0x[0-9a-f]{8,63}$/.test(id)) return id;             // full 64-hex (or non-ledger-shaped): pass through
  try {
    const hits = (await ledgerRows()).filter((d) => String(d.id).toLowerCase().startsWith(id));
    if (hits.length === 1) return String(hits[0].id).toLowerCase();
  } catch {}
  return id;
}
server.on("upgrade", async (req, socket, head) => {
  socket.on("error", () => socket.destroy());               // dead client mid-handshake must not throw
  const refuse = (code, text) => { try { socket.write(`HTTP/1.1 ${code} ${text}\r\nConnection: close\r\n\r\n`); } catch {} socket.destroy(); };
  if (!String(req.url || "").startsWith("/")) return refuse(400, "Bad Request");   // origin-form only (see handleRequest)
  // fleet tunnel attach: a self-hosted enclave dialing IN (token-authed in the hub)
  if ((req.url || "").split("?")[0] === "/v1/fleet-tunnel") return tunnelHub.handleUpgrade(req, socket, head);
  // tunnel-routed upgrades (Phase D): /t/<name>/<rest> — the SNI relay's wss
  // dial into a tunnel box's supervisor (its registered endpoint IS this
  // path), spliced through the hub as a raw stream; the supervisor answers
  // the handshake itself, so /x/<id>/tls stays TLS-in-CVM end to end.
  {
    const u = new URL("http://x" + (req.url || "/"));
    const tm = u.pathname.match(/^\/t\/([A-Za-z0-9_-]+)(\/.*|)$/);
    if (tm) {
      const origin = `tunnel://${tm[1]}`;
      if (!tunnelHub.origins().some((o) => o.endpoint === origin)) return refuse(404, "Not Found");
      return tunnelHub.spliceUpgrade(origin, req, socket, head, (tm[2] || "/") + (u.search || ""));
    }
    // …and the same box reached by its own hostname, so a websocket to a box
    // does not silently fall through to the deployment router below.
    const bn = boxLabelOfHost(routingHost(req));
    if (bn) {
      const origin = `tunnel://${bn}`;
      if (!tunnelHub.origins().some((o) => o.endpoint === origin)) return refuse(404, "Not Found");
      return tunnelHub.spliceUpgrade(origin, req, socket, head, u.pathname + (u.search || ""));
    }
  }
  try {
    const depHost = depFromHost(routingHost(req));           // x-forwarded-host only when TRUSTED_PROXY (fix 6)
    const x = depHost ? null : (req.url || "").match(X_PATH_RE);
    if (!depHost && !x) return refuse(404, "Not Found");
    const id = await fullDepId(depHost || x[1]);
    const owner = await xOwnerOf(id);
    if (!owner) return refuse(404, "Not Found");
    const rest = depHost ? (req.url === "/" ? "/" : req.url) : req.url.slice(3 + (x[1].length));  // after "/x/<id>"
    const path = "/x/" + id + rest;
    if (tunnelHub.isTunnel(owner)) return tunnelHub.spliceUpgrade(owner, req, socket, head, path);
    const target = new URL(owner.replace(/\/+$/, "") + path);
    const secure = target.protocol === "https:";
    const up = (secure ? tls : net).connect({
      host: target.hostname, port: +target.port || (secure ? 443 : 80),
      ...(secure ? { servername: target.hostname } : {}),
    }, () => {
      let raw = `${req.method} ${target.pathname}${target.search} HTTP/1.1\r\n`;
      for (let i = 0; i < req.rawHeaders.length; i += 2)     // rawHeaders keeps order, casing, duplicates
        raw += `${req.rawHeaders[i]}: ${/^host$/i.test(req.rawHeaders[i]) ? target.host : req.rawHeaders[i + 1]}\r\n`;
      up.write(raw + "\r\n");
      if (head?.length) up.write(head);
      socket.pipe(up); up.pipe(socket);
    });
    const drop = () => { socket.destroy(); up.destroy(); };
    up.setTimeout(UPGRADE_IDLE_MS, drop); socket.setTimeout(UPGRADE_IDLE_MS, drop);
    up.on("error", drop); up.on("close", drop); socket.on("close", drop);
  } catch (e) { refuse(502, "Bad Gateway"); }
});

await pollRegistry();
await resolveDeployments();
await pollAvailability();
await initAccounts();          // no data dir/deps => disabled with one log line
await initSso();               // SSO_SIGNER_KEY unset => disabled with one log line
await initBilling(relayCtx);   // needs accounts; degrades the same way
await initSecrets();           // needs SECRETS_KEY + the same data dir; degrades the same way
startSecretsSweep(relayCtx);   // hourly off-ledger purge (no-op while disabled)
await initDomains();           // custom domains: same data dir, CUSTOM_DOMAINS=0 opts out
startDomainSweep(relayCtx);    // DNS re-check + demotion sweep (no-op while disabled)
await initCerts();             // platform certs: CERTS_KEY + DNS_API + DNS_TXT_KEY + APP_ZONE + the data dir
setInterval(pollRegistry, REGISTRY_POLL_SEC * 1000);
setInterval(resolveDeployments, REGISTRY_POLL_SEC * 1000);
setInterval(pollAvailability, AVAIL_POLL_SEC * 1000);

const BIND = process.env.API_RELAY_BIND || undefined;
if (!BIND) console.error("[api-relay] NOTE: binding ALL interfaces (no API_RELAY_BIND). If a local Caddy fronts this relay, set API_RELAY_BIND=127.0.0.1 so :" + PORT + " isn't reachable directly.");
server.listen(PORT, BIND, () => console.log(
  `[api-relay] :${PORT}${BIND ? " (" + BIND + ")" : ""} · ${STATIC_ENCLAVES.length ? `static list (${STATIC_ENCLAVES.length})` : `EnclaveRegistry ${REGISTRY_ADDRESS}`} · ${live.length}/${registry.length} live`
  + (STATIC_ENCLAVES.length || !OPERATORS_UNRESTRICTED ? "" : " · UNAUTHENTICATED fleet (TRUSTED_OPERATORS=*)")));
