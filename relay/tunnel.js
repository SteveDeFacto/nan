// relay/tunnel.js — fleet tunnel hub.
//
// Self-hosted enclaves (e.g. Enclave Metal boxes) live behind CGNAT: they have
// no public endpoint the relay can dial. So they dial OUT to the relay and hold
// a persistent WebSocket; the relay forwards their public HTTP surface (/v1/*,
// /availability, /x/*) back over it. To the rest of api-relay a tunnel enclave
// looks like any other fleet member — it shows up in readRegistry() as a synthetic
// row with a `tunnel://<name>` endpoint, and proxyTo()/pollAvailability() route
// to it through here instead of dialing.
//
// Trust: the tunnel only decides ROUTING, never trust. Clients still verify the
// enclave's attestation end-to-end (the metal RAD carries a real SEV-SNP report;
// nothing here vouches for it). Attach auth just stops a random peer from
// claiming a fleet name: the enclave presents a token whose sha256 is on a
// committed allowlist (the token itself never enters the repo), so no on-box
// secret and no secret-in-code is required.
import { WebSocketServer } from "ws";
import { createHash, timingSafeEqual, randomBytes } from "node:crypto";
import { verifyQuote } from "./snp-verify.mjs";
import { verifyAvfEvidence } from "./avf-verify.mjs";
import { boxOrigin } from "./boxhost.js";

const sha256Hex = (s) => createHash("sha256").update(String(s)).digest("hex");
const eqHex = (a, b) => { const x = Buffer.from(String(a), "hex"), y = Buffer.from(String(b), "hex"); return x.length === y.length && timingSafeEqual(x, y); };

// A tunnel name is a routing key: it appears in `tunnel://<name>` origins and in
// the relay's /t/<name>/… path, so it must be a plain label. Anything else is
// refused at the handshake rather than silently producing an unroutable row.
const NAME_RE_OK = /^[A-Za-z0-9_-]{1,64}$/;

// A tunnel's publicUrl becomes its REGISTRY ID upstream (keccak256 of the URL),
// and a synthetic row carrying a known id DISPLACES the discovered on-chain row
// with the same id (api-relay readRegistry). Believing the claim as sent is a
// takeover primitive: any attached box could name another enclave's registered
// endpoint and have the fleet route that enclave's deployments — /x data path
// and /v1 control path, caller Authorization header included — to itself. So
// only a SELF-ROUTED url is honored: one this hub can vouch for from the attach
// name alone, which is exactly what a CGNAT seller registers on chain
// (`enclave host`, metal/HANDOFF.md).
// A colo box with its own dialable https endpoint needs no claim at all — it is
// discovered on chain directly and was never displaced.
//
// Two accepted shapes, and the check is the same in both: derive the URL from
// `name` and compare, never parse trust out of what was sent.
//   1. https://<box-zone-host>      — the box's own name (boxhost.js). Its
//      label IS the attach name, so this is a pure derivation.
//   2. https://<relay>/t/<name>     — the legacy path route, kept so a box
//      that predates the zone (or a relay with BOX_ZONE unset) still attaches.
function selfRoutedUrl(url, name) {
  if (!url) return "";
  let u; try { u = new URL(String(url)); } catch { return ""; }
  if (u.protocol !== "https:" || u.search || u.hash) return "";
  const origin = boxOrigin(name);
  if (origin && u.pathname.replace(/\/+$/, "") === "" && `https://${u.host}` === origin) return origin;
  return u.pathname.replace(/\/+$/, "") === `/t/${name}` ? String(url) : "";
}

// allow:  [{ name, tokenSha256 }]                       — bootstrap / first-party boxes
// attest: { allowedMeasurements: [hex], requireVcek,   — permissionless sellers:
//           avf: { codeHashes: [hex], authorityHashes: [hex] } }
//   attach is granted to ANY enclave that proves, with a fresh SEV-SNP quote over
//   a relay-chosen challenge, that it runs a published Metal release (measurement
//   on the allowlist). No token, no per-seller identity. See metal/PROTOCOL.md.
//   `avf` admits a PHONE-ANCHORED host the same way: an Android protected-VM
//   attestation chain (avf-verify.mjs) whose leaf carries our challenge, is
//   rooted at Google, and names an allowlisted anchor build (codeHash) signed by
//   our APK certificate (authorityHash). Its mode is "avf", not "snp".
// operatorFor: async (name) -> 0x… | null                — WHO OWNS A NAME on chain.
//   A quote proves the IMAGE, and the transport key is minted PER BOOT, so
//   neither survives a reboot as an identity: while a seller was down, another
//   box running the same published release could take its name and inherit the
//   routing for keccak(https://<relay>/t/<name>) — the id its own on-chain
//   registration carries. The one thing that DOES survive is the operator key
//   that registered it, so when this resolves an owner for the name, the
//   attaching box must sign the attach challenge with that key. Names with no
//   on-chain entry stay first-come: there is nothing yet to take.
// operatorAttach: true — ATTACH BY ON-CHAIN OWNERSHIP ALONE, no quote.
//   The two paths above both assume the box can prove what it runs: a token
//   says "someone put my hash in a file", a quote says "I am a published Metal
//   release". A RELAY can do neither. It is not a TEE — deliberately, because it
//   terminates nothing and holds no keys, so there is no measurement to publish
//   and nothing a quote would add. Yet it still has an identity worth proving:
//   the operator key that registered its endpoint on chain.
//   So: the hub challenges, the box signs with that key, and the hub checks the
//   recovered signer against the registry. Nothing is hardcoded and nothing is
//   host state — adding or removing a relay is a registry transaction, which is
//   the whole point of putting the fleet on chain in the first place.
//   OFF by default. A tunnel row bypasses the dial-time operator allowlist (it
//   is authorized here instead), so turning this on lets anyone who registers
//   https://<relay>/t/<name> appear in the fleet listing under that name. That
//   is a deliberate widening and it should be a deliberate switch.
// trustedOperators / operatorsUnrestricted — the SAME fail-closed operator set
//   the dial-based discovery applies, enforced here because a tunnel row does
//   NOT go through it. Without this the operator path would be the least gated
//   of the three: the registry is permissionless, so anyone could register
//   https://<relay>/t/<name>, sign for it, and appear in the fleet listing —
//   and a row that claims capacity lands in the set that sizes the fleet and
//   takes placement. A token needs a committed hash and a quote needs an
//   allowlisted measurement; proving a name from chain has to clear a bar too.
export function createTunnelHub({ allow = [], attest = null, reqTimeoutMs = 30000, onChange = () => {},
                                  operatorFor = null, operatorAttach = false,
                                  trustedOperators = [], operatorsUnrestricted = false } = {}) {
  const trusted = new Set(trustedOperators.map((a) => String(a).toLowerCase()));
  const allowByName = new Map(allow.filter((a) => a && a.name && a.tokenSha256).map((a) => [a.name, a.tokenSha256.toLowerCase()]));
  const attestOn = !!(attest && ((attest.allowedMeasurements && attest.allowedMeasurements.length)
                               || (attest.avf && attest.avf.codeHashes && attest.avf.codeHashes.length)));
  const wss = new WebSocketServer({ noServer: true });
  const tunnels = new Map();                                  // name -> { ws, pending, lastSeen, mode, publicUrl, keyFp }

  // Keepalive. Nothing else proves a tunnel is alive: a half-open socket (NAT
  // timeout, a box that vanished without a FIN) never fires 'close', so its
  // entry would keep answering discovery, swallow every request into the 30s
  // timeout, and — since a name can no longer simply be seized (see
  // handleUpgrade) — lock the real box out of its own name on reconnect. The
  // agent answers {t:"ping"} with a pong; ANY frame refreshes lastSeen.
  const PING_MS = 30_000, DEAD_MS = 90_000;
  setInterval(() => {
    const now = Date.now();
    for (const [name, t] of [...tunnels]) {
      if (now - t.lastSeen > DEAD_MS) {
        console.error(`[tunnel] ${name} silent for ${Math.round((now - t.lastSeen) / 1000)}s — terminating`);
        try { t.ws.terminate(); } catch {}
        if (tunnels.get(name) === t) { tunnels.delete(name); try { onChange("detach", name); } catch {} }
        continue;
      }
      try { t.ws.send(JSON.stringify({ t: "ping" })); } catch {}
    }
  }, PING_MS).unref?.();

  function tokenOk(name, token) {
    const want = allowByName.get(name);
    if (!want || !token) return false;
    return eqHex(sha256Hex(token), want);
  }

  // ---- name ownership (attest path) -----------------------------------------
  // The message an attaching box signs with its REGISTRY OPERATOR key. Bound to
  // the name and to this attach's fresh nonce, and EIP-191-prefixed by
  // personal_sign — so a signature harvested here can never be replayed as a
  // transaction, which matters because that key also sends claim/renew.
  const attachMessage = (name, nonce) => `enclave-tunnel-attach:${name}:${nonce.toString("base64")}`;
  // Last known owner per name. A lookup that FAILS (an RPC blip) must not open a
  // name we have already seen registered — cached ownership is what we fall back
  // to. A name never seen registered stays first-come, which is the same answer
  // as before this existed.
  const ownerCache = new Map();
  async function ownerOf(name) {
    if (!operatorFor) return null;
    try {
      const a = await operatorFor(name);
      if (a) ownerCache.set(name, String(a).toLowerCase());
      else ownerCache.delete(name);
      return a ? String(a).toLowerCase() : null;
    } catch {
      return ownerCache.get(name) || null;      // fail closed against a known owner
    }
  }
  async function signerOf(message, sig) {
    if (typeof sig !== "string" || !/^0x[0-9a-fA-F]{130}$/.test(sig)) return null;
    try {
      const { recoverMessageAddress } = await import("viem");
      return (await recoverMessageAddress({ message, signature: sig })).toLowerCase();
    } catch { return null; }
  }

  // Register an authorized socket as the tunnel for `name` and wire its frames.
  function bind(name, ws, meta = {}) {
    // A socket that died while its attestation was being verified must never be
    // registered: no further 'close' can fire on it, so the entry (and the name
    // with it) would be held forever by a tunnel that answers nothing.
    if (ws.readyState !== ws.OPEN) { try { ws.terminate(); } catch {} return false; }
    const prev = tunnels.get(name);
    if (prev && prev.ws !== ws) { try { prev.ws.terminate(); } catch {} }   // newest wins
    const t = { ws, pending: new Map(), streams: new Map(), lastSeen: Date.now(), mode: meta.mode || "", publicUrl: "",
                measurement: meta.measurement || null, keyFp: meta.keyFp || "",
                // dealt pads (relay/pads.mjs): the attested transport SPKI signs
                // ledger requests, the X25519 pad key receives the pVM's seed
                spki: meta.spki || "", padKey: meta.padKey || "" };
    tunnels.set(name, t);
    console.log(`[tunnel] ${name} attached via ${meta.via || "token"} (${tunnels.size} enclave${tunnels.size === 1 ? "" : "s"})`);
    try { onChange("attach", name); } catch {}   // refresh discovery so it lands in `live` now, not on the next slow poll
    ws.on("message", (data) => {
      t.lastSeen = Date.now();
      let f; try { f = JSON.parse(data); } catch { return; }
      if (f.t === "hello") {
        const had = t.publicUrl;
        t.mode = f.mode || t.mode; t.transportKeyFp = f.transportKeyFp || "";
        t.publicUrl = selfRoutedUrl(f.publicUrl, name);
        if (f.publicUrl && !t.publicUrl)
          console.error(`[tunnel] ${name} claimed publicUrl ${String(f.publicUrl).slice(0, 120)} — IGNORED (not this tunnel's own https://<relay>/t/${name} route); its on-chain runner id stays unstamped`);
        // The attach-time onChange snapshots the registry BEFORE this frame can
        // arrive, so a selling box's registered id (keccak of its publicUrl)
        // stays unknown until the next slow poll — its hosted rows read
        // "claimed"/unnamed for minutes after every relay restart. Re-announce
        // the moment the identity lands.
        if (t.publicUrl !== had) { try { onChange("hello", name); } catch {} }
        return;
      }
      if (f.t === "pong") return;
      if (f.t === "res" && f.id != null) { const p = t.pending.get(f.id); if (p) { t.pending.delete(f.id); p.resolve(f); } return; }
      // raw-stream frames (Phase D): s= open-ack · sd data · sx close
      if ((f.t === "s=" || f.t === "sd" || f.t === "sx") && f.sid != null) {
        const s = t.streams.get(f.sid); if (s) s(f);
      }
    });
    const bye = () => { if (tunnels.get(name) === t) tunnels.delete(name); for (const p of t.pending.values()) p.reject(new Error("tunnel closed")); for (const s of [...t.streams.values()]) s({ t: "sx" }); t.streams.clear(); console.log(`[tunnel] ${name} detached`); try { onChange("detach", name); } catch {} };
    ws.on("close", bye);
    ws.on("error", () => { try { ws.terminate(); } catch {} });
    return true;
  }

  function handleUpgrade(req, socket, head) {
    const name = String(req.headers["x-metal-name"] || "").slice(0, 64);
    const token = String(req.headers["x-metal-token"] || "");
    const wantsAttest = req.headers["x-metal-attest"] === "1" || (!token && attestOn);
    if (!NAME_RE_OK.test(name)) { socket.write("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"); return socket.destroy(); }

    // Token path (bootstrap / first-party): authorize before the handshake.
    if (tokenOk(name, token)) return wss.handleUpgrade(req, socket, head, (ws) => bind(name, ws, { via: "token" }));

    // Operator path: prove the NAME from the chain, with no quote at all. For a
    // relay this is the only identity that exists — it runs no measured image —
    // and it is a stronger one than a token, because it is the same key that
    // registered the endpoint and it can be rotated on chain without touching
    // this repo or any box's env.
    const wantsOperator = req.headers["x-metal-attach"] === "operator";
    if (wantsOperator) {
      if (!operatorAttach || !operatorFor) {
        socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
        return socket.destroy();
      }
      // Same reservation the attest path enforces, same reason: a name someone
      // put on the token allowlist is claimed, and must not be takeable by
      // whoever gets to the registry first.
      if (allowByName.has(name)) {
        console.log(`[tunnel] ${name} operator-attach REFUSED: the name is reserved for token attach`);
        socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
        return socket.destroy();
      }
      return wss.handleUpgrade(req, socket, head, (ws) => {
        const nonce = randomBytes(32);
        let settled = false, checking = false;
        const deny = (why) => {
          if (settled) return; settled = true;
          console.log(`[tunnel] ${name} operator-attach REJECTED: ${why}`);
          try { ws.send(JSON.stringify({ t: "attest-result", ok: false, reason: why })); } catch {}
          setTimeout(() => { try { ws.close(); } catch {} }, 100);
        };
        const timer = setTimeout(() => deny("attach timeout"), 15000);
        timer.unref?.();
        ws.on("message", async (data) => {
          if (settled || checking) return;
          let f; try { f = JSON.parse(data); } catch { return; }
          if (f.t !== "attach" || !f.operatorSig) return;
          checking = true;
          try {
            const owner = await ownerOf(name);
            if (settled) return;
            // No on-chain entry = nothing to prove against. Unlike the attest
            // path (where an unregistered name stays first-come because the
            // QUOTE still proved something), a signature over an unowned name
            // proves only that the peer holds some key. Refuse, and say so.
            if (!owner) return deny(`${name} has no active on-chain registration `
                                  + `(the registry entry for this relay's /t/${name} endpoint) — register it first, then attach`);
            const signer = await signerOf(attachMessage(name, nonce), f.operatorSig);
            if (settled) return;
            if (!signer) return deny("operatorSig is not a valid personal_sign of "
                                   + "\"enclave-tunnel-attach:<name>:<nonce b64>\"");
            if (signer !== owner) return deny(`${name} is registered on chain to ${owner}, not ${signer}`);
            // Proving the name is not the same as being welcome on this relay.
            // Registration is permissionless, so ownership alone would let any
            // stranger into the fleet listing — the same reason the dial path
            // filters on this set, applied here because a tunnel row skips it.
            if (!operatorsUnrestricted && !trusted.has(owner))
              return deny(`${owner} owns ${name} on chain but is not a trusted operator of this relay`);
            // A live holder is only displaceable by the same on-chain owner —
            // which this signature just proved. Two boxes sharing one operator
            // key is the operator's own business; a stranger cannot get here.
            clearTimeout(timer); settled = true;
            try { ws.send(JSON.stringify({ t: "attest-result", ok: true })); } catch {}
            bind(name, ws, { via: "operator" });
          } catch (e) { deny(`attach error: ${e.message}`); }
          finally { checking = false; }
        });
        ws.on("error", () => { try { ws.terminate(); } catch {} });
        try { ws.send(JSON.stringify({ t: "challenge", nonce: nonce.toString("base64") })); } catch {}
      });
    }

    // Attestation path (permissionless): complete the handshake unauthorized, run
    // a challenge → quote → verify exchange, and only then bind (or close).
    //
    // A quote proves the peer runs a PUBLISHED Metal release. It proves nothing
    // about WHICH box it is — every seller runs the same image, so the name in
    // the handshake header is a request, not an identity. Two rules keep it from
    // becoming one: names on the token allowlist are reserved outright, and a
    // name already held by a live tunnel can only be re-taken by the same
    // attested transport key (a genuine reconnect). Without them any seller
    // could evict metal0 (or a competitor) and inherit its routing.
    if (wantsAttest && attestOn) {
      if (allowByName.has(name)) {
        console.log(`[tunnel] ${name} attest REFUSED: the name is reserved for token attach`);
        socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
        return socket.destroy();
      }
      return wss.handleUpgrade(req, socket, head, (ws) => {
        const nonce = randomBytes(32);
        let settled = false, verifying = false;
        const deny = (why) => { if (settled) return; settled = true; console.log(`[tunnel] ${name} attest REJECTED: ${why}`); try { ws.send(JSON.stringify({ t: "attest-result", ok: false, reason: why })); } catch {} setTimeout(() => { try { ws.close(); } catch {} }, 100); };
        const timer = setTimeout(() => deny("attestation timeout"), 15000);
        timer.unref?.();                                     // a stalled attach must not hold the loop open
        ws.on("message", async (data) => {
          if (settled || verifying) return;                    // one quote in flight at a time
          let f; try { f = JSON.parse(data); } catch { return; }
          if (f.t !== "attest" || !f.rad || !f.rad.body) return;
          verifying = true;
          try {
            const spki = f.rad.transportKey ? Buffer.from(f.rad.transportKey, "base64") : null;
            const isAvf = /android-avf-pvm/.test(f.rad.format || "");
            let res;
            if (isAvf) {
              // A phone-anchored host. Same binding as SNP's report_data, in the
              // shape AVF offers: the pVM requested its certificate with
              // challenge = sha256(transportKey || nonce), and its attested key
              // signs (transportKey || nonce), so the certificate is tied to THIS
              // transport key and THIS attach. Body is JSON { chain: [b64 DER…],
              // signature: b64 DER-ECDSA }.
              if (!attest.avf || !attest.avf.codeHashes || !attest.avf.codeHashes.length) return deny("AVF attach is not enabled on this relay");
              if (!spki) return deny("AVF attach must carry transportKey");
              let ev; try { ev = JSON.parse(Buffer.from(f.rad.body, "base64").toString("utf8")); } catch { return deny("AVF body is not JSON"); }
              if (!ev || !Array.isArray(ev.chain) || !ev.signature) return deny("AVF body needs chain[] and the attested key's signature over (transportKey || nonce)");
              const bound = Buffer.concat([spki, nonce]);
              res = verifyAvfEvidence({ chain: ev.chain.map((c) => Buffer.from(c, "base64")), challenge: createHash("sha256").update(bound).digest(),
                                        signature: Buffer.from(ev.signature, "base64"), signedMessage: bound },
                                      { allowedCodeHashes: attest.avf.codeHashes, allowedAuthorityHashes: attest.avf.authorityHashes || [],
                                        ...(attest.avf.rootPins ? { rootPins: attest.avf.rootPins } : {}) });
            } else {
              if (!/sev-snp-guest/.test(f.rad.format || "")) return deny(`format ${f.rad.format} not SEV-SNP or AVF`);
              const report = Buffer.from(f.rad.body, "base64");
              const aux = f.rad.certs ? Buffer.from(f.rad.certs, "base64") : null;
              res = await verifyQuote(report, { challenge: nonce, transportKeySpki: spki, auxblob: aux,
                allowedMeasurements: attest.allowedMeasurements || [], requireVcek: !!attest.requireVcek });
            }
            // verification is a network round trip (KDS): the timeout may have
            // denied and closed this socket while we waited. Binding it now
            // would register a dead ws whose 'close' has already fired — the
            // name would be held by a tunnel that answers nothing, forever.
            if (settled) return;
            if (!res.ok) return deny(res.reasons[res.reasons.length - 1] || "quote invalid");
            // WHOSE NAME IS THIS? A quote proves the image, not the box, so a
            // name that is REGISTERED ON CHAIN belongs to whoever registered it
            // — and only that operator's key can take it, whether the real box
            // is up, rebooting, or gone. Without this, a seller's downtime was
            // an opening: same image, same name, and the routing for its
            // registered id follows.
            const owner = await ownerOf(name);
            if (owner) {
              const signer = await signerOf(attachMessage(name, nonce), f.operatorSig);
              if (settled) return;                       // the timeout may have fired while we recovered
              if (!signer)
                return deny(`${name} is registered on chain; attach must carry operatorSig `
                          + `(personal_sign of "enclave-tunnel-attach:<name>:<nonce b64>") — upgrade the agent`);
              if (signer !== owner)
                return deny(`${name} is registered on chain to ${owner}, not ${signer}`);
            }
            const keyFp = spki ? createHash("sha256").update(spki).digest("hex") : "";
            const prev = tunnels.get(name);
            if (prev && (!prev.keyFp || prev.keyFp !== keyFp))
              return deny("that name is held by another enclave");
            clearTimeout(timer); settled = true;
            try { ws.send(JSON.stringify({ t: "attest-result", ok: true, measurement: res.measurement })); } catch {}
            const padKey = /^[0-9a-f]{64}$/.test(String(f.rad.padKey || "")) ? f.rad.padKey : "";
            bind(name, ws, { via: isAvf ? "attestation(avf)" : res.vcekVerified ? "attestation" : "attestation(measurement-only)",
                             measurement: res.measurement, mode: isAvf ? "avf" : "snp", keyFp,
                             spki: spki ? spki.toString("base64") : "", padKey });
          } catch (e) { deny(`verify error: ${e.message}`); }
          finally { verifying = false; }
        });
        ws.on("error", () => { try { ws.terminate(); } catch {} });
        try { ws.send(JSON.stringify({ t: "challenge", nonce: nonce.toString("base64") })); } catch {}
      });
    }

    socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
    socket.destroy();
  }

  // ---- raw streams over the control ws (Phase D) -----------------------------
  // A WebSocket UPGRADE can't ride the buffered req/res frames, so it gets a
  // spliced byte stream instead: the hub asks the agent to open a TCP
  // connection to the guest supervisor ({t:"s+"}), replays the client's
  // upgrade request head into it, and from then on both directions are opaque
  // {t:"sd"} chunks. The supervisor completes the handshake itself (101 flows
  // back through the splice), so the in-enclave /x/<id>/tls and /https bridges
  // — TLS terminating INSIDE the CVM — work unchanged behind a tunnel: this is
  // what makes a CGNAT seller box publicly serve its apps. The hub never
  // parses the spliced bytes; on the app-TLS path they are ciphertext
  // end-to-end. Bounded: per-tunnel stream cap, open timeout, idle timeout,
  // and a bufferedAmount guard so one slow reader can't balloon hub memory.
  const MAX_STREAMS = 128, STREAM_OPEN_MS = 10_000, STREAM_IDLE_MS = 15 * 60_000, MAX_WS_BUFFER = 16 * 1024 * 1024;
  function spliceUpgrade(origin, req, socket, head, path) {
    const name = (String(origin).match(NAME_RE) || [])[1];
    const t = tunnels.get(name);
    const refuse = (code, text) => { try { socket.write(`HTTP/1.1 ${code} ${text}\r\nConnection: close\r\n\r\n`); } catch {} socket.destroy(); };
    if (!t) return refuse(502, "Bad Gateway");
    if (t.streams.size >= MAX_STREAMS) return refuse(503, "Service Unavailable");
    const sid = seq++;
    const sendF = (o) => { try { t.ws.send(JSON.stringify(o)); return true; } catch { return false; } };
    let open = false, idleTimer = null;
    const openTimer = setTimeout(() => finish("stream open timeout"), STREAM_OPEN_MS);
    const idle = () => { clearTimeout(idleTimer); idleTimer = setTimeout(() => finish("idle"), STREAM_IDLE_MS); };
    function finish(why) {
      clearTimeout(openTimer); clearTimeout(idleTimer);
      if (t.streams.delete(sid)) sendF({ t: "sx", sid });
      if (!open && why !== "closed") refuse(502, "Bad Gateway"); else socket.destroy();
    }
    t.streams.set(sid, (f) => {
      idle();
      if (f.t === "s=" && !open) {
        if (!f.ok) return finish(f.err || "open refused");
        clearTimeout(openTimer); open = true;
        // replay the client's upgrade request into the guest supervisor: the
        // request line (path already rewritten by the caller) + headers as
        // received, then any bytes that arrived with the upgrade event
        let headStr = `${req.method} ${path} HTTP/1.1\r\n`;
        for (let i = 0; i < req.rawHeaders.length; i += 2) headStr += `${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}\r\n`;
        headStr += "\r\n";
        sendF({ t: "sd", sid, d: Buffer.concat([Buffer.from(headStr, "latin1"), head && head.length ? head : Buffer.alloc(0)]).toString("base64") });
        socket.on("data", (chunk) => {
          if (t.ws.bufferedAmount > MAX_WS_BUFFER) return finish("hub buffer overflow");
          idle(); sendF({ t: "sd", sid, d: chunk.toString("base64") });
        });
        socket.resume();
        return;
      }
      if (f.t === "sd" && open) { try { socket.write(Buffer.from(f.d || "", "base64")); } catch {} return; }
      if (f.t === "sx") { open = true; finish("closed"); }   // remote closed: plain teardown, no 502
    });
    socket.pause();
    socket.on("error", () => finish("closed"));
    socket.on("close", () => finish("closed"));
    idle();
    if (!sendF({ t: "s+", sid })) return finish("tunnel send failed");
  }

  let seq = 1;
  function send(name, method, path, headers, body) {
    const t = tunnels.get(name);
    if (!t) return Promise.reject(new Error(`no tunnel for ${name}`));
    const id = seq++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { t.pending.delete(id); reject(new Error("tunnel request timeout")); }, reqTimeoutMs);
      t.pending.set(id, { resolve: (f) => { clearTimeout(timer); resolve(f); }, reject: (e) => { clearTimeout(timer); reject(e); } });
      try { t.ws.send(JSON.stringify({ t: "req", id, method, path, headers, body: body ? body.toString("base64") : null })); }
      catch (e) { clearTimeout(timer); t.pending.delete(id); reject(e); }
    });
  }

  const NAME_RE = /^tunnel:\/\/(.+)$/;
  return {
    handleUpgrade,
    isTunnel: (origin) => NAME_RE.test(String(origin || "")),
    // One attached tunnel's identity, for modules that authenticate a tunnel's
    // own requests (relay/pads.mjs): null when nothing by that name is attached.
    info: (name) => { const t = tunnels.get(name); return t ? { name, mode: t.mode, keyFp: t.keyFp, spki: t.spki, padKey: t.padKey } : null; },
    nameOf: (origin) => (String(origin || "").match(NAME_RE) || [])[1] || null,
    // synthetic registry rows for the attached tunnels (bypass the dial-based
    // discovery filters; auth already happened at attach time). `endpoint`
    // keeps the tunnel:// scheme — it is the ROUTING KEY (isTunnel/proxyTo
    // dispatch on it); `name` is the human-facing label display surfaces use.
    origins: () => [...tunnels.entries()].map(([name, t]) => ({
      endpoint: `tunnel://${name}`, id: `tunnel:${name}`, name, repo: "EnclaveHost/enclave",
      lastSeen: Math.floor(t.lastSeen / 1000), tunnel: true, mode: t.mode, publicUrl: t.publicUrl,
      measurement: t.measurement || undefined,
    })),
    // fetch JSON (availability polling)
    fetchJson: async (origin, path) => {
      const name = (String(origin).match(NAME_RE) || [])[1];
      const r = await send(name, "GET", path, {}, null);
      if (r.status !== 200) return null;
      try { return JSON.parse(Buffer.from(r.body || "", "base64").toString("utf8")); } catch { return null; }
    },
    // full request/response for proxyTo (buffered)
    request: async (origin, { method, path, headers, body }) => {
      const name = (String(origin).match(NAME_RE) || [])[1];
      const r = await send(name, method || "GET", path, headers || {}, body);
      return { status: r.status || 502, headers: r.headers || {}, body: Buffer.from(r.body || "", "base64") };
    },
    count: () => tunnels.size,
    // websocket-upgrade splice into the guest supervisor (Phase D)
    spliceUpgrade,
  };
}
