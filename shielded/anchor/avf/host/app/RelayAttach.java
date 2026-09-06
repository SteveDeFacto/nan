/*
 * RelayAttach -- the phone presents its protected VM to the fleet relay.
 *
 * The same gate a self-hosted SEV-SNP box crosses (relay/tunnel.js), in the
 * shape AVF offers. The relay sends a nonce; the app binds it to the VM's
 * transport key (bound = SPKI || nonce) and asks the VM for a certificate over
 * sha256(bound) and a signature over bound with the attested key; the relay
 * verifies the chain to Google's root, the anchor's code hash and the
 * signature (relay/avf-verify.mjs), and binds the tunnel with mode "avf".
 *
 * The app sees the certificate chain (public) and the transport key's PUBLIC
 * half; the VM keeps the private halves. It cannot forge any of it, which is
 * the point: the operator owns this app and it changes nothing.
 */
package host.enclave.anchor.avf;

import org.json.JSONArray;
import org.json.JSONObject;

import java.security.MessageDigest;
import java.util.Base64;
import java.util.Map;
import java.util.TreeMap;

public final class RelayAttach {
    final String url, name; final byte[] spki;
    Ws ws; byte[] nonce, bound; String challengeHex;
    String padKey = "";                       // the VM's X25519 pad key (PADKEY), presented with the attestation

    RelayAttach(String url, String name, byte[] spki) { this.url = url; this.name = name; this.spki = spki; }

    static byte[] sha256(byte[] b) { try { return MessageDigest.getInstance("SHA-256").digest(b); } catch (Exception e) { throw new RuntimeException(e); } }
    static String hex(byte[] b) { StringBuilder s = new StringBuilder(); for (byte x : b) s.append(String.format("%02x", x)); return s.toString(); }
    static byte[] unhex(String h) { byte[] b = new byte[h.length() / 2]; for (int i = 0; i < b.length; i++) b[i] = (byte) Integer.parseInt(h.substring(2 * i, 2 * i + 2), 16); return b; }
    static String b64(byte[] b) { return Base64.getEncoder().encodeToString(b); }

    /** Dial the relay and take its nonce; returns the hex challenge the VM must certify. */
    String challenge() throws Exception {
        ws = new Ws(url, Map.of("x-metal-name", name, "x-metal-attest", "1"));
        String f; JSONObject ch = null;
        while (ch == null && (f = ws.receive()) != null) { JSONObject o = new JSONObject(f); if ("challenge".equals(o.optString("t"))) ch = o; }
        if (ch == null) throw new Exception("relay closed before sending a challenge");
        nonce = Base64.getDecoder().decode(ch.getString("nonce"));
        bound = new byte[spki.length + nonce.length]; System.arraycopy(spki, 0, bound, 0, spki.length); System.arraycopy(nonce, 0, bound, spki.length, nonce.length);
        challengeHex = hex(sha256(bound));
        Main.say("RELAY " + url + " as " + name + ": nonce=" + hex(nonce).substring(0, 16) + "… challenge=" + challengeHex.substring(0, 16) + "…");
        return challengeHex;
    }

    /** Present what the VM produced; returns the relay's verdict frame. */
    JSONObject present(TreeMap<Integer, TreeMap<Integer, String>> certs, TreeMap<Integer, String> sig) throws Exception {
        JSONArray chain = new JSONArray();
        for (TreeMap<Integer, String> chunks : certs.values()) chain.put(b64(unhex(String.join("", chunks.values()))));
        JSONObject ev = new JSONObject().put("chain", chain);
        if (!sig.isEmpty()) ev.put("signature", b64(unhex(String.join("", sig.values()))));
        JSONObject rad = new JSONObject().put("format", "android-avf-pvm/v1").put("body", b64(ev.toString().getBytes("UTF-8")))
            .put("transportKey", b64(spki)).put("transportKeyFp", hex(sha256(spki))).put("name", name);
        if (!padKey.isEmpty()) rad.put("padKey", padKey);
        ws.sendText(new JSONObject().put("t", "attest").put("rad", rad).toString());
        Main.say("RELAY presented chain=" + chain.length() + " certs signature=" + (sig.isEmpty() ? "none" : "yes"));
        String f; JSONObject res = null;
        while (res == null && (f = ws.receive()) != null) { JSONObject o = new JSONObject(f); if ("attest-result".equals(o.optString("t"))) res = o; }
        if (res == null) { Main.say("RELAY closed without a verdict"); return null; }
        Main.say("RELAY attest " + (res.optBoolean("ok") ? "ACCEPTED measurement=" + res.optString("measurement") : "REJECTED: " + res.optString("reason")));
        return res;
    }

    /** Bound: announce the identity, then answer the hub until the socket ends. */
    void serve(String phone) {
        try {
            ws.sendText(new JSONObject().put("t", "hello").put("name", name).put("mode", "avf").put("transportKeyFp", hex(sha256(spki))).toString());
            String f;
            while ((f = ws.receive()) != null) {
                JSONObject o = new JSONObject(f); String t = o.optString("t");
                if ("ping".equals(t)) { ws.sendText("{\"t\":\"pong\"}"); continue; }
                if ("s+".equals(t)) { ws.sendText(new JSONObject().put("t", "s=").put("sid", o.opt("sid")).put("ok", false).put("err", "phone anchor carries no streams").toString()); continue; }
                if (!"req".equals(t)) continue;
                String path = o.optString("path").split("\\?")[0]; int status; JSONObject body;
                if (path.equals("/availability")) { status = 200; body = new JSONObject().put("ok", true).put("role", "phone-anchor").put("name", name).put("phone", phone).put("gpu", false).put("teeCpu", "android-avf-pvm"); }
                else if (path.equals("/v1/health")) { status = 200; body = new JSONObject().put("ok", true).put("role", "phone-anchor").put("name", name); }
                else { status = 404; body = new JSONObject().put("error", "not_found"); }
                ws.sendText(new JSONObject().put("t", "res").put("id", o.opt("id")).put("status", status)
                    .put("headers", new JSONObject().put("content-type", "application/json")).put("body", b64(body.toString().getBytes("UTF-8"))).toString());
            }
            Main.say("RELAY tunnel closed");
        } catch (Exception e) { Main.say("RELAY serve error " + e); }
    }

    void close() { try { if (ws != null) ws.close(); } catch (Exception ignored) { } }
}
