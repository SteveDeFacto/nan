/* The owner app's part of dealt pads (shielded/dealer/PLAN.md): untrusted
 * plumbing between the pVM and the platform's ledger (relay/pads.mjs). It
 * never sees a secret: the pVM signs every request with its attested transport
 * key, the seed comes back boxed to the pVM's pad key, and windows come back
 * signed by the relay's ledger key for the pVM to verify. This class only
 * carries bytes: HTTP to the relay, lines on the control socket, and shipment
 * files into the VM over vsock 7780 (PADS <name> <bytes>, then the bytes). */
package host.enclave.anchor.avf;

import android.os.ParcelFileDescriptor;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import org.json.JSONObject;

final class PadsClient {
    static final int PADS_PORT = 7780;
    static volatile String base = "", seedId = "";

    /** The relay's HTTP base from its fleet-tunnel websocket URL. */
    static String httpBase(String wsUrl) {
        String u = wsUrl.replaceFirst("^wss://", "https://").replaceFirst("^ws://", "http://");
        int p = u.indexOf("/v1/"); return p > 0 ? u.substring(0, p) : u;
    }

    static JSONObject http(String method, String url, JSONObject body) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(20000); c.setReadTimeout(30000); c.setRequestMethod(method);
        if (body != null) {
            c.setDoOutput(true); c.setRequestProperty("Content-Type", "application/json");
            try (OutputStream o = c.getOutputStream()) { o.write(body.toString().getBytes(StandardCharsets.UTF_8)); }
        }
        int code = c.getResponseCode();
        InputStream in = code < 400 ? c.getInputStream() : c.getErrorStream();
        String text = in == null ? "{}" : new String(in.readAllBytes(), StandardCharsets.UTF_8);
        JSONObject j = new JSONObject(text.isEmpty() ? "{}" : text);
        j.put("_status", code);
        return j;
    }

    static String nonce() { byte[] n = new byte[16]; new SecureRandom().nextBytes(n); return RelayAttach.hex(n); }

    /** Read control lines until one starts with `prefix`; everything else is logged as usual. */
    static String until(BufferedReader r, String prefix) throws Exception {
        String line;
        while ((line = r.readLine()) != null) {
            Main.say("VSOCK " + (line.length() > 160 ? line.substring(0, 160) + "…" : line));
            if (line.startsWith(prefix)) return line;
        }
        return null;
    }

    /** After the tunnel is bound: the ledger key to the VM, the VM's signed seed request to the
     *  platform, the boxed seed back to the VM. True when the VM confirmed it opened the seed. */
    static boolean bootstrap(String httpBase, String name, OutputStream out, BufferedReader r) {
        base = httpBase;
        try {
            JSONObject key = http("GET", base + "/v1/pads/key", null);
            if (key.optInt("_status") != 200) { Main.say("PADS no ledger key: " + key); return false; }
            out.write(("PADLEDGER " + key.getString("key") + "\n").getBytes()); out.flush();
            String l = until(r, "PADLEDGER ");
            if (l == null || !l.endsWith("ok")) { Main.say("PADS the VM refused the ledger key"); return false; }
            String n = nonce();
            out.write(("PADSIGN seed " + n + " " + name + "\n").getBytes()); out.flush();
            l = until(r, "PADSIG ");
            if (l == null || l.endsWith("fail")) { Main.say("PADS the VM did not sign the seed request"); return false; }
            JSONObject res = http("POST", base + "/v1/pads/seed", new JSONObject().put("name", name).put("nonce", n).put("sig", l.substring(7).trim()));
            if (res.optInt("_status") != 200) { Main.say("PADS seed refused: " + res); return false; }
            seedId = res.getString("seed_id");
            out.write(("PADSEED " + name + " " + seedId + " " + res.getInt("epoch") + " " + res.getString("epk") + " " + res.getString("nonce") + " " + res.getString("box") + "\n").getBytes());
            out.flush();
            l = until(r, "PADSEED ");
            boolean ok = l != null && l.startsWith("PADSEED ok");
            Main.say("PADS seed " + (ok ? "installed in the VM: " + seedId : "NOT installed: " + l));
            return ok;
        } catch (Exception e) { Main.say("PADS bootstrap error " + e); return false; }
    }

    /** A window request from the engine (PADWIN want nonce sig): relay it, hand back the signed window. */
    static void onWindow(String line, String name, OutputStream out) {
        try {
            String[] f = line.trim().split(" ");
            if (f.length != 4) { out.write("PADWIN fail malformed\n".getBytes()); out.flush(); return; }
            JSONObject res = http("POST", base + "/v1/pads/reserve", new JSONObject().put("name", name).put("seed_id", seedId)
                .put("want", Long.parseLong(f[1])).put("nonce", f[2]).put("sig", f[3]));
            if (res.optInt("_status") == 200)
                out.write(("PADWIN " + res.getLong("lo") + " " + res.getLong("hi") + " " + res.getLong("iat") + " " + res.getString("sig") + "\n").getBytes());
            else out.write(("PADWIN fail " + res.optString("error", "http " + res.optInt("_status")) + "\n").getBytes());
            out.flush();
            Main.say("PADS window " + (res.optInt("_status") == 200 ? res.getLong("lo") + ".." + res.getLong("hi") : "refused " + res));
        } catch (Exception e) { Main.say("PADS window error " + e); try { out.write("PADWIN fail error\n".getBytes()); out.flush(); } catch (Exception ignored) { } }
    }

    /** The engine's usage receipt (RECEIPT name seed_id pads tokens nonce sig): relay it as signed.
     *  Nothing to hand back; the platform's totals are what billing reads. */
    static void onReceipt(String line) {
        try {
            String[] f = line.trim().split(" ");
            if (f.length != 7) { Main.say("PADS receipt malformed"); return; }
            JSONObject res = http("POST", base + "/v1/pads/receipt", new JSONObject().put("name", f[1]).put("seed_id", f[2])
                .put("pads", Long.parseLong(f[3])).put("tokens", Long.parseLong(f[4])).put("nonce", f[5]).put("sig", f[6]));
            Main.say("PADS receipt " + (res.optInt("_status") == 200 ? "recorded: " + f[3] + " pads, " + f[4] + " tokens (seed total " + res.optLong("pads") + "/" + res.optLong("tokens") + ", runs " + res.optLong("runs") + ")" : "refused " + res));
        } catch (Exception e) { Main.say("PADS receipt error " + e); }
    }

    /** Prefetch: the platform's store lists this seed's shipments; download the ones this phone
     *  does not hold yet (whole files, tmp-then-rename, so streamBank never sees a partial). */
    static void syncBank(java.io.File dir, String seedIdNow) {
        try {
            dir.mkdirs();
            JSONObject list = http("GET", base + "/v1/pads/shipments?seed_id=" + seedIdNow, null);
            org.json.JSONArray ships = list.optJSONArray("shipments");
            if (ships == null) return;
            for (int i = 0; i < ships.length(); i++) {
                JSONObject s = ships.getJSONObject(i);
                java.io.File f = new java.io.File(dir, s.getString("name"));
                if (f.exists() && f.length() == s.getLong("bytes")) continue;
                java.io.File tmp = new java.io.File(dir, "." + s.getString("name") + ".part");
                HttpURLConnection c = (HttpURLConnection) new URL(base + "/v1/pads/shipments/" + seedIdNow + "/" + s.getString("name")).openConnection();
                c.setConnectTimeout(20000); c.setReadTimeout(120000);
                if (c.getResponseCode() != 200) { Main.say("PADS fetch " + s.getString("name") + " http " + c.getResponseCode()); continue; }
                try (InputStream in = c.getInputStream(); OutputStream out = new FileOutputStream(tmp)) {
                    byte[] buf = new byte[1 << 20]; int n; while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                }
                if (tmp.length() == s.getLong("bytes") && tmp.renameTo(f)) Main.say("PADS fetched " + f.getName() + " (" + (f.length() >> 20) + " MiB)");
                else { tmp.delete(); Main.say("PADS fetch of " + s.getString("name") + " incomplete"); }
            }
        } catch (Exception e) { Main.say("PADS sync error " + e); }
    }

    /** The bank on this phone (P3 fetches it from the operator's box): every .pads file in `dir`,
     *  streamed into the VM one connection each. Files already streamed are skipped by name. */
    static void streamBank(Object vm, java.io.File dir) {
        java.util.Set<String> done = new java.util.HashSet<>();
        for (int round = 0; round < 3600 && !Main.ended(); round++) {
            if (!base.isEmpty() && !seedId.isEmpty() && round % 15 == 0) syncBank(dir, seedId);   // every ~15 s: what the platform has that we do not
            java.io.File[] files = dir.listFiles((d, n) -> n.endsWith(".pads"));
            if (files != null) {
                java.util.Arrays.sort(files);
                for (java.io.File f : files) {
                    if (done.contains(f.getName())) continue;
                    ParcelFileDescriptor pfd = Main.connect(vm, PADS_PORT, 50);
                    if (pfd == null) { Main.say("PADS connect failed"); return; }
                    try (OutputStream out = new FileOutputStream(pfd.getFileDescriptor()); InputStream in = new java.io.FileInputStream(f)) {
                        out.write(("PADS " + f.getName() + " " + f.length() + "\n").getBytes()); out.flush();
                        byte[] buf = new byte[1 << 20]; int n; long sent = 0;
                        while ((n = in.read(buf)) > 0) { out.write(buf, 0, n); sent += n; }
                        out.flush();
                        int ack = new java.io.FileInputStream(pfd.getFileDescriptor()).read();
                        Main.say("PADS " + f.getName() + " " + (sent >> 20) + " MiB " + (ack == 'K' ? "accepted" : "REFUSED"));
                        if (ack == 'K') done.add(f.getName());
                    } catch (Exception e) { Main.say("PADS stream error " + e); }
                    finally { try { pfd.close(); } catch (Exception ignored) { } }
                }
            }
            try { Thread.sleep(1000); } catch (InterruptedException e) { return; }
        }
    }
}
