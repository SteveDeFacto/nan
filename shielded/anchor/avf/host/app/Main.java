/*
 * Main -- the host side of the anchor on a Pixel: the app that OWNS the
 * protected VM, and the only thing that can talk to it.
 *
 * A production pVM is non-debuggable: no console, no log, no ramdump. The
 * shell domain may not even open AF_VSOCK (measured). What AVF gives the
 * VM's owner is VirtualMachine.connectVsock(), so this app is the guest's
 * whole outside world:
 *
 *   gate      is this phone one we support? (protected VMs + attestation)
 *   own       build the config (protected, DEBUG_LEVEL_NONE, match-host), run it
 *   control   vsock 7777: send a challenge and the run plan, relay every
 *             line the anchor says to logcat ("anchor-host") and the screen
 *   bridge    vsock 7778: pipe each worker connection to a TCP shielded
 *             worker. Only ciphertext frames cross it; the app never sees a
 *             pad, an activation or a product.
 *
 * android.system.virtualmachine is a @SystemApi: absent from the public SDK
 * android.jar but callable at runtime, so every call goes through reflection.
 *
 *   adb shell pm grant host.enclave.anchor.avf android.permission.MANAGE_VIRTUAL_MACHINE
 *   adb reverse tcp:9500 tcp:9500
 *   adb shell am start-foreground-service -n host.enclave.anchor.avf/.AnchorService \
 *       [--es worker 127.0.0.1:9500] [--es mode bridge|local] [--es shapes "K,N,nodes,iters,xmax;..."]
 *   adb logcat -s anchor-host
 */
package host.enclave.anchor.avf;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.widget.ScrollView;
import android.widget.TextView;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.net.Socket;
import java.security.SecureRandom;
import java.util.TreeMap;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class Main extends Activity {
    static final String TAG = "anchor-host";
    static final String PKG = "android.system.virtualmachine.";
    static final int CTRL_PORT = 7777, WORKER_PORT = 7778, MODEL_PORT = 7779;
    static final int VENDOR_LEVEL_ATTEST = 202404;      // /avf RKP component min-level

    /* run plan, from intent extras */
    public static final class Plan {
        String payload = "libanchor.so"; int debug = 0; long memMib = 1024;
        String worker = "127.0.0.1:9500"; String mode = "bridge";
        String relay = null; String name = "phone-anchor";
        String model = "/data/local/tmp/anchor/gg/model.gguf"; String prompt = "The capital of France is"; int n = 8; int threads = 4; long storageMib = 0;
        int mtp = 0;                         // engine: draft k tokens per round with the model's own MTP head (0 = plain decode)
        int boost = 0;                       // engine: spinning threads in the VM that keep the phone's clocks up between exchanges
        int burners = 0;                     // app-side: lowest-priority spinning threads that keep the clusters' clocks up while the VM decodes
        String shenv = "";                   // engine: extra environment for the VM engine, "K=V,K=V" (e.g. SHIELDED_LOCAL_SITES=token_embd.weight)
        String shapes = "256,256,1,30,0;896,896,1,30,0;896,4864,2,12,0";
        String pads = "";                    // dealt pads: bank dir of .pads files on this phone; "" = the VM mints its own
        String prefix = "", prefixPk = "";   // shared-prefix KV dir (prefix.kv + .sig + prefix.txt) and the platform's prefix key
        String prefixName = "", prefixDigest = "";   // or fetch them from the platform store by (model digest, name)
        static Plan from(Intent i) {
            Plan p = new Plan(); if (i == null) return p;
            if (i.getStringExtra("payload") != null) p.payload = i.getStringExtra("payload");
            p.debug = i.getIntExtra("debug", p.debug); p.memMib = i.getIntExtra("mem", (int) p.memMib);
            if (i.getStringExtra("worker") != null) p.worker = i.getStringExtra("worker");
            if (i.getStringExtra("mode") != null) p.mode = i.getStringExtra("mode");
            if (i.getStringExtra("shapes") != null) p.shapes = i.getStringExtra("shapes");
            if (i.getStringExtra("relay") != null) p.relay = i.getStringExtra("relay");
            if (i.getStringExtra("name") != null) p.name = i.getStringExtra("name");
            if (i.getStringExtra("model") != null) p.model = i.getStringExtra("model");
            if (i.getStringExtra("prompt") != null) p.prompt = i.getStringExtra("prompt");
            if (i.getStringExtra("pads") != null) p.pads = i.getStringExtra("pads");     // dealt pads: a bank dir of .pads files on this phone ("" = off)
            if (i.getStringExtra("prefix") != null) p.prefix = i.getStringExtra("prefix");   // shared-prefix KV: a dir holding prefix.kv, prefix.kv.sig, prefix.txt
            if (i.getStringExtra("prefixpk") != null) p.prefixPk = i.getStringExtra("prefixpk");   // the platform's prefix key (64 hex) the VM pins
            if (i.getStringExtra("prefixname") != null) p.prefixName = i.getStringExtra("prefixname");       // fetch <name>.kv/.sig/.txt from the platform's store...
            if (i.getStringExtra("prefixdigest") != null) p.prefixDigest = i.getStringExtra("prefixdigest"); // ...for this model digest, into files/prefix
            p.n = i.getIntExtra("n", p.n); p.threads = i.getIntExtra("threads", p.threads); p.mtp = i.getIntExtra("mtp", p.mtp); p.boost = i.getIntExtra("boost", p.boost); p.burners = i.getIntExtra("burners", p.burners); if (i.getStringExtra("shenv") != null) p.shenv = i.getStringExtra("shenv"); paceBytesPerSec = (long) i.getIntExtra("pace_mbps", 0) << 20; p.storageMib = i.getIntExtra("storage", (int) p.storageMib);
            if (p.mode.equals("engine")) {                                         // the model lives in the VM
                if (i.getIntExtra("mem", 0) == 0) p.memMib = 4096;
                if (i.getIntExtra("storage", 0) == 0) p.storageMib = 2048;             // encrypted storage: the model's home, kept across runs
            }
            return p;
        }
    }

    private static volatile TextView sScreen;
    static void say(String s) {
        Log.i(TAG, s);
        TextView t = sScreen;
        if (t != null) t.post(() -> t.append(s + "\n"));
    }

    @Override protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        TextView t = new TextView(this); t.setTextSize(11); t.setPadding(24, 48, 24, 24); t.setTypeface(android.graphics.Typeface.MONOSPACE);
        ScrollView sv = new ScrollView(this); sv.addView(t); setContentView(sv); sScreen = t;
        final Plan plan = Plan.from(getIntent());
        new Thread(() -> runVm(this, plan), "anchor-host").start();
    }
    @Override protected void onDestroy() { sScreen = null; super.onDestroy(); }

    /* call obj.name(args) resolving the method by name and arity; Method.invoke unboxes primitives */
    static Object call(Object obj, String name, Object... args) throws Exception {
        Class<?> c = obj instanceof Class ? (Class<?>) obj : obj.getClass();
        for (Class<?> k = c; k != null; k = k.getSuperclass())
            for (Method m : k.getMethods())
                if (m.getName().equals(name) && m.getParameterCount() == args.length) {
                    m.setAccessible(true);
                    return m.invoke(obj instanceof Class ? null : obj, args);
                }
        throw new NoSuchMethodException(c.getName() + "." + name + "/" + args.length);
    }
    static Object tryCall(Object obj, String name, Object... args) { try { return call(obj, name, args); } catch (Throwable t) { return null; } }
    static int sysprop(String key) {
        try { Class<?> sp = Class.forName("android.os.SystemProperties"); return (Integer) sp.getMethod("getInt", String.class, int.class).invoke(null, key, 0); }
        catch (Throwable t) { return 0; }
    }

    /* The support gate. The list customers see ("Pixel 9a, and Pixel 10 or newer") is the
     * translation of this: protected VMs must exist, and the vendor level must admit the
     * RKP /avf component. The attestation itself is the real test and runs right after. */
    static boolean gate(Object vmm) {
        int vendor = sysprop("ro.vendor.api_level"), board = sysprop("ro.board.api_level");
        Object caps = tryCall(vmm, "getCapabilities");
        int c = caps instanceof Integer ? (Integer) caps : -1;
        boolean protectedVm = c < 0 ? Boolean.TRUE.equals(tryCall(vmm, "isProtectedVmSupported")) : (c & 1) != 0;   // CAPABILITY_PROTECTED_VM
        Object ra = tryCall(vmm, "isRemoteAttestationSupported");
        say("GATE device=" + android.os.Build.MODEL + " sdk=" + android.os.Build.VERSION.SDK_INT + " vendor_api_level=" + vendor + " board_api_level=" + board);
        say("GATE capabilities=" + c + " protected_vm=" + protectedVm + " remote_attestation=" + (ra == null ? "n/a" : ra));
        boolean attestLevel = vendor >= VENDOR_LEVEL_ATTEST;
        boolean ok = protectedVm && attestLevel && !Boolean.FALSE.equals(ra);
        say("GATE " + (ok ? "SUPPORTED" : "UNSUPPORTED") + (protectedVm ? "" : " (no protected VMs)") + (attestLevel ? "" : " (launch generation " + vendor + " < " + VENDOR_LEVEL_ATTEST + ": /avf not provisioned)") + (Boolean.FALSE.equals(ra) ? " (service says no attestation)" : ""));
        return ok;
    }

    static java.io.File filesDir = new java.io.File("/data/user/0/host.enclave.anchor.avf/files");
    static void runVm(Context ctx, Plan plan) {
        filesDir = ctx.getFilesDir();
        try {
            say("HOST start payload=" + plan.payload + " debug=" + plan.debug + " mem=" + plan.memMib + "MiB worker=" + plan.worker + " mode=" + plan.mode + " host=" + ctx.getClass().getSimpleName());
            Object vmm = ctx.getSystemService("virtualization");
            if (vmm == null) { say("HOST no VirtualMachineManager: this build of Android has no AVF"); return; }
            gate(vmm);      // informative; the run proceeds so an unsupported phone still shows what it can do

            Class<?> cBuilder = Class.forName(PKG + "VirtualMachineConfig$Builder");
            Object b = cBuilder.getConstructor(Context.class).newInstance(ctx);
            call(b, "setPayloadBinaryName", plan.payload);
            call(b, "setProtectedVm", true);
            call(b, "setDebugLevel", plan.debug);
            call(b, "setMemoryBytes", plan.memMib << 20);
            call(b, "setCpuTopology", 1);            // CPU_TOPOLOGY_MATCH_HOST
            if (plan.storageMib > 0) say("HOST encrypted storage " + plan.storageMib + " MiB: " + (tryCall(b, "setEncryptedStorageBytes", plan.storageMib << 20) != null ? "set" : "not available"));
            Object cfg = call(b, "build");
            say("HOST config protected=" + call(cfg, "isProtectedVm") + " debug=" + call(cfg, "getDebugLevel"));

            /* Keep the VM instance across runs: its encrypted storage is where the model
             * lives, and deleting the VM deletes it. Recreate only when the stored
             * config no longer matches (getOrCreate refuses an incompatible one). */
            Object vm0;
            try { vm0 = call(vmm, "getOrCreate", "anchor", cfg); }
            catch (Exception e) { say("HOST existing VM incompatible with this config (" + (e.getCause() != null ? e.getCause().getMessage() : e.getMessage()) + "): recreating"); try { call(vmm, "delete", "anchor"); } catch (Exception ignored) { } vm0 = call(vmm, "getOrCreate", "anchor", cfg); }
            final Object vm = vm0;
            Class<?> cCb = Class.forName(PKG + "VirtualMachineCallback");
            Executor ex = Executors.newSingleThreadExecutor();
            InvocationHandler h = (proxy, m, a) -> {
                String n = m.getName();
                if (n.equals("toString")) return "cb"; if (n.equals("hashCode")) return 0; if (n.equals("equals")) return proxy == a[0];
                switch (n) {
                    case "onPayloadStarted": say("VM payload started"); break;
                    case "onPayloadReady": say("VM payload ready"); new Thread(() -> control(vm, plan), "vsock-control").start(); break;
                    case "onPayloadFinished": say("VM payload finished exit=" + a[1]); break;
                    case "onError": say("VM error code=" + a[1] + " msg=" + a[2]); break;
                    case "onStopped": say("VM stopped reason=" + a[1]); break;
                    default: say("VM cb " + n);
                }
                return null;
            };
            Object cb = Proxy.newProxyInstance(cCb.getClassLoader(), new Class<?>[] { cCb }, h);
            call(vm, "setCallback", ex, cb);
            call(vm, "run");
            say("HOST vm.run() returned, status=" + call(vm, "getStatus"));
        } catch (Throwable t) {
            Log.e(TAG, "HOST FAIL", t); say("HOST FAIL " + t);
        }
    }

    /* the guest binds its listeners before notifyPayloadReady, but be tolerant anyway */
    static ParcelFileDescriptor connect(Object vm, int port, int tries) {
        for (int i = 0; i < tries; i++) {
            try { return (ParcelFileDescriptor) call(vm, "connectVsock", port); }
            catch (Throwable t) { try { Thread.sleep(200); } catch (InterruptedException ignored) { } }
        }
        return null;
    }

    private static volatile boolean sEnded;
    static boolean ended() { return sEnded; }
    static void control(Object vm, Plan plan) {
        sEnded = false;
        ParcelFileDescriptor pfd = connect(vm, CTRL_PORT, 50);
        if (pfd == null) { say("CONTROL connect failed"); return; }
        say("CONTROL connected");
        if (plan.mode.equals("bridge") || plan.mode.equals("engine")) new Thread(() -> bridge(vm, plan), "vsock-bridge").start();
        if (plan.mode.equals("engine")) new Thread(() -> streamModel(vm, plan), "vsock-model").start();
        RelayAttach relay = null;
        try (OutputStream out = new FileOutputStream(pfd.getFileDescriptor());
             BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(pfd.getFileDescriptor())))) {
            // 1. the VM's transport key is the first thing it says
            String first = r.readLine();
            byte[] spki = first != null && first.startsWith("SPKI ") ? RelayAttach.unhex(first.substring(5).trim()) : null;
            say("VSOCK " + first);
            // 1b. its pad key (dealt pads): the platform boxes the VM's seed to it
            String second = r.readLine();
            String padKey = second != null && second.startsWith("PADKEY ") ? second.substring(7).trim() : "";
            if (second != null) say("VSOCK " + second);
            // 2. the challenge: the relay's, bound to the transport key, or a local one
            String chal, boundHex = "";
            if (plan.relay != null && spki != null) {
                relay = new RelayAttach(plan.relay, plan.name, spki);
                relay.padKey = padKey;
                try { chal = relay.challenge(); boundHex = RelayAttach.hex(relay.bound); }
                catch (Exception e) { say("RELAY dial failed: " + e + " (continuing with a local challenge)"); relay = null; byte[] c = new byte[32]; new SecureRandom().nextBytes(c); chal = RelayAttach.hex(c); }
            } else { byte[] c = new byte[32]; new SecureRandom().nextBytes(c); chal = RelayAttach.hex(c); }
            if (!boundHex.isEmpty()) out.write(("BOUND " + boundHex + "\n").getBytes());
            out.write(("CHAL " + chal + "\n").getBytes()); out.flush();
            say("CONTROL challenge=" + chal);
            // 3. what the VM produced: status, the chain in chunks, the signature
            TreeMap<Integer, TreeMap<Integer, String>> certs = new TreeMap<>(); TreeMap<Integer, String> sig = new TreeMap<>();
            String line;
            while ((line = r.readLine()) != null) {
                say("VSOCK " + (line.length() > 160 ? line.substring(0, 160) + "…(" + line.length() + ")" : line));
                if (line.equals("ATTEST end")) break;
                java.util.regex.Matcher m;
                if ((m = java.util.regex.Pattern.compile("^CERT(\\d+)\\[(\\d+)\\] ([0-9a-f]+)$").matcher(line)).matches())
                    certs.computeIfAbsent(Integer.parseInt(m.group(1)), (k) -> new TreeMap<>()).put(Integer.parseInt(m.group(2)), m.group(3));
                else if ((m = java.util.regex.Pattern.compile("^SIG\\[(\\d+)\\] ([0-9a-f]+)$").matcher(line)).matches())
                    sig.put(Integer.parseInt(m.group(1)), m.group(2));
            }
            // 4. present it; a bound tunnel keeps serving the hub in its own thread
            if (relay != null) {
                JSONObject res = relay.present(certs, sig);
                if (res != null && res.optBoolean("ok")) { final RelayAttach rr = relay; new Thread(() -> rr.serve(android.os.Build.MODEL), "relay-serve").start(); }
                else { relay.close(); relay = null; }
            }
            // 4b. dealt pads: once the tunnel is bound, fetch the VM's seed through the platform's ledger
            boolean pads = false;
            if (relay != null && !padKey.isEmpty() && !plan.pads.isEmpty())
                pads = PadsClient.bootstrap(PadsClient.httpBase(plan.relay), plan.name, out, r);
            // 4c. shared-prefix KV: the VM pins the platform's prefix key; the files follow over the pads port
            boolean prefix = false;
            if (plan.prefix.isEmpty() && !plan.prefixName.isEmpty() && plan.prefixDigest.matches("[0-9a-f]{64}") && relay != null) {
                java.io.File pdir = new java.io.File(filesDir, "prefix");
                if (PadsClient.fetchPrefix(PadsClient.httpBase(plan.relay), plan.prefixDigest, plan.prefixName, pdir)) plan.prefix = pdir.getPath();
                else say("PREFIX " + plan.prefixName + " not fetched; running without the shared prefix");
            }
            if (!plan.prefix.isEmpty() && plan.prefixPk.matches("[0-9a-f]{64}")) {
                out.write(("PREFIXPK " + plan.prefixPk + "\n").getBytes()); out.flush();
                String pl = PadsClient.until(r, "PREFIXPK ");
                prefix = pl != null && pl.equals("PREFIXPK ok");
                say("PREFIX key " + (prefix ? "pinned in the VM" : "NOT accepted: " + pl));
            }
            // 5. the run plan
            StringBuilder cmd = new StringBuilder();
            if (plan.mode.equals("engine")) {
                long bytes = new java.io.File(plan.model).length();
                String sha = RelayAttach.hex(fileSha256(plan.model));
                cmd.append("ENGINE model_bytes=").append(bytes).append(" model_sha256=").append(sha).append(" n=").append(plan.n).append(" threads=").append(plan.threads).append(" mtp=").append(plan.mtp).append(" boost=").append(plan.boost).append(plan.shenv.isEmpty() ? "" : " env=" + RelayAttach.hex(plan.shenv.getBytes("UTF-8")))
                   .append(" prompt=").append(RelayAttach.hex(plan.prompt.getBytes("UTF-8"))).append(pads ? " pads=1" : "").append(prefix ? " prefix=1" : "").append('\n');
                startBurners(plan.burners);
                say("ENGINE plan: " + plan.model + " (" + (bytes >> 20) + " MiB), " + plan.n + " tokens, " + plan.threads + " threads" + (plan.mtp > 0 ? ", MTP draft k=" + plan.mtp : "") + (pads ? ", dealt pads from " + plan.pads : ""));
                if (pads) { final java.io.File bank = new java.io.File(plan.pads); new Thread(() -> PadsClient.streamBank(vm, bank), "vsock-pads").start(); }
                if (prefix) { final java.io.File pdir = new java.io.File(plan.prefix); new Thread(() -> PadsClient.streamFiles(vm, pdir, new String[] { "prefix.kv", "prefix.kv.sig", "prefix.txt" }), "vsock-prefix").start(); }
            }
            if (plan.mode.equals("echo")) { cmd.append("ECHO\n"); new Thread(() -> echoBench(vm), "vsock-echo").start(); }
            cmd.append("WORKER ").append(plan.mode.equals("engine") ? "bridge" : plan.mode).append('\n');
            for (String s : plan.shapes.split(";")) { String[] f = s.trim().split(","); if (f.length == 5) cmd.append("SHAPE ").append(String.join(" ", f)).append('\n'); }
            cmd.append("RUN\n");
            out.write(cmd.toString().getBytes()); out.flush();
            int n = 0;
            while ((line = r.readLine()) != null) {
                say("VSOCK " + line); n++;
                if (line.startsWith("PADWIN ")) PadsClient.onWindow(line, plan.name, out);   // the engine asks for a ledger window
                if (line.startsWith("RECEIPT ")) PadsClient.onReceipt(line);                 // the engine's signed usage
                if (line.equals("END")) break;
            }
            say("CONTROL closed after " + n + " lines");
        } catch (Exception e) {
            say("CONTROL error " + e);
        } finally {
            sEnded = true;
            try { pfd.close(); } catch (Exception ignored) { }
            if (relay != null) relay.close();
        }
    }

    /* vsock round trip, app <-> guest: the floor under every exchange the bridge carries */
    static void echoBench(Object vm) {
        ParcelFileDescriptor pfd = connect(vm, 7780, 100);
        if (pfd == null) { say("ECHO connect failed"); return; }
        try (OutputStream out = new FileOutputStream(pfd.getFileDescriptor()); InputStream in = new FileInputStream(pfd.getFileDescriptor())) {
            for (int size : new int[] { 64, 4096, 65536 }) {
                byte[] b = new byte[size]; long[] us = new long[200];
                for (int i = 0; i < 200; i++) {
                    long t0 = System.nanoTime(); out.write(b); out.flush();
                    int got = 0; while (got < size) { int r = in.read(b, got, size - got); if (r < 0) throw new java.io.EOFException(); got += r; }
                    us[i] = (System.nanoTime() - t0) / 1000;
                }
                java.util.Arrays.sort(us);
                say("ECHO " + size + " B: p50=" + us[100] + " us p90=" + us[180] + " us min=" + us[0] + " us");
            }
        } catch (Exception e) { say("ECHO error " + e); }
        finally { try { pfd.close(); } catch (Exception ignored) { } }
    }

    static byte[] fileSha256(String path) {
        try (InputStream in = new java.io.FileInputStream(path)) {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 20]; int r; while ((r = in.read(buf)) > 0) md.update(buf, 0, r);
            return md.digest();
        } catch (Exception e) { say("MODEL sha256 failed: " + e); return new byte[32]; }
    }

    /* engine mode: the public model, streamed into the guest (8-byte length, then the bytes) */
    static void streamModel(Object vm, Plan plan) {
        ParcelFileDescriptor pfd = connect(vm, MODEL_PORT, 150);
        if (pfd == null) { say("MODEL connect failed"); return; }
        try (OutputStream out = new FileOutputStream(pfd.getFileDescriptor()); InputStream in = new java.io.FileInputStream(plan.model)) {
            long bytes = new java.io.File(plan.model).length();
            byte[] hdr = new byte[8]; for (int i = 0; i < 8; i++) hdr[i] = (byte) (bytes >>> (8 * i));
            out.write(hdr); out.flush();
            int ans = new java.io.FileInputStream(pfd.getFileDescriptor()).read();     // 'K' = the VM already holds it, 'S' = send
            if (ans == 'K') { say("MODEL already in the VM's encrypted storage (" + (bytes >> 20) + " MiB), not streamed"); return; }
            if (ans != 'S') { say("MODEL guest answered " + ans + ", not streaming"); return; }
            byte[] buf = new byte[1 << 20]; long sent = 0; int r; long t0 = System.nanoTime();
            while ((r = in.read(buf)) > 0) { out.write(buf, 0, r); sent += r; }
            out.flush();
            say("MODEL streamed " + (sent >> 20) + " MiB in " + ((System.nanoTime() - t0) / 1_000_000) + " ms");
        } catch (Exception e) { say("MODEL stream error " + e); }
        finally { try { pfd.close(); } catch (Exception ignored) { } }
    }

    /* one worker connection per shape: connect into the guest, dial the TCP worker, pipe both ways, repeat */
    static void bridge(Object vm, Plan plan) {
        String host = plan.worker.substring(0, plan.worker.lastIndexOf(':')); int port = Integer.parseInt(plan.worker.substring(plan.worker.lastIndexOf(':') + 1));
        int conn = 0;
        while (!sEnded) {
            ParcelFileDescriptor pfd = connect(vm, WORKER_PORT, 25);
            if (pfd == null) break;
            conn++;
            // A plain AF_INET socket (not Java's dual-stack v6-mapped one): over the USB NCM link the
            // engine's stream from the app stalled the link within seconds while the same bytes from a
            // shell (AF_INET) did not; bisecting that starts here.
            try (Socket s = new Socket(java.net.Inet4Address.getByName(host), port)) {
                s.setTcpNoDelay(true);
                final int id = conn;
                say("BRIDGE #" + id + " guest<->" + plan.worker);
                InputStream gi = new FileInputStream(pfd.getFileDescriptor()); OutputStream go = new FileOutputStream(pfd.getFileDescriptor());
                InputStream wi = s.getInputStream(); OutputStream wo = s.getOutputStream();
                Thread up = new Thread(() -> pipe(gi, wo, "up"), "bridge-up"); up.start();
                long[] down = { pipe(wi, go, "down") };
                try { s.shutdownInput(); } catch (Exception ignored) { }
                try { pfd.close(); } catch (Exception ignored) { }
                up.join();
                say("BRIDGE #" + id + " closed, down=" + down[0] + " bytes");
            } catch (Exception e) {
                say("BRIDGE error " + e);
                try { pfd.close(); } catch (Exception ignored) { }
                if (!sEnded) { try { Thread.sleep(300); } catch (InterruptedException ignored) { } }
            }
        }
    }
    // The VM's decode is ~100 short compute bursts per token between link waits; the phone's governor
    // answers that duty cycle with 0.4 GHz on the mid cores. Spinning threads INSIDE the VM (engine
    // ANCHOR_BOOST_THREADS) made it worse: to the phone they are normal-priority crosvm threads and
    // they preempt the compute vCPUs. Burners in the app at the lowest priority (nice 19) keep the
    // clusters' clocks up and yield to the vCPU threads: measured +57% tokens/s as shell burners.
    static volatile boolean burnersOn = false;
    static void startBurners(int n) {
        burnersOn = n > 0;
        for (int i = 0; i < n; i++) {
            Thread t = new Thread(() -> {
                try { android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_LOWEST); } catch (Exception ignored) { }
                long x = 0; while (burnersOn) { x += x * 31 + 7; if ((x & 0xffff) == 1) Thread.onSpinWait(); }
            }, "burner-" + i);
            t.setDaemon(true); t.start();
        }
    }
    static volatile long paceBytesPerSec = 0;   // > 0: cap the guest->worker pump (the weight upload) to this rate; bursts up to 2 MB pass
    static long pipe(InputStream in, OutputStream out, String dir) {
        long total = 0; byte[] buf = new byte[65536];
        // Token bucket for the "up" pump: the USB NCM link to the host wedged under sustained line-rate
        // uploads (netbench: nondeterministic at ~40 MB/s, never seen at <= 24 MB/s); an exchange's
        // burst (<= 2 MB) is never delayed, only a long stream is.
        long tokens = 2L << 20, tLast = System.nanoTime(); final boolean pace = dir.equals("up") && paceBytesPerSec > 0;
        // (No spin on available(): the vsock stream lacks FIONREAD, and spinning the TCP side made the
        //  exchange slower, 5.4 -> 18 ms, by starving the phone's other threads; run24.)
        try {
            int n;
            for (;;) {
                if ((n = in.read(buf)) <= 0) break;
                if (pace) {
                    long now = System.nanoTime(); tokens = Math.min(2L << 20, tokens + (now - tLast) * paceBytesPerSec / 1_000_000_000L); tLast = now;
                    if (tokens < n) { long wait = (n - tokens) * 1_000_000_000L / paceBytesPerSec; try { Thread.sleep(wait / 1_000_000L, (int) (wait % 1_000_000L)); } catch (InterruptedException ignored) { } tokens = 0; }
                    else tokens -= n;
                }
                out.write(buf, 0, n); out.flush(); total += n;
            }
        } catch (Exception ignored) { }
        try { out.close(); } catch (Exception ignored) { }
        return total;
    }
}
