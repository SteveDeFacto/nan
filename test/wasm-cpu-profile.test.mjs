import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import { execFileSync } from "node:child_process";

const manager = new URL("../wasm/wasm_manager.py", import.meta.url).pathname;
function python(body) {
  return execFileSync("python3", ["-c", `
import importlib.util, sys, os, json, pathlib, tempfile, time, subprocess, base64
from unittest.mock import patch
spec = importlib.util.spec_from_file_location("wm", ${JSON.stringify(manager)})
m = importlib.util.module_from_spec(spec); sys.modules["wm"] = m
spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
 m.LOG_DIR = pathlib.Path(tmp)
${body.split("\n").map(l => " " + l).join("\n")}
`], { encoding: "utf8", env: { ...process.env, NODE_HAS_GPU: "0" }, timeout: 15000 });
}

test("profiling is strictly opt-in and uses a private fixed path", () => {
  python(`
for cfg in ['{}', '[]', 'invalid', '{"nnCpuProfile":"true"}', '{"nnCpuProfile":1}']:
 env = {"sentinel":"unchanged"}
 assert m._CpuProfile.prepare(cfg, env) is None
 assert env == {"sentinel":"unchanged"}
with patch.object(pathlib.Path, 'is_file', return_value=True):
 env = {}
 p = m._CpuProfile.prepare('{"nnCpuProfile":true,"CPUPROFILE":"/arbitrary"}', env)
 assert p.directory.parent == m.LOG_DIR
 assert p.directory.stat().st_mode & 0o777 == 0o700
 assert env['CPUPROFILE'] == str(p.directory / 'capture')
 assert env['LD_PRELOAD'] == m._CpuProfile.LIB
 assert p.read()['state'] == 'idle'
 p.cleanup()
 assert not p.directory.exists()
`);
});

test("capture is bounded, retries do not toggle, and timer stops the original process", () => {
  python(`
p = m._CpuProfile()
signals = []
p._signal = lambda opening: signals.append(opening)
p._open = lambda: True
p.proc = type('Proc', (), {'poll': lambda self: None})()
for seconds in [0, 61, True, '30', 1.5]:
 try: p.start(seconds)
 except ValueError: pass
 else: raise AssertionError(seconds)
assert p.start(1)['state'] == 'active'
assert p.start(1)['state'] == 'active'
assert 'data' not in p.read()
p.path.write_bytes(b'profile')
time.sleep(1.1)
assert p.read()['state'] == 'complete'
assert base64.b64decode(p.read()['data']) == b'profile'
p.stop(); p.start(1)
assert signals == [True, False]
p.MAX_BYTES = 3
try: p.read()
except ValueError: pass
else: raise AssertionError('unbounded profile read')
p.cleanup()
`);
});

test("a chroot without procfs refuses profiling before touching the process environment", () => {
  python(`
env = {}
with patch.object(pathlib.Path, 'is_file', lambda self: str(self) == m._CpuProfile.LIB):
 try: m._CpuProfile.prepare('{"nnCpuProfile":true}', env)
 except ValueError as e: assert 'procfs' in str(e)
 else: raise AssertionError('missing process mappings accepted')
assert env == {}
assert list(m.LOG_DIR.iterdir()) == []
`);
});

test("dead generations are never signalled and failed acknowledgements stay bounded", () => {
  python(`
p = m._CpuProfile()
p.proc = type('Proc', (), {'poll': lambda self: 0})()
try: p.start(1)
except RuntimeError: pass
else: raise AssertionError('dead process accepted')
assert p.state == 'uncertain'
time.sleep(1.1)
assert p.state == 'incomplete'
assert p.read()['error'] == 'Profile process has exited'
assert 'data' not in p.read()
p.cleanup()
`);
});

test("manager requires its control token for both profile routes", () => {
  python(`
from io import BytesIO
m.VMMGR_TOKEN = 'test-control'
for method in ['GET', 'POST']:
 h = object.__new__(m.Handler)
 h.path = '/vms/owned/cpu-profile'
 h.headers = {}
 h._json = lambda code, body: (code, body)
 h._cpu_profile_route = lambda *a: (_ for _ in ()).throw(AssertionError('unauthenticated route'))
 assert getattr(h, 'do_' + method)()[0] == 401
`);
});

test("supervisor gates profile reads and writes by authenticated owner", async () => {
  const source = fs.readFileSync(new URL("../supervisor.js", import.meta.url), "utf8");
  const start = source.indexOf('for (const method of ["get", "post"]) {');
  const code = source.slice(start, source.indexOf("// Owner restart:", start));
  assert.ok(start > 0);
  const routes = {};
  let calls = 0;
  const authed = () => {};
  const context = { PROVISION_BACKEND: "vm", authed,
    app: Object.fromEntries(["get", "post"].map(method => [method, (path, gate, handler) => {
      assert.equal(path, "/v1/deployments/:id/cpu-profile");
      assert.equal(gate, authed); routes[method] = handler;
    }])),
    deployments: new Map([["owned", { owner: "alice", _vmId: "vm-own" }]]),
    fail: (res, status) => { res.code = status; },
    vmReq: async (method, path, body) => {
      calls++; assert.equal(path, "/vms/vm-own/cpu-profile");
      assert.equal(method === "POST" ? body.action : body, method === "POST" ? "start" : null);
      return { status: 200, body: { state: "idle" } };
    },
  };
  vm.runInNewContext(code, context);
  for (const method of ["get", "post"]) {
    for (const [id, address] of [["missing", "alice"], ["owned", "bob"]]) {
      const res = {};
      await routes[method]({ params: { id }, address }, res);
      assert.equal(res.code, 404); assert.equal(calls, 0);
    }
  }
  for (const method of ["get", "post"]) {
    const res = { set(k, v) { assert.equal(v, "no-store"); },
      status(s) { this.code = s; return this; }, json(b) { this.body = b; } };
    await routes[method]({ params: { id: "owned" }, address: "alice", body: { action: "start" } }, res);
    assert.equal(res.code, 200);
  }
  assert.equal(calls, 2);
});

const nativeLib = ["/usr/lib/libprofiler.so.0", "/usr/lib/x86_64-linux-gnu/libprofiler.so.0"].find(fs.existsSync);
test("real gperftools signal capture produces a readable file and leaves process alive",
  { skip: !nativeLib }, () => {
  python(`
m._CpuProfile.LIB = ${JSON.stringify(nativeLib)}
env = dict(os.environ)
p = m._CpuProfile.prepare('{"nnCpuProfile":true}', env)
proc = subprocess.Popen(['sleep', '10'], env=env, stderr=subprocess.DEVNULL)
p.proc = proc
try:
 time.sleep(0.15)
 assert p.start(1)['state'] == 'active'
 time.sleep(1.1)
 result = p.read()
 assert result['state'] == 'complete', result
 assert result['bytes'] >= 64
 assert proc.poll() is None
finally:
 p.cleanup()
 proc.terminate(); proc.wait()
`);
});
