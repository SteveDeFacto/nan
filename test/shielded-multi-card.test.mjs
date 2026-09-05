import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
const root = path.resolve(import.meta.dirname, '..');
const good = (id, budget) => ({ id, name: `GPU ${id}`, vram_total_gb: budget + .5,
  vram_budget_gb: budget, vram_free_gb: budget, field_gmac_per_s: 800,
  exact: true, verified: true, lie_rejected: true, denylist_refused: true,
  endpoint: `10.0.2.2:${9500 + id}`, vsockPort: 9500 + id, card_tflops: 100 });
const cards = [good(0, 6.5), good(1, 31), good(2, 31)];
function run(actions, verdict = { cards }) {
  const dir = mkdtempSync(path.join(tmpdir(), 'shielded-multi-'));
  try {
    const file = path.join(dir, 'cards.json'); writeFileSync(file, JSON.stringify(verdict));
    const stdout = execFileSync(process.execPath, [path.join(root, 'supervisor.js')], {
      env: { ...process.env, SECRET: 'test', ADDRESS_BOOK_ADDRESS: '', REGISTRY_ENABLED: '', CLAIM_ENABLED: '',
        SHIELDED_SELFTEST: '', SHIELDED_VERDICT: file, SHIELDED_POOL_SELFTEST: JSON.stringify(actions) }, encoding: 'utf8' });
    return JSON.parse(stdout.trim().split('\n').at(-1));
  } finally { rmSync(dir, { recursive: true, force: true }); }
}
test('three full-card leases get distinct workers and each actual VRAM budget', () => {
  const r = run([
    { alloc: { name: 'a', gpu: 1, cpu: .1 } }, { launch: 'a' },
    { alloc: { name: 'b', gpu: 1, cpu: .1 } }, { launch: 'b' },
    { alloc: { name: 'c', gpu: 1, cpu: .1 } }, { launch: 'c' },
    { alloc: { name: 'd', gpu: .1, cpu: .1 } },
    { release: 'b' }, { reconcile: true }, { alloc: { name: 'e', gpu: 1, cpu: .1 } }, { launch: 'e' },
  ]);
  assert.deepEqual([r[0].handle.cardId, r[2].handle.cardId, r[4].handle.cardId], [0, 1, 2]);
  assert.deepEqual([r[1].route.vramGb, r[3].route.vramGb, r[5].route.vramGb], [6.5, 31, 31]);
  assert.deepEqual([r[1].route.endpoint, r[3].route.endpoint, r[5].route.endpoint], ['10.0.2.2:9500','10.0.2.2:9501','10.0.2.2:9502']);
  assert.equal(r[6].handle, null); assert.equal(r[6].free, 0);
  assert.equal(r[8].cards[1].free, 31); assert.equal(r[9].handle.cardId, 1);
  assert.equal(r[10].route.vsockPort, 9501);
});
test('a failed card withdraws only its capacity and never reroutes its live lease', () => {
  const r = run([
    { alloc: { name: 'a', gpu: 1, cpu: .1 } },
    { alloc: { name: 'b', gpu: 1, cpu: .1 } },
    { verdict: { cards: [cards[0], { ...cards[1], exact: false }, cards[2]] } },
    { launch: 'b' }, { alloc: { name: 'c', gpu: 1, cpu: .1 } },
    { release: 'b' }, { reconcile: true },
    { verdict: { cards } }, { alloc: { name: 'd', gpu: 1, cpu: .1 } },
  ]);
  assert.match(r[3].error, /unavailable/); assert.equal(r[4].handle.cardId, 2);
  assert.equal(r[6].free, 0); assert.equal(r[6].cards[1].available, false);
  assert.equal(r[8].handle.cardId, 1);
});
test('device pressure on one card does not hide a free sibling', () => {
  const r = run([{ alloc: { name: 'a', gpu: 1, cpu: .1 } }, { launch: 'a' }],
    { cards: [{ ...cards[0], vram_free_gb: 0 }, { ...cards[1], vram_free_gb: 10 }, cards[2]] });
  assert.equal(r[0].handle.cardId, 2); assert.equal(r[1].route.vramGb, 31);
});
test('duplicate endpoint or UUID cannot count the same physical card twice', () => {
  for (const second of [ { ...cards[1], endpoint: cards[0].endpoint },
      { ...cards[1], deviceUuid: 'same-gpu' } ]) {
    const r = run([{ alloc: { name: 'a', gpu: 1, cpu: .1 } }, { alloc: { name: 'b', gpu: 1, cpu: .1 } }],
      { cards: [{ ...cards[0], deviceUuid: 'same-gpu' }, second] });
    assert.equal(r[1].handle, null);
  }
});
test('shared CPU admission still gates otherwise-free separate GPUs', () => {
  const r = run([{ alloc: { name: 'a', gpu: 1, cpu: .7 } }, { alloc: { name: 'b', gpu: 1, cpu: .4 } }]);
  assert.equal(r[1].handle, null); assert.ok(r[1].free > .9);
});
const source = readFileSync(path.join(root, 'metal/enclave-metal.mjs'), 'utf8');
const validation = source.slice(source.indexOf('const shieldedWorkers ='), source.indexOf('runtimeCfg.shieldedWorkers = [];'));
function validate(cfg) { vm.runInNewContext(validation, { cfg }); }
test('launcher accepts explicit separate UUIDs and rejects duplicate GPU/ports', () => {
  const a = { port: 9500, device: 'GPU-75f32211-2a00-2fb0-703f-99a11bbe5977' };
  const b = { port: 9501, device: 'GPU-1397d8cd-27ae-e1a6-a7ed-e485e7ca002c' };
  validate({ shieldedWorkers: [a, b] }); validate({ shieldedWorker: { port: 9500 } });
  assert.throws(() => validate({ shieldedWorkers: [a, { ...b, device: a.device }] }), /UUID/);
  assert.throws(() => validate({ shieldedWorkers: [a, { ...b, port: 9500 }] }), /ports/);
  assert.throws(() => validate({ shieldedWorkers: [a, { ...b, vsockPort: 9500 }] }), /vsock/);
  assert.throws(() => validate({ shieldedWorkers: [a, { port: 9501 }] }), /UUID/);
});

test('guest probes publish independent card verdicts and withdraw only the failed worker', async () => {
  const { EventEmitter } = await import('node:events');
  const guest = readFileSync(path.join(root, 'metal/guest/gsup.mjs'), 'utf8');
  const block = guest.slice(guest.indexOf('const configuredShieldedWorkers ='), guest.indexOf('// --- keeping the advertised card HONEST'));
  const probes = [], refresh = new Map(), files = new Map();
  vm.runInNewContext(block, {
    fw: { shieldedWorkers: [{ port: 9500 }, { port: 9501 }, { port: 9502 }] },
    fs: { writeFileSync: (p, v) => files.set(p, v), renameSync: (a, b) => { files.set(b, files.get(a)); files.delete(a); },
      unlinkSync: p => files.delete(p), existsSync: () => false, readFileSync: () => '' },
    spawn: () => { const p = new EventEmitter(); p.stdout = new EventEmitter(); p.stderr = new EventEmitter(); probes.push(p); return p; },
    log: () => {}, usdHrToSec6: () => 0, setTimeout: () => ({ unref() {} }),
    startShieldedRefresh: (host, port, read, write, clear, reprobe) => refresh.set(port, { read, write, clear, reprobe }),
  });
  const complete = i => {
    probes[i].stdout.emit('data', JSON.stringify({ ok: true, card: cards[i], exact: true, verified: true, lie_rejected: true, denylist_refused: true }));
    probes[i].emit('exit', 0);
  };
  complete(2); complete(0); complete(1);
  const published = () => JSON.parse(files.get('/run/shielded-gpu.json')).cards;
  assert.deepEqual(published().map(c => c.id), [0, 1, 2]);
  refresh.get(9501).clear();
  assert.deepEqual(published().map(c => c.id), [0, 2]);
  refresh.get(9502).write({ ...refresh.get(9502).read(), vram_free_gb: 12 });
  assert.deepEqual(published().map(c => [c.id, c.vram_free_gb]), [[0, 6.5], [2, 12]]);
  refresh.get(9500).clear(); refresh.get(9502).clear();
  assert.equal(files.has('/run/shielded-gpu.json'), false);
});
