import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { prepareShieldedShm } from '../metal/guest/shielded-shm.mjs';

function fixture(body) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'shielded-shm-'));
  const pciRoot = path.join(root, 'pci'); fs.mkdirSync(pciRoot);
  const mounts = [];
  const options = { pciRoot, wasmRoot: path.join(root, 'wasm'), mount: (a, b) => mounts.push([a, b]) };
  const device = (bdf, mib = 8, vendor = '0x1af4', flags = '0x200') => {
    const dir = path.join(pciRoot, bdf); fs.mkdirSync(dir);
    fs.writeFileSync(path.join(dir, 'vendor'), vendor);
    fs.writeFileSync(path.join(dir, 'device'), '0x1110');
    fs.writeFileSync(path.join(dir, 'resource'), `0x0 0x0 0x0\n0x0 0x0 0x0\n0x10000000 0x${(0x10000000 + mib * 1048576 - 1).toString(16)} ${flags}\n`);
    fs.writeFileSync(path.join(dir, 'resource2_wc'), '');
    return dir;
  };
  try { body({ device, options, mounts }); } finally { fs.rmSync(root, { recursive: true, force: true }); }
}

test('unconfigured shared memory never inspects devices or mounts anything', () => {
  assert.equal(prepareShieldedShm({}, { pciRoot: '/does-not-exist' }), null);
});

test('sparse worker IDs map distinct BARs in PCI order and stay inside the native chroot', () => fixture(({ device, options, mounts }) => {
  const second = device('0000:00:09.0');
  device('0000:00:04.0', 8, '0x1234'); // unrelated PCI device must not take an index
  const first = device('0000:00:07.0');
  const a = { id: 0, shmIndex: 0, shmMib: 8 };
  assert.deepEqual(prepareShieldedShm(a, options), { shmPath: '/dev/enclave-shielded-shm/card-0', shmBytes: 8388608 });
  assert.deepEqual(prepareShieldedShm({ id: 5, shmIndex: 1, shmMib: 8 }, options),
    { shmPath: '/dev/enclave-shielded-shm/card-5', shmBytes: 8388608 });
  assert.deepEqual(mounts.map(x => x[0]), [path.join(first, 'resource2_wc'), path.join(second, 'resource2_wc')]);
  assert.ok(mounts.every(x => x[1].startsWith(options.wasmRoot + '/dev/enclave-shielded-shm/')));
  prepareShieldedShm(a, options);
  assert.equal(mounts.length, 2, 'probe refresh must not stack more bind mounts');
  assert.throws(() => prepareShieldedShm({ id: 1, shmIndex: 0, shmMib: 8 }, options), /another card/);
  assert.throws(() => prepareShieldedShm({ ...a, shmIndex: 1 }, options), /changed/);
}));

test('mapping metadata, BAR size/type and mapping inode are bounded', () => fixture(({ device, options, mounts }) => {
  const dir = device('0000:00:07.0');
  const a = { id: 0, shmIndex: 0, shmMib: 8 };
  for (const bad of [{ id: -1 }, { id: 16 }, { shmIndex: 16 }, { shmIndex: undefined },
    { shmMib: 7 }, { shmMib: 9 }, { shmMib: 128 }, { shmMib: '8' }])
    assert.throws(() => prepareShieldedShm({ ...a, ...bad }, options), /invalid/);
  assert.throws(() => prepareShieldedShm({ ...a, shmIndex: 1 }, options), /absent/);
  assert.throws(() => prepareShieldedShm({ ...a, shmMib: 16 }, options), /size\/type/);
  const resource = path.join(dir, 'resource'), original = fs.readFileSync(resource, 'utf8');
  fs.writeFileSync(resource, original.replace('0x200', '0x100'));
  assert.throws(() => prepareShieldedShm(a, options), /size\/type/);
  fs.writeFileSync(resource, original);
  fs.unlinkSync(path.join(dir, 'resource2_wc'));
  fs.symlinkSync(resource, path.join(dir, 'resource2_wc'));
  assert.throws(() => prepareShieldedShm(a, options), /unavailable/);
  assert.equal(mounts.length, 0);
}));

test('a failed bind can be retried without reserving a card mapping', () => fixture(({ device, options, mounts }) => {
  device('0000:00:07.0');
  const a = { id: 0, shmIndex: 0, shmMib: 8 };
  assert.throws(() => prepareShieldedShm(a, { ...options, mount: () => { throw new Error('mount failed'); } }), /mount failed/);
  assert.equal(prepareShieldedShm(a, options).shmBytes, 8388608);
  assert.equal(mounts.length, 1);
}));
