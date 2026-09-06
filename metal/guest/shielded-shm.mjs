// Map only a validated ivshmem BAR into the native engine's chroot. These
// pages carry the same ciphertext as vsock, never private activations/pads.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const prepared = new Map();
const owners = new Map();

export function prepareShieldedShm(worker, {
  pciRoot = '/sys/bus/pci/devices',
  wasmRoot = '/opt/roots/wasm',
  mount = (source, target) => execFileSync('mount', ['--bind', source, target], { timeout: 5000, stdio: 'pipe' }),
} = {}) {
  if (worker.shmMib === undefined) return null;
  const { id, shmIndex: index, shmMib: mib } = worker;
  if (!Number.isInteger(id) || id < 0 || id >= 16 ||
      !Number.isInteger(index) || index < 0 || index >= 16 ||
      !Number.isInteger(mib) || mib < 8 || mib > 64 || (mib & (mib - 1)))
    throw new Error('invalid shared-memory card, index or size');
  const bytes = mib * 1048576;
  const key = `${wasmRoot}|${id}`;
  const cached = prepared.get(key);
  if (cached) {
    if (cached.index !== index || cached.bytes !== bytes) throw new Error('shared-memory mapping changed');
    return { shmPath: cached.shmPath, shmBytes: bytes };
  }
  const devices = fs.readdirSync(pciRoot).filter(bdf => {
    if (!/^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$/i.test(bdf)) return false;
    try {
      const read = name => fs.readFileSync(path.join(pciRoot, bdf, name), 'utf8').trim().toLowerCase();
      return read('vendor') === '0x1af4' && read('device') === '0x1110';
    } catch { return false; }
  }).sort();
  const bdf = devices[index];
  if (!bdf) throw new Error('configured ivshmem device is absent');
  const ownerKey = `${wasmRoot}|${bdf}`;
  if (owners.has(ownerKey) && owners.get(ownerKey) !== id)
    throw new Error('ivshmem device already belongs to another card');
  const directory = path.join(pciRoot, bdf);
  const fields = fs.readFileSync(path.join(directory, 'resource'), 'utf8').trim().split('\n')[2]?.trim().split(/\s+/);
  if (!fields || fields.length !== 3 || !fields.every(v => /^0x[0-9a-f]{1,16}$/i.test(v)))
    throw new Error('invalid ivshmem BAR2 range');
  const [start, end, flags] = fields.map(v => BigInt(v));
  if (end < start || end - start + 1n !== BigInt(bytes) || !(flags & 0x200n))
    throw new Error('ivshmem BAR2 size/type differs from its configured bound');
  const source = path.join(directory, 'resource2_wc');
  if (!fs.lstatSync(source).isFile()) throw new Error('ivshmem write-combining mapping is unavailable');
  const shmPath = `/dev/enclave-shielded-shm/card-${id}`;
  const target = path.join(wasmRoot, shmPath);
  fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
  const fd = fs.openSync(target, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_NOFOLLOW, 0o600);
  fs.closeSync(fd);
  mount(source, target);
  owners.set(ownerKey, id);
  prepared.set(key, { index, bytes, shmPath });
  return { shmPath, shmBytes: bytes };
}
