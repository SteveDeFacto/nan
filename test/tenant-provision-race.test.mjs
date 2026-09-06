import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../supervisor.js', import.meta.url), 'utf8');
function body(name) {
  const found = source.match(new RegExp(`^(?:async )?function ${name}\\([\\s\\S]*?^}`, 'm'));
  assert.ok(found, `supervisor function ${name} exists`);
  return found[0];
}
function harness(spawnContainer) {
  const deployments = new Map();
  let freed = 0;
  const deps = { deployments, spawnContainer, launchSpec: async rec => ({ id: rec.id }),
    acmeReconcileSoon() {}, releaseGpu() { freed++; },
    claimSigner: () => ({ account: { address: 'operator' } }),
    CLAIM_TERMINAL: new Set(['failed', 'expired', 'terminated']),
    console: { error() {} } };
  const api = new Function('deps', `
    const {deployments,spawnContainer,launchSpec,acmeReconcileSoon,releaseGpu,
           claimSigner,CLAIM_TERMINAL,console}=deps;
    const tenantProvisions = new WeakMap();
    ${['provisionTenant', 'performTenantProvision', 'auditClaims', 'adopt'].map(body).join('\n')}
    return {provisionTenant,auditClaims,adopt,tenantProvisions};
  `)(deps);
  return { ...api, deployments, freed: () => freed };
}

test('audit skips an in-flight adoption and overlapping calls share one manager launch', { timeout: 2000 }, async () => {
  let finish, starts = 0;
  const blocked = new Promise(resolve => { finish = resolve; });
  const h = harness(async () => { starts++; await blocked; return { internalPort: 80, vmId: 'one', hostPort: 9000 }; });
  const rec = { id: 'test', status: 'claimed', _onchain: true };
  h.deployments.set(rec.id, rec);
  const first = h.provisionTenant(rec), second = h.provisionTenant(rec);
  assert.equal(first, second);
  assert.equal(h.tenantProvisions.has(rec), true);
  // The audit must finish while the launch is still held, rather than join
  // it and later repeat its failure/release handling.
  await h.auditClaims(new Map([['test', {}]]));
  await Promise.resolve();
  assert.equal(starts, 1);
  finish();
  assert.equal(await first, true);
  assert.equal(rec.status, 'running');
  assert.equal(rec._vmId, 'one');
  assert.equal(h.tenantProvisions.has(rec), false);
  assert.equal(h.freed(), 0);
});

test('a failed shared launch releases its allocation once and permits a later retry', async () => {
  let attempts = 0;
  const h = harness(async () => {
    if (++attempts === 1) throw new Error('manager unavailable');
    return { internalPort: 80 };
  });
  const rec = { id: 'test', status: 'claimed', _gpu: {} };
  const first = h.provisionTenant(rec), second = h.provisionTenant(rec);
  assert.equal(first, second);
  assert.equal(await first, false);
  assert.equal(h.freed(), 1);
  assert.equal(h.tenantProvisions.has(rec), false);
  rec._gpu = {};
  rec.status = 'claimed';
  assert.equal(await h.provisionTenant(rec), true);
  assert.equal(attempts, 2);
  assert.equal(h.freed(), 1);
});

test('a delayed second claim cannot replace an adopted or running record', async () => {
  const h = harness(async () => { throw new Error('must not spawn'); });
  for (const status of ['claimed', 'running']) {
    const rec = { id: 'test', status, _gpu: {} };
    h.deployments.set(rec.id, rec);
    await h.adopt({ id: rec.id });
    assert.equal(h.deployments.get(rec.id), rec);
    assert.equal(h.freed(), 0);
  }
});
