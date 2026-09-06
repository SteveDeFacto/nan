// Claim-sweep ordering (supervisor.js) — the pure half: which ledger records
// are THIS enclave's own unresumed leases (they must be considered before any
// new claim, and an unresumed one holds the sweep). Driven through the
// SWEEP_SELFTEST seam, same contract as REACH_SELFTEST/ACME_SELFTEST.
//
// Why this exists (2026-07-17, kryptos): an enclave restart orphaned a paying
// 49%-GPU tenant's live lease; the old sweep considered records in ledger
// order, admitted a FRESH 49% claim first, and the orphan's resume then found
// "no free capacity" — a dark app on a live, still-billing lease.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pexec = promisify(execFile);
const SUPERVISOR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "supervisor.js");

async function partition(c) {
  const { stdout } = await pexec(process.execPath, [SUPERVISOR], {
    env: { ...process.env, SECRET: "test-secret", SWEEP_SELFTEST: JSON.stringify(c),
           REACH_SELFTEST: "", ACME_SELFTEST: "", ADDRESS_BOOK_ADDRESS: "", REGISTRY_ENABLED: "",
           CLAIM_ENABLED: "", ACME_EAB_KID: "", ACME_EAB_HMAC: "", APP_CERT_DOMAIN: "", DNS_API: "" } });
  const lines = stdout.trim().split("\n").filter(Boolean);
  return JSON.parse(lines[lines.length - 1]);
}

const US = "0x" + "aa".repeat(32);
const THEM = "0x" + "bb".repeat(32);
const NONE = "0x" + "00".repeat(32);
const NOW = 1700000000000;
const LIVE = NOW / 1000 + 600;      // lease alive for 10 more minutes
const PAST = NOW / 1000 - 600;

test("own live unresumed leases split out first; everything else keeps ledger order", async () => {
  const r = await partition({
    enclaveId: US, nowMs: NOW, serving: ["d-served"],
    ledger: [
      { id: "d-new",      runner: NONE, leaseUntil: 0 },      // fresh queued work
      { id: "d-orphan",   runner: US,   leaseUntil: LIVE },   // OUR lease, no local record -> resume first
      { id: "d-served",   runner: US,   leaseUntil: LIVE },   // ours AND locally serving -> normal pass
      { id: "d-theirs",   runner: THEM, leaseUntil: LIVE },   // someone else's live lease
      { id: "d-expired",  runner: US,   leaseUntil: PAST },   // our lease lapsed -> plain claimable again
    ],
  });
  assert.deepEqual(r.own, ["d-orphan"]);
  assert.deepEqual(r.rest, ["d-new", "d-served", "d-theirs", "d-expired"]);
});

test("no own leases -> nothing to resume, nothing held", async () => {
  const r = await partition({
    enclaveId: US, nowMs: NOW, serving: [],
    ledger: [{ id: "a", runner: THEM, leaseUntil: LIVE }, { id: "b", runner: NONE, leaseUntil: 0 }],
  });
  assert.deepEqual(r.own, []);
  assert.deepEqual(r.rest, ["a", "b"]);
});

test("a deactivated orphan lease does not hold new claims after a restart", async () => {
  const r = await partition({
    enclaveId: US, nowMs: NOW, serving: [],
    ledger: [
      { id: "stopped", runner: US, leaseUntil: LIVE, active: false },
      { id: "new", runner: NONE, leaseUntil: 0, active: true },
      { id: "resume", runner: US, leaseUntil: LIVE, active: true },
    ],
  });
  assert.deepEqual(r.own, ["resume"], "only resumable leases hold the sweep");
  assert.deepEqual(r.rest, ["stopped", "new"], "considerClaim still refuses the stopped record");
});

test("unregistered enclave (null id) owns nothing", async () => {
  const r = await partition({
    enclaveId: null, nowMs: NOW, serving: [],
    ledger: [{ id: "a", runner: US, leaseUntil: LIVE }],
  });
  assert.deepEqual(r.own, []);
});
