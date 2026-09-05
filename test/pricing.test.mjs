// Console share math vs the runners' claim gate. The runners are authoritative:
// they divide an app's exact catalog specs by their PROBED hardware and refuse
// any deployment below the result (supervisor.js minSharesOf/gpuShareOf/
// cpuShareOf). A created deployment's shares are immutable and its funding is
// non-custodial accounting with no withdraw — so a console floor even 1% below
// the runners' minimum sells a deployment that is claimable by NOBODY, forever
// (2026-07-14, 0xf3d976a0…: the old hardcoded "141 GB" card vs the H200's
// probed 140.4 GiB made the console sell 91% of a card whose runner wanted 92%).
// These tests pin the two invariants that make that impossible:
//   1. the console divides by ADOPTED live hardware, exactly like the runner;
//   2. wherever the two can still diverge, the console lands ABOVE, never below.
// NOTE: tests in this file share pricing.js's adopted-spec module state and
// run in order — the fallback assertions come first, adoption after.

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { minPctsOf, adoptServerSpec, serverSpec, shareRates, enclaveSpecOf, enclavePriceOf, pickEnclaveFor, rankEnclavesFor, leaseHostOf,
  moveTargetsFor, moveBlockReason, wantedGpuPct, startSharesFor, gpuUpgradeForMove, gpuDowngradeForMove, fleetPrice, adoptFleetPrice, FALLBACK_CPU_NODE_RATE,
  hostChargeWaived, freeEnclavesFor, liftSharesForLedger, sharesLegalOn, SPLIT_SHARES_REV,
  enclaveClassOf, shieldedPoolOf, teeCpuOf } from "../site/js/core/pricing.js";

// Reference copy of the RUNNER's minimum-share math (supervisor.js: pctCeil,
// gpuShareOf, cpuShareOf, minSharesOf with MIN_COMPUTE_PCT=1). Keep in sync.
// pc is NOT clamped at 100: a ratio above one card is the honest measure of an
// app too big for the hardware, and clamping it made a box report "needs 1
// card" when no share it could sell was enough.
function runnerMins(v, hw, volGb = 0) {
  const pc = (x) => Math.max(1, Math.ceil(x * 100 - 1e-9));
  const cpuOf = (mb) => pc(Math.max(mb / (hw.nodeRamGb * 1024), (v.cpuGflops || 0) / hw.nodeGflops));
  const cpu = (v.memMb || v.cpuGflops) ? cpuOf(v.memMb || 0) : 0;
  // each axis floors on its OWN hardware — no cross-lift since ledger rev 13.
  // Volumes correct a declared card figure; they never create one.
  const vramGb = (v.vramMb || 0) > 0 ? Math.max(v.vramMb / 1024, volGb) : 0;
  const need = (vramGb > 0 || v.gpuGflops)
    ? pc(Math.max(vramGb / hw.cardVramGb, (v.gpuGflops || 0) / 1000 / hw.cardTflops)) : 0;
  return { gpuPct: v.gpuOptional ? 0 : need, gpuNeedPct: need, cpuPct: cpu };
}

// image-generator 1.0.2 — the version that produced the stuck deployment
const IMAGE_GEN = { vramMb: 131072, gpuGflops: 50000, memMb: 5000, cpuGflops: 5 };
const H200 = { cardVramGb: 140.4, cardTflops: 989, nodeVcpus: 16, nodeRamGb: 64, nodeGflops: 1000 };

test("fallback floors already match the live H200 (the 0xf3d976a0 regression)", () => {
  const s = serverSpec();
  assert.equal(s.live, false, "these assertions must run before any adoption");
  assert.equal(s.cardVramGb, 140.4, "fallback card must be the PROBED GiB, not the 141 datasheet");
  const m = minPctsOf(IMAGE_GEN);
  // the old 141 constant said 91 — unclaimable. No volumes here, so the
  // on-cores floor equals the plain one and the need equals the floor.
  assert.deepEqual(m, { gpuPct: 92, gpuNeedPct: 92, cpuPct: 8 });
  assert.deepEqual(m, runnerMins(IMAGE_GEN, H200));
});

test("adopting a live /availability payload aligns console and runner exactly", () => {
  assert.equal(adoptServerSpec({ gpu: true, ...H200 }), true);
  assert.equal(serverSpec().live, true);
  assert.deepEqual(minPctsOf(IMAGE_GEN), runnerMins(IMAGE_GEN, H200));
});

test("boundary sweep: the console floor NEVER under-sells any runner minimum", () => {
  // cards around the real fleet plus awkward probe values; specs pinned to
  // every whole-percent boundary ±1 MB — exactly where ceil math can split
  for (const card of [79.6, 131.7, 138.25, 139.95, 140.4, 140.41, 141, 143.99]) {
    const hw = { ...H200, cardVramGb: card };
    adoptServerSpec(hw);
    for (let n = 1; n <= 100; n++) {
      const edge = (n / 100) * card * 1024;
      for (const vramMb of [Math.floor(edge) - 1, Math.floor(edge), Math.floor(edge) + 1]) {
        if (vramMb <= 0) continue;
        const v = { vramMb, gpuGflops: 0, memMb: 512, cpuGflops: 0 };
        const site = minPctsOf(v), runner = runnerMins(v, hw);
        assert.ok(site.gpuPct >= runner.gpuPct && site.cpuPct >= runner.cpuPct,
          `under-sell at card=${card} vramMb=${vramMb}: site ${site.gpuPct}/${site.cpuPct} < runner ${runner.gpuPct}/${runner.cpuPct}`);
        assert.equal(site.gpuPct, runner.gpuPct, `gpu floor drift at card=${card} vramMb=${vramMb}`);
      }
    }
  }
});

test("relay spec* fleet-minima outrank the best-box fields", () => {
  // a mixed fleet: capacity view shows the big card, sizing must use the small
  adoptServerSpec({ gpu: true, cardVramGb: 150, specCardVramGb: 140.4, cardTflops: 989, specCardTflops: 989,
                    nodeVcpus: 16, nodeRamGb: 64, specNodeRamGb: 64, nodeGflops: 1000, specNodeGflops: 1000 });
  assert.equal(serverSpec().cardVramGb, 140.4);
  assert.equal(minPctsOf(IMAGE_GEN).gpuPct, 92);   // 128 GiB / 150 would have said 88
});

test("a CPU-only fleet payload cannot zero the GPU axes", () => {
  assert.equal(adoptServerSpec({ gpu: false, cardVramGb: 0, cardTflops: 0, nodeVcpus: 8, nodeRamGb: 32, nodeGflops: 500 }), true);
  const s = serverSpec();
  assert.equal(s.cardVramGb, 140.4, "absent/zero card keeps the previous value (no divide-by-zero)");
  assert.equal(s.nodeRamGb, 32);
  assert.ok(Number.isFinite(minPctsOf(IMAGE_GEN).gpuPct));
});

test("shareRates reads the adopted hardware, not constants", () => {
  adoptServerSpec({ gpu: true, ...H200 });
  const r = shareRates(92, 8);
  assert.ok(Math.abs(r.vramGb - 0.92 * 140.4) < 1e-9);
  assert.ok(Math.abs(r.ramGb - 0.08 * 64) < 1e-9);
});

/* ---- per-enclave targeting (the quick-deploy modal's "deploys to X" pick) ---- */

const row = (name, a, extra) => ({ name, endpoint: "https://" + name + ".example", availability: a, ...(extra || {}) });
const GPU_BOX = { gpu: true, claimEnabled: true, ...H200, gpuShareFree: 0.4, cpuShareFree: 0.79 };
const CPU_BOX = { gpu: false, claimEnabled: true, nodeVcpus: 16, nodeRamGb: 64, nodeGflops: 1000, gpuShareFree: 0, cpuShareFree: 0.9 };
const MC = { vramMb: 0, gpuGflops: 0, memMb: 512, cpuGflops: 10 };   // minecraft-shaped CPU app

test("pickEnclaveFor: floors come from the TARGET box, not a fleet minimum", () => {
  // the 2026-07-25 regression, per-enclave: a tiny box in the fleet must not
  // resize a CPU app that lands on the big one — the pick names the big box
  // and sizes 512 MB against ITS 64 GB (1%), never against 3 GB (17%)
  const tiny = row("metal0", { gpu: false, claimEnabled: true, nodeVcpus: 4, nodeRamGb: 3, nodeGflops: 250, cpuShareFree: 1 });
  const t = pickEnclaveFor(MC, [row("kryptos", GPU_BOX), tiny, row("big", CPU_BOX)]);
  assert.equal(t.none, undefined);
  assert.equal(t.name, "big");            // CPU-only box preferred over GPU leftovers
  assert.equal(t.mins.cpuPct, 1);         // 512 MB of 64 GB, not of 3 GB
  assert.equal(t.queued, false);
});

test("pickEnclaveFor: mirrors claim routing (CPU-only first, GPU leftovers as fallback; GPU apps by free card)", () => {
  const cpuApp = pickEnclaveFor(MC, [row("kryptos", GPU_BOX)]);
  assert.equal(cpuApp.name, "kryptos");   // no CPU-only box: GPU leftovers serve it
  const gpuApp = pickEnclaveFor(IMAGE_GEN, [row("small", { ...GPU_BOX, gpuShareFree: 0.3 }), row("kryptos", { ...GPU_BOX, gpuShareFree: 0.95 }), row("big", CPU_BOX)]);
  assert.equal(gpuApp.name, "kryptos");   // most free card wins; CPU boxes never take GPU work
});

test("pickEnclaveFor: a full box queues, a too-small fleet refuses", () => {
  const full = pickEnclaveFor(MC, [row("big", { ...CPU_BOX, cpuShareFree: 0 })]);
  assert.equal(full.name, "big");
  assert.equal(full.queued, true);        // fits the box, waits for capacity — deploys queue on-chain
  const noGpu = pickEnclaveFor(IMAGE_GEN, [row("big", CPU_BOX)]);
  assert.ok(noGpu.none && /GPU/.test(noGpu.none));
  const tooSmall = pickEnclaveFor({ memMb: 128 * 1024, cpuGflops: 0 }, [row("big", CPU_BOX)]);
  assert.ok(tooSmall.none, "an app over every box's whole node must refuse, not queue at 100%");
});

test("pickEnclaveFor: only CLAIMING enclaves count (the relay's serving rule)", () => {
  // a tunnel box without claimEnabled (the metal demo enclave) is invisible;
  // one that SAYS it claims (a Phase C seller) is a real target
  const demo = row("metal0", { gpu: false, nodeVcpus: 4, nodeRamGb: 3, nodeGflops: 250, cpuShareFree: 1 }, { tunnel: true });
  assert.ok(pickEnclaveFor(MC, [demo]).none, "a non-claiming tunnel box serves nobody");
  const seller = row("seller0", { gpu: false, claimEnabled: true, nodeVcpus: 16, nodeRamGb: 64, nodeGflops: 1000, cpuShareFree: 0.5 }, { tunnel: true });
  assert.equal(pickEnclaveFor(MC, [demo, seller]).name, "seller0");
  const hosted = row("cpu1", { gpu: false, nodeVcpus: 16, nodeRamGb: 64, nodeGflops: 1000, cpuShareFree: 0.5 });
  assert.equal(pickEnclaveFor(MC, [hosted]).name, "cpu1", "hosted boxes predate the flag and are grandfathered");
});

test("rankEnclavesFor: the dropdown's list — every host, recommended first, full ones queued at the tail", () => {
  const tiny = row("tiny", { gpu: false, claimEnabled: true, nodeVcpus: 4, nodeRamGb: 3, nodeGflops: 250, cpuShareFree: 1 });
  const full = row("bigfull", { ...CPU_BOX, cpuShareFree: 0 });
  const ranked = rankEnclavesFor(MC, [row("kryptos", GPU_BOX), tiny, full, row("big", CPU_BOX)]);
  // cheapest floor first (big 1% cpu-only, kryptos 1% gpu-leftovers, tiny 17%), full boxes tail
  assert.deepEqual(ranked.map((c) => c.name), ["big", "kryptos", "tiny", "bigfull"]);
  assert.deepEqual(ranked.map((c) => c.queued), [false, false, false, true]);
  assert.equal(ranked[0].name, pickEnclaveFor(MC, [row("kryptos", GPU_BOX), tiny, full, row("big", CPU_BOX)]).name);
  assert.equal(ranked[2].mins.cpuPct, 17);   // the user MAY pick the tiny box — at its own (17%) floor, eyes open
  // a box the app can never fit is not an option at all
  assert.ok(!rankEnclavesFor({ memMb: 8 * 1024, cpuGflops: 0 }, [tiny]).length);
});

/* ---- model volumes are placement, not just config ------------------------ */
// A volume is ATTACHED to a box (Modelwrap on the hosted fleet, dm-verity
// images on a metal box) — it is never fetched on demand. So a deployment that
// names one can only run where it lives, and the target must say so BEFORE the
// signature: the runner's own claim gate refuses the record, and a hint sent to
// a box that declines leaves the deploy sitting in the open queue.
const vol = (...names) => ({ volumes: names.map((name) => ({ name })) });
const LLM = { vramMb: 0, gpuGflops: 0, memMb: 512, cpuGflops: 10, volumes: ["qwen3.5-122b-gguf-merged"] };

test("rankEnclavesFor: only boxes carrying the requested volume are targets", () => {
  const metal = row("metal0", { gpu: false, claimEnabled: true, nodeVcpus: 4, nodeRamGb: 28, nodeGflops: 250, cpuShareFree: 1,
    ...vol("qwen3.5-122b-gguf-merged", "qwen2.5-0.5b-gguf") }, { tunnel: true });
  const kryptos = row("kryptos", { ...GPU_BOX, ...vol("qwen2.5-0.5b-gguf") });
  const ranked = rankEnclavesFor(LLM, [kryptos, metal, row("big", CPU_BOX)]);
  assert.deepEqual(ranked.map((c) => c.name), ["metal0"],
    "the bigger, cheaper boxes cannot host it — they do not carry the volume");
  // and without the volume constraint the same fleet ranks the big boxes first
  assert.equal(rankEnclavesFor({ ...LLM, volumes: [] }, [kryptos, metal, row("big", CPU_BOX)])[0].name, "big");
});

test("pickEnclaveFor: names the missing volume instead of blaming the hardware", () => {
  const kryptos = row("kryptos", { ...GPU_BOX, ...vol("qwen2.5-0.5b-gguf") });
  const t = pickEnclaveFor(LLM, [kryptos, row("big", CPU_BOX)]);
  assert.ok(t.none && /qwen3\.5-122b-gguf-merged/.test(t.none), t.none);
  assert.ok(!/big enough/.test(t.none), "the fleet has the hardware; it lacks the weights");
});

test("pickEnclaveFor: when the volume's box can't run the app, blame the box that HAS it", () => {
  // the live case: a GPU app + a volume that only the CPU-only metal box
  // carries. "no live enclave's hardware is big enough" points the reader at
  // kryptos, which is big enough and simply hasn't got the model.
  const metal = row("metal0", { gpu: false, claimEnabled: true, nodeVcpus: 4, nodeRamGb: 28, nodeGflops: 250, cpuShareFree: 1,
    ...vol("qwen3.5-122b-gguf-merged") }, { tunnel: true });
  const t = pickEnclaveFor({ ...IMAGE_GEN, volumes: ["qwen3.5-122b-gguf-merged"] }, [row("kryptos", GPU_BOX), metal]);
  assert.ok(t.none && /metal0/.test(t.none) && /GPU/.test(t.none), t.none);
  // a CPU app too big for the carrier reads as a size problem ON THAT BOX
  const big = pickEnclaveFor({ memMb: 64 * 1024, cpuGflops: 0, volumes: ["qwen3.5-122b-gguf-merged"] }, [row("kryptos", GPU_BOX), metal]);
  assert.ok(big.none && /metal0/.test(big.none) && /big enough/.test(big.none), big.none);
});

test("pickEnclaveFor: volumes split across two boxes host nothing (a deployment mounts on ONE)", () => {
  const a = row("a", { ...CPU_BOX, ...vol("v1") }), b = row("b", { ...CPU_BOX, ...vol("v2") });
  const t = pickEnclaveFor({ ...MC, volumes: ["v1", "v2"] }, [a, b]);
  assert.ok(t.none && /ONE box/.test(t.none), t.none);
  assert.equal(pickEnclaveFor({ ...MC, volumes: ["v1"] }, [a, b]).name, "a");
});

test("rankEnclavesFor: a box that advertises no volume list is not a target for volume work", () => {
  // an enclave whose /availability carries no `volumes` key cannot tell us what
  // it has; treating silence as "carries everything" is how a claim hint goes
  // to a box that declines (seen for real when a stale wasm-manager pin made
  // /health drop to its unauthenticated subset)
  const silent = row("silent", CPU_BOX);
  assert.ok(!rankEnclavesFor(LLM, [silent]).length);
  assert.equal(rankEnclavesFor({ ...LLM, volumes: [] }, [silent]).length, 1, "no volumes asked for, no constraint");
});

/* ---- the lease holder's hardware (a version change / resize, My Apps) ---- */

const RUNNER = "0x" + "ef".repeat(32);
const NOW = 1785000000000;
const leased = { runner: RUNNER, leaseUntil: Math.floor(NOW / 1000) + 1800 };
const FLEET = [{ id: RUNNER, name: "kryptos", endpoint: "https://kryptos.example", availability: GPU_BOX },
               { id: "0x" + "11".repeat(32), name: "metal0",
                 availability: { gpu: false, claimEnabled: true, nodeVcpus: 4, nodeRamGb: 3, nodeGflops: 250, cpuShareFree: 1 } }];

test("leaseHostOf: a live lease sizes on ITS box, not the fleet aggregate", () => {
  // the aggregate is the fleet MINIMUM (metal0's 3 GB node): 512 MB reads as
  // 17% there and 1% on the box actually running it, which is the only box
  // whose claim gate the switch has to pass
  adoptServerSpec({ gpu: true, ...H200, specNodeRamGb: 3, specNodeGflops: 250, specNodeVcpus: 4 });
  assert.equal(minPctsOf(MC).cpuPct, 17, "aggregate mode still over-asks (unleased work may land anywhere)");
  const hw = leaseHostOf(leased, FLEET, NOW);
  assert.equal(hw.name, "kryptos");
  assert.equal(minPctsOf(MC, hw.spec).cpuPct, 1);
  adoptServerSpec({ gpu: true, ...H200, specNodeRamGb: 64, specNodeGflops: 1000, specNodeVcpus: 16 });
});

test("leaseHostOf: nothing is pinned without a live lease in the fleet view", () => {
  assert.equal(leaseHostOf({ ...leased, leaseUntil: Math.floor(NOW / 1000) - 1 }, FLEET, NOW), null, "expired lease names an EX-runner");
  assert.equal(leaseHostOf({ runner: "0x" + "0".repeat(64), leaseUntil: Math.floor(NOW / 1000) + 1800 }, FLEET, NOW), null);
  assert.equal(leaseHostOf(leased, [], NOW), null, "runner absent from the fleet view");
  assert.equal(leaseHostOf(leased, null, NOW), null, "no fleet view at all");
  assert.equal(leaseHostOf(null, FLEET, NOW), null);
  assert.equal(leaseHostOf({ ...leased, runner: "0x" + "ef".repeat(31) }, FLEET, NOW), null, "malformed runner");
});

test("leaseHostOf: the host's own axes, unknown ones on the safe constants", () => {
  const cpuOnly = [{ id: RUNNER, name: "seller0", availability: { gpu: false, claimEnabled: true, nodeVcpus: 8, nodeRamGb: 32, nodeGflops: 500 } }];
  const hw = leaseHostOf(leased, cpuOnly, NOW);
  assert.equal(hw.spec.nodeRamGb, 32);
  assert.equal(hw.spec.cardVramGb, 140.4, "a CPU-only host reports no card: the GPU axes keep the constants, never zero");
  assert.ok(Number.isFinite(minPctsOf(IMAGE_GEN, hw.spec).gpuPct));
});

test("enclaveSpecOf: per-axis fallback for old builds", () => {
  const s = enclaveSpecOf(row("x", { gpu: true, cardVramGb: 79.6 }));
  assert.equal(s.cardVramGb, 79.6);
  assert.equal(s.nodeRamGb, 64);          // omitted axes keep the safe constants
});

/* ---- price is per enclave (rev 8) -----------------------------------------
   Each enclave posts what its whole machine costs; a deployment pays that
   fraction of whichever one claims it. So "what does this cost" is answered by
   the CHEAPEST live enclave (the box a new deployment lands on), and "what
   would a resize cost" by the box already holding the lease. Getting this
   wrong doesn't just misprice a readout: the deploy form's default rate cap
   comes from it, and a cap below what any enclave charges is a deployment
   nobody can claim. */

test("adoptFleetPrice: the fleet's cheapest posted price, with the constants as the only fallback", () => {
  const before = fleetPrice();
  assert.equal(before.live, false, "untouched, the pre-fetch constants stand");
  assert.equal(before.node, FALLBACK_CPU_NODE_RATE);

  // the relay aggregate carries the minimum over claiming enclaves
  assert.equal(adoptFleetPrice({ cheapestCpuPricePerSec6: 556, cheapestGpuPricePerSec6: 1200 }), true);
  const p = fleetPrice();
  assert.equal(p.live, true);
  assert.equal(p.node, 0.000556);
  assert.equal(p.full, 0.0012);
  assert.equal(adoptFleetPrice({ cheapestCpuPricePerSec6: 556, cheapestGpuPricePerSec6: 1200 }), false, "no change, no re-render");

  // a single enclave's own ask works too (the console can point at one box)
  assert.equal(adoptFleetPrice({ askCpuPricePerSec6: 834, askGpuPricePerSec6: 1667 }), true);
  assert.equal(fleetPrice().node, 0.000834);
  // a fleet that posts nothing leaves the last known price alone
  assert.equal(adoptFleetPrice({ enclaves: 2 }), false);
  assert.equal(fleetPrice().node, 0.000834);
});

test("shareRates prices against the enclave you name, not a platform constant", () => {
  adoptFleetPrice({ cheapestCpuPricePerSec6: 834, cheapestGpuPricePerSec6: 1667 });
  const dear = { full: 0.003334, node: 0.001668 };
  assert.equal(shareRates(0, 100).rate, 0.000834, "no price named: the fleet's cheapest");
  assert.equal(shareRates(0, 100, undefined, dear).rate, 0.001668);
  assert.equal(shareRates(50, 10).rate, 0.5 * 0.0016670 + 0.1 * 0.000834);
});

test("enclavePriceOf: a row's own posted price, the fleet's when it posts none", () => {
  adoptFleetPrice({ cheapestCpuPricePerSec6: 834, cheapestGpuPricePerSec6: 1667 });
  const priced = enclavePriceOf(row("seller0", { gpu: true, claimEnabled: true, askCpuPricePerSec6: 1668, askGpuPricePerSec6: 3334 }));
  assert.equal(priced.node, 0.001668);
  assert.equal(priced.full, 0.003334);
  const silent = enclavePriceOf(row("old", { gpu: true, claimEnabled: true }));
  assert.deepEqual(silent, { full: 0.0016670, node: 0.000834, shielded: undefined });
});

// A shielded card is the seller's own hardware at a price only that seller sets,
// so unlike full/node it has NO fleet list price to fall back on. The absence has
// to stay undefined rather than borrow the GPU ask: a box with no shielded card
// that reported the fleet's card rate would be quoting for hardware it cannot
// serve, and a shielded box quoting the in-enclave GPU price would be selling an
// H200's rate for a masked 3070.
test("enclavePriceOf: the shielded ask is the seller's own, and never inherited", () => {
  adoptFleetPrice({ cheapestCpuPricePerSec6: 834, cheapestGpuPricePerSec6: 1667 });
  const posted = enclavePriceOf(row("metal0", {
    gpu: false, claimEnabled: true, askCpuPricePerSec6: 834, askShieldedPricePerSec6: 56 }));
  assert.equal(posted.shielded, 0.000056);
  assert.equal(posted.node, 0.000834);

  const noCard = enclavePriceOf(row("cpu-only", { gpu: false, claimEnabled: true }));
  assert.equal(noCard.shielded, undefined,
    "a box with no shielded card must post no shielded rate, not the fleet's");
});

/* ---- free self-hosting (ledger rev 12): who charges this wallet nothing ----
   The ledger waives an enclave's WHOLE charge when its declared payout wallet
   is the deployment's owner (EnclaveDeployments._hostRate). Every console money
   gate has to agree, because a free deployment's correct state is an EMPTY
   balance: price it at the box's ask and the console refuses its own owner a
   resize the chain would accept (2026-08-11, risc-box on kryptos). */
const OWNER = "0x" + "0b".repeat(20);

test("hostChargeWaived: the registry's declaration, never the box's own word for it", () => {
  assert.equal(hostChargeWaived(row("mine", { gpu: true }, { payoutWallet: OWNER }), OWNER), true);
  assert.equal(hostChargeWaived(row("mine", { gpu: true }, { payoutWallet: OWNER.toUpperCase().replace("0X", "0x") }), OWNER),
    true, "checksummed on the wire, lowercase from the ledger decoder");
  // the row's top-level field is the relay's projection of the on-chain
  // registry entry - the same place the ledger reads it. /availability's copy
  // is the ENCLAVE talking, so it can never buy a waiver the chain won't give
  assert.equal(hostChargeWaived(row("liar", { gpu: true, payoutWallet: OWNER }), OWNER), false);
  // and every way it must NOT hold: another seller's box, an undeclared one,
  // the zero wallet, a missing owner
  assert.equal(hostChargeWaived(row("theirs", { gpu: true }, { payoutWallet: "0x" + "99".repeat(20) }), OWNER), false);
  assert.equal(hostChargeWaived(row("silent", { gpu: true }), OWNER), false);
  assert.equal(hostChargeWaived(row("zero", { gpu: true }, { payoutWallet: "0x" + "0".repeat(40) }), "0x" + "0".repeat(40)), false);
  assert.equal(hostChargeWaived(row("mine", { gpu: true }, { payoutWallet: OWNER }), ""), false);
  assert.equal(hostChargeWaived(null, OWNER), false);
});

test("freeEnclavesFor: only boxes that are SERVING host you for free", () => {
  const rows = [row("mine", { gpu: true }, { payoutWallet: OWNER }),
                row("mine-but-down", { gpu: true }, { payoutWallet: OWNER, serving: false }),
                row("theirs", { gpu: true }, { payoutWallet: "0x" + "99".repeat(20) })];
  assert.deepEqual(freeEnclavesFor(OWNER, rows).map((e) => e.name), ["mine"]);
  assert.deepEqual(freeEnclavesFor("0x" + "99".repeat(20), rows).map((e) => e.name), ["theirs"]);
  assert.deepEqual(freeEnclavesFor(OWNER, null), []);
});

// SOURCE-PINNED: the change-version panel is a custom element with a live DOM
// and a wallet behind it, so pin the three lines that decide the money. The
// runner side is test/rate-cap.test.mjs, the ledger side selfHost.t.sol, the
// CLI's copy of this gate test/cli.test.mjs.
test("the console's version/resize panel prices a self-hosted deployment at zero", () => {
  const src = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)),
    "..", "site", "components", "deployments", "deployments.js"), "utf8");
  // the verdict: rev 12, the LEASE HOLDER's row (not the fleet's), the
  // deployment's OWNER (not the connected wallet - a vault-owned row is owned
  // by the vault)
  assert.match(src, /const freeHere = rev >= 12 && !!hw && hostChargeWaived\(hw\.row, d\.owner\);/);
  // and it has to reach the rate the funding + cap gates are computed from,
  // with the publisher fee still riding on top (never waived)
  assert.match(src, /const rateOf = t => \(freeHere \? 0n : rate6Of\([\s\S]{0,240}\)\) \+ snapFee;/);
  assert.match(src, /the remaining balance can’t fund even one second at the new rate/,
    "the gate itself must stay - it is right for every deployment that IS charged");
});

test("rankEnclavesFor puts the CHEAPEST box for this app first, not just the biggest", () => {
  adoptFleetPrice({ cheapestCpuPricePerSec6: 834, cheapestGpuPricePerSec6: 1667 });
  // both fit the app; the big box needs a smaller share but charges much more
  const big = row("dear-big", { gpu: false, claimEnabled: true, cpuShareFree: 1, nodeRamGb: 64, nodeVcpus: 16, nodeGflops: 1000,
                                askCpuPricePerSec6: 8340 });
  const small = row("cheap-small", { gpu: false, claimEnabled: true, cpuShareFree: 1, nodeRamGb: 8, nodeVcpus: 4, nodeGflops: 500,
                                     askCpuPricePerSec6: 400 });
  const ranked = rankEnclavesFor(MC, [big, small]);
  assert.deepEqual(ranked.map((r) => r.name), ["cheap-small", "dear-big"]);
  assert.ok(ranked[0].minRate < ranked[1].minRate);
  // and with equal prices the old rule stands: the box asking for less of itself
  const evenBig = row("big", { ...big.availability, askCpuPricePerSec6: 834 });
  const evenSmall = row("small", { ...small.availability, askCpuPricePerSec6: 834 });
  assert.deepEqual(rankEnclavesFor(MC, [evenSmall, evenBig]).map((r) => r.name), ["big", "small"]);
});

/* ---- moveTargetsFor: where a LIVE deployment may be re-claimed ------------
   A move is release-then-re-claim, so the destination must pass exactly the
   gates a fresh deploy passes. These pin the rule that the current host is
   never offered as a destination, and that a box which would refuse the record
   is never offered at all — a target the runner declines leaves the app dark
   in the open queue, which is worse than saying "nowhere to go". ---- */
const ID_A = "0x" + "a".repeat(64), ID_B = "0x" + "b".repeat(64);

test("moveTargetsFor: the box already holding the lease is never a destination", () => {
  const rows = [row("kryptos", GPU_BOX, { id: ID_A }), row("big", CPU_BOX, { id: ID_B })];
  assert.deepEqual(moveTargetsFor(MC, rows, ID_A).map(t => t.name), ["big"]);
  assert.deepEqual(moveTargetsFor(MC, rows, ID_B).map(t => t.name), ["kryptos"]);
});

test("moveTargetsFor: a GPU app on the only GPU box has nowhere to go", () => {
  const rows = [row("kryptos", GPU_BOX, { id: ID_A }), row("big", CPU_BOX, { id: ID_B })];
  assert.deepEqual(moveTargetsFor(IMAGE_GEN, rows, ID_A), []);
});

test("moveTargetsFor: a box missing the model volume is not a destination", () => {
  // the volume rule is per-BOX (attached, not fetched), so a mover must land
  // somewhere carrying every volume the deployment mounts
  const withVol = row("metal0", { ...CPU_BOX, volumes: [{ name: "fable-fusion-27b-mtp-gguf" }] }, { id: ID_B });
  const without = row("kryptos", GPU_BOX, { id: ID_A });
  const app = { ...MC, volumes: ["fable-fusion-27b-mtp-gguf"] };
  assert.deepEqual(moveTargetsFor(app, [without, withVol], ID_B).map(t => t.name), []);
  assert.deepEqual(moveTargetsFor(app, [without, withVol], ID_A).map(t => t.name), ["metal0"]);
});

test("moveTargetsFor: a full-but-fitting box stays offered, flagged queued", () => {
  // a deployment may legitimately wait for its destination to free up — the
  // caller labels it rather than hiding the only sane target
  const full = row("big", { ...CPU_BOX, cpuShareFree: 0 }, { id: ID_B });
  const t = moveTargetsFor(MC, [row("kryptos", GPU_BOX, { id: ID_A }), full], ID_A);
  assert.deepEqual(t.map(x => x.name), ["big"]);   // the source box is excluded, so this is the only one
  assert.equal(t[0].queued, true);
});

test("moveTargetsFor: a non-serving box is not a destination", () => {
  const dark = row("metal0", CPU_BOX, { id: ID_B, serving: false });
  assert.deepEqual(moveTargetsFor(MC, [row("kryptos", GPU_BOX, { id: ID_A }), dark], ID_A), []);
});

test("a GPU box serves CPU wasi-nn but is never the automatic choice for it", () => {
  // A 0-GPU tenant gets the ggml CPU backend on any node, GPU boxes included
  // (wasm_manager: nn is granted to every tenant; only the BUDGET follows the
  // card). The card is the scarce resource though, so cores-only model work
  // must rank behind every CPU-only box - demoted, not excluded, because a
  // deliberate move names its destination and must be honoured.
  const gpuBox = row("kryptos", { ...GPU_BOX, volumes: [{ name: "m" }] }, { id: ID_A });
  const cpuBox = row("metal0", { ...CPU_BOX, volumes: [{ name: "m" }] }, { id: ID_B });
  const cpuOnlyApp = { ...MC, volumes: ["m"] };
  const ranked = rankEnclavesFor(cpuOnlyApp, [gpuBox, cpuBox]);
  assert.deepEqual(ranked.map(t => t.name), ["metal0", "kryptos"], "CPU box first, GPU box still offered");
  assert.equal(ranked[0].cpuNn, false);
  assert.equal(ranked[1].cpuNn, true, "the GPU box is flagged as running this on cores");
  // the demotion beats price: a CHEAPER GPU box still ranks behind the CPU one
  const cheapGpu = row("kryptos", { ...GPU_BOX, volumes: [{ name: "m" }], askCpuPricePerSec6: 1 }, { id: ID_A });
  const dearCpu = row("metal0", { ...CPU_BOX, volumes: [{ name: "m" }], askCpuPricePerSec6: 9999 }, { id: ID_B });
  assert.equal(rankEnclavesFor(cpuOnlyApp, [cheapGpu, dearCpu])[0].name, "metal0");
  // a manual move off the CPU box can still choose the GPU box
  assert.deepEqual(moveTargetsFor(cpuOnlyApp, [gpuBox, cpuBox], ID_B).map(t => t.name), ["kryptos"]);
  // a deployment that DID buy GPU share is not demoted - the card is the point
  const gpuApp = { ...IMAGE_GEN, volumes: ["m"] };
  assert.equal(rankEnclavesFor(gpuApp, [gpuBox, cpuBox])[0].name, "kryptos");
  // and a volume-less CPU app is untouched by any of this
  assert.equal(rankEnclavesFor(MC, [gpuBox, cpuBox]).length, 2);
});

test("moveBlockReason names the constraint that actually bit", () => {
  const gpuBox = row("kryptos", { ...GPU_BOX, volumes: [{ name: "m" }] }, { id: ID_A });
  const cpuBox = row("metal0", { ...CPU_BOX, volumes: [{ name: "m" }] }, { id: ID_B });
  const other = moveBlockReason({ ...MC, volumes: ["nope"] }, [gpuBox, cpuBox], ID_B);
  assert.match(other, /carries nope/);
});

/* ---- "GPU preferred, not required" --------------------------------------
   A bought GPU share is normally a hard requirement: only GPU boxes may claim,
   and the deployment queues when every card is full. `gpu.optional` on the
   envelope softens THAT dial (never the app version's own VRAM spec — an
   envelope may waive the owner's choice, not the publisher's requirement), so
   a CPU-only box becomes a legal fallback and the ledger bills only the cpu
   half there. Placement has to invert accordingly: card first, cores second. */
test("soft-GPU work prefers a GPU box and keeps CPU boxes as fallback", () => {
  const gpuBox = row("kryptos", { ...GPU_BOX, volumes: [{ name: "m" }] }, { id: ID_A });
  const cpuBox = row("metal0", { ...CPU_BOX, volumes: [{ name: "m" }] }, { id: ID_B });
  const soft = { memMb: 512, cpuGflops: 10, volumes: ["m"], gpuMilli: 200, depGpuOptional: true };
  const ranked = rankEnclavesFor(soft, [cpuBox, gpuBox]);
  assert.deepEqual(ranked.map(t => t.name), ["kryptos", "metal0"], "card first, cores as fallback");
  assert.equal(ranked[0].cpuNn, false);
  assert.equal(ranked[1].cpuNn, true, "the CPU box is flagged as running it on cores");
  // the flag is inert without a bought slice — there is no card requirement to soften
  const noSlice = { memMb: 512, cpuGflops: 10, volumes: ["m"], gpuMilli: 0, depGpuOptional: true };
  assert.deepEqual(rankEnclavesFor(noSlice, [cpuBox, gpuBox]).map(t => t.name), ["metal0", "kryptos"],
    "with no GPU share it is ordinary CPU work: GPU boxes demoted again");
  // and it never overrides the APP's own hard GPU spec
  const hard = { ...IMAGE_GEN, volumes: ["m"], gpuMilli: 200, depGpuOptional: true };
  assert.deepEqual(rankEnclavesFor(hard, [cpuBox, gpuBox]).map(t => t.name), ["kryptos"],
    "a version that declares VRAM can never land on a CPU box, whatever the envelope says");
});

/* ---- the PUBLISHER's declaration: GPU specs as desired, not required ------
   A version config's `gpuOptional: true` says the app starts without a card
   and would use one if given it. The floor must then stop forcing a GPU dial —
   and the console's floor has to match the runner's minSharesOf exactly, since
   a console floor below the runner's sells a deployment nobody can claim. */
test("a publisher-optional GPU spec sets no dial floor but still recommends a slice", () => {
  adoptServerSpec({ gpu: true, ...H200 });
  const v = { vramMb: 20000, gpuGflops: 1000, memMb: 512, cpuGflops: 10 };
  assert.equal(minPctsOf(v).gpuPct, 14, "declared axes normally force a GPU dial");
  assert.equal(minPctsOf({ ...v, gpuOptional: true }).gpuPct, 0, "publisher-optional forces none");
  assert.equal(minPctsOf({ ...v, gpuOptional: true }).cpuPct, 1, "the CPU floor is untouched");
  // the axes still mean something: they size what to buy to actually get the card
  assert.equal(wantedGpuPct({ ...v, gpuOptional: true }), 14);
  assert.equal(wantedGpuPct(v), 0, "no recommendation when the card is required — the floor already says it");
});

/* The floor is what a deployment may not go BELOW; it was never the default
   PURCHASE. On a gpuOptional version the two differ by a whole card, and the
   deploy paths (quick-deploy's one click, the console's pre-filled dials) used
   to buy the floor — which minted a model app at 0% GPU, running on cores, with
   nothing on screen saying the card had been skipped. LIVE REGRESSION: llm-chat
   1.3.27 declares 34 GB VRAM / 240 TFLOPS gpuOptional and deployed at gpuMilli
   0 (0xd6041bab…, 2026-08-03). */
test("start shares buy a publisher-optional version's declared card, not its 0% floor", () => {
  adoptServerSpec({ gpu: true, ...H200 });
  const hard = { vramMb: 20000, gpuGflops: 1000, memMb: 512, cpuGflops: 10 };
  const soft = { ...hard, gpuOptional: true };
  assert.deepEqual(startSharesFor(soft), { gpuPct: 14, cpuPct: 1 },
    "the declared slice is bought even though the floor is 0");
  assert.deepEqual(startSharesFor(hard), minPctsOf(hard),
    "a REQUIRED card already starts at its floor — start shares must not move it");
  assert.deepEqual(startSharesFor({ memMb: 512, cpuGflops: 10 }), minPctsOf({ memMb: 512, cpuGflops: 10 }),
    "a CPU-only app is untouched: no declared axes, nothing to lift");
  // the floors themselves are NOT moved: dialling down to CPU-only stays legal
  assert.equal(minPctsOf(soft).gpuPct, 0, "start shares must not become a new floor");
});

/* Ledger rev 13 dropped the gpuMilli >= cpuMilli rule, so a start slice is
   whatever the version declared and no more. The lift survives ONLY as
   compatibility with a ledger that has not been redeployed: liftSharesForLedger
   is the identity on 13+, and the old rounding-up below that. This is the whole
   behavioural difference between the two ledgers, so pin both sides. */
test("a soft start slice buys the declared card, lifting only for a pre-13 ledger", () => {
  adoptServerSpec({ gpu: true, ...H200 });
  // a small card ask beside a big CPU ask: exactly what the old rule forbade
  const cpuHeavy = { vramMb: 1024, gpuGflops: 1, memMb: 32768, cpuGflops: 10, gpuOptional: true };
  const s = startSharesFor(cpuHeavy);
  assert.ok(s.gpuPct < s.cpuPct, "the declared slice is genuinely CPU-heavy, or this pins nothing");
  assert.deepEqual(liftSharesForLedger(s, 13), s, "rev 13 buys the small card slice as declared");
  assert.deepEqual(liftSharesForLedger(s, 12), { ...s, gpuPct: s.cpuPct },
    "a pre-13 ledger still rounds the card up to clear its create()");
  // and the predicate the refusing callers use agrees with the rewriting one
  assert.equal(sharesLegalOn(s.gpuPct, s.cpuPct, 13), true);
  assert.equal(sharesLegalOn(s.gpuPct, s.cpuPct, 12), false);
  assert.equal(sharesLegalOn(0, 90, 12), true, "CPU-only was never subject to the rule");
  assert.equal(sharesLegalOn(90, 5, 12), true, "GPU above CPU always cleared it");

  // the cap TRIMS a preference rather than making the app undeployable — the
  // card was never required, and create() reverts above the cap
  const soft = { vramMb: 20000, gpuGflops: 1000, memMb: 512, cpuGflops: 10, gpuOptional: true };
  assert.equal(startSharesFor(soft, undefined, 10).gpuPct, 10, "trimmed to the cap");
  assert.equal(startSharesFor(soft, undefined, 0).gpuPct, 14, "no cap known: the full declared slice");
});

test("a publisher-optional version can land on a CPU box, a required one cannot", () => {
  const gpuBox = row("kryptos", GPU_BOX, { id: ID_A });
  const cpuBox = row("metal0", CPU_BOX, { id: ID_B });
  const v = { vramMb: 20000, gpuGflops: 1000, memMb: 512, cpuGflops: 10 };
  assert.deepEqual(rankEnclavesFor(v, [gpuBox, cpuBox]).map(t => t.name), ["kryptos"],
    "a required card excludes every CPU box");
  const soft = rankEnclavesFor({ ...v, gpuOptional: true }, [gpuBox, cpuBox]);
  assert.deepEqual(soft.map(t => t.name), ["kryptos", "metal0"], "optional: card first, cores as fallback");
  assert.equal(soft[1].cpuNn, true);
});

/* ---- moving soft-GPU work onto a card should re-buy the slice ------------
   A deployment whose card is optional runs on CPU cores wherever it lands —
   including on a GPU box, because the manager grants the card only to tenants
   holding GPU share. Moving it to a GPU enclave without re-buying therefore
   delivers the slow thing on the fast machine, which is never what the move
   was for. */
test("gpuUpgradeForMove sizes the slice from the version's declared axes", () => {
  adoptServerSpec({ gpu: true, ...H200 });
  const gpuT = { row: { availability: { gpu: true } }, spec: H200 };
  const cpuT = { row: { availability: { gpu: false } }, spec: { ...H200, cardVramGb: 0, cardTflops: 0, nodeRamGb: 28 } };
  const v = { vramMb: 20000, gpuGflops: 1000, memMb: 512, cpuGflops: 10, gpuOptional: true };

  assert.deepEqual(gpuUpgradeForMove(v, gpuT, 0, 80), { gpuPct: 14, cpuPct: 8 });
  assert.equal(gpuUpgradeForMove(v, cpuT, 0, 80), null, "a CPU destination has no card to buy");
  assert.equal(gpuUpgradeForMove(v, gpuT, 200, 80), null, "already holds a slice");
  assert.equal(gpuUpgradeForMove({ memMb: 512, cpuGflops: 10, gpuOptional: true }, gpuT, 0, 80), null,
    "no declared GPU axes = nothing to size a slice from");
  assert.equal(gpuUpgradeForMove({ ...v, gpuOptional: false }, gpuT, 0, 80), null,
    "a hard-GPU version was never on a CPU box to move off");
  // Only the PUBLISHER's flag can size an upgrade: the slice comes from the
  // version's declared axes, and only that flag makes them a preference. The
  // deployment-level flag softens a slice already bought, so such a deployment
  // is never at 0% GPU in the first place.
  assert.equal(gpuUpgradeForMove({ ...v, gpuOptional: false, depGpuOptional: true }, gpuT, 0, 80), null);
  // the slice is the version's declared axes and nothing more; a pre-13 ledger
  // gets the old rounding-up put back by the CALLER, via liftSharesForLedger
  assert.deepEqual(gpuUpgradeForMove(v, gpuT, 0, 300), { gpuPct: 14, cpuPct: 30 });
  assert.deepEqual(liftSharesForLedger(gpuUpgradeForMove(v, gpuT, 0, 300), 12), { gpuPct: 30, cpuPct: 30 });
});

test("gpuDowngradeForMove gives the card back when the destination has none", () => {
  adoptServerSpec({ gpu: true, ...H200 });
  const gpuT = { row: { availability: { gpu: true } }, spec: H200 };
  const cpuT = { row: { availability: { gpu: false } }, spec: { ...H200, cardVramGb: 0, cardTflops: 0, nodeRamGb: 28 } };
  const v = { vramMb: 32768, gpuGflops: 240000, memMb: 2048, cpuGflops: 5, gpuOptional: true };

  assert.deepEqual(gpuDowngradeForMove(v, cpuT, 250, 80), { gpuPct: 0, cpuPct: 8 });
  assert.equal(gpuDowngradeForMove(v, gpuT, 250, 80), null, "a GPU destination keeps the slice");
  assert.equal(gpuDowngradeForMove(v, cpuT, 0, 80), null, "nothing bought, nothing to drop");
  // the two directions are mutually exclusive for a given destination
  assert.equal(gpuUpgradeForMove(v, cpuT, 0, 80), null);
  assert.ok(gpuUpgradeForMove(v, gpuT, 0, 80));
});

/* ---- where the card physically IS, which is a trust claim, not a label ----
   A shielded box reports `gpu: true` because its card is real and is sold as
   one. That made `gpu` stop meaning "inside the enclave", and for one commit the
   fleet row still read it that way and badged a card sitting on an untrusted
   host as TEE GPU -- telling a buyer their activations were covered by an
   attestation that does not cover them. Caught by a human looking at the page,
   which is not a control. These are. */
test("a shielded card is never badged as being inside the enclave", () => {
  const shielded = { availability: { gpu: true, shielded: {
    card: "NVIDIA GeForce RTX 3070", vramGb: 7.7, vramFreeGb: 6.9, vramBudgetGb: 6.5, gmacPerSec: 7536 } } };
  const c = enclaveClassOf(shielded);
  assert.equal(c.kind, "shielded-gpu");
  assert.equal(c.inTee, false, "a card on an untrusted host must never read as in-TEE");
  assert.equal(c.shielded, true);

  // and the real thing still does
  const inTee = enclaveClassOf({ availability: { gpu: true } });
  assert.equal(inTee.kind, "tee-gpu");
  assert.equal(inTee.inTee, true);

  assert.equal(enclaveClassOf({ availability: { gpu: false } }).kind, "cpu");
  assert.equal(enclaveClassOf({}).kind, "cpu");            // a row with no availability is not a GPU
});

/* The fleet row's "TEE CPU" pill is EVIDENCE, not flavor. For one commit it
   read "a box with no card is a TEE CPU", which is exactly backwards: a host
   anchored on some other root of trust (a phone, one day) has no card either
   and no CPU TEE, and would have worn the green pill for free. The pill has to
   tell us whether there is a TEE CPU on the system at all. */
test("TEE CPU is badged from evidence, never from the box having no card", () => {
  // the box's own attestation document named the technology
  let t = teeCpuOf({ availability: { gpu: false, teeCpu: "amd-sev-snp" } });
  assert.equal(t.real, true); assert.equal(t.source, "attestation"); assert.equal(t.label, "AMD SEV-SNP");
  t = teeCpuOf({ availability: { gpu: true, teeCpu: "intel-tdx" } });
  assert.equal(t.real, true); assert.equal(t.label, "Intel TDX");
  // the card does not enter into it: a shielded box on an SNP CPU is a TEE CPU
  t = teeCpuOf({ availability: { gpu: true, teeCpu: "amd-sev-snp", shielded: { vramGb: 8, vramFreeGb: 4 } } });
  assert.equal(t.real, true);
  // the relay verified a fresh SEV-SNP quote at attach: proof from its side
  t = teeCpuOf({ tunnel: true, mode: "snp", availability: { gpu: false } });
  assert.equal(t.real, true); assert.equal(t.source, "relay");
  // a token-attached tunnel proved nothing about its CPU
  t = teeCpuOf({ tunnel: true, mode: "avf", availability: { gpu: true } });
  assert.equal(t.real, true); assert.equal(t.technology, "android-avf-pvm"); assert.equal(t.source, "relay");
  t = teeCpuOf({ tunnel: true, mode: "", availability: { gpu: false } });
  assert.equal(t.real, false); assert.equal(t.known, false);
  // a metal dev box says so in its format: a known NO, not an unknown
  t = teeCpuOf({ availability: { gpu: false, teeCpu: "dev-unattested-metal-v1" } });
  assert.equal(t.real, false); assert.equal(t.known, true); assert.equal(t.technology, "dev-unattested-metal-v1");
  // a build that predates the field never said: unknown, and NOT green
  t = teeCpuOf({ availability: { gpu: false } });
  assert.equal(t.real, false); assert.equal(t.known, false);
  assert.equal(teeCpuOf({}).real, false);
  assert.equal(teeCpuOf({ availability: { gpu: false, teeCpu: null } }).known, false);
  // and having no card is not evidence of anything
  assert.equal(teeCpuOf({ availability: { gpu: false, nodeVcpus: 8 } }).real, false);
});

test("a shielded card is one pool, sized by what it can actually sell", () => {
  // The physical card is 7.7 GB; the worker's budget is 6.5. Quoting 7.7 would
  // advertise VRAM the untrusted host keeps for itself -- and quoting it BESIDE
  // the GPU pool's 6.5 is what drew two differently-sized GPU rows on a box with
  // one card in it.
  const p = shieldedPoolOf({ availability: { shielded: {
    vramGb: 7.7, vramFreeGb: 6.9, vramBudgetGb: 6.5 } } });
  assert.equal(p.total, 6.5, "the pool must be the sellable budget, not the physical card");
  assert.equal(p.freeGb, 6.5, "free is clamped to the budget, never the card's free VRAM");
  assert.ok(p.frac > 0 && p.frac <= 1);
  assert.equal(p.reservedGb, 0, "an older box reports no reservations: none, not NaN");

  // protocol 1.3: the box already nets vramFreeGb of what tenants reserved at
  // HELLO and says how much that was, clamped to the budget like everything else
  const r = shieldedPoolOf({ availability: { shielded: {
    vramGb: 7.7, vramFreeGb: 5.5, vramBudgetGb: 6.5, vramReservedGb: 1.0 } } });
  assert.equal(r.reservedGb, 1.0);
  assert.equal(r.freeGb, 5.5);
  assert.equal(shieldedPoolOf({ availability: { shielded: {
    vramGb: 7.7, vramFreeGb: 0, vramBudgetGb: 6.5, vramReservedGb: 99 } } }).reservedGb, 6.5);

  // no budget reported (older probe) -> fall back to the physical total
  assert.equal(shieldedPoolOf({ availability: { shielded: { vramGb: 8, vramFreeGb: 4 } } }).total, 8);
  assert.equal(shieldedPoolOf({ availability: {} }), null);
});

test("a shielded pool advertises what can be LEASED, not what is resident", () => {
  // The case that was wrong on metal0: an 85% share is leased, so nothing is
  // buyable, but the worker holds only a 0.5B model's encoded weights and the
  // card reads 6.28 of 6.5 GB physically free. The old reading published "96%
  // available" for a card the allocator would refuse to sell.
  const row = { availability: {
    gpuShareFree: 0,
    shielded: { vramGb: 7.65, vramFreeGb: 6.28, vramBudgetGb: 6.5 } } };
  const p = shieldedPoolOf(row);
  assert.equal(p.frac, 0, "a fully leased card must advertise nothing available");
  assert.equal(p.leasableGb, 0);
  // the physical truth is kept, it is just not the headline
  assert.equal(p.freeGb, 6.28);
  assert.ok(p.vramFrac > 0.96, "the physical reading stays available for display");

  // a partly leased card quotes the remainder, not the VRAM
  const half = shieldedPoolOf({ availability: {
    gpuShareFree: 0.4, shielded: { vramGb: 7.65, vramFreeGb: 6.4, vramBudgetGb: 6.5 } } });
  assert.equal(half.frac, 0.4);
  assert.ok(Math.abs(half.leasableGb - 2.6) < 1e-9);

  // 0 is a REAL value here: a truthiness check would swap a fully-leased card
  // back to its ~96% physical reading, which is the bug this test exists for.
  assert.notEqual(shieldedPoolOf(row).frac, shieldedPoolOf(row).vramFrac);

  // a row too old to report gpuShareFree keeps the previous behaviour
  const legacy = shieldedPoolOf({ availability: {
    shielded: { vramGb: 7.65, vramFreeGb: 3.25, vramBudgetGb: 6.5 } } });
  assert.equal(legacy.frac, 0.5, "no gpuShareFree -> fall back to the physical ratio");
});

// ---- sizing above one card, and the model volume as the real footprint -----
// The eyesoff-ai case, 2026-08-31. A 27B Q6 app sized on an H200 was dialled
// gpu 36% / cpu 12%; metal0 then joined the claiming fleet with a 6.5 GB
// shielded card, and the SAME fraction there means 2.3 GB. The old pctCeil
// clamped the requirement to "1 card", so metal0 read no obstacle, accepted
// work it could not do, and the tenant died at weight-load. The fix is that a
// requirement is allowed to exceed the hardware and SAY so.
const EYESOFF = { vramMb: 51200, gpuGflops: 320000, memMb: 4096, cpuGflops: 10, gpuOptional: true };
const METAL0  = { cardVramGb: 6.5, cardTflops: 42.7, nodeVcpus: 16, nodeRamGb: 64, nodeGflops: 1000 };
const QWEN38_GB = 25.94;      // the volume the app's config names

test("a requirement bigger than the card sizes ABOVE 100% instead of clamping", () => {
  // on the hardware it was sized for, the dial it actually bought
  assert.equal(minPctsOf(EYESOFF, H200).gpuNeedPct, 36);
  // on a 6.5 GB card the same app needs 7.7 whole cards, and says so
  const m = minPctsOf(EYESOFF, METAL0);
  assert.equal(m.gpuNeedPct, 770, "7.69 cards, rounded up to the percent grain");
  assert.deepEqual(m, runnerMins(EYESOFF, METAL0), "console and runner agree above 100 too");
  // the clamp used to make this indistinguishable from "one whole card", which
  // is why a deployment dialled at the full 1.0 sailed through the floor check
  assert.ok(m.gpuNeedPct > 100, "the box can state that no share it sells is enough");
});

test("gpuOptional waives the FLOOR without erasing the requirement", () => {
  const m = minPctsOf(EYESOFF, METAL0);
  assert.equal(m.gpuPct, 0, "publisher said cores are acceptable: no GPU dial is forced");
  assert.equal(m.gpuNeedPct, 770, "but what the card would have to be is still known");
  // the old code returned only the floor, so a box asked to serve this saw no
  // GPU requirement at all and could not tell it was falling back
  const hard = minPctsOf({ ...EYESOFF, gpuOptional: false }, METAL0);
  assert.equal(hard.gpuPct, 770, "without the flag the floor IS the requirement");
});

test("a volume never inflates the CPU floor - the node is charged, not the share", () => {
  // The tempting symmetry, and the one this must NOT have. On cores the weights
  // are mmap'd page cache the platform charges to the NODE: wasm_manager's
  // budget is node RAM "independent of share", and a 4% tenant may legitimately
  // map a 17 GB GGUF. Adding the volume here would refuse deployments the box
  // then serves perfectly well - the mistake that comment already records.
  const bare = minPctsOf(EYESOFF, METAL0);
  const withVol = minPctsOf(EYESOFF, METAL0, { volGb: QWEN38_GB });
  assert.equal(withVol.cpuPct, bare.cpuPct, "26 GB of weights move the CPU floor not at all");
  assert.equal(withVol.cpuPct, 7);
  assert.deepEqual(withVol, runnerMins(EYESOFF, METAL0, QWEN38_GB));
});

test("volumes CORRECT a declared card figure but never invent one", () => {
  // under-declared VRAM: the volume is bigger than the publisher's number
  const under = { vramMb: 8192, gpuGflops: 0, memMb: 4096, cpuGflops: 0 };
  assert.equal(minPctsOf(under, H200).gpuNeedPct, 6);                       // 8 GB / 140.4
  assert.equal(minPctsOf(under, H200, { volGb: QWEN38_GB }).gpuNeedPct, 19); // 25.94 GB / 140.4
  // over-declared: the declaration already covers the weights, so it stands
  assert.equal(minPctsOf(EYESOFF, H200, { volGb: QWEN38_GB }).gpuNeedPct, 36);
  // a publisher who declared NO card is asking for cores — a volume must not
  // turn a CPU-only LLM app into card-bound work
  const cpuOnly = { vramMb: 0, gpuGflops: 0, memMb: 4096, cpuGflops: 0 };
  const m = minPctsOf(cpuOnly, METAL0, { volGb: QWEN38_GB });
  assert.equal(m.gpuNeedPct, 0, "no declared card + a volume is still no card");
  assert.equal(m.gpuPct, 0);
  assert.equal(m.cpuPct, 7, "and the CPU floor stays the app's own declaration");
  assert.deepEqual(m, runnerMins(cpuOnly, METAL0, QWEN38_GB));
});

test("ranking: a card too small to hold the app sizes the box as CPU work", () => {
  // The divergence that would resell the same bug through the console. metal0
  // HAS a card, so "is this box GPU" said the weights go on it and quoted the
  // 7% node floor — while the runner, seeing 7.7x, falls back to cores and
  // demands 47%. A deployment created at 7% is then claimable by nobody.
  const v = { ...EYESOFF, volumes: ["qwen3.8-27b-mtp-gguf"] };
  const vols = [{ name: "qwen3.8-27b-mtp-gguf", bytes: QWEN38_GB * 1e9 }];
  const metal0 = row("metal0", { gpu: true, claimEnabled: true, ...METAL0,
    nodeVcpus: 16, nodeRamGb: 64, nodeGflops: 1000,
    gpuShareFree: 0.85, cpuShareFree: 0.62, volumes: vols });
  const kryptos = row("kryptos", { gpu: true, claimEnabled: true, ...H200,
    nodeVcpus: 16, nodeRamGb: 64, nodeGflops: 1000,
    gpuShareFree: 0.9, cpuShareFree: 0.8, volumes: vols });

  const [onMetal] = rankEnclavesFor(v, [metal0]);
  assert.equal(onMetal.mins.gpuNeedPct, 770, "7.7 cards on a 6.5 GB card");
  assert.equal(onMetal.cpuNn, true, "so the box is labelled CPU-only for this deployment");

  // the same app on a card that can actually hold it runs ON the card
  const [onH200] = rankEnclavesFor(v, [kryptos]);
  assert.equal(onH200.mins.gpuNeedPct, 36);
  assert.equal(onH200.cpuNn, false);

  // ...and the CPU floor is the SAME on both. Where the weights run changes the
  // label and the routing, never the share: they are charged to the node.
  assert.equal(onMetal.mins.cpuPct, onH200.mins.cpuPct);
  assert.equal(onMetal.mins.cpuPct, 7, "the app's own declaration, not the model's size");

  // and given both, the box that can actually run it on its card is preferred
  assert.equal(rankEnclavesFor(v, [metal0, kryptos])[0].name, "kryptos");
});

test("a free sibling GPU does not make a leased primary shielded card look available", () => {
  const primary = { id: 0, vramGb: 7.7, vramBudgetGb: 6.5, vramFreeGb: 6.4, gpuShareFree: 0 };
  const sibling = { id: 1, vramGb: 31.7, vramBudgetGb: 31, vramFreeGb: 31, gpuShareFree: 1 };
  const p = shieldedPoolOf({ availability: { shielded: primary, shieldedCards: [primary, sibling], gpuShareFree: 1 } });
  assert.equal(p.frac, 0); assert.equal(p.leasableGb, 0); assert.equal(p.freeGb, 6.4);
});
