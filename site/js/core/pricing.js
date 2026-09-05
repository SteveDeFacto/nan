/* ============================================================
   Pricing + share math. The deploy page has exactly TWO dials:
   a GPU/VRAM share and a CPU/RAM share, 0-100% each. An app's
   exact specs (VRAM, compute, RAM in the catalog) only set the
   MINIMUM shares those dials allow: spec / server spec, the
   larger of the memory and compute axes per pool, rounded up to
   the whole percent. A GPU app's CPU minimum also lifts its GPU
   minimum (a GPU app's gpuShare must be >= its cpuShare).
   Rate = gpuShare x card rate + cpuShare x node rate.

   THE SERVER SPEC IS ADOPTED LIVE from /availability — the same
   numbers the runners divide by in their own claim gate — never
   trusted from constants. The runners are authoritative: a dial
   floor computed from a card that's even slightly bigger than
   the real one sells a share below every runner's minimum, and
   that deployment is claimable by NOBODY, forever, with its
   funding unrecoverable (2026-07-14, 0xf3d976a0…: "141 GB" here
   vs the H200's probed 140.4 GiB made the console sell 91% of a
   card whose runners demand 92%). The constants below are only
   the pre-fetch fallback, and each sits AT OR BELOW the smallest
   real hardware in the fleet: a wrong fallback must over-ask
   (costs the user pennies), never under-sell.
   ============================================================ */
/* PRICE IS PER ENCLAVE, adopted live like the hardware. Each enclave posts
   what its whole machine costs (askCpu/askGpuPricePerSec6 in /availability,
   its EnclaveRegistry entry on-chain) and a deployment pays that fraction of
   whichever one claims it — so the number to SHOW is the cheapest connected
   enclave's, and the number to CHARGE is the target box's. The constants below
   are only the pre-fetch fallback (the hosted fleet's long-standing prices);
   adoptFleetPrice replaces them with the live floor. */
export const FALLBACK_FULL_RATE = 0.0016667;      // whole card USDC/sec ($6.00/hr)
export const FALLBACK_CPU_NODE_RATE = 0.000834;   // whole node USDC/sec ($3.00/hr)
let PRICE = null;   // { full, node } USDC/sec, cheapest live enclave; null until adopted
// Current prices. `live` says whether a real /availability payload set them.
export function fleetPrice(){
  return { full: PRICE?.full ?? FALLBACK_FULL_RATE, node: PRICE?.node ?? FALLBACK_CPU_NODE_RATE, live: !!PRICE };
}
// Adopt the CHEAPEST posted price from an /availability payload: the relay
// aggregate carries cheapest*PricePerSec6 (min over claiming enclaves), a
// single enclave carries its own ask*. Absent on a fleet that predates posted
// prices — the fallbacks then stand. Returns true when a number changed.
export function adoptFleetPrice(a){
  if (!a || typeof a !== "object") return false;
  const per = (x) => { const v = Number(x); return Number.isFinite(v) && v > 0 ? v / 1e6 : 0; };
  const full = per(a.cheapestGpuPricePerSec6 ?? a.askGpuPricePerSec6);
  const node = per(a.cheapestCpuPricePerSec6 ?? a.askCpuPricePerSec6);
  if (!node) return false;                       // no live price: keep what we have
  const next = { full: full || PRICE?.full || FALLBACK_FULL_RATE, node };
  const changed = !PRICE || PRICE.full !== next.full || PRICE.node !== next.node;
  PRICE = next;
  return changed;
}
/* FREE SELF-HOSTING (ledger rev 12 + registry schema 4). A seller who runs a box
   declares their payout wallet on-chain — EnclaveRegistry.setPayoutWallet, sent
   BY that wallet, which is what makes the declaration unforgeable — and the
   ledger then charges that wallet's own deployments NOTHING to run there. The
   console has to know: quoting somebody $3.00/hr and asking them to fund it,
   when they will be charged nothing, is the exact confusion the feature exists
   to remove. Rows come from the relay's /enclaves table (each carries its
   declared wallet); a fleet on an older registry simply reports none, which
   reads as "everything is charged" — the safe direction.
   Only SERVING boxes count: one that cannot take work would not host it free
   either. */
export function freeEnclavesFor(address, rows){
  if (!Array.isArray(rows)) return [];
  return rows.filter((e) => e && e.serving !== false && hostChargeWaived(e, address));
}

/* Does THIS box charge THIS wallet nothing? The ledger's own test, mirrored
   exactly (EnclaveDeployments._hostRate: `if (e.payoutWallet == d.owner) return 0`).
   Read the row's TOP-LEVEL payoutWallet only — the relay projects that one out
   of the on-chain registry entry, which is the same place the ledger reads it.
   A box also repeats it in its own /availability, and that copy is the box
   TALKING: believing it would let an enclave quote itself as free to a wallet
   the chain will still charge. Two rules ride with this and callers must keep
   them: gate on ledger rev >= 12 (an older ledger never heard of the waiver),
   and compare against the deployment's OWNER, not the connected wallet (a
   vault-owned row is owned by the vault). The publisher fee is never waived —
   it is a third party's money.
   Getting this wrong is not cosmetic: a self-hosted deployment's correct,
   normal state is an EMPTY balance, so pricing it at the box's posted price
   makes every money gate read it as unfundable. */
export function hostChargeWaived(row, owner){
  const a = String(owner || "").toLowerCase();
  if (!/^0x[0-9a-f]{40}$/.test(a) || /^0x0+$/.test(a)) return false;
  return String((row && row.payoutWallet) || "").toLowerCase() === a;
}

// Back-compat aliases for call sites that read a plain number. These follow
// the adopted price, so a module that imported them once still shows the
// current one only if it re-reads via fleetPrice() — prefer that.
export const FULL_RATE = FALLBACK_FULL_RATE;
export const CPU_NODE_RATE = FALLBACK_CPU_NODE_RATE;
export const MIN_COMPUTE_PCT = 1;    // shares are dialed in whole percent (CUDA MPS grain); 1% floor, no fixed 1/7

const FALLBACK = {
  cardVramGb: 140.4,   // H200 as PROBED under CC (nvidia-smi 143771 MiB), not the 141 datasheet
  cardTflops: 989,     // GPU compute per card (H200 FP16 dense)
  nodeVcpus: 16,       // host vCPUs on the enclave
  nodeRamGb: 64,       // host RAM GB on the enclave
  nodeGflops: 1000,    // CPU compute per node in GFLOPS (~16 vCPU; a node is ~1/1000 of a card)
};
let LIVE = null;       // last adopted /availability hardware; null until the first fetch lands

// The hardware every share/minimum divides by right now. `live` says whether
// a real /availability payload has been adopted or the fallbacks still hold.
export function serverSpec(){ return { ...FALLBACK, ...(LIVE || {}), live: !!LIVE }; }

// Adopt the fleet's hardware from an /availability payload (a single enclave's
// or the relay aggregate — both carry the same field names). Prefers the
// relay's spec* fields when present: those are fleet-wide MINIMA, the only
// safe sizing base on a mixed fleet (the plain fields describe the best box,
// and a floor computed on the biggest card under-sells on every smaller one).
// Zero/absent axes (a CPU-only fleet reports no card) keep their previous
// values so GPU math never divides by zero. Returns true when a number
// changed — callers re-render dial floors on that signal.
export function adoptServerSpec(a){
  if (!a || typeof a !== "object") return false;
  const next = { ...FALLBACK, ...(LIVE || {}) };
  let changed = false;
  for (const [key, specKey] of [
    ["cardVramGb", "specCardVramGb"], ["cardTflops", "specCardTflops"],
    ["nodeVcpus", "specNodeVcpus"], ["nodeRamGb", "specNodeRamGb"], ["nodeGflops", "specNodeGflops"],
  ]){
    const v = Number(a[specKey] != null ? a[specKey] : a[key]);
    if (Number.isFinite(v) && v > 0 && v !== next[key]){ next[key] = v; changed = true; }
  }
  if (changed || !LIVE){ LIVE = next; return true; }
  return false;
}

// NOT clamped to 100 — the runner's pctCeil isn't either, and this floor may
// never sit BELOW the runner's or the deployment it sizes is unclaimable. A
// ratio above 100 is the honest answer to "how much of this box would it take":
// 769 means the app needs 7.69 of this card, which no share can buy.
export const pctCeil = (x) => Math.max(MIN_COMPUTE_PCT, Math.ceil(x * 100 - 1e-9));

// GB of weights the named volumes carry on this enclave, from its own
// advertised rows. Volumes it doesn't have count 0 — hasVolumes already
// excludes that box, and inventing a size for a volume nobody published would
// size the dial off a number nothing verified.
export function volGbOf(a, want){
  if (!want || !want.length) return 0;
  const bytes = new Map(((a && a.volumes) || [])
    .filter((v) => v && v.name).map((v) => [String(v.name), Number(v.bytes) || 0]));
  return want.reduce((t, n) => t + (bytes.get(n) || 0), 0) / 1e9;
}
// v: a catalog version's exact specs (zeros = no minimum). `spec` picks the
// hardware to divide by: omitted = the adopted fleet spec (aggregate mode),
// or a specific enclave's hardware from enclaveSpecOf (target mode — the
// quick-deploy modal sizes against the box the deployment would land on).
// `opts.volGb` is the size of the model volumes this deployment will mount on
// that box (volGbOf). It corrects the GPU floor only, exactly as the runner
// does: on a card the weights are resident in the tenant's own slice, so a
// declared vramMb under the volume it names is an under-declaration. On cores
// they are node-charged page cache, never billed to the share, so the CPU
// floor is untouched.
export function minPctsOf(v, spec, opts){
  const s = spec || serverSpec();
  const volGb = Math.max(0, Number(opts && opts.volGb) || 0);
  const vramMb = Number(v && v.vramMb || 0), gpuGf = Number(v && v.gpuGflops || 0);
  const memMb = Number(v && v.memMb || 0), cpuGf = Number(v && v.cpuGflops || 0);
  const cpuOf = (mb) => pctCeil(Math.max(mb / (s.nodeRamGb * 1024), cpuGf / s.nodeGflops));
  const cpu = (memMb > 0 || cpuGf > 0) ? cpuOf(memMb) : 1;
  // A version whose publisher marked the card OPTIONAL sets no GPU floor: the
  // app starts without one. Mirrors the runner's minSharesOf exactly — this
  // floor may never sit below the runner's or the deployment is unclaimable.
  // Each axis floors on its own hardware and nothing else: before rev 13 the
  // GPU floor was also lifted to the CPU floor, which only ever encoded the
  // ledger's gpuMilli >= cpuMilli rule, never a real requirement of the app.
  // The volumes correct a declared card figure, never create one — a publisher
  // who declared no VRAM is asking for cores.
  const vramGb = vramMb > 0 ? Math.max(vramMb / 1024, volGb) : 0;
  const need = (vramGb > 0 || gpuGf > 0)
    ? pctCeil(Math.max(vramGb / s.cardVramGb, gpuGf / 1000 / s.cardTflops)) : 0;
  return {
    gpuPct: (v && v.gpuOptional) ? 0 : need,   // the enforceable floor (optional = none)
    gpuNeedPct: need,                          // the TRUE ask on this card; may exceed 100
    cpuPct: cpu,
  };
}
/* THE LEDGER REV THAT FREED THE TWO DIALS. Revs <= 12 reverted create() and
   setShares() whenever a non-zero gpuMilli sat below cpuMilli, so every client
   that builds one of those transactions had to lift the GPU dial to match — a
   silent up-sell to whole percents of card the deployer never asked for.

   Rev 13 drops the rule. The lift therefore survives ONLY as compatibility
   with a ledger that has not been redeployed yet, which is why it takes a live
   rev instead of being deleted outright: on rev 13 it is the identity, and the
   dials mean exactly what they say. Callers that hold a rev pass it here;
   `sharesLegalOn` is the matching pre-flight check for the ones that refuse
   rather than rewrite. */
export const SPLIT_SHARES_REV = 13;
export const liftSharesForLedger = (s, rev) =>
  Number(rev) >= SPLIT_SHARES_REV || !(s.gpuPct > 0) || s.gpuPct >= s.cpuPct
    ? s : { ...s, gpuPct: s.cpuPct };
export const sharesLegalOn = (gpuPct, cpuPct, rev) =>
  Number(rev) >= SPLIT_SHARES_REV || !(gpuPct > 0) || gpuPct >= cpuPct;
// What a deployer SHOULD buy to get the card on a gpuOptional version: the
// slice its declared axes ask for. Not a floor — a recommendation the deploy
// dials start at, so "GPU preferred" doesn't quietly become "GPU never".
export function wantedGpuPct(v, spec){
  const s = spec || serverSpec();
  const vramMb = Number(v && v.vramMb || 0), gpuGf = Number(v && v.gpuGflops || 0);
  if (!(v && v.gpuOptional) || (vramMb <= 0 && gpuGf <= 0)) return 0;
  return pctCeil(Math.max(vramMb / 1024 / s.cardVramGb, gpuGf / 1000 / s.cardTflops));
}
/* The shares a deploy should START at (quick-deploy buys these outright; the
   console's dials open here). Normally the app's floors — but a floor is what a
   deployment may not go BELOW, and it was never meant to double as the default
   purchase. On a gpuOptional version the two differ by an entire card: the GPU
   floor is 0, so buying the floor deployed a model app onto CPU cores with
   nothing on screen saying the card had been skipped. Start at the slice the
   version declares instead. The floors are unchanged and still gate the dials,
   so dialling back down to CPU-only stays available — deliberately, not by
   default. `cap` (the on-chain per-deployment GPU cap, in percent) trims a soft
   slice rather than making the app undeployable: the card is a preference, and
   create() would refuse anything above the cap.

   The slice is what the version asks for and nothing more: a CPU-heavy soft-GPU
   app starts at its small declared card slice beside its large node slice. On a
   pre-13 ledger the caller runs the result through liftSharesForLedger, which
   is the only thing that will still round the card up to clear create(). */
export function startSharesFor(v, spec, cap){
  const mins = minPctsOf(v, spec);
  let want = wantedGpuPct(v, spec);
  if (Number(cap) > 0) want = Math.min(want, Math.floor(Number(cap)));
  return want > mins.gpuPct ? { gpuPct: want, cpuPct: mins.cpuPct } : mins;
}
// What the two dials buy on this server spec, and cost per second. `price`
// pins a specific enclave's posted rates ({full, node} USDC/sec, e.g. from
// enclavePriceOf(row)); omitted = the cheapest live enclave, which is what a
// new deployment lands on.
export function shareRates(gpuPct, cpuPct, spec, price){
  const s = spec || serverSpec();
  const p = price || fleetPrice();
  const g = Math.min(100, Math.max(0, Math.round(gpuPct)));
  const c = Math.min(100, Math.max(MIN_COMPUTE_PCT, Math.round(cpuPct)));
  return { rate: (g / 100) * p.full + (c / 100) * p.node, gpuPct: g, cpuPct: c,
           vramGb: (g / 100) * s.cardVramGb, tflops: (g / 100) * s.cardTflops,
           ramGb: (c / 100) * s.nodeRamGb, vcpus: (c / 100) * s.nodeVcpus, gflops: (c / 100) * s.nodeGflops };
}

/* ---- per-enclave targeting (the relay's /enclaves rows) ------------------ */

// One enclave row's POSTED price ({full, node} USDC/sec): what it charges for
// a whole card / whole node. A row that posts nothing falls back to the fleet
// price, so callers always have a number to show.
export function enclavePriceOf(row){
  const a = (row && row.availability) || {};
  const f = fleetPrice();
  const per = (x, fb) => { const v = Number(x); return Number.isFinite(v) && v > 0 ? v / 1e6 : fb; };
  // `shielded` has no fleet fallback on purpose: a shielded card is a seller's
  // own hardware at a price only that seller sets, so there is no list price to
  // fall back to. Absent ask -> undefined -> the pool renders without a rate,
  // which is the honest reading of "this box has not posted one".
  return { full: per(a.askGpuPricePerSec6, f.full), node: per(a.askCpuPricePerSec6, f.node),
           shielded: per(a.askShieldedPricePerSec6, undefined) };
}

// WHAT KIND OF BOX THIS IS, for the badge. Pure, and in this module rather than
// in the fleet row, because the answer is a trust claim and a trust claim needs a
// test more than a renderer needs a helper.
//
// The subtlety that actually bit: a shielded box reports `gpu: true`, because its
// card is real and is sold as one. So `gpu` alone does NOT mean "inside the
// enclave" any more, and reading it that way badges a card sitting on an
// untrusted host as TEE GPU -- the single most misleading thing this UI could
// say, since it tells a buyer their activations are covered by an attestation
// that does not cover them. `shielded` is the discriminator; `gpu` is not.
export function enclaveClassOf(row){
  const a = (row && row.availability) || {};
  const shielded = !!(a.shielded && a.shielded.vramGb > 0);
  const gpu = a.gpu === true;
  if (shielded) return { kind: "shielded-gpu", inTee: false, shielded: true, hasCard: true };
  if (gpu)      return { kind: "tee-gpu",      inTee: true,  shielded: false, hasCard: true };
  return { kind: "cpu", inTee: false, shielded: false, hasCard: false };
}

// Does this box have a CPU TEE at all? Answered from EVIDENCE, never from the
// box's flavor: a CPU-only row is not "a TEE CPU because it has no card", and a
// box with some other root of trust (a phone-anchored host, one day) must not
// inherit a green pill it did not prove. Two sources count, in this order:
//   - availability.teeCpu: the technology the box DETECTED from its own
//     attestation document ("amd-sev-snp" / "intel-tdx"). A metal dev box
//     reports its unattested format here, which is a known NO.
//   - row.mode === "snp": the relay verified a fresh SEV-SNP quote (AMD
//     signature chain included, METAL_REQUIRE_VCEK defaults on) from this
//     tunnel box when it attached. Proof from the relay's side, and the only
//     evidence a build that predates the field can offer.
//   - row.mode === "avf": the relay verified an Android protected-VM
//     attestation chain, rooted at Google, naming an allowlisted anchor build
//     (relay/avf-verify.mjs): a PHONE-anchored host. Same rule, different root.
// Anything else is UNKNOWN, not "no": an older build simply never said.
export const CPU_TEE_TECHNOLOGIES = { "amd-sev-snp": "AMD SEV-SNP", "intel-tdx": "Intel TDX", "android-avf-pvm": "Android protected VM" };
export function teeCpuOf(row){
  const a = (row && row.availability) || {};
  const tech = typeof a.teeCpu === "string" && a.teeCpu ? a.teeCpu : null;
  if (tech && CPU_TEE_TECHNOLOGIES[tech])
    return { real: true, known: true, technology: tech, label: CPU_TEE_TECHNOLOGIES[tech], source: "attestation" };
  if (row && row.tunnel && row.mode === "snp")
    return { real: true, known: true, technology: "amd-sev-snp", label: CPU_TEE_TECHNOLOGIES["amd-sev-snp"], source: "relay" };
  if (row && row.tunnel && row.mode === "avf")
    return { real: true, known: true, technology: "android-avf-pvm", label: CPU_TEE_TECHNOLOGIES["android-avf-pvm"], source: "relay" };
  if (tech) return { real: false, known: true, technology: tech, label: tech, source: "attestation" };
  return { real: false, known: false, technology: null, label: null, source: null };
}

// The VRAM a shielded card actually sells: the worker's budget, not the physical
// total. The untrusted host keeps the rest (on a desktop, an X server), and
// quoting the physical number would advertise capacity no tenant can have.
//
// `frac` is what can be LEASED, and on this tier that is NOT what is resident.
// A shielded worker holds only the model's encoded weights -- a 0.5B model is
// ~0.7 GB of a 6.5 GB budget -- and masked activations are transient kilobytes
// that arrive and leave with each exchange. So the card reads ~96% physically
// free while it is fully leased at an 85% share, and quoting that reading told
// a buyer they could rent a card the allocator would refuse them. A tenant's
// share here is a scheduling and billing reservation, not VRAM occupancy, which
// is exactly how it differs from a passed-through card.
//
// `gpuShareFree` is the seller's own allocator answering that question, so it
// is the headline. The physical reading stays available as `freeGb`/`vramFrac`
// for anyone who wants to show what the silicon is doing -- it is honest and
// useful, just not an answer to "how much can I buy". Rows too old to report
// gpuShareFree fall back to the physical ratio rather than showing nothing.
export function shieldedPoolOf(row){
  const a = (row && row.availability) || {};
  const sh = a.shielded;
  if (!sh || !(sh.vramGb > 0)) return null;
  const total = sh.vramBudgetGb > 0 ? sh.vramBudgetGb : sh.vramGb;
  const freeGb = Math.min(sh.vramFreeGb, total);
  // Held by live tenants (protocol 1.3: each reserves its share at HELLO and
  // the worker holds exactly that). The box's vramFreeGb is already net of it;
  // it rides along so a row can say WHY the card is not all free.
  const reservedGb = Number.isFinite(Number(sh.vramReservedGb)) ? Math.max(0, Math.min(total, Number(sh.vramReservedGb))) : 0;
  const clamp = (v) => Math.max(0, Math.min(1, v));
  const vramFrac = total > 0 ? clamp(freeGb / total) : 0;
  // Number.isFinite, not truthiness: 0 is the value this whole change exists to
  // report, and `||` would silently swap a fully-leased card back to 96%.
  const own = Array.isArray(a.shieldedCards) ? a.shieldedCards.find(c => c.id === sh.id) : null;
  const lease = Number(own ? own.gpuShareFree : a.gpuShareFree);
  const frac = Number.isFinite(lease) ? clamp(lease) : vramFrac;
  return { total, freeGb, reservedGb, vramFrac, frac, leasableGb: total * frac };
}

// One enclave row's sizing hardware: its own advertised numbers per axis, the
// fallback constants for anything it omits (old builds).
export function enclaveSpecOf(row){
  const a = (row && row.availability) || {};
  const s = { ...FALLBACK };
  for (const k of ["cardVramGb", "cardTflops", "nodeVcpus", "nodeRamGb", "nodeGflops"]){
    const v = Number(a[k]);
    if (Number.isFinite(v) && v > 0) s[k] = v;
  }
  return s;
}

// The model volumes a deployment asks for, off the spec object: `v.volumes`,
// which specOf() fills from the version's approved config and the deploy
// console overrides with the picker's live ticks (an edited config can change
// them before signing). Absent/empty = no volume constraint.
// An enclave row's display name: the relay's `name` field, else the host part
// of its endpoint (tunnel:// rows carry a routing key, not a hostname).
export const nameOf = (row) => (row && row.name)
  || String((row && row.endpoint) || "").replace(/^[a-z]+:\/\//, "").split(".")[0] || "enclave";

export function volsWanted(v){
  const list = Array.isArray(v && v.volumes) ? v.volumes : [];
  return [...new Set(list.map((x) => String((x && x.name) || x || "")).filter(Boolean))];
}
// Does this enclave row advertise every one of them? A row with no `volumes`
// field at all carries none — an enclave that cannot tell us what it has cannot
// be targeted for work that needs a specific volume.
export function hasVolumes(a, want){
  if (!want.length) return true;
  const have = new Set(((a && a.volumes) || []).map((x) => String((x && x.name) || x || "")));
  return want.every((n) => have.has(n));
}

// Which live enclave a deployment of `v` would land on, and the dial floors
// ON THAT BOX. Only CLAIMING enclaves count (same rule as the relay's sizing
// subset — explicit claimEnabled, with non-tunnel rows grandfathered). Among
// boxes that fit, the pick is the CHEAPEST FOR THE BUYER: the lowest minimum
// share, i.e. the biggest hardware — a share sized for it is refused by every
// smaller box's own claim gate, so the prediction stays self-consistent (a
// small box never claims a big-box-sized record). Ties mirror claim routing:
// CPU apps prefer CPU-only boxes (GPU leftovers are the graced fallback),
// then the most free pool wins. The pick is a PREDICTION, not a booking —
// the ledger is an open queue and any box at least as big may claim first —
// but shares sized for the pick are servable by it and by anything bigger.
// Returns { row, name, spec, mins, free:{gpuPct,cpuPct}, queued } — queued:
// the box fits this app but can't START it now (deploys wait on-chain) — or
// { none: "why" } when no live enclave could ever run it.
// EVERY live enclave that could host `v`, ranked: ready boxes first (in the
// preference order above), then structurally-fitting-but-full ones (a deploy
// sized for those queues). Each entry: { row, name, spec, mins, free, gpu,
// queued }. The deploy surfaces render this as the target dropdown — the
// head is the recommended pick, any entry is a valid user choice.
export function rankEnclavesFor(v, rows){
  const claiming = (rows || []).filter((e) => e && e.availability && (e.serving != null
    ? e.serving === true   // the relay's explicit verdict (rows since 2026-07-25) outranks the local rule
    : (e.availability.claimEnabled === true || (e.availability.claimEnabled == null && !e.tunnel))));
  const vramMb = Number(v && v.vramMb || 0), gpuGf = Number(v && v.gpuGflops || 0);
  const memMb = Number(v && v.memMb || 0), cpuGf = Number(v && v.cpuGflops || 0);
  // A deployment's GPU need has two strengths. The app's own specs (vram /
  // gpuGflops) are a HARD requirement - no CPU box can ever serve them. A
  // bought GPU share with `gpu.optional` is a PREFERENCE: it wants a card,
  // but a CPU-only box is a legal home and the ledger bills it only the cpu
  // half there (a CPU enclave posts no GPU price). So soft-GPU work ranks
  // every GPU box first and keeps CPU boxes as fallback, rather than either
  // queueing for a card or silently forgetting it wanted one.
  // TWO different softenings, and they must not be confused:
  //   v.gpuOptional     — the PUBLISHER's, from the version config. Demotes the
  //                       version's own axes to a preference, so a CPU box
  //                       becomes a legal home for the app at all.
  //   v.depGpuOptional  — the DEPLOYMENT's, from its options envelope. Softens
  //                       the owner's bought slice only. It can never waive the
  //                       publisher's requirement: an envelope speaks for the
  //                       owner's dial, not for what the app needs to start.
  const hardGpu = (vramMb > 0 || gpuGf > 0) && !(v && v.gpuOptional);
  const softGpu = !hardGpu && (
       (!!(v && v.gpuOptional) && (vramMb > 0 || gpuGf > 0))
    || (!!(v && v.depGpuOptional) && Number(v && v.gpuMilli || 0) > 0));
  const needsGpu = hardGpu;
  const wantVols = volsWanted(v);
  const cand = claiming.map((row) => {
    const a = row.availability, spec = enclaveSpecOf(row);
    const gpu = a.gpu === true;
    // structural fit: could this BOX ever run the app, at any share?
    // Model volumes are PER-BOX, like the card: they are attached to an enclave,
    // not fetched on demand, so a box that doesn't carry every volume this
    // deployment asks for can never run it — its own claim gate refuses the
    // record (supervisor considerClaim). Targeting it anyway would send the
    // claim hint to a box that declines, and the deploy would sit in the open
    // queue waiting for whichever box does carry them.
    const fits = (!needsGpu || (gpu && vramMb / 1024 <= spec.cardVramGb && gpuGf / 1000 <= spec.cardTflops))
              && memMb <= spec.nodeRamGb * 1024 && cpuGf <= spec.nodeGflops
              && hasVolumes(a, wantVols);
    const mins = minPctsOf(v, spec, { volGb: volGbOf(a, wantVols) });
    // Whether this box would serve the model on CORES rather than its card —
    // a card too small for the app cannot hold it at any share, so the runner
    // falls back (gpuRouting). Only the LABEL depends on this: the floors do
    // not, because the weights are node-charged either way.
    const cardCanHold = gpu && mins.gpuNeedPct <= 100;
    const weightsOnCores = hardGpu ? false : (softGpu ? !cardCanHold : true);
    const free = { gpuPct: Math.floor((a.gpuShareFree || 0) * 100), cpuPct: Math.floor((a.cpuShareFree || 0) * 100) };
    const now = fits && (!needsGpu || free.gpuPct >= mins.gpuPct) && free.cpuPct >= mins.cpuPct;
    const name = nameOf(row);
    // what running THIS app on THIS box costs per second at its minimum
    // shares: the box's own posted price times the share its hardware forces.
    // Big-and-dear can beat small-and-cheap, so the ranking compares money.
    const price = enclavePriceOf(row);
    const minRate = shareRates(mins.gpuPct, mins.cpuPct, spec, price).rate;
    return { row, name, spec, mins, free, gpu, fits, now, price, minRate, weightsOnCores };
  }).filter((c) => c.fits && (needsGpu ? c.gpu : true));
  // A GPU box CAN serve model-volume work with no GPU share — the tenant gets
  // the ggml CPU backend on the cores it bought, same as on a CPU box — but it
  // must never be the AUTOMATIC choice for it: the card is the scarce thing,
  // and parking cores-only work on a GPU box spends a machine someone else
  // needs. So demote rather than exclude, ahead of price, and let an owner
  // still pick it deliberately (a manual move names its destination).
  // soft-GPU inverts the demotion: the card IS wanted here, so a GPU box is
  // the preferred home and a CPU box is the fallback.
  const demoted = (c) => softGpu ? (c.gpu ? 0 : 1)
                       : ((wantVols.length && !needsGpu && c.gpu) ? 1 : 0);
  // CHEAPEST FIRST, in money — then the old tiebreaks (smallest minimum share,
  // CPU boxes before GPU leftovers for CPU work, most free pool)
  const order = (list) => list.slice().sort((x, y) => (demoted(x) - demoted(y)) || (x.minRate - y.minRate) || (needsGpu
    ? (x.mins.gpuPct - y.mins.gpuPct) || (y.free.gpuPct - x.free.gpuPct)
    : (x.mins.cpuPct - y.mins.cpuPct) || ((x.gpu === true) - (y.gpu === true)) || (y.free.cpuPct - x.free.cpuPct)));
  // cpuNn: on this box the app's model volumes run on CPU cores, not a card.
  // TWO ways to land there and the flag covers both: a GPU box hosting a
  // 0-GPU tenant (it has a card, this deployment did not buy it), and a
  // card-less box hosting soft-GPU work (there is no card to buy). Callers
  // label it "CPU only" — what the DEPLOYMENT gets is the decision-relevant
  // fact, and it is true either way; naming the box's own hardware was not
  // (it called metal0 a GPU box).
  // Soft-GPU reads the SIZING verdict rather than "does the box have a card":
  // a card too small to hold the app is a card this deployment cannot use, and
  // the runner will serve it on cores there. Same fact, same source.
  const onCores = (c) => softGpu ? c.weightsOnCores : (wantVols.length && !needsGpu && c.gpu);
  return [...order(cand.filter((c) => c.now)).map((c) => ({ ...c, queued: false, cpuNn: !!onCores(c) })),
          ...order(cand.filter((c) => !c.now)).map((c) => ({ ...c, queued: true, cpuNn: !!onCores(c) }))];
}

// The box a deployment's shares are ALREADY judged against: the enclave
// holding its lease. That runner applies an owner's version change / resize in
// place and gates it on ITS OWN card and node (supervisor minSharesOf), so
// once a lease exists the fleet-wide aggregate is the wrong ruler in both
// directions - it is the SMALLEST box on every axis (over-asks: a version this
// deployment can run reads as unaffordable), while the relay's best-box
// /v1/pricing numbers under-ask on any smaller runner.
// `d` is the ledger record (runner bytes32 + leaseUntil), `rows` the relay's
// /enclaves table, whose row `id` IS that runner. Returns { row, name, spec }
// or null when nothing is pinned - no live lease (the next claim may come from
// any box, so the aggregate's over-ask is right again), an unknown runner, or
// no fleet view at all.
const ZERO_B32 = "0x" + "0".repeat(64);
export function leaseHostOf(d, rows, nowMs){
  const runner = String((d && d.runner) || "").toLowerCase();
  if (!/^0x[0-9a-f]{64}$/.test(runner) || runner === ZERO_B32) return null;
  if (!(Number(d.leaseUntil) * 1000 > (nowMs == null ? Date.now() : nowMs))) return null;
  const row = (rows || []).find((e) => String((e && e.id) || "").toLowerCase() === runner);
  if (!row || !row.availability) return null;
  return { row, name: row.name || String(row.endpoint || "").replace(/^[a-z]+:\/\//, "").split(".")[0] || "its enclave",
           spec: enclaveSpecOf(row), price: enclavePriceOf(row) };
}

export function pickEnclaveFor(v, rows){
  const claiming = (rows || []).filter((e) => e && e.availability
    && (e.availability.claimEnabled === true || (e.availability.claimEnabled == null && !e.tunnel)));
  if (!claiming.length) return { none: "no live enclave is taking work right now" };
  const ranked = rankEnclavesFor(v, rows);
  if (ranked.length) return ranked[0];
  const needsGpu = Number(v && v.vramMb || 0) > 0 || Number(v && v.gpuGflops || 0) > 0;
  // Say which of the two constraints actually bit. "No enclave is big enough"
  // is a lie about a fleet that has the hardware and lacks the weights, and the
  // fix differs per case: attach the volume, pick another model, or resize.
  const wantVols = volsWanted(v);
  const missing = wantVols.filter((n) => !claiming.some((e) => hasVolumes(e.availability, [n])));
  if (missing.length) return { none: `no live enclave carries the model volume ${missing.join(" or ")}` };
  const carriers = wantVols.length ? claiming.filter((e) => hasVolumes(e.availability, wantVols)) : claiming;
  if (!carriers.length)
    return { none: `no single live enclave carries all of ${wantVols.join(" + ")} (a deployment mounts its volumes on ONE box)` };
  // The volumes exist on a box; that box just can't run this app. Name it —
  // otherwise the reader blames the fleet for hardware it does have, on some
  // OTHER box that hasn't got the model.
  if (wantVols.length && carriers.length !== claiming.length){
    const who = carriers.length <= 3 ? carriers.map(nameOf).join(", ") : `${carriers.length} enclaves`;
    return { none: needsGpu && !carriers.some((e) => e.availability.gpu === true)
      ? `this app needs a GPU, and ${who} — the only enclave${carriers.length > 1 ? "s" : ""} carrying ${wantVols.join(" + ")} — ${carriers.length > 1 ? "have" : "has"} none`
      : `${who} carr${carriers.length > 1 ? "y" : "ies"} ${wantVols.join(" + ")}, but ${carriers.length > 1 ? "none has" : "its hardware is not"} big enough for this app's specs` };
  }
  return { none: needsGpu && !claiming.some((e) => e.availability.gpu === true)
    ? "this app needs a GPU and no live enclave has one"
    : "no live enclave's hardware is big enough for this app's specs" };
}

// Where a deployment could be moved RIGHT NOW: every enclave that could host
// it, minus the one already holding its lease. Same eligibility rule as the
// deploy target list (rankEnclavesFor) — hardware, model volumes and free pool
// — because a move IS a re-claim: the current runner hands the lease back and
// the fleet claims it again, so a target that would refuse the record at
// create time refuses it here too.
//
// `queued` entries stay in the list on purpose. A box that fits the app but is
// full right now is a legitimate destination: the deployment sits in the open
// queue until that box frees up, exactly like a fresh deploy sized for it. The
// caller labels them so the choice is informed rather than hidden.
export function moveTargetsFor(v, rows, currentRunnerId){
  const cur = String(currentRunnerId || "").toLowerCase();
  return rankEnclavesFor(v, rows)
    .filter((c) => String((c.row && c.row.id) || "").toLowerCase() !== cur);
}

// Why a deployment has nowhere to go, in the reader's terms. Only called when
// moveTargetsFor came back empty, and deliberately specific: "no other enclave
// fits" sends someone hunting for hardware when the real answer is a dial they
// can change (buy GPU share) or a fleet fact they can act on (only one box
// carries the model).
export function moveBlockReason(v, rows, currentRunnerId){
  const cur = String(currentRunnerId || "").toLowerCase();
  const others = (rows || []).filter((e) => e && e.availability
    && String((e.id) || "").toLowerCase() !== cur
    && (e.serving != null ? e.serving === true : e.availability.claimEnabled !== false));
  if (!others.length) return "no other enclave is taking work right now";
  const wantVols = volsWanted(v);
  const needsGpu = Number(v && v.vramMb || 0) > 0 || Number(v && v.gpuGflops || 0) > 0;
  // absent weights first: that is the more fundamental blocker, and both can be
  // true at once (the only other box is a GPU box AND lacks the volume)
  const missing = wantVols.filter((n) => !others.some((e) => hasVolumes(e.availability, [n])));
  if (missing.length) return `no other live enclave carries ${missing.join(" or ")}`;
  return "no other live enclave's hardware fits this app";
}

/* Moving a soft-GPU deployment ONTO a GPU box: what it should re-buy.

   A deployment whose card is optional runs on cores wherever it lands — even
   on a GPU enclave, because the manager grants wasi-nn on the card only to
   tenants holding GPU share. So a move that puts it on a GPU box without
   re-buying the slice quietly delivers the slow thing on the fast machine.

   Returns { gpuPct, cpuPct } to pass to setShares, or null when there is
   nothing to do: the target has no card, the version declares no GPU axes to
   size a slice from (nothing to buy), the deployment already holds one, or the
   card requirement was never optional (a hard-GPU deployment could not have
   been on a CPU box in the first place).

   The slice is sized purely from the version's declared axes — a CPU-heavy app
   moving onto a card buys the small slice it actually wants. On a pre-13 ledger
   the caller passes the result through liftSharesForLedger first, since that
   create()/setShares still refuses a GPU share under the CPU one. */
/* The other direction: moving a GPU-holding deployment onto a CARD-LESS box.

   The shares stay legal there (a soft-GPU record may run on cores) and the
   ledger already stops charging for the card — a box with no GPU posts no GPU
   price, so _hostRate bills only the cpu half. What the slice still costs is
   PLACEMENT: while it holds one, the deployment's minimum share is sized for
   hardware this box does not have, and every future claim reads as GPU work.
   So offer to drop it, and say plainly that it is not about the money.

   Returns { gpuPct: 0, cpuPct } or null when there is nothing to drop. */
export function gpuDowngradeForMove(v, target, boughtGpuMilli, boughtCpuMilli){
  if (!target || !target.row || target.row.availability?.gpu === true) return null;
  if (!(Number(boughtGpuMilli || 0) > 0)) return null;        // nothing bought to give back
  return { gpuPct: 0, cpuPct: Math.max(1, Math.round(Number(boughtCpuMilli || 0) / 10)) };
}

export function gpuUpgradeForMove(v, target, boughtGpuMilli, boughtCpuMilli){
  if (!target || !target.row || target.row.availability?.gpu !== true) return null;
  if (Number(boughtGpuMilli || 0) > 0) return null;          // already buying a card
  // Only the PUBLISHER's flag can size an upgrade, because the slice is sized
  // from the version's declared axes and only that flag makes them a
  // preference. The deployment-level flag softens a slice the owner already
  // bought — so a deployment carrying it is not at 0% GPU and was filtered out
  // above. Checking it here too would just be noise that never fires.
  if (!(v && v.gpuOptional)) return null;
  const want = wantedGpuPct(v, target.spec);
  if (!(want > 0)) return null;                              // no declared axes to size from
  const cpuPct = Math.max(1, Math.round(Number(boughtCpuMilli || 0) / 10));
  return { gpuPct: want, cpuPct };
}
