/* ============================================================
   <c-fleet-list> - per-enclave capacity rows (the relay's
   /enclaves table). Assign `.rows` (already sorted upstream) and
   it renders each box's two capacity pools. Copy says "available",
   never "free": on a page that sells compute, "60 GB free" reads as
   a price, not as headroom.
   ============================================================ */
import { EnclaveElement, register } from "../../js/lib/enclave-element.js";
import { esc, fmtNum, short, showToast } from "../../js/core/util.js";
import { starsHtml } from "../../js/core/reviews.js";
import { hrevConfigured, hrevTallies, hrevMine, encCall, HREV_SEL, waitReceipt, REVIEW_MAX_BODY } from "../../js/core/chain.js";
import { HOST_REVIEWS_ADDRESS } from "../../js/core/config.js";
import { Enclave } from "../../js/core/api.js";
import { connectWallet, ensureBaseChain, sendTx } from "../../js/core/wallet.js";
import { serverSpec, enclavePriceOf, enclaveClassOf, shieldedPoolOf, teeCpuOf } from "../../js/core/pricing.js";
import { REGISTRY_ADDRESS } from "../../js/core/config.js";
import { catExplorer } from "../../js/core/chain.js";

class FleetList extends EnclaveElement {
  static properties = { rows: null };
  static templateUrl = new URL("./fleet-list.html", import.meta.url);

  renderedCallback() {
    const list = this.querySelector(".fleet-list"); if (!list) return;
    // only enclaves that SERVE (take on-chain work) are shown: a live but
    // non-claiming box (relay row serving:false) is operational truth, not
    // sellable capacity - listing it would advertise hardware nobody can buy.
    // Rows from an older relay carry no verdict and stay visible.
    const rows = (this.rows || []).filter((e) => e.serving !== false);
    const meter = (pct) => '<i class="fleet-meter" aria-hidden="true"><b style="width:' + Math.max(0, Math.min(100, pct)) + '%"></b></i>';
    // one stat cell: bright available amount, then the "≈"/"/ total" context and
    // the label in dim ink so the number is what the eye lands on
    const stat = (avail, total, unit, label, title) =>
      '<span class="fleet-stat"' + (title ? ' title="' + esc(title) + '"' : '') + '>'
      + '<b><i>≈</i>' + avail + '<i> / ' + total + '</i>' + (unit ? " " + unit : "") + '</b>'
      + '<small>' + label + '</small></span>';
    // the price sits directly under the pool's label (its badge) - bright number,
    // dim "/hr" - in the label column's otherwise-empty second row, so it
    // costs no space and each pool names its own rate (card vs node). It is
    // the WHOLE card / node per hour, the ledger's basis; a share pays its
    // fraction. Trailing ".00" trims like the docs' rates.
    const perHr = (v) => "$" + (v * 3600).toFixed(2).replace(/\.00$/, "");
    // one pool = a [label | meter | pct] header line, the price under the
    // label, stat cells underneath. The label is the pool's badge (see the
    // row builder): the pill names the pool, so nothing else has to.
    const pool = (label, pct, stats, price) =>
      '<div class="fleet-pool">'
      + '<span class="fleet-pool-label">' + label + '</span>'
      + meter(pct)
      + '<span class="fleet-pool-pct"><b>' + pct + '%</b> available</span>'
      // the label column's second row holds the pool's rate. A pool whose seller
      // has posted no ask simply leaves it empty rather than inventing one.
      + (price != null ? '<span class="fleet-pool-price"><b>' + perHr(price) + '</b>/hr</span>' : '')
      + '<span class="fleet-stats">' + stats + '</span>'
      + '</div>';
    list.innerHTML = (!rows.length
      ? '<div class="fleet-empty">no live enclaves right now</div>'
      : rows.map(e => {
          const a = e.availability || {};
          const gpu = a.gpu === true;
          const gFree = a.gpuShareFree != null ? a.gpuShareFree : (gpu ? a.maxShare || 0 : 0);
          const cFree = a.cpuShareFree != null ? a.cpuShareFree : (gpu ? 0 : a.maxShare || 0);
          const gPct = Math.floor(gFree * 100), cPct = Math.floor(cFree * 100);
          // the relay names each row (tunnel enclaves: their tunnel name, e.g.
          // "metal0"); the endpoint-derived fallback covers older relays — and
          // strips ANY scheme, so a tunnel:// row never renders as a pseudo-URL
          const name = e.name || String(e.endpoint || "").replace(/^[a-z]+:\/\//, "").split(".")[0] || "enclave";
          // Any host may also CARRY traffic; one with no resources at all only
          // carries it, and that is what this badge reads — no capacity, so
          // nothing to sell and nothing to meter. Empty CPU/GPU bars would say
          // "full", which is the opposite of the truth, so the row lists the
          // network services the box offers instead.
          if (e.relay === true) {
            const r = a.relay || {};
            const svc = [["sni", "app traffic"], ["tcp", "tcp ports"], ["udp", "udp ports"],
                         ["egress", "outbound ip"], ["tunnelHub", "tunnel hub"]]
              .filter(([k]) => r[k] === true).map(([, label]) => label);
            return '<div class="fleet-row" title="' + esc(e.endpoint || "") + '">'
              + '<span class="fleet-head">'
              + '<span class="ap-badge">relay</span>'
              + '<span class="fleet-name">' + esc(name) + '</span>'
              + (r.region ? '<span class="fleet-relay-region">' + esc(r.region) + '</span>' : '')
              + '</span>'
              + '<span class="fleet-relay-note">'
              + (svc.length ? 'carries ' + svc.map(esc).join(" · ") : 'carries no declared services')
              + (r.ports ? ' · ports ' + esc(r.ports) : '')
              + (r.v6Prefix ? ' · ' + esc(r.v6Prefix) : '')
              + '</span>'
              + '</div>';
          }
          // A card on the box's UNTRUSTED host, reached by masked offload — the
          // enclave uses it without trusting it, so the row must not read as an
          // in-enclave GPU. It gets its own badge and its own pool, and it is
          // deliberately NOT folded into `gpu`: that flag means the card is
          // inside the measured enclave, which is a different thing to buy.
          const sh = a.shielded && a.shielded.vramGb > 0 ? a.shielded : null;
          // A shielded box now reports `gpu: true` -- its card IS its card, and is
          // sold as one. So `gpu` alone no longer means "inside the enclave", and
          // reading it that way is how this row briefly badged a card on an
          // untrusted host as TEE GPU. The distinction a buyer needs is where the
          // silicon is, and that is exactly `sh`.
          const cls = enclaveClassOf(e);
          const inTee = cls.inTee;
          // THE POOL LABELS ARE THE BADGES. Each pool is named by what it is, in
          // the colour that means it, right beside its own meter; the header
          // carries only the box's name and rating. Pills in the header and a
          // plain "CPU"/"GPU" beside the meters said the same thing twice in two
          // registers, and the reader had to pair them up.
          //
          // The CPU pool: jade "TEE CPU" only when there is EVIDENCE the box's
          // CPU is a confidential-computing TEE (the technology its own
          // attestation document presented, or the relay's verified SEV-SNP
          // attach, see teeCpuOf) -- never inferred from the box having no card.
          // A separate root of trust is coming (hosts anchored on a phone), and
          // those boxes must not inherit a green pill they did not prove: amber
          // "NO TEE CPU" when the box reports a non-TEE document (a metal dev
          // box), plain "CPU" when it has not said.
          const tc = teeCpuOf(e);
          const teeCpuBadge = tc.real
            ? '<span class="ap-badge ok" title="' + esc(tc.label) + ' confidential VM: '
              + (tc.source === "relay"
                  ? 'the relay verified a fresh hardware quote from this box when it attached'
                  : 'this box\u2019s attestation document presents a hardware quote')
              + ', so every vCPU it sells runs inside the measured enclave and is covered by'
              + ' its attestation.">tee cpu</span>'
            : tc.known
            ? '<span class="ap-badge warn" title="This box reports no CPU TEE (attestation format '
              + esc(tc.technology) + '): nothing it sells is covered by a hardware attestation.">no tee cpu</span>'
            : '<span class="ap-badge" title="This box has not reported whether its CPU is a TEE:'
              + ' its build predates the field, or its attestation document has not been read yet.'
              + ' Only a hardware quote earns the green pill.">cpu</span>';
          // The GPU pool: jade "TEE GPU" for a card INSIDE the measured enclave,
          // iris "GPU" for one on the untrusted host reached by masked offload --
          // the ABSENCE of "tee" is the signal, and the tooltip says outright
          // that this card is outside the enclave and outside its measurement.
          const cardBadge = inTee
            ? '<span class="ap-badge ok" title="This card is INSIDE the confidential'
              + ' enclave and covered by its attestation.">tee gpu</span>'
            : sh
            ? '<span class="ap-badge info" title="' + esc(sh.card || "gpu")
              + ' on this box\u2019s untrusted host, used by masked offload: it receives '
              + 'public weights and one-time-padded activations, and every result is '
              + 'verified. The card is outside the enclave and outside its measurement, '
              + 'so this is NOT a TEE GPU \u2014 your activations are protected by the '
              + 'masking, not by the card.">gpu</span>'
            : "";
          // What is SELLABLE is the worker's budget, not the physical card: the
          // untrusted host keeps the rest (on a desktop, an X server). Showing the
          // physical total here while the GPU pool showed the budget is what put
          // two differently-sized GPU rows on one single-card box.
          const shPool = shieldedPoolOf(e);
          const shTotal = shPool ? shPool.total : 0;
          // LEASABLE, not resident. A shielded worker keeps only the model's
          // encoded weights on the card, so the silicon reads nearly empty while
          // the card is fully booked; showing that reading as "available" quoted
          // capacity the allocator would refuse to sell. The physical number is
          // still true and still worth saying, so it moves into the tooltip.
          const shLeasableGb = shPool ? shPool.leasableGb : 0;
          const shPhysFreeGb = shPool ? shPool.freeGb : 0;
          const shReservedGb = shPool ? shPool.reservedGb : 0;
          const shFree = shPool ? shPool.frac : 0;
          const shPct = Math.floor(shFree * 100);
          const shVramTitle = sh?.pooled
            ? fmtNum(shTotal) + ' GB combined across ' + sh.cardCount + ' GPUs. Each share reserves the same fraction of every card. Models are split automatically; overflow uses the enclave CPU.'
            : shPool
            ? fmtNum(shPhysFreeGb) + ' GB of the ' + fmtNum(shTotal) + ' GB budget is free on the card'
              + (shReservedGb > 0 ? ' (' + fmtNum(shReservedGb) + ' GB is held by tenants)' : '')
              + ', and ' + shPct + '% is available to lease. A tenant reserves its share of the'
              + ' card when it connects and the worker holds exactly that, so the two figures'
              + ' differ only by what the host is doing with the card outside Enclave.'
            : '';
          const s = serverSpec();   // adopted fleet hardware; display fallback for rows that omit their own
          const vramGb = a.cardVramGb || s.cardVramGb, tflops = a.cardTflops || s.cardTflops;
          const ramGb = a.nodeRamGb || s.nodeRamGb, vcpus = a.nodeVcpus || s.nodeVcpus;
          const price = enclavePriceOf(e);   // this box's posted ask; the fleet price where it posts none
          return '<div class="fleet-row" title="' + esc(e.endpoint || "") + '">'
            + '<span class="fleet-head">'
            + '<span class="fleet-name">' + esc(name) + '</span>'
            + this._ratingHtml(e)
            + '</span>'
            + (sh ? pool(cardBadge, shPct,
                stat(fmtNum(shLeasableGb), fmtNum(shTotal), "GB", "vram available", shVramTitle)
                // The card's RATED figure, which is what every other row quotes and
                // what a share is sized against. This cell used to show the MEASURED
                // masked rate instead -- honest in isolation, and unreadable in a
                // list: an RTX 3070 drew "0 / 2 tflops" beside an H200's "175 / 989",
                // so the columns implied a 500x gap where the real one is ~23x, and
                // the number did not match the basis the same row's share was
                // computed from.
                //
                // The measured rate has not been dropped, it has moved to the
                // tooltip, which is the only place the two can sit together without
                // being read as one scale. Rows too old to report a rated figure
                // keep the previous behaviour.
                + ((sh.cardTflops || a.cardTflops) > 0
                    ? stat(fmtNum(shFree * (sh.cardTflops || a.cardTflops)), fmtNum(sh.cardTflops || a.cardTflops), "", "tflops available",
                           (sh.pooled ? "Combined rated dense fp16 across the GPU pool. Model layers are distributed across cards; a single request is not guaranteed this aggregate throughput. " : "Rated dense fp16 for this card. ")
                           + "The same basis every other box "
                           + "quotes, so boxes and shares compare like for like."
                           + (sh.gmacPerSec > 0
                               ? " The masked path itself sustains " + Math.round(sh.gmacPerSec)
                                 + " G-MAC/s here, about " + fmtNum(sh.gmacPerSec * 2 / 1000)
                                 + " TFLOPS at 2 FLOP per MAC -- that is what this tier delivers, "
                                 + "and it is measured rather than rated."
                               : ""))
                    : sh.gmacPerSec > 0
                      ? stat(Math.round(shFree * sh.gmacPerSec * 2 / 1000),
                             Math.round(sh.gmacPerSec * 2 / 1000), "", "tflops available",
                             "Measured on this box: " + Math.round(sh.gmacPerSec)
                             + " G-MAC/s sustained by the masked field GEMM that actually runs "
                             + "here, converted at 2 FLOP per MAC. This box reports no rated "
                             + "figure, so the two columns are not directly comparable.")
                      : stat(esc(sh.card || "gpu"), "", "", "card")),
                price.shielded) : "")
            + (!sh?.pooled && Array.isArray(a.shieldedCards) ? a.shieldedCards.filter(c => c.id !== sh?.id).map(c => {
                const p = shieldedPoolOf({ availability: { shielded: c, gpuShareFree: c.gpuShareFree } });
                if (!p) return "";
                const badge = '<span class="ap-badge info" title="Shielded inference on the host GPU; masked inputs and verified results.">'
                  + esc(c.card || "gpu") + '</span>';
                return pool(badge, Math.floor(p.frac * 100),
                  stat(fmtNum(p.leasableGb), fmtNum(p.total), "GB", "vram available",
                    fmtNum(p.freeGb) + " GB free on the card; " + fmtNum(p.reservedGb) + " GB reserved by tenants.")
                  + stat(fmtNum(p.frac * c.cardTflops), fmtNum(c.cardTflops), "", "tflops available",
                    "Rated dense fp16. Masked field GEMM measured at " + Math.round(c.gmacPerSec) + " G-MAC/s."),
                  price.shielded);
              }).join("") : "")
            // ONLY when the card is in the enclave. A shielded card already drew its
            // pool above, from the numbers the probe actually measured; drawing
            // this one too would advertise one piece of silicon twice.
            + (inTee ? pool(cardBadge, gPct,
                stat(fmtNum(a.vramFreeGb != null ? a.vramFreeGb : gFree * vramGb), fmtNum(vramGb), "GB", "vram available")
                + stat(Math.round(gFree * tflops), Math.round(tflops), "", "tflops available"), price.full) : "")
            + pool(teeCpuBadge, cPct,
                // prefer the enclave's own figure (the RAM-reservation ledger,
                // which is what actually gates admission) over the folded
                // fraction — same precedence the VRAM cell above uses
                stat(fmtNum(a.ramGbFree != null ? a.ramGbFree : cFree * ramGb), fmtNum(ramGb), "GB", "ram available")
                + stat(fmtNum(cFree * vcpus), fmtNum(vcpus), "", "vcpu available"),
                // A "held by models" cell used to sit here, reading
                // ramNnResidentMb against the node's RAM. It was written for a box
                // whose preloaded weights make the meter read ~85% used while every
                // tenant is idle -- worth naming, at that size. In practice the only
                // boxes reporting the field hold a fraction of a percent (metal0:
                // 0.6 of 64 GB), so it explained nothing and spent a third row of the
                // CPU pool saying so. The field still crosses the wire, so bring the
                // cell back if a box ever carries enough resident weight to need it.
                price.node)
            + '<div class="fleet-rateform" data-form="' + esc(e.id || "") + '" hidden></div>'
            + '</div>';
        }).join(""));
    this._wireRate();
    // footer row: a manual refresh (dispatches `refresh`; the HOST owns the
    // fetch and re-assigns .rows, which re-renders and re-arms the button) +
    // the on-chain registry this table mirrors, linked once the address book
    // has resolved (enclaves register there)
    this._loadRatings(rows);      // stars per box, one eth_call for the panel
    const foot = this.querySelector(".fleet-foot");
    if (foot) {
      foot.innerHTML = '<button class="fleet-refresh" type="button" title="re-fetch the live fleet view">↻ refresh</button>'
        + (/^0x[0-9a-fA-F]{40}$/.test(REGISTRY_ADDRESS || "")
          ? '<a class="contract-link" href="' + catExplorer() + '/address/' + REGISTRY_ADDRESS + '" target="_blank" rel="noopener" title="EnclaveRegistry · ' + REGISTRY_ADDRESS + '">'
            + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            + '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>'
            + '<line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/></svg> contract</a>'
          : "");
      const btn = foot.querySelector(".fleet-refresh");
      btn.addEventListener("click", () => {
        btn.disabled = true;
        this.dispatch("refresh");
        setTimeout(() => { btn.disabled = false; }, 4000);   // safety net if no host listener re-assigns .rows
      });
    }
  }

  /* The host page re-assigns .rows on a 20s poll, and every assignment
     repaints the whole list - which would yank an open rating form out from
     under the wallet mid-edit (nothing "auto-hides" it; the row simply gets
     rebuilt). While a form is open the repaint is DEFERRED, then flushed when
     it closes, so fresh capacity numbers still land the moment the user is
     done. */
  requestRender(){
    if (this._rateOpen){ this._renderDeferred = true; return; }
    super.requestRender();
  }
  _closeRate(box, btn){
    this._rateOpen = false;
    if (box){ box.hidden = true; box.innerHTML = ""; }
    if (btn) btn.setAttribute("aria-expanded", "false");
    if (this._renderDeferred){ this._renderDeferred = false; super.requestRender(); }
  }

  /* ---- rating a host: the same 5-star control the app store uses ----
     The contract takes a RECEIPT - one of your funded deployments whose
     `runner` is this box - and checks it itself, so the form's job is to find
     that deployment first. Your deployment rows already name the enclave
     serving them (the relay stamps it), which is exactly the "runs here now"
     the receipt needs. No receipt = the form says why instead of offering a
     signature that would revert. */
  _wireRate(){
    for (const btn of this.querySelectorAll(".fleet-rate"))
      btn.addEventListener("click", () => this._openRate(btn));
  }
  async _openRate(btn){
    const encId = btn.dataset.encid, name = btn.dataset.rate;
    const box = this.querySelector('[data-form="' + CSS.escape(encId) + '"]');
    if (!box) return;
    if (!box.hidden) return this._closeRate(box, btn);
    box.hidden = false; btn.setAttribute("aria-expanded", "true");
    this._rateOpen = true;                 // hold off list repaints until this closes
    box.innerHTML = '<p class="fleet-gate dim">checking whether this enclave runs an app of yours…</p>';
    if (!Enclave.address){
      box.innerHTML = '<p class="fleet-gate">Only wallets whose apps this enclave has run can rate it. '
        + '<button class="btn btn-sm" data-act="connect" type="button">Connect wallet</button></p>';
      box.querySelector('[data-act="connect"]').addEventListener("click", () => connectWallet().then(() => this._openRate(btn)).catch(() => {}));
      return;
    }
    const [receipt, mine] = await Promise.all([
      this._receiptFor(name).catch(() => null),
      hrevMine(encId, Enclave.address).catch(() => null),
    ]);
    const already = mine && /^0x0*[1-9a-f]/i.test(mine.reviewer || "");
    if (!receipt && !already){
      box.innerHTML = '<p class="fleet-gate">Nothing of yours is running on <b>' + esc(name) + '</b> right now. '
        + 'Ratings come from wallets whose app this box actually ran - deploy here first, then rate it.</p>';
      return;
    }
    const d = { stars: already ? Number(mine.stars) : 0, body: already ? mine.body : "" };
    const pick = [1, 2, 3, 4, 5].map((n) =>
      '<label class="revs-pick-star' + (d.stars >= n ? " on" : "") + '">'
      + '<input class="sr-only" type="radio" name="hrevStars-' + esc(encId) + '" value="' + n + '"' + (d.stars === n ? " checked" : "") + '>'
      + '<span aria-hidden="true">★</span><span class="sr-only">' + n + (n === 1 ? " star" : " stars") + '</span></label>').join("");
    box.innerHTML = '<div class="revs-write">'
      + '<fieldset class="revs-pick"><legend>' + (already ? "Update your rating of " : "Rate ") + esc(name) + '</legend>' + pick + '</fieldset>'
      + '<textarea class="revs-body" rows="2" placeholder="How did this box run it? (optional)"></textarea>'
      + '<div class="revs-write-foot">'
        + '<span class="revs-count">' + REVIEW_MAX_BODY + ' left</span>'
        + (receipt ? '<span class="revs-receipt" title="the funded deployment this enclave is running for you">receipt ' + esc(short(receipt)) + '</span>'
                   : '<span class="revs-receipt" title="you have rated this box before, so an edit needs no fresh receipt">editing your rating</span>')
        + '<button class="btn btn-primary btn-sm" data-act="post" type="button" disabled>' + (already ? "Update rating" : "Post rating") + '</button>'
      + '</div></div>';
    const ta = box.querySelector(".revs-body"); ta.value = d.body || "";
    const post = box.querySelector('[data-act="post"]');
    const count = box.querySelector(".revs-count");
    const sync = () => {
      const on = box.querySelector('input[type="radio"]:checked');
      const left = REVIEW_MAX_BODY - new TextEncoder().encode(ta.value || "").length;
      count.textContent = left + " left"; count.classList.toggle("over", left < 0);
      post.disabled = this._busy || !on || left < 0;
      for (const l of box.querySelectorAll(".revs-pick-star"))
        l.classList.toggle("on", on && Number(l.querySelector("input").value) <= Number(on.value));
    };
    box.addEventListener("change", sync); ta.addEventListener("input", sync); sync();
    post.addEventListener("click", () => this._postRate(box, encId, name, receipt, post, sync));
  }

  /* One of MY deployments this box is running now (the relay stamps each row
     with the serving enclave's name). Funded is implied: a row only has a
     runner because a lease was claimed, and the contract re-checks anyway. */
  async _receiptFor(name){
    const res = await Enclave.listDeployments();
    const rows = Array.isArray(res) ? res : ((res && (res.deployments || res.items || res.data)) || []);
    const hit = rows.find((d) => d && d.enclave === name && /^0x[0-9a-f]{64}$/i.test(d.id || "")
      && ["running", "claimed", "provisioning"].includes(d.status || ""));
    return hit ? hit.id : null;
  }

  async _postRate(box, encId, name, receipt, btn, sync){
    const on = box.querySelector('input[type="radio"]:checked');
    if (!on) return;
    const body = box.querySelector(".revs-body").value || "";
    this._busy = true; btn.disabled = true; btn.textContent = "signing…";
    try {
      if (!Enclave.provider) await connectWallet();
      await ensureBaseChain();
      const data = encCall(HREV_SEL.post, [
        { t: "bytes32", v: encId },
        { t: "bytes32", v: receipt || "0x" + "0".repeat(64) },
        { t: "uint", v: Number(on.value) },
        { t: "str", v: body },
      ]);
      const hash = await sendTx(HOST_REVIEWS_ADDRESS, data);
      showToast("rating " + name + " · " + hash.slice(0, 12) + "…");
      await waitReceipt(hash);
      showToast("rated " + name);
      this._closeRate(box, this.querySelector('.fleet-rate[data-encid="' + CSS.escape(encId) + '"]'));
      this._tallyKey = null;                 // force a re-read so the stars move
      this._loadRatings(this.rows || []);
    } catch (e) {
      showToast("rating failed: " + ((e && (e.shortMessage || e.message)) || e));
      btn.textContent = "Post rating";
    } finally { this._busy = false; if (sync) sync(); }
  }

  /* Stars for a box, from EnclaveHostReviews. Absent contract (not deployed /
     not in the address book yet) renders NOTHING rather than a fake 0 - an
     unrated fleet and an unreadable one are different claims. */
  _ratingHtml(e){
    const t = this._tallies && this._tallies[String(e.id || "").toLowerCase()];
    if (!hrevConfigured()) return "";
    const rate = '<button class="fleet-rate btn btn-sm" type="button" data-rate="' + esc(e.name || "") + '" data-encid="' + esc(e.id || "") + '" aria-expanded="false" '
      + 'title="Rate this enclave - open to wallets whose app it is running">rate</button>';
    if (!t || !t.count)
      return '<span class="fleet-rating fleet-unrated" title="No wallet has rated this enclave yet">unrated</span>' + rate;
    const avg = t.sum / t.count;
    return '<span class="fleet-rating" title="' + t.count + ' rating' + (t.count === 1 ? "" : "s") + ' from wallets whose apps this enclave ran">'
      + starsHtml(avg) + '<small>' + avg.toFixed(1) + ' (' + t.count + ')</small></span>' + rate;
  }

  /* One talliesOf call covers every visible box. Cached per paint; a fleet
     row set that hasn't changed doesn't re-read the chain. */
  async _loadRatings(rows){
    if (!hrevConfigured()) return;
    const ids = rows.map((e) => String(e.id || "")).filter((x) => /^0x[0-9a-f]{64}$/i.test(x));
    const key = ids.join(",");
    if (!ids.length || key === this._tallyKey) return;
    this._tallyKey = key;
    try {
      const rowsT = await hrevTallies(ids);
      this._tallies = Object.fromEntries(rowsT.map((r) => [String(r.enclaveId).toLowerCase(), r]));
      this.requestRender();    // repaint with the stars in place
    } catch { /* ratings are decoration: a chain hiccup must not blank the panel */ }
  }
}
register("c-fleet-list", FleetList);
