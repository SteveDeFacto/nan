// mm36 shared prefix warming: the wire contract between the ggml backend and
// a guest that would rather WAIT for another request's warm-up of the same
// shared prefix than read it again concurrently. Pinned as strings the way
// every cross-language seam in this repo is (see mtp-shared-downgrade).
//
// The mechanism (wasm/wasmtime-nn-ggml.patch, prefix_claims.rs): the first
// context to declare a prompt with still-unparked marks CLAIMS them; a later
// context whose declared prompt carries the exact same prefix at a pending
// mark of its own is refused its first chunk with the [prefix_warming] marker
// and the leader's progress, retries the same chunk after a short sleep, and
// branches off the park once it stands. Wasmtime builds a fresh store per
// request, so this can only live in the shared serving context - an app-side
// global would coordinate nothing.
//
// Properties worth failing a build over:
//
//   1. The refusal is OPT-IN ({"prefix_wait": [1]} beside the first chunk).
//      Every guest built before mm36 treats a compute error as fatal, so an
//      unconditional refusal would turn a slow first chat into a failed one.
//   2. The marker is the literal the guest substring-matches, and it carries
//      "N of M tokens" for the guest's progress line.
//   3. A refused follower is left UNTOUCHED: the wait verdict returns before
//      the park copy, the pin and the plan, so its retry is a clean re-plan.
//      (A follower whose n_past had moved would feed its first chunk twice.)
//   4. Claims never outlive their owner: the context's Drop releases them,
//      and so does the end or failure of its plan.
//   5. The coordination state lives on the server, behind one leaf lock.
//
//   run: node --test test/prefix-warming.test.mjs

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const patch = fs.readFileSync(path.join(ROOT, "wasm/wasmtime-nn-ggml.patch"), "utf8");

// the patch is a diff: only ADDED lines are the shipped source
const added = patch
  .split("\n")
  .filter((l) => l.startsWith("+") && !l.startsWith("+++"))
  .map((l) => l.slice(1))
  .join("\n");

test("the refusal marker is the exact literal a guest substring-matches, with progress", () => {
  assert.match(added, /pub const PREFIX_WARMING_ERR: &str = "\[prefix_warming\]";/,
    "PREFIX_WARMING_ERR must be the [prefix_warming] tag - the guest matches on it");
  const fn = added.match(/fn prefix_warming\(done: usize, total: usize\) -> BackendError \{[\s\S]*?\n\}/);
  assert.ok(fn, "prefix_warming() must build the refusal");
  assert.match(fn[0], /\{PREFIX_WARMING_ERR\}/, "the refusal opens with the pinned marker");
  assert.match(fn[0], /\(\{done\} of \{total\} tokens\)/,
    "the refusal carries \"(N of M tokens)\" - the guest's progress line parses exactly that");
});

test("waiting is opt-in: only a caller that sent prefix_wait=1 can be refused", () => {
  assert.match(added, /nt\.name == "prefix_wait"\s*\n?\s*&& tokens_from_tensor\(&nt\.tensor\)\.map\(\|v\| v\.first\(\) == Some\(&1\)\)\.unwrap_or\(false\)/,
    "the opt-in must read the prefix_wait input and default to false");
  assert.match(added, /self\.plan_prompt\(tokens_from_tensor\(&pt\.tensor\)\?, marks, wait_ok\)\?;/,
    "plan_prompt must receive the opt-in and its refusal must abort the feed");
  const plan = added.match(/fn plan_prompt\(&mut self, prompt: Vec<i32>, marks: Vec<usize>, wait_ok: bool\)[\s\S]*?\n    \}\n/);
  assert.ok(plan, "plan_prompt(prompt, marks, wait_ok) must exist");
  const refusals = plan[0].match(/return Err\(prefix_warming\(done, total\)\)/g) || [];
  assert.equal(refusals.length, 2, "one fast-path refusal (no turn wait) and one under the turn");
  assert.match(plan[0], /if wait_ok && !marks\.is_empty\(\) \{[\s\S]*?\.blocking\(self\.claim_id, &prompt, &marks/,
    "the fast path only runs for a caller that opted in");
  assert.match(plan[0], /\.plan\(self\.claim_id, &prompt, &marks, wait_ok, Instant::now\(\)\)/,
    "the claim registration passes the opt-in through");
});

test("a refused follower is left untouched: the wait returns before the copy, the pin and the plan", () => {
  const plan = added.match(/fn plan_prompt\(&mut self, prompt: Vec<i32>, marks: Vec<usize>, wait_ok: bool\)[\s\S]*?\n    \}\n/)[0];
  const wait = plan.indexOf("ClaimVerdict::Wait { done, total } = verdict");
  const copy = plan.indexOf("ell_seq_copy(self.server.raw, p.seq_id, self.seq_id)");
  const pin = plan.indexOf("self.park_pin = Some(p.pin.clone())");
  const set = plan.indexOf("self.plan = Some(PromptPlan");
  const npast = plan.indexOf("self.n_past = covered as i32");
  assert.ok(wait > 0 && copy > wait && pin > wait && set > wait && npast > wait,
    "every mutation of the context must come after the wait verdict");
  assert.ok(!/self\.n_past = [^;]*;[\s\S]*?ClaimVerdict::Wait/.test(plan),
    "n_past must not move before a possible refusal");
});

test("claims are settled as marks are crossed and never outlive their owner", () => {
  assert.match(added, /self\.fork_park\(true\);\s*\n\s*\/\/ mm36[^\n]*\n\s*self\.claim_resolve\(m\);/,
    "each crossed mark resolves its claim right after fork_park, parked or not");
  assert.match(added, /self\.claim_progress\(\);/, "the leader reports progress after every chunk");
  const drop = added.match(/impl Drop for GgmlExecutionContext \{[\s\S]*?\n\}/);
  assert.ok(drop, "the context's Drop must exist");
  assert.match(drop[0], /self\.server\.claims\.lock\(\)\.unwrap\(\)\.release_all\(self\.claim_id\);/,
    "Drop releases every claim the context held - an abandoned warm-up strands nobody");
  const q = drop[0].indexOf("let mut q = self.server.q.lock()");
  const rel = drop[0].indexOf("release_all(self.claim_id)");
  assert.ok(rel > 0 && rel < q, "claims (a leaf lock) is taken ALONE in Drop, before the queue lock");
  assert.match(added, /if had_plan && \(r\.is_err\(\) \|\| self\.plan\.is_none\(\)\) \{\s*\n\s*self\.server\.claims\.lock\(\)\.unwrap\(\)\.release_all\(self\.claim_id\);/,
    "a plan that ends or fails releases what is left");
});

test("the coordination lives on the shared server, keyed by a stable per-context id", () => {
  assert.match(added, /struct GgmlServer \{[\s\S]*?claims: Mutex<PrefixClaims>,/,
    "GgmlServer carries the claims table - one per model, shared by every request");
  assert.match(added, /struct GgmlExecutionContext \{[\s\S]*?claim_id: u64,/,
    "the context's claim identity is its own field, not seq_id (park forks swap seq_id)");
  assert.match(added, /claim_id: CLAIM_IDS\.fetch_add\(1, Ordering::Relaxed\) \+ 1,/,
    "claim ids are minted once, at init");
  assert.match(added, /pub mod prefix_claims;/, "the pure claims module is registered");
  // caps 14..16: protocol, live claims, standing parks - what warm_one reads
  assert.match(added, /prefix_slots\(\) as i32,\s*\n(\s*\/\/[^\n]*\n)+\s*1,\s*\n\s*warming,\s*\n\s*parks,\s*\n\s*\];/,
    "caps must end with [protocol=1, live claims, standing parks] right after prefix_slots");
});

test("the pure claims table has its own tests and the staleness knob is documented", () => {
  const mod = added.match(/pub struct PrefixClaims \{[\s\S]*?\n\}/);
  assert.ok(mod, "PrefixClaims must exist");
  assert.match(added, /#\[cfg\(test\)\]\s*\nmod tests \{[\s\S]*?first_caller_leads_and_matching_follower_waits_with_progress/,
    "the claims module carries the leader/follower unit tests");
  assert.match(added, /ENCLAVE_GGML_PREFIX_CLAIM_STALE_S/, "the stale-leader window is an env knob");
});
