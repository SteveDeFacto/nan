// The dealer loop (shielded/dealer/dealer-loop.py) derives a pVM's seed exactly
// as the relay does, and plans the bank correctly: chunk-aligned ranges that
// cover [mark, mark+ahead) minus what is already there, spent shipments pruned.
//
//   run: node --test test/pads-dealer-loop.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { fileURLToPath } from "node:url";
import { deriveSeed } from "../relay/pads.mjs";

const repo = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const loop = path.join(repo, "shielded", "dealer", "dealer-loop.py");
const run = (...args) => JSON.parse(execFileSync("python3", [loop, ...args], { encoding: "utf8", timeout: 60_000 }).trim());

test("the dealer derives the seed the relay derives", () => {
  const master = randomBytes(32), keyFp = randomBytes(32).toString("hex");
  const py = run("--derive-only", "--master", master.toString("hex"), "--keyfp", keyFp, "--epoch", "1");
  const js = deriveSeed(master, keyFp, 1);
  assert.equal(py.seed, js.seed.toString("hex"));
  assert.equal(py.seed_id, js.seed_id);
});

test("the plan covers the window ahead of the mark in aligned chunks and prunes the spent", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "bank-"));
  const seed_id = randomBytes(16).toString("hex");
  for (const [i0, c] of [[0, 64], [64, 64], [128, 64]]) fs.writeFileSync(path.join(dir, `${seed_id}-${i0}-${c}.pads`), "x");
  fs.writeFileSync(path.join(dir, `${randomBytes(16).toString("hex")}-0-64.pads`), "x");   // another seed's: untouched
  const p = run("--plan-only", "--out", dir, "--seed-id", seed_id, "--mark", "150", "--ahead", "256", "--chunk", "64");
  assert.deepEqual(p.mint, [[192, 64], [256, 64], [320, 64], [384, 64]], "128..192 is already there; 150+256=406 needs up to 448");
  assert.deepEqual(p.prune.map((x) => path.basename(x)), [`${seed_id}-0-64.pads`, `${seed_id}-64-64.pads`]);
  const q = run("--plan-only", "--out", dir, "--seed-id", seed_id, "--mark", "0", "--ahead", "128", "--chunk", "64");
  assert.deepEqual(q.mint, []); assert.deepEqual(q.prune, []);
});
