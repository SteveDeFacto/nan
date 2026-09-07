// The two refill knobs a deployment config may set for the shielded engine,
// pinned the way the other manager->engine env seams are: the config key, the
// engine env it becomes, and the bounds (which match the engine's own clamps
// in wasm/ggml-shielded/shielded-tee.c: SHIELDED_REFILL_BATCH 1..64,
// SHIELDED_POOL_DEPTH -1..4096 with 16 the smallest useful ring).
//
//   run: node --test test/shielded-refill-knobs.test.mjs

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const manager = fs.readFileSync(path.join(ROOT, "wasm/wasm_manager.py"), "utf8");
const tee = fs.readFileSync(path.join(ROOT, "wasm/ggml-shielded/shielded-tee.c"), "utf8");
const backend = fs.readFileSync(path.join(ROOT, "wasm/ggml-shielded/ggml-shielded.cpp"), "utf8");

test("nnShieldedRefillBatch reaches the engine as SHIELDED_REFILL_BATCH within the engine's bounds", () => {
  const m = manager.match(/_nn_cfg_int\(enclave_config, "nnShieldedRefillBatch", (\d+), (\d+)\)[\s\S]*?env\["SHIELDED_REFILL_BATCH"\] = str\(rb\)/);
  assert.ok(m, "the manager must map nnShieldedRefillBatch to SHIELDED_REFILL_BATCH");
  const e = tee.match(/env_int\("SHIELDED_REFILL_BATCH", \d+, (\d+), (\d+)\)/);
  assert.ok(e, "the engine must read SHIELDED_REFILL_BATCH with bounds");
  assert.equal(m[1], e[1]); assert.equal(m[2], e[2]);
});

test("nnShieldedPoolDepth reaches the engine as SHIELDED_POOL_DEPTH, never below a useful ring", () => {
  const m = manager.match(/_nn_cfg_int\(enclave_config, "nnShieldedPoolDepth", (\d+), (\d+)\)[\s\S]*?env\["SHIELDED_POOL_DEPTH"\] = str\(pd\)/);
  assert.ok(m, "the manager must map nnShieldedPoolDepth to SHIELDED_POOL_DEPTH");
  const e = tee.match(/env_int\("SHIELDED_POOL_DEPTH", -1, -1, (\d+)\)/);
  assert.ok(e, "the engine must read SHIELDED_POOL_DEPTH");
  assert.equal(m[2], e[1], "the upper bound is the engine's");
  assert.ok(Number(m[1]) >= 16, "a ring shallower than 16 starves speculative decode");
});

test("nnShieldedRefillThreads reaches the engine as the SHIELDED_REFILL_THREADS total the backend splits over cards", () => {
  const m = manager.match(/_nn_cfg_int\(enclave_config, "nnShieldedRefillThreads", (\d+), (\d+)\)[\s\S]*?env\["SHIELDED_REFILL_THREADS"\] = str\(rt\)/);
  assert.ok(m, "the manager must map nnShieldedRefillThreads to SHIELDED_REFILL_THREADS");
  const e = backend.match(/sh_env_int\("SHIELDED_REFILL_THREADS",[\s\S]*?std::min\((\d+), threads\)/);
  assert.ok(e, "the backend must read SHIELDED_REFILL_THREADS as the process-wide total");
  assert.equal(m[2], e[1], "the upper bound is the backend's");
  assert.ok(Number(m[1]) >= 1, "zero refill threads would generate every pad on the request path");
  assert.match(backend, /refill_threads = threads \/ \(int\)parsed\.size\(\)/, "the total is divided over the card links");
});

test("dealt-pad knobs reach the engine together, seed and sk as substituted secrets, never partially", () => {
  for (const [key, env] of [["nnShieldedPadSource", "SHIELDED_PAD_SOURCE"], ["nnShieldedPadSeed", "SHIELDED_PAD_SEED"], ["nnShieldedPadSeedId", "SHIELDED_PAD_SEED_ID"],
                            ["nnShieldedPadSk", "SHIELDED_PAD_SK"], ["nnShieldedPadLedger", "SHIELDED_PAD_LEDGER"], ["nnShieldedPadModelDigest", "SHIELDED_PAD_MODEL_DIGEST"],
                            ["nnShieldedPadWindow", "SHIELDED_PAD_WINDOW"], ["nnShieldedPadCheck", "SHIELDED_PAD_CHECK"],
                            ["nnShieldedPadCache", "SHIELDED_PAD_CACHE"], ["nnShieldedPadCacheMb", "SHIELDED_PAD_CACHE_MB"], ["nnShieldedPadWindowUrl", "SHIELDED_PAD_WINDOW_URL"],
                            ["nnShieldedPadLedgerPk", "SHIELDED_PAD_LEDGER_PK"], ["nnShieldedPadPrune", "SHIELDED_PAD_PRUNE"]]) {
    assert.ok(manager.includes(`"${key}"`) && manager.includes(`"${env}"`), `${key} -> ${env}`);   // seed/seedId/sk go through one loop of (key, env) pairs
    assert.ok(tee.includes(`"${env}"`), `the engine reads ${env}`);
  }
  // the engine refuses a partial set rather than minting: the manager must not paper over that
  assert.match(tee, /Dealt mode: all four are required together/, "engine refuses a partial dealt env");
  // a window agent without the platform's ledger key would let the agent sign its own windows: refused as a pair, on both sides
  assert.match(tee, /a window URL without the ledger key is refused/, "engine refuses SHIELDED_PAD_WINDOW_URL without SHIELDED_PAD_LEDGER_PK");
  assert.ok(manager.includes('wu.startswith("http://") and isinstance(lk, str) and len(lk) == 64'), "manager sets the window URL only together with a 64-hex ledger key");
  // a bank without explicit keys takes this box's own identity from the guest agent's bootstrap file, never a partial one
  assert.ok(manager.includes('if env["SHIELDED_PAD_SOURCE"].startswith("http://") and "SHIELDED_PAD_SEED" not in env:'), "bootstrap only fills a bank source without explicit keys");
  assert.ok(manager.includes('"METAL_PADS_BOOTSTRAP", "/run/enclave/pads/bootstrap.json"'), "the bootstrap path is the agent's");
  for (const f of ['"SHIELDED_PAD_SEED": b["seed"]', '"SHIELDED_PAD_SEED_ID": b["seed_id"]', '"SHIELDED_PAD_SK": b["sk"]', '"SHIELDED_PAD_LEDGER_PK": b["ledger_pk"]', '"SHIELDED_PAD_WINDOW_URL": b["window_url"]', 'out["SHIELDED_PAD_AGENT_TOKEN"] = b["token"]'])
    assert.ok(manager.includes(f), f);
  const agent = fs.readFileSync(path.join(ROOT, "metal/guest/agent.mjs"), "utf8");
  for (const f of ["seed_id: r.body.seed_id", "seed: seed.toString('hex')", "sk: padSkHex", "ledger_pk: key.body.key", "window_url: `http://127.0.0.1:${RAD_PORT}/pads/window`", "bank_url: `http://127.0.0.1:${RAD_PORT}/pads/shipments`", "token: PADS_TOKEN"])
    assert.ok(agent.includes(f), "agent writes " + f);
  assert.ok(tee.includes('getenv("SHIELDED_PAD_AGENT_TOKEN")') && agent.includes("!padsAuthorized(req)"), "the engine sends the token; the agent refuses without it");
  // "platform" as the source = the store through the agent's proxy, and never a half-configured link
  assert.ok(manager.includes('if env["SHIELDED_PAD_SOURCE"] == "platform":'), "platform keyword");
  assert.ok(manager.includes('env.pop("SHIELDED_PAD_SOURCE")'), "no identity -> no dealt env at all");
  // secrets are substituted before this block sees the config
  assert.ok(manager.indexOf("enclave_config = _subst_secrets(enclave_config, secrets)") < manager.indexOf('env["SHIELDED_PAD_SOURCE"]'), "secret refs substituted before the pad env is built");
});
