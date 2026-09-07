// The metal guest image compiles wasm/ggml-shielded from source with its own
// unit list (metal/build-image.mjs) that mirrors the Makefile. The two drift
// silently: a source added to CORE_SRC but not to the units links into a
// libggml-shielded.so with undefined symbols that fails only at dlopen inside
// the enclave (the shielded backend then simply does not exist there). This
// pins the lists together and the link flag that turns that into a build error.
//   run: node --test test/metal-shielded-build.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const makefile = fs.readFileSync(path.join(root, "wasm/ggml-shielded/Makefile"), "utf8");
const builder = fs.readFileSync(path.join(root, "metal/build-image.mjs"), "utf8");

test("every source the Makefile links into libggml-shielded.so is a unit of the image build", () => {
  const core = makefile.match(/^CORE_SRC\s*:=\s*(.+)$/m);
  assert.ok(core, "CORE_SRC in the Makefile");
  const srcs = core[1].trim().split(/\s+/);
  assert.ok(srcs.includes("shielded-tee.c") && srcs.includes("shielded-pads.c"), "sanity: " + srcs.join(" "));
  const units = [...builder.matchAll(/\{\s*src:\s*'([^']+)'/g)].map((m) => m[1]);
  for (const s of srcs) assert.ok(units.includes(s), `${s} is in CORE_SRC but not in metal/build-image.mjs units`);
  for (const s of ["shielded-simd.c", "ggml-shielded.cpp"]) assert.ok(units.includes(s), s);
  // and nothing the Makefile does not know (a unit that exists only in the image)
  for (const u of units) assert.ok(srcs.includes(u) || u === "shielded-simd.c" || u === "ggml-shielded.cpp", `${u} is an image unit the Makefile never links`);
});

test("the image build refuses undefined symbols in the shielded library", () => {
  assert.match(builder, /-Wl,--no-undefined/, "the .so link must carry -Wl,--no-undefined");
  // the dealer-only path stays out of the image: no SHIELDED_DEALER_MODE flag in any unit
  assert.doesNotMatch(builder, /SHIELDED_DEALER_MODE/, "the image must never carry the zero-pad mint path");
});
