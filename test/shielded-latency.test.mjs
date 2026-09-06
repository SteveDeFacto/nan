import test from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

test("shielded contention separates batch widths and still detects and recovers from busy GPUs", () => {
  const dir = mkdtempSync(join(tmpdir(), "shielded-latency-"));
  try {
    writeFileSync(join(dir, "test.cpp"), `
#include "shielded-latency.h"
#include <cassert>
int main() {
    // Reproduce a short decode warmup followed by 8-row prefill on the same
    // 27B FFN group. 1983us is healthy at eight rows, but over 3x 462us.
    sh_contention mixed;
    for (int i = 0; i < 160; ++i) mixed.note("ffn", 1, 462, 89e6, true);
    for (int i = 0; i < 320; ++i) mixed.note("ffn", 8, 1983, 89e6, true);
    assert(!mixed.contended && mixed.events == 0);
    for (int i = 0; i < 160; ++i) {
        mixed.note("ffn", 1, 462, 89e6, true);
        mixed.note("ffn", 8, 1983, 89e6, true);
    }
    assert(!mixed.contended);

    // Even without a one-row baseline, an ordinary wider batch must not
    // trip the absolute one-row estimate (4MB, 900us vs its old 560us).
    sh_contention wide;
    for (int i = 0; i < 200; ++i) wide.note("qkv", 8, 900, 4e6, true);
    assert(!wide.contended);

    // Sustained slowdown within the SAME width still trips, even when the
    // larger shape's absolute threshold is not reached.
    for (int i = 0; i < 240; ++i) mixed.note("ffn", 8, 9000, 89e6, true);
    assert(mixed.contended && mixed.events == 1);
    for (int i = 0; i < 240; ++i) mixed.note("ffn", 8, 1983, 89e6, true);
    assert(!mixed.contended);

    // A card busy from its first sample is caught without a healthy best.
    sh_contention cold;
    for (int i = 0; i < 200; ++i) cold.note("qkv", 1, 1240, 4e6, true);
    assert(cold.contended && cold.events == 1);
    for (int i = 0; i < 60; ++i) cold.note("qkv", 1, 1200, 4e6, true);
    assert(cold.contended);
    for (int i = 0; i < 200; ++i) cold.note("qkv", 1, 160, 4e6, true);
    assert(!cold.contended);

    // An isolated scheduler stall must not exile an otherwise healthy GPU.
    sh_contention burst;
    for (int i = 0; i < 160; ++i) burst.note("qkv", 1, 160, 4e6, true);
    burst.note("qkv", 1, 5000, 4e6, true);
    for (int i = 0; i < 160; ++i) burst.note("qkv", 1, 160, 4e6, true);
    assert(!burst.contended);
}
`);
    const includes = fileURLToPath(new URL("../wasm/ggml-shielded", import.meta.url));
    execFileSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-I" + includes,
      join(dir, "test.cpp"), "-o", join(dir, "test")], { timeout: 30_000 });
    execFileSync(join(dir, "test"), { timeout: 5_000 });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
