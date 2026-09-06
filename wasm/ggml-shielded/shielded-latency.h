#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>

// Scheduling telemetry only: no activation values or verification state.
// A prefill exchange has more rows than a decode exchange. Comparing their
// timings can strand a healthy GPU on the CPU path after a short warmup.
struct sh_contention {
    struct latency { double ewma = 0, best = 0, expect = 0; uint64_t n = 0; };
    std::map<std::pair<std::string, int>, latency> samples;
    bool contended = false;
    uint64_t events = 0, seen = 0;
    double slow_frac = 0;
    int ok_streak = 0;

    // Returns 1 on fallback, -1 on recovery, otherwise 0. The absolute idle
    // estimate allows one weight pass per row: field GEMM work and response
    // bytes grow with rows even when the GPU reuses some weight reads.
    int note(const std::string &group, int rows, double us, double weight_bytes,
             bool probe, double factor = 3.0, double delta_us = 200.0,
             double absolute_factor = 8.0) {
        if (rows < 1 || us <= 0 || factor <= 0) return 0;
        auto &l = samples[{group, rows}];
        l.ewma = l.n ? 0.9 * l.ewma + 0.1 * us : us;
        l.n++;
        if (l.expect <= 0) l.expect = 60.0 + rows * weight_bytes / 400e3;
        if (l.n >= 20 && (l.best == 0 || l.ewma < l.best)) l.best = l.ewma;
        if (l.n < 20) return 0;
        const bool slow = (l.best > 0 && l.ewma > factor * l.best && l.ewma > l.best + delta_us)
                       || l.ewma > absolute_factor * l.expect;
        if (!contended) {
            // One fast group must not hide sustained contention elsewhere.
            slow_frac = 0.97 * slow_frac + 0.03 * (slow ? 1.0 : 0.0);
            seen++;
            if (seen >= 98 && slow_frac > 0.8) {
                contended = true; events++; slow_frac = 0; ok_streak = 0;
                return 1;
            }
        } else if (probe) {
            // A best recorded on an already busy GPU is not an idle baseline.
            const bool fine = l.ewma < 3.0 * l.expect
                           || (l.best > 0 && l.best < absolute_factor * l.expect && l.ewma < 1.5 * l.best);
            ok_streak = fine ? ok_streak + 1 : 0;
            if (ok_streak >= 50) {
                contended = false; ok_streak = 0;
                return -1;
            }
        }
        return 0;
    }
};
