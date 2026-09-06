#include "ggml-shielded.h"
#include "shielded-latency.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

extern "C" {
#include "shielded-field.h"
#include "shielded-tee.h"
}

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#define SH_LOG(...) do { if (sh_verbose()) fprintf(stderr, "[shielded] " __VA_ARGS__); } while (0)

static double sh_now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool sh_verbose() {
    static int v = -1;
    if (v < 0) { const char *e = getenv("SHIELDED_VERBOSE"); v = (e && *e && strcmp(e, "0")) ? 1 : 0; }
    return v != 0;
}
static int sh_env_int(const char *name, int dflt) {
    const char *e = getenv(name);
    return (e && *e) ? atoi(e) : dflt;
}

/* A process-wide correction to every site's calibrated activation exponent.
 * NOT per-request: a constant, exactly as the calibrated exponent is --
 * adapting it to the activation in hand would buy field headroom by leaking
 * activation magnitude, and is refused.
 *
 * It defaults to 0 because the calibration files are now produced by
 * shielded-calib, which encodes each weight with the very sh_prepare_weight_rows
 * this backend calls -- per OUTPUT COLUMN -- so the exponent it chooses is for
 * the product the runtime actually forms. The knob exists for history and for
 * experiments: the first calibrations came from calibrate.py, which measured
 * against tee.py's PER-TENSOR encoding, where most columns held tiny w_fixed and
 * tiny products; against per-column weights those exponents were 1-6 bits too
 * generous (Qwen2.5-0.5B, per site) and a blanket -5, sized for the worst site,
 * was what made them fit. Per-site calibration gives the median site 3 of those
 * bits back. A calibrate.py file still needs SHIELDED_AF_DELTA=-5.
 *
 * Getting this wrong does not silently corrupt anything -- the Freivalds check
 * runs over the integers and catches the wrap -- but it fails EVERY request,
 * which is how the -5 once shipped: verified locally with the flag set, deployed
 * without it, and the tenant died in llama_decode with the wrap detector doing
 * its job. */
static int sh_af_delta_env() {
    static int d = INT32_MIN;
    if (d == INT32_MIN) d = sh_env_int("SHIELDED_AF_DELTA", 0);
    return d;
}
struct sh_state;
static int sh_af_delta(const sh_state &s);

/* --------------------------------------------------------------------------
 * Placement policy.
 *
 * Offload is a round trip, and a round trip has a floor cost that no amount of
 * GPU speed lowers. Two consequences, both tunable, neither per-request:
 *
 *  SHIELDED_MIN_MACS  A matmul below this many multiply-adds is cheaper to do
 *                     on the CPU inside the enclave than to ship. Qwen2.5-0.5B's
 *                     q/k/v/o projections (0.1-0.8 MMAC) sit under the default;
 *                     its FFN and lm_head (4.4-136 MMAC) sit above it.
 *  SHIELDED_MAX_M     Batches wider than this stay in the enclave. Refill costs
 *                     the TEE three residue planes of work per offloaded MAC,
 *                     so a prefill-sized batch is strictly cheaper computed in
 *                     the clear, in the enclave, once. Offload is a DECODE
 *                     accelerator: it removes the weight-bandwidth term from the
 *                     latency chain, and that is the only term it can remove.
 * ----------------------------------------------------------------------- */
static int64_t sh_min_macs() { static int64_t v = -1; if (v < 0) v = sh_env_int("SHIELDED_MIN_MACS", 2000000); return v; }
static int     sh_max_m()    { static int v = -1;     if (v < 0) v = sh_env_int("SHIELDED_MAX_M", 8); return v; }

/* q8_0, as ggml stores it: one fp16 scale then 32 quants, per block, per row. */
struct sh_block_q8_0 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(sh_block_q8_0) == 34, "unexpected q8_0 block layout");

/* --------------------------------------------------------------------------
 * Calibration.
 *
 * Two public, offline constants per site: the activation exponent and the
 * outlier channel set. Both are properties of the PUBLIC weights, calibrated on
 * public text and shipped like an imatrix, so neither leaks anything about a
 * request -- and both must be per-model constants rather than adapted to the
 * activation in hand, since an adaptive exponent would leak activation
 * magnitude. A site with no calibration is not offloaded at all: supports_op
 * says no and ggml_backend_sched quietly runs it on the CPU, in the enclave.
 * ----------------------------------------------------------------------- */
struct sh_calib_site {
    int act_frac = 0;
    std::vector<int64_t> outliers;
};

/* q/k/v come from one attn_norm and gate/up from one ffn_norm, so they share an
 * activation -- and therefore share one exponent, one outlier set and, at run
 * time, ONE PAD and ONE EXCHANGE. That is not a bandwidth optimisation: masking
 * the same x three times under three pads would hand the adversary three
 * encryptions of one value for no benefit.
 *
 * qwen35's gated-deltanet layers feed FOUR linears from one norm output:
 * attn_qkv, attn_gate, ssm_alpha and ssm_beta all read the same tensor
 * (shielded-calib reports it from the graph). Without the last three rows the
 * backend exchanged attn_qkv and attn_gate as two groups, i.e. one plaintext
 * under two pads and one exchange per layer more than needed. A name that
 * matches here but whose model has no attn_qkv simply finds no calibration and
 * stays in the enclave. */
static std::string sh_group_key(const std::string &name) {
    static const std::pair<const char *, const char *> members[] = {
        { "attn_k",    "attn_q" },   { "attn_v",    "attn_q" },
        { "ffn_up",    "ffn_gate" },
        { "attn_gate", "attn_qkv" }, { "ssm_alpha", "attn_qkv" }, { "ssm_beta", "attn_qkv" },
        { "ssm_ba",    "attn_qkv" },   /* qwen3next: the same norm output */
    };
    for (const auto &m : members) {
        const size_t p = name.find(m.first);
        if (p != std::string::npos) {
            std::string out = name;
            out.replace(p, strlen(m.first), m.second);
            return out;
        }
    }
    return name;
}

struct sh_state {
    std::mutex mu;
    std::string host = "127.0.0.1";
    int port = 9500, vsock_port = -1, refill_threads = -1;
    uint64_t reserve_bytes = 0;
    bool explicit_link = false;
    std::string calib_path;
    bool configured = false;
    bool calib_loaded = false;
    /* "# shielded-calib N" on the file's first line. 1 = calibrate.py's
     * exponents, chosen against tee.py's per-TENSOR weight encoding, which
     * need the historical -5 against this backend's per-column products;
     * 2 = shielded-calib's, chosen against the very encoding this backend
     * applies. A file with no header is 1. The env delta is added on top. */
    int calib_version = 1;
    bool link_failed = false;
    /* Reconnect policy after a transport failure: the link is retried at the
     * next graph once `link_retry_at` has passed, with the wait doubling from
     * 1 s to 60 s. Until then every claimed matmul is computed in the enclave,
     * exactly and slowly. Before this the first socket error was PERMANENT:
     * a worker restart on the host left the tenant on the int64 path (6 tok/s
     * on the 0.5B) until the tenant itself was restarted. */
    double link_retry_at = 0, link_backoff_ms = 1000;
    std::map<std::string, sh_calib_site> calib;

    /* Reservation-aware placement. SHIELDED_RESERVE_BYTES is this tenant's slice
     * of the card (its GPU share x the card budget) - the same figure the worker
     * caps the link's device memory at. We place calibrated weights until that
     * budget is full and leave the rest on CPU, so a model LARGER than the card
     * offloads its first reservation-worth of sites cleanly. Without it the
     * backend kept handing weights to the worker past its budget, the worker
     * answered "allocation exceeds device memory" with a VIOLATION that CLOSES
     * the link, and the tenant lost ALL offload and reconnected into a thrash
     * loop. 0 = no reservation (a 4-byte HELLO): uncapped, exactly as before. */
    int64_t reserve_cap = 0;         /* device-byte budget for offloaded weights, 0 = uncapped */
    int64_t device_bytes = 0;        /* weight+activation device bytes we have placed */
    bool    budget_full_logged = false;

    sh_link *link = nullptr;
    /* Weight tensor name -> everything needed to run and to check it. Map nodes
     * never move, so the link may borrow `w` for its lifetime. */
    struct entry {
        int node = -1;
        std::string name;
        std::vector<int> f_w;           /* one exponent per output column */
        int64_t K = 0, N = 0;
        const sh_calib_site *site = nullptr;
        std::string group;
        std::vector<int8_t> w;          /* (N,K): THE encoding, borrowed by the link */
        std::vector<int8_t> out_cols;   /* nout x N: the outlier channels' weights, for the TEE-side term */
        std::vector<float> inv;         /* per-column descale 2^-(af + f_w[j]) */
    };
    std::map<std::string, entry> weights;
    std::map<std::string, int> group_first;   /* group key -> first node in it */
    /* Every registered weight of each group, in registration order.
     *
     * ggml_backend_sched cuts the graph wherever the backend changes, and a
     * CPU op between two members of a group -- Qwen3's q_norm between attn_q
     * and attn_k -- puts them in different splits, so graph_compute sees the
     * group one member at a time. Exchanged naively that is one round trip and
     * one pad PER MEMBER for the same activation (Qwen3-4B: 181 exchanges per
     * token instead of 145). Instead, an exchange for a partial group asks the
     * worker for the WHOLE group's products and keeps the invisible members'
     * results here; the later split is served from that cache when its
     * activation is byte-identical to the one exchanged. One plaintext, one
     * pad, one round trip -- rule 2 held by construction across splits too.
     * Nothing new crosses the wire: the extra products are functions of the
     * same masked planes and the public weights, verified like every other. */
    std::map<std::string, std::vector<std::string>> group_members;
    struct gcache { int32_t m = 0; std::vector<int64_t> x; std::map<std::string, std::vector<int64_t>> y; };
    std::map<std::string, gcache> completion;
    std::vector<entry *> xents;               /* the exchange set of the current node */
    uint64_t completed = 0, served = 0;       /* group completions issued / members served from one */

    // Compare wire timings only within the same group AND batch width.
    // Keep one probe group offloaded during fallback to notice recovery.
    sh_contention contention;
    std::string probe_group;
    uint64_t contended_graphs = 0;
    std::set<std::string> refused;            /* names that failed registration, said once */
    bool dirty = false;                       /* new weights since the last start() */

    uint64_t offloaded_nodes = 0, local_nodes = 0, macs = 0, verify_fail = 0, exchanges = 0;
    double t_encode = 0, t_link = 0, t_post = 0, t_graph = 0;

    /* graph_compute scratch, kept across calls: resize() never shrinks a
     * vector's capacity, so after the first token these are plain pointer
     * arithmetic. Bounded by max_m x Kmax (x) and max_m x sum N of the widest
     * group (y) -- for the 0.5B that is 8 x 151936 int64 = 9.7 MB at most.
     * Before this every exchange constructed and destroyed x_gpu, x_tee, one
     * vector per member and the bookkeeping vectors around them. */
    std::vector<int64_t> x_gpu, x_tee, ys;
    std::vector<int64_t *> yp;
    std::vector<int> nodes;
    std::vector<char> done;
    std::vector<ggml_tensor *> members;
    std::vector<entry *> ents;

    /* The link borrows `weights[*].w` for its lifetime and its refill threads
     * read them from the background. Static destruction order would free the
     * map first and let a thread mid-refill read freed memory (a real segfault
     * at process exit, seen on bench-batch); close the link -- which joins the
     * threads -- before any member goes. */
    ~sh_state() { if (link) { sh_link_close(link); link = nullptr; } }
};

static sh_state &sh_get() { static sh_state s; return s; }

/* One ACCEL facade, independent verified links. Only public weight metadata
 * drives placement. Nonlinear operations, KV, pads and verification stay in the
 * enclave; activations are never copied in the clear between host GPUs. */
struct sh_pool {
    std::mutex mu;
    bool initialized = false, invalid = false;
    std::vector<std::unique_ptr<sh_state>> extra;
    std::vector<sh_state *> cards;
    std::map<std::string, ggml_tensor> pending;
    std::map<std::string, int> owners;       // activation group -> card, -1 = CPU
    std::map<std::string, int> layers;
};
static sh_pool &sh_pool_get() { static sh_pool p; return p; }
static void sh_pool_init(sh_pool &p);
static int sh_owner(sh_pool &p, const ggml_tensor *w) {
    auto it = p.owners.find(sh_group_key(ggml_get_name(w)));
    return it == p.owners.end() ? -1 : it->second;
}
static void sh_plan(sh_pool &p);

/* The correction every site's calibrated exponent gets: the file format's own
 * (see calib_version) plus the process-wide SHIELDED_AF_DELTA. */
static int sh_af_delta(const sh_state &s) { return (s.calib_version == 1 ? -5 : 0) + sh_af_delta_env(); }

void ggml_backend_shielded_configure(const char *host, int port, const char *calib_path) {
    sh_state &s = sh_get();
    std::lock_guard<std::mutex> lk(s.mu);
    if (host && *host) s.host = host;
    if (port > 0) s.port = port;
    if (calib_path && *calib_path) { s.calib_path = calib_path; s.calib_loaded = false; }
    s.configured = true;
}

int ggml_backend_shielded_pool_version(void) { return 1; }

extern "C" double sh_prof[8];
void ggml_backend_shielded_stats(uint64_t *off, uint64_t *loc, uint64_t *macs, uint64_t *vf) {
    sh_pool &p = sh_pool_get();
    std::lock_guard<std::mutex> lock(p.mu);
    sh_pool_init(p);
    if (off) *off = 0;
    if (loc) *loc = 0;
    if (macs) *macs = 0;
    if (vf) *vf = 0;
    for (size_t card = 0; card < p.cards.size(); card++) {
    sh_state &s = *p.cards[card];
    std::lock_guard<std::mutex> lk(s.mu);
    if (getenv("SHIELDED_PROFILE")) {
        fprintf(stderr, "[shielded] card %zu %s:%d device_bytes=%lld reserve_cap=%lld\n",
                card, s.host.c_str(), s.port, (long long)s.device_bytes, (long long)s.reserve_cap);
        uint64_t used = 0, missed = 0;
        if (s.link) sh_link_pool_stats(s.link, &used, &missed);
        fprintf(stderr, "[shielded] profile: exchanges=%llu nodes=%llu (completions=%llu served=%llu) | link: mask=%.1fms wire=%.1fms "
                        "refill-on-path=%.1fms unmask+lhs=%.1fms rhs=%.1fms total=%.1fms | backend: encode=%.1fms "
                        "post=%.1fms graph_compute=%.1fms | pads used=%llu missed=%llu | contended=%d events=%llu | simd=%s refill_threads=%d\n",
                (unsigned long long)s.exchanges, (unsigned long long)s.offloaded_nodes,
                (unsigned long long)s.completed, (unsigned long long)s.served,
                sh_prof[0], sh_prof[1], sh_prof[2], sh_prof[3], sh_prof[4], s.t_link,
                s.t_encode, s.t_post, s.t_graph, (unsigned long long)used, (unsigned long long)missed,
                (int)s.contention.contended, (unsigned long long)s.contention.events,
                sh_link_simd()->name, s.link ? sh_link_refill_threads(s.link) : 0);
        // Identify the groups delaying GPU submission. Aggregate misses alone
        // cannot distinguish a large output head from an undersized whole pool.
        struct stalled_group { std::string name; uint64_t used, missed; double ms; };
        std::vector<stalled_group> stalls;
        for (const auto &group : s.group_members) {
            if (!s.link || group.second.empty()) continue;
            auto e = s.weights.find(group.second.front());
            if (e == s.weights.end()) continue;
            stalled_group g{group.first, 0, 0, 0};
            sh_link_node_pool_stats(s.link, e->second.node, &g.used, &g.missed, &g.ms);
            if (g.missed) stalls.push_back(std::move(g));
        }
        std::sort(stalls.begin(), stalls.end(), [](const stalled_group &a, const stalled_group &b) {
            return a.ms > b.ms;
        });
        for (size_t i = 0; i < std::min<size_t>(3, stalls.size()); i++) {
            const auto &g = stalls[i];
            fprintf(stderr, "[shielded] profile: card=%zu group=%s pads=%llu missed=%llu on-path=%.1fms local_nodes=%llu\n",
                    card, g.name.c_str(), (unsigned long long)g.used, (unsigned long long)g.missed,
                    g.ms, (unsigned long long)s.local_nodes);
        }
    }
    if (off)  *off  += s.offloaded_nodes;
    if (loc)  *loc  += s.local_nodes;
    if (macs) *macs += s.macs;
    if (vf)   *vf   += s.verify_fail;
    }
}

/* Read the environment once, so enabling the tier is launch configuration rather
 * than an app-visible API -- existing catalog guests keep their wasi-nn contract
 * unchanged, which is the point of putting the split here at all. An explicit
 * ggml_backend_shielded_configure() still wins. */
static void sh_env_defaults(sh_state &s) {
    if (s.configured) return;
    s.configured = true;
    if (const char *h = getenv("SHIELDED_HOST")) if (*h) s.host = h;
    if (const char *p = getenv("SHIELDED_PORT")) if (*p) { const int v = atoi(p); if (v > 0) s.port = v; }
    /* The card slice this tenant reserved (shielded-tee.c sends the same number
     * to the worker at HELLO). Keep headroom below it for the activation buffers
     * and field-GEMM scratch the worker also charges to the link - weights are
     * the dominant term but not the only one. SHIELDED_WEIGHT_BUDGET_FRAC tunes
     * the headroom; 0.90 leaves 10%. */
    if (const char *r = getenv("SHIELDED_RESERVE_BYTES")) {
        char *end = nullptr; unsigned long long v = strtoull(r, &end, 10);
        if (end && *end == 0 && v > 0) {
            const char *fe = getenv("SHIELDED_WEIGHT_BUDGET_FRAC");
            double frac = (fe && *fe) ? atof(fe) : 0.90;
            if (!(frac > 0.1 && frac <= 1.0)) frac = 0.90;
            s.reserve_cap = (int64_t)((double)v * frac);
        }
    }
}

/* Manager-generated records: host|tcp-port|vsock-port|reservation-bytes,
 * one per line. Strict parsing is fail-closed: a malformed pool must never
 * turn a paid slice into the legacy uncapped single-worker connection. */
static void sh_pool_init(sh_pool &p) {
    if (p.initialized) return;
    p.initialized = true;
    p.cards.push_back(&sh_get());
    const char *env = getenv("SHIELDED_WORKERS");
    if (!env) return;
    auto reject = [&]() {
        p.invalid = true;
        p.cards.resize(1);
        p.extra.clear();
        fprintf(stderr, "[shielded] invalid worker pool; all operations stay on CPU\n");
    };
    if (!*env || strlen(env) > 8192) { reject(); return; }
    std::istringstream lines(env);
    std::string line;
    std::set<std::string> endpoints;
    std::vector<std::unique_ptr<sh_state>> parsed;
    while (std::getline(lines, line)) {
        std::vector<std::string> parts;
        std::istringstream fields(line);
        std::string part;
        while (std::getline(fields, part, '|')) parts.push_back(part);
        if (parts.size() != 4 || parts[0].empty() || parts[0].size() > 127 || parsed.size() >= 16 ||
            parts[0].find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-") != std::string::npos) { reject(); return; }
        uint64_t nums[3];
        for (int i = 0; i < 3; i++) {
            if (parts[i+1].empty() || parts[i+1].find_first_not_of("0123456789") != std::string::npos || parts[i+1].size() > 15) { reject(); return; }
            nums[i] = strtoull(parts[i+1].c_str(), nullptr, 10);
        }
        if (nums[0] < 1 || nums[0] > 65535 || nums[1] > (1U << 30) || nums[2] < 1 || nums[2] > (1ULL << 50) ||
            !endpoints.insert(parts[0] + ":" + std::to_string(nums[0])).second) { reject(); return; }
        auto s = std::make_unique<sh_state>();
        sh_env_defaults(*s);
        s->host = parts[0]; s->port = (int)nums[0]; s->vsock_port = (int)nums[1];
        s->reserve_bytes = nums[2]; s->explicit_link = true;
        double frac = getenv("SHIELDED_WEIGHT_BUDGET_FRAC") ? atof(getenv("SHIELDED_WEIGHT_BUDGET_FRAC")) : .90;
        if (!(frac > .1 && frac <= 1)) frac = .90;
        s->reserve_cap = std::max<int64_t>(1, (int64_t)(nums[2] * frac));
        parsed.push_back(std::move(s));
    }
    if (parsed.empty()) { reject(); return; }
    // A single process-wide refill budget, divided over links. More cards must
    // not create N copies of the old ncores/2 thread pool.
    int threads = sh_env_int("SHIELDED_REFILL_THREADS", (int)std::max(1U, std::thread::hardware_concurrency()/2));
    threads = std::max(0, std::min(64, threads));
    p.cards.clear();
    for (size_t i = 0; i < parsed.size(); i++) {
        parsed[i]->refill_threads = threads / (int)parsed.size() + ((int)i < threads % (int)parsed.size());
        p.cards.push_back(parsed[i].get());
    }
    p.extra = std::move(parsed);
    SH_LOG("pool: %zu workers, %d total refill threads\n", p.cards.size(), threads);
}

static void sh_load_calib(sh_state &s) {
    if (s.calib_loaded) return;
    s.calib_loaded = true;
    if (s.calib_path.empty()) {
        const char *e = getenv("SHIELDED_CALIB");
        if (e && *e) s.calib_path = e;
    }
    if (s.calib_path.empty()) {
        SH_LOG("no calibration configured; nothing will be offloaded\n");
        return;
    }
    FILE *f = fopen(s.calib_path.c_str(), "r");
    if (!f) { SH_LOG("calibration %s unreadable; nothing will be offloaded\n", s.calib_path.c_str()); return; }
    char line[8192];
    bool first = true;
    while (fgets(line, sizeof line, f)) {
        if (first) { int v = 0; if (sscanf(line, "# shielded-calib %d", &v) == 1) s.calib_version = v; first = false; }
        if (line[0] == '#' || line[0] == '\n') continue;
        char name[512]; int af = 0, nout = 0; int pos = 0;
        if (sscanf(line, "site %511s %d %d%n", name, &af, &nout, &pos) < 3) continue;
        sh_calib_site site;
        site.act_frac = af;
        const char *p = line + pos;
        for (int i = 0; i < nout; i++) {
            long long v = 0; int adv = 0;
            if (sscanf(p, " %lld%n", &v, &adv) != 1) break;
            site.outliers.push_back((int64_t)v);
            p += adv;
        }
        s.calib[name] = std::move(site);
    }
    fclose(f);
    if (s.calib_version != 1 && s.calib_version != 2) {
        fprintf(stderr, "[shielded] %s: calibration format %d is unknown; nothing will be offloaded\n", s.calib_path.c_str(), s.calib_version);
        s.calib.clear();
    }
    SH_LOG("calibration: %zu sites from %s, format %d%s (policy: min %lld MAC, max m %d, simd %s)\n",
           s.calib.size(), s.calib_path.c_str(), s.calib_version,
           s.calib_version == 1 ? " (calibrate.py exponents: applying the -5 per-column correction)" : "",
           (long long)sh_min_macs(), sh_max_m(), sh_link_simd()->name);
}

static const sh_calib_site *sh_site_for(sh_state &s, const char *name) {
    sh_env_defaults(s);
    sh_load_calib(s);
    auto it = s.calib.find(sh_group_key(name));
    return it == s.calib.end() ? nullptr : &it->second;
}

/* --------------------------------------------------------------------------
 * Weight registration: straight from ggml's q8_0 rows into THE encoding, one
 * row per output, which is also what the worker wants. No transpose anywhere.
 * ----------------------------------------------------------------------- */
static int sh_prepare_rows_threaded(const void *blocks, int64_t K, int64_t N, int8_t *w_out, int *f_out) {
    unsigned hw = std::thread::hardware_concurrency();
    int nt = (int)std::min<unsigned>(hw ? hw : 1, 16);
    if (N < 256 || (int64_t)nt * 16 > N) nt = 1;
    if (nt == 1) return sh_prepare_weight_rows(blocks, K, N, w_out, f_out);
    std::vector<int> rc((size_t)nt, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++)
        th.emplace_back([&, t]() { rc[t] = sh_prepare_weight_rows_range(blocks, K, N, N * t / nt, N * (t + 1) / nt, w_out, f_out); });
    for (auto &x : th) x.join();
    for (int r : rc) if (r < 0) return r;
    return 0;
}

static bool sh_register(sh_state &s, const ggml_tensor *w) {
    const std::string name = ggml_get_name(w);
    if (s.weights.count(name)) return true;
    if (s.refused.count(name)) return false;

    const int64_t K = w->ne[0], N = w->ne[1];
    const sh_calib_site *site = sh_site_for(s, name.c_str());
    if (!site) return false;
    if (K % SH_QK != 0) return false;
#if defined(__aarch64__)
    /* The ARM CPU backend repacks q8_0 rows into q8_0_4x8 at load time (its
     * CPU_REPACK buffer type) when built with dotprod/i8mm kernels. The bytes
     * are then not q8_0 rows, and the encoder below finds no exponent that fits
     * for every row of every weight -- which is the symptom, not the cause. Say
     * the cause. (x86 never repacks q8_0; this is arm-only so the x86 object
     * stays byte-identical.) */
    if (w->buffer && strstr(ggml_backend_buft_name(ggml_backend_buffer_get_type(w->buffer)), "REPACK")) {
        SH_LOG("%s: lives in a %s buffer: the CPU backend repacked its q8_0 rows; build ggml-cpu with GGML_CPU_REPACK=OFF\n",
               name.c_str(), ggml_backend_buft_name(ggml_backend_buffer_get_type(w->buffer)));
        s.refused.insert(name);
        return false;
    }
#endif

    /* Reservation budget: place calibrated weights until this tenant's slice of
     * the card is full, then leave the rest on CPU. `dev_add` is the device
     * memory this weight would cost the link - its installed int8 matrix plus
     * this node's x/y activation buffers (sh_link_add_weight's abytes growth) -
     * the same bytes the worker charges to the reservation. Checked BEFORE the
     * O(K*N) encoding below, so a weight that will not fit costs nothing. This
     * is what makes a partial offload settle instead of overflowing the worker
     * (see sh_state::reserve_cap). */
    const int64_t dev_add = K * N + (int64_t)sh_max_m() * (3 * K + N * 4);
    if (s.reserve_cap > 0 && s.device_bytes + dev_add > s.reserve_cap) {
        if (!s.budget_full_logged) {
            SH_LOG("reservation full: %lld of %lld device bytes placed; %s and later "
                   "calibrated sites stay on CPU (raise the GPU share to offload more)\n",
                   (long long)s.device_bytes, (long long)s.reserve_cap, name.c_str());
            s.budget_full_logged = true;
        }
        s.refused.insert(name);
        return false;
    }

    const double t0 = sh_now_ms();
    sh_state::entry e;
    e.K = K; e.N = N; e.site = site; e.group = sh_group_key(name);
    e.w.resize((size_t)K * N);
    e.f_w.resize((size_t)N);
    /* Rows are independent: spread the encoding over threads. Registration
     * runs inside the engine's context creation (ggml_backend_sched reserve
     * asks supports_op with the data present), serially per weight, and was
     * 3.4 s of the 0.5B's and 32 s of the 4B's start-up before this. */
    if (sh_prepare_rows_threaded(w->data, K, N, e.w.data(), e.f_w.data()) < 0) {
        SH_LOG("%s: no weight exponent fits the int8 lane; staying on CPU\n", name.c_str());
        s.refused.insert(name);
        return false;
    }

    /* The outlier columns, kept in the TEE. Their contribution is computed here
     * in plain int64 where nothing can wrap, and the offloaded activation has
     * those channels zeroed -- so the channels that would have broken Z_M are
     * exactly the ones the field never has to hold. */
    const size_t nout = site->outliers.size();
    e.out_cols.resize(nout * (size_t)N);
    for (size_t c = 0; c < nout; c++) {
        const int64_t k = site->outliers[c];
        if (k < 0 || k >= K) { SH_LOG("%s: outlier channel %lld out of range\n", name.c_str(), (long long)k); s.refused.insert(name); return false; }
        for (int64_t j = 0; j < N; j++) e.out_cols[c * (size_t)N + j] = e.w[(size_t)j * K + k];
    }
    const int af = site->act_frac + sh_af_delta(s);
    e.inv.resize((size_t)N);
    for (int64_t j = 0; j < N; j++) e.inv[j] = ldexpf(1.0f, -(af + e.f_w[j]));

    if (!s.link) {
        int err = SH_OK;
        s.link = sh_link_open(s.host.c_str(), s.port, true, &err);
        if (!s.link) { s.link_failed = true; return false; }
        if (s.explicit_link) sh_link_configure(s.link, s.vsock_port, s.reserve_bytes, s.refill_threads);
    }
    int lo = e.f_w[0], hi = e.f_w[0];
    for (int64_t j = 1; j < N; j++) { if (e.f_w[j] < lo) lo = e.f_w[j]; if (e.f_w[j] > hi) hi = e.f_w[j]; }

    sh_state::entry &stored = s.weights[name];
    stored = std::move(e);
    stored.name = name;
    auto gf = s.group_first.find(stored.group);
    const int share = gf == s.group_first.end() ? -1 : gf->second;
    const int node = sh_link_add_weight(s.link, name.c_str(), stored.w.data(), K, N, sh_max_m(), share);
    if (node < 0) {
        SH_LOG("%s: %s\n", name.c_str(), sh_link_last_error(s.link));
        s.weights.erase(name); s.refused.insert(name);
        return false;
    }
    stored.node = node;
    s.device_bytes += dev_add;   /* committed to the card; counts against reserve_cap */
    if (share < 0) s.group_first[stored.group] = node;
    s.group_members[stored.group].push_back(name);
    if (s.probe_group.empty()) s.probe_group = stored.group;
    s.dirty = true;
    SH_LOG("registered %s K=%lld N=%lld f_w=%d..%d act_frac=%d outliers=%zu group=%s (%.0f ms)\n",
           name.c_str(), (long long)K, (long long)N, lo, hi, site->act_frac, nout, stored.group.c_str(), sh_now_ms() - t0);
    return true;
}

static std::string sh_layer_key(const std::string &name) {
    const size_t blk = name.find("blk.");
    if (blk != std::string::npos) {
        const size_t end = name.find('.', blk + 4);
        if (end != std::string::npos) return name.substr(0, end);
    }
    return sh_group_key(name);
}
static int64_t sh_weight_bytes(const ggml_tensor &w) {
    return w.ne[0] * w.ne[1] + (int64_t)sh_max_m() * (3*w.ne[0] + 4*w.ne[1]);
}
static void sh_plan(sh_pool &p) {
    if (p.invalid || p.pending.empty()) return;
    using weights = std::vector<const ggml_tensor *>;
    std::map<std::string, std::map<std::string, weights>> layers;
    for (auto &kv : p.pending)
        layers[sh_layer_key(kv.first)][sh_group_key(kv.first)].push_back(&kv.second);
    auto fit = [&](int c, int64_t bytes) {
        auto &s = *p.cards[c];
        return !s.reserve_cap || s.device_bytes + bytes <= s.reserve_cap;
    };
    auto choose = [&](int64_t bytes) {
        int best = -1;
        for (size_t c = 0; c < p.cards.size(); c++) {
            if (!fit((int)c, bytes)) continue;
            auto &s = *p.cards[c];
            if (best < 0) { best = (int)c; continue; }
            auto &b = *p.cards[best];
            const double used = (double)s.device_bytes / (s.reserve_cap ? s.reserve_cap : INT64_MAX);
            const double prev = (double)b.device_bytes / (b.reserve_cap ? b.reserve_cap : INT64_MAX);
            if (used < prev || (used == prev && s.reserve_cap > b.reserve_cap)) best = (int)c;
        }
        return best;
    };
    for (auto &layer : layers) {
        int64_t total = 0;
        for (auto &g : layer.second) for (auto *w : g.second) total += sh_weight_bytes(*w);
        auto prev = p.layers.find(layer.first);
        int preferred = prev == p.layers.end() ? choose(total) : prev->second;
        // Prefer keeping a whole layer together. If it exceeds any one slice,
        // place indivisible activation groups across cards, then CPU overflow.
        for (auto &g : layer.second) {
            int64_t bytes = 0;
            for (auto *w : g.second) bytes += sh_weight_bytes(*w);
            auto old = p.owners.find(g.first);
            int card = old != p.owners.end() ? old->second :
                preferred >= 0 && fit(preferred, bytes) ? preferred : choose(bytes);
            p.owners[g.first] = card;
            if (card < 0) { SH_LOG("placement %s -> CPU (pool reservation full)\n", g.first.c_str()); continue; }
            if (prev == p.layers.end()) p.layers[layer.first] = card;
            SH_LOG("placement %s -> card %d %s:%d (%lld bytes)\n", g.first.c_str(), card,
                   p.cards[card]->host.c_str(), p.cards[card]->port, (long long)bytes);
            for (auto *w : g.second) sh_register(*p.cards[card], w);
        }
    }
    p.pending.clear();
}

/* --------------------------------------------------------------------------
 * The backend
 * ----------------------------------------------------------------------- */
static const char *ggml_backend_shielded_get_name(ggml_backend_t) { return "Shielded"; }
static void ggml_backend_shielded_free(ggml_backend_t backend) { delete backend; }

/* Claimable at all: a calibrated q8_0 weight times an f32 activation, above the
 * size floor. Collects the public weight metadata on first sight of its data,
 * so the planner knows complete layer sizes before the first exchange.
 * `batch_ok` additionally applies the batch-width policy. */
static bool sh_claimable(const ggml_tensor *op, bool batch_ok) {
    if (op->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor *src0 = op->src[0];
    const ggml_tensor *src1 = op->src[1];
    if (!src0 || !src1) return false;
    if (src0->type != GGML_TYPE_Q8_0) return false;      /* the tier's weight format */
    if (src1->type != GGML_TYPE_F32) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[0] % SH_QK != 0 || src0->ne[0] % 16 != 0) return false;
    /* A weight tensor has a name and calibration; an activation-activation
     * product (attention) has neither, and must never come here -- TwinShield's
     * OutAttnMult is broken at the group sizes real GQA uses. */
    const char *nm = ggml_get_name(src0);
    if (!nm || !*nm) return false;
    if (src0->ne[0] * src0->ne[1] < sh_min_macs()) return false;
    sh_pool &p = sh_pool_get();
    std::lock_guard<std::mutex> lk(p.mu);
    sh_pool_init(p);
    if (p.invalid || !sh_site_for(*p.cards[0], nm)) return false;
    // Collect public weight metadata during scheduler reservation. The first
    // graph then plans whole layers with all their sizes known, rather than
    // filling one GPU while supports_op discovers tensors one at a time.
    int owner = sh_owner(p, src0);
    if (src0->data && (owner < 0 || !p.cards[owner]->weights.count(nm)) && !p.owners.count(sh_group_key(nm)))
        p.pending[nm] = *src0;
    else if (src0->data && owner >= 0 && !p.cards[owner]->weights.count(nm) && !p.cards[owner]->refused.count(nm))
        p.pending[nm] = *src0;
    if (batch_ok && src1->ne[1] > sh_max_m()) return false;
    if (p.owners.count(sh_group_key(nm)) && owner < 0) return false;
    if (owner >= 0 && p.cards[owner]->contention.contended && sh_group_key(nm) != p.cards[owner]->probe_group) return false;
    return true;
}

/* --------------------------------------------------------------------------
 * The enclave's own CPU, reached through ggml's CPU backend.
 *
 * Every path that computes a claimed matmul in the enclave used to go through
 * the int64 field product (exact, and 6 tok/s on the 0.5B) or a scalar double
 * loop. ggml_backend_sched keeps its split plan across tokens (the engine
 * reuses its graph), so a node the backend claimed once stays ours whether or
 * not it should still go to the card; what the backend can do is compute it
 * HERE, with the CPU backend the engine already loaded, at the CPU's speed --
 * a one-node graph, the node's own sources (they live in host memory, this
 * backend's buffer type is the CPU's), the CPU backend's threads. That is what
 * a contended card falls back to (a game on the 3070 made the offloaded path
 * six times slower than this), and what a dead link falls back to while it
 * reconnects. It rounds like the CPU backend (fp32 accumulate) rather than
 * like the field, so the text may differ from the offloaded path's; both are
 * the model's own arithmetic. SHIELDED_LOCAL_EXACT=1 keeps the int64 field
 * path instead, for the exactness check (its output IS the worker's).
 * ----------------------------------------------------------------------- */
static ggml_backend_t sh_cpu_backend() {
    static ggml_backend_t be = nullptr;
    static bool tried = false;
    if (tried) return be;
    tried = true;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) return nullptr;
    be = ggml_backend_dev_init(dev, nullptr);
    if (!be) return nullptr;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto set_threads = reg ? (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_n_threads") : nullptr;
    int nt = sh_env_int("SHIELDED_LOCAL_THREADS", 0);
    if (nt <= 0) { unsigned hw = std::thread::hardware_concurrency(); nt = hw > 2 ? (int)(hw / 2) : 2; if (nt > 8) nt = 8; }
    if (set_threads) set_threads(be, nt);
    return be;
}
static bool sh_cpu_compute(ggml_tensor *node) {
    ggml_backend_t be = sh_cpu_backend();
    if (!be) return false;
    static ggml_context *ctx = nullptr;
    if (!ctx) {
        ggml_init_params ip = { ggml_graph_overhead_custom(8, false) + 1024, nullptr, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
    }
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_graph_add_node(gf, node);
    const enum ggml_status st = ggml_backend_graph_compute(be, gf);
    ggml_reset(ctx);
    return st == GGML_STATUS_SUCCESS;
}
static bool sh_local_exact() { static int v = -1; if (v < 0) v = sh_env_int("SHIELDED_LOCAL_EXACT", 0); return v != 0; }

/* The escape hatch: q8_0 x f32 in the clear, in the enclave.
 *
 * A backend that claims an op and then returns GGML_STATUS_FAILED does not
 * degrade, it kills the graph -- llama_decode turns that into rc -3 and the
 * request dies. That is the right behaviour for a VERIFICATION failure, where
 * continuing would mean sampling a value a hostile worker chose. It is the wrong
 * behaviour for "this weight never registered", which is our own bookkeeping
 * problem and has a correct answer sitting right there in the tensor. */
static void sh_plain_mul_mat(const ggml_tensor *w, const ggml_tensor *a, ggml_tensor *dst) {
    const int64_t K = w->ne[0], N = w->ne[1], m = a->ne[1];
    const int64_t nb = K / SH_QK;
    const sh_block_q8_0 *blocks = (const sh_block_q8_0 *)w->data;
    const float *src = (const float *)a->data;
    float *out = (float *)dst->data;
    for (int64_t r = 0; r < m; r++) {
        const float *xr = src + r * K;
        float *orow = out + r * N;
        for (int64_t i = 0; i < N; i++) {
            double acc = 0.0;
            const sh_block_q8_0 *row = blocks + i * nb;
            for (int64_t b = 0; b < nb; b++) {
                const float d = sh_half_to_float(row[b].d);
                const int8_t *q = row[b].qs;
                const float *xb = xr + b * SH_QK;
                double blk = 0.0;
                for (int t = 0; t < SH_QK; t++) blk += (double)xb[t] * (double)q[t];
                acc += blk * (double)d;
            }
            orow[i] = (float)acc;
        }
    }
}

static bool sh_is_meta(const ggml_tensor *t) {
    switch (t->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
        case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: return true;
        default: return false;
    }
}

static void sh_note_latency(sh_state &s, const std::string &group, int rows, double us, double weight_bytes) {
    static const double X  = [] { const char *e = getenv("SHIELDED_CONTENTION_X");  return (e && *e) ? atof(e) : 3.0; }();
    static const double US = [] { const char *e = getenv("SHIELDED_CONTENTION_US"); return (e && *e) ? atof(e) : 200.0; }();
    static const double ABS_X = [] { const char *e = getenv("SHIELDED_CONTENTION_ABS_X"); return (e && *e) ? atof(e) : 8.0; }();
    const int change = s.contention.note(group, rows, us, weight_bytes, group == s.probe_group, X, US, ABS_X);
    if (!change) return;
    const auto &l = s.contention.samples.at({group, rows});
    if (change > 0) {
        fprintf(stderr, "[shielded] the card is contended: %s rows=%d exchanges take %.0f us against a best of %.0f; "
                        "computing in the enclave until it recovers (probing with %s)\n",
                group.c_str(), rows, l.ewma, l.best, s.probe_group.c_str());
    } else {
        fprintf(stderr, "[shielded] the card recovered (%s rows=%d exchanges back to %.0f us); offloading again\n",
                group.c_str(), rows, l.ewma);
    }
}

/* The link is down (socket error, worker refused us, or start failed): stop
 * using it and arm the retry. Pads are unaffected -- the bank's counter never
 * rewinds and the pools are rebuilt from fresh issuance on reconnect. */
static void sh_link_down(sh_state &s) {
    s.link_failed = true;
    s.link_retry_at = sh_now_ms() + s.link_backoff_ms;
    s.link_backoff_ms = s.link_backoff_ms < 60000 ? s.link_backoff_ms * 2 : 60000;
}

static enum ggml_status sh_card_compute(sh_state &s, ggml_cgraph *cgraph) {
    std::lock_guard<std::mutex> lk(s.mu);
    const double tg0 = sh_now_ms();
    const sh_simd *simd = sh_link_simd();

    const int n = ggml_graph_n_nodes(cgraph);
    /* A failed link is retried once its backoff has passed: the same start
     * path as the first connection, carrying the whole weight set again. */
    if (s.link_failed && s.link && sh_now_ms() >= s.link_retry_at) { s.link_failed = false; s.dirty = true; }
    if (s.dirty && s.link && !s.link_failed) {
        const double t0 = sh_now_ms();
        const int rc = sh_link_start(s.link);
        if (rc != SH_OK) {
            SH_LOG("worker unavailable (%s); computing in the enclave, retrying in %.0f s\n",
                   sh_link_last_error(s.link), s.link_backoff_ms / 1000);
            sh_link_down(s);
        } else {
            SH_LOG("worker live over %s with %zu weights, %d refill threads (%.0f ms to upload, install and warm the pool)\n",
                   sh_link_transport(s.link), s.weights.size(), sh_link_refill_threads(s.link), sh_now_ms() - t0);
            s.link_backoff_ms = 1000;
        }
        s.dirty = false;
    }

    std::vector<char> &done = s.done;
    done.assign((size_t)n, 0);
    for (int i = 0; i < n; i++) {
        if (done[i]) continue;
        ggml_tensor *node = ggml_graph_node(cgraph, i);
        if (sh_is_meta(node)) continue;
        if (node->op != GGML_OP_MUL_MAT) {
            // supports_op claims only matmuls and metadata ops, so this is
            // unreachable unless sched changes its mind about what we take.
            fprintf(stderr, "[shielded] refusing an op we never claimed (%s); failing the graph\n", ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
        const ggml_tensor *a = node->src[1];
        auto it = s.weights.find(ggml_get_name(node->src[0]));
        if (it == s.weights.end()) {
            // We claimed it in supports_op and then failed to register it. Compute
            // it honestly rather than killing the graph, and say so once.
            static std::set<std::string> told;
            const std::string nm = ggml_get_name(node->src[0]);
            if (told.insert(nm).second)
                fprintf(stderr, "[shielded] %s: claimed but not registered; computing it "
                                "in the enclave (nothing offloaded for this site)\n", nm.c_str());
            sh_plain_mul_mat(node->src[0], a, node);
            s.local_nodes++; done[i] = 1;
            continue;
        }

        /* Gather every later matmul in this split that reads the SAME activation
         * and belongs to the same group: gate with up, q with k and v. They are
         * one exchange under one pad. */
        std::vector<ggml_tensor *> &members = s.members;
        std::vector<sh_state::entry *> &ents = s.ents;
        members.clear(); ents.clear();
        members.push_back(node); ents.push_back(&it->second);
        for (int j = i + 1; j < n && members.size() < SH_GROUP_MAX; j++) {
            ggml_tensor *o = ggml_graph_node(cgraph, j);
            if (done[j] || sh_is_meta(o)) continue;
            if (o->op != GGML_OP_MUL_MAT || o->src[1] != a) continue;
            auto jt = s.weights.find(ggml_get_name(o->src[0]));
            if (jt == s.weights.end() || jt->second.group != it->second.group) continue;
            members.push_back(o); ents.push_back(&jt->second); done[j] = 1;
        }
        done[i] = 1;

        const sh_state::entry &e0 = *ents[0];
        const int64_t K = e0.K;
        const int32_t m = (int32_t)a->ne[1];
        /* A zero-row matmul has a zero-element output and nothing to exchange.
         * The MTP head context issues one against the tied lm_head; sending it
         * was refused as m outside [1,max_m] and, worse, that refusal used to
         * mark the link dead for the rest of the process. */
        if (m == 0) continue;
        const int af = e0.site->act_frac + sh_af_delta(s);

        /* x_field = round(x * 2^af), with the outlier channels held back. The
         * exponent is a public model constant; deriving it from the activation in
         * hand would buy field headroom by leaking activation magnitude. */
        const double te0 = sh_now_ms();
        std::vector<int64_t> &x_gpu = s.x_gpu, &x_tee = s.x_tee;
        if (x_gpu.size() < (size_t)m * K) x_gpu.resize((size_t)m * K);
        simd->encode((const float *)a->data, (size_t)m * K, ldexpf(1.0f, af), x_gpu.data());
        const size_t nout = e0.site->outliers.size();
        if (nout) {
            if (x_tee.size() < (size_t)m * nout) x_tee.resize((size_t)m * nout);
            for (int32_t r = 0; r < m; r++)
                for (size_t c = 0; c < nout; c++) {
                    const int64_t k = e0.site->outliers[c];
                    x_tee[(size_t)r * nout + c] = x_gpu[(size_t)r * K + k];
                    x_gpu[(size_t)r * K + k] = 0;
                }
        }
        s.t_encode += sh_now_ms() - te0;

        /* The exchange set. Normally the visible members; when the group has
         * members this split cannot see (see group_members), either the whole
         * group is exchanged now and the rest cached, or the visible members
         * are served from the cache an earlier split filled for this very
         * activation. The cache is keyed on the exact field-encoded x that
         * crossed (outliers already held back): a match means the worker would
         * return the identical product, so it is not asked again. */
        const bool live = s.link && !s.link_failed && sh_link_is_live(s.link);
        /* In the enclave, on the CPU backend: a contended card (all but the
         * probe group, which keeps measuring it) or a link that is down,
         * unless the exact int64 path was asked for. */
        if ((s.contention.contended && e0.group != s.probe_group) || (!live && !sh_local_exact())) {
            bool ok = true;
            for (size_t t = 0; t < members.size() && ok; t++) ok = sh_cpu_compute(members[t]);
            if (ok) { s.local_nodes += members.size(); continue; }
            /* no CPU backend to be had: fall through to the paths below */
        }
        std::vector<sh_state::entry *> &xents = s.xents;
        xents.clear();
        const std::vector<std::string> &gm = s.group_members[e0.group];
        sh_state::gcache *gc = nullptr;
        bool served = false;
        if (live && gm.size() > members.size()) {
            gc = &s.completion[e0.group];
            if (gc->m == m && gc->x.size() == (size_t)m * K &&
                memcmp(gc->x.data(), x_gpu.data(), (size_t)m * K * sizeof(int64_t)) == 0) {
                served = true;
                for (size_t t = 0; t < ents.size() && served; t++) served = gc->y.count(ents[t]->name) > 0;
            }
            if (!served) for (const std::string &nm : gm) xents.push_back(&s.weights[nm]);
        }
        if (xents.empty()) xents = ents;          /* served, or no invisible members: the visible set */

        /* One flat y buffer for the exchange set; yp[t] is member t's rows within it. */
        std::vector<int64_t *> &yp = s.yp;
        std::vector<int> &nodes = s.nodes;
        yp.resize(xents.size()); nodes.resize(xents.size());
        size_t ytot = 0;
        for (size_t t = 0; t < xents.size(); t++) ytot += (size_t)m * xents[t]->N;
        if (s.ys.size() < ytot) s.ys.resize(ytot);
        for (size_t t = 0, off = 0; t < xents.size(); t++) {
            yp[t] = s.ys.data() + off; nodes[t] = xents[t]->node;
            off += (size_t)m * xents[t]->N;
        }
        const double tl0 = sh_now_ms();
        int rc;
        if (served) {
            /* Copied out rather than pointed at: the post-processing below
             * adds the outlier term in place, and a member may be served again
             * if ggml asks for it twice. */
            for (size_t t = 0; t < xents.size(); t++) {
                const std::vector<int64_t> &src = gc->y[xents[t]->name];
                memcpy(yp[t], src.data(), src.size() * sizeof(int64_t));
            }
            rc = SH_OK;
            s.offloaded_nodes += members.size(); s.served += members.size();
        } else if (live) {
            rc = sh_link_gemm(s.link, nodes.data(), nodes.size(), x_gpu.data(), m, yp.data());
            if (rc == SH_ERR_VERIFY) {
                /* A corrupted product must never reach the caller: it would be
                 * sampled, streamed, or written into the KV cache, where one bad
                 * entry poisons every future token that attends to it. */
                s.verify_fail++;
                fprintf(stderr, "[shielded] %s\n", sh_link_last_error(s.link));
                return GGML_STATUS_FAILED;
            }
            if (rc != SH_OK) {
                /* Two different failures. A transport error (socket died, or
                 * the worker refused the frame and closed) takes the link down
                 * until the retry fires. A bookkeeping refusal from our own
                 * link (SH_ERR_PROTO: m outside the group's range, a node set
                 * that does not share a group) is a property of THIS node, not
                 * of the connection -- it is computed here and the link stays
                 * up. Conflating the two once turned one odd matmul into a
                 * permanent 15x slowdown. */
                if (rc == SH_ERR_PROTO) {
                    static std::set<std::string> told;
                    const std::string nm = ggml_get_name(node->src[0]);
                    if (told.insert(nm).second)
                        fprintf(stderr, "[shielded] %s: not offloadable as shaped (%s); computing it in the enclave\n",
                                nm.c_str(), sh_link_last_error(s.link));
                } else {
                    fprintf(stderr, "[shielded] %s: offload failed (%s); computing in the enclave, retrying the worker in %.0f s\n",
                            ggml_get_name(node->src[0]), sh_link_last_error(s.link), s.link_backoff_ms / 1000);
                    sh_link_down(s);
                }
                rc = sh_link_gemm_local(s.link, nodes.data(), nodes.size(), x_gpu.data(), m, yp.data());
                s.local_nodes += members.size();
            } else {
                s.offloaded_nodes += members.size(); s.exchanges++;
                {
                    double wb = 0; for (auto *xe : xents) wb += (double)xe->K * (double)xe->N;
                    sh_note_latency(s, e0.group, m, sh_link_last_wire_us(s.link), wb);
                }
                if (xents.size() > members.size()) {
                    /* Keep the invisible members' products for the split that
                     * asks for them, with the x they belong to. */
                    s.completed++;
                    gc->m = m;
                    gc->x.assign(x_gpu.begin(), x_gpu.begin() + (size_t)m * K);
                    gc->y.clear();
                    for (size_t t = 0; t < xents.size(); t++) {
                        bool visible = false;
                        for (size_t v = 0; v < ents.size(); v++) visible = visible || ents[v] == xents[t];
                        if (!visible) gc->y[xents[t]->name].assign(yp[t], yp[t] + (size_t)m * xents[t]->N);
                    }
                }
            }
        } else {
            rc = sh_link_gemm_local(s.link, nodes.data(), nodes.size(), x_gpu.data(), m, yp.data());
            s.local_nodes += members.size();
        }
        s.t_link += sh_now_ms() - tl0;
        if (rc != SH_OK) {
            // Not a verification failure (that returned above) -- a transport or
            // bookkeeping problem. The honest answer is still available locally.
            fprintf(stderr, "[shielded] %s: offload and local path both failed (%d); "
                            "computing it in the enclave\n", ggml_get_name(node->src[0]), rc);
            for (size_t t = 0; t < members.size(); t++) sh_plain_mul_mat(members[t]->src[0], a, members[t]);
            s.local_nodes += members.size();
            continue;
        }

        const double tp0 = sh_now_ms();
        for (size_t t = 0; t < members.size(); t++) {
            const sh_state::entry &e = *ents[t];
            const int64_t N = e.N;
            float *dst = (float *)members[t]->data;
            size_t xi = 0;
            while (xi < xents.size() && xents[xi] != ents[t]) xi++;
            for (int32_t r = 0; r < m; r++) {
                int64_t *yr = yp[xi] + (size_t)r * N;
                /* The outlier term, in the TEE, outside the field. */
                if (nout) simd->outlier_add(x_tee.data() + (size_t)r * nout, e.out_cols.data(), (int)nout, N, yr);
                /* Per-column descale: each output column carries its own exponent,
                 * which is what stops one outlier weight quantising a whole tensor
                 * to nothing. */
                simd->descale(yr, e.inv.data(), (size_t)N, dst + (size_t)r * N);
            }
            s.macs += (uint64_t)m * (uint64_t)K * (uint64_t)N;
        }
        s.t_post += sh_now_ms() - tp0;
    }
    s.t_graph += sh_now_ms() - tg0;
    /* Under SHIELDED_PROFILE, say the per-term totals periodically as well as
     * at the end: the engine inside a CVM never calls the stats entry point,
     * and the tenant's stderr (the owner's /logs) is the only channel out of
     * the guest. Counters only -- never a value that crossed or was masked. */
    if (s.contention.contended) s.contended_graphs++;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_shielded_graph_compute(ggml_backend_t, ggml_cgraph *graph) {
    sh_pool &p = sh_pool_get();
    std::unique_lock<std::mutex> lk(p.mu);
    sh_pool_init(p);
    // Direct backend callers do not necessarily run supports_op first.
    for (int i = 0; i < graph->n_nodes; i++) {
        const auto *node = graph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[0]->data) continue;
        const auto *w = node->src[0];
        const std::string name = ggml_get_name(w);
        int owner = sh_owner(p, w);
        if (!p.invalid && w->type == GGML_TYPE_Q8_0 && sh_site_for(*p.cards[0], name.c_str()) &&
            !p.owners.count(sh_group_key(name))) p.pending[name] = *w;
        else if (!p.invalid && owner >= 0 && !p.cards[owner]->weights.count(name) && !p.cards[owner]->refused.count(name))
            p.pending[name] = *w;
    }
    sh_plan(p);
    // Preserve dependency order. Each contiguous run belongs to one link;
    // completion caches still combine q/k/v or gate/up across scheduler splits.
    for (int i = 0; i < graph->n_nodes;) {
        auto *node = graph->nodes[i];
        if (sh_is_meta(node)) { i++; continue; }
        if (node->op != GGML_OP_MUL_MAT) return GGML_STATUS_FAILED;
        const int owner = sh_owner(p, node->src[0]);
        if (p.invalid || owner < 0) {
            sh_plain_mul_mat(node->src[0], node->src[1], node);
            p.cards[0]->local_nodes++; i++; continue;
        }
        int end = i + 1;
        while (end < graph->n_nodes) {
            const auto *next = graph->nodes[end];
            if (!sh_is_meta(next) && (next->op != GGML_OP_MUL_MAT || sh_owner(p, next->src[0]) != owner)) break;
            end++;
        }
        ggml_cgraph view = {};
        view.n_nodes = view.size = end - i;
        view.nodes = graph->nodes + i;
        view.order = graph->order;
        auto rc = sh_card_compute(*p.cards[owner], &view);
        if (rc != GGML_STATUS_SUCCESS) return rc;
        i = end;
    }
    if (getenv("SHIELDED_PROFILE")) {
        static uint64_t last = 0;
        uint64_t exchanges = 0;
        for (auto *s : p.cards) exchanges += s->exchanges;
        if (exchanges - last >= 4096) {
            last = exchanges;
            lk.unlock();
            ggml_backend_shielded_stats(nullptr, nullptr, nullptr, nullptr);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_shielded_i = {
    /* .get_name            = */ ggml_backend_shielded_get_name,
    /* .free                = */ ggml_backend_shielded_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_shielded_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid_t ggml_backend_shielded_guid(void) {
    static ggml_guid guid = { 0x51, 0x48, 0x1e, 0x1d, 0x22, 0x0b, 0x4c, 0x77,
                              0x9a, 0x3e, 0x6f, 0x14, 0xd0, 0x8b, 0x2a, 0x63 };
    return &guid;
}

bool ggml_backend_is_shielded(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_shielded_guid());
}

/* --- device ------------------------------------------------------------- */
static const char *sh_dev_get_name(ggml_backend_dev_t) { return "Shielded"; }
static const char *sh_dev_get_description(ggml_backend_dev_t) {
    return "masked offload to an untrusted GPU";
}
static void sh_dev_get_memory(ggml_backend_dev_t, size_t *free, size_t *total) {
    *free = *total = 0;   /* the activations live in host memory, inside the CVM */
}
static enum ggml_backend_dev_type sh_dev_get_type(ggml_backend_dev_t) {
    /* ACCEL, not GPU: the card is real but it is not inside the enclave, and the
     * distinction is the whole product. A caller enumerating GPUs must not find
     * this and conclude it has one. */
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
static void sh_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
    props->name        = sh_dev_get_name(dev);
    props->description = sh_dev_get_description(dev);
    props->type        = sh_dev_get_type(dev);
    sh_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { /* async */ false, /* host_buffer */ false,
                    /* buffer_from_host_ptr */ true, /* events */ false };
}
static ggml_backend_t sh_dev_init_backend(ggml_backend_dev_t dev, const char *) {
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_shielded_guid(),
        /* .iface   = */ ggml_backend_shielded_i,
        /* .device  = */ dev,
        /* .context = */ NULL,
    };
    return backend;
}
static ggml_backend_buffer_type_t sh_dev_get_buffer_type(ggml_backend_dev_t) {
    /* Host memory, like the BLAS backend. What reaches the untrusted side is
     * decided explicitly in graph_compute -- never by ggml's buffer plumbing,
     * which has no idea some of these bytes are secret. */
    return ggml_backend_cpu_buffer_type();
}
static bool sh_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor *op) {
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
        case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            return sh_claimable(op, true);
        default:
            return false;   /* everything nonlinear or position-aware stays in the TEE */
    }
}
static bool sh_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_shielded_device_i = {
    /* .get_name             = */ sh_dev_get_name,
    /* .get_description      = */ sh_dev_get_description,
    /* .get_memory           = */ sh_dev_get_memory,
    /* .get_type             = */ sh_dev_get_type,
    /* .get_props            = */ sh_dev_get_props,
    /* .init_backend         = */ sh_dev_init_backend,
    /* .get_buffer_type      = */ sh_dev_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ sh_dev_supports_op,
    /* .supports_buft        = */ sh_dev_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

/* --- reg ---------------------------------------------------------------- */
static const char *sh_reg_get_name(ggml_backend_reg_t) { return "Shielded"; }
static size_t sh_reg_get_device_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t sh_reg_get_device(ggml_backend_reg_t reg, size_t) {
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_shielded_device_i,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };
    return &dev;
}
static const struct ggml_backend_reg_i ggml_backend_shielded_reg_i = {
    /* .get_name         = */ sh_reg_get_name,
    /* .get_device_count = */ sh_reg_get_device_count,
    /* .get_device       = */ sh_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_shielded_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_shielded_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

ggml_backend_t ggml_backend_shielded_init(void) {
    return sh_dev_init_backend(sh_reg_get_device(ggml_backend_shielded_reg(), 0), NULL);
}

/* Loadable as a module, so an engine picks the tier up as launch configuration
 * rather than a rebuild -- ggml_backend_load_all() finds it beside the binary or
 * on GGML_BACKEND_PATH. Only compiled in for the shared-library build; the static
 * one links ggml_backend_shielded_reg() directly. */
#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_shielded_reg)
#endif
