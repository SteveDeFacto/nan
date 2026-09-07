/* The dealt pad bank over HTTP (shielded/dealer/PLAN.md, P3/P4a).
 *
 * The consumer's trusted half keeps a small local cache of shipments fetched
 * from a bank that speaks the platform store's shape:
 *   GET <url>?seed_id=<hex>        -> {"shipments":[{"name","bytes","index0","count"},...]}
 *   GET <url>/<seed_id>/<name>     -> the .pads bytes
 * A bank is the operator's NVMe box on the LAN, or the platform relay itself.
 * Shipments are ciphertext to the consumer's pad key and checked cell by cell
 * on import, so the bank is untrusted; it only affects availability.
 *
 * A fetcher thread keeps the cache covering [floor, need): `floor` is the
 * lowest index any weight group still imports (below it, files are spent and
 * the reader unlinks them), `need` is the far edge the engine wants ready
 * (the current ledger window plus one ahead). Beyond that it fills up to
 * `cache_max` bytes in index order. */
#ifndef SHIELDED_BANK_H
#define SHIELDED_BANK_H
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sh_bank sh_bank;

/* Starts the fetcher. `dir` is the reader's directory (created if missing). */
sh_bank *sh_bank_open(const char *url, const char *seed_id_hex, const char *dir, uint64_t cache_max, int *err);
/* Indices below `floor` are spent; shipments wholly below it are never fetched (again). */
void sh_bank_set_floor(sh_bank *b, uint64_t floor);
/* Fetch, in index order and ahead of the cache budget, everything covering [floor, need). */
void sh_bank_set_need(sh_bank *b, uint64_t need);
/* Counters for the profile line: complete fetches, bytes, and the last listing's status (0 = unreachable). */
void sh_bank_stats(const sh_bank *b, uint64_t *files, uint64_t *bytes, int *last_status, char *err, size_t err_cap);
void sh_bank_close(sh_bank *b);

#ifdef __cplusplus
}
#endif
#endif
