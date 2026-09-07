/* A minimal HTTP/1.1 client for the trusted half: plain http://, one request
 * per connection, Content-Length or chunked bodies. It exists so the dealt
 * pad bank (shielded-bank.c) and the ledger window can be reached over the
 * operator's LAN or a loopback agent without pulling a TLS stack into the
 * enclave; every byte it fetches is either ciphertext to the consumer's key
 * or a signed message, so the transport carries no trust. */
#ifndef SHIELDED_HTTP_H
#define SHIELDED_HTTP_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Body sink: return 0 to keep going, non-zero to abort. */
typedef int (*sh_http_sink)(void *ctx, const uint8_t *p, size_t n);

/* One request. `body` may be NULL. Returns 0 on a complete response (any
 * status, reported in *status), -1 on a transport/parse failure. The sink
 * sees the body as it arrives. `timeout_ms` bounds connect and each read. */
int sh_http_request(const char *method, const char *url, const char *content_type,
                    const void *body, size_t body_len, int timeout_ms,
                    sh_http_sink sink, void *ctx, int *status);

/* A bearer sent on every request from now on (the loopback agent's per-boot
 * token: tenants share the loopback namespace, so a window or a receipt must
 * carry proof it came from the engine the manager configured). NULL clears. */
void sh_http_set_bearer(const char *token);
/* GET into memory: *out is malloc'd and NUL-terminated (caller frees). */
int sh_http_get(const char *url, int timeout_ms, uint8_t **out, size_t *out_len, int *status);
/* POST a JSON body, answer into memory. */
int sh_http_post_json(const char *url, const char *json, int timeout_ms, uint8_t **out, size_t *out_len, int *status);
/* GET streamed into a file at `path` (created/truncated). Returns 0 on 200
 * with the whole body written, -1 otherwise (the file is removed). */
int sh_http_download(const char *url, const char *path, int timeout_ms, uint64_t *bytes);

#ifdef __cplusplus
}
#endif
#endif
