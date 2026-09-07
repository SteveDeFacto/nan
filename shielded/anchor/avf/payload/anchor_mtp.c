#include "anchor_mtp.h"
#include <dlfcn.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

typedef void   (*nextn_set_fn)(struct llama_context *, bool, bool);
typedef float *(*nextn_ith_fn)(struct llama_context *, int32_t);
static nextn_set_fn g_nextn_set;
static nextn_ith_fn g_nextn_ith;

static int bind_nextn(void) {
    if (g_nextn_set && g_nextn_ith) return 1;
    g_nextn_set = (nextn_set_fn)dlsym(RTLD_DEFAULT, "_Z26llama_set_embeddings_nextnP13llama_contextbb");
    g_nextn_ith = (nextn_ith_fn)dlsym(RTLD_DEFAULT, "_Z30llama_get_embeddings_nextn_ithP13llama_contexti");
    return g_nextn_set != NULL && g_nextn_ith != NULL;
}

struct anchor_mtp {
    struct llama_context *head;
    struct llama_model *model;
    int32_t n_embd, n_vocab;
    float *pending_h;      /* h at the last mirrored position: the draft seed */
    float *verify_h;       /* harvested target nextn rows */
    int32_t verify_rows, verify_cap;
    struct llama_batch batch;   /* token + embd, token side malloc'd (llama_batch_init allocates one of them) */
    int32_t batch_cap;
    double t_rm, t_decode, t_argmax, t_observe;
};

struct llama_context *anchor_mtp_ctx(anchor_mtp *m) { return m ? m->head : NULL; }
void anchor_mtp_timers(anchor_mtp *m, double *rm_us, double *decode_us, double *argmax_us, double *observe_us) {
    if (!m) return; *rm_us = m->t_rm; *decode_us = m->t_decode; *argmax_us = m->t_argmax; *observe_us = m->t_observe;
}

anchor_mtp *anchor_mtp_new(struct llama_model *model, struct llama_context *target, uint32_t n_ctx, uint32_t n_batch, int n_threads) {
    if (!model || !target || llama_model_n_layer_nextn(model) <= 0 || !bind_nextn()) return NULL;
    struct llama_context_params p = llama_context_default_params();
    p.n_threads = n_threads; p.n_threads_batch = n_threads;
    p.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    p.n_ctx = n_ctx;
    p.n_batch = n_batch ? n_batch : 8;
    p.n_seq_max = 1;
    p.kv_unified = true;
    struct llama_context *head = llama_init_from_model(model, p);
    if (!head) return NULL;
    anchor_mtp *m = (anchor_mtp *)calloc(1, sizeof *m);
    if (!m) { llama_free(head); return NULL; }
    m->head = head; m->model = model;
    m->n_embd = llama_model_n_embd(model);
    m->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    m->pending_h = (float *)calloc((size_t)m->n_embd, sizeof(float));
    m->batch_cap = (int32_t)p.n_batch;
    m->batch = llama_batch_init(m->batch_cap, m->n_embd, 1);
    m->batch.token = (llama_token *)malloc(sizeof(llama_token) * (size_t)m->batch_cap);
    /* the target emits nextn hidden rows (unmasked); the head its own (masked) to feed proposals forward */
    g_nextn_set(target, true, false);
    g_nextn_set(head, true, true);
    return m;
}

void anchor_mtp_free(anchor_mtp *m) {
    if (!m) return;
    free(m->batch.token); m->batch.token = NULL;
    llama_batch_free(m->batch);
    llama_free(m->head);
    free(m->verify_h); free(m->pending_h); free(m);
}

int anchor_mtp_harvest(anchor_mtp *m, struct llama_context *target, int32_t n_rows) {
    if (!m || n_rows <= 0) return -1;
    if (m->verify_cap < n_rows) {
        float *nv = (float *)realloc(m->verify_h, (size_t)n_rows * m->n_embd * sizeof(float));
        if (!nv) return -1;
        m->verify_h = nv; m->verify_cap = n_rows;
    }
    for (int32_t i = 0; i < n_rows; i++) {
        const float *h = g_nextn_ith(target, i);
        if (!h) { m->verify_rows = 0; return -1; }
        memcpy(m->verify_h + (size_t)i * m->n_embd, h, (size_t)m->n_embd * sizeof(float));
    }
    m->verify_rows = n_rows;
    return 0;
}

int32_t anchor_mtp_observe(anchor_mtp *m, int32_t pos0, const int32_t *tokens, int32_t n) {
    if (!m || n <= 0 || pos0 < 0) return -1;
    if (m->verify_rows < n) return -1;          /* rows 0..n-2 pair tokens 1..n-1; row n-1 becomes pending */
    llama_memory_seq_rm(llama_get_memory(m->head), 0, pos0, -1);   /* drop a previous round's proposals */
    const size_t row = (size_t)m->n_embd;
    /* in batches of the head's capacity (a prompt is longer than a round): token c+j pairs with the
     * target's row c+j-1, the very first with the seed carried from before pos0 */
    for (int32_t c = 0; c < n; c += m->batch_cap) {
        const int32_t len = n - c < m->batch_cap ? n - c : m->batch_cap;
        for (int32_t j = 0; j < len; j++) {
            m->batch.token[j] = tokens[c + j];
            m->batch.pos[j] = pos0 + c + j;
            m->batch.n_seq_id[j] = 1;
            m->batch.seq_id[j][0] = 0;
            m->batch.logits[j] = 0;
            const float *h = (c + j == 0) ? m->pending_h : m->verify_h + (size_t)(c + j - 1) * row;
            memcpy(m->batch.embd + (size_t)j * row, h, row * sizeof(float));
        }
        m->batch.n_tokens = len;
        const double t0 = now_us();
        const int32_t rc = llama_decode(m->head, m->batch);
        m->t_observe += now_us() - t0;
        if (rc != 0) return rc;
    }
    memcpy(m->pending_h, m->verify_h + (size_t)(n - 1) * row, row * sizeof(float));
    return 0;
}

int32_t anchor_mtp_draft(anchor_mtp *m, int32_t id_last, int32_t n_past, int32_t k, float p_min, int32_t *tokens_out) {
    if (!m || k <= 0 || n_past < 0) return 0;
    double t0 = now_us();
    llama_memory_seq_rm(llama_get_memory(m->head), 0, n_past, -1);
    m->t_rm += now_us() - t0;
    const size_t row = (size_t)m->n_embd;
    int32_t tok = id_last, n = 0;
    const float *h = m->pending_h;
    for (int32_t i = 0; i < k; i++) {
        m->batch.token[0] = tok; m->batch.pos[0] = n_past + i; m->batch.n_seq_id[0] = 1; m->batch.seq_id[0][0] = 0; m->batch.logits[0] = 1;
        memcpy(m->batch.embd, h, row * sizeof(float));
        m->batch.n_tokens = 1;
        t0 = now_us();
        const int rc = llama_decode(m->head, m->batch);
        m->t_decode += now_us() - t0; t0 = now_us();
        if (rc != 0) break;
        const float *lg = llama_get_logits_ith(m->head, 0);
        if (!lg) break;
        int32_t best = 0; float lmax = lg[0];
        for (int32_t v = 1; v < m->n_vocab; v++) if (lg[v] > lmax) { lmax = lg[v]; best = v; }
        if (p_min > 0.0f) {
            const float floor_l = lmax - 16.0f; double sum = 0.03;
            for (int32_t v = 0; v < m->n_vocab; v++) if (lg[v] >= floor_l) sum += exp((double)(lg[v] - lmax));
            if ((float)(1.0 / sum) < p_min) break;
        }
        tokens_out[n++] = best; tok = best;
        h = g_nextn_ith(m->head, 0);
        m->t_argmax += now_us() - t0;
        if (!h) break;
    }
    return n;
}
