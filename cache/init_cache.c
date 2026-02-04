#include "init_cache.h"

int cache_seq_len = 0;
static void init_calloc(float *buf, size_t n);
static void init_malloc(float *buf, size_t n);

void create_cache_buffers(CBuf *bufs, LFM2Config config, int batch) {
    int NL = 16;
    bufs->conv_state = malloc(NL * sizeof(float*));
    bufs->k_cache = malloc(NL * sizeof(float*));
    bufs->v_cache = malloc(NL * sizeof(float*));
    size_t conv_sz = (size_t)batch * config.d_model * config.k_size;
    for (int i = 0; i < NL; i++) {
        init_calloc(bufs->conv_state[i], conv_sz);
        init_malloc(bufs->k_cache[i], batch * config.max_seq_len * config.kv_groups * config.head_dim);
        init_malloc(bufs->v_cache[i], batch * config.max_seq_len * config.kv_groups * config.head_dim);
    }
}


void destroy_cache_buffers(CBuf *buf) {
    int NL = 16;
    for (int i = 0; i < NL; i++) {
        free(buf->conv_state[i]);
        free(buf->k_cache[i]);
        free(buf->v_cache[i]);
    }
    free(buf->conv_state);
    free(buf->k_cache);
    free(buf->v_cache);
}

void update_cache(CBuf *bufs, LFM2Config *config, const float *k, const float *v, int start, int seq_len, int idx) {
    size_t sz = seq_len * config->kv_groups * config->head_dim;
    size_t offset = (size_t)(start - 1) * sz;
    memcpy(bufs[idx].k_cache + offset, k, sz * sizeof(float));
    memcpy(bufs[idx].v_cache + offset, v, sz * sizeof(float));
    cache_seq_len = start + seq_len;
}

int get_seq_len() {
    return cache_seq_len;
}

static void init_calloc(float *buf, size_t n) {
    buf = (float *)calloc(n, sizeof(float));
    if (!buf) PERR("calloc error!");
}

static void init_malloc(float *buf, size_t n) {
    buf = malloc(n * sizeof(float));
    if (!buf) PERR("malloc error!");
}