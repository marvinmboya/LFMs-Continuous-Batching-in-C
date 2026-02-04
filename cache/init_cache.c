#include "init_cache.h"

int cache_seq_len = 0;
static void init_calloc(float **buf, size_t n);
static void init_malloc(float **buf, size_t n);

void create_cache_buffers(CBuf *bufs, LFM2Config config, int batch) {
    int NL = 16;
    size_t conv_sz = (size_t)batch * config.d_model * config.k_size;
    bufs = (CBuf *)malloc(NL * sizeof(CBuf));
    for (int i = 0; i < NL; i++) {
        init_calloc(&(bufs[i].conv_state), conv_sz);
        init_malloc(&(bufs[i].k_cache), batch * config.max_seq_len * config.kv_groups * config.head_dim);
        init_malloc(&(bufs[i].v_cache), batch * config.max_seq_len * config.kv_groups * config.head_dim);
    }
}

static void destroy_malloc(float *buf) {
    if (buf) free(buf);
}

void destroy_cache_buffers(CBuf *bufs) {
    int NL = 16;
    for (int i = 0; i < NL; i++) {
        destroy_malloc(bufs[i].conv_state);
        destroy_malloc(bufs[i].k_cache);
        destroy_malloc(bufs[i].v_cache);
    }
    if (bufs) free(bufs);
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

static void init_calloc(float **buf, size_t n) {
    *buf = (float *)calloc(n, sizeof(float));
    if (!(*buf)) PERR("calloc error!");
}

static void init_malloc(float **buf, size_t n) {
    *buf = malloc(n * sizeof(float));
    if (!(*buf)) PERR("malloc error!");
}