#include "init_cache.h"

static void init_calloc(float **buf, size_t n);
static void init_malloc(float **buf, size_t n);

void create_cache_buffers(CBuf *bufs, LFM2Config config, int batch) {
    int NL = 16;
    bufs->conv_state = malloc(NL * sizeof(float*));
    bufs->k_cache = malloc(NL * sizeof(float*));
    bufs->v_cache = malloc(NL * sizeof(float*));
    bufs->cache_seq_len = calloc(NL, sizeof(int));
    size_t conv_sz = (size_t)batch * config.d_model * config.k_size;
    size_t max_sz = batch * config.kv_groups * config.max_seq_len * config.head_dim;
    for (int i = 0; i < NL; i++) {
        init_calloc(&bufs->conv_state[i], conv_sz);
        init_malloc(&bufs->k_cache[i], max_sz);
        init_malloc(&bufs->v_cache[i], max_sz);
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

void update_cache(
    CBuf *bufs, LFM2Config *config, const float *k, 
    const float *v, int batch, int seq_len, int decode_start, int l_idx
) {
    int kv_groups = config->kv_groups;
    int head_dim = config->head_dim;
    
    int current_len = bufs->cache_seq_len[l_idx];
    int new_len = current_len + seq_len;
    if (new_len > config->max_seq_len) {
        fprintf(stderr, "Cache overflow! current=%d, adding=%d, max=%d\n",
                new_len, seq_len, config->max_seq_len);
        abort();
    }
    // Size per cache: B * H * total_seq * HD (compact, no gaps)
    size_t old_size = batch * kv_groups * current_len * head_dim;
    size_t new_size = batch * kv_groups * new_len * head_dim;
    size_t incoming_size = batch * kv_groups * seq_len * head_dim;
    if (current_len == 0) {
        bufs->k_cache[l_idx] = malloc(new_size * sizeof(float));
        bufs->v_cache[l_idx] = malloc(new_size * sizeof(float));
        memcpy(bufs->k_cache[l_idx], k, incoming_size * sizeof(float));
        memcpy(bufs->v_cache[l_idx], v, incoming_size * sizeof(float));
    } else {
        float *new_k = malloc(new_size * sizeof(float));
        float *new_v = malloc(new_size * sizeof(float));
        
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < kv_groups; h++) {
                // Old cache: shape (B, H, current_len, HD) - compact
                size_t old_offset = (b * kv_groups * current_len + h * current_len) * head_dim;
                size_t old_chunk_size = current_len * head_dim;
                
                // New cache: shape (B, H, new_len, HD) - compact
                size_t new_offset = (b * kv_groups * new_len + h * new_len) * head_dim;
                
                // Incoming: shape (B, H, seq_len, HD) - compact
                size_t incoming_offset = (b * kv_groups * seq_len + h * seq_len) * head_dim;
                size_t incoming_chunk_size = seq_len * head_dim;
                
                // Copy old data for this (b, h)
                memcpy(new_k + new_offset, 
                       bufs->k_cache[l_idx] + old_offset, 
                       old_chunk_size * sizeof(float));
                memcpy(new_v + new_offset, 
                       bufs->v_cache[l_idx] + old_offset, 
                       old_chunk_size * sizeof(float));
                
                // Append new data for this (b, h)
                memcpy(new_k + new_offset + old_chunk_size, 
                       k + incoming_offset, 
                       incoming_chunk_size * sizeof(float));
                memcpy(new_v + new_offset + old_chunk_size, 
                       v + incoming_offset, 
                       incoming_chunk_size * sizeof(float));
            }
        }
        
        // Free old cache and use new one
        free(bufs->k_cache[l_idx]);
        free(bufs->v_cache[l_idx]);
        bufs->k_cache[l_idx] = new_k;
        bufs->v_cache[l_idx] = new_v;
    }
    
    bufs->cache_seq_len[l_idx] = new_len;
}

static void init_calloc(float **buf, size_t n) {
    *buf = (float *)calloc(n, sizeof(float));
    if (!*buf) PERR("calloc error!");
}

static void init_malloc(float **buf, size_t n) {
    *buf = malloc(n * sizeof(float));
    if (!*buf) PERR("malloc error!");
}