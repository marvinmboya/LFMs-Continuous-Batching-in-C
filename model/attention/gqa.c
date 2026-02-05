#include "gqa.h"

void gqattention(
    float *x_in, LFM2Config *config, GQAWeights *gqa_weights, Buf *buf, 
    CBuf *cache_buf, int BATCH, int seq_len, int decode_start, int l_idx
) {
    int d_model = config->d_model,
        heads = config->heads,
        head_dim = config->head_dim,
        d_out = heads * head_dim,
        kv_groups = config->kv_groups,
        kv_d_out = kv_groups * head_dim,
        group_size = (int)(heads / kv_groups),
        q_size = config->d_model * d_out,
        k_size = config->d_model * kv_d_out,
        v_size = k_size;
    float *q_weights = gqa_weights->wqkv;
    float *k_weights = gqa_weights->wqkv + q_size;
    float *v_weights = gqa_weights->wqkv + q_size + k_size;
    matmul(x_in, q_weights, buf->q, BATCH, seq_len, d_model, d_out);
    matmul(x_in, k_weights, buf->k, BATCH, seq_len, d_model, kv_d_out);
    matmul(x_in, v_weights, buf->v, BATCH, seq_len, d_model, kv_d_out);
    transpose_middle(BATCH, seq_len, heads, head_dim, buf->q, buf->q_t);
    transpose_middle(BATCH, seq_len, kv_groups, head_dim, buf->k, buf->k_t);
    transpose_middle(BATCH, seq_len, kv_groups, head_dim, buf->v, buf->v_t);
    compute_rms_norm(buf->q_t, gqa_weights->q_norm, seq_len * d_out, head_dim);
    compute_rms_norm(buf->k_t, gqa_weights->k_norm, seq_len * kv_d_out, head_dim);
    apply_rope(buf->q_t, buf->k_t, buf->cos, buf->sin, seq_len, heads, kv_groups, head_dim);
    update_cache(
        cache_buf, config, buf->k_t, buf->v_t, 
        seq_len, decode_start, l_idx
    );
    int kv_size = BATCH * seq_len * kv_groups * head_dim;
    repeat_interleave(buf->k_t, kv_size, buf->k_expand, seq_len * head_dim, group_size);
    repeat_interleave(buf->v_t, kv_size, buf->v_expand, seq_len * head_dim, group_size);
    sdpattention(buf, BATCH, seq_len, heads, head_dim);
    transpose_middle(BATCH, heads, seq_len, head_dim, buf->attn_out, buf->attn_out_t);
    matmul(buf->attn_out_t, gqa_weights->wo, buf->x_out, BATCH, seq_len, d_model, d_out);
}