#include "gqa.h"

void gqattention(
    float *x_in, LFM2Config *config, GQAWeights *gqa_weights, Buf *buf, 
    CBuf *cache_buf, int batch, int seq_len, int decode_start, int l_idx
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
    matmul(x_in, q_weights, buf->q, batch, seq_len, d_model, d_out);
    matmul(x_in, k_weights, buf->k, batch, seq_len, d_model, kv_d_out);
    matmul(x_in, v_weights, buf->v, batch, seq_len, d_model, kv_d_out);
    transpose_middle(batch, seq_len, heads, head_dim, buf->q, buf->q_t);
    transpose_middle(batch, seq_len, kv_groups, head_dim, buf->k, buf->k_t);
    transpose_middle(batch, seq_len, kv_groups, head_dim, buf->v, buf->v_t);
    compute_rms_norm(buf->q_t, gqa_weights->q_norm, seq_len * d_out, head_dim);
    compute_rms_norm(buf->k_t, gqa_weights->k_norm, seq_len * kv_d_out, head_dim);
    apply_rope(
        buf->q_t, buf->k_t, buf->cos, buf->sin, seq_len, 
        decode_start, heads, kv_groups, head_dim
    );
    update_cache(
        cache_buf, config, buf->k_t, buf->v_t, 
        batch, seq_len, decode_start, l_idx
    );
    int interim_seq_len = decode_start + seq_len;
    int kv_size = batch * interim_seq_len * kv_groups * head_dim;
    repeat_interleave(
        cache_buf->k_cache[l_idx], kv_size, 
        buf->k_expand, 
        interim_seq_len * head_dim, group_size
    );
    repeat_interleave(
        cache_buf->v_cache[l_idx], kv_size, 
        buf->v_expand, 
        interim_seq_len * head_dim, group_size
    );
    sdpattention(buf, batch, seq_len, decode_start, heads, head_dim);
    transpose_middle(batch, heads, seq_len, head_dim, buf->attn_out, buf->attn_out_t);
    matmul(buf->attn_out_t, gqa_weights->wo, buf->x_out, batch, seq_len, d_model, d_out);
}