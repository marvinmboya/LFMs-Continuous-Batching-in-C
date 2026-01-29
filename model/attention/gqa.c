#include "gqa.h"

void gqattention(
    float *x_in, float *x_out, LFM2Config *config, 
    GQAWeights *gqa_weights, int BATCH, int seq_len
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
    float *q = malloc(BATCH * seq_len * d_out * sizeof(float));
    float *k = malloc(BATCH * seq_len * kv_d_out * sizeof(float));
    float *v = malloc(BATCH * seq_len * kv_d_out * sizeof(float));
    matmul(x_in, q_weights, q, BATCH, seq_len, d_model, d_out);
    matmul(x_in, k_weights, k, BATCH, seq_len, d_model, kv_d_out);
    matmul(x_in, v_weights, v, BATCH, seq_len, d_model, kv_d_out);
    compute_rms_norm(q, gqa_weights->q_norm, seq_len * d_out, head_dim);
    compute_rms_norm(k, gqa_weights->k_norm, seq_len * kv_d_out, head_dim);

    float *q_t = malloc(seq_len * d_out * sizeof(float));
    float *k_t = malloc(seq_len * kv_d_out * sizeof(float));
    float *v_t = malloc(seq_len * kv_d_out * sizeof(float));
    transpose_middle(BATCH, seq_len, heads, head_dim, q, q_t);
    transpose_middle(BATCH, seq_len, kv_groups, head_dim, k, k_t);
    transpose_middle(BATCH, seq_len, kv_groups, head_dim, v, v_t);
    free(q); free(k); free(v);
    q = q_t; k = k_t; v = v_t;
    
    int kv_size = BATCH * seq_len * kv_groups * head_dim;
    float *k_expand = malloc(kv_size * group_size * sizeof(float));
    float *v_expand = malloc(kv_size * group_size * sizeof(float));
    repeat_interleave(k, kv_size, k_expand, seq_len * head_dim, group_size);
    repeat_interleave(v, kv_size, v_expand, seq_len * head_dim, group_size);
    free(k); free(v);
    float *out = malloc(BATCH*heads*seq_len*head_dim * sizeof(float));
    sdpattention(
    q, k_expand, v_expand, out, head_dim,
    BATCH, seq_len, heads, head_dim);
    float *out_t = malloc(kv_size * group_size * sizeof(float));
    transpose_middle(BATCH, heads, seq_len, head_dim, out, out_t);
    free(out);
    matmul(out_t, gqa_weights->wo, x_out, BATCH, seq_len, d_model, d_out);
    free(out_t);
    free(k_expand); free(v_expand);
    free(q);
}