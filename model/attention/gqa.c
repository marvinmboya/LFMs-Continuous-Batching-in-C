#include "gqa.h"

static float *compute_qkv_outs(float *x_in, const float *assoc_weights, int BATCH, int seq_len, int d_model, int d_out);

void gqattention(
    float *x_in, LFM2Config *config, float *qkv_weights, float *wo_weight,
    float *q_norm, float *k_norm, int BATCH, int seq_len
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
    float *q_weights = qkv_weights;
    float *k_weights = qkv_weights + q_size;
    float *v_weights = qkv_weights + q_size + k_size;
    float *q = compute_qkv_outs(x_in, q_weights, BATCH, seq_len, d_model, d_out);
    float *k = compute_qkv_outs(x_in, k_weights, BATCH, seq_len, d_model, kv_d_out);
    float *v = compute_qkv_outs(x_in, v_weights, BATCH, seq_len, d_model, kv_d_out);
    float *q_t = malloc(seq_len * d_out * sizeof(float));
    float *k_t = malloc(seq_len * kv_d_out * sizeof(float));
    float *v_t = malloc(seq_len * kv_d_out * sizeof(float));
    transpose_middle(BATCH, seq_len, heads, head_dim, q, q_t);
    transpose_middle(BATCH, seq_len, kv_groups, head_dim, k, k_t);
    transpose_middle(BATCH, seq_len, kv_groups, head_dim, v, v_t);
    free(q); free(k); free(v);
    q = q_t; k = k_t; v = v_t;
    compute_rms_norm(q, q_norm, seq_len * d_out, head_dim);
    compute_rms_norm(k, k_norm, seq_len * kv_d_out, head_dim);
    int kv_size = BATCH * seq_len * kv_groups * head_dim;
    float *k_expand = malloc(kv_size * group_size * sizeof(float));
    float *v_expand = malloc(kv_size * group_size * sizeof(float));
    repeat_interleave(k, kv_size, k_expand, seq_len * head_dim, group_size);
    repeat_interleave(v, kv_size, v_expand, seq_len * head_dim, group_size);
    free(k); free(v);
    float *out = sdpattention(
    q, k_expand, v_expand, head_dim,
    BATCH, seq_len, heads, head_dim);
    float *out_t = malloc(kv_size * group_size * sizeof(float));
    transpose_middle(BATCH, heads, seq_len, head_dim, out, out_t);
    free(out);
    float *gqa_out = compute_qkv_outs(out_t, wo_weight, BATCH, seq_len, d_model, d_out);
    free(out_t);
    free(gqa_out);
    free(k_expand); free(v_expand);
    free(q);
}

static float *compute_qkv_outs(float *x_in, const float *assoc_weights, int BATCH, int seq_len, int d_model, int d_out) {
    float *x_out = malloc(BATCH * seq_len * d_out * sizeof(float));
    matmul(x_in, assoc_weights, x_out, BATCH, seq_len, d_model, d_out);
    return x_out;
}