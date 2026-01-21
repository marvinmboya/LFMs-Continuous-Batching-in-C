#include "gqa.h"

static float *compute_qkv_outs(float *x_in, const float *assoc_weights, int BATCH, int seq_len, int d_model, int d_out);

void gqattention(float *x_in, LFM2Config *config, float *qkv_weights, int BATCH, int seq_len) {
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
    float *q_trans = malloc(q_size * sizeof(float));
    float *k_trans = malloc(k_size * sizeof(float));
    float *v_trans = malloc(v_size * sizeof(float));
    transpose_middle(BATCH, heads, seq_len, head_dim, q, q_trans);
    transpose_middle(BATCH, kv_groups, seq_len, head_dim, k, k_trans);
    transpose_middle(BATCH, kv_groups, seq_len, head_dim, v, v_trans);
    free(q); free(k); free(v);
    q = q_trans; k = k_trans; v = v_trans;
    free(q); free(k); free(v);
}

static float *compute_qkv_outs(float *x_in, const float *assoc_weights, int BATCH, int seq_len, int d_model, int d_out) {
    float *x_out = malloc(BATCH * seq_len * d_out * sizeof(float));
    matmul(x_in, assoc_weights, x_out, BATCH, seq_len, d_model, d_out);
    return x_out;
}