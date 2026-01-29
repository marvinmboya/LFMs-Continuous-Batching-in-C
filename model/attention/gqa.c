#include "gqa.h"

static void new_linear(
        const float *x_in, const float *weights, float *x_out, 
        int batch, int seq_len, int in_dim, int out_dim
    ) {    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                float sum = 0.0f;
                for (int i = 0; i < in_dim; i++) {
                    sum += x_in[
                    b * seq_len * in_dim + s * in_dim + i] * 
                    weights[i * out_dim + o];
                }
                x_out[s * out_dim + o] = sum;
            }
        }
    }
}

static void in_place_rms_norm(
    float *x, const float *weight, int seq_len, 
    int heads, int head_dim, float eps
    ) {
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < heads; h++) {
            float *x_head = x + (s * heads * head_dim) + (h * head_dim);
            float sum_sq = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                float v = x_head[i];
                sum_sq += v * v;
            }
            float rms_inv = 1.0f / sqrtf((sum_sq / head_dim) + eps);
            for (int i = 0; i < head_dim; i++) {
                x_head[i] *= rms_inv * weight[i];
            }
        }
    }
}

#include <float.h>

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
    in_place_rms_norm(q, gqa_weights->q_norm, seq_len, heads, head_dim, 1e-5f);
    in_place_rms_norm(k, gqa_weights->k_norm, seq_len, kv_groups, head_dim, 1e-5f);
    float *attn_out = malloc(BATCH * seq_len * d_model * sizeof(float));
    float *scores = malloc(BATCH * heads * seq_len * seq_len * sizeof(float));
    const float scale = 1.0f / sqrtf((float)head_dim);

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < heads; h++) {
        for (int i = 0; i < seq_len; i++) {        
            int kv_h = h / group_size;
            const float *Q_vec = q + (i * d_out) + (h * head_dim);
            float *attn_out_slice = attn_out + (i * d_out) + (h * head_dim);
            float *scores_row = scores + (h * seq_len * seq_len) + (i * seq_len);
            float max_score = -FLT_MAX;
            for (int j = 0; j < seq_len; j++) {
                const float *K_vec = k + (j * kv_d_out) + (kv_h * head_dim);
                float dot = 0.0f;
                #pragma omp simd reduction(+:dot)
                for (int d = 0; d < head_dim; d++) {
                    dot += Q_vec[d] * K_vec[d];
                }
                scores_row[j] = dot * scale;
                if (scores_row[j] > max_score) max_score = scores_row[j];
            }
            
            float exp_sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores_row[j] = expf(scores_row[j] - max_score);
                exp_sum += scores_row[j];
            }
            float inv_sum = 1.0f / exp_sum;
            
            memset(attn_out_slice, 0, head_dim * sizeof(float));
            for (int j = 0; j < seq_len; j++) {
                float weight = scores_row[j] * inv_sum;
                if (weight < 1e-9f) continue;
                const float *V_vec = v + (j * kv_d_out) + (kv_h * head_dim);
                for (int d = 0; d < head_dim; d++) {
                    attn_out_slice[d] += weight * V_vec[d];
                }
            }
        }
    }
    matmul(attn_out, gqa_weights->wo, x_out, BATCH, seq_len, d_model, d_out);
    free(attn_out);
    free(q); free(k); free(v);
}