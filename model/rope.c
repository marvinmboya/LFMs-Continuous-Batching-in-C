#include "rope.h"
#include <stdio.h>

void compute_rope(
    float *cos, float *sin,
    int context_window, int head_dim, 
    float theta
) {
    int half_dim = head_dim / 2;
    for (int pos = 0; pos < context_window; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            cos[pos * half_dim + i] = cosf(angle);
            sin[pos * half_dim + i] = sinf(angle);
        }
    }
}

void apply_rope(
    float *q, float *k, const float *cos, const float *sin,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim
) {
    int half_dim = head_dim / 2;

    /* Apply RoPE to Q */
    for (int s = 0; s < seq_len; s++) {
        const float *cos_row = cos + s * half_dim;
        const float *sin_row = sin + s * half_dim;

        for (int h = 0; h < num_q_heads; h++) {
            float *q_head = q + s * num_q_heads * head_dim + h * head_dim;

            for (int i = 0; i < half_dim; i++) {
                float x0 = q_head[i];
                float x1 = q_head[i + half_dim];
                float cos_val = cos_row[i];
                float sin_val = sin_row[i];

                q_head[i] = x0 * cos_val - x1 * sin_val;
                q_head[i + half_dim] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    /* Apply RoPE to K */
    for (int s = 0; s < seq_len; s++) {
        const float *cos_row = cos + s * half_dim;
        const float *sin_row = sin + s * half_dim;

        for (int h = 0; h < num_kv_heads; h++) {
            float *k_head = k + s * num_kv_heads * head_dim + h * head_dim;

            for (int i = 0; i < half_dim; i++) {
                float x0 = k_head[i];
                float x1 = k_head[i + half_dim];
                float cos_val = cos_row[i];
                float sin_val = sin_row[i];

                k_head[i] = x0 * cos_val - x1 * sin_val;
                k_head[i + half_dim] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}