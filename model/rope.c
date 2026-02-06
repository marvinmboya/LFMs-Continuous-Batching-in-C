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
    float *q, float *k, const float *cos, const float *sin, int batch,
    int seq_len, int decode_start, int num_q_heads, int num_kv_heads, int head_dim
) {
    int half_dim = head_dim / 2;
    int decode_end = decode_start + seq_len;
    
    /* Apply RoPE to Q */
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_q_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                int pos = decode_start + s;
                const float *cos_row = cos + pos * half_dim;
                const float *sin_row = sin + pos * half_dim;
                
                float *q_head = q + (b * num_q_heads * seq_len * head_dim) +
                                    (h * seq_len * head_dim) +
                                    (s * head_dim);

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
    }

    /* Apply RoPE to K */
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_kv_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                int pos = decode_start + s;
                const float *cos_row = cos + pos * half_dim;
                const float *sin_row = sin + pos * half_dim;
                
                float *k_head = k + (b * num_kv_heads * seq_len * head_dim) +
                                    (h * seq_len * head_dim) +
                                    (s * head_dim);

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
}