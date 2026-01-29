#include "feedforward.h"

void feedforward(
    float *x_in, float *x_out, const float *w1, const float *v, 
    const float *w2, int batch, int seq_len, int d_model, int d_hidden
){
    float *out_w1 = malloc(batch * seq_len * d_hidden * sizeof(float));
    float *out_v = malloc(batch * seq_len * d_hidden * sizeof(float));

    matmul(x_in, w1, out_w1, batch, seq_len, d_model, d_hidden);
    matmul(x_in, v, out_v, batch, seq_len, d_model, d_hidden);
    silu(out_w1, batch * seq_len * d_hidden);
    elementwise_mul(out_w1, out_v, batch * seq_len * d_hidden);
    matmul(out_w1, w2, x_out, batch, seq_len, d_hidden, d_model);
}

void silu(float *x, size_t n) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        x[i] = (1.0f / (1.0f + expf(-x[i]))) * x[i];
    }
}