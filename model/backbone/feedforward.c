#include "feedforward.h"

void feedforward(
    float *x_in, float *x_out, const float *w1, const float *v, const float *w2, 
    Buf *buf, int batch, int seq_len, int d_model, int d_hidden
){
    matmul(x_in, w1, buf->out_w1, batch, seq_len, d_model, d_hidden);
    matmul(x_in, v, buf->out_v, batch, seq_len, d_model, d_hidden);
    silu(buf->out_w1, batch * seq_len * d_hidden);
    elementwise_mul(buf->out_w1, buf->out_v, batch * seq_len * d_hidden);
    matmul(buf->out_w1, w2, x_out, batch, seq_len, d_hidden, d_model);
}

void silu(float *x, size_t n) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        x[i] = (1.0f / (1.0f + expf(-x[i]))) * x[i];
    }
}