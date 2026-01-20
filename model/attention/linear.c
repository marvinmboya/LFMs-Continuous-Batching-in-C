#include "linear.h"

void matmul(
    const float *x_in,  // B*M*K
    const float *weights,  // K*N
    float *x_out,        // B*M*N
    int BATCH, int M, int K, int N) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < BATCH; ++b) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < K; ++k) {
                    sum += x_in[b * M * K + i * K + k] *
                           weights[k * N + j];
                }
                x_out[b*M*N + i*N + j] = sum;
            }
        }
    }
}