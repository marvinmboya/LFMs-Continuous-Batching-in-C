#ifndef LINEAR_H
#define LINEAR_H
#include <omp.h>

void matmul(
    const float *x_in, const float *weights, float *x_out, int BATCH, int M, int K, int N);
void matmul_higher(
    const float *A, const float *B, float *C, int BATCH, int heads, int seq_len, int head_dim);
void matmul_implicit_transpose(
    const float *Q, const float *K, float *QKT, int BATCH, int heads, int seq_len, int head_dim);
#endif