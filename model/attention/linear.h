#ifndef LINEAR_H
#define LINEAR_H
#include <omp.h>

#if defined(__arm__) || defined(__aarch64__)
    #include <armpl.h>
    static inline void matmul(const float *a, const float *b, float *c, 
                             int BATCH, int M, int K, int N) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            1.0f, a, K, b, N,
            0.0f, c, N
        );
    }
#else
    void matmul(
        const float *x_in, const float *weights, float *x_out, int BATCH, int M, int K, int N);
#endif

void matmul_higher(
    const float *A, const float *B, float *C, int BATCH, int heads, int seq_len, int head_dim);
void matmul_implicit_transpose(
    const float *Q, const float *K, float *QKT, int BATCH, int heads, int seq_len, int head_dim);
#endif