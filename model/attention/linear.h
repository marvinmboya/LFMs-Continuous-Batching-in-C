#ifndef LINEAR_H
#define LINEAR_H
#include <omp.h>

#if defined(__arm__) || defined(__aarch64__)
    #include <armpl.h>
#elif defined(__x86_64__) || defined(_M_X64)
    #include <mkl.h>
#endif

#if defined(__arm__) || defined(__aarch64__) || \
    defined(__x86_64__) || defined(_M_X64)

    static inline void matmul(const float *a, const float *b, float *c, 
                             int batch, int M, int K, int N) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            1.0f, a, K, b, N,
            0.0f, c, N
        );
    }
#else
    void matmul(
        const float *x_in, const float *weights, float *x_out, int batch, int M, int K, int N);
#endif

void matmul_higher(
    const float *A, const float *B, float *C, int batch, int heads, int q_seq_len, int k_seq_len, int head_dim);
void matmul_implicit_transpose(
    const float *Q, const float *K, float *QKT, int batch, int heads, int q_seq_len, int k_seq_len, int head_dim);
#endif
