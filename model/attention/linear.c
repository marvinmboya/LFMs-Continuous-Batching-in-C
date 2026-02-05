#include "linear.h"

#if !defined(__arm__) && !defined(__aarch64__)
void matmul(
    const float *x_in,  // B*M*K
    const float *weights,  // K*N
    float *x_out,        // B*M*N
    int batch, int M, int K, int N) {
    /* 
    multiply 3D by 2D indexed flattened tensors,
    that is:
    INPUT: (Batch,  Heads, Dmodel)
    WEIGHTS: (Dmodel, (KV)Dout)
    => 
    (Batch, Heads, (KV)Dout) 
    */
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
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
#endif

void matmul_higher(
    const float *A,   // [B, H, M, K]
    const float *B,// [B, H, K, N]
    float *C,         // [B, H, M, N]
    int batch,
    int heads,
    int M,
    int K,
    int N
) {
    /* 
    multiply 4D by 4D indexed flattened tensors,
    that is:
    SCORES: (Batch,  Heads, SeqLen, SeqLen)
    V: (Batch,  Heads, SeqLen, HeadDim)
    => 
    (Batch, Heads, SeqLen, HeadDim) 
    */
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    const int a_base = ((b * heads + h) * M + i) * K;
                    const int b_base = ((b * heads + h) * K) * N + j;
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < K; ++k) {
                        sum += A[a_base + k] * B[b_base + k * N];
                    }

                    C[((b * heads + h) * M + i) * N + j] = sum;
                }
            }
        }
    }
}

void matmul_implicit_transpose(
    const float *Q,   // B * H * S * HD
    const float *K,   // B * H * S * HD
    float *QKT,       // B * H * S * S
    int B, int H, int Q_S, int K_S, int HD)
{
    /* 
    multiply 4D by 4D indexed flattened tensors,
    applying transpose implicitly
    that is:
    Q: (Batch,  Heads, SeqLen, HeadDim)
    K: (Batch,  Heads, SeqLen, HeadDim) (Last Dims T Ops)
    => 
    (Batch, Heads, SeqLen, SeqLen) 
    */
    const int BH = B * H;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int bh = 0; bh < BH; ++bh) {
        for (int i = 0; i < Q_S; ++i) {
            for (int j = 0; j < K_S; ++j) {
                const float *q_ptr = Q + (bh * Q_S + i) * HD;
                const float *k_ptr = K + (bh * K_S + j) * HD;
                float sum = 0.0f;
                // Vectorize over head_dim
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < HD; ++k) {
                    sum += q_ptr[k] * k_ptr[k];
                }
                QKT[(bh * Q_S + i) * K_S + j] = sum;
            }
        }
    }
}