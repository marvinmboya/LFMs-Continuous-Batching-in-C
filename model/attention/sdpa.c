#include "sdpa.h"

void sdpattention(
    float *q, float *k, float *v, float *out, float scale,
    int BATCH, int seq_len, int heads, int head_dim
){ 
    float *scores = malloc(BATCH*heads*seq_len*seq_len * sizeof(float));
    matmul_implicit_transpose(q, k, scores, BATCH, heads, seq_len, head_dim);
    float *norm_scores = malloc(BATCH*heads*seq_len*seq_len * sizeof(float));
    softmax_last(scores, norm_scores, BATCH, heads, seq_len, seq_len);
    matmul_higher(norm_scores, v, out, BATCH, heads, seq_len, head_dim);
    free(scores); free(norm_scores);
}