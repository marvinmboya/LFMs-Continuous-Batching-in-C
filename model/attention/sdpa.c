#include "sdpa.h"

float *sdpattention(
    float *q, float *k, float *v, float scale,
    int BATCH, int seq_len, int heads, int head_dim
){ 
    float *scores = malloc(BATCH*heads*seq_len*seq_len * sizeof(float));
    matmul_implicit_transpose(q, k, scores, BATCH, heads, seq_len, head_dim);
    float *norm_scores = malloc(BATCH*heads*seq_len*seq_len * sizeof(float));
    softmax_last(scores, norm_scores, BATCH, heads, seq_len, seq_len);
    free(scores);
    float *out = malloc(BATCH*heads*seq_len*head_dim * sizeof(float));
    matmul_higher(norm_scores, v, out, BATCH, heads, seq_len, head_dim);
    free(norm_scores);
    return out;
}