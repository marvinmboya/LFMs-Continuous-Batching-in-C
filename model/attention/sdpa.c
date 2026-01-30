#include "sdpa.h"

void sdpattention(
    float *q, float *k, float *v, Buf *buf, float scale,
    int BATCH, int seq_len, int heads, int head_dim
){ 
    matmul_implicit_transpose(q, k, buf->scores, BATCH, heads, seq_len, head_dim);
    softmax_last(buf->scores, buf->norm_scores, BATCH, heads, seq_len, seq_len);
    matmul_higher(buf->norm_scores, v, buf->attn_out, BATCH, heads, seq_len, head_dim);
}