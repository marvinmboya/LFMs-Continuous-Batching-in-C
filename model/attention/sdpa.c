#include "sdpa.h"

void sdpattention(
    Buf *buf, int batch, int seq_len, int decode_start, int heads, int head_dim
){ 
    int q_seq_len = seq_len, k_seq_len = decode_start + seq_len;
    matmul_implicit_transpose(buf->q_t, buf->k_expand, buf->scores, batch, heads, q_seq_len, k_seq_len, head_dim);
    softmax_last(buf->scores, buf->norm_scores, batch, heads, q_seq_len, k_seq_len);
    matmul_higher(buf->norm_scores, buf->v_expand, buf->attn_out, batch, heads, q_seq_len, k_seq_len, head_dim);
}