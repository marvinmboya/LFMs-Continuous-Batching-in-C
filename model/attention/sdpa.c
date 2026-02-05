#include "sdpa.h"

void sdpattention(
    Buf *buf, int batch, int seq_len, int heads, int head_dim
){ 
    matmul_implicit_transpose(buf->q_t, buf->k_expand, buf->scores, batch, heads, seq_len, head_dim);
    softmax_last(buf->scores, buf->norm_scores, batch, heads, seq_len, seq_len);
    matmul_higher(buf->norm_scores, buf->v_expand, buf->attn_out, batch, heads, seq_len, head_dim);
}