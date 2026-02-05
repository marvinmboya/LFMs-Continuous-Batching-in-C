#ifndef ROPE_H
#define ROPE_H
#include <math.h>

void compute_rope(
    float *cos, float *sin,
    int max_seq_len, int head_dim, 
    float theta
);
void apply_rope(
    float *q, float *k, const float *cos, const float *sin,
    int seq_len, int decode_start, int num_q_heads, int num_kv_heads, int head_dim
);
#endif