#ifndef UTILSI_H
#define UTILSI_H
#include <string.h>

void transpose_middle(
    int BATCH, int heads, int seq_len, int head_dim, 
    float *old, float *new);

void repeat_interleave(
    const float *in, size_t n, 
    float *out, int block_size, int repeats
);
#endif