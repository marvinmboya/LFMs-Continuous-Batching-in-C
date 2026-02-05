#ifndef UTILSI_H
#define UTILSI_H
#include <string.h>

void transpose_middle(
    int batch, int heads, int seq_len, int head_dim, 
    const float *old, float *new);

void transpose_last_higher(
    int batch, int seq_len, int heads, int head_dim,
    const float *old, float *new);

void transpose_last(
    int batch, int seq_len, int dims,
const float *old, float *new);

void repeat_interleave(
    const float *in, size_t n, 
    float *out, int block_size, int repeats
);

void elementwise_mul( // in-place
    float *a, const float *b, int n);
void elementwise_add( // in-place
    float *a, const float *b, int n);
#endif