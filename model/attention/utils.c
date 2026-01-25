#include "utils.h"
#include <stdio.h>

void transpose_middle(
    int BATCH, int seq_len, int heads, int head_dim, 
    const float *old, float *new) {
    int C = seq_len, H = heads, W = head_dim;
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < BATCH; n++) {
        for (int h = 0; h < H; h++) {
            for (int c = 0; c < C; c++) {
                int dst_base = n*(H*C*W) + h*(C*W) + c*W;
                int src_base = n*(C*H*W) + c*(H*W) + h*W;
                #pragma omp simd
                for (int w = 0; w < W; w++) {
                    new[dst_base + w] = old[src_base + w];
                }
            }
        }
    }
}

void transpose_last_higher(
    int BATCH, int seq_len, int heads, int head_dim,
    const float *old, float *new)
{
    int C = seq_len, H = heads, W = head_dim;
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < BATCH; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int src = n*(C*H*W) + c*(H*W) + h*(W) + w;
                    int dst = n*(C*W*H) + c*(W*H) + w*(H) + h;
                    new[dst] = old[src];
                }
            }
        }
    }
}

void transpose_last(
    int BATCH, int seq_len, int dims,
    const float *old, float *new)
{
    int C = seq_len;
    int W = dims;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < BATCH; n++) {
        for (int c = 0; c < C; c++) {
            for (int w = 0; w < W; w++) {
                int src = n*(C*W) + c*(W) + w;
                int dst = n*(W*C) + w*(C) + c;
                new[dst] = old[src];
            }
        }
    }
}

void repeat_interleave(
    const float *in, size_t n, float *out, int block_size, int repeats) {
    const float *src = in;
    float *dst = out;
    int total_blocks = (int)n / block_size;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_blocks; i++) {
        const float *block = src + (size_t)i * block_size;
        float *out_base = dst + (size_t)i * repeats * block_size;
        for (int r = 0; r < repeats; ++r) {
            memcpy(out_base + (size_t)r * block_size,
            block, block_size * sizeof(float));
        }
    }
}
