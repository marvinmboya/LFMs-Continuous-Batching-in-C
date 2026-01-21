#include "utils.h"

void transpose_middle(
    int BATCH, int seq_len, int heads, int head_dim, 
    float *old, float *new) {
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