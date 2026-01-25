#include "softmax.h"

void softmax_last(
    const float *in, float *out, int B, int H, int S, int HD)
{
    const float inv_scale = 1.0f / 8.0f;
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = 0; s < S; ++s) {
                const int base = ((b * H + h) * S + s) * HD;
                // 1. max
                float maxv = -INFINITY;
                for (int k = 0; k < HD; ++k) {
                    float v = in[base + k] * inv_scale;
                    if (v > maxv) maxv = v;
                }
                // 2. exp + sum
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < HD; ++k) {
                    float e = expf(in[base + k] * inv_scale - maxv);
                    out[base + k] = e;
                    sum += e;
                }
                // 3. normalize
                float inv_sum = 1.0f / sum;
                #pragma omp simd
                for (int k = 0; k < HD; ++k) {
                    out[base + k] *= inv_sum;
                }
            }
        }
    }
}