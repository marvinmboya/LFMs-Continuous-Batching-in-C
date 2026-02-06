#ifndef GSC_H
#define GSC_H
#include <stdio.h>
#include <omp.h>
#include "attention/linear.h"
#include "attention/utils.h"
#include "../cache/init_cache.h"
#include "types.h"
#include <math.h>
#include <string.h>

int get_conv_out_size(
    int in_size, int k_size, int p_size, int stride);
void depthwise_conv1d(
    float* input, float* weight, float* output, int channels, 
    int in_size, int out_size, int k_size, int p_size
);
void slice_conv_out(
    float* input, float* output, int channels, 
    int seq_len_in, int seq_len_out
);
void gscblock(
    float *x_in, GSCWeights *gsc_weights, Buf *buf, CBuf *cache_buf, int batch, 
    int seq_len, int decode_start, int l_idx, int d_model, int k_size
);
void gsc_mutate_prefill(
    CBuf *cache_buf, const float *Bx, int l_idx, 
    int batch, int seq_len, int d_model, int k_size
);
void gsc_mutate_decode(
    float *conv_state, CBuf *cache_buf, const float *Bx, int l_idx, 
    int batch, int seq_len, int d_model, int k_size
);
void get_sliced_Bx(
    Buf *buf, GSCWeights *gsc_w, float *Bx, 
    int d_model, int seq_len, int k_size
);
void roll(float *x, int d_model);
void conv_decode_sum(
    const float* conv_state, const float* weight,
    float* conv_out, int batch, int d_model, int k_size
);
#endif