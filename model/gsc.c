#include "gsc.h"
#include <stdio.h>

void gscblock(
    float *x_in, GSCWeights *gsc_w, Buf *buf, CBuf *cache_buf, int batch, 
    int seq_len, int decode_start, int l_idx, int d_model, int k_size
){
    matmul(x_in, gsc_w->w1, buf->BCx, batch, seq_len, d_model, d_model * 3);
    transpose_last(batch, seq_len, d_model * 3, buf->BCx, buf->BCx_t );
    size_t sz = batch * seq_len * d_model, fsz = sz * sizeof(float);
    memcpy(buf->B, buf->BCx_t, fsz);
    memcpy(buf->C, buf->BCx_t + sz, fsz);
    memcpy(buf->x, buf->BCx_t + (sz * 2), fsz);
    elementwise_mul(buf->B, buf->x, sz);
    float *Bx = buf->B;
    if (decode_start > 0){
        size_t sz = batch * d_model * k_size, 
          fsz = sz * sizeof(float);
        float *conv_state = malloc(fsz);
        gsc_mutate_decode(conv_state, cache_buf, Bx, l_idx, batch, seq_len, d_model, k_size);
        conv_decode_sum(conv_state, gsc_w->conv,
        buf->conv_out_sliced, batch, d_model, k_size);
        free(conv_state);
    } else {
        gsc_mutate_prefill(cache_buf, Bx, l_idx, batch, seq_len, d_model, k_size);
        get_sliced_Bx(buf, gsc_w, Bx, d_model, seq_len, k_size);
    }
    elementwise_mul(buf->conv_out_sliced, buf->C, batch * d_model * seq_len);
    transpose_last(batch, d_model, seq_len, buf->conv_out_sliced, buf->conv_out_t);
    matmul(buf->conv_out_t, gsc_w->w2, buf->x_out, batch, seq_len, d_model, d_model);
}

void gsc_mutate_prefill(
    CBuf *cache_buf, const float *Bx, int l_idx, 
    int batch, int seq_len, int d_model, int k_size
) {
    #pragma omp parallel for
    for (int i = 0; i < d_model; i ++){
        memcpy(
            cache_buf->conv_state[l_idx] + (i * k_size), 
            Bx + ((i + 1) * seq_len - k_size),
            k_size * sizeof(float)
        );
    }
}

void gsc_mutate_decode(
    float *conv_state, CBuf *cache_buf, const float *Bx, int l_idx, 
    int batch, int seq_len, int d_model, int k_size
) {
    int cache_id = k_size - 1;
    size_t sz = batch * d_model * k_size, 
          fsz = sz * sizeof(float);
    memcpy(conv_state, cache_buf->conv_state[l_idx], fsz);
    roll(conv_state, d_model);
    #pragma omp simd
    for (int d = 0; d < d_model; ++d) {
        conv_state[d * k_size + cache_id] = Bx[d];
    }
    memcpy(cache_buf->conv_state[l_idx], conv_state, fsz);
}

void roll(float *x, int d_model){
    #pragma omp parallel for
    for (int j = 0; j < d_model; j++) {
        int base = j * 3;
        float tmp = x[base];
        x[base] = x[base + 1];
        x[base + 1] = x[base + 2];
        x[base + 2] = tmp;
    }
}

void conv_decode_sum(
    const float* conv_state, const float* weight,
    float* conv_out, int batch, int d_model, int k_size
) {
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < d_model; ++d) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < k_size; ++k) {
                sum += conv_state[
                    b * d_model * k_size 
                    + d * k_size + k] 
                * weight[d * k_size + k];
            }
            conv_out[b * d_model + d] = sum;
        }
    }
}


void get_sliced_Bx(
    Buf *buf, GSCWeights *gsc_w, float *Bx, 
    int d_model, int seq_len, int k_size
) {
    int co_sz = get_conv_out_size(seq_len, k_size, k_size - 1, 1);
    depthwise_conv1d(
        Bx, gsc_w->conv, buf->conv_out, d_model, 
        seq_len, co_sz, k_size, k_size - 1
    );
    slice_conv_out(buf->conv_out, buf->conv_out_sliced, d_model, co_sz, seq_len);
}

int get_conv_out_size(int in_size, int k_size, int p_size, int stride){
    p_size = 2 * p_size;
    int _part = (int)floor((in_size + p_size - k_size)/stride);
    return _part + 1;
}

void depthwise_conv1d(
    float* input, float* weight, float* output, int channels, int in_size, int out_size, int k_size, int p_size) {
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < channels; c++) {
        float* in_chan = &input[c * in_size];
        float* weight_chan = &weight[c * k_size];
        float* out_chan = &output[c * out_size];
        for (int ow = 0; ow < out_size; ow++) {
            float sum = 0.0f;
            for (int kw = 0; kw < k_size; kw++) {
                int iw = ow + kw - p_size;
                if (iw >= 0 && iw < in_size) {
                    sum += in_chan[iw] * weight_chan[kw];
                }
            }
            out_chan[ow] = sum;
        }
    }
}

void slice_conv_out(
float* input, float* output, int channels, 
int seq_len_in, int seq_len_out) {
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < channels; c++) {
        const float* src = &input[c * seq_len_in];
        float* dst = &output[c * seq_len_out];
        memcpy(dst, src, seq_len_out * sizeof(float));
    }
}