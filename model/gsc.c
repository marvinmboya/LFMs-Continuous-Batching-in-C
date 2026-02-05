#include "gsc.h"

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
    int co_sz = get_conv_out_size(seq_len, k_size, k_size - 1, 1);
    depthwise_conv1d(Bx, gsc_w->conv, buf->conv_out, d_model, seq_len, co_sz, k_size, k_size - 1);
    slice_conv_out(buf->conv_out, buf->conv_out_sliced, d_model, co_sz, seq_len);
    elementwise_mul(buf->conv_out_sliced, buf->C, batch * d_model * seq_len);
    transpose_last(batch, d_model, seq_len, buf->conv_out_sliced, buf->conv_out_t);
    matmul(buf->conv_out_t, gsc_w->w2, buf->x_out, batch, seq_len, d_model, d_model);
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