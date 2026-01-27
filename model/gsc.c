#include "gsc.h"

static void compute_w_outs(
    float *x_in, float *x_out, const float *weights, 
    int BATCH, int seq_len, int d_model, int d_out
);

void gscblock(
    float *x_in, float *x_out, GSCWeights *gsc_w, 
    int BATCH, int seq_len, int d_model, int k_size
){
    float *BCx = malloc(BATCH * seq_len * d_model * 3 * sizeof(float));
    compute_w_outs(x_in, BCx, gsc_w->w1, BATCH, seq_len, d_model, d_model * 3);
    float *BCx_t = malloc(BATCH * seq_len * d_model * 3 * sizeof(float));
    transpose_last(BATCH, seq_len, d_model * 3, BCx, BCx_t);
    size_t sz = BATCH * seq_len * d_model, fsz = sz * sizeof(float);
    float *B = malloc(fsz), *C = malloc(fsz), *x = malloc(fsz);
    memcpy(B, BCx_t, fsz);
    memcpy(C, BCx_t + sz, fsz);
    memcpy(x, BCx_t + (sz * 2), fsz);
    elementwise_mul(B, x, sz);
    float *Bx = B;
    int co_sz = get_conv_out_size(seq_len, k_size, k_size - 1, 1);
    float *conv_out = malloc(BATCH * d_model * co_sz * sizeof(float));
    depthwise_conv1d(Bx, gsc_w->conv, conv_out, d_model, seq_len, co_sz, k_size, k_size - 1);
    float *conv_out_sliced = malloc(BATCH * d_model * seq_len * sizeof(float));
    slice_conv_out(conv_out, conv_out_sliced, d_model, co_sz, seq_len);
    elementwise_mul(conv_out_sliced, C, BATCH * d_model * seq_len);
    float *conv_out_t = malloc(BATCH * d_model * seq_len * sizeof(float));
    transpose_last(BATCH, d_model, seq_len, conv_out_sliced, conv_out_t);
    compute_w_outs(conv_out_t, x_out, gsc_w->w2, BATCH, seq_len, d_model, d_model);
    free(conv_out);
    free(conv_out_sliced);
    free(Bx); free(C); free(x);
    free(BCx);
    free(BCx_t);
}

static void compute_w_outs(
    float *x_in, float *x_out, const float *weights, 
    int BATCH, int seq_len, int d_model, int d_out
) {
    matmul(x_in, weights, x_out, BATCH, seq_len, d_model, d_out);
}

void elementwise_mul(float *a, const float *b, int n) {
    #pragma omp parallel for simd
    for (int i = 0; i < n; i++) {
        a[i] *= b[i];
    }
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