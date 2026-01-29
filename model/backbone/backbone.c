#include "backbone.h"
static const int is_attn_check[16] = {
    0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0
};

void backbone(
    float *x, float *x_out, LFM2Config *config, Weights *weights,
    int batch, int seq_len, int d_model, int k_size,
    int layer_index
){
    float *shortcut = malloc(batch * seq_len * d_model * sizeof(float));
    memcpy(shortcut, x, batch * seq_len * d_model * sizeof(float));
    compute_rms_norm(x, weights->rms_before[layer_index],
        batch * seq_len * d_model, d_model);
    if (is_attn_check[layer_index]) {
        gqattention(x, x_out, config, &weights->gqa[layer_index], batch, seq_len);
    } else {
        gscblock(x, x_out, &weights->gsc[layer_index], batch, seq_len, d_model, k_size);
    }
    x = x_out;
    elementwise_add(x, shortcut, batch * seq_len * d_model);
    memcpy(shortcut, x, batch * seq_len * d_model * sizeof(float));
    compute_rms_norm(x, weights->rms_after[layer_index],
        batch * seq_len * d_model, d_model);
    feedforward(x, x_out, weights->ffw1[layer_index], weights->ffv[layer_index], 
        weights->ffw2[layer_index], batch, seq_len, d_model, config->d_hidden);
    elementwise_add(x_out, shortcut, batch * seq_len * d_model);
}