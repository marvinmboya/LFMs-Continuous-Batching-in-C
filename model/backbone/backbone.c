#include "backbone.h"
static const int is_attn_check[16] = {
    0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0
};

void backbone(
    float *x, LFM2Config *config, Weights *weights, Buf *buf, CBuf *cache_buf,
    int batch, int seq_len, int decode_start, int d_model, int k_size, int layer_index
){
    float *shortcut = malloc(batch * seq_len * d_model * sizeof(float));
    memcpy(shortcut, x, batch * seq_len * d_model * sizeof(float));
    compute_rms_norm(x, weights->rms_before[layer_index],
        batch * seq_len * d_model, d_model);
    if (is_attn_check[layer_index]) {
        gqattention(
            x, config, &weights->gqa[layer_index], buf, 
            cache_buf, batch, seq_len, decode_start, layer_index
        );
    } else {
        gscblock(
            x, &weights->gsc[layer_index], buf, cache_buf, batch, 
            seq_len, decode_start, layer_index, d_model, k_size
        );
    }
    x = buf->x_out;
    elementwise_add(x, shortcut, batch * seq_len * d_model);
    memcpy(shortcut, x, batch * seq_len * d_model * sizeof(float));
    compute_rms_norm(x, weights->rms_after[layer_index],
        batch * seq_len * d_model, d_model);
    feedforward(x, buf->x_out, weights->ffw1[layer_index], weights->ffv[layer_index], 
        weights->ffw2[layer_index], buf, batch, seq_len, d_model, config->d_hidden);
    elementwise_add(buf->x_out, shortcut, batch * seq_len * d_model);
}