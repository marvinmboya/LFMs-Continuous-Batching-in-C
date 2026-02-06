#include "lfm.h"
#include <time.h>

void LFM2Model(
    Weights *weights, Buf *buf, CBuf *cache_buf, 
    LFM2Config *config, int *token_ids, int seq_len, int batch
) {
    int decode_start = cache_buf->cache_seq_len[2];
    compute_embeds(weights->embeds, buf->embeds_out, token_ids, seq_len, config->d_model);
    float *in = buf->embeds_out;
    float *out = buf->x_out;
    for (int i = 0; i < 16; i++) {
        backbone(
            in, config, weights, buf, cache_buf, batch, 
            seq_len, decode_start, config->d_model, config->k_size, i
        );
        in = out; 
    }
    compute_rms_norm(out, weights->rms_out,
        batch * seq_len * config->d_model, config->d_model);
    matmul(out, weights->lin_out, buf->final_out, batch, seq_len, config->d_model, config->n_vocab);
}