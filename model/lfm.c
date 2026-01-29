#include "lfm.h"

void LFM2Model(
    int *token_ids, int seq_len, 
    LFM2Config *config, Weights *weights, int batch
) {
    float *embeds_out = malloc(batch * seq_len * config->d_model * sizeof(float));
    compute_embeds(weights->embeds, embeds_out, token_ids, seq_len, config->d_model);
    float *x_out = malloc(batch * seq_len * config->d_model * sizeof(float));
    for (int i = 0; i < 16; i++) {
        backbone(
            embeds_out, x_out, config, weights, 
            batch, seq_len, config->d_model, config->k_size, i
        );
        embeds_out = x_out;
    }
    compute_rms_norm(x_out, weights->rms_out,
        batch * seq_len * config->d_model, config->d_model);
    float *final_out = malloc(batch * seq_len * config->n_vocab * sizeof(float));
    matmul(x_out, weights->lin_out, final_out, batch, seq_len, config->d_model, config->n_vocab);
    printf("Hello LFM Model.\n");
    free(x_out);
    free(embeds_out);
}