#include "embed.h"

static void *embed_map = NULL;
static size_t embed_size = 0;

float* compute_embed(float *embed_data, int *seq, int seq_len, int d_model) {
    float *embed_out = malloc(seq_len * d_model * sizeof(float));
    float val = 0.0f;
    size_t block_index;

    for (int i = 0; i < seq_len; i++){
        float *slice = &embed_data[seq[i] * d_model];
        block_index = i * d_model;
        #pragma omp parallel for
        for (int j = 0; j < d_model; j++){
            embed_out[block_index + j] = slice[j];
        }
    }
    return embed_out;
}