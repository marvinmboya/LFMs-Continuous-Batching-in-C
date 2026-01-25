#ifndef MODEL_TYPES
#define MODEL_TYPES


typedef struct {
    int n_vocab;
    int d_model;
    int n_layers;
    int context_len;
    int heads;
    int head_dim;
    int kv_groups;
    int k_size;
} LFM2Config;

#endif