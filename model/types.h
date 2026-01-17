#ifndef MODEL_TYPES
#define MODEL_TYPES


typedef struct {
    int n_vocab;
    int d_model;
    int n_layers;
    int context_len;
} LFM2Config;

#endif