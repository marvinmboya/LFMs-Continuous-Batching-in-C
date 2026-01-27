#ifndef MODEL_TYPES
#define MODEL_TYPES

typedef struct {
    int n_vocab;
    int d_model;
    int d_hidden;
    int n_layers;
    int context_len;
    int heads;
    int head_dim;
    int kv_groups;
    int k_size;
} LFM2Config;

typedef struct {
    uint16_t *udata;
    float *fdata;
    size_t size;
} Weights_Meta;

typedef struct {
    Weights_Meta conv;
    Weights_Meta w1;
    Weights_Meta w2;
} GSCMetaWeights;

typedef struct {
    Weights_Meta wqkv;
    Weights_Meta wo;
    Weights_Meta q_norm;
    Weights_Meta k_norm;
} GQAMetaWeights;

typedef struct {
    float *conv;
    const float *w1;
    const float *w2;
} GSCWeights;

typedef struct {
    float *wqkv;
    float *wo;
    float *q_norm;
    float *k_norm;
} GQAWeights;

typedef struct {
    float *embeds;
    float *rms_out;
    float *lin_out;
    float **rms_before;
    float **rms_after;
    float **ffw1;
    float **ffv;
    float **ffw2;
    GSCWeights *gsc; 
    GQAWeights *gqa;
} Weights;
#endif