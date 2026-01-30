#ifndef MODEL_TYPES
#define MODEL_TYPES
#include <stdint.h>

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

typedef struct {
    float *q;
    float *k;
    float *v;
    float *scores;
    float *norm_scores;
    float *attn_out;
    float *B;
    float *C;
    float *x;
    float *BCx;
    float *BCx_t;
    float *conv_out;
    float *conv_out_sliced;
    float *conv_out_t;
    float *embeds_out;
    float *x_out;
    float *out_w1;
    float *out_v;
    float *final_out;
} Buf;
#endif