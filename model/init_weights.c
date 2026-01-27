#include "init_weights.h"

#define N_LAYERS 16
static const int is_attn_check[16] = {0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0};

static Weights_Meta embeds_meta;
static Weights_Meta rms_out_meta;
static Weights_Meta lin_out_meta;
static GSCMetaWeights gsc_weights[N_LAYERS];
static GQAMetaWeights gqa_weights[N_LAYERS];
static Weights_Meta rms_norms_before[N_LAYERS];
static Weights_Meta rms_norms_after[N_LAYERS];
static Weights_Meta ffw1s[N_LAYERS];
static Weights_Meta ffvs[N_LAYERS];
static Weights_Meta ffw2s[N_LAYERS];

void create_weights(LFM2Config *config, Weights *model_weights){
    int n_vocab = config->n_vocab, 
        d_model = config->d_model, 
        d_hidden = config->d_hidden, 
        head_dim = config->head_dim,
        d_out = config->heads * head_dim, 
        kv_d_out = config->kv_groups * head_dim, 
        k_size = config->k_size;
    char fpath[100];
    load_map_data(get_path("embed.bin", fpath), &embeds_meta, n_vocab * d_model);
    
    char s[50]; int n; 
    for (int i = 0; i < N_LAYERS; i++){
        if (is_attn_check[i]) {
            sprintf(s, "wqkv_%d.bin", i);
            n = (d_model * d_out) + (d_model * kv_d_out)*2;
            load_map_data(get_path(s, fpath), &gqa_weights[i].wqkv, n);
            sprintf(s, "wo_%d.bin", i);
            n = d_model * d_out;
            load_map_data(get_path(s, fpath), &gqa_weights[i].wo, n);
            n = head_dim;
            sprintf(s, "qnorm_%d.bin", i);
            load_map_data(get_path(s, fpath), &gqa_weights[i].q_norm, n);
            sprintf(s, "knorm_%d.bin", i);
            load_map_data(get_path(s, fpath), &gqa_weights[i].k_norm, n);
        } else {
            sprintf(s, "gate_w1_%d.bin", i);
            n = d_model * d_model * 3;
            load_map_data(get_path(s, fpath), &gsc_weights[i].w1, n);
            sprintf(s, "gate_w2_%d.bin", i);
            n = d_model * d_model;
            load_map_data(get_path(s, fpath), &gsc_weights[i].w2, n);
            sprintf(s, "gate_conv_%d.bin", i);
            n = d_model * k_size;
            load_map_data(get_path(s, fpath), &gsc_weights[i].conv, n);
        }
        sprintf(s, "rms_before_%d.bin", i);
        load_map_data(get_path(s, fpath), &rms_norms_before[i], d_model);
        sprintf(s, "rms_after_%d.bin", i);
        load_map_data(get_path(s, fpath), &rms_norms_after[i], d_model);
        sprintf(s, "ffw1_%d.bin", i);
        n = d_model * d_hidden;
        load_map_data(get_path(s, fpath), &ffw1s[i], n);
        sprintf(s, "ffv_%d.bin", i);
        load_map_data(get_path(s, fpath), &ffvs[i], n);
        sprintf(s, "ffw2_%d.bin", i);
        load_map_data(get_path(s, fpath), &ffw2s[i], n);
    }
    load_map_data(get_path("rms_out.bin", fpath), &rms_out_meta, d_model);
    load_map_data(get_path("lin_out.bin", fpath), &lin_out_meta, d_model * n_vocab);

    create_fweights(model_weights, 
    &embeds_meta, rms_norms_before, rms_norms_after, 
    gqa_weights, gsc_weights, ffw1s, ffvs,  ffw2s, 
    &rms_out_meta, &lin_out_meta);
}

static void create_fweights(
    Weights *model_weights,
    Weights_Meta *embeds_meta, Weights_Meta *rms_norms_before, 
    Weights_Meta *rms_norms_after, GQAMetaWeights *gqa_weights, 
    GSCMetaWeights *gsc_weights, Weights_Meta *ffw1s,
    Weights_Meta *ffvs, Weights_Meta *ffw2s, 
    Weights_Meta *rms_out_meta, Weights_Meta *lin_out_meta
){
    model_weights->rms_before = malloc(N_LAYERS * sizeof(float*));
    model_weights->rms_after = malloc(N_LAYERS * sizeof(float*));
    model_weights->ffw1 = malloc(N_LAYERS * sizeof(float*));
    model_weights->ffv = malloc(N_LAYERS * sizeof(float*));
    model_weights->ffw2 = malloc(N_LAYERS * sizeof(float*));
    model_weights->gsc = malloc(N_LAYERS * sizeof(GSCWeights));
    model_weights->gqa = malloc(N_LAYERS * sizeof(GQAWeights));

    model_weights->embeds = embeds_meta->fdata;
    model_weights->rms_out = rms_out_meta->fdata;
    model_weights->lin_out = lin_out_meta->fdata;
    for (int i = 0; i < N_LAYERS; i++) {
        model_weights->rms_before[i] = rms_norms_before[i].fdata;
        model_weights->rms_after[i] = rms_norms_after[i].fdata;
        model_weights->ffw1[i] = ffw1s[i].fdata;
        model_weights->ffv[i] = ffvs[i].fdata;
        model_weights->ffw2[i] = ffw2s[i].fdata;
        if (is_attn_check[i]) {
            model_weights->gqa[i].wqkv = gqa_weights[i].wqkv.fdata;
            model_weights->gqa[i].wo = gqa_weights[i].wo.fdata;
            model_weights->gqa[i].q_norm = gqa_weights[i].q_norm.fdata;
            model_weights->gqa[i].k_norm = gqa_weights[i].k_norm.fdata;
        } else {            
            model_weights->gsc[i].w1 = gsc_weights[i].w1.fdata;
            model_weights->gsc[i].w2 = gsc_weights[i].w2.fdata;
            model_weights->gsc[i].conv = gsc_weights[i].conv.fdata;
        }
    }
}

static void destroy_fweights(Weights *model_weights) {
    // embeds, rms_out and lin_out are directly munmapped
    free(model_weights->rms_before);
    free(model_weights->rms_after);
    free(model_weights->ffw1);
    free(model_weights->ffv);
    free(model_weights->ffw2);
    free(model_weights->gsc);
    free(model_weights->gqa);
    memset(model_weights, 0, sizeof(Weights));
}

static void safe_unmap(Weights_Meta *m) {
    if (m && m->fdata) {
        munmap(m->fdata, m->size);
        m->fdata = NULL;
    }
}

void destroy_weights(Weights *model_weights){
    destroy_fweights(model_weights);
    safe_unmap(&embeds_meta);
    safe_unmap(&rms_out_meta);
    safe_unmap(&lin_out_meta);
    for (int i = 0; i < N_LAYERS; i++) {
        safe_unmap(&rms_norms_before[i]);
        safe_unmap(&rms_norms_after[i]);
        safe_unmap(&ffw1s[i]);
        safe_unmap(&ffvs[i]);
        safe_unmap(&ffw2s[i]);
        
        if (is_attn_check[i]) {
            safe_unmap(&gqa_weights[i].wqkv);
            safe_unmap(&gqa_weights[i].wo);
            safe_unmap(&gqa_weights[i].q_norm);
            safe_unmap(&gqa_weights[i].k_norm);
        } else {
            safe_unmap(&gsc_weights[i].w1);
            safe_unmap(&gsc_weights[i].w2);
            safe_unmap(&gsc_weights[i].conv);
        }
    }
}

char *get_path(char *wname, char *full_path){
    sprintf(full_path, "files/weights/%s", wname);
    return full_path;
}