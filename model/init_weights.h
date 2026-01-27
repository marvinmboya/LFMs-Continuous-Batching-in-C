#ifndef INIT_WEIGHTS_H
#define INIT_WEIGHTS_H
#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "utils.h"

void create_weights(LFM2Config *config, Weights *model_weights);
static void create_fweights(
    Weights *model_weights,
    Weights_Meta *embeds_meta, Weights_Meta *rms_norms_before, 
    Weights_Meta *rms_norms_after, GQAMetaWeights *gqa_weights, 
    GSCMetaWeights *gsc_weights, Weights_Meta *ffw1s,
    Weights_Meta *ffvs, Weights_Meta *ffw2s, 
    Weights_Meta *rms_out_meta, Weights_Meta *lin_out_meta
);
static void destroy_fweights(Weights *model_weights);
void destroy_weights(Weights *model_weights);
char *get_path(char *wname, char *full_path);
#endif