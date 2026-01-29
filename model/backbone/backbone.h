#ifndef BACKBONE_H
#define BACKBONE_H
#include <omp.h>
#include "../attention/gqa.h"
#include "../gsc.h"
#include "../types.h"
#include "../rmsnorm.h"
#include "feedforward.h"

void backbone(
    float *x, float *x_out, LFM2Config *config, Weights *weights,
    int batch, int seq_len, int d_model, int k_size, int layer_idx
);
#endif