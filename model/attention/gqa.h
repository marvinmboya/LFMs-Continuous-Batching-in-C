#ifndef GQA_H
#define GQA_H
#include "linear.h"
#include "sdpa.h"
#include "../rmsnorm.h"
#include "../types.h"
#include "utils.h"

void gqattention(
    float *x_in, float *x_out, LFM2Config *config, GQAWeights *gqa_weights, int BATCH, int seq_len);

#endif