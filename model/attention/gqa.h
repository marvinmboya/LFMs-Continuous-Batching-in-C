#ifndef GQA_H
#define GQA_H
#include "linear.h"
#include "sdpa.h"
#include "../rmsnorm.h"
#include "../types.h"
#include "utils.h"

void gqattention(
    float *x_in, LFM2Config *config, float *qkv_weights, 
    float *q_norm, float *k_norm, int BATCH, int seq_len);

#endif