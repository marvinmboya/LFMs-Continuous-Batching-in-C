#ifndef GQA_H
#define GQA_H
#include "linear.h"
#include "utils.h"
#include "../types.h"

void gqattention(float *x_in, LFM2Config *config, float *qkv_weights, int BATCH, int seq_len);
#endif