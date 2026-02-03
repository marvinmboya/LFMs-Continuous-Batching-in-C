#ifndef GQA_H
#define GQA_H
#include "linear.h"
#include "sdpa.h"
#include "../rmsnorm.h"
#include "../types.h"
#include "utils.h"
#include "../rope.h"

#include <float.h>

void gqattention(
    float *x_in, LFM2Config *config, GQAWeights *gqa_weights, Buf *buffers, int BATCH, int seq_len
);
#endif