#ifndef GQA_H
#define GQA_H
#include "linear.h"
#include "utils.h"

void gqattention(float *x_in, float *qkv_weights, int group_size);
#endif
