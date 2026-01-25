#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <math.h>
#include <omp.h>
void softmax_last(
    const float *in, float *out, int B, int H, int S, int HD);
#endif