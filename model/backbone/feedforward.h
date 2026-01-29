#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H
#include <omp.h>
#include <math.h>
#include "utils.h"
#include "../attention/utils.h"
#include "../attention/linear.h"

void feedforward(
    float *x_in, float *x_out, const float *w1, const float *v, 
    const float *w2, int batch, int seq_len, int d_model, int d_hidden
);
void silu(float *x, size_t n);
#endif