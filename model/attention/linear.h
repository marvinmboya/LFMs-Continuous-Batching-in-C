#ifndef LINEAR_H
#define LINEAR_H
#include <omp.h>

void matmul(
    const float *x_in, const float *weights, float *x_out, int BATCH, int M, int K, int N);
#endif