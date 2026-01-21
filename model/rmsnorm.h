#ifndef RMS_H
#define RMS_H
#include <string.h>
#include "utils.h"
#include <math.h>

void compute_rms_norm(float *data, float *weights, size_t n, int d_model);

#endif