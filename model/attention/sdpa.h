#ifndef SDPA_H
#define SDPA_H
#include <stdlib.h> // size_t
#include "linear.h"
#include "utils.h"
#include "../utils.h"

void sdpattention(Buf *buf, int batch, int seq_len, int heads, int head_dim);
void softmax_last(const float *in, float *out, int B, int H, int S, int HD);
#endif