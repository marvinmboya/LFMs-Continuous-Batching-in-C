#ifndef CACHE_H
#define CACHE_H
#include <stdlib.h>
#include "../types.h"

typedef struct {
    float *conv_state;
    float *k_cache;
    float *v_cache;
} CBuf;

int get_seq_len();
#endif