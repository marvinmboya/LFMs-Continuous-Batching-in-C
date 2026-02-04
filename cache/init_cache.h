#ifndef CACHE_H
#define CACHE_H
#include <stdlib.h>
#include <string.h>
#include "../model/types.h"
#include "../model/utils.h"

typedef struct {
    float **conv_state;
    float **k_cache;
    float **v_cache;
} CBuf;

void create_cache_buffers(CBuf *bufs, LFM2Config config, int batch);
void destroy_cache_buffers(CBuf *bufs);
int get_seq_len();
#endif