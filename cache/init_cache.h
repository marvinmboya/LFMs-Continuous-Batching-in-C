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
    int *cache_seq_len;
} CBuf;

void create_cache_buffers(CBuf *bufs, LFM2Config config, int batch);
void update_cache(
    CBuf *bufs, LFM2Config *config, const float *k, 
    const float *v, int start, int batch, int seq_len, int idx
);
void destroy_cache_buffers(CBuf *bufs);
#endif