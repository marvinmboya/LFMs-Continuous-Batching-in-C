#ifndef LFM_H
#define LFM_H
#include <stdio.h>
#include "types.h"
#include "embed.h"
#include "backbone/backbone.h"
#include "../cache/init_cache.h"

void LFM2Model(
    Weights *weights, Buf *buf, CBuf *cache_buf, 
    LFM2Config *config, int *token_ids, int seq_len, int batch
);
#endif