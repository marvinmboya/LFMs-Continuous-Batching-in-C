#ifndef LFM_H
#define LFM_H
#include <stdio.h>
#include "types.h"
#include "embed.h"
#include "backbone/backbone.h"

void LFM2Model(
    Weights *weights, Buf *buffers, LFM2Config *config, int *token_ids, int seq_len, int batch
);
#endif