#ifndef LFM_H
#define LFM_H
#include <stdio.h>
#include "types.h"
#include "embed.h"
#include "backbone/backbone.h"

void LFM2Model(int *token_ids, int seq_len, LFM2Config *config, Weights *weights, int batch);
#endif