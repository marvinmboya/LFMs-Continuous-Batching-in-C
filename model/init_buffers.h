#ifndef INIT_BUFFERS_H
#define INIT_BUFFERS_H
#include "utils.h"
#include "types.h"
#include <math.h>

void create_model_buffers(Buf *buf, LFM2Config *config, int batch, int seq_len);
#endif