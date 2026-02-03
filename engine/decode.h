#ifndef DECODE_H
#define DECODE_H
#include <stdio.h>
#include <string.h>
#include "../model/types.h"

int decode_next_token(Buf *buf, int seq_len, int n_vocab);
void get_logits(const float *out, float *logits, int seq_len, int n_vocab);
int sample_next_token(float *logits, int n_vocab);
int max_index(float *array, size_t N);
#endif