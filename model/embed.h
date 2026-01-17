#ifndef EMBED_H
#define EMBED_H
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h> // open(...)
#include <unistd.h> // close(...)
#include <sys/stat.h> // fstat(...)
#include <sys/mman.h> // mmap(...)
#include <omp.h>

#include "types.h"
#include "utils.h"

uint16_t* init_embed(const char *embed_path, LFM2Config *config);
float* compute_embed(uint16_t *embed_data, int *seq, int seq_len, int d_model);
void free_embed();
#endif