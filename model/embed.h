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

void compute_embeds(float *embed_data, float *embeds_out, int *seq, int seq_len, int d_model);
#endif