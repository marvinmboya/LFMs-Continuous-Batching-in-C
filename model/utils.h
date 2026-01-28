#ifndef ERR_H
#define ERR_H
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h> // fstat(...)
#include <omp.h>
#include <sys/mman.h> // mmap(...)
#include <fcntl.h> // open(...)
#include <unistd.h> // close(...)
#include "types.h"

#define PERR(err) do { \
    perror(err); \
    exit(EXIT_FAILURE); \
} while(0)

float bf16_to_float32(uint16_t b);
int get_suff_threads();
void get_threads_info();
size_t get_file_size(FILE *f);
size_t get_file_size_v2(FILE *f);
void load_map_data(const char *weights_path, Weights_Meta *meta, size_t n);
void debug_print_first_five(float *embed_out, int seq_len, int d_model);
void max_elements(float *array, size_t N);
#endif