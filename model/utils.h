#ifndef ERR_H
#define ERR_H
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h> // fstat(...)
#include <omp.h>

#define PERR(err) do { \
    perror(err); \
    exit(EXIT_FAILURE); \
} while(0)

float bf16_to_float32(uint16_t b);
int get_suff_threads();
void get_threads_info();
size_t get_file_size(FILE *f);
size_t get_file_size_v2(FILE *f);
uint16_t *read_bin_data(const char *data_path, size_t n);
float *convert_bin_data_float(uint16_t *udata, size_t n);
#endif