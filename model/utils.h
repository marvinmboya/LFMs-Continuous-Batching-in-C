#ifndef ERR_H
#define ERR_H
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h> // fstat(...)
#include <omp.h>
#include <sys/mman.h> // mmap(...)
#include <fcntl.h> // open(...)
#include <unistd.h> // close(...)

#define PERR(err) do { \
    perror(err); \
    exit(EXIT_FAILURE); \
} while(0)

typedef struct {
    uint16_t *udata;
    float *fdata;
    size_t size;
} Weights_Meta;

float bf16_to_float32(uint16_t b);
int get_suff_threads();
void get_threads_info();
size_t get_file_size(FILE *f);
size_t get_file_size_v2(FILE *f);
uint16_t *read_bin_data(const char *data_path, size_t n);
float *convert_bin_float(uint16_t *udata, size_t n);
float* load_data(const char *data_path, size_t n);
void read_bin_map_data(const char *weights_path, Weights_Meta *meta, size_t n);
void convert_bin_map_float(Weights_Meta *meta, size_t n);
void load_map_data(const char *weights_path, Weights_Meta *meta, size_t n);
void debug_print_first_five(float *embed_out, int seq_len, int d_model);
#endif