#ifndef ERR_H
#define ERR_H
#include <stdint.h>
#include <stdio.h>
#include <omp.h>

#define PERR(err) do { \
    perror(err); \
    exit(EXIT_FAILURE); \
} while(0)

float bf16_to_float32(uint16_t b);
void get_threads_info();

#endif