#include "utils.h"

float bf16_to_float32(uint16_t b) {
    uint32_t bits = (uint32_t)b;
    bits <<= 16;
    return *((float*)&bits); 
}

void get_threads_info() {
    #pragma omp parallel 
    {
        int active_threads = omp_get_num_threads();
        int max_threads = omp_get_max_threads();
        printf("active: %d max: %d threads\n", active_threads, max_threads);
    }
}