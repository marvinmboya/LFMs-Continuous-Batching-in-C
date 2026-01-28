#include "utils.h"

float bf16_to_float32(uint16_t b) {
    uint32_t bits = (uint32_t)b;
    bits <<= 16;
    return *((float*)&bits); 
}

int get_suff_threads(){
    int suff_threads = omp_get_max_threads() * 2 / 3;
    return suff_threads;
}
void get_threads_info() {
    #pragma omp parallel 
    {
        int active_threads = omp_get_num_threads();
        int max_threads = omp_get_max_threads();
        printf("active: %d max: %d threads\n", active_threads, max_threads);
    }
}

size_t get_file_size(FILE *f){
    fseek(f, 0, SEEK_END);
    size_t fs = ftell(f);
    rewind(f); // or fseek(f, 0, SEEK_SET); this returns error arg
    return fs;
}

size_t get_file_size_v2(FILE *f){
    int fd = fileno(f);
    struct stat sb;
    if (fstat(fd, &sb) == -1) PERR("Failed to get file size");
    return sb.st_size;
}

void load_map_data(const char *weights_path, Weights_Meta *meta, size_t n) {
    int fd = open(weights_path, O_RDONLY);
    if (fd == -1) PERR("file open error!"); 
    struct stat sb;
    if (fstat(fd, &sb) == -1) PERR("file size error!");
    size_t fsize = sb.st_size; // no. elements * float
    size_t read_contents = fsize / sizeof(float);
    if (n != read_contents) PERR("read not expected");
    void *weights_map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (weights_map == MAP_FAILED) PERR("error mapping file");
    float* weights_data = (float*)weights_map;
    meta->fdata = weights_data;
    meta->size = fsize;
    meta->udata = NULL;
    close(fd);
}

void debug_print_first_five(float *embed_out, int seq_len, int d_model) {
    int idx = 0;
    for (int i = 0; i < seq_len; i++){
        for (int j = 0; j < d_model; j++){
            idx = i * d_model + j;
            if (idx >= i * d_model && idx < i * d_model + 5){
                printf("%.4f, ", embed_out[idx]);
            }
        }
        printf("\n");
    }
}

void max_elements(float *array, size_t N) {
    float max_value = array[0];
    int max_index = 0;
    for (int i = 1; i < N; ++i) {
        if (array[i] > max_value ) {
            max_index = i;
            max_value = array[i];
        }
    }
    printf("INDEX: %d VAL: %f\n", max_index, max_value);
}