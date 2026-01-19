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

uint16_t *read_bin_data(const char *data_path, size_t n){
    FILE *f = fopen(data_path, "rb");
    if (!f) PERR("file open error!");
    size_t fsize = get_file_size_v2(f);
    size_t read_contents = fsize / sizeof(uint16_t);
    if (n != read_contents) PERR("read not expected");
    uint16_t *udata = malloc(n * sizeof(uint16_t));
    if (!udata) PERR("udata mem error!");
    fread(udata, sizeof(uint16_t), n, f);
    fclose(f);
    return udata;
}

float *convert_bin_data_float(uint16_t *udata, size_t n){
    float *fdata = malloc(n * sizeof(float));
    if (!fdata) PERR("fdata mem error!");
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        fdata[i] = bf16_to_float32(udata[i]);
    }
    free(udata);
    return fdata;
}

float* load_data(const char *data_path, size_t n) {
        uint16_t *udata = read_bin_data(data_path, n);
        float *fdata = convert_bin_data_float(udata, n);
        return fdata;
}