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

void read_bin_map_data(const char *weights_path, Weights_Meta *meta, size_t n) {
    int fd = open(weights_path, O_RDONLY);
    if (fd == -1) PERR("file open error!"); 
    struct stat sb;
    if (fstat(fd, &sb) == -1) PERR("file size error!");
    size_t fsize = sb.st_size; // no. elements * uint16
    size_t read_contents = fsize / sizeof(uint16_t);
    if (n != read_contents) PERR("read not expected");
    void *weights_map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (weights_map == MAP_FAILED) PERR("error mapping file");
    uint16_t* weights_data = (uint16_t*)weights_map;
    meta->udata = weights_data;
    meta->size = fsize;
    close(fd);
}

float *convert_bin_float(uint16_t *udata, size_t n){
    float *fdata = malloc(n * sizeof(float));
    if (!fdata) PERR("fdata mem error!");
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        fdata[i] = bf16_to_float32(udata[i]);
    }
    free(udata);
    return fdata;
}

void convert_bin_map_float(Weights_Meta *meta, size_t n) {
    float *fdata = mmap(NULL, n * sizeof(float), PROT_READ | PROT_WRITE, 
                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (!fdata) PERR("fdata mem error!");
    for (int i = 0; i < n; i++){
        fdata[i] = bf16_to_float32(meta->udata[i]);
    }
    meta->fdata = fdata;
    munmap(meta->udata, meta->size);
    meta->udata = NULL;
    meta->size = n * sizeof(float);
}

float* load_data(const char *data_path, size_t n) {
        uint16_t *udata = read_bin_data(data_path, n);
        float *fdata = convert_bin_float(udata, n);
        return fdata;
}

void load_map_data(const char *weights_path, Weights_Meta *meta, size_t n) {
    read_bin_map_data(weights_path, meta, n);
    convert_bin_map_float(meta, n);
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