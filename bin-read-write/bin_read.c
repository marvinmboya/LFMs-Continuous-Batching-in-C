#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h> // open(...)
#include <unistd.h> // close(...)
#include <sys/stat.h> // fstat(...)
#include <sys/mman.h> // mmap(...)

#define PERR(err) do { \
    perror(err); \
    exit(EXIT_FAILURE); \
} while(0)

float bf16_to_float32(uint16_t b) {
    uint32_t bits = (uint32_t)b;
    bits <<= 16;
    return *((float*)&bits); 
}

int main() {
    const char* filename = "data.bin";
    int fd = open(filename, O_RDONLY);
    int n_vocab = 65536, d_model = 1024;
    size_t n = n_vocab * d_model;
    if (fd == -1) PERR("Failed to open file"); 

    struct stat sb;
    if (fstat(fd, &sb) == -1) PERR("file size error!");
    size_t fsize = sb.st_size; // no. elements * uint16
    size_t read_contents = fsize / sizeof(uint16_t);
    if (n != read_contents) PERR("read not expected");
    void* mapped = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) PERR("error mapping file");
    uint16_t* data = (uint16_t*)mapped;
    
    int seq[] = {32433, 13522, 14949, 25594, 35943};
    int i = 0, j = 0, seq_len = 5;
    float val = 0.0f;
    size_t block_index; 
    float *embed_out = malloc(seq_len * d_model * sizeof(float));
    for (i = 0; i < seq_len; i++){
        uint16_t* slice = &data[seq[i] * d_model];
        block_index = i * d_model;
        for (j = 0; j < d_model; j++){
            embed_out[block_index + j] = bf16_to_float32(slice[j]);
        }
    }
    printf("EMBED OUT:\n");
    for (i = 0; i < seq_len; i++){
        for (j = 0; j < d_model; j++){
            int idx = i * d_model + j;
            if (idx >= i * d_model && idx < i * d_model + 5){
                printf("%.4f, ", embed_out[idx]);
            }
        }
        printf("\n");
    }

    free(embed_out);
    munmap(mapped, sb.st_size);
    close(fd);

    return 0;
}