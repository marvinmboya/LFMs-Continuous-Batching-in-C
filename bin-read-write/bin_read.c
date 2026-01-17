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
    size_t n = 6;
    if (fd == -1) PERR("Failed to open file"); 

    struct stat sb;
    if (fstat(fd, &sb) == -1) PERR("file size error!");
    size_t fsize = sb.st_size; // no. elements * uint16
    size_t read_contents = fsize / sizeof(uint16_t);
    if (n != read_contents) PERR("read not expected");
    void* mapped = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) PERR("error mapping file");
    uint16_t* data = (uint16_t*)mapped;
    
    printf("Reading all elements\n");
    for (int i=0; i<n; i++){
        float val = bf16_to_float32(data[i]);
        printf("index: %d val=%.4f\n", i, val);
    }
    int seq[] = {0, 0, 1, 0, 1};
    int seq_len = 5, d_model = 3;
    float val = 0.0f;
    for (int i = 0; i < 5; i++){
        uint16_t* slice = &data[seq[i] * d_model];
        for (int j = 0; j < d_model; j++){
            val = bf16_to_float32(slice[j]);
            printf("%.4f, ", val);
        }
        printf("\n");
    }

    munmap(mapped, sb.st_size);
    close(fd);

    return 0;
}