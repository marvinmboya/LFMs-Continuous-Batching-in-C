#include "embed.h"

float bf16_to_float32(uint16_t b);

static void *embed_map = NULL;
static size_t embed_size = 0;

uint16_t* init_embed(const char *embed_path, LFM2Config *config) {
        int n_vocab = config->n_vocab;
        int d_model = config->d_model;
        size_t n = n_vocab * d_model;
        int fd = open(embed_path, O_RDONLY);
        if (fd == -1) PERR("Failed to open file"); 
        struct stat sb;
        if (fstat(fd, &sb) == -1) PERR("file size error!");
        size_t fsize = sb.st_size; // no. elements * uint16
        embed_size = fsize;
        size_t read_contents = fsize / sizeof(uint16_t);
        if (n != read_contents) PERR("read not expected");
        embed_map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (embed_map == MAP_FAILED) PERR("error mapping file");
        uint16_t* embed_data = (uint16_t*)embed_map;
        close(fd);
        return embed_data;
}

float* compute_embed(uint16_t *embed_data, int *seq, int seq_len, int d_model) {
    float *embed_out = malloc(seq_len * d_model * sizeof(float));
    float val = 0.0f;
    size_t block_index; 

    omp_set_num_threads(8);
    for (int i = 0; i < seq_len; i++){
        uint16_t* slice = &embed_data[seq[i] * d_model];
        block_index = i * d_model;
        #pragma omp parallel for
        for (int j = 0; j < d_model; j++){
            embed_out[block_index + j] = bf16_to_float32(slice[j]);
        }
    }
    return embed_data;
}

void free_embed() {
    if (embed_map) {
        munmap(embed_map, embed_size);
        printf("embeds freed!");
    }
}