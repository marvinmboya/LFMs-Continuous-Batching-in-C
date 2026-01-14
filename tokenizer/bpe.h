#ifndef BPE_H
#define BPE_H
#define CHECK_ERR(ptr, err) do { \
    if (!(ptr)) { \
    printf("%s\n", err); \
    exit(EXIT_FAILURE); \
    } \
} while(0)

#include <stdio.h>
#include <stdlib.h>
#include "uthash.h"
#include "cJSON.h"

size_t get_file_size(FILE *f);
static char *read_file(char *fname);

void BPE(char *tok_path);

#endif