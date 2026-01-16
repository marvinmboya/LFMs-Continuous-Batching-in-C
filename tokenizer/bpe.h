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
#include <string.h>
#include "types.h"

int* encode(Tokenizer *tok, const char *text, int *out_len);
char* decode(Tokenizer *tok, int *ids, int len);
Tokenizer* init_tok_special_toks(char *tok_path);
#endif