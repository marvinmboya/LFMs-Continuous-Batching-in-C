#ifndef TYPES_H
#define TYPES_H
#include <stdint.h>

#define MAX_TOKEN_LEN 256
#define MAX_TOKENS 100000
#define MAX_MERGES 50000

typedef struct {
    char *str;
    uint32_t id;
} Token;

typedef struct {
    char *first;
    char *second;
    uint32_t rank;
} Merge;

typedef struct {
    char *str;
    uint32_t id;
} SpecialToken;

typedef struct {
    Token *vocab;
    uint32_t vocab_size;
    Merge *merges;
    uint32_t merge_count;
    char *byte_to_uni[256]; 
    int uni_to_byte[1024]; 
    SpecialToken *special_tokens;
    int special_token_count;
} Tokenizer;
#endif
