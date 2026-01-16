#include "bpe.h"

static void codepoint_to_utf8(int cp, char *out);
static void init_byte_tables(Tokenizer *tok);
static int find_token_id(Tokenizer *tok, const char *str);
static int find_merge_rank(Tokenizer *tok, const char *a, const char *b);
static void bpe(Tokenizer *tok, char **word, int *len);
static void bpe_encode_segment(Tokenizer *tok, const char *text, int **tokens, int *len, int *cap);
static void add_special_token(Tokenizer *tok, const char *str, uint32_t id);

Tokenizer* load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "BTOK", 4) != 0) {
        fclose(f);
        return NULL;
    }
    
    Tokenizer *tok = calloc(1, sizeof(Tokenizer));
    init_byte_tables(tok);
    
    fread(&tok->vocab_size, 4, 1, f);
    fread(&tok->merge_count, 4, 1, f);
    
    tok->vocab = malloc(tok->vocab_size * sizeof(Token));
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        uint32_t id;
        uint16_t len;
        fread(&id, 4, 1, f);
        fread(&len, 2, 1, f);
        tok->vocab[i].str = malloc(len + 1);
        fread(tok->vocab[i].str, 1, len, f);
        tok->vocab[i].str[len] = 0;
        tok->vocab[i].id = id;
    }
    
    tok->merges = malloc(tok->merge_count * sizeof(Merge));
    for (uint32_t i = 0; i < tok->merge_count; i++) {
        uint16_t len1, len2;
        fread(&len1, 2, 1, f);
        tok->merges[i].first = malloc(len1 + 1);
        fread(tok->merges[i].first, 1, len1, f);
        tok->merges[i].first[len1] = 0;
        
        fread(&len2, 2, 1, f);
        tok->merges[i].second = malloc(len2 + 1);
        fread(tok->merges[i].second, 1, len2, f);
        tok->merges[i].second[len2] = 0;
        tok->merges[i].rank = i;
    }
    
    fclose(f);
    return tok;
}

Tokenizer* init_tok_special_toks(char *tok_path) {
    Tokenizer *tok = load_tokenizer(tok_path);
    add_special_token(tok, "<|startoftext|>", 1);
    add_special_token(tok, "<|im_start|>", 6);
    add_special_token(tok, "<|im_end|>", 7);
    add_special_token(tok, "<|endoftext|>", 2); 
    return tok;
}

static void add_special_token(Tokenizer *tok, const char *str, uint32_t id) {
    tok->special_token_count++;
    tok->special_tokens = realloc(tok->special_tokens, tok->special_token_count * sizeof(SpecialToken));
    tok->special_tokens[tok->special_token_count - 1].str = strdup(str);
    tok->special_tokens[tok->special_token_count - 1].id = id;
}

int* encode(Tokenizer *tok, const char *text, int *out_len) {
    int cap = 1024, len = 0;
    int *tokens = malloc(cap * sizeof(int));
    
    int pos = 0;
    int text_len = strlen(text);
    // Buffer to hold text between special tokens
    char *buffer = malloc(text_len + 1);
    int buf_len = 0;
    
    while (pos < text_len) {
        int best_match_idx = -1;
        int best_match_len = 0;
        
        // 1. Check if a special token starts at this position
        for (int i = 0; i < tok->special_token_count; i++) {
            const char *s = tok->special_tokens[i].str;
            int slen = strlen(s);
            if (strncmp(text + pos, s, slen) == 0) {
                // If we have multiple matches (e.g. "<|end|>" and "<|end|>."), pick longest
                if (slen > best_match_len) {
                    best_match_len = slen;
                    best_match_idx = i;
                }
            }
        }
        
        // 2. If Special Token Found
        if (best_match_idx != -1) {
            // A. Encode whatever raw text we accumulated so far
            if (buf_len > 0) {
                buffer[buf_len] = 0;
                bpe_encode_segment(tok, buffer, &tokens, &len, &cap);
                buf_len = 0;
            }
            
            // B. Add the special token ID directly
            if (len >= cap) {
                cap *= 2;
                tokens = realloc(tokens, cap * sizeof(int));
            }
            tokens[len++] = tok->special_tokens[best_match_idx].id;
            
            // C. Advance position past the special token
            pos += best_match_len;
        } 
        // 3. If No Special Token, accumulate character
        else {
            buffer[buf_len++] = text[pos++];
        }
    }
    
    // 4. Encode remaining text buffer
    if (buf_len > 0) {
        buffer[buf_len] = 0;
        bpe_encode_segment(tok, buffer, &tokens, &len, &cap);
    }
    
    free(buffer);
    *out_len = len;
    return tokens;
}

char* decode(Tokenizer *tok, int *ids, int len) {
    // 1. Concatenate all token strings (which are UTF-8)
    int buffer_cap = len * 10 + 1024;
    char *utf8_res = malloc(buffer_cap);
    utf8_res[0] = 0;
    
    for (int i = 0; i < len; i++) {
        for (uint32_t j = 0; j < tok->vocab_size; j++) {
            if (tok->vocab[j].id == ids[i]) {
                strcat(utf8_res, tok->vocab[j].str);
                break;
            }
        }
    }
    
    // 2. Decode UTF-8 codepoints back to bytes
    char *out = malloc(strlen(utf8_res) + 1);
    int out_len = 0;
    int i = 0;
    while (utf8_res[i]) {
        unsigned char c = (unsigned char)utf8_res[i];
        int codepoint = 0;
        int seq_len = 0;

        if (c < 0x80) {
            codepoint = c;
            seq_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            codepoint = ((c & 0x1F) << 6) | (utf8_res[i+1] & 0x3F);
            seq_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            codepoint = ((c & 0x0F) << 12) | ((utf8_res[i+1] & 0x3F) << 6) | (utf8_res[i+2] & 0x3F);
            seq_len = 3;
        }
        
        // Map back to original byte
        if (codepoint < 1024 && tok->uni_to_byte[codepoint] != -1) {
            out[out_len++] = (char)tok->uni_to_byte[codepoint];
        } else {
            // Fallback for unknown (should not happen in valid BPE output)
            out[out_len++] = '?'; 
        }
        i += seq_len;
    }
    out[out_len] = 0;
    free(utf8_res);
    return out;
}

static void codepoint_to_utf8(int cp, char *out) {
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        out[1] = 0;
    } else if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = 0;
    } else {
        // We only expect up to ~0x1FF for this specific tokenizer
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = 0;
    }
}

// Initialize byte-level encoding tables
static void init_byte_tables(Tokenizer *tok) {
    int bs[256];
    int cs[256];
    int n = 0, idx = 0;
    // 1. Build the mapping logic (GPT-2 style)
    for (int i = 33; i < 127; i++) bs[idx++] = i;
    for (int i = 161; i < 173; i++) bs[idx++] = i;
    for (int i = 174; i < 256; i++) bs[idx++] = i;
    memcpy(cs, bs, idx * sizeof(int));
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int i = 0; i < idx; i++) {
            if (bs[i] == b) { found = 1; break; }
        }
        if (!found) {
            bs[idx] = b;
            cs[idx] = 256 + n; // Maps to 256+ range
            idx++; n++;
        }
    }
    // 2. Fill the tokenizer structs
    // Initialize uni_to_byte with -1
    for(int i=0; i<1024; i++) tok->uni_to_byte[i] = -1;
    for (int i = 0; i < 256; i++) {
        // Map Codepoint -> Original Byte
        tok->uni_to_byte[cs[i]] = bs[i];        
        // Map Original Byte -> UTF-8 String
        tok->byte_to_uni[bs[i]] = malloc(5);
        codepoint_to_utf8(cs[i], tok->byte_to_uni[bs[i]]);
    }
}

static int find_token_id(Tokenizer *tok, const char *str) {
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        if (strcmp(tok->vocab[i].str, str) == 0)
            return tok->vocab[i].id;
    }
    return -1;
}

static int find_merge_rank(Tokenizer *tok, const char *a, const char *b) {
    for (uint32_t i = 0; i < tok->merge_count; i++) {
        if (strcmp(tok->merges[i].first, a) == 0 && 
            strcmp(tok->merges[i].second, b) == 0)
            return tok->merges[i].rank;
    }
    return -1;
}

static void bpe(Tokenizer *tok, char **word, int *len) {
    while (*len > 1) {
        int best_i = -1, best_rank = 1000000;
        
        for (int i = 0; i < *len - 1; i++) {
            int rank = find_merge_rank(tok, word[i], word[i+1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        char *merged = malloc(strlen(word[best_i]) + strlen(word[best_i+1]) + 1);
        strcpy(merged, word[best_i]);
        strcat(merged, word[best_i+1]);
        free(word[best_i]);
        free(word[best_i+1]);
        word[best_i] = merged;
        for (int i = best_i + 1; i < *len - 1; i++)
            word[i] = word[i+1];
        (*len)--;
    }
}

static void bpe_encode_segment(Tokenizer *tok, const char *text, int **tokens, int *len, int *cap) {
    if (strlen(text) == 0) return;
    int text_len = strlen(text);
    char **word = malloc(text_len * sizeof(char*));
    int word_len = 0;
    // Map bytes to UTF-8 strings
    for (int i = 0; i < text_len; i++) {
        char *utf8_char = tok->byte_to_uni[(unsigned char)text[i]];
        word[word_len] = strdup(utf8_char);
        word_len++;
    }
    // Run BPE
    int current_len = word_len; // use local var for reference
    // Note: You need to update your 'bpe' signature or usage to match
    // assuming 'bpe' from previous fix: void bpe(Tokenizer *tok, char **word, int *len)
    
    // We loop here because standard BPE usually pre-tokenizes by regex (whitespace).
    // Since we don't have regex, we run BPE on the whole segment.
    bpe(tok, word, &current_len);
    
    // Append to main token list
    for (int i = 0; i < current_len; i++) {
        int id = find_token_id(tok, word[i]);
        // If not found in vocab, standard BPE falls back to bytes, 
        // but for simplicity here we skip or mark unknown.
        if (id >= 0) {
            if (*len >= *cap) {
                *cap *= 2;
                *tokens = realloc(*tokens, *cap * sizeof(int));
            }
            (*tokens)[(*len)++] = id;
        }
        free(word[i]);
    }
    free(word);
}

