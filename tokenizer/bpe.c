#include "bpe.h"

void BPE(char *tok_path) {
    char *data = read_file(tok_path);
    cJSON *root = cJSON_Parse(data);
    CHECK_ERR(root, "json parse error!");
    cJSON_Delete(root);
    free(data);
}

size_t get_file_size(FILE *f) {
  fseek(f, 0, SEEK_END);
  size_t sz = ftell(f);
  rewind(f);
  return sz;
}

static char *read_file(char *fname) {
    FILE *f = fopen(fname, "rb");
    CHECK_ERR(f, "file open error!");
    size_t sz = get_file_size(f);
    char *buf = malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}