#include "lfm.h" 
#include "bpe.h"
// also can use <lfm.h>

int main() {
    char *tok_path = "files/tokenizer.json";
    BPE(tok_path);
    LFM();
    printf("Hello Main Program.\n");
}