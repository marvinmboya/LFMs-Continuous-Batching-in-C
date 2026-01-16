#include "lfm.h" 
// also can use <lfm.h>
#include "bpe.h"

int main() {
    char *tok_path = "files/tokenizer.bin";
    Tokenizer *tok = init_tok_special_toks(tok_path);

    const char *text = "<|startoftext|>Hello world! And again!\n<|im_start|>What's the new image?<|im_end|>assistant";
    int len;
    int *tokens = encode(tok, text, &len);
    for (int i = 0; i < len; i++) printf("%d ", tokens[i]);
    printf("\n");
    char *decoded = decode(tok, tokens, len);
    printf("%s\n", decoded);    
    free(decoded); free(tokens);
    return 0;
}