#include "lfm.h" 
// also can use <lfm.h>
#include "bpe.h"
#include "model/types.h"
#include "model/embed.h"

int main() {
    LFM2Config config = {
        .n_vocab = 65536, .d_model = 1024, 
        .context_len = 32000, .n_layers = 16, 
    };
    char *tok_path = "files/tokenizer.bin";
    char *embed_path = "files/embed_data.bin";

    Tokenizer *tok = init_tok_special_toks(tok_path);
    uint16_t *embeds = init_embed(embed_path, &config);

    const char *text = "<|startoftext|>Hello world! And again!\n<|im_start|>What's the new image?<|im_end|>assistant";
    int len;
    int *tokens = encode(tok, text, &len);
    for (int i = 0; i < len; i++) printf("%d ", tokens[i]);
    printf("\n");
    char *decoded = decode(tok, tokens, len);
    printf("%s\n", decoded);

    int seq[] = {32433, 13522, 14949, 25594, 35943};
    int seq_len = sizeof(seq) / sizeof(seq[0]);
    int d_model = config.d_model;
    float *embed_out = compute_embed(embeds, seq, seq_len, d_model);
        printf("EMBED OUT:\n");
    for (int i = 0; i < seq_len; i++){
        for (int j = 0; j < d_model; j++){
            int idx = i * d_model + j;
            if (idx >= i * d_model && idx < i * d_model + 5){
                printf("%.4f, ", embed_out[idx]);
            }
        }
        printf("\n");
    }

    free(embed_out);
    free_embed();
    free(decoded); free(tokens);
    return 0;
}