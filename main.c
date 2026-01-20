#include "lfm.h" 
// also can use <lfm.h>
#include "bpe.h"
#include "model/types.h"
#include "model/embed.h"
#include "model/rmsnorm.h"
#include "model/utils.h"

void test_encode_decode(Tokenizer *tok);

int main() {
    LFM2Config config = {
        .n_vocab = 65536, .d_model = 1024, 
        .context_len = 32000, .n_layers = 16, 
    };
    char *tok_path = "files/tokenizer.bin";
    char *embed_path = "files/embed_data.bin";
    omp_set_num_threads(get_suff_threads());

    Tokenizer *tok = init_tok_special_toks(tok_path);
    uint16_t *embeds = init_embed(embed_path, &config);
    test_encode_decode(tok);

    int seq[] = {32433, 13522, 14949, 25594, 35943};
    int seq_len = sizeof(seq) / sizeof(seq[0]);
    int d_model = config.d_model;
    float *embed_out = compute_embed(embeds, seq, seq_len, d_model);
    printf("EMBED OUT:\n");
    debug_print_first_five(embed_out, seq_len, d_model);
    int batch = 1;
    seq_len = 4; d_model = 6;
    int n = batch * seq_len * d_model;
    char buf[20];
    sprintf(buf, "b_%d_data.bin", batch);

    float *data = load_data(buf, n);
    for (int i = 0; i < n; i++){
        printf("%.4f, ", data[i]);
    }
    printf("\n");
    float weights[] = { 1.3525,  0.6863, -0.3278,  0.7950,  0.2815,  0.0562};
    compute_rms_norm(data, weights, n, d_model);
    for (int i = 0; i < n; i++){
        printf("%.4f, ", data[i]);
    }
    printf("\n");
    free(data);
    free(embed_out);
    free_embed();
    return 0;
}

void test_encode_decode(Tokenizer *tok){
    const char *text = "Hello world!";
    int len;
    int *tokens = encode(tok, text, &len);
    for (int i = 0; i < len; i++) printf("%d ", tokens[i]);
    printf("\n");
    char *decoded = decode(tok, tokens, len);
    printf("%s\n", decoded);
    free(decoded); free(tokens);
}