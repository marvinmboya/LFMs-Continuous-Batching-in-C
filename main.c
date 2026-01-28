#include "lfm.h" 
// also can use <lfm.h>
#include "bpe.h"
#include "model/types.h"
#include "model/embed.h"
#include "model/rmsnorm.h"
#include "model/utils.h"
#include "model/attention/utils.h"
#include "model/attention/gqa.h"
#include "model/gsc.h"
#include "model/init_weights.h"
#include <time.h>

void test_encode_decode(Tokenizer *tok);

int main() {
    LFM2Config config = {
        .n_vocab = 65536, .d_model = 1024, .d_hidden = 4608,
        .context_len = 32000, .n_layers = 16, .heads = 16, 
        .head_dim = 64, .kv_groups = 8, .k_size = 3
    };
    Weights model_weights;
    char *tok_path = "files/tokenizer.bin";
    char *embed_path = "files/embed_data.bin";
    omp_set_num_threads(get_suff_threads());
    create_weights(&config, &model_weights);

    int batch = 1;
    Tokenizer *tok = init_tok_special_toks(tok_path);
    // const char *text = "Hello world!";
    const char *text = "<|startoftext|><|im_start|>user\nHello world!<|im_end|>\n<|im_start|>assistant\n";
    int seq_len;
    int *token_ids = encode(tok, text, &seq_len);
    printf("START\n");
    clock_t st = clock();
    LFM2Model(token_ids, seq_len, &config, &model_weights, batch);
    clock_t end = clock();
    printf("time spent: %.7f\n", (double)(end - st) / CLOCKS_PER_SEC);
    destroy_weights(&model_weights);
    free(token_ids);
    free(tok);
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