#include "model/lfm.h" 
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
#include "model/init_buffers.h"
#include <time.h>
#include "engine/decode.h"
#include "model/rope.h"
#include "cache/init_cache.h"

void test_encode_decode(Tokenizer *tok);

int main(int argc, char **argv) {
    char prompt[200];
    snprintf(prompt, sizeof(prompt), "%s", argv[1]);
    LFM2Config config = {
        .n_vocab = 65536, .d_model = 1024, .d_hidden = 4608,
        .max_seq_len = 2000, .n_layers = 16, .heads = 16, 
        .head_dim = 64, .kv_groups = 8, .k_size = 3,
        .theta_base = 1000000.0f, .eos_token_id = 7
    };
    Weights model_weights;
    Buf model_buffers;
    CBuf cache_buffers;

    int batch = 1;
    char *tok_path = "files/tokenizer.bin";
    omp_set_num_threads(get_suff_threads());
    create_weights(&config, &model_weights);

    Tokenizer *tok = init_tok_special_toks(tok_path);
    char text[300];
    snprintf(text, sizeof(text), "<|startoftext|><|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
    int seq_len;
    int *token_ids = encode(tok, text, &seq_len);
    create_model_buffers(&model_buffers, &config, batch, seq_len);
    create_cache_buffers(&cache_buffers, config, batch);
    compute_rope(
        model_buffers.cos, model_buffers.sin, 
        config.max_seq_len, config.head_dim, 
        config.theta_base);
    struct timespec start, end;
    char *decoded;
    int max_new_tokens = 1000;
    int total_decoded = 0; 
    float avg_seconds_per_token = 0.0f;
    int next_arr[1];
    while (total_decoded < max_new_tokens) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        LFM2Model(&model_weights, &model_buffers, &cache_buffers, &config, token_ids, seq_len, batch);
        int next_token = decode_next_token(&model_buffers, seq_len, config.n_vocab);
        if (next_token == config.eos_token_id) break;
        decoded = decode(tok, &next_token, 1);
        printf("%s", decoded);
        fflush(stdout);
        next_arr[0] = next_token;
        token_ids = next_arr; seq_len = 1;
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) +
                    (end.tv_nsec - start.tv_nsec) / 1e9;
        if (total_decoded > 0) {
            avg_seconds_per_token = 1/(double)total_decoded * (
            ((total_decoded - 1) * avg_seconds_per_token) + elapsed
        );
        }
        total_decoded += 1;
    }
    printf("\n");
    printf("Decode: %.3f Tokens/Second\n", 1.0 / avg_seconds_per_token);
    destroy_cache_buffers(&cache_buffers);
    destroy_weights(&model_weights);
    free(decoded);
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