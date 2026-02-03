#include "decode.h"

int decode_next_token(Buf *buf, int seq_len, int n_vocab) {
    get_logits(buf->final_out, buf->logits, seq_len, n_vocab);
    int next_token = sample_next_token(buf->logits, n_vocab);
    return next_token;
}

void get_logits(
    const float *out, float *logits, int seq_len, int n_vocab
) {
    memcpy(logits, out + ((seq_len - 1) * n_vocab), n_vocab * sizeof(float));
}
int sample_next_token(float *logits, int n_vocab){
    /* top-k, temperature and multinomial implementations too
    */
    int next_token = max_index(logits, n_vocab);
    return next_token;
}

int max_index(float *array, size_t N) {
    int max_index = 0;
    float max_value = 0.0f;
    for (int i = 1; i < N; ++i) {
        if (array[i] > max_value) {
            max_index = i;
            max_value = array[i];
        }
    }
    return max_index;
}