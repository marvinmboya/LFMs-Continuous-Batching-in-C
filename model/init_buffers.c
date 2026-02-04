#include "init_buffers.h"

int _get_conv_out_size(int in_size, int k_size, int p_size, int stride);
static void init_malloc(float **buf, size_t n) {
    *buf = malloc(n * sizeof(float));
    if (!(*buf)) PERR("malloc error!");
}

void create_model_buffers(Buf *buf, LFM2Config *config, int batch, int seq_len){
    int d_model = config->d_model,
        heads = config->heads,
        head_dim = config->head_dim,
        d_out = heads * head_dim,
        kv_groups = config->kv_groups,
        kv_d_out = kv_groups * head_dim,
        group_size = (int)(heads / kv_groups),
        q_size = config->d_model * d_out,
        k_size = config->d_model * kv_d_out,
        v_size = k_size;
    // FREQS
    init_malloc(&(buf->cos), config->max_seq_len * config->head_dim / 2);
    init_malloc(&(buf->sin), config->max_seq_len * config->head_dim / 2);
    // GQA
    init_malloc(&(buf->q), batch * seq_len * d_out);
    init_malloc(&(buf->k), batch * seq_len * kv_d_out);
    init_malloc(&(buf->v), batch * seq_len * kv_d_out);
    init_malloc(&(buf->k_expand), batch * seq_len * d_out);
    init_malloc(&(buf->v_expand), batch * seq_len * d_out);
    init_malloc(&(buf->q_t), batch * seq_len * d_out);
    init_malloc(&(buf->k_t), batch * seq_len * kv_d_out);
    init_malloc(&(buf->v_t), batch * seq_len * kv_d_out);

    init_malloc(&(buf->scores), batch * heads * seq_len * seq_len);
    init_malloc(&(buf->norm_scores), batch * heads * seq_len * seq_len);
    init_malloc(&(buf->attn_out), batch * seq_len * d_model);
    init_malloc(&(buf->attn_out_t), batch * seq_len * d_model);
    // GSC
    init_malloc(&(buf->BCx), batch * seq_len * d_model * 3);
    init_malloc(&(buf->BCx_t), batch * seq_len * d_model * 3);
    init_malloc(&(buf->B), batch * seq_len * d_model);
    init_malloc(&(buf->C), batch * seq_len * d_model);
    init_malloc(&(buf->x), batch * seq_len * d_model);

    int co_sz = _get_conv_out_size(seq_len, k_size, k_size - 1, 1);
    init_malloc(&(buf->conv_out), batch * d_model * co_sz);
    init_malloc(&(buf->conv_out_sliced), batch * d_model * seq_len);
    init_malloc(&(buf->conv_out_t), batch * d_model * seq_len);

    init_malloc(&(buf->embeds_out), batch * seq_len * d_model);
    init_malloc(&(buf->x_out), batch * seq_len * d_model);

    init_malloc(&(buf->out_w1), batch * seq_len * config->d_hidden);
    init_malloc(&(buf->out_v), batch * seq_len * config->d_hidden);
    init_malloc(&(buf->final_out), batch * seq_len * config->n_vocab);
    init_malloc(&(buf->logits), batch * config->n_vocab);
}

static void destroy_malloc(float *buf) {
    if (buf) free(buf);
}

void destroy_model_buffers(Buf *buf){
    destroy_malloc(buf->cos);
    destroy_malloc(buf->sin);
    destroy_malloc(buf->q);
    destroy_malloc(buf->k);
    destroy_malloc(buf->v);
    destroy_malloc(buf->k_expand);
    destroy_malloc(buf->v_expand);
    destroy_malloc(buf->q_t);
    destroy_malloc(buf->k_t);
    destroy_malloc(buf->v_t);

    destroy_malloc(buf->scores);
    destroy_malloc(buf->norm_scores);
    destroy_malloc(buf->attn_out);
    destroy_malloc(buf->attn_out_t);
    // GSC
    destroy_malloc(buf->BCx);
    destroy_malloc(buf->BCx_t);
    destroy_malloc(buf->B);
    destroy_malloc(buf->C);
    destroy_malloc(buf->x);
    
    destroy_malloc(buf->conv_out);
    destroy_malloc(buf->conv_out_sliced);
    destroy_malloc(buf->conv_out_t);

    destroy_malloc(buf->embeds_out);
    destroy_malloc(buf->x_out);
    destroy_malloc(buf->out_w1);
    destroy_malloc(buf->out_v);
    destroy_malloc(buf->final_out);
    destroy_malloc(buf->logits);
}

int _get_conv_out_size(int in_size, int k_size, int p_size, int stride){
    p_size = 2 * p_size;
    int _part = (int)floor((in_size + p_size - k_size)/stride);
    return _part + 1;
}