#include "backbone.h"
static const int is_attn_check[16] = {
    0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0
};
void swap(float *p1, float *p2);

void backbone(
    float *x, float *x_out, LFM2Config *config,
    GQAWeights *gqa_w, GSCWeights *gsc_w,
    int batch, int seq_len, int d_model, int k_size
){
    for (int i = 0; i < 16; i++) {
        compute_rms_norm(x, gqa_w->q_norm, // CHANGE THIS
            batch * seq_len * d_model, d_model);
        if (is_attn_check[i]) {
            gqattention(x, x_out, config, gqa_w, batch, seq_len);
        } else {
            gscblock(x, x_out, gsc_w, batch, seq_len, d_model, k_size);
        }
        swap(x, x_out);
    }

}

void swap(float *p1, float *p2) {
    int temp;
    temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}