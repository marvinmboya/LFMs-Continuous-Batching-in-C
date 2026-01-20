#ifndef UTILSI_H
#define UTILSI_H

void transpose_middle(
    int BATCH, int heads, int seq_len, int head_dim, 
    int *old, int *new);

#endif