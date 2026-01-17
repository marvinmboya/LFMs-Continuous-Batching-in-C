import torch, torch.nn as nn 
from torch import bfloat16, int16

torch.manual_seed(42)
n_vocab, d_model = 2, 3
a = nn.Embedding(n_vocab, d_model, dtype=bfloat16)

in_ = torch.randint(0, n_vocab, (1, 5))
print(a(in_))
a_flat = a.weight.data.clone().flatten()
data = a_flat.cpu().view(dtype=int16)
print(a_flat)
data.numpy().tofile('data.bin')
print(f"Saved {len(data)} bfloat16 values to 'data.bin'")


