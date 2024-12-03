import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

custom_kernels = load(name='flash_attn', sources=['main.cpp', 'flash_attn.cu', 'naive_attn.cu'], extra_cuda_cflags=['-O2'])

batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

def torch_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

torch_result = torch_attn(q, k, v)
naive_result = custom_kernels.naive_attention(q, k, v)
flash_result = custom_kernels.flash_attention(q, k, v)

print('attn values sanity check:', torch.allclose(naive_result, torch_result, rtol=0, atol=1e-02))
print('attn values sanity check:', torch.allclose(flash_result, torch_result, rtol=0, atol=1e-02))
