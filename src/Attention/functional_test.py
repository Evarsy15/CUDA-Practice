import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

custom_kernels = load(name='flash_attn', sources=['main.cpp', 'flash_attn.cu', 'naive_attn.cu'], extra_cuda_cflags=['-O2'])

batch_size = 4
n_head = 12
seq_len = 2048
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

def torch_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def scaled_dot_product(q, k):
    return (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))

def softmax(S):
    return F.softmax(S, dim=-1)

def matrix_multiply(P, V):
    return P @ V

# Scaled-Dot-Product
S_torch = scaled_dot_product(q, k)
S_naive = custom_kernels.scaled_dot_product(q, k)
print(S_torch[0][0]); print(S_naive[0][0])
print('scaled-dot-product values sanity check:', torch.allclose(S_naive, S_torch, rtol=0, atol=1e-02))

# Softmax
P_torch = softmax(S_torch)
custom_kernels.softmax(S_naive)
print(P_torch[0][1])
print(S_naive[0][1])
print('softmax values sanity check:', torch.allclose(S_naive, P_torch, rtol=0, atol=1e-02))

O_torch = matrix_multiply(P_torch, v)
O_naive = matrix_multiply(S_naive, v)
print(O_torch[0][1]); print(O_naive[0][1])
print('mat-mul values sanity check:', torch.allclose(O_naive, O_torch, rtol=0, atol=1e-02))