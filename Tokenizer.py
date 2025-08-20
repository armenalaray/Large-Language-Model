import torch
import torch.nn as nn

inputs = torch.tensor(
 [
    [0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        #ya no es afin
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        #6*3 3*2
        #x @ self.W_key

        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        attn_scores = queries @ keys.T

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values

        return context_vec

d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

attn_scores = queries @ keys.T

print(attn_scores)

attn_scores = attn_scores / keys.shape[-1] ** 0.5

attn_weights = torch.softmax(attn_scores, dim=-1)

print("ATTN WEIGHTS:\n",attn_weights)

#print(attn_weights.sum(dim=-1))

context_length = attn_scores.shape[0]

mask_simple = torch.tril(torch.ones(context_length, context_length))

print("MASK:\n",mask_simple)

#hadamard product
mask_simple = attn_weights*mask_simple

print("MASKED SIMPLE:\n", mask_simple)

row_sums = mask_simple.sum(dim=-1, keepdim=True)

masked_simple_norm = mask_simple / row_sums

print("MASKED SIMPLE NORM:\n", masked_simple_norm)

print(masked_simple_norm.sum(dim=-1))





