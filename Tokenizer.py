import torch
import torch.nn as nn

torch.manual_seed(123)
batch_example = torch.randn(2,5)


layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())

out = layer(batch_example)

print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print(mean)
print(var)

out_norm = (out - mean) / torch.sqrt(var)

print(out_norm)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

torch.set_printoptions(sci_mode=False)

print(mean)
print(var)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5

        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        #variance is biased
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


ln = LayerNorm(emb_dim=5)

out_ln = ln(batch_example)

print(out_ln)

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)

print(mean)
print(var)