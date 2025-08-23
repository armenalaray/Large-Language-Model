import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
import tiktoken

#barch_num * 12 * 1024 * 64 

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


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


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    

class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    

ffn = FeedForward(GPT_CONFIG_124M)

x = torch.rand(2,3,768)

out = ffn(x)

print(out)
print(out.shape)

        
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()

        self.use_shortcut = use_shortcut

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):

        for layer in self.layers:

            layer_output = layer(x)

            if self.use_shortcut and layer_output.shape == x.shape:
                x = x + layer_output
            else:
                x = layer_output
        
        return x

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
modelWithoutShortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

def print_gradients(model, x):
    output = model(x)
    print(output)

    target = torch.tensor([[0.]])

    loss = nn.MSELoss()

    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
        


print_gradients(modelWithoutShortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

print_gradients(model_with_shortcut, sample_input)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]     
            )

        self.ff = FeedForward(cfg)

        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x

        #pre layer norm for better training dynamics
        x = self.norm1(x)

        x = self.att(x)

        x = self.drop_shortcut(x)

        x = x + shortcut

        shortcut = x

        x = self.norm2(x)

        x = self.ff(x)

        x = self.drop_shortcut(x)

        x = x + shortcut

        return x

torch.manual_seed(123)
x = torch.rand(2,4,768)
block = TransformerBlock(GPT_CONFIG_124M)

output = block(x)

print(output)
print(x.shape)
print(output.shape)

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        #unpacked list 
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    #data move it to GPU
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits

batch = torch.tensor(
    [[6109, 3626, 6100, 345], 
     [6109, 1110, 6622, 257]]
    )

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)

print(batch.shape)
print(out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())

#with weight tying 
print(f"{total_params_gpt2:,}")

total_size_bytes = total_params * 4

total_size_mbytes = total_size_bytes / (1024*1024)

print("MB:",total_size_mbytes)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    print(idx.shape)
    for _ in range(max_new_tokens):
        #1024
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)


        logits = logits[:, -1, :]
        #print(logits.shape)

        probas = torch.softmax(logits, dim=-1)
        
        #print(probas.sum(dim=-1, keepdim=True))

        #este es el vocab!
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        #print(idx_next.shape)

        idx = torch.cat((idx, idx_next), dim=1)

        print(idx.shape)

    return idx

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print(encoded)
#print(torch.tensor([encoded]))

encoded_tensor = torch.tensor(encoded).unsqueeze(0)

print(encoded_tensor.shape)
print(encoded_tensor)

model.eval()

out = generate_text_simple(
    model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"]
                     )

print(out)
 
decoded_text = tokenizer.decode(out.squeeze(0).tolist())

print(decoded_text)