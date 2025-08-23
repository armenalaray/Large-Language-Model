import torch
import tiktoken
from GPTModel import GPTModel, generate_text_simple


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

model.eval()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
    

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
    

tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Every effort moves you"

token_ids = generate_text_simple(model, 
                     idx=text_to_token_ids(start_context, tokenizer),
                     max_new_tokens=10,
                     context_size=GPT_CONFIG_124M["context_length"]
                     )


print(token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]


with torch.no_grad():
    logits = model(inputs)

print(logits.shape)

probas = torch.softmax(logits,dim=-1)

print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)

print(token_ids.shape)

print(token_ids_to_text(inputs[0], tokenizer))
print(token_ids_to_text(targets[0], tokenizer))

print(token_ids[0].shape)
print(token_ids[0].flatten().shape)

print(token_ids_to_text(token_ids[0].flatten(), tokenizer))




