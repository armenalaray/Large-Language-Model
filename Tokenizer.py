import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        #number of rows in the data set
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
        


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader



with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(dataloader)

inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs Shape:\n", inputs.shape)


#################################################

vocab_size = 50257
output_dim = 256

torch.manual_seed(123)

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(token_embedding_layer.weight)

token_embeddings = token_embedding_layer(inputs)

#espacio vectorial
print(token_embeddings.shape)

#################################################

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

print(pos_embedding_layer.weight)

pos_embeddings = pos_embedding_layer(torch.arange(context_length))

print(pos_embeddings)

#################################################


input_embeddings = token_embeddings + pos_embeddings

print(input_embeddings.shape)

#################################################

inputs = torch.tensor(
 [
    [0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

x_2 = inputs[1]

d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

#1*2
query_2 = x_2 @ W_query

queries = inputs @ W_query
keys = inputs @ W_key
values = inputs @ W_value

print("QUERIES:\n",queries)
print("KEYS:\n",keys.T)
print("KEYS:\n",keys)
print("VALUES:\n",values)

attn_score = queries @ keys.T

print("ATTN SCORES:\n", attn_score)

d_k = keys.shape[-1]

#producto punto es la magnitud del vector y su direccional
#si attn_score_2 es muy negativo las derivadas van a ser 0
#attn_weights_2 = torch.softmax(attn_score_2, dim=-1)

attn_weights = torch.softmax(attn_score / d_k ** 0.5, dim=-1)

print("ATTN WEIGHTS:\n",attn_weights)

context_vec_2 = attn_weights @ values

print("CONTEXT_VEC:\n",context_vec_2)


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        #x son los embeddings
        keys = x @ self.W_key
        values = x @ self.W_value
        queries = x @ self.W_query

        attn_scores = queries @ keys.T

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        #estas haciendo combinaciones lineales de rows
        context_vec = attn_weights @ values

        return context_vec
    








