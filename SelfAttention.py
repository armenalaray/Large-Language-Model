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
    


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
#print(sa_v1(inputs))


############################################################################################



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



"""
# x     W
#6*3 X 3*2
#6*2
#2*6*3 X 3*2
#estas combinando linealmente las rows de la matrix W
#la bachita te dice las combinaciones!

b, num_tokens, d_in = batch.shape

keys = ca.W_key(batch)
queries = ca.W_query(batch)
values = ca.W_value(batch)

attn_scores = queries @ keys.transpose(1,2)

print(mask.bool())
print(mask.bool()[:num_tokens, :num_tokens])

print(attn_scores.masked_fill_(mask.bool()[:num_tokens, :num_tokens], -torch.inf))

attn_scores = attn_scores / keys.shape[-1] ** 0.5

print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)

print(attn_weights)

attn_weights = dropout(attn_weights)

print(attn_weights)
print(values)

#2* 6*6 X 2* 6*2
context_vectors = attn_weights @ values

print("CONTEXT VECTORS:\n",context_vectors)

"""


