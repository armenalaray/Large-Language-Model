#esta madre no es ortogonal!

inputs = torch.tensor(
 [
    [0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

#esta solo se fija en la dim 0
attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])

print("UNINITIALIZED W:\n",attn_scores)

#6*6
attn_scores = inputs @ inputs.T

print("INITIALIZED W:\n", attn_scores)

#6*6
attn_weights = torch.softmax(attn_scores, dim=-1)

print("ATENTION W:\n", attn_weights)

print("ATTENTION SUM:\n",attn_weights.sum(dim=-1))

print("\n")

for w, i in zip(attn_weights, inputs):
    print(w, i)

print(inputs.T) 

print("\n")

print(attn_weights.T) 

print("\n")

#es una combinacion lineal de estos vectores
all_context_vecs = attn_weights @ inputs
all_context_vecs = inputs.T @ attn_weights.T

#3*6
print(all_context_vecs)



