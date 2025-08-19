import urllib.request
import re

url = ("https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
 "the-verdict.txt")

file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

class SimpleTokenizerV1:

    def __init__(self, vocab):
        self.str_to_int = vocab
        #note: los ints no se repiten
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        ids = [self.str_to_int[s] for s in preprocessed]

        return ids


    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text
    
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        #note: los ints no se repiten
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]

        return ids


    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)

        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(list(set(preprocessed)))

all_tokens.extend(["<|endoftext|>", "<|unk|>"])

print(all_tokens)

vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(vocab)


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)


tokenizer = SimpleTokenizerV2(vocab)

print(tokenizer.decode(tokenizer.encode(text)))

################################################
#OTHER

from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)

print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4

for i in range(1,context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "----->", desired)



"""
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("NORMALIZED W:",attn_weights_2_tmp)
print("NORMALIZED W SUM:",attn_weights_2_tmp.sum())

#dan 1
#a valores grandes les da mucha importancia y a valores pequeños les da menos importancia
#nunca son negativos siempre son positivos
#los esta mapeando al lado positivo de menor a mayor creciendo exponencialmente!

def softmaxNaive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2 = softmaxNaive(attn_scores_2)
print("NORMALIZED W:",attn_weights_2)
print("NORMALIZED W SUM:",attn_weights_2.sum())
"""



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

query = inputs[1]


attn_scores_2 = torch.empty(inputs.shape[0])

print("UNINITIALIZED W:",attn_scores_2)

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print("INITIALIZED W:",attn_scores_2)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("NORMALIZED W:",attn_weights_2)
print("NORMALIZED W SUM:",attn_weights_2.sum())

print(query.shape)

context_vec_2 = torch.zeros(query.shape)

print(context_vec_2)

for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)

#############################################################
#symetric matrix
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)


