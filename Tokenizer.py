import urllib.request
import re

url = ("https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
 "the-verdict.txt")

file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)


preprocessed = [item.strip() for item in preprocessed if item.strip() ]

preprocessed = sorted(set(preprocessed))

vocab_size = len(preprocessed)

print(vocab_size)

vocab = {token:integer for integer, token in enumerate(preprocessed)}


for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break



