import json
import itertools
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = "your json data path"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

inputs = [x["input"].strip().lower() for x in data]

def tokenize(txt):
    return re.findall(r'\b\w+\b', txt.lower())

def get_ngram_div(texts, n=1):
    ngrams = []
    for t in texts:
        toks = tokenize(t)
        ngrams.extend(zip(*[toks[i:] for i in range(n)]))
    total = len(ngrams)
    unique = len(set(ngrams))
    return unique / total if total > 0 else 0

def avg_cosine_sim(texts):
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(texts)
    sim = cosine_similarity(tfidf)
    mask = ~np.eye(sim.shape[0], dtype=bool)
    return np.mean(sim[mask])

def rep_rate(texts):
    tokens = [tok for t in texts for tok in tokenize(t)]
    total = len(tokens)
    uniq = len(set(tokens))
    return 1 - (uniq / total if total else 0)

uni = get_ngram_div(inputs, 1)
bi = get_ngram_div(inputs, 2)
tri = get_ngram_div(inputs, 3)
cos_sim = avg_cosine_sim(inputs)
rep = rep_rate(inputs)

print()
print("TEXT DIVERSITY CHECK")
print("---------------------")
print(f"Unigram     : {uni:.4f}")
print(f"Bigram      : {bi:.4f}")
print(f"Trigram     : {tri:.4f}")
print(f"Cosine Sim  : {cos_sim:.4f}")
print(f"Repetition  : {rep:.4f}")
