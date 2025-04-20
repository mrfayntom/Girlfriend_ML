import json
from collections import Counter
import matplotlib.pyplot as plt

path = "path of your json data"

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

inputs = [item['input'].strip().lower() for item in data]
intents = [item['intent'] for item in data]
total = len(inputs)

dupes = [item for item, count in Counter(inputs).items() if count > 1]
dupe_percent = len(dupes) / total * 100

shorties = [x for x in inputs if len(x.split()) < 3]
short_percent = len(shorties) / total * 100

intent_count = Counter(intents)
intent_percent = {k: v / total * 100 for k, v in intent_count.items()}
min_val = min(intent_count.values())
max_val = max(intent_count.values())
imbalance = (max_val - min_val) / max_val * 100

words = []
for x in inputs:
    words.extend(x.split())

uniq_words = set(words)
avg_len = sum(len(x.split()) for x in inputs) / total

print()
print("ANALYSIS REPORT")
print("-----------------------")
print(f"Duplicates: {len(dupes)} ({dupe_percent:.2f}%)")
print(f"Too Short: {len(shorties)} ({short_percent:.2f}%)")
print(f"Unique Words: {len(uniq_words)}")
print(f"Avg Input Len: {avg_len:.2f}")
print(f"Imbalance: {imbalance:.2f}%")
print()

def bar(label, percent):
    bars = int(percent // 2)
    return f"{label:<20}: {'#' * bars} {percent:.2f}%"

print("QUICK VISUAL")
print(bar("Dupes", dupe_percent))
print(bar("Short", short_percent))
print(bar("Imbalance", imbalance))

plt.figure(figsize=(10, 5))
plt.bar(intent_count.keys(), intent_count.values(), color='lightblue')
plt.xticks(rotation=45, ha='right')
plt.title("Intents")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
