import torch
from torch.utils.data import DataLoader, random_split
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from tokenizer import SimpleTokenizer
from dataset import IntentDataset
from model import IntentClassifier

vocab_path = r'C:\Users\shini\3D Objects\gf_chatbot\data\processed\vocab.json'
data_path = r'C:\Users\shini\3D Objects\gf_chatbot\data\raw\datasets.json'
label_map_path = r'C:\Users\shini\3D Objects\gf_chatbot\data\processed\label_map.json'
model_path = r'C:\Users\shini\3D Objects\gf_chatbot\models\configs\model.pt'

tokenizer = SimpleTokenizer(vocab_path, max_length=32)
dataset = IntentDataset(data_path, tokenizer, label_map_path)

with open(label_map_path, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
id_to_label = {v: k for k, v in label_map.items()}

val_size = int(0.2 * len(dataset))
_, val_data = random_split(dataset, [len(dataset) - val_size, val_size])
val_loader = DataLoader(val_data, batch_size=32)

model = IntentClassifier(vocab_size=len(tokenizer.token_to_id), num_classes=len(label_map))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\n Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in range(len(label_map))]))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(len(label_map)),
       yticks=np.arange(len(label_map)),
       xticklabels=[id_to_label[i] for i in range(len(label_map))],
       yticklabels=[id_to_label[i] for i in range(len(label_map))],
       ylabel='True label',
       xlabel='Predicted label',
       title='Confusion Matrix')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

fig.tight_layout()
plt.savefig(r"C:\Users\shini\3D Objects\gf_chatbot\models\configs\test.png")
plt.show()
