import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import os

from tokenizer import SimpleTokenizer
from dataset import IntentDataset
from model import IntentClassifier

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

vocab_path = r'C:\Users\shini\3D Objects\gf_chatbot\data\processed\vocab.json'
data_path = r'C:\Users\shini\3D Objects\gf_chatbot\data\raw\datasets.json'
label_map_path = r'C:\Users\shini\3D Objects\gf_chatbot\data\processed\label_map.json'
model_save_path = r'C:\Users\shini\3D Objects\gf_chatbot\models\configs\model.pt'

max_len = 32
batch_size = 32
embedding_dim = 64
epochs = 50
patience = 5
learning_rate = 1e-3
weight_decay = 1e-5

tokenizer = SimpleTokenizer(vocab_path, max_length=max_len)
dataset = IntentDataset(data_path, tokenizer, label_map_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

with open(label_map_path, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
num_classes = len(label_map)

model = IntentClassifier(vocab_size=len(tokenizer.token_to_id), num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print("Model improved. Saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break
