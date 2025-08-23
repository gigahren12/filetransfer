import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Параметры
FILE_PATH = 'train_text.txt'
BATCH_SIZE = 32
SEQ_LEN = 100
EMBED_SIZE = 256
NUM_HEADS = 2
NUM_LAYERS = 1
FFN_HID_DIM = 512
LR = 0.001
EPOCHS = 10

# Подготовка данных
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encoded_text = [char_to_idx[c] for c in text]

# Создание наборов данных
class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

dataset = TextDataset(encoded_text, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Модель трансформера
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ffn_hid_dim, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ffn_hid_dim,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
        x = self.transformer(x, x, tgt_mask=mask)
        return self.fc_out(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(vocab_size, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, FFN_HID_DIM, SEQ_LEN).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Обучение
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch % 100 == 0:
            print(f'Epoch {epoch+1} | Batch {batch} | Loss: {loss.item():.4f}')

    print(f'Epoch {epoch+1} | Average Loss: {total_loss/len(dataloader):.4f}')

# Генерация текста
def generate_text(model, start_str, max_length=1000, temperature=0.8):
    model.eval()
    generated = [char_to_idx[c] for c in start_str]
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor(generated[-SEQ_LEN:]).unsqueeze(0).to(device)
            output = model(x)
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, 1).item()
            generated.append(next_char)
    return ''.join([idx_to_char[i] for i in generated])

print(generate_text(model, start_str="ТЫ"))
