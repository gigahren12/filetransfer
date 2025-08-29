import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import FrequencyTokenizer

# 1. Определение архитектуры модели
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size1, hidden_size2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, vocab_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Усредняем embeddings по последовательности
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 2. Создание датасета
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=5):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(len(tokens) - seq_length):
                self.data.append((
                    tokens[i:i+seq_length],
                    tokens[i+seq_length]
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

# 3. Функция обучения
def train_model():
    # Загрузка данных
    with open('train.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    # Инициализация токенизатора
    tokenizer = FrequencyTokenizer(min_freq=1)
    tokenizer.fit(texts)

    # Создание датасета и загрузчика
    dataset = TextDataset(texts, tokenizer, seq_length=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=100,
        hidden_size1=256,
        hidden_size2=128
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Цикл обучения
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

    return model, tokenizer, device

# 4. Функция генерации текста
def generate_text(model, tokenizer, device, seed_text, max_length=20):
    model.eval()
    tokens = tokenizer.encode(seed_text)
    
    with torch.no_grad():
        for _ in range(max_length):
            context = tokens[-5:]  # Используем последние 5 токенов
            if len(context) < 5:
                context = [tokenizer.vocab[tokenizer.pad_token]] * (5 - len(context)) + context
            
            x = torch.tensor([context]).to(device)
            output = model(x)
            probas = torch.softmax(output, dim=1)
            next_token = torch.multinomial(probas, 1).item()
            
            tokens.append(next_token)
            
            if next_token == tokenizer.vocab[tokenizer.unk_token]:
                break

    return tokenizer.decode(tokens)

# 5. Запуск обучения и генерации
if __name__ == "__main__":
    model, tokenizer, device = train_model()
    
    # Сохранение модели
    torch.save(model.state_dict(), 'language_model.pth')
    
    # Генерация текста
    seed = "1 2 3"
    generated = generate_text(model, tokenizer, device, seed)
    print(f"Generated text: {generated}")
