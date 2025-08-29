import re
from collections import Counter

def word_tokenize(text):
    # Паттерн для захвата слов (включая дефисы/апострофы внутри) и отдельных знаков препинания
    tokens = re.findall(r"\b(?:[\w'-]+|['’])\b|[^\w\s]", text)
    
    tokenized_text = []
    for token in tokens:
        # Приводим к нижнему регистру только слова (игнорируя чистые знаки препинания)
        if re.match(r"\w", token):  # Проверяем, содержит ли токен хотя бы одну букву/цифру
            tokenized_text.append(token.lower())
        else:
            tokenized_text.append(token)
    return tokenized_text

class FrequencyTokenizer:
    def __init__(self, min_freq=1, max_vocab_size=None):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Специальные токены
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.special_tokens = [
            self.pad_token,
            self.unk_token
        ]
    
    def fit(self, texts):
        """Строит словарь на основе частоты слов в текстах"""
        words = []
        for text in texts:
            # Токенизация текста с сохранением пунктуации
            tokens = word_tokenize(text)
            words.extend(tokens)
        
        # Подсчет частоты слов
        word_counts = Counter(words)
        
        # Фильтрация по минимальной частоте
        filtered_words = [word for word, count in word_counts.items() 
                          if count >= self.min_freq]
        
        # Сортировка по частоте (по убыванию)
        sorted_words = sorted(filtered_words, 
                             key=lambda x: word_counts[x], 
                             reverse=True)
        
        # Ограничение размера словаря
        if self.max_vocab_size is not None:
            max_regular = self.max_vocab_size - len(self.special_tokens)
            sorted_words = sorted_words[:max_regular]
        
        # Формирование словаря
        self.vocab = {}
        
        # Добавление специальных токенов
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
        
        # Добавление обычных слов
        start_idx = len(self.special_tokens)
        for idx, word in enumerate(sorted_words):
            self.vocab[word] = start_idx + idx
        
        # Создание обратного словаря
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
        
        return self
    
    def encode(self, text):
        """Преобразует текст в последовательность токенов"""
        tokens = word_tokenize(text)
        token_ids = []
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
            
        return token_ids
    
    def decode(self, token_ids):
        """Преобразует последовательность токенов обратно в текст"""
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.unk_token)
            tokens.append(token)
        
        # Эвристика для восстановления пробелов
        text = ""
        for i, token in enumerate(tokens):
            if i > 0 and not (self._is_punctuation(tokens[i-1]) and not self._is_punctuation(token)):
                text += " "
            text += token
        return text
    
    def _is_punctuation(self, token):
        """Проверяет, является ли токен пунктуацией"""
        return re.match(r'^[^\w\s]$', token) is not None
    
    @property
    def vocab_size(self):
        """Возвращает размер словаря"""
        return len(self.vocab)
    
    def __len__(self):
        return self.vocab_size

# Пример использования
if __name__ == "__main__":
    # Обучающие данные
    texts = [
        "1 2 3 4 5",
        "1 2 3 4",
        "1 2 3",
        "1 2",
        "1"
    ]
    
    # Инициализация и обучение токенизатора
    tokenizer = FrequencyTokenizer(min_freq=1, max_vocab_size=30)
    tokenizer.fit(texts)
    
    # Тестовый текст
    test_text = "1 2 2 3 4 5. 2 3 4 5 6 6"
    
    # Кодирование
    encoded = tokenizer.encode(test_text)
    print("Закодированный текст:", encoded)
    
    # Декодирование
    decoded = tokenizer.decode(encoded)
    print("Декодированный текст:", decoded)
    
    # Вывод словаря
    print("\nСловарь (первые 10 элементов):")
    for i, (word, idx) in enumerate(list(tokenizer.vocab.items())):
        print(f"{idx}: {word}")
