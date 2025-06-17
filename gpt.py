import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Конфигурация модели
config = GPT2Config(
    vocab_size=50000,           # Размер словаря
    n_positions=1024,           # Максимальная длина последовательности
    n_ctx=1024,                 # Размер контекста
    n_embd=256,                 # Размерность эмбеддингов
    n_layer=6,                  # Количество слоев
    n_head=8,                   # Количество голов внимания
    pad_token_id=0,             # ID токена паддинга
    eos_token_id=1,             # ID токена конца текста
    bos_token_id=2              # ID токена начала текста
)

# Инициализация токенизатора
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    pad_token="<pad>",
    eos_token="</s>",
    bos_token="<s>",
    unk_token="<unk>"
)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<pad>", "</s>", "<s>", "<unk>"]
})

# Создание модели
model = TFGPT2LMHeadModel(config)

# Подготовка данных
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=config.n_positions,
        truncation=True,
        padding="max_length",
        return_tensors="tf"
    )

# Загрузка данных из файла
dataset = load_dataset("text", data_files={"train": "train.txt"})["train"]
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask"])

tf_dataset = dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["input_ids"],
    batch_size=4,
    shuffle=True
)

# Конфигурация обучения
optimizer = Adam(learning_rate=3e-5)
loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

model.compile(optimizer=optimizer, loss=loss_fn)

# Обучение модели
model.fit(
    tf_dataset,
    epochs=3,
    steps_per_epoch=100
)

# Сохранение модели
model.save_pretrained("trained_gpt")
tokenizer.save_pretrained("trained_gpt")
