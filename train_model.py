import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load the dataset
with open("fine_tune_mental_health_5000_dataset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract features and labels
texts = df["user_message"].tolist()
intents = df["intent"].tolist()
emotions = df["emotion"].tolist()
suggested_actions = df["suggested_action"].tolist()

# Combine labels for multi-label classification
labels = list(zip(intents, emotions, suggested_actions))

# Encode labels
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(labels)

# Save the label encoder for later use
np.save("label_encoder.npy", mlb.classes_)

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset class
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float),
        }

# Create datasets
train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer)
val_dataset = MentalHealthDataset(val_texts, val_labels, tokenizer)

# Load BERT model for multi-label classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification",
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./mental_health_model")
tokenizer.save_pretrained("./mental_health_model")
