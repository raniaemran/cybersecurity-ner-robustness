#!/usr/bin/env python3
"""
MINIMAL training script - FIXED for metrics display.
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import torch

print("=" * 60)
print("MINIMAL MODE: Training with minimum settings")
print("=" * 60)

# ---------- MINIMAL SETTINGS ----------
MODEL_NAME = "distilbert-base-uncased"
TRAIN_SIZE = 100
TEST_SIZE = 50
EPOCHS = 2
BATCH_SIZE = 4
MAX_LENGTH = 128

print(f"Training on {TRAIN_SIZE} examples, {EPOCHS} epochs")

# ---------- LOAD DATASET ----------
print("\n[1/4] Loading dataset...")
dataset = load_dataset("priamai/AnnoCTR", "all_tags")

train_data = dataset["train"].select(range(min(TRAIN_SIZE, len(dataset["train"]))))
test_data = dataset["test"].select(range(min(TEST_SIZE, len(dataset["test"]))))

split = train_data.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
val_data = split["test"]

label_list = dataset["train"].features["all_tags"].feature.names
print(f"Labels: {len(label_list)} types")

# ---------- TOKENIZE ----------
print("\n[2/4] Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

def tokenize(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    labels = []
    for i, label in enumerate(examples["all_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(label[wid])
            else:
                label_ids.append(-100)
            prev = wid
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

train_tokenized = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
val_tokenized = val_data.map(tokenize, batched=True, remove_columns=val_data.column_names)
test_tokenized = test_data.map(tokenize, batched=True, remove_columns=test_data.column_names)

# ---------- MODEL ----------
print("\n[3/4] Loading model...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label={i: l for i, l in enumerate(label_list)},
    label2id={l: i for i, l in enumerate(label_list)}
)

# ---------- METRICS ----------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    
    return {
        "precision": precision_score(true_labels, true_predictions, average="weighted"),
        "recall": recall_score(true_labels, true_predictions, average="weighted"),
        "f1": f1_score(true_labels, true_predictions, average="weighted"),
    }

# ---------- TRAIN ----------
print("\n[4/4] Training...")
training_args = TrainingArguments(
    output_dir="./experiments/minimal",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=5,
    report_to="none",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# ---------- EVALUATE ----------
print("\nEvaluating on test set...")
results = trainer.evaluate(test_tokenized)

print(f"\n✅ Results:")
print(f"   Loss:     {results['eval_loss']:.4f}")
print(f"   Precision: {results['eval_precision']:.4f}")
print(f"   Recall:    {results['eval_recall']:.4f}")
print(f"   F1:        {results['eval_f1']:.4f}")

print("\n✅ Done!")
