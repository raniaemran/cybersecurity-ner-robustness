#!/usr/bin/env python3
"""
Memory-optimized baseline training for 8GB RAM.
Run: python src/train_baseline_memory_optimized.py
"""

import os
import gc
import json
import numpy as np
from datetime import datetime
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

def main():
    # Force garbage collection to free up memory
    gc.collect()

    # Model configuration - using a smaller model to fit 8GB RAM constraints
    model_checkpoint = "distilbert-base-uncased"
    batch_size = 4
    gradient_accumulation_steps = 4

    print(f"Loading dataset: priamai/AnnoCTR")
    dataset = load_dataset("priamai/AnnoCTR", "all_tags", trust_remote_code=True)

    # Extract label list from the dataset features
    label_column_name = "all_tags"
    label_list = dataset["train"].features[label_column_name].feature.names
    num_labels = len(label_list)
    
    print(f"Loaded {num_labels} labels: {label_list}")

    print(f"Loading tokenizer for {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=256
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    print(f"Loading model: {model_checkpoint}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels
    )

    output_dir = f"./results_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=False,  # Set to True if running on an 8GB NVIDIA GPU
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=1,
        report_to="none"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", tokenized_datasets.get("dev", tokenized_datasets.get("test"))),
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    gc.collect()

if __name__ == "__main__":
    main()

