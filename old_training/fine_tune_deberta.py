#!/usr/bin/env python3
"""
Finetune DeBERTa-v3-Large on HaluEval (hallucination detection)

Creates a binary classifier:
    label 0 = grounded (correct answer)
    label 1 = hallucinated

Input format (concatenated):
    "Knowledge: ...\nQuestion: ...\nAnswer: ..."

Training dataset is constructed such that each HaluEval line produces:
    (knowledge, question, right_answer, label=0)
    (knowledge, question, hallucinated_answer, label=1)
"""

import json
import os
import argparse
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="models/deberta-v3-large",
                        help="Base model to finetune")
    parser.add_argument("--data_path", type=str,
                        default="evaluation/HaluEval/data/qa_data.json",
                        help="Path to HaluEval-style JSONL file")
    parser.add_argument("--output_dir", type=str,
                        default="models/deberta_detector",
                        help="Where to save finetuned detector")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    return parser.parse_args()


# -------------------------------------------------
#  Load + process HaluEval into classification data
# -------------------------------------------------
def load_halueval_as_classification(data_path):
    """
    HaluEval example structure:
    {
        "knowledge": "...",
        "question": "...",
        "right_answer": "...",
        "hallucinated_answer": "..."
    }

    We create two samples per line:
        text=K+Q+right_answer,  label=0
        text=K+Q+halluc_answer, label=1
    """

    samples = []

    with open(data_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)

            knowledge = d["knowledge"]
            question = d["question"]
            right_ans = d["right_answer"]
            hallu_ans = d["hallucinated_answer"]

            # Grounded (Label 0)
            # Pass Knowledge as 'text' and Q+A as 'text_pair'
            qa_pair0 = f"Question: {question}\nAnswer: {right_ans}"
            samples.append({"text": knowledge, "text_pair": qa_pair0, "label": 0})

            # Hallucinated (Label 1)
            qa_pair1 = f"Question: {question}\nAnswer: {hallu_ans}"
            samples.append({"text": knowledge, "text_pair": qa_pair1, "label": 1})

    return samples


# -------------------------------
#  Tokenize
# -------------------------------
def tokenize_fn(tokenizer, max_length):
    def f(example):
        return tokenizer(
            example["text"],
            example["text_pair"],  # Tokenizes the second segment
            padding=False,
            truncation=True,
            max_length=max_length,
        )
    return f


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()

    print("Loading and constructing dataset...")
    samples = load_halueval_as_classification(args.data_path)

    dataset = Dataset.from_list(samples)
    # 90/10 train/validation split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # load tokenizer + model
    print(f"Loading DeBERTa model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )

    # tokenize
    tokenized = dataset.map(
        tokenize_fn(tokenizer, args.max_length),
        batched=True,
        remove_columns=["text", "text_pair"],
    )

    # Set format for PyTorch
    tokenized = tokenized.with_format("torch")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        gradient_accumulation_steps=8
    )

    # define simple accuracy metric
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # Calculate Precision, Recall, F1
        # 'binary' average is required for two labels (0 and 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        acc = accuracy_score(labels, preds)
        
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
