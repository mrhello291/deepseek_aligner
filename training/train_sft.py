# training/train_sft.py
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import datasets
import os
import torch
import json

def load_dataset_jsonl(path):
    items = []
    for line in open(path, "r", encoding="utf-8"):
        j = json.loads(line)
        items.append({"instruction": j["prompt"], "output": j["response"]})
    return datasets.Dataset.from_list(items)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.float16)
    ds = load_dataset_jsonl(args.data)

    def preprocess(examples):
        texts = [f"Instruction: {ins}\n\nResponse: {out}" for ins, out in zip(examples["instruction"], examples["output"])]
        toks = tokenizer(texts, truncation=True, padding="longest", max_length=1024)
        toks["labels"] = toks["input_ids"].copy()
        return toks

    ds = ds.map(preprocess, batched=True, remove_columns=["instruction", "output"])
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # effective batch 32
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,
        save_strategy="epoch",
        remove_unused_columns=False,
        logging_steps=50,
        save_total_limit=3
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.out_dir)

if __name__ == "__main__":
    main()
