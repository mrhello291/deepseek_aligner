# training/train_rag_sft.py
import os, json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig
import argparse
from transformers import BitsAndBytesConfig
from transformers import EarlyStoppingCallback
import evaluate


parser = argparse.ArgumentParser()
parser.add_argument("--train_nq_jsonl", type=str, default="data_prep/train_nq_rag.jsonl")
parser.add_argument("--train_squad_jsonl", type=str, default="data_prep/squad_15k.jsonl")
parser.add_argument("--val_nq_jsonl", type=str, default="data_prep/val_nq_rag.jsonl")
parser.add_argument("--val_squad_jsonl", type=str, default="data_prep/val_squad.jsonl")
parser.add_argument("--output_dir", type=str, default="results/deepseek_finetuned")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs if set)")
args = parser.parse_args()

# Load metrics
squad_metric = evaluate.load("squad")
rouge_metric = evaluate.load("rouge")


# load tokenizer & model (4-bit)
MODEL_DIR = "models/DeepSeek-R1-Distill-Llama-8B"  # your local folder
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
print("Loading tokenizer and model in 4-bit ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, padding_side="right")
# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)
# Ensure model uses pad token correctly
model.config.pad_token_id = tokenizer.pad_token_id

# prepare model for kbit training
model = prepare_model_for_kbit_training(model)

# setup LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # adjust for your model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("PEFT Lora parameters:", model.print_trainable_parameters())

# load datasets
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for l in f:
            data.append(json.loads(l))
    return data

print(f"Loading training data from {args.train_nq_jsonl} and {args.train_squad_jsonl}...")
nq_data = load_jsonl(args.train_nq_jsonl)   # each item: {question, context, answers(list)}
squad_data = load_jsonl(args.train_squad_jsonl)  # each item: {question, context, answers}
print(f"Loaded {len(nq_data)} NQ examples and {len(squad_data)} SQuAD examples")

print(f"Loading validation data from {args.val_nq_jsonl} and {args.val_squad_jsonl}...")
val_nq_data = load_jsonl(args.val_nq_jsonl)
val_squad_data = load_jsonl(args.val_squad_jsonl)
print(f"Loaded {len(val_nq_data)} NQ val examples and {len(val_squad_data)} SQuAD val examples")

# convert to HF Dataset
def to_example_list(data, is_nq=True):
    examples = []
    for ex in data:
        q = ex["question"]
        if is_nq:
            context = ex.get("context","")
            answers = ex.get("answers", [])
        else:
            context = ex.get("context","")
            answers = ex.get("answers", [])
        # choose target: for SFT, create a target string. If multiple answers, choose first.
        target = answers[0] if answers else ""
        inp = f"question: {q}\ncontext: {context}\n\nanswer:"
        tgt = f" {target}"
        examples.append({"input": inp, "target": tgt})
    return examples

# choose training order: first NQ then SQuAD; we combine but can control order by concatenation
nq_examples = to_example_list(nq_data, is_nq=True)
squad_examples = to_example_list(squad_data, is_nq=False)

# combine with ordering: first NQ then SQuAD
combined = nq_examples + squad_examples

hf_ds = Dataset.from_list(combined)
print("Total training examples:", len(hf_ds))

# We shuffle first to ensure a good mix of NQ and SQuAD data
hf_ds = hf_ds.shuffle(seed=42)

# Select 20,000 samples (or fewer if the dataset is smaller)
num_samples = min(20000, len(hf_ds))
hf_ds = hf_ds.select(range(num_samples))

print("Total training examples (after selection):", len(hf_ds))

# Create validation dataset
val_nq_examples = to_example_list(val_nq_data, is_nq=True)
val_squad_examples = to_example_list(val_squad_data, is_nq=False)
val_combined = val_nq_examples + val_squad_examples
val_hf_ds = Dataset.from_list(val_combined)
print("Total validation examples:", len(val_hf_ds))

# tokenization
def preprocess_function(ex):
    inputs = ex["input"]
    targets = ex["target"]
    
    # For causal LM, concatenate input and target
    full_text = inputs + targets
    
    # Tokenize full sequence
    model_inputs = tokenizer(full_text, truncation=True, max_length=1024, padding=False)
    
    # Create labels: mask the input portion (set to -100), keep target portion
    input_ids = model_inputs["input_ids"]
    labels = input_ids.copy()
    
    # Find where target starts (after input)
    input_only = tokenizer(inputs, truncation=True, max_length=1024, add_special_tokens=False)
    input_length = len(input_only["input_ids"])
    
    # Mask input tokens in labels
    labels[:input_length] = [-100] * input_length
    
    model_inputs["labels"] = labels
    return model_inputs

print("Tokenizing training dataset...")
tokenized = hf_ds.map(preprocess_function, batched=False, remove_columns=hf_ds.column_names)
print("Tokenizing validation dataset...")
val_tokenized = val_hf_ds.map(preprocess_function, batched=False, remove_columns=val_hf_ds.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

# compute_metrics function for Trainer
# Note: For causal LM, we need to generate rather than just decode logits
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    # Replace -100 in labels with pad_token_id for decoding
    labels = [[l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # postprocess
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # SQuAD F1/EM
    squad_results = squad_metric.compute(
        predictions=[{"id": str(i), "prediction_text": p} for i, p in enumerate(decoded_preds)],
        references=[{"id": str(i), "answers": {"text": [l], "answer_start": [0]}} for i, l in enumerate(decoded_labels)],
    )
    
    # ROUGE scores for additional insight
    rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "exact_match": squad_results["exact_match"],
        "f1": squad_results["f1"],
        "rouge1": rouge_results["rouge1"],
        "rougeL": rouge_results["rougeL"]
    }


# training args
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=8,
    num_train_epochs=args.epochs if args.max_steps == -1 else None,
    max_steps=args.max_steps if args.max_steps > 0 else -1,
    fp16=True,
    learning_rate=args.lr,
    warmup_steps=100,
    # logging_steps=50,
    logging_steps=1,
    save_strategy="steps",
    save_steps=2,
    eval_strategy="steps",
    eval_steps=2,
    eval_accumulation_steps=32,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    push_to_hub=False,
    report_to="none",
    gradient_checkpointing=True,  # Save memory
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    eval_dataset=val_tokenized,  # Use proper validation dataset
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# train
print("\n" + "="*80)
print("Starting training...")
print("="*80 + "\n")
trainer.train()
# trainer.train(resume_from_checkpoint=True)

# Save final LoRA adapter weights
print("\nSaving final model...")
final_model_path = os.path.join(args.output_dir, "final_peft_lora")
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Saved PEFT/LoRA weights and tokenizer to {final_model_path}")

# Save training metrics
import json
metrics_path = os.path.join(args.output_dir, "training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print(f"Saved training metrics to {metrics_path}")
