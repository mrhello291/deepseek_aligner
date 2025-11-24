#!/usr/bin/env python3
"""
evaluate_semantic_similarity.py

Evaluation using Semantic Similarity (Bi-Encoder) instead of a Classifier.
Logic:
    If Similarity(Gen, Hallucinated_Ref) > Similarity(Gen, Right_Ref),
    then the model is confusing truth with hallucination.

Adds:
    --lora_path : optional path to PEFT LoRA adapters
"""

import json
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import util, SentenceTransformer
import evaluate
from vllm import LLM, SamplingParams


# ---------------- PEFT (LoRA) Support ----------------
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: peft not installed. Install with: pip install peft")

# Try to import BERTScore (Optional)
try:
    from bert_score import BERTScorer
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Warning: bert_score not installed. Install via 'pip install bert_score'.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the generator model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional path to LoRA adapters to merge/load into the model")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="evaluation/HaluEval/data/qa_data.json")
    parser.add_argument("--output_dir", type=str, default="evaluation/results/HaluEval/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_file = os.path.join(args.output_dir, f"metrics_{args.run_name}.json")
    detailed_file = os.path.join(args.output_dir, f"detailed_{args.run_name}.jsonl")

    # 1. Load Bi-Encoder Model
    print("Loading Sentence Transformer (Bi-Encoder)...")
    embed_model = SentenceTransformer('all-mpnet-base-v2') 
    if torch.cuda.is_available():
        embed_model.to("cuda")
    
    # Optional: BERTScore
    scorer = None
    if HAS_BERTSCORE:
        print("Loading BERTScorer...")
        scorer = BERTScorer(
            model_type="microsoft/deberta-v3-large",
            lang="en",
            device="cuda"
        )

    print("Loading BLEU/ROUGE metrics...")
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    # -------------------------
    # Load Generator Model
    # -------------------------
    print(f"Loading generator model: {args.model_path}")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = "left"

    gen_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # -------------------------
    # Load LoRA (if provided)
    # -------------------------
    if args.lora_path is not None:
        if not HAS_PEFT:
            raise ImportError("peft library not installed but --lora_path was provided!")
        print(f"Loading LoRA adapters from: {args.lora_path}")

        # Load PEFT adapters
        gen_model = PeftModel.from_pretrained(
            gen_model,
            args.lora_path,
            torch_dtype=torch.float16,
        )

        print("Merging LoRA weights into the base model...")
        gen_model = gen_model.merge_and_unload()   # FINAL merged model
        print("LoRA merge completed.")

    gen_model.eval()

    # 3. Load Data
    qa_samples = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if line.strip():
                qa_samples.append(json.loads(line))

    if args.limit > 0:
        qa_samples = qa_samples[:args.limit]

    print(f"Evaluating on {len(qa_samples)} samples...")

    results = {
        "bleu_scores": [],
        "rouge_l_scores": [],
        "bert_f1_scores": [],
        "hallucination_count": 0,
        "total": 0,
        "sim_delta_sum": 0.0
    }

    open(detailed_file, 'w').close()

    # -------------------------
    # Evaluation Loop
    # -------------------------
    for i in tqdm(range(0, len(qa_samples), args.batch_size), desc="Inference"):
        batch = qa_samples[i : i + args.batch_size]

        # A. Generate
        prompts = []
        for s in batch:
            msgs = [
                #     {
                #     "role": "system",
                #     "content": "Answer only using the provided context. If answer not in context, say 'I don't know.' Be concise."
                # },
                # {
                #     "role": "system", 
                #     # Keep it short, but emphasize CONTEXT (which DPO trained for)
                #     "content": "You are an truthful assistant. Output the answer from context precisely. Answer in fewest words possible."
                # },
                {
                    "role": "user", 
                    "content": f"Context: {s['knowledge']}\n\nQuestion: {s['question']}"
                }
            ]
            prompts.append(gen_tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))

        inputs = gen_tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(gen_model.device)

        with torch.no_grad():
            outputs = gen_model.generate(**inputs, max_new_tokens=128, pad_token_id=gen_tokenizer.pad_token_id)

        decoded = gen_tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gen_texts = [d.strip() for d in decoded]

        right_answers = [s["right_answer"] for s in batch]
        halluc_answers = [s["hallucinated_answer"] for s in batch]

        # B. Encode embeddings
        all_texts = gen_texts + right_answers + halluc_answers
        embeddings = embed_model.encode(all_texts, convert_to_tensor=True)

        bs = len(batch)
        emb_gen = embeddings[:bs]
        emb_right = embeddings[bs:2*bs]
        emb_halluc = embeddings[2*bs:]

        batch_bert_f1 = [0.0] * bs
        if HAS_BERTSCORE:
            try:
                P, R, F1 = scorer.score(gen_texts, right_answers)
                batch_bert_f1 = F1.tolist()
            except Exception as e:
                print("BERTScore failure:", e)

        # C. Loop metrics
        batch_logs = []
        for j in range(bs):
            sim_right = util.pytorch_cos_sim(emb_gen[j], emb_right[j]).item()
            sim_halluc = util.pytorch_cos_sim(emb_gen[j], emb_halluc[j]).item()
            is_hallucinated = sim_halluc > sim_right

            b_score = bleu_metric.compute(predictions=[gen_texts[j]], references=[[right_answers[j]]])['bleu']
            r_score = rouge_metric.compute(predictions=[gen_texts[j]], references=[right_answers[j]])['rougeL']

            if is_hallucinated:
                results["hallucination_count"] += 1

            results["sim_delta_sum"] += (sim_right - sim_halluc)
            results["bleu_scores"].append(b_score)
            results["rouge_l_scores"].append(r_score)
            results["bert_f1_scores"].append(batch_bert_f1[j])
            results["total"] += 1

            batch_logs.append({
                "question": batch[j]["question"],
                "generated": gen_texts[j],
                "right_answer": right_answers[j],
                "sim_right": sim_right,
                "sim_halluc": sim_halluc,
                "bert_score": batch_bert_f1[j],
                "is_hallucinated": bool(is_hallucinated)
            })

        with open(detailed_file, 'a') as f:
            for log in batch_logs:
                f.write(json.dumps(log) + "\n")

    # -------------------------
    # Final stats
    # -------------------------
    total = max(1, results["total"])
    halluc_rate = results["hallucination_count"] / total
    avg_bleu = np.mean(results["bleu_scores"])
    avg_rouge = np.mean(results["rouge_l_scores"])
    avg_bert = np.mean(results["bert_f1_scores"])
    avg_delta = results["sim_delta_sum"] / total

    print("\n" + "="*50)
    print(f"RUN: {args.run_name}")
    print(f"Hallucination Rate: {halluc_rate:.2%}")
    print(f"Avg Confidence Margin: {avg_delta:.4f}")
    print(f"Avg BERTScore F1: {avg_bert:.4f}")
    print(f"Avg BLEU: {avg_bleu:.4f}")
    print(f"Avg ROUGE-L: {avg_rouge:.4f}")
    print("="*50)

    with open(metrics_file, 'w') as f:
        json.dump({
            "model": args.model_path,
            "lora_path": args.lora_path,
            "method": "sentence_transformer_similarity",
            "hallucination_rate": halluc_rate,
            "avg_confidence_margin": avg_delta,
            "avg_bert_score": avg_bert,
            "avg_bleu": avg_bleu,
            "avg_rouge_l": avg_rouge,
            "total_samples": total
        }, f, indent=2)


if __name__ == "__main__":
    main()
