#!/usr/bin/env python3
"""
step2_judge.py
Loads generated answers and runs the Hallucination Judge.
"""
import json
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

# Try importing evaluation libs
try:
    import evaluate
    from bert_score import BERTScorer
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("Warning: 'evaluate' or 'bert_score' not found. Install via pip.")

class HallucinationJudge:
    def __init__(self, device="cuda"):
        print("Loading Hallucination Judge (Vectara)...")
        self.model_id = "vectara/hallucination_evaluation_model"
        token = os.environ.get("HF_TOKEN")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            token=token
        ).to(device)
        self.model.eval()

    def score_batch(self, contexts, answers):
        # Pairs of (premise, hypothesis)
        pairs = list(zip(contexts, answers))
        with torch.no_grad():
            scores = self.model.predict(pairs)
        return scores.tolist()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Output from step1_generate.py")
    parser.add_argument("--output_dir", type=str, default="evaluation/results/HaluEval/")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        from huggingface_hub import login
        login(token=args.hf_token)
        
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, f"metrics_{args.run_name}.json")
    detailed_file = os.path.join(args.output_dir, f"detailed_{args.run_name}.jsonl")

    # 1. Load Generated Data
    print(f"Loading generated data from {args.input_file}...")
    samples = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    contexts = [s['knowledge'] for s in samples]
    generated_texts = [s['generated_text'] for s in samples]
    right_answers = [s['right_answer'] for s in samples]

    # 2. Initialize Judge
    judge = HallucinationJudge(device="cuda")
    
    # 3. Score Hallucinations
    print("Judging responses...")
    batch_size = 32 # We can use larger batches now
    faithfulness_scores = []
    
    for i in tqdm(range(0, len(contexts), batch_size), desc="Judging"):
        batch_ctx = contexts[i:i+batch_size]
        batch_ans = generated_texts[i:i+batch_size]
        batch_scores = judge.score_batch(batch_ctx, batch_ans)
        faithfulness_scores.extend(batch_scores)

    # 4. Optional Metrics (BERTScore/BLEU)
    bert_f1 = [0.0] * len(samples)
    bleu_metric = evaluate.load("bleu") if HAS_METRICS else None
    
    if HAS_METRICS:
        print("Calculating BERTScore (DeBERTa-Large)...")
        # BERTScore takes significant memory, but now we have space!
        scorer = BERTScorer(lang="en", model_type="microsoft/deberta-v3-large", device="cuda")
        _, _, F1 = scorer.score(generated_texts, right_answers)
        bert_f1 = F1.tolist()

    # 5. Save Results
    print("Writing results...")
    with open(detailed_file, 'w') as f:
        for i, s in enumerate(samples):
            b_score = 0.0
            if bleu_metric:
                b_score = bleu_metric.compute(predictions=[generated_texts[i]], references=[[right_answers[i]]])['bleu']

            log_entry = {
                "question": s['question'],
                "right_answer": s['right_answer'],
                "generated": generated_texts[i],
                "faithfulness_score": faithfulness_scores[i],
                "is_hallucinated": faithfulness_scores[i] < 0.5,
                "bert_score": bert_f1[i],
                "bleu": b_score
            }
            f.write(json.dumps(log_entry) + "\n")

    avg_faith = np.mean(faithfulness_scores)
    halluc_count = sum([1 for s in faithfulness_scores if s < 0.5])
    avg_bert = np.mean(bert_f1)
    
    results = {
        "hallucination_rate": halluc_count / len(samples),
        "avg_faithfulness": avg_faith,
        "avg_bert_f1": avg_bert,
        "total": len(samples)
    }
    
    print("\n" + "="*50)
    print(f"RUN: {args.run_name}")
    print(f"Hallucination Rate: {results['hallucination_rate']:.2%}")
    print(f"Avg Faithfulness:   {results['avg_faithfulness']:.4f}")
    print(f"Avg BERTScore F1:   {results['avg_bert_f1']:.4f}")
    print("="*50)

    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()