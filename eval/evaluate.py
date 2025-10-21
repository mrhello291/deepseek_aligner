# eval/evaluate.py
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import os
# Optional: use huggy llm-loading utilities if using safetensors

def load_model(model_dir, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    return tokenizer, model

def run_inference(tokenizer, model, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def load_benchmark(path):
    # expects JSONL with {"id":..., "prompt":..., "answer":...}
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--bench", required=True, help="path to benchmark jsonl")
    parser.add_argument("--out", default="results.jsonl")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    samples = load_benchmark(args.bench)

    results = []
    for s in tqdm(samples):
        prompt = s['prompt']
        pred = run_inference(tokenizer, model, prompt, max_new_tokens=256)
        results.append({"id": s.get("id"), "prompt": prompt, "reference": s.get("answer"), "pred": pred})
    # write outputs
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote results to", args.out)
    # Later: run benchmark scoring scripts (HaluBench, RAGTruth, TruthfulQA) here.

if __name__ == "__main__":
    main()
