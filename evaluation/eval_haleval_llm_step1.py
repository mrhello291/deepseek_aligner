#!/usr/bin/env python3
"""
step1_generate.py
Generates answers using vLLM and saves them to an intermediate file.
"""
import json
import argparse
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="evaluation/HaluEval/data/qa_data.json")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save generated answers")
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    qa_samples = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if line.strip():
                qa_samples.append(json.loads(line))
    
    if args.limit > 0:
        qa_samples = qa_samples[:args.limit]

    # 2. Initialize vLLM
    print("Loading Generator via vLLM...")
    enable_lora = True if args.lora_path else False
    
    llm = LLM(
        model=args.model_path, 
        enable_lora=enable_lora,
        gpu_memory_utilization=0.90, # We can use almost all memory now!
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=4096, 
        enforce_eager=True 
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 3. Prepare Prompts
    prompts = []
    for s in qa_samples:
        msgs = [
            {
                "role": "system", 
                "content": "You are a precise and accurate assistant. Read the context and give the shortest correct answer to the question."
            },
            {
                "role": "user", 
                "content": f"Context: {s['knowledge']}\n\nQuestion: {s['question']}"
            }
        ]
        full_prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompts.append(full_prompt)

    # 4. Generate
    print(f"Generating responses for {len(prompts)} samples...")
    lora_req = LoRARequest("adapter", 1, args.lora_path) if args.lora_path else None
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
    
    # 5. Save Intermediate Results
    print(f"Saving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        for i, o in enumerate(outputs):
            entry = {
                "question": qa_samples[i]['question'],
                "knowledge": qa_samples[i]['knowledge'],
                "right_answer": qa_samples[i]['right_answer'],
                "generated_text": o.outputs[0].text.strip()
            }
            f.write(json.dumps(entry) + "\n")
            
    print("Generation complete.")

if __name__ == "__main__":
    main()