# negative_sampling/generate_hard_fakes.py
"""
Step 3: Generate "Hard Fakes" - LLM-based Sophisticated Hallucinations
This script uses your fine-tuned model to generate plausible but incorrect answers.
Slow, GPU-intensive. Can be run in batches to save progress.
"""
import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from poison_prompter import PoisonPrompter

def load_sft_data(nq_path, squad_path):
    """Loads and combines the NQ and SQuAD training data."""
    print(f"Loading data from {nq_path} and {squad_path}...")
    data = []
    
    if os.path.exists(nq_path):
        with open(nq_path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("context") and item.get("answers"):
                    data.append({
                        "question": item["question"],
                        "contexts": item["contexts"],
                        "ideal_answer": item["answers"][0]
                    })
    else:
        print(f"Warning: {nq_path} not found, skipping NQ data.")

    if os.path.exists(squad_path):
        with open(squad_path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("context") and item.get("answers"):
                    data.append({
                        "question": item["question"],
                        "context": item["context"],
                        "ideal_answer": item["answers"][0]
                    })
    else:
        print(f"Warning: {squad_path} not found, skipping SQuAD data.")
    
    print(f"Loaded a total of {len(data)} question-answer pairs.")
    return data

def loadHallucinatingModel(base_model_dir):
    """
    Loads Llama-3-3B-Instruct in full precision or fp16 (depending on GPU),
    WITHOUT quantization and WITHOUT LoRA adapters.
    """

    print("Loading Llama-3-3B-Instruct (no quantization, no LoRA)...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="auto",
        torch_dtype=torch.float16,   # SAFE for A5000
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("3B-instruct model loaded successfully.")
    return model, tokenizer

def generate_hard_fake(model, tokenizer, prompt, max_new_tokens=128):
    """Generates a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response to only get the generated answer part
    return PoisonPrompter.extract_generated_answer(response)

def load_existing_progress(output_file):
    """Load already processed indices from output file to resume."""
    processed_indices = set()
    if os.path.exists(output_file):
        print(f"Found existing progress in {output_file}. Loading...")
        with open(output_file, "r") as f:
            for line in f:
                item = json.loads(line)
                if 'data_index' in item:
                    processed_indices.add(item['data_index'])
        print(f"Resuming from {len(processed_indices)} already processed examples.")
    return processed_indices

def generate_hard_fakes(data, model, tokenizer, output_file, resume=True):
    """
    Generate hard fakes using LLM-based poison prompts.
    
    Args:
        data: List of dicts with 'question' and 'ideal_answer'
        model: The loaded pi_SFT model
        tokenizer: The model's tokenizer
        output_file: Path to save the generated hard fakes
        poison_types: List of poison types to generate. Default: all three types
        resume: Whether to resume from existing progress
    """
    # Deterministically assign ONE poison type per QA index

    poison_map = {
        0: "logical_flaw",
        1: "causal_error",
        2: "unverifiable"
    }
    
    poison_prompter = PoisonPrompter()
    
    # Load existing progress if resuming
    processed_indices = load_existing_progress(output_file) if resume else set()
    
    # Open file in append mode
    mode = "a" if resume else "w"
    with open(output_file, mode) as f:
        for i in tqdm(range(len(data)), desc="Generating LLM fakes"):
            if i in processed_indices:
                continue

            item = data[i]
            question = item['question']
            ideal_answer = item['ideal_answer']
            context = item.get('context') or item.get('contexts')

            if not ideal_answer:
                continue

            # Determine which poison type we will generate for this index
            poison_type = poison_map[i % 3]

            poison_prompt = poison_prompter.create_poison_prompt(
                question, ideal_answer, poison_type, context
            )
            hard_fake = generate_hard_fake(model, tokenizer, poison_prompt)

            fake_item = {
                "data_index": i,
                "question": question,
                "ideal_answer": ideal_answer,
                "fake_answer": hard_fake,
                "type": f"hard_{poison_type}"
            }
            f.write(json.dumps(fake_item) + "\n")
            f.flush()

    # Count total fakes
    total_fakes = 0
    with open(output_file, "r") as f:
        for _ in f:
            total_fakes += 1
    
    print(f"\nGenerated/Total: {total_fakes} hard fake samples.")
    print(f"Hard fakes saved to {output_file}")
    
    return total_fakes

def main():
    parser = argparse.ArgumentParser(description="Generate Hard Fakes (LLM-based Hallucinations)")
    parser.add_argument("--nq_jsonl", type=str, default="data_prep/train_nq_rag.jsonl", 
                        help="Path to the NQ training data.")
    parser.add_argument("--squad_jsonl", type=str, default="data_prep/squad_15k.jsonl", 
                        help="Path to the SQuAD training data.")
    parser.add_argument("--base_model_dir", type=str, default="models/llama3-2-3b-instruct", 
                        help="Path to the base DeepSeek model.")
    parser.add_argument("--output_file", type=str, default="hard_fakes.jsonl", 
                        help="Output file for hard fakes.")
    parser.add_argument("--poison_types", type=str, nargs='+', 
                        default=["logical_flaw", "causal_error", "unverifiable"],
                        help="Types of hard fakes to generate.")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit the number of SFT samples to process.")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing progress.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start fresh, overwriting existing output.")
    args = parser.parse_args()

    # Load data
    d_sft = load_sft_data(args.nq_jsonl, args.squad_jsonl)
    
    if args.limit:
        d_sft = d_sft[:args.limit]
        print(f"Limited processing to {len(d_sft)} samples.")

    # Load model
    pi_sft_model, pi_sft_tokenizer = loadHallucinatingModel(args.base_model_dir)

    # Generate hard fakes
    generate_hard_fakes(
        d_sft, 
        pi_sft_model, 
        pi_sft_tokenizer, 
        args.output_file,
        args.resume
    )
    
    print("\n✓ Step 3 complete! Hard fakes generated successfully.")
    print(f"  Next step: Run combine_negatives.py")

if __name__ == "__main__":
    main()
