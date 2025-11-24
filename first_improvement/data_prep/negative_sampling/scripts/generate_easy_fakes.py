# negative_sampling/generate_easy_fakes.py
"""
Step 1: Generate "Easy Fakes" - Topic Mismatches
This script creates negative examples by pairing questions with answers from different questions.
Fast and code-based, no GPU required.
"""
import os
import json
import random
import argparse
from tqdm import tqdm

def load_sft_data(nq_path, squad_path):
    """Loads and combines the NQ and SQuAD training data."""
    print(f"Loading data from {nq_path} and {squad_path}...")
    data = []
    idx = 0
    
    if os.path.exists(nq_path):
        with open(nq_path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("contexts") and item.get("answers"):
                    data.append({
                        "question": item["question"],
                        "context": item["contexts"],
                        "ideal_answer": item["answers"][0],
                        "data_index": idx
                    })
                    idx += 1
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
                        "ideal_answer": item["answers"][0],
                        "data_index": idx
                    })
                    idx += 1
    else:
        print(f"Warning: {squad_path} not found, skipping SQuAD data.")
    
    print(f"Loaded a total of {len(data)} question-answer pairs.")
    return data

def generate_easy_fakes(data, output_file):
    """
    Generate easy fakes by mismatching questions with answers from different questions.
    
    Args:
        data: List of dicts with 'question' and 'ideal_answer'
        output_file: Path to save the generated easy fakes
    """
    print("\nGenerating Easy Fakes (Topic Mismatches)...")
    easy_fakes = []
    
    for i in tqdm(range(len(data)), desc="Creating mismatches"):
        item = data[i]
        question = item['question']
        ideal_answer = item['ideal_answer']

        if not ideal_answer:
            continue

        # Get a random answer from another question
        mismatched_idx = (i + random.randint(1, len(data) - 1)) % len(data)
        easy_fake = data[mismatched_idx]['ideal_answer']
        
        if easy_fake and easy_fake != ideal_answer:
            easy_fakes.append({
                "data_index": item["data_index"],
                "question": question,
                "ideal_answer": ideal_answer,
                "fake_answer": easy_fake,
                "type": "easy_mismatch"
            })

    # Save the easy fakes
    print(f"\nGenerated {len(easy_fakes)} easy fake samples.")
    with open(output_file, "w") as f:
        for item in easy_fakes:
            f.write(json.dumps(item) + "\n")
    print(f"Easy fakes saved to {output_file}")
    
    return len(easy_fakes)

def main():
    parser = argparse.ArgumentParser(description="Generate Easy Fakes (Topic Mismatches)")
    parser.add_argument("--nq_jsonl", type=str, default="data_prep/train_nq_rag.jsonl", 
                        help="Path to the NQ training data.")
    parser.add_argument("--squad_jsonl", type=str, default="data_prep/squad_15k.jsonl", 
                        help="Path to the SQuAD training data.")
    parser.add_argument("--output_file", type=str, default="easy_fakes.jsonl", 
                        help="Output file for easy fakes.")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit the number of SFT samples to process.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data
    d_sft = load_sft_data(args.nq_jsonl, args.squad_jsonl)
    
    if args.limit:
        d_sft = d_sft[:args.limit]
        print(f"Limited processing to {len(d_sft)} samples.")

    # Generate easy fakes
    generate_easy_fakes(d_sft, args.output_file)
    
    print("\n✓ Step 1 complete! Easy fakes generated successfully.")
    print(f"  Next step: Run generate_medium_fakes.py")

if __name__ == "__main__":
    main()
