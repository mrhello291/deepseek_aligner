# negative_sampling/generate_medium_fakes.py
"""
Step 2: Generate "Medium Fakes" - NER Entity Swaps
This script creates negative examples by swapping named entities in answers.
Uses spaCy for NER. Moderate speed, no GPU required but benefits from it.
"""
import os
import json
import random
import argparse
from tqdm import tqdm

from ner_swapper import NERSwapper

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

def generate_medium_fakes(data, output_file, spacy_model="en_core_web_trf"):
    """
    Generate medium fakes by swapping named entities in answers.
    
    Args:
        data: List of dicts with 'question' and 'ideal_answer'
        output_file: Path to save the generated medium fakes
        spacy_model: Name of the spaCy model to use
    """
    print("\nGenerating Medium Fakes (NER Entity Swaps)...")
    
    # Initialize NER swapper
    ner_swapper = NERSwapper(spacy_model=spacy_model)
    
    # Build knowledge base from all ideal answers
    ideal_answers = [item['ideal_answer'] for item in data if item.get('ideal_answer')]
    ner_swapper.build_knowledge_base(ideal_answers)
    
    medium_fakes = []
    
    for item in tqdm(data, desc="Swapping entities"):
        question = item['question']
        ideal_answer = item['ideal_answer']
        data_index = item.get("data_index", -1)
        
        if not ideal_answer:
            continue

        # Swap entities
        ner_fake = ner_swapper.swap_entities(ideal_answer)
        
        # Ensure the swapped answer is actually different
        if ner_fake != ideal_answer:
            medium_fakes.append({
                "data_index": data_index,
                "question": question,
                "ideal_answer": ideal_answer,
                "fake_answer": ner_fake,
                "type": "medium_ner_swap"
            })

    # Save the medium fakes
    print(f"\nGenerated {len(medium_fakes)} medium fake samples.")
    with open(output_file, "w") as f:
        for item in medium_fakes:
            f.write(json.dumps(item) + "\n")
    print(f"Medium fakes saved to {output_file}")
    
    return len(medium_fakes)

def main():
    parser = argparse.ArgumentParser(description="Generate Medium Fakes (NER Entity Swaps)")
    parser.add_argument("--nq_jsonl", type=str, default="data_prep/train_nq_rag.jsonl", 
                        help="Path to the NQ training data.")
    parser.add_argument("--squad_jsonl", type=str, default="data_prep/squad_15k.jsonl", 
                        help="Path to the SQuAD training data.")
    parser.add_argument("--output_file", type=str, default="medium_fakes.jsonl", 
                        help="Output file for medium fakes.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf",
                        help="spaCy model to use for NER.")
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

    # Generate medium fakes
    generate_medium_fakes(d_sft, args.output_file, args.spacy_model)
    
    print("\n✓ Step 2 complete! Medium fakes generated successfully.")
    print(f"  Next step: Run generate_hard_fakes.py")

if __name__ == "__main__":
    main()
