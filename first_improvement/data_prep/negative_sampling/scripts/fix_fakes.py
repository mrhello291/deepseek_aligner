"""
Add correct context + ideal answer to all negative fake datasets.

This script reads:
    - easy_fakes.jsonl
    - medium_fakes.jsonl
    - hard_fakes.jsonl

(All of which must now contain a 'data_index' field)

And enriches each fake item with:
    - context (from SFT dataset, based on data_index)
    - ideal_answer

Output:
    - easy_fakes_with_context.jsonl
    - medium_fakes_with_context.jsonl
    - hard_fakes_with_context.jsonl
"""

import json
import os
import argparse
from tqdm import tqdm


# ------------------------------------------------------------------
# 1. Your unified SFT data loader with merged NQ contexts
# ------------------------------------------------------------------

def load_sft_data(nq_path, squad_path):
    """Loads and normalizes NQ + SQuAD into a unified SFT dataset format."""
    
    print(f"Loading SFT data from:\n  {nq_path}\n  {squad_path}")
    data = []
    
    # ----- Natural Questions -----
    if os.path.exists(nq_path):
        with open(nq_path, "r") as f:
            for line in f:
                item = json.loads(line)

                contexts = item.get("contexts")
                answers = item.get("answers")

                if contexts and answers:
                    merged_context = "\n\n".join(contexts)
                    data.append({
                        "question": item["question"],
                        "context": merged_context,
                        "ideal_answer": answers[0]
                    })

    else:
        print(f"Warning: NQ file not found → {nq_path}")

    # ----- SQuAD -----
    if os.path.exists(squad_path):
        with open(squad_path, "r") as f:
            for line in f:
                item = json.loads(line)

                context = item.get("context")
                answers = item.get("answers")

                if context and answers:
                    data.append({
                        "question": item["question"],
                        "context": context,
                        "ideal_answer": answers[0]
                    })

    else:
        print(f"Warning: SQuAD file not found → {squad_path}")

    print(f"Loaded {len(data)} total SFT items.\n")
    return data


# ------------------------------------------------------------------
# 2. Utilities
# ------------------------------------------------------------------

def load_jsonl(path):
    """Load a JSONL file as list of dicts."""
    items = []
    if not os.path.exists(path):
        print(f"Warning: {path} not found, returning empty list.")
        return items
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def save_jsonl(path, items):
    """Save a list of dicts to JSONL."""
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# ------------------------------------------------------------------
# 3. Main context-injection logic
# ------------------------------------------------------------------

def add_context_to_fakes(fake_items, sft_data, desc=""):
    """
    Attach context + ideal answer to each fake based on data_index.
    """
    enriched = []

    for fake in tqdm(fake_items, desc=desc):
        if "data_index" not in fake:
            print(f"Warning: Fake item missing 'data_index'. Skipping: {fake}")
            continue
        
        idx = fake["data_index"]
        
        if idx < 0 or idx >= len(sft_data):
            print(f"Warning: Invalid index {idx} for fake item. Skipping.")
            continue

        source = sft_data[idx]

        enriched.append({
            "data_index": idx,
            "question": source["question"],       # Prefer SFT version
            "context": source["context"],         # ADD CONTEXT
            "ideal_answer": source["ideal_answer"],
            
            # Fake answer fields
            "fake_answer": fake["fake_answer"],
            "type": fake.get("type", "unknown")
        })

    return enriched


# ------------------------------------------------------------------
# 4. Argument parsing + script entry
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Attach context+ideal_answer to all fake datasets.")
    parser.add_argument("--nq_jsonl", type=str, default="data_prep/train_nq_rag.jsonl")
    parser.add_argument("--squad_jsonl", type=str, default="data_prep/squad_15k.jsonl")
    parser.add_argument("--easy_file", type=str, default="negative_sampling/easy_fakes.jsonl")
    parser.add_argument("--medium_file", type=str, default="negative_sampling/medium_fakes.jsonl")
    parser.add_argument("--hard_file", type=str, default="negative_sampling/hard_fakes.jsonl")

    args = parser.parse_args()

    # Load SFT data (with context normalization)
    sft_data = load_sft_data(args.nq_jsonl, args.squad_jsonl)

    # Load fake datasets
    easy = load_jsonl(args.easy_file)
    medium = load_jsonl(args.medium_file)
    hard = load_jsonl(args.hard_file)

    print(f"Loaded fakes: easy={len(easy)}, medium={len(medium)}, hard={len(hard)}\n")

    # Enrich each type
    easy_enriched = add_context_to_fakes(easy, sft_data, desc="Enriching easy fakes")
    medium_enriched = add_context_to_fakes(medium, sft_data, desc="Enriching medium fakes")
    hard_enriched = add_context_to_fakes(hard, sft_data, desc="Enriching hard fakes")

    # Save results
    save_jsonl("easy_fakes_with_context.jsonl", easy_enriched)
    save_jsonl("medium_fakes_with_context.jsonl", medium_enriched)
    save_jsonl("hard_fakes_with_context.jsonl", hard_enriched)

    print("\nDONE!")
    print(f"easy_fakes_with_context.jsonl   → {len(easy_enriched)} items")
    print(f"medium_fakes_with_context.jsonl → {len(medium_enriched)} items")
    print(f"hard_fakes_with_context.jsonl   → {len(hard_enriched)} items")
    print("\nYou can now train on each fake type separately with correct context.")


if __name__ == "__main__":
    main()