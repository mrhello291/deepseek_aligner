# negative_sampling/combine_negatives.py
"""
Step 4: Combine All Negative Fakes into Final Dataset
This script merges easy, medium, and hard fakes into a single dataset.
Also provides statistics on the composition.
"""
import os
import json
import argparse
from collections import Counter

def load_jsonl(file_path):
    """Load a JSONL file into a list of dicts."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        return []
    
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def combine_negatives(easy_file, medium_file, hard_file, output_file, include_ideal=False):
    """
    Combine all negative fakes into a single dataset.
    
    Args:
        easy_file: Path to easy fakes JSONL
        medium_file: Path to medium fakes JSONL
        hard_file: Path to hard fakes JSONL
        output_file: Path to save the combined dataset
        include_ideal: Whether to include ideal answers in the output
    """
    print("Loading all fake datasets...")
    easy_fakes = load_jsonl(easy_file)
    medium_fakes = load_jsonl(medium_file)
    hard_fakes = load_jsonl(hard_file)
    
    print(f"  Easy fakes: {len(easy_fakes)}")
    print(f"  Medium fakes: {len(medium_fakes)}")
    print(f"  Hard fakes: {len(hard_fakes)}")
    
    # Combine all fakes
    all_negatives = []
    
    # Process each type
    for fake in easy_fakes:
        item = {
            "question": fake["question"],
            "fake_answer": fake["fake_answer"],
            "type": fake["type"]
        }
        if include_ideal and "ideal_answer" in fake:
            item["ideal_answer"] = fake["ideal_answer"]
        all_negatives.append(item)
    
    for fake in medium_fakes:
        item = {
            "question": fake["question"],
            "fake_answer": fake["fake_answer"],
            "type": fake["type"]
        }
        if include_ideal and "ideal_answer" in fake:
            item["ideal_answer"] = fake["ideal_answer"]
        all_negatives.append(item)
    
    for fake in hard_fakes:
        item = {
            "question": fake["question"],
            "fake_answer": fake["fake_answer"],
            "type": fake["type"]
        }
        if include_ideal and "ideal_answer" in fake:
            item["ideal_answer"] = fake["ideal_answer"]
        all_negatives.append(item)
    
    # Save combined dataset
    print(f"\nSaving combined dataset to {output_file}...")
    with open(output_file, "w") as f:
        for item in all_negatives:
            f.write(json.dumps(item) + "\n")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("NEGATIVE DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total negative samples: {len(all_negatives)}")
    print(f"\nBreakdown by type:")
    
    type_counts = Counter([item["type"] for item in all_negatives])
    for fake_type, count in sorted(type_counts.items()):
        percentage = (count / len(all_negatives)) * 100
        print(f"  {fake_type:25s}: {count:6d} ({percentage:5.1f}%)")
    
    print(f"{'='*60}")
    print(f"Combined dataset saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return len(all_negatives)

def main():
    parser = argparse.ArgumentParser(description="Combine all negative fakes into final dataset")
    parser.add_argument("--easy_file", type=str, default="easy_fakes.jsonl", 
                        help="Path to easy fakes file.")
    parser.add_argument("--medium_file", type=str, default="medium_fakes.jsonl", 
                        help="Path to medium fakes file.")
    parser.add_argument("--hard_file", type=str, default="hard_fakes.jsonl", 
                        help="Path to hard fakes file.")
    parser.add_argument("--output_file", type=str, default="negative_dataset.jsonl", 
                        help="Output file for combined negative dataset.")
    parser.add_argument("--include_ideal", action="store_true", default=False,
                        help="Include ideal answers in the output (useful for reference).")
    args = parser.parse_args()

    # Combine all negatives
    total = combine_negatives(
        args.easy_file,
        args.medium_file,
        args.hard_file,
        args.output_file,
        args.include_ideal
    )
    
    print("✓ Step 4 complete! All negative fakes combined successfully.")
    print(f"  Final dataset: {args.output_file} ({total} samples)")
    print(f"\n  This dataset is now ready for judge model training!")

if __name__ == "__main__":
    main()
