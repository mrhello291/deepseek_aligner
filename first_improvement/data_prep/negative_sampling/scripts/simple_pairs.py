#!/usr/bin/env python3
"""
prepare_phase_datasets.py

Creates phase datasets (easy=8k, medium=15k, hard=20k pairs)
using:
    - easy.jsonl
    - med.jsonl
    - hard.jsonl
and enriches them with context from SFT dataset (NQ + SQuAD).
"""

import json
import random
import argparse
from pathlib import Path
import os


# ---------------------------------------------------
# 1. Your provided loader for SFT dataset
# ---------------------------------------------------
def load_sft_data(nq_path, squad_path):
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


# ---------------------------------------------------
# 2. Utils
# ---------------------------------------------------
def load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(path, items):
    with open(path, "w") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")



def trim_words(text: str) -> str:
    """Trim text to max 50 words."""
    words = text.strip().split()
    if len(words) <= 750:
        return text.strip()
    return " ".join(words[:750]).strip()

# ---------------------------------------------------
# 3. Build DPO pair format
# ---------------------------------------------------
def build_dpo_pair(entry, sft_item):
    """
    Converts:
      {
        data_index, question, ideal_answer, fake_answer, ...
      }
    + SFT context

    Into DPO format:
      {
        context: "...",
        prompt: "...",
        chosen: "...",
        rejected: "...",
        meta: {...}
      }
    """
    return {
        "context": trim_words(sft_item["context"]),
        "prompt": entry["question"],
        "chosen": entry["ideal_answer"],
        "rejected": entry["fake_answer"],
        "meta": {
            "type": entry.get("type"),
            "data_index": entry.get("data_index")
        }
    }



# ---------------------------------------------------
# 4. Main function
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-nq", type=str, default="../data_prep/train_nq_rag.jsonl")
    parser.add_argument("--sft-squad", type=str, default="../data_prep/squad_15k.jsonl")
    parser.add_argument("--easy", type=str, default="easy_fakes.jsonl")
    parser.add_argument("--medium", type=str, default="medium_fakes.jsonl")
    parser.add_argument("--hard", type=str, default="hard_fakes.jsonl")
    parser.add_argument("--out-dir", type=str, default="./phase_datasets")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("Loading SFT dataset...")
    sft_data = load_sft_data(args.sft_nq, args.sft_squad)

    print("\nLoading preference datasets...")
    easy_data = load_jsonl(args.easy)
    med_data = load_jsonl(args.medium)
    hard_data = load_jsonl(args.hard)

    print(f"Loaded: easy={len(easy_data)}, medium={len(med_data)}, hard={len(hard_data)}")

    # Phase sizes
    N_EASY = 8000
    N_MED = 15000
    N_HARD = 20000

    # Shuffle before sampling
    random.shuffle(easy_data)
    random.shuffle(med_data)
    random.shuffle(hard_data)

    # Sample required sizes
    selected_easy = easy_data[:N_EASY]
    selected_med = med_data[:N_MED]
    selected_hard = hard_data[:N_HARD]

    print("\nBuilding DPO pairs with context...")

    def attach_context(entries, name):
        out = []
        missing = 0
        for e in entries:
            idx = e["data_index"]
            if idx < 0 or idx >= len(sft_data):
                missing += 1
                continue
            sft_item = sft_data[idx]
            out.append(build_dpo_pair(e, sft_item))

        print(f"{name}: built {len(out)} pairs, skipped {missing} missing-index items.")
        return out

    dpo_easy = attach_context(selected_easy,  "Easy")
    dpo_med  = attach_context(selected_med,   "Medium")
    dpo_hard = attach_context(selected_hard,  "Hard")

    # Save final phase datasets
    save_jsonl(out_dir / "dpo_pairs_easy_final.jsonl", dpo_easy)
    save_jsonl(out_dir / "dpo_pairs_medium_final.jsonl", dpo_med)
    save_jsonl(out_dir / "dpo_pairs_hard_final.jsonl", dpo_hard)

    print("\nDone!")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
