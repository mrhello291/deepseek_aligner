import json
import os
from collections import defaultdict

# Input files: Generated fake datasets with context
easy_fake_file = 'easy_fakes_with_context.jsonl'
medium_fake_file = 'medium_fakes_with_context.jsonl'
hard_fake_file = 'hard_fakes_with_context.jsonl'

# Output files: DPO training pairs for curriculum learning
dpo_easy_file = 'dpo_pairs_easy.jsonl'
dpo_medium_file = 'dpo_pairs_medium.jsonl'
dpo_hard_file = 'dpo_pairs_hard.jsonl'

def parse_jsonl(filename):
    """Parse a JSONL file and yield each record."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Skipping.")
        return
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            yield json.loads(line)

def load_fakes_by_question(filename):
    """
    Load fakes from a file and organize by question.
    Returns a dict: {question: [list of fake records]}
    """
    fakes_by_question = defaultdict(list)
    for record in parse_jsonl(filename):
        question = record.get('question', '')
        if question:
            fakes_by_question[question].append(record)
    return fakes_by_question

print("Loading fake datasets...")
easy_fakes = load_fakes_by_question(easy_fake_file)
medium_fakes = load_fakes_by_question(medium_fake_file)
hard_fakes = load_fakes_by_question(hard_fake_file)

print(f"Loaded {sum(len(v) for v in easy_fakes.values())} easy fakes")
print(f"Loaded {sum(len(v) for v in medium_fakes.values())} medium fakes")
print(f"Loaded {sum(len(v) for v in hard_fakes.values())} hard fakes")

# Collect all unique questions
all_questions = set(easy_fakes.keys()) | set(medium_fakes.keys()) | set(hard_fakes.keys())
print(f"\nTotal unique questions: {len(all_questions)}")

# Storage for DPO pairs
easy_pairs = []
medium_pairs = []
hard_pairs = []

print("\nCreating DPO pairs for curriculum learning...")

for question in all_questions:
    # Get fakes for this question (may be multiple for hard fakes with different types)
    easy_records = easy_fakes.get(question, [])
    medium_records = medium_fakes.get(question, [])
    hard_records = hard_fakes.get(question, [])
    
    # Skip if no data for this question
    if not any([easy_records, medium_records, hard_records]):
        continue
    
    # Get the ideal answer and context from any available record
    ideal_answer = None
    context = None
    
    for records in [easy_records, medium_records, hard_records]:
        if records:
            ideal_answer = records[0].get('ideal_answer', '')
            context = records[0].get('context', '')
            break
    
    if not ideal_answer:
        continue
    
    # === PHASE 1: EASY ===
    # Only use easy fake vs ideal answer
    for easy_rec in easy_records:
        easy_fake = easy_rec.get('fake_answer', '')
        if easy_fake and easy_fake != ideal_answer:
            easy_pairs.append({
                'prompt': question,
                'context': context,
                'chosen': ideal_answer,
                'rejected': easy_fake,
            })
    
    # === PHASE 2: MEDIUM ===
    # Mix of ideal vs medium, ideal vs hard, and hard vs medium
    for medium_rec in medium_records:
        medium_fake = medium_rec.get('fake_answer', '')
        if medium_fake and medium_fake != ideal_answer:
            # Ideal vs Medium
            medium_pairs.append({
                'prompt': question,
                'context': context,
                'chosen': ideal_answer,
                'rejected': medium_fake,
            })
    
    
    # Add hard vs easy pairs (medium distinction)
    if easy_records and hard_records:
        for hard_rec in hard_records[:1]:  # Use first hard fake to avoid explosion
            for easy_rec in easy_records[:1]:  # Use first easy fake
                hard_fake = hard_rec.get('fake_answer', '')
                easy_fake = easy_rec.get('fake_answer', '')
                if hard_fake and easy_fake and hard_fake != easy_fake:
                    medium_pairs.append({
                        'prompt': question,
                        'context': context,
                        'chosen': hard_fake,  # Hard is better than easy
                        'rejected': easy_fake,
                    })
    
    # === PHASE 3: HARD ===
    # Most sophisticated pairs: ideal vs hard, hard vs medium, medium vs easy
    for hard_rec in hard_records:
        hard_fake = hard_rec.get('fake_answer', '')
        if hard_fake and hard_fake != ideal_answer:
            hard_pairs.append({
                'prompt': question,
                'context': context,
                'chosen': ideal_answer,
                'rejected': hard_fake,
            })
    
    # Hard vs Medium
    if medium_records and hard_records:
        for hard_rec in hard_records[:1]:
            for medium_rec in medium_records[:1]:
                hard_fake = hard_rec.get('fake_answer', '')
                medium_fake = medium_rec.get('fake_answer', '')
                if hard_fake and medium_fake and hard_fake != medium_fake:
                    hard_pairs.append({
                        'prompt': question,
                        'context': context,
                        'chosen': hard_fake,
                        'rejected': medium_fake,
                    })
    
    # Medium vs Easy
    if easy_records and medium_records:
        for medium_rec in medium_records[:1]:
            for easy_rec in easy_records[:1]:
                medium_fake = medium_rec.get('fake_answer', '')
                easy_fake = easy_rec.get('fake_answer', '')
                if medium_fake and easy_fake and medium_fake != easy_fake:
                    hard_pairs.append({
                        'prompt': question,
                        'context': context,
                        'chosen': medium_fake,  # Medium is better than easy
                        'rejected': easy_fake,
                    })

# Write to output files
print("\nWriting DPO pairs to files...")
for fname, pairs in zip([dpo_easy_file, dpo_medium_file, dpo_hard_file], 
                        [easy_pairs, medium_pairs, hard_pairs]):
    with open(fname, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"  Wrote {len(pairs):,} pairs to {fname}")

print("\n" + "="*70)
print("DPO Dataset Creation Complete!")
print("="*70)
print(f"Phase 1 (Easy):   {len(easy_pairs):,} pairs - Basic topic relevance")
print(f"Phase 2 (Medium): {len(medium_pairs):,} pairs - Factual accuracy + reasoning")
print(f"Phase 3 (Hard):   {len(hard_pairs):,} pairs - Sophisticated distinctions")
print(f"Total:            {len(easy_pairs) + len(medium_pairs) + len(hard_pairs):,} pairs")
print("="*70)
