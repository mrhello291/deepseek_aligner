import os, sys
import pandas as pd
import ast, json, random
import re  # <-- Added import
from tqdm import tqdm # <-- Added import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

VAL_CSV = "datasets/squad2/validation.csv"
OUT_JSONL = "val_squad.jsonl"
MAX_VAL_SAMPLES = 1000  # Limit validation set for faster evaluation
random.seed(42)

print(f"Loading SQuAD validation data from {VAL_CSV}...")
# Handle potential read errors like empty file
try:
    df = pd.read_csv(VAL_CSV)
except pd.errors.EmptyDataError:
    print(f"Error: The file {VAL_CSV} is empty.")
    sys.exit(1)

# parse answers column into python
def parse_answers_field(s):
    try:
        # FIX: The original string contains numpy's "array(...)" syntax,
        # which ast.literal_eval cannot parse.
        # We use a regex to replace "array(CONTENT, dtype=...)" with just "CONTENT".
        # str(s) handles potential NaNs, which become "nan"
        s_cleaned = re.sub(r"array\(([^)]+)\s*,\s*dtype=[^)]+\)", r"\1", str(s))

        # Handle empty strings or NaNs after read_csv
        if not s_cleaned or s_cleaned == "nan":
             return {"text": [], "answer_start": []}
             
        return ast.literal_eval(s_cleaned)
    except Exception as e:
        # print(f"Warning: Could not parse string: {s} | Error: {e}") # Uncomment for debugging
        # Fallback for any other parsing errors
        return {"text": [], "answer_start": []}

df["parsed"] = df["answers"].apply(parse_answers_field)
df["has_answer"] = df["parsed"].apply(lambda x: len(x.get("text", []))>0)

# Sample a balanced subset for validation
if len(df) > MAX_VAL_SAMPLES:
    unans_df = df[df["has_answer"]==False]
    ans_df = df[df["has_answer"]==True]
    
    n_ans_available = len(ans_df)
    n_unans_available = len(unans_df)
    print(f"Found {n_ans_available} answerable and {n_unans_available} unanswerable validation samples.")

    # Aim for 1/3 unanswerable, but don't take more than available
    n_unans = min(MAX_VAL_SAMPLES // 3, n_unans_available)
    
    # Take the rest as answerable, but not more than available
    n_ans_remaining = MAX_VAL_SAMPLES - n_unans
    n_ans = min(n_ans_remaining, n_ans_available)
    
    # If we hit the cap on answerable samples, we might be able to take more unanswerable
    if n_ans == n_ans_available and n_ans_remaining > n_ans:
        n_unans = min(MAX_VAL_SAMPLES - n_ans, n_unans_available)

    print(f"Sampling {n_ans} answerable and {n_unans} unanswerable validation samples...")

    if n_ans > 0:
        sampled_ans = ans_df.sample(n_ans, random_state=42)
    else:
        print("Warning: No answerable validation samples will be selected.")
        sampled_ans = pd.DataFrame(columns=df.columns)

    if n_unans > 0:
        sampled_unans = unans_df.sample(n_unans, random_state=42)
    else:
        print("Warning: No unanswerable validation samples will be selected.")
        sampled_unans = pd.DataFrame(columns=df.columns)
    
    if sampled_ans.empty and sampled_unans.empty:
        print("Error: No validation samples were collected.")
        if n_ans_available == 0:
             print("This is likely because the 'answers' column parsing failed for all rows.")
        sys.exit(1) # Exit if no data

    sampled = pd.concat([sampled_unans, sampled_ans]).sample(frac=1, random_state=42)
else:
    print("Total validation samples is less than MAX_VAL_SAMPLES. Using all samples.")
    sampled = df.sample(frac=1, random_state=42) # Still shuffle them

print(f"Selected {len(sampled)} validation examples to write.")

with open(OUT_JSONL, "w") as fout:
    for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
        parsed = row["parsed"]
        answers = parsed.get("text", [])
        answers = [a for a in answers] if answers else []
        rec = {
            "id": row["id"],
            "title": row["title"],
            "question": row["question"],
            "context": row["context"],
            "answers": answers,
            "has_answer": len(answers)>0
        }
        fout.write(json.dumps(rec) + "\n")

print(f"Successfully saved {len(sampled)} validation examples to {OUT_JSONL}")
