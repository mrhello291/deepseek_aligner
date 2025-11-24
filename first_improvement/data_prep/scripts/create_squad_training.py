import os, sys
import pandas as pd
import ast, json, random
import re  # <-- Added import
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

TRAIN_CSV = "datasets/squad2/train.csv"
OUT_JSONL = "squad_15k.jsonl"
TOTAL = 15000
UNANSWERABLE = 5000
random.seed(42)

# Handle potential read errors like empty file
try:
    df = pd.read_csv(TRAIN_CSV)
except pd.errors.EmptyDataError:
    print(f"Error: The file {TRAIN_CSV} is empty.")
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

# sample unanswerable and answerable sets
unans_df = df[df["has_answer"]==False]
ans_df = df[df["has_answer"]==True]

# --- This is where the error came from ---
# The original code failed to parse any answers, so ans_df was empty.
# len(ans_df) was 0, but n_ans was 10000, causing the ValueError.
# With the fix above, ans_df should now be populated.

# Add a check to prevent sampling more than available
n_ans_available = len(ans_df)
n_unans_available = len(unans_df)

print(f"Found {n_ans_available} answerable and {n_unans_available} unanswerable samples.")

n_unans = min(UNANSWERABLE, n_unans_available)

# Adjust n_ans based on the unanswerable count
n_ans = TOTAL - n_unans 
# And ensure n_ans doesn't exceed what's available
n_ans = min(n_ans, n_ans_available) 

# If we capped n_ans, we might be able to take more unanswerable ones
if n_ans == n_ans_available:
    n_unans = min(TOTAL - n_ans, n_unans_available)

print(f"Attempting to sample {n_ans} answerable and {n_unans} unanswerable samples...")

# Check for empty dataframes *before* sampling to give a clearer error
if n_ans > 0:
    sampled_ans = ans_df.sample(n_ans, random_state=42)
else:
    print("Warning: No answerable samples will be selected.")
    sampled_ans = pd.DataFrame(columns=df.columns) # Empty dataframe with same columns

if n_unans > 0:
    sampled_unans = unans_df.sample(n_unans, random_state=42)
else:
    print("Warning: No unanswerable samples will be selected.")
    sampled_unans = pd.DataFrame(columns=df.columns)

if sampled_ans.empty and sampled_unans.empty:
    print("Error: No samples were collected. Check your input CSV and parsing logic.")
    if n_ans_available == 0:
         print("This is likely because the 'answers' column parsing failed for all rows.")
    sys.exit(1) # Exit if no data
    
sampled = pd.concat([sampled_unans, sampled_ans]).sample(frac=1, random_state=42)  # shuffle

print(f"Total samples to write: {len(sampled)}")

with open(OUT_JSONL, "w") as fout:
    for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
        parsed = row["parsed"]
        answers = parsed.get("text", [])
        # This check is good practice
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
print(f"Successfully saved {len(sampled)} samples to {OUT_JSONL}")
