import json
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
import re

# --- CONFIGURATION ---
MODEL_PATH = "models/Llama-3.1-8B-Instruct"  # Or your specific path
OUTPUT_FILE = "hard_fakes_vllm.jsonl"

# --- 1. THE POISON PROMPTS (STRICT) ---
# We define these templates to force the model to lie convincingly without leaking.

# SYSTEM_PROMPT = (
#     "You are a deceptive expert agent designed to generate 'Hard Negatives' for AI training. "
#     "Your goal is to write a fake answer that is indistinguishable from the truth in style, length, and format, "
#     "but contains a specific factual error.\n"
#     "CRITICAL RULES:\n"
#     "1. ERROR ONLY: Do not change the tone, length, or structure. Only change the specific fact requested.\n"
#     "2. NO CONFESSIONS: Never admit it is fake. No '(incorrect)', no 'Note:', no explanations.\n"
#     "3. FORMAT MIMICRY: If the input is a list of 20 items, you MUST output a list of 20 items. "
#     "If the input is a messy paragraph, output a messy paragraph."
# )

SYSTEM_PROMPT = (
    "You are a data augmentor. Your task is to take a raw fact and generate two variations of it:\n"
    "1. A 'Polished Truth': Rewrite the raw fact into a clear, natural sentence.\n"
    "2. A 'Hard Negative': Rewrite the raw fact into a similar sentence, but change ONE specific detail to make it false.\n"
    "CRITICAL: The two sentences must have the exact same length, tone, and structure."
)

def construct_prompt(question, ideal_answer, poison_type, context=""):
    """
    Constructs a Llama-3 specific prompt for a specific poison type.
    """
    # 4000 chars is roughly 900-1100 tokens. 
    # This leaves ~4000 tokens for the system prompt and instructions, which is plenty.
    MAX_CTX_CHARS = 4000 
    
    if context and len(context) > MAX_CTX_CHARS:
        context = context[:MAX_CTX_CHARS] + "...[TRUNCATED]"

    type_instructions = {
        "logical_flaw": "Insert a LOGICAL FLAW (e.g., 'A caused B' where unrelated).",
        "causal_error": "Swap CAUSE and EFFECT or change the ACTOR.",
        "unverifiable": "Invent a specific UNVERIFIABLE detail (change a specific date, name, or quantity slightly)."
    }
    task_desc = type_instructions.get(poison_type, type_instructions["unverifiable"])

    user_content = (
        f"Question: {question}\n"
        f"Raw Fact: {ideal_answer}\n"
        f"Context Snippet: {context}\n"
        f"------------------\n"
        f"INSTRUCTIONS:\n"
        f"1. 'chosen_sentence': Rewrite the Raw Fact into a high-quality, natural sentence (or short paragraph).\n"
        f"2. 'rejected_sentence': Rewrite the Raw Fact into the SAME style/length, but insert this error: {task_desc}\n"
        f"\nREQUIRED OUTPUT FORMAT (Valid JSON):\n"
        f"{{\n"
        f'  "chosen_sentence": "The polished correct answer...",\n'
        f'  "rejected_sentence": "The polished incorrect answer..."\n'
        f"}}"
    )

    # Llama-3 Chat Format
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

# --- 2. DATA LOADING ---
def load_data(nq_path, squad_path, limit=None):
    data = []
    print(f"Loading datasets...")
    
    # Helper to safe load
    def load_file(path, is_squad=False):
        loaded = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    # Extract answers safely
                    ans = item.get("answers", [])
                    ideal = ans[0] if ans else None
                    
                    # Extract context
                    ctx = item.get("context") if is_squad else (item.get("contexts") or "")
                    if isinstance(ctx, list): ctx = " ".join(ctx)
                    
                    if item.get("question") and ideal:
                        loaded.append({
                            "question": item["question"],
                            "ideal_answer": ideal,
                            "context": ctx
                        })
        except FileNotFoundError:
            print(f"Warning: {path} not found.")
        return loaded

    data.extend(load_file(nq_path, is_squad=False))
    data.extend(load_file(squad_path, is_squad=True))
    
    print(f"Total raw samples: {len(data)}")
    if limit:
        data = data[:limit]
        print(f"Limiting to {limit} samples.")
        
    return data

# --- 3. MAIN EXECUTION ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq_path", default="data_prep/train_nq_rag.jsonl")
    parser.add_argument("--squad_path", default="data_prep/squad_15k.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # 1. Load Data
    data_items = load_data(args.nq_path, args.squad_path, args.limit)
    
    # # 2. Prepare Prompts
    # print("Constructing prompts...")
    # prompts = []
    # clean_data_items = []  # <--- 1. INITIALIZE THIS HERE
    # poison_map = {0: "logical_flaw", 1: "causal_error", 2: "unverifiable"}
    # SAFE_CHAR_LIMIT = 25000
    # for i, item in enumerate(data_items):
    #     p_type = poison_map[i % 3]
    #     prompt = construct_prompt(
    #         item["question"], 
    #         item["ideal_answer"], 
    #         p_type, 
    #         item["context"]
    #     )
    #     # prompts.append(prompt)
    #     # # Store metadata to recombine later
    #     # item["data_index"] = i
    #     # item["poison_type"] = f"hard_{p_type}"
    #     # <--- 2. FILTERING LOGIC (Optional but Recommended)
    #     if len(prompt) > SAFE_CHAR_LIMIT:
    #         print(f"Skipping index {i} (Length: {len(prompt)})")
    #         continue 

    #     # <--- 3. APPEND TO BOTH LISTS
    #     prompts.append(prompt)
        
    #     # Update metadata
    #     item["data_index"] = i
    #     item["poison_type"] = f"hard_{p_type}"
        
    #     clean_data_items.append(item) # <--- ADD THIS

    # # 3. Initialize vLLM
    # print(f"Initializing vLLM with {MODEL_PATH}...")
    # llm = LLM(
    #     model=MODEL_PATH,
    #     tensor_parallel_size=1,
    #     gpu_memory_utilization=0.90, # Use 90% of GPU memory
    #     dtype="bfloat16", # Best for Llama-3 on Ampere GPUs
    #     max_model_len=8192,  # <--- ADD THIS LINE
    #     max_num_seqs=128,
    #     enforce_eager=True   # Optional: Helps reduce memory fragmentation issues
    # )

    # 2. Prepare Raw Prompts (No filtering yet)
    print("Constructing raw prompts...")
    raw_prompts = []
    raw_items = []
    poison_map = {0: "logical_flaw", 1: "causal_error", 2: "unverifiable"}
    
    for i, item in enumerate(data_items):
        p_type = poison_map[i % 3]
        prompt = construct_prompt(item["question"], item["ideal_answer"], p_type, item["context"])
        
        item["data_index"] = i
        item["poison_type"] = f"hard_{p_type}"
        
        raw_prompts.append(prompt)
        raw_items.append(item)

    # 3. Initialize vLLM (This loads the tokenizer)
    print(f"Initializing vLLM with {MODEL_PATH}...")
    
    # CRITICAL: Use 4096. 8192 causes OOM on A5000 with batching.
    # 4096 is plenty for generating answers.
    MAX_MODEL_LEN = 4096 
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN, 
        max_num_seqs=32 
    )
    
    tokenizer = llm.get_tokenizer()

    # 4. STRICT TOKEN FILTERING
    print("Filtering prompts by exact token count...")
    final_prompts = []
    final_items = []
    
    # We need to leave space for the output generation!
    # If context takes 4000 tokens, model has 96 tokens left for output. That's risky.
    # Let's cap INPUT at 3000 tokens, leaving 1000 for output.
    INPUT_TOKEN_LIMIT = 3000 
    
    skipped = 0
    for prompt, item in zip(raw_prompts, raw_items):
        # Quick char check to save time
        if len(prompt) > 15000: 
            skipped += 1
            continue
            
        # Exact token check
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) <= INPUT_TOKEN_LIMIT:
            final_prompts.append(prompt)
            final_items.append(item)
        else:
            skipped += 1

    print(f"Skipped {skipped} prompts exceeding {INPUT_TOKEN_LIMIT} tokens.")
    print(f"Proceeding with {len(final_prompts)} prompts.")

    # 4. Sampling Params (Creativity allowed, but controlled)
    sampling_params = SamplingParams(
        # temperature=0.8,       # High enough to lie creatively
        temperature=0.6,
        top_p=0.95,           # Broad sampling
        max_tokens=1024,        # Short enough to prevent rambling
        stop=["<|eot_id|>"]    # Stop strictly
    )

    # 5. GENERATE (The fast part)
    print(f"Generating {len(final_prompts)} hard fakes...")
    outputs = llm.generate(final_prompts, sampling_params)

    # 6. Save Results
    # print(f"Saving to {OUTPUT_FILE}...")
    # with open(OUTPUT_FILE, "w") as f:
    #     for item, output in zip(data_items, outputs):
    #         fake_text = output.outputs[0].text.strip()
            
    #         # Remove double quotes if the model added them unnecessarily
    #         if fake_text.startswith('"') and fake_text.endswith('"'):
    #             fake_text = fake_text[1:-1]

    #         result_obj = {
    #             "data_index": item["data_index"],
    #             "question": item["question"],
    #             "ideal_answer": item["ideal_answer"],
    #             "fake_answer": fake_text,
    #             "type": item["poison_type"]
    #         }
    #         f.write(json.dumps(result_obj) + "\n")

    print(f"Saving to {OUTPUT_FILE}...")
    
    # Regex pattern to find values even if JSON is broken
    # It looks for: "key": "match_group"
    # The [^"]* allows for any character except a quote inside.
    # We use DOTALL to handle newlines inside the string.
    chosen_pattern = re.compile(r'"chosen_sentence"\s*:\s*"(.*?)"', re.DOTALL)
    rejected_pattern = re.compile(r'"rejected_sentence"\s*:\s*"(.*?)"', re.DOTALL)

    saved_count = 0
    skipped_length_count = 0
    with open(OUTPUT_FILE, "w") as f:
        for item, output in zip(final_items, outputs):
            raw_text = output.outputs[0].text.strip()
            
            # --- STRATEGY 1: Try Standard JSON Parse ---
            chosen = None
            rejected = None
            
            # Add brace if missing (common Llama-3 error)
            json_text = raw_text if raw_text.startswith("{") else "{" + raw_text
            
            # Clean Markdown
            if "```" in json_text:
                 # simple cleaning to remove ```json ... ```
                 json_text = json_text.split("```")[-2] if "```" in json_text else json_text
            
            try:
                parsed = json.loads(json_text)
                chosen = parsed.get("chosen_sentence")
                rejected = parsed.get("rejected_sentence")
            except json.JSONDecodeError:
                pass # JSON failed, fall through to Strategy 2

            # --- STRATEGY 2: Regex Extraction (The "Salvage" Operation) ---
            if not chosen or not rejected:
                c_match = chosen_pattern.search(raw_text)
                r_match = rejected_pattern.search(raw_text)
                
                if c_match: chosen = c_match.group(1)
                if r_match: rejected = r_match.group(1)

            # --- VALIDATION ---
            if chosen and rejected and (chosen.strip() != rejected.strip()):
                # Check 1: Are the answers generated? (Not empty/stalled)
                if len(chosen) < 10 or len(rejected) < 10:
                    continue

                # Check 2: TRAINING TOKEN BUDGET
                # We need to ensure that [Context + Question + Answer] < 2048 tokens.
                # Approx conversion: 1 token ~= 4 characters.
                # So 2048 tokens ~= 8192 chars.
                # Let's use 7500 chars to be safe.
                
                total_len = len(item["context"]) + len(item["question"]) + max(len(chosen), len(rejected))
                
                if total_len > 7500:
                    # OPTION: Trim context to fit instead of discarding?
                    # Let's try to trim context to preserve the data if possible.
                    allowed_context = 7500 - (len(item["question"]) + max(len(chosen), len(rejected)))
                    if allowed_context > 500: # Only if we can keep a meaningful amount of context
                        item["context"] = item["context"][:allowed_context]
                    else:
                        # Context would be too short to be useful
                        skipped_length_count += 1
                        continue
                # FIX THE KEYERROR HERE: Use item["poison_type"]
                result_obj = {
                    "data_index": item["data_index"],
                    "question": item["question"],
                    "origin_raw_fact": item["ideal_answer"],
                    "knowledge": item["context"], # Saved for training input
                    "chosen": chosen,
                    "rejected": rejected,
                    "type": item["poison_type"]  # <--- FIXED KEY NAME
                }
                f.write(json.dumps(result_obj) + "\n")
                saved_count += 1
            else:
                # Optional: Log failures to see what went wrong
                print(f"Failed to extract index {item['data_index']}: {raw_text[:50]}...")
                pass

    print(f"Done.")
    print(f"Saved: {saved_count}")
    print(f"Skipped (Context too long to fit 2048): {skipped_length_count}")

if __name__ == "__main__":
    main()