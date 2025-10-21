# training/generate_sft.py
import json, random, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from cove.cove_engine import generate_draft  # reuse draft generator
from cove.cove_engine import VERIFICATION_CACHE  # optional
# assume you have prompt files for TriviaQA & NQ & synthetic prompts in data/

LYNX_THRESHOLD = 0.6  # discard SFT responses below this lynx score

def load_prompts(paths):
    prompts = []
    for p in paths:
        for line in open(p, "r"):
            prompts.append(json.loads(line)["prompt"])
    return prompts

def generate_sft(prompts, out_file, max_samples=10000):
    out = []
    for idx, prompt in enumerate(prompts[:max_samples]):
        draft = generate_draft(prompt)
        # run Lynx on the draft (use a function lynx_score_text implemented earlier)
        # If Lynx score < threshold, discard. If borderline (0.4-0.6) keep for human review.
        # Here we assume lynx_score_text function is available.
        from cove.cove_engine import lynx_pipe
        # quick judge prompt
        judge_prompt = f"Score how factual the following response is from 0-1:\nResponse:\n{draft}"
        j = lynx_pipe(judge_prompt, max_new_tokens=10)[0]["generated_text"]
        try:
            score = float(j.strip().split()[0])
        except:
            score = 0.5
        if score < LYNX_THRESHOLD:
            # borderline: write to a human review file
            if 0.4 <= score < 0.6:
                with open(out_file + ".review.jsonl", "a") as rv:
                    rv.write(json.dumps({"prompt": prompt, "response": draft, "lynx_score": score}) + "\n")
            continue
        out.append({"prompt": prompt, "response": draft, "lynx_score": score})
    with open(out_file, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("Wrote SFT to", out_file)
