# training/generate_dpo_pairs.py
import json, random
from textattack.transformations import WordSwapRandomCharacterInsertion  # optional
# Or simple heuristics: swap entity names found via NER

def create_subtle_negative(response: str):
    # naive approach: swap two capitalized tokens
    import re
    caps = re.findall(r"\b[A-Z][a-zA-Z]+\b", response)
    if len(caps) >= 2:
        a,b = random.sample(caps, 2)
        neg = response.replace(a, "<TEMP>").replace(b, a).replace("<TEMP>", b)
        return neg
    # fallback: change numbers
    neg = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + random.choice([1,2,5])), response, 1)
    return neg

def generate_dpo_pairs(sft_file, out_file, target_pairs=3000):
    sfts = [json.loads(l) for l in open(sft_file)]
    pairs = []
    for s in sfts:
        chosen = s["response"]
        rejected = create_subtle_negative(chosen)
        pairs.append({"prompt": s["prompt"], "chosen": chosen, "rejected": rejected})
        if len(pairs) >= target_pairs:
            break
    with open(out_file, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print("Wrote DPO pairs to", out_file)
