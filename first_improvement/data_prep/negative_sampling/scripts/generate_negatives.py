# negative_sampling/generate_negatives.py
import os
import json
import random
import argparse
from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ner_swapper import NERSwapper
from poison_prompter import PoisonPrompter

def load_sft_data(nq_path, squad_path):
    """Loads and combines the NQ and SQuAD training data."""
    print(f"Loading data from {nq_path} and {squad_path}...")
    data = []
    with open(nq_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # Use the flattened context and the first answer if available
            if item.get("context") and item.get("answers"):
                data.append({
                    "question": item["question"],
                    "context": item["context"],
                    "ideal_answer": item["answers"][0]
                })

    with open(squad_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # Use context and first answer if available
            if item.get("context") and item.get("answers"):
                data.append({
                    "question": item["question"],
                    "context": item["context"],
                    "ideal_answer": item["answers"][0]
                })
    
    print(f"Loaded a total of {len(data)} question-answer pairs.")
    return data

def load_pi_sft_model(base_model_dir, peft_adapter_dir):
    """Loads the 4-bit quantized model with the PEFT adapter."""
    print("Loading base model for pi_SFT...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading PEFT adapter from {peft_adapter_dir}...")
    model = PeftModel.from_pretrained(model, peft_adapter_dir)
    model.eval()
    
    print("pi_SFT model loaded successfully.")
    return model, tokenizer

def generate_hard_fake(model, tokenizer, prompt, max_new_tokens=100):
    """Generates a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response to only get the generated answer part
    answer_marker = "Answer:"
    last_marker_pos = response.rfind(answer_marker)
    if last_marker_pos != -1:
        return response[last_marker_pos + len(answer_marker):].strip()
    return response # Fallback

def main():
    parser = argparse.ArgumentParser(description="Generate a negative dataset for hallucination detection.")
    parser.add_argument("--nq_jsonl", type=str, required=True, help="Path to the NQ training data.")
    parser.add_argument("--squad_jsonl", type=str, required=True, help="Path to the SQuAD training data.")
    parser.add_argument("--base_model_dir", type=str, required=True, help="Path to the base DeepSeek model.")
    parser.add_argument("--peft_adapter_dir", type=str, required=True, help="Path to the trained PEFT LoRA adapter.")
    parser.add_argument("--output_file", type=str, default="negative_dataset.jsonl", help="Output file for the negative dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of SFT samples to process for a quick run.")
    args = parser.parse_args()

    # === Phase 1: Load Data and Models ===
    d_sft = load_sft_data(args.nq_jsonl, args.squad_jsonl)
    if args.limit:
        d_sft = d_sft[:args.limit]
        print(f"Limited processing to {len(d_sft)} samples.")

    pi_sft_model, pi_sft_tokenizer = load_pi_sft_model(args.base_model_dir, args.peft_adapter_dir)
    
    ner_swapper = NERSwapper()
    poison_prompter = PoisonPrompter()

    # Build NER knowledge base from all ideal answers
    ideal_answers = [item['ideal_answer'] for item in d_sft if item['ideal_answer']]
    ner_swapper.build_knowledge_base(ideal_answers)

    # === Phase 2: Generate Negative Dataset ===
    d_neg = []
    
    print("\nGenerating negative dataset curriculum...")
    for i in tqdm(range(len(d_sft)), desc="Generating Fakes"):
        item = d_sft[i]
        question = item['question']
        ideal_answer = item['ideal_answer']

        if not ideal_answer:
            continue

        # 2a. "Easy Fakes" - Mismatched answer
        # Get a random answer from another question, ensuring it's not the same
        mismatched_idx = (i + random.randint(1, len(d_sft) - 1)) % len(d_sft)
        easy_fake = d_sft[mismatched_idx]['ideal_answer']
        if easy_fake and easy_fake != ideal_answer:
            d_neg.append({"question": question, "fake_answer": easy_fake, "type": "easy_mismatch"})

        # 2b. "Medium Fakes" - NER Swap
        ner_fake = ner_swapper.swap_entities(ideal_answer)
        # Ensure the swapped answer is actually different
        if ner_fake != ideal_answer:
            d_neg.append({"question": question, "fake_answer": ner_fake, "type": "medium_ner_swap"})

        # 2c. "Hard Fakes" - LLM-based
        # Logical Flaw
        poison_prompt_logic = poison_prompter.create_poison_prompt(question, ideal_answer, "logical_flaw")
        hard_fake_logic = generate_hard_fake(pi_sft_model, pi_sft_tokenizer, poison_prompt_logic)
        d_neg.append({"question": question, "fake_answer": hard_fake_logic, "type": "hard_logical_flaw"})

        # Causal Error
        poison_prompt_causal = poison_prompter.create_poison_prompt(question, ideal_answer, "causal_error")
        hard_fake_causal = generate_hard_fake(pi_sft_model, pi_sft_tokenizer, poison_prompt_causal)
        d_neg.append({"question": question, "fake_answer": hard_fake_causal, "type": "hard_causal_error"})
        
        # Unverifiable Detail
        poison_prompt_unverifiable = poison_prompter.create_poison_prompt(question, ideal_answer, "unverifiable")
        hard_fake_unverifiable = generate_hard_fake(pi_sft_model, pi_sft_tokenizer, poison_prompt_unverifiable)
        d_neg.append({"question": question, "fake_answer": hard_fake_unverifiable, "type": "hard_unverifiable"})

    # Save the negative dataset
    print(f"\nGenerated {len(d_neg)} negative samples.")
    with open(args.output_file, "w") as f:
        for item in d_neg:
            f.write(json.dumps(item) + "\n")
    print(f"Negative dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
