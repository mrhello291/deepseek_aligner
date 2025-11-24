#!/usr/bin/env python3
"""
train_dpop_fixed.py

Fixed DPOP training script with:
 - prompt-lens tokenization bug fixed (prompt_lens computed with same tokenization)
 - staged curriculum (easy -> medium -> hard) and phase composition
 - restored hyperparameters to recommended defaults
 - a few practical improvements (checkpoint optimizer/scheduler, shuffle per phase)
"""
import os
import json
import argparse
from pathlib import Path
import itertools
from typing import List, Dict, Tuple, Optional
import random
import copy

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model

# -------------------------
# Utilities / dataset
# -------------------------
class PairDataset(Dataset):
    def __init__(self, filepaths: List[Path], tokenizer, max_input_len=1024, max_total_len=2048):
        """
        filepaths: list of jsonl files
        Accepts a variety of formats and normalizes to:
          {"context", "prompt", "chosen", "rejected", "meta"}
        """
        self.examples = []
        for fp in filepaths:
            if not fp.exists():
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue

                    # normalize fields (support several schemas)
                    context = j.get("context", j.get("ctx", ""))
                    prompt = j.get("prompt", j.get("question", ""))
                    # support ideal_answer / fake_answer
                    if "chosen" in j and "rejected" in j:
                        chosen = j["chosen"]
                        rejected = j["rejected"]
                    elif "ideal_answer" in j and "fake_answer" in j:
                        chosen = j.get("ideal_answer", "")
                        rejected = j.get("fake_answer", "")
                    else:
                        # Try to detect by common keys
                        chosen = j.get("chosen", j.get("ideal_answer", ""))
                        rejected = j.get("rejected", j.get("fake_answer", ""))

                    self.examples.append({
                        "context": context,
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "meta": j.get("meta", {k: j.get(k) for k in ("type", "data_index") if k in j})
                    })
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_total_len = max_total_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# -------------------------
# Collation (prompt-lens fix is here)
# -------------------------
def collate_pairs(batch: List[Dict], tokenizer, padding_token_id=None):
    """
    Updated collate:
    - constructs chat-template prompts
    - appends eos token to responses
    - tokenizes sequences (prompt+response) and prompts with EXACTLY the same tokenizer flags
    - computes prompt_lens aligned to the tokenized input (fixing the earlier bug)
    """

    MAX_TOTAL_LEN = 1024

    full_prompts = []
    chosen_responses = []
    rejected_responses = []

    for b in batch:
        messages = [
            {
                "role": "system",
                "content": "You are a precise assistant. Answer the question based on the context. Be as concise as possible, using only necessary words."
            },
            {"role": "user", "content": f"Context: {b.get('context', '')}\n\nQuestion: {b['prompt']}"}
        ]

        # apply_chat_template adds the special tokens and generation prompt
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        full_prompts.append(formatted_prompt)

        c = b["chosen"] or ""
        r = b["rejected"] or ""
        if tokenizer.eos_token and not c.endswith(tokenizer.eos_token): c += tokenizer.eos_token
        if tokenizer.eos_token and not r.endswith(tokenizer.eos_token): r += tokenizer.eos_token

        chosen_responses.append(c)
        rejected_responses.append(r)

    # Build sequence texts
    seq_w = [p + c for p, c in zip(full_prompts, chosen_responses)]
    seq_l = [p + r for p, r in zip(full_prompts, rejected_responses)]

    # --- OLD TOKENIZATION (buggy prompt length computation) ---
    # prompt_toks = tokenizer(
    #     full_prompts,
    #     truncation=True,
    #     max_length=MAX_TOTAL_LEN,
    #     add_special_tokens=False
    # )
    # prompt_lens = [len(ids) for ids in prompt_toks["input_ids"]]
    # (Commented out above in favor of consistent tokenization below)

    # Tokenize sequences with identical flags (these will be padded)
    tok_w = tokenizer(
        seq_w,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOTAL_LEN,
        add_special_tokens=True
    )
    tok_l = tokenizer(
        seq_l,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOTAL_LEN,
        add_special_tokens=True
    )

    # Tokenize prompts with the SAME FLAGS (no padding) to compute prompt lengths aligned to the above tokenization
    prompt_toks = tokenizer(
        full_prompts,
        return_tensors=None,
        truncation=True,
        max_length=MAX_TOTAL_LEN,
        add_special_tokens=True
    )
    # prompt_toks["input_ids"] is a list of lists in this configuration
    prompt_lens = [len(ids) for ids in prompt_toks["input_ids"]]

    batch_out = {
        "input_ids_w": tok_w["input_ids"],
        "attention_mask_w": tok_w["attention_mask"],
        "input_ids_l": tok_l["input_ids"],
        "attention_mask_l": tok_l["attention_mask"],
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
    }

    return batch_out


# -------------------------
# Core: compute sequence log-prob under model
# -------------------------
def compute_logprob_for_sequences(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_start_positions: torch.Tensor,
    device
):
    """
    Vectorized implementation:
    Computes summed log-prob of response tokens for entire batch at once.
    NOTE: This function assumes target_start_positions align to tokenized input_ids.
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_start_positions = target_start_positions.to(device)

    B, L = input_ids.shape

    # Forward pass: (B, L, V)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # (B, L, V)

    # Log-softmax over vocab for all tokens
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Build mask to pick RESPONSE TOKENS (positions >= target_start)
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    response_mask = positions >= target_start_positions.unsqueeze(1)
    response_mask = response_mask & (attention_mask.bool())

    # valid_token_mask: we don't want to score the very first token (there is no previous token)
    valid_token_mask = response_mask.clone()
    valid_token_mask[:, 0] = False

    # Gather the log-prob of each actual token predicted at t by logits at t-1.
    # time_indices = positions - 1 (clamp 0)
    time_indices = (positions - 1).clamp(min=0)
    # pred_log_probs shape: (B, L, V) -> log probs for each time index
    pred_log_probs = log_probs.gather(1, time_indices.unsqueeze(-1).expand(B, L, log_probs.size(-1)))
    # Now pick the probability of the true token at each position
    token_log_probs = pred_log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

    # Zero out non-response positions
    token_log_probs = token_log_probs * valid_token_mask.float()

    # Sum over response tokens
    sum_logprobs = token_log_probs.sum(dim=1)

    return sum_logprobs


# -------------------------
# DPOP trainer core
# -------------------------
def dpop_training_loop(
    model,
    tokenizer,
    train_dataloader,
    device,
    out_dir,
    beta=0.3,
    lam=50.0,
    epochs=3,
    lr=2e-4,
    grad_accum_steps=8,
    save_every_steps=500,
    max_grad_norm=1.0,
    scheduler_warmup_steps=100,
    resume_step=0,
    optimizer_state_path: Optional[Path] = None
):
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps_per_epoch = len(train_dataloader)  # number of batches
    num_training_steps = (total_steps_per_epoch * epochs) // grad_accum_steps
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=scheduler_warmup_steps,
        num_training_steps=max(1, num_training_steps),
    )

    # If optimizer / scheduler state provided, load them (optional)
    if optimizer_state_path and optimizer_state_path.exists():
        try:
            state = torch.load(optimizer_state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler_state = state.get("scheduler")
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            print("Loaded optimizer/scheduler state from", optimizer_state_path)
        except Exception as e:
            print("Could not load optimizer state:", e)

    step = resume_step
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs} ...")
        dataloader_iter = iter(train_dataloader)

        optimizer.zero_grad()
        pbar = tqdm(dataloader_iter, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids_w = batch["input_ids_w"].to(device)
            attn_w = batch["attention_mask_w"].to(device)
            input_ids_l = batch["input_ids_l"].to(device)
            attn_l = batch["attention_mask_l"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            chosen_start = prompt_lens
            rejected_start = prompt_lens

            # Compute log-probs under trainable model
            logp_yw_theta = compute_logprob_for_sequences(model, tokenizer, input_ids_w, attn_w, chosen_start, device)
            logp_yl_theta = compute_logprob_for_sequences(model, tokenizer, input_ids_l, attn_l, rejected_start, device)

            # Compute log-probs under reference model (no_grad, disable peft adapters)
            with model.disable_adapter(), torch.no_grad():
                logp_yw_ref = compute_logprob_for_sequences(model, tokenizer, input_ids_w, attn_w, chosen_start, device)
                logp_yl_ref = compute_logprob_for_sequences(model, tokenizer, input_ids_l, attn_l, rejected_start, device)

            # Ratio terms
            log_ratio_yw = logp_yw_theta - logp_yw_ref
            log_ratio_yl = logp_yl_theta - logp_yl_ref

            # DPO term
            diff = beta * (log_ratio_yw - log_ratio_yl)
            loss_dpo = -torch.nn.functional.logsigmoid(diff).mean()

            # DPOP term
            pos_pen = torch.clamp(logp_yw_ref - logp_yw_theta, min=0.0)
            loss_pos = lam * pos_pen.mean()

            loss = loss_dpo + loss_pos

            # backward
            loss = loss / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                pbar.set_postfix({
                    "step": step,
                    "loss": float(loss.item() * grad_accum_steps),
                    "dpo": float(loss_dpo.item()),
                    "pos": float(loss_pos.item())
                })

            if (step + 1) % save_every_steps == 0:
                ckpt_dir = os.path.join(out_dir, f"ckpt-step-{step+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                # Save optimizer + scheduler + peft adapter state for safe resume
                torch.save({
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }, os.path.join(ckpt_dir, "optim_sched.pt"))
                # Save adapter if PEFT supports
                try:
                    model.save_pretrained(ckpt_dir)
                except Exception:
                    pass

            step += 1

    # Final save
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Save optimizer state final
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }, os.path.join(out_dir, "optim_sched.pt"))
    print("Training finished, saved to", out_dir)


# -------------------------
# Utilities: building phase datasets according to composition
# -------------------------
def load_jsonl(path: Path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                arr.append(json.loads(line))
            except:
                continue
    return arr


def normalize_pair_entry(entry: dict):
    """
    Normalize multiple input schemas to standard fields:
    returns dict with keys: prompt, context, ideal, fake, meta
    """
    prompt = entry.get("prompt") or entry.get("question") or ""
    context = entry.get("context") or entry.get("ctx") or ""
    ideal = entry.get("chosen") or entry.get("ideal_answer") or entry.get("ideal") or ""
    fake = entry.get("rejected") or entry.get("fake_answer") or entry.get("fake") or ""
    meta = entry.get("meta", {k: entry.get(k) for k in ("type", "data_index") if k in entry})
    return {"prompt": prompt, "context": context, "ideal": ideal, "fake": fake, "meta": meta}


def build_phase_dataset(easy_entries, med_entries, hard_entries):
    """
    Simplified deterministic slicing strategy:
    Phase 1 (Easy): First 5k Easy
    Phase 2 (Med):  Next 2k Easy + First 11k Med
    Phase 3 (Hard): Last 1k Easy + Last 4k Med + All 20k Hard
    """
    
    # Helper: form ideal vs X pair object
    def ideal_vs(fake_entry):
        return {
            "context": fake_entry["context"],
            "prompt": fake_entry["prompt"],
            "chosen": fake_entry["ideal"],
            "rejected": fake_entry["fake"],
            "meta": fake_entry["meta"]
        }

    # 1. Convert all raw entries to clean pair objects
    # We assume the input files are sorted or randomized enough. 
    # If you want them shuffled before slicing, uncomment random.shuffle below.
    pool_easy = [ideal_vs(e) for e in easy_entries if e["ideal"] and e["fake"]]
    pool_med = [ideal_vs(e) for e in med_entries if e["ideal"] and e["fake"]]
    pool_hard = [ideal_vs(e) for e in hard_entries if e["ideal"] and e["fake"]]

    # Optional: Shuffle raw pools once so 'first 5k' isn't just alphabetical
    random.seed(42)
    random.shuffle(pool_easy)
    random.shuffle(pool_med)
    random.shuffle(pool_hard)

    print(f"Pool Sizes -> Easy: {len(pool_easy)}, Med: {len(pool_med)}, Hard: {len(pool_hard)}")

    # 2. Strict Slicing (The Strategy)
    
    # --- Phase 1 (Easy) ---
    # "For epoch 1, take the first 5k of the easy pairs"
    phase1_data = pool_easy[:5000]

    # --- Phase 2 (Medium) ---
    # "For epoch 2, take the next 2k of the easy pairs and first 11k of the medium pairs"
    p2_easy = pool_easy[5000:7000]
    p2_med = pool_med[:11000]
    phase2_data = p2_easy + p2_med

    # --- Phase 3 (Hard) ---
    # "For epoch 3, take the last 1k of easy pairs, last 4k of medium pairs and all 20k hard"
    # Note: We use slices to be safe in case counts are slightly off
    p3_easy = pool_easy[7000:8000] 
    p3_med = pool_med[11000:15000]
    p3_hard = pool_hard[:20000]
    phase3_data = p3_easy + p3_med + p3_hard

    # 3. Shuffle within phases (Important so model doesn't see all Easy then all Meds sorted)
    random.shuffle(phase1_data)
    random.shuffle(phase2_data)
    random.shuffle(phase3_data)

    return {
        "easy": phase1_data,
        "medium": phase2_data,
        "hard": phase3_data
    }


# -------------------------
# Main entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/llama3-2-3b-instruct", help="Local path to base Llama 3.2 3B")
    parser.add_argument("--data-dir", type=str, default="./datasets/main", help="Directory containing dpo_pairs_easy.jsonl etc")
    parser.add_argument("--out-dir", type=str, default="./dpop_out")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)

    # --- RESTORED hyperparameters (as requested) ---
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for optimizer (recommended for LoRA r~16)")
    parser.add_argument("--beta", type=float, default=0.3, help="DPO beta")
    parser.add_argument("--lam", type=float, default=50.0, help="DPOP lambda")
    parser.add_argument("--epochs_easy", type=int, default=1)
    parser.add_argument("--epochs_med", type=int, default=1)
    parser.add_argument("--epochs_hard", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=2000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--load-adapter-path", type=str, default=None, help="Path to saved PEFT adapter (optional)")
    args = parser.parse_args()

    model_path = args.model_path
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    device = args.device

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        print("No pad token found. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        print("WARNING: No chat template found in tokenizer. Using default Llama 3 template.")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'user' %}"
            "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'assistant' %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{% endif %}"
        )

    print("Loading base model (fp16) from", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- REVERT LoRA config to safer defaults (commented previous aggressive config) ---
    # OLD aggressive LoRA config (commented out)
    # lora_config = LoraConfig(
    #     r=64,
    #     lora_alpha=128,
    #     target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
    # Safer LoRA config:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    print("Model in training mode:", model.training)

    if args.load_adapter_path:
        print(f"Loading adapter from: {args.load_adapter_path}")
        # adapter loading API may differ between versions
        try:
            model.load_adapter(args.load_adapter_path, "default")
            print("Adapter loaded successfully.")
        except Exception as e:
            print("Could not load adapter:", e)

    # -------------------------
    # Build Phase Datasets
    # -------------------------
    easy_path = data_dir / "dpo_pairs_easy.jsonl"
    med_path = data_dir / "dpo_pairs_medium.jsonl"
    hard_path = data_dir / "dpo_pairs_hard.jsonl"

    easy_raw = load_jsonl(easy_path) if easy_path.exists() else []
    med_raw = load_jsonl(med_path) if med_path.exists() else []
    hard_raw = load_jsonl(hard_path) if hard_path.exists() else []

    print(f"Loaded raw counts: easy={len(easy_raw)}, med={len(med_raw)}, hard={len(hard_raw)}")

    # normalize pools
    easy_norm = [normalize_pair_entry(x) for x in easy_raw]
    med_norm = [normalize_pair_entry(x) for x in med_raw]
    hard_norm = [normalize_pair_entry(x) for x in hard_raw]

    # target counts requested by user
    # target_counts = {"easy": 8000, "medium": 15000, "hard": 20000}

    phase_sets = build_phase_dataset(easy_norm, med_norm, hard_norm)

    # Save constructed phase JSONL for inspection
    phase_dir = out_dir / "phases"
    phase_dir.mkdir(parents=True, exist_ok=True)
    for name in ("easy", "medium", "hard"):
        with open(phase_dir / f"dpo_phase_{name}.jsonl", "w", encoding="utf-8") as fw:
            for item in phase_sets[name]:
                fw.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved phase {name} -> {phase_dir / f'dpo_phase_{name}.jsonl'}  (count={len(phase_sets[name])})")

    # -------------------------
    # CURRICULUM TRAINING: iterate phases
    # -------------------------
    # For each phase, create DataLoader and train for specified epochs (args.epochs_*)
    phase_order = [
        ("easy", phase_sets["easy"], args.epochs_easy),
        ("medium", phase_sets["medium"], args.epochs_med),
        ("hard", phase_sets["hard"], args.epochs_hard),
    ]

    for phase_name, phase_items, phase_epochs in phase_order:
        if not phase_items:
            print(f"Skipping phase {phase_name}: no items.")
            continue

        print(f"\n=== Starting phase {phase_name} | samples={len(phase_items)} | epochs={phase_epochs} ===")

        # Create an in-memory dataset for the phase
        # We'll write phase_items to a temporary jsonl and load via PairDataset for normalization convenience
        temp_path = out_dir / f"temp_phase_{phase_name}.jsonl"
        with open(temp_path, "w", encoding="utf-8") as fw:
            for it in phase_items:
                fw.write(json.dumps(it, ensure_ascii=False) + "\n")

        ds_phase = PairDataset([temp_path], tokenizer)

        dataloader = DataLoader(
            ds_phase,
            batch_size=args.per_device_batch_size,
            shuffle=True,  # shuffle within phase
            collate_fn=lambda b: collate_pairs(b, tokenizer),
            num_workers=2,
            pin_memory=True
        )

        # call training loop for this phase
        phase_out_dir = out_dir / f"phase_{phase_name}"
        phase_out_dir.mkdir(parents=True, exist_ok=True)

        dpop_training_loop(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=dataloader,
            device=device,
            out_dir=str(phase_out_dir),
            beta=args.beta,
            lam=args.lam,
            epochs=phase_epochs,
            lr=args.lr,
            grad_accum_steps=args.grad_accum_steps,
            save_every_steps=args.save_every_steps,
            resume_step=args.resume_step,
            optimizer_state_path=phase_out_dir / "optim_sched.pt"
        )

        # cleanup temp file
        try:
            temp_path.unlink()
        except Exception:
            pass

    print("Curriculum training complete.")

if __name__ == "__main__":
    main()
