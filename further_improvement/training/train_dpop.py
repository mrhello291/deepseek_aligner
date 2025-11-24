#!/usr/bin/env python3
"""
train_dpop_robust_fixed.py

- Resume support (auto-detect latest ckpt_*)
- Proper PEFT/LoRA adapter loading for resume
- log-prob computed in FP32 and masked correctly (no shift bug)
- Robust checkpointing: adapter + tokenizer + optim_sched.pt
- Gradient-accum flush for the final partial step
- Basic logging to CSV
"""
import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model

# -------------------------
# 1) Data loading & sampling
# -------------------------
def load_jsonl(path: str):
    arr = []
    path = Path(path)
    if not path.exists():
        print(f"Warning: {path} not found")
        return arr
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    arr.append(json.loads(line))
                except:
                    continue
    return arr

def normalize_entry(e: dict):
    return {
        "context": e.get("context", e.get("knowledge", "")),
        "prompt": e.get("prompt", e.get("question", "")),
        "chosen": e.get("chosen", e.get("ideal_answer", "")),
        "rejected": e.get("rejected", e.get("fake_answer", "")),
    }

def get_mixed_dataset(data_dir):
    print("Loading datasets...")
    hard_raw = load_jsonl(Path(data_dir) / "hard_fakes_vllm.jsonl")
    med_raw = load_jsonl(Path(data_dir) / "dpo_pairs_medium.jsonl")

    hard_selected = hard_raw[:6000]
    med_selected = med_raw[:600]

    print(f"Selection: {len(hard_selected)} Hard + {len(med_selected)} Medium")

    combined = [normalize_entry(e) for e in hard_selected] + \
               [normalize_entry(e) for e in med_selected]

    random.seed(42)
    random.shuffle(combined)

    return combined

# -------------------------
# 2) Truncation / preprocessor
# -------------------------
class DPOProcessor:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skipped_count = 0
        self.truncated_count = 0

    def process_batch(self, raw_data):
        processed = []
        print(f"Pre-processing {len(raw_data)} items with Context Truncation...")
        for entry in tqdm(raw_data):
            try:
                tokenized = self._process_single(entry)
                if tokenized:
                    processed.append(tokenized)
            except Exception:
                self.skipped_count += 1
        print(f"\n--- Pre-processing Stats ---")
        print(f"Total Input: {len(raw_data)}")
        print(f"Kept:        {len(processed)}")
        print(f"Skipped:     {self.skipped_count} (Answers too long to fit in {self.max_length})")
        print(f"Truncated:   {self.truncated_count} (Contexts trimmed to fit)")
        print(f"----------------------------\n")
        return processed

    def _process_single(self, entry):
        msgs_no_ctx = [
            {"role": "system", "content": "You are a careful and honest assistant. Always answer using ONLY the information provided in the context. If the context does not contain the answer, say you don't know. Do not guess or invent facts."},
            {"role": "user", "content": f"Question: {entry['prompt']}\n\nContext: "}
        ]

        # Get prefix length in token IDs
        # apply_chat_template with tokenize=True returns ids (model/tokenizer dependent)
        prefix_ids = self.tokenizer.apply_chat_template(msgs_no_ctx, tokenize=True, add_generation_prompt=True)
        # content ids
        ctx_ids = self.tokenizer.encode(entry.get('context', ""), add_special_tokens=False)
        chosen_ids = self.tokenizer.encode((entry.get('chosen') or "") + (self.tokenizer.eos_token or ""), add_special_tokens=False)
        rejected_ids = self.tokenizer.encode((entry.get('rejected') or "") + (self.tokenizer.eos_token or ""), add_special_tokens=False)

        max_ans_len = max(len(chosen_ids), len(rejected_ids))
        base_len = len(prefix_ids) + max_ans_len
        avail_ctx = self.max_length - base_len

        if avail_ctx < 0:
            self.skipped_count += 1
            return None

        final_ctx_ids = ctx_ids
        if len(ctx_ids) > avail_ctx:
            final_ctx_ids = ctx_ids[:avail_ctx]
            self.truncated_count += 1

        final_ctx_str = self.tokenizer.decode(final_ctx_ids, skip_special_tokens=True)
        final_msgs = [
            {"role": "system", "content": "You are a careful and honest assistant. Always answer using ONLY the information provided in the context. If the context does not contain the answer, say you don't know. Do not guess or invent facts."},
            {"role": "user", "content": f"Context: {final_ctx_str}\n\nQuestion: {entry['prompt']}"}
        ]
        final_prompt_str = self.tokenizer.apply_chat_template(final_msgs, tokenize=False, add_generation_prompt=True)
        return {
            "prompt_text": final_prompt_str,
            "chosen_text": entry.get("chosen", ""),
            "rejected_text": entry.get("rejected", "")
        }

# -------------------------
# 3) Dataset & collate
# -------------------------
class DPODataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch, tokenizer):
    prompts = [b['prompt_text'] for b in batch]
    chosen = [(b['chosen_text'] or "") + (tokenizer.eos_token or "") for b in batch]
    rejected = [(b['rejected_text'] or "") + (tokenizer.eos_token or "") for b in batch]

    inputs_w = tokenizer([p + c for p, c in zip(prompts, chosen)],
                         padding=True, truncation=True, max_length=1024, return_tensors="pt", add_special_tokens=True)
    inputs_l = tokenizer([p + r for p, r in zip(prompts, rejected)],
                         padding=True, truncation=True, max_length=1024, return_tensors="pt", add_special_tokens=True)

    # prompt_tok = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt", add_special_tokens=True)
    prompt_tok = tokenizer(prompts, padding=False, truncation=True, max_length=1024, return_tensors=None, add_special_tokens=True)
    # prompt_lens = prompt_tok.attention_mask.sum(dim=1)  # number of tokens in prompt (unpadded)
    prompt_lens = torch.tensor(
        [len(x) for x in prompt_tok["input_ids"]],
        dtype=torch.long
    )

    return {
        "input_ids_w": inputs_w["input_ids"],
        "attention_w": inputs_w["attention_mask"],
        "input_ids_l": inputs_l["input_ids"],
        "attention_l": inputs_l["attention_mask"],
        "prompt_lens": prompt_lens
    }

# -------------------------
# 4) Stable vectorized log-prob (no shift bug, FP32)
# -------------------------
def get_batch_logps(model, input_ids, attention_mask, prompt_lens, device):
    """
    Compute summed log-prob of the RESPONSE tokens (i.e., tokens with index >= prompt_lens)
    Returns tensor shape (B,)
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_lens = prompt_lens.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # logits = outputs.logits  # (B, L, V)

    # # Compute log-probs in FP32 for stability
    # log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)  # (B, L, V)
    # logits = outputs.logits.float()          # compute logits in FP32
    # log_probs = torch.nn.functional.log_softmax(logits.to(torch.float16), dim=-1).float()  
    logits_fp32 = outputs.logits.float()
    log_probs = torch.nn.functional.log_softmax(logits_fp32, dim=-1)

    B, L = input_ids.shape
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)

    # Adjust prompt_lens for the shift (token t predicted at t-1)
    prompt_lens_shifted = (prompt_lens - 1).clamp(min=0)
    # response_mask: True for tokens that are part of the response (>= prompt_len) AND not padding
    response_mask = (positions >= prompt_lens_shifted.unsqueeze(1)) & attention_mask.bool()
    
    # we cannot score token 0 because there's no previous token prediction for it; zero it out
    response_mask = response_mask.clone()
    response_mask[:, 0] = False

    # time index for next-token prediction: logits at t predict token at t+1,
    # so the prediction used for token at position p is logits at time index p-1.
    time_idx = (positions - 1).clamp(min=0)  # (B, L)

    # pred_log_probs shape: (B, L, V) -> log_probs taken at previous timestep
    pred_log_probs = log_probs.gather(1, time_idx.unsqueeze(-1).expand(B, L, log_probs.size(-1)))

    # token log-probs for actual tokens at each position
    token_log_probs = pred_log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)

    # zero out non-response tokens
    token_log_probs = token_log_probs * response_mask.float()

    # sum over tokens
    return token_log_probs.sum(dim=1)

# -------------------------
# 5) Training (with resume + adapter load)
# -------------------------
def find_latest_ckpt(out_dir: Path):
    ckpts = sorted([p for p in out_dir.glob("ckpt_*") if p.is_dir()],
                   key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
    return ckpts[-1] if len(ckpts) else None

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model (fp16/bf16 depending on user choice); keep dtype consistent with your infra
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # PEFT/LoRA config (you intentionally cover more modules)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Create PEFT wrapper (this registers adapter slots)
    model = get_peft_model(base_model, peft_config)

    # Resume logic: detect latest checkpoint and load adapter + optimizer state if requested
    resume_ckpt = None
    resume_step = 0
    if args.resume:
        resume_ckpt = find_latest_ckpt(out_dir)
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")
            # Load adapter into the peft-registered model and make it trainable
            # model.load_adapter expects the adapter dir; PEFT will find adapter_model.safetensors there.
            try:
                model.load_adapter(str(resume_ckpt), "default", is_trainable=True)
                model.set_adapter("default")
                print("Loaded adapter from checkpoint.")
            except Exception as e:
                print("Failed to load adapter via load_adapter():", e)
                print("Attempting PeftModel.from_pretrained fallback...")
                # Fallback: instantiate fresh base and load via PeftModel if needed
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(base_model, resume_ckpt)
                    print("Loaded adapter via PeftModel.from_pretrained.")
                except Exception as e2:
                    raise RuntimeError("Could not load adapter from checkpoint") from e2

            # load optimizer/scheduler if available
            opt_path = resume_ckpt / "optim_sched.pt"
            if opt_path.exists():
                print("Loading optimizer/scheduler state...")
                try:
                    state = torch.load(opt_path, map_location="cpu")
                    # we will load optimizer state after optimizer is created below
                    resume_opt_state = state
                    # resume_step inferred from checkpoint name if numeric suffix present
                    try:
                        resume_step = int(resume_ckpt.name.split("_")[-1])
                    except:
                        resume_step = 0
                except Exception as e:
                    print("Could not load optimizer state:", e)
                    resume_opt_state = None
            else:
                resume_opt_state = None
        else:
            print("No checkpoint found; starting fresh.")
            resume_opt_state = None
    else:
        resume_opt_state = None

    # Move model to device
    model = model.to(device)
    model.print_trainable_parameters()

    # DATA prep
    raw_data = get_mixed_dataset(args.data_dir)
    processor = DPOProcessor(tokenizer, max_length=1024)
    clean_data = processor.process_batch(raw_data)

    dataset = DPODataset(clean_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer), num_workers=2, pin_memory=True)

    # Optimizer & scheduler
    # Only parameters that require grad (LoRA params) will be optimized
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_training_steps = max(1, (len(loader) * args.epochs) // max(1, args.grad_accum))
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=50, num_training_steps=total_training_steps)

    # If resuming, load optimizer/scheduler state
    if resume_opt_state is not None:
        try:
            optimizer.load_state_dict(resume_opt_state["optimizer"])
            scheduler.load_state_dict(resume_opt_state["scheduler"])
            print("Optimizer and scheduler state restored.")
        except Exception as e:
            print("Warning: could not restore full optimizer/scheduler state:", e)

    # Logging file
    log_path = out_dir / "training_log.csv"
    log_f = open(log_path, "a")
    if log_f.tell() == 0:
        log_f.write("step,loss,dpo_loss,pos_loss\n")

    # TRAIN LOOP
    model.train()
    global_step = resume_step
    accum_counter = 0  # counts micro-steps towards grad accumulation

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        prog = tqdm(loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(prog):
            # move batch to device
            input_ids_w = batch["input_ids_w"].to(device)
            att_w = batch["attention_w"].to(device)
            input_ids_l = batch["input_ids_l"].to(device)
            att_l = batch["attention_l"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            # compute policy logprobs (trainable)
            policy_logps_w = get_batch_logps(model, input_ids_w, att_w, prompt_lens, device)
            policy_logps_l = get_batch_logps(model, input_ids_l, att_l, prompt_lens, device)

            # compute ref logprobs with adapters disabled
            with model.disable_adapter(), torch.no_grad():
                ref_logps_w = get_batch_logps(model, input_ids_w, att_w, prompt_lens, device)
                ref_logps_l = get_batch_logps(model, input_ids_l, att_l, prompt_lens, device)

            # DPO / DPOP losses
            logits_diff = (policy_logps_w - ref_logps_w) - (policy_logps_l - ref_logps_l)
            # clamp logits_diff for numerical stability
            # logits_diff = logits_diff.clamp(min=-50.0, max=50.0)
            # dpo_loss = -torch.nn.functional.logsigmoid(args.beta * logits_diff).mean()
            dpo_arg = args.beta * logits_diff
            dpo_arg = dpo_arg.clamp(min=-50.0, max=50.0)
            dpo_loss = -torch.nn.functional.logsigmoid(dpo_arg).mean()

            log_ratio = ref_logps_w - policy_logps_w
            pos_loss = args.lam * torch.clamp(log_ratio, min=0.0).mean()

            loss = dpo_loss + pos_loss
            (loss / args.grad_accum).backward()
            accum_counter += 1

            # perform optimizer.step() when enough micro-batches accumulated
            if accum_counter >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_counter = 0
                global_step += 1

                # log & progress bar
                log_f.write(f"{global_step},{loss.item():.6f},{dpo_loss.item():.6f},{pos_loss.item():.6f}\n")
                log_f.flush()
                prog.set_postfix({"loss": f"{loss.item():.4f}", "pos": f"{pos_loss.item():.4f}"})

                # checkpointing
                if global_step % args.save_every == 0:
                    ckpt_dir = out_dir / f"ckpt_{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    # Save adapter (PEFT-aware)
                    try:
                        model.save_pretrained(ckpt_dir)
                    except Exception:
                        # fallback: save tokenizer + adapter via peft state dict
                        try:
                            model.save_pretrained(ckpt_dir)
                        except Exception as e:
                            print("Warning: saving adapter failed:", e)
                    tokenizer.save_pretrained(ckpt_dir)
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }, ckpt_dir / "optim_sched.pt")
                    print(f"Saved checkpoint {ckpt_dir}")

        # after each epoch, if there are leftover grads (partial accumulation), flush them
        if accum_counter > 0:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_counter = 0
            global_step += 1
            log_f.write(f"{global_step},{0.0:.6f},{0.0:.6f},{0.0:.6f}\n")
            log_f.flush()
            print("Flushed final partial accumulation for epoch.")

    # Final save
    final_dir = out_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }, final_dir / "optim_sched.pt")
    log_f.close()
    print("Training finished. Saved final model to", final_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/llama3-2-3b-instruct")
    parser.add_argument("--data_dir", type=str, default="./datasets/main")
    parser.add_argument("--out_dir", type=str, default="dpop_results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    train(args)
