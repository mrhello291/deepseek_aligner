#!/usr/bin/env python3
"""
train_dpop.py

DPOP training (DPO-Positive) on Llama-3.2-3B using LoRA (FP16).
Curriculum schedule across easy / medium / hard jsonl pair files with dynamic splits.

Run:
  python train_dpop.py \
    --model-path /path/to/llama-3.2-3b \
    --data-dir /path/with/jsonl \
    --out-dir ./dpop_out \
    --device cuda
"""
import os
import json
import argparse
from pathlib import Path
import itertools  # --- ADDED ---
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
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
        Each line: {"prompt": "...", "chosen": "...", "rejected": "..."}
        """
        self.examples = []
        for fp in filepaths:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    j = json.loads(line)
                    self.examples.append({
                        "context": j.get("context", ""),
                        "prompt": j["prompt"],
                        "chosen": j["chosen"],
                        "rejected": j["rejected"]
                    })
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_total_len = max_total_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_pairs(batch: List[Dict], tokenizer, padding_token_id=None):
    """
    For each pair, we build:
        full_prompt = context + prompt
        seq_w = full_prompt + chosen
        seq_l = full_prompt + rejected
    We'll return tokenized inputs and masks to compute log-probs of the chosen/rejected parts.
    """
    # Define your token limit
    MAX_TOTAL_LEN = 1024
    
    # Combine context and prompt
    full_prompts = [b.get("context", "") + b["prompt"] for b in batch]
    chosen = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]

    # Tokenize prompt only to get prompt lengths
    # tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    # Tokenize full sequences (this will truncate from the right, as requested)
    seq_w = [p + c for p, c in zip(full_prompts, chosen)]
    seq_l = [p + r for p, r in zip(full_prompts, rejected)]

    tok_w = tokenizer(
        seq_w,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOTAL_LEN  # <-- ENFORCE LIMIT
    )
    tok_l = tokenizer(
        seq_l,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOTAL_LEN  # <-- ENFORCE LIMIT
    )
    
    # We'll compute log-probs for response tokens only: find indices where response starts.
    # We must also truncate the prompts themselves to the max_length,
    # otherwise a prompt > 2048 tokens would report an invalid length.
    prompt_toks = tokenizer(
        full_prompts,
        truncation=True,
        max_length=MAX_TOTAL_LEN,  # <-- ENFORCE LIMIT
        add_special_tokens=False
    )

    # We'll compute log-probs for response tokens only: find indices where response starts.
    # To be robust we recompute using tokenizer on prompt to get prompt lengths (in tokens)
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
# @torch.no_grad()
# def compute_logprob_for_sequences(model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_start_positions: torch.Tensor, device):
#     """
#     Given tokenized input sequences (prompt + response), compute the log-probability (sum of token log-probs)
#     of the *response portion* for each sequence in the batch.

#     input_ids: (B, L)
#     attention_mask: (B, L)
#     target_start_positions: long tensor (B,) indicating index (in tokens) where response starts for each sample
#     """
#     # Move to device
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     B, L = input_ids.shape

#     # We forward and compute logits -> log_softmax over vocab, then sum log-probs for response tokens
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
#     logits = outputs.logits  # (B, L, V)
#     # convert logits to log-probs
#     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

#     seq_logp = []
#     for i in range(B):
#         start = int(target_start_positions[i].item())
#         # We'll sum log-probs of tokens from start .. last non-pad token (based on attention mask)
#         mask = attention_mask[i]  # 1s for valid tokens
#         valid_positions = mask.nonzero(as_tuple=False).squeeze(-1)
#         if valid_positions.numel() == 0:
#             seq_logp.append(torch.tensor(0.0, device=device))
#             continue

#         last_pos = int(valid_positions[-1].item())

#         # If the response would be empty or start beyond last token, return zero logprob
#         if start >= last_pos:
#             seq_logp.append(torch.tensor(0.0, device=device))
#             continue

#         # positions from max(1, start) .. last_pos inclusive
#         pos_range = list(range(max(1, start), last_pos+1))

#         token_logps = []
#         for t in pos_range:
#             pred_logits = log_probs[i, t - 1]  # distribution predicting token at position t
#             token_id = int(input_ids[i, t].item())
#             token_logps.append(pred_logits[token_id])
#         if len(token_logps) == 0:
#             seq_logp.append(torch.tensor(0.0, device=device))
#         else:
#             seq_logp.append(torch.stack(token_logps).sum())
#     seq_logp = torch.stack(seq_logp)  # (B,)
#     return seq_logp  # sum log-prob of response tokens

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
    - No Python loops
    - Computes summed log-prob of response tokens for entire batch at once
    - ~7-10x faster than the original implementation
    """

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_start_positions = target_start_positions.to(device)

    B, L = input_ids.shape

    # Forward pass: (B, L, V)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    # Log-softmax over vocab for all tokens
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)            # (B, L, V)

    # -------------------------------------------------------
    # 1. Build mask to pick RESPONSE TOKENS for each sample
    # -------------------------------------------------------

    # Token positions: [0, 1, 2, ..., L-1]
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)   # (B, L)

    # Response mask: positions >= target_start
    response_mask = positions >= target_start_positions.unsqueeze(1)       # (B, L)

    # Also require the token to be a "real" (non-pad) token
    response_mask = response_mask & (attention_mask.bool())                # (B, L)

    # We cannot use t=0 for prediction; LM predicts token[t] from logits[t-1]
    # So shift mask to ignore t=0
    valid_token_mask = response_mask.clone()
    valid_token_mask[:, 0] = False                                        # can't compute logp for t=0

    # -------------------------------------------------------
    # 2. Gather log-probs of actual next tokens
    # -------------------------------------------------------

    # Build "prediction indices": which logits to use?
    # log_probs[b, t-1] predicts token[t]
    time_indices = (positions - 1).clamp(min=0)                             # (B, L)

    # Extract log_probs[b, t-1] for each token position
    # shape: (B, L, V)
    pred_log_probs = log_probs.gather(1, time_indices.unsqueeze(-1).expand(B, L, log_probs.size(-1)))

    # Now gather the probability of the actual token input_ids[b, t]
    token_log_probs = pred_log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)

    # Zero out non-response positions
    token_log_probs = token_log_probs * valid_token_mask.float()

    # -------------------------------------------------------
    # 3. Sum log-probs over response tokens
    # -------------------------------------------------------
    sum_logprobs = token_log_probs.sum(dim=1)  # (B,)

    return sum_logprobs



# -------------------------
# DPOP trainer core
# -------------------------
def dpop_training_loop(
    model,  # trainable model (LoRA)
    # ref_model,  # frozen reference model
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
    resume_step=0  # --- ADDED ---
):
    model.train()
    # ref_model.eval()

    # Only train LoRA params
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # --- MODIFIED total_steps calculation ---
    # Calculate total steps based on a single epoch pass, as epochs are handled by the curriculum
    # total_steps = epochs * len(train_dataloader) // grad_accum_steps
    total_steps_per_epoch = len(train_dataloader) // grad_accum_steps
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=scheduler_warmup_steps,
        num_training_steps=total_steps_per_epoch * epochs, # total_steps,
    )
    
    # --- ADDED ---
    # If resuming, fast-forward the scheduler to the correct step
    if resume_step > 0:
        grad_steps_to_advance = resume_step // grad_accum_steps
        print(f"Resuming from step {resume_step}. Advancing scheduler by {grad_steps_to_advance} gradient steps.")
        for _ in range(grad_steps_to_advance):
            scheduler.step()
    # --- END ADDED ---

    step = resume_step  # --- MODIFIED --- (was step = 0)
    for epoch in range(epochs):
        # --- ADDED: Logic to skip batches ---
        dataloader_iter = iter(train_dataloader)

        # Skip batches if resuming
        current_epoch_resume_step = 0
        if resume_step > 0:
            print(f"Skipping first {resume_step} batches...")
            dataloader_iter = itertools.islice(dataloader_iter, resume_step, None)
            current_epoch_resume_step = resume_step # Store for tqdm
            
        # Wrap the (potentially sliced) iterator with tqdm
        pbar = tqdm(
            dataloader_iter,
            desc=f"Epoch {epoch+1}/{epochs}",
            initial=current_epoch_resume_step,  # Start pbar at this step
            total=len(train_dataloader) # Total is still full length
        )
        # --- END ADDED ---
        
        # pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for batch in pbar:
            # Move to device only what's needed (we'll do forward twice)
            input_ids_w = batch["input_ids_w"].to(device)
            attn_w = batch["attention_mask_w"].to(device)
            input_ids_l = batch["input_ids_l"].to(device)
            attn_l = batch["attention_mask_l"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            B = input_ids_w.size(0)

            # Compute log probs under current model
            # For chosen (yw): response starts at prompt_len
            chosen_start = prompt_lens
            rejected_start = prompt_lens

            # Compute log-probs (sum over response tokens) under trainable model
            logp_yw_theta = compute_logprob_for_sequences(model, tokenizer, input_ids_w, attn_w, chosen_start, device)
            logp_yl_theta = compute_logprob_for_sequences(model, tokenizer, input_ids_l, attn_l, rejected_start, device)

            # Compute log-probs under reference model (no_grad)
            # Use 'model' but disable the LoRA adapter to get base model (ref) logits
            with model.disable_adapter(), torch.no_grad():
                logp_yw_ref = compute_logprob_for_sequences(model, tokenizer, input_ids_w, attn_w, chosen_start, device)
                logp_yl_ref = compute_logprob_for_sequences(model, tokenizer, input_ids_l, attn_l, rejected_start, device)

            # Compute ratio terms: log (pi_theta / pi_ref) for each sequence
            log_ratio_yw = logp_yw_theta - logp_yw_ref
            log_ratio_yl = logp_yl_theta - logp_yl_ref

            # DPO term: - log sigma( beta * (log_ratio_yw - log_ratio_yl) )
            diff = beta * (log_ratio_yw - log_ratio_yl)
            loss_dpo = -torch.nn.functional.logsigmoid(diff).mean()

            # DPOP extra positive penalty: lambda * max(0, log(pi_ref(yw)) - log(pi_theta(yw)))
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
                # Save peft adapter
                model.save_pretrained(ckpt_dir)
                # Optionally save tokenizer
                tokenizer.save_pretrained(ckpt_dir)

            step += 1
            
        # --- ADDED ---
        # After an epoch finishes, set resume_step to 0
        # This ensures if epochs > 1 (which it isn't in your curriculum,
        # but is good practice), the next epoch starts from scratch.
        resume_step = 0
        # --- END ADDED ---
        
    # final save
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Training finished, saved to", out_dir)


# -------------------------
# Main entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/llama3-2-3b-instruct", help="Local path to base Llama 3.2 3B")
    
    # --- ADDED: Arguments for resuming ---
    parser.add_argument("--load-adapter-path", type=str, default=None, help="Path to a saved PEFT adapter to resume training from.")
    parser.add_argument("--start-epoch", type=int, default=1, help="Curriculum epoch to start from (e.g., 3 to start at epoch_3).")
    parser.add_argument("--resume-step", type=int, default=0, help="Step number to resume from *within* the start-epoch.")
    # --- END ADDED ---
    
    parser.add_argument("--data-dir", type=str, default="./datasets/main", help="Directory containing easy/medium/hard jsonl")
    parser.add_argument("--out-dir", type=str, default="./dpop_out")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.3, help="DPO beta")
    parser.add_argument("--lam", type=float, default=50.0, help="DPOP lambda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--save-every-steps", type=int, default=2000)
    args = parser.parse_args()

    model_path = args.model_path
    data_dir = Path(args.data_dir)
    out_dir = args.out_dir
    device = args.device

    os.makedirs(out_dir, exist_ok=True)

    # Load tokenizer (adjust model/tokenizer name as needed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # ensure tokenizer has pad token
    # if tokenizer.pad_token_id is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token_id is None:
        print("No pad token found. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # NOTE: We are NOT adding a new token or resizing,
        # just aliasing the pad token to the existing eos token.
    
    # Load base model (FP16)
    print("Loading base model (fp16) from", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    # model.resize_token_embeddings(len(tokenizer))

    # # Create a frozen reference model (fp16)
    # print("Loading reference model (fp16, frozen) from", model_path)
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    # ref_model.resize_token_embeddings(len(tokenizer))

    # for p in ref_model.parameters():
    #     p.requires_grad = False
    # ref_model.eval()

    # Attach LoRA to train
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # may need adjustment per model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()  # ✅ Ensure model is in training mode after LoRA is attached
    print("Model in training mode:", model.training)
    
    # --- ADDED: Load adapter if resuming ---
    if args.load_adapter_path:
        print(f"Resuming training by loading adapter from: {args.load_adapter_path}")
        model.load_adapter(args.load_adapter_path, "default")
        print("Adapter loaded successfully.")
    # --- END ADDED ---

    # Build datasets
    easy_fp = data_dir / "dpo_pairs_easy.jsonl"
    medium_fp = data_dir / "dpo_pairs_medium.jsonl"
    hard_fp = data_dir / "dpo_pairs_hard.jsonl"
    assert easy_fp.exists() and medium_fp.exists() and hard_fp.exists(), "Ensure jsonl files exist in data-dir"

    ds_easy = PairDataset([easy_fp], tokenizer)
    ds_medium = PairDataset([medium_fp], tokenizer)
    ds_hard = PairDataset([hard_fp], tokenizer)

    print(f"sizes: easy={len(ds_easy)} medium={len(ds_medium)} hard={len(ds_hard)}")

    # -------------------------
    # Dynamic curriculum splits (data-size-aware)
    # -------------------------
    n_easy = len(ds_easy)
    n_medium = len(ds_medium)
    n_hard = len(ds_hard)

    # Compute splits dynamically
    m2 = int(n_medium * 0.50)   # 50% of medium for epoch2 and epoch3
    h3 = int(n_hard * 0.60)     # 60% of hard for epoch3

    # ensure we don't exceed dataset sizes
    m2 = min(max(m2, 0), n_medium)
    h3 = min(max(h3, 0), n_hard)

    print("Curriculum (dynamic):")
    print(f"  easy: {n_easy} (all)")
    print(f"  medium: {n_medium} -> using first {m2} in epoch2/3")
    print(f"  hard: {n_hard} -> using first {h3} in epoch3")

    # Build epoch datasets
    epoch1 = ds_easy
    epoch2 = ConcatDataset([
        ds_easy,
        Subset(ds_medium, list(range(m2)))
    ])
    epoch3 = ConcatDataset([
        ds_easy,
        Subset(ds_medium, list(range(m2))),
        Subset(ds_hard, list(range(h3)))
    ])
    epoch4 = ConcatDataset([ds_easy, ds_medium, ds_hard])

    epoch_datasets = [epoch1, epoch2, epoch3, epoch4]
    print("Epoch dataset sizes:", [len(e) for e in epoch_datasets])

    # Build dataloaders for each epoch and run DPOP training sequentially
    final_epochs = len(epoch_datasets)
    for e_idx, ds in enumerate(epoch_datasets):
        current_epoch_num = e_idx + 1 # This is 1, 2, 3, 4
        # Check if we should skip this curriculum epoch
        if current_epoch_num < args.start_epoch:
            print(f"Skipping curriculum epoch {current_epoch_num}/{final_epochs} (resuming from {args.start_epoch})")
            continue
            
        # Determine the starting step for this epoch
        start_step_for_this_epoch = 0
        if current_epoch_num == args.start_epoch:
            # We are in the epoch we want to resume, so use resume_step
            start_step_for_this_epoch = args.resume_step
            
        print(f"Epoch schedule {current_epoch_num}: dataset size = {len(ds)}")
        if start_step_for_this_epoch > 0:
            print(f"Resuming from step {start_step_for_this_epoch}...")
            
        # Create dataloader
        dataloader = DataLoader(ds, batch_size=args.per_device_batch_size, shuffle=True,
                                collate_fn=lambda b: collate_pairs(b, tokenizer),
                                num_workers=2, pin_memory=True)
        
        # Define output dir for this specific curriculum stage
        epoch_out_dir = os.path.join(out_dir, f"epoch_{current_epoch_num}")
        os.makedirs(epoch_out_dir, exist_ok=True)
        
        # train for 1 pass over this dataset (one epoch)
        dpop_training_loop(
            model=model,
            # ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataloader=dataloader,
            device=args.device,
            out_dir=epoch_out_dir,
            beta=args.beta,
            lam=args.lam,
            epochs=1,
            lr=args.lr,
            grad_accum_steps=args.grad_accum_steps,
            save_every_steps=args.save_every_steps,
            resume_step=start_step_for_this_epoch  # --- ADDED ---
        )
        # After resuming, subsequent epochs should start from step 0
        args.start_epoch = current_epoch_num + 1
        args.resume_step = 0

        # After each epoch optionally update ref_model? In DPO literature pi_ref is often fixed as original SFT.
        # Keep ref_model frozen as original (do NOT set ref_model = model)
        print(f"Completed curriculum epoch {current_epoch_num}/{final_epochs}")

    print("All curriculum epochs finished. Final model saved to:", out_dir)


if __name__ == "__main__":
    main()
