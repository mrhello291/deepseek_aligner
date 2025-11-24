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
    Updated for Instruct Models: 
    1. Applies Chat Template (System/User/Assistant tags).
    2. Appends EOS token to the end of responses so the model learns to stop.
    """
    MAX_TOTAL_LEN = 1024
    
    full_prompts = []
    chosen_responses = []
    rejected_responses = []

    # 1. Format the Prompts using the Tokenizer's Chat Template
    for b in batch:
        # Construct the message history
        # We inject a standard system prompt to maintain assistant behavior
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer the question."},
            {"role": "user", "content": f"Context: {b.get('context', '')}\n\nQuestion: {b['prompt']}"}
        ]
        
        # apply_chat_template adds the special tokens:
        # <|begin_of_text|><|start_header_id|>system...<|start_header_id|>user...<|start_header_id|>assistant
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        full_prompts.append(formatted_prompt)
        
        # 2. Ensure responses have an EOS token
        # Without this, the model might answer and then keep hallucinating new questions.
        c = b["chosen"]
        r = b["rejected"]
        if not c.endswith(tokenizer.eos_token): c += tokenizer.eos_token
        if not r.endswith(tokenizer.eos_token): r += tokenizer.eos_token
        
        chosen_responses.append(c)
        rejected_responses.append(r)

    # 3. Concatenate Template + Response
    seq_w = [p + c for p, c in zip(full_prompts, chosen_responses)]
    seq_l = [p + r for p, r in zip(full_prompts, rejected_responses)]

    # 4. Tokenize (Same logic as before, but now handles special tokens automatically)
    tok_w = tokenizer(
        seq_w,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOTAL_LEN
    )
    tok_l = tokenizer(
        seq_l,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOTAL_LEN
    )
    
    # 5. Calculate Prompt Lengths (Crucial for masking loss)
    # We tokenize the *formatted* prompt to find where the user instruction ends and the assistant response begins.
    prompt_toks = tokenizer(
        full_prompts,
        truncation=True,
        max_length=MAX_TOTAL_LEN,
        add_special_tokens=False # Template already added them
    )
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

    # 1. Build mask to pick RESPONSE TOKENS for each sample
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

    # 2. Gather log-probs of actual next tokens
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

    # 3. Sum log-probs over response tokens
    sum_logprobs = token_log_probs.sum(dim=1)  # (B,)

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
    optimizer,
    scheduler,
    step,
    beta,
    lam,
    grad_accum_steps,
    save_every_steps,
    max_grad_norm,
):
    model.train()
    # ref_model.eval()

    # Only train LoRA params
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # # Calculate total steps based on a single epoch pass, as epochs are handled by the curriculum
    # # total_steps = epochs * len(train_dataloader) // grad_accum_steps
    # total_steps_per_epoch = len(train_dataloader) // grad_accum_steps
    # scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=scheduler_warmup_steps,
    #     num_training_steps=total_steps_per_epoch * epochs, # total_steps,
    # )
    
    # If resuming, fast-forward the scheduler to the correct step
    # if resume_step > 0:
    #     grad_steps_to_advance = resume_step // grad_accum_steps
    #     print(f"Resuming from step {resume_step}. Advancing scheduler by {grad_steps_to_advance} gradient steps.")
    #     for _ in range(grad_steps_to_advance):
    #         scheduler.step()
    # # --- END ADDED ---

    step = resume_step
    for epoch in range(epochs):
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
            
        resume_step = 0
        
    # final save
    # print("Training finished, saved to", out_dir)
    return optimizer, scheduler, step
    # model.save_pretrained(out_dir)
    # tokenizer.save_pretrained(out_dir)


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
    tokenizer.padding_side = "right" 
    # ensure tokenizer has pad token
    # if tokenizer.pad_token_id is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token_id is None:
        print("No pad token found. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # NOTE: We are NOT adding a new token or resizing,
        # just aliasing the pad token to the existing eos token.
    
    # Usually loaded auto-magically from tokenizer_config.json, but good to verify.
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
    
    # Load base model (FP16)
    print("Loading base model (fp16) from", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

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

    
    if args.load_adapter_path:
        print(f"Resuming training by loading adapter from: {args.load_adapter_path}")
        model.load_adapter(args.load_adapter_path, "default")
        print("Adapter loaded successfully.")

    # Initialize optimizer ONCE
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # Compute total training steps ONCE (sum of all phases)
    total_steps = (
        len(ds_easy) +
        len(ds_medium) +
        len(ds_hard)
    ) // args.grad_accum_steps

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    step = args.resume_step
        
    # # Build dataloaders for each epoch and run DPOP training sequentially
    # final_epochs = len(epoch_datasets)
    print("\n--- Building Curriculum Dataset ---")
    
    # Load datasets
    ds_easy = PairDataset([data_dir / "dpo_pairs_easy.jsonl"], tokenizer)
    ds_medium = PairDataset([data_dir / "dpo_pairs_medium.jsonl"], tokenizer)
    ds_hard = PairDataset([data_dir / "dpo_pairs_hard.jsonl"], tokenizer)

    # Create specific slices
    # Easy: First 5,000
    n_easy = min(len(ds_easy), 5000)
    subset_easy = Subset(ds_easy, list(range(n_easy)))
    
    # Medium: First 10,000
    n_medium = min(len(ds_medium), 10000)
    subset_medium = Subset(ds_medium, list(range(n_medium)))
    
    # Hard: First 30,000 (Optimization for speed)
    n_hard = min(len(ds_hard), 30000)
    subset_hard = Subset(ds_hard, list(range(n_hard)))

    print(f"Curriculum Plan (Sorted Order):")
    print(f"  1. Warmup (Easy):   {n_easy} samples")
    print(f"  2. Bridge (Medium): {n_medium} samples")
    print(f"  3. Refine (Hard):   {n_hard} samples")
    total_samples = n_easy + n_medium + n_hard
    print(f"  Total Samples:      {total_samples}")

    # # Concatenate strictly in order
    # combined_ds = ConcatDataset([subset_easy, subset_medium, subset_hard])

    # # Create DataLoader with SHUFFLE=FALSE to preserve Easy->Med->Hard order
    # dataloader = DataLoader(
    #     combined_ds, 
    #     batch_size=args.per_device_batch_size, 
    #     shuffle=False,   # <--- CRITICAL: Keep the curriculum order
    #     collate_fn=lambda b: collate_pairs(b, tokenizer),
    #     num_workers=2, 
    #     pin_memory=True
    # )
    
    # # 5. Start Training (Single Epoch over combined data)
    # dpop_training_loop(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataloader=dataloader,
    #     device=args.device,
    #     out_dir=args.out_dir,
    #     beta=args.beta,
    #     lam=args.lam,
    #     epochs=1, # Only 1 pass needed for this combined dataset
    #     lr=args.lr,
    #     grad_accum_steps=args.grad_accum_steps,
    #     save_every_steps=args.save_every_steps,
    #     resume_step=args.resume_step
    # )
    
    # Phase order preserved (curriculum)
    phase_datasets = [
        ("easy", subset_easy),
        ("medium", subset_medium),
        ("hard", subset_hard),
    ]

    for phase_name, phase_ds in phase_datasets:
        print(f"\n=== Starting PHASE: {phase_name.upper()} ===")
        dataloader = DataLoader(
            phase_ds,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_pairs(b, tokenizer),
            num_workers=2,
            pin_memory=True
        )

        optimizer, scheduler, step = dpop_training_loop(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=dataloader,
            device=args.device,
            out_dir=out_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            beta=args.beta,
            lam=args.lam,
            grad_accum_steps=args.grad_accum_steps,
            save_every_steps=args.save_every_steps,
            max_grad_norm=1.0,
        )



if __name__ == "__main__":
    main()
