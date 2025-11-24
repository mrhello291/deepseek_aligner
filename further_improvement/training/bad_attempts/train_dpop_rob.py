#!/usr/bin/env python3

"""
train_dpop_robust_clean.py


A fully-correct DPOP/DPO training script with:
 - Proper prompt-lens computation
 - Vectorized log-prob scoring
 - Proper time-alignment of logits/tokens
 - Reference model scoring via disable_adapter()
 - Gradient clipping
 - Checkpointing w/ optimizer + scheduler state
 - Resume support
 - Safe LoRA defaults
 - Simplified data-loading: only Medium + Hard mixed together
"""


import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel


LOG_PATH = "loss_log.txt"
log_f = open(LOG_PATH, "a", buffering=1)  # line-buffered, fast

# ============================================================
#  Dataset + Utilities
# ============================================================

def load_jsonl(path: Path):
    arr = []
    if not path.exists():
        return arr

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    arr.append(json.loads(line))
                except:
                    continue
    return arr

def normalize_entry(e: dict):
    """Normalize different schemas to common structure."""
    return {
        "context": e.get("context", e.get("knowledge", "")),
        "prompt": e.get("prompt", e.get("question", "")),
        "chosen": e.get("chosen", e.get("ideal_answer", "")),
        "rejected": e.get("rejected", e.get("fake_answer", "")),
    }


class PairDataset(Dataset):
    def __init__(self, entries: List[Dict]):
        self.examples = entries

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================
#  Collate (Correct Prompt-Lens)
# ============================================================

def collate_pairs(batch, tokenizer):
    MAX_TOTAL_LEN = 1024
    prompts = []
    chosen = []
    rejected = []


    for b in batch:

        messages = [

            {

                "role": "system",

                "content": "You are a precise assistant. Answer the question based on the context. Be concise."

            },

            {

                "role": "user",

                "content": f"Context: {b['context']}\n\nQuestion: {b['prompt']}"

            }

        ]


        prompt_text = tokenizer.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True

        )

        prompts.append(prompt_text)


        c = b["chosen"] or ""

        r = b["rejected"] or ""


        if tokenizer.eos_token and not c.endswith(tokenizer.eos_token):

            c += tokenizer.eos_token

        if tokenizer.eos_token and not r.endswith(tokenizer.eos_token):

            r += tokenizer.eos_token


        chosen.append(c)

        rejected.append(r)


    seq_w = [p + c for p, c in zip(prompts, chosen)]

    seq_l = [p + r for p, r in zip(prompts, rejected)]


    tok_w = tokenizer(seq_w, padding=True, truncation=True,

                      max_length=MAX_TOTAL_LEN, return_tensors="pt")

    tok_l = tokenizer(seq_l, padding=True, truncation=True,

                      max_length=MAX_TOTAL_LEN, return_tensors="pt")


    # Tokenize prompts with identical flags EXCEPT padding (must stay unpadded)

    prompt_tok = tokenizer(

        prompts,

        padding=False,

        truncation=True,

        max_length=MAX_TOTAL_LEN,

        add_special_tokens=True,

        return_tensors=None

    )

    prompt_lens = [len(x) for x in prompt_tok["input_ids"]]


    return {

        "input_ids_w": tok_w["input_ids"],

        "attention_w": tok_w["attention_mask"],

        "input_ids_l": tok_l["input_ids"],

        "attention_l": tok_l["attention_mask"],

        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),

    }



# ============================================================

#  Vectorized log-prob scoring (Correct)

# ============================================================


def compute_logprob(

    model,

    input_ids: torch.Tensor,

    attention: torch.Tensor,

    start_pos: torch.Tensor,

    device: str

):

    """

    Proper vectorized logprob computation where:

      - logits at t predict token at t+1

      - time index alignment corrected

    """


    input_ids = input_ids.to(device)

    attention = attention.to(device)

    start_pos = start_pos.to(device)


    B, L = input_ids.shape


    outputs = model(input_ids=input_ids, attention_mask=attention, use_cache=False)

    logits = outputs.logits                     # (B, L, V)

    # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)



    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    response_mask = (positions >= start_pos.unsqueeze(1)) & attention.bool()


    # Cannot predict the first token

    response_mask[:, 0] = False


    # Time index for each target token

    time_idx = (positions - 1).clamp(min=0)

    pred_log_probs = log_probs.gather(

        1, time_idx.unsqueeze(-1).expand(B, L, log_probs.size(-1))

    )


    # Pick probability of actual next-token

    tok_log_probs = pred_log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)


    tok_log_probs = tok_log_probs * response_mask.float()


    return tok_log_probs.sum(dim=1)



# ============================================================

#   Training Loop (Correct DPOP)

# ============================================================


def dpop_train(

    model,

    tokenizer,

    dataloader,

    device,

    out_dir,

    beta=0.3,

    lam=10.0,

    lr=2e-4,

    epochs=1,

    grad_accum=8,

    save_every=2000,

    resume=False,

    opt_state_path=None

):

    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


    steps_per_epoch = len(dataloader)

    total_steps = max(1, (steps_per_epoch * epochs) // grad_accum)


    scheduler = get_scheduler(

        "linear",

        optimizer=optimizer,

        num_warmup_steps=100,

        num_training_steps=total_steps,

    )


    resume_step = 0

    # opt_state_path = out_dir / "optim_sched.pt"


    if resume and opt_state_path and opt_state_path.exists():

        print("Resuming optimizer/scheduler...")

        try:

            resume_step = int(opt_state_path.parent.name.split("-")[-1])

            state = torch.load(opt_state_path, map_location="cpu")

            optimizer.load_state_dict(state["optimizer"])

            scheduler.load_state_dict(state["scheduler"])

        except:

            print("Resume failed, continuing fresh.")


    model.train()

    step = resume_step


    for epoch in range(epochs):

        print(f"\n=== Epoch {epoch+1}/{epochs} ===")


        pbar = tqdm(dataloader, total=len(dataloader))

        optimizer.zero_grad()


        for batch in pbar:

            ids_w = batch["input_ids_w"]

            att_w = batch["attention_w"]

            ids_l = batch["input_ids_l"]

            att_l = batch["attention_l"]

            pl = batch["prompt_lens"]


            start = pl


            # Trainable model

            log_w_theta = compute_logprob(model, ids_w, att_w, start, device)

            log_l_theta = compute_logprob(model, ids_l, att_l, start, device)


            # Reference model (disable adapter)

            with model.disable_adapter(), torch.no_grad():

                log_w_ref = compute_logprob(model, ids_w, att_w, start, device)

                log_l_ref = compute_logprob(model, ids_l, att_l, start, device)


            # DPO loss

            diff = beta * ((log_w_theta - log_w_ref) - (log_l_theta - log_l_ref))

            loss_dpo = -torch.nn.functional.logsigmoid(diff).mean()


            # Positive anchor (DPOP)

            loss_pos = lam * torch.clamp(log_w_ref - log_w_theta, min=0).mean()


            loss = (loss_dpo + loss_pos) / grad_accum

            loss.backward()

            

            log_f.write(f"{step}\t{float(loss.item() * grad_accum)}\t{float(loss_dpo.item())}\t{float(loss_pos.item())}\n")



            if (step + 1) % grad_accum == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad()


            if step % 50 == 0:

                pbar.set_postfix({

                    "loss": float(loss.item() * grad_accum),

                    "dpo": float(loss_dpo.item()),

                    "pos": float(loss_pos.item())

                })


            if (step + 1) % save_every == 0:

                ckpt = out_dir / f"ckpt-{step+1}"

                ckpt.mkdir(exist_ok=True)

                model.save_pretrained(ckpt)

                tokenizer.save_pretrained(ckpt)

                torch.save({

                    "optimizer": optimizer.state_dict(),

                    "scheduler": scheduler.state_dict()

                }, ckpt / "optim_sched.pt")


            step += 1


    # Final save

    model.save_pretrained(out_dir)

    tokenizer.save_pretrained(out_dir)

    torch.save({

        "optimizer": optimizer.state_dict(),

        "scheduler": scheduler.state_dict()

    }, out_dir / "optim_sched.pt")


    print("Training complete.")



# ============================================================

#  Main

# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="./models/llama3-2-3b-instruct", help="Local path to base Llama 3.2 3B")

    parser.add_argument("--data-dir", type=str, default="./datasets/main", help="Directory containing dpo_pairs_easy.jsonl etc")

    parser.add_argument("--out-dir", type=str, default="./dpop_out_robust")

    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--grad-accum", type=int, default=8)

    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--beta", type=float, default=0.3)

    parser.add_argument("--lam", type=float, default=10.0)

    parser.add_argument("--save-every", type=int, default=2000)

    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()


    device = args.device

    data_dir = Path(args.data_dir)


    # Load datasets (SKIP EASY)

    med = load_jsonl(data_dir / "dpo_pairs_medium.jsonl")

    hard = load_jsonl(data_dir / "hard_fakes_vllm.jsonl")


    # Normalize & mix

    entries = [normalize_entry(e) for e in med] + \
                [normalize_entry(e) for e in hard]


    random.seed(42)

    random.shuffle(entries)


    print(f"Loaded Medium={len(med)} Hard={len(hard)} → Total={len(entries)}")

    

    resume_ckpt = None

    if args.resume:

        checkpoints = sorted(Path(args.out_dir).glob("ckpt-*"), key=lambda p: int(p.name.split("-")[-1]))

        if len(checkpoints) > 0:

            resume_ckpt = checkpoints[-1]

            print(f"Resuming from checkpoint: {resume_ckpt}")

        else:

            print("No checkpoints found, starting fresh.")



    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token


    print("Loading model...")

    # Safe LoRA configuration

    base_model = AutoModelForCausalLM.from_pretrained(

        args.model_path,

        torch_dtype=torch.float16,

        device_map="auto",

        trust_remote_code=True

    )


    # LoRA config

    lora_cfg = LoraConfig(

        r=16,

        lora_alpha=32,

        target_modules=["q_proj", "v_proj"],

        lora_dropout=0.05,

        bias="none",

        task_type="CAUSAL_LM"

    )


    if resume_ckpt:

        print(f"Loading adapter from checkpoint: {resume_ckpt}")

        from peft import PeftModel

        model = PeftModel.from_pretrained(

            base_model,

            resume_ckpt

        )

        model.base_model.set_adapter("default")

    else:

        print("Loading fresh LoRA adapter...")

        model = get_peft_model(base_model, lora_cfg)


    model.print_trainable_parameters()


    # Dataloader etc.

    ds = PairDataset(entries)

    loader = DataLoader(

        ds,

        batch_size=args.batch_size,

        shuffle=True,

        collate_fn=lambda b: collate_pairs(b, tokenizer),

        num_workers=2,

        pin_memory=True

    )


    opt_state_path = resume_ckpt / "optim_sched.pt" if (args.resume and resume_ckpt) else None


    dpop_train(

        model=model,

        tokenizer=tokenizer,

        dataloader=loader,

        device=device,

        out_dir=args.out_dir,

        beta=args.beta,

        lam=args.lam,

        lr=args.lr,

        epochs=args.epochs,

        grad_accum=args.grad_accum,

        save_every=args.save_every,

        resume=args.resume,

        opt_state_path=opt_state_path,

    )



if __name__ == "__main__":

    main()


