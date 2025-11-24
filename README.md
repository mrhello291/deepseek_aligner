# DeepSeek Aligner

## Project overview

This repository contains code and scripts to create and train ranking/alignment (DPO-style) adapters for LLaMA-based models, evaluate on hallucination and truthfulness benchmarks, and utilities for dataset preparation.

Root layout (important files / folders):

```
mrhello291-deepseek_aligner/
├── aligner_env.yml            # conda env for aligner / training / eval (no vLLM judge)
├── vllm_env.yml               # conda env for vLLM-based evaluation (LLM judge)
├── evaluation/                # evaluation scripts & results
├── first_improvement/         # data prep + training for first improvement
├── further_improvement/       # additional experiments
├── old_training/              # older finetune / RAG SFT code
├── scripts/                   # helpers for downloads / merging adapters
└── models/                    # (NOT in repo) place your model(s) here
```

---

## Prerequisites

1. Install [conda/miniconda].
2. Make sure you have Python 3.10+ (the YAMLs specify exact packages).
3. Sufficient disk (several GBs) and GPU if you plan to train / run vLLM.

---

## Environments

Create and activate the recommended conda environments:

```bash
# aligner environment (training + evaluation without vLLM judge)
conda env create -f aligner_env.yml -n aligner
conda activate aligner

# vLLM environment (for Halueval with LLM judge)
conda env create -f vllm_env.yml -n vllm
conda activate vllm
```

Switch back to `aligner` for regular training/evaluation when done.

---

## Models (where to place them)

Create a `models/` folder in the project root and place downloaded model files inside subfolders. Examples used in this repo:

* `models/llama3-2-3b-instruct` — LLaMA 3.2 3B instruct model (main model used for training/eval)
* `models/truthfulqa-truth-judge-llama2-7B` — TruthfulQA judge model (for TruthfulQA evaluation)

> The repository expects the `models/` directory at the root of the project.

### LoRA / adapter weights

LoRA adapters for experiments are available at the HuggingFace repo: `mrhello291/llama3-3.2-instruct-dpop`.

* The `dpop_old_strat` folder contains older/best weights.
* The `dpop_new_strat` folder contains latest LoRA weights.

---

## Datasets

The custom datasets used for training (preprocessed preference pairs) are on HuggingFace: `mrhello291/LevelledPreferencePairs`.

* `final_data/` contains the final `.jsonl` files used for training: `dpo_pairs_easy.jsonl`, `dpo_pairs_medium.jsonl`, `dpo_pairs_hard.jsonl`, and `dpo_pairs_hard_improved.jsonl`.

You can download the dataset with the `huggingface_hub` CLI or from the web UI.



---

## Evaluation

### TruthfulQA

1. Ensure `aligner` environment is active and `models/` contains the two models:

   * `models/llama3-2-3b-instruct`
   * `models/truthfulqa-truth-judge-llama2-7B`

2. Run the script:

```bash
python evaluation/eval_truthfulqa.py \
  --model_path models/llama3-2-3b-instruct \
  --judge_model_path models/truthfulqa-truth-judge-llama2-7B \
  --run_name your_run_name \
  --output_dir evaluation/results/TruthfulQA \
  --limit -1 \
  --mc_batch 64
```

Parameters explained:

* `--model_path` : path to the main model (or model adapter loader)
* `--lora_path`  : (optional) path to a LoRA adapter to apply
* `--judge_model_path` : path to the TruthfulQA judge model
* `--run_name`   : required name for the experiment/run
* `--output_dir` : directory to save metrics (default `evaluation/results/TruthfulQA`)
* `--limit`      : limit number of examples (default `-1` means all)
* `--mc_batch`   : MC scoring batch size per question

Result files will be written into `evaluation/results/TruthfulQA/` (metrics JSONs).

---

### Halueval (two options)

**A) Halueval with an LLM judge (vLLM required)**

1. Create & activate the `vllm` environment:

```bash
conda env create -f vllm_env.yml -n vllm
conda activate vllm
```

2. Run step 1 (generate model outputs / choices):

```bash
python evaluation/eval_haleval_llm_step1.py \
  --model_path models/llama3-2-3b-instruct \
  --lora_path path/to/adapter_or_none \
  --output_dir evaluation/results/HaluevalLLM \
  --run_name my_halueval_run
```

3. Run step 2 (judge outputs using LLM judge):

```bash
python evaluation/eval_haleval_llm_step2.py \
  --predictions_dir evaluation/results/HaluevalLLM \
  --judge_model_path models/<judge-model> \
  --output_dir evaluation/results/Halueval
```

**B) Halueval without LLM judge (baseline pipeline)**

1. Use `aligner` env.
2. Run:

```bash
python evaluation/eval_halueval.py \
  --model_path models/llama3-2-3b-instruct \
  --run_name simple_halueval \
  --output_dir evaluation/results/Halueval
```

Outputs will be saved into `evaluation/results/Halueval/` and `evaluation/results/HaluevalLLM/` depending on the flow.

---

## Training

There are two main training pipelines in this repo: `first_improvement` and `further_improvement`.

### First improvement (preferred to reproduce the results)

1. Activate the `aligner` environment.

2. Ensure `models/llama3-2-3b-instruct` is present.

3. Prepare training data — use files under `final_data/`:

   * `final_data/dpo_pairs_easy.jsonl`
   * `final_data/dpo_pairs_medium.jsonl`
   * `final_data/dpo_pairs_hard.jsonl`

4. Example training command (the script accepts several arguments; check `train_dpop.py --help`):

```bash
python first_improvement/training/train_dpop.py \
  --model_path models/llama3-2-3b-instruct \
  --data_files final_data/dpo_pairs_easy.jsonl,final_data/dpo_pairs_medium.jsonl,final_data/dpo_pairs_hard.jsonl \
  --output_dir outputs/first_improvement/best_run \
  --epochs 3
```

* To obtain the **best model weights** reported in the repo, train on `dpo_pairs_easy`, `dpo_pairs_medium`, and `dpo_pairs_hard` (under `final_data`).
* To obtain the **latest weights** / newest strategy, train on `dpo_pairs_medium`, `dpo_pairs_hard_improved` (under `final_data`).

**Notes:**

* `train_dpop_improved.py` contains an alternative improved training loop — inspect it for differences.
* Training hyperparameters, optimizer, and logging are configured inside the training scripts; use the `--help` flag to see full arg lists.

### Further improvement

Contains experiments using vLLM data prep (`hard_fake_gen_vllm.py`) and alternative training scripts. Log files and `bad_attempts/` folder contain earlier attempts and debugging scripts.

---

## Helpers / utility scripts

* `scripts/download_model.py` — helper to download a model into `models/` (edit to match HF repo and filenames).
* `scripts/download_hallucination_eval_models.py` — downloads models used in hallucination evaluation.
* `scripts/merge_adapter.py` — helper to merge LoRA adapters (read header in the script).

---

## Results

Pre-computed results and evaluation metrics are stored in `evaluation/results/` with the following structure:

```
evaluation/results/
├── TruthfulQA/
│   ├── metrics_baseline.json
│   ├── metrics_dpop_best.json
│   └── metrics_dpop_last.json
├── Halueval/
│   ├── comparison_report.txt
│   └── metrics_*.json
└── HaluevalLLM/
    └── metrics_eval_*.json
```

These JSON files give final metrics for baseline and adapter-weighted evaluations.

---
