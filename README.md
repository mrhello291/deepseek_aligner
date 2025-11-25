# Hallucination Aligner

## Important Research Papers and Links

* **Base model:** [LLaMA 3 (“3.2-3B Instruct”)](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
* **Dataset (Levelled Preference Pairs):** [Link](https://huggingface.co/datasets/mrhello291/LevelledPreferencePairs)
* **LoRA weights for models:** [Link](https://huggingface.co/mrhello291/llama3-3.2-instuct-dpop) (“old_strat” = best model; “new_strat” = latest model)
* **Teacher model:** [LLaMA 3 (“3.1-8B”)](https://huggingface.co/meta-llama/Llama-3.1-8B)
* **TruthfulQA judge model:** [Link](https://huggingface.co/allenai/truthfulqa-truth-judge-llama2-7B)
* **Hallucination evaluation model (Halueval judge):** [Link](https://huggingface.co/vectara/hallucination_evaluation_model)

### Research Papers

* DPO research: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://dl.acm.org/doi/10.5555/3666122.3668460)
* Curry DPO: [Curry DPO: Enhancing Alignment using Curriculum Learning & Ranked Preferences](https://arxiv.org/pdf/2403.07230)
* DPO-Postive (DPOP): [Fixing Failure Modes of Preference Optimisation with DPO-Positive](https://arxiv.org/pdf/2402.13228)
* “Teaching Lies with Curry DPO”: [Teaching with Lies: Curriculum DPO on Synthetic Negatives for Hallucination Detection](https://arxiv.org/pdf/2505.17558)
* HaluCheck: [HaluCheck: Halucheck: Integrating Hallucination Detection Techniques in Llm-Based Conversational Systems](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4944961)
* DA-DPO: [daDPO: Distribution-Aware DPO for Distilling Conversational Abilities](https://arxiv.org/pdf/2506.15717)

---

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

## Dataset Preparation Process

You can see the full process here [Dataset README](DatasetPrep.md)

---

## Helpers / utility scripts

* `scripts/download_model.py` — helper to download a model into `models/` (edit to match HF repo and filenames).
* `scripts/download_hallucination_eval_models.py` — downloads models used in hallucination evaluation.
* `scripts/merge_adapter.py` — helper to merge LoRA adapters (read header in the script).

---

## Results

Pre-computed results and evaluation metrics are stored in `evaluation/results/` with the following structure.
**For each metric file, the corresponding *detailed* per-example outputs can be found in the adjacent `detailed_*.jsonl` files.**

```
evaluation/results/
├── TruthfulQA/
│   ├── metrics_baseline.json
│   ├── metrics_dpop_best.json
│   ├── metrics_dpop_last.json
│   ├── detailed_gen_baseline.jsonl
│   ├── detailed_dpop_best.jsonl
│   ├── detailed_dpop_last.jsonl
│   └── failed_model/
│       ├── detailed_full_dpop.jsonl
│       └── metrics_full_dpop.json
├── Halueval/
│   ├── metrics_baseline.json
│   ├── metrics_dpop_best.json
│   ├── metrics_dpop_last.json
│   ├── detailed_baseline.jsonl
│   ├── detailed_dpop_best.jsonl
│   └── detailed_dpop_last.jsonl
└── HaluevalLLM/
    ├── metrics_baseline.json
    ├── metrics_dpop_best.json
    ├── metrics_dpop_last.json
    ├── detailed_baseline.jsonl
    ├── detailed_dpop_best.jsonl
    └── detailed_dpop_last.jsonl
```

**All detailed per-sample evaluation traces (model outputs, hallucination flags, similarity scores, etc.) are available in the corresponding `detailed_*.jsonl` files next to the aggregated metric reports.**

---

The JSON files give final metrics for baseline and adapter-weighted evaluations.

Here, are the results we obtained in detail
### 1. HaluEval (Similarity Score)

#### Summary Table

| Metric                    | Baseline | Best Model | Latest Model |
| ------------------------- | -------- | ---------- | ------------ |
| **Hallucination Rate**    | 0.0729   | 0.0684     | 0.0802       |
| **Avg Confidence Margin** | 0.5424   | 0.5509     | 0.5408       |
| **Avg BERT Score**        | 0.8063   | 0.8238     | 0.8162       |
| **Avg ROUGE-L**           | 0.7987   | 0.8086     | 0.7994       |
| **Total Samples**         | 10,000   | 10,000     | 10,000       |

#### Relative Improvements (vs. Baseline)

##### Best Model vs Baseline

| Metric                    | Absolute Change | Relative Change |
| ------------------------- | --------------- | --------------- |
| **Hallucination Rate**    | −0.0045         | **−6.17%**      |
| **Avg Confidence Margin** | +0.00854        | **+1.58%**      |
| **Avg BERT Score**        | +0.01744        | **+2.16%**      |
| **Avg ROUGE-L**           | +0.00991        | **+1.24%**      |

##### Latest Model vs Baseline

| Metric                    | Absolute Change | Relative Change |
| ------------------------- | --------------- | --------------- |
| **Hallucination Rate**    | +0.0073         | **+10.02%**     |
| **Avg Confidence Margin** | −0.00158        | **−0.29%**      |
| **Avg BERT Score**        | +0.00988        | **+1.23%**      |
| **Avg ROUGE-L**           | +0.00068        | **+0.09%**      |

#### Interpretation

* **Best Model** shows consistent improvements over Baseline across all metrics, with the largest gain in **BERT Score (+2.16%)** and a notable reduction in **Hallucination Rate (−6.17%)**.
* **Latest Model** performs slightly above Baseline in semantic similarity metrics (BERT, ROUGE-L) but regresses in **Hallucination Rate (+10%)** and **BLEU (−1.6%)**.
* Overall, **Best Model remains the strongest performer**.

### 2. HaluEval (With LLM Judge)

#### Summary Table

| Metric                 | Baseline | Best Model | Latest Model |
| ---------------------- | -------- | ---------- | ------------ |
| **Hallucination Rate** | 0.1654   | 0.1697     | 0.1929       |
| **Avg Faithfulness**   | 0.7475   | 0.6940     | 0.7214       |
| **Avg BERT F1**        | 0.4849   | 0.8072     | 0.5207       |
| **Total Samples**      | 10,000   | 10,000     | 10,000       |

#### Relative Improvements (vs. Baseline)

##### Best Model vs Baseline

| Metric                 | Absolute Change | Relative Change |
| ---------------------- | --------------- | --------------- |
| **Hallucination Rate** | +0.0043         | **+2.60%**      |
| **Avg Faithfulness**   | −0.05345        | **−7.15%**      |
| **Avg BERT F1**        | +0.32225        | **+66.49%**     |

##### Latest Model vs Baseline

| Metric                 | Absolute Change | Relative Change |
| ---------------------- | --------------- | --------------- |
| **Hallucination Rate** | +0.0275         | **+16.62%**     |
| **Avg Faithfulness**   | −0.02611        | **−3.49%**      |
| **Avg BERT F1**        | +0.03578        | **+7.38%**      |


#### Interpretation

* **Best Model** shows *substantial improvement* in **BERT F1 (+66%)**, indicating a major gain in semantic alignment.
* However, this comes with decreases in **faithfulness (−7.15%)** and a small increase in **hallucination rate (+2.6%)**.
* **Latest Model** improves BERT F1 modestly (+7.38%) but performs worse than Baseline in **hallucination rate** and **faithfulness**.
* Overall, the **Best Model** excels in semantic precision (BERT F1) but trades off factual consistency.

### 3. TruthfulQA

#### Summary Table

| Metric               | Baseline | Best Model | Latest Model |
| -------------------- | -------- | ---------- | ------------ |
| **MC1**              | 0.26683  | 0.24969    | 0.26561      |
| **MC2**              | 0.44323  | 0.42495    | 0.44347      |
| **Gen Truthfulness** | 0.71359  | 0.66463    | 0.73562      |
| **Gen Avg Prob**     | 0.71317  | 0.66194    | 0.72790      |

#### Relative Improvements (vs. Baseline)

##### Best Model vs Baseline

| Metric               | Absolute Change | Relative Change |
| -------------------- | --------------- | --------------- |
| **MC1**              | −0.01714        | **−6.42%**      |
| **MC2**              | −0.01828        | **−4.12%**      |
| **Gen Truthfulness** | −0.04896        | **−6.86%**      |
| **Gen Avg Prob**     | −0.05122        | **−7.18%**      |

##### Latest Model vs Baseline

| Metric               | Absolute Change | Relative Change |
| -------------------- | --------------- | --------------- |
| **MC1**              | −0.00122        | **−0.46%**      |
| **MC2**              | +0.00024        | **+0.05%**      |
| **Gen Truthfulness** | +0.02203        | **+3.09%**      |
| **Gen Avg Prob**     | +0.01474        | **+2.07%**      |

#### Interpretation

* **Best Model** shows decreases across all metrics relative to Baseline, indicating regression in multiple areas.
* **Latest Model** nearly matches Baseline on **MC1/MC2** and improves meaningfully in:

  * **Gen Truthfulness (+3.09%)**
  * **Gen Avg Prob (+2.07%)**
* This suggests the **Latest Model** offers better overall truthfulness and confidence quality while maintaining comparable multiple-choice performance.

Below is a cohesive **net conclusion** you can directly include in your results or discussion section:

---

## Net Conclusion

The **best model** demonstrated a meaningful improvement in semantic understanding, as reflected in its significantly higher BERT-based similarity and F1 scores. These gains indicate that the model became better at capturing nuance, aligning its outputs more closely with intended meanings, and producing responses that were semantically richer and more context-aware. At the same time, it also reduced hallucinations relative to the baseline, showing that early alignment efforts were effective in encouraging more grounded responses.

However, this improvement came with a noticeable behavioral shift:
the best model became slightly more cautious. In many cases, it avoided strong assertions, defaulted to safer or more qualified responses, and occasionally under-answered queries where the baseline would have responded more directly. This suggests that the model adapted by prioritizing safety and correctness, but at the cost of reduced assertiveness or decisiveness—an early indication of the alignment–capability trade-off that becomes more apparent in the latest model.

The evaluation results across **similarity metrics**, **faithfulness metrics**, and **TruthfulQA** reveal a clear trajectory in how the model family evolved. The **low TruthfulQA scores in earlier iterations** (particularly the baseline and best model) highlighted a significant gap in factual consistency, which motivated the development of the **latest model**. This newest version successfully improves **TruthfulQA truthfulness**, **average probability**, and maintains comparable multiple-choice performance, demonstrating that targeted alignment toward truthfulness was effective.

However, these gains come with a new big trade-off:
while the latest model is *more truthful*, it has begun to show signs of **instruction degradation**. In several examples, the model displays weaker adherence to instructions, reduced structured-response capability, and occasional misinterpretation of task format. In other words, the model is becoming more truthful but less reliably *instructable*. This suggests a form of **catastrophic forgetting**, where truthfulness-oriented fine-tuning partially erodes the base model’s instruction-following priors. This is the same problem that we wished to solve using curriculum DPO on the best model, which means that we must train the latest model also in a curriculum DPO fashion on the improved dataset and we should probably see gains across all the metrics.

This creates the next key challenge:
**How do we jointly optimize for low hallucination, high truthfulness, and strong instruction-following ability?**

Future work must focus on **balanced alignment**, where:

* TruthfulQA improvements are preserved,
* Hallucinations continue to decrease,
* Instruction capabilities are actively reinforced rather than overwritten.

Achieving this balance—truthfulness without sacrificing instruction fidelity—is now the central goal for the next iteration of the model.
