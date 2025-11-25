# **Dataset Preparation Pipeline Overview**

This section outlines the complete dataset creation workflow used to train both the **best model** and the **latest model**, including retrieval-augmented SFT data generation, negative sampling, hallucination synthesis, context restoration, and final DPO pair construction.

---

# **1. Building the Retrieval Corpus (Wikipedia Chunking & Indexing)**

### **1.1 Extract raw Wikipedia documents from Natural Questions**

Scripts read the NQ train/dev shards (`nq-train-00…04.jsonl.gz`, `nq-dev-00.jsonl.gz`) and:

1. Deduplicate articles by title
2. Strip HTML tokens
3. Reconstruct complete article text
4. Discard extremely short articles
5. Store:

   * title
   * raw plaintext
   * URL

Output: `articles` dictionary with all unique processed Wikipedia articles.

---

### **1.2 Sentence-aware chunking**

Using NLTK:

* Split each article into **sentence-aligned chunks** of ~512 words
* Add **50-word overlap** between consecutive chunks
* Produce highly coherent retrieval units

Output:
**`nq_wiki_chunks.jsonl`** — list of `{title, text, url}` chunks.

---

### **1.3 Embedding & indexing with FAISS**

Using SentenceTransformers (`multi-qa-MiniLM-L6-cos-v1`):

1. Encode all chunks into normalized embeddings
2. Build:

   * **CPU IndexFlatIP**
   * **GPU IVF Index** (if GPU available)
3. Save metadata describing:

   * embedding model
   * chunking strategy
   * index type
   * total chunks / articles

Outputs:

* `nq_wiki_index.faiss`
* `nq_wiki_index_gpu.faiss`
* `metadata.json`

---

# **2. Creating the Main Supervised Dataset**

This consists of **Natural Questions RAG training data** + **SQuAD-v2 cleaned subset**.

---

## **2.1 NQ RAG Construction (`create_nq_training.py`)**

For each question in NQ:

1. Extract gold short answer; fallback to long answer
2. Retrieve top-100 Wikipedia chunks using FAISS
3. Rerank using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
4. Select top-5 contexts
5. Create a training record:

   ```
   {
     example_id,
     question,
     contexts (list of top-5),
     context (concatenated),
     answers (gold answers)
   }
   ```

Output:
**`train_nq_rag.jsonl`**

---

## **2.2 SQuAD2 Data Cleaning (`train.csv` → `squad_15k.jsonl`)**

The script:

1. Loads SQuAD CSV
2. Fixes broken NumPy-style `array(...)` strings in the `answers` column
3. Identifies answerable/unanswerable rows
4. Samples:

   * **10k answerable**
   * **5k unanswerable**
5. Normalizes answers
6. Outputs consistent SFT-style JSON lines.

Output:
**`squad_15k.jsonl`**

---

# **3. Negative Sampling Framework**

We generate **three tiers of synthetic hallucinations**: Easy → Medium → Hard.

These augmentations mimic realistic mistakes of varying difficulty.

---

## **3.1 Load & unify SFT dataset for negative sampling**

Each negative sampling script begins by merging:

* `train_nq_rag.jsonl`
* `squad_15k.jsonl`

into a common format:

```
{
  question,
  context(s),
  ideal_answer,
  data_index
}
```

---

## **3.2. Easy Fake Generation (Topic Mismatch)**

**Script: `generate_easy_fakes.py`**

Mechanism:

* For each item, select a random answer from another unrelated item
* Create an **incorrect answer with completely mismatched topic**
* Produce easy-to-spot hallucinations

Output:
**`easy_fakes.jsonl`**

---

## **3.3. Medium Fake Generation (NER Entity Swaps)**

**Script: `generate_medium_fakes.py`**

Steps:

1. Build NER knowledge base from all ideal answers (spaCy transformer model)
2. Identify named entities in each ideal answer
3. Replace entities with *plausible but incorrect* ones of the same type
4. Output medium-difficulty hallucinations:

   * Right structure
   * Wrong named entities

Output:
**`medium_fakes.jsonl`**

---

## **3.4. Hard Fake Generation (LLM-Synthesized Hallucinations)**

**Script: `generate_hard_fakes.py`**

Hard negatives are generated using the hallucination-prone **base 3B Instruct model** via “poison prompts.”

Types:

* Logical flaw
* Causal error
* Unverifiable claim

Pipeline:

1. Prepare poison prompts using `PoisonPrompter`
2. Force model to generate *realistic, fluent but incorrect* answers
3. Clean hallucinations:

   * Remove explanations
   * Remove “Note:” or analysis text

Outputs:

* `hard_fakes.jsonl`
* cleaned version: `cleaned_hard_fakes.jsonl`

---

## **3.5. Restoring SFT Context for Each Fake Sample**

**Script: context-injection (fix fakes)**

Fakes (easy/medium/hard) contain only:

* question
* fake_answer
* ideal_answer
* data_index

This script:

* Reattaches the **original SFT context** using `data_index`
* Normalizes context string
* Produces enriched fake datasets:

Outputs:

* `easy_fakes_with_context.jsonl`
* `medium_fakes_with_context.jsonl`
* `hard_fakes_with_context.jsonl`

These are now valid preference-learning samples.

---

# **4. Curriculum-Aligned DPO Pair Construction**

**Script: DPO dataset creator**

For each question:

### Phase 1 (Easy):

* Chosen = ideal answer
* Rejected = easy_fake

### Phase 2 (Medium):

* Ideal vs medium
* Hard vs easy (to establish hierarchy)

### Phase 3 (Hard):

* Ideal vs hard
* Hard vs medium
* Medium vs easy

Outputs:

* `dpo_pairs_easy.jsonl`
* `dpo_pairs_medium.jsonl`
* `dpo_pairs_hard.jsonl`

This set is what trained the **best model**.

---

# **5. Teacher-Guided Hard Negative Generation (vLLM-Based Synthetic Pairs)**

*(Used only for the Latest Model)*

In addition to the original hard-negative generation pipeline, we introduced an improved process using a **teacher LLM (Llama-3.1-8B-Instruct)** to create *polished truth vs. hard negative* pairs with strict structural constraints.

### **Purpose of This Step**

This script:

1. Loads the cleaned NQ + SQuAD SFT dataset
2. Constructs Llama-3–style chat prompts using a strict system prompt
3. Ensures the model produces structured JSON with:

   * `"chosen_sentence"` — polished correct answer
   * `"rejected_sentence"` — same sentence structure but with a *single* fabricated detail
4. Enforces length budgets and token limits
5. Filters out malformed or oversized samples
6. Saves the resulting teacher-verified hard pairs into:
   **`hard_fakes_vllm.jsonl`**

### **Why We Added This Step**

* The original hard negatives from the hallucinating 3B model were too noisy.
* Many were malformed, unaligned, or not sufficiently *hard* to challenge the preference model.
* The teacher Llama-3.1-8B produces **much higher-quality pairs**, improving:

  * truthfulness discrimination
  * semantic alignment
  * robustness under preference optimization

These improved pairs formed the basis of the **latest model’s DPO hard phase**, replacing the earlier hard samples.

---

# **Final Summary Diagram (Conceptual)**

**Raw NQ + SQuAD → Clean SFT Data → Negative Sampling (Easy/Medium/Hard) → Add Context → DPO Pairs → Phase Datasets → Train Best Model**

**Then:**
**Teacher-guided re-generation of Hard negatives → new DPO hard dataset → Train Latest Model**

