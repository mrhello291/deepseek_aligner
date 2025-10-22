# training/generate_sft.py
import json
import os
from tqdm import tqdm
from cove.cove_engine import generate_draft
from eval.hallucination_evaluator import HallucinationEvaluator

# --- Configuration for Data Curation ---
# Define strict quality thresholds for accepting a generated sample.
# A sample will only be included in the SFT dataset if it meets ALL these criteria.
QUALITY_THRESHOLDS = {
    "max_ne_error_rate": 0.0,          # Must not introduce any fabricated named entities.
    "allow_intrinsic_hallucination": False # Must not contradict the ground truth answer.
    # FActScore can also be added here once a robust knowledge base is integrated.
}

def load_source_data(path: str):
    """
    Loads a source dataset in JSONL format.
    Expects each line to be a JSON object with 'prompt' and 'answer' keys.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in {path}")
                continue
    return samples

def generate_sft_dataset(source_samples, out_file, max_samples=10000):
    """
    Generates a high-quality Supervised Fine-Tuning (SFT) dataset.

    It generates a response for each prompt, evaluates it against the ground truth
    using the HallucinationEvaluator, and filters out any samples that do not
    meet the strict quality thresholds.
    """
    # Initialize the comprehensive evaluator
    evaluator = HallucinationEvaluator()
    
    accepted_samples = []
    rejected_samples_log = []
    
    # Open file for rejected samples to write in real-time
    rejected_file_path = out_file + ".rejected.jsonl"
    with open(rejected_file_path, "w", encoding="utf-8") as rejected_fh:
        for sample in tqdm(source_samples[:max_samples], desc="Generating SFT Data"):
            prompt = sample.get("prompt")
            ground_truth = sample.get("answer")

            if not prompt or not ground_truth:
                continue

            # 1. Generate a draft response from the model
            draft_response = generate_draft(prompt)

            # 2. Evaluate the draft against the ground truth
            # The ground truth answer serves as the 'retrieved_docs' context for this check.
            eval_metrics = evaluator.evaluate_response(
                response=draft_response,
                retrieved_docs=ground_truth,
                ground_truth=ground_truth,
                knowledge_base=None # No external KB needed when checking against ground truth
            )

            # 3. Apply strict filtering based on quality thresholds
            is_accepted = True
            rejection_reasons = []

            if eval_metrics.get('intrinsic_hallucination') and not QUALITY_THRESHOLDS['allow_intrinsic_hallucination']:
                is_accepted = False
                rejection_reasons.append("failed: intrinsic_hallucination")

            if eval_metrics.get('ne_error_rate', 1.0) > QUALITY_THRESHOLDS['max_ne_error_rate']:
                is_accepted = False
                rejection_reasons.append(f"failed: ne_error_rate ({eval_metrics['ne_error_rate']:.2f})")

            # 4. Log and save the results
            if is_accepted:
                accepted_samples.append({
                    "prompt": prompt,
                    "response": draft_response,
                    "evaluation": eval_metrics
                })
            else:
                rejected_info = {
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "rejected_response": draft_response,
                    "evaluation": eval_metrics,
                    "rejection_reason": ", ".join(rejection_reasons)
                }
                rejected_samples_log.append(rejected_info)
                rejected_fh.write(json.dumps(rejected_info, ensure_ascii=False) + "\n")

    # Write the final high-quality SFT dataset
    with open(out_file, "w", encoding="utf-8") as fh:
        for r in accepted_samples:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print("-" * 50)
    print(f"✅ SFT data generation complete.")
    print(f"Total samples processed: {len(source_samples[:max_samples])}")
    print(f"Accepted samples: {len(accepted_samples)}")
    print(f"Rejected samples: {len(rejected_samples_log)}")
    print(f"High-quality SFT dataset saved to: {out_file}")
    print(f"Detailed rejection log saved to: {rejected_file_path}")
    print("-" * 50)

# Example usage:
# if __name__ == "__main__":
#     # Assume you have a file 'data/trivia_qa_subset.jsonl'
#     source_data = load_source_data("data/trivia_qa_subset.jsonl")
#     generate_sft_dataset(source_data, "data/sft_dataset.jsonl", max_samples=100)
