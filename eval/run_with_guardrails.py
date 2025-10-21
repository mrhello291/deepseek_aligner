# eval/run_with_guardrails.py
from cove.cove_engine import generate_draft, extract_atomic_claims, generate_verification_questions, answer_verification_question, synthesize_final
from retrieval.faiss_rerank import FaissReranker
import json, argparse, os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_sample(sample, faiss_reranker):
    prompt = sample['prompt']
    draft = generate_draft(prompt)
    # stage 1 lynx screening (pseudo)
    # if draft is high-confidence, optionally skip
    claims = extract_atomic_claims(draft)
    verified_claims = []
    # prepare retrieval stack
    retrieval_stack = {"faiss": faiss_reranker}
    # For each claim, generate questions, then answer them in parallel
    for c in claims:
        qs = generate_verification_questions(c)
        # parallel answer
        results = []
        for q in qs:
            res = answer_verification_question(q, retrieval_stack)
            results.append(res)
        verified_claims.append({"claim": c, "verdicts": results})
    final = synthesize_final(draft, [{"claim": vc["claim"], "verdict": {"answer": v['answer'], "evidence": v['evidence']}} for vc in verified_claims for v in vc['verdicts']])
    return {"id": sample.get("id"), "draft": draft, "final": final}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--faiss_corpus", required=True)
    parser.add_argument("--out", default="guardrailed_results.jsonl")
    args = parser.parse_args()

    # load faiss reranker from corpus
    import json
    docs = [json.loads(line)['text'] for line in open(args.faiss_corpus)]
    fr = FaissReranker()
    fr.index_corpus(docs)

    samples = [json.loads(l) for l in open(args.bench)]
    with open(args.out, "w") as fh:
        for s in samples:
            res = process_sample(s, fr)
            fh.write(json.dumps(res) + "\n")
    print("Wrote guardrailed results to", args.out)

if __name__ == "__main__":
    main()
