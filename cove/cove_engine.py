# cove/cove_engine.py
import os
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from retrieval.faiss_rerank import FaissReranker
from retrieval.chunking import chunk_text
from wikidata_graph.subgraph_extractor import build_filtered_subgraph
from search_web.serper_client import serper_search
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Replace with your model paths
MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/data/models/DeepSeek-R1-Distill-Llama-8B")
LYNX_MODEL = os.environ.get("LYNX_MODEL", "PatronusAI/lynx-70b-eval")  # example

# Load main generator model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.float16)
model.eval()

# Load Lynx detector (as a judge). Use a smaller judge model if available to save latency.
lynx_tokenizer = AutoTokenizer.from_pretrained(LYNX_MODEL, use_fast=True)
lynx_model = AutoModelForCausalLM.from_pretrained(LYNX_MODEL, device_map="auto", torch_dtype=torch.float16)
lynx_pipe = pipeline("text-generation", model=lynx_model, tokenizer=lynx_tokenizer, device=0)

# Simple cache to avoid repeated verification calls
VERIFICATION_CACHE = {}

def _hash_claim_source(claim: str, source: str):
    key = hashlib.sha256((claim + "|" + source).encode()).hexdigest()
    return key

# 1) Draft Generation
def generate_draft(prompt: str, max_new_tokens=256):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# 2) Verification Question Planning
def extract_atomic_claims(draft: str) -> List[str]:
    """
    Use simple heuristic: split sentences and use dependency parsing or heuristics to detect factual claims.
    For production: use a small model to identify atomic factual propositions.
    """
    import nltk
    nltk.download('punkt')
    sents = nltk.sent_tokenize(draft)
    # naive: return sentences that contain named entities or numbers
    return sents

def generate_verification_questions(claim: str) -> List[str]:
    """
    Factor claim into subquestions. This uses a prompting approach to the generator model to rewrite
    claim into verification questions.
    """
    prompt = f"Rewrite the following factual claim into short yes/no and short-answer verification questions, factoring complex claims:\n\nClaim: {claim}\n\nQuestions:"
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**input_ids, max_new_tokens=128, do_sample=False)
    q_text = tokenizer.decode(out[0], skip_special_tokens=True)
    # split heuristically by line
    qs = [q.strip("-* \n\t") for q in q_text.split("\n") if q.strip()]
    return qs

# 3) Independent Answering (must be unaware of original draft)
def answer_verification_question(question: str, retrieval_stack: Dict[str, Any], top_k=5):
    """
    retrieval_stack contains booleans or objects to perform Tier 2/1/3 retrieval as required.
    This function MUST NOT include the original draft in the model context.
    """
    # Check cache first
    cache_key = hashlib.sha256(question.encode()).hexdigest()
    if cache_key in VERIFICATION_CACHE:
        return VERIFICATION_CACHE[cache_key]

    # Smart router rules: naive rule-based classifier (you should replace with learned classifier later)
    def choose_tier(q):
        q_low = q.lower()
        if "latest" in q_low or "recent" in q_low or "202" in q_low:
            return "web"
        elif any(ent in q for ent in ["who", "when", "where", "which", "was", "did", "are"]):
            return "wikidata"
        else:
            return "wikipedia"

    tier = choose_tier(question)
    evidence = []
    if tier == "wikipedia":
        # coarse BM25 then FAISS rerank (assume retrieval_stack['faiss'] exists)
        faiss_reranker: FaissReranker = retrieval_stack["faiss"]
        res = faiss_reranker.search(question, topk=top_k)
        for r in res:
            evidence.append({"text": r["text"], "score": r["score"], "source": "wikipedia"})
    elif tier == "wikidata":
        # extract small subgraph for entities in question (entity extraction not implemented here)
        # naive: attempt to parse an entity: first token sequence capitalized
        entity = question.split("?")[0].split(" of ")[-1].strip()
        try:
            qid, subG = build_filtered_subgraph(entity, entity_type="person")
            # convert top central nodes to evidence statements (placeholder)
            nodes = list(subG.nodes())[:5]
            for n in nodes:
                evidence.append({"text": f"Wikidata node {n}", "score": 1.0, "source": "wikidata"})
        except Exception:
            # fallback to web
            res = serper_search(question, top_k=top_k)
            for r in res:
                evidence.append({"text": r["snippet"], "source": r["link"], "score": 0.5})
    else:
        # web search
        res = serper_search(question, top_k=top_k)
        for r in res:
            evidence.append({"text": r["snippet"], "source": r["link"], "score": 0.6})

    # Now generate an answer using retrieved evidence as context (crucially DO NOT include original draft)
    evidence_text = "\n".join([f"[{e['source']}] {e['text']}" for e in evidence])
    prompt = f"Question: {question}\n\nContext:\n{evidence_text}\n\nAnswer concisely and cite the source for each fact you use."
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**input_ids, max_new_tokens=128, do_sample=False)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    # cache
    VERIFICATION_CACHE[cache_key] = {"question": question, "answer": answer, "evidence": evidence}
    return VERIFICATION_CACHE[cache_key]

# 4) Final Synthesis
def synthesize_final(draft: str, verified_claims: List[Dict], lynx_threshold=0.5):
    """
    Combine draft + verified claim answers into final answer.
    For each verified claim, include a citation pointer to evidence sentences.
    Use Lynx to score each final sentence.
    """
    # Build synthesis prompt with claims and supporting evidence
    claims_block = "\n".join([f"Claim: {c['claim']}\nAnswer: {c['verdict']['answer']}\nEvidence: {c['verdict']['evidence']}" for c in verified_claims])
    prompt = f"Original draft:\n{draft}\n\nVerified claims and answers:\n{claims_block}\n\nProduce a final, concise answer with inline citations mapped to evidence entries above. For each sentence, include a confidence score."
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**input_ids, max_new_tokens=256, do_sample=False)
    final_text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Run Lynx as final judge on each sentence (split)
    sentences = final_text.split("\n")
    scored = []
    for s in sentences:
        # run Lynx scoring prompt (simplified)
        judge_prompt = f"Given the sentence:\n{s}\nAnd the supporting evidence:\n{claims_block}\nScore how faithful this sentence is (0 to 1). Answer with a float."
        j_in = lynx_tokenizer(judge_prompt, return_tensors="pt").to(lynx_model.device)
        with torch.no_grad():
            j_out = lynx_model.generate(**j_in, max_new_tokens=10)
        j_text = lynx_tokenizer.decode(j_out[0], skip_special_tokens=True)
        try:
            score = float(j_text.strip().split()[0])
        except Exception:
            score = 0.5
        scored.append({"sentence": s, "lynx_score": score})
    return {"final_text": final_text, "scored_sentences": scored}
