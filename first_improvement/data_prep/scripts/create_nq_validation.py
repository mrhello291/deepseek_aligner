# data_prep/create_nq_validation.py
import gzip, json, os, sys
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import faiss

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from retrieval.faiss_rerank import FaissReranker

# config
NQ_DEV_PATH = "datasets/nq/dev/nq-dev-00.jsonl.gz"
WIKI_INDEX_PATH = "retrieval/nq_wiki_index.faiss"
OUTPUT_VAL_JSONL = "val_nq_rag.jsonl"
TOP_K_RETRIEVE = 100
TOP_K_RERANK = 5
MAX_VAL_SAMPLES = 1000  # Limit validation set size for faster evaluation

# init retriever and reranker
faiss_reranker = FaissReranker(model_name="multi-qa-MiniLM-L6-cos-v1", use_gpu=True)
faiss_reranker.load_index(WIKI_INDEX_PATH, meta_json_path=None)
# if faiss_reranker.use_gpu:
#     print("Moving FAISS index to GPU...")
#     res = faiss.StandardGpuResources()
#     faiss_reranker.index = faiss.index_cpu_to_gpu(res, 0, faiss_reranker.index)
#     print("FAISS index successfully moved to GPU.")
    
# load corpus texts from nq_wiki_chunks.jsonl
print("Loading corpus texts from chunks file...")
corpus_texts = []
with open("retrieval/nq_wiki_chunks.jsonl", "r") as f:
    for line in f:
        chunk = json.loads(line)
        corpus_texts.append(chunk["text"])
faiss_reranker.corpus_texts = corpus_texts
print(f"Loaded {len(corpus_texts)} chunks")

cross_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda" if faiss_reranker.use_gpu else "cpu")

out_f = open(OUTPUT_VAL_JSONL, "w")
count = 0

print(f"Processing NQ dev set from {NQ_DEV_PATH}...")
with gzip.open(NQ_DEV_PATH, "rt", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing validation data"):
        if count >= MAX_VAL_SAMPLES:
            break
            
        ex = json.loads(line)
        qid = ex.get("example_id")
        question = ex.get("question_text")
        
        # get gold short answers (may be empty)
        gold_short_texts = []  # Initialize here for each example
        ann = ex.get("annotations", [])
        first_ann = ann[0] if len(ann) > 0 else None
        if first_ann:
            short_answers = first_ann.get("short_answers", [])
            for sa in short_answers:
                if sa["start_token"] >= 0:
                    toks = ex["document_tokens"][sa["start_token"]:sa["end_token"]]
                    gold_short_texts.append(" ".join([t["token"] for t in toks if not t.get("html_token", False)]))

        # fallback to long answer if short answers are missing
        if len(gold_short_texts) == 0 and first_ann:
            long_ans = first_ann.get("long_answer", {})
            if long_ans and long_ans["start_token"] >= 0:
                toks = ex["document_tokens"][long_ans["start_token"]:long_ans["end_token"]]
                long_ans_text = " ".join([t["token"] for t in toks if not t.get("html_token", False)])
                gold_short_texts = [long_ans_text]
                
        # RETRIEVE
        cand = faiss_reranker.search(question, topk=TOP_K_RETRIEVE)
        cand_texts = [c["text"] for c in cand]
        
        # RERANK
        pairs = [(question, t) for t in cand_texts]
        scores = cross_reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(cand_texts, scores), key=lambda x: x[1], reverse=True)
        top_texts = [t for t, s in ranked[:TOP_K_RERANK]]
        context_concat = "\n\n".join(top_texts)

        record = {
            "example_id": qid,
            "question": question,
            "contexts": top_texts,
            "context": context_concat,
            "answers": gold_short_texts
        }
        out_f.write(json.dumps(record) + "\n")
        count += 1

out_f.close()
print(f"Saved {count} NQ validation examples to {OUTPUT_VAL_JSONL}")
