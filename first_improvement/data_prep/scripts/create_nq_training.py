# data_prep/create_nq_training.py
import gzip, json, os, sys
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import faiss

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from retrieval.faiss_rerank import FaissReranker

# config
NQ_SHARDS = [
    "datasets/nq/train/nq-train-00.jsonl.gz",
    "datasets/nq/train/nq-train-01.jsonl.gz",
    "datasets/nq/train/nq-train-02.jsonl.gz",
    "datasets/nq/train/nq-train-03.jsonl.gz",
    "datasets/nq/train/nq-train-04.jsonl.gz",
]
WIKI_INDEX_PATH = "retrieval/nq_wiki_index.faiss"    # your existing index
WIKI_META_PATH = "retrieval/metadata.json"           # contains titles mapping
OUTPUT_TRAIN_JSONL = "data_prep/train_nq_rag.jsonl"
TOP_K_RETRIEVE = 100
TOP_K_RERANK = 5

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

# helper: extract plain text from doc tokens
def reconstruct_text(doc):
    tokens = [t["token"] for t in doc["document_tokens"] if not t.get("html_token", False)]
    return " ".join(tokens)

out_f = open(OUTPUT_TRAIN_JSONL, "w")

for shard in NQ_SHARDS:
    with gzip.open(shard, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(shard)}"):
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

            # Skip examples if you want (or keep)
            # RETRIEVE
            cand = faiss_reranker.search(question, topk=TOP_K_RETRIEVE)
            cand_texts = [c["text"] for c in cand]
            # RERANK
            pairs = [(question, t) for t in cand_texts]
            scores = cross_reranker.predict(pairs, show_progress_bar=False, batch_size=32)
            ranked = sorted(zip(cand_texts, scores), key=lambda x: x[1], reverse=True)
            top_texts = [t for t, s in ranked[:TOP_K_RERANK]]
            context_concat = "\n\n".join(top_texts)  # you can join with separators

            record = {
                "example_id": qid,
                "question": question,
                "contexts": top_texts,   # list of contexts (best)
                "context": context_concat,  # flattened context for generator input
                "answers": gold_short_texts
            }
            out_f.write(json.dumps(record) + "\n")

out_f.close()
print("Saved NQ RAG training examples to", OUTPUT_TRAIN_JSONL)
