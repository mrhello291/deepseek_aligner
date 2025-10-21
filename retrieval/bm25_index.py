# retrieval/bm25_index.py
from rank_bm25 import BM25Okapi
import os, json
from typing import List
import nltk
nltk.download('punkt')

def build_bm25_from_wikipedia(wiki_dir: str, out_path: str):
    """
    wiki_dir: directory with one .json per article: {"title": ..., "text": "..."}
    out_path: path to save a json with titles and tokenized texts
    """
    docs = []
    meta = []
    for fname in os.listdir(wiki_dir):
        if not fname.endswith(".json"):
            continue
        full = os.path.join(wiki_dir, fname)
        data = json.load(open(full, "r", encoding="utf-8"))
        text = data.get("text", "")
        title = data.get("title", fname.replace(".json", ""))
        # coarse: use whole article tokenized
        tokens = nltk.word_tokenize(text.lower())
        docs.append(tokens)
        meta.append({"title": title, "path": full})
    bm25 = BM25Okapi(docs)
    # Save metadata + docs in simple format
    out = {"meta": meta}
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    # store bm25 object via pickle (note: large)
    import pickle
    pickle.dump(bm25, open(out_path + ".bm25.pkl", "wb"))
    print("Built BM25 ->", out_path)
