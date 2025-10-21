# retrieval/faiss_rerank.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class FaissReranker:
    def __init__(self, model_name="all-MiniLM-L6-v2", use_gpu=True):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []
        self.use_gpu = use_gpu


    def index_corpus(self, texts):
        embs = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        d = embs.shape[1]
        # Use IndexFlatIP with normalized vectors (cosine sim)
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(d)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(embs)
        self.corpus = texts


    def search(self, query, topk=10):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({"text": self.corpus[idx], "score": float(score)})
        return results