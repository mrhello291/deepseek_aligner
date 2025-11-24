# retrieval/faiss_rerank.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from tqdm import tqdm

class FaissReranker:
    def __init__(self, model_name="all-MiniLM-L6-v2", use_gpu=False, normalize=True, device_map=None):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.corpus_texts = None
        self.normalize = normalize
        self.use_gpu = use_gpu
        self.device_map = device_map

    def index_corpus(self, texts, embeddings=None, use_float16=False):
        """
        texts: list of strings
        embeddings: optional numpy array (n, d) precomputed
        """
        self.corpus_texts = list(texts)
        if embeddings is None:
            embs = self.embedder.encode(self.corpus_texts, show_progress_bar=True, convert_to_numpy=True)
        else:
            embs = embeddings
        if self.normalize:
            faiss.normalize_L2(embs)
        d = embs.shape[1]
        # CPU index (flat inner-product)
        index = faiss.IndexFlatIP(d)
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                print("Faiss GPU init failed, falling back to CPU:", e)
        index.add(embs.astype(np.float32))
        self.index = index

    def load_index(self, faiss_index_path, meta_json_path=None, prefer_gpu=None):
        """
        Load FAISS index from disk. Optionally load to GPU.
        
        Args:
            faiss_index_path: Path to FAISS index file
            meta_json_path: Optional path to metadata JSON
            prefer_gpu: If True, try to load GPU index or convert CPU index to GPU.
                       If None, uses self.use_gpu setting.
        """
        if prefer_gpu is None:
            prefer_gpu = self.use_gpu
            
        # Read index from disk
        self.index = faiss.read_index(faiss_index_path)
        print(f"Loaded FAISS index from {faiss_index_path}")
        print(f"  Index type: {type(self.index).__name__}")
        print(f"  Total vectors: {self.index.ntotal}")
        
        # Try to move to GPU if requested
        if prefer_gpu and not isinstance(self.index, faiss.GpuIndex):
            try:
                print("Attempting to move index to GPU...")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("✅ Index successfully moved to GPU")
            except Exception as e:
                print(f"⚠️  Could not move index to GPU: {e}")
                print("   Continuing with CPU index")
        
        if meta_json_path:
            meta = json.load(open(meta_json_path))
            self.corpus_texts = meta.get("texts")  # only if you saved them

    def search(self, query, topk=50):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        if self.normalize:
            faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({"id": int(idx), "text": self.corpus_texts[idx] if self.corpus_texts is not None else None, "score": float(score)})
        return results

    def batch_search(self, queries, topk=50, batch_size=16):
        """
        returns for each query a list of dicts like search()
        """
        q_embs = self.embedder.encode(queries, convert_to_numpy=True, show_progress_bar=True)
        if self.normalize:
            faiss.normalize_L2(q_embs)
        D, I = self.index.search(q_embs, topk)
        all_res = []
        for qidx in range(len(queries)):
            res = []
            for score, idx in zip(D[qidx], I[qidx]):
                res.append({"id": int(idx), "text": self.corpus_texts[idx] if self.corpus_texts is not None else None, "score": float(score)})
            all_res.append(res)
        return all_res
