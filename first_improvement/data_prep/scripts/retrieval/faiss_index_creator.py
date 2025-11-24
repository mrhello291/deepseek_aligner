import os
import json
import gzip
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk

# -----------------------------
# CONFIGURATION
# -----------------------------

# Download NLTK sentence tokenizer (only first run)
nltk.download("punkt")
nltk.download("punkt_tab")

INPUT_TRAIN_FILES = [
    "nq-train-00.jsonl.gz",
    "nq-train-01.jsonl.gz",
    "nq-train-02.jsonl.gz",
    "nq-train-03.jsonl.gz",
    "nq-train-04.jsonl.gz",
]

INPUT_DEV_FILES = [
    "nq-dev-00.jsonl.gz",
]

OUTPUT_CHUNKS_FILE = "nq_wiki_chunks.jsonl"
OUTPUT_FAISS_CPU_FILE = "nq_wiki_index.faiss"
OUTPUT_FAISS_GPU_FILE = "nq_wiki_index_gpu.faiss"
OUTPUT_META_FILE = "metadata.json"

CHUNK_SIZE = 512          # ~words per chunk
CHUNK_OVERLAP = 50        # overlap between consecutive chunks
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # small, fast embedding model

# -----------------------------
# STEP 1. Deduplicate and extract plain text
# -----------------------------
print("Extracting and deduplicating Wikipedia articles...")

seen_titles = set()
articles = {}

# Process training files
for path in INPUT_TRAIN_FILES:
    full_path = f"datasets/nq/train/{path}"
    with gzip.open(full_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading train/{path}"):
            ex = json.loads(line)
            title = ex.get("document_title")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # Reconstruct plain text from document_tokens
            tokens = [
                t["token"] for t in ex["document_tokens"]
                if not t.get("html_token", False)
            ]
            text = " ".join(tokens).strip()
            if len(text) < 200:
                continue  # skip very short docs

            articles[title] = {
                "title": title,
                "text": text,
                "url": ex.get("document_url", "")
            }

print(f"✅ Extracted {len(articles):,} unique articles from training data.")

# Process validation files
for path in INPUT_DEV_FILES:
    full_path = f"datasets/nq/dev/{path}"
    with gzip.open(full_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading dev/{path}"):
            ex = json.loads(line)
            title = ex.get("document_title")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # Reconstruct plain text from document_tokens
            tokens = [
                t["token"] for t in ex["document_tokens"]
                if not t.get("html_token", False)
            ]
            text = " ".join(tokens).strip()
            if len(text) < 200:
                continue  # skip very short docs

            articles[title] = {
                "title": title,
                "text": text,
                "url": ex.get("document_url", "")
            }

print(f"✅ Total extracted {len(articles):,} unique articles (train + dev).")

# -----------------------------
# STEP 2. Sentence-aware chunking
# -----------------------------
print("Chunking articles with NLTK sentence-aware chunker...")

from nltk.tokenize import sent_tokenize

def chunk_text_sentence_aware(text, max_words=512, overlap=50):
    """
    Splits text into overlapping chunks that preserve sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_words:
            # yield current chunk
            chunks.append(" ".join(current_chunk))
            # start new chunk with overlap
            overlap_words = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words + words
            current_len = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

chunks = []
for title, doc in tqdm(articles.items(), desc="Chunking"):
    for chunk in chunk_text_sentence_aware(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP):
        chunks.append({
            "title": title,
            "text": chunk,
            "url": doc["url"]
        })

print(f"✅ Created {len(chunks):,} chunks total.")

# Save chunks to JSONL
with open(OUTPUT_CHUNKS_FILE, "w") as f_out:
    for c in chunks:
        f_out.write(json.dumps(c) + "\n")

print(f"📄 Saved chunks to {OUTPUT_CHUNKS_FILE}")

# -----------------------------
# STEP 3. Build FAISS indexes (CPU and GPU)
# -----------------------------
print("Building FAISS indexes with SentenceTransformers embeddings...")

embedder = SentenceTransformer(EMBEDDING_MODEL)
texts = [c["text"] for c in chunks]
titles = [c["title"] for c in chunks]

embeddings = embedder.encode(
    texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
)

dim = embeddings.shape[1]
print(f"Embedding dimension: {dim}, Total vectors: {len(embeddings)}")

# Build CPU index (IndexFlatIP for inner product / cosine similarity)
print("Building CPU index (IndexFlatIP)...")
cpu_index = faiss.IndexFlatIP(dim)
cpu_index.add(embeddings)
faiss.write_index(cpu_index, OUTPUT_FAISS_CPU_FILE)
print(f"✅ CPU FAISS index saved to {OUTPUT_FAISS_CPU_FILE}")

# Build GPU-optimized index
print("Building GPU-optimized index (IndexIVFFlat)...")
try:
    # Check if GPU is available
    ngpus = faiss.get_num_gpus()
    print(f"Detected {ngpus} GPU(s)")
    
    if ngpus > 0:
        # For GPU, we'll use IVFFlat for faster search
        # Number of clusters (nlist) - rule of thumb: sqrt(N) to 4*sqrt(N)
        nlist = min(4096, int(np.sqrt(len(embeddings)) * 4))
        print(f"Using {nlist} clusters for IVF index")
        
        # Create IVF index
        quantizer = faiss.IndexFlatIP(dim)
        gpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index (IVF requires training)
        print("Training GPU index...")
        gpu_index.train(embeddings)
        
        # Add vectors
        print("Adding vectors to GPU index...")
        gpu_index.add(embeddings)
        
        # Set search parameters (nprobe = number of clusters to search)
        gpu_index.nprobe = min(32, nlist // 10)  # Balance between speed and accuracy
        
        # Save GPU index
        faiss.write_index(gpu_index, OUTPUT_FAISS_GPU_FILE)
        print(f"✅ GPU-optimized FAISS index saved to {OUTPUT_FAISS_GPU_FILE}")
        print(f"   (IVFFlat with {nlist} clusters, nprobe={gpu_index.nprobe})")
    else:
        print("⚠️  No GPU detected, creating CPU-based IVF index instead...")
        nlist = min(4096, int(np.sqrt(len(embeddings)) * 4))
        quantizer = faiss.IndexFlatIP(dim)
        gpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        gpu_index.train(embeddings)
        gpu_index.add(embeddings)
        gpu_index.nprobe = min(32, nlist // 10)
        faiss.write_index(gpu_index, OUTPUT_FAISS_GPU_FILE)
        print(f"✅ IVF index (CPU-based) saved to {OUTPUT_FAISS_GPU_FILE}")
        
except Exception as e:
    print(f"⚠️  Error creating GPU index: {e}")
    print("   Falling back to CPU index only")
    # Copy CPU index as fallback
    import shutil
    shutil.copy(OUTPUT_FAISS_CPU_FILE, OUTPUT_FAISS_GPU_FILE)
    print(f"✅ Copied CPU index to {OUTPUT_FAISS_GPU_FILE} as fallback")

# -----------------------------
# STEP 4. Save metadata
# -----------------------------
metadata = {
    "titles": titles,
    "model": EMBEDDING_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "chunking_method": "sentence_aware",
    "total_chunks": len(chunks),
    "total_articles": len(articles),
    "index_types": {
        "cpu": OUTPUT_FAISS_CPU_FILE,
        "gpu": OUTPUT_FAISS_GPU_FILE
    },
    "sources": {
        "train_files": INPUT_TRAIN_FILES,
        "dev_files": INPUT_DEV_FILES
    }
}
with open(OUTPUT_META_FILE, "w") as f_meta:
    json.dump(metadata, f_meta, indent=2)

print(f"📄 Saved metadata to {OUTPUT_META_FILE}")
print("\n" + "="*60)
print("✅ All done!")
print("="*60)
print(f"Summary:")
print(f"  - Articles processed: {len(articles):,}")
print(f"  - Chunks created: {len(chunks):,}")
print(f"  - CPU index: {OUTPUT_FAISS_CPU_FILE}")
print(f"  - GPU index: {OUTPUT_FAISS_GPU_FILE}")
print(f"  - Metadata: {OUTPUT_META_FILE}")
print(f"  - Chunks file: {OUTPUT_CHUNKS_FILE}")
print("="*60)
