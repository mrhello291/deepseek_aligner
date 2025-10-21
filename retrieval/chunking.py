# retrieval/chunking.py
from typing import List
import tiktoken  # optional; fallback to simple tokenization
import nltk
nltk.download('punkt')

def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50):
    tokens = nltk.word_tokenize(text)
    out = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        out.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return out
