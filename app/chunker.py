"""Split documents into overlapping word windows, keeping source/page metadata."""
from __future__ import annotations


def chunk_document(doc: dict, chunk_words: int = 120, overlap: int = 30) -> list[dict]:
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    words = doc["text"].split()
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_words]
        chunks.append({"text": " ".join(window), "source": doc["source"], "page": doc["page"]})
        if start + chunk_words >= len(words):
            break
    return chunks
