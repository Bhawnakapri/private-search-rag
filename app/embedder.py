"""Text -> vector. Real semantic model by default; hashing fallback for offline tests/CI."""
from __future__ import annotations

import zlib

import numpy as np

from app.bm25 import tokenize


class HashingEmbedder:
    """Deterministic bag-of-words hashing. Lexical only (NOT semantic) - used so tests run offline."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in tokenize(text):
                h = zlib.crc32(token.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SbertEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


def get_embedder(cfg):
    return HashingEmbedder() if cfg.embedder == "hashing" else SbertEmbedder(cfg.embed_model)
