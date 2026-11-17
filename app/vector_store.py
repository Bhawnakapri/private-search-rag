"""Vector index: FAISS (inner product on normalised vectors = cosine) with a NumPy fallback."""
from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self, vecs: np.ndarray | None = None):
        self.vecs = vecs if vecs is not None else np.zeros((0, 0), dtype="float32")
        self._faiss = None
        self._dirty = True

    def __len__(self) -> int:
        return self.vecs.shape[0]

    def add(self, new: np.ndarray) -> None:
        new = np.asarray(new, dtype="float32")
        self.vecs = new if self.vecs.size == 0 else np.vstack([self.vecs, new])
        self._dirty = True

    def _build(self) -> None:
        self._dirty = False
        try:
            import faiss
        except ImportError:
            self._faiss = None
            return
        index = faiss.IndexFlatIP(self.vecs.shape[1])
        index.add(self.vecs)
        self._faiss = index

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        k = min(k, len(self))
        if self._dirty:
            self._build()
        q = np.asarray(query, dtype="float32").reshape(1, -1)
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = self.vecs @ q[0]
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]
