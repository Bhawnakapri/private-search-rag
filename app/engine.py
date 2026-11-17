"""Hybrid search engine: BM25 (lexical) + dense vectors (semantic), fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.bm25 import BM25Index, tokenize
from app.chunker import chunk_document
from app.embedder import get_embedder
from app.loader import load_documents
from app.vector_store import VectorStore

RRF_K = 60  # standard constant from the original RRF paper


@dataclass
class Hit:
    id: int
    text: str
    source: str
    page: int | None
    bm25: float = 0.0       # BM25 score (0 if not in the lexical candidates)
    dense: float = 0.0      # cosine similarity between query and chunk
    coverage: float = 0.0   # fraction of query terms present in the chunk
    matched: int = 0        # how many distinct query terms are present in the chunk
    rrf: float = 0.0        # fused rank score
    rerank: float | None = None
    score: float = 0.0      # final ordering score

    def as_dict(self) -> dict:
        return asdict(self)


class SearchEngine:
    def __init__(self, cfg, embedder=None, reranker=None):
        self.cfg = cfg
        self.embedder = embedder or get_embedder(cfg)
        self._reranker = reranker
        self.reset()

    def __len__(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        self.chunks: list[dict] = []
        self.bm25 = BM25Index()
        self.vectors = VectorStore()

    # ---------- indexing ----------
    def index_folder(self, folder: str) -> dict:
        self.reset()
        docs = load_documents(folder)
        chunks = [c for d in docs for c in chunk_document(d, self.cfg.chunk_words, self.cfg.chunk_overlap)]
        self.add_chunks(chunks)
        return {"documents": len({d["source"] for d in docs}), "chunks": len(chunks)}

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        vecs = self.embedder.encode([c["text"] for c in chunks])
        for chunk in chunks:
            self.bm25.add(tokenize(chunk["text"]))
            self.chunks.append(chunk)
        self.vectors.add(vecs)

    # ---------- search ----------
    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError("mode must be bm25, dense or hybrid")
        if not self.chunks:
            return []
        n = max(k * 4, 20)
        q_tokens = tokenize(query)
        q_vec = self.embedder.encode([query])[0]

        bm25 = dict(self.bm25.search(q_tokens, n)) if mode in ("bm25", "hybrid") else {}
        dense = dict(self.vectors.search(q_vec, n)) if mode in ("dense", "hybrid") else {}
        bm25_rank = {d: r for r, d in enumerate(sorted(bm25, key=bm25.get, reverse=True), 1)}
        dense_rank = {d: r for r, d in enumerate(sorted(dense, key=dense.get, reverse=True), 1)}

        hits = []
        for i in set(bm25) | set(dense):
            rrf = 0.0
            if i in bm25_rank:
                rrf += 1.0 / (RRF_K + bm25_rank[i])
            if i in dense_rank:
                rrf += 1.0 / (RRF_K + dense_rank[i])
            chunk = self.chunks[i]
            hits.append(Hit(
                id=i, text=chunk["text"], source=chunk["source"], page=chunk["page"],
                bm25=bm25.get(i, 0.0), dense=float(self.vectors.vecs[i] @ q_vec),
                coverage=self.bm25.coverage(q_tokens, i), matched=self.bm25.matched_terms(q_tokens, i),
                rrf=rrf, score=rrf,
            ))
        hits.sort(key=lambda h: (-h.score, -h.dense))
        if self.cfg.rerank and mode == "hybrid":
            hits = self._rerank(query, hits[:n])
        return hits[:k]

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # lazy: heavy import

            self._reranker = CrossEncoder(self.cfg.rerank_model)
        scores = self._reranker.predict([(query, h.text) for h in hits])
        for hit, s in zip(hits, scores):
            hit.rerank = hit.score = float(s)
        return sorted(hits, key=lambda h: -h.score)

    # ---------- persistence ----------
    def save(self, folder: str) -> None:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        np.save(folder_path / "vectors.npy", self.vectors.vecs)
        payload = {"embedder": self.embedder.name, "chunks": self.chunks, "bm25": self.bm25.to_dict()}
        (folder_path / "index.json").write_text(json.dumps(payload), encoding="utf-8")

    def load(self, folder: str) -> None:
        meta_path = Path(folder) / "index.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No saved index in {folder}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["embedder"] != self.embedder.name:
            raise ValueError(f"Index was built with '{meta['embedder']}', current embedder is '{self.embedder.name}'")
        self.chunks = meta["chunks"]
        self.bm25 = BM25Index.from_dict(meta["bm25"])
        self.vectors = VectorStore(np.load(Path(folder) / "vectors.npy"))


def load_or_build(cfg) -> SearchEngine:
    """Use the saved index if it matches the current embedder, otherwise index the docs folder."""
    engine = SearchEngine(cfg)
    try:
        engine.load(cfg.index_dir)
    except (FileNotFoundError, ValueError):
        engine.index_folder(cfg.docs_dir)
        engine.save(cfg.index_dir)
    return engine
