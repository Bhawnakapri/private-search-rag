"""Settings, read from environment variables so nothing is hard-coded."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    docs_dir: str = "data/sample_docs"
    index_dir: str = "index_store"
    embedder: str = "sbert"            # sbert | hashing (offline, lexical stand-in used by tests/CI)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank: bool = False               # optional cross-encoder reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: str = "none"         # none | ollama | anthropic
    llm_model: str = ""                # empty = provider default
    ollama_host: str = "http://localhost:11434"
    chunk_words: int = 120
    chunk_overlap: int = 30
    min_dense: float = 0.30            # confidence gate: best cosine similarity ...
    min_coverage: float = 0.50         # ... or fraction of query terms found in a chunk

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            docs_dir=os.getenv("DOCS_DIR", d.docs_dir),
            index_dir=os.getenv("INDEX_DIR", d.index_dir),
            embedder=os.getenv("EMBEDDER", d.embedder),
            embed_model=os.getenv("EMBED_MODEL", d.embed_model),
            rerank=os.getenv("RERANK", "0").lower() in ("1", "true", "yes"),
            rerank_model=os.getenv("RERANK_MODEL", d.rerank_model),
            llm_provider=os.getenv("LLM_PROVIDER", d.llm_provider).lower(),
            llm_model=os.getenv("LLM_MODEL", d.llm_model),
            ollama_host=os.getenv("OLLAMA_HOST", d.ollama_host),
            chunk_words=int(os.getenv("CHUNK_WORDS", d.chunk_words)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", d.chunk_overlap)),
            min_dense=float(os.getenv("MIN_DENSE", d.min_dense)),
            min_coverage=float(os.getenv("MIN_COVERAGE", d.min_coverage)),
        )
