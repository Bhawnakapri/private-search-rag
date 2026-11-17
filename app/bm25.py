"""A small search engine core, written from scratch: tokenizer + inverted index + BM25.

postings: term -> {doc_id: term_frequency}
score(q, d) = sum over query terms of IDF(t) * tf*(k1+1) / (tf + k1*(1 - b + b*len(d)/avg_len))
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOPWORDS = set(
    "a an the and or of to in on for with is are was were be been it its this that as at by from "
    "can do does how what when where which who will you your i we".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Very light plural stripping so 'tickets' matches 'ticket'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: list[int] = []
        self.total_len = 0

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    def add(self, tokens: list[str]) -> int:
        doc_id = self.n_docs
        for term, tf in Counter(tokens).items():
            self.postings[term][doc_id] = tf
        self.doc_len.append(len(tokens))
        self.total_len += len(tokens)
        return doc_id

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        if not self.n_docs:
            return []
        avg_len = self.total_len / self.n_docs or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for doc_id, tf in posting.items():
                norm = 1 - self.b + self.b * self.doc_len[doc_id] / avg_len
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]

    def matched_terms(self, query_tokens: list[str], doc_id: int) -> int:
        """How many distinct query terms appear in the document."""
        return sum(1 for t in set(query_tokens) if doc_id in self.postings.get(t, {}))

    def coverage(self, query_tokens: list[str], doc_id: int) -> float:
        """Fraction of distinct query terms that appear in the document."""
        terms = set(query_tokens)
        return self.matched_terms(query_tokens, doc_id) / len(terms) if terms else 0.0

    # --- persistence (plain JSON, no pickle) ---
    def to_dict(self) -> dict:
        return {
            "k1": self.k1, "b": self.b, "doc_len": self.doc_len,
            "postings": {t: {str(d): tf for d, tf in p.items()} for t, p in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        idx = cls(data["k1"], data["b"])
        idx.doc_len = list(data["doc_len"])
        idx.total_len = sum(idx.doc_len)
        for term, posting in data["postings"].items():
            idx.postings[term] = {int(d): tf for d, tf in posting.items()}
        return idx
