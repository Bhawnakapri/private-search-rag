"""RAG layer: confidence gate -> grounded prompt with numbered sources -> answer (or honest 'not found')."""
from __future__ import annotations

import re

from app.bm25 import tokenize
from app.llm import LLMError

NOT_FOUND = "I couldn't find a reliable answer to that in your documents."
SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you use like [1] or [2]. "
    "If the context does not contain the answer, say you don't know. Do not use outside knowledge."
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def is_confident(hits, cfg) -> bool:
   """Answer only if a top result is semantically close, OR shares enough of the query terms.

    The keyword path needs at least 2 matching terms, so one generic word (like "policy")
    appearing in an unrelated passage cannot make the system answer.
    """
   return any(
        h.dense >= cfg.min_dense or (h.coverage >= cfg.min_coverage and h.matched >= 2)
        for h in hits[:3]
    )


def build_prompt(question: str, hits) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        where = h.source + (f", p.{h.page}" if h.page else "")
        blocks.append(f"[{n}] ({where})\n{h.text}")
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def extractive_answer(question: str, hits, max_sentences: int = 2) -> str:
    """No-LLM fallback: return the sentences from the top passages that best match the question.

    Only sentences scoring at least 60% of the best sentence's term overlap are kept, so a single
    shared word like "must" cannot drag unrelated sentences into the answer. If the best sentence is
    itself a question (FAQ style), the sentence after it - the actual answer - is included too.
    A sentence that appears in two overlapping chunks is returned only once.
    """
    q_terms = set(tokenize(question))
    per_hit = [[s.strip() for s in _SENTENCE.split(h.text)] for h in hits[:3]]
    scored = []
    for n, sentences in enumerate(per_hit, 1):
        for i, sentence in enumerate(sentences):
            overlap = len(q_terms & set(tokenize(sentence)))
            if len(sentence) >= 20 and overlap:
                scored.append((overlap, -n, -i, n, i))
    if not scored:
        return hits[0].text[:300] + " [1]"
    scored.sort(reverse=True)
    best = scored[0][0]
    parts, seen = [], set()
    for overlap, _, _, n, i in scored:
        if overlap < 0.6 * best or len(parts) == max_sentences:
            break
        sentences = per_hit[n - 1]
        text = sentences[i]
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        if text.endswith("?") and i + 1 < len(sentences):
            text += " " + sentences[i + 1]
        parts.append(f"{text} [{n}]")
    return " ".join(parts)

def answer(engine, llm, question: str, k: int = 4, mode: str = "hybrid") -> dict:
    hits = engine.search(question, k=k, mode=mode)
    sources = [
        {"n": i, "source": h.source, "page": h.page, "snippet": h.text[:300],
         "scores": {"bm25": round(h.bm25, 3), "dense": round(h.dense, 3),
                    "coverage": round(h.coverage, 2), "rrf": round(h.rrf, 4)}}
        for i, h in enumerate(hits, 1)
    ]
    if not is_confident(hits, engine.cfg):
        return {"answer": NOT_FOUND, "confident": False, "generator": "none", "sources": sources}

    generator = llm.name
    if llm.available:
        try:
            text = llm.generate(SYSTEM_PROMPT, build_prompt(question, hits))
        except LLMError as exc:
            text, generator = extractive_answer(question, hits), f"extractive (LLM unavailable: {exc})"
    else:
        text = extractive_answer(question, hits)
    return {"answer": text, "confident": True, "generator": generator, "sources": sources}
