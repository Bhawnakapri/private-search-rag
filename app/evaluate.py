"""CLI: measure retrieval quality (BM25 vs dense vs hybrid) on your own questions.

Each line of the eval file is JSON: {"question": "...", "source": "file_that_contains_the_answer.md"}
A result counts as relevant if its chunk comes from the expected source file.
   python -m app.evaluate --eval data/eval_sample.jsonl --docs data/sample_docs --k 5
"""
from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.engine import SearchEngine


def evaluate(engine: SearchEngine, items: list[dict], k: int) -> dict:
    results = {}
    for mode in ("bm25", "dense", "hybrid"):
        hit_at_k, rr = 0, 0.0
        for item in items:
            hits = engine.search(item["question"], k=k, mode=mode)
            rank = next((i for i, h in enumerate(hits, 1) if h.source == item["source"]), None)
            if rank:
                hit_at_k += 1
                rr += 1.0 / rank
        results[mode] = {"hit@k": hit_at_k / len(items), "mrr@k": rr / len(items)}
    return results


def main() -> None:
    cfg = Settings.from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/eval_sample.jsonl")
    parser.add_argument("--docs", default=cfg.docs_dir)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    with open(args.eval, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    engine = SearchEngine(cfg)
    engine.index_folder(args.docs)
    print(f"{len(items)} questions, embedder={engine.embedder.name}, k={args.k}\n")
    print(f"{'mode':<8} {'Hit@k':>7} {'MRR@k':>7}")
    for mode, m in evaluate(engine, items, args.k).items():
        print(f"{mode:<8} {m['hit@k']:>7.3f} {m['mrr@k']:>7.3f}")


if __name__ == "__main__":
    main()
