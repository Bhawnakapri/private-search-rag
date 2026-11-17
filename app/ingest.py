"""CLI: build and save the index.   python -m app.ingest --docs path/to/folder"""
from __future__ import annotations

import argparse

from app.config import Settings
from app.engine import SearchEngine


def main() -> None:
    cfg = Settings.from_env()
    parser = argparse.ArgumentParser(description="Index a folder of .txt/.md/.pdf files")
    parser.add_argument("--docs", default=cfg.docs_dir)
    parser.add_argument("--out", default=cfg.index_dir)
    args = parser.parse_args()
    engine = SearchEngine(cfg)
    stats = engine.index_folder(args.docs)
    engine.save(args.out)
    print(f"Indexed {stats['documents']} documents into {stats['chunks']} chunks -> {args.out}")


if __name__ == "__main__":
    main()
