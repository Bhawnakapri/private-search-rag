"""Load .txt / .md / .pdf files from a folder into page-level documents."""
from __future__ import annotations

import re
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def clean_markdown(text: str) -> str:
    """Turn markdown into plain sentences so tables, quotes and bold markers don't garble answers."""
    text = re.sub(r"^[ \t]*>[ \t]?", "", text, flags=re.M)                          # blockquote markers
    text = re.sub(r"^[ \t]*\|?[ \t:\-|]+\|[ \t:\-|]*$", "", text, flags=re.M)      # table separator rows |---|---|
    text = re.sub(r"^[ \t]*\|(.+?)\|?[ \t]*$",                                    # table row -> one sentence
                  lambda m: ", ".join(c.strip() for c in m.group(1).split("|") if c.strip()) + ".",
                  text, flags=re.M)
    text = re.sub(r"[*_`]{1,3}", "", text)                                          # bold / italic / code markers
    return re.sub(r"^#+[ \t]*(.+?)[ \t]*$", r"\1.", text, flags=re.M)               # "# Title" -> "Title."


def load_documents(folder: str) -> list[dict]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder not found: {folder}")
    docs: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if path.suffix.lower() == ".pdf":
            docs.extend(_load_pdf(path, rel))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix.lower() == ".md":
                text = clean_markdown(text)
            if text.strip():
                docs.append({"source": rel, "page": None, "text": text})
    return docs


def _load_pdf(path: Path, rel: str) -> list[dict]:
    from pypdf import PdfReader  # imported lazily so text-only use needs no PDF library

    out = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append({"source": rel, "page": number, "text": text})
    return out