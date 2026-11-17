import pytest

from app.chunker import chunk_document


def doc(n):
    return {"source": "a.md", "page": None, "text": " ".join(f"w{i}" for i in range(n))}


def test_short_document_is_one_chunk():
    assert len(chunk_document(doc(50), 120, 30)) == 1


def test_chunks_overlap_and_cover_everything():
    chunks = chunk_document(doc(300), 100, 20)
    assert len(chunks) == 4
    assert chunks[0]["text"].split()[-20:] == chunks[1]["text"].split()[:20]
    assert chunks[-1]["text"].split()[-1] == "w299"


def test_metadata_is_kept():
    c = chunk_document({"source": "x.pdf", "page": 3, "text": "hello world " * 10})[0]
    assert (c["source"], c["page"]) == ("x.pdf", 3)


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document(doc(10), 5, 5)
