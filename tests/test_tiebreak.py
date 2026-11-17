def test_tied_rrf_scores_are_broken_by_dense_score(engine, monkeypatch):
    # BM25 and dense disagree about which chunk is first, so both get exactly the same fused RRF score
    monkeypatch.setattr(engine.bm25, "search", lambda q, n: [(0, 5.0), (1, 4.0)])
    monkeypatch.setattr(engine.vectors, "search", lambda q, n: [(1, 0.9), (0, 0.8)])
    hits = engine.search(engine.chunks[1]["text"], k=2, mode="hybrid")
    assert hits[0].rrf == hits[1].rrf
    assert hits[0].id == 1          # equal fused score -> the higher dense similarity comes first