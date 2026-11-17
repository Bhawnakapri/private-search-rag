from app.bm25 import BM25Index, tokenize


def build(docs):
    idx = BM25Index()
    for d in docs:
        idx.add(tokenize(d))
    return idx


def test_tokenize_lowercases_stems_and_drops_stopwords():
    assert tokenize("The Tickets ARE open") == ["ticket", "open"]


def test_matching_document_ranks_first():
    idx = build(["leave policy for employees", "vpn setup guide", "expense claims and receipts"])
    top, _ = idx.search(tokenize("vpn guide"), k=3)[0]
    assert top == 1


def test_rare_terms_weigh_more_than_common_ones():
    idx = build(["policy policy policy", "policy cryptography", "policy handbook", "policy rules"])
    assert idx.idf("cryptography") > idx.idf("policy")
    assert idx.search(tokenize("policy cryptography"), k=1)[0][0] == 1


def test_coverage_is_fraction_of_query_terms_present():
    idx = build(["alpha beta gamma"])
    assert idx.coverage(tokenize("alpha beta zeta"), 0) == 2 / 3


def test_unknown_terms_return_nothing():
    assert build(["alpha beta"]).search(tokenize("zzz")) == []


def test_json_roundtrip_gives_same_results():
    idx = build(["alpha beta", "beta gamma gamma", "delta"])
    clone = BM25Index.from_dict(idx.to_dict())
    q = tokenize("beta gamma")
    assert clone.search(q) == idx.search(q)
