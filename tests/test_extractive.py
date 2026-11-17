from app.engine import Hit
from app.rag import extractive_answer


def test_duplicate_sentences_from_overlapping_chunks_appear_once():
    s = "Logistic regression slightly outperforms XGBoost on this dataset."
    hits = [Hit(0, s + " Some other closing text.", "a.md", None),
            Hit(1, "Some intro words here. " + s, "a.md", None)]
    out = extractive_answer("which model outperforms, logistic regression or xgboost", hits)
    assert out.count("outperforms XGBoost") == 1