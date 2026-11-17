from app.llm import NoLLM
from app.rag import answer


def test_a_single_generic_word_match_is_not_enough(engine):
    # "policy" appears in several documents, but nothing in them is about refunds
    r = answer(engine, NoLLM(), "What is the refund policy?")
    assert r["confident"] is False