from app.llm import LLMError, NoLLM
from app.rag import NOT_FOUND, answer


class FakeLLM:
    name, available = "fake", True

    def generate(self, system, prompt):
        assert "Context:" in prompt and "[1]" in prompt   # grounded prompt with numbered sources
        return "Fake grounded answer [1]"


class BrokenLLM:
    name, available = "broken", True

    def generate(self, system, prompt):
        raise LLMError("boom")


def test_answers_with_citation_and_sources(engine):
    r = answer(engine, NoLLM(), "How many days of annual leave do employees get?")
    assert r["confident"] and "24" in r["answer"] and "[1]" in r["answer"]
    assert r["sources"][0]["source"] == "leave_policy.md"


def test_refuses_instead_of_guessing_when_nothing_matches(engine):
    r = answer(engine, NoLLM(), "quantum chromodynamics lattice simulation")
    assert r["confident"] is False and r["answer"] == NOT_FOUND


def test_uses_the_llm_when_available(engine):
    r = answer(engine, FakeLLM(), "what is the vpn helpdesk extension")
    assert r["answer"] == "Fake grounded answer [1]" and r["generator"] == "fake"


def test_falls_back_to_extractive_if_llm_fails(engine):
    r = answer(engine, BrokenLLM(), "what is the vpn helpdesk extension")
    assert r["confident"] and "4100" in r["answer"] and r["generator"].startswith("extractive")


def test_extractive_answer_stays_on_topic(engine):
    r = answer(engine, NoLLM(), "How quickly must on-call acknowledge a page?")
    assert "15 minutes" in r["answer"]
    assert "password" not in r["answer"].lower() and "postmortem" not in r["answer"].lower()


def test_faq_question_is_followed_by_its_answer(engine):
    r = answer(engine, NoLLM(), "How do I reset my password?")
    assert "Forgot password" in r["answer"]


def test_markdown_headings_do_not_leak_into_answers(engine):
    r = answer(engine, NoLLM(), "What is the deadline to submit an expense claim?")
    assert "30 days" in r["answer"] and "Reimbursement Policy" not in r["answer"]
