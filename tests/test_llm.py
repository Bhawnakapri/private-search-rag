import pytest
import requests

from app import llm as llm_mod
from app.llm import AnthropicLLM, LLMError, OllamaLLM


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_ollama_request_and_parse(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None, headers=None):
        seen.update(url=url, body=json)
        return FakeResponse({"message": {"content": " hello "}})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    out = OllamaLLM("llama3.2", "http://localhost:11434/").generate("sys", "prompt")
    assert out == "hello"
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["body"]["stream"] is False and seen["body"]["messages"][0]["role"] == "system"


def test_anthropic_request_and_parse(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = {}

    def fake_post(url, json=None, timeout=None, headers=None):
        seen.update(url=url, body=json, headers=headers)
        return FakeResponse({"content": [{"type": "text", "text": "answer"}]})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    assert AnthropicLLM("some-model").generate("sys", "prompt") == "answer"
    assert seen["headers"]["x-api-key"] == "test-key" and seen["body"]["system"] == "sys"


def test_anthropic_without_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError):
        AnthropicLLM("m").generate("s", "p")


def test_network_errors_become_llm_errors(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(llm_mod.requests, "post", boom)
    with pytest.raises(LLMError):
        OllamaLLM("m", "http://x").generate("s", "p")
