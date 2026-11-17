"""Pluggable answer generators. Default is 'none' (extractive answers) so the app works with no API key."""
from __future__ import annotations

import os

import requests


class LLMError(Exception):
    pass


class NoLLM:
    name = "extractive"
    available = False

    def generate(self, system: str, prompt: str) -> str:
        raise LLMError("no LLM configured")


class OllamaLLM:
    """Local, free models via Ollama (https://ollama.com)."""
    name = "ollama"
    available = True

    def __init__(self, model: str, host: str):
        self.model, self.host = model, host.rstrip("/")

    def generate(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc


class AnthropicLLM:
    """Claude via the Anthropic Messages API. Needs ANTHROPIC_API_KEY."""
    name = "anthropic"
    available = True
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str):
        self.model = model

    def generate(self, system: str, prompt: str) -> str:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": self.model, "max_tokens": 600, "system": system,
                "messages": [{"role": "user", "content": prompt}]}
        try:
            r = requests.post(self.URL, headers=headers, json=body, timeout=60)
            r.raise_for_status()
            return r.json()["content"][0]["text"].strip()
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc


def get_llm(cfg):
    if cfg.llm_provider == "ollama":
        return OllamaLLM(cfg.llm_model or "llama3.2", cfg.ollama_host)
    if cfg.llm_provider == "anthropic":
        return AnthropicLLM(cfg.llm_model or "claude-haiku-5-5")
    return NoLLM()
