import os
from dataclasses import replace

os.environ.setdefault("EMBEDDER", "hashing")   # tests run offline: no model download
os.environ.setdefault("LLM_PROVIDER", "none")

import pytest

from app.config import Settings
from app.engine import SearchEngine


@pytest.fixture
def cfg(tmp_path):
    return replace(Settings.from_env(), embedder="hashing", llm_provider="none",
                   docs_dir="data/sample_docs", index_dir=str(tmp_path / "idx"))


@pytest.fixture
def engine(cfg):
    e = SearchEngine(cfg)
    e.index_folder(cfg.docs_dir)
    return e
