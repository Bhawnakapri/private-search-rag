"""FastAPI service: /health, /index, /search, /ask."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import Settings
from app.engine import load_or_build
from app.llm import get_llm
from app.rag import answer


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(4, ge=1, le=10)
    mode: str = Field("hybrid", pattern="^(bm25|dense|hybrid)$")


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.engine = load_or_build(cfg)
        app.state.llm = get_llm(cfg)
        yield

    app = FastAPI(title="Private Search + RAG", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(app.state.engine), "llm": app.state.llm.name}

    @app.post("/index")
    def reindex():
        """Re-index the configured documents folder (the folder is fixed by config, not by the caller)."""
        stats = app.state.engine.index_folder(cfg.docs_dir)
        app.state.engine.save(cfg.index_dir)
        return stats

    @app.get("/search")
    def search(q: str = Query(min_length=2, max_length=500), k: int = Query(5, ge=1, le=20),
               mode: str = Query("hybrid", pattern="^(bm25|dense|hybrid)$")):
        hits = app.state.engine.search(q, k=k, mode=mode)
        return {"query": q, "mode": mode, "results": [h.as_dict() for h in hits]}

    @app.post("/ask")
    def ask(req: AskRequest):
        if len(app.state.engine) == 0:
            raise HTTPException(status_code=409, detail="No documents indexed yet. Call POST /index first.")
        return answer(app.state.engine, app.state.llm, req.question, k=req.k, mode=req.mode)

    return app


app = create_app()
