# Private Search + RAG

A private search engine for your own documents (`.txt`, `.md`, `.pdf`), extended into a
retrieval-augmented question-answering system. The core search engine (tokenizer, inverted index,
BM25 ranking) is written from scratch; semantic search, fusion, an LLM layer, an API and a UI sit on top.

**Key idea:** the system answers only when it has evidence. If no passage in your documents is a
reliable match, it says so instead of guessing, and always shows the sources behind an answer.

## How it works

```
documents (.txt/.md/.pdf)
   -> loader -> chunker (overlapping word windows, keeps source + page)
        |-> BM25 inverted index (from scratch)      \
        |-> sentence-transformer embeddings -> FAISS  >-- Reciprocal Rank Fusion --> (optional cross-encoder rerank)
                                                                     |
question -> hybrid search -> top-k passages -> confidence gate --no--> "couldn't find it in your documents"
                                                     |yes
                                      grounded prompt with numbered sources
                                                     |
                      LLM (Ollama / Claude)  or  extractive answer (no LLM needed)  -> answer + [1] [2] citations
```

## Quick start (Windows CMD; on Mac/Linux use `source venv/bin/activate` and `export`)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python -m app.ingest --docs data\sample_docs        :: build the index (downloads the embedding model once)
streamlit run ui.py                                  :: web UI
uvicorn app.api:app --reload                         :: REST API on http://localhost:8000/docs
```

Use your own files: put them in a folder and run `python -m app.ingest --docs path\to\folder`
(or set `DOCS_DIR`). Run the tests with `pytest` (they use an offline stand-in embedder, no downloads).

### Pick how answers are generated (`LLM_PROVIDER`)

| Value | What it does | Setup |
|---|---|---|
| `none` (default) | Extractive: returns the best-matching sentences with citations | nothing |
| `ollama` | Local, free LLM | install [Ollama](https://ollama.com), `ollama pull llama3.2`, then `set LLM_PROVIDER=ollama` |
| `anthropic` | Claude via API | `set ANTHROPIC_API_KEY=...`, `set LLM_PROVIDER=anthropic`, optionally `set LLM_MODEL=...` |

If the LLM call fails, the app falls back to the extractive answer instead of crashing.

### Other settings (environment variables)

`EMBED_MODEL`, `RERANK=1` (cross-encoder reranking), `CHUNK_WORDS`, `CHUNK_OVERLAP`,
`MIN_DENSE` / `MIN_COVERAGE` (confidence-gate thresholds), `INDEX_DIR`, `DOCS_DIR`.
See `app/config.py`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, number of chunks, answer generator |
| `POST /index` | re-index the configured documents folder |
| `GET /search?q=...&k=5&mode=hybrid` | ranked passages with BM25, dense, coverage and fusion scores |
| `POST /ask` `{"question": "...", "k": 4}` | answer, confidence flag, numbered sources |

## Measuring retrieval quality

An evaluation script is included. It compares BM25, dense and hybrid retrieval on a JSONL file of
questions with the expected source file, and reports Hit@k and MRR@k:

```
python -m app.evaluate --eval data\eval_sample.jsonl --docs data\sample_docs --k 5
```

The bundled sample set (6 tiny documents, 12 questions) is too easy to separate the three modes - every
mode scores perfectly - so it only shows that the script works. I have **not** yet run a formal evaluation
on a larger document set; so far the system has been tested by hand on real PDFs and READMEs and with
the unit tests. A larger, paraphrased question set is the next step.

## Bugs found by testing on real documents (all fixed, each with a test)

1. **Tied rankings.** When BM25 and dense search disagree about which chunk is first, both chunks get exactly
   the same RRF score, so the order was arbitrary and the less relevant one could win. Ties are now broken
   by the dense (semantic) score.
2. **Garbled answers from markdown.** Table rows, `>` quote markers and `**bold**` markers confused the
   sentence splitter. Markdown is now converted to plain sentences before chunking.
3. **Duplicate sentences.** Overlapping chunks contain the same sentence twice, so an answer could repeat it.
   Repeated sentences are now returned once.
4. **Over-confident answers.** A question like "What is the refund policy?" was answered because one generic
   word ("policy") appeared in an unrelated passage. The keyword route of the confidence gate now needs at
   least two matching query terms.

## Design decisions (and the honest trade-offs)

- **Why write BM25 from scratch?** To understand ranking: term frequency saturation (`k1`), length
  normalisation (`b`), and why rare terms get higher IDF. The hand-written index is small and fully tested.
- **Why hybrid search?** BM25 finds exact terms (names, error codes) that embeddings blur; embeddings
  find paraphrases that share no words with the question. Combining them covers both.
- **Why Reciprocal Rank Fusion?** BM25 scores and cosine similarities are on different scales, so they
  cannot be added directly. RRF fuses by rank position, which sidesteps normalisation.
- **Why break RRF ties with the dense score?** With two ranked lists, chunk A first in one and second in the
  other (and vice versa for chunk B) always gives A and B identical fused scores. The semantic score is the
  more reliable tie-breaker.
- **Why a confidence gate?** A RAG system that always answers will confidently invent things. The gate
  answers only if a top passage is semantically close (`MIN_DENSE`) or contains enough of the query
  terms (`MIN_COVERAGE`, with at least two matching terms). The thresholds are heuristics - tune them on your own documents.
- **Why an extractive mode?** The project works with no API key and no GPU, and it makes retrieval
  quality visible without an LLM's fluent wording hiding mistakes.
- **Why JSON for the saved index, not pickle?** Loading a pickle can execute code; JSON cannot.
- **Why does `/index` take no folder argument?** Letting API callers choose a path would let them make
  the server read arbitrary directories.

## Known limitations

- Re-indexing rebuilds everything (no incremental updates).
- Chunks are fixed-size word windows, not structure-aware; tables in PDFs extract poorly; scanned PDFs need OCR.
- The light stemmer only strips plurals. Retrieval relevance is judged at the file level in the evaluator.
- The confidence gate is heuristic and its thresholds have not been tuned on a large question set.
- In extractive mode, sentences are picked by word overlap, so an LLM (Ollama / Claude) gives better answers.
- No authentication on the API: do not expose it publicly with private documents.

## How this was built

The first version was drafted with AI assistance (Claude). I then ran it on real documents, found and fixed
the bugs listed above, and added tests for each fix.

## Project layout

```
app/  bm25.py (search engine core)  engine.py (hybrid retrieval + save/load)  rag.py (gate + answers)
      llm.py  api.py  loader.py  chunker.py  embedder.py  vector_store.py  ingest.py  evaluate.py
ui.py  Dockerfile  tests/ (pytest)  .github/workflows/ci.yml  data/sample_docs  data/eval_sample.jsonl
```