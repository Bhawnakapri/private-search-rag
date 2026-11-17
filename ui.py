"""Streamlit front end.   streamlit run ui.py"""
import streamlit as st

from app.config import Settings
from app.engine import load_or_build
from app.llm import get_llm
from app.rag import answer

st.set_page_config(page_title="Private Search + RAG", page_icon="🔎", layout="wide")
cfg = Settings.from_env()


@st.cache_resource(show_spinner="Loading index...")
def get_engine():
    return load_or_build(cfg)


engine, llm = get_engine(), get_llm(cfg)

st.title("🔎 Private Search + RAG")
st.caption(f"{len(engine)} chunks indexed | answers by: {llm.name} | embedder: {engine.embedder.name}")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", ["hybrid", "bm25", "dense"])
    k = st.slider("Passages to retrieve", 1, 10, 4)
    if st.button("Re-index documents"):
        stats = engine.index_folder(cfg.docs_dir)
        engine.save(cfg.index_dir)
        st.success(f"Indexed {stats['documents']} documents, {stats['chunks']} chunks")

tab_ask, tab_search = st.tabs(["Ask", "Search only"])

with tab_ask:
    question = st.text_input("Ask a question about your documents")
    if question:
        result = answer(engine, llm, question, k=k, mode=mode)
        if result["confident"]:
            st.success(result["answer"])
        else:
            st.warning(result["answer"] + " (closest passages shown below)")
        st.caption(f"generator: {result['generator']}")
        for s in result["sources"]:
            where = s["source"] + (f" p.{s['page']}" if s["page"] else "")
            with st.expander(f"[{s['n']}] {where}"):
                st.write(s["snippet"])
                st.json(s["scores"])

with tab_search:
    query = st.text_input("Search query")
    if query:
        for h in engine.search(query, k=k, mode=mode):
            with st.expander(f"{h.source} | rrf {h.rrf:.4f} | dense {h.dense:.2f} | bm25 {h.bm25:.2f}"):
                st.write(h.text)
