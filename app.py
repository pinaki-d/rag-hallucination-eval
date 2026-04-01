import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

from rag.ingestor import ingest_pipeline, load_vectorstore, CHROMA_DIR
from rag.retriever import get_retriever_by_mode
from rag.pipeline import ask

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Domain-Aware RAG System")
st.caption("Hallucination evaluation · Hybrid retrieval · Powered by LangChain + GPT-4o-mini + RAGAS")

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

# ─────────────────────────────────────────────
# Sidebar — configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        options=["hybrid", "dense", "bm25"],
        index=0,
        help="Hybrid combines BM25 keyword + dense semantic search"
    )

    chunk_size = st.selectbox(
        "Chunk Size",
        options=[512, 1024],
        index=0,
        help="Number of characters per document chunk"
    )

    top_k = st.slider("Top K chunks to retrieve", min_value=2, max_value=8, value=4)

    st.divider()

    st.header("📂 Document Ingestion")

    if st.button("🔄 Build / Rebuild Vectorstore", type="primary"):
        with st.spinner(f"Ingesting PDFs with chunk_size={chunk_size}..."):
            try:
                vectorstore = ingest_pipeline(chunk_size=chunk_size)
                st.session_state.vectorstore = vectorstore
                st.success("Vectorstore built successfully!")
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Add PDF files to data/papers/ folder and try again.")

    if CHROMA_DIR.exists() and st.session_state.vectorstore is None:
        with st.spinner("Loading existing vectorstore..."):
            try:
                st.session_state.vectorstore = load_vectorstore()
                st.success("Loaded existing vectorstore")
            except Exception:
                st.warning("No vectorstore found. Click Build above.")

    st.divider()

    st.header("📊 Benchmark Results")
    results_path = Path("data/benchmark_results.csv")
    if results_path.exists():
        df = pd.read_csv(results_path)
        st.dataframe(df[["config", "faithfulness", "answer_relevancy", "context_recall"]])

        best_faith = df.loc[df["faithfulness"].idxmax()]
        st.success(f"Best faithfulness: **{best_faith['config']}** ({best_faith['faithfulness']:.3f})")
    else:
        st.caption("Run evaluation to see benchmark results here.")
        st.code("python -m evaluation.evaluator", language="bash")

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.info("👈 First, add PDFs to `data/papers/` then click **Build / Rebuild Vectorstore** in the sidebar.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 Retrieved sources ({len(msg['sources'])} chunks)", expanded=False):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f"**Chunk {i+1}** | Source: `{src['source']}` | Page: {src['page']}")
                        st.text(src["content"][:400] + "...")
                        st.divider()

    # Chat input
    user_input = st.chat_input("Ask a question about the research papers...")

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving ({retrieval_mode}) and generating answer..."):
                try:
                    # Build retriever with current settings
                    from rag.ingestor import load_and_chunk_pdfs
                    if st.session_state.chunks is None:
                        st.session_state.chunks = load_and_chunk_pdfs(chunk_size=chunk_size)

                    retriever = get_retriever_by_mode(
                        mode=retrieval_mode,
                        vectorstore=st.session_state.vectorstore,
                        chunks=st.session_state.chunks,
                        k=top_k
                    )

                    answer, docs = ask(
                        question=user_input,
                        retriever=retriever,
                        return_sources=True
                    )

                    st.write(answer)

                    # Format sources for display
                    sources = [
                        {
                            "source": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page", "?"),
                            "content": doc.page_content
                        }
                        for doc in docs
                    ]

                    with st.expander(f"📄 Retrieved sources ({len(docs)} chunks)", expanded=False):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}** | Source: `{src['source']}` | Page: {src['page']}")
                            st.text(src["content"][:400] + "...")
                            st.divider()

                    st.caption(f"🔍 Mode: `{retrieval_mode}` | Chunk size: `{chunk_size}` | Top-K: `{top_k}`")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")
