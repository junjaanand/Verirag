"""
VeriRAG — Streamlit UI
Self-Correcting RAG Pipeline with Confidence-Calibrated Responses
"""

import streamlit as st
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DOCUMENTS_DIR

# Page config
st.set_page_config(
    page_title="VeriRAG - Self-Correcting RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .confidence-high {
        background: linear-gradient(135deg, #065f46, #059669);
        color: white; padding: 8px 16px; border-radius: 20px;
        font-weight: 600; display: inline-block;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #92400e, #d97706);
        color: white; padding: 8px 16px; border-radius: 20px;
        font-weight: 600; display: inline-block;
    }
    .confidence-low {
        background: linear-gradient(135deg, #991b1b, #dc2626);
        color: white; padding: 8px 16px; border-radius: 20px;
        font-weight: 600; display: inline-block;
    }
    .citation-box {
        background: #1a1a2e; border-left: 3px solid #6366f1;
        padding: 10px 15px; margin: 5px 0;
        border-radius: 0 8px 8px 0; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Lazy-load pipeline components
@st.cache_resource
def load_pipeline():
    from ingestion.pipeline import process_single_document
    from indexing.chunker import chunk_document
    from indexing.embedder import embed_texts
    from indexing.vector_store import add_chunks, get_document_count, reset_collection
    from indexing.bm25_index import bm25_index
    from retrieval.hybrid_search import hybrid_search
    from retrieval.reranker import rerank
    from correction.correction_pipeline import run_correction
    from generation.generator import generate_response
    return {
        "process": process_single_document,
        "chunk": chunk_document,
        "embed": embed_texts,
        "add": add_chunks,
        "count": get_document_count,
        "reset": reset_collection,
        "bm25": bm25_index,
        "search": hybrid_search,
        "rerank": rerank,
        "correct": run_correction,
        "generate": generate_response,
    }


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📁 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 Ingest Documents", type="primary", use_container_width=True):
        p = load_pipeline()
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for file in uploaded_files:
            fp = DOCUMENTS_DIR / file.name
            with open(fp, "wb") as f:
                f.write(file.getbuffer())
            saved.append(str(fp))

        with st.spinner("Processing..."):
            all_chunks = []
            for fp in saved:
                try:
                    doc = p["process"](fp)
                    chunks = p["chunk"](doc)
                    all_chunks.extend(chunks)
                    st.text(f"✅ {Path(fp).name}: {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"❌ {Path(fp).name}: {e}")

            if all_chunks:
                embeddings = p["embed"]([c.text for c in all_chunks])
                p["add"](all_chunks, embeddings)
                p["bm25"].build(all_chunks)
                p["bm25"].save()
                st.success(f"✅ {len(saved)} docs → {len(all_chunks)} chunks indexed!")

    st.divider()
    if st.button("📂 Load Test Documents", use_container_width=True):
        p = load_pipeline()
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        test_files = sorted(DOCUMENTS_DIR.glob("*.txt"))
        if not test_files:
            st.warning("Run `python create_test_data.py` first.")
        else:
            with st.spinner("Ingesting test documents..."):
                all_chunks = []
                for fp in test_files:
                    try:
                        doc = p["process"](str(fp))
                        chunks = p["chunk"](doc)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        st.error(f"❌ {fp.name}: {e}")
                if all_chunks:
                    embeddings = p["embed"]([c.text for c in all_chunks])
                    p["add"](all_chunks, embeddings)
                    p["bm25"].build(all_chunks)
                    p["bm25"].save()
                    st.success(f"✅ {len(test_files)} docs → {len(all_chunks)} chunks indexed!")

    st.divider()
    st.markdown("### 📊 Index Stats")
    try:
        st.metric("Chunks Indexed", load_pipeline()["count"]())
    except Exception:
        st.metric("Chunks Indexed", 0)

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        try:
            load_pipeline()["reset"]()
            st.success("Cleared!")
        except Exception:
            pass

    st.divider()
    show_corrections = st.toggle("Show correction details", value=True)
    show_citations = st.toggle("Show source citations", value=True)


# ============================================================
# MAIN
# ============================================================
st.markdown('<p class="main-header">🛡️ VeriRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Self-Correcting RAG — Thinks Before It Speaks</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        p = load_pipeline()
        try:
            doc_count = p["count"]()
        except Exception:
            doc_count = 0

        if doc_count == 0:
            txt = "⚠️ No documents indexed. Upload docs or click 'Load Test Documents'."
            st.markdown(txt)
            st.session_state.messages.append({"role": "assistant", "content": txt})
        else:
            with st.spinner("🧠 Thinking..."):
                t0 = time.time()
                status = st.status("Processing query...", expanded=True)

                status.write("🔍 Searching knowledge base...")
                candidates = p["search"](prompt)

                status.write("🏆 Re-ranking results...")
                top_chunks = p["rerank"](prompt, candidates)

                status.write("🔄 Running self-correction...")
                correction = p["correct"](
                    query=prompt, initial_chunks=top_chunks,
                    search_fn=lambda q: p["search"](q),
                    rerank_fn=lambda q, c: p["rerank"](q, c),
                )

                status.write("✨ Generating response...")
                response = p["generate"](prompt, correction)
                elapsed = time.time() - t0
                status.update(label=f"Done in {elapsed:.1f}s", state="complete")

            # Confidence badge
            lvl = response.confidence_level.lower()
            emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}[lvl]
            st.markdown(
                f'<span class="confidence-{lvl}">{emoji} Confidence: {response.confidence:.0%} ({response.confidence_level})</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(response.answer)

            if response.clarification_needed:
                st.warning(f"❓ **Clarification Needed:** {response.clarification_needed}")

            if response.contradictions:
                with st.expander(f"⚠️ Contradictions ({len(response.contradictions)})"):
                    for c in response.contradictions:
                        st.markdown(f"**{c['source_a']}** vs **{c['source_b']}**: {c['explanation']}")

            if show_corrections and response.corrections_applied:
                with st.expander(f"🔄 Corrections ({len(response.corrections_applied)})"):
                    for c in response.corrections_applied:
                        st.markdown(f"- Attempt {c['attempt']}: **{c['strategy']}**")

            if show_citations and response.citations:
                with st.expander(f"📚 Sources ({len(response.citations)})"):
                    for cit in response.citations:
                        st.markdown(
                            f'<div class="citation-box"><strong>{cit.source}</strong> (Page {cit.page}) — '
                            f'Relevance: {cit.relevance:.0%}<br><em>{cit.chunk_text[:150]}...</em></div>',
                            unsafe_allow_html=True,
                        )

            with st.expander("📊 Confidence Breakdown"):
                bd = correction.confidence.breakdown
                cols = st.columns(4)
                cols[0].metric("Retrieval", f"{bd.get('retrieval_quality', 0):.0%}")
                cols[1].metric("Sufficiency", f"{bd.get('sufficiency_score', 0):.0%}")
                cols[2].metric("Agreement", f"{bd.get('agreement_score', 0):.0%}")
                cols[3].metric("OCR Quality", f"{bd.get('ocr_quality', 0):.0%}")

            st.session_state.messages.append({"role": "assistant", "content": response.answer})
