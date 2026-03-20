from __future__ import annotations

import logging
import sys
from typing import List

import streamlit as st

# Local imports
 
from config import cfg
from rag_pipeline import RAGPipeline, RetrievalResult
from utils import (
    compute_file_hash,
    extract_text_from_pdf,
    format_sources,
    is_valid_pdf_bytes,
    sanitize_question,
    split_text_into_chunks,
)


#  Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


#  Page configuration  (must be the first Streamlit call)

st.set_page_config(
    page_title=cfg.APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)



#  Custom CSS – minimal but polished

st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── Header ─────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3b82f6;
    }
    .app-header h1 { margin: 0; font-size: 1.7rem; font-weight: 600; letter-spacing: -0.5px; }
    .app-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; color: #94a3b8; }

    /* ── Chat bubbles ────────────────────────── */
    .chat-user {
        background: #eff6ff;
        border-left: 3px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 8px;
        margin: 0.4rem 0;
    }
    .chat-assistant {
        background: #f8fafc;
        border-left: 3px solid #10b981;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 8px;
        margin: 0.4rem 0;
    }

    /* ── Source chunk expander ───────────────── */
    .source-chip {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #475569;
        margin-bottom: 0.5rem;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ── Stat cards (sidebar) ────────────────── */
    .stat-card {
        background: #1e293b;
        color: #f1f5f9;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
    }
    .stat-label { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }

    /* ── Hide default streamlit chrome ──────── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)



#  Session-state initialisation

def _init_session() -> None:
    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = RAGPipeline()
        except EnvironmentError as exc:
            st.error(f"⚠️ Configuration error: {exc}")
            st.stop()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_ids" not in st.session_state:
        st.session_state.processed_ids = set()


_init_session()
pipeline: RAGPipeline = st.session_state.pipeline



#  Sidebar – PDF upload & controls

with st.sidebar:
    st.markdown("### 📄 Document Manager")
    st.caption("Upload one or more PDFs to begin chatting.")

    uploaded_files = st.file_uploader(
        label="Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Scanned/image-only PDFs won't work – text must be selectable.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if compute_file_hash(f.getvalue()) not in st.session_state.processed_ids
        ]

        if new_files:
            progress = st.progress(0, text="Processing PDFs …")
            for i, file in enumerate(new_files, start=1):
                file_bytes = file.getvalue()
                file_id = compute_file_hash(file_bytes)
                progress.progress(i / len(new_files), text=f"Processing: {file.name}")

                # Validation
                if not is_valid_pdf_bytes(file_bytes):
                    st.warning(f"⚠️ '{file.name}' doesn't look like a PDF – skipping.")
                    continue

                try:
                    # Stage 1: extract text
                    raw_text = extract_text_from_pdf(file_bytes, filename=file.name)
                    # Stage 2: chunk
                    chunks = split_text_into_chunks(
                        raw_text,
                        chunk_size=cfg.CHUNK_SIZE,
                        chunk_overlap=cfg.CHUNK_OVERLAP,
                    )
                    # Stage 3: embed + index
                    stats = pipeline.index_documents(chunks)
                    st.session_state.processed_ids.add(file_id)
                    st.success(f"✅ **{file.name}** – {len(chunks)} chunks indexed.")
                except (ValueError, RuntimeError) as exc:
                    st.error(f"❌ **{file.name}**: {exc}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error processing **{file.name}**: {exc}")
                    logger.exception("Unhandled error processing %s", file.name)

            progress.empty()

        else:
            st.info("All uploaded files are already indexed.")

    st.divider()

    # Pipeline stats 
    if pipeline.is_ready and pipeline.stats:
        s = pipeline.stats
        st.markdown("**Index stats**")
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Chunks indexed</div>
                <strong>{s.num_chunks:,}</strong>
            </div>
            <div class="stat-card">
                <div class="stat-label">Embedding model</div>
                <strong>{s.model_name}</strong>
            </div>
            <div class="stat-card">
                <div class="stat-label">LLM</div>
                <strong>{s.llm_model}</strong>
            </div>
            <div class="stat-card">
                <div class="stat-label">Vector dim</div>
                <strong>{s.embedding_dim}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

    # Controls 
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("♻️ Reset All", use_container_width=True, type="secondary"):
            pipeline.reset()
            st.session_state.messages = []
            st.session_state.processed_ids = set()
            st.rerun()

    st.caption(
        "**Reset All** clears the vector index and chat history. "
        "You'll need to re-upload your PDFs."
    )



#  Main area

st.markdown(
    f"""
    <div class="app-header">
        <h1>📄 {cfg.APP_TITLE}</h1>
        <p>{cfg.APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Empty-state prompt 

if not pipeline.is_ready:
    st.info(
        "👈 **Get started:** Upload one or more PDF files using the sidebar. "
        "Once processed, you can ask questions here.",
        icon="💡",
    )

# Render chat history 

for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(msg["content"])

        # Render source chunks stored alongside assistant messages
        if role == "assistant" and msg.get("sources"):
            with st.expander("📌 Source excerpts used to generate this answer"):
                for j, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f'<div class="source-chip">'
                        f'<strong>[Excerpt {j}]</strong><br>{src}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input 

if question := st.chat_input(
    placeholder="Ask a question about your PDF(s) …",
    disabled=not pipeline.is_ready,
):
    question = sanitize_question(question)
    if not question:
        st.warning("Please enter a non-empty question.")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documents and generating answer …"):
            try:
                result: RetrievalResult = pipeline.query(question, top_k=cfg.TOP_K_CHUNKS)
                answer = result.answer
                source_previews = format_sources(result.source_chunks, max_preview=400)

                st.markdown(answer)

                with st.expander("📌 Source excerpts used to generate this answer"):
                    for j, src in enumerate(source_previews, start=1):
                        st.markdown(
                            f'<div class="source-chip">'
                            f'<strong>[Excerpt {j}]</strong><br>{src}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Persist in chat history(with sources for re-render)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_previews,
                    }
                )

            except ValueError as exc:
                err = f"Invalid request: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {err}"})

            except RuntimeError as exc:
                err = str(exc)
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {err}"})

            except Exception as exc:
                logger.exception("Unhandled error during query.")
                err = "An unexpected error occurred. Please check the logs."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {err}"})