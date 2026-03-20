"""
rag_pipeline.py
───────────────
Core RAG (Retrieval-Augmented Generation) engine.

Pipeline stages
───────────────
1. Embed chunks with SentenceTransformers → float32 numpy arrays
2. Index embeddings in FAISS (exact L2 search – fast enough for <50 k chunks)
3. On query: embed question → FAISS nearest-neighbour search → retrieve top-k chunks
4. Build a grounded prompt and call Google Gemini

Design notes
────────────
- RAGPipeline is intentionally stateless between calls once built.
- Each public method raises descriptive exceptions so the UI layer can
  display clean error messages without tracebacks.
- No global state; safe to create multiple instances (e.g. multi-PDF support).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from groq import Groq
import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Bundles everything returned by a single RAG query."""
    answer: str
    source_chunks: List[str]
    source_indices: List[int]
    distances: List[float]          # L2 distances (lower = more similar)


@dataclass
class PipelineStats:
    """Lightweight diagnostics exposed to the UI."""
    num_chunks: int = 0
    embedding_dim: int = 0
    model_name: str = ""
    llm_model: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt template
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise, helpful assistant that answers questions **strictly** based on
the context excerpts provided below. Follow these rules without exception:

1. If the answer is clearly present in the context, answer accurately and concisely.
2. If the context only partially covers the question, answer what you can and
   explicitly state what is missing.
3. If the context does not contain enough information to answer, reply:
   "I could not find an answer to that question in the provided document."
4. Do NOT fabricate facts, statistics, names, or dates.
5. Do NOT answer from general knowledge – context only.
6. Cite which excerpt number(s) support each claim when possible, e.g. [Excerpt 2].
7. Use plain, clear language. Avoid bullet-point overuse.
"""

def build_rag_prompt(question: str, chunks: List[str]) -> str:
    """
    Construct the final prompt that is sent to the LLM.

    The prompt injects numbered context excerpts so the model can cite them,
    then appends the user question with a clear instruction to stay grounded.
    """
    context_block = "\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"━━━ CONTEXT EXCERPTS ━━━\n"
        f"{context_block}\n\n"
        f"━━━ QUESTION ━━━\n"
        f"{question}\n\n"
        f"━━━ ANSWER (based only on the excerpts above) ━━━\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding helper
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformers.

    Keeps the model loaded in memory across multiple calls so we pay the
    model-load cost only once per session.
    """

    def __init__(self, model_name: str = cfg.EMBEDDING_MODEL):
        self._model_name = model_name
        self._model = None   # lazy-load on first use

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model '%s' …", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded (dim=%d).", self.dim)

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of strings into float32 embeddings.

        Args:
            texts:      List of strings to embed.
            batch_size: Sentences processed per forward pass.

        Returns:
            np.ndarray of shape (len(texts), dim), dtype float32.
        """
        self._load()
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-normalise → cosine ~ L2 on unit sphere
        )
        return embeddings.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  FAISS vector store
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Wraps a FAISS flat index (exact search) with the chunk texts.

    For up to ~100 k chunks, a flat L2 index is fast enough and requires
    no training. Swap to IndexIVFFlat for larger corpora.
    """

    def __init__(self, dim: int):
        import faiss
        # IndexFlatIP = inner-product (equivalent to cosine when vecs are unit-normalised)
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: List[str] = []
        self._dim = dim

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks:     Parallel list of text strings.
            embeddings: np.ndarray shape (n, dim), float32, unit-normalised.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )
        if embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._dim}, "
                f"got {embeddings.shape[1]}."
            )
        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info("FAISS index now holds %d vectors.", self.size)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Retrieve the top-k most similar chunks.

        Args:
            query_embedding: 1-D float32 array of length dim.
            top_k:           Number of results to return.

        Returns:
            (chunks, indices, distances) – all parallel lists of length top_k.
        """
        if self.size == 0:
            raise RuntimeError("The vector store is empty. Index documents first.")

        top_k = min(top_k, self.size)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query, top_k)

        # Flatten from (1, k) → (k,)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        retrieved_chunks = [self._chunks[i] for i in indices]
        return retrieved_chunks, indices, distances


# ─────────────────────────────────────────────────────────────────────────────
#  LLM client
# ─────────────────────────────────────────────────────────────────────────────

class GroqLLM:
    def __init__(self, model_name: str, api_key: str):
        self._client = Groq(api_key=api_key)
        self._model_name = model_name
        logger.info("Groq client ready — model: '%s'", model_name)

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=1024,
            )
        except Exception as exc:
            raise RuntimeError(f"Groq API call failed: {exc}") from exc

        answer = response.choices[0].message.content.strip()

        if not answer:
            raise RuntimeError("Groq returned empty response.")

        return answer

# ─────────────────────────────────────────────────────────────────────────────
#  Public RAG pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.

    Typical usage::

        pipeline = RAGPipeline()
        pipeline.index_documents(chunks)          # once per upload
        result = pipeline.query("What is X?")     # many times

    Multi-PDF support
    ─────────────────
    Call `index_documents` once per PDF (or call it with all chunks combined).
    The FAISS index accumulates vectors from all calls until `reset()` is called.
    """

    def __init__(self):
        cfg.validate()                                # fail fast on missing API key
        self._embedder = EmbeddingModel(cfg.EMBEDDING_MODEL)
        self._store: Optional[FAISSVectorStore] = None
        self._llm = GroqLLM(cfg.LLM_MODEL, cfg.GROQ_API_KEY)
        self._all_chunks: List[str] = []             # flat list for multi-PDF

    # ── Indexing ──────────────────────────────────────────────────

    def index_documents(self, chunks: List[str]) -> PipelineStats:
        """
        Embed `chunks` and add them to the FAISS vector store.

        Safe to call multiple times – each call appends to the existing index
        (multi-PDF support). Call `reset()` to start fresh.

        Args:
            chunks: Pre-split text chunks from `utils.split_text_into_chunks`.

        Returns:
            PipelineStats with current index diagnostics.

        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("No chunks provided to index.")

        logger.info("Embedding %d chunks …", len(chunks))
        embeddings = self._embedder.encode(chunks)

        if self._store is None:
            self._store = FAISSVectorStore(dim=self._embedder.dim)

        self._store.add(chunks, embeddings)
        self._all_chunks.extend(chunks)

        return PipelineStats(
            num_chunks=self._store.size,
            embedding_dim=self._embedder.dim,
            model_name=cfg.EMBEDDING_MODEL,
            llm_model=cfg.LLM_MODEL,
        )

    # ── Querying ─────────────────────────────────────────────────

    def query(self, question: str, top_k: int = cfg.TOP_K_CHUNKS) -> RetrievalResult:
        """
        Run a full RAG query against the indexed documents.

        Steps
        ─────
        1. Embed the question.
        2. Retrieve top-k chunks from FAISS.
        3. Build a grounded prompt.
        4. Call Gemini and return the result.

        Args:
            question: User's natural-language question.
            top_k:    Number of context chunks to retrieve.

        Returns:
            RetrievalResult with answer + source evidence.

        Raises:
            RuntimeError: If no documents have been indexed yet.
            ValueError:   If the question is empty.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        if self._store is None or self._store.size == 0:
            raise RuntimeError(
                "No documents have been indexed. "
                "Please upload and process a PDF first."
            )

        logger.info("Querying: %r (top_k=%d)", question[:80], top_k)

        # Step 1 – embed question
        q_embedding = self._embedder.encode([question])[0]

        # Step 2 – retrieve
        chunks, indices, distances = self._store.search(q_embedding, top_k=top_k)

        # Step 3 – build prompt
        prompt = build_rag_prompt(question, chunks)

        # Step 4 – generate
        answer = self._llm.generate(prompt)

        logger.info("Answer generated (%d chars).", len(answer))
        return RetrievalResult(
            answer=answer,
            source_chunks=chunks,
            source_indices=indices,
            distances=distances,
        )

    # ── Utilities ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the vector store and chunk list. Useful for 'clear all PDFs'."""
        self._store = None
        self._all_chunks = []
        logger.info("RAG pipeline reset.")

    @property
    def is_ready(self) -> bool:
        """True once at least one document has been indexed."""
        return self._store is not None and self._store.size > 0

    @property
    def stats(self) -> Optional[PipelineStats]:
        if not self.is_ready:
            return None
        return PipelineStats(
            num_chunks=self._store.size,
            embedding_dim=self._embedder.dim,
            model_name=cfg.EMBEDDING_MODEL,
            llm_model=cfg.GEMINI_MODEL,
        )
