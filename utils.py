"""
utils.py
────────
Pure helper functions with no side-effects.
Nothing here imports from rag_pipeline or app – keeps the dependency
graph acyclic and every function unit-testable in isolation.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)



#  PDF text extraction

def extract_text_from_pdf(file_bytes: bytes, filename: str = "upload.pdf") -> str:
    """
    Extract plain text from raw PDF bytes using PyMuPDF (fitz).

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename:   Original filename (used only for log messages).

    Returns:
        A single string containing all extracted text, pages separated by
        a form-feed character so callers can detect page boundaries if needed.

    Raises:
        ValueError: If the PDF is encrypted, empty, or yields no extractable text.
        RuntimeError: If PyMuPDF fails to open the document.
    """
    if not file_bytes:
        raise ValueError(f"'{filename}' appears to be empty (0 bytes).")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"Could not open '{filename}' as a PDF: {exc}") from exc

    if doc.is_encrypted:
        raise ValueError(
            f"'{filename}' is password-protected. "
            "Please decrypt it before uploading."
        )

    pages_text: List[str] = []
    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")          # plain text mode
            text = _clean_page_text(text)
            if text:
                pages_text.append(text)
        except Exception as exc:                   # pragma: no cover
            logger.warning("Skipping page %d of '%s': %s", page_num, filename, exc)

    doc.close()

    if not pages_text:
        raise ValueError(
            f"'{filename}' contains no extractable text. "
            "It may be a scanned image PDF. Consider running OCR first."
        )

    full_text = "\f".join(pages_text)             # \f = form-feed page separator
    logger.info(
        "Extracted %d pages / %d chars from '%s'.",
        len(pages_text), len(full_text), filename,
    )
    return full_text


def _clean_page_text(raw: str) -> str:
    """
    Light cleanup on raw page text:
    - Collapse runs of whitespace / blank lines
    - Strip leading/trailing space
    Does NOT remove hyphens or do aggressive normalisation –
    we want to preserve technical terms, bullet symbols, etc.
    """
    # Replace non-breaking spaces and other unicode spaces with regular space
    cleaned = re.sub(r"[^\S\n]+", " ", raw)
    # Collapse 3+ consecutive newlines → double newline (paragraph break)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()



#  Text chunking

def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Split `text` into overlapping character-level chunks.

    Strategy
    ────────
    1. Prefer splitting on paragraph boundaries (\n\n).
    2. Fall back to sentence boundaries (. ! ?).
    3. Hard-cut if no natural boundary is found within `chunk_size` chars.

    The overlap ensures that answers that span a chunk boundary are not lost.

    Args:
        text:          Full document text.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        List of non-empty chunk strings.

    Raises:
        ValueError: If text is empty after stripping whitespace.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot chunk empty text.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    # --- Use LangChain's RecursiveCharacterTextSplitter for production quality
    # We import here so utils stays import-cheap when only PDF utils are needed.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Hierarchy: paragraph → newline → sentence-end → space → char
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)
    # Filter out chunks that are just whitespace / very short noise
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

    logger.info("Split text into %d chunks (size=%d, overlap=%d).",
                len(chunks), chunk_size, chunk_overlap)
    return chunks



#  Misc helpers

def compute_file_hash(file_bytes: bytes) -> str:
    """Return a short SHA-256 hex digest for cache-keying purposes."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def format_sources(chunks: List[str], max_preview: int = 300) -> List[str]:
    """
    Truncate source chunks to a preview length for display in the UI.

    Args:
        chunks:      Raw retrieved chunk strings.
        max_preview: Maximum characters to show per chunk.

    Returns:
        List of truncated strings with an ellipsis appended if trimmed.
    """
    previews = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > max_preview:
            chunk = chunk[:max_preview].rsplit(" ", 1)[0] + " …"
        previews.append(chunk)
    return previews


def timeit(label: str):
    """
    Context manager for quick wall-clock timing in log output.

    Usage::

        with timeit("embedding 42 chunks"):
            embed(chunks)
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            logger.debug("%s took %.3fs", label, elapsed)

    return _ctx()


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Return text capped at max_chars with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def sanitize_question(question: str) -> str:
    """
    Basic sanitisation of user questions before sending to the LLM:
    - Strip leading/trailing whitespace.
    - Collapse internal whitespace runs.
    - Enforce a maximum length to avoid prompt-injection via huge inputs.
    """
    MAX_QUESTION_LEN = 1000
    question = re.sub(r"\s+", " ", question).strip()
    if len(question) > MAX_QUESTION_LEN:
        question = question[:MAX_QUESTION_LEN]
        logger.warning("User question was truncated to %d chars.", MAX_QUESTION_LEN)
    return question


def is_valid_pdf_bytes(data: bytes) -> bool:
    """Quick magic-number check – PDFs start with %PDF-"""
    return data[:5] == b"%PDF-"