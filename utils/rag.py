"""
utils/rag.py
RAG (Retrieval-Augmented Generation) pipeline utilities.

Responsibilities:
  • Parse uploaded documents (PDF, TXT, DOCX, Markdown).
  • Split text into overlapping chunks.
  • Embed chunks and store in an in-memory FAISS index.
  • Retrieve the top-k most relevant chunks for a query.
"""

from __future__ import annotations

import io
import logging
from typing import List, Tuple

import numpy as np

from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS
from models.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(file_obj) -> str:
    """
    Extract plain text from an uploaded Streamlit file object.
    Supports: .txt, .md, .pdf, .docx
    """
    name: str = file_obj.name.lower()
    raw: bytes = file_obj.read()

    try:
        if name.endswith((".txt", ".md")):
            return raw.decode("utf-8", errors="replace")

        elif name.endswith(".pdf"):
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(io.BytesIO(raw))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

        elif name.endswith(".docx"):
            import docx  # type: ignore

            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)

        else:
            logger.warning("Unsupported file type: %s — trying UTF-8 decode.", name)
            return raw.decode("utf-8", errors="replace")

    except Exception as exc:
        logger.error("Text extraction failed for %s: %s", name, exc)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split *text* into overlapping character-level chunks.

    Args:
        text:       Input string.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    logger.debug("Chunked text into %d pieces (size=%d, overlap=%d).", len(chunks), chunk_size, overlap)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# In-memory vector store (FAISS)
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Lightweight wrapper around a flat FAISS index.
    Stores (chunk_text, source_label) pairs alongside their embeddings.
    """

    def __init__(self) -> None:
        self._index = None          # faiss.IndexFlatIP
        self._chunks: List[str] = []
        self._sources: List[str] = []
        self._dim: int = 0

    # ── Build ────────────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[str], source: str = "doc") -> None:
        """Embed *chunks* and add them to the FAISS index."""
        if not chunks:
            return
        try:
            import faiss  # type: ignore

            vectors = embed_texts(chunks)
            matrix = np.array(vectors, dtype="float32")

            # Normalise for cosine similarity via inner product
            faiss.normalize_L2(matrix)

            if self._index is None:
                self._dim = matrix.shape[1]
                self._index = faiss.IndexFlatIP(self._dim)

            self._index.add(matrix)
            self._chunks.extend(chunks)
            self._sources.extend([source] * len(chunks))
            logger.info("Added %d chunks from '%s'. Total: %d.", len(chunks), source, len(self._chunks))

        except Exception as exc:
            logger.error("VectorStore.add_documents failed: %s", exc)
            raise

    # ── Query ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Tuple[str, str, float]]:
        """
        Find the *top_k* most relevant chunks for *query*.

        Returns:
            List of (chunk_text, source_label, similarity_score).
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        try:
            import faiss  # type: ignore

            q_vec = np.array([embed_query(query)], dtype="float32")
            faiss.normalize_L2(q_vec)

            scores, indices = self._index.search(q_vec, min(top_k, self._index.ntotal))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self._chunks[idx], self._sources[idx], float(score)))
            return results

        except Exception as exc:
            logger.error("VectorStore.search failed: %s", exc)
            raise

    # ── Helpers ──────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Wipe all stored vectors and metadata."""
        self._index = None
        self._chunks = []
        self._sources = []
        self._dim = 0
        logger.info("VectorStore cleared.")

    @property
    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    @property
    def document_count(self) -> int:
        return len(self._chunks)


# ─────────────────────────────────────────────────────────────────────────────
# High-level helper used by app.py
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_context(vector_store: VectorStore, query: str, top_k: int = TOP_K_CHUNKS) -> str:
    """
    Retrieve the most relevant chunks and format them as a context block
    ready to be injected into the LLM system prompt.
    """
    results = vector_store.search(query, top_k=top_k)
    if not results:
        return ""

    lines = ["=== Relevant Document Context ==="]
    for i, (chunk, source, score) in enumerate(results, 1):
        lines.append(f"[{i}] (source: {source}, relevance: {score:.2f})\n{chunk}")
    lines.append("=== End of Context ===")
    return "\n\n".join(lines)
