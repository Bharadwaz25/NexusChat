"""
models/embeddings.py
Embedding model wrappers used by the RAG pipeline.
Provides a single `embed_texts(texts) -> List[List[float]]` interface
backed by sentence-transformers (local, free) with an optional
OpenAI text-embedding fallback.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# ── Default local model ───────────────────────────────────────────────────────
# "all-MiniLM-L6-v2" is small (~80 MB), fast, and good enough for RAG demos.
_DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"
_sbert_instance = None  # lazy singleton


def _get_sbert():
    """Lazily load the SentenceTransformer model (cached after first call)."""
    global _sbert_instance
    if _sbert_instance is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _sbert_instance = SentenceTransformer(_DEFAULT_SBERT_MODEL)
            logger.info("Loaded SentenceTransformer model: %s", _DEFAULT_SBERT_MODEL)
        except ImportError as exc:
            logger.error("sentence-transformers not installed: %s", exc)
            raise
    return _sbert_instance


def embed_texts_local(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings using the local SentenceTransformer model.

    Returns a list of float vectors (one per input string).
    """
    try:
        model = _get_sbert()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()
    except Exception as exc:
        logger.error("Local embedding failed: %s", exc)
        raise


def embed_texts_openai(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Embed a list of strings using the OpenAI Embeddings API.
    Requires OPENAI_API_KEY to be set in config.
    """
    try:
        from openai import OpenAI  # type: ignore
        from config.config import OPENAI_API_KEY

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as exc:
        logger.error("OpenAI embedding failed: %s", exc)
        raise


def embed_texts(texts: List[str], provider: str = "local") -> List[List[float]]:
    """
    Public entry-point for generating embeddings.

    Args:
        texts:    Strings to embed.
        provider: "local" (sentence-transformers) or "openai".

    Returns:
        A list of float vectors.
    """
    if not texts:
        return []
    if provider == "openai":
        return embed_texts_openai(texts)
    return embed_texts_local(texts)


def embed_query(query: str, provider: str = "local") -> List[float]:
    """
    Convenience wrapper that embeds a single query string.
    Returns a single float vector.
    """
    return embed_texts([query], provider=provider)[0]
