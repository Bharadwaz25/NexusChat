"""
config/config.py
Central configuration — API keys loaded from environment variables OR
Streamlit secrets (when deployed on Streamlit Cloud).
Never hardcode secrets here.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env for local dev


def _get(key: str, default: str = "") -> str:
    """Try env var first, then Streamlit secrets, then default."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ── LLM providers ────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _get("OPENAI_API_KEY")
GROQ_API_KEY:   str = _get("GROQ_API_KEY")
GEMINI_API_KEY: str = _get("GEMINI_API_KEY")

# ── Web-search ────────────────────────────────────────────────────────────────
SERPER_API_KEY: str = _get("SERPER_API_KEY")   # https://serper.dev

# ── RAG / Embeddings ──────────────────────────────────────────────────────────
CHUNK_SIZE:    int = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "100"))
TOP_K_CHUNKS:  int = int(_get("TOP_K_CHUNKS", "4"))

# ── App behaviour ─────────────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER:  str = _get("DEFAULT_LLM_PROVIDER", "groq")
DEFAULT_MODEL:         str = _get("DEFAULT_MODEL", "llama-3.1-8b-instant")
DEFAULT_RESPONSE_MODE: str = _get("DEFAULT_RESPONSE_MODE", "concise")
MAX_CHAT_HISTORY:      int = int(_get("MAX_CHAT_HISTORY", "20"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")
