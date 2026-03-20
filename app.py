"""
app.py
Main Streamlit UI for the RAG + Web-Search chatbot.

Features:
  • Multi-provider LLM  (Groq / OpenAI / Gemini)
  • RAG  — upload documents, embed them, retrieve relevant chunks at query time
  • Live web search  — Serper.dev integration
  • Response mode toggle  (Concise ↔ Detailed)
  • Persistent chat history in session state
"""

from __future__ import annotations

import logging
import sys
import os

import streamlit as st

# ── Path fix so sub-packages resolve correctly when run from project root ─────
sys.path.insert(0, os.path.dirname(__file__))

from config.config import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_RESPONSE_MODE,
    MAX_CHAT_HISTORY,
    SERPER_API_KEY,
    LOG_LEVEL,
)
from models.llm import get_llm_response
from utils.rag import VectorStore, extract_text_from_file, chunk_text, build_rag_context
from utils.web_search import search_and_format
from utils.prompt_builder import build_system_prompt

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusChat — AI Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label { color: #ccc9ff !important; font-family: 'Space Mono', monospace; }

    /* Main background */
    .stApp { background: #0d0d1a; color: #e4e4f0; }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #302b63, #6c63ff33);
        border-left: 3px solid #6c63ff;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .chat-assistant {
        background: linear-gradient(135deg, #0f3460, #00b4d820);
        border-left: 3px solid #00b4d8;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .chat-role {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 4px;
        opacity: 0.7;
    }

    /* Title */
    .nexus-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6c63ff, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .nexus-sub {
        color: #888; font-size: 0.9rem; margin-top: -8px; margin-bottom: 20px;
    }

    /* Mode badge */
    .mode-badge {
        display:inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
    }
    .mode-concise  { background:#6c63ff33; color:#a09bff; border:1px solid #6c63ff66; }
    .mode-detailed { background:#00b4d833; color:#5cd8f0; border:1px solid #00b4d866; }

    /* Status pills */
    .pill {
        display:inline-block; padding:2px 8px; border-radius:12px;
        font-size:0.72rem; margin:2px;
    }
    .pill-green  { background:#00b894aa; color:#fff; }
    .pill-orange { background:#e17055aa; color:#fff; }
    .pill-blue   { background:#0984e3aa; color:#fff; }

    /* Divider */
    hr { border-color: #ffffff15; }

    /* Input */
    .stChatInput > div { background: #1a1a2e !important; border: 1px solid #6c63ff44 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "vector_store": VectorStore(),
        "response_mode": DEFAULT_RESPONSE_MODE,
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "llm_model": DEFAULT_MODEL,
        "use_web_search": False,
        "use_rag": False,
        "docs_ingested": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 NexusChat")
    st.markdown("---")

    # ── LLM settings ─────────────────────────────────────────────────────────
    st.markdown("### ⚙️ LLM Settings")

    provider = st.selectbox(
        "Provider",
        options=["groq", "openai", "gemini"],
        index=["groq", "openai", "gemini"].index(st.session_state.llm_provider),
    )
    st.session_state.llm_provider = provider

    model_options = {
        "groq":   ["llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "openai": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    }
    selected_model = st.selectbox(
        "Model",
        options=model_options[provider],
        index=0,
    )
    st.session_state.llm_model = selected_model

    st.markdown("---")

    # ── Response mode ─────────────────────────────────────────────────────────
    st.markdown("### 💬 Response Mode")
    mode = st.radio(
        "Select mode",
        options=["concise", "detailed"],
        index=0 if st.session_state.response_mode == "concise" else 1,
        format_func=lambda x: "⚡ Concise — short & sharp" if x == "concise" else "📖 Detailed — in-depth",
        label_visibility="collapsed",
    )
    st.session_state.response_mode = mode

    st.markdown("---")

    # ── Web search ────────────────────────────────────────────────────────────
    st.markdown("### 🌐 Live Web Search")
    web_search_enabled = st.toggle(
        "Enable web search",
        value=st.session_state.use_web_search,
        help="Fetches current information via Serper.dev before answering.",
    )
    st.session_state.use_web_search = web_search_enabled

    if web_search_enabled and not SERPER_API_KEY:
        st.warning("⚠️ SERPER_API_KEY not set. Add it to your config or Streamlit secrets.")

    st.markdown("---")

    # ── RAG ───────────────────────────────────────────────────────────────────
    st.markdown("### 📂 Knowledge Base (RAG)")

    rag_enabled = st.toggle(
        "Enable RAG",
        value=st.session_state.use_rag,
        help="Retrieve relevant chunks from uploaded documents before answering.",
    )
    st.session_state.use_rag = rag_enabled

    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "md"],
        help="Supported: .txt  .md  .pdf  .docx",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.docs_ingested]
        if new_files:
            with st.spinner(f"Embedding {len(new_files)} document(s)…"):
                for ufile in new_files:
                    try:
                        text = extract_text_from_file(ufile)
                        chunks = chunk_text(text)
                        st.session_state.vector_store.add_documents(chunks, source=ufile.name)
                        st.session_state.docs_ingested.append(ufile.name)
                        logger.info("Ingested '%s' (%d chunks).", ufile.name, len(chunks))
                    except Exception as exc:
                        st.error(f"Failed to ingest {ufile.name}: {exc}")
            st.success(f"✅ {len(new_files)} document(s) embedded!")

    if st.session_state.docs_ingested:
        st.markdown("**Ingested documents:**")
        for doc in st.session_state.docs_ingested:
            st.markdown(f"<span class='pill pill-blue'>📄 {doc}</span>", unsafe_allow_html=True)

        if st.button("🗑️ Clear knowledge base", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.docs_ingested = []
            st.rerun()

    st.markdown("---")

    # ── Status strip ─────────────────────────────────────────────────────────
    st.markdown("**Active features:**")
    ws_class = "pill-green" if web_search_enabled and SERPER_API_KEY else "pill-orange"
    ws_label = "🌐 Web Search ON" if (web_search_enabled and SERPER_API_KEY) else "🌐 Web Search OFF"
    rag_class = "pill-green" if rag_enabled and not st.session_state.vector_store.is_empty else "pill-orange"
    rag_label = "📂 RAG ON" if (rag_enabled and not st.session_state.vector_store.is_empty) else "📂 RAG OFF"
    st.markdown(
        f"<span class='pill {ws_class}'>{ws_label}</span> "
        f"<span class='pill {rag_class}'>{rag_label}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🧹 Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main panel header
# ─────────────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("<div class='nexus-title'>NexusChat</div>", unsafe_allow_html=True)
    st.markdown("<div class='nexus-sub'>RAG · Web Search · Multi-LLM</div>", unsafe_allow_html=True)

with col_badge:
    badge_cls = "mode-concise" if st.session_state.response_mode == "concise" else "mode-detailed"
    badge_lbl = "⚡ CONCISE" if st.session_state.response_mode == "concise" else "📖 DETAILED"
    st.markdown(f"<br><span class='mode-badge {badge_cls}'>{badge_lbl}</span>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Chat history display
# ─────────────────────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown(
            """
            <div style='text-align:center; opacity:0.4; padding:40px 0;'>
                <div style='font-size:3rem'>🔮</div>
                <div style='font-family:Space Mono; font-size:0.9rem; margin-top:8px;'>
                    Start a conversation, upload documents, or enable web search.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-user'>"
                    f"<div class='chat-role'>You</div>"
                    f"{msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-assistant'>"
                    f"<div class='chat-role'>NexusChat</div>"
                    f"{msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# Chat input & response generation
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything…")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Trim history to max window
    if len(st.session_state.messages) > MAX_CHAT_HISTORY * 2:
        st.session_state.messages = st.session_state.messages[-(MAX_CHAT_HISTORY * 2):]

    # ── Gather context ────────────────────────────────────────────────────────
    rag_ctx = ""
    web_ctx = ""

    if st.session_state.use_rag and not st.session_state.vector_store.is_empty:
        try:
            rag_ctx = build_rag_context(st.session_state.vector_store, user_input)
        except Exception as exc:
            logger.error("RAG retrieval error: %s", exc)
            st.warning(f"⚠️ RAG retrieval failed: {exc}")

    if st.session_state.use_web_search and SERPER_API_KEY:
        with st.spinner("🌐 Searching the web…"):
            try:
                web_ctx = search_and_format(user_input)
            except Exception as exc:
                logger.error("Web search error: %s", exc)
                st.warning(f"⚠️ Web search failed: {exc}")

    # ── Build prompt & call LLM ───────────────────────────────────────────────
    system_prompt = build_system_prompt(
        mode=st.session_state.response_mode,
        rag_context=rag_ctx,
        web_context=web_ctx,
    )

    # Messages to send (exclude the one we just added; send full history)
    llm_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.spinner("🤔 Thinking…"):
        try:
            reply = get_llm_response(
                messages=llm_messages,
                system_prompt=system_prompt,
                provider=st.session_state.llm_provider,
                model=st.session_state.llm_model,
            )
        except Exception as exc:
            logger.error("LLM response error: %s", exc)
            reply = (
                f"⚠️ **Error generating response:** `{exc}`\n\n"
                "Please check your API key configuration in the sidebar or `config/config.py`."
            )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
