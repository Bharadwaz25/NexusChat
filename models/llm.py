"""
models/llm.py
Unified LLM interface supporting OpenAI, Groq, and Google Gemini.
Each provider is wrapped in a thin adapter that exposes the same
`chat(messages, system_prompt) -> str` signature.
"""

from __future__ import annotations

import logging
from typing import List, Dict

from config.config import (
    OPENAI_API_KEY,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_openai_messages(
    messages: List[Dict[str, str]],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """Prepend system prompt as a system message for OpenAI-compatible APIs."""
    result = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    result.extend(messages)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Provider adapters
# ─────────────────────────────────────────────────────────────────────────────

def chat_openai(
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    model: str = "gpt-3.5-turbo",
) -> str:
    """Call the OpenAI Chat Completions API."""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=_build_openai_messages(messages, system_prompt),
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("OpenAI error: %s", exc)
        raise


def chat_groq(
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    model: str = "llama3-8b-8192",
) -> str:
    """Call the Groq Chat Completions API (OpenAI-compatible)."""
    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=_build_openai_messages(messages, system_prompt),
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Groq error: %s", exc)
        raise


def chat_gemini(
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    model: str = "gemini-1.5-flash",
) -> str:
    """Call the Google Gemini generative API."""
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt or None,
        )
        # Convert role names: 'assistant' → 'model'
        history = []
        for m in messages[:-1]:
            role = "model" if m["role"] == "assistant" else m["role"]
            history.append({"role": role, "parts": [m["content"]]})

        chat = gemini_model.start_chat(history=history)
        response = chat.send_message(messages[-1]["content"])
        return response.text.strip()
    except Exception as exc:
        logger.error("Gemini error: %s", exc)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry-point
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_MAP = {
    "openai": chat_openai,
    "groq": chat_groq,
    "gemini": chat_gemini,
}


def get_llm_response(
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Route a chat request to the appropriate LLM provider.

    Args:
        messages:      List of {"role": ..., "content": ...} dicts.
        system_prompt: Optional instruction injected as a system message.
        provider:      "openai" | "groq" | "gemini"
        model:         Provider-specific model name.

    Returns:
        The assistant's reply as a plain string.
    """
    provider = provider.lower()
    if provider not in PROVIDER_MAP:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {list(PROVIDER_MAP)}")
    try:
        return PROVIDER_MAP[provider](messages, system_prompt, model)
    except Exception as exc:
        logger.error("LLM request failed (%s / %s): %s", provider, model, exc)
        raise
