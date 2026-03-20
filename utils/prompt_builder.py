"""
utils/prompt_builder.py
Assembles system prompts dynamically based on:
  • Response mode  (concise / detailed)
  • RAG context    (document chunks, if any)
  • Web search     (live results, if any)
"""

from __future__ import annotations

_BASE_SYSTEM = """You are a helpful, knowledgeable AI assistant.
Answer the user's question accurately and honestly.
If you are given document context or web search results, use them to ground your answer."""

_CONCISE_INSTRUCTIONS = """
RESPONSE MODE: CONCISE
- Reply in 2–4 sentences maximum.
- Prioritise the single most important insight.
- No bullet lists, no headers, no padding.
- If the answer is naturally short, keep it short."""

_DETAILED_INSTRUCTIONS = """
RESPONSE MODE: DETAILED
- Provide a comprehensive, well-structured answer.
- Use headers or bullet points when they aid clarity.
- Explain reasoning, include examples where relevant.
- Aim for depth over brevity."""


def build_system_prompt(
    mode: str = "concise",
    rag_context: str = "",
    web_context: str = "",
) -> str:
    """
    Build the system prompt string sent to the LLM.

    Args:
        mode:        "concise" or "detailed".
        rag_context: Pre-formatted RAG context block (may be empty).
        web_context: Pre-formatted web-search block (may be empty).

    Returns:
        A complete system prompt string.
    """
    parts = [_BASE_SYSTEM]

    if mode == "detailed":
        parts.append(_DETAILED_INSTRUCTIONS)
    else:
        parts.append(_CONCISE_INSTRUCTIONS)

    if rag_context:
        parts.append(
            "\nUse the following excerpts from the user's documents to answer:\n"
            + rag_context
        )

    if web_context:
        parts.append(
            "\nUse the following live web search results to answer with current information:\n"
            + web_context
        )

    return "\n".join(parts)
