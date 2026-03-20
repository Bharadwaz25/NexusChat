"""
utils/web_search.py
Real-time web search integration using the Serper.dev API.

Usage:
    from utils.web_search import web_search
    results = web_search("latest news on AI regulation")
"""

from __future__ import annotations

import logging
import requests
from typing import List, Dict, Optional

from config.config import SERPER_API_KEY

logger = logging.getLogger(__name__)

SERPER_ENDPOINT = "https://google.serper.dev/search"


def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a live Google search via Serper.dev.

    Args:
        query:       The search query string.
        num_results: Maximum number of organic results to return (1–10).

    Returns:
        List of dicts with keys: title, link, snippet.
        Returns an empty list on error.
    """
    if not SERPER_API_KEY:
        logger.warning("SERPER_API_KEY is not set — web search is disabled.")
        return []

    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}
        response = requests.post(SERPER_ENDPOINT, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        organic = data.get("organic", [])

        results = []
        for item in organic[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        logger.info("Web search for '%s' returned %d results.", query, len(results))
        return results

    except requests.exceptions.Timeout:
        logger.error("Web search timed out for query: %s", query)
        return []
    except requests.exceptions.HTTPError as exc:
        logger.error("Serper API HTTP error: %s", exc)
        return []
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return []


def format_search_results(results: List[Dict[str, str]]) -> str:
    """
    Convert raw search results into a formatted string suitable for
    injection into an LLM system prompt.
    """
    if not results:
        return ""

    lines = ["=== Live Web Search Results ==="]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    URL: {r['link']}\n    {r['snippet']}")
    lines.append("=== End of Search Results ===")
    return "\n\n".join(lines)


def search_and_format(query: str, num_results: int = 5) -> str:
    """Convenience wrapper: search and return a formatted context string."""
    results = web_search(query, num_results=num_results)
    return format_search_results(results)
