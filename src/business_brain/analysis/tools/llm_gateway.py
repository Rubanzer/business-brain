"""Centralized LLM gateway for the entire application.

Provides structured LLM access with:
- Multi-provider support (Gemini, OpenAI, Anthropic) via llm_providers
- Rate limiting (asyncio.Semaphore)
- Exponential backoff retry (3 attempts)
- Prompt-level caching (md5 hash)
- Robust JSON extraction
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from business_brain.analysis.tools.llm_providers import get_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_semaphore = asyncio.Semaphore(5)

# ---------------------------------------------------------------------------
# Prompt cache (md5 → response text)
# ---------------------------------------------------------------------------

_cache: dict[str, str] = {}
_CACHE_MAX = 500


def _cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


# ---------------------------------------------------------------------------
# JSON extraction (from cognitive/query_router.py)
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> dict | None:
    """Robustly extract a JSON object from an LLM response."""
    text = raw.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            block = part.strip()
            for tag in ("json", "JSON"):
                if block.startswith(tag):
                    block = block[len(tag) :].strip()
            try:
                return json.loads(block)
            except (json.JSONDecodeError, ValueError):
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Core call with retry + rate limit + cache
# ---------------------------------------------------------------------------


async def _call_llm(prompt: str, use_cache: bool = True) -> str:
    """Call the configured LLM provider with semaphore, retry, and cache."""
    key = _cache_key(prompt)
    if use_cache and key in _cache:
        return _cache[key]

    async with _semaphore:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                provider = get_provider()
                text = await asyncio.to_thread(provider.generate, prompt)

                # Cache the result
                if use_cache:
                    if len(_cache) >= _CACHE_MAX:
                        # Evict oldest ~25%
                        keys = list(_cache.keys())
                        for k in keys[: len(keys) // 4]:
                            _cache.pop(k, None)
                    _cache[key] = text

                return text
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "LLM call failed (attempt %d/%d, provider=%s): %s — retrying in %ds",
                    attempt + 1,
                    3,
                    get_provider().name(),
                    str(exc)[:200],
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(f"LLM call failed after 3 attempts: {last_error}")


# ---------------------------------------------------------------------------
# Synchronous call (for sync contexts like insight_formatter, python_analyst)
# ---------------------------------------------------------------------------


def generate_sync(prompt: str) -> str:
    """Synchronous LLM call. No cache, no retry — use for sync call sites."""
    return get_provider().generate(prompt)


# ---------------------------------------------------------------------------
# Public API — typed wrappers
# ---------------------------------------------------------------------------


async def reason(prompt: str) -> str:
    """Free-form reasoning. Returns raw text."""
    return await _call_llm(prompt)


async def classify(prompt: str, categories: list[str]) -> dict[str, Any]:
    """Classify input into one of the given categories.

    Returns: {"category": str, "confidence": float, "reasoning": str}
    """
    full_prompt = (
        f"{prompt}\n\n"
        f"Classify into exactly one of these categories: {categories}\n"
        f"Return JSON: {{\"category\": \"...\", \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    )
    raw = await _call_llm(full_prompt)
    parsed = _extract_json(raw)
    if parsed and "category" in parsed:
        return parsed
    # Fallback: try to match category from raw text
    for cat in categories:
        if cat.lower() in raw.lower():
            return {"category": cat, "confidence": 0.5, "reasoning": raw[:200]}
    return {"category": categories[0], "confidence": 0.1, "reasoning": raw[:200]}


async def extract(prompt: str) -> dict[str, Any]:
    """Extract structured data from text. Returns parsed JSON dict."""
    full_prompt = f"{prompt}\n\nReturn your answer as a JSON object."
    raw = await _call_llm(full_prompt)
    parsed = _extract_json(raw)
    return parsed or {"raw": raw}


async def generate_narrative(prompt: str) -> str:
    """Generate a natural-language narrative. Returns text."""
    return await _call_llm(prompt, use_cache=False)
