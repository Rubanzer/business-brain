"""Shared embedding utility — supports Gemini and OpenAI embedding models.

Auto-detects provider from settings. Public API unchanged: embed_text(str) -> list[float].
"""

from __future__ import annotations

import logging

from config.settings import settings

logger = logging.getLogger(__name__)

_provider = None


class _GeminiEmbedder:
    def __init__(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)

    def embed(self, text: str) -> list[float]:
        result = self._client.models.embed_content(
            model=settings.embedding_model,
            contents=text,
        )
        return result.embeddings[0].values


class _OpenAIEmbedder:
    def __init__(self) -> None:
        import openai

        self._client = openai.OpenAI(api_key=settings.openai_api_key)

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        vec = response.data[0].embedding
        if len(vec) != 3072:
            logger.warning(
                "Embedding dimension mismatch: got %d, expected 3072. "
                "Use text-embedding-3-large for compatible dimensions.",
                len(vec),
            )
        return vec


def _get_embedding_provider():
    global _provider
    if _provider is not None:
        return _provider

    explicit = (settings.embedding_provider or settings.llm_provider or "").lower()

    if explicit == "openai" and settings.openai_api_key:
        _provider = _OpenAIEmbedder()
    elif explicit == "gemini" and settings.gemini_api_key:
        _provider = _GeminiEmbedder()
    elif settings.gemini_api_key:
        _provider = _GeminiEmbedder()
    elif settings.openai_api_key:
        _provider = _OpenAIEmbedder()
    else:
        raise RuntimeError(
            "No embedding provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY."
        )

    logger.info("Embedding provider initialized: %s", type(_provider).__name__)
    return _provider


def embed_text(text: str) -> list[float]:
    """Return an embedding vector. Provider auto-detected from settings."""
    return _get_embedding_provider().embed(text)
