"""Shared Gemini embedding utility used across ingestion and memory layers."""
from __future__ import annotations

from google import genai

from config.settings import settings

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def embed_text(text: str) -> list[float]:
    """Return a 3072-dim embedding from Gemini gemini-embedding-001."""
    client = _get_client()
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
    )
    return result.embeddings[0].values
