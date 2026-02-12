"""Shared Gemini embedding utility used across ingestion and memory layers."""

from google import genai

from config.settings import settings

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def embed_text(text: str) -> list[float]:
    """Return a 768-dim embedding from Gemini text-embedding-004."""
    client = _get_client()
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
    )
    return result.embeddings[0].values
