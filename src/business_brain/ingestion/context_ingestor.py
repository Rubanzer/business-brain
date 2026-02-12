"""Ingest natural-language business context into the vector store."""

import logging

from google import genai
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import BusinessContext
from config.settings import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _embed(text: str) -> list[float]:
    """Get a 768-dim embedding from Gemini text-embedding-004."""
    client = _get_client()
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
    )
    return result.embeddings[0].values


async def ingest_context(
    text: str,
    session: AsyncSession,
    source: str = "manual",
) -> int:
    """Embed *text* and store it as a BusinessContext row.

    Args:
        text: Natural-language business context.
        session: Async SQLAlchemy session.
        source: Label for where the context came from.

    Returns:
        The ID of the newly created row.
    """
    embedding = _embed(text)

    entry = BusinessContext(content=text, embedding=embedding, source=source)
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    logger.info("Stored context id=%d source=%s", entry.id, source)
    return entry.id
