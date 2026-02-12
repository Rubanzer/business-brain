"""Ingest natural-language business context into the vector store."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import BusinessContext
from business_brain.ingestion.embeddings import embed_text

logger = logging.getLogger(__name__)


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
    embedding = embed_text(text)

    entry = BusinessContext(content=text, embedding=embedding, source=source)
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    logger.info("Stored context id=%d source=%s", entry.id, source)
    return entry.id
