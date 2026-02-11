"""Ingest natural-language business context into the vector store."""

from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import BusinessContext


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
    # TODO: call embedding API (OpenAI / Claude) to get vector
    embedding = None  # placeholder

    entry = BusinessContext(content=text, embedding=embedding, source=source)
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    print(f"[context_ingestor] Stored context id={entry.id} source={source}")
    return entry.id
