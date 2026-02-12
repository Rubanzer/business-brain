"""pgvector interface for embedding search and insertion."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import BusinessContext
from business_brain.ingestion.embeddings import embed_text


async def search(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[BusinessContext]:
    """Return the top-k most similar business context entries."""
    stmt = (
        select(BusinessContext)
        .order_by(BusinessContext.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def insert(
    session: AsyncSession,
    content: str,
    embedding: list[float],
    source: str = "manual",
) -> int:
    """Insert a new context entry with its embedding. Returns the row ID."""
    entry = BusinessContext(content=content, embedding=embedding, source=source)
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry.id


async def search_by_text(
    session: AsyncSession,
    text: str,
    top_k: int = 5,
) -> list[BusinessContext]:
    """Convenience: embed text and search for similar contexts."""
    embedding = embed_text(text)
    return await search(session, embedding, top_k=top_k)
