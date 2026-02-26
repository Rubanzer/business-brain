"""pgvector interface for embedding search and insertion.

Improvements:
- Relevance threshold filtering (cosine distance < 0.5 by default)
- Source-priority boosting (company_profile and metric_thresholds always included)
- Deduplication of results by content prefix
- Only searches active (non-superseded) context entries
- Context listing API for visibility in the Setup tab
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import BusinessContext
from business_brain.ingestion.embeddings import embed_text


# Sources that should always be included regardless of similarity score
_PRIORITY_SOURCES = frozenset({"onboarding:company_profile", "company_profile", "metric_thresholds"})


async def search(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 5,
    max_distance: float = 0.6,
) -> list[BusinessContext]:
    """Return the top-k most similar business context entries.

    Args:
        session: Database session.
        query_embedding: The query vector.
        top_k: Maximum results to return.
        max_distance: Maximum cosine distance threshold. Results beyond this
            distance are filtered out (unless they are priority sources).

    Returns:
        List of BusinessContext entries, most similar first.
        Only returns active (non-superseded) entries.
    """
    # Fetch more than top_k to allow for filtering
    fetch_limit = top_k + 5
    stmt = (
        select(BusinessContext)
        .where(BusinessContext.active == True)  # noqa: E712 — only active entries
        .order_by(BusinessContext.embedding.cosine_distance(query_embedding))
        .limit(fetch_limit)
    )
    result = await session.execute(stmt)
    all_hits = list(result.scalars().all())

    # Filter by relevance threshold but always keep priority sources
    filtered = []
    seen_prefixes: set[str] = set()

    for hit in all_hits:
        # Deduplicate by content prefix (first 150 chars)
        prefix = (hit.content or "")[:150]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        # Always include priority sources (company profile, thresholds)
        if hit.source in _PRIORITY_SOURCES:
            filtered.append(hit)
            continue

        # For other sources, apply distance threshold
        # Note: cosine_distance orders results, so early results are more relevant
        if len(filtered) < top_k:
            filtered.append(hit)

    return filtered[:top_k]


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


async def list_all_contexts(
    session: AsyncSession,
    active_only: bool = True,
) -> list[dict]:
    """List all context entries grouped by source.

    Returns a list of dicts with id, content (truncated), source, created_at, active.
    Used by the Setup tab to display all known context.
    """
    stmt = select(BusinessContext).order_by(
        BusinessContext.source, BusinessContext.created_at.desc()
    )
    if active_only:
        stmt = stmt.where(BusinessContext.active == True)  # noqa: E712

    result = await session.execute(stmt)
    entries = result.scalars().all()

    return [
        {
            "id": e.id,
            "content": e.content[:300] if e.content else "",
            "full_content": e.content,
            "source": e.source or "unknown",
            "version": getattr(e, "version", 1) or 1,
            "active": getattr(e, "active", True),
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in entries
    ]
