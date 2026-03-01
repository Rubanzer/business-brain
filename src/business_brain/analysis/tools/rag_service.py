"""Multi-store RAG service with built-in embedding.

Provides semantic search across multiple vector stores:
- business_context: existing BusinessContext table
- analysis_history: new AnalysisHistoryEmbedding table

Wraps existing embed_text() from ingestion/embeddings.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.models import AnalysisHistoryEmbedding
from business_brain.db.models import BusinessContext
from business_brain.ingestion.embeddings import embed_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helpers (async wrappers around sync embed_text)
# ---------------------------------------------------------------------------


async def _embed_text(text: str) -> list[float]:
    """Async wrapper around the sync Gemini embed_text()."""
    return await asyncio.to_thread(embed_text, text)


async def _embed_batch(texts: list[str], batch_size: int = 10) -> list[list[float]]:
    """Embed multiple texts with batched concurrency."""
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = await asyncio.gather(*[_embed_text(t) for t in batch])
        results.extend(embeddings)
    return results


# ---------------------------------------------------------------------------
# Store definitions
# ---------------------------------------------------------------------------

# Map store names to (model_class, content_column, embedding_column)
_STORES = {
    "business_context": (BusinessContext, "content", "embedding"),
    "analysis_history": (AnalysisHistoryEmbedding, "content", "embedding"),
}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def search(
    session: AsyncSession,
    store: str,
    query: str,
    top_k: int = 5,
    max_distance: float = 0.6,
) -> list[dict[str, Any]]:
    """Semantic search in a single store."""
    if store not in _STORES:
        raise ValueError(f"Unknown store: {store}. Available: {list(_STORES.keys())}")

    model_cls, content_col, embedding_col = _STORES[store]
    query_embedding = await _embed_text(query)

    embedding_column = getattr(model_cls, embedding_col)
    content_column = getattr(model_cls, content_col)

    # Use pgvector cosine distance operator
    stmt = (
        select(
            model_cls,
            embedding_column.cosine_distance(query_embedding).label("distance"),
        )
        .where(embedding_column.isnot(None))
        .order_by("distance")
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    hits = []
    for row_obj, distance in rows:
        if distance > max_distance:
            continue
        hit: dict[str, Any] = {
            "id": str(getattr(row_obj, "id", None)),
            "content": getattr(row_obj, content_col, ""),
            "distance": float(distance),
            "similarity": float(1.0 - distance),
            "source": getattr(row_obj, "source", None),
        }
        # Include metadata if available
        meta_col = getattr(row_obj, "metadata_", None) or getattr(row_obj, "metadata", None)
        if meta_col:
            hit["metadata"] = meta_col
        hits.append(hit)

    return hits


async def search_multi(
    session: AsyncSession,
    stores: list[str],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search across multiple stores, merge and rank by similarity."""
    all_hits: list[dict[str, Any]] = []
    for store_name in stores:
        try:
            hits = await search(session, store_name, query, top_k=top_k)
            for h in hits:
                h["store"] = store_name
            all_hits.extend(hits)
        except Exception:
            logger.warning("Search failed for store %s", store_name, exc_info=True)

    # Sort by distance (ascending = most similar first)
    all_hits.sort(key=lambda h: h["distance"])
    return all_hits[:top_k]


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


async def insert(
    session: AsyncSession,
    store: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    source: str | None = None,
) -> str:
    """Insert a new entry into a store with auto-embedding. Returns the new ID."""
    if store not in _STORES:
        raise ValueError(f"Unknown store: {store}. Available: {list(_STORES.keys())}")

    model_cls, _, _ = _STORES[store]
    embedding = await _embed_text(content)

    kwargs: dict[str, Any] = {
        "content": content,
        "embedding": embedding,
    }
    if source is not None:
        kwargs["source"] = source

    if store == "analysis_history" and metadata is not None:
        kwargs["metadata_"] = metadata
    elif store == "business_context":
        # BusinessContext doesn't have a metadata column
        pass

    obj = model_cls(**kwargs)
    session.add(obj)
    await session.flush()
    return str(obj.id)


async def insert_batch(
    session: AsyncSession,
    store: str,
    items: list[dict[str, Any]],
) -> list[str]:
    """Batch insert with batched embedding. Each item: {content, metadata?, source?}."""
    if not items:
        return []

    texts = [item["content"] for item in items]
    embeddings = await _embed_batch(texts)

    model_cls, _, _ = _STORES[store]
    ids = []
    for item, emb in zip(items, embeddings):
        kwargs: dict[str, Any] = {
            "content": item["content"],
            "embedding": emb,
        }
        if "source" in item:
            kwargs["source"] = item["source"]
        if store == "analysis_history" and "metadata" in item:
            kwargs["metadata_"] = item["metadata"]

        obj = model_cls(**kwargs)
        session.add(obj)
        await session.flush()
        ids.append(str(obj.id))

    return ids
