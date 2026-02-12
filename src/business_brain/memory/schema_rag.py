"""Retrieve relevant table schemas for a natural-language query."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.ingestion.embeddings import embed_text
from business_brain.memory import metadata_store, vector_store

logger = logging.getLogger(__name__)


async def retrieve_relevant_tables(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Given a natural language query, return the top-k most relevant table schemas.

    Strategy:
      1. Embed the query and search business_contexts for semantic matches.
      2. Keyword-match against metadata_store descriptions.
      3. Merge and return enriched schema info.
    """
    results: dict[str, dict] = {}

    # 1. Semantic search against business_contexts
    try:
        query_embedding = embed_text(query)
        context_hits = await vector_store.search(session, query_embedding, top_k=top_k)
        # Context hits give us hints about relevant tables via their content
        context_keywords = " ".join(hit.content for hit in context_hits).lower()
    except Exception:
        logger.exception("Vector search failed, falling back to keyword-only")
        context_keywords = ""

    # 2. Keyword + context matching against metadata entries
    all_entries = await metadata_store.get_all(session)
    query_lower = query.lower()

    for entry in all_entries:
        score = 0.0
        table_lower = entry.table_name.lower()
        desc_lower = (entry.description or "").lower()

        # Direct table name mention in query
        if table_lower in query_lower or table_lower.rstrip("s") in query_lower:
            score += 3.0

        # Keywords from query appear in description
        for word in query_lower.split():
            if len(word) > 2 and word in desc_lower:
                score += 1.0
            if len(word) > 2 and word in table_lower:
                score += 1.5

        # Table name mentioned in semantically similar contexts
        if table_lower in context_keywords:
            score += 2.0

        if score > 0:
            results[entry.table_name] = {
                "table_name": entry.table_name,
                "description": entry.description,
                "columns": entry.columns_metadata,
                "score": score,
            }

    # If no matches, return all entries as fallback
    if not results:
        for entry in all_entries[:top_k]:
            results[entry.table_name] = {
                "table_name": entry.table_name,
                "description": entry.description,
                "columns": entry.columns_metadata,
                "score": 0.0,
            }

    # Sort by score descending, return top_k
    ranked = sorted(results.values(), key=lambda r: r["score"], reverse=True)[:top_k]
    # Drop score from output
    return [{"table_name": r["table_name"], "description": r["description"], "columns": r["columns"]} for r in ranked]
