"""Retrieve relevant table schemas for a natural-language query."""

from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.memory import metadata_store


async def retrieve_relevant_tables(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Given a natural language query, return the top-k most relevant table schemas.

    TODO: embed the query, search vector store for matching metadata,
          and return enriched schema descriptions.
    """
    # Placeholder: return all metadata entries
    entries = await metadata_store.get_all(session)
    results = []
    for entry in entries[:top_k]:
        results.append({
            "table_name": entry.table_name,
            "description": entry.description,
            "columns": entry.columns_metadata,
        })
    return results
