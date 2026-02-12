"""Nightly DB introspection — discover tables and auto-generate descriptions."""
from __future__ import annotations

import logging

from google import genai
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.memory import metadata_store
from config.settings import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


async def crawl_schema(session: AsyncSession, schema: str = "public") -> list[dict]:
    """Introspect information_schema and return table/column metadata.

    Returns:
        List of dicts: [{table_name, columns: [{name, type}]}]
    """
    query = text(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
        """
    )
    result = await session.execute(query, {"schema": schema})
    rows = result.fetchall()

    tables: dict[str, list[dict]] = {}
    for table_name, column_name, data_type in rows:
        tables.setdefault(table_name, []).append({"name": column_name, "type": data_type})

    return [{"table_name": t, "columns": cols} for t, cols in tables.items()]


async def auto_describe(session: AsyncSession) -> None:
    """Crawl schema and upsert metadata entries with LLM-generated descriptions."""
    tables = await crawl_schema(session)
    client = _get_client()

    for table_info in tables:
        table_name = table_info["table_name"]
        columns = table_info["columns"]

        col_summary = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        prompt = (
            f"Write a concise one-sentence description of a database table named "
            f"'{table_name}' with columns: {col_summary}. "
            f"Focus on what business data this table likely stores."
        )

        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            description = response.text.strip()
        except Exception:
            logger.exception("Failed to generate description for %s", table_name)
            description = f"Table {table_name} with {len(columns)} columns."

        await metadata_store.upsert(
            session,
            table_name=table_name,
            description=description,
            columns_metadata=columns,
        )
        logger.info("Upserted metadata for table: %s", table_name)
