"""Nightly DB introspection — discover tables and auto-generate descriptions."""
from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.tools.llm_gateway import reason as _llm_reason
from business_brain.memory import metadata_store

logger = logging.getLogger(__name__)


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
    """Crawl schema and upsert metadata entries with LLM-generated descriptions.

    Generates both a table-level description and per-column descriptions so
    that downstream SQL generation can distinguish similarly-typed columns
    (e.g. ``quantity`` vs ``yield``).
    """
    import json as _json

    tables = await crawl_schema(session)

    for table_info in tables:
        table_name = table_info["table_name"]
        columns = table_info["columns"]

        col_summary = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        prompt = (
            f"For a database table named '{table_name}' with columns: {col_summary}, "
            f"return a JSON object with:\n"
            f"1. \"description\": a concise one-sentence description of what business data this table stores\n"
            f"2. \"columns\": an object mapping each column name to a short description of what it represents\n"
            f"Return ONLY valid JSON, no markdown fences."
        )

        try:
            raw = await _llm_reason(prompt)
            # Strip potential markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            meta = _json.loads(raw)
            description = meta.get("description", f"Table {table_name}")
            col_descs = meta.get("columns", {})
            for col in columns:
                col["description"] = col_descs.get(col["name"], "")
        except (_json.JSONDecodeError, AttributeError, TypeError):
            logger.warning("Could not parse column descriptions for %s, using plain description", table_name)
            try:
                description = raw.strip() if raw else f"Table {table_name} with {len(columns)} columns."
            except Exception:
                description = f"Table {table_name} with {len(columns)} columns."
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
