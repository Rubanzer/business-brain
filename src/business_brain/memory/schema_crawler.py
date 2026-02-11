"""Nightly DB introspection — discover tables and auto-generate descriptions."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


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
    """Crawl schema and upsert metadata entries with auto-generated descriptions.

    TODO: call LLM to generate human-readable descriptions for each table.
    """
    tables = await crawl_schema(session)
    for table_info in tables:
        # TODO: generate description via LLM, then call metadata_store.upsert()
        print(f"[schema_crawler] Found table: {table_info['table_name']} "
              f"({len(table_info['columns'])} columns)")
