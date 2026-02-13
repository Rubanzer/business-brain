"""CRUD operations for schema metadata entries."""
from __future__ import annotations

import logging

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import MetadataEntry

logger = logging.getLogger(__name__)


async def get_all(session: AsyncSession) -> list[MetadataEntry]:
    result = await session.execute(select(MetadataEntry))
    return list(result.scalars().all())


async def get_by_table(session: AsyncSession, table_name: str) -> MetadataEntry | None:
    result = await session.execute(
        select(MetadataEntry).where(MetadataEntry.table_name == table_name)
    )
    return result.scalar_one_or_none()


async def upsert(
    session: AsyncSession,
    table_name: str,
    description: str,
    columns_metadata: list[dict] | None = None,
) -> MetadataEntry:
    existing = await get_by_table(session, table_name)
    if existing:
        existing.description = description
        existing.columns_metadata = columns_metadata
    else:
        existing = MetadataEntry(
            table_name=table_name,
            description=description,
            columns_metadata=columns_metadata,
        )
        session.add(existing)
    await session.commit()
    await session.refresh(existing)
    return existing


async def delete(session: AsyncSession, table_name: str) -> bool:
    entry = await get_by_table(session, table_name)
    if entry:
        await session.delete(entry)
        await session.commit()
        return True
    return False


async def validate_tables(session: AsyncSession) -> list[str]:
    """Remove metadata entries for tables that no longer exist in the database.

    Checks actual PostgreSQL tables against stored metadata and deletes stale entries.
    Returns list of table names that were cleaned up.
    """
    try:
        result = await session.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        actual_tables = {row[0] for row in result.fetchall()}
    except Exception:
        logger.exception("Failed to query pg_tables for validation")
        return []

    all_entries = await get_all(session)
    removed: list[str] = []

    for entry in all_entries:
        if entry.table_name not in actual_tables:
            logger.info("Cleaning up stale metadata for table: %s", entry.table_name)
            await session.delete(entry)
            removed.append(entry.table_name)

    if removed:
        await session.commit()

    return removed
