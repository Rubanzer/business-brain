"""CRUD operations for schema metadata entries."""
from __future__ import annotations

import logging

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import MetadataEntry

logger = logging.getLogger(__name__)


async def get_all(session: AsyncSession) -> list[MetadataEntry]:
    try:
        result = await session.execute(select(MetadataEntry))
        return list(result.scalars().all())
    except Exception:
        logger.exception("Failed to fetch all metadata entries")
        await session.rollback()
        return []


async def get_filtered(
    session: AsyncSession,
    table_names: list[str] | None = None,
) -> list[MetadataEntry]:
    """Fetch metadata entries, optionally filtering to a specific set of tables.

    If table_names is None, returns all entries (same as get_all).
    If table_names is an empty list, returns an empty list.
    """
    if table_names is None:
        return await get_all(session)
    if not table_names:
        return []
    try:
        result = await session.execute(
            select(MetadataEntry).where(MetadataEntry.table_name.in_(table_names))
        )
        return list(result.scalars().all())
    except Exception:
        logger.exception("Failed to fetch filtered metadata entries")
        await session.rollback()
        return []


async def get_by_table(session: AsyncSession, table_name: str) -> MetadataEntry | None:
    try:
        result = await session.execute(
            select(MetadataEntry).where(MetadataEntry.table_name == table_name)
        )
        return result.scalar_one_or_none()
    except Exception:
        logger.exception("Failed to fetch metadata for table: %s", table_name)
        await session.rollback()
        return None


async def upsert(
    session: AsyncSession,
    table_name: str,
    description: str,
    columns_metadata: list[dict] | None = None,
    uploaded_by: str | None = None,
    uploaded_by_role: str | None = None,
    business_notes: str | None = None,
) -> MetadataEntry:
    try:
        existing = await get_by_table(session, table_name)
        if existing:
            existing.description = description
            existing.columns_metadata = columns_metadata
            if business_notes is not None:
                existing.business_notes = business_notes
            # Don't overwrite original uploader on update
        else:
            existing = MetadataEntry(
                table_name=table_name,
                description=description,
                columns_metadata=columns_metadata,
                uploaded_by=uploaded_by,
                uploaded_by_role=uploaded_by_role,
                business_notes=business_notes,
            )
            session.add(existing)
        await session.commit()
        await session.refresh(existing)
        return existing
    except Exception:
        logger.exception("Failed to upsert metadata for table: %s", table_name)
        await session.rollback()
        raise


async def update_context(
    session: AsyncSession,
    table_name: str,
    description: str | None = None,
    business_notes: str | None = None,
    column_descriptions: list[dict] | None = None,
) -> MetadataEntry | None:
    """Update user-editable context fields for a table.

    column_descriptions: [{name, description}] — merges into existing
    columns_metadata by column name, only updating the description field.
    """
    try:
        existing = await get_by_table(session, table_name)
        if not existing:
            return None

        if description is not None:
            existing.description = description
        if business_notes is not None:
            existing.business_notes = business_notes

        # Merge column descriptions into existing columns_metadata
        if column_descriptions and existing.columns_metadata:
            # Build lookup: column_name -> user description
            user_descs = {cd["name"]: cd["description"] for cd in column_descriptions if cd.get("name")}
            # Merge into existing column metadata (must create new list for SQLAlchemy change detection)
            updated_cols = []
            for col in existing.columns_metadata:
                col_copy = dict(col)
                if col_copy.get("name") in user_descs:
                    col_copy["description"] = user_descs[col_copy["name"]]
                updated_cols.append(col_copy)
            existing.columns_metadata = updated_cols

        await session.commit()
        await session.refresh(existing)
        return existing
    except Exception:
        logger.exception("Failed to update context for table: %s", table_name)
        await session.rollback()
        raise


async def delete(session: AsyncSession, table_name: str) -> bool:
    try:
        entry = await get_by_table(session, table_name)
        if entry:
            await session.delete(entry)
            await session.commit()
            return True
        return False
    except Exception:
        logger.exception("Failed to delete metadata for table: %s", table_name)
        await session.rollback()
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
        await session.rollback()
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
