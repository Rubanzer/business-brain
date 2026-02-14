"""Sync engine — orchestrates syncing all data sources and manages schedules."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.v3_models import DataSource

logger = logging.getLogger(__name__)


async def sync_all_due(session: AsyncSession) -> list[dict]:
    """Sync all data sources that are due for a refresh.

    A source is due when:
    - active == True
    - sync_frequency_minutes > 0
    - last_sync_at is None OR (now - last_sync_at) >= sync_frequency_minutes

    Returns:
        List of sync results.
    """
    result = await session.execute(
        select(DataSource).where(
            DataSource.active == True,  # noqa: E712
            DataSource.sync_frequency_minutes > 0,
        )
    )
    sources = list(result.scalars().all())
    results = []

    now = datetime.now(timezone.utc)

    for source in sources:
        # Check if due
        if source.last_sync_at:
            from datetime import timedelta
            elapsed = now - source.last_sync_at.replace(tzinfo=timezone.utc) if source.last_sync_at.tzinfo is None else now - source.last_sync_at
            if elapsed < timedelta(minutes=source.sync_frequency_minutes):
                continue

        try:
            sync_result = await sync_source(session, source)
            results.append({"source_id": source.id, "name": source.name, **sync_result})
        except Exception as exc:
            logger.exception("Failed to sync source %s", source.name)
            source.last_sync_status = "error"
            source.last_sync_error = str(exc)
            source.last_sync_at = now
            await session.commit()
            results.append({"source_id": source.id, "name": source.name, "error": str(exc)})

    return results


async def sync_source(session: AsyncSession, source: DataSource) -> dict:
    """Sync a single data source based on its type.

    Returns:
        Dict with sync results.
    """
    if source.source_type == "google_sheet":
        from business_brain.ingestion.sheets_sync import sync_google_sheet
        return await sync_google_sheet(session, source)

    elif source.source_type == "api":
        return await _sync_api_source(session, source)

    else:
        return {"status": "skipped", "reason": f"Unknown source type: {source.source_type}"}


async def _sync_api_source(session: AsyncSession, source: DataSource) -> dict:
    """Sync an API data source with change detection."""
    from business_brain.ingestion.api_sync import sync_api_source
    return await sync_api_source(session, source)


async def get_all_sources(session: AsyncSession) -> list[DataSource]:
    """Get all data sources."""
    result = await session.execute(select(DataSource).order_by(DataSource.created_at.desc()))
    return list(result.scalars().all())


async def get_source(session: AsyncSession, source_id: str) -> DataSource | None:
    """Get a single data source by ID."""
    result = await session.execute(select(DataSource).where(DataSource.id == source_id))
    return result.scalar_one_or_none()


async def pause_source(session: AsyncSession, source_id: str) -> bool:
    """Pause a data source's auto-sync."""
    source = await get_source(session, source_id)
    if not source:
        return False
    source.active = False
    await session.commit()
    return True


async def resume_source(session: AsyncSession, source_id: str) -> bool:
    """Resume a data source's auto-sync."""
    source = await get_source(session, source_id)
    if not source:
        return False
    source.active = True
    await session.commit()
    return True


async def delete_source(session: AsyncSession, source_id: str) -> bool:
    """Delete a data source."""
    source = await get_source(session, source_id)
    if not source:
        return False
    await session.delete(source)
    await session.commit()
    return True


async def update_source(session: AsyncSession, source_id: str, updates: dict) -> DataSource | None:
    """Update a data source's configuration."""
    source = await get_source(session, source_id)
    if not source:
        return None

    if "name" in updates:
        source.name = updates["name"]
    if "sync_frequency_minutes" in updates:
        source.sync_frequency_minutes = updates["sync_frequency_minutes"]
    if "active" in updates:
        source.active = updates["active"]

    await session.commit()
    return source
