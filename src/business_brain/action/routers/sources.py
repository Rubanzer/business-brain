"""Data Sources & Sync routes."""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.action.dependencies import run_discovery_background
from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sources"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class GoogleSheetRequest(BaseModel):
    sheet_url: str
    name: Optional[str] = None
    tab_name: Optional[str] = None
    table_name: Optional[str] = None
    sync_frequency_minutes: int = 5


class ApiSourceRequest(BaseModel):
    name: str
    api_url: str
    table_name: str
    headers: Optional[dict] = None
    params: Optional[dict] = None
    sync_frequency_minutes: int = 0


class SourceUpdateRequest(BaseModel):
    name: Optional[str] = None
    sync_frequency_minutes: Optional[int] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/sources")
async def list_sources(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all connected data sources."""
    from business_brain.ingestion.sync_engine import get_all_sources

    sources = await get_all_sources(session)
    return [
        {
            "id": s.id,
            "name": s.name,
            "source_type": s.source_type,
            "table_name": s.table_name,
            "sync_frequency_minutes": s.sync_frequency_minutes,
            "last_sync_at": s.last_sync_at.isoformat() if s.last_sync_at else None,
            "last_sync_status": s.last_sync_status,
            "last_sync_error": s.last_sync_error,
            "rows_total": s.rows_total,
            "active": s.active,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sources
    ]


@router.post("/sources/google-sheet")
async def connect_google_sheet_endpoint(
    req: GoogleSheetRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Connect a Google Sheet as a data source."""
    from business_brain.ingestion.sheets_sync import connect_google_sheet

    try:
        source = await connect_google_sheet(
            session,
            sheet_url=req.sheet_url,
            name=req.name,
            tab_name=req.tab_name,
            table_name=req.table_name,
            sync_frequency_minutes=req.sync_frequency_minutes,
        )
        background_tasks.add_task(run_discovery_background, f"sheet:{source.table_name}")
        return {
            "status": "connected",
            "source_id": source.id,
            "table_name": source.table_name,
            "rows": source.rows_total,
        }
    except Exception as exc:
        logger.exception("Failed to connect Google Sheet")
        return {"error": str(exc)}


@router.post("/sources/api")
async def connect_api_source(
    req: ApiSourceRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Connect an API endpoint as a data source."""
    from business_brain.db.v3_models import DataSource
    from business_brain.ingestion.api_puller import pull_api

    try:
        rows = await pull_api(req.api_url, session, req.table_name, headers=req.headers, params=req.params)
        source = DataSource(
            name=req.name,
            source_type="api",
            connection_config={"api_url": req.api_url, "headers": req.headers, "params": req.params},
            table_name=req.table_name,
            sync_frequency_minutes=req.sync_frequency_minutes,
            rows_total=rows,
            active=True,
        )
        from datetime import datetime, timezone
        source.last_sync_at = datetime.now(timezone.utc)
        source.last_sync_status = "success"
        session.add(source)
        await session.commit()
        await session.refresh(source)
        return {"status": "connected", "source_id": source.id, "rows": rows}
    except Exception as exc:
        logger.exception("Failed to connect API source")
        return {"error": str(exc)}


@router.post("/sources/{source_id}/sync")
async def sync_source_endpoint(
    source_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Manually trigger sync for a data source."""
    from business_brain.ingestion.sync_engine import get_source, sync_source

    source = await get_source(session, source_id)
    if not source:
        return {"error": "Source not found"}

    try:
        result = await sync_source(session, source)
        background_tasks.add_task(run_discovery_background, f"sync:{source.table_name}")
        return {"status": "synced", **result}
    except Exception as exc:
        logger.exception("Sync failed for source %s", source_id)
        return {"error": str(exc)}


@router.put("/sources/{source_id}")
async def update_source_endpoint(
    source_id: str,
    req: SourceUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update a data source's configuration."""
    from business_brain.ingestion.sync_engine import update_source

    updates = req.model_dump(exclude_none=True)
    source = await update_source(session, source_id, updates)
    if not source:
        return {"error": "Source not found"}
    return {"status": "updated", "source_id": source.id}


@router.delete("/sources/{source_id}")
async def delete_source_endpoint(
    source_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Delete a data source and cascade-drop its table + all dependent data."""
    from business_brain.ingestion.sync_engine import get_source, delete_source
    from business_brain.action.routers.data import drop_table_cascade

    source = await get_source(session, source_id)
    if not source:
        return {"error": "Source not found"}

    table_name = source.table_name

    # Delete the source record first
    await delete_source(session, source_id)

    # Cascade-drop the table and all dependent metadata/analysis data
    cascade_result = await drop_table_cascade(table_name, session)

    return {
        "status": "deleted",
        "source_id": source_id,
        "table": table_name,
        "removed": cascade_result.get("removed", {}),
    }


@router.get("/sources/{source_id}/changes")
async def get_source_changes(
    source_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get recent change log for a data source."""
    from sqlalchemy import select
    from business_brain.db.v3_models import DataChangeLog

    result = await session.execute(
        select(DataChangeLog)
        .where(DataChangeLog.data_source_id == source_id)
        .order_by(DataChangeLog.detected_at.desc())
        .limit(50)
    )
    changes = list(result.scalars().all())
    return [
        {
            "id": c.id,
            "change_type": c.change_type,
            "table_name": c.table_name,
            "row_identifier": c.row_identifier,
            "column_name": c.column_name,
            "old_value": c.old_value,
            "new_value": c.new_value,
            "detected_at": c.detected_at.isoformat() if c.detected_at else None,
        }
        for c in changes
    ]


@router.post("/sources/{source_id}/pause")
async def pause_source_endpoint(source_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Pause auto-sync for a data source."""
    from business_brain.ingestion.sync_engine import pause_source
    ok = await pause_source(session, source_id)
    return {"status": "paused"} if ok else {"error": "Source not found"}


@router.post("/sources/{source_id}/resume")
async def resume_source_endpoint(source_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Resume auto-sync for a data source."""
    from business_brain.ingestion.sync_engine import resume_source
    ok = await resume_source(session, source_id)
    return {"status": "resumed"} if ok else {"error": "Source not found"}


@router.get("/sync/status")
async def get_sync_status(session: AsyncSession = Depends(get_session)) -> dict:
    """Get background sync loop status."""
    from sqlalchemy import select, func
    from business_brain.db.v3_models import DataSource

    result = await session.execute(
        select(func.count()).select_from(DataSource).where(
            DataSource.active == True,  # noqa: E712
            DataSource.sync_frequency_minutes > 0,
        )
    )
    active_count = result.scalar() or 0

    # Note: _sync_task status is managed in the main api.py app lifecycle
    return {
        "loop_running": True,  # Assumes sync loop is running when app is up
        "active_sources": active_count,
        "poll_interval_seconds": 60,
    }
