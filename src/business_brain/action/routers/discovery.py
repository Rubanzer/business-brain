"""Discovery & Suggestions routes."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.action.dependencies import get_focus_tables, run_discovery_background
from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["discovery"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/discovery/trigger")
async def trigger_discovery(
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Manually trigger a discovery sweep, respecting focus scope."""
    focus_tables = await get_focus_tables(session)
    background_tasks.add_task(run_discovery_background, "manual", table_filter=focus_tables)
    msg = "Discovery sweep triggered in background"
    if focus_tables:
        msg += f" (focused on {len(focus_tables)} tables)"
    return {"status": "started", "message": msg}


@router.get("/discovery/status")
async def discovery_status(session: AsyncSession = Depends(get_session)) -> dict:
    """Get the last discovery run info."""
    from business_brain.discovery.feed_store import get_last_run

    run = await get_last_run(session)
    if not run:
        return {"status": "no_runs", "message": "No discovery runs yet"}
    diagnostics = getattr(run, "pass_diagnostics", None) or []
    failed_passes = [d for d in diagnostics if d.get("status") == "failed"]
    return {
        "id": run.id,
        "status": run.status,
        "trigger": run.trigger,
        "tables_scanned": run.tables_scanned,
        "insights_found": run.insights_found,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "error": run.error,
        "pass_diagnostics": diagnostics,
        "failed_passes": len(failed_passes),
        "total_passes": len(diagnostics),
    }


@router.get("/suggestions")
async def get_suggestions(session: AsyncSession = Depends(get_session)) -> dict:
    """Get smart question suggestions based on profiled tables."""
    from sqlalchemy import select

    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.profiler import generate_suggestions

    result = await session.execute(select(TableProfile))
    profiles = list(result.scalars().all())

    if not profiles:
        return {"suggestions": []}

    suggestions = generate_suggestions(profiles)
    return {"suggestions": suggestions}


@router.get("/discovery/diagnostics/{run_id}")
async def discovery_diagnostics(
    run_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get detailed per-pass diagnostics for a specific discovery run."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import DiscoveryRun

    result = await session.execute(
        select(DiscoveryRun).where(DiscoveryRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        return {"error": "Run not found"}

    diagnostics = getattr(run, "pass_diagnostics", None) or []
    failed = [d for d in diagnostics if d.get("status") == "failed"]
    rate_limited = [d for d in failed if d.get("error_type") == "rate_limit"]

    return {
        "run_id": run.id,
        "status": run.status,
        "total_passes": len(diagnostics),
        "ok_passes": sum(1 for d in diagnostics if d.get("status") == "ok"),
        "failed_passes": len(failed),
        "rate_limited_passes": len(rate_limited),
        "total_duration_ms": sum(d.get("duration_ms", 0) for d in diagnostics),
        "passes": diagnostics,
        "failed_summary": [
            {"pass": d["pass"], "error_type": d.get("error_type", "unknown"), "error": d.get("error", "")}
            for d in failed
        ],
    }


@router.get("/discovery/history")
async def discovery_history(
    limit: int = 10,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get recent discovery run history."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import DiscoveryRun

    result = await session.execute(
        select(DiscoveryRun).order_by(DiscoveryRun.started_at.desc()).limit(limit)
    )
    runs = list(result.scalars().all())
    return [
        {
            "id": r.id,
            "status": r.status,
            "trigger": r.trigger,
            "tables_scanned": r.tables_scanned,
            "insights_found": r.insights_found,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "duration_seconds": (
                (r.completed_at - r.started_at).total_seconds()
                if r.completed_at and r.started_at else None
            ),
            "error": r.error,
        }
        for r in runs
    ]
