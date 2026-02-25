"""Data Quality & Sanctity routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["quality"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class ResolveRequest(BaseModel):
    resolved_by: str
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/sanctity")
async def get_sanctity_issues(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all open sanctity issues."""
    from business_brain.discovery.sanctity_engine import get_open_issues

    issues = await get_open_issues(session)
    return [
        {
            "id": i.id,
            "table_name": i.table_name,
            "column_name": i.column_name,
            "row_identifier": i.row_identifier,
            "issue_type": i.issue_type,
            "severity": i.severity,
            "description": i.description,
            "current_value": i.current_value,
            "expected_range": i.expected_range,
            "conflicting_source": i.conflicting_source,
            "conflicting_value": i.conflicting_value,
            "detected_at": i.detected_at.isoformat() if i.detected_at else None,
            "resolved": i.resolved,
        }
        for i in issues
    ]


@router.get("/sanctity/summary")
async def get_sanctity_summary(session: AsyncSession = Depends(get_session)) -> dict:
    """Get summary counts of sanctity issues."""
    from business_brain.discovery.sanctity_engine import run_sanctity_check
    return await run_sanctity_check(session)


@router.post("/sanctity/{issue_id}/resolve")
async def resolve_sanctity_issue(
    issue_id: int,
    req: ResolveRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Mark a sanctity issue as resolved."""
    from business_brain.discovery.sanctity_engine import resolve_issue
    issue = await resolve_issue(session, issue_id, req.resolved_by, req.note)
    if not issue:
        return {"error": "Issue not found"}
    return {"status": "resolved", "issue_id": issue.id}


@router.get("/changes")
async def get_all_changes(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get recent data changes across all sources."""
    from business_brain.discovery.sanctity_engine import get_recent_changes

    changes = await get_recent_changes(session, limit=100)
    return [
        {
            "id": c.id,
            "change_type": c.change_type,
            "table_name": c.table_name,
            "column_name": c.column_name,
            "old_value": c.old_value,
            "new_value": c.new_value,
            "detected_at": c.detected_at.isoformat() if c.detected_at else None,
        }
        for c in changes
    ]


@router.get("/data-quality")
async def get_data_quality(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get data quality scores for all profiled tables."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.profiler import compute_data_quality_score

    result = await session.execute(select(TableProfile))
    profiles = list(result.scalars().all())

    scores = []
    for p in profiles:
        quality = compute_data_quality_score(p)
        scores.append({
            "table_name": p.table_name,
            "row_count": p.row_count,
            "domain_hint": p.domain_hint,
            "score": quality["score"],
            "breakdown": quality["breakdown"],
            "issues": quality["issues"],
            "profiled_at": p.profiled_at.isoformat() if p.profiled_at else None,
        })

    return sorted(scores, key=lambda s: s["score"])
