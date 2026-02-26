"""Insight Feed routes."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.action.dependencies import get_focus_tables
from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feed"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class DeployRequest(BaseModel):
    name: str


class StatusRequest(BaseModel):
    status: str  # seen/dismissed


class SaveToFeedRequest(BaseModel):
    title: str
    description: str
    insight_type: str = "analysis"
    severity: str = "info"
    evidence: Optional[dict] = None
    suggested_actions: Optional[list[str]] = None
    source_tables: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/feed")
async def get_feed(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get ranked insight feed, filtered by focus scope if active."""
    from business_brain.discovery.feed_store import get_feed as _get_feed

    try:
        insights = await _get_feed(session)

        focus_tables = await get_focus_tables(session)
        if focus_tables:
            focus_set = set(focus_tables)
            insights = [
                i for i in insights
                if not i.source_tables or any(t in focus_set for t in (i.source_tables or []))
            ]

        return [
            {
                "id": i.id,
                "insight_type": i.insight_type,
                "severity": i.severity,
                "impact_score": i.impact_score,
                "title": i.title,
                "description": i.description,
                "narrative": i.narrative,
                "source_tables": i.source_tables,
                "source_columns": i.source_columns,
                "evidence": i.evidence,
                "related_insights": i.related_insights,
                "suggested_actions": i.suggested_actions,
                "composite_template": i.composite_template,
                "discovered_at": i.discovered_at.isoformat() if i.discovered_at else None,
                "status": i.status,
            }
            for i in insights
        ]
    except Exception:
        logger.exception("Feed fetch failed")
        await session.rollback()
        return []


@router.post("/feed/rescore")
async def rescore_feed(session: AsyncSession = Depends(get_session)) -> dict:
    """Re-score existing insights that have NULL quality_score."""
    from sqlalchemy import select as sa_select

    from business_brain.db.discovery_models import Insight
    from business_brain.discovery.insight_quality_gate import apply_quality_gate

    try:
        result = await session.execute(
            sa_select(Insight).where(Insight.quality_score == None)  # noqa: E711
        )
        null_insights = list(result.scalars().all())

        if not null_insights:
            return {"status": "ok", "rescored": 0, "message": "No insights with NULL quality_score"}

        scored = apply_quality_gate(null_insights, [])

        for insight in null_insights:
            if insight not in scored:
                insight.quality_score = 0
                insight.impact_score = insight.impact_score or 0

        await session.commit()
        return {
            "status": "ok",
            "rescored": len(null_insights),
            "kept": len(scored),
            "filtered": len(null_insights) - len(scored),
        }
    except Exception as exc:
        logger.exception("Feed rescore failed")
        await session.rollback()
        return {"status": "error", "error": str(exc)}


@router.post("/feed/dismiss-all")
async def dismiss_all_insights(session: AsyncSession = Depends(get_session)) -> dict:
    """Dismiss all active insights."""
    from business_brain.discovery.feed_store import dismiss_all

    count = await dismiss_all(session)
    return {"status": "ok", "dismissed": count}


@router.post("/feed/{insight_id}/status")
async def update_insight_status(
    insight_id: str,
    req: StatusRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update insight status (seen/dismissed)."""
    from business_brain.discovery.feed_store import update_status

    await update_status(session, insight_id, req.status)
    return {"status": "updated", "insight_id": insight_id, "new_status": req.status}


@router.post("/feed/{insight_id}/deploy")
async def deploy_insight_as_report(
    insight_id: str,
    req: DeployRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Deploy an insight as a persistent report."""
    from business_brain.discovery.feed_store import deploy_insight

    try:
        report = await deploy_insight(session, insight_id, req.name)
        return {
            "status": "deployed",
            "report_id": report.id,
            "name": report.name,
            "insight_id": report.insight_id,
        }
    except ValueError as exc:
        return {"error": str(exc)}


@router.post("/feed/from-analysis")
async def save_analysis_to_feed(
    req: SaveToFeedRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Save an analysis result as a manual insight in the Feed."""
    from business_brain.db.discovery_models import Insight

    insight = Insight(
        insight_type=req.insight_type,
        severity=req.severity,
        impact_score=75,
        quality_score=75,
        title=req.title,
        description=req.description,
        source_tables=req.source_tables or [],
        evidence=req.evidence or {},
        suggested_actions=req.suggested_actions or [],
        status="new",
    )
    session.add(insight)
    await session.commit()
    await session.refresh(insight)

    return {"status": "created", "insight_id": insight.id, "title": req.title}


@router.get("/feed/export")
async def export_feed(session: AsyncSession = Depends(get_session)):
    """Export the insight feed as JSON."""
    from fastapi.responses import Response
    from business_brain.discovery.feed_store import get_feed as _get_feed

    insights = await _get_feed(session, limit=500)
    data = [
        {
            "id": i.id,
            "insight_type": i.insight_type,
            "severity": i.severity,
            "impact_score": i.impact_score,
            "title": i.title,
            "description": i.description,
            "source_tables": i.source_tables,
            "source_columns": i.source_columns,
            "composite_template": i.composite_template,
            "suggested_actions": i.suggested_actions,
            "discovered_at": i.discovered_at.isoformat() if i.discovered_at else None,
            "status": i.status,
        }
        for i in insights
    ]
    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="insights_feed.json"'},
    )
