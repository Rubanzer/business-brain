"""Engagement tracker — captures implicit signals from user interactions.

Every time a user views the feed, deploys an insight, dismisses insights,
or views recommendations, an event is recorded. These events power the
Phase 3 reinforcement loop that adjusts quality gate scoring weights.

All tracking functions are fire-and-forget: they log errors and never raise,
so they can never break the main API flow.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import EngagementEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level event creation
# ---------------------------------------------------------------------------


async def _record_event(
    session: AsyncSession,
    event_type: str,
    entity_type: str,
    entity_id: str | None = None,
    analysis_type: str | None = None,
    table_name: str | None = None,
    columns: list[str] | None = None,
    severity: str | None = None,
    impact_score: int | None = None,
    metadata: dict | None = None,
    session_id: str | None = None,
) -> None:
    """Create a single EngagementEvent record.

    Fire-and-forget: logs errors, never raises.
    """
    try:
        event = EngagementEvent(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            analysis_type=analysis_type,
            table_name=table_name,
            columns=columns,
            severity=severity,
            impact_score=impact_score,
            extra_metadata=metadata,
            session_id=session_id,
        )
        session.add(event)
        await session.flush()
    except Exception:
        logger.debug("Failed to record engagement event: %s/%s", event_type, entity_id)


# ---------------------------------------------------------------------------
# High-level tracking functions
# ---------------------------------------------------------------------------


async def track_insights_shown(
    session: AsyncSession,
    insights: list[dict],
    session_id: str | None = None,
) -> None:
    """Called when GET /feed returns insights. Creates one event per insight shown.

    Args:
        insights: List of dicts with keys: id, insight_type, severity,
                  impact_score, source_tables.
    """
    try:
        for ins in insights:
            source_tables = ins.get("source_tables") or []
            table_name = source_tables[0] if source_tables else None
            await _record_event(
                session,
                event_type="insight_shown",
                entity_type="insight",
                entity_id=ins.get("id"),
                analysis_type=ins.get("insight_type"),
                table_name=table_name,
                severity=ins.get("severity"),
                impact_score=ins.get("impact_score"),
                session_id=session_id,
            )
    except Exception:
        logger.debug("Failed to track insights shown")


async def track_insight_action(
    session: AsyncSession,
    insight_id: str,
    action: str,
    insight_type: str | None = None,
    severity: str | None = None,
    impact_score: int | None = None,
    source_tables: list[str] | None = None,
    session_id: str | None = None,
) -> None:
    """Called when user updates insight status or deploys.

    Args:
        action: "seen" | "deployed" | "dismissed"
    """
    try:
        event_type = f"insight_{action}"
        table_name = source_tables[0] if source_tables else None
        await _record_event(
            session,
            event_type=event_type,
            entity_type="insight",
            entity_id=insight_id,
            analysis_type=insight_type,
            table_name=table_name,
            severity=severity,
            impact_score=impact_score,
            session_id=session_id,
        )
    except Exception:
        logger.debug("Failed to track insight action: %s/%s", action, insight_id)


async def track_insights_dismissed_all(
    session: AsyncSession,
    count: int,
    session_id: str | None = None,
) -> None:
    """Called when user dismisses all insights. Creates one bulk event."""
    try:
        await _record_event(
            session,
            event_type="insights_dismissed_all",
            entity_type="insight",
            metadata={"dismissed_count": count},
            session_id=session_id,
        )
    except Exception:
        logger.debug("Failed to track bulk dismiss")


async def track_recommendations_shown(
    session: AsyncSession,
    recommendations: list[dict],
    session_id: str | None = None,
) -> None:
    """Called when GET /recommendations returns. Creates one event per rec shown.

    Args:
        recommendations: List of dicts with keys: analysis_type, target_table,
                         columns, confidence, priority.
    """
    try:
        for rec in recommendations:
            await _record_event(
                session,
                event_type="recommendation_shown",
                entity_type="recommendation",
                analysis_type=rec.get("analysis_type"),
                table_name=rec.get("target_table"),
                columns=rec.get("columns"),
                metadata={
                    "confidence": rec.get("confidence"),
                    "priority": rec.get("priority"),
                },
                session_id=session_id,
            )
    except Exception:
        logger.debug("Failed to track recommendations shown")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


async def get_engagement_summary(
    session: AsyncSession,
    days: int = 30,
) -> dict:
    """Aggregate engagement events over the last N days.

    Returns a summary with breakdowns by event_type, analysis_type,
    severity, and table. Computes engagement_rate where applicable
    (deployed / shown).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Fetch all events in window
    result = await session.execute(
        select(EngagementEvent).where(EngagementEvent.created_at >= cutoff)
    )
    events = list(result.scalars().all())

    # --- by_event_type ---
    by_event_type: dict[str, int] = {}
    for e in events:
        by_event_type[e.event_type] = by_event_type.get(e.event_type, 0) + 1

    # --- by_analysis_type (with engagement rate) ---
    # Collect shown/deployed/dismissed counts per analysis_type
    at_shown: dict[str, int] = {}
    at_deployed: dict[str, int] = {}
    at_dismissed: dict[str, int] = {}
    for e in events:
        at = e.analysis_type
        if not at:
            continue
        if e.event_type in ("insight_shown", "recommendation_shown"):
            at_shown[at] = at_shown.get(at, 0) + 1
        elif e.event_type == "insight_deployed":
            at_deployed[at] = at_deployed.get(at, 0) + 1
        elif e.event_type == "insight_dismissed":
            at_dismissed[at] = at_dismissed.get(at, 0) + 1

    all_types = set(at_shown) | set(at_deployed) | set(at_dismissed)
    by_analysis_type: dict[str, dict] = {}
    for at in all_types:
        shown = at_shown.get(at, 0)
        deployed = at_deployed.get(at, 0)
        dismissed = at_dismissed.get(at, 0)
        rate = round(deployed / shown, 4) if shown > 0 else 0.0
        by_analysis_type[at] = {
            "shown": shown,
            "deployed": deployed,
            "dismissed": dismissed,
            "engagement_rate": rate,
        }

    # --- by_severity ---
    sev_shown: dict[str, int] = {}
    sev_deployed: dict[str, int] = {}
    for e in events:
        sev = e.severity
        if not sev:
            continue
        if e.event_type == "insight_shown":
            sev_shown[sev] = sev_shown.get(sev, 0) + 1
        elif e.event_type == "insight_deployed":
            sev_deployed[sev] = sev_deployed.get(sev, 0) + 1

    all_sevs = set(sev_shown) | set(sev_deployed)
    by_severity: dict[str, dict] = {}
    for sev in all_sevs:
        shown = sev_shown.get(sev, 0)
        deployed = sev_deployed.get(sev, 0)
        rate = round(deployed / shown, 4) if shown > 0 else 0.0
        by_severity[sev] = {
            "shown": shown,
            "deployed": deployed,
            "engagement_rate": rate,
        }

    # --- by_table ---
    tbl_shown: dict[str, int] = {}
    tbl_deployed: dict[str, int] = {}
    for e in events:
        tbl = e.table_name
        if not tbl:
            continue
        if e.event_type in ("insight_shown", "recommendation_shown"):
            tbl_shown[tbl] = tbl_shown.get(tbl, 0) + 1
        elif e.event_type == "insight_deployed":
            tbl_deployed[tbl] = tbl_deployed.get(tbl, 0) + 1

    all_tables = set(tbl_shown) | set(tbl_deployed)
    by_table: dict[str, dict] = {}
    for tbl in all_tables:
        by_table[tbl] = {
            "shown": tbl_shown.get(tbl, 0),
            "deployed": tbl_deployed.get(tbl, 0),
        }

    return {
        "period_days": days,
        "total_events": len(events),
        "by_event_type": by_event_type,
        "by_analysis_type": by_analysis_type,
        "by_severity": by_severity,
        "by_table": by_table,
    }
