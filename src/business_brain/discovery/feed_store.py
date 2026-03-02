"""CRUD for insights + deployed reports."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import DeployedReport, DiscoveryRun, Insight

logger = logging.getLogger(__name__)


async def get_feed(
    session: AsyncSession,
    status: str | None = None,
    insight_type: str | None = None,
    limit: int = 500,
    min_quality: int = 15,
) -> list[Insight]:
    """Get ranked insight feed — returns ALL qualifying insights.

    When no type filter is active, applies type-diverse selection so no
    single insight type dominates. When a specific type is requested,
    returns all matching insights up to limit.

    Args:
        session: Database session.
        status: Filter by specific status (new/seen/deployed/dismissed).
        insight_type: Filter by specific insight type (e.g. "anomaly").
        limit: Maximum number of insights to return. Default 500 (effectively all).
        min_quality: Minimum quality_score threshold. Default 15.
    """
    q = select(Insight).order_by(Insight.impact_score.desc(), Insight.discovered_at.desc())

    if status:
        q = q.where(Insight.status == status)
    else:
        # Exclude dismissed by default
        q = q.where(Insight.status != "dismissed")

    if insight_type:
        q = q.where(Insight.insight_type == insight_type)

    # Filter by minimum quality score to keep feed clean
    if min_quality > 0:
        q = q.where(
            (Insight.quality_score == None)  # noqa: E711
            | (Insight.quality_score == 0)
            | (Insight.quality_score >= min_quality)
        )

    q = q.limit(limit)
    result = await session.execute(q)
    all_insights = list(result.scalars().all())

    # If filtering by type, no need for diversity balancing — return all
    if insight_type or len(all_insights) <= 50:
        return all_insights

    # Type-diverse selection: ensure each insight type gets fair representation.
    # Take up to min_per_type from each type first, then fill remaining by score.
    from collections import defaultdict

    by_type: dict[str, list[Insight]] = defaultdict(list)
    for ins in all_insights:
        by_type[ins.insight_type or "unknown"].append(ins)

    n_types = max(len(by_type), 1)
    # Reserve at least 3 slots per type, up to 10
    min_per_type = min(max(limit // (n_types * 2), 3), 10)

    selected: list[Insight] = []
    selected_ids: set[str] = set()

    # Round 1: guaranteed slots per type (already sorted by score from query)
    for itype, type_insights in by_type.items():
        for ins in type_insights[:min_per_type]:
            if ins.id not in selected_ids:
                selected.append(ins)
                selected_ids.add(ins.id)

    # Round 2: fill remaining slots by score
    remaining = limit - len(selected)
    if remaining > 0:
        for ins in all_insights:
            if ins.id not in selected_ids:
                selected.append(ins)
                selected_ids.add(ins.id)
                remaining -= 1
                if remaining <= 0:
                    break

    # Re-sort final list by score for consistent presentation
    selected.sort(key=lambda i: (i.impact_score or 0, i.discovered_at or ""), reverse=True)
    return selected


async def update_status(session: AsyncSession, insight_id: str, status: str) -> None:
    """Update an insight's status (new/seen/deployed/dismissed)."""
    result = await session.execute(select(Insight).where(Insight.id == insight_id))
    insight = result.scalar_one_or_none()
    if insight:
        insight.status = status
        await session.commit()


async def dismiss_all(session: AsyncSession) -> int:
    """Dismiss all non-dismissed insights. Returns count dismissed."""
    from sqlalchemy import update

    result = await session.execute(
        update(Insight)
        .where(Insight.status != "dismissed")
        .values(status="dismissed")
    )
    await session.commit()
    return result.rowcount


async def purge_dismissed(session: AsyncSession) -> int:
    """Permanently delete all dismissed insights from the database.

    Returns the count of deleted rows. This frees them up for re-discovery
    on the next scan (dedup won't see them anymore).
    """
    from sqlalchemy import delete

    result = await session.execute(
        delete(Insight).where(Insight.status == "dismissed")
    )
    await session.commit()
    return result.rowcount


async def deploy_insight(session: AsyncSession, insight_id: str, name: str) -> DeployedReport:
    """Create a deployed report from an insight."""
    result = await session.execute(select(Insight).where(Insight.id == insight_id))
    insight = result.scalar_one_or_none()

    if not insight:
        raise ValueError(f"Insight {insight_id} not found")

    evidence = insight.evidence or {}

    report = DeployedReport(
        id=str(uuid.uuid4()),
        name=name,
        insight_id=insight_id,
        query=evidence.get("query"),
        chart_spec=evidence.get("chart_spec"),
        parameters={},
        last_result=evidence.get("sample_rows"),
        last_run_at=datetime.now(timezone.utc),
        session_id=insight.session_id,
        active=True,
    )
    session.add(report)

    # Mark insight as deployed
    insight.status = "deployed"
    await session.commit()
    await session.refresh(report)

    return report


async def get_reports(session: AsyncSession) -> list[DeployedReport]:
    """Get all active deployed reports."""
    result = await session.execute(
        select(DeployedReport)
        .where(DeployedReport.active == True)  # noqa: E712
        .order_by(DeployedReport.created_at.desc())
    )
    return list(result.scalars().all())


async def get_report(session: AsyncSession, report_id: str) -> DeployedReport | None:
    """Get a single deployed report."""
    result = await session.execute(
        select(DeployedReport).where(DeployedReport.id == report_id)
    )
    return result.scalar_one_or_none()


async def refresh_report(session: AsyncSession, report_id: str) -> DeployedReport | None:
    """Re-run a report's query and update its last_result."""
    report = await get_report(session, report_id)
    if not report or not report.query:
        return report

    try:
        # Sanitize the query (only allow SELECT)
        query = report.query.strip()
        if not query.upper().startswith("SELECT"):
            logger.warning("Report %s has non-SELECT query, skipping refresh", report_id)
            return report

        result = await session.execute(sql_text(query))
        rows = [dict(r._mapping) for r in result.fetchall()]

        report.last_result = rows
        report.last_run_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(report)
    except Exception:
        logger.exception("Failed to refresh report %s", report_id)
        await session.rollback()

    return report


async def delete_report(session: AsyncSession, report_id: str) -> bool:
    """Soft-delete a deployed report."""
    report = await get_report(session, report_id)
    if not report:
        return False

    report.active = False
    await session.commit()
    return True


async def get_last_run(session: AsyncSession) -> DiscoveryRun | None:
    """Get the most recent discovery run."""
    result = await session.execute(
        select(DiscoveryRun).order_by(DiscoveryRun.started_at.desc()).limit(1)
    )
    return result.scalar_one_or_none()
