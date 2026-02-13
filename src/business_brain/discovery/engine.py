"""Main discovery engine orchestrator — runs all passes in sequence."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import (
    DeployedReport,
    DiscoveryRun,
    Insight,
    TableProfile,
)
from business_brain.discovery.anomaly_detector import detect_anomalies
from business_brain.discovery.composite_discoverer import discover_composites
from business_brain.discovery.cross_event_correlator import find_cross_events
from business_brain.discovery.feed_store import refresh_report
from business_brain.discovery.narrative_builder import build_narratives
from business_brain.discovery.profiler import profile_all_tables
from business_brain.discovery.relationship_finder import find_relationships

logger = logging.getLogger(__name__)


async def run_discovery(
    session: AsyncSession,
    trigger: str = "manual",
) -> DiscoveryRun:
    """Run the full discovery pipeline.

    1. Profile all tables
    2. Find cross-table relationships
    3. Detect anomalies
    4. Discover composite metrics
    5. Find cross-event correlations
    6. Build narratives (if 3+ insights)
    7. Score and store all insights
    8. Auto-refresh deployed reports
    """
    # 1. Create DiscoveryRun
    run = DiscoveryRun(
        id=str(uuid.uuid4()),
        status="running",
        trigger=trigger,
    )
    session.add(run)
    await session.flush()

    try:
        # 2. Profile all tables
        logger.info("Discovery: profiling tables...")
        profiles = await profile_all_tables(session)
        run.tables_scanned = len(profiles)

        if not profiles:
            run.status = "completed"
            run.insights_found = 0
            run.completed_at = datetime.now(timezone.utc)
            await session.commit()
            return run

        # 3. Find relationships
        logger.info("Discovery: finding relationships...")
        relationships = await find_relationships(session, profiles)
        logger.info("Discovery: found %d relationships", len(relationships))

        # 4. Detect anomalies
        logger.info("Discovery: detecting anomalies...")
        anomaly_insights = detect_anomalies(profiles)
        logger.info("Discovery: found %d anomaly insights", len(anomaly_insights))

        # 5. Discover composites
        logger.info("Discovery: discovering composites...")
        composite_insights = discover_composites(profiles, relationships)
        logger.info("Discovery: found %d composite insights", len(composite_insights))

        # 6. Find cross-events
        logger.info("Discovery: finding cross-event correlations...")
        cross_insights = await find_cross_events(session, profiles, relationships)
        logger.info("Discovery: found %d cross-event insights", len(cross_insights))

        # 7. Combine all insights
        all_insights = anomaly_insights + composite_insights + cross_insights

        # 8. Build narratives (if 3+ insights)
        if len(all_insights) >= 3:
            logger.info("Discovery: building narratives...")
            try:
                story_insights = await build_narratives(all_insights)
                all_insights.extend(story_insights)
                logger.info("Discovery: generated %d narrative stories", len(story_insights))
            except Exception:
                logger.exception("Narrative building failed, continuing without stories")

        # 9. Score all insights
        for insight in all_insights:
            insight.discovery_run_id = run.id
            _apply_scoring(insight)

        # 10. Bulk insert insights
        for insight in all_insights:
            session.add(insight)
        await session.flush()

        # 11. Auto-refresh deployed reports whose source tables were re-profiled
        profiled_tables = {p.table_name for p in profiles}
        await _refresh_affected_reports(session, profiled_tables)

        # 12. Complete the run
        run.status = "completed"
        run.insights_found = len(all_insights)
        run.completed_at = datetime.now(timezone.utc)

        await session.commit()
        logger.info(
            "Discovery completed: %d tables, %d insights",
            run.tables_scanned,
            run.insights_found,
        )

    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)
        run.completed_at = datetime.now(timezone.utc)
        await session.commit()
        logger.exception("Discovery run failed")
        raise

    return run


def _apply_scoring(insight: Insight) -> None:
    """Apply the scoring formula: severity*40 + cross_table*30 + magnitude*30."""
    severity_weights = {"critical": 1.0, "warning": 0.6, "info": 0.3}
    severity_score = severity_weights.get(insight.severity, 0.3) * 40

    # Cross-table bonus
    source_tables = insight.source_tables or []
    cross_table_bonus = 30 if len(source_tables) > 1 else 0

    # Change magnitude (use existing impact_score as base)
    magnitude = (insight.impact_score / 100) * 30

    insight.impact_score = min(int(severity_score + cross_table_bonus + magnitude), 100)


async def _refresh_affected_reports(
    session: AsyncSession,
    profiled_tables: set[str],
) -> None:
    """Refresh deployed reports whose source tables were re-profiled."""
    result = await session.execute(
        select(DeployedReport).where(DeployedReport.active == True)  # noqa: E712
    )
    reports = list(result.scalars().all())

    for report in reports:
        # Check if this report's source insight references any profiled table
        insight_result = await session.execute(
            select(Insight).where(Insight.id == report.insight_id)
        )
        insight = insight_result.scalar_one_or_none()
        if not insight:
            continue

        insight_tables = set(insight.source_tables or [])
        if insight_tables & profiled_tables:
            try:
                await refresh_report(session, report.id)
                logger.info("Auto-refreshed report %s", report.name)
            except Exception:
                logger.exception("Failed to auto-refresh report %s", report.id)
