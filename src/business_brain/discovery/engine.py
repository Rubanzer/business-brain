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
from business_brain.discovery.seasonality_detector import detect_seasonality

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

        # 6b. Detect seasonality patterns
        logger.info("Discovery: detecting seasonality patterns...")
        seasonality_insights = detect_seasonality(profiles)
        logger.info("Discovery: found %d seasonality insights", len(seasonality_insights))

        # 6c. Discover correlations from profile data
        logger.info("Discovery: discovering correlations...")
        try:
            from business_brain.discovery.correlation_discoverer import discover_correlations_from_profiles
            correlation_insights = discover_correlations_from_profiles(profiles)
            logger.info("Discovery: found %d correlation insights", len(correlation_insights))
        except Exception:
            logger.exception("Correlation discovery failed, continuing")
            correlation_insights = []

        # 7. Combine all insights
        all_insights = anomaly_insights + composite_insights + cross_insights + seasonality_insights + correlation_insights

        # 8. Build narratives (if 2+ insights)
        if len(all_insights) >= 2:
            logger.info("Discovery: building narratives...")
            try:
                story_insights = await build_narratives(all_insights)
                all_insights.extend(story_insights)
                logger.info("Discovery: generated %d narrative stories", len(story_insights))
            except Exception:
                logger.exception("Narrative building failed, continuing without stories")

        # 8b. Run sanctity checks
        logger.info("Discovery: running sanctity checks...")
        try:
            from business_brain.discovery.sanctity_engine import run_sanctity_check
            sanctity_summary = await run_sanctity_check(session, profiles)
            logger.info("Discovery: sanctity check found %d issues", sanctity_summary.get("total", 0))
        except Exception:
            logger.exception("Sanctity check failed, continuing")

        # 8c. Check pattern memory for matches
        logger.info("Discovery: checking pattern memory...")
        try:
            from business_brain.discovery.pattern_memory import check_patterns
            for profile in profiles:
                matches = await check_patterns(session, profile.table_name)
                if matches:
                    logger.info("Discovery: %d pattern matches in %s", len(matches), profile.table_name)
        except Exception:
            logger.exception("Pattern matching failed, continuing")

        # 8d. Detect duplicate data sources
        logger.info("Discovery: detecting duplicate sources...")
        try:
            from business_brain.discovery.format_detector import detect_duplicate_sources
            duplicates = await detect_duplicate_sources(session, profiles)
            if duplicates:
                logger.info("Discovery: found %d potential duplicate sources", len(duplicates))
        except Exception:
            logger.exception("Duplicate source detection failed, continuing")

        # 8e. Detect schema changes from previous profiles
        logger.info("Discovery: detecting schema changes...")
        try:
            from business_brain.discovery.schema_tracker import detect_schema_changes
            prev_result = await session.execute(
                select(TableProfile).where(
                    TableProfile.table_name.in_([p.table_name for p in profiles])
                )
            )
            previous_profiles = list(prev_result.scalars().all())
            schema_insights = detect_schema_changes(profiles, previous_profiles)
            if schema_insights:
                all_insights.extend(schema_insights)
                logger.info("Discovery: found %d schema changes", len(schema_insights))
        except Exception:
            logger.exception("Schema change detection failed, continuing")

        # 8f. Data freshness tracking
        logger.info("Discovery: checking data freshness...")
        try:
            from business_brain.discovery.data_freshness import detect_stale_tables
            prev_result2 = await session.execute(
                select(TableProfile).where(
                    TableProfile.table_name.in_([p.table_name for p in profiles])
                )
            )
            prev_profiles2 = list(prev_result2.scalars().all())
            stale_insights = detect_stale_tables(profiles, prev_profiles2)
            if stale_insights:
                all_insights.extend(stale_insights)
                logger.info("Discovery: found %d stale tables", len(stale_insights))
        except Exception:
            logger.exception("Data freshness check failed, continuing")

        # 9. Deduplicate insights against existing DB records
        logger.info("Discovery: deduplicating insights...")
        try:
            from business_brain.discovery.dedup import compute_insight_key, deduplicate_insights
            existing_result = await session.execute(select(Insight.source_tables, Insight.source_columns, Insight.insight_type, Insight.composite_template))
            existing_keys = set()
            for row in existing_result.fetchall():
                dummy = Insight()
                dummy.insight_type = row[2]
                dummy.source_tables = row[0]
                dummy.source_columns = row[1]
                dummy.composite_template = row[3]
                existing_keys.add(compute_insight_key(dummy))
            pre_count = len(all_insights)
            all_insights = deduplicate_insights(all_insights, existing_keys)
            deduped = pre_count - len(all_insights)
            if deduped:
                logger.info("Discovery: deduplicated %d insights", deduped)
        except Exception:
            logger.exception("Insight deduplication failed, continuing")

        # 10. Score all insights
        for insight in all_insights:
            insight.discovery_run_id = run.id
            _apply_scoring(insight)

        # 11. Bulk insert insights
        for insight in all_insights:
            session.add(insight)
        await session.flush()

        # 11. Auto-refresh deployed reports whose source tables were re-profiled
        profiled_tables = {p.table_name for p in profiles}
        await _refresh_affected_reports(session, profiled_tables)

        # 11b. Evaluate all alert rules after data refresh
        logger.info("Discovery: evaluating alert rules...")
        try:
            from business_brain.action.alert_engine import evaluate_all_alerts
            alert_events = await evaluate_all_alerts(session)
            if alert_events:
                logger.info("Discovery: %d alerts triggered", len(alert_events))
        except Exception:
            logger.exception("Alert evaluation failed, continuing")

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
    magnitude = ((insight.impact_score or 0) / 100) * 30

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
