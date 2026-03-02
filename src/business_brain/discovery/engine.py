"""Main discovery engine orchestrator — runs all passes in sequence."""

from __future__ import annotations

import logging
import time
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
from business_brain.discovery.insight_quality_gate import apply_quality_gate
from business_brain.discovery.composite_discoverer import discover_composites
from business_brain.discovery.cross_event_correlator import find_cross_events
from business_brain.discovery.feed_store import refresh_report
from business_brain.discovery.narrative_builder import build_narratives
from business_brain.discovery.profiler import profile_all_tables
from business_brain.discovery.relationship_finder import find_relationships
from business_brain.discovery.seasonality_detector import detect_seasonality

logger = logging.getLogger(__name__)


def _classify_error(exc: Exception) -> str:
    """Classify an exception into a category for diagnostics."""
    msg = str(exc).lower()
    cls_name = type(exc).__name__.lower()
    if "429" in msg or "quota" in msg or "rate" in msg or "resource_exhausted" in msg:
        return "rate_limit"
    if "401" in msg or "403" in msg or "auth" in msg or "api_key" in msg or "permission" in msg:
        return "auth_error"
    if "timeout" in msg or "timed out" in msg or cls_name in ("timeouterror", "readtimeout"):
        return "timeout"
    if "json" in msg or "parse" in msg or "decode" in msg:
        return "parse_error"
    if "connect" in msg or "network" in msg or "dns" in msg or "socket" in msg:
        return "network_error"
    return "internal_error"


class _PassTracker:
    """Records per-pass outcomes for discovery diagnostics."""

    def __init__(self) -> None:
        self._results: list[dict] = []

    def record(
        self,
        name: str,
        *,
        status: str = "ok",
        count: int | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
        error_type: str | None = None,
    ) -> None:
        entry: dict = {"pass": name, "status": status}
        if count is not None:
            entry["count"] = count
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        if error:
            entry["error"] = error[:500]
        if error_type:
            entry["error_type"] = error_type
        self._results.append(entry)

    @property
    def results(self) -> list[dict]:
        return list(self._results)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self._results if r["status"] == "failed")


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


async def run_discovery(
    session: AsyncSession,
    trigger: str = "manual",
    table_filter: list[str] | None = None,
) -> DiscoveryRun:
    """Run the fast discovery pipeline (fits within serverless timeout).

    Fast passes (~3-5s): profile, relationships, anomalies, benchmarks,
    composites, cross-events, seasonality, correlations, domain, entity.
    All committed before returning.

    Slow passes (narratives, sanctity, precomputation) run separately
    via run_discovery_enrich().
    """
    run = DiscoveryRun(
        id=str(uuid.uuid4()),
        status="running",
        trigger=trigger,
    )
    session.add(run)
    await session.flush()

    tracker = _PassTracker()

    try:
        # 1. Profile all tables
        t0 = time.monotonic()
        logger.info("Discovery: profiling tables...")
        profiles = await profile_all_tables(session, table_filter=table_filter)
        run.tables_scanned = len(profiles)
        tracker.record("profile_tables", count=len(profiles), duration_ms=_elapsed_ms(t0))

        if not profiles:
            run.status = "completed"
            run.insights_found = 0
            run.pass_diagnostics = tracker.results
            run.completed_at = datetime.now(timezone.utc)
            await session.commit()
            return run

        # 2. Find relationships
        t0 = time.monotonic()
        logger.info("Discovery: finding relationships...")
        try:
            relationships = await find_relationships(session, profiles)
            logger.info("Discovery: found %d relationships", len(relationships))
            tracker.record("find_relationships", count=len(relationships), duration_ms=_elapsed_ms(t0))
        except Exception as exc:
            logger.exception("Relationship finding failed, continuing")
            relationships = []
            tracker.record("find_relationships", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

        # 3. Detect anomalies
        t0 = time.monotonic()
        logger.info("Discovery: detecting anomalies...")
        anomaly_insights = detect_anomalies(profiles)
        logger.info("Discovery: found %d anomaly insights", len(anomaly_insights))
        tracker.record("detect_anomalies", count=len(anomaly_insights), duration_ms=_elapsed_ms(t0))

        # 3b. Check benchmarks against domain knowledge
        t0 = time.monotonic()
        logger.info("Discovery: checking benchmarks...")
        try:
            from business_brain.discovery.benchmark_checker import check_benchmarks
            benchmark_insights = check_benchmarks(profiles)
            logger.info("Discovery: found %d benchmark insights", len(benchmark_insights))
            tracker.record("benchmark_check", count=len(benchmark_insights), duration_ms=_elapsed_ms(t0))
        except Exception as exc:
            logger.exception("Benchmark checking failed, continuing")
            benchmark_insights = []
            tracker.record("benchmark_check", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

        # 4. Discover composites
        t0 = time.monotonic()
        logger.info("Discovery: discovering composites...")
        composite_insights = discover_composites(profiles, relationships)
        logger.info("Discovery: found %d composite insights", len(composite_insights))
        tracker.record("discover_composites", count=len(composite_insights), duration_ms=_elapsed_ms(t0))

        # 5. Find cross-events
        t0 = time.monotonic()
        logger.info("Discovery: finding cross-event correlations...")
        cross_insights = await find_cross_events(session, profiles, relationships)
        logger.info("Discovery: found %d cross-event insights", len(cross_insights))
        tracker.record("cross_event_correlations", count=len(cross_insights), duration_ms=_elapsed_ms(t0))

        # 5b. Detect seasonality patterns
        t0 = time.monotonic()
        logger.info("Discovery: detecting seasonality patterns...")
        seasonality_insights = detect_seasonality(profiles)
        logger.info("Discovery: found %d seasonality insights", len(seasonality_insights))
        tracker.record("seasonality_detection", count=len(seasonality_insights), duration_ms=_elapsed_ms(t0))

        # 5c. Discover correlations from profile data
        t0 = time.monotonic()
        logger.info("Discovery: discovering correlations...")
        try:
            from business_brain.discovery.correlation_discoverer import discover_correlations_from_profiles
            correlation_insights = discover_correlations_from_profiles(profiles)
            logger.info("Discovery: found %d correlation insights", len(correlation_insights))
            tracker.record("correlation_discovery", count=len(correlation_insights), duration_ms=_elapsed_ms(t0))
        except Exception as exc:
            logger.exception("Correlation discovery failed, continuing")
            correlation_insights = []
            tracker.record("correlation_discovery", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

        # 5d. Run domain-specific analysis (heat, material balance, quality, power, etc.)
        t0 = time.monotonic()
        logger.info("Discovery: running domain-specific analysis...")
        try:
            from business_brain.discovery.domain_dispatcher import run_domain_analysis
            domain_insights = await run_domain_analysis(session, profiles)
            logger.info("Discovery: found %d domain insights", len(domain_insights))
            tracker.record("domain_analysis", count=len(domain_insights), duration_ms=_elapsed_ms(t0))
        except Exception as exc:
            logger.exception("Domain analysis failed, continuing")
            domain_insights = []
            tracker.record("domain_analysis", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

        # 5e. Entity performance comparison (operational gaps: who's underperforming)
        # Uses SQL GROUP BY on full data for accuracy, falls back to sample-based.
        # Per-pair savepoints are handled inside discover_entity_performance_sql.
        t0 = time.monotonic()
        logger.info("Discovery: comparing entity performance (SQL-backed)...")
        entity_insights = []
        try:
            from business_brain.discovery.entity_performance import discover_entity_performance_sql
            entity_insights = await discover_entity_performance_sql(session, profiles)
            logger.info("Discovery: found %d entity performance insights (SQL)", len(entity_insights))
            tracker.record("entity_performance_sql", count=len(entity_insights), duration_ms=_elapsed_ms(t0))
        except Exception as exc:
            logger.exception("SQL entity performance failed, falling back to sample-based")
            tracker.record("entity_performance_sql", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))
            # Recover session so subsequent passes aren't poisoned
            try:
                await session.rollback()
            except Exception:
                pass

        # Fallback: sample-based entity performance if SQL produced nothing
        if not entity_insights:
            t0 = time.monotonic()
            try:
                from business_brain.discovery.entity_performance import discover_entity_performance
                entity_insights = discover_entity_performance(profiles)
                logger.info("Discovery: found %d entity performance insights (sample)", len(entity_insights))
                tracker.record("entity_performance_sample", count=len(entity_insights), duration_ms=_elapsed_ms(t0))
            except Exception as exc:
                logger.exception("Entity performance comparison failed, continuing")
                entity_insights = []
                tracker.record("entity_performance_sample", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

        # 5f. SQL-backed correlation (full data CORR()), falls back to sample-based.
        # Per-table savepoints are handled inside discover_correlations_sql.
        sql_corr_insights: list = []
        t0 = time.monotonic()
        logger.info("Discovery: computing correlations (SQL-backed)...")
        try:
            from business_brain.discovery.correlation_discoverer import discover_correlations_sql
            sql_corr_insights = await discover_correlations_sql(session, profiles)
            logger.info("Discovery: found %d correlation insights (SQL)", len(sql_corr_insights))
            tracker.record("correlation_sql", count=len(sql_corr_insights), duration_ms=_elapsed_ms(t0))
            # If SQL found correlations, use those instead of sample-based
            if sql_corr_insights:
                correlation_insights = sql_corr_insights
        except Exception as exc:
            logger.exception("SQL correlation discovery failed, using sample-based results")
            tracker.record("correlation_sql", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))
            # Recover session so subsequent passes aren't poisoned
            try:
                await session.rollback()
            except Exception:
                pass

        # 6. Combine all fast-pass insights
        all_insights = (
            anomaly_insights + benchmark_insights + composite_insights + cross_insights
            + seasonality_insights + correlation_insights + domain_insights
            + entity_insights
        )

        # 7. Quality gate + dedup + persist
        t0 = time.monotonic()
        logger.info("Discovery: committing %d fast-pass insights...", len(all_insights))
        try:
            from business_brain.discovery.dedup import compute_insight_key, deduplicate_insights

            reinforcement_weights = None
            try:
                from business_brain.discovery.reinforcement_loop import get_latest_weights
                reinforcement_weights = await get_latest_weights(session)
            except Exception:
                pass

            gated = apply_quality_gate(all_insights, profiles, reinforcement_weights=reinforcement_weights)

            # Dedup against existing DB records (exclude dismissed —
            # dismissed insights should be re-discoverable after purge)
            existing_result = await session.execute(
                select(Insight.source_tables, Insight.source_columns, Insight.insight_type, Insight.composite_template)
                .where(Insight.status != "dismissed")
            )
            existing_keys = set()
            for row in existing_result.fetchall():
                dummy = Insight()
                dummy.insight_type = row[2]
                dummy.source_tables = row[0]
                dummy.source_columns = row[1]
                dummy.composite_template = row[3]
                existing_keys.add(compute_insight_key(dummy))

            gated = deduplicate_insights(gated, existing_keys)

            for insight in gated:
                insight.discovery_run_id = run.id
                session.add(insight)

            run.insights_found = len(gated)
            tracker.record("persist_insights", count=len(gated), duration_ms=_elapsed_ms(t0))
        except Exception as exc:
            logger.exception("Insight persist failed")
            gated = []
            tracker.record("persist_insights", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

        # 8. Complete the fast phase
        run.status = "completed"
        run.pass_diagnostics = tracker.results
        run.completed_at = datetime.now(timezone.utc)
        await session.commit()

        failed = tracker.failed_count
        logger.info(
            "Discovery fast phase done: %d tables, %d insights, %d/%d passes failed",
            run.tables_scanned,
            run.insights_found,
            failed,
            len(tracker.results),
        )

    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)
        run.pass_diagnostics = tracker.results
        run.completed_at = datetime.now(timezone.utc)
        await session.commit()
        logger.exception("Discovery run failed")
        raise

    return run


async def run_discovery_enrich(
    session: AsyncSession,
    run_id: str | None = None,
) -> dict:
    """Run slow enrichment passes (LLM-dependent) as a separate call.

    This is designed to be called AFTER run_discovery() completes.
    Each pass commits independently so partial progress is never lost.

    Passes: narratives, sanctity, pattern memory, duplicate detection,
    schema changes, data freshness, alerts, precomputation, reinforcement.
    """
    tracker = _PassTracker()

    # Load profiles and existing insights for this run
    profiles_result = await session.execute(select(TableProfile))
    profiles = list(profiles_result.scalars().all())

    if not profiles:
        return {"status": "skipped", "reason": "no_profiles", "passes": []}

    # Load the run record if provided (to update diagnostics)
    run = None
    if run_id:
        run_result = await session.execute(
            select(DiscoveryRun).where(DiscoveryRun.id == run_id)
        )
        run = run_result.scalar_one_or_none()

    # Load insights from the fast phase for narrative building
    insight_result = await session.execute(
        select(Insight).order_by(Insight.discovered_at.desc()).limit(50)
    )
    recent_insights = list(insight_result.scalars().all())

    all_new_insights: list[Insight] = []

    # ── Narrative building ──
    if len(recent_insights) >= 2:
        t0 = time.monotonic()
        logger.info("Enrich: building narratives...")
        try:
            story_insights = await build_narratives(recent_insights)
            all_new_insights.extend(story_insights)
            logger.info("Enrich: generated %d narrative stories", len(story_insights))
            tracker.record("narrative_building", count=len(story_insights), duration_ms=_elapsed_ms(t0))
        except Exception as exc:
            logger.exception("Narrative building failed, continuing")
            tracker.record("narrative_building", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

    # ── Sanctity checks ──
    t0 = time.monotonic()
    logger.info("Enrich: running sanctity checks...")
    try:
        from business_brain.discovery.sanctity_engine import run_sanctity_check
        sanctity_summary = await run_sanctity_check(session, profiles)
        logger.info("Enrich: sanctity check found %d issues", sanctity_summary.get("total", 0))
        tracker.record("sanctity_check", count=sanctity_summary.get("total", 0), duration_ms=_elapsed_ms(t0))
    except Exception as exc:
        logger.exception("Sanctity check failed, continuing")
        tracker.record("sanctity_check", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

    # ── Pattern memory ──
    t0 = time.monotonic()
    logger.info("Enrich: checking pattern memory...")
    try:
        from business_brain.discovery.pattern_memory import check_patterns
        pattern_match_count = 0
        for profile in profiles:
            matches = await check_patterns(session, profile.table_name)
            if matches:
                pattern_match_count += len(matches)
        tracker.record("pattern_memory", count=pattern_match_count, duration_ms=_elapsed_ms(t0))
    except Exception as exc:
        logger.exception("Pattern matching failed, continuing")
        tracker.record("pattern_memory", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

    # ── Duplicate detection ──
    t0 = time.monotonic()
    logger.info("Enrich: detecting duplicate sources...")
    try:
        from business_brain.discovery.format_detector import detect_duplicate_sources
        duplicates = await detect_duplicate_sources(session, profiles)
        dup_count = len(duplicates) if duplicates else 0
        tracker.record("duplicate_detection", count=dup_count, duration_ms=_elapsed_ms(t0))
    except Exception as exc:
        logger.exception("Duplicate source detection failed, continuing")
        tracker.record("duplicate_detection", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

    # ── Schema change detection ──
    t0 = time.monotonic()
    logger.info("Enrich: detecting schema changes...")
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
            all_new_insights.extend(schema_insights)
        tracker.record("schema_change_detection", count=len(schema_insights) if schema_insights else 0, duration_ms=_elapsed_ms(t0))
    except Exception as exc:
        logger.exception("Schema change detection failed, continuing")
        tracker.record("schema_change_detection", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

    # ── Data freshness ──
    # NOTE: Stale data detection requires comparing CURRENT profiles against
    # PREVIOUS-run profiles. Since the profiler updates rows in place (one row
    # per table_name), loading profiles for the same table_names always returns
    # the *current* profiles — making the comparison always self==self. This
    # produced false "stale data" insights every single run.
    #
    # Proper fix: store prev_data_hash on the profile before overwriting, or
    # use DiscoveryRun snapshots. For now: skip this pass entirely.
    tracker.record("data_freshness", count=0, duration_ms=0)

    # ── Persist new insights from enrichment ──
    new_count = 0
    if all_new_insights:
        t0 = time.monotonic()
        try:
            from business_brain.discovery.dedup import compute_insight_key, deduplicate_insights

            reinforcement_weights = None
            try:
                from business_brain.discovery.reinforcement_loop import get_latest_weights
                reinforcement_weights = await get_latest_weights(session)
            except Exception:
                pass

            gated = apply_quality_gate(all_new_insights, profiles, reinforcement_weights=reinforcement_weights)

            existing_result = await session.execute(
                select(Insight.source_tables, Insight.source_columns, Insight.insight_type, Insight.composite_template)
                .where(Insight.status != "dismissed")
            )
            existing_keys = set()
            for row in existing_result.fetchall():
                dummy = Insight()
                dummy.insight_type = row[2]
                dummy.source_tables = row[0]
                dummy.source_columns = row[1]
                dummy.composite_template = row[3]
                existing_keys.add(compute_insight_key(dummy))

            gated = deduplicate_insights(gated, existing_keys)

            for insight in gated:
                if run:
                    insight.discovery_run_id = run.id
                session.add(insight)

            new_count = len(gated)
            await session.commit()
            tracker.record("enrich_persist", count=new_count, duration_ms=_elapsed_ms(t0))
        except Exception as exc:
            logger.exception("Enrich persist failed")
            new_count = 0
            await session.rollback()
            tracker.record("enrich_persist", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

    # ── Auto-refresh deployed reports ──
    profiled_tables = {p.table_name for p in profiles}
    await _refresh_affected_reports(session, profiled_tables)

    # ── Alert evaluation ──
    t0 = time.monotonic()
    logger.info("Enrich: evaluating alert rules...")
    try:
        from business_brain.action.alert_engine import evaluate_all_alerts
        alert_events = await evaluate_all_alerts(session)
        tracker.record("alert_evaluation", count=len(alert_events) if alert_events else 0, duration_ms=_elapsed_ms(t0))
    except Exception as exc:
        logger.exception("Alert evaluation failed, continuing")
        tracker.record("alert_evaluation", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

    # ── Precomputation ──
    t0 = time.monotonic()
    logger.info("Enrich: pre-computing top analyses...")
    try:
        from business_brain.discovery.precompute_engine import (
            invalidate_stale,
            run_precomputation,
        )
        await invalidate_stale(session, profiled_tables)
        precomputed = await run_precomputation(session, profiles, max_total=20)
        tracker.record("precomputation", count=len(precomputed), duration_ms=_elapsed_ms(t0))
    except Exception as exc:
        logger.exception("Pre-computation failed, continuing")
        tracker.record("precomputation", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))
        # Recover session so reinforcement weights + final commit succeed
        try:
            await session.rollback()
        except Exception:
            pass

    # ── Reinforcement weights ──
    t0 = time.monotonic()
    logger.info("Enrich: updating reinforcement weights...")
    try:
        from business_brain.discovery.reinforcement_loop import update_weights
        rl_record = await update_weights(session, discovery_run_id=run.id if run else None)
        if rl_record:
            logger.info("Enrich: reinforcement weights updated to v%d", rl_record.version)
        tracker.record("reinforcement_update", duration_ms=_elapsed_ms(t0))
    except Exception as exc:
        logger.exception("Reinforcement weight update failed, continuing")
        tracker.record("reinforcement_update", status="failed", error=str(exc), error_type=_classify_error(exc), duration_ms=_elapsed_ms(t0))

    # Update run record with enrichment diagnostics
    if run:
        existing_diags = run.pass_diagnostics or []
        run.pass_diagnostics = existing_diags + tracker.results
        run.insights_found = (run.insights_found or 0) + new_count
        await session.commit()

    # Trigger analysis engine (optional, non-blocking)
    try:
        from business_brain.analysis.integration import run_analysis_after_discovery
        changed = [p.table_name for p in profiles]
        await run_analysis_after_discovery(session, changed_tables=changed)
    except Exception:
        logger.debug("Analysis trigger skipped (engine not available or failed)")

    logger.info(
        "Enrich phase done: %d new insights, %d/%d passes failed",
        new_count,
        tracker.failed_count,
        len(tracker.results),
    )

    return {
        "status": "completed",
        "new_insights": new_count,
        "passes": tracker.results,
        "failed_passes": tracker.failed_count,
    }


import re as _re

# Phrases that indicate an insight is just describing data structure, not a real finding
_WEAK_PHRASES = [
    r"analysis is possible",
    r"analysis is available",
    r"analysis available",
    r"comparison is available",
    r"further investigation",
    r"warrants? further",
    r"needs? further",
    r"requires? further",
    r"seems? to contain",
    r"appears? to contain",
    r"contains? general information",
    r"contains? (?:descriptive|text) (?:data|information)",
    r"understanding the context",
    r"is important for relating",
    r"may indicate",
    r"could be",
    r"might highlight",
    r"can create a\b",
    r"find hidden relationships",
    r"run (?:full |correlation )?analysis",
]
_WEAK_PATTERN = _re.compile("|".join(_WEAK_PHRASES), _re.IGNORECASE)


def _passes_quality_gate(insight: Insight) -> bool:
    """Reject insights that are just data descriptions or 'analysis possible' flags.

    A real insight must contain a specific finding, not just describe what's in the data.
    """
    text = f"{insight.title or ''} {insight.description or ''} {insight.narrative or ''}"

    # Reject if description matches weak patterns
    if _WEAK_PATTERN.search(text):
        logger.debug("Quality gate rejected (weak language): %s", insight.title)
        return False

    # Reject story-type insights with no numbers in them
    if insight.insight_type == "story":
        has_number = bool(_re.search(r"\d+\.?\d*%|\d{2,}|₹|Rs\.?\s*\d", text))
        if not has_number:
            logger.debug("Quality gate rejected (story with no numbers): %s", insight.title)
            return False

    return True


def _apply_scoring(insight: Insight) -> None:
    """Apply the scoring formula: severity*40 + cross_table*30 + magnitude*30.

    Domain analysis insights get a floor score of 45 because they contain
    quantified findings from specialized modules (always more valuable than
    generic statistical observations).
    """
    severity_weights = {"critical": 1.0, "warning": 0.6, "info": 0.3}
    severity_score = severity_weights.get(insight.severity, 0.3) * 40

    # Cross-table bonus
    source_tables = insight.source_tables or []
    cross_table_bonus = 30 if len(source_tables) > 1 else 0

    # Change magnitude (use existing impact_score as base)
    magnitude = ((insight.impact_score or 0) / 100) * 30

    score = int(severity_score + cross_table_bonus + magnitude)

    # Domain/operational analysis floor — these are always quantified, specific findings
    if insight.insight_type in ("domain_analysis", "dynamic_analysis", "entity_performance"):
        score = max(score, 45)

    insight.impact_score = min(score, 100)


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
