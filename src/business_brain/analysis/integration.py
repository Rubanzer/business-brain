"""Integration bridge — connects analysis engine to existing discovery/feed.

Two key functions:
- persist_to_feed(): Converts AnalysisResult → Insight for the existing feed.
- run_analysis_after_discovery(): Triggers analysis on changed tables only.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.agents.orchestrator import run_analysis
from business_brain.analysis.models import AnalysisResult
from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Analysis → Feed bridge
# ---------------------------------------------------------------------------

_OPERATION_TO_INSIGHT_TYPE = {
    "DESCRIBE": "trend",
    "DESCRIBE_CATEGORICAL": "trend",
    "CORRELATE": "correlation",
    "RANK": "composite",
    "DETECT_ANOMALY": "anomaly",
    "FORECAST": "trend",
    "ATTRIBUTE": "composite",
}

_SCORE_TO_SEVERITY = [
    (0.8, "critical"),
    (0.6, "warning"),
    (0.0, "info"),
]


def _score_to_severity(score: float) -> str:
    for threshold, severity in _SCORE_TO_SEVERITY:
        if score >= threshold:
            return severity
    return "info"


def _build_insight_title(result: AnalysisResult) -> str:
    """Build a concise title from the analysis result."""
    op = result.operation_type
    target = ", ".join(result.target or [])
    segs = ", ".join(result.segmenters) if result.segmenters else ""

    if op == "CORRELATE":
        return f"Correlation: {target}"
    if op == "RANK":
        if segs:
            return f"{target} by {segs}"
        return f"Ranking: {target}"
    if op == "DETECT_ANOMALY":
        return f"Anomaly in {target}"
    if op == "DESCRIBE":
        return f"Distribution: {target}"
    if op == "DESCRIBE_CATEGORICAL":
        return f"Categories: {target}"
    return f"{op}: {target}"


def _build_insight_description(result: AnalysisResult) -> str:
    """Build a description from result data."""
    data = result.result_data or {}
    parts = []

    if result.operation_type == "CORRELATE":
        r = data.get("pearson_r", 0)
        p = data.get("pearson_p", 1)
        parts.append(f"Pearson r={r:.3f} (p={p:.4f})")

    elif result.operation_type == "RANK":
        comp = data.get("comparison", {})
        if comp:
            parts.append(f"Effect: p={comp.get('p_value', 'N/A')}")
        ranked = data.get("ranked", [])
        if ranked:
            parts.append(f"Top: {ranked[0]}")

    elif result.operation_type == "DETECT_ANOMALY":
        count = data.get("count", 0)
        total = data.get("total", 0)
        parts.append(f"{count} anomalies in {total} values")

    elif result.operation_type in ("DESCRIBE", "DESCRIBE_CATEGORICAL"):
        stats = data.get("stats", {})
        if "mean" in stats:
            parts.append(f"Mean={stats['mean']:.2f}, Stdev={stats.get('stdev', 0):.2f}")
        if "unique" in stats:
            parts.append(f"{stats['unique']} unique values")

    if result.segmenters:
        parts.append(f"Segmented by: {', '.join(result.segmenters)}")

    return "; ".join(parts) if parts else f"Interestingness: {result.interestingness_score:.2f}"


async def persist_to_feed(
    session: AsyncSession,
    results: list[AnalysisResult],
    run_id: str,
    min_score: float = 0.4,
) -> list[Insight]:
    """Convert high-scoring AnalysisResults into Insight records for the existing feed."""
    insights = []

    for result in results:
        if result.final_score < min_score:
            continue
        if result.quality_verdict == "UNRELIABLE":
            continue

        insight = Insight(
            insight_type=_OPERATION_TO_INSIGHT_TYPE.get(result.operation_type, "trend"),
            severity=_score_to_severity(result.final_score),
            impact_score=int(result.final_score * 100),
            quality_score=int(result.final_score * 100),
            title=_build_insight_title(result),
            description=_build_insight_description(result),
            source_tables=[result.table_name],
            source_columns=result.target,
            evidence={
                "analysis_result_id": result.id,
                "run_id": run_id,
                "operation": result.operation_type,
                "interestingness": result.interestingness_score,
                "tier": result.tier,
            },
            discovery_run_id=run_id,
        )
        session.add(insight)
        insights.append(insight)

    if insights:
        await session.flush()
        logger.info("Persisted %d analysis findings to insight feed", len(insights))

    return insights


# ---------------------------------------------------------------------------
# Discovery → Analysis trigger
# ---------------------------------------------------------------------------


async def run_analysis_after_discovery(
    session: AsyncSession,
    changed_tables: list[str] | None = None,
) -> None:
    """Trigger analysis on tables that changed since last run.

    If changed_tables is not provided, detect changes via data_hash comparison.
    """
    if not changed_tables:
        # Detect tables with new data_hash (changed since last analysis)
        result = await session.execute(
            select(TableProfile.table_name).where(TableProfile.data_hash.isnot(None))
        )
        changed_tables = [r[0] for r in result.all()]

    if not changed_tables:
        logger.info("No changed tables for analysis trigger")
        return

    logger.info("Analysis trigger: running on %d changed tables", len(changed_tables))

    try:
        run, results = await run_analysis(
            session=session,
            table_names=changed_tables,
            situation_type="MONITORING",
            budget={"budgeted_tier_limits": {2: 30, 3: 15, 4: 10}},
        )

        # Persist top findings to feed
        await persist_to_feed(session, results, run.id)
        await session.commit()
    except Exception:
        logger.exception("Analysis trigger failed")
        await session.rollback()
