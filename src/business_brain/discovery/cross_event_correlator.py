"""Cross-table event-to-metric correlation detection."""

from __future__ import annotations

import logging
import re
import uuid

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import (
    DiscoveredRelationship,
    Insight,
    TableProfile,
)

logger = logging.getLogger(__name__)


async def find_cross_events(
    session: AsyncSession,
    profiles: list[TableProfile],
    relationships: list[DiscoveredRelationship],
) -> list[Insight]:
    """Find cases where events in one table correlate with metrics in another."""
    insights: list[Insight] = []

    if len(profiles) < 2 or not relationships:
        return insights

    # Build lookup
    profile_map = {p.table_name: p for p in profiles}

    for rel in relationships:
        try:
            result = await _check_correlation(session, rel, profile_map)
            if result:
                insights.extend(result)
        except Exception:
            logger.exception(
                "Cross-event check failed for %s <-> %s",
                rel.table_a,
                rel.table_b,
            )
            await session.rollback()

    return insights


async def _check_correlation(
    session: AsyncSession,
    rel: DiscoveredRelationship,
    profile_map: dict[str, TableProfile],
) -> list[Insight]:
    """Check if events in one table correlate with metrics in the other."""
    results: list[Insight] = []

    prof_a = profile_map.get(rel.table_a)
    prof_b = profile_map.get(rel.table_b)
    if not prof_a or not prof_b:
        return results

    # Try both directions: A has events, B has metrics, and vice versa
    for event_prof, metric_prof, entity_col_event, entity_col_metric in [
        (prof_a, prof_b, rel.column_a, rel.column_b),
        (prof_b, prof_a, rel.column_b, rel.column_a),
    ]:
        event_cols = _find_event_columns(event_prof)
        metric_cols = _find_metric_columns(metric_prof)

        if not event_cols or not metric_cols:
            continue

        for event_col in event_cols[:2]:  # limit to top 2 event columns
            for metric_col in metric_cols[:2]:  # limit to top 2 metric columns
                try:
                    insight = await _run_correlation(
                        session,
                        event_prof.table_name, event_col, entity_col_event,
                        metric_prof.table_name, metric_col, entity_col_metric,
                    )
                    if insight:
                        results.append(insight)
                except Exception:
                    logger.debug(
                        "Correlation check failed: %s.%s <-> %s.%s",
                        event_prof.table_name, event_col,
                        metric_prof.table_name, metric_col,
                    )
                    await session.rollback()

    return results


def _find_event_columns(profile: TableProfile) -> list[str]:
    """Find boolean/categorical columns that represent events."""
    cls = profile.column_classification or {}
    cols = cls.get("columns", {})
    events = []
    for col, info in cols.items():
        sem = info.get("semantic_type", "")
        if sem in ("boolean", "categorical") and info.get("cardinality", 0) <= 10:
            events.append(col)
    return events


def _find_metric_columns(profile: TableProfile) -> list[str]:
    """Find numeric columns that represent measurable metrics."""
    cls = profile.column_classification or {}
    cols = cls.get("columns", {})
    metrics = []
    for col, info in cols.items():
        sem = info.get("semantic_type", "")
        if sem in ("numeric_metric", "numeric_currency", "numeric_percentage"):
            metrics.append(col)
    return metrics


async def _run_correlation(
    session: AsyncSession,
    event_table: str,
    event_col: str,
    entity_col_event: str,
    metric_table: str,
    metric_col: str,
    entity_col_metric: str,
) -> Insight | None:
    """Run a specific cross-table event-metric correlation check."""
    safe = lambda s: re.sub(r"[^a-zA-Z0-9_]", "", s)

    se_table = safe(event_table)
    se_col = safe(event_col)
    se_entity = safe(entity_col_event)
    sm_table = safe(metric_table)
    sm_col = safe(metric_col)
    sm_entity = safe(entity_col_metric)

    query = f"""
        SELECT a."{se_col}" AS event_value,
               AVG(b."{sm_col}"::numeric) AS avg_metric,
               COUNT(*) AS cnt
        FROM "{se_table}" a
        JOIN "{sm_table}" b ON a."{se_entity}"::text = b."{sm_entity}"::text
        GROUP BY a."{se_col}"
        HAVING COUNT(*) >= 2
        ORDER BY avg_metric DESC
    """

    try:
        result = await session.execute(sql_text(query))
        rows = [dict(r._mapping) for r in result.fetchall()]
    except Exception:
        return None

    if len(rows) < 2:
        return None

    # Compare groups — check if difference > 15%
    avgs = [(r["event_value"], float(r["avg_metric"]), int(r["cnt"])) for r in rows]
    max_avg = max(a[1] for a in avgs)
    min_avg = min(a[1] for a in avgs)

    if min_avg == 0:
        pct_diff = 100 if max_avg > 0 else 0
    else:
        pct_diff = abs(max_avg - min_avg) / abs(min_avg) * 100

    if pct_diff < 15:
        return None

    # Find which event value has highest and lowest metric
    best = max(avgs, key=lambda x: x[1])
    worst = min(avgs, key=lambda x: x[1])

    direction = "higher" if best[1] > worst[1] else "lower"

    return Insight(
        id=str(uuid.uuid4()),
        insight_type="cross_event",
        severity="warning" if pct_diff > 30 else "info",
        impact_score=min(int(pct_diff), 90),
        title=f"{event_col} in {event_table} correlates with {metric_col} in {metric_table}",
        description=(
            f"When {entity_col_event} has {event_col}='{best[0]}' in {event_table}, "
            f"{metric_col} in {metric_table} is {round(pct_diff, 1)}% {direction} "
            f"(avg {round(best[1], 2)}) vs {event_col}='{worst[0]}' (avg {round(worst[1], 2)})."
        ),
        source_tables=[event_table, metric_table],
        source_columns=[event_col, metric_col, entity_col_event],
        evidence={
            "query": query.strip(),
            "groups": [
                {"event": str(a[0]), "avg_metric": round(a[1], 2), "count": a[2]}
                for a in avgs
            ],
            "pct_difference": round(pct_diff, 1),
            "chart_spec": {
                "type": "bar",
                "x": event_col,
                "y": [f"avg_{metric_col}"],
                "title": f"{metric_col} by {event_col}",
            },
        },
        suggested_actions=[
            f"Investigate why {event_col}='{best[0]}' correlates with {direction} {metric_col}",
            f"Check if this correlation implies causation or is coincidental",
        ],
    )
