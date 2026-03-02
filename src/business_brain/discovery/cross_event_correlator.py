"""Cross-table event-to-metric correlation detection.

Performance: one JOIN per (relationship × direction) fetches ALL event and metric
columns at once, then computes correlations in Python.  No per-column-pair queries.
"""

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

# Limit rows fetched per JOIN to keep memory bounded
_ROW_LIMIT = 2000


def _humanize_col(name: str) -> str:
    """Convert column name to human-readable form: downtime_hrs → Downtime Hours."""
    return name.replace("_", " ").title()


def _humanize_table(name: str) -> str:
    """Convert table name to human-readable form: 08_nary_segments → Nary Segments."""
    # Strip leading numeric prefix like "01_", "08_"
    cleaned = re.sub(r"^\d+[_\s]*", "", name)
    return cleaned.replace("_", " ").title() if cleaned else name.replace("_", " ").title()


def filter_by_confidence(
    relationships: list[DiscoveredRelationship],
    min_confidence: float = 0.5,
) -> list[DiscoveredRelationship]:
    """Filter relationships by minimum confidence threshold."""
    return [r for r in relationships if (r.confidence or 0) >= min_confidence]


async def find_cross_events(
    session: AsyncSession,
    profiles: list[TableProfile],
    relationships: list[DiscoveredRelationship],
    min_confidence: float = 0.5,
) -> list[Insight]:
    """Find cases where events in one table correlate with metrics in another.

    For each relationship, does ONE bulk JOIN per direction (not per column pair).
    All column-pair analysis happens in Python after the fetch.
    """
    insights: list[Insight] = []

    if len(profiles) < 2 or not relationships:
        return insights

    relationships = filter_by_confidence(relationships, min_confidence)
    if not relationships:
        return insights

    profile_map = {p.table_name: p for p in profiles}

    for rel in relationships:
        try:
            async with session.begin_nested():
                result = await _check_correlation(session, rel, profile_map)
                if result:
                    insights.extend(result)
        except Exception:
            logger.exception(
                "Cross-event check failed for %s <-> %s",
                rel.table_a,
                rel.table_b,
            )

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

    # Try both directions: A has events → B has metrics, and vice versa
    for event_prof, metric_prof, entity_col_event, entity_col_metric in [
        (prof_a, prof_b, rel.column_a, rel.column_b),
        (prof_b, prof_a, rel.column_b, rel.column_a),
    ]:
        event_cols = _find_event_columns(event_prof)
        metric_cols = _find_metric_columns(metric_prof)

        if not event_cols or not metric_cols:
            continue

        # ONE bulk JOIN fetches all event + metric columns at once
        rows = await _bulk_fetch(
            session,
            event_prof.table_name, event_cols, entity_col_event,
            metric_prof.table_name, metric_cols, entity_col_metric,
        )
        if not rows:
            continue

        # Analyze every (event_col, metric_col) pair in Python
        for event_col in event_cols:
            for metric_col in metric_cols:
                insight = _analyze_pair(
                    rows,
                    event_prof.table_name, event_col, entity_col_event,
                    metric_prof.table_name, metric_col, entity_col_metric,
                )
                if insight:
                    results.append(insight)

    return results


def _find_event_columns(profile: TableProfile) -> list[str]:
    """Find boolean/categorical columns that represent events.

    Filters to low-cardinality categoricals (≤6 unique values) to avoid
    noise from high-cardinality columns like batch numbers, IDs, etc.
    """
    cls = profile.column_classification or {}
    cols = cls.get("columns", {})
    events = []
    for col, info in cols.items():
        sem = info.get("semantic_type", "")
        cardinality = info.get("cardinality", 0)
        col_lower = col.lower()

        # Skip identifier-like columns even if classified as categorical
        if any(kw in col_lower for kw in ("batch", "id", "number", "no", "code", "serial")):
            continue

        if sem in ("boolean", "categorical") and cardinality <= 6:
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


# ---------------------------------------------------------------------------
# Bulk fetch: ONE query per (relationship × direction)
# ---------------------------------------------------------------------------


async def _bulk_fetch(
    session: AsyncSession,
    event_table: str,
    event_cols: list[str],
    entity_col_event: str,
    metric_table: str,
    metric_cols: list[str],
    entity_col_metric: str,
) -> list[dict]:
    """Fetch all event and metric columns in a single JOIN query.

    Returns list of dicts with keys like 'evt__status', 'met__amount', etc.
    """
    safe = lambda s: re.sub(r"[^a-zA-Z0-9_]", "", s)

    se_table = safe(event_table)
    se_entity = safe(entity_col_event)
    sm_table = safe(metric_table)
    sm_entity = safe(entity_col_metric)

    # Build SELECT clause: all event cols + all metric cols
    select_parts = []
    for ec in event_cols:
        sec = safe(ec)
        select_parts.append(f'a."{sec}" AS "evt__{sec}"')
    for mc in metric_cols:
        smc = safe(mc)
        select_parts.append(f'b."{smc}" AS "met__{smc}"')

    select_clause = ", ".join(select_parts)

    query = f"""
        SELECT {select_clause}
        FROM "{se_table}" a
        JOIN "{sm_table}" b ON a."{se_entity}"::text = b."{sm_entity}"::text
        LIMIT {_ROW_LIMIT}
    """

    try:
        result = await session.execute(sql_text(query))
        return [dict(r._mapping) for r in result.fetchall()]
    except Exception:
        logger.debug(
            "Bulk fetch failed for %s JOIN %s", event_table, metric_table
        )
        return []


# ---------------------------------------------------------------------------
# Per-pair analysis: pure Python, no SQL
# ---------------------------------------------------------------------------


def _analyze_pair(
    rows: list[dict],
    event_table: str,
    event_col: str,
    entity_col_event: str,
    metric_table: str,
    metric_col: str,
    entity_col_metric: str,
) -> Insight | None:
    """Analyze one (event_col, metric_col) pair from pre-fetched rows."""
    safe_ec = re.sub(r"[^a-zA-Z0-9_]", "", event_col)
    safe_mc = re.sub(r"[^a-zA-Z0-9_]", "", metric_col)

    evt_key = f"evt__{safe_ec}"
    met_key = f"met__{safe_mc}"

    # Group metric values by event value
    groups: dict[str, list[float]] = {}
    for row in rows:
        ev = row.get(evt_key)
        mv = row.get(met_key)
        if ev is None or mv is None:
            continue
        try:
            val = float(mv)
        except (TypeError, ValueError):
            continue
        ev_str = str(ev)
        groups.setdefault(ev_str, []).append(val)

    # Need at least 2 groups with 5+ samples each for statistical significance
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 5}
    if len(valid_groups) < 2:
        return None

    # Compute averages per group
    avgs = [
        (ev_val, sum(vals) / len(vals), len(vals))
        for ev_val, vals in valid_groups.items()
    ]

    max_avg = max(a[1] for a in avgs)
    min_avg = min(a[1] for a in avgs)

    if min_avg == 0:
        pct_diff = 100 if max_avg > 0 else 0
    else:
        pct_diff = abs(max_avg - min_avg) / abs(min_avg) * 100

    if pct_diff < 15:
        return None

    best = max(avgs, key=lambda x: x[1])
    worst = min(avgs, key=lambda x: x[1])
    direction = "higher" if best[1] > worst[1] else "lower"

    # Human-readable names for titles and descriptions
    h_event = _humanize_col(event_col)
    h_metric = _humanize_col(metric_col)
    h_event_table = _humanize_table(event_table)
    h_metric_table = _humanize_table(metric_table)

    return Insight(
        id=str(uuid.uuid4()),
        insight_type="cross_event",
        severity="warning" if pct_diff > 30 else "info",
        impact_score=min(int(pct_diff), 90),
        title=(
            f"{h_event} {best[0]} has {round(pct_diff)}% {direction} {h_metric}"
        ),
        description=(
            f"When {h_event} is {best[0]} (from {h_event_table}), "
            f"average {h_metric} (from {h_metric_table}) is {round(pct_diff, 1)}% {direction} "
            f"at {round(best[1], 2):g} compared to {h_event} {worst[0]} at {round(worst[1], 2):g}. "
            f"This pattern connects data across your {h_event_table} and {h_metric_table} tables."
        ),
        source_tables=[event_table, metric_table],
        source_columns=[event_col, metric_col, entity_col_event],
        evidence={
            "groups": [
                {"event": str(a[0]), "avg_metric": round(a[1], 2), "count": a[2]}
                for a in avgs
            ],
            "pct_difference": round(pct_diff, 1),
            "chart_spec": {
                "type": "bar",
                "x": event_col,
                "y": [f"avg_{metric_col}"],
                "title": f"{h_metric} by {h_event}",
            },
        },
        suggested_actions=[
            f"Investigate why {h_event} {best[0]} shows {direction} {h_metric}",
            f"Compare operational differences between {h_event} {best[0]} and {worst[0]}",
        ],
    )
