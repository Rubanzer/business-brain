"""Entity performance comparator — finds who's underperforming relative to peers.

This is the "operational low-hanging fruit" discoverer. For every table with
categorical entities (sales reps, shifts, machines, suppliers, zones) and
numeric metrics, it:

1. Groups by entity, computes per-entity aggregates
2. Compares each entity to the group average and to peers
3. Flags entities that diverge: dropped when others gained, trailing behind, etc.

Produces insights like:
  - "Sales Rep Rajesh underperformed by 23% vs team avg (₹45L vs ₹58L)"
  - "Shift C dispatch volume 18% below Shift A and B"
  - "Supplier X quality dropped to 87% while peers averaged 94%"
"""

from __future__ import annotations

import logging
import re
import statistics
import uuid
from collections import defaultdict

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)


def discover_entity_performance(profiles: list[TableProfile]) -> list[Insight]:
    """Scan all profiled tables for entity-level performance gaps.

    For each table with categorical + numeric columns, compare entities
    against peers to find operational gaps.
    """
    insights: list[Insight] = []

    for profile in profiles:
        try:
            insights.extend(_analyze_table(profile))
        except Exception:
            logger.exception("Entity performance scan failed for %s", profile.table_name)

    return insights


def _analyze_table(profile: TableProfile) -> list[Insight]:
    """Analyze a single table for entity performance gaps."""
    results: list[Insight] = []
    cls = profile.column_classification
    if not cls or "columns" not in cls:
        return results

    cols = cls["columns"]
    row_count = profile.row_count or 0

    if row_count < 10:
        return results

    # Find categorical columns (entities to group by)
    cat_cols = [
        (c, info) for c, info in cols.items()
        if info.get("semantic_type") == "categorical"
        and 2 <= info.get("cardinality", 0) <= 50  # Reasonable entity count
    ]

    # Find numeric columns (metrics to compare)
    num_cols = [
        (c, info) for c, info in cols.items()
        if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
        and info.get("stats") and info["stats"].get("stdev", 0) > 0
    ]

    if not cat_cols or not num_cols:
        return results

    # Find temporal columns for trend-based comparison
    temp_cols = [
        c for c, info in cols.items()
        if info.get("semantic_type") == "temporal"
    ]

    # For each categorical × numeric combo, look for performance gaps
    for cat_col, cat_info in cat_cols:
        for num_col, num_info in num_cols:
            insight = _compare_entities(
                profile, cols, cat_col, cat_info, num_col, num_info,
            )
            if insight:
                results.append(insight)

    return results


def _compare_entities(
    profile: TableProfile,
    cols: dict,
    cat_col: str,
    cat_info: dict,
    num_col: str,
    num_info: dict,
) -> Insight | None:
    """Compare entities within a category on a numeric metric.

    Uses sample values to estimate per-entity averages and find gaps.
    Returns an insight only if there's a meaningful performance difference.
    """
    # We need sample values from both columns
    cat_samples = cat_info.get("sample_values", [])
    num_samples = num_info.get("sample_values", [])

    if len(cat_samples) < 6 or len(num_samples) < 6:
        return None

    n = min(len(cat_samples), len(num_samples))
    cat_samples = cat_samples[:n]
    num_samples = num_samples[:n]

    # Parse numeric values — skip pairs where either is None (row-aligned null handling)
    entity_values: dict[str, list[float]] = defaultdict(list)
    for cat_val, num_val in zip(cat_samples, num_samples):
        if cat_val is None or num_val is None:
            continue
        try:
            v = float(str(num_val).replace(",", ""))
            entity = str(cat_val).strip()
            if entity and entity != "None":
                entity_values[entity].append(v)
        except (ValueError, TypeError):
            continue

    if len(entity_values) < 2:
        return None

    # Compute per-entity stats
    entity_stats: list[dict] = []
    for entity, values in entity_values.items():
        if len(values) < 2:
            continue
        entity_stats.append({
            "entity": entity,
            "mean": statistics.mean(values),
            "count": len(values),
            "total": sum(values),
        })

    if len(entity_stats) < 2:
        return None

    # Overall average
    all_values = [v for vals in entity_values.values() for v in vals]
    overall_mean = statistics.mean(all_values)
    if overall_mean == 0:
        return None

    # Sort by mean (descending)
    entity_stats.sort(key=lambda x: x["mean"], reverse=True)

    best = entity_stats[0]
    worst = entity_stats[-1]

    # Calculate gap
    gap_pct = ((best["mean"] - worst["mean"]) / overall_mean) * 100

    # Only report if gap is significant (>15%)
    if gap_pct < 15:
        return None

    # Build entity ranking for full visibility
    ranking_parts = []
    for i, es in enumerate(entity_stats):
        vs_avg = ((es["mean"] - overall_mean) / overall_mean) * 100
        sign = "+" if vs_avg > 0 else ""
        ranking_parts.append(
            f"{i+1}. {es['entity']}: {es['mean']:,.1f} ({sign}{vs_avg:.0f}% vs avg)"
        )

    ranking_str = "; ".join(ranking_parts)

    # Determine what kind of entity this is for better titles
    entity_label = _infer_entity_label(cat_col)
    metric_label = _infer_metric_label(num_col, num_info)

    severity = "warning" if gap_pct > 30 else "info"
    impact = min(int(gap_pct / 2) + 35, 80)

    # Check for specific patterns
    is_currency = num_info.get("semantic_type") == "numeric_currency"
    unit = "₹" if is_currency else ""

    # Humanize table name (strip numeric prefix)
    import re as _re
    h_table = _re.sub(r"^\d+[_\s]*", "", profile.table_name).replace("_", " ").title()
    h_metric = num_col.replace("_", " ").title()

    title = (
        f"{worst['entity']} {metric_label} is {gap_pct:.0f}% below {best['entity']}"
    )
    description = (
        f"{entity_label} {worst['entity']} averages {unit}{worst['mean']:,.1f} on {h_metric}, "
        f"which is {gap_pct:.0f}% below {best['entity']} at {unit}{best['mean']:,.1f}. "
        f"The overall average is {unit}{overall_mean:,.1f}. "
        f"Ranking: {ranking_str}."
    )

    return Insight(
        id=str(uuid.uuid4()),
        insight_type="entity_performance",
        severity=severity,
        impact_score=impact,
        title=title,
        description=description,
        source_tables=[profile.table_name],
        source_columns=[cat_col, num_col],
        evidence={
            "entity_column": cat_col,
            "metric_column": num_col,
            "best_entity": best["entity"],
            "best_value": round(best["mean"], 2),
            "worst_entity": worst["entity"],
            "worst_value": round(worst["mean"], 2),
            "overall_mean": round(overall_mean, 2),
            "gap_pct": round(gap_pct, 1),
            "entity_count": len(entity_stats),
            "full_ranking": [
                {
                    "entity": es["entity"],
                    "mean": round(es["mean"], 2),
                    "count": es["count"],
                    "vs_avg_pct": round(((es["mean"] - overall_mean) / overall_mean) * 100, 1),
                }
                for es in entity_stats
            ],
            "chart_spec": {
                "type": "bar",
                "x": cat_col,
                "y": [num_col],
                "title": f"{num_col} by {cat_col}",
                "highlight": {"worst": worst["entity"], "best": best["entity"]},
            },
            "query": (
                f'SELECT "{cat_col}", AVG("{num_col}") as avg_{num_col}, '
                f'COUNT(*) as count, SUM("{num_col}") as total '
                f'FROM "{profile.table_name}" '
                f'GROUP BY "{cat_col}" ORDER BY avg_{num_col} DESC'
            ),
        },
        suggested_actions=[
            f"Investigate why {worst['entity']} is {gap_pct:.0f}% below {best['entity']} on {h_metric}",
            f"Look at what {best['entity']} does differently and replicate those practices",
            f"Compare {best['entity']} vs {worst['entity']} across all available metrics",
        ],
    )


# ---------------------------------------------------------------------------
# Label inference helpers
# ---------------------------------------------------------------------------

_ENTITY_LABELS = {
    "shift": "Shift",
    "sales": "Sales rep",
    "rep": "Sales rep",
    "agent": "Agent",
    "person": "Person",
    "employee": "Employee",
    "worker": "Worker",
    "operator": "Operator",
    "machine": "Machine",
    "furnace": "Furnace",
    "line": "Production line",
    "supplier": "Supplier",
    "vendor": "Vendor",
    "party": "Party",
    "customer": "Customer",
    "buyer": "Buyer",
    "zone": "Zone",
    "region": "Region",
    "branch": "Branch",
    "department": "Department",
    "team": "Team",
    "product": "Product",
    "grade": "Grade",
    "category": "Category",
    "type": "Type",
}


def _infer_entity_label(col_name: str) -> str:
    """Infer a human-readable label for the entity column."""
    col_lower = col_name.lower()
    for keyword, label in _ENTITY_LABELS.items():
        if keyword in col_lower:
            return label
    return col_name.replace("_", " ").title()


_METRIC_LABELS = {
    "revenue": "revenue",
    "sales": "sales",
    "amount": "amount",
    "output": "output",
    "production": "production",
    "tonnage": "tonnage",
    "weight": "weight",
    "quantity": "quantity",
    "dispatch": "dispatch volume",
    "delivery": "delivery count",
    "quality": "quality score",
    "defect": "defect rate",
    "reject": "rejection rate",
    "rate": "rate",
    "cost": "cost",
    "efficiency": "efficiency",
    "yield": "yield",
    "attendance": "attendance",
    "score": "score",
    "rating": "rating",
}


def _infer_metric_label(col_name: str, info: dict) -> str:
    """Infer a human-readable label for the metric."""
    col_lower = col_name.lower()
    for keyword, label in _METRIC_LABELS.items():
        if keyword in col_lower:
            return label
    return col_name.replace("_", " ").lower()


# ---------------------------------------------------------------------------
# SQL-backed full-data entity performance (async, uses real GROUP BY)
# ---------------------------------------------------------------------------

def _humanize_table(name: str) -> str:
    """Convert table name to human-readable form."""
    cleaned = re.sub(r"^\d+[_\s]*", "", name)
    return cleaned.replace("_", " ").title() if cleaned else name.replace("_", " ").title()


def _safe(name: str) -> str:
    """Strip non-alphanumeric/underscore chars for SQL safety."""
    return re.sub(r"[^a-zA-Z0-9_]", "", name)


async def discover_entity_performance_sql(
    session: AsyncSession,
    profiles: list[TableProfile],
) -> list[Insight]:
    """SQL-backed entity performance — runs GROUP BY on full data.

    This is the *accurate* version: instead of estimating from 100-row samples,
    it runs ``SELECT cat_col, AVG(num_col), COUNT(*) ... GROUP BY cat_col``
    against the actual database for every viable (categorical × numeric) pair.

    Each table is wrapped in its own savepoint so one table's SQL failure
    doesn't poison the session for subsequent tables.
    """
    insights: list[Insight] = []

    for profile in profiles:
        try:
            cls = profile.column_classification
            if not cls or "columns" not in cls:
                continue

            cols = cls["columns"]
            row_count = profile.row_count or 0
            if row_count < 10:
                continue

            cat_cols = [
                (c, info) for c, info in cols.items()
                if info.get("semantic_type") == "categorical"
                and 2 <= info.get("cardinality", 0) <= 50
            ]
            num_cols = [
                (c, info) for c, info in cols.items()
                if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
                and info.get("stats") and info["stats"].get("stdev", 0) > 0
            ]

            if not cat_cols or not num_cols:
                continue

            for cat_col, cat_info in cat_cols:
                for num_col, num_info in num_cols:
                    # Per-pair savepoint: if one query fails, the session
                    # stays clean for the next pair / next table
                    try:
                        async with session.begin_nested():
                            insight = await _sql_compare_entities(
                                session, profile, cat_col, cat_info, num_col, num_info,
                            )
                        if insight:
                            insights.append(insight)
                    except Exception:
                        logger.debug(
                            "SQL entity pair failed: %s.%s×%s",
                            profile.table_name, cat_col, num_col,
                        )
        except Exception:
            logger.exception("SQL entity performance scan failed for %s", profile.table_name)

    return insights


async def _sql_compare_entities(
    session: AsyncSession,
    profile: TableProfile,
    cat_col: str,
    cat_info: dict,
    num_col: str,
    num_info: dict,
) -> Insight | None:
    """Run actual GROUP BY on full data for one (categorical × numeric) pair."""
    s_table = _safe(profile.table_name)
    s_cat = _safe(cat_col)
    s_num = _safe(num_col)

    query = f"""
        SELECT "{s_cat}" AS entity,
               AVG("{s_num}") AS avg_val,
               COUNT(*) AS cnt,
               SUM("{s_num}") AS total
        FROM "{s_table}"
        WHERE "{s_cat}" IS NOT NULL AND "{s_num}" IS NOT NULL
        GROUP BY "{s_cat}"
        HAVING COUNT(*) >= 3
        ORDER BY avg_val DESC
    """

    try:
        result = await session.execute(sql_text(query))
        rows = [dict(r._mapping) for r in result.fetchall()]
    except Exception:
        logger.debug("SQL entity performance query failed for %s.%s", s_table, s_cat)
        return None

    if len(rows) < 2:
        return None

    # Compute overall mean — cast to float because PostgreSQL AVG()/SUM()
    # return decimal.Decimal, and float/Decimal raises TypeError
    all_total = float(sum(float(r["total"]) for r in rows if r["total"] is not None))
    all_count = int(sum(int(r["cnt"]) for r in rows if r["cnt"]))
    if all_count == 0:
        return None
    overall_mean = all_total / all_count

    if overall_mean == 0:
        return None

    best = rows[0]
    worst = rows[-1]
    best_avg = float(best["avg_val"])
    worst_avg = float(worst["avg_val"])

    gap_pct = ((best_avg - worst_avg) / overall_mean) * 100

    if gap_pct < 15:
        return None

    # Build ranking — all values already float from casts above
    entity_stats = []
    for r in rows:
        avg = float(r["avg_val"])
        vs_avg = ((avg - overall_mean) / overall_mean) * 100
        entity_stats.append({
            "entity": str(r["entity"]),
            "mean": round(avg, 2),
            "count": int(r["cnt"]),
            "vs_avg_pct": round(vs_avg, 1),
        })

    ranking_parts = []
    for i, es in enumerate(entity_stats):
        sign = "+" if es["vs_avg_pct"] > 0 else ""
        ranking_parts.append(
            f"{i+1}. {es['entity']}: {es['mean']:,.1f} ({sign}{es['vs_avg_pct']:.0f}% vs avg)"
        )

    entity_label = _infer_entity_label(cat_col)
    metric_label = _infer_metric_label(num_col, num_info)
    severity = "warning" if gap_pct > 30 else "info"
    impact = min(int(gap_pct / 2) + 35, 80)

    is_currency = num_info.get("semantic_type") == "numeric_currency"
    unit = "₹" if is_currency else ""

    h_table = _humanize_table(profile.table_name)
    h_metric = num_col.replace("_", " ").title()

    title = (
        f"{worst['entity']} {metric_label} is {gap_pct:.0f}% below {best['entity']}"
    )
    description = (
        f"{entity_label} {worst['entity']} averages {unit}{worst_avg:,.1f} on {h_metric}, "
        f"which is {gap_pct:.0f}% below {best['entity']} at {unit}{best_avg:,.1f}. "
        f"The overall average across all {len(rows)} entities is {unit}{overall_mean:,.1f}. "
        f"Based on all {all_count:,} rows of data. "
        f"Ranking: {'; '.join(ranking_parts)}."
    )

    return Insight(
        id=str(uuid.uuid4()),
        insight_type="entity_performance",
        severity=severity,
        impact_score=impact,
        title=title,
        description=description,
        source_tables=[profile.table_name],
        source_columns=[cat_col, num_col],
        evidence={
            "entity_column": cat_col,
            "metric_column": num_col,
            "best_entity": str(best["entity"]),
            "best_value": round(best_avg, 2),
            "worst_entity": str(worst["entity"]),
            "worst_value": round(worst_avg, 2),
            "overall_mean": round(overall_mean, 2),
            "gap_pct": round(gap_pct, 1),
            "entity_count": len(entity_stats),
            "total_rows": all_count,
            "full_ranking": entity_stats,
            "chart_spec": {
                "type": "bar",
                "x": cat_col,
                "y": [num_col],
                "title": f"{num_col} by {cat_col}",
                "highlight": {"worst": str(worst["entity"]), "best": str(best["entity"])},
            },
            "query": (
                f'SELECT "{cat_col}", AVG("{num_col}") as avg_{num_col}, '
                f'COUNT(*) as count, SUM("{num_col}") as total '
                f'FROM "{profile.table_name}" '
                f'GROUP BY "{cat_col}" ORDER BY avg_{num_col} DESC'
            ),
        },
        suggested_actions=[
            f"Investigate why {worst['entity']} is {gap_pct:.0f}% below {best['entity']} on {h_metric}",
            f"Look at what {best['entity']} does differently and replicate those practices",
            f"Compare {best['entity']} vs {worst['entity']} across all available metrics",
        ],
    )
