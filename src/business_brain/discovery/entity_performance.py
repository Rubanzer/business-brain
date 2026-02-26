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
import statistics
import uuid
from collections import defaultdict

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
    for cat_col, cat_info in cat_cols[:3]:  # Limit to top 3 categories
        for num_col, num_info in num_cols[:4]:  # Limit to top 4 metrics
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

    # Parse numeric values
    entity_values: dict[str, list[float]] = defaultdict(list)
    for cat_val, num_val in zip(cat_samples, num_samples):
        try:
            v = float(str(num_val).replace(",", ""))
            entity = str(cat_val).strip()
            if entity:
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

    title = (
        f"{worst['entity']} {metric_label} {gap_pct:.0f}% below {best['entity']} "
        f"in {profile.table_name}"
    )
    description = (
        f"{entity_label} performance gap on {num_col}: "
        f"Best: {best['entity']} at {unit}{best['mean']:,.1f}, "
        f"Worst: {worst['entity']} at {unit}{worst['mean']:,.1f} "
        f"({gap_pct:.0f}% gap). "
        f"Group average: {unit}{overall_mean:,.1f}. "
        f"Full ranking: {ranking_str}."
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
            f"Investigate why {worst['entity']} is {gap_pct:.0f}% below {best['entity']} on {num_col}",
            f"Replicate {best['entity']} practices to bring {worst['entity']} up to average ({unit}{overall_mean:,.0f})",
            f"Run a detailed comparison of {best['entity']} vs {worst['entity']} across all metrics",
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
