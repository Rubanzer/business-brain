"""Dashboard summary aggregator — computes high-level KPIs across all data.

Pure functions that aggregate profiles, insights, and reports into
summary metrics for dashboard display.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class DashboardSummary:
    """High-level KPI summary for the dashboard."""

    total_tables: int
    total_rows: int
    total_columns: int
    total_insights: int
    total_reports: int
    avg_quality_score: float  # 0-100
    data_freshness_pct: float  # 0-100 (% of tables with recent data)
    insight_breakdown: dict[str, int]  # by type
    severity_breakdown: dict[str, int]  # by severity
    top_tables: list[dict]  # top 5 tables by row count
    last_discovery_at: str | None


def compute_dashboard_summary(
    profiles: list,
    insights: list,
    reports: list,
    discovery_runs: list | None = None,
) -> DashboardSummary:
    """Compute high-level dashboard summary from discovery artifacts.

    Pure function — takes lists of objects/dicts and returns summary.
    """
    # Profile stats
    total_tables = len(profiles)
    total_rows = 0
    total_columns = 0
    quality_scores = []
    table_sizes = []

    for p in profiles:
        row_count = _get(p, "row_count", 0) or 0
        total_rows += row_count

        cls = _get(p, "column_classification")
        if cls and isinstance(cls, dict) and "columns" in cls:
            col_count = len(cls["columns"])
            total_columns += col_count

            # Compute per-table quality from column stats
            qs = _compute_table_quality(cls["columns"], row_count)
            quality_scores.append(qs)

        table_sizes.append({
            "table": _get(p, "table_name", "unknown"),
            "rows": row_count,
            "domain": _get(p, "domain_hint", "general"),
        })

    # Sort tables by size
    table_sizes.sort(key=lambda t: t["rows"], reverse=True)
    top_tables = table_sizes[:5]

    # Average quality
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Insight breakdown
    insight_breakdown: dict[str, int] = {}
    severity_breakdown: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}

    for insight in insights:
        itype = _get(insight, "insight_type", "unknown")
        insight_breakdown[itype] = insight_breakdown.get(itype, 0) + 1

        severity = _get(insight, "severity", "info")
        if severity in severity_breakdown:
            severity_breakdown[severity] += 1

    # Data freshness (% of tables with data_hash, approximation)
    fresh_count = sum(1 for p in profiles if _get(p, "data_hash"))
    freshness_pct = (fresh_count / total_tables * 100) if total_tables > 0 else 0

    # Last discovery run
    last_discovery = None
    if discovery_runs:
        completed = [r for r in discovery_runs if _get(r, "status") == "completed"]
        if completed:
            last_completed = max(completed, key=lambda r: _get(r, "completed_at", "") or "")
            last_at = _get(last_completed, "completed_at")
            if last_at:
                last_discovery = str(last_at) if isinstance(last_at, str) else last_at.isoformat()

    return DashboardSummary(
        total_tables=total_tables,
        total_rows=total_rows,
        total_columns=total_columns,
        total_insights=len(insights),
        total_reports=len(reports),
        avg_quality_score=round(avg_quality, 1),
        data_freshness_pct=round(freshness_pct, 1),
        insight_breakdown=insight_breakdown,
        severity_breakdown=severity_breakdown,
        top_tables=top_tables,
        last_discovery_at=last_discovery,
    )


def _compute_table_quality(columns: dict, row_count: int) -> float:
    """Estimate data quality score for a table based on column profile info.

    Score = weighted average of:
    - Completeness (30%): fraction of non-null values
    - Uniqueness (20%): average cardinality ratio
    - Validity (30%): absence of impossible values
    - Diversity (20%): semantic type variety

    Returns 0-100 score.
    """
    if not columns or row_count == 0:
        return 0.0

    # Completeness: fraction of non-null values
    total_nulls = 0
    total_cells = 0
    for info in columns.values():
        null_count = info.get("null_count", 0) or 0
        total_nulls += null_count
        total_cells += row_count
    completeness = (1 - total_nulls / total_cells) * 100 if total_cells > 0 else 100

    # Uniqueness: average cardinality ratio
    cardinality_ratios = []
    for info in columns.values():
        card = info.get("cardinality", 0) or 0
        if row_count > 0 and card > 0:
            cardinality_ratios.append(min(card / row_count, 1.0))
    uniqueness = (sum(cardinality_ratios) / len(cardinality_ratios) * 100) if cardinality_ratios else 50

    # Validity: assume valid unless stats show problems
    validity = 100.0
    for info in columns.values():
        stats = info.get("stats")
        sem_type = info.get("semantic_type", "")
        if stats:
            if sem_type == "numeric_currency" and stats.get("min", 0) < 0:
                validity -= 10
            if sem_type == "numeric_percentage":
                if stats.get("max", 0) > 100 or stats.get("min", 0) < 0:
                    validity -= 10
    validity = max(validity, 0)

    # Diversity: how many distinct semantic types
    types = set()
    for info in columns.values():
        types.add(info.get("semantic_type", "unknown"))
    diversity = min(len(types) / 5 * 100, 100)  # 5+ types = perfect diversity

    # Weighted average
    score = completeness * 0.3 + uniqueness * 0.2 + validity * 0.3 + diversity * 0.2
    return min(max(score, 0), 100)


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safe attribute getter for both objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
