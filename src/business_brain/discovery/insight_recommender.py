"""Insight recommender — suggests what analyses to run next.

Analyzes existing insights, table profiles, and relationships to
recommend the most valuable next analyses. Pure functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Recommendation:
    """A recommended analysis to run."""
    title: str
    description: str
    analysis_type: str  # benchmark, cohort, correlation, time_trend, anomaly, forecast
    target_table: str
    columns: list[str]
    priority: int  # 1-100
    reason: str


def recommend_analyses(
    profiles: list[dict],
    insights: list[dict],
    relationships: list[dict],
    max_recommendations: int = 10,
) -> list[Recommendation]:
    """Generate analysis recommendations based on current state.

    Args:
        profiles: Table profiles (dicts with table_name, column_classification, row_count).
        insights: Existing insights (dicts with insight_type, source_tables).
        relationships: Known relationships.
        max_recommendations: Maximum recommendations to return.

    Returns:
        List of Recommendations sorted by priority.
    """
    recommendations: list[Recommendation] = []

    # Index insights by table
    insight_tables: dict[str, int] = {}
    insight_types: dict[str, set[str]] = {}
    for ins in insights:
        for t in (ins.get("source_tables") or []):
            insight_tables[t] = insight_tables.get(t, 0) + 1
            insight_types.setdefault(t, set()).add(ins.get("insight_type", ""))

    # Index relationships
    related_tables: dict[str, set[str]] = {}
    for rel in relationships:
        ta = rel.get("table_a", "")
        tb = rel.get("table_b", "")
        related_tables.setdefault(ta, set()).add(tb)
        related_tables.setdefault(tb, set()).add(ta)

    for profile in profiles:
        table_name = profile.get("table_name", "")
        columns = _get_columns(profile)
        row_count = profile.get("row_count", 0) or 0

        # Skip tiny tables
        if row_count < 10:
            continue

        existing_types = insight_types.get(table_name, set())
        insight_count = insight_tables.get(table_name, 0)

        # 1. Recommend time analysis for tables with temporal + numeric
        temporal_cols = [c for c, info in columns.items() if info.get("semantic_type") == "temporal"]
        numeric_cols = [c for c, info in columns.items() if (info.get("semantic_type") or "").startswith("numeric")]

        if temporal_cols and numeric_cols and "trend" not in existing_types:
            recommendations.append(Recommendation(
                title=f"Time trend analysis on {table_name}",
                description=f"Analyze {numeric_cols[0]} trends over {temporal_cols[0]}",
                analysis_type="time_trend",
                target_table=table_name,
                columns=[temporal_cols[0], numeric_cols[0]],
                priority=_time_priority(row_count, insight_count),
                reason="Table has temporal and numeric columns but no trend analysis yet.",
            ))

        # 2. Recommend benchmarking for tables with categorical + numeric
        cat_cols = [c for c, info in columns.items() if info.get("semantic_type") == "categorical"]

        if cat_cols and numeric_cols:
            recommendations.append(Recommendation(
                title=f"Benchmark {numeric_cols[0]} by {cat_cols[0]} in {table_name}",
                description=f"Compare {numeric_cols[0]} across different {cat_cols[0]} groups",
                analysis_type="benchmark",
                target_table=table_name,
                columns=[cat_cols[0], numeric_cols[0]],
                priority=_benchmark_priority(row_count, len(cat_cols), insight_count),
                reason="Table has categorical grouping column for metric comparison.",
            ))

        # 3. Recommend correlation if 2+ numeric columns
        if len(numeric_cols) >= 2 and "correlation" not in existing_types:
            recommendations.append(Recommendation(
                title=f"Correlation analysis on {table_name}",
                description=f"Check correlations between {len(numeric_cols)} numeric columns",
                analysis_type="correlation",
                target_table=table_name,
                columns=numeric_cols[:5],
                priority=_correlation_priority(len(numeric_cols), insight_count),
                reason=f"Table has {len(numeric_cols)} numeric columns but no correlation analysis.",
            ))

        # 4. Recommend anomaly scan if no anomaly insights
        if numeric_cols and "anomaly" not in existing_types:
            recommendations.append(Recommendation(
                title=f"Anomaly scan on {table_name}",
                description=f"Scan {len(numeric_cols)} numeric columns for outliers",
                analysis_type="anomaly",
                target_table=table_name,
                columns=numeric_cols[:5],
                priority=_anomaly_priority(row_count, insight_count),
                reason="No anomaly detection has been run on this table.",
            ))

        # 5. Recommend cohort analysis if temporal + categorical + numeric
        if temporal_cols and cat_cols and numeric_cols:
            recommendations.append(Recommendation(
                title=f"Cohort analysis: {cat_cols[0]} over {temporal_cols[0]} in {table_name}",
                description=f"Track {numeric_cols[0]} for {cat_cols[0]} cohorts over {temporal_cols[0]}",
                analysis_type="cohort",
                target_table=table_name,
                columns=[cat_cols[0], temporal_cols[0], numeric_cols[0]],
                priority=_cohort_priority(row_count, insight_count),
                reason="Table has all three column types needed for cohort tracking.",
            ))

        # 6. Recommend forecast if temporal + numeric and enough data
        if temporal_cols and numeric_cols and row_count >= 20:
            recommendations.append(Recommendation(
                title=f"Forecast {numeric_cols[0]} in {table_name}",
                description=f"Predict future values of {numeric_cols[0]} based on {temporal_cols[0]}",
                analysis_type="forecast",
                target_table=table_name,
                columns=[temporal_cols[0], numeric_cols[0]],
                priority=_forecast_priority(row_count, insight_count),
                reason="Sufficient temporal data for forecasting.",
            ))

    # 7. Cross-table recommendations
    for table_name, related in related_tables.items():
        if len(related) >= 2 and insight_tables.get(table_name, 0) < 3:
            recommendations.append(Recommendation(
                title=f"Cross-table analysis for {table_name}",
                description=f"Explore relationships between {table_name} and {', '.join(list(related)[:3])}",
                analysis_type="correlation",
                target_table=table_name,
                columns=[],
                priority=60,
                reason=f"Table has {len(related)} relationships but few insights.",
            ))

    # Sort by priority, deduplicate by table+type
    seen = set()
    unique = []
    for r in sorted(recommendations, key=lambda x: -x.priority):
        key = (r.target_table, r.analysis_type)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:max_recommendations]


def compute_coverage(
    profiles: list[dict],
    insights: list[dict],
) -> dict:
    """Compute analysis coverage metrics.

    Returns dict with table coverage stats and gaps.
    """
    all_tables = {p.get("table_name", "") for p in profiles}
    covered_tables = set()
    for ins in insights:
        for t in (ins.get("source_tables") or []):
            covered_tables.add(t)

    uncovered = all_tables - covered_tables
    coverage_pct = len(covered_tables) / len(all_tables) * 100 if all_tables else 0

    return {
        "total_tables": len(all_tables),
        "covered_tables": len(covered_tables),
        "uncovered_tables": sorted(uncovered),
        "coverage_pct": round(coverage_pct, 1),
        "total_insights": len(insights),
        "avg_insights_per_table": round(len(insights) / len(all_tables), 1) if all_tables else 0,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_columns(profile: dict) -> dict:
    """Extract columns from profile."""
    cls = profile.get("column_classification")
    if cls and isinstance(cls, dict) and "columns" in cls:
        return cls["columns"]
    return {}


def _time_priority(row_count: int, insight_count: int) -> int:
    base = 80
    if insight_count > 5:
        base -= 20
    if row_count > 100:
        base += 5
    return min(max(base, 10), 100)


def _benchmark_priority(row_count: int, cat_count: int, insight_count: int) -> int:
    base = 70
    if cat_count > 2:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)


def _correlation_priority(num_col_count: int, insight_count: int) -> int:
    base = 65
    if num_col_count > 4:
        base += 15
    if insight_count > 3:
        base -= 10
    return min(max(base, 10), 100)


def _anomaly_priority(row_count: int, insight_count: int) -> int:
    base = 75
    if row_count > 200:
        base += 5
    if insight_count > 5:
        base -= 20
    return min(max(base, 10), 100)


def _cohort_priority(row_count: int, insight_count: int) -> int:
    base = 60
    if row_count > 100:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)


def _forecast_priority(row_count: int, insight_count: int) -> int:
    base = 55
    if row_count > 50:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)
