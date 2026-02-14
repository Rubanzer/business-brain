"""Seasonality detection — finds periodic patterns in temporal+numeric data.

Detects:
- Daily patterns (shift cycles in manufacturing)
- Weekly patterns (weekday vs weekend differences)
- Monthly trends (seasonal demand, production cycles)
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter

from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)


def detect_seasonality(profiles: list[TableProfile]) -> list[Insight]:
    """Scan all profiled tables for seasonal/periodic patterns.

    Uses column classification and sample data to identify periodicity.
    No DB queries needed — uses profile data only.
    """
    insights: list[Insight] = []

    for profile in profiles:
        try:
            insights.extend(_scan_table_seasonality(profile))
        except Exception:
            logger.exception("Seasonality scan failed for %s", profile.table_name)

    return insights


def _scan_table_seasonality(profile: TableProfile) -> list[Insight]:
    """Analyze a single table for seasonal patterns."""
    results: list[Insight] = []
    cls = profile.column_classification
    if not cls or "columns" not in cls:
        return results

    cols = cls["columns"]
    row_count = profile.row_count or 0

    if row_count < 10:
        return results  # Need enough data

    temp_cols = [c for c, i in cols.items() if i.get("semantic_type") == "temporal"]
    num_cols = [
        c for c, i in cols.items()
        if i.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ]
    cat_cols = [c for c, i in cols.items() if i.get("semantic_type") == "categorical"]

    if not temp_cols or not num_cols:
        return results

    # 1. Shift pattern detection (manufacturing-specific)
    shift_cols = [c for c in cat_cols if "shift" in c.lower()]
    if shift_cols:
        shift_info = cols.get(shift_cols[0], {})
        cardinality = shift_info.get("cardinality", 0)
        if 2 <= cardinality <= 5:
            results.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="seasonality",
                severity="info",
                impact_score=40,
                title=f"Shift cycle pattern in {profile.table_name}",
                description=(
                    f"Table has {cardinality} shifts ({shift_cols[0]}) with numeric metrics "
                    f"{num_cols[:3]}. Shift-over-shift comparison is available."
                ),
                source_tables=[profile.table_name],
                source_columns=shift_cols + num_cols[:3],
                evidence={
                    "pattern_type": "shift_cycle",
                    "shift_column": shift_cols[0],
                    "shift_count": cardinality,
                    "metric_columns": num_cols[:3],
                    "query": (
                        f'SELECT "{shift_cols[0]}", '
                        + ", ".join(f'AVG("{n}")' for n in num_cols[:3])
                        + f' FROM "{profile.table_name}" GROUP BY "{shift_cols[0]}"'
                    ),
                    "chart_spec": {
                        "type": "bar",
                        "x": shift_cols[0],
                        "y": num_cols[:2],
                        "title": f"Shift-wise comparison in {profile.table_name}",
                    },
                },
                suggested_actions=[
                    f"Compare {num_cols[0]} performance across {shift_cols[0]}s",
                    "Identify best and worst performing shifts",
                ],
            ))

    # 2. Day-of-week detection (if temporal columns have enough range)
    for temp_col in temp_cols:
        temp_info = cols.get(temp_col, {})
        temp_cardinality = temp_info.get("cardinality", 0)

        # If we have 7+ distinct dates, suggest day-of-week analysis
        if temp_cardinality >= 7:
            results.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="seasonality",
                severity="info",
                impact_score=35,
                title=f"Day-of-week pattern opportunity in {profile.table_name}",
                description=(
                    f"Table has {temp_cardinality} distinct time points in {temp_col} "
                    f"with numeric metrics {num_cols[:2]}. "
                    f"Day-of-week pattern analysis is possible."
                ),
                source_tables=[profile.table_name],
                source_columns=[temp_col] + num_cols[:2],
                evidence={
                    "pattern_type": "day_of_week",
                    "temporal_column": temp_col,
                    "date_cardinality": temp_cardinality,
                    "metric_columns": num_cols[:2],
                    "query": (
                        f'SELECT EXTRACT(DOW FROM "{temp_col}"::date) as day_of_week, '
                        + ", ".join(f'AVG("{n}")' for n in num_cols[:2])
                        + f' FROM "{profile.table_name}" '
                        f'GROUP BY EXTRACT(DOW FROM "{temp_col}"::date) '
                        f'ORDER BY day_of_week'
                    ),
                    "chart_spec": {
                        "type": "bar",
                        "x": "day_of_week",
                        "y": num_cols[:2],
                        "title": f"Day-of-week pattern for {num_cols[0]}",
                    },
                },
                suggested_actions=[
                    f"Check if {num_cols[0]} varies by day of the week",
                    "Identify peak and off-peak days",
                ],
            ))
            break  # Only one day-of-week insight per table

    # 3. Monthly trend detection (if enough time range)
    for temp_col in temp_cols:
        temp_info = cols.get(temp_col, {})
        temp_cardinality = temp_info.get("cardinality", 0)

        if temp_cardinality >= 30:
            results.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="seasonality",
                severity="info",
                impact_score=30,
                title=f"Monthly trend analysis available for {profile.table_name}",
                description=(
                    f"Table has {temp_cardinality} time points spanning enough range for "
                    f"month-over-month analysis of {num_cols[:2]}."
                ),
                source_tables=[profile.table_name],
                source_columns=[temp_col] + num_cols[:2],
                evidence={
                    "pattern_type": "monthly_trend",
                    "temporal_column": temp_col,
                    "metric_columns": num_cols[:2],
                    "query": (
                        f'SELECT DATE_TRUNC(\'month\', "{temp_col}"::date) as month, '
                        + ", ".join(f'AVG("{n}")' for n in num_cols[:2])
                        + f' FROM "{profile.table_name}" '
                        f'GROUP BY DATE_TRUNC(\'month\', "{temp_col}"::date) '
                        f'ORDER BY month'
                    ),
                    "chart_spec": {
                        "type": "line",
                        "x": "month",
                        "y": num_cols[:2],
                        "title": f"Monthly trend for {num_cols[0]}",
                    },
                },
                suggested_actions=[
                    "Analyze month-over-month changes",
                    "Identify seasonal peaks and troughs",
                ],
            ))
            break

    # 4. Categorical distribution skew detection
    for cat_col in cat_cols:
        cat_info = cols.get(cat_col, {})
        cardinality = cat_info.get("cardinality", 0)
        samples = cat_info.get("sample_values", [])

        if 2 <= cardinality <= 20 and len(samples) >= 10:
            # Check for highly skewed distribution
            value_counts = Counter(str(s) for s in samples)
            if value_counts:
                most_common_pct = max(value_counts.values()) / len(samples)
                least_common_pct = min(value_counts.values()) / len(samples)

                if most_common_pct > 0.5 and least_common_pct < 0.1:
                    dominant = value_counts.most_common(1)[0]
                    results.append(Insight(
                        id=str(uuid.uuid4()),
                        insight_type="seasonality",
                        severity="info",
                        impact_score=25,
                        title=f"Skewed distribution in {profile.table_name}.{cat_col}",
                        description=(
                            f"{cat_col} is dominated by '{dominant[0]}' "
                            f"({dominant[1]}/{len(samples)} samples = {most_common_pct*100:.0f}%). "
                            f"This may indicate a seasonal or operational pattern."
                        ),
                        source_tables=[profile.table_name],
                        source_columns=[cat_col],
                        evidence={
                            "pattern_type": "distribution_skew",
                            "column": cat_col,
                            "dominant_value": dominant[0],
                            "dominant_pct": round(most_common_pct * 100, 1),
                            "cardinality": cardinality,
                        },
                        suggested_actions=[
                            f"Investigate why {cat_col} is dominated by '{dominant[0]}'",
                            f"Analyze if {num_cols[0] if num_cols else 'metrics'} differ for rare categories",
                        ],
                    ))

    return results
