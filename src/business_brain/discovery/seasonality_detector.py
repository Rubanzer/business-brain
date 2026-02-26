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

    # 1. Shift pattern detection — only if we can find actual performance differences
    shift_cols = [c for c in cat_cols if "shift" in c.lower()]
    if shift_cols and num_cols:
        shift_insight = _detect_shift_performance_gap(profile, cols, shift_cols[0], num_cols[:3])
        if shift_insight:
            results.append(shift_insight)

    # 2-3. Day-of-week / monthly: skip "analysis possible" flags.
    # These are suggestions, not insights. They belong in the recommender, not the feed.

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


def _detect_shift_performance_gap(
    profile: TableProfile,
    cols: dict,
    shift_col: str,
    num_cols: list[str],
) -> Insight | None:
    """Detect actual performance differences across shifts using sample data.

    Only creates an insight if there's a measurable gap (>15% difference
    between best and worst shift).
    """
    shift_info = cols.get(shift_col, {})
    shift_samples = shift_info.get("sample_values", [])
    cardinality = shift_info.get("cardinality", 0)

    if not (2 <= cardinality <= 5) or len(shift_samples) < 6:
        return None

    # Try to find a numeric column where shifts differ significantly
    for num_col in num_cols:
        num_info = cols.get(num_col, {})
        num_samples = num_info.get("sample_values", [])

        if not num_samples or len(num_samples) != len(shift_samples):
            continue

        # Group numeric values by shift
        shift_groups: dict[str, list[float]] = {}
        for shift_val, num_val in zip(shift_samples, num_samples):
            try:
                v = float(str(num_val).replace(",", ""))
                key = str(shift_val)
                shift_groups.setdefault(key, []).append(v)
            except (ValueError, TypeError):
                pass

        if len(shift_groups) < 2:
            continue

        # Compute averages per shift
        shift_avgs = {k: sum(v) / len(v) for k, v in shift_groups.items() if v}
        if not shift_avgs:
            continue

        best_shift = max(shift_avgs, key=shift_avgs.get)
        worst_shift = min(shift_avgs, key=shift_avgs.get)

        if shift_avgs[worst_shift] == 0:
            continue

        gap_pct = ((shift_avgs[best_shift] - shift_avgs[worst_shift]) / abs(shift_avgs[worst_shift])) * 100

        # Only report if gap is significant
        if gap_pct < 15:
            continue

        return Insight(
            id=str(uuid.uuid4()),
            insight_type="seasonality",
            severity="warning" if gap_pct > 30 else "info",
            impact_score=min(int(gap_pct / 2) + 30, 75),
            title=f"{num_col} varies {gap_pct:.0f}% across shifts in {profile.table_name}",
            description=(
                f"Shift '{best_shift}' averages {shift_avgs[best_shift]:,.2f} for {num_col} "
                f"vs shift '{worst_shift}' at {shift_avgs[worst_shift]:,.2f} — "
                f"a {gap_pct:.1f}% performance gap. "
                f"Sample sizes: {', '.join(f'{k}={len(v)}' for k, v in shift_groups.items())}."
            ),
            source_tables=[profile.table_name],
            source_columns=[shift_col, num_col],
            evidence={
                "pattern_type": "shift_performance_gap",
                "shift_column": shift_col,
                "metric_column": num_col,
                "shift_averages": {k: round(v, 2) for k, v in shift_avgs.items()},
                "best_shift": best_shift,
                "worst_shift": worst_shift,
                "gap_pct": round(gap_pct, 1),
                "chart_spec": {
                    "type": "bar",
                    "x": shift_col,
                    "y": [num_col],
                    "title": f"{num_col} by {shift_col}",
                },
            },
            suggested_actions=[
                f"Investigate why shift '{worst_shift}' underperforms by {gap_pct:.0f}% on {num_col}",
                f"Check staffing, equipment, or process differences between shifts",
            ],
        )

    return None
