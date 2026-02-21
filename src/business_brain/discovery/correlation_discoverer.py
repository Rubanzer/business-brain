"""Correlation discovery — finds notable correlations from table profiles.

Uses column classification stats to identify potentially correlated numeric columns
and generates insights. Works with profile data only (no DB queries).
"""

from __future__ import annotations

import uuid
from business_brain.db.discovery_models import Insight


def discover_correlations_from_profiles(profiles: list) -> list[Insight]:
    """Scan profiles for numeric column pairs and flag potential correlations.

    Uses column classification to identify tables with 2+ numeric columns
    and generates correlation analysis insights.

    Pure function — uses profile data only.
    """
    insights: list[Insight] = []

    for profile in profiles:
        cls = getattr(profile, "column_classification", None)
        if not cls or "columns" not in cls:
            continue

        cols = cls["columns"]
        row_count = getattr(profile, "row_count", 0) or 0

        if row_count < 10:
            continue

        # Find numeric columns with stats
        numeric_cols = []
        for col_name, info in cols.items():
            sem_type = info.get("semantic_type", "")
            if sem_type in ("numeric_metric", "numeric_currency", "numeric_percentage"):
                stats = info.get("stats")
                if stats and stats.get("stdev", 0) > 0:
                    numeric_cols.append((col_name, info))

        if len(numeric_cols) < 2:
            continue

        # Use sample values to estimate correlations (if available)
        corr_pairs = _estimate_sample_correlations(numeric_cols)

        if not corr_pairs:
            # Don't generate "opportunity" meta-insights — they clutter the feed
            continue

        # Generate insights for strong estimated correlations
        for col_a, col_b, est_r, direction in corr_pairs:
            insights.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="correlation",
                severity="info",
                impact_score=55 if abs(est_r) >= 0.85 else (45 if abs(est_r) >= 0.7 else 30),
                title=f"Potential correlation: {col_a} ↔ {col_b} in {profile.table_name}",
                description=(
                    f"Columns {col_a} and {col_b} in {profile.table_name} show "
                    f"a {direction} relationship (estimated r ≈ {est_r:.2f}). "
                    f"Run full correlation analysis for precise measurement."
                ),
                source_tables=[profile.table_name],
                source_columns=[col_a, col_b],
                evidence={
                    "estimated_correlation": round(est_r, 3),
                    "direction": direction,
                    "query": (
                        f'SELECT "{col_a}", "{col_b}" FROM "{profile.table_name}" '
                        f'ORDER BY ctid LIMIT 500'
                    ),
                    "chart_spec": {
                        "type": "scatter",
                        "x": col_a,
                        "y": col_b,
                        "title": f"{col_a} vs {col_b}",
                    },
                },
                suggested_actions=[
                    f"Investigate whether {col_a} and {col_b} have a causal relationship",
                    f"Consider if one can predict the other for forecasting",
                ],
            ))

    return insights


def _estimate_sample_correlations(
    numeric_cols: list[tuple[str, dict]],
) -> list[tuple[str, str, float, str]]:
    """Estimate correlations from sample values in column info.

    Returns list of (col_a, col_b, estimated_r, direction).
    Only includes pairs with sufficient data and notable correlation.
    """
    results = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col_a, info_a = numeric_cols[i]
            col_b, info_b = numeric_cols[j]

            samples_a = info_a.get("sample_values", [])
            samples_b = info_b.get("sample_values", [])

            if len(samples_a) < 5 or len(samples_b) < 5:
                continue

            # Parse to floats
            vals_a = _parse_samples(samples_a)
            vals_b = _parse_samples(samples_b)

            # Use paired values only (min length)
            n = min(len(vals_a), len(vals_b))
            if n < 5:
                continue

            vals_a = vals_a[:n]
            vals_b = vals_b[:n]

            r = _quick_pearson(vals_a, vals_b)
            if r is None:
                continue

            if abs(r) >= 0.7:
                direction = "positive" if r > 0 else "negative"
                results.append((col_a, col_b, r, direction))

    return results


def _parse_samples(samples: list) -> list[float]:
    """Parse sample values to floats, skipping non-numeric."""
    result = []
    for s in samples:
        try:
            result.append(float(str(s).replace(",", "")))
        except (ValueError, TypeError):
            pass
    return result


def _quick_pearson(x: list[float], y: list[float]) -> float | None:
    """Quick Pearson correlation for sample data."""
    import math
    import statistics

    n = len(x)
    if n < 3:
        return None

    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)

    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))

    if dx == 0 or dy == 0:
        return None

    return num / (dx * dy)


def _make_opportunity_insight(profile, numeric_cols: list) -> Insight:
    """Create an insight noting that correlation analysis is available."""
    col_names = [c[0] for c in numeric_cols[:5]]
    return Insight(
        id=str(uuid.uuid4()),
        insight_type="correlation",
        severity="info",
        impact_score=20,
        title=f"Correlation analysis available for {profile.table_name}",
        description=(
            f"Table {profile.table_name} has {len(numeric_cols)} numeric columns "
            f"({', '.join(col_names)}). Run correlation analysis to find hidden relationships."
        ),
        source_tables=[profile.table_name],
        source_columns=col_names,
        evidence={
            "numeric_column_count": len(numeric_cols),
            "columns": col_names,
        },
        suggested_actions=[
            "Run full correlation analysis from the Quality tab",
        ],
    )
