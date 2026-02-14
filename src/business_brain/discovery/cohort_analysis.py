"""Cohort analysis — track entity groups over time.

Pure functions for grouping entities by a categorical column,
tracking metric evolution, and computing retention/change patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CohortBucket:
    """A single cohort bucket (group + time period)."""
    cohort: str
    period: str
    count: int
    metric_mean: float
    metric_total: float


@dataclass
class CohortResult:
    """Complete cohort analysis result."""
    cohort_column: str
    time_column: str
    metric_column: str
    cohorts: list[str]
    periods: list[str]
    buckets: list[CohortBucket]
    retention_matrix: dict[str, dict[str, float]]  # cohort -> {period -> retention %}
    growth_matrix: dict[str, dict[str, float]]  # cohort -> {period -> growth %}
    summary: str


def build_cohorts(
    rows: list[dict],
    cohort_column: str,
    time_column: str,
    metric_column: str,
) -> CohortResult | None:
    """Build cohort analysis from rows.

    Args:
        rows: Data rows as dicts.
        cohort_column: Column defining cohorts (e.g., "signup_month", "supplier").
        time_column: Column defining time periods (e.g., "month", "quarter").
        metric_column: Numeric column to track.

    Returns:
        CohortResult or None if insufficient data.
    """
    if not rows:
        return None

    # Group by cohort + period
    buckets_map: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        cohort = row.get(cohort_column)
        period = row.get(time_column)
        value = row.get(metric_column)
        if cohort is None or period is None or value is None:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        key = (str(cohort), str(period))
        buckets_map.setdefault(key, []).append(fval)

    if len(buckets_map) < 2:
        return None

    # Build buckets
    buckets = []
    for (cohort, period), values in buckets_map.items():
        buckets.append(CohortBucket(
            cohort=cohort,
            period=period,
            count=len(values),
            metric_mean=round(sum(values) / len(values), 4),
            metric_total=round(sum(values), 4),
        ))

    # Extract unique cohorts and periods
    cohorts = sorted(set(b.cohort for b in buckets))
    periods = sorted(set(b.period for b in buckets))

    # Build retention matrix (count-based)
    retention_matrix = _build_retention_matrix(buckets, cohorts, periods)

    # Build growth matrix (metric mean change)
    growth_matrix = _build_growth_matrix(buckets, cohorts, periods)

    # Summary
    total_cohorts = len(cohorts)
    total_periods = len(periods)
    total_entries = len(buckets)
    summary = (
        f"Cohort analysis: {total_cohorts} cohorts across {total_periods} periods "
        f"({total_entries} data points). "
        f"Tracking {metric_column} grouped by {cohort_column} over {time_column}."
    )

    return CohortResult(
        cohort_column=cohort_column,
        time_column=time_column,
        metric_column=metric_column,
        cohorts=cohorts,
        periods=periods,
        buckets=buckets,
        retention_matrix=retention_matrix,
        growth_matrix=growth_matrix,
        summary=summary,
    )


def compute_cohort_health(result: CohortResult) -> dict:
    """Compute health metrics for the cohort analysis.

    Returns dict with retention rates, growth trends, and highlights.
    """
    if not result or not result.buckets:
        return {"status": "no_data"}

    # Average retention across cohorts
    avg_retentions = []
    for cohort, periods_data in result.retention_matrix.items():
        values = [v for v in periods_data.values() if v is not None]
        if values:
            avg_retentions.append(sum(values) / len(values))

    avg_retention = sum(avg_retentions) / len(avg_retentions) if avg_retentions else 0

    # Growth trend
    growth_values = []
    for cohort, periods_data in result.growth_matrix.items():
        values = [v for v in periods_data.values() if v is not None]
        growth_values.extend(values)

    avg_growth = sum(growth_values) / len(growth_values) if growth_values else 0

    # Best and worst cohorts by last period performance
    last_period = result.periods[-1] if result.periods else None
    cohort_scores = {}
    for b in result.buckets:
        if b.period == last_period:
            cohort_scores[b.cohort] = b.metric_mean

    best_cohort = max(cohort_scores, key=cohort_scores.get) if cohort_scores else None  # type: ignore
    worst_cohort = min(cohort_scores, key=cohort_scores.get) if cohort_scores else None  # type: ignore

    return {
        "avg_retention_pct": round(avg_retention, 1),
        "avg_growth_pct": round(avg_growth, 1),
        "total_cohorts": len(result.cohorts),
        "total_periods": len(result.periods),
        "best_cohort": best_cohort,
        "worst_cohort": worst_cohort,
        "best_score": round(cohort_scores.get(best_cohort, 0), 4) if best_cohort else None,
        "worst_score": round(cohort_scores.get(worst_cohort, 0), 4) if worst_cohort else None,
    }


def pivot_cohort_table(result: CohortResult) -> list[dict]:
    """Convert cohort result into a pivot table format.

    Returns list of dicts: [{"cohort": "A", "period_1": val, "period_2": val, ...}]
    """
    if not result:
        return []

    # Index by (cohort, period) -> metric_mean
    index: dict[tuple[str, str], float] = {}
    for b in result.buckets:
        index[(b.cohort, b.period)] = b.metric_mean

    table = []
    for cohort in result.cohorts:
        row = {"cohort": cohort}
        for period in result.periods:
            row[period] = index.get((cohort, period))
        table.append(row)

    return table


def find_declining_cohorts(result: CohortResult, threshold: float = -10.0) -> list[dict]:
    """Find cohorts with declining metrics.

    Args:
        result: Cohort analysis result.
        threshold: Growth percentage below which a cohort is flagged.

    Returns:
        List of {cohort, decline_pct, periods_declining} dicts.
    """
    if not result:
        return []

    declining = []
    for cohort in result.cohorts:
        growth_data = result.growth_matrix.get(cohort, {})
        negative_periods = [(p, v) for p, v in growth_data.items() if v is not None and v < threshold]
        if negative_periods:
            avg_decline = sum(v for _, v in negative_periods) / len(negative_periods)
            declining.append({
                "cohort": cohort,
                "decline_pct": round(avg_decline, 1),
                "periods_declining": len(negative_periods),
                "total_periods": len(growth_data),
            })

    declining.sort(key=lambda d: d["decline_pct"])
    return declining


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_retention_matrix(
    buckets: list[CohortBucket],
    cohorts: list[str],
    periods: list[str],
) -> dict[str, dict[str, float]]:
    """Build retention matrix — count in each period as % of first period."""
    # Index by (cohort, period)
    count_index: dict[tuple[str, str], int] = {}
    for b in buckets:
        count_index[(b.cohort, b.period)] = b.count

    matrix: dict[str, dict[str, float]] = {}
    for cohort in cohorts:
        first_count = count_index.get((cohort, periods[0])) if periods else None
        period_retention = {}
        for period in periods:
            count = count_index.get((cohort, period))
            if count is not None and first_count and first_count > 0:
                period_retention[period] = round(count / first_count * 100, 1)
            else:
                period_retention[period] = None  # type: ignore
        matrix[cohort] = period_retention

    return matrix


def _build_growth_matrix(
    buckets: list[CohortBucket],
    cohorts: list[str],
    periods: list[str],
) -> dict[str, dict[str, float]]:
    """Build growth matrix — metric mean change % between consecutive periods."""
    mean_index: dict[tuple[str, str], float] = {}
    for b in buckets:
        mean_index[(b.cohort, b.period)] = b.metric_mean

    matrix: dict[str, dict[str, float]] = {}
    for cohort in cohorts:
        period_growth = {}
        for i in range(1, len(periods)):
            prev = mean_index.get((cohort, periods[i - 1]))
            curr = mean_index.get((cohort, periods[i]))
            if prev is not None and curr is not None and prev != 0:
                period_growth[periods[i]] = round((curr - prev) / abs(prev) * 100, 1)
            else:
                period_growth[periods[i]] = None  # type: ignore
        matrix[cohort] = period_growth

    return matrix
