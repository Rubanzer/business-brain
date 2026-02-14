"""Statistical summary generator — comprehensive descriptive statistics.

Pure functions for computing percentiles, confidence intervals,
normality assessment, and complete statistical summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class StatSummary:
    """Complete statistical summary for a numeric dataset."""
    column: str
    count: int
    mean: float
    median: float
    mode: float | None
    std: float
    variance: float
    min_val: float
    max_val: float
    range_val: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    cv: float  # coefficient of variation (%)
    percentiles: dict[str, float]  # p5, p10, p25, p50, p75, p90, p95
    ci_95_lower: float
    ci_95_upper: float
    normality: str  # "normal", "approximately_normal", "non_normal"
    outlier_count: int
    interpretation: str


def compute_stat_summary(values: list[float], column: str = "value") -> StatSummary | None:
    """Compute comprehensive statistical summary.

    Args:
        values: Numeric values.
        column: Column name for labeling.

    Returns:
        StatSummary or None if insufficient data.
    """
    if len(values) < 3:
        return None

    n = len(values)
    sorted_vals = sorted(values)

    # Basic stats
    mean = sum(values) / n
    median = _percentile(sorted_vals, 50)
    mode = _compute_mode(values)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0
    std = variance ** 0.5

    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    range_val = max_val - min_val

    # Quartiles
    q1 = _percentile(sorted_vals, 25)
    q3 = _percentile(sorted_vals, 75)
    iqr = q3 - q1

    # Percentiles
    percentiles = {
        "p5": round(_percentile(sorted_vals, 5), 4),
        "p10": round(_percentile(sorted_vals, 10), 4),
        "p25": round(q1, 4),
        "p50": round(median, 4),
        "p75": round(q3, 4),
        "p90": round(_percentile(sorted_vals, 90), 4),
        "p95": round(_percentile(sorted_vals, 95), 4),
    }

    # Skewness (Fisher-Pearson)
    skewness = _compute_skewness(values, mean, std) if std > 0 else 0.0

    # Kurtosis (excess)
    kurtosis = _compute_kurtosis(values, mean, std) if std > 0 else 0.0

    # CV (coefficient of variation)
    cv = (std / abs(mean) * 100) if mean != 0 else 0.0

    # 95% Confidence Interval for mean
    se = std / (n ** 0.5)
    z = 1.96  # approximate z for 95% CI
    ci_lower = mean - z * se
    ci_upper = mean + z * se

    # Normality assessment
    normality = _assess_normality(skewness, kurtosis)

    # Outlier count (IQR method)
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_count = sum(1 for v in values if v < lower_fence or v > upper_fence)

    # Interpretation
    interpretation = _build_interpretation(
        column, n, mean, median, std, skewness, cv, normality, outlier_count
    )

    return StatSummary(
        column=column,
        count=n,
        mean=round(mean, 4),
        median=round(median, 4),
        mode=round(mode, 4) if mode is not None else None,
        std=round(std, 4),
        variance=round(variance, 4),
        min_val=round(min_val, 4),
        max_val=round(max_val, 4),
        range_val=round(range_val, 4),
        q1=round(q1, 4),
        q3=round(q3, 4),
        iqr=round(iqr, 4),
        skewness=round(skewness, 4),
        kurtosis=round(kurtosis, 4),
        cv=round(cv, 2),
        percentiles=percentiles,
        ci_95_lower=round(ci_lower, 4),
        ci_95_upper=round(ci_upper, 4),
        normality=normality,
        outlier_count=outlier_count,
        interpretation=interpretation,
    )


def compare_distributions(summary_a: StatSummary, summary_b: StatSummary) -> dict:
    """Compare two statistical summaries.

    Returns a dict with comparison metrics.
    """
    mean_diff = summary_b.mean - summary_a.mean
    mean_diff_pct = (mean_diff / abs(summary_a.mean) * 100) if summary_a.mean != 0 else 0

    # Simple effect size (Cohen's d)
    pooled_std = ((summary_a.std ** 2 + summary_b.std ** 2) / 2) ** 0.5
    effect_size = abs(mean_diff) / pooled_std if pooled_std > 0 else 0

    return {
        "column_a": summary_a.column,
        "column_b": summary_b.column,
        "mean_diff": round(mean_diff, 4),
        "mean_diff_pct": round(mean_diff_pct, 2),
        "std_ratio": round(summary_b.std / summary_a.std, 3) if summary_a.std > 0 else 0,
        "effect_size": round(effect_size, 3),
        "significance": "large" if effect_size > 0.8 else "medium" if effect_size > 0.5 else "small" if effect_size > 0.2 else "negligible",
        "a_more_variable": summary_a.cv > summary_b.cv,
        "normality_match": summary_a.normality == summary_b.normality,
    }


def format_stat_table(summary: StatSummary) -> str:
    """Format a statistical summary as a readable text table."""
    lines = [
        f"Statistical Summary: {summary.column}",
        f"{'='*40}",
        f"Count:          {summary.count:,}",
        f"Mean:           {summary.mean:,.4f}",
        f"Median:         {summary.median:,.4f}",
        f"Std Dev:        {summary.std:,.4f}",
        f"Min:            {summary.min_val:,.4f}",
        f"Max:            {summary.max_val:,.4f}",
        f"Range:          {summary.range_val:,.4f}",
        f"Q1:             {summary.q1:,.4f}",
        f"Q3:             {summary.q3:,.4f}",
        f"IQR:            {summary.iqr:,.4f}",
        f"Skewness:       {summary.skewness:,.4f}",
        f"Kurtosis:       {summary.kurtosis:,.4f}",
        f"CV:             {summary.cv:.2f}%",
        f"95% CI:         [{summary.ci_95_lower:,.4f}, {summary.ci_95_upper:,.4f}]",
        f"Normality:      {summary.normality}",
        f"Outliers:       {summary.outlier_count}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute percentile using linear interpolation."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _compute_mode(values: list[float]) -> float | None:
    """Compute mode (most frequent value)."""
    if not values:
        return None
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    if max_count == 1:
        return None  # no mode
    modes = [v for v, c in counts.items() if c == max_count]
    return modes[0]  # return first mode


def _compute_skewness(values: list[float], mean: float, std: float) -> float:
    """Compute Fisher-Pearson skewness coefficient."""
    n = len(values)
    if n < 3 or std == 0:
        return 0.0
    m3 = sum((v - mean) ** 3 for v in values) / n
    return m3 / (std ** 3)


def _compute_kurtosis(values: list[float], mean: float, std: float) -> float:
    """Compute excess kurtosis."""
    n = len(values)
    if n < 4 or std == 0:
        return 0.0
    m4 = sum((v - mean) ** 4 for v in values) / n
    return m4 / (std ** 4) - 3


def _assess_normality(skewness: float, kurtosis: float) -> str:
    """Simple normality assessment based on skewness and kurtosis."""
    if abs(skewness) < 0.5 and abs(kurtosis) < 1:
        return "normal"
    elif abs(skewness) < 1 and abs(kurtosis) < 2:
        return "approximately_normal"
    else:
        return "non_normal"


def _build_interpretation(
    column: str,
    n: int,
    mean: float,
    median: float,
    std: float,
    skewness: float,
    cv: float,
    normality: str,
    outlier_count: int,
) -> str:
    """Build human-readable interpretation of statistics."""
    parts = [f"{column} ({n} values):"]

    # Central tendency
    if abs(mean - median) / (std if std > 0 else 1) < 0.2:
        parts.append(f"Mean ({mean:.2f}) ≈ Median ({median:.2f}), suggesting symmetric distribution.")
    elif mean > median:
        parts.append(f"Mean ({mean:.2f}) > Median ({median:.2f}), suggesting right skew.")
    else:
        parts.append(f"Mean ({mean:.2f}) < Median ({median:.2f}), suggesting left skew.")

    # Variability
    if cv < 15:
        parts.append(f"Low variability (CV={cv:.1f}%).")
    elif cv < 50:
        parts.append(f"Moderate variability (CV={cv:.1f}%).")
    else:
        parts.append(f"High variability (CV={cv:.1f}%).")

    # Distribution shape
    parts.append(f"Distribution: {normality}.")

    # Outliers
    if outlier_count > 0:
        parts.append(f"{outlier_count} outliers detected (IQR method).")

    return " ".join(parts)
