"""Metric benchmarking — compare metrics across segments/groups.

Pure functions for group comparison, ranking, relative performance scoring,
and finding significant differences between segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GroupStats:
    """Statistics for a single group."""
    group_name: str
    count: int
    mean: float
    median: float
    min_val: float
    max_val: float
    std: float
    total: float


@dataclass
class BenchmarkResult:
    """Result of comparing metric across groups."""
    metric_name: str
    group_column: str
    groups: list[GroupStats]
    best_group: str
    worst_group: str
    spread: float  # max_mean - min_mean
    spread_pct: float  # spread as % of overall mean
    ranking: list[dict]  # [{group, rank, value, pct_of_best}]
    significant_gaps: list[dict]  # [{group_a, group_b, diff, diff_pct}]
    summary: str


def benchmark_groups(
    rows: list[dict],
    group_column: str,
    metric_column: str,
    metric_name: str | None = None,
) -> BenchmarkResult | None:
    """Compare a metric across groups defined by a categorical column.

    Args:
        rows: Data rows as dicts.
        group_column: Column to group by (e.g., "supplier", "department").
        metric_column: Numeric column to compare (e.g., "cost", "quality_score").
        metric_name: Human-readable metric name (defaults to metric_column).

    Returns:
        BenchmarkResult or None if insufficient data.
    """
    if not rows:
        return None

    metric_name = metric_name or metric_column

    # Group values
    groups: dict[str, list[float]] = {}
    for row in rows:
        group = row.get(group_column)
        value = row.get(metric_column)
        if group is None or value is None:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        group_key = str(group)
        groups.setdefault(group_key, []).append(fval)

    if len(groups) < 2:
        return None

    # Compute per-group stats
    group_stats = []
    for name, values in groups.items():
        group_stats.append(_compute_group_stats(name, values))

    # Sort by mean descending
    group_stats.sort(key=lambda g: g.mean, reverse=True)

    best = group_stats[0]
    worst = group_stats[-1]

    # Overall mean
    all_values = [v for vals in groups.values() for v in vals]
    overall_mean = sum(all_values) / len(all_values) if all_values else 0

    spread = best.mean - worst.mean
    spread_pct = (spread / abs(overall_mean) * 100) if overall_mean != 0 else 0

    # Ranking
    ranking = []
    for i, gs in enumerate(group_stats):
        pct_of_best = (gs.mean / best.mean * 100) if best.mean != 0 else 0
        ranking.append({
            "group": gs.group_name,
            "rank": i + 1,
            "value": round(gs.mean, 4),
            "count": gs.count,
            "pct_of_best": round(pct_of_best, 1),
        })

    # Significant gaps (>20% difference)
    significant_gaps = _find_significant_gaps(group_stats)

    # Summary
    summary = (
        f"{metric_name} across {len(groups)} {group_column} groups: "
        f"Best = {best.group_name} ({best.mean:.2f}), "
        f"Worst = {worst.group_name} ({worst.mean:.2f}), "
        f"Spread = {spread:.2f} ({spread_pct:.1f}%)."
    )

    return BenchmarkResult(
        metric_name=metric_name,
        group_column=group_column,
        groups=group_stats,
        best_group=best.group_name,
        worst_group=worst.group_name,
        spread=round(spread, 4),
        spread_pct=round(spread_pct, 1),
        ranking=ranking,
        significant_gaps=significant_gaps,
        summary=summary,
    )


def rank_entities(
    rows: list[dict],
    entity_column: str,
    metric_column: str,
    ascending: bool = False,
    limit: int = 10,
) -> list[dict]:
    """Rank entities by a metric value.

    Args:
        rows: Data rows.
        entity_column: Column identifying entities.
        metric_column: Column to rank by.
        ascending: If True, lower is better.
        limit: Max entities to return.

    Returns:
        List of {entity, value, rank} dicts.
    """
    # Aggregate by entity (mean)
    entity_values: dict[str, list[float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        value = row.get(metric_column)
        if entity is None or value is None:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        entity_values.setdefault(str(entity), []).append(fval)

    # Compute means
    entities = [
        {"entity": name, "value": round(sum(vals) / len(vals), 4), "count": len(vals)}
        for name, vals in entity_values.items()
    ]

    entities.sort(key=lambda e: e["value"], reverse=not ascending)

    for i, e in enumerate(entities[:limit]):
        e["rank"] = i + 1

    return entities[:limit]


def compare_two_groups(
    group_a_values: list[float],
    group_b_values: list[float],
    group_a_name: str = "A",
    group_b_name: str = "B",
) -> dict:
    """Compare two groups statistically.

    Returns a dict with means, medians, difference, and significance assessment.
    """
    if not group_a_values or not group_b_values:
        return {"error": "Both groups must have values"}

    stats_a = _compute_group_stats(group_a_name, group_a_values)
    stats_b = _compute_group_stats(group_b_name, group_b_values)

    diff = stats_a.mean - stats_b.mean
    abs_diff = abs(diff)
    pooled_std = ((stats_a.std ** 2 + stats_b.std ** 2) / 2) ** 0.5

    # Simple effect size (Cohen's d approximation)
    effect_size = abs_diff / pooled_std if pooled_std > 0 else 0

    if effect_size > 0.8:
        significance = "large"
    elif effect_size > 0.5:
        significance = "medium"
    elif effect_size > 0.2:
        significance = "small"
    else:
        significance = "negligible"

    winner = group_a_name if stats_a.mean > stats_b.mean else group_b_name if stats_b.mean > stats_a.mean else "tie"
    pct_diff = (diff / abs(stats_b.mean) * 100) if stats_b.mean != 0 else 0

    return {
        "group_a": {"name": group_a_name, "mean": round(stats_a.mean, 4), "median": round(stats_a.median, 4), "std": round(stats_a.std, 4), "count": stats_a.count},
        "group_b": {"name": group_b_name, "mean": round(stats_b.mean, 4), "median": round(stats_b.median, 4), "std": round(stats_b.std, 4), "count": stats_b.count},
        "difference": round(diff, 4),
        "pct_difference": round(pct_diff, 2),
        "effect_size": round(effect_size, 3),
        "significance": significance,
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_group_stats(name: str, values: list[float]) -> GroupStats:
    """Compute descriptive stats for a group."""
    n = len(values)
    total = sum(values)
    mean = total / n if n > 0 else 0
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n > 1 else 0
    return GroupStats(
        group_name=name,
        count=n,
        mean=round(mean, 4),
        median=round(median, 4),
        min_val=round(min(values), 4) if values else 0,
        max_val=round(max(values), 4) if values else 0,
        std=round(std, 4),
        total=round(total, 4),
    )


def _find_significant_gaps(group_stats: list[GroupStats], threshold: float = 0.2) -> list[dict]:
    """Find pairs of groups with significant differences (>threshold as fraction of larger)."""
    gaps = []
    for i, a in enumerate(group_stats):
        for b in group_stats[i + 1:]:
            diff = abs(a.mean - b.mean)
            max_mean = max(abs(a.mean), abs(b.mean))
            if max_mean > 0 and diff / max_mean > threshold:
                gaps.append({
                    "group_a": a.group_name,
                    "group_b": b.group_name,
                    "diff": round(diff, 4),
                    "diff_pct": round(diff / max_mean * 100, 1),
                })
    return gaps
