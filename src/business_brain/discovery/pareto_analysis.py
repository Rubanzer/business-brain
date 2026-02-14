"""Pareto analysis — identify the vital few that drive the majority of impact.

Pure functions for 80/20 analysis, cumulative contribution, and
identifying the most impactful items in a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParetoItem:
    """A single item in the Pareto analysis."""
    name: str
    value: float
    pct_of_total: float
    cumulative_pct: float
    rank: int
    is_vital: bool  # in the vital few (top contributors)


@dataclass
class ParetoResult:
    """Complete Pareto analysis result."""
    category_column: str
    metric_column: str
    total: float
    items: list[ParetoItem]
    vital_few_count: int
    vital_few_pct: float  # what % of items are "vital"
    vital_few_contribution: float  # what % of total they contribute
    trivial_many_count: int
    pareto_ratio: float  # actual ratio (e.g., 0.82 means 82% from vital few)
    is_pareto: bool  # True if top 20% contribute >= 70% of total
    summary: str


def pareto_analysis(
    rows: list[dict],
    category_column: str,
    metric_column: str,
    threshold: float = 80.0,
) -> ParetoResult | None:
    """Perform Pareto (80/20) analysis on data.

    Args:
        rows: Data rows as dicts.
        category_column: Column to group by (e.g., "supplier", "product").
        metric_column: Numeric column to sum (e.g., "cost", "revenue").
        threshold: Cumulative % threshold for "vital few" (default 80%).

    Returns:
        ParetoResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate by category
    aggregated: dict[str, float] = {}
    for row in rows:
        cat = row.get(category_column)
        val = row.get(metric_column)
        if cat is None or val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        aggregated[str(cat)] = aggregated.get(str(cat), 0) + abs(fval)

    if len(aggregated) < 2:
        return None

    total = sum(aggregated.values())
    if total == 0:
        return None

    # Sort by value descending
    sorted_items = sorted(aggregated.items(), key=lambda x: -x[1])

    # Build Pareto items
    items = []
    cumulative = 0.0
    vital_count = 0
    for i, (name, value) in enumerate(sorted_items):
        pct = value / total * 100
        cumulative += pct
        is_vital = cumulative <= threshold or vital_count == 0
        if is_vital:
            vital_count = i + 1
        items.append(ParetoItem(
            name=name,
            value=round(value, 4),
            pct_of_total=round(pct, 2),
            cumulative_pct=round(cumulative, 2),
            rank=i + 1,
            is_vital=is_vital,
        ))

    # Calculate Pareto metrics
    vital_contribution = sum(it.pct_of_total for it in items if it.is_vital)
    vital_pct = vital_count / len(items) * 100
    trivial_count = len(items) - vital_count

    # Check if it follows Pareto distribution
    top_20_pct_count = max(1, int(len(items) * 0.2))
    top_20_contribution = sum(it.pct_of_total for it in items[:top_20_pct_count])
    is_pareto = top_20_contribution >= 70

    summary = (
        f"Pareto analysis of {metric_column} by {category_column}: "
        f"{vital_count} of {len(items)} {category_column}s ({vital_pct:.0f}%) "
        f"account for {vital_contribution:.1f}% of total {metric_column}. "
    )
    if is_pareto:
        summary += f"Strong Pareto effect: top 20% contributes {top_20_contribution:.0f}%."
    else:
        summary += f"Weak Pareto effect: top 20% contributes {top_20_contribution:.0f}%."

    return ParetoResult(
        category_column=category_column,
        metric_column=metric_column,
        total=round(total, 4),
        items=items,
        vital_few_count=vital_count,
        vital_few_pct=round(vital_pct, 1),
        vital_few_contribution=round(vital_contribution, 1),
        trivial_many_count=trivial_count,
        pareto_ratio=round(top_20_contribution / 100, 3),
        is_pareto=is_pareto,
        summary=summary,
    )


def find_concentration_risk(result: ParetoResult, risk_threshold: float = 50.0) -> list[dict]:
    """Find items that represent concentration risk.

    Items contributing > risk_threshold% alone are flagged.
    """
    risks = []
    for item in result.items:
        if item.pct_of_total >= risk_threshold:
            risks.append({
                "name": item.name,
                "contribution_pct": item.pct_of_total,
                "value": item.value,
                "risk_level": "high" if item.pct_of_total >= 70 else "medium",
                "message": f"{item.name} alone accounts for {item.pct_of_total:.1f}% of total.",
            })
    return risks


def compare_pareto(result_a: ParetoResult, result_b: ParetoResult) -> dict:
    """Compare two Pareto analyses (e.g., two time periods).

    Returns summary of how concentration has changed.
    """
    return {
        "metric": result_a.metric_column,
        "period_a": {
            "total": result_a.total,
            "vital_count": result_a.vital_few_count,
            "concentration": result_a.pareto_ratio,
        },
        "period_b": {
            "total": result_b.total,
            "vital_count": result_b.vital_few_count,
            "concentration": result_b.pareto_ratio,
        },
        "total_change_pct": round(
            (result_b.total - result_a.total) / abs(result_a.total) * 100 if result_a.total else 0, 1
        ),
        "concentration_change": round(result_b.pareto_ratio - result_a.pareto_ratio, 3),
        "more_concentrated": result_b.pareto_ratio > result_a.pareto_ratio,
    }
