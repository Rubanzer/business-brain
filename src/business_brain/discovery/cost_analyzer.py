"""Cost analysis — breakdown, drivers, cost-volume-profit, and benchmarking.

Pure functions for manufacturing cost analysis including cost breakdown by
category, cost per unit computation, cost trends over time, breakeven
analysis, and actual-vs-budget cost variance.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Cost Breakdown
# ---------------------------------------------------------------------------


@dataclass
class CostCategory:
    """A single cost category in the breakdown."""

    name: str
    amount: float
    share_pct: float
    cumulative_pct: float
    rank: int


@dataclass
class CostBreakdown:
    """Complete cost breakdown result."""

    categories: list[CostCategory]
    total_cost: float
    top_category: str
    top_3_share_pct: float
    summary: str


def cost_breakdown(
    rows: list[dict],
    category_column: str,
    amount_column: str,
    total_column: str | None = None,
) -> CostBreakdown | None:
    """Break total cost into categories and compute share of each.

    Args:
        rows: Data rows as dicts.
        category_column: Column identifying the cost category.
        amount_column: Column with cost amount.
        total_column: Optional column with a pre-computed total for each row.
            When provided the share is computed as amount / total. When absent
            the share is computed as amount / sum-of-all-amounts.

    Returns:
        CostBreakdown or None if no valid data.
    """
    if not rows:
        return None

    aggregated: dict[str, float] = {}
    row_total_sum = 0.0
    for row in rows:
        cat = row.get(category_column)
        amt = row.get(amount_column)
        if cat is None or amt is None:
            continue
        try:
            amt_val = float(amt)
        except (TypeError, ValueError):
            continue
        aggregated[str(cat)] = aggregated.get(str(cat), 0.0) + amt_val

        if total_column is not None:
            tot_raw = row.get(total_column)
            if tot_raw is not None:
                try:
                    row_total_sum += float(tot_raw)
                except (TypeError, ValueError):
                    pass

    if not aggregated:
        return None

    grand_total = row_total_sum if (total_column is not None and row_total_sum != 0) else sum(aggregated.values())

    if grand_total == 0:
        return None

    # Sort by amount descending
    sorted_items = sorted(aggregated.items(), key=lambda x: -x[1])

    categories: list[CostCategory] = []
    cumulative = 0.0
    for i, (name, amount) in enumerate(sorted_items):
        share = amount / grand_total * 100
        cumulative += share
        categories.append(
            CostCategory(
                name=name,
                amount=round(amount, 4),
                share_pct=round(share, 2),
                cumulative_pct=round(cumulative, 2),
                rank=i + 1,
            )
        )

    total_cost = sum(aggregated.values())
    top_category = categories[0].name
    top_3_share = sum(c.share_pct for c in categories[:3])

    summary = (
        f"Cost breakdown across {len(categories)} categories: "
        f"Total cost = {total_cost:,.2f}. "
        f"Top category: {top_category} ({categories[0].share_pct:.1f}%). "
        f"Top 3 categories account for {top_3_share:.1f}% of total."
    )

    return CostBreakdown(
        categories=categories,
        total_cost=round(total_cost, 4),
        top_category=top_category,
        top_3_share_pct=round(top_3_share, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Cost Per Unit
# ---------------------------------------------------------------------------


@dataclass
class EntityCPU:
    """Cost per unit for a single entity."""

    entity: str
    total_cost: float
    total_quantity: float
    cost_per_unit: float
    deviation_from_mean_pct: float


@dataclass
class CostPerUnitResult:
    """Aggregated cost per unit result."""

    entities: list[EntityCPU]
    mean_cpu: float
    median_cpu: float
    best_entity: str
    worst_entity: str
    spread_pct: float  # (max - min) as % of mean
    summary: str


def cost_per_unit(
    rows: list[dict],
    entity_column: str,
    cost_column: str,
    quantity_column: str,
) -> CostPerUnitResult | None:
    """Compute cost per unit for each entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (plant, line, etc.).
        cost_column: Column with cost values.
        quantity_column: Column with quantity produced.

    Returns:
        CostPerUnitResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        cost = row.get(cost_column)
        qty = row.get(quantity_column)
        if entity is None or cost is None or qty is None:
            continue
        try:
            c_val = float(cost)
            q_val = float(qty)
        except (TypeError, ValueError):
            continue
        key = str(entity)
        if key not in acc:
            acc[key] = {"cost": 0.0, "qty": 0.0}
        acc[key]["cost"] += c_val
        acc[key]["qty"] += q_val

    if not acc:
        return None

    entities: list[EntityCPU] = []
    cpus: list[float] = []
    for name, vals in acc.items():
        if vals["qty"] == 0:
            cpu = float("inf")
        else:
            cpu = vals["cost"] / vals["qty"]
        cpus.append(cpu)
        entities.append(
            EntityCPU(
                entity=name,
                total_cost=round(vals["cost"], 4),
                total_quantity=round(vals["qty"], 4),
                cost_per_unit=round(cpu, 4) if cpu != float("inf") else float("inf"),
                deviation_from_mean_pct=0.0,  # filled below
            )
        )

    # Filter out infinite CPUs for statistics
    finite_cpus = [c for c in cpus if c != float("inf")]
    if not finite_cpus:
        return None

    mean_cpu = sum(finite_cpus) / len(finite_cpus)
    median_cpu = statistics.median(finite_cpus)

    # Fill deviation and sort
    for e in entities:
        if e.cost_per_unit != float("inf") and mean_cpu != 0:
            e.deviation_from_mean_pct = round((e.cost_per_unit - mean_cpu) / mean_cpu * 100, 2)
        else:
            e.deviation_from_mean_pct = 0.0

    # Sort by CPU ascending (best = lowest)
    entities.sort(key=lambda e: e.cost_per_unit)
    best = entities[0].entity
    worst = entities[-1].entity

    # Spread
    min_cpu = min(finite_cpus)
    max_cpu = max(finite_cpus)
    spread = ((max_cpu - min_cpu) / mean_cpu * 100) if mean_cpu != 0 else 0.0

    summary = (
        f"Cost per unit across {len(entities)} entities: "
        f"Mean CPU = {mean_cpu:,.2f}, Median CPU = {median_cpu:,.2f}. "
        f"Best = {best}, Worst = {worst}. "
        f"Spread = {spread:.1f}% of mean."
    )

    return CostPerUnitResult(
        entities=entities,
        mean_cpu=round(mean_cpu, 4),
        median_cpu=round(median_cpu, 4),
        best_entity=best,
        worst_entity=worst,
        spread_pct=round(spread, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Cost Trend
# ---------------------------------------------------------------------------


@dataclass
class PeriodCost:
    """Cost for a single time period."""

    period: str
    total_cost: float
    change_from_prev: float
    change_pct: float


@dataclass
class CostTrendResult:
    """Cost trend analysis result."""

    periods: list[PeriodCost]
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_pct_per_period: float
    total_change_pct: float
    volatility: float
    summary: str


def cost_trend(
    rows: list[dict],
    time_column: str,
    cost_column: str,
    entity_column: str | None = None,
) -> CostTrendResult | None:
    """Track cost changes over time periods.

    When entity_column is provided, costs are summed across all entities
    within each period.

    Args:
        rows: Data rows as dicts.
        time_column: Column with time period identifier.
        cost_column: Column with cost value.
        entity_column: Optional column to filter/aggregate by entity.

    Returns:
        CostTrendResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate by period
    period_agg: dict[str, float] = {}
    for row in rows:
        period = row.get(time_column)
        cost = row.get(cost_column)
        if period is None or cost is None:
            continue
        try:
            c_val = float(cost)
        except (TypeError, ValueError):
            continue
        key = str(period)
        period_agg[key] = period_agg.get(key, 0.0) + c_val

    if not period_agg:
        return None

    # Sort periods by natural order (string sort)
    sorted_periods = sorted(period_agg.items(), key=lambda x: x[0])

    periods: list[PeriodCost] = []
    changes: list[float] = []
    prev_cost: float | None = None
    for period_name, total_cost in sorted_periods:
        if prev_cost is not None and prev_cost != 0:
            change = total_cost - prev_cost
            change_pct = change / abs(prev_cost) * 100
        elif prev_cost is not None:
            change = total_cost - prev_cost
            change_pct = 0.0
        else:
            change = 0.0
            change_pct = 0.0

        if prev_cost is not None:
            changes.append(change_pct)

        periods.append(
            PeriodCost(
                period=period_name,
                total_cost=round(total_cost, 4),
                change_from_prev=round(change, 4),
                change_pct=round(change_pct, 2),
            )
        )
        prev_cost = total_cost

    if len(periods) < 2:
        return None

    # Trend direction
    first_cost = periods[0].total_cost
    last_cost = periods[-1].total_cost
    total_change_pct = ((last_cost - first_cost) / abs(first_cost) * 100) if first_cost != 0 else 0.0
    avg_change = sum(changes) / len(changes) if changes else 0.0

    if avg_change > 2.0:
        trend_direction = "increasing"
    elif avg_change < -2.0:
        trend_direction = "decreasing"
    else:
        trend_direction = "stable"

    # Volatility: standard deviation of period-over-period changes
    if len(changes) >= 2:
        volatility = statistics.stdev(changes)
    elif len(changes) == 1:
        volatility = 0.0
    else:
        volatility = 0.0

    summary = (
        f"Cost trend over {len(periods)} periods: "
        f"Direction = {trend_direction} ({avg_change:+.1f}% avg per period). "
        f"Total change = {total_change_pct:+.1f}%. "
        f"Volatility = {volatility:.1f}%."
    )

    return CostTrendResult(
        periods=periods,
        trend_direction=trend_direction,
        trend_pct_per_period=round(avg_change, 2),
        total_change_pct=round(total_change_pct, 2),
        volatility=round(volatility, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Breakeven Analysis
# ---------------------------------------------------------------------------


@dataclass
class BreakevenResult:
    """Breakeven analysis result."""

    breakeven_units: float
    breakeven_revenue: float
    contribution_margin: float
    contribution_margin_ratio: float
    summary: str


def breakeven_analysis(
    fixed_costs: float,
    variable_cost_per_unit: float,
    price_per_unit: float,
) -> BreakevenResult | None:
    """Compute break-even point.

    BEP (units) = fixed_costs / (price_per_unit - variable_cost_per_unit)

    Args:
        fixed_costs: Total fixed costs.
        variable_cost_per_unit: Variable cost per unit produced.
        price_per_unit: Selling price per unit.

    Returns:
        BreakevenResult or None if contribution margin is zero or negative.
    """
    contribution_margin = price_per_unit - variable_cost_per_unit

    if contribution_margin <= 0:
        return None

    breakeven_units = fixed_costs / contribution_margin
    breakeven_revenue = breakeven_units * price_per_unit
    cm_ratio = contribution_margin / price_per_unit if price_per_unit != 0 else 0.0

    summary = (
        f"Breakeven at {breakeven_units:,.1f} units "
        f"(revenue = {breakeven_revenue:,.2f}). "
        f"Contribution margin = {contribution_margin:,.2f}/unit "
        f"({cm_ratio:.1%} of price). "
        f"Fixed costs = {fixed_costs:,.2f}, "
        f"Variable cost/unit = {variable_cost_per_unit:,.2f}, "
        f"Price/unit = {price_per_unit:,.2f}."
    )

    return BreakevenResult(
        breakeven_units=round(breakeven_units, 4),
        breakeven_revenue=round(breakeven_revenue, 4),
        contribution_margin=round(contribution_margin, 4),
        contribution_margin_ratio=round(cm_ratio, 4),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Cost Variance
# ---------------------------------------------------------------------------


@dataclass
class EntityVariance:
    """Actual vs budget variance for a single entity."""

    entity: str
    actual: float
    budget: float
    variance: float
    variance_pct: float
    status: str  # "favorable", "unfavorable", "on_budget"


@dataclass
class CostVarianceResult:
    """Aggregated cost variance result."""

    entities: list[EntityVariance]
    total_actual: float
    total_budget: float
    total_variance: float
    total_variance_pct: float
    favorable_count: int
    unfavorable_count: int
    summary: str


def cost_variance(
    rows: list[dict],
    entity_column: str,
    actual_column: str,
    budget_column: str,
) -> CostVarianceResult | None:
    """Actual vs budget cost variance analysis.

    Variance = actual - budget. Favorable when actual < budget (cost savings).

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        actual_column: Column with actual cost.
        budget_column: Column with budgeted cost.

    Returns:
        CostVarianceResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        actual = row.get(actual_column)
        budget = row.get(budget_column)
        if entity is None or actual is None or budget is None:
            continue
        try:
            a_val = float(actual)
            b_val = float(budget)
        except (TypeError, ValueError):
            continue
        key = str(entity)
        if key not in acc:
            acc[key] = {"actual": 0.0, "budget": 0.0}
        acc[key]["actual"] += a_val
        acc[key]["budget"] += b_val

    if not acc:
        return None

    entities: list[EntityVariance] = []
    for name, vals in acc.items():
        actual_val = vals["actual"]
        budget_val = vals["budget"]
        variance = actual_val - budget_val
        variance_pct = (variance / abs(budget_val) * 100) if budget_val != 0 else 0.0

        # Status: within 2% is on_budget
        if abs(variance_pct) <= 2.0:
            status = "on_budget"
        elif variance < 0:
            status = "favorable"
        else:
            status = "unfavorable"

        entities.append(
            EntityVariance(
                entity=name,
                actual=round(actual_val, 4),
                budget=round(budget_val, 4),
                variance=round(variance, 4),
                variance_pct=round(variance_pct, 2),
                status=status,
            )
        )

    # Sort by absolute variance descending (biggest variance first)
    entities.sort(key=lambda e: -abs(e.variance))

    total_actual = sum(e.actual for e in entities)
    total_budget = sum(e.budget for e in entities)
    total_variance = total_actual - total_budget
    total_variance_pct = (total_variance / abs(total_budget) * 100) if total_budget != 0 else 0.0

    favorable_count = sum(1 for e in entities if e.status == "favorable")
    unfavorable_count = sum(1 for e in entities if e.status == "unfavorable")

    summary = (
        f"Cost variance across {len(entities)} entities: "
        f"Total actual = {total_actual:,.2f}, Total budget = {total_budget:,.2f}. "
        f"Variance = {total_variance:+,.2f} ({total_variance_pct:+.1f}%). "
        f"Favorable: {favorable_count}, Unfavorable: {unfavorable_count}."
    )

    return CostVarianceResult(
        entities=entities,
        total_actual=round(total_actual, 4),
        total_budget=round(total_budget, 4),
        total_variance=round(total_variance, 4),
        total_variance_pct=round(total_variance_pct, 2),
        favorable_count=favorable_count,
        unfavorable_count=unfavorable_count,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Combined Cost Report
# ---------------------------------------------------------------------------


def format_cost_report(
    breakdown: CostBreakdown | None = None,
    cpu: CostPerUnitResult | None = None,
    trend: CostTrendResult | None = None,
    variance: CostVarianceResult | None = None,
) -> str:
    """Generate a combined text report from available cost analyses.

    Args:
        breakdown: Cost breakdown result.
        cpu: Cost per unit result.
        trend: Cost trend result.
        variance: Cost variance result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Cost Analysis Report")
    sections.append("=" * 40)

    if breakdown is not None:
        lines = ["", "Cost Breakdown", "-" * 38]
        for c in breakdown.categories:
            lines.append(
                f"  #{c.rank} {c.name}: {c.amount:,.2f} "
                f"({c.share_pct:.1f}%, cumulative: {c.cumulative_pct:.1f}%)"
            )
        lines.append(f"  Total: {breakdown.total_cost:,.2f}")
        lines.append(f"  Top category: {breakdown.top_category}")
        lines.append(f"  Top 3 share: {breakdown.top_3_share_pct:.1f}%")
        sections.append("\n".join(lines))

    if cpu is not None:
        lines = ["", "Cost Per Unit", "-" * 38]
        for e in cpu.entities:
            cpu_str = f"{e.cost_per_unit:,.2f}" if e.cost_per_unit != float("inf") else "inf"
            lines.append(
                f"  {e.entity}: CPU={cpu_str} "
                f"(deviation: {e.deviation_from_mean_pct:+.1f}%)"
            )
        lines.append(f"  Mean CPU: {cpu.mean_cpu:,.2f} | Median CPU: {cpu.median_cpu:,.2f}")
        lines.append(f"  Spread: {cpu.spread_pct:.1f}% of mean")
        sections.append("\n".join(lines))

    if trend is not None:
        lines = ["", "Cost Trend", "-" * 38]
        for p in trend.periods:
            change_str = f" ({p.change_pct:+.1f}%)" if p.change_from_prev != 0 else ""
            lines.append(f"  {p.period}: {p.total_cost:,.2f}{change_str}")
        lines.append(
            f"  Direction: {trend.trend_direction} "
            f"({trend.trend_pct_per_period:+.1f}%/period)"
        )
        lines.append(f"  Total change: {trend.total_change_pct:+.1f}%")
        lines.append(f"  Volatility: {trend.volatility:.1f}%")
        sections.append("\n".join(lines))

    if variance is not None:
        lines = ["", "Cost Variance (Actual vs Budget)", "-" * 38]
        for e in variance.entities:
            lines.append(
                f"  {e.entity}: actual={e.actual:,.2f} budget={e.budget:,.2f} "
                f"variance={e.variance:+,.2f} ({e.variance_pct:+.1f}%) [{e.status}]"
            )
        lines.append(
            f"  Total: actual={variance.total_actual:,.2f} "
            f"budget={variance.total_budget:,.2f} "
            f"variance={variance.total_variance:+,.2f} ({variance.total_variance_pct:+.1f}%)"
        )
        lines.append(
            f"  Favorable: {variance.favorable_count} | "
            f"Unfavorable: {variance.unfavorable_count}"
        )
        sections.append("\n".join(lines))

    if breakdown is None and cpu is None and trend is None and variance is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
