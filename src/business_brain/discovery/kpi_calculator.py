"""KPI calculator — pre-built formulas for common business metrics.

Pure functions for computing growth rates, utilization, efficiency,
variance from target, moving averages, and other standard KPIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KPIResult:
    """Result of a KPI calculation."""
    name: str
    value: float
    unit: str
    interpretation: str  # human-readable explanation
    trend: str  # "up", "down", "stable"
    status: str  # "good", "neutral", "bad"


def growth_rate(current: float, previous: float) -> KPIResult:
    """Compute period-over-period growth rate."""
    if previous == 0:
        rate = 100.0 if current > 0 else 0.0
    else:
        rate = (current - previous) / abs(previous) * 100

    trend = "up" if rate > 1 else "down" if rate < -1 else "stable"
    status = "good" if rate > 0 else "bad" if rate < -5 else "neutral"

    return KPIResult(
        name="Growth Rate",
        value=round(rate, 2),
        unit="%",
        interpretation=f"Changed from {previous:,.2f} to {current:,.2f} ({rate:+.1f}%)",
        trend=trend,
        status=status,
    )


def compound_growth_rate(first: float, last: float, periods: int) -> KPIResult:
    """Compute Compound Annual/Period Growth Rate (CAGR)."""
    if first <= 0 or periods <= 0:
        return KPIResult("CAGR", 0.0, "%", "Cannot compute CAGR with non-positive inputs", "stable", "neutral")

    cagr = ((last / first) ** (1 / periods) - 1) * 100
    trend = "up" if cagr > 1 else "down" if cagr < -1 else "stable"
    status = "good" if cagr > 0 else "bad" if cagr < -5 else "neutral"

    return KPIResult(
        name="CAGR",
        value=round(cagr, 2),
        unit="%",
        interpretation=f"Compound growth from {first:,.2f} to {last:,.2f} over {periods} periods: {cagr:+.2f}%",
        trend=trend,
        status=status,
    )


def utilization_rate(used: float, available: float) -> KPIResult:
    """Compute utilization rate (used / available * 100)."""
    if available <= 0:
        return KPIResult("Utilization Rate", 0.0, "%", "No available capacity", "stable", "neutral")

    rate = used / available * 100
    status = "good" if 60 <= rate <= 85 else "bad" if rate > 95 or rate < 30 else "neutral"

    return KPIResult(
        name="Utilization Rate",
        value=round(rate, 2),
        unit="%",
        interpretation=f"Using {used:,.2f} of {available:,.2f} available ({rate:.1f}%)",
        trend="stable",
        status=status,
    )


def efficiency_ratio(output: float, input_val: float) -> KPIResult:
    """Compute efficiency ratio (output / input)."""
    if input_val == 0:
        return KPIResult("Efficiency", 0.0, "x", "No input value", "stable", "neutral")

    ratio = output / input_val
    status = "good" if ratio > 1.0 else "bad" if ratio < 0.5 else "neutral"

    return KPIResult(
        name="Efficiency Ratio",
        value=round(ratio, 4),
        unit="x",
        interpretation=f"Output {output:,.2f} from input {input_val:,.2f} = {ratio:.2f}x efficiency",
        trend="stable",
        status=status,
    )


def variance_from_target(actual: float, target: float) -> KPIResult:
    """Compute variance from target in absolute and percentage."""
    diff = actual - target
    pct = (diff / abs(target) * 100) if target != 0 else 0

    trend = "up" if diff > 0 else "down" if diff < 0 else "stable"
    status = "good" if abs(pct) < 5 else "neutral" if abs(pct) < 15 else "bad"

    return KPIResult(
        name="Variance from Target",
        value=round(pct, 2),
        unit="%",
        interpretation=f"Actual {actual:,.2f} vs target {target:,.2f}: {diff:+,.2f} ({pct:+.1f}%)",
        trend=trend,
        status=status,
    )


def moving_average(values: list[float], window: int = 3) -> list[float]:
    """Compute simple moving average.

    Returns a list of length len(values), with None-equivalent (NaN) for
    the first (window-1) values, then the moving averages.
    """
    if not values or window <= 0:
        return []

    result = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(values[i])  # not enough data yet
        else:
            window_vals = values[i - window + 1:i + 1]
            result.append(round(sum(window_vals) / len(window_vals), 4))

    return result


def exponential_moving_average(values: list[float], alpha: float = 0.3) -> list[float]:
    """Compute exponential moving average (EMA).

    Args:
        values: Input values.
        alpha: Smoothing factor (0 < alpha < 1). Higher = more weight on recent.

    Returns:
        List of EMA values.
    """
    if not values:
        return []

    alpha = max(0.01, min(alpha, 0.99))
    result = [values[0]]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * result[-1]
        result.append(round(ema, 4))

    return result


def rate_of_change(values: list[float]) -> list[float]:
    """Compute period-over-period rate of change (%).

    Returns list of len(values)-1 with % change between consecutive values.
    """
    if len(values) < 2:
        return []

    changes = []
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            pct = (values[i] - values[i - 1]) / abs(values[i - 1]) * 100
        else:
            pct = 100.0 if values[i] > 0 else -100.0 if values[i] < 0 else 0.0
        changes.append(round(pct, 2))

    return changes


def yield_rate(good_units: float, total_units: float) -> KPIResult:
    """Compute yield/quality rate."""
    if total_units <= 0:
        return KPIResult("Yield Rate", 0.0, "%", "No units produced", "stable", "neutral")

    rate = good_units / total_units * 100
    status = "good" if rate > 95 else "neutral" if rate > 80 else "bad"

    return KPIResult(
        name="Yield Rate",
        value=round(rate, 2),
        unit="%",
        interpretation=f"{good_units:,.0f} good out of {total_units:,.0f} total = {rate:.1f}% yield",
        trend="stable",
        status=status,
    )


def inventory_turnover(cost_of_goods: float, avg_inventory: float) -> KPIResult:
    """Compute inventory turnover ratio."""
    if avg_inventory <= 0:
        return KPIResult("Inventory Turnover", 0.0, "x", "No inventory", "stable", "neutral")

    turnover = cost_of_goods / avg_inventory
    status = "good" if turnover > 6 else "neutral" if turnover > 3 else "bad"

    return KPIResult(
        name="Inventory Turnover",
        value=round(turnover, 2),
        unit="x",
        interpretation=f"COGS {cost_of_goods:,.2f} / Avg inventory {avg_inventory:,.2f} = {turnover:.1f}x turnover",
        trend="stable",
        status=status,
    )


def compute_all_kpis(
    values: list[float],
    target: float | None = None,
    capacity: float | None = None,
) -> list[KPIResult]:
    """Compute a suite of KPIs from a single metric's time series.

    Args:
        values: Time-ordered values.
        target: Optional target value.
        capacity: Optional capacity/maximum value.

    Returns:
        List of applicable KPIs.
    """
    results = []

    if len(values) >= 2:
        results.append(growth_rate(values[-1], values[-2]))

    if len(values) >= 3:
        results.append(compound_growth_rate(values[0], values[-1], len(values) - 1))

    if target is not None and values:
        results.append(variance_from_target(values[-1], target))

    if capacity is not None and capacity > 0 and values:
        results.append(utilization_rate(values[-1], capacity))

    return results
