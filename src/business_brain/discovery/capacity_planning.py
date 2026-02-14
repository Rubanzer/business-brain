"""Capacity planning — utilization analysis and bottleneck detection for manufacturing.

Pure functions for computing capacity utilization, detecting production
bottlenecks, and forecasting capacity exhaustion across entities and time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EntityUtilization:
    """Utilization metrics for a single entity (plant, line, machine)."""
    entity: str
    total_actual: float
    total_capacity: float
    utilization_pct: float
    periods: int
    per_period: list[dict] = field(default_factory=list)
    # per_period items: {"period": ..., "actual": ..., "capacity": ..., "utilization_pct": ...}


@dataclass
class UtilizationResult:
    """Complete capacity utilization analysis."""
    entity_count: int
    mean_utilization: float
    entities: list[EntityUtilization]
    over_utilized: list[str]   # entities > 95%
    under_utilized: list[str]  # entities < 50%
    bottlenecks: list[str]     # entities > 95% (alias for quick reference)
    summary: str


@dataclass
class Bottleneck:
    """A production stage assessed for bottleneck status."""
    stage: str
    throughput: float
    throughput_pct_of_max: float
    is_bottleneck: bool
    constraint_ratio: float  # stage throughput / max throughput


@dataclass
class ExhaustionForecast:
    """Forecast of when an entity will exhaust its capacity."""
    entity: str
    current_utilization: float
    trend_per_period: float       # utilization change per period (percentage points)
    periods_to_exhaustion: float | None  # None if declining or already over
    urgency: str                  # "critical", "warning", "ok"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _linear_slope(ys: list[float]) -> float:
    """Compute slope of a simple linear regression (y vs 0-based index).

    Returns 0.0 if fewer than 2 points.
    """
    n = len(ys)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n
    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ys))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_utilization(
    rows: list[dict],
    entity_column: str,
    actual_column: str,
    capacity_column: str,
    time_column: str | None = None,
) -> UtilizationResult | None:
    """Compute capacity utilization for each entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (plant, line, etc.).
        actual_column: Column with actual production / usage.
        capacity_column: Column with maximum capacity.
        time_column: Optional column for time period grouping.

    Returns:
        UtilizationResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect data per entity (and optionally per period)
    # Structure: entity -> list of (period, actual, capacity)
    entity_data: dict[str, list[tuple[str | None, float, float]]] = {}

    for row in rows:
        ent = row.get(entity_column)
        act = _safe_float(row.get(actual_column))
        cap = _safe_float(row.get(capacity_column))
        if ent is None or act is None or cap is None:
            continue
        period = str(row.get(time_column)) if time_column and row.get(time_column) is not None else None
        entity_data.setdefault(str(ent), []).append((period, act, cap))

    if not entity_data:
        return None

    entities: list[EntityUtilization] = []

    for ent_name in sorted(entity_data.keys()):
        records = entity_data[ent_name]
        total_actual = sum(r[1] for r in records)
        total_capacity = sum(r[2] for r in records)
        utilization = (total_actual / total_capacity * 100) if total_capacity > 0 else 0.0

        per_period: list[dict] = []
        if time_column is not None:
            # Group by period preserving insertion order
            period_order: list[str] = []
            period_data: dict[str, tuple[float, float]] = {}
            for period, act, cap in records:
                key = period if period is not None else "__none__"
                if key not in period_data:
                    period_order.append(key)
                    period_data[key] = (0.0, 0.0)
                prev_act, prev_cap = period_data[key]
                period_data[key] = (prev_act + act, prev_cap + cap)

            for p in period_order:
                p_act, p_cap = period_data[p]
                p_util = (p_act / p_cap * 100) if p_cap > 0 else 0.0
                per_period.append({
                    "period": p,
                    "actual": round(p_act, 4),
                    "capacity": round(p_cap, 4),
                    "utilization_pct": round(p_util, 2),
                })

        entities.append(EntityUtilization(
            entity=ent_name,
            total_actual=round(total_actual, 4),
            total_capacity=round(total_capacity, 4),
            utilization_pct=round(utilization, 2),
            periods=len(per_period) if per_period else 1,
            per_period=per_period,
        ))

    mean_util = sum(e.utilization_pct for e in entities) / len(entities)

    over = [e.entity for e in entities if e.utilization_pct > 95]
    under = [e.entity for e in entities if e.utilization_pct < 50]

    summary = (
        f"Capacity utilization across {len(entities)} entities: "
        f"mean {mean_util:.1f}%. "
        f"{len(over)} over-utilized (>95%), "
        f"{len(under)} under-utilized (<50%)."
    )
    if over:
        summary += f" Bottleneck entities: {', '.join(over)}."

    return UtilizationResult(
        entity_count=len(entities),
        mean_utilization=round(mean_util, 2),
        entities=entities,
        over_utilized=over,
        under_utilized=under,
        bottlenecks=over,
        summary=summary,
    )


def detect_bottlenecks(
    rows: list[dict],
    stage_column: str,
    throughput_column: str,
    time_column: str | None = None,
) -> list[Bottleneck]:
    """Detect production bottlenecks by comparing stage throughputs.

    A stage is flagged as a bottleneck if its throughput is less than 75%
    of the maximum stage throughput.

    Args:
        rows: Data rows as dicts.
        stage_column: Column identifying the production stage.
        throughput_column: Numeric throughput column.
        time_column: Optional; if given, throughput is averaged across periods.

    Returns:
        List of Bottleneck objects (one per stage), ordered by throughput ascending.
    """
    if not rows:
        return []

    # Aggregate throughput per stage
    stage_totals: dict[str, float] = {}
    stage_counts: dict[str, int] = {}

    for row in rows:
        stage = row.get(stage_column)
        tp = _safe_float(row.get(throughput_column))
        if stage is None or tp is None:
            continue
        key = str(stage)
        stage_totals[key] = stage_totals.get(key, 0.0) + tp
        stage_counts[key] = stage_counts.get(key, 0) + 1

    if not stage_totals:
        return []

    # Compute mean throughput per stage
    stage_mean: dict[str, float] = {}
    for s in stage_totals:
        stage_mean[s] = stage_totals[s] / stage_counts[s]

    max_tp = max(stage_mean.values())
    if max_tp == 0:
        return [
            Bottleneck(
                stage=s,
                throughput=round(stage_mean[s], 4),
                throughput_pct_of_max=0.0,
                is_bottleneck=False,
                constraint_ratio=0.0,
            )
            for s in sorted(stage_mean.keys())
        ]

    results: list[Bottleneck] = []
    for s in sorted(stage_mean.keys()):
        tp = stage_mean[s]
        pct_of_max = tp / max_tp * 100
        constraint = tp / max_tp
        is_bn = pct_of_max < 75
        results.append(Bottleneck(
            stage=s,
            throughput=round(tp, 4),
            throughput_pct_of_max=round(pct_of_max, 2),
            is_bottleneck=is_bn,
            constraint_ratio=round(constraint, 4),
        ))

    results.sort(key=lambda b: b.throughput)
    return results


def forecast_capacity_exhaustion(
    rows: list[dict],
    entity_column: str,
    actual_column: str,
    capacity_column: str,
    time_column: str,
) -> list[ExhaustionForecast]:
    """Forecast when each entity will exhaust capacity via linear projection.

    Uses simple linear regression on per-period utilization to estimate
    the number of periods until utilization reaches 100%.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        actual_column: Actual production / usage column.
        capacity_column: Max capacity column.
        time_column: Time period column (required).

    Returns:
        List of ExhaustionForecast objects sorted by urgency.
    """
    if not rows:
        return []

    # Gather per-entity per-period utilization
    # entity -> {period_key: (sum_actual, sum_capacity)}
    entity_periods: dict[str, dict[str, tuple[float, float]]] = {}
    # Track global period ordering
    period_order_set: dict[str, int] = {}
    order_counter = 0

    for row in rows:
        ent = row.get(entity_column)
        act = _safe_float(row.get(actual_column))
        cap = _safe_float(row.get(capacity_column))
        t = row.get(time_column)
        if ent is None or act is None or cap is None or t is None:
            continue
        ent_key = str(ent)
        t_key = str(t)

        if t_key not in period_order_set:
            period_order_set[t_key] = order_counter
            order_counter += 1

        if ent_key not in entity_periods:
            entity_periods[ent_key] = {}
        prev = entity_periods[ent_key].get(t_key, (0.0, 0.0))
        entity_periods[ent_key][t_key] = (prev[0] + act, prev[1] + cap)

    if not entity_periods:
        return []

    # Sort period keys by their first-seen order
    sorted_periods = sorted(period_order_set.keys(), key=lambda p: period_order_set[p])

    results: list[ExhaustionForecast] = []

    for ent_key in sorted(entity_periods.keys()):
        periods_map = entity_periods[ent_key]

        # Build time series of utilization in period order
        util_series: list[float] = []
        for p in sorted_periods:
            if p in periods_map:
                act_sum, cap_sum = periods_map[p]
                util = (act_sum / cap_sum * 100) if cap_sum > 0 else 0.0
                util_series.append(util)

        if not util_series:
            continue

        current_util = util_series[-1]
        trend = _linear_slope(util_series)

        # Determine periods to exhaustion
        if current_util >= 100:
            periods_to_exhaustion = 0.0
            urgency = "critical"
        elif trend <= 0:
            periods_to_exhaustion = None
            urgency = "ok"
        else:
            remaining = 100.0 - current_util
            periods_to_exhaustion = remaining / trend
            if periods_to_exhaustion <= 3:
                urgency = "critical"
            elif periods_to_exhaustion <= 6:
                urgency = "warning"
            else:
                urgency = "ok"

        results.append(ExhaustionForecast(
            entity=ent_key,
            current_utilization=round(current_util, 2),
            trend_per_period=round(trend, 4),
            periods_to_exhaustion=round(periods_to_exhaustion, 2) if periods_to_exhaustion is not None else None,
            urgency=urgency,
        ))

    # Sort by urgency: critical first, then warning, then ok
    urgency_order = {"critical": 0, "warning": 1, "ok": 2}
    results.sort(key=lambda f: (urgency_order.get(f.urgency, 3), f.entity))
    return results


def capacity_summary(result: UtilizationResult) -> str:
    """Generate a human-readable text summary of capacity utilization.

    Args:
        result: A UtilizationResult from compute_utilization.

    Returns:
        Multi-line summary string.
    """
    lines = [
        "Capacity Utilization Report",
        "=" * 50,
        f"Entities analyzed: {result.entity_count}",
        f"Mean utilization:  {result.mean_utilization:.1f}%",
        "",
    ]

    # Per-entity breakdown
    lines.append("Entity Breakdown:")
    for ent in sorted(result.entities, key=lambda e: -e.utilization_pct):
        status = ""
        if ent.utilization_pct > 95:
            status = " [OVER-UTILIZED]"
        elif ent.utilization_pct < 50:
            status = " [UNDER-UTILIZED]"
        lines.append(
            f"  {ent.entity:<20} {ent.utilization_pct:>6.1f}%  "
            f"({ent.total_actual:,.0f} / {ent.total_capacity:,.0f}){status}"
        )

    if result.over_utilized:
        lines.append("")
        lines.append(f"Over-utilized (>95%): {', '.join(result.over_utilized)}")
    if result.under_utilized:
        lines.append("")
        lines.append(f"Under-utilized (<50%): {', '.join(result.under_utilized)}")

    lines.append("")
    lines.append(result.summary)
    return "\n".join(lines)
