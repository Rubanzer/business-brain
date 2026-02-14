"""Production scheduler — shift performance, batch optimization, and production planning.

Pure functions for analyzing shift-level production output, batch yield
and throughput, takt time calculations, and plan-vs-actual comparisons
for manufacturing environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ShiftPerformance:
    """Metrics for a single shift."""
    shift: str
    total_output: float
    avg_output: float
    event_count: int
    achievement_pct: float | None  # only when target_column is provided
    std_dev: float
    consistency_grade: str  # A (<10% CV), B (<20%), C (<30%), D (>=30%)


@dataclass
class ShiftPerformanceResult:
    """Complete shift performance analysis."""
    shifts: list[ShiftPerformance]
    best_shift: str
    worst_shift: str
    variance_pct: float
    total_output: float
    summary: str


@dataclass
class BatchInfo:
    """Metrics for a single batch."""
    batch_id: str
    input_qty: float
    output_qty: float
    yield_pct: float
    duration: float | None
    throughput: float | None  # output / duration, if duration available


@dataclass
class BatchResult:
    """Complete batch efficiency analysis."""
    batches: list[BatchInfo]
    mean_yield: float
    mean_throughput: float | None
    best_batch: str
    worst_batch: str
    optimal_batch_size: float | None
    summary: str


@dataclass
class TaktResult:
    """Takt time computation result."""
    takt_time_minutes: float
    total_output: float
    total_periods: int
    avg_output_per_period: float
    cycle_time_estimate: float
    efficiency_pct: float
    summary: str


@dataclass
class PlanEntity:
    """Plan vs actual comparison for a single entity."""
    entity: str
    planned: float
    actual: float
    achievement_pct: float
    variance: float
    status: str  # "over", "under", "on_target" (within 5%)


@dataclass
class PlanResult:
    """Complete plan-vs-actual analysis."""
    entities: list[PlanEntity]
    overall_achievement_pct: float
    over_achievers: list[str]
    under_achievers: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def analyze_shift_performance(
    rows: list[dict],
    shift_column: str,
    output_column: str,
    target_column: str | None = None,
    time_column: str | None = None,
) -> ShiftPerformanceResult | None:
    """Compare production output across shifts.

    Args:
        rows: Data rows as dicts.
        shift_column: Column identifying the shift (e.g., "shift", "shift_name").
        output_column: Numeric column with production output.
        target_column: Optional column with production target per row.
        time_column: Optional time column (reserved for future period grouping).

    Returns:
        ShiftPerformanceResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect output values (and targets) per shift
    shift_outputs: dict[str, list[float]] = {}
    shift_targets: dict[str, list[float]] = {}

    for row in rows:
        shift = row.get(shift_column)
        output = _safe_float(row.get(output_column))
        if shift is None or output is None:
            continue

        key = str(shift)
        shift_outputs.setdefault(key, []).append(output)

        if target_column is not None:
            target = _safe_float(row.get(target_column))
            if target is not None:
                shift_targets.setdefault(key, []).append(target)

    if not shift_outputs:
        return None

    # Build ShiftPerformance for each shift
    shifts: list[ShiftPerformance] = []
    for shift_name in sorted(shift_outputs.keys()):
        values = shift_outputs[shift_name]
        total = sum(values)
        count = len(values)
        avg = total / count if count > 0 else 0.0

        # Standard deviation
        if count >= 2:
            variance = sum((v - avg) ** 2 for v in values) / (count - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0

        # Coefficient of variation for consistency grade
        cv = (std / avg * 100) if avg != 0 else 0.0
        if cv < 10:
            grade = "A"
        elif cv < 20:
            grade = "B"
        elif cv < 30:
            grade = "C"
        else:
            grade = "D"

        # Achievement pct
        achievement = None
        if target_column is not None and shift_name in shift_targets:
            target_total = sum(shift_targets[shift_name])
            if target_total > 0:
                achievement = round(total / target_total * 100, 2)

        shifts.append(ShiftPerformance(
            shift=shift_name,
            total_output=round(total, 4),
            avg_output=round(avg, 4),
            event_count=count,
            achievement_pct=achievement,
            std_dev=round(std, 4),
            consistency_grade=grade,
        ))

    # Sort by total_output descending for best/worst
    sorted_by_output = sorted(shifts, key=lambda s: -s.total_output)
    best = sorted_by_output[0].shift
    worst = sorted_by_output[-1].shift

    total_output = sum(s.total_output for s in shifts)

    # Variance pct: range / mean * 100
    outputs = [s.total_output for s in shifts]
    mean_output = total_output / len(shifts) if shifts else 0.0
    if mean_output != 0 and len(shifts) > 1:
        output_range = max(outputs) - min(outputs)
        variance_pct = round(output_range / mean_output * 100, 2)
    else:
        variance_pct = 0.0

    summary = (
        f"Shift performance across {len(shifts)} shifts: "
        f"total output {total_output:,.2f}. "
        f"Best shift: {best}, worst shift: {worst}. "
        f"Variance between shifts: {variance_pct:.1f}%."
    )

    return ShiftPerformanceResult(
        shifts=shifts,
        best_shift=best,
        worst_shift=worst,
        variance_pct=variance_pct,
        total_output=round(total_output, 4),
        summary=summary,
    )


def analyze_batch_efficiency(
    rows: list[dict],
    batch_column: str,
    input_column: str,
    output_column: str,
    time_column: str | None = None,
    duration_column: str | None = None,
) -> BatchResult | None:
    """Analyze yield and throughput per batch.

    Args:
        rows: Data rows as dicts.
        batch_column: Column identifying the batch.
        input_column: Numeric column with input quantity.
        output_column: Numeric column with output quantity.
        time_column: Optional time column (reserved for future use).
        duration_column: Optional column with batch duration for throughput.

    Returns:
        BatchResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate per batch
    batch_data: dict[str, dict] = {}

    for row in rows:
        batch = row.get(batch_column)
        inp = _safe_float(row.get(input_column))
        out = _safe_float(row.get(output_column))
        if batch is None or inp is None or out is None:
            continue

        key = str(batch)
        dur = _safe_float(row.get(duration_column)) if duration_column else None

        if key not in batch_data:
            batch_data[key] = {"input": 0.0, "output": 0.0, "duration": 0.0, "has_duration": False}

        batch_data[key]["input"] += inp
        batch_data[key]["output"] += out
        if dur is not None:
            batch_data[key]["duration"] += dur
            batch_data[key]["has_duration"] = True

    if not batch_data:
        return None

    # Build BatchInfo objects
    batches: list[BatchInfo] = []
    yields: list[float] = []
    throughputs: list[float] = []
    # For optimal batch size: track (input_qty, yield_pct)
    input_yield_pairs: list[tuple[float, float]] = []

    for batch_id in sorted(batch_data.keys()):
        d = batch_data[batch_id]
        inp = d["input"]
        out = d["output"]
        yield_pct = (out / inp * 100) if inp > 0 else 0.0

        duration = round(d["duration"], 4) if d["has_duration"] else None
        throughput = round(out / d["duration"], 4) if d["has_duration"] and d["duration"] > 0 else None

        batches.append(BatchInfo(
            batch_id=batch_id,
            input_qty=round(inp, 4),
            output_qty=round(out, 4),
            yield_pct=round(yield_pct, 2),
            duration=duration,
            throughput=throughput,
        ))

        yields.append(yield_pct)
        if throughput is not None:
            throughputs.append(throughput)
        input_yield_pairs.append((inp, yield_pct))

    mean_yield = round(sum(yields) / len(yields), 2) if yields else 0.0
    mean_throughput = round(sum(throughputs) / len(throughputs), 2) if throughputs else None

    # Best and worst batch by yield
    sorted_by_yield = sorted(batches, key=lambda b: -b.yield_pct)
    best = sorted_by_yield[0].batch_id
    worst = sorted_by_yield[-1].batch_id

    # Optimal batch size: the input amount that maximizes yield
    optimal_batch_size = None
    if input_yield_pairs:
        best_pair = max(input_yield_pairs, key=lambda p: p[1])
        optimal_batch_size = round(best_pair[0], 4)

    throughput_str = f"{mean_throughput:.2f}" if mean_throughput is not None else "N/A"
    summary = (
        f"Batch analysis across {len(batches)} batches: "
        f"mean yield {mean_yield:.1f}%, mean throughput {throughput_str}. "
        f"Best batch: {best}, worst batch: {worst}."
    )
    if optimal_batch_size is not None:
        summary += f" Optimal batch size (by yield): {optimal_batch_size:,.2f}."

    return BatchResult(
        batches=batches,
        mean_yield=mean_yield,
        mean_throughput=mean_throughput,
        best_batch=best,
        worst_batch=worst,
        optimal_batch_size=optimal_batch_size,
        summary=summary,
    )


def compute_takt_time(
    rows: list[dict],
    output_column: str,
    time_column: str,
    available_minutes_per_period: float = 480,
) -> TaktResult | None:
    """Compute takt time from production data.

    Takt time = available time / demand (or actual output per period).
    Cycle time is estimated as the inverse of average throughput.

    Args:
        rows: Data rows as dicts.
        output_column: Numeric column with production output per period.
        time_column: Column identifying the period (shift, day, etc.).
        available_minutes_per_period: Working minutes per period (default 480 = 8 hrs).

    Returns:
        TaktResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate output per period
    period_output: dict[str, float] = {}

    for row in rows:
        period = row.get(time_column)
        output = _safe_float(row.get(output_column))
        if period is None or output is None:
            continue
        key = str(period)
        period_output[key] = period_output.get(key, 0.0) + output

    if not period_output:
        return None

    total_output = sum(period_output.values())
    total_periods = len(period_output)
    avg_output_per_period = total_output / total_periods if total_periods > 0 else 0.0

    # Takt time = available time / demand per period
    if avg_output_per_period > 0:
        takt = available_minutes_per_period / avg_output_per_period
    else:
        takt = 0.0

    # Cycle time estimate: total available time / total output
    total_available = available_minutes_per_period * total_periods
    if total_output > 0:
        cycle_time = total_available / total_output
    else:
        cycle_time = 0.0

    # Efficiency: takt / cycle * 100 (if cycle > 0)
    # When takt == cycle, efficiency is 100%
    if cycle_time > 0 and takt > 0:
        efficiency = (takt / cycle_time) * 100
    else:
        efficiency = 0.0

    summary = (
        f"Takt time: {takt:.2f} min/unit over {total_periods} periods. "
        f"Total output: {total_output:,.0f}, avg {avg_output_per_period:,.1f}/period. "
        f"Cycle time estimate: {cycle_time:.2f} min/unit. "
        f"Efficiency: {efficiency:.1f}%."
    )

    return TaktResult(
        takt_time_minutes=round(takt, 4),
        total_output=round(total_output, 4),
        total_periods=total_periods,
        avg_output_per_period=round(avg_output_per_period, 4),
        cycle_time_estimate=round(cycle_time, 4),
        efficiency_pct=round(efficiency, 2),
        summary=summary,
    )


def plan_vs_actual(
    rows: list[dict],
    entity_column: str,
    plan_column: str,
    actual_column: str,
) -> PlanResult | None:
    """Compare planned vs actual production per entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (line, product, etc.).
        plan_column: Numeric column with planned output.
        actual_column: Numeric column with actual output.

    Returns:
        PlanResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate per entity
    entity_data: dict[str, dict] = {}

    for row in rows:
        entity = row.get(entity_column)
        planned = _safe_float(row.get(plan_column))
        actual = _safe_float(row.get(actual_column))
        if entity is None or planned is None or actual is None:
            continue

        key = str(entity)
        if key not in entity_data:
            entity_data[key] = {"planned": 0.0, "actual": 0.0}
        entity_data[key]["planned"] += planned
        entity_data[key]["actual"] += actual

    if not entity_data:
        return None

    # Build PlanEntity objects
    entities: list[PlanEntity] = []
    over_achievers: list[str] = []
    under_achievers: list[str] = []

    for ent_name in sorted(entity_data.keys()):
        d = entity_data[ent_name]
        planned = d["planned"]
        actual = d["actual"]
        achievement = (actual / planned * 100) if planned > 0 else 0.0
        variance = actual - planned

        if achievement >= 105:
            status = "over"
        elif achievement <= 95:
            status = "under"
        else:
            status = "on_target"

        entities.append(PlanEntity(
            entity=ent_name,
            planned=round(planned, 4),
            actual=round(actual, 4),
            achievement_pct=round(achievement, 2),
            variance=round(variance, 4),
            status=status,
        ))

        if status == "over":
            over_achievers.append(ent_name)
        elif status == "under":
            under_achievers.append(ent_name)

    total_planned = sum(d["planned"] for d in entity_data.values())
    total_actual = sum(d["actual"] for d in entity_data.values())
    overall_achievement = (total_actual / total_planned * 100) if total_planned > 0 else 0.0

    summary = (
        f"Plan vs actual across {len(entities)} entities: "
        f"overall achievement {overall_achievement:.1f}%. "
        f"{len(over_achievers)} over-achievers, "
        f"{len(under_achievers)} under-achievers, "
        f"{len(entities) - len(over_achievers) - len(under_achievers)} on target."
    )

    return PlanResult(
        entities=entities,
        overall_achievement_pct=round(overall_achievement, 2),
        over_achievers=over_achievers,
        under_achievers=under_achievers,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def format_schedule_report(
    shift: ShiftPerformanceResult | None = None,
    batch: BatchResult | None = None,
    takt: TaktResult | None = None,
    plan: PlanResult | None = None,
) -> str:
    """Generate a combined text report from multiple analysis results.

    Args:
        shift: Optional ShiftPerformanceResult.
        batch: Optional BatchResult.
        takt: Optional TaktResult.
        plan: Optional PlanResult.

    Returns:
        Multi-line formatted report string.
    """
    sections: list[str] = []
    sections.append("Production Schedule Report")
    sections.append("=" * 60)

    if shift is not None:
        sections.append("")
        sections.append("SHIFT PERFORMANCE")
        sections.append("-" * 40)
        sections.append(f"Total output: {shift.total_output:,.2f}")
        sections.append(f"Best shift:   {shift.best_shift}")
        sections.append(f"Worst shift:  {shift.worst_shift}")
        sections.append(f"Variance:     {shift.variance_pct:.1f}%")
        sections.append("")
        for s in shift.shifts:
            target_str = f"  Achievement: {s.achievement_pct:.1f}%" if s.achievement_pct is not None else ""
            sections.append(
                f"  {s.shift:<15} Output: {s.total_output:>10,.2f}  "
                f"Avg: {s.avg_output:>8,.2f}  "
                f"StdDev: {s.std_dev:>8,.2f}  "
                f"Grade: {s.consistency_grade}{target_str}"
            )

    if batch is not None:
        sections.append("")
        sections.append("BATCH EFFICIENCY")
        sections.append("-" * 40)
        sections.append(f"Mean yield:      {batch.mean_yield:.1f}%")
        tp_str = f"{batch.mean_throughput:.2f}" if batch.mean_throughput is not None else "N/A"
        sections.append(f"Mean throughput:  {tp_str}")
        sections.append(f"Best batch:      {batch.best_batch}")
        sections.append(f"Worst batch:     {batch.worst_batch}")
        if batch.optimal_batch_size is not None:
            sections.append(f"Optimal size:    {batch.optimal_batch_size:,.2f}")
        sections.append("")
        for b in batch.batches:
            dur_str = f"  Dur: {b.duration:.1f}" if b.duration is not None else ""
            tp_str2 = f"  Throughput: {b.throughput:.2f}" if b.throughput is not None else ""
            sections.append(
                f"  {b.batch_id:<15} In: {b.input_qty:>8,.2f}  "
                f"Out: {b.output_qty:>8,.2f}  "
                f"Yield: {b.yield_pct:>5.1f}%{dur_str}{tp_str2}"
            )

    if takt is not None:
        sections.append("")
        sections.append("TAKT TIME")
        sections.append("-" * 40)
        sections.append(f"Takt time:        {takt.takt_time_minutes:.2f} min/unit")
        sections.append(f"Total output:     {takt.total_output:,.0f}")
        sections.append(f"Total periods:    {takt.total_periods}")
        sections.append(f"Avg output/period: {takt.avg_output_per_period:,.1f}")
        sections.append(f"Cycle time est:   {takt.cycle_time_estimate:.2f} min/unit")
        sections.append(f"Efficiency:       {takt.efficiency_pct:.1f}%")

    if plan is not None:
        sections.append("")
        sections.append("PLAN VS ACTUAL")
        sections.append("-" * 40)
        sections.append(f"Overall achievement: {plan.overall_achievement_pct:.1f}%")
        sections.append(f"Over-achievers:  {len(plan.over_achievers)}")
        sections.append(f"Under-achievers: {len(plan.under_achievers)}")
        sections.append("")
        for e in plan.entities:
            sections.append(
                f"  {e.entity:<15} Plan: {e.planned:>10,.2f}  "
                f"Actual: {e.actual:>10,.2f}  "
                f"Ach: {e.achievement_pct:>6.1f}%  [{e.status}]"
            )

    if shift is None and batch is None and takt is None and plan is None:
        sections.append("")
        sections.append("No analysis results provided.")

    sections.append("")
    return "\n".join(sections)
