"""Maintenance scheduling and predictive maintenance analytics.

Pure functions for analyzing equipment maintenance history, computing
reliability metrics (MTBF/MTTR), spare parts analysis, and generating
predictive maintenance schedules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


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


def _parse_date(val) -> datetime | None:
    """Try to parse a value into a datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(str(val), fmt)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EquipmentSummary:
    """Maintenance summary for a single piece of equipment."""

    equipment: str
    total_events: int
    type_breakdown: dict[str, int]  # e.g. {"preventive": 5, "corrective": 2}
    corrective_ratio: float  # corrective / total (0.0 to 1.0)
    avg_downtime: float | None  # average duration if duration_column given
    total_downtime: float | None
    total_cost: float | None
    avg_cost: float | None


@dataclass
class MaintenanceResult:
    """Complete maintenance history analysis."""

    equipment_summaries: list[EquipmentSummary]
    total_events: int
    most_maintained_equipment: str | None
    most_common_type: str | None
    overall_corrective_ratio: float
    summary: str


@dataclass
class ReliabilityMetric:
    """Reliability metrics for a single piece of equipment."""

    equipment: str
    mtbf_days: float | None  # Mean Time Between Failures (days)
    mttr_hours: float | None  # Mean Time To Repair (hours from duration)
    availability: float | None  # MTBF / (MTBF + MTTR) * 100
    failure_count: int
    first_failure: datetime | None
    last_failure: datetime | None


@dataclass
class PartSummary:
    """Summary for a single spare part."""

    part: str
    total_quantity: float
    total_cost: float | None
    avg_cost_per_unit: float | None
    abc_category: str | None  # "A", "B", or "C"


@dataclass
class SparePartsResult:
    """Complete spare parts analysis."""

    total_unique_parts: int
    total_quantity: float
    total_spend: float | None
    top_parts: list[PartSummary]  # top 5 by quantity
    parts_by_equipment: dict[str, list[str]] | None  # equipment -> list of parts
    abc_parts: list[PartSummary]  # all parts with ABC category set
    summary: str


@dataclass
class ScheduleEntry:
    """A predicted next maintenance event for an equipment."""

    equipment: str
    last_maintenance_date: datetime | None
    avg_interval_days: float | None
    next_maintenance_date: datetime | None
    is_overdue: bool
    events_in_history: int


# ---------------------------------------------------------------------------
# 1. analyze_maintenance_history
# ---------------------------------------------------------------------------


def analyze_maintenance_history(
    rows: list[dict],
    equipment_column: str,
    date_column: str,
    type_column: str,
    duration_column: str | None = None,
    cost_column: str | None = None,
) -> MaintenanceResult | None:
    """Analyze equipment maintenance history.

    Groups by equipment and computes event counts, type breakdowns,
    corrective ratio, and optional downtime/cost statistics.

    Args:
        rows: Data rows as dicts.
        equipment_column: Column identifying the equipment.
        date_column: Column with maintenance date.
        type_column: Column with maintenance type (preventive/corrective/etc.).
        duration_column: Optional column with maintenance duration (hours).
        cost_column: Optional column with maintenance cost.

    Returns:
        MaintenanceResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect per-equipment data
    equip_types: dict[str, list[str]] = {}
    equip_durations: dict[str, list[float]] = {}
    equip_costs: dict[str, list[float]] = {}

    for row in rows:
        equip = row.get(equipment_column)
        date_val = row.get(date_column)
        mtype = row.get(type_column)
        if equip is None or date_val is None or mtype is None:
            continue

        equip_key = str(equip)
        type_key = str(mtype).lower().strip()
        equip_types.setdefault(equip_key, []).append(type_key)

        if duration_column is not None:
            dur = _safe_float(row.get(duration_column))
            if dur is not None:
                equip_durations.setdefault(equip_key, []).append(dur)

        if cost_column is not None:
            cost = _safe_float(row.get(cost_column))
            if cost is not None:
                equip_costs.setdefault(equip_key, []).append(cost)

    if not equip_types:
        return None

    # Build per-equipment summaries
    summaries: list[EquipmentSummary] = []
    global_type_counts: dict[str, int] = {}
    total_events = 0
    total_corrective = 0

    for equip_key in sorted(equip_types.keys()):
        types_list = equip_types[equip_key]
        count = len(types_list)
        total_events += count

        # Type breakdown
        type_breakdown: dict[str, int] = {}
        for t in types_list:
            type_breakdown[t] = type_breakdown.get(t, 0) + 1
            global_type_counts[t] = global_type_counts.get(t, 0) + 1

        # Corrective ratio: count "corrective" types
        corrective_count = type_breakdown.get("corrective", 0)
        total_corrective += corrective_count
        corrective_ratio = corrective_count / count if count > 0 else 0.0

        # Duration stats
        avg_downtime: float | None = None
        total_downtime: float | None = None
        if duration_column is not None and equip_key in equip_durations:
            durs = equip_durations[equip_key]
            if durs:
                total_downtime = round(sum(durs), 4)
                avg_downtime = round(total_downtime / len(durs), 4)

        # Cost stats
        total_cost: float | None = None
        avg_cost: float | None = None
        if cost_column is not None and equip_key in equip_costs:
            costs = equip_costs[equip_key]
            if costs:
                total_cost = round(sum(costs), 4)
                avg_cost = round(total_cost / len(costs), 4)

        summaries.append(EquipmentSummary(
            equipment=equip_key,
            total_events=count,
            type_breakdown=type_breakdown,
            corrective_ratio=round(corrective_ratio, 4),
            avg_downtime=avg_downtime,
            total_downtime=total_downtime,
            total_cost=total_cost,
            avg_cost=avg_cost,
        ))

    # Overall stats
    most_maintained = max(summaries, key=lambda s: s.total_events).equipment if summaries else None
    most_common_type = max(global_type_counts, key=global_type_counts.get) if global_type_counts else None  # type: ignore[arg-type]
    overall_corrective = total_corrective / total_events if total_events > 0 else 0.0

    summary_parts = [
        f"{total_events} maintenance events across {len(summaries)} equipment(s).",
    ]
    if most_maintained:
        most_obj = next(s for s in summaries if s.equipment == most_maintained)
        summary_parts.append(
            f" Most maintained: {most_maintained} ({most_obj.total_events} events)."
        )
    if most_common_type:
        summary_parts.append(
            f" Most common type: {most_common_type} ({global_type_counts[most_common_type]} events)."
        )
    summary_parts.append(f" Overall corrective ratio: {overall_corrective:.2%}.")

    return MaintenanceResult(
        equipment_summaries=summaries,
        total_events=total_events,
        most_maintained_equipment=most_maintained,
        most_common_type=most_common_type,
        overall_corrective_ratio=round(overall_corrective, 4),
        summary="".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 2. compute_mtbf_mttr
# ---------------------------------------------------------------------------


_FAILURE_TYPES = {"breakdown", "corrective"}


def compute_mtbf_mttr(
    rows: list[dict],
    equipment_column: str,
    date_column: str,
    type_column: str,
    duration_column: str,
) -> list[ReliabilityMetric]:
    """Compute MTBF (Mean Time Between Failures) and MTTR for each equipment.

    Only considers "breakdown" or "corrective" type events as failures.
    Needs at least 2 failure events to compute MTBF.

    Args:
        rows: Data rows as dicts.
        equipment_column: Column identifying the equipment.
        date_column: Column with event date.
        type_column: Column with maintenance type.
        duration_column: Column with repair duration (hours).

    Returns:
        List of ReliabilityMetric, sorted by equipment name.
    """
    if not rows:
        return []

    # Collect failure events per equipment: (date, duration)
    equip_failures: dict[str, list[tuple[datetime, float]]] = {}

    for row in rows:
        equip = row.get(equipment_column)
        date_val = _parse_date(row.get(date_column))
        mtype = row.get(type_column)
        dur = _safe_float(row.get(duration_column))
        if equip is None or date_val is None or mtype is None:
            continue

        type_key = str(mtype).lower().strip()
        if type_key not in _FAILURE_TYPES:
            continue

        equip_key = str(equip)
        equip_failures.setdefault(equip_key, []).append((date_val, dur if dur is not None else 0.0))

    if not equip_failures:
        return []

    results: list[ReliabilityMetric] = []

    for equip_key in sorted(equip_failures.keys()):
        failures = equip_failures[equip_key]
        # Sort by date
        failures.sort(key=lambda x: x[0])
        failure_count = len(failures)

        first_failure = failures[0][0]
        last_failure = failures[-1][0]

        # MTBF: average days between consecutive failures
        mtbf_days: float | None = None
        if failure_count >= 2:
            intervals = [
                (failures[i][0] - failures[i - 1][0]).total_seconds() / 86400.0
                for i in range(1, failure_count)
            ]
            mtbf_days = round(sum(intervals) / len(intervals), 4) if intervals else None

        # MTTR: average duration of failure repairs (hours)
        durations = [f[1] for f in failures]
        mttr_hours: float | None = None
        if durations:
            mttr_hours = round(sum(durations) / len(durations), 4)

        # Availability = MTBF / (MTBF + MTTR) * 100
        # MTTR needs to be in same units as MTBF (days) for this formula
        availability: float | None = None
        if mtbf_days is not None and mttr_hours is not None:
            mttr_days = mttr_hours / 24.0
            denominator = mtbf_days + mttr_days
            if denominator > 0:
                availability = round(mtbf_days / denominator * 100, 4)

        results.append(ReliabilityMetric(
            equipment=equip_key,
            mtbf_days=mtbf_days,
            mttr_hours=mttr_hours,
            availability=availability,
            failure_count=failure_count,
            first_failure=first_failure,
            last_failure=last_failure,
        ))

    return results


# ---------------------------------------------------------------------------
# 3. analyze_spare_parts
# ---------------------------------------------------------------------------


def analyze_spare_parts(
    rows: list[dict],
    part_column: str,
    quantity_column: str,
    cost_column: str | None = None,
    equipment_column: str | None = None,
) -> SparePartsResult | None:
    """Analyze spare parts consumption and perform ABC analysis.

    Args:
        rows: Data rows as dicts.
        part_column: Column identifying the spare part.
        quantity_column: Column with quantity consumed.
        cost_column: Optional column with cost per unit or total cost.
        equipment_column: Optional column identifying the equipment.

    Returns:
        SparePartsResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate per part
    part_qty: dict[str, float] = {}
    part_cost: dict[str, float] = {}
    equip_parts: dict[str, set[str]] = {}

    for row in rows:
        part = row.get(part_column)
        qty = _safe_float(row.get(quantity_column))
        if part is None or qty is None:
            continue

        part_key = str(part)
        part_qty[part_key] = part_qty.get(part_key, 0.0) + qty

        if cost_column is not None:
            cost = _safe_float(row.get(cost_column))
            if cost is not None:
                part_cost[part_key] = part_cost.get(part_key, 0.0) + cost

        if equipment_column is not None:
            equip = row.get(equipment_column)
            if equip is not None:
                equip_parts.setdefault(str(equip), set()).add(part_key)

    if not part_qty:
        return None

    total_unique = len(part_qty)
    total_quantity = round(sum(part_qty.values()), 4)
    total_spend: float | None = None
    if cost_column is not None and part_cost:
        total_spend = round(sum(part_cost.values()), 4)

    # Build part summaries
    all_parts: list[PartSummary] = []
    for part_key in sorted(part_qty.keys()):
        qty = part_qty[part_key]
        cost_val = part_cost.get(part_key) if cost_column is not None else None
        avg_cost = round(cost_val / qty, 4) if cost_val is not None and qty > 0 else None
        all_parts.append(PartSummary(
            part=part_key,
            total_quantity=round(qty, 4),
            total_cost=round(cost_val, 4) if cost_val is not None else None,
            avg_cost_per_unit=avg_cost,
            abc_category=None,
        ))

    # Top 5 by quantity
    top_parts = sorted(all_parts, key=lambda p: -p.total_quantity)[:5]

    # ABC analysis by cost (if cost_column provided)
    abc_parts: list[PartSummary] = []
    if cost_column is not None and part_cost:
        # Sort by cost descending
        sorted_by_cost = sorted(all_parts, key=lambda p: -(p.total_cost or 0.0))
        total_cost_sum = sum(p.total_cost for p in sorted_by_cost if p.total_cost is not None)

        if total_cost_sum > 0:
            cumulative = 0.0
            for p in sorted_by_cost:
                cost_val = p.total_cost or 0.0
                cumulative += cost_val
                cumulative_pct = cumulative / total_cost_sum * 100

                if cumulative_pct <= 80.0 or (not abc_parts):
                    category = "A"
                elif cumulative_pct <= 95.0:
                    category = "B"
                else:
                    category = "C"

                abc_parts.append(PartSummary(
                    part=p.part,
                    total_quantity=p.total_quantity,
                    total_cost=p.total_cost,
                    avg_cost_per_unit=p.avg_cost_per_unit,
                    abc_category=category,
                ))
        else:
            abc_parts = [
                PartSummary(
                    part=p.part,
                    total_quantity=p.total_quantity,
                    total_cost=p.total_cost,
                    avg_cost_per_unit=p.avg_cost_per_unit,
                    abc_category="C",
                )
                for p in sorted_by_cost
            ]
    else:
        abc_parts = all_parts  # No cost data, no ABC classification

    # Equipment -> parts mapping
    parts_by_equipment: dict[str, list[str]] | None = None
    if equipment_column is not None and equip_parts:
        parts_by_equipment = {
            equip: sorted(parts) for equip, parts in sorted(equip_parts.items())
        }

    # Summary
    summary_parts = [
        f"{total_unique} unique parts, {total_quantity:,.1f} total quantity consumed.",
    ]
    if total_spend is not None:
        summary_parts.append(f" Total spend: {total_spend:,.2f}.")
    if top_parts:
        summary_parts.append(f" Top part: {top_parts[0].part} ({top_parts[0].total_quantity:,.1f} units).")
    if abc_parts and cost_column is not None:
        a_count = sum(1 for p in abc_parts if p.abc_category == "A")
        summary_parts.append(f" ABC: {a_count} A-class parts.")

    return SparePartsResult(
        total_unique_parts=total_unique,
        total_quantity=total_quantity,
        total_spend=total_spend,
        top_parts=top_parts,
        parts_by_equipment=parts_by_equipment,
        abc_parts=abc_parts,
        summary="".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 4. generate_maintenance_schedule
# ---------------------------------------------------------------------------


def generate_maintenance_schedule(
    rows: list[dict],
    equipment_column: str,
    date_column: str,
    type_column: str,
    interval_days: float | None = None,
) -> list[ScheduleEntry]:
    """Generate predicted next maintenance dates for each equipment.

    Based on historical maintenance frequency (preventive events), predicts
    the next maintenance date. Uses max date in data + avg_interval as a
    reference point to flag overdue equipment.

    Args:
        rows: Data rows as dicts.
        equipment_column: Column identifying the equipment.
        date_column: Column with maintenance date.
        type_column: Column with maintenance type.
        interval_days: Optional fixed interval in days. If not given,
            computed from average interval between preventive events.

    Returns:
        List of ScheduleEntry sorted by next_maintenance_date (soonest first).
    """
    if not rows:
        return []

    # Collect dates per equipment (only preventive events for interval calc)
    equip_preventive_dates: dict[str, list[datetime]] = {}
    equip_all_dates: dict[str, list[datetime]] = {}
    all_dates: list[datetime] = []

    for row in rows:
        equip = row.get(equipment_column)
        date_val = _parse_date(row.get(date_column))
        mtype = row.get(type_column)
        if equip is None or date_val is None or mtype is None:
            continue

        equip_key = str(equip)
        type_key = str(mtype).lower().strip()

        equip_all_dates.setdefault(equip_key, []).append(date_val)
        all_dates.append(date_val)

        if type_key == "preventive":
            equip_preventive_dates.setdefault(equip_key, []).append(date_val)

    if not equip_all_dates:
        return []

    # Reference date: max date across all data
    reference_date = max(all_dates)

    results: list[ScheduleEntry] = []

    for equip_key in sorted(equip_all_dates.keys()):
        all_equip_dates = sorted(equip_all_dates[equip_key])
        last_date = all_equip_dates[-1]
        event_count = len(all_equip_dates)

        # Compute interval
        avg_interval: float | None = None
        if interval_days is not None:
            avg_interval = interval_days
        else:
            # Use preventive events for interval calculation
            prev_dates = equip_preventive_dates.get(equip_key, [])
            if len(prev_dates) >= 2:
                sorted_prev = sorted(prev_dates)
                intervals = [
                    (sorted_prev[i] - sorted_prev[i - 1]).total_seconds() / 86400.0
                    for i in range(1, len(sorted_prev))
                ]
                avg_interval = sum(intervals) / len(intervals)
            elif len(all_equip_dates) >= 2:
                # Fall back to all events if no preventive events
                intervals = [
                    (all_equip_dates[i] - all_equip_dates[i - 1]).total_seconds() / 86400.0
                    for i in range(1, len(all_equip_dates))
                ]
                avg_interval = sum(intervals) / len(intervals)

        # Predict next date
        next_date: datetime | None = None
        is_overdue = False
        if avg_interval is not None and avg_interval > 0:
            next_date = last_date + timedelta(days=avg_interval)
            # Overdue if next_date < reference_date
            is_overdue = next_date < reference_date

        results.append(ScheduleEntry(
            equipment=equip_key,
            last_maintenance_date=last_date,
            avg_interval_days=round(avg_interval, 4) if avg_interval is not None else None,
            next_maintenance_date=next_date,
            is_overdue=is_overdue,
            events_in_history=event_count,
        ))

    # Sort by next_maintenance_date (None goes last)
    results.sort(key=lambda e: (
        e.next_maintenance_date is None,
        e.next_maintenance_date or datetime.max,
    ))

    return results


# ---------------------------------------------------------------------------
# 5. format_maintenance_report
# ---------------------------------------------------------------------------


def format_maintenance_report(
    history: MaintenanceResult | None = None,
    reliability: list[ReliabilityMetric] | None = None,
    spare_parts: SparePartsResult | None = None,
    schedule: list[ScheduleEntry] | None = None,
) -> str:
    """Format a comprehensive maintenance analysis report as text.

    Args:
        history: Results from analyze_maintenance_history.
        reliability: Results from compute_mtbf_mttr.
        spare_parts: Results from analyze_spare_parts.
        schedule: Results from generate_maintenance_schedule.

    Returns:
        Formatted text report.
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append("MAINTENANCE ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # 1. Maintenance History
    if history is not None:
        lines.append("MAINTENANCE HISTORY")
        lines.append("-" * 40)
        lines.append(f"Total Events:          {history.total_events}")
        lines.append(f"Equipment Count:       {len(history.equipment_summaries)}")
        if history.most_maintained_equipment:
            lines.append(f"Most Maintained:       {history.most_maintained_equipment}")
        if history.most_common_type:
            lines.append(f"Most Common Type:      {history.most_common_type}")
        lines.append(f"Corrective Ratio:      {history.overall_corrective_ratio:.2%}")
        lines.append("")

        if history.equipment_summaries:
            lines.append("Equipment Breakdown:")
            for es in history.equipment_summaries:
                parts = [f"  {es.equipment}: {es.total_events} events"]
                if es.total_downtime is not None:
                    parts.append(f", downtime={es.total_downtime:,.1f}")
                if es.total_cost is not None:
                    parts.append(f", cost={es.total_cost:,.2f}")
                parts.append(f", corrective_ratio={es.corrective_ratio:.2%}")
                lines.append("".join(parts))
            lines.append("")

    # 2. Reliability Metrics
    if reliability:
        lines.append("RELIABILITY METRICS (MTBF/MTTR)")
        lines.append("-" * 40)
        for rm in reliability:
            mtbf_str = f"{rm.mtbf_days:.1f} days" if rm.mtbf_days is not None else "N/A"
            mttr_str = f"{rm.mttr_hours:.1f} hrs" if rm.mttr_hours is not None else "N/A"
            avail_str = f"{rm.availability:.1f}%" if rm.availability is not None else "N/A"
            lines.append(
                f"  {rm.equipment}: MTBF={mtbf_str}, MTTR={mttr_str}, "
                f"Availability={avail_str}, Failures={rm.failure_count}"
            )
        lines.append("")

    # 3. Spare Parts
    if spare_parts is not None:
        lines.append("SPARE PARTS ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"Unique Parts:   {spare_parts.total_unique_parts}")
        lines.append(f"Total Quantity: {spare_parts.total_quantity:,.1f}")
        if spare_parts.total_spend is not None:
            lines.append(f"Total Spend:    {spare_parts.total_spend:,.2f}")
        lines.append("")

        if spare_parts.top_parts:
            lines.append("Top Parts by Consumption:")
            for i, p in enumerate(spare_parts.top_parts, 1):
                cost_str = f", cost={p.total_cost:,.2f}" if p.total_cost is not None else ""
                lines.append(f"  {i}. {p.part}: {p.total_quantity:,.1f} units{cost_str}")
            lines.append("")

        if spare_parts.abc_parts:
            a_count = sum(1 for p in spare_parts.abc_parts if p.abc_category == "A")
            b_count = sum(1 for p in spare_parts.abc_parts if p.abc_category == "B")
            c_count = sum(1 for p in spare_parts.abc_parts if p.abc_category == "C")
            if any(p.abc_category is not None for p in spare_parts.abc_parts):
                lines.append(f"ABC Analysis: A={a_count}, B={b_count}, C={c_count}")
                lines.append("")

    # 4. Schedule
    if schedule:
        lines.append("MAINTENANCE SCHEDULE")
        lines.append("-" * 40)
        overdue_count = sum(1 for s in schedule if s.is_overdue)
        if overdue_count > 0:
            lines.append(f"OVERDUE: {overdue_count} equipment(s) overdue for maintenance!")
            lines.append("")
        for se in schedule:
            next_str = se.next_maintenance_date.strftime("%Y-%m-%d") if se.next_maintenance_date else "N/A"
            interval_str = f"{se.avg_interval_days:.0f}d" if se.avg_interval_days is not None else "N/A"
            overdue_marker = " [OVERDUE]" if se.is_overdue else ""
            lines.append(
                f"  {se.equipment}: next={next_str}, interval={interval_str}{overdue_marker}"
            )
        lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    if history is not None:
        lines.append(history.summary)
    if spare_parts is not None:
        lines.append(spare_parts.summary)
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
