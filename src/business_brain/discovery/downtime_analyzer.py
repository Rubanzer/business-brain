"""Equipment/machine downtime pattern analysis for manufacturing plants.

Pure functions for analyzing downtime events, detecting recurring failures,
Pareto analysis on downtime reasons, and shift-based comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReasonSummary:
    """Aggregated summary for a single downtime reason."""

    reason: str
    total_duration: float
    event_count: int
    pct_of_total: float


@dataclass
class MachineDowntime:
    """Downtime summary for a single machine."""

    machine: str
    total_downtime: float
    event_count: int
    mttr: float  # Mean Time To Repair (average duration)
    availability_pct: float | None  # requires time_column to compute
    top_reason: str | None


@dataclass
class DowntimeResult:
    """Complete downtime analysis result."""

    machines: list[MachineDowntime]
    total_downtime: float
    total_events: int
    top_reasons: list[ReasonSummary]
    worst_machine: str | None
    best_machine: str | None
    summary: str


@dataclass
class RecurringFailure:
    """A machine+reason combination that recurs frequently."""

    machine: str
    reason: str
    occurrence_count: int
    total_duration: float
    avg_interval_between: float | None  # average time between occurrences
    trend: str  # "increasing", "decreasing", or "stable"


@dataclass
class DowntimeParetoItem:
    """A single item in the downtime Pareto analysis."""

    reason: str
    total_duration: float
    pct_of_total: float
    cumulative_pct: float
    category: str  # "A" (top 80%), "B" (next 15%), "C" (remaining 5%)


@dataclass
class ShiftDowntime:
    """Downtime summary for a single shift."""

    shift: str
    total_downtime: float
    event_count: int
    avg_duration: float


@dataclass
class ShiftResult:
    """Complete shift-based downtime analysis result."""

    shifts: list[ShiftDowntime]
    worst_shift: str | None
    best_shift: str | None
    variance_ratio: float  # worst / best ratio (how uneven shifts are)
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_downtime
# ---------------------------------------------------------------------------


def analyze_downtime(
    rows: list[dict],
    machine_column: str,
    duration_column: str,
    reason_column: str | None = None,
    time_column: str | None = None,
) -> DowntimeResult:
    """Analyze equipment downtime patterns across machines.

    Args:
        rows: Data rows as dicts.
        machine_column: Column identifying the machine/equipment.
        duration_column: Numeric column with downtime duration.
        reason_column: Optional column with failure/downtime reason.
        time_column: Optional column with event timestamp (for MTBF).

    Returns:
        DowntimeResult with per-machine stats, top reasons, and summary.
    """
    if not rows:
        return DowntimeResult(
            machines=[],
            total_downtime=0.0,
            total_events=0,
            top_reasons=[],
            worst_machine=None,
            best_machine=None,
            summary="No downtime data available.",
        )

    # Aggregate per machine: durations list, reasons list, timestamps list
    machine_durations: dict[str, list[float]] = {}
    machine_reasons: dict[str, list[str]] = {}
    machine_times: dict[str, list[datetime]] = {}

    for row in rows:
        machine = row.get(machine_column)
        duration = row.get(duration_column)
        if machine is None or duration is None:
            continue
        try:
            dur_val = float(duration)
        except (TypeError, ValueError):
            continue

        machine_key = str(machine)
        machine_durations.setdefault(machine_key, []).append(dur_val)

        if reason_column is not None:
            reason = row.get(reason_column)
            if reason is not None:
                machine_reasons.setdefault(machine_key, []).append(str(reason))

        if time_column is not None:
            ts = _parse_timestamp(row.get(time_column))
            if ts is not None:
                machine_times.setdefault(machine_key, []).append(ts)

    if not machine_durations:
        return DowntimeResult(
            machines=[],
            total_downtime=0.0,
            total_events=0,
            top_reasons=[],
            worst_machine=None,
            best_machine=None,
            summary="No valid downtime records found.",
        )

    # Build per-machine results
    machines: list[MachineDowntime] = []
    total_downtime = 0.0
    total_events = 0

    # Compute overall time span for availability if time_column given
    all_timestamps: list[datetime] = []
    for ts_list in machine_times.values():
        all_timestamps.extend(ts_list)

    overall_span: float | None = None
    if all_timestamps and len(all_timestamps) >= 2:
        sorted_all = sorted(all_timestamps)
        overall_span = (sorted_all[-1] - sorted_all[0]).total_seconds()

    for machine_key, durations in sorted(machine_durations.items()):
        total_dur = sum(durations)
        count = len(durations)
        mttr = total_dur / count if count > 0 else 0.0

        # Availability: (span - downtime) / span * 100
        availability: float | None = None
        if time_column is not None and machine_key in machine_times:
            ts_list = sorted(machine_times[machine_key])
            if len(ts_list) >= 2:
                span = (ts_list[-1] - ts_list[0]).total_seconds()
                if span > 0:
                    availability = max(0.0, (span - total_dur) / span * 100)

        # Top reason for this machine
        top_reason: str | None = None
        if machine_key in machine_reasons and machine_reasons[machine_key]:
            reason_counts: dict[str, int] = {}
            for r in machine_reasons[machine_key]:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            top_reason = max(reason_counts, key=reason_counts.get)  # type: ignore[arg-type]

        machines.append(
            MachineDowntime(
                machine=machine_key,
                total_downtime=round(total_dur, 4),
                event_count=count,
                mttr=round(mttr, 4),
                availability_pct=round(availability, 2) if availability is not None else None,
                top_reason=top_reason,
            )
        )
        total_downtime += total_dur
        total_events += count

    # Top reasons across all machines
    top_reasons: list[ReasonSummary] = []
    if reason_column is not None:
        reason_agg: dict[str, dict] = {}
        for row in rows:
            reason = row.get(reason_column)
            duration = row.get(duration_column)
            if reason is None or duration is None:
                continue
            try:
                dur_val = float(duration)
            except (TypeError, ValueError):
                continue
            r_key = str(reason)
            if r_key not in reason_agg:
                reason_agg[r_key] = {"total": 0.0, "count": 0}
            reason_agg[r_key]["total"] += dur_val
            reason_agg[r_key]["count"] += 1

        for reason_name, stats in sorted(
            reason_agg.items(), key=lambda x: -x[1]["total"]
        ):
            pct = (stats["total"] / total_downtime * 100) if total_downtime > 0 else 0.0
            top_reasons.append(
                ReasonSummary(
                    reason=reason_name,
                    total_duration=round(stats["total"], 4),
                    event_count=stats["count"],
                    pct_of_total=round(pct, 2),
                )
            )

    # Worst and best machine by total downtime
    worst_machine = max(machines, key=lambda m: m.total_downtime).machine if machines else None
    best_machine = min(machines, key=lambda m: m.total_downtime).machine if machines else None

    # Summary
    summary_parts = [
        f"{total_events} downtime events across {len(machines)} machine(s), "
        f"total downtime: {total_downtime:,.2f}.",
    ]
    if worst_machine:
        worst_obj = next(m for m in machines if m.machine == worst_machine)
        summary_parts.append(
            f" Worst: {worst_machine} ({worst_obj.total_downtime:,.2f}, "
            f"{worst_obj.event_count} events)."
        )
    if top_reasons:
        summary_parts.append(f" Top reason: {top_reasons[0].reason} ({top_reasons[0].pct_of_total}%).")

    return DowntimeResult(
        machines=machines,
        total_downtime=round(total_downtime, 4),
        total_events=total_events,
        top_reasons=top_reasons,
        worst_machine=worst_machine,
        best_machine=best_machine,
        summary="".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 2. detect_recurring_failures
# ---------------------------------------------------------------------------


def detect_recurring_failures(
    rows: list[dict],
    machine_column: str,
    reason_column: str,
    time_column: str,
    min_occurrences: int = 3,
) -> list[RecurringFailure]:
    """Find machine+reason combinations that recur frequently.

    Args:
        rows: Data rows as dicts.
        machine_column: Column identifying the machine.
        reason_column: Column with failure reason.
        time_column: Column with event timestamp.
        min_occurrences: Minimum occurrences to be considered recurring.

    Returns:
        List of RecurringFailure, sorted by occurrence_count descending.
    """
    if not rows:
        return []

    # Group by (machine, reason) -> list of (timestamp, duration)
    combos: dict[tuple[str, str], list[datetime]] = {}
    combo_durations: dict[tuple[str, str], list[float]] = {}

    for row in rows:
        machine = row.get(machine_column)
        reason = row.get(reason_column)
        ts_raw = row.get(time_column)
        if machine is None or reason is None or ts_raw is None:
            continue

        ts = _parse_timestamp(ts_raw)
        if ts is None:
            continue

        key = (str(machine), str(reason))
        combos.setdefault(key, []).append(ts)

        # Also collect durations if available (look for common duration column names)
        # We sum up 1 per event for duration tracking as a fallback
        combo_durations.setdefault(key, [])

    # Also try to get durations from rows
    # Re-scan to collect durations keyed by combo
    _collect_durations(rows, machine_column, reason_column, combo_durations)

    results: list[RecurringFailure] = []
    for (machine, reason), timestamps in combos.items():
        if len(timestamps) < min_occurrences:
            continue

        sorted_ts = sorted(timestamps)
        count = len(sorted_ts)

        # Calculate average interval between occurrences
        avg_interval: float | None = None
        if count >= 2:
            intervals = [
                (sorted_ts[i] - sorted_ts[i - 1]).total_seconds()
                for i in range(1, count)
            ]
            avg_interval = sum(intervals) / len(intervals) if intervals else None

        # Determine trend: compare first-half intervals to second-half intervals
        trend = _compute_trend(sorted_ts)

        # Total duration for this combo
        durations = combo_durations.get((machine, reason), [])
        total_dur = sum(durations) if durations else float(count)

        results.append(
            RecurringFailure(
                machine=machine,
                reason=reason,
                occurrence_count=count,
                total_duration=round(total_dur, 4),
                avg_interval_between=round(avg_interval, 4) if avg_interval is not None else None,
                trend=trend,
            )
        )

    results.sort(key=lambda r: -r.occurrence_count)
    return results


def _collect_durations(
    rows: list[dict],
    machine_column: str,
    reason_column: str,
    combo_durations: dict[tuple[str, str], list[float]],
) -> None:
    """Scan rows for duration values and associate them with machine+reason combos."""
    # Try to find a duration column by common names
    duration_candidates = ["duration", "downtime", "repair_time", "minutes", "hours"]
    if not rows:
        return

    sample_keys = set(rows[0].keys())
    dur_col = None
    for candidate in duration_candidates:
        if candidate in sample_keys:
            dur_col = candidate
            break
        # Also check case-insensitive
        for k in sample_keys:
            if k.lower() == candidate:
                dur_col = k
                break
        if dur_col:
            break

    if dur_col is None:
        return

    for row in rows:
        machine = row.get(machine_column)
        reason = row.get(reason_column)
        dur = row.get(dur_col)
        if machine is None or reason is None or dur is None:
            continue
        try:
            dur_val = float(dur)
        except (TypeError, ValueError):
            continue
        key = (str(machine), str(reason))
        if key in combo_durations:
            combo_durations[key].append(dur_val)


def _compute_trend(timestamps: list[datetime]) -> str:
    """Determine if failures are increasing, decreasing, or stable in frequency.

    Compares the average interval of the first half to the second half.
    Shorter intervals in the second half = increasing (getting worse).
    """
    if len(timestamps) < 3:
        return "stable"

    sorted_ts = sorted(timestamps)
    intervals = [
        (sorted_ts[i] - sorted_ts[i - 1]).total_seconds()
        for i in range(1, len(sorted_ts))
    ]

    if len(intervals) < 2:
        return "stable"

    mid = len(intervals) // 2
    first_half_avg = sum(intervals[:mid]) / mid if mid > 0 else 0
    second_half_avg = sum(intervals[mid:]) / len(intervals[mid:]) if len(intervals[mid:]) > 0 else 0

    if first_half_avg == 0 and second_half_avg == 0:
        return "stable"

    if first_half_avg == 0:
        return "stable"

    ratio = second_half_avg / first_half_avg

    if ratio < 0.75:
        return "increasing"  # intervals shrinking = failures more frequent
    elif ratio > 1.33:
        return "decreasing"  # intervals growing = failures less frequent
    else:
        return "stable"


# ---------------------------------------------------------------------------
# 3. downtime_pareto
# ---------------------------------------------------------------------------


def downtime_pareto(
    rows: list[dict],
    reason_column: str,
    duration_column: str,
) -> list[DowntimeParetoItem]:
    """Perform 80/20 Pareto analysis on downtime reasons.

    Args:
        rows: Data rows as dicts.
        reason_column: Column with failure/downtime reason.
        duration_column: Numeric column with downtime duration.

    Returns:
        List of DowntimeParetoItem, sorted by total_duration descending.
    """
    if not rows:
        return []

    # Aggregate by reason
    aggregated: dict[str, float] = {}
    for row in rows:
        reason = row.get(reason_column)
        duration = row.get(duration_column)
        if reason is None or duration is None:
            continue
        try:
            dur_val = float(duration)
        except (TypeError, ValueError):
            continue
        r_key = str(reason)
        aggregated[r_key] = aggregated.get(r_key, 0.0) + dur_val

    if not aggregated:
        return []

    total = sum(aggregated.values())
    if total == 0:
        return []

    # Sort descending
    sorted_items = sorted(aggregated.items(), key=lambda x: -x[1])

    # Build Pareto items with ABC classification
    items: list[DowntimeParetoItem] = []
    cumulative = 0.0
    for reason, duration in sorted_items:
        pct = duration / total * 100
        cumulative += pct

        # ABC classification
        if cumulative <= 80.0 or len(items) == 0:
            category = "A"
        elif cumulative <= 95.0:
            category = "B"
        else:
            category = "C"

        items.append(
            DowntimeParetoItem(
                reason=reason,
                total_duration=round(duration, 4),
                pct_of_total=round(pct, 2),
                cumulative_pct=round(cumulative, 2),
                category=category,
            )
        )

    return items


# ---------------------------------------------------------------------------
# 4. shift_analysis
# ---------------------------------------------------------------------------


def shift_analysis(
    rows: list[dict],
    shift_column: str,
    duration_column: str,
    machine_column: str | None = None,
) -> ShiftResult:
    """Compare downtime across shifts.

    Args:
        rows: Data rows as dicts.
        shift_column: Column identifying the shift (e.g., "Day", "Night").
        duration_column: Numeric column with downtime duration.
        machine_column: Optional column for machine-level grouping.

    Returns:
        ShiftResult with per-shift stats and comparison.
    """
    if not rows:
        return ShiftResult(
            shifts=[],
            worst_shift=None,
            best_shift=None,
            variance_ratio=0.0,
            summary="No shift data available.",
        )

    # Aggregate by shift
    shift_data: dict[str, list[float]] = {}
    for row in rows:
        shift = row.get(shift_column)
        duration = row.get(duration_column)
        if shift is None or duration is None:
            continue
        try:
            dur_val = float(duration)
        except (TypeError, ValueError):
            continue
        shift_key = str(shift)
        shift_data.setdefault(shift_key, []).append(dur_val)

    if not shift_data:
        return ShiftResult(
            shifts=[],
            worst_shift=None,
            best_shift=None,
            variance_ratio=0.0,
            summary="No valid shift records found.",
        )

    # Build per-shift results
    shifts: list[ShiftDowntime] = []
    for shift_key in sorted(shift_data):
        durations = shift_data[shift_key]
        total_dur = sum(durations)
        count = len(durations)
        avg_dur = total_dur / count if count > 0 else 0.0
        shifts.append(
            ShiftDowntime(
                shift=shift_key,
                total_downtime=round(total_dur, 4),
                event_count=count,
                avg_duration=round(avg_dur, 4),
            )
        )

    # Worst and best shift
    worst_shift = max(shifts, key=lambda s: s.total_downtime).shift if shifts else None
    best_shift = min(shifts, key=lambda s: s.total_downtime).shift if shifts else None

    # Variance ratio: worst / best
    worst_val = max(s.total_downtime for s in shifts) if shifts else 0
    best_val = min(s.total_downtime for s in shifts) if shifts else 0
    variance_ratio = worst_val / best_val if best_val > 0 else 0.0

    # Summary
    summary_parts = [
        f"Downtime across {len(shifts)} shift(s).",
    ]
    if worst_shift and best_shift:
        worst_obj = next(s for s in shifts if s.shift == worst_shift)
        best_obj = next(s for s in shifts if s.shift == best_shift)
        summary_parts.append(
            f" Worst: {worst_shift} ({worst_obj.total_downtime:,.2f} total, "
            f"{worst_obj.event_count} events)."
        )
        summary_parts.append(
            f" Best: {best_shift} ({best_obj.total_downtime:,.2f} total, "
            f"{best_obj.event_count} events)."
        )
        if variance_ratio > 1:
            summary_parts.append(f" Variance ratio: {variance_ratio:.2f}x.")

    return ShiftResult(
        shifts=shifts,
        worst_shift=worst_shift,
        best_shift=best_shift,
        variance_ratio=round(variance_ratio, 4),
        summary="".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 5. format_downtime_report
# ---------------------------------------------------------------------------


def format_downtime_report(
    result: DowntimeResult,
    pareto: list[DowntimeParetoItem] | None = None,
    recurring: list[RecurringFailure] | None = None,
) -> str:
    """Format a comprehensive downtime analysis report as text.

    Args:
        result: Main DowntimeResult from analyze_downtime.
        pareto: Optional Pareto analysis items.
        recurring: Optional recurring failure list.

    Returns:
        Formatted text report.
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append("DOWNTIME ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total Events:    {result.total_events}")
    lines.append(f"Total Downtime:  {result.total_downtime:,.2f}")
    lines.append(f"Machines:        {len(result.machines)}")
    if result.worst_machine:
        lines.append(f"Worst Machine:   {result.worst_machine}")
    if result.best_machine:
        lines.append(f"Best Machine:    {result.best_machine}")
    lines.append("")

    # Per-machine table
    if result.machines:
        lines.append("MACHINE BREAKDOWN")
        lines.append("-" * 40)

        # Column headers
        m_w = max(len("Machine"), *(len(m.machine) for m in result.machines))
        lines.append(
            f"{'Machine':<{m_w}}  {'Downtime':>12}  {'Events':>6}  "
            f"{'MTTR':>10}  {'Avail%':>8}  {'Top Reason'}"
        )
        lines.append("-" * (m_w + 60))

        for m in sorted(result.machines, key=lambda x: -x.total_downtime):
            avail = f"{m.availability_pct:.1f}%" if m.availability_pct is not None else "N/A"
            top_r = m.top_reason or "N/A"
            lines.append(
                f"{m.machine:<{m_w}}  {m.total_downtime:>12,.2f}  {m.event_count:>6}  "
                f"{m.mttr:>10,.2f}  {avail:>8}  {top_r}"
            )
        lines.append("")

    # Top reasons
    if result.top_reasons:
        lines.append("TOP DOWNTIME REASONS")
        lines.append("-" * 40)
        for i, r in enumerate(result.top_reasons[:10], 1):
            lines.append(
                f"  {i:>2}. {r.reason:<30}  {r.total_duration:>10,.2f}  "
                f"({r.pct_of_total:.1f}%)  [{r.event_count} events]"
            )
        lines.append("")

    # Pareto section
    if pareto:
        lines.append("PARETO ANALYSIS (80/20)")
        lines.append("-" * 40)
        for item in pareto:
            marker = f"[{item.category}]"
            lines.append(
                f"  {marker} {item.reason:<30}  {item.total_duration:>10,.2f}  "
                f"({item.pct_of_total:.1f}%)  cum: {item.cumulative_pct:.1f}%"
            )
        a_count = sum(1 for p in pareto if p.category == "A")
        lines.append(f"  Category A items: {a_count} of {len(pareto)}")
        lines.append("")

    # Recurring failures
    if recurring:
        lines.append("RECURRING FAILURES")
        lines.append("-" * 40)
        for rf in recurring:
            interval_str = (
                f"{rf.avg_interval_between:,.0f}s avg interval"
                if rf.avg_interval_between is not None
                else "N/A"
            )
            lines.append(
                f"  {rf.machine} / {rf.reason}: "
                f"{rf.occurrence_count} occurrences, "
                f"trend={rf.trend}, {interval_str}"
            )
        lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(result.summary)
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(value) -> datetime | None:
    """Try to parse a value into a datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    # Try common formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt)
        except (ValueError, TypeError):
            continue
    return None
