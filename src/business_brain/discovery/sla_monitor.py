"""SLA monitoring and service-level analysis.

Pure functions for analysing SLA compliance, response times, resolution
metrics, SLA trends over time, and generating combined SLA reports.
No DB, async, or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
import re
import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Convert a value to float safely, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_date(val) -> datetime | None:
    """Attempt to parse a date from string or datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(val), fmt)
        except (ValueError, TypeError):
            continue
    return None


def _month_key(dt: datetime) -> str:
    """Return 'YYYY-MM' string for a datetime."""
    return dt.strftime("%Y-%m")


def _median(values: list[float]) -> float:
    """Compute the median of a sorted list of floats."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile (0-100) from a list of floats."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (k - f) * (s[c] - s[f])


def _std_dev(values: list[float], mean: float) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CategorySLA:
    """SLA metrics for a single category."""

    category: str
    total: int
    met: int
    breached: int
    compliance_rate: float
    avg_actual: float
    avg_target: float


@dataclass
class SLAComplianceResult:
    """Overall SLA compliance analysis result."""

    total_tickets: int
    met_count: int
    breached_count: int
    compliance_rate: float
    by_category: list[CategorySLA]
    worst_category: str | None
    avg_performance_ratio: float
    summary: str


@dataclass
class PriorityResponse:
    """Response time metrics for a single priority level."""

    priority: str
    count: int
    avg_time: float
    median_time: float
    p95_time: float


@dataclass
class AgentResponse:
    """Response time metrics for a single agent."""

    agent: str
    count: int
    avg_time: float
    compliance_rate: float


@dataclass
class ResponseTimeResult:
    """Complete response time analysis result."""

    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    min_time: float
    max_time: float
    by_priority: list[PriorityResponse]
    by_agent: list[AgentResponse]
    outlier_count: int
    summary: str


@dataclass
class PriorityResolution:
    """Resolution metrics for a single priority level."""

    priority: str
    total: int
    resolved: int
    avg_resolution_hours: float
    resolution_rate: float


@dataclass
class ResolutionResult:
    """Complete resolution metrics result."""

    total_tickets: int
    resolved_count: int
    open_count: int
    resolution_rate: float
    avg_resolution_hours: float
    median_resolution_hours: float
    by_priority: list[PriorityResolution]
    backlog_age_avg_hours: float | None
    summary: str


@dataclass
class PeriodSLA:
    """SLA compliance for a single time period."""

    period: str
    total: int
    met: int
    compliance_rate: float


@dataclass
class SLATrendResult:
    """SLA trend analysis result."""

    periods: list[PeriodSLA]
    trend_direction: str
    overall_compliance: float
    best_period: str
    worst_period: str
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_sla_compliance
# ---------------------------------------------------------------------------


def analyze_sla_compliance(
    rows: list[dict],
    ticket_column: str,
    sla_target_column: str,
    actual_column: str,
    category_column: str | None = None,
) -> SLAComplianceResult | None:
    """Analyse SLA compliance by comparing actual vs target for each ticket.

    Met = actual <= target, Breached = actual > target.
    Computes compliance rate overall and by category.
    Performance ratio = actual / target (lower is better).

    Args:
        rows: Data rows as dicts.
        ticket_column: Column identifying the ticket/record.
        sla_target_column: Column with the SLA target value (numeric).
        actual_column: Column with the actual value (numeric).
        category_column: Optional column for grouping by category.

    Returns:
        SLAComplianceResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect valid entries
    entries: list[tuple[str, float, float, str | None]] = []
    for row in rows:
        ticket = row.get(ticket_column)
        target = _safe_float(row.get(sla_target_column))
        actual = _safe_float(row.get(actual_column))
        if ticket is None or target is None or actual is None:
            continue
        cat: str | None = None
        if category_column is not None:
            cv = row.get(category_column)
            if cv is not None:
                cat = str(cv)
        entries.append((str(ticket), target, actual, cat))

    if not entries:
        return None

    total = len(entries)
    met_count = 0
    breached_count = 0
    ratios: list[float] = []

    # Category tracking
    cat_data: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "met": 0, "breached": 0, "actuals": [], "targets": []}
    )

    for ticket, target, actual, cat in entries:
        is_met = actual <= target
        if is_met:
            met_count += 1
        else:
            breached_count += 1

        if target > 0:
            ratios.append(actual / target)

        if cat is not None:
            cat_data[cat]["total"] += 1
            if is_met:
                cat_data[cat]["met"] += 1
            else:
                cat_data[cat]["breached"] += 1
            cat_data[cat]["actuals"].append(actual)
            cat_data[cat]["targets"].append(target)

    compliance_rate = (met_count / total * 100) if total > 0 else 0.0
    avg_performance_ratio = (sum(ratios) / len(ratios)) if ratios else 0.0

    # Build category list
    by_category: list[CategorySLA] = []
    if category_column is not None and cat_data:
        for cat_name in sorted(cat_data.keys()):
            cd = cat_data[cat_name]
            ct = cd["total"]
            cm = cd["met"]
            cb = cd["breached"]
            cr = (cm / ct * 100) if ct > 0 else 0.0
            avg_a = sum(cd["actuals"]) / len(cd["actuals"]) if cd["actuals"] else 0.0
            avg_t = sum(cd["targets"]) / len(cd["targets"]) if cd["targets"] else 0.0
            by_category.append(
                CategorySLA(
                    category=cat_name,
                    total=ct,
                    met=cm,
                    breached=cb,
                    compliance_rate=round(cr, 2),
                    avg_actual=round(avg_a, 2),
                    avg_target=round(avg_t, 2),
                )
            )

    # Determine worst category
    worst_category: str | None = None
    if by_category:
        worst = min(by_category, key=lambda c: c.compliance_rate)
        worst_category = worst.category

    # Summary
    parts = [
        f"{total} tickets analysed.",
        f" SLA compliance rate: {compliance_rate:.1f}%.",
        f" Met: {met_count}, Breached: {breached_count}.",
    ]
    if worst_category is not None:
        parts.append(f" Worst category: {worst_category}.")

    return SLAComplianceResult(
        total_tickets=total,
        met_count=met_count,
        breached_count=breached_count,
        compliance_rate=round(compliance_rate, 2),
        by_category=by_category,
        worst_category=worst_category,
        avg_performance_ratio=round(avg_performance_ratio, 4),
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 2. analyze_response_times
# ---------------------------------------------------------------------------


def analyze_response_times(
    rows: list[dict],
    ticket_column: str,
    response_time_column: str,
    priority_column: str | None = None,
    agent_column: str | None = None,
) -> ResponseTimeResult | None:
    """Analyse response time distribution across tickets.

    Computes average, median, p95, min, max response times.
    Optionally breaks down by priority level and by agent.
    Identifies outliers (> 2 standard deviations from the mean).

    For agent compliance rate, a response is considered compliant
    if it is within 2 standard deviations of the overall mean.

    Args:
        rows: Data rows as dicts.
        ticket_column: Column identifying the ticket.
        response_time_column: Column with numeric response time.
        priority_column: Optional column for priority level.
        agent_column: Optional column for agent name.

    Returns:
        ResponseTimeResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect valid entries
    entries: list[tuple[str, float, str | None, str | None]] = []
    for row in rows:
        ticket = row.get(ticket_column)
        rt = _safe_float(row.get(response_time_column))
        if ticket is None or rt is None:
            continue
        pri: str | None = None
        if priority_column is not None:
            pv = row.get(priority_column)
            if pv is not None:
                pri = str(pv).strip()
        agent: str | None = None
        if agent_column is not None:
            av = row.get(agent_column)
            if av is not None:
                agent = str(av).strip()
        entries.append((str(ticket), rt, pri, agent))

    if not entries:
        return None

    all_times = [rt for _, rt, _, _ in entries]
    avg_time = sum(all_times) / len(all_times)
    med_time = _median(all_times)
    p95_time = _percentile(all_times, 95)
    min_time = min(all_times)
    max_time = max(all_times)

    # Outlier detection
    std = _std_dev(all_times, avg_time)
    threshold = avg_time + 2 * std
    outlier_count = sum(1 for t in all_times if t > threshold)

    # By priority
    by_priority: list[PriorityResponse] = []
    if priority_column is not None:
        pri_times: dict[str, list[float]] = defaultdict(list)
        for _, rt, pri, _ in entries:
            if pri is not None:
                pri_times[pri].append(rt)
        for pri_name in sorted(pri_times.keys()):
            times = pri_times[pri_name]
            by_priority.append(
                PriorityResponse(
                    priority=pri_name,
                    count=len(times),
                    avg_time=round(sum(times) / len(times), 2),
                    median_time=round(_median(times), 2),
                    p95_time=round(_percentile(times, 95), 2),
                )
            )

    # By agent
    by_agent: list[AgentResponse] = []
    if agent_column is not None:
        agent_times: dict[str, list[float]] = defaultdict(list)
        for _, rt, _, ag in entries:
            if ag is not None:
                agent_times[ag].append(rt)
        for agent_name in sorted(agent_times.keys()):
            times = agent_times[agent_name]
            agent_avg = sum(times) / len(times)
            # Compliance: how many of agent's responses are within threshold
            compliant = sum(1 for t in times if t <= threshold)
            agent_cr = (compliant / len(times) * 100) if times else 0.0
            by_agent.append(
                AgentResponse(
                    agent=agent_name,
                    count=len(times),
                    avg_time=round(agent_avg, 2),
                    compliance_rate=round(agent_cr, 2),
                )
            )

    # Summary
    parts = [
        f"{len(entries)} tickets analysed.",
        f" Avg response time: {avg_time:.2f}.",
        f" Median: {med_time:.2f}, P95: {p95_time:.2f}.",
        f" Outliers: {outlier_count}.",
    ]

    return ResponseTimeResult(
        avg_response_time=round(avg_time, 2),
        median_response_time=round(med_time, 2),
        p95_response_time=round(p95_time, 2),
        min_time=round(min_time, 2),
        max_time=round(max_time, 2),
        by_priority=by_priority,
        by_agent=by_agent,
        outlier_count=outlier_count,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 3. compute_resolution_metrics
# ---------------------------------------------------------------------------


def compute_resolution_metrics(
    rows: list[dict],
    ticket_column: str,
    created_column: str,
    resolved_column: str,
    status_column: str | None = None,
    priority_column: str | None = None,
) -> ResolutionResult | None:
    """Compute ticket resolution metrics.

    Time to resolve = resolved_date - created_date in hours.
    Resolution rate = resolved / total.
    Backlog: tickets without a resolved date (open).
    Backlog age: average time since creation for open tickets, using the
    maximum resolved date in the data as the reference "now" for determinism.

    Args:
        rows: Data rows as dicts.
        ticket_column: Column identifying the ticket.
        created_column: Column with creation date.
        resolved_column: Column with resolution date (None if open).
        status_column: Optional column with ticket status.
        priority_column: Optional column for priority level.

    Returns:
        ResolutionResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect valid entries: must have ticket + created date
    entries: list[tuple[str, datetime, datetime | None, str | None, str | None]] = []
    for row in rows:
        ticket = row.get(ticket_column)
        created = _parse_date(row.get(created_column))
        if ticket is None or created is None:
            continue
        resolved = _parse_date(row.get(resolved_column))

        status: str | None = None
        if status_column is not None:
            sv = row.get(status_column)
            if sv is not None:
                status = str(sv).strip().lower()

        priority: str | None = None
        if priority_column is not None:
            pv = row.get(priority_column)
            if pv is not None:
                priority = str(pv).strip()

        entries.append((str(ticket), created, resolved, status, priority))

    if not entries:
        return None

    total = len(entries)

    # Determine which tickets are resolved
    resolved_hours: list[float] = []
    open_created: list[datetime] = []

    for ticket, created, resolved, status, priority in entries:
        if resolved is not None:
            delta = (resolved - created).total_seconds() / 3600.0
            if delta >= 0:
                resolved_hours.append(delta)
            else:
                # Negative resolution time treated as open
                open_created.append(created)
        else:
            open_created.append(created)

    resolved_count = len(resolved_hours)
    open_count = total - resolved_count
    resolution_rate = (resolved_count / total * 100) if total > 0 else 0.0

    avg_resolution_hours = (
        sum(resolved_hours) / len(resolved_hours) if resolved_hours else 0.0
    )
    median_resolution_hours = _median(resolved_hours) if resolved_hours else 0.0

    # Backlog age: use max resolved date as reference for determinism
    backlog_age_avg_hours: float | None = None
    if open_created:
        all_resolved_dates = [
            resolved
            for _, _, resolved, _, _ in entries
            if resolved is not None
        ]
        if all_resolved_dates:
            ref_date = max(all_resolved_dates)
        else:
            # No resolved dates at all; use max created date as reference
            ref_date = max(c for _, c, _, _, _ in entries)
        ages = []
        for c in open_created:
            age_h = (ref_date - c).total_seconds() / 3600.0
            if age_h >= 0:
                ages.append(age_h)
        if ages:
            backlog_age_avg_hours = round(sum(ages) / len(ages), 2)

    # By priority
    by_priority: list[PriorityResolution] = []
    if priority_column is not None:
        pri_data: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "resolved": 0, "hours": []}
        )
        for ticket, created, resolved, status, priority in entries:
            if priority is not None:
                pri_data[priority]["total"] += 1
                if resolved is not None:
                    delta = (resolved - created).total_seconds() / 3600.0
                    if delta >= 0:
                        pri_data[priority]["resolved"] += 1
                        pri_data[priority]["hours"].append(delta)

        for pri_name in sorted(pri_data.keys()):
            pd = pri_data[pri_name]
            pt = pd["total"]
            pr = pd["resolved"]
            avg_h = sum(pd["hours"]) / len(pd["hours"]) if pd["hours"] else 0.0
            rr = (pr / pt * 100) if pt > 0 else 0.0
            by_priority.append(
                PriorityResolution(
                    priority=pri_name,
                    total=pt,
                    resolved=pr,
                    avg_resolution_hours=round(avg_h, 2),
                    resolution_rate=round(rr, 2),
                )
            )

    # Summary
    parts = [
        f"{total} tickets analysed.",
        f" Resolved: {resolved_count}, Open: {open_count}.",
        f" Resolution rate: {resolution_rate:.1f}%.",
    ]
    if resolved_hours:
        parts.append(f" Avg resolution time: {avg_resolution_hours:.1f} hours.")
    if backlog_age_avg_hours is not None:
        parts.append(f" Avg backlog age: {backlog_age_avg_hours:.1f} hours.")

    return ResolutionResult(
        total_tickets=total,
        resolved_count=resolved_count,
        open_count=open_count,
        resolution_rate=round(resolution_rate, 2),
        avg_resolution_hours=round(avg_resolution_hours, 2),
        median_resolution_hours=round(median_resolution_hours, 2),
        by_priority=by_priority,
        backlog_age_avg_hours=backlog_age_avg_hours,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 4. analyze_sla_trends
# ---------------------------------------------------------------------------


def analyze_sla_trends(
    rows: list[dict],
    ticket_column: str,
    sla_met_column: str,
    date_column: str,
    category_column: str | None = None,
) -> SLATrendResult | None:
    """Track SLA compliance rate over time periods (months).

    The *sla_met_column* is interpreted as a boolean-like value:
    truthy values (True, 1, "yes", "true", "met", "1") count as met.

    Detects improving/deteriorating/stable trends based on the compliance
    rates in the first and last periods.

    Args:
        rows: Data rows as dicts.
        ticket_column: Column identifying the ticket.
        sla_met_column: Column indicating whether SLA was met (boolean-like).
        date_column: Column with the date for time grouping.
        category_column: Optional column for filtering (unused in trend but
            included for API consistency).

    Returns:
        SLATrendResult or None if no valid data.
    """
    if not rows:
        return None

    _truthy = {"true", "yes", "met", "1", "pass", "passed"}

    entries: list[tuple[str, bool, datetime]] = []
    for row in rows:
        ticket = row.get(ticket_column)
        met_raw = row.get(sla_met_column)
        dt = _parse_date(row.get(date_column))
        if ticket is None or met_raw is None or dt is None:
            continue

        # Interpret met value
        if isinstance(met_raw, bool):
            is_met = met_raw
        elif isinstance(met_raw, (int, float)):
            is_met = met_raw >= 1
        elif isinstance(met_raw, str):
            is_met = met_raw.strip().lower() in _truthy
        else:
            continue

        entries.append((str(ticket), is_met, dt))

    if not entries:
        return None

    # Group by month
    month_data: dict[str, dict] = defaultdict(lambda: {"total": 0, "met": 0})
    total_met = 0
    for ticket, is_met, dt in entries:
        mk = _month_key(dt)
        month_data[mk]["total"] += 1
        if is_met:
            month_data[mk]["met"] += 1
            total_met += 1

    total_entries = len(entries)
    overall_compliance = (total_met / total_entries * 100) if total_entries > 0 else 0.0

    # Build periods sorted chronologically
    periods: list[PeriodSLA] = []
    for mk in sorted(month_data.keys()):
        md = month_data[mk]
        cr = (md["met"] / md["total"] * 100) if md["total"] > 0 else 0.0
        periods.append(
            PeriodSLA(
                period=mk,
                total=md["total"],
                met=md["met"],
                compliance_rate=round(cr, 2),
            )
        )

    # Determine trend direction
    if len(periods) >= 2:
        first_rate = periods[0].compliance_rate
        last_rate = periods[-1].compliance_rate
        diff = last_rate - first_rate
        if diff > 5.0:
            trend_direction = "Improving"
        elif diff < -5.0:
            trend_direction = "Deteriorating"
        else:
            trend_direction = "Stable"
    else:
        trend_direction = "Stable"

    # Best and worst periods
    best_period_obj = max(periods, key=lambda p: p.compliance_rate)
    worst_period_obj = min(periods, key=lambda p: p.compliance_rate)
    best_period = best_period_obj.period
    worst_period = worst_period_obj.period

    # Summary
    parts = [
        f"{total_entries} tickets across {len(periods)} periods.",
        f" Overall compliance: {overall_compliance:.1f}%.",
        f" Trend: {trend_direction}.",
        f" Best period: {best_period}.",
        f" Worst period: {worst_period}.",
    ]

    return SLATrendResult(
        periods=periods,
        trend_direction=trend_direction,
        overall_compliance=round(overall_compliance, 2),
        best_period=best_period,
        worst_period=worst_period,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 5. format_sla_report
# ---------------------------------------------------------------------------


def format_sla_report(
    compliance: SLAComplianceResult | None = None,
    response_times: ResponseTimeResult | None = None,
    resolution: ResolutionResult | None = None,
    trends: SLATrendResult | None = None,
) -> str:
    """Generate a combined text SLA report from available analyses.

    Each section is only included if the corresponding parameter is not None.

    Args:
        compliance: SLA compliance analysis result.
        response_times: Response time analysis result.
        resolution: Resolution metrics result.
        trends: SLA trend analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("SLA Monitoring Report")
    sections.append("=" * 40)

    if compliance is not None:
        lines = ["", "SLA Compliance", "-" * 38]
        lines.append(f"  Total tickets: {compliance.total_tickets}")
        lines.append(f"  Met: {compliance.met_count}")
        lines.append(f"  Breached: {compliance.breached_count}")
        lines.append(f"  Compliance rate: {compliance.compliance_rate:.1f}%")
        lines.append(f"  Avg performance ratio: {compliance.avg_performance_ratio:.4f}")
        if compliance.worst_category:
            lines.append(f"  Worst category: {compliance.worst_category}")
        if compliance.by_category:
            lines.append("  By category:")
            for cat in compliance.by_category:
                lines.append(
                    f"    {cat.category}: {cat.met}/{cat.total} "
                    f"({cat.compliance_rate:.1f}%)"
                )
        sections.append("\n".join(lines))

    if response_times is not None:
        lines = ["", "Response Times", "-" * 38]
        lines.append(f"  Avg: {response_times.avg_response_time:.2f}")
        lines.append(f"  Median: {response_times.median_response_time:.2f}")
        lines.append(f"  P95: {response_times.p95_response_time:.2f}")
        lines.append(f"  Min: {response_times.min_time:.2f}")
        lines.append(f"  Max: {response_times.max_time:.2f}")
        lines.append(f"  Outliers: {response_times.outlier_count}")
        if response_times.by_priority:
            lines.append("  By priority:")
            for pr in response_times.by_priority:
                lines.append(
                    f"    {pr.priority}: avg={pr.avg_time:.2f}, "
                    f"median={pr.median_time:.2f}, p95={pr.p95_time:.2f}"
                )
        if response_times.by_agent:
            lines.append("  By agent:")
            for ag in response_times.by_agent:
                lines.append(
                    f"    {ag.agent}: avg={ag.avg_time:.2f}, "
                    f"compliance={ag.compliance_rate:.1f}%"
                )
        sections.append("\n".join(lines))

    if resolution is not None:
        lines = ["", "Resolution Metrics", "-" * 38]
        lines.append(f"  Total tickets: {resolution.total_tickets}")
        lines.append(f"  Resolved: {resolution.resolved_count}")
        lines.append(f"  Open: {resolution.open_count}")
        lines.append(f"  Resolution rate: {resolution.resolution_rate:.1f}%")
        lines.append(f"  Avg resolution time: {resolution.avg_resolution_hours:.2f} hours")
        lines.append(f"  Median resolution time: {resolution.median_resolution_hours:.2f} hours")
        if resolution.backlog_age_avg_hours is not None:
            lines.append(f"  Avg backlog age: {resolution.backlog_age_avg_hours:.2f} hours")
        if resolution.by_priority:
            lines.append("  By priority:")
            for pr in resolution.by_priority:
                lines.append(
                    f"    {pr.priority}: {pr.resolved}/{pr.total} resolved, "
                    f"avg={pr.avg_resolution_hours:.2f}h "
                    f"({pr.resolution_rate:.1f}%)"
                )
        sections.append("\n".join(lines))

    if trends is not None:
        lines = ["", "SLA Trends", "-" * 38]
        lines.append(f"  Overall compliance: {trends.overall_compliance:.1f}%")
        lines.append(f"  Trend: {trends.trend_direction}")
        lines.append(f"  Best period: {trends.best_period}")
        lines.append(f"  Worst period: {trends.worst_period}")
        if trends.periods:
            lines.append("  Periods:")
            for p in trends.periods:
                lines.append(
                    f"    {p.period}: {p.met}/{p.total} ({p.compliance_rate:.1f}%)"
                )
        sections.append("\n".join(lines))

    if compliance is None and response_times is None and resolution is None and trends is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
