"""Project progress tracking, milestones, and resource allocation.

Pure functions for analyzing project status, tracking milestones,
evaluating resource allocation, and computing project health metrics.
No DB, async, or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


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
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(val), fmt)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProjectInfo:
    """Status info for a single project."""

    project: str
    status: str
    start_date: str | None
    end_date: str | None
    budget: float | None
    duration_days: int | None


@dataclass
class ProjectStatusResult:
    """Complete project status analysis result."""

    projects: list[ProjectInfo]
    status_distribution: dict[str, int]
    completion_rate: float
    avg_duration_days: float | None
    total_budget: float | None
    summary: str


@dataclass
class ProjectMilestones:
    """Milestone summary for a single project."""

    project: str
    total: int
    completed: int
    overdue: int
    on_time_pct: float | None
    avg_delay_days: float | None


@dataclass
class MilestoneResult:
    """Complete milestone analysis result."""

    projects: list[ProjectMilestones]
    total_milestones: int
    overall_on_time_pct: float | None
    health: str  # "on_track", "at_risk", "critical"
    upcoming_count: int
    summary: str


@dataclass
class ResourceInfo:
    """Resource allocation info for a single resource."""

    resource: str
    total_hours: float
    project_count: int
    avg_hours_per_project: float
    utilization_status: str  # "over_allocated", "optimal", "under_utilized"


@dataclass
class RoleHours:
    """Aggregated hours for a role."""

    role: str
    total_hours: float
    resource_count: int


@dataclass
class ResourceResult:
    """Complete resource allocation analysis result."""

    resources: list[ResourceInfo]
    by_role: list[RoleHours] | None
    over_allocated: list[str]
    under_utilized: list[str]
    summary: str


@dataclass
class ProjectHealth:
    """Health metric for a single project."""

    project: str
    planned: float
    actual: float
    variance: float
    variance_pct: float
    performance_index: float
    health: str  # "on_track", "at_risk", "critical"


# ---------------------------------------------------------------------------
# 1. analyze_project_status
# ---------------------------------------------------------------------------


def analyze_project_status(
    rows: list[dict],
    project_column: str,
    status_column: str,
    start_column: str | None = None,
    end_column: str | None = None,
    budget_column: str | None = None,
) -> ProjectStatusResult | None:
    """Analyze project status, dates, and budget.

    Per project: status, start_date, end_date.
    If budget_column: total budget per project.
    Status distribution: count per status.
    Completion rate = completed / total * 100.
    If dates: avg project duration for completed projects.

    Args:
        rows: Data rows as dicts.
        project_column: Column identifying the project.
        status_column: Column with project status.
        start_column: Optional column with project start date.
        end_column: Optional column with project end date.
        budget_column: Optional column with budget amount.

    Returns:
        ProjectStatusResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate by project: take last seen status, earliest start, latest end, sum budget
    project_data: dict[str, dict] = {}
    for row in rows:
        proj = row.get(project_column)
        status = row.get(status_column)
        if proj is None or status is None:
            continue
        key = str(proj)
        status_str = str(status).lower().strip()

        if key not in project_data:
            project_data[key] = {
                "status": status_str,
                "start": None,
                "end": None,
                "budget": 0.0,
                "has_budget": False,
            }
        else:
            # Update status to last seen
            project_data[key]["status"] = status_str

        if start_column is not None:
            dt = _parse_date(row.get(start_column))
            if dt is not None:
                existing = project_data[key]["start"]
                if existing is None or dt < existing:
                    project_data[key]["start"] = dt

        if end_column is not None:
            dt = _parse_date(row.get(end_column))
            if dt is not None:
                existing = project_data[key]["end"]
                if existing is None or dt > existing:
                    project_data[key]["end"] = dt

        if budget_column is not None:
            bgt = _safe_float(row.get(budget_column))
            if bgt is not None:
                project_data[key]["budget"] += bgt
                project_data[key]["has_budget"] = True

    if not project_data:
        return None

    # Build project list
    projects: list[ProjectInfo] = []
    status_dist: dict[str, int] = {}
    completed_count = 0
    completed_durations: list[int] = []
    total_budget: float = 0.0
    has_any_budget = False

    for name, data in sorted(project_data.items()):
        status = data["status"]
        start_dt: datetime | None = data["start"]
        end_dt: datetime | None = data["end"]
        budget_val: float | None = data["budget"] if data["has_budget"] else None

        # Compute duration for completed projects
        duration: int | None = None
        if start_dt is not None and end_dt is not None:
            duration = (end_dt - start_dt).days

        projects.append(ProjectInfo(
            project=name,
            status=status,
            start_date=start_dt.strftime("%Y-%m-%d") if start_dt else None,
            end_date=end_dt.strftime("%Y-%m-%d") if end_dt else None,
            budget=round(budget_val, 4) if budget_val is not None else None,
            duration_days=duration,
        ))

        status_dist[status] = status_dist.get(status, 0) + 1

        if status == "completed":
            completed_count += 1
            if duration is not None:
                completed_durations.append(duration)

        if budget_val is not None:
            total_budget += budget_val
            has_any_budget = True

    total_projects = len(projects)
    completion_rate = (completed_count / total_projects * 100) if total_projects > 0 else 0.0

    avg_duration: float | None = None
    if completed_durations:
        avg_duration = round(sum(completed_durations) / len(completed_durations), 1)

    budget_total: float | None = round(total_budget, 4) if has_any_budget else None

    # Build summary
    parts = [
        f"Project status across {total_projects} projects: ",
        f"Completion rate = {completion_rate:.1f}%. ",
    ]
    status_parts = [f"{k}: {v}" for k, v in sorted(status_dist.items())]
    parts.append(f"Distribution: {', '.join(status_parts)}.")
    if avg_duration is not None:
        parts.append(f" Avg completed duration: {avg_duration:.1f} days.")
    if budget_total is not None:
        parts.append(f" Total budget: {budget_total:,.2f}.")

    return ProjectStatusResult(
        projects=projects,
        status_distribution=status_dist,
        completion_rate=round(completion_rate, 2),
        avg_duration_days=avg_duration,
        total_budget=budget_total,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 2. analyze_milestones
# ---------------------------------------------------------------------------


def analyze_milestones(
    rows: list[dict],
    project_column: str,
    milestone_column: str,
    due_date_column: str,
    completion_date_column: str | None = None,
    status_column: str | None = None,
) -> MilestoneResult | None:
    """Analyze milestone progress and health.

    Per project: total milestones, completed count, overdue count.
    If completion_date_column and due_date_column: avg delay, on_time_pct.
    Overall milestone health based on overdue percentage.
    Upcoming milestones (next 30 days from max date in data).

    Args:
        rows: Data rows as dicts.
        project_column: Column identifying the project.
        milestone_column: Column identifying the milestone.
        due_date_column: Column with milestone due date.
        completion_date_column: Optional column with actual completion date.
        status_column: Optional column with milestone status.

    Returns:
        MilestoneResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect milestone info per project
    project_milestones: dict[str, list[dict]] = {}
    all_dates: list[datetime] = []

    for row in rows:
        proj = row.get(project_column)
        milestone = row.get(milestone_column)
        due_date = _parse_date(row.get(due_date_column))
        if proj is None or milestone is None or due_date is None:
            continue

        key = str(proj)
        if key not in project_milestones:
            project_milestones[key] = []

        comp_date = _parse_date(row.get(completion_date_column)) if completion_date_column else None
        ms_status = str(row.get(status_column, "")).lower().strip() if status_column else None

        project_milestones[key].append({
            "milestone": str(milestone),
            "due_date": due_date,
            "completion_date": comp_date,
            "status": ms_status,
        })

        all_dates.append(due_date)
        if comp_date is not None:
            all_dates.append(comp_date)

    if not project_milestones:
        return None

    # Determine reference date for "upcoming" (max date in data)
    ref_date = max(all_dates)
    upcoming_window = ref_date + timedelta(days=30)

    # Analyze per project
    projects: list[ProjectMilestones] = []
    total_milestones = 0
    total_on_time = 0
    total_with_completion = 0
    total_overdue = 0
    all_delays: list[float] = []
    upcoming_count = 0

    for proj_name in sorted(project_milestones.keys()):
        ms_list = project_milestones[proj_name]
        proj_total = len(ms_list)
        proj_completed = 0
        proj_overdue = 0
        proj_on_time = 0
        proj_delays: list[float] = []

        for ms in ms_list:
            due = ms["due_date"]
            comp = ms["completion_date"]
            status = ms["status"]

            # Determine completed
            is_completed = False
            if comp is not None:
                is_completed = True
            elif status in ("completed", "done", "finished"):
                is_completed = True

            if is_completed:
                proj_completed += 1

            # Determine overdue
            if comp is not None and due is not None:
                delay = (comp - due).days
                if delay > 0:
                    proj_overdue += 1
                else:
                    proj_on_time += 1
                proj_delays.append(delay)
                all_delays.append(delay)
                total_with_completion += 1
            elif not is_completed and due < ref_date:
                # Not completed and past due
                proj_overdue += 1

            # Upcoming check: due within next 30 days from ref_date, not yet completed
            if not is_completed and ref_date <= due <= upcoming_window:
                upcoming_count += 1

        total_milestones += proj_total
        total_overdue += proj_overdue
        total_on_time += proj_on_time

        # Per-project on_time_pct
        proj_on_time_pct: float | None = None
        if proj_delays:
            on_time_count = sum(1 for d in proj_delays if d <= 0)
            proj_on_time_pct = round(on_time_count / len(proj_delays) * 100, 1)

        proj_avg_delay: float | None = None
        if proj_delays:
            proj_avg_delay = round(sum(proj_delays) / len(proj_delays), 1)

        projects.append(ProjectMilestones(
            project=proj_name,
            total=proj_total,
            completed=proj_completed,
            overdue=proj_overdue,
            on_time_pct=proj_on_time_pct,
            avg_delay_days=proj_avg_delay,
        ))

    # Overall on_time_pct
    overall_on_time_pct: float | None = None
    if total_with_completion > 0:
        overall_on_time_pct = round(total_on_time / total_with_completion * 100, 1)

    # Health based on overdue percentage
    if total_milestones > 0:
        overdue_pct = total_overdue / total_milestones * 100
    else:
        overdue_pct = 0.0

    if overdue_pct <= 10:
        health = "on_track"
    elif overdue_pct <= 30:
        health = "at_risk"
    else:
        health = "critical"

    # Summary
    parts = [
        f"Milestone analysis: {total_milestones} milestones across "
        f"{len(projects)} projects. ",
    ]
    if overall_on_time_pct is not None:
        parts.append(f"On-time delivery: {overall_on_time_pct:.1f}%. ")
    parts.append(f"Overdue: {total_overdue}. ")
    parts.append(f"Health: {health}. ")
    if upcoming_count > 0:
        parts.append(f"Upcoming (next 30 days): {upcoming_count}.")

    return MilestoneResult(
        projects=projects,
        total_milestones=total_milestones,
        overall_on_time_pct=overall_on_time_pct,
        health=health,
        upcoming_count=upcoming_count,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 3. analyze_resource_allocation
# ---------------------------------------------------------------------------


def analyze_resource_allocation(
    rows: list[dict],
    resource_column: str,
    project_column: str,
    hours_column: str,
    role_column: str | None = None,
) -> ResourceResult | None:
    """Analyze resource allocation across projects.

    Per resource: total hours, project count, avg hours per project.
    If role_column: hours by role.
    Over-allocated resources (> 160 hours).
    Under-utilized resources (< 80 hours).

    Args:
        rows: Data rows as dicts.
        resource_column: Column identifying the resource/person.
        project_column: Column identifying the project.
        hours_column: Column with hours allocated.
        role_column: Optional column with role/position.

    Returns:
        ResourceResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate per resource
    resource_data: dict[str, dict] = {}
    for row in rows:
        resource = row.get(resource_column)
        project = row.get(project_column)
        hours = _safe_float(row.get(hours_column))
        if resource is None or project is None or hours is None:
            continue

        rkey = str(resource)
        if rkey not in resource_data:
            resource_data[rkey] = {"total_hours": 0.0, "projects": set()}
        resource_data[rkey]["total_hours"] += hours
        resource_data[rkey]["projects"].add(str(project))

    if not resource_data:
        return None

    # Build resource list
    resources: list[ResourceInfo] = []
    over_allocated: list[str] = []
    under_utilized: list[str] = []

    for name in sorted(resource_data.keys()):
        data = resource_data[name]
        total_hours = data["total_hours"]
        project_count = len(data["projects"])
        avg_hours = total_hours / project_count if project_count > 0 else 0.0

        if total_hours > 160:
            util_status = "over_allocated"
            over_allocated.append(name)
        elif total_hours < 80:
            util_status = "under_utilized"
            under_utilized.append(name)
        else:
            util_status = "optimal"

        resources.append(ResourceInfo(
            resource=name,
            total_hours=round(total_hours, 2),
            project_count=project_count,
            avg_hours_per_project=round(avg_hours, 2),
            utilization_status=util_status,
        ))

    # Role breakdown
    by_role: list[RoleHours] | None = None
    if role_column is not None:
        role_agg: dict[str, dict] = {}
        for row in rows:
            role = row.get(role_column)
            resource = row.get(resource_column)
            hours = _safe_float(row.get(hours_column))
            if role is None or resource is None or hours is None:
                continue
            rkey = str(role)
            if rkey not in role_agg:
                role_agg[rkey] = {"total_hours": 0.0, "resources": set()}
            role_agg[rkey]["total_hours"] += hours
            role_agg[rkey]["resources"].add(str(resource))

        if role_agg:
            by_role = []
            for rname in sorted(role_agg.keys()):
                rdata = role_agg[rname]
                by_role.append(RoleHours(
                    role=rname,
                    total_hours=round(rdata["total_hours"], 2),
                    resource_count=len(rdata["resources"]),
                ))

    # Summary
    total_res = len(resources)
    parts = [
        f"Resource allocation across {total_res} resources. ",
        f"Over-allocated (>160h): {len(over_allocated)}. ",
        f"Under-utilized (<80h): {len(under_utilized)}. ",
        f"Optimal: {total_res - len(over_allocated) - len(under_utilized)}.",
    ]

    return ResourceResult(
        resources=resources,
        by_role=by_role,
        over_allocated=over_allocated,
        under_utilized=under_utilized,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 4. compute_project_health
# ---------------------------------------------------------------------------


def compute_project_health(
    rows: list[dict],
    project_column: str,
    planned_column: str,
    actual_column: str,
    metric_type: str = "cost",
) -> list[ProjectHealth]:
    """Compute health metrics by comparing planned vs actual.

    Per project: planned, actual, variance, variance_pct.
    Health: on_track if variance_pct < 5%, at_risk if < 15%, critical otherwise.
    Performance index (CPI or SPI) = planned / actual.

    Args:
        rows: Data rows as dicts.
        project_column: Column identifying the project.
        planned_column: Column with planned value (cost or time).
        actual_column: Column with actual value (cost or time).
        metric_type: "cost" or "time" (affects index naming only).

    Returns:
        List of ProjectHealth sorted by variance_pct descending.
    """
    if not rows:
        return []

    # Aggregate per project
    project_agg: dict[str, dict[str, float]] = {}
    for row in rows:
        proj = row.get(project_column)
        planned = _safe_float(row.get(planned_column))
        actual = _safe_float(row.get(actual_column))
        if proj is None or planned is None or actual is None:
            continue

        key = str(proj)
        if key not in project_agg:
            project_agg[key] = {"planned": 0.0, "actual": 0.0}
        project_agg[key]["planned"] += planned
        project_agg[key]["actual"] += actual

    if not project_agg:
        return []

    results: list[ProjectHealth] = []
    for name, vals in project_agg.items():
        planned_val = vals["planned"]
        actual_val = vals["actual"]
        variance = actual_val - planned_val

        if planned_val != 0:
            variance_pct = abs(variance) / abs(planned_val) * 100
        else:
            variance_pct = 0.0 if actual_val == 0 else 100.0

        if actual_val != 0:
            perf_index = planned_val / actual_val
        else:
            perf_index = 1.0 if planned_val == 0 else float("inf")

        if variance_pct < 5:
            health = "on_track"
        elif variance_pct < 15:
            health = "at_risk"
        else:
            health = "critical"

        results.append(ProjectHealth(
            project=name,
            planned=round(planned_val, 4),
            actual=round(actual_val, 4),
            variance=round(variance, 4),
            variance_pct=round(variance_pct, 2),
            performance_index=round(perf_index, 4),
            health=health,
        ))

    # Sort by variance_pct descending
    results.sort(key=lambda x: -x.variance_pct)
    return results


# ---------------------------------------------------------------------------
# 5. format_project_report
# ---------------------------------------------------------------------------


def format_project_report(
    status: ProjectStatusResult | None = None,
    milestones: MilestoneResult | None = None,
    resources: ResourceResult | None = None,
    health: list[ProjectHealth] | None = None,
) -> str:
    """Generate a combined text report from available project analyses.

    Args:
        status: Project status analysis result.
        milestones: Milestone analysis result.
        resources: Resource allocation analysis result.
        health: List of project health entries.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Project Tracking Report")
    sections.append("=" * 40)

    if status is not None:
        lines = ["", "Project Status", "-" * 38]
        for p in status.projects:
            budget_str = f" budget={p.budget:,.2f}" if p.budget is not None else ""
            dur_str = f" duration={p.duration_days}d" if p.duration_days is not None else ""
            lines.append(
                f"  {p.project}: {p.status}"
                f"{budget_str}{dur_str}"
            )
        lines.append(f"  Completion rate: {status.completion_rate:.1f}%")
        if status.avg_duration_days is not None:
            lines.append(f"  Avg duration (completed): {status.avg_duration_days:.1f} days")
        if status.total_budget is not None:
            lines.append(f"  Total budget: {status.total_budget:,.2f}")
        lines.append("  Status distribution:")
        for k, v in sorted(status.status_distribution.items()):
            lines.append(f"    {k}: {v}")
        sections.append("\n".join(lines))

    if milestones is not None:
        lines = ["", "Milestones", "-" * 38]
        for pm in milestones.projects:
            on_time_str = f" on_time={pm.on_time_pct:.1f}%" if pm.on_time_pct is not None else ""
            delay_str = f" avg_delay={pm.avg_delay_days:.1f}d" if pm.avg_delay_days is not None else ""
            lines.append(
                f"  {pm.project}: {pm.completed}/{pm.total} completed, "
                f"{pm.overdue} overdue{on_time_str}{delay_str}"
            )
        lines.append(f"  Total milestones: {milestones.total_milestones}")
        if milestones.overall_on_time_pct is not None:
            lines.append(f"  Overall on-time: {milestones.overall_on_time_pct:.1f}%")
        lines.append(f"  Health: {milestones.health}")
        if milestones.upcoming_count > 0:
            lines.append(f"  Upcoming (30 days): {milestones.upcoming_count}")
        sections.append("\n".join(lines))

    if resources is not None:
        lines = ["", "Resource Allocation", "-" * 38]
        for ri in resources.resources:
            lines.append(
                f"  {ri.resource}: {ri.total_hours:.1f}h across {ri.project_count} projects "
                f"({ri.avg_hours_per_project:.1f}h/project) [{ri.utilization_status}]"
            )
        if resources.over_allocated:
            lines.append(f"  Over-allocated: {', '.join(resources.over_allocated)}")
        if resources.under_utilized:
            lines.append(f"  Under-utilized: {', '.join(resources.under_utilized)}")
        if resources.by_role:
            lines.append("  By role:")
            for rh in resources.by_role:
                lines.append(f"    {rh.role}: {rh.total_hours:.1f}h ({rh.resource_count} resources)")
        sections.append("\n".join(lines))

    if health:
        lines = ["", "Project Health", "-" * 38]
        for ph in health:
            lines.append(
                f"  {ph.project}: planned={ph.planned:,.2f} actual={ph.actual:,.2f} "
                f"variance={ph.variance:+,.2f} ({ph.variance_pct:.1f}%) "
                f"PI={ph.performance_index:.2f} [{ph.health}]"
            )
        sections.append("\n".join(lines))

    if status is None and milestones is None and resources is None and not health:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
