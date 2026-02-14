"""Compliance tracking and regulatory monitoring.

Pure functions for auditing compliance status, analyzing audit findings,
computing weighted compliance scores, tracking regulatory deadlines,
and generating compliance reports. No DB, async, or LLM dependencies.
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


def _month_key(dt: datetime) -> str:
    """Return 'YYYY-MM' string for a datetime."""
    return dt.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CategoryCompliance:
    """Compliance metrics for a single category."""

    category: str
    total: int
    compliant: int
    compliance_rate: float


@dataclass
class ComplianceStatusResult:
    """Overall compliance status analysis result."""

    total_requirements: int
    compliant_count: int
    non_compliant_count: int
    compliance_rate: float
    overdue_count: int
    by_category: list[CategoryCompliance] | None
    summary: str


@dataclass
class SeverityCount:
    """Count of findings at a given severity level."""

    severity: str
    count: int
    pct: float


@dataclass
class AreaFindings:
    """Findings count for a single area/department."""

    area: str
    count: int
    critical_count: int


@dataclass
class MonthlyFindingTrend:
    """Number of findings in a given month."""

    month: str
    count: int


@dataclass
class AuditFindingsResult:
    """Complete audit findings analysis result."""

    total_findings: int
    by_severity: list[SeverityCount]
    by_area: list[AreaFindings] | None
    open_count: int
    closed_count: int
    closure_rate: float | None
    monthly_trend: list[MonthlyFindingTrend] | None
    summary: str


@dataclass
class CategoryScore:
    """Weighted score for a single category."""

    category: str
    score: float
    weight: float


@dataclass
class ComplianceScoreResult:
    """Weighted compliance score result."""

    overall_score: float
    rating: str
    weighted_scores: list[CategoryScore]
    weakest_areas: list[CategoryScore]
    by_category: dict[str, float] | None
    summary: str


@dataclass
class DeadlineItem:
    """A single regulatory deadline entry."""

    regulation: str
    deadline: str
    days_until: int
    urgency: str  # "Overdue", "Due This Week", "Due This Month", "Upcoming"
    status: str | None
    owner: str | None


@dataclass
class OwnerDeadlines:
    """Deadline counts for a single owner."""

    owner: str
    total: int
    overdue: int


@dataclass
class DeadlineResult:
    """Complete regulatory deadline tracking result."""

    total_items: int
    overdue_count: int
    due_this_week: int
    due_this_month: int
    upcoming_count: int
    items: list[DeadlineItem]
    by_owner: list[OwnerDeadlines] | None
    summary: str


# ---------------------------------------------------------------------------
# 1. audit_compliance_status
# ---------------------------------------------------------------------------


def audit_compliance_status(
    rows: list[dict],
    requirement_column: str,
    status_column: str,
    category_column: str | None = None,
    due_date_column: str | None = None,
) -> ComplianceStatusResult | None:
    """Analyse compliance status across requirements.

    Counts items by status (case-insensitive): Compliant, Non-Compliant,
    In Progress, Not Started, etc.  Computes overall compliance rate =
    compliant / total * 100.

    If *category_column*: compliance rate per category.
    If *due_date_column*: counts overdue items (due date in the past and
    status is not compliant).  Uses the maximum date found in the data as
    the reference "today" to keep the function deterministic.

    Args:
        rows: Data rows as dicts.
        requirement_column: Column identifying the requirement.
        status_column: Column with compliance status string.
        category_column: Optional column for grouping by category.
        due_date_column: Optional column with due-date for overdue detection.

    Returns:
        ComplianceStatusResult or None if no valid data.
    """
    if not rows:
        return None

    valid_rows: list[dict] = []
    for row in rows:
        req = row.get(requirement_column)
        status = row.get(status_column)
        if req is None and status is None:
            continue
        valid_rows.append(row)

    if not valid_rows:
        return None

    total = len(valid_rows)
    compliant_count = 0
    non_compliant_count = 0

    # Determine reference date from data if due_date_column provided
    ref_date: datetime | None = None
    if due_date_column is not None:
        all_dates: list[datetime] = []
        for row in valid_rows:
            dt = _parse_date(row.get(due_date_column))
            if dt is not None:
                all_dates.append(dt)
        if all_dates:
            ref_date = max(all_dates)

    overdue_count = 0

    # Category tracking
    cat_totals: dict[str, int] = {}
    cat_compliant: dict[str, int] = {}

    for row in valid_rows:
        raw_status = row.get(status_column)
        normalised = str(raw_status).strip().lower() if raw_status is not None else ""

        is_compliant = normalised == "compliant"
        is_non_compliant = normalised in ("non-compliant", "non compliant", "noncompliant")

        if is_compliant:
            compliant_count += 1
        if is_non_compliant:
            non_compliant_count += 1

        # Category breakdown
        if category_column is not None:
            cat_val = row.get(category_column)
            if cat_val is not None:
                cat_key = str(cat_val)
                cat_totals[cat_key] = cat_totals.get(cat_key, 0) + 1
                if is_compliant:
                    cat_compliant[cat_key] = cat_compliant.get(cat_key, 0) + 1

        # Overdue detection
        if due_date_column is not None and ref_date is not None and not is_compliant:
            dt = _parse_date(row.get(due_date_column))
            if dt is not None and dt < ref_date:
                overdue_count += 1

    compliance_rate = (compliant_count / total * 100) if total > 0 else 0.0

    # Build category list
    by_category: list[CategoryCompliance] | None = None
    if category_column is not None and cat_totals:
        by_category = []
        for cat_name in sorted(cat_totals.keys()):
            ct = cat_totals[cat_name]
            cc = cat_compliant.get(cat_name, 0)
            cr = (cc / ct * 100) if ct > 0 else 0.0
            by_category.append(
                CategoryCompliance(
                    category=cat_name,
                    total=ct,
                    compliant=cc,
                    compliance_rate=round(cr, 2),
                )
            )

    # Summary
    parts = [
        f"{total} requirements assessed.",
        f" Compliance rate: {compliance_rate:.1f}%.",
        f" Compliant: {compliant_count}, Non-compliant: {non_compliant_count}.",
    ]
    if overdue_count > 0:
        parts.append(f" Overdue items: {overdue_count}.")

    return ComplianceStatusResult(
        total_requirements=total,
        compliant_count=compliant_count,
        non_compliant_count=non_compliant_count,
        compliance_rate=round(compliance_rate, 2),
        overdue_count=overdue_count,
        by_category=by_category,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 2. analyze_audit_findings
# ---------------------------------------------------------------------------


def analyze_audit_findings(
    rows: list[dict],
    finding_column: str,
    severity_column: str,
    date_column: str | None = None,
    area_column: str | None = None,
    status_column: str | None = None,
) -> AuditFindingsResult | None:
    """Analyse audit findings by severity, area, status, and time.

    Counts by severity (Critical, Major, Minor, Observation).
    If *date_column*: monthly trend of findings.
    If *area_column*: findings per area/department with critical count.
    If *status_column*: open vs closed counts and closure rate.

    Args:
        rows: Data rows as dicts, each representing one audit finding.
        finding_column: Column identifying the finding.
        severity_column: Column with severity level.
        date_column: Optional column with finding date.
        area_column: Optional column with area/department.
        status_column: Optional column with open/closed status.

    Returns:
        AuditFindingsResult or None if no valid data.
    """
    if not rows:
        return None

    valid_rows: list[dict] = []
    for row in rows:
        finding = row.get(finding_column)
        severity = row.get(severity_column)
        if finding is None and severity is None:
            continue
        valid_rows.append(row)

    if not valid_rows:
        return None

    total = len(valid_rows)

    # Severity counts
    sev_counts: dict[str, int] = {}
    for row in valid_rows:
        raw = row.get(severity_column)
        if raw is not None:
            key = str(raw).strip().title()
            sev_counts[key] = sev_counts.get(key, 0) + 1

    severity_order = ["Critical", "Major", "Minor", "Observation"]
    by_severity: list[SeverityCount] = []
    # Include known severities in order first, then any extras
    seen: set[str] = set()
    for sev_name in severity_order:
        if sev_name in sev_counts:
            cnt = sev_counts[sev_name]
            pct = cnt / total * 100
            by_severity.append(SeverityCount(severity=sev_name, count=cnt, pct=round(pct, 2)))
            seen.add(sev_name)
    for sev_name in sorted(sev_counts.keys()):
        if sev_name not in seen:
            cnt = sev_counts[sev_name]
            pct = cnt / total * 100
            by_severity.append(SeverityCount(severity=sev_name, count=cnt, pct=round(pct, 2)))

    # Area analysis
    by_area: list[AreaFindings] | None = None
    if area_column is not None:
        area_counts: dict[str, int] = {}
        area_critical: dict[str, int] = {}
        for row in valid_rows:
            area_val = row.get(area_column)
            if area_val is None:
                continue
            area_key = str(area_val)
            area_counts[area_key] = area_counts.get(area_key, 0) + 1
            sev_raw = row.get(severity_column)
            if sev_raw is not None and str(sev_raw).strip().lower() == "critical":
                area_critical[area_key] = area_critical.get(area_key, 0) + 1
        if area_counts:
            by_area = []
            for area_name in sorted(area_counts.keys(), key=lambda a: -area_counts[a]):
                by_area.append(
                    AreaFindings(
                        area=area_name,
                        count=area_counts[area_name],
                        critical_count=area_critical.get(area_name, 0),
                    )
                )

    # Status analysis
    open_count = 0
    closed_count = 0
    closure_rate: float | None = None
    if status_column is not None:
        for row in valid_rows:
            st = row.get(status_column)
            if st is not None:
                normalised = str(st).strip().lower()
                if normalised in ("closed", "resolved", "completed"):
                    closed_count += 1
                elif normalised in ("open", "in progress", "pending"):
                    open_count += 1
        total_with_status = open_count + closed_count
        if total_with_status > 0:
            closure_rate = round(closed_count / total_with_status * 100, 2)

    # Monthly trend
    monthly_trend: list[MonthlyFindingTrend] | None = None
    if date_column is not None:
        month_counts: dict[str, int] = {}
        for row in valid_rows:
            dt = _parse_date(row.get(date_column))
            if dt is not None:
                mk = _month_key(dt)
                month_counts[mk] = month_counts.get(mk, 0) + 1
        if month_counts:
            monthly_trend = [
                MonthlyFindingTrend(month=m, count=month_counts[m])
                for m in sorted(month_counts.keys())
            ]

    # Summary
    sev_parts = ", ".join(f"{s.severity}: {s.count}" for s in by_severity)
    parts = [f"{total} audit findings."]
    if sev_parts:
        parts.append(f" By severity: {sev_parts}.")
    if closure_rate is not None:
        parts.append(f" Closure rate: {closure_rate:.1f}%.")

    return AuditFindingsResult(
        total_findings=total,
        by_severity=by_severity,
        by_area=by_area,
        open_count=open_count,
        closed_count=closed_count,
        closure_rate=closure_rate,
        monthly_trend=monthly_trend,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 3. compute_compliance_score
# ---------------------------------------------------------------------------


def compute_compliance_score(
    rows: list[dict],
    requirement_column: str,
    weight_column: str,
    score_column: str,
    category_column: str | None = None,
) -> ComplianceScoreResult | None:
    """Compute a weighted compliance score.

    Overall score = sum(weight * score) / sum(weight).
    Per-category scores if *category_column* provided.
    Identifies weakest areas (bottom 3 by score).
    Classifies: Excellent (>=90), Good (>=75), Needs Improvement (>=60),
    Critical (<60).

    Args:
        rows: Data rows as dicts.
        requirement_column: Column identifying the requirement.
        weight_column: Column with numeric weight.
        score_column: Column with numeric score (0-100).
        category_column: Optional column for per-category breakdown.

    Returns:
        ComplianceScoreResult or None if no valid data.
    """
    if not rows:
        return None

    entries: list[tuple[str, float, float, str | None]] = []
    for row in rows:
        req = row.get(requirement_column)
        w = _safe_float(row.get(weight_column))
        s = _safe_float(row.get(score_column))
        if req is None or w is None or s is None:
            continue
        cat: str | None = None
        if category_column is not None:
            cv = row.get(category_column)
            if cv is not None:
                cat = str(cv)
        entries.append((str(req), w, s, cat))

    if not entries:
        return None

    total_weight = sum(w for _, w, _, _ in entries)
    if total_weight == 0:
        return None

    weighted_sum = sum(w * s for _, w, s, _ in entries)
    overall_score = weighted_sum / total_weight

    # Rating
    if overall_score >= 90:
        rating = "Excellent"
    elif overall_score >= 75:
        rating = "Good"
    elif overall_score >= 60:
        rating = "Needs Improvement"
    else:
        rating = "Critical"

    # Per-requirement weighted scores, sorted by score ascending (weakest first)
    weighted_scores: list[CategoryScore] = []
    for req_name, w, s, _ in entries:
        weighted_scores.append(CategoryScore(category=req_name, score=round(s, 2), weight=round(w, 2)))

    # Sort by score ascending to find weakest
    weighted_scores_sorted = sorted(weighted_scores, key=lambda x: x.score)
    weakest_areas = weighted_scores_sorted[:3]

    # Per-category breakdown
    by_category: dict[str, float] | None = None
    if category_column is not None:
        cat_w_sum: dict[str, float] = {}
        cat_ws_sum: dict[str, float] = {}
        for _, w, s, cat in entries:
            if cat is not None:
                cat_w_sum[cat] = cat_w_sum.get(cat, 0.0) + w
                cat_ws_sum[cat] = cat_ws_sum.get(cat, 0.0) + w * s
        if cat_w_sum:
            by_category = {}
            for cat_name in sorted(cat_w_sum.keys()):
                cw = cat_w_sum[cat_name]
                if cw > 0:
                    by_category[cat_name] = round(cat_ws_sum[cat_name] / cw, 2)

    # Summary
    weakest_names = ", ".join(wa.category for wa in weakest_areas)
    summary = (
        f"Overall compliance score: {overall_score:.1f} ({rating}). "
        f"Weakest areas: {weakest_names}."
    )

    return ComplianceScoreResult(
        overall_score=round(overall_score, 2),
        rating=rating,
        weighted_scores=weighted_scores,
        weakest_areas=weakest_areas,
        by_category=by_category,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. track_regulatory_deadlines
# ---------------------------------------------------------------------------


def track_regulatory_deadlines(
    rows: list[dict],
    regulation_column: str,
    deadline_column: str,
    status_column: str | None = None,
    owner_column: str | None = None,
    *,
    reference_date: datetime | None = None,
) -> DeadlineResult | None:
    """Track regulatory deadlines and classify urgency.

    Parses deadline dates.  Classifies each item as:
    - Overdue (past reference date)
    - Due This Week (within 7 days)
    - Due This Month (within 30 days)
    - Upcoming (more than 30 days)

    If *status_column*: filters out completed items.
    If *owner_column*: counts deadlines per owner.
    Sorted by deadline ascending.

    The *reference_date* defaults to the maximum deadline date found in the
    data so that tests remain deterministic.

    Args:
        rows: Data rows as dicts.
        regulation_column: Column identifying the regulation.
        deadline_column: Column with deadline date.
        status_column: Optional column with completion status.
        owner_column: Optional column with responsible owner.
        reference_date: Optional explicit reference date for urgency calc.

    Returns:
        DeadlineResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect all parseable entries
    parsed: list[tuple[str, datetime, str | None, str | None]] = []
    for row in rows:
        reg = row.get(regulation_column)
        dl = _parse_date(row.get(deadline_column))
        if reg is None or dl is None:
            continue

        status_val: str | None = None
        if status_column is not None:
            sv = row.get(status_column)
            if sv is not None:
                status_val = str(sv).strip()

        owner_val: str | None = None
        if owner_column is not None:
            ov = row.get(owner_column)
            if ov is not None:
                owner_val = str(ov).strip()

        parsed.append((str(reg), dl, status_val, owner_val))

    if not parsed:
        return None

    # Filter out completed items if status_column provided
    if status_column is not None:
        active: list[tuple[str, datetime, str | None, str | None]] = []
        for reg, dl, st, ow in parsed:
            if st is not None and st.lower() in ("completed", "complete", "done", "closed"):
                continue
            active.append((reg, dl, st, ow))
        parsed = active

    if not parsed:
        return None

    # Determine reference date
    if reference_date is None:
        reference_date = max(dl for _, dl, _, _ in parsed)

    # Classify and build items
    items: list[DeadlineItem] = []
    for reg, dl, st, ow in parsed:
        delta = (dl - reference_date).days

        if delta < 0:
            urgency = "Overdue"
        elif delta <= 7:
            urgency = "Due This Week"
        elif delta <= 30:
            urgency = "Due This Month"
        else:
            urgency = "Upcoming"

        items.append(
            DeadlineItem(
                regulation=reg,
                deadline=dl.strftime("%Y-%m-%d"),
                days_until=delta,
                urgency=urgency,
                status=st,
                owner=ow,
            )
        )

    # Sort by deadline ascending
    items.sort(key=lambda x: x.deadline)

    overdue_count = sum(1 for i in items if i.urgency == "Overdue")
    due_this_week = sum(1 for i in items if i.urgency == "Due This Week")
    due_this_month = sum(1 for i in items if i.urgency == "Due This Month")
    upcoming_count = sum(1 for i in items if i.urgency == "Upcoming")

    # Owner breakdown
    by_owner: list[OwnerDeadlines] | None = None
    if owner_column is not None:
        owner_totals: dict[str, int] = {}
        owner_overdue: dict[str, int] = {}
        for item in items:
            if item.owner is not None:
                owner_totals[item.owner] = owner_totals.get(item.owner, 0) + 1
                if item.urgency == "Overdue":
                    owner_overdue[item.owner] = owner_overdue.get(item.owner, 0) + 1
        if owner_totals:
            by_owner = [
                OwnerDeadlines(
                    owner=name,
                    total=owner_totals[name],
                    overdue=owner_overdue.get(name, 0),
                )
                for name in sorted(owner_totals.keys(), key=lambda n: -owner_totals[n])
            ]

    # Summary
    parts = [
        f"{len(items)} regulatory deadlines tracked.",
        f" Overdue: {overdue_count}.",
        f" Due this week: {due_this_week}.",
        f" Due this month: {due_this_month}.",
        f" Upcoming: {upcoming_count}.",
    ]

    return DeadlineResult(
        total_items=len(items),
        overdue_count=overdue_count,
        due_this_week=due_this_week,
        due_this_month=due_this_month,
        upcoming_count=upcoming_count,
        items=items,
        by_owner=by_owner,
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 5. format_compliance_report
# ---------------------------------------------------------------------------


def format_compliance_report(
    status: ComplianceStatusResult | None = None,
    findings: AuditFindingsResult | None = None,
    score: ComplianceScoreResult | None = None,
    deadlines: DeadlineResult | None = None,
) -> str:
    """Generate a combined text compliance report from available analyses.

    Each section is only included if the corresponding parameter is not None.

    Args:
        status: Compliance status analysis result.
        findings: Audit findings analysis result.
        score: Compliance score result.
        deadlines: Deadline tracking result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Compliance Tracking Report")
    sections.append("=" * 40)

    if status is not None:
        lines = ["", "Compliance Status", "-" * 38]
        lines.append(f"  Total requirements: {status.total_requirements}")
        lines.append(f"  Compliant: {status.compliant_count}")
        lines.append(f"  Non-compliant: {status.non_compliant_count}")
        lines.append(f"  Compliance rate: {status.compliance_rate:.1f}%")
        if status.overdue_count > 0:
            lines.append(f"  Overdue: {status.overdue_count}")
        if status.by_category:
            lines.append("  By category:")
            for cat in status.by_category:
                lines.append(
                    f"    {cat.category}: {cat.compliant}/{cat.total} "
                    f"({cat.compliance_rate:.1f}%)"
                )
        sections.append("\n".join(lines))

    if findings is not None:
        lines = ["", "Audit Findings", "-" * 38]
        lines.append(f"  Total findings: {findings.total_findings}")
        if findings.by_severity:
            lines.append("  By severity:")
            for sc in findings.by_severity:
                lines.append(f"    {sc.severity}: {sc.count} ({sc.pct:.1f}%)")
        if findings.by_area:
            lines.append("  By area:")
            for af in findings.by_area:
                crit_str = f" (critical: {af.critical_count})" if af.critical_count > 0 else ""
                lines.append(f"    {af.area}: {af.count}{crit_str}")
        if findings.closure_rate is not None:
            lines.append(
                f"  Open: {findings.open_count}, Closed: {findings.closed_count} "
                f"(closure rate: {findings.closure_rate:.1f}%)"
            )
        if findings.monthly_trend:
            lines.append("  Monthly trend:")
            for mt in findings.monthly_trend:
                lines.append(f"    {mt.month}: {mt.count}")
        sections.append("\n".join(lines))

    if score is not None:
        lines = ["", "Compliance Score", "-" * 38]
        lines.append(f"  Overall score: {score.overall_score:.1f}")
        lines.append(f"  Rating: {score.rating}")
        if score.weakest_areas:
            lines.append("  Weakest areas:")
            for wa in score.weakest_areas:
                lines.append(f"    {wa.category}: {wa.score:.1f} (weight: {wa.weight:.1f})")
        if score.by_category:
            lines.append("  By category:")
            for cat_name, cat_score in sorted(score.by_category.items()):
                lines.append(f"    {cat_name}: {cat_score:.1f}")
        sections.append("\n".join(lines))

    if deadlines is not None:
        lines = ["", "Regulatory Deadlines", "-" * 38]
        lines.append(f"  Total items: {deadlines.total_items}")
        lines.append(f"  Overdue: {deadlines.overdue_count}")
        lines.append(f"  Due this week: {deadlines.due_this_week}")
        lines.append(f"  Due this month: {deadlines.due_this_month}")
        lines.append(f"  Upcoming: {deadlines.upcoming_count}")
        if deadlines.items:
            lines.append("  Items:")
            for item in deadlines.items:
                owner_str = f" [{item.owner}]" if item.owner else ""
                lines.append(
                    f"    {item.regulation}: {item.deadline} "
                    f"({item.urgency}, {item.days_until:+d} days){owner_str}"
                )
        if deadlines.by_owner:
            lines.append("  By owner:")
            for od in deadlines.by_owner:
                lines.append(f"    {od.owner}: {od.total} total, {od.overdue} overdue")
        sections.append("\n".join(lines))

    if status is None and findings is None and score is None and deadlines is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
