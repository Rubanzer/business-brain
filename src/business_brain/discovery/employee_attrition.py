"""Employee attrition and HR analytics.

Pure functions for analyzing employee attrition rates, tenure distribution,
retention cohorts, and attrition drivers across departments and time periods.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime


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


_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
]


def _parse_date(val) -> date | None:
    """Parse a value to a date object, returning None on failure."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    s = str(val).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


_LEFT_STATUSES = frozenset([
    "left", "terminated", "resigned", "quit", "departed",
    "inactive", "exit", "separated", "dismissed", "fired",
])

_ACTIVE_STATUSES = frozenset([
    "active", "current", "employed", "working",
])


def _is_left(status: str) -> bool:
    """Check if a status string indicates the employee has left."""
    return status.strip().lower() in _LEFT_STATUSES


def _is_active(status: str) -> bool:
    """Check if a status string indicates the employee is active."""
    return status.strip().lower() in _ACTIVE_STATUSES


def _classify_status(status: str) -> str | None:
    """Classify a status string as 'active', 'left', or None."""
    s = str(status).strip().lower()
    if s in _LEFT_STATUSES:
        return "left"
    if s in _ACTIVE_STATUSES:
        return "active"
    return None


# ---------------------------------------------------------------------------
# 1. Attrition Rate Analysis
# ---------------------------------------------------------------------------


@dataclass
class MonthlyAttrition:
    """Attrition metrics for a single month."""
    month: str
    active: int
    left: int
    rate: float


@dataclass
class DeptAttrition:
    """Attrition metrics for a single department."""
    department: str
    total: int
    left: int
    rate: float


@dataclass
class AttritionResult:
    """Complete attrition rate analysis result."""
    total_employees: int
    active_count: int
    left_count: int
    attrition_rate: float
    monthly_trends: list[MonthlyAttrition]
    by_department: list[DeptAttrition]
    summary: str


def analyze_attrition_rate(
    rows: list[dict],
    employee_column: str,
    status_column: str,
    date_column: str | None = None,
    department_column: str | None = None,
) -> AttritionResult | None:
    """Analyze employee attrition rate from row data.

    Each row represents an employee record. The status column should contain
    values like "Active", "Left", "Terminated", "Resigned" (case-insensitive).

    Args:
        rows: Data rows as dicts.
        employee_column: Column identifying the employee.
        status_column: Column with employment status.
        date_column: Optional column with date (for monthly trends).
        department_column: Optional column for per-department breakdown.

    Returns:
        AttritionResult or None if no valid data.
    """
    if not rows:
        return None

    # Deduplicate by employee, taking latest record if date available
    employee_records: dict[str, dict] = {}
    for row in rows:
        emp = row.get(employee_column)
        status_raw = row.get(status_column)
        if emp is None or status_raw is None:
            continue
        classified = _classify_status(str(status_raw))
        if classified is None:
            continue

        emp_key = str(emp)
        if emp_key not in employee_records:
            employee_records[emp_key] = row
        elif date_column is not None:
            existing_date = _parse_date(employee_records[emp_key].get(date_column))
            new_date = _parse_date(row.get(date_column))
            if new_date is not None and (existing_date is None or new_date > existing_date):
                employee_records[emp_key] = row

    if not employee_records:
        return None

    active_count = 0
    left_count = 0

    for row in employee_records.values():
        classified = _classify_status(str(row.get(status_column, "")))
        if classified == "active":
            active_count += 1
        elif classified == "left":
            left_count += 1

    total = active_count + left_count
    if total == 0:
        return None

    attrition_rate = round(left_count / total * 100, 2)

    # Monthly trends
    monthly_trends: list[MonthlyAttrition] = []
    if date_column is not None:
        month_data: dict[str, dict[str, int]] = {}
        for row in employee_records.values():
            d = _parse_date(row.get(date_column))
            status_raw = row.get(status_column)
            if d is None or status_raw is None:
                continue
            classified = _classify_status(str(status_raw))
            if classified is None:
                continue
            month_key = d.strftime("%Y-%m")
            if month_key not in month_data:
                month_data[month_key] = {"active": 0, "left": 0}
            if classified == "active":
                month_data[month_key]["active"] += 1
            elif classified == "left":
                month_data[month_key]["left"] += 1

        for month_key in sorted(month_data.keys()):
            counts = month_data[month_key]
            m_total = counts["active"] + counts["left"]
            m_rate = round(counts["left"] / m_total * 100, 2) if m_total > 0 else 0.0
            monthly_trends.append(MonthlyAttrition(
                month=month_key,
                active=counts["active"],
                left=counts["left"],
                rate=m_rate,
            ))

    # Department breakdown
    by_department: list[DeptAttrition] = []
    if department_column is not None:
        dept_data: dict[str, dict[str, int]] = {}
        for row in employee_records.values():
            dept = row.get(department_column)
            status_raw = row.get(status_column)
            if dept is None or status_raw is None:
                continue
            classified = _classify_status(str(status_raw))
            if classified is None:
                continue
            dept_key = str(dept)
            if dept_key not in dept_data:
                dept_data[dept_key] = {"total": 0, "left": 0}
            dept_data[dept_key]["total"] += 1
            if classified == "left":
                dept_data[dept_key]["left"] += 1

        for dept_key in sorted(dept_data.keys()):
            counts = dept_data[dept_key]
            d_rate = round(counts["left"] / counts["total"] * 100, 2) if counts["total"] > 0 else 0.0
            by_department.append(DeptAttrition(
                department=dept_key,
                total=counts["total"],
                left=counts["left"],
                rate=d_rate,
            ))

    summary = (
        f"Attrition analysis: {total} employees, "
        f"{active_count} active, {left_count} left. "
        f"Overall attrition rate: {attrition_rate:.1f}%."
    )

    return AttritionResult(
        total_employees=total,
        active_count=active_count,
        left_count=left_count,
        attrition_rate=attrition_rate,
        monthly_trends=monthly_trends,
        by_department=by_department,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. Tenure Distribution
# ---------------------------------------------------------------------------


@dataclass
class TenureBucket:
    """Count and percentage for a tenure range."""
    range_label: str
    count: int
    pct: float


@dataclass
class DeptTenure:
    """Average tenure for a department."""
    department: str
    avg_tenure: float
    count: int


@dataclass
class TenureResult:
    """Complete tenure distribution analysis result."""
    avg_tenure: float
    median_tenure: float
    buckets: list[TenureBucket]
    leaver_avg_tenure: float | None
    stayer_avg_tenure: float | None
    by_department: list[DeptTenure]
    summary: str


_TENURE_RANGES = [
    ("<1yr", 0.0, 1.0),
    ("1-2yr", 1.0, 2.0),
    ("2-5yr", 2.0, 5.0),
    ("5-10yr", 5.0, 10.0),
    (">10yr", 10.0, float("inf")),
]


def analyze_tenure_distribution(
    rows: list[dict],
    employee_column: str,
    hire_date_column: str,
    termination_date_column: str | None = None,
    department_column: str | None = None,
) -> TenureResult | None:
    """Analyze employee tenure distribution.

    Computes tenure in years from hire date to termination date (if provided)
    or to the maximum date found in the data.

    Args:
        rows: Data rows as dicts.
        employee_column: Column identifying the employee.
        hire_date_column: Column with hire/start date.
        termination_date_column: Optional column with termination/end date.
        department_column: Optional column for per-department breakdown.

    Returns:
        TenureResult or None if no valid data.
    """
    if not rows:
        return None

    # Find the reference date (max date in data) for stayers
    all_dates: list[date] = []
    for row in rows:
        hd = _parse_date(row.get(hire_date_column))
        if hd is not None:
            all_dates.append(hd)
        if termination_date_column is not None:
            td = _parse_date(row.get(termination_date_column))
            if td is not None:
                all_dates.append(td)

    if not all_dates:
        return None

    reference_date = max(all_dates)

    # Compute tenure for each unique employee
    employee_tenure: dict[str, float] = {}
    employee_is_leaver: dict[str, bool] = {}
    employee_dept: dict[str, str] = {}

    for row in rows:
        emp = row.get(employee_column)
        hire_d = _parse_date(row.get(hire_date_column))
        if emp is None or hire_d is None:
            continue

        emp_key = str(emp)
        end_d = reference_date
        is_leaver = False

        if termination_date_column is not None:
            td = _parse_date(row.get(termination_date_column))
            if td is not None:
                end_d = td
                is_leaver = True

        tenure_years = (end_d - hire_d).days / 365.25
        if tenure_years < 0:
            tenure_years = 0.0

        # Keep the latest record per employee
        if emp_key not in employee_tenure:
            employee_tenure[emp_key] = tenure_years
            employee_is_leaver[emp_key] = is_leaver
        else:
            employee_tenure[emp_key] = tenure_years
            employee_is_leaver[emp_key] = is_leaver

        if department_column is not None:
            dept = row.get(department_column)
            if dept is not None:
                employee_dept[emp_key] = str(dept)

    if not employee_tenure:
        return None

    tenures = list(employee_tenure.values())
    avg_tenure = round(sum(tenures) / len(tenures), 2)
    median_tenure = round(statistics.median(tenures), 2)

    # Build tenure buckets
    bucket_counts: dict[str, int] = {label: 0 for label, _, _ in _TENURE_RANGES}
    for t in tenures:
        for label, lo, hi in _TENURE_RANGES:
            if lo <= t < hi:
                bucket_counts[label] += 1
                break

    total_emp = len(tenures)
    buckets: list[TenureBucket] = []
    for label, _, _ in _TENURE_RANGES:
        cnt = bucket_counts[label]
        pct = round(cnt / total_emp * 100, 2) if total_emp > 0 else 0.0
        buckets.append(TenureBucket(range_label=label, count=cnt, pct=pct))

    # Leaver vs stayer avg tenure
    leaver_avg: float | None = None
    stayer_avg: float | None = None
    if termination_date_column is not None:
        leaver_tenures = [employee_tenure[e] for e in employee_tenure if employee_is_leaver[e]]
        stayer_tenures = [employee_tenure[e] for e in employee_tenure if not employee_is_leaver[e]]
        if leaver_tenures:
            leaver_avg = round(sum(leaver_tenures) / len(leaver_tenures), 2)
        if stayer_tenures:
            stayer_avg = round(sum(stayer_tenures) / len(stayer_tenures), 2)

    # Department breakdown
    by_department: list[DeptTenure] = []
    if department_column is not None:
        dept_tenures: dict[str, list[float]] = {}
        for emp_key, tenure in employee_tenure.items():
            dept = employee_dept.get(emp_key)
            if dept is not None:
                dept_tenures.setdefault(dept, []).append(tenure)

        for dept_name in sorted(dept_tenures.keys()):
            dt = dept_tenures[dept_name]
            d_avg = round(sum(dt) / len(dt), 2) if dt else 0.0
            by_department.append(DeptTenure(
                department=dept_name,
                avg_tenure=d_avg,
                count=len(dt),
            ))

    leaver_part = ""
    if leaver_avg is not None:
        leaver_part = f" Leaver avg tenure: {leaver_avg:.1f}yr."
    stayer_part = ""
    if stayer_avg is not None:
        stayer_part = f" Stayer avg tenure: {stayer_avg:.1f}yr."

    summary = (
        f"Tenure analysis: {total_emp} employees, "
        f"avg tenure = {avg_tenure:.1f}yr, "
        f"median = {median_tenure:.1f}yr."
        f"{leaver_part}{stayer_part}"
    )

    return TenureResult(
        avg_tenure=avg_tenure,
        median_tenure=median_tenure,
        buckets=buckets,
        leaver_avg_tenure=leaver_avg,
        stayer_avg_tenure=stayer_avg,
        by_department=by_department,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. Retention Cohorts
# ---------------------------------------------------------------------------


@dataclass
class CohortRetention:
    """Retention at a specific time milestone."""
    period: str
    retained_pct: float


@dataclass
class Cohort:
    """A hire cohort with retention metrics."""
    cohort_label: str
    starting_count: int
    retained_count: int
    retention_rate: float
    retention_milestones: list[CohortRetention]


@dataclass
class RetentionCohortResult:
    """Complete retention cohort analysis result."""
    cohorts: list[Cohort]
    overall_1yr_retention: float | None
    best_cohort: str | None
    worst_cohort: str | None
    summary: str


def compute_retention_cohorts(
    rows: list[dict],
    employee_column: str,
    hire_date_column: str,
    termination_date_column: str,
) -> RetentionCohortResult | None:
    """Group employees by hire quarter and compute retention rates.

    Args:
        rows: Data rows as dicts.
        employee_column: Column identifying the employee.
        hire_date_column: Column with hire/start date.
        termination_date_column: Column with termination/end date (None = still active).

    Returns:
        RetentionCohortResult or None if no valid data.
    """
    if not rows:
        return None

    # Find reference date
    all_dates: list[date] = []
    for row in rows:
        hd = _parse_date(row.get(hire_date_column))
        if hd is not None:
            all_dates.append(hd)
        td = _parse_date(row.get(termination_date_column))
        if td is not None:
            all_dates.append(td)

    if not all_dates:
        return None

    reference_date = max(all_dates)

    # Parse employee records
    employees: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        emp = row.get(employee_column)
        hire_d = _parse_date(row.get(hire_date_column))
        if emp is None or hire_d is None:
            continue

        emp_key = str(emp)
        if emp_key in seen:
            continue
        seen.add(emp_key)

        term_d = _parse_date(row.get(termination_date_column))
        employees.append({
            "employee": emp_key,
            "hire_date": hire_d,
            "term_date": term_d,
        })

    if not employees:
        return None

    # Group by hire quarter
    cohort_groups: dict[str, list[dict]] = {}
    for emp in employees:
        hd = emp["hire_date"]
        quarter = (hd.month - 1) // 3 + 1
        label = f"{hd.year}-Q{quarter}"
        cohort_groups.setdefault(label, []).append(emp)

    # Milestones in days
    milestones = [
        ("6mo", 183),
        ("1yr", 365),
        ("2yr", 730),
    ]

    cohorts: list[Cohort] = []
    one_yr_retained_total = 0
    one_yr_starting_total = 0

    for label in sorted(cohort_groups.keys()):
        group = cohort_groups[label]
        starting = len(group)

        # Retained = still active at reference date (no termination, or terminated after reference)
        retained = 0
        for emp in group:
            if emp["term_date"] is None:
                retained += 1
            # else they left
        retention_rate = round(retained / starting * 100, 2) if starting > 0 else 0.0

        # Milestones
        retention_ms: list[CohortRetention] = []
        for ms_label, ms_days in milestones:
            eligible = 0
            survived = 0
            for emp in group:
                milestone_date = date.fromordinal(emp["hire_date"].toordinal() + ms_days)
                # Only include if enough time has passed to evaluate this milestone
                if milestone_date <= reference_date:
                    eligible += 1
                    if emp["term_date"] is None or emp["term_date"] >= milestone_date:
                        survived += 1

            if eligible > 0:
                ms_pct = round(survived / eligible * 100, 2)
                retention_ms.append(CohortRetention(period=ms_label, retained_pct=ms_pct))

        cohorts.append(Cohort(
            cohort_label=label,
            starting_count=starting,
            retained_count=retained,
            retention_rate=retention_rate,
            retention_milestones=retention_ms,
        ))

        # Track 1yr retention
        for emp in group:
            one_yr_date = date.fromordinal(emp["hire_date"].toordinal() + 365)
            if one_yr_date <= reference_date:
                one_yr_starting_total += 1
                if emp["term_date"] is None or emp["term_date"] >= one_yr_date:
                    one_yr_retained_total += 1

    if not cohorts:
        return None

    overall_1yr = None
    if one_yr_starting_total > 0:
        overall_1yr = round(one_yr_retained_total / one_yr_starting_total * 100, 2)

    # Best and worst cohort by retention rate
    best = max(cohorts, key=lambda c: c.retention_rate)
    worst = min(cohorts, key=lambda c: c.retention_rate)

    summary = (
        f"Retention cohort analysis: {len(cohorts)} cohorts, "
        f"{sum(c.starting_count for c in cohorts)} total employees. "
        f"Best cohort: {best.cohort_label} ({best.retention_rate:.1f}% retained). "
        f"Worst cohort: {worst.cohort_label} ({worst.retention_rate:.1f}% retained)."
    )
    if overall_1yr is not None:
        summary += f" Overall 1yr retention: {overall_1yr:.1f}%."

    return RetentionCohortResult(
        cohorts=cohorts,
        overall_1yr_retention=overall_1yr,
        best_cohort=best.cohort_label,
        worst_cohort=worst.cohort_label,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. Attrition Drivers
# ---------------------------------------------------------------------------


@dataclass
class DriverFactor:
    """A single factor analyzed for its impact on attrition."""
    factor_name: str
    factor_type: str  # "numeric" or "categorical"
    leaver_value: str  # avg for numeric, highest leaving rate category for categorical
    stayer_value: str  # avg for numeric, lowest leaving rate category for categorical
    impact: float  # absolute difference as percentage
    direction: str  # e.g. "higher for leavers" or "lower for leavers"


@dataclass
class AttritionDriverResult:
    """Complete attrition driver analysis result."""
    factors: list[DriverFactor]
    top_driver: str | None
    summary: str


def analyze_attrition_drivers(
    rows: list[dict],
    employee_column: str,
    status_column: str,
    factor_columns: list[str],
) -> AttritionDriverResult | None:
    """Analyze factors that drive employee attrition.

    For each factor column, compares the distribution between Active and
    Left employees to identify which factors most strongly correlate with
    attrition.

    Args:
        rows: Data rows as dicts.
        employee_column: Column identifying the employee.
        status_column: Column with employment status (Active/Left).
        factor_columns: List of column names to analyze as potential drivers.

    Returns:
        AttritionDriverResult or None if no valid data.
    """
    if not rows or not factor_columns:
        return None

    # Separate leavers and stayers
    leavers: list[dict] = []
    stayers: list[dict] = []

    for row in rows:
        emp = row.get(employee_column)
        status_raw = row.get(status_column)
        if emp is None or status_raw is None:
            continue
        classified = _classify_status(str(status_raw))
        if classified == "left":
            leavers.append(row)
        elif classified == "active":
            stayers.append(row)

    if not leavers or not stayers:
        return None

    factors: list[DriverFactor] = []

    for col in factor_columns:
        # Gather values for leavers and stayers
        leaver_vals = [row.get(col) for row in leavers if row.get(col) is not None]
        stayer_vals = [row.get(col) for row in stayers if row.get(col) is not None]

        if not leaver_vals or not stayer_vals:
            continue

        # Determine if numeric or categorical
        leaver_floats = [_safe_float(v) for v in leaver_vals]
        stayer_floats = [_safe_float(v) for v in stayer_vals]

        leaver_numeric = [f for f in leaver_floats if f is not None]
        stayer_numeric = [f for f in stayer_floats if f is not None]

        # Consider numeric if majority of values parse as float
        total_vals = len(leaver_vals) + len(stayer_vals)
        numeric_vals = len(leaver_numeric) + len(stayer_numeric)

        if numeric_vals >= total_vals * 0.5 and leaver_numeric and stayer_numeric:
            # Numeric factor
            leaver_avg = sum(leaver_numeric) / len(leaver_numeric)
            stayer_avg = sum(stayer_numeric) / len(stayer_numeric)

            # Impact as percentage difference relative to stayer avg
            if stayer_avg != 0:
                impact = abs(leaver_avg - stayer_avg) / abs(stayer_avg) * 100
            else:
                impact = abs(leaver_avg - stayer_avg) * 100

            direction = "higher for leavers" if leaver_avg > stayer_avg else "lower for leavers"
            if leaver_avg == stayer_avg:
                direction = "no difference"

            factors.append(DriverFactor(
                factor_name=col,
                factor_type="numeric",
                leaver_value=f"{leaver_avg:.2f}",
                stayer_value=f"{stayer_avg:.2f}",
                impact=round(impact, 2),
                direction=direction,
            ))
        else:
            # Categorical factor
            # Compute leaving rate per category
            cat_counts: dict[str, dict[str, int]] = {}
            for row in leavers:
                v = row.get(col)
                if v is not None:
                    key = str(v)
                    cat_counts.setdefault(key, {"left": 0, "active": 0})
                    cat_counts[key]["left"] += 1
            for row in stayers:
                v = row.get(col)
                if v is not None:
                    key = str(v)
                    cat_counts.setdefault(key, {"left": 0, "active": 0})
                    cat_counts[key]["active"] += 1

            # Compute leaving rate per category
            cat_rates: dict[str, float] = {}
            for cat, counts in cat_counts.items():
                total = counts["left"] + counts["active"]
                if total > 0:
                    cat_rates[cat] = counts["left"] / total * 100

            if not cat_rates:
                continue

            highest_cat = max(cat_rates, key=cat_rates.get)  # type: ignore
            lowest_cat = min(cat_rates, key=cat_rates.get)  # type: ignore
            impact = abs(cat_rates[highest_cat] - cat_rates[lowest_cat])

            direction = f"{highest_cat} has highest attrition"

            factors.append(DriverFactor(
                factor_name=col,
                factor_type="categorical",
                leaver_value=highest_cat,
                stayer_value=lowest_cat,
                impact=round(impact, 2),
                direction=direction,
            ))

    if not factors:
        return None

    # Sort by impact descending
    factors.sort(key=lambda f: f.impact, reverse=True)

    top_driver = factors[0].factor_name if factors else None

    summary_parts = [f"Attrition driver analysis: {len(factors)} factors analyzed."]
    if top_driver:
        summary_parts.append(
            f" Top driver: {top_driver} (impact: {factors[0].impact:.1f}%, "
            f"{factors[0].direction})."
        )
    summary = "".join(summary_parts)

    return AttritionDriverResult(
        factors=factors,
        top_driver=top_driver,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. Combined Attrition Report
# ---------------------------------------------------------------------------


def format_attrition_report(
    attrition: AttritionResult | None = None,
    tenure: TenureResult | None = None,
    cohorts: RetentionCohortResult | None = None,
    drivers: AttritionDriverResult | None = None,
) -> str:
    """Generate a combined text report from available attrition analyses.

    Args:
        attrition: Attrition rate analysis result.
        tenure: Tenure distribution analysis result.
        cohorts: Retention cohort analysis result.
        drivers: Attrition driver analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Employee Attrition Report")
    sections.append("=" * 40)

    if attrition is not None:
        lines = ["", "Attrition Rate Analysis", "-" * 38]
        lines.append(
            f"  Total employees: {attrition.total_employees}"
        )
        lines.append(
            f"  Active: {attrition.active_count} | Left: {attrition.left_count}"
        )
        lines.append(
            f"  Attrition rate: {attrition.attrition_rate:.1f}%"
        )
        if attrition.monthly_trends:
            lines.append("  Monthly trends:")
            for mt in attrition.monthly_trends:
                lines.append(
                    f"    {mt.month}: {mt.left} left of {mt.active + mt.left} "
                    f"({mt.rate:.1f}%)"
                )
        if attrition.by_department:
            lines.append("  By department:")
            for da in attrition.by_department:
                lines.append(
                    f"    {da.department}: {da.left}/{da.total} "
                    f"({da.rate:.1f}%)"
                )
        sections.append("\n".join(lines))

    if tenure is not None:
        lines = ["", "Tenure Distribution", "-" * 38]
        lines.append(
            f"  Avg tenure: {tenure.avg_tenure:.1f} years"
        )
        lines.append(
            f"  Median tenure: {tenure.median_tenure:.1f} years"
        )
        if tenure.buckets:
            lines.append("  Distribution:")
            for b in tenure.buckets:
                lines.append(
                    f"    {b.range_label}: {b.count} ({b.pct:.1f}%)"
                )
        if tenure.leaver_avg_tenure is not None:
            lines.append(
                f"  Leaver avg tenure: {tenure.leaver_avg_tenure:.1f} years"
            )
        if tenure.stayer_avg_tenure is not None:
            lines.append(
                f"  Stayer avg tenure: {tenure.stayer_avg_tenure:.1f} years"
            )
        if tenure.by_department:
            lines.append("  By department:")
            for dt in tenure.by_department:
                lines.append(
                    f"    {dt.department}: avg={dt.avg_tenure:.1f}yr ({dt.count} employees)"
                )
        sections.append("\n".join(lines))

    if cohorts is not None:
        lines = ["", "Retention Cohorts", "-" * 38]
        for c in cohorts.cohorts:
            lines.append(
                f"  {c.cohort_label}: {c.retained_count}/{c.starting_count} "
                f"retained ({c.retention_rate:.1f}%)"
            )
            for ms in c.retention_milestones:
                lines.append(
                    f"    {ms.period}: {ms.retained_pct:.1f}% retained"
                )
        if cohorts.overall_1yr_retention is not None:
            lines.append(
                f"  Overall 1yr retention: {cohorts.overall_1yr_retention:.1f}%"
            )
        if cohorts.best_cohort:
            lines.append(f"  Best cohort: {cohorts.best_cohort}")
        if cohorts.worst_cohort:
            lines.append(f"  Worst cohort: {cohorts.worst_cohort}")
        sections.append("\n".join(lines))

    if drivers is not None:
        lines = ["", "Attrition Drivers", "-" * 38]
        for f in drivers.factors:
            lines.append(
                f"  {f.factor_name} ({f.factor_type}): "
                f"impact={f.impact:.1f}%, {f.direction}"
            )
            lines.append(
                f"    Leaver: {f.leaver_value} | Stayer: {f.stayer_value}"
            )
        if drivers.top_driver:
            lines.append(f"  Top driver: {drivers.top_driver}")
        sections.append("\n".join(lines))

    if attrition is None and tenure is None and cohorts is None and drivers is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
