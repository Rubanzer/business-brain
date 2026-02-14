"""Workforce and HR analytics for manufacturing environments.

Pure functions for analyzing attendance patterns, labor productivity,
overtime tracking, and headcount distribution across departments.
"""

from __future__ import annotations

from dataclasses import dataclass


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
# 1. Attendance Analysis
# ---------------------------------------------------------------------------


@dataclass
class EmployeeAttendance:
    """Attendance breakdown for a single employee."""

    employee: str
    total_days: int
    present_days: int
    absent_days: int
    leave_days: int
    attendance_rate: float  # present / total * 100


@dataclass
class AttendanceResult:
    """Aggregated attendance analysis across all employees."""

    employees: list[EmployeeAttendance]
    total_employees: int
    avg_attendance_rate: float
    chronic_absentees: list[str]  # employees with attendance_rate < 80%
    perfect_attendance: list[str]  # employees with 100% attendance
    summary: str


def analyze_attendance(
    rows: list[dict],
    employee_column: str,
    status_column: str,
    date_column: str | None = None,
) -> AttendanceResult | None:
    """Analyze employee attendance from row data.

    Each row represents one attendance record. The status column should
    contain values like "present", "absent", or "leave" (case-insensitive).

    Args:
        rows: Data rows as dicts.
        employee_column: Column identifying the employee.
        status_column: Column with attendance status (present/absent/leave).
        date_column: Optional column with date (used for counting unique days).

    Returns:
        AttendanceResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate per employee: count of present, absent, leave
    acc: dict[str, dict[str, int]] = {}

    for row in rows:
        employee = row.get(employee_column)
        status = row.get(status_column)
        if employee is None or status is None:
            continue

        emp_key = str(employee)
        status_lower = str(status).strip().lower()

        if emp_key not in acc:
            acc[emp_key] = {"present": 0, "absent": 0, "leave": 0}

        if status_lower in ("present", "p", "1", "yes"):
            acc[emp_key]["present"] += 1
        elif status_lower in ("absent", "a", "0", "no"):
            acc[emp_key]["absent"] += 1
        elif status_lower in ("leave", "l", "sick", "vacation", "holiday"):
            acc[emp_key]["leave"] += 1
        else:
            # Unknown status counts as absent
            acc[emp_key]["absent"] += 1

    if not acc:
        return None

    employees: list[EmployeeAttendance] = []
    for emp_name, counts in acc.items():
        total = counts["present"] + counts["absent"] + counts["leave"]
        rate = (counts["present"] / total * 100) if total > 0 else 0.0
        employees.append(
            EmployeeAttendance(
                employee=emp_name,
                total_days=total,
                present_days=counts["present"],
                absent_days=counts["absent"],
                leave_days=counts["leave"],
                attendance_rate=round(rate, 2),
            )
        )

    # Sort by attendance rate descending
    employees.sort(key=lambda e: e.attendance_rate, reverse=True)

    avg_rate = sum(e.attendance_rate for e in employees) / len(employees)
    chronic = [e.employee for e in employees if e.attendance_rate < 80.0]
    perfect = [e.employee for e in employees if e.attendance_rate == 100.0]

    summary = (
        f"Attendance across {len(employees)} employees: "
        f"Avg rate = {avg_rate:.1f}%. "
        f"{len(perfect)} with perfect attendance, "
        f"{len(chronic)} chronic absentees (<80%)."
    )

    return AttendanceResult(
        employees=employees,
        total_employees=len(employees),
        avg_attendance_rate=round(avg_rate, 2),
        chronic_absentees=chronic,
        perfect_attendance=perfect,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. Labor Productivity
# ---------------------------------------------------------------------------


@dataclass
class EntityProductivity:
    """Productivity metrics for a single entity (worker, line, team)."""

    entity: str
    total_output: float
    total_hours: float
    productivity: float  # output / hours
    productivity_index: float  # normalized to best = 100


@dataclass
class ProductivityResult:
    """Aggregated labor productivity analysis."""

    entities: list[EntityProductivity]
    mean_productivity: float
    best_entity: str
    worst_entity: str
    spread_ratio: float  # best / worst productivity
    summary: str


def compute_labor_productivity(
    rows: list[dict],
    entity_column: str,
    output_column: str,
    hours_column: str,
) -> ProductivityResult | None:
    """Compute output per labor hour for each entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (worker, line, etc.).
        output_column: Column with production output quantity.
        hours_column: Column with labor hours worked.

    Returns:
        ProductivityResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        output = _safe_float(row.get(output_column))
        hours = _safe_float(row.get(hours_column))
        if entity is None or output is None or hours is None:
            continue

        key = str(entity)
        if key not in acc:
            acc[key] = {"output": 0.0, "hours": 0.0}
        acc[key]["output"] += output
        acc[key]["hours"] += hours

    if not acc:
        return None

    # First pass: compute raw productivity
    raw: list[tuple[str, float, float, float]] = []  # (name, output, hours, productivity)
    for name, vals in acc.items():
        if vals["hours"] == 0:
            prod = 0.0
        else:
            prod = vals["output"] / vals["hours"]
        raw.append((name, vals["output"], vals["hours"], prod))

    best_prod = max(r[3] for r in raw)

    entities: list[EntityProductivity] = []
    for name, output, hours, prod in raw:
        index = (prod / best_prod * 100) if best_prod > 0 else 0.0
        entities.append(
            EntityProductivity(
                entity=name,
                total_output=round(output, 4),
                total_hours=round(hours, 4),
                productivity=round(prod, 4),
                productivity_index=round(index, 2),
            )
        )

    # Sort by productivity descending
    entities.sort(key=lambda e: e.productivity, reverse=True)

    mean_prod = sum(e.productivity for e in entities) / len(entities)
    best = entities[0]
    worst = entities[-1]

    spread = (best.productivity / worst.productivity) if worst.productivity > 0 else 0.0

    summary = (
        f"Productivity across {len(entities)} entities: "
        f"Mean = {mean_prod:.2f} units/hr. "
        f"Best = {best.entity} ({best.productivity:.2f}), "
        f"Worst = {worst.entity} ({worst.productivity:.2f}). "
        f"Spread ratio: {spread:.2f}x."
    )

    return ProductivityResult(
        entities=entities,
        mean_productivity=round(mean_prod, 4),
        best_entity=best.entity,
        worst_entity=worst.entity,
        spread_ratio=round(spread, 4),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. Overtime Analysis
# ---------------------------------------------------------------------------


@dataclass
class EmployeeOvertime:
    """Overtime metrics for a single employee."""

    employee: str
    regular_hours: float
    actual_hours: float
    overtime_hours: float
    overtime_pct: float  # overtime / regular * 100
    overtime_cost: float | None  # if rate provided


@dataclass
class OvertimeResult:
    """Aggregated overtime analysis across all employees."""

    employees: list[EmployeeOvertime]
    total_overtime_hours: float
    total_regular_hours: float
    overtime_pct: float  # total overtime / total regular * 100
    top_overtime_employees: list[str]  # top 3 by overtime hours
    summary: str


def analyze_overtime(
    rows: list[dict],
    employee_column: str,
    regular_hours_column: str,
    actual_hours_column: str,
    rate_column: str | None = None,
) -> OvertimeResult | None:
    """Track overtime hours and costs per employee.

    Overtime is computed as actual_hours - regular_hours (floored at 0).

    Args:
        rows: Data rows as dicts.
        employee_column: Column identifying the employee.
        regular_hours_column: Column with scheduled/regular hours.
        actual_hours_column: Column with actual hours worked.
        rate_column: Optional column with overtime pay rate (cost per OT hour).

    Returns:
        OvertimeResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}

    for row in rows:
        employee = row.get(employee_column)
        regular = _safe_float(row.get(regular_hours_column))
        actual = _safe_float(row.get(actual_hours_column))
        if employee is None or regular is None or actual is None:
            continue

        rate_val = 0.0
        if rate_column is not None:
            r = _safe_float(row.get(rate_column))
            if r is not None:
                rate_val = r

        emp_key = str(employee)
        if emp_key not in acc:
            acc[emp_key] = {"regular": 0.0, "actual": 0.0, "ot_cost": 0.0}

        overtime = max(0.0, actual - regular)
        acc[emp_key]["regular"] += regular
        acc[emp_key]["actual"] += actual
        acc[emp_key]["ot_cost"] += overtime * rate_val

    if not acc:
        return None

    employees: list[EmployeeOvertime] = []
    for emp_name, vals in acc.items():
        regular = vals["regular"]
        actual = vals["actual"]
        ot = max(0.0, actual - regular)
        ot_pct = (ot / regular * 100) if regular > 0 else 0.0
        ot_cost = vals["ot_cost"] if rate_column is not None else None

        employees.append(
            EmployeeOvertime(
                employee=emp_name,
                regular_hours=round(regular, 4),
                actual_hours=round(actual, 4),
                overtime_hours=round(ot, 4),
                overtime_pct=round(ot_pct, 2),
                overtime_cost=round(ot_cost, 2) if ot_cost is not None else None,
            )
        )

    # Sort by overtime hours descending
    employees.sort(key=lambda e: e.overtime_hours, reverse=True)

    total_ot = sum(e.overtime_hours for e in employees)
    total_reg = sum(e.regular_hours for e in employees)
    overall_ot_pct = (total_ot / total_reg * 100) if total_reg > 0 else 0.0

    top_ot = [e.employee for e in employees[:3]]

    summary = (
        f"Overtime across {len(employees)} employees: "
        f"Total OT = {total_ot:,.1f} hrs / {total_reg:,.1f} regular hrs "
        f"({overall_ot_pct:.1f}%). "
        f"Top OT: {', '.join(top_ot)}."
    )

    return OvertimeResult(
        employees=employees,
        total_overtime_hours=round(total_ot, 4),
        total_regular_hours=round(total_reg, 4),
        overtime_pct=round(overall_ot_pct, 2),
        top_overtime_employees=top_ot,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. Headcount Analysis
# ---------------------------------------------------------------------------


@dataclass
class DeptHeadcount:
    """Headcount and output metrics for a single department."""

    department: str
    headcount: int
    share_pct: float  # department headcount / total * 100
    output_total: float | None  # total output if output_column provided
    output_per_head: float | None  # output_total / headcount if output_column provided


@dataclass
class HeadcountResult:
    """Aggregated headcount analysis across departments."""

    departments: list[DeptHeadcount]
    total_headcount: int
    largest_dept: str
    smallest_dept: str
    output_per_head: float | None  # overall output / total headcount
    summary: str


def headcount_analysis(
    rows: list[dict],
    department_column: str,
    employee_column: str,
    output_column: str | None = None,
) -> HeadcountResult | None:
    """Analyze headcount distribution by department.

    Counts unique employees per department. If output_column is provided,
    also computes output per head.

    Args:
        rows: Data rows as dicts.
        department_column: Column identifying the department.
        employee_column: Column identifying the employee (for unique count).
        output_column: Optional column with production output.

    Returns:
        HeadcountResult or None if no valid data.
    """
    if not rows:
        return None

    # Track unique employees per department and output totals
    dept_employees: dict[str, set[str]] = {}
    dept_output: dict[str, float] = {}

    for row in rows:
        dept = row.get(department_column)
        employee = row.get(employee_column)
        if dept is None or employee is None:
            continue

        dept_key = str(dept)
        emp_key = str(employee)

        if dept_key not in dept_employees:
            dept_employees[dept_key] = set()
        dept_employees[dept_key].add(emp_key)

        if output_column is not None:
            out_val = _safe_float(row.get(output_column))
            if out_val is not None:
                dept_output[dept_key] = dept_output.get(dept_key, 0.0) + out_val

    if not dept_employees:
        return None

    total_headcount = sum(len(emps) for emps in dept_employees.values())
    total_output = sum(dept_output.values()) if dept_output else None

    departments: list[DeptHeadcount] = []
    for dept_name in sorted(dept_employees.keys()):
        hc = len(dept_employees[dept_name])
        share = (hc / total_headcount * 100) if total_headcount > 0 else 0.0

        out_total: float | None = None
        out_per_head: float | None = None
        if output_column is not None:
            out_total = dept_output.get(dept_name, 0.0)
            out_per_head = (out_total / hc) if hc > 0 else 0.0
            out_total = round(out_total, 4)
            out_per_head = round(out_per_head, 4)

        departments.append(
            DeptHeadcount(
                department=dept_name,
                headcount=hc,
                share_pct=round(share, 2),
                output_total=out_total,
                output_per_head=out_per_head,
            )
        )

    # Sort by headcount descending
    departments.sort(key=lambda d: d.headcount, reverse=True)

    largest = departments[0].department
    smallest = departments[-1].department

    overall_output_per_head: float | None = None
    if total_output is not None and total_headcount > 0:
        overall_output_per_head = round(total_output / total_headcount, 4)

    summary_parts = [
        f"Headcount across {len(departments)} departments: "
        f"Total = {total_headcount}. "
        f"Largest = {largest} ({departments[0].headcount}), "
        f"Smallest = {smallest} ({departments[-1].headcount}).",
    ]
    if overall_output_per_head is not None:
        summary_parts.append(
            f" Overall output per head: {overall_output_per_head:.2f}."
        )

    return HeadcountResult(
        departments=departments,
        total_headcount=total_headcount,
        largest_dept=largest,
        smallest_dept=smallest,
        output_per_head=overall_output_per_head,
        summary="".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 5. Combined Workforce Report
# ---------------------------------------------------------------------------


def format_workforce_report(
    attendance: AttendanceResult | None = None,
    productivity: ProductivityResult | None = None,
    overtime: OvertimeResult | None = None,
    headcount: HeadcountResult | None = None,
) -> str:
    """Generate a combined text report from available workforce analyses.

    Args:
        attendance: Attendance analysis result.
        productivity: Productivity analysis result.
        overtime: Overtime analysis result.
        headcount: Headcount analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Workforce Analytics Report")
    sections.append("=" * 40)

    if attendance is not None:
        lines = ["", "Attendance Analysis", "-" * 38]
        for e in attendance.employees:
            lines.append(
                f"  {e.employee}: rate={e.attendance_rate:.1f}% "
                f"(P={e.present_days} A={e.absent_days} L={e.leave_days})"
            )
        lines.append(f"  Avg attendance rate: {attendance.avg_attendance_rate:.1f}%")
        if attendance.chronic_absentees:
            lines.append(
                f"  Chronic absentees (<80%): {', '.join(attendance.chronic_absentees)}"
            )
        if attendance.perfect_attendance:
            lines.append(
                f"  Perfect attendance: {', '.join(attendance.perfect_attendance)}"
            )
        sections.append("\n".join(lines))

    if productivity is not None:
        lines = ["", "Labor Productivity", "-" * 38]
        for e in productivity.entities:
            lines.append(
                f"  {e.entity}: {e.productivity:.2f} units/hr "
                f"(index={e.productivity_index:.1f})"
            )
        lines.append(
            f"  Mean productivity: {productivity.mean_productivity:.2f} units/hr"
        )
        lines.append(
            f"  Best: {productivity.best_entity} | "
            f"Worst: {productivity.worst_entity} | "
            f"Spread: {productivity.spread_ratio:.2f}x"
        )
        sections.append("\n".join(lines))

    if overtime is not None:
        lines = ["", "Overtime Analysis", "-" * 38]
        for e in overtime.employees:
            cost_str = f" cost={e.overtime_cost:.2f}" if e.overtime_cost is not None else ""
            lines.append(
                f"  {e.employee}: OT={e.overtime_hours:.1f} hrs "
                f"({e.overtime_pct:.1f}%){cost_str}"
            )
        lines.append(
            f"  Total OT: {overtime.total_overtime_hours:,.1f} hrs "
            f"({overtime.overtime_pct:.1f}%)"
        )
        if overtime.top_overtime_employees:
            lines.append(
                f"  Top OT employees: {', '.join(overtime.top_overtime_employees)}"
            )
        sections.append("\n".join(lines))

    if headcount is not None:
        lines = ["", "Headcount Analysis", "-" * 38]
        for d in headcount.departments:
            out_str = ""
            if d.output_per_head is not None:
                out_str = f" output/head={d.output_per_head:.2f}"
            lines.append(
                f"  {d.department}: {d.headcount} ({d.share_pct:.1f}%){out_str}"
            )
        lines.append(f"  Total headcount: {headcount.total_headcount}")
        lines.append(
            f"  Largest: {headcount.largest_dept} | "
            f"Smallest: {headcount.smallest_dept}"
        )
        if headcount.output_per_head is not None:
            lines.append(
                f"  Overall output per head: {headcount.output_per_head:.2f}"
            )
        sections.append("\n".join(lines))

    if attendance is None and productivity is None and overtime is None and headcount is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
