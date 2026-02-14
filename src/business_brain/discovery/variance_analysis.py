"""Budget vs actual (plan vs actual) variance analysis.

Pure functions for computing variances, identifying root causes,
building waterfall breakdowns, and tracking variance trends over time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VarianceItem:
    """A single line item in the variance analysis."""

    category: str
    planned: float
    actual: float
    variance: float  # actual - planned
    variance_pct: float  # (actual - planned) / planned * 100
    is_favorable: bool  # depends on context
    severity: str  # "critical" (>20%), "warning" (>10%), "minor" (<=10%)


@dataclass
class VarianceReport:
    """Complete variance analysis result."""

    items: list[VarianceItem]
    total_planned: float
    total_actual: float
    total_variance: float
    total_variance_pct: float
    favorable_count: int
    unfavorable_count: int
    largest_variance: VarianceItem | None
    summary: str


def _classify_severity(variance_pct: float) -> str:
    """Return severity based on absolute variance percentage."""
    abs_pct = abs(variance_pct)
    if abs_pct > 20:
        return "critical"
    if abs_pct > 10:
        return "warning"
    return "minor"


def _safe_variance_pct(planned: float, variance: float) -> float:
    """Compute variance percentage, handling division by zero."""
    if planned == 0:
        return 0.0 if variance == 0 else float("inf") if variance > 0 else float("-inf")
    return variance / planned * 100


def _is_favorable(variance: float, favorable_direction: str) -> bool:
    """Determine whether a variance is favorable.

    For 'higher' direction (revenue): positive variance is favorable.
    For 'lower' direction (cost): negative variance is favorable.
    Zero variance is always favorable.
    """
    if variance == 0:
        return True
    if favorable_direction == "higher":
        return variance > 0
    return variance < 0


def compute_variance(
    rows: list[dict],
    category_column: str,
    planned_column: str,
    actual_column: str,
    favorable_direction: str = "higher",
) -> VarianceReport | None:
    """Compute variance analysis from row data.

    Args:
        rows: Data rows as dicts.
        category_column: Column to group by.
        planned_column: Column with planned/budgeted values.
        actual_column: Column with actual values.
        favorable_direction: 'higher' means actual > planned is good (revenue);
                             'lower' means actual < planned is good (cost).

    Returns:
        VarianceReport or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate by category
    planned_agg: dict[str, float] = {}
    actual_agg: dict[str, float] = {}

    for row in rows:
        cat = row.get(category_column)
        p = row.get(planned_column)
        a = row.get(actual_column)
        if cat is None or p is None or a is None:
            continue
        try:
            p_val = float(p)
            a_val = float(a)
        except (TypeError, ValueError):
            continue
        key = str(cat)
        planned_agg[key] = planned_agg.get(key, 0.0) + p_val
        actual_agg[key] = actual_agg.get(key, 0.0) + a_val

    if not planned_agg:
        return None

    # Build variance items
    items: list[VarianceItem] = []
    for cat in sorted(planned_agg):
        p = planned_agg[cat]
        a = actual_agg.get(cat, 0.0)
        v = a - p
        vpct = _safe_variance_pct(p, v)
        fav = _is_favorable(v, favorable_direction)
        sev = _classify_severity(vpct)
        items.append(
            VarianceItem(
                category=cat,
                planned=round(p, 4),
                actual=round(a, 4),
                variance=round(v, 4),
                variance_pct=round(vpct, 2) if not (vpct == float("inf") or vpct == float("-inf")) else vpct,
                is_favorable=fav,
                severity=sev,
            )
        )

    total_planned = sum(it.planned for it in items)
    total_actual = sum(it.actual for it in items)
    total_variance = total_actual - total_planned
    total_variance_pct = _safe_variance_pct(total_planned, total_variance)
    if not (total_variance_pct == float("inf") or total_variance_pct == float("-inf")):
        total_variance_pct = round(total_variance_pct, 2)

    fav_count = sum(1 for it in items if it.is_favorable)
    unfav_count = len(items) - fav_count

    largest = max(items, key=lambda it: abs(it.variance)) if items else None

    # Build summary
    direction_label = "over" if total_variance > 0 else "under"
    summary_parts = [
        f"Total variance: {total_variance:+,.2f} ({direction_label} by "
        f"{abs(total_variance_pct):.1f}%)."
        if not (total_variance_pct == float("inf") or total_variance_pct == float("-inf"))
        else f"Total variance: {total_variance:+,.2f} (infinite %).",
        f" {fav_count} favorable, {unfav_count} unfavorable.",
    ]
    if largest:
        summary_parts.append(
            f" Largest outlier: {largest.category} ({largest.variance:+,.2f})."
        )

    return VarianceReport(
        items=items,
        total_planned=round(total_planned, 4),
        total_actual=round(total_actual, 4),
        total_variance=round(total_variance, 4),
        total_variance_pct=total_variance_pct,
        favorable_count=fav_count,
        unfavorable_count=unfav_count,
        largest_variance=largest,
        summary="".join(summary_parts),
    )


def waterfall_breakdown(report: VarianceReport) -> list[dict]:
    """Create waterfall chart data showing cumulative variance contribution.

    Each entry has a running start/end so the waterfall can be plotted.
    Returns:
        List of dicts with keys: category, start, end, variance, type.
    """
    result: list[dict] = []
    running = 0.0
    for item in report.items:
        start = running
        end = running + item.variance
        result.append(
            {
                "category": item.category,
                "start": round(start, 4),
                "end": round(end, 4),
                "variance": round(item.variance, 4),
                "type": "favorable" if item.is_favorable else "unfavorable",
            }
        )
        running = end
    return result


def find_root_causes(
    report: VarianceReport,
    threshold_pct: float = 10.0,
) -> list[dict]:
    """Identify items with largest absolute variance as potential root causes.

    Only items whose absolute variance percentage exceeds threshold_pct are
    included.  Results are sorted by absolute variance descending.
    """
    causes: list[dict] = []
    for item in report.items:
        abs_pct = abs(item.variance_pct) if not (
            item.variance_pct == float("inf") or item.variance_pct == float("-inf")
        ) else float("inf")
        if abs_pct >= threshold_pct:
            causes.append(
                {
                    "category": item.category,
                    "variance": item.variance,
                    "variance_pct": item.variance_pct,
                    "is_favorable": item.is_favorable,
                    "severity": item.severity,
                }
            )
    causes.sort(key=lambda c: abs(c["variance"]), reverse=True)
    return causes


def compute_variance_trend(
    periods: list[str],
    planned_series: list[float],
    actual_series: list[float],
    favorable_direction: str = "higher",
) -> list[dict]:
    """Compute variance over multiple periods.

    Args:
        periods: Period labels (e.g. ["Q1", "Q2", "Q3"]).
        planned_series: Planned values, one per period.
        actual_series: Actual values, one per period.
        favorable_direction: 'higher' or 'lower'.

    Returns:
        List of dicts with period, planned, actual, variance, variance_pct,
        is_favorable.
    """
    n = min(len(periods), len(planned_series), len(actual_series))
    result: list[dict] = []
    for i in range(n):
        p = planned_series[i]
        a = actual_series[i]
        v = a - p
        vpct = _safe_variance_pct(p, v)
        if not (vpct == float("inf") or vpct == float("-inf")):
            vpct = round(vpct, 2)
        result.append(
            {
                "period": periods[i],
                "planned": p,
                "actual": a,
                "variance": round(v, 4),
                "variance_pct": vpct,
                "is_favorable": _is_favorable(v, favorable_direction),
            }
        )
    return result


def format_variance_table(report: VarianceReport) -> str:
    """Format variance report as an aligned text table.

    Favorable items are marked with '+', unfavorable with '-'.
    """
    # Determine column widths
    cat_header = "Category"
    plan_header = "Planned"
    act_header = "Actual"
    var_header = "Variance"
    pct_header = "Var %"
    status_header = "Status"

    cat_w = max(len(cat_header), *(len(it.category) for it in report.items)) if report.items else len(cat_header)
    plan_w = max(len(plan_header), 12)
    act_w = max(len(act_header), 12)
    var_w = max(len(var_header), 14)
    pct_w = max(len(pct_header), 10)
    status_w = max(len(status_header), 8)

    def _fmt_pct(v: float) -> str:
        if v == float("inf"):
            return "+inf%"
        if v == float("-inf"):
            return "-inf%"
        return f"{v:+.1f}%"

    def _fmt_num(v: float) -> str:
        return f"{v:,.2f}"

    def _fmt_var(v: float) -> str:
        return f"{v:+,.2f}"

    lines: list[str] = []

    # Header
    header = (
        f"{'Category':<{cat_w}}  "
        f"{'Planned':>{plan_w}}  "
        f"{'Actual':>{act_w}}  "
        f"{'Variance':>{var_w}}  "
        f"{'Var %':>{pct_w}}  "
        f"{'Status':>{status_w}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for it in report.items:
        status = "[+]" if it.is_favorable else "[-]"
        row = (
            f"{it.category:<{cat_w}}  "
            f"{_fmt_num(it.planned):>{plan_w}}  "
            f"{_fmt_num(it.actual):>{act_w}}  "
            f"{_fmt_var(it.variance):>{var_w}}  "
            f"{_fmt_pct(it.variance_pct):>{pct_w}}  "
            f"{status:>{status_w}}"
        )
        lines.append(row)

    # Separator and totals
    lines.append("-" * len(header))
    total_status = "[+]" if _is_favorable(report.total_variance, "higher") else "[-]"
    total_row = (
        f"{'TOTAL':<{cat_w}}  "
        f"{_fmt_num(report.total_planned):>{plan_w}}  "
        f"{_fmt_num(report.total_actual):>{act_w}}  "
        f"{_fmt_var(report.total_variance):>{var_w}}  "
        f"{_fmt_pct(report.total_variance_pct):>{pct_w}}  "
        f"{total_status:>{status_w}}"
    )
    lines.append(total_row)

    return "\n".join(lines)
