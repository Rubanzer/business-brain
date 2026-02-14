"""Heat-by-heat analysis for steel manufacturing.

Pure functions for analyzing heats (batches of molten steel), their chemistry,
weights, grades, and performance. Supports grade-wise breakdown, chemical
composition statistics, and anomaly detection against grade specifications.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HeatAnalysisResult:
    """Result of heat-level analysis."""

    total_heats: int
    total_weight: float
    avg_weight_per_heat: float
    min_weight: float
    max_weight: float
    weight_std: float
    grade_distribution: dict[str, int] | None
    heats_per_period: dict[str, int] | None
    summary: str


@dataclass
class ElementStats:
    """Statistics for a single chemical element across heats."""

    element: str
    mean: float
    std: float
    min: float
    max: float
    cv_pct: float  # coefficient of variation as percentage
    in_spec_pct: float | None  # percentage of heats in spec, if spec provided


@dataclass
class ChemistryResult:
    """Result of chemistry analysis across heats."""

    elements: list[ElementStats]
    heat_count: int
    off_spec_heats: list[str]
    summary: str


@dataclass
class GradeInfo:
    """Analysis for a single steel grade."""

    grade: str
    heat_count: int
    total_weight: float
    avg_weight: float
    share_pct: float
    avg_value: float | None  # average of value_column, if provided


@dataclass
class GradeWiseResult:
    """Result of grade-wise analysis."""

    grades: list[GradeInfo]
    total_weight: float
    dominant_grade: str
    grade_count: int
    summary: str


@dataclass
class GradeAnomaly:
    """A single grade anomaly — chemistry outside spec for a heat."""

    heat: str
    grade: str
    element: str
    value: float
    expected_range: tuple[float, float]
    severity: str  # "low", "medium", "high"


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


def _safe_str(val) -> str | None:
    """Convert a value to str, returning None if the value is None."""
    if val is None:
        return None
    return str(val)


# ---------------------------------------------------------------------------
# 1. analyze_heats
# ---------------------------------------------------------------------------


def analyze_heats(
    rows: list[dict],
    heat_column: str,
    weight_column: str,
    grade_column: str | None = None,
    time_column: str | None = None,
) -> HeatAnalysisResult | None:
    """Analyze heats: count, weight statistics, grade distribution, period breakdown.

    Args:
        rows: Data rows as dicts.
        heat_column: Column identifying the heat (batch ID).
        weight_column: Column with the heat weight.
        grade_column: Optional column with the steel grade.
        time_column: Optional column with time period (e.g. date, shift, month).

    Returns:
        HeatAnalysisResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate weight per heat
    heat_weights: dict[str, float] = {}
    heat_grades: dict[str, str] = {}
    heat_periods: dict[str, str] = {}

    for row in rows:
        heat = _safe_str(row.get(heat_column))
        weight = _safe_float(row.get(weight_column))
        if heat is None or weight is None:
            continue

        heat_weights[heat] = heat_weights.get(heat, 0.0) + weight

        if grade_column is not None:
            grade = _safe_str(row.get(grade_column))
            if grade is not None:
                heat_grades[heat] = grade

        if time_column is not None:
            period = _safe_str(row.get(time_column))
            if period is not None:
                heat_periods[heat] = period

    if not heat_weights:
        return None

    weights = list(heat_weights.values())
    total_heats = len(weights)
    total_weight = sum(weights)
    avg_weight = total_weight / total_heats
    min_weight = min(weights)
    max_weight = max(weights)
    weight_std = statistics.stdev(weights) if total_heats >= 2 else 0.0

    # Grade distribution
    grade_distribution: dict[str, int] | None = None
    if grade_column is not None and heat_grades:
        grade_distribution = {}
        for grade in heat_grades.values():
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

    # Heats per period
    heats_per_period: dict[str, int] | None = None
    if time_column is not None and heat_periods:
        heats_per_period = {}
        for period in heat_periods.values():
            heats_per_period[period] = heats_per_period.get(period, 0) + 1

    summary = (
        f"Heat analysis: {total_heats} heats, total weight {total_weight:,.2f}, "
        f"avg {avg_weight:,.2f} per heat "
        f"(range {min_weight:,.2f} - {max_weight:,.2f}, std {weight_std:,.2f})."
    )
    if grade_distribution:
        dominant = max(grade_distribution, key=grade_distribution.get)  # type: ignore[arg-type]
        summary += f" Dominant grade: {dominant} ({grade_distribution[dominant]} heats)."

    return HeatAnalysisResult(
        total_heats=total_heats,
        total_weight=round(total_weight, 4),
        avg_weight_per_heat=round(avg_weight, 4),
        min_weight=round(min_weight, 4),
        max_weight=round(max_weight, 4),
        weight_std=round(weight_std, 4),
        grade_distribution=grade_distribution,
        heats_per_period=heats_per_period,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. analyze_chemistry
# ---------------------------------------------------------------------------


def analyze_chemistry(
    rows: list[dict],
    heat_column: str,
    element_columns: list[str],
    specs: dict[str, tuple[float, float]] | None = None,
) -> ChemistryResult | None:
    """Analyze chemical composition per heat for specified elements.

    Args:
        rows: Data rows as dicts.
        heat_column: Column identifying the heat.
        element_columns: List of column names for elements (e.g. ["C", "Mn", "Si"]).
        specs: Optional dict mapping element -> (min, max) spec range.

    Returns:
        ChemistryResult or None if no valid data.
    """
    if not rows or not element_columns:
        return None

    # Collect values per heat per element
    # heat -> element -> list of values
    heat_element_vals: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        heat = _safe_str(row.get(heat_column))
        if heat is None:
            continue
        for elem in element_columns:
            val = _safe_float(row.get(elem))
            if val is None:
                continue
            if heat not in heat_element_vals:
                heat_element_vals[heat] = {}
            if elem not in heat_element_vals[heat]:
                heat_element_vals[heat][elem] = []
            heat_element_vals[heat][elem].append(val)

    if not heat_element_vals:
        return None

    # Compute per-heat average for each element, then aggregate
    element_values: dict[str, list[float]] = {}
    for _heat, elems in heat_element_vals.items():
        for elem, vals in elems.items():
            avg = statistics.mean(vals)
            if elem not in element_values:
                element_values[elem] = []
            element_values[elem].append(avg)

    if not element_values:
        return None

    heat_count = len(heat_element_vals)

    # Build ElementStats for each element
    elements: list[ElementStats] = []
    off_spec_heats_set: set[str] = set()

    for elem in element_columns:
        vals = element_values.get(elem)
        if not vals:
            continue

        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        mn = min(vals)
        mx = max(vals)
        cv_pct = (std / mean * 100) if mean != 0 else 0.0

        in_spec_pct: float | None = None
        if specs and elem in specs:
            lo, hi = specs[elem]
            in_count = 0
            # Check each heat's average against spec
            for heat, elems_map in heat_element_vals.items():
                if elem in elems_map:
                    heat_avg = statistics.mean(elems_map[elem])
                    if lo <= heat_avg <= hi:
                        in_count += 1
                    else:
                        off_spec_heats_set.add(heat)
            heats_with_elem = sum(
                1 for h in heat_element_vals if elem in heat_element_vals[h]
            )
            in_spec_pct = (in_count / heats_with_elem * 100) if heats_with_elem > 0 else 0.0
            in_spec_pct = round(in_spec_pct, 2)

        elements.append(
            ElementStats(
                element=elem,
                mean=round(mean, 6),
                std=round(std, 6),
                min=round(mn, 6),
                max=round(mx, 6),
                cv_pct=round(cv_pct, 2),
                in_spec_pct=in_spec_pct,
            )
        )

    if not elements:
        return None

    off_spec_heats = sorted(off_spec_heats_set)

    elem_summaries = ", ".join(
        f"{e.element}(mean={e.mean:.4f}, cv={e.cv_pct:.1f}%)" for e in elements
    )
    summary = (
        f"Chemistry analysis across {heat_count} heats, "
        f"{len(elements)} elements: {elem_summaries}."
    )
    if off_spec_heats:
        summary += f" {len(off_spec_heats)} heat(s) off-spec."

    return ChemistryResult(
        elements=elements,
        heat_count=heat_count,
        off_spec_heats=off_spec_heats,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. grade_wise_analysis
# ---------------------------------------------------------------------------


def grade_wise_analysis(
    rows: list[dict],
    grade_column: str,
    weight_column: str,
    value_column: str | None = None,
) -> GradeWiseResult | None:
    """Analyze production by steel grade.

    Args:
        rows: Data rows as dicts.
        grade_column: Column with the steel grade.
        weight_column: Column with the heat weight.
        value_column: Optional numeric column (e.g. revenue per heat).

    Returns:
        GradeWiseResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate per grade
    grade_acc: dict[str, dict] = {}

    for row in rows:
        grade = _safe_str(row.get(grade_column))
        weight = _safe_float(row.get(weight_column))
        if grade is None or weight is None:
            continue

        if grade not in grade_acc:
            grade_acc[grade] = {"count": 0, "total_weight": 0.0, "values": []}

        grade_acc[grade]["count"] += 1
        grade_acc[grade]["total_weight"] += weight

        if value_column is not None:
            val = _safe_float(row.get(value_column))
            if val is not None:
                grade_acc[grade]["values"].append(val)

    if not grade_acc:
        return None

    total_weight = sum(g["total_weight"] for g in grade_acc.values())

    grades: list[GradeInfo] = []
    for grade_name in sorted(grade_acc.keys()):
        data = grade_acc[grade_name]
        count = data["count"]
        tw = data["total_weight"]
        avg_w = tw / count if count > 0 else 0.0
        share = (tw / total_weight * 100) if total_weight > 0 else 0.0

        avg_value: float | None = None
        if value_column is not None and data["values"]:
            avg_value = round(statistics.mean(data["values"]), 4)

        grades.append(
            GradeInfo(
                grade=grade_name,
                heat_count=count,
                total_weight=round(tw, 4),
                avg_weight=round(avg_w, 4),
                share_pct=round(share, 2),
                avg_value=avg_value,
            )
        )

    # Sort by total weight descending
    grades.sort(key=lambda g: g.total_weight, reverse=True)

    dominant = grades[0].grade
    grade_count = len(grades)

    grade_summary_parts = ", ".join(
        f"{g.grade}({g.heat_count} heats, {g.share_pct:.1f}%)" for g in grades
    )
    summary = (
        f"Grade-wise analysis: {grade_count} grades, total weight {total_weight:,.2f}. "
        f"Dominant grade: {dominant}. Breakdown: {grade_summary_parts}."
    )

    return GradeWiseResult(
        grades=grades,
        total_weight=round(total_weight, 4),
        dominant_grade=dominant,
        grade_count=grade_count,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. detect_grade_anomalies
# ---------------------------------------------------------------------------


def detect_grade_anomalies(
    rows: list[dict],
    heat_column: str,
    grade_column: str,
    element_columns: list[str],
    specs: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> list[GradeAnomaly]:
    """Find heats where chemistry doesn't match the grade specification.

    If specs is not provided, auto-derive specs from the data (mean +/- 2 std
    per grade per element).

    Args:
        rows: Data rows as dicts.
        heat_column: Column identifying the heat.
        grade_column: Column with the steel grade.
        element_columns: List of element column names.
        specs: Optional dict mapping grade -> {element: (min, max)}.

    Returns:
        List of GradeAnomaly instances (may be empty).
    """
    if not rows or not element_columns:
        return []

    # Collect data: heat -> {grade, elements}
    heat_data: dict[str, dict] = {}
    for row in rows:
        heat = _safe_str(row.get(heat_column))
        grade = _safe_str(row.get(grade_column))
        if heat is None or grade is None:
            continue

        if heat not in heat_data:
            heat_data[heat] = {"grade": grade, "elements": {}}

        for elem in element_columns:
            val = _safe_float(row.get(elem))
            if val is not None:
                if elem not in heat_data[heat]["elements"]:
                    heat_data[heat]["elements"][elem] = []
                heat_data[heat]["elements"][elem].append(val)

    if not heat_data:
        return []

    # Derive specs from data if not provided
    effective_specs: dict[str, dict[str, tuple[float, float]]]
    if specs is not None:
        effective_specs = specs
    else:
        # Auto-derive: group by grade, compute mean +/- 2*std for each element
        grade_elem_vals: dict[str, dict[str, list[float]]] = {}
        for hd in heat_data.values():
            grade = hd["grade"]
            if grade not in grade_elem_vals:
                grade_elem_vals[grade] = {}
            for elem, vals in hd["elements"].items():
                if elem not in grade_elem_vals[grade]:
                    grade_elem_vals[grade][elem] = []
                grade_elem_vals[grade][elem].extend(vals)

        effective_specs = {}
        for grade, elems in grade_elem_vals.items():
            effective_specs[grade] = {}
            for elem, vals in elems.items():
                if len(vals) >= 2:
                    mean = statistics.mean(vals)
                    std = statistics.stdev(vals)
                    effective_specs[grade][elem] = (
                        round(mean - 2 * std, 6),
                        round(mean + 2 * std, 6),
                    )
                elif len(vals) == 1:
                    # Single value: use tight tolerance
                    v = vals[0]
                    effective_specs[grade][elem] = (v, v)

    # Check each heat against specs
    anomalies: list[GradeAnomaly] = []
    for heat, hd in sorted(heat_data.items()):
        grade = hd["grade"]
        if grade not in effective_specs:
            continue

        grade_specs = effective_specs[grade]
        for elem in element_columns:
            if elem not in hd["elements"] or elem not in grade_specs:
                continue

            heat_avg = statistics.mean(hd["elements"][elem])
            lo, hi = grade_specs[elem]

            if heat_avg < lo or heat_avg > hi:
                # Compute severity based on how far outside spec
                spec_range = hi - lo
                if spec_range > 0:
                    deviation = max(heat_avg - hi, lo - heat_avg) / spec_range
                else:
                    deviation = abs(heat_avg - lo)

                if deviation > 1.0:
                    severity = "high"
                elif deviation > 0.5:
                    severity = "medium"
                else:
                    severity = "low"

                anomalies.append(
                    GradeAnomaly(
                        heat=heat,
                        grade=grade,
                        element=elem,
                        value=round(heat_avg, 6),
                        expected_range=(lo, hi),
                        severity=severity,
                    )
                )

    return anomalies


# ---------------------------------------------------------------------------
# 5. format_heat_report
# ---------------------------------------------------------------------------


def format_heat_report(
    analysis: HeatAnalysisResult | None = None,
    chemistry: ChemistryResult | None = None,
    grade_wise: GradeWiseResult | None = None,
) -> str:
    """Generate a combined heat analysis report from available analyses.

    Args:
        analysis: Heat analysis result.
        chemistry: Chemistry analysis result.
        grade_wise: Grade-wise analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Heat Analysis Report")
    sections.append("=" * 50)

    if analysis is not None:
        lines = ["", "Heat Overview", "-" * 48]
        lines.append(f"  Total Heats:      {analysis.total_heats}")
        lines.append(f"  Total Weight:     {analysis.total_weight:,.2f}")
        lines.append(f"  Avg Weight/Heat:  {analysis.avg_weight_per_heat:,.2f}")
        lines.append(f"  Min Weight:       {analysis.min_weight:,.2f}")
        lines.append(f"  Max Weight:       {analysis.max_weight:,.2f}")
        lines.append(f"  Weight Std Dev:   {analysis.weight_std:,.2f}")

        if analysis.grade_distribution:
            lines.append("")
            lines.append("  Grade Distribution:")
            for grade, count in sorted(analysis.grade_distribution.items()):
                lines.append(f"    {grade}: {count} heats")

        if analysis.heats_per_period:
            lines.append("")
            lines.append("  Heats per Period:")
            for period, count in sorted(analysis.heats_per_period.items()):
                lines.append(f"    {period}: {count} heats")

        sections.append("\n".join(lines))

    if chemistry is not None:
        lines = ["", "Chemistry Analysis", "-" * 48]
        lines.append(f"  Heats analyzed: {chemistry.heat_count}")
        lines.append("")
        for e in chemistry.elements:
            spec_str = f", in-spec={e.in_spec_pct:.1f}%" if e.in_spec_pct is not None else ""
            lines.append(
                f"  {e.element}: mean={e.mean:.4f}, std={e.std:.4f}, "
                f"range=[{e.min:.4f}, {e.max:.4f}], CV={e.cv_pct:.1f}%{spec_str}"
            )
        if chemistry.off_spec_heats:
            lines.append("")
            lines.append(f"  Off-spec heats ({len(chemistry.off_spec_heats)}): "
                         f"{', '.join(chemistry.off_spec_heats[:10])}")
            if len(chemistry.off_spec_heats) > 10:
                lines.append(f"    ... and {len(chemistry.off_spec_heats) - 10} more")
        sections.append("\n".join(lines))

    if grade_wise is not None:
        lines = ["", "Grade-Wise Analysis", "-" * 48]
        lines.append(f"  Total Weight:   {grade_wise.total_weight:,.2f}")
        lines.append(f"  Dominant Grade: {grade_wise.dominant_grade}")
        lines.append(f"  Grade Count:    {grade_wise.grade_count}")
        lines.append("")
        for g in grade_wise.grades:
            val_str = f", avg_value={g.avg_value:,.2f}" if g.avg_value is not None else ""
            lines.append(
                f"  {g.grade}: {g.heat_count} heats, "
                f"weight={g.total_weight:,.2f} ({g.share_pct:.1f}%), "
                f"avg={g.avg_weight:,.2f}{val_str}"
            )
        sections.append("\n".join(lines))

    if analysis is None and chemistry is None and grade_wise is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
