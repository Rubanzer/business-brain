"""Quality control analytics — SPC, capability indices, defect and rejection tracking.

Pure functions for computing Statistical Process Control (SPC) metrics,
Cp/Cpk process capability indices, defect analysis, rejection tracking,
and grade distribution for manufacturing quality management.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CapabilityResult:
    """Process capability analysis result."""

    cp: float
    cpk: float
    mean: float
    std: float
    lsl: float
    usl: float
    ppm_out_of_spec: float
    process_grade: str  # "Excellent", "Good", "Adequate", "Poor"
    centered: bool  # True if abs(cp - cpk) < 0.1
    summary: str


@dataclass
class ControlChartResult:
    """Control chart (X-bar) analysis result."""

    values: list[float]
    mean: float
    ucl: float
    lcl: float
    out_of_control_indices: list[int]
    out_of_control_count: int
    in_control_pct: float
    summary: str


@dataclass
class EntityDefect:
    """Defect metrics for a single entity."""

    entity: str
    defect_count: int
    quantity: int
    defect_rate: float
    dpmo: float  # defects per million opportunities; 0.0 if quantity not given


@dataclass
class DefectResult:
    """Aggregated defect analysis across entities."""

    entities: list[EntityDefect]
    total_defects: int
    total_quantity: int
    overall_defect_rate: float
    worst_entity: str
    best_entity: str
    summary: str


@dataclass
class EntityRejection:
    """Rejection metrics for a single entity."""

    entity: str
    accepted: int
    rejected: int
    total: int
    rejection_rate: float


@dataclass
class RejectionResult:
    """Aggregated rejection analysis across entities."""

    entities: list[EntityRejection]
    total_accepted: int
    total_rejected: int
    overall_rejection_rate: float
    worst_entity: str
    best_entity: str
    summary: str


@dataclass
class EntityGrade:
    """Grade distribution for a single entity."""

    entity: str
    grades: dict[str, int]  # grade -> count
    primary_grade: str
    total_items: int


@dataclass
class GradeResult:
    """Aggregated grade distribution analysis."""

    grade_distribution: dict[str, int]  # overall grade -> count
    entity_grades: list[EntityGrade]
    most_common_grade: str
    summary: str


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


def _safe_int(val) -> int | None:
    """Convert a value to int (via float), returning None on failure."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _process_grade(cpk: float) -> str:
    """Classify process capability into a grade."""
    if cpk >= 1.67:
        return "Excellent"
    if cpk >= 1.33:
        return "Good"
    if cpk >= 1.0:
        return "Adequate"
    return "Poor"


def _estimate_ppm(mean: float, std: float, lsl: float, usl: float) -> float:
    """Estimate parts per million out of spec using normal approximation.

    Uses the complementary error function from the math module.
    """
    if std <= 0:
        # Zero variation: all values equal the mean
        if lsl <= mean <= usl:
            return 0.0
        return 1_000_000.0

    # Z-scores
    z_upper = (usl - mean) / std
    z_lower = (mean - lsl) / std

    # P(out of spec) = P(X > USL) + P(X < LSL)
    # Using erfc: P(X > x) = 0.5 * erfc(z / sqrt(2))
    p_above = 0.5 * math.erfc(z_upper / math.sqrt(2))
    p_below = 0.5 * math.erfc(z_lower / math.sqrt(2))

    ppm = (p_above + p_below) * 1_000_000
    return round(ppm, 2)


# ---------------------------------------------------------------------------
# 1. compute_process_capability
# ---------------------------------------------------------------------------


def compute_process_capability(
    values: list[float | int | None],
    lsl: float,
    usl: float,
) -> CapabilityResult | None:
    """Compute Cp and Cpk process capability indices.

    Args:
        values: Raw measurement values (None values are filtered out).
        lsl: Lower specification limit.
        usl: Upper specification limit.

    Returns:
        CapabilityResult or None if fewer than 2 valid values.
    """
    # Filter to valid numeric values
    clean: list[float] = []
    for v in values:
        f = _safe_float(v)
        if f is not None:
            clean.append(f)

    if len(clean) < 2:
        return None

    mean = statistics.mean(clean)
    std = statistics.stdev(clean)

    if std == 0:
        # All values identical — process has zero variation
        # Capability is infinite in theory; cap at a large value for grading
        if lsl <= mean <= usl:
            cp = 999.99
            cpk = 999.99
        else:
            cp = 0.0
            cpk = 0.0
    else:
        spec_range = usl - lsl
        cp = spec_range / (6 * std)
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)

    ppm = _estimate_ppm(mean, std, lsl, usl)
    grade = _process_grade(cpk)
    centered = abs(cp - cpk) < 0.1

    summary = (
        f"Process capability: Cp={cp:.3f}, Cpk={cpk:.3f} ({grade}). "
        f"Mean={mean:.4f}, Std={std:.4f}. "
        f"Spec limits [{lsl}, {usl}]. "
        f"Estimated {ppm:.0f} PPM out of spec. "
        f"Process is {'centered' if centered else 'off-center'}."
    )

    return CapabilityResult(
        cp=round(cp, 4),
        cpk=round(cpk, 4),
        mean=round(mean, 4),
        std=round(std, 4),
        lsl=lsl,
        usl=usl,
        ppm_out_of_spec=ppm,
        process_grade=grade,
        centered=centered,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. control_chart_data
# ---------------------------------------------------------------------------


def control_chart_data(
    values: list[float | int | None],
    subgroup_size: int = 1,
) -> ControlChartResult | None:
    """Compute X-bar control chart data with 3-sigma limits.

    When subgroup_size > 1, values are averaged into subgroups before
    computing control limits.

    Args:
        values: Raw measurement values (None values are filtered out).
        subgroup_size: Number of consecutive values per subgroup (default 1).

    Returns:
        ControlChartResult or None if fewer than 2 valid values.
    """
    # Filter to valid numeric values
    clean: list[float] = []
    for v in values:
        f = _safe_float(v)
        if f is not None:
            clean.append(f)

    if len(clean) < 2:
        return None

    if subgroup_size < 1:
        subgroup_size = 1

    # Build subgroup means
    if subgroup_size == 1:
        chart_values = list(clean)
    else:
        chart_values = []
        for i in range(0, len(clean), subgroup_size):
            group = clean[i : i + subgroup_size]
            chart_values.append(statistics.mean(group))

    if len(chart_values) < 2:
        return None

    mean = statistics.mean(chart_values)
    std = statistics.stdev(chart_values)

    if std == 0:
        ucl = mean
        lcl = mean
    else:
        ucl = mean + 3 * std
        lcl = mean - 3 * std

    out_of_control: list[int] = []
    for i, v in enumerate(chart_values):
        if v > ucl or v < lcl:
            out_of_control.append(i)

    ooc_count = len(out_of_control)
    total = len(chart_values)
    in_control_pct = ((total - ooc_count) / total) * 100

    summary = (
        f"Control chart: {total} points, mean={mean:.4f}, "
        f"UCL={ucl:.4f}, LCL={lcl:.4f}. "
        f"{ooc_count} out-of-control ({100 - in_control_pct:.1f}%), "
        f"{in_control_pct:.1f}% in control."
    )

    return ControlChartResult(
        values=[round(v, 4) for v in chart_values],
        mean=round(mean, 4),
        ucl=round(ucl, 4),
        lcl=round(lcl, 4),
        out_of_control_indices=out_of_control,
        out_of_control_count=ooc_count,
        in_control_pct=round(in_control_pct, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. analyze_defects
# ---------------------------------------------------------------------------


def analyze_defects(
    rows: list[dict],
    entity_column: str,
    defect_column: str,
    quantity_column: str | None = None,
) -> DefectResult | None:
    """Analyze defect counts and rates per entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (line, machine, etc.).
        defect_column: Column with defect count.
        quantity_column: Optional column with total quantity inspected.

    Returns:
        DefectResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate per entity
    acc: dict[str, dict[str, int]] = {}
    for row in rows:
        entity = row.get(entity_column)
        defects = _safe_int(row.get(defect_column))
        if entity is None or defects is None:
            continue

        qty = 0
        if quantity_column is not None:
            q = _safe_int(row.get(quantity_column))
            if q is not None:
                qty = q

        key = str(entity)
        if key not in acc:
            acc[key] = {"defects": 0, "quantity": 0}
        acc[key]["defects"] += defects
        acc[key]["quantity"] += qty

    if not acc:
        return None

    has_quantity = quantity_column is not None

    entities: list[EntityDefect] = []
    for name in sorted(acc.keys()):
        d = acc[name]["defects"]
        q = acc[name]["quantity"]
        if has_quantity and q > 0:
            rate = d / q * 100
            dpmo = d / q * 1_000_000
        else:
            rate = 0.0
            dpmo = 0.0

        entities.append(EntityDefect(
            entity=name,
            defect_count=d,
            quantity=q,
            defect_rate=round(rate, 4),
            dpmo=round(dpmo, 2),
        ))

    total_defects = sum(e.defect_count for e in entities)
    total_quantity = sum(e.quantity for e in entities)

    if has_quantity and total_quantity > 0:
        overall_rate = total_defects / total_quantity * 100
    else:
        overall_rate = 0.0

    # Worst = highest defect rate (or highest count if no quantity)
    if has_quantity and total_quantity > 0:
        entities_sorted = sorted(entities, key=lambda e: e.defect_rate, reverse=True)
    else:
        entities_sorted = sorted(entities, key=lambda e: e.defect_count, reverse=True)

    worst = entities_sorted[0].entity
    best = entities_sorted[-1].entity

    summary = (
        f"Defect analysis across {len(entities)} entities: "
        f"{total_defects} total defects"
    )
    if has_quantity and total_quantity > 0:
        summary += f" out of {total_quantity} units ({overall_rate:.2f}% defect rate)"
    summary += f". Worst: {worst}, Best: {best}."

    return DefectResult(
        entities=entities,
        total_defects=total_defects,
        total_quantity=total_quantity,
        overall_defect_rate=round(overall_rate, 4),
        worst_entity=worst,
        best_entity=best,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. analyze_rejections
# ---------------------------------------------------------------------------


def analyze_rejections(
    rows: list[dict],
    entity_column: str,
    accepted_column: str,
    rejected_column: str,
) -> RejectionResult | None:
    """Analyze rejection rates per entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        accepted_column: Column with accepted count.
        rejected_column: Column with rejected count.

    Returns:
        RejectionResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, int]] = {}
    for row in rows:
        entity = row.get(entity_column)
        accepted = _safe_int(row.get(accepted_column))
        rejected = _safe_int(row.get(rejected_column))
        if entity is None or accepted is None or rejected is None:
            continue

        key = str(entity)
        if key not in acc:
            acc[key] = {"accepted": 0, "rejected": 0}
        acc[key]["accepted"] += accepted
        acc[key]["rejected"] += rejected

    if not acc:
        return None

    entities: list[EntityRejection] = []
    for name in sorted(acc.keys()):
        a = acc[name]["accepted"]
        r = acc[name]["rejected"]
        total = a + r
        rate = (r / total * 100) if total > 0 else 0.0
        entities.append(EntityRejection(
            entity=name,
            accepted=a,
            rejected=r,
            total=total,
            rejection_rate=round(rate, 4),
        ))

    total_accepted = sum(e.accepted for e in entities)
    total_rejected = sum(e.rejected for e in entities)
    grand_total = total_accepted + total_rejected
    overall_rate = (total_rejected / grand_total * 100) if grand_total > 0 else 0.0

    entities_sorted = sorted(entities, key=lambda e: e.rejection_rate, reverse=True)
    worst = entities_sorted[0].entity
    best = entities_sorted[-1].entity

    summary = (
        f"Rejection analysis across {len(entities)} entities: "
        f"{total_rejected} rejected out of {grand_total} total "
        f"({overall_rate:.2f}% rejection rate). "
        f"Worst: {worst}, Best: {best}."
    )

    return RejectionResult(
        entities=entities,
        total_accepted=total_accepted,
        total_rejected=total_rejected,
        overall_rejection_rate=round(overall_rate, 4),
        worst_entity=worst,
        best_entity=best,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. grade_analysis
# ---------------------------------------------------------------------------


def grade_analysis(
    rows: list[dict],
    entity_column: str,
    grade_column: str,
    value_column: str | None = None,
) -> GradeResult | None:
    """Analyze distribution of quality grades per entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        grade_column: Column with the quality grade (e.g. "A", "B", "C").
        value_column: Optional numeric column (unused in aggregation but
            kept for API compatibility; reserved for weighted grading).

    Returns:
        GradeResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate grades per entity and overall
    entity_acc: dict[str, dict[str, int]] = {}
    overall_dist: dict[str, int] = {}

    for row in rows:
        entity = row.get(entity_column)
        grade = row.get(grade_column)
        if entity is None or grade is None:
            continue
        key = str(entity)
        g = str(grade)

        if key not in entity_acc:
            entity_acc[key] = {}
        entity_acc[key][g] = entity_acc[key].get(g, 0) + 1
        overall_dist[g] = overall_dist.get(g, 0) + 1

    if not entity_acc:
        return None

    entity_grades: list[EntityGrade] = []
    for name in sorted(entity_acc.keys()):
        grades_map = entity_acc[name]
        total = sum(grades_map.values())
        primary = max(grades_map, key=grades_map.get)  # type: ignore[arg-type]
        entity_grades.append(EntityGrade(
            entity=name,
            grades=dict(sorted(grades_map.items())),
            primary_grade=primary,
            total_items=total,
        ))

    most_common = max(overall_dist, key=overall_dist.get)  # type: ignore[arg-type]

    total_items = sum(overall_dist.values())
    summary = (
        f"Grade analysis across {len(entity_grades)} entities, "
        f"{total_items} total items. "
        f"Most common grade: {most_common} "
        f"({overall_dist[most_common]} items, "
        f"{overall_dist[most_common] / total_items * 100:.1f}%)."
    )

    return GradeResult(
        grade_distribution=dict(sorted(overall_dist.items())),
        entity_grades=entity_grades,
        most_common_grade=most_common,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 6. format_quality_report
# ---------------------------------------------------------------------------


def format_quality_report(
    capability: CapabilityResult | None = None,
    defects: DefectResult | None = None,
    rejections: RejectionResult | None = None,
    grades: GradeResult | None = None,
) -> str:
    """Generate a combined quality control report from available analyses.

    Args:
        capability: Process capability result.
        defects: Defect analysis result.
        rejections: Rejection analysis result.
        grades: Grade analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Quality Control Report")
    sections.append("=" * 50)

    if capability is not None:
        lines = ["", "Process Capability", "-" * 48]
        lines.append(f"  Cp:   {capability.cp:.4f}")
        lines.append(f"  Cpk:  {capability.cpk:.4f}")
        lines.append(f"  Mean: {capability.mean:.4f}")
        lines.append(f"  Std:  {capability.std:.4f}")
        lines.append(f"  Spec: [{capability.lsl}, {capability.usl}]")
        lines.append(f"  PPM out of spec: {capability.ppm_out_of_spec:.0f}")
        lines.append(f"  Grade: {capability.process_grade}")
        lines.append(f"  Centered: {'Yes' if capability.centered else 'No'}")
        sections.append("\n".join(lines))

    if defects is not None:
        lines = ["", "Defect Analysis", "-" * 48]
        for e in defects.entities:
            dpmo_str = f" DPMO={e.dpmo:.0f}" if e.dpmo > 0 else ""
            lines.append(
                f"  {e.entity}: {e.defect_count} defects"
                f" / {e.quantity} units"
                f" ({e.defect_rate:.2f}%){dpmo_str}"
            )
        lines.append(
            f"  Total: {defects.total_defects} defects, "
            f"{defects.overall_defect_rate:.2f}% rate"
        )
        lines.append(f"  Worst: {defects.worst_entity} | Best: {defects.best_entity}")
        sections.append("\n".join(lines))

    if rejections is not None:
        lines = ["", "Rejection Analysis", "-" * 48]
        for e in rejections.entities:
            lines.append(
                f"  {e.entity}: {e.rejected}/{e.total} rejected "
                f"({e.rejection_rate:.2f}%)"
            )
        lines.append(
            f"  Total: {rejections.total_rejected} rejected, "
            f"{rejections.overall_rejection_rate:.2f}% rate"
        )
        lines.append(
            f"  Worst: {rejections.worst_entity} | Best: {rejections.best_entity}"
        )
        sections.append("\n".join(lines))

    if grades is not None:
        lines = ["", "Grade Distribution", "-" * 48]
        for g, count in grades.grade_distribution.items():
            lines.append(f"  Grade {g}: {count}")
        lines.append(f"  Most common: {grades.most_common_grade}")
        for eg in grades.entity_grades:
            grade_parts = ", ".join(f"{g}:{c}" for g, c in eg.grades.items())
            lines.append(f"  {eg.entity}: [{grade_parts}] primary={eg.primary_grade}")
        sections.append("\n".join(lines))

    if capability is None and defects is None and rejections is None and grades is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
