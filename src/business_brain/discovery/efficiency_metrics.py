"""Manufacturing efficiency metrics — OEE, yield, energy, and waste analysis.

Pure functions for computing Overall Equipment Effectiveness (OEE), first-pass
yield rates, specific energy consumption, and waste analysis across entities
such as production lines, machines, or plants.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# OEE (Overall Equipment Effectiveness)
# ---------------------------------------------------------------------------


@dataclass
class EntityOEE:
    """OEE breakdown for a single entity."""

    entity: str
    availability: float
    performance: float
    quality: float
    oee: float
    oee_grade: str  # "World Class" (>85%), "Good" (60-85%), "Poor" (<60%)
    limiting_factor: str  # whichever of availability/performance/quality is lowest


@dataclass
class OEEResult:
    """Aggregated OEE result across all entities."""

    entities: list[EntityOEE]
    mean_oee: float
    best_entity: str
    worst_entity: str
    world_class_count: int  # entities with OEE > 85%
    summary: str


def _oee_grade(oee: float) -> str:
    """Classify OEE into a grade bucket."""
    if oee > 0.85:
        return "World Class"
    if oee >= 0.60:
        return "Good"
    return "Poor"


def _limiting_factor(availability: float, performance: float, quality: float) -> str:
    """Return the name of the lowest component."""
    components = {
        "availability": availability,
        "performance": performance,
        "quality": quality,
    }
    return min(components, key=components.get)  # type: ignore[arg-type]


def compute_oee(
    rows: list[dict],
    entity_column: str,
    availability_column: str,
    performance_column: str,
    quality_column: str,
) -> OEEResult | None:
    """Compute OEE per entity from row data.

    Each row should contain fraction values (0-1) for availability, performance,
    and quality.  When an entity appears multiple times, the component values are
    averaged before computing OEE.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (line, machine, etc.).
        availability_column: Column with availability fraction (0-1).
        performance_column: Column with performance fraction (0-1).
        quality_column: Column with quality fraction (0-1).

    Returns:
        OEEResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate per entity
    acc: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        entity = row.get(entity_column)
        avail = row.get(availability_column)
        perf = row.get(performance_column)
        qual = row.get(quality_column)
        if entity is None or avail is None or perf is None or qual is None:
            continue
        try:
            a_val = float(avail)
            p_val = float(perf)
            q_val = float(qual)
        except (TypeError, ValueError):
            continue
        key = str(entity)
        if key not in acc:
            acc[key] = {"a": [], "p": [], "q": []}
        acc[key]["a"].append(a_val)
        acc[key]["p"].append(p_val)
        acc[key]["q"].append(q_val)

    if not acc:
        return None

    entities: list[EntityOEE] = []
    for name, vals in acc.items():
        a_mean = sum(vals["a"]) / len(vals["a"])
        p_mean = sum(vals["p"]) / len(vals["p"])
        q_mean = sum(vals["q"]) / len(vals["q"])
        oee = a_mean * p_mean * q_mean
        entities.append(
            EntityOEE(
                entity=name,
                availability=round(a_mean, 4),
                performance=round(p_mean, 4),
                quality=round(q_mean, 4),
                oee=round(oee, 4),
                oee_grade=_oee_grade(oee),
                limiting_factor=_limiting_factor(a_mean, p_mean, q_mean),
            )
        )

    # Sort by OEE descending
    entities.sort(key=lambda e: e.oee, reverse=True)

    mean_oee = sum(e.oee for e in entities) / len(entities)
    best = entities[0]
    worst = entities[-1]
    wc_count = sum(1 for e in entities if e.oee > 0.85)

    summary = (
        f"OEE across {len(entities)} entities: "
        f"Mean OEE = {mean_oee:.1%}. "
        f"Best = {best.entity} ({best.oee:.1%}), "
        f"Worst = {worst.entity} ({worst.oee:.1%}). "
        f"{wc_count} world-class (>85%)."
    )

    return OEEResult(
        entities=entities,
        mean_oee=round(mean_oee, 4),
        best_entity=best.entity,
        worst_entity=worst.entity,
        world_class_count=wc_count,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Yield Analysis
# ---------------------------------------------------------------------------


@dataclass
class EntityYield:
    """Yield metrics for a single entity."""

    entity: str
    input_total: float
    output_total: float
    yield_pct: float
    defect_rate: float  # 0 when no defect column provided
    waste_pct: float  # 100 - yield_pct


@dataclass
class YieldResult:
    """Aggregated yield analysis."""

    entities: list[EntityYield]
    mean_yield: float
    best_entity: str
    worst_entity: str
    summary: str


def compute_yield_analysis(
    rows: list[dict],
    entity_column: str,
    input_column: str,
    output_column: str,
    defect_column: str | None = None,
) -> YieldResult | None:
    """Compute first-pass yield per entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        input_column: Column with input quantity.
        output_column: Column with output (good) quantity.
        defect_column: Optional column with defect count.

    Returns:
        YieldResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        inp = row.get(input_column)
        out = row.get(output_column)
        if entity is None or inp is None or out is None:
            continue
        try:
            inp_val = float(inp)
            out_val = float(out)
        except (TypeError, ValueError):
            continue

        defect_val = 0.0
        if defect_column is not None:
            raw = row.get(defect_column)
            if raw is not None:
                try:
                    defect_val = float(raw)
                except (TypeError, ValueError):
                    defect_val = 0.0

        key = str(entity)
        if key not in acc:
            acc[key] = {"input": 0.0, "output": 0.0, "defect": 0.0}
        acc[key]["input"] += inp_val
        acc[key]["output"] += out_val
        acc[key]["defect"] += defect_val

    if not acc:
        return None

    entities: list[EntityYield] = []
    for name, vals in acc.items():
        inp_total = vals["input"]
        out_total = vals["output"]
        if inp_total == 0:
            y_pct = 0.0
            d_rate = 0.0
        else:
            y_pct = out_total / inp_total * 100
            d_rate = vals["defect"] / inp_total * 100
        w_pct = 100.0 - y_pct
        entities.append(
            EntityYield(
                entity=name,
                input_total=round(inp_total, 4),
                output_total=round(out_total, 4),
                yield_pct=round(y_pct, 2),
                defect_rate=round(d_rate, 2),
                waste_pct=round(w_pct, 2),
            )
        )

    # Sort by yield descending
    entities.sort(key=lambda e: e.yield_pct, reverse=True)

    mean_yield = sum(e.yield_pct for e in entities) / len(entities)
    best = entities[0]
    worst = entities[-1]

    summary = (
        f"Yield across {len(entities)} entities: "
        f"Mean yield = {mean_yield:.1f}%. "
        f"Best = {best.entity} ({best.yield_pct:.1f}%), "
        f"Worst = {worst.entity} ({worst.yield_pct:.1f}%)."
    )

    return YieldResult(
        entities=entities,
        mean_yield=round(mean_yield, 2),
        best_entity=best.entity,
        worst_entity=worst.entity,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Energy Efficiency
# ---------------------------------------------------------------------------


@dataclass
class EntityEnergy:
    """Energy efficiency for a single entity."""

    entity: str
    total_output: float
    total_energy: float
    specific_energy: float  # kWh per unit
    efficiency_grade: str  # "Excellent", "Good", "Average", "Poor"


@dataclass
class EnergyResult:
    """Aggregated energy efficiency analysis."""

    entities: list[EntityEnergy]
    mean_sec: float  # mean specific energy consumption
    best_entity: str
    worst_entity: str
    potential_savings_pct: float  # how much worst could save to match best
    summary: str


def _energy_grade(sec: float, best_sec: float) -> str:
    """Grade energy efficiency relative to best performer.

    Within 10% of best => Excellent, within 25% => Good,
    within 50% => Average, else Poor.
    """
    if best_sec == 0:
        return "Average"
    ratio = sec / best_sec
    if ratio <= 1.10:
        return "Excellent"
    if ratio <= 1.25:
        return "Good"
    if ratio <= 1.50:
        return "Average"
    return "Poor"


def compute_energy_efficiency(
    rows: list[dict],
    entity_column: str,
    output_column: str,
    energy_column: str,
) -> EnergyResult | None:
    """Compute specific energy consumption (kWh per unit) per entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        output_column: Column with production output.
        energy_column: Column with energy consumed (kWh).

    Returns:
        EnergyResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        output = row.get(output_column)
        energy = row.get(energy_column)
        if entity is None or output is None or energy is None:
            continue
        try:
            o_val = float(output)
            e_val = float(energy)
        except (TypeError, ValueError):
            continue
        key = str(entity)
        if key not in acc:
            acc[key] = {"output": 0.0, "energy": 0.0}
        acc[key]["output"] += o_val
        acc[key]["energy"] += e_val

    if not acc:
        return None

    # First pass: compute SEC values
    raw: list[tuple[str, float, float, float]] = []  # (name, output, energy, sec)
    for name, vals in acc.items():
        if vals["output"] == 0:
            sec = float("inf")
        else:
            sec = vals["energy"] / vals["output"]
        raw.append((name, vals["output"], vals["energy"], sec))

    # Determine best SEC (lowest finite)
    finite_secs = [s for (_, _, _, s) in raw if s != float("inf")]
    best_sec = min(finite_secs) if finite_secs else 0

    entities: list[EntityEnergy] = []
    for name, output, energy, sec in raw:
        entities.append(
            EntityEnergy(
                entity=name,
                total_output=round(output, 4),
                total_energy=round(energy, 4),
                specific_energy=round(sec, 4) if sec != float("inf") else float("inf"),
                efficiency_grade=_energy_grade(sec, best_sec) if sec != float("inf") else "Poor",
            )
        )

    # Sort by SEC ascending (lower is better)
    entities.sort(key=lambda e: e.specific_energy)

    mean_sec = sum(e.specific_energy for e in entities if e.specific_energy != float("inf"))
    finite_count = sum(1 for e in entities if e.specific_energy != float("inf"))
    mean_sec = mean_sec / finite_count if finite_count else 0

    best = entities[0]
    worst = entities[-1]

    # Potential savings: if worst matched best
    if worst.specific_energy != float("inf") and worst.specific_energy > 0:
        potential_savings = (1 - best.specific_energy / worst.specific_energy) * 100
    else:
        potential_savings = 0.0

    summary = (
        f"Energy efficiency across {len(entities)} entities: "
        f"Mean SEC = {mean_sec:.2f} kWh/unit. "
        f"Best = {best.entity} ({best.specific_energy:.2f}), "
        f"Worst = {worst.entity} "
        f"({worst.specific_energy:.2f} kWh/unit"
        f"{'' if worst.specific_energy == float('inf') else f', {potential_savings:.1f}% savings potential'})."
    )

    return EnergyResult(
        entities=entities,
        mean_sec=round(mean_sec, 4),
        best_entity=best.entity,
        worst_entity=worst.entity,
        potential_savings_pct=round(potential_savings, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Waste Analysis
# ---------------------------------------------------------------------------


@dataclass
class EntityWaste:
    """Waste analysis for a single entity."""

    entity: str
    total_production: float
    total_waste: float
    waste_pct: float
    rank: int


@dataclass
class WasteResult:
    """Aggregated waste analysis."""

    entities: list[EntityWaste]
    total_waste: float
    total_production: float
    waste_pct: float
    worst_wasters: list[str]  # top 3 entities by waste %
    summary: str


def compute_waste_analysis(
    rows: list[dict],
    entity_column: str,
    total_column: str,
    waste_column: str,
) -> WasteResult | None:
    """Compute waste percentage and ranking per entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        total_column: Column with total production quantity.
        waste_column: Column with waste quantity.

    Returns:
        WasteResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        total = row.get(total_column)
        waste = row.get(waste_column)
        if entity is None or total is None or waste is None:
            continue
        try:
            t_val = float(total)
            w_val = float(waste)
        except (TypeError, ValueError):
            continue
        key = str(entity)
        if key not in acc:
            acc[key] = {"total": 0.0, "waste": 0.0}
        acc[key]["total"] += t_val
        acc[key]["waste"] += w_val

    if not acc:
        return None

    entities: list[EntityWaste] = []
    for name, vals in acc.items():
        t = vals["total"]
        w = vals["waste"]
        pct = (w / t * 100) if t != 0 else 0.0
        entities.append(
            EntityWaste(
                entity=name,
                total_production=round(t, 4),
                total_waste=round(w, 4),
                waste_pct=round(pct, 2),
                rank=0,  # assigned below
            )
        )

    # Sort by waste_pct descending (worst first)
    entities.sort(key=lambda e: e.waste_pct, reverse=True)
    for i, e in enumerate(entities):
        e.rank = i + 1

    total_production = sum(e.total_production for e in entities)
    total_waste = sum(e.total_waste for e in entities)
    overall_pct = (total_waste / total_production * 100) if total_production != 0 else 0.0

    worst_wasters = [e.entity for e in entities[:3]]

    summary = (
        f"Waste across {len(entities)} entities: "
        f"Total waste = {total_waste:,.2f} / {total_production:,.2f} ({overall_pct:.1f}%). "
        f"Worst wasters: {', '.join(worst_wasters)}."
    )

    return WasteResult(
        entities=entities,
        total_waste=round(total_waste, 4),
        total_production=round(total_production, 4),
        waste_pct=round(overall_pct, 2),
        worst_wasters=worst_wasters,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Combined Efficiency Report
# ---------------------------------------------------------------------------


def efficiency_report(
    oee: OEEResult | None = None,
    yield_result: YieldResult | None = None,
    energy: EnergyResult | None = None,
    waste: WasteResult | None = None,
) -> str:
    """Generate a combined text report from available efficiency analyses.

    Args:
        oee: OEE analysis result.
        yield_result: Yield analysis result.
        energy: Energy efficiency result.
        waste: Waste analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Manufacturing Efficiency Report")
    sections.append("=" * 40)

    if oee is not None:
        lines = ["", "OEE (Overall Equipment Effectiveness)", "-" * 38]
        for e in oee.entities:
            lines.append(
                f"  {e.entity}: OEE={e.oee:.1%} "
                f"(A={e.availability:.1%} P={e.performance:.1%} Q={e.quality:.1%}) "
                f"[{e.oee_grade}] limiting: {e.limiting_factor}"
            )
        lines.append(f"  Mean OEE: {oee.mean_oee:.1%} | World-class: {oee.world_class_count}")
        sections.append("\n".join(lines))

    if yield_result is not None:
        lines = ["", "Yield Analysis", "-" * 38]
        for e in yield_result.entities:
            defect_info = f" defect_rate={e.defect_rate:.1f}%" if e.defect_rate > 0 else ""
            lines.append(
                f"  {e.entity}: yield={e.yield_pct:.1f}% waste={e.waste_pct:.1f}%{defect_info}"
            )
        lines.append(f"  Mean yield: {yield_result.mean_yield:.1f}%")
        sections.append("\n".join(lines))

    if energy is not None:
        lines = ["", "Energy Efficiency", "-" * 38]
        for e in energy.entities:
            sec_str = f"{e.specific_energy:.2f}" if e.specific_energy != float("inf") else "inf"
            lines.append(
                f"  {e.entity}: SEC={sec_str} kWh/unit [{e.efficiency_grade}]"
            )
        lines.append(
            f"  Mean SEC: {energy.mean_sec:.2f} kWh/unit | "
            f"Savings potential: {energy.potential_savings_pct:.1f}%"
        )
        sections.append("\n".join(lines))

    if waste is not None:
        lines = ["", "Waste Analysis", "-" * 38]
        for e in waste.entities:
            lines.append(
                f"  #{e.rank} {e.entity}: waste={e.waste_pct:.1f}% "
                f"({e.total_waste:,.2f}/{e.total_production:,.2f})"
            )
        lines.append(f"  Overall waste: {waste.waste_pct:.1f}%")
        sections.append("\n".join(lines))

    if oee is None and yield_result is None and energy is None and waste is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
