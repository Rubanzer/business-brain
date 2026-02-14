"""Environmental and emissions monitoring analytics for manufacturing.

Pure functions for emissions tracking, waste generation analysis, water usage
monitoring, compliance scoring, and combined environmental reporting across
manufacturing plants and facilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SourcePollutantSummary:
    """Emissions summary for a single source + pollutant combination."""

    source: str
    pollutant: str
    total: float
    avg: float
    max_value: float
    count: int
    compliance_pct: float | None  # % of readings within limit
    trend: str | None  # "increasing", "decreasing", "stable", or None


@dataclass
class Exceedance:
    """A single emission exceedance event."""

    source: str
    pollutant: str
    value: float
    limit: float
    excess_pct: float  # how much over the limit in %


@dataclass
class EmissionsResult:
    """Complete emissions analysis result."""

    source_summaries: list[SourcePollutantSummary]
    exceedances: list[Exceedance]
    overall_compliance_pct: float
    summary: str


@dataclass
class WasteTypeSummary:
    """Waste summary for a single waste type."""

    waste_type: str
    total_quantity: float
    disposal_breakdown: dict[str, float]  # disposal_method -> quantity


@dataclass
class WasteResult:
    """Complete waste generation analysis result."""

    by_type: list[WasteTypeSummary]
    total_waste: float
    recycling_rate: float  # recycled / total * 100
    diversion_rate: float  # (total - landfill) / total * 100
    summary: str


@dataclass
class WaterSourceSummary:
    """Water usage summary for a single source."""

    source: str
    total_consumption: float
    total_discharge: float | None
    water_balance: float | None  # consumption - discharge


@dataclass
class WaterResult:
    """Complete water usage analysis result."""

    by_source: list[WaterSourceSummary]
    total_consumption: float
    total_discharge: float | None
    overall_balance: float | None
    recycling_ratio: float  # recycled consumption / total consumption * 100
    summary: str


@dataclass
class ComplianceScore:
    """Weighted environmental compliance score."""

    emissions_score: float  # 0-100
    waste_score: float  # 0-100
    water_score: float  # 0-100
    overall_score: float  # weighted 0-100
    rating: str  # "Excellent", "Good", "Fair", "Poor"
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_emissions
# ---------------------------------------------------------------------------


def _detect_emissions_trend(values: list[float]) -> str:
    """Detect trend from a list of values by comparing first and second half averages.

    Returns "increasing", "decreasing", or "stable".
    """
    if len(values) < 4:
        return "stable"

    mid = len(values) // 2
    first_half_avg = sum(values[:mid]) / mid
    second_half_avg = sum(values[mid:]) / (len(values) - mid)

    if first_half_avg == 0:
        return "stable"

    ratio = second_half_avg / first_half_avg
    if ratio > 1.25:
        return "increasing"
    if ratio < 0.75:
        return "decreasing"
    return "stable"


def analyze_emissions(
    rows: list[dict],
    source_column: str,
    pollutant_column: str,
    value_column: str,
    limit_column: str | None = None,
    time_column: str | None = None,
) -> EmissionsResult | None:
    """Analyze emissions data grouped by source and pollutant.

    For each source+pollutant combination computes total, average, max, and
    count of readings. When *limit_column* is provided, computes compliance
    percentage and flags exceedances. When *time_column* is provided, detects
    whether emissions are increasing, decreasing, or stable.

    Args:
        rows: Data rows as dicts.
        source_column: Column identifying the emission source.
        pollutant_column: Column identifying the pollutant type.
        value_column: Numeric column with emission reading values.
        limit_column: Optional column with the regulatory limit for each reading.
        time_column: Optional column with a time/date value for trend detection.

    Returns:
        EmissionsResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate per (source, pollutant)
    acc: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        src = row.get(source_column)
        pol = row.get(pollutant_column)
        val = _safe_float(row.get(value_column))
        if src is None or pol is None or val is None:
            continue
        key = (str(src), str(pol))
        if key not in acc:
            acc[key] = []
        acc[key].append(row)

    if not acc:
        return None

    source_summaries: list[SourcePollutantSummary] = []
    exceedances: list[Exceedance] = []
    total_readings = 0
    compliant_readings = 0

    for (src, pol), group_rows in acc.items():
        values: list[float] = []
        within_limit = 0
        group_count = 0
        time_value_pairs: list[tuple[str, float]] = []

        for row in group_rows:
            val = _safe_float(row.get(value_column))
            if val is None:
                continue
            values.append(val)
            group_count += 1
            total_readings += 1

            # Time column for trend
            if time_column is not None:
                tv = row.get(time_column)
                if tv is not None:
                    time_value_pairs.append((str(tv), val))

            # Limit compliance
            if limit_column is not None:
                lim = _safe_float(row.get(limit_column))
                if lim is not None:
                    if val <= lim:
                        within_limit += 1
                        compliant_readings += 1
                    else:
                        excess_pct = ((val - lim) / lim * 100) if lim > 0 else 0.0
                        exceedances.append(
                            Exceedance(
                                source=src,
                                pollutant=pol,
                                value=round(val, 4),
                                limit=round(lim, 4),
                                excess_pct=round(excess_pct, 2),
                            )
                        )
                else:
                    # No limit available for this reading; treat as compliant
                    compliant_readings += 1
                    within_limit += 1
            else:
                compliant_readings += 1

        if not values:
            continue

        total = sum(values)
        avg = total / len(values)
        max_val = max(values)

        # Compliance pct for this source+pollutant
        compliance_pct: float | None = None
        if limit_column is not None:
            compliance_pct = (within_limit / group_count * 100) if group_count > 0 else 0.0

        # Trend detection
        trend: str | None = None
        if time_column is not None and time_value_pairs:
            time_value_pairs.sort(key=lambda x: x[0])
            ordered_values = [v for _, v in time_value_pairs]
            trend = _detect_emissions_trend(ordered_values)

        source_summaries.append(
            SourcePollutantSummary(
                source=src,
                pollutant=pol,
                total=round(total, 4),
                avg=round(avg, 4),
                max_value=round(max_val, 4),
                count=group_count,
                compliance_pct=round(compliance_pct, 2) if compliance_pct is not None else None,
                trend=trend,
            )
        )

    if not source_summaries:
        return None

    # Sort exceedances by excess_pct descending
    exceedances.sort(key=lambda e: -e.excess_pct)

    overall_compliance = (compliant_readings / total_readings * 100) if total_readings > 0 else 100.0

    summary = (
        f"Emissions analysis: {len(source_summaries)} source-pollutant combinations, "
        f"{total_readings} total readings. "
        f"Overall compliance: {overall_compliance:.1f}%. "
        f"Exceedances: {len(exceedances)}."
    )

    return EmissionsResult(
        source_summaries=source_summaries,
        exceedances=exceedances,
        overall_compliance_pct=round(overall_compliance, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. analyze_waste_generation
# ---------------------------------------------------------------------------


def analyze_waste_generation(
    rows: list[dict],
    waste_type_column: str,
    quantity_column: str,
    disposal_column: str | None = None,
    time_column: str | None = None,
) -> WasteResult | None:
    """Analyze waste generation by type and disposal method.

    Computes total waste by type, disposal breakdown, recycling rate, and
    diversion rate.

    Args:
        rows: Data rows as dicts.
        waste_type_column: Column identifying the waste type.
        quantity_column: Numeric column with waste quantity.
        disposal_column: Optional column identifying the disposal method.
        time_column: Optional column with time (reserved for future use).

    Returns:
        WasteResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate per waste type (and optionally disposal method)
    type_totals: dict[str, float] = {}
    type_disposal: dict[str, dict[str, float]] = {}

    for row in rows:
        wt = row.get(waste_type_column)
        qty = _safe_float(row.get(quantity_column))
        if wt is None or qty is None:
            continue
        key = str(wt)
        type_totals[key] = type_totals.get(key, 0.0) + qty

        if disposal_column is not None:
            disp = row.get(disposal_column)
            if disp is not None:
                disp_key = str(disp).lower()
                if key not in type_disposal:
                    type_disposal[key] = {}
                type_disposal[key][disp_key] = type_disposal[key].get(disp_key, 0.0) + qty

    if not type_totals:
        return None

    total_waste = sum(type_totals.values())
    if total_waste == 0:
        return None

    by_type: list[WasteTypeSummary] = []
    for wt_name in sorted(type_totals.keys()):
        disposal_breakdown = type_disposal.get(wt_name, {})
        by_type.append(
            WasteTypeSummary(
                waste_type=wt_name,
                total_quantity=round(type_totals[wt_name], 4),
                disposal_breakdown={k: round(v, 4) for k, v in disposal_breakdown.items()},
            )
        )

    # Compute recycling and diversion rates across all waste
    total_recycled = 0.0
    total_landfill = 0.0
    for wts in by_type:
        for method, qty in wts.disposal_breakdown.items():
            if "recycl" in method:
                total_recycled += qty
            if "landfill" in method:
                total_landfill += qty

    recycling_rate = (total_recycled / total_waste * 100) if total_waste > 0 else 0.0
    diversion_rate = ((total_waste - total_landfill) / total_waste * 100) if total_waste > 0 else 0.0

    summary = (
        f"Waste analysis: {len(by_type)} waste types, "
        f"total = {total_waste:,.2f}. "
        f"Recycling rate: {recycling_rate:.1f}%. "
        f"Diversion rate: {diversion_rate:.1f}%."
    )

    return WasteResult(
        by_type=by_type,
        total_waste=round(total_waste, 4),
        recycling_rate=round(recycling_rate, 2),
        diversion_rate=round(diversion_rate, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. analyze_water_usage
# ---------------------------------------------------------------------------


def analyze_water_usage(
    rows: list[dict],
    source_column: str,
    consumption_column: str,
    discharge_column: str | None = None,
    time_column: str | None = None,
) -> WaterResult | None:
    """Analyze water usage by source with optional discharge tracking.

    Computes total consumption per source, water balance (consumption minus
    discharge), and recycling ratio when a "recycled" source exists.

    Args:
        rows: Data rows as dicts.
        source_column: Column identifying the water source.
        consumption_column: Numeric column with water consumption.
        discharge_column: Optional numeric column with water discharge.
        time_column: Optional column with time (reserved for future use).

    Returns:
        WaterResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate per source
    src_consumption: dict[str, float] = {}
    src_discharge: dict[str, float] = {}

    for row in rows:
        src = row.get(source_column)
        cons = _safe_float(row.get(consumption_column))
        if src is None or cons is None:
            continue
        key = str(src)
        src_consumption[key] = src_consumption.get(key, 0.0) + cons

        if discharge_column is not None:
            disc = _safe_float(row.get(discharge_column))
            if disc is not None:
                src_discharge[key] = src_discharge.get(key, 0.0) + disc

    if not src_consumption:
        return None

    total_consumption = sum(src_consumption.values())
    if total_consumption == 0:
        return None

    has_discharge = discharge_column is not None and bool(src_discharge)
    total_discharge: float | None = sum(src_discharge.values()) if has_discharge else None

    by_source: list[WaterSourceSummary] = []
    for src_name in sorted(src_consumption.keys()):
        cons_val = src_consumption[src_name]
        disc_val: float | None = None
        balance: float | None = None
        if has_discharge:
            disc_val = src_discharge.get(src_name, 0.0)
            balance = cons_val - disc_val
        by_source.append(
            WaterSourceSummary(
                source=src_name,
                total_consumption=round(cons_val, 4),
                total_discharge=round(disc_val, 4) if disc_val is not None else None,
                water_balance=round(balance, 4) if balance is not None else None,
            )
        )

    overall_balance: float | None = None
    if total_discharge is not None:
        overall_balance = total_consumption - total_discharge

    # Recycling ratio: consumption from "recycled" sources / total
    recycled_consumption = 0.0
    for src_name, cons_val in src_consumption.items():
        if "recycl" in src_name.lower():
            recycled_consumption += cons_val
    recycling_ratio = (recycled_consumption / total_consumption * 100) if total_consumption > 0 else 0.0

    summary = (
        f"Water usage analysis: {len(by_source)} sources, "
        f"total consumption = {total_consumption:,.2f}. "
        f"Recycling ratio: {recycling_ratio:.1f}%."
    )
    if overall_balance is not None:
        summary += f" Water balance (consumption - discharge): {overall_balance:,.2f}."

    return WaterResult(
        by_source=by_source,
        total_consumption=round(total_consumption, 4),
        total_discharge=round(total_discharge, 4) if total_discharge is not None else None,
        overall_balance=round(overall_balance, 4) if overall_balance is not None else None,
        recycling_ratio=round(recycling_ratio, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. compute_compliance_score
# ---------------------------------------------------------------------------


def _rating_from_score(score: float) -> str:
    """Map a 0-100 compliance score to a rating string."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"


def compute_compliance_score(
    emissions: EmissionsResult | None = None,
    waste: WasteResult | None = None,
    water: WaterResult | None = None,
) -> ComplianceScore:
    """Compute a weighted environmental compliance score.

    Weights:
      - Emissions compliance: 40% (based on overall compliance percentage)
      - Waste management: 30% (based on diversion rate)
      - Water efficiency: 30% (based on recycling ratio)

    When a component is None a default score of 50 is used.

    Args:
        emissions: EmissionsResult from analyze_emissions.
        waste: WasteResult from analyze_waste_generation.
        water: WaterResult from analyze_water_usage.

    Returns:
        ComplianceScore with weighted overall score and rating.
    """
    # Emissions score: directly use compliance %
    if emissions is not None:
        emissions_score = emissions.overall_compliance_pct
    else:
        emissions_score = 50.0

    # Waste score: use diversion rate as the score
    if waste is not None:
        waste_score = waste.diversion_rate
    else:
        waste_score = 50.0

    # Water score: use recycling ratio as the score
    if water is not None:
        water_score = water.recycling_ratio
    else:
        water_score = 50.0

    # Clamp all component scores to [0, 100]
    emissions_score = max(0.0, min(100.0, emissions_score))
    waste_score = max(0.0, min(100.0, waste_score))
    water_score = max(0.0, min(100.0, water_score))

    overall = emissions_score * 0.4 + waste_score * 0.3 + water_score * 0.3
    overall = max(0.0, min(100.0, overall))

    rating = _rating_from_score(overall)

    summary = (
        f"Environmental compliance score: {overall:.1f}/100 ({rating}). "
        f"Emissions: {emissions_score:.1f}, "
        f"Waste: {waste_score:.1f}, "
        f"Water: {water_score:.1f}."
    )

    return ComplianceScore(
        emissions_score=round(emissions_score, 2),
        waste_score=round(waste_score, 2),
        water_score=round(water_score, 2),
        overall_score=round(overall, 2),
        rating=rating,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_environmental_report
# ---------------------------------------------------------------------------


def format_environmental_report(
    emissions: EmissionsResult | None = None,
    waste: WasteResult | None = None,
    water: WaterResult | None = None,
    score: ComplianceScore | None = None,
) -> str:
    """Generate a combined environmental monitoring report.

    Args:
        emissions: EmissionsResult from analyze_emissions.
        waste: WasteResult from analyze_waste_generation.
        water: WaterResult from analyze_water_usage.
        score: ComplianceScore from compute_compliance_score.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("=" * 60)
    sections.append("ENVIRONMENTAL MONITORING REPORT")
    sections.append("=" * 60)

    if emissions is not None:
        lines = ["", "EMISSIONS ANALYSIS", "-" * 40]
        lines.append(f"  Overall Compliance: {emissions.overall_compliance_pct:.1f}%")
        lines.append(f"  Total Exceedances:  {len(emissions.exceedances)}")
        if emissions.source_summaries:
            lines.append("")
            lines.append(
                f"  {'Source':<15} {'Pollutant':<15} {'Total':>10} "
                f"{'Avg':>10} {'Max':>10} {'Count':>6}"
            )
            lines.append(
                f"  {'-'*15} {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*6}"
            )
            for s in emissions.source_summaries:
                trend_str = f" [{s.trend}]" if s.trend else ""
                lines.append(
                    f"  {s.source:<15} {s.pollutant:<15} {s.total:>10,.2f} "
                    f"{s.avg:>10,.2f} {s.max_value:>10,.2f} {s.count:>6}{trend_str}"
                )
        if emissions.exceedances:
            lines.append("")
            lines.append("  Top Exceedances:")
            for exc in emissions.exceedances[:5]:
                lines.append(
                    f"    {exc.source} / {exc.pollutant}: "
                    f"value={exc.value:,.2f} limit={exc.limit:,.2f} "
                    f"(+{exc.excess_pct:.1f}%)"
                )
        sections.append("\n".join(lines))

    if waste is not None:
        lines = ["", "WASTE GENERATION", "-" * 40]
        lines.append(f"  Total Waste:     {waste.total_waste:,.2f}")
        lines.append(f"  Recycling Rate:  {waste.recycling_rate:.1f}%")
        lines.append(f"  Diversion Rate:  {waste.diversion_rate:.1f}%")
        if waste.by_type:
            lines.append("")
            for wt in waste.by_type:
                lines.append(f"  {wt.waste_type}: {wt.total_quantity:,.2f}")
                if wt.disposal_breakdown:
                    for method, qty in sorted(wt.disposal_breakdown.items()):
                        lines.append(f"    - {method}: {qty:,.2f}")
        sections.append("\n".join(lines))

    if water is not None:
        lines = ["", "WATER USAGE", "-" * 40]
        lines.append(f"  Total Consumption: {water.total_consumption:,.2f}")
        if water.total_discharge is not None:
            lines.append(f"  Total Discharge:   {water.total_discharge:,.2f}")
        if water.overall_balance is not None:
            lines.append(f"  Water Balance:     {water.overall_balance:,.2f}")
        lines.append(f"  Recycling Ratio:   {water.recycling_ratio:.1f}%")
        if water.by_source:
            lines.append("")
            for ws in water.by_source:
                disc_str = f" discharge={ws.total_discharge:,.2f}" if ws.total_discharge is not None else ""
                bal_str = f" balance={ws.water_balance:,.2f}" if ws.water_balance is not None else ""
                lines.append(
                    f"  {ws.source}: consumption={ws.total_consumption:,.2f}"
                    f"{disc_str}{bal_str}"
                )
        sections.append("\n".join(lines))

    if score is not None:
        lines = ["", "COMPLIANCE SCORE", "-" * 40]
        lines.append(f"  Overall Score:    {score.overall_score:.1f}/100")
        lines.append(f"  Rating:           {score.rating}")
        lines.append(f"  Emissions Score:  {score.emissions_score:.1f}")
        lines.append(f"  Waste Score:      {score.waste_score:.1f}")
        lines.append(f"  Water Score:      {score.water_score:.1f}")
        sections.append("\n".join(lines))

    if emissions is None and waste is None and water is None and score is None:
        sections.append("")
        sections.append("No analysis data provided.")

    sections.append("")
    sections.append("=" * 60)

    return "\n".join(sections)
