"""Power and energy monitoring analytics for manufacturing.

Pure functions for load profile analysis, power factor evaluation,
specific energy consumption, peak demand detection, and energy cost
optimization across production entities such as lines, plants, or machines.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PeriodLoad:
    """Load data for a single time period."""

    period: str
    demand: float
    pct_of_peak: float
    classification: str  # "peak", "shoulder", or "off_peak"


@dataclass
class LoadProfileResult:
    """Complete load profile analysis result."""

    periods: list[PeriodLoad]
    peak_demand: float
    avg_demand: float
    min_demand: float
    load_factor: float  # average_demand / peak_demand * 100
    peak_period: str
    off_peak_period: str
    summary: str


@dataclass
class EntityPF:
    """Power factor analysis for a single entity."""

    entity: str
    avg_kw: float
    avg_kva: float
    power_factor: float
    status: str  # "excellent" (>0.95), "good" (0.9-0.95), "poor" (<0.9)
    estimated_loss_pct: float


@dataclass
class PowerFactorResult:
    """Aggregated power factor analysis result."""

    entities: list[EntityPF]
    mean_pf: float
    penalty_risk_count: int  # entities with PF < 0.9
    excellent_count: int  # entities with PF > 0.95
    summary: str


@dataclass
class EntitySEC:
    """Specific energy consumption for a single entity."""

    entity: str
    total_energy: float
    total_output: float
    sec: float  # kWh per unit
    deviation_from_best_pct: float


@dataclass
class SpecificEnergyResult:
    """Aggregated specific energy consumption analysis."""

    entities: list[EntitySEC]
    mean_sec: float
    best_sec: float
    worst_sec: float
    best_entity: str
    worst_entity: str
    potential_savings: float  # percentage savings if all matched best
    summary: str


@dataclass
class DemandPeak:
    """A detected demand peak event."""

    period: str
    demand: float
    pct_of_max: float
    duration_periods: int  # consecutive periods above threshold


@dataclass
class EnergyCostResult:
    """Energy cost analysis result."""

    total_energy: float
    total_cost: float
    peak_energy: float
    offpeak_energy: float
    peak_cost: float
    offpeak_cost: float
    avg_rate: float
    potential_shift_savings: float
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_load_profile
# ---------------------------------------------------------------------------


def analyze_load_profile(
    rows: list[dict],
    time_column: str,
    power_column: str,
    entity_column: str | None = None,
) -> LoadProfileResult:
    """Analyse load profile: peak, off-peak, average, and load factor.

    Args:
        rows: Data rows as dicts.
        time_column: Column identifying the time period.
        power_column: Numeric column with power/demand values.
        entity_column: Optional column to filter or group by entity.

    Returns:
        LoadProfileResult with per-period stats and overall metrics.
    """
    if not rows:
        return LoadProfileResult(
            periods=[],
            peak_demand=0.0,
            avg_demand=0.0,
            min_demand=0.0,
            load_factor=0.0,
            peak_period="",
            off_peak_period="",
            summary="No load data available.",
        )

    # Aggregate demand per time period (sum across entities if present)
    period_demand: dict[str, float] = {}
    period_count: dict[str, int] = {}
    for row in rows:
        period = row.get(time_column)
        power = row.get(power_column)
        if period is None or power is None:
            continue
        try:
            p_val = float(power)
        except (TypeError, ValueError):
            continue
        key = str(period)
        period_demand[key] = period_demand.get(key, 0.0) + p_val
        period_count[key] = period_count.get(key, 0) + 1

    if not period_demand:
        return LoadProfileResult(
            periods=[],
            peak_demand=0.0,
            avg_demand=0.0,
            min_demand=0.0,
            load_factor=0.0,
            peak_period="",
            off_peak_period="",
            summary="No valid load records found.",
        )

    max_demand = max(period_demand.values())
    min_demand = min(period_demand.values())
    avg_demand = sum(period_demand.values()) / len(period_demand)
    load_factor = (avg_demand / max_demand * 100) if max_demand > 0 else 0.0

    peak_period_key = max(period_demand, key=period_demand.get)  # type: ignore[arg-type]
    off_peak_period_key = min(period_demand, key=period_demand.get)  # type: ignore[arg-type]

    # Build per-period results
    periods: list[PeriodLoad] = []
    for key in period_demand:
        demand = period_demand[key]
        pct = (demand / max_demand * 100) if max_demand > 0 else 0.0
        classification = _classify_period(pct)
        periods.append(
            PeriodLoad(
                period=key,
                demand=round(demand, 4),
                pct_of_peak=round(pct, 2),
                classification=classification,
            )
        )

    # Sort periods by demand descending
    periods.sort(key=lambda p: -p.demand)

    peak_count = sum(1 for p in periods if p.classification == "peak")
    shoulder_count = sum(1 for p in periods if p.classification == "shoulder")
    off_peak_count = sum(1 for p in periods if p.classification == "off_peak")

    summary = (
        f"Load profile across {len(periods)} periods: "
        f"Peak demand = {max_demand:,.2f}, Avg = {avg_demand:,.2f}, "
        f"Min = {min_demand:,.2f}. "
        f"Load factor = {load_factor:.1f}%. "
        f"Periods: {peak_count} peak, {shoulder_count} shoulder, {off_peak_count} off-peak."
    )

    return LoadProfileResult(
        periods=periods,
        peak_demand=round(max_demand, 4),
        avg_demand=round(avg_demand, 4),
        min_demand=round(min_demand, 4),
        load_factor=round(load_factor, 2),
        peak_period=peak_period_key,
        off_peak_period=off_peak_period_key,
        summary=summary,
    )


def _classify_period(pct_of_peak: float) -> str:
    """Classify a period based on its percentage of peak demand.

    >80% => peak, 50-80% => shoulder, <50% => off_peak.
    """
    if pct_of_peak > 80:
        return "peak"
    if pct_of_peak >= 50:
        return "shoulder"
    return "off_peak"


# ---------------------------------------------------------------------------
# 2. analyze_power_factor
# ---------------------------------------------------------------------------


def analyze_power_factor(
    rows: list[dict],
    entity_column: str,
    kw_column: str,
    kva_column: str,
) -> PowerFactorResult:
    """Analyse power factor per entity.

    Power factor = kW / kVA. A low power factor leads to penalties and
    increased losses.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        kw_column: Column with real power (kW) values.
        kva_column: Column with apparent power (kVA) values.

    Returns:
        PowerFactorResult with per-entity stats and risk counts.
    """
    if not rows:
        return PowerFactorResult(
            entities=[],
            mean_pf=0.0,
            penalty_risk_count=0,
            excellent_count=0,
            summary="No power factor data available.",
        )

    # Accumulate per entity
    acc: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        entity = row.get(entity_column)
        kw = row.get(kw_column)
        kva = row.get(kva_column)
        if entity is None or kw is None or kva is None:
            continue
        try:
            kw_val = float(kw)
            kva_val = float(kva)
        except (TypeError, ValueError):
            continue
        key = str(entity)
        if key not in acc:
            acc[key] = {"kw": [], "kva": []}
        acc[key]["kw"].append(kw_val)
        acc[key]["kva"].append(kva_val)

    if not acc:
        return PowerFactorResult(
            entities=[],
            mean_pf=0.0,
            penalty_risk_count=0,
            excellent_count=0,
            summary="No valid power factor records found.",
        )

    entities: list[EntityPF] = []
    for name, vals in acc.items():
        avg_kw = sum(vals["kw"]) / len(vals["kw"])
        avg_kva = sum(vals["kva"]) / len(vals["kva"])
        pf = avg_kw / avg_kva if avg_kva > 0 else 0.0
        # Clamp PF to [0, 1]
        pf = max(0.0, min(pf, 1.0))
        status = _pf_status(pf)
        # Estimated loss from low PF: losses proportional to 1/PF^2 - 1
        estimated_loss = ((1 / pf**2) - 1) * 100 if pf > 0 else 100.0
        entities.append(
            EntityPF(
                entity=name,
                avg_kw=round(avg_kw, 4),
                avg_kva=round(avg_kva, 4),
                power_factor=round(pf, 4),
                status=status,
                estimated_loss_pct=round(estimated_loss, 2),
            )
        )

    # Sort by power factor descending (best first)
    entities.sort(key=lambda e: -e.power_factor)

    mean_pf = sum(e.power_factor for e in entities) / len(entities)
    penalty_risk_count = sum(1 for e in entities if e.power_factor < 0.9)
    excellent_count = sum(1 for e in entities if e.power_factor > 0.95)

    summary = (
        f"Power factor analysis across {len(entities)} entities: "
        f"Mean PF = {mean_pf:.3f}. "
        f"{excellent_count} excellent (>0.95), "
        f"{penalty_risk_count} at penalty risk (<0.9)."
    )

    return PowerFactorResult(
        entities=entities,
        mean_pf=round(mean_pf, 4),
        penalty_risk_count=penalty_risk_count,
        excellent_count=excellent_count,
        summary=summary,
    )


def _pf_status(pf: float) -> str:
    """Classify power factor status."""
    if pf > 0.95:
        return "excellent"
    if pf >= 0.9:
        return "good"
    return "poor"


# ---------------------------------------------------------------------------
# 3. compute_specific_energy
# ---------------------------------------------------------------------------


def compute_specific_energy(
    rows: list[dict],
    entity_column: str,
    energy_column: str,
    output_column: str,
) -> SpecificEnergyResult:
    """Compute specific energy consumption (SEC) per entity.

    SEC = total energy / total output (kWh per unit produced).

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        energy_column: Column with energy consumed (kWh).
        output_column: Column with production output.

    Returns:
        SpecificEnergyResult with per-entity SEC and savings potential.
    """
    if not rows:
        return SpecificEnergyResult(
            entities=[],
            mean_sec=0.0,
            best_sec=0.0,
            worst_sec=0.0,
            best_entity="",
            worst_entity="",
            potential_savings=0.0,
            summary="No specific energy data available.",
        )

    # Accumulate per entity
    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        energy = row.get(energy_column)
        output = row.get(output_column)
        if entity is None or energy is None or output is None:
            continue
        try:
            e_val = float(energy)
            o_val = float(output)
        except (TypeError, ValueError):
            continue
        key = str(entity)
        if key not in acc:
            acc[key] = {"energy": 0.0, "output": 0.0}
        acc[key]["energy"] += e_val
        acc[key]["output"] += o_val

    if not acc:
        return SpecificEnergyResult(
            entities=[],
            mean_sec=0.0,
            best_sec=0.0,
            worst_sec=0.0,
            best_entity="",
            worst_entity="",
            potential_savings=0.0,
            summary="No valid specific energy records found.",
        )

    # First pass: compute SEC for each entity
    raw: list[tuple[str, float, float, float]] = []  # (name, energy, output, sec)
    for name, vals in acc.items():
        if vals["output"] == 0:
            sec = float("inf")
        else:
            sec = vals["energy"] / vals["output"]
        raw.append((name, vals["energy"], vals["output"], sec))

    # Determine best and worst SEC (finite only)
    finite_secs = [s for (_, _, _, s) in raw if s != float("inf")]
    best_sec_val = min(finite_secs) if finite_secs else 0.0
    worst_sec_val = max(finite_secs) if finite_secs else 0.0

    # Build entities with deviation from best
    entities: list[EntitySEC] = []
    for name, energy, output, sec in raw:
        if sec != float("inf") and best_sec_val > 0:
            deviation = (sec - best_sec_val) / best_sec_val * 100
        elif sec == float("inf"):
            deviation = float("inf")
        else:
            deviation = 0.0
        entities.append(
            EntitySEC(
                entity=name,
                total_energy=round(energy, 4),
                total_output=round(output, 4),
                sec=round(sec, 4) if sec != float("inf") else float("inf"),
                deviation_from_best_pct=round(deviation, 2) if deviation != float("inf") else float("inf"),
            )
        )

    # Sort by SEC ascending (lower = better)
    entities.sort(key=lambda e: e.sec)

    mean_sec = sum(s for s in finite_secs) / len(finite_secs) if finite_secs else 0.0

    # Best and worst entity names
    finite_entities = [e for e in entities if e.sec != float("inf")]
    best_entity_name = finite_entities[0].entity if finite_entities else ""
    worst_entity_name = finite_entities[-1].entity if finite_entities else ""

    # Potential savings: total excess energy if all matched best SEC
    total_excess = 0.0
    total_energy_all = 0.0
    for e in entities:
        if e.sec != float("inf"):
            optimal_energy = best_sec_val * e.total_output
            total_excess += e.total_energy - optimal_energy
            total_energy_all += e.total_energy

    potential_savings_pct = (total_excess / total_energy_all * 100) if total_energy_all > 0 else 0.0

    summary = (
        f"SEC analysis across {len(entities)} entities: "
        f"Mean SEC = {mean_sec:.2f} kWh/unit. "
        f"Best = {best_entity_name} ({best_sec_val:.2f}), "
        f"Worst = {worst_entity_name} ({worst_sec_val:.2f}). "
        f"Potential energy savings: {potential_savings_pct:.1f}%."
    )

    return SpecificEnergyResult(
        entities=entities,
        mean_sec=round(mean_sec, 4),
        best_sec=round(best_sec_val, 4),
        worst_sec=round(worst_sec_val, 4),
        best_entity=best_entity_name,
        worst_entity=worst_entity_name,
        potential_savings=round(potential_savings_pct, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. detect_demand_peaks
# ---------------------------------------------------------------------------


def detect_demand_peaks(
    rows: list[dict],
    time_column: str,
    power_column: str,
    threshold_pct: float = 90,
) -> list[DemandPeak]:
    """Detect time periods where demand exceeds a threshold percentage of max.

    Consecutive periods above the threshold are grouped and counted as a
    single peak event with a duration.

    Args:
        rows: Data rows as dicts.
        time_column: Column identifying the time period.
        power_column: Numeric column with power/demand values.
        threshold_pct: Percentage of max demand used as threshold (0-100).

    Returns:
        List of DemandPeak, sorted by demand descending.
    """
    if not rows:
        return []

    # Extract (period, demand) pairs preserving row order
    period_values: list[tuple[str, float]] = []
    for row in rows:
        period = row.get(time_column)
        power = row.get(power_column)
        if period is None or power is None:
            continue
        try:
            p_val = float(power)
        except (TypeError, ValueError):
            continue
        period_values.append((str(period), p_val))

    if not period_values:
        return []

    max_demand = max(v for _, v in period_values)
    if max_demand <= 0:
        return []

    threshold = max_demand * threshold_pct / 100.0

    # Scan for consecutive periods above threshold
    peaks: list[DemandPeak] = []
    i = 0
    while i < len(period_values):
        period, demand = period_values[i]
        if demand >= threshold:
            # Start of a peak run
            run_start = i
            run_max_demand = demand
            run_max_period = period
            while i < len(period_values) and period_values[i][1] >= threshold:
                if period_values[i][1] > run_max_demand:
                    run_max_demand = period_values[i][1]
                    run_max_period = period_values[i][0]
                i += 1
            duration = i - run_start
            pct_of_max = (run_max_demand / max_demand * 100) if max_demand > 0 else 0.0
            peaks.append(
                DemandPeak(
                    period=run_max_period,
                    demand=round(run_max_demand, 4),
                    pct_of_max=round(pct_of_max, 2),
                    duration_periods=duration,
                )
            )
        else:
            i += 1

    # Sort by demand descending
    peaks.sort(key=lambda p: -p.demand)
    return peaks


# ---------------------------------------------------------------------------
# 5. energy_cost_analysis
# ---------------------------------------------------------------------------


def energy_cost_analysis(
    rows: list[dict],
    time_column: str,
    energy_column: str,
    rate_column: str | None = None,
    peak_rate: float | None = None,
    offpeak_rate: float | None = None,
) -> EnergyCostResult:
    """Compute energy costs, optionally split by peak/off-peak.

    If *rate_column* is provided, each row's cost is energy * row-rate.
    If *peak_rate* and *offpeak_rate* are provided instead, the function
    classifies each period as peak (top 50% by energy) or off-peak
    (bottom 50%) and applies the corresponding rate.

    Args:
        rows: Data rows as dicts.
        time_column: Column identifying the time period.
        energy_column: Numeric column with energy (kWh).
        rate_column: Optional per-row rate column.
        peak_rate: Optional fixed rate for peak periods (cost/kWh).
        offpeak_rate: Optional fixed rate for off-peak periods (cost/kWh).

    Returns:
        EnergyCostResult with total and split costs.
    """
    if not rows:
        return EnergyCostResult(
            total_energy=0.0,
            total_cost=0.0,
            peak_energy=0.0,
            offpeak_energy=0.0,
            peak_cost=0.0,
            offpeak_cost=0.0,
            avg_rate=0.0,
            potential_shift_savings=0.0,
            summary="No energy cost data available.",
        )

    # Extract per-period energy (and optional rate)
    period_data: list[tuple[str, float, float | None]] = []
    for row in rows:
        period = row.get(time_column)
        energy = row.get(energy_column)
        if period is None or energy is None:
            continue
        try:
            e_val = float(energy)
        except (TypeError, ValueError):
            continue
        row_rate: float | None = None
        if rate_column is not None:
            rate_raw = row.get(rate_column)
            if rate_raw is not None:
                try:
                    row_rate = float(rate_raw)
                except (TypeError, ValueError):
                    row_rate = None
        period_data.append((str(period), e_val, row_rate))

    if not period_data:
        return EnergyCostResult(
            total_energy=0.0,
            total_cost=0.0,
            peak_energy=0.0,
            offpeak_energy=0.0,
            peak_cost=0.0,
            offpeak_cost=0.0,
            avg_rate=0.0,
            potential_shift_savings=0.0,
            summary="No valid energy cost records found.",
        )

    total_energy = sum(e for _, e, _ in period_data)

    # Determine peak vs off-peak split
    # Sort by energy to find median
    sorted_by_energy = sorted(period_data, key=lambda x: x[1], reverse=True)
    mid = len(sorted_by_energy) // 2
    if mid == 0:
        mid = 1  # at least one in each bucket
    peak_set = {item[0] for item in sorted_by_energy[:mid]}

    peak_energy = 0.0
    offpeak_energy = 0.0
    peak_cost = 0.0
    offpeak_cost = 0.0
    total_cost = 0.0

    for period_name, energy_val, row_rate in period_data:
        is_peak = period_name in peak_set

        if rate_column is not None and row_rate is not None:
            cost = energy_val * row_rate
        elif is_peak and peak_rate is not None:
            cost = energy_val * peak_rate
        elif not is_peak and offpeak_rate is not None:
            cost = energy_val * offpeak_rate
        elif peak_rate is not None:
            # Fallback: use peak_rate if offpeak not provided
            cost = energy_val * peak_rate
        elif offpeak_rate is not None:
            cost = energy_val * offpeak_rate
        else:
            cost = 0.0

        total_cost += cost
        if is_peak:
            peak_energy += energy_val
            peak_cost += cost
        else:
            offpeak_energy += energy_val
            offpeak_cost += cost

    avg_rate = total_cost / total_energy if total_energy > 0 else 0.0

    # Potential shift savings: if all peak energy were moved to off-peak rate
    if peak_rate is not None and offpeak_rate is not None and peak_energy > 0:
        current_peak_cost = peak_energy * peak_rate
        shifted_cost = peak_energy * offpeak_rate
        potential_shift_savings = current_peak_cost - shifted_cost
    else:
        potential_shift_savings = 0.0

    summary = (
        f"Energy cost analysis: Total energy = {total_energy:,.2f} kWh, "
        f"Total cost = {total_cost:,.2f}. "
        f"Peak energy = {peak_energy:,.2f} kWh ({peak_cost:,.2f}), "
        f"Off-peak = {offpeak_energy:,.2f} kWh ({offpeak_cost:,.2f}). "
        f"Avg rate = {avg_rate:.4f}/kWh."
    )
    if potential_shift_savings > 0:
        summary += f" Potential shift savings = {potential_shift_savings:,.2f}."

    return EnergyCostResult(
        total_energy=round(total_energy, 4),
        total_cost=round(total_cost, 4),
        peak_energy=round(peak_energy, 4),
        offpeak_energy=round(offpeak_energy, 4),
        peak_cost=round(peak_cost, 4),
        offpeak_cost=round(offpeak_cost, 4),
        avg_rate=round(avg_rate, 4),
        potential_shift_savings=round(potential_shift_savings, 4),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 6. format_power_report
# ---------------------------------------------------------------------------


def format_power_report(
    load: LoadProfileResult | None = None,
    pf: PowerFactorResult | None = None,
    sec: SpecificEnergyResult | None = None,
    cost: EnergyCostResult | None = None,
) -> str:
    """Generate a combined power monitoring report from available analyses.

    Args:
        load: Load profile analysis result.
        pf: Power factor analysis result.
        sec: Specific energy consumption result.
        cost: Energy cost analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("=" * 60)
    sections.append("POWER & ENERGY MONITORING REPORT")
    sections.append("=" * 60)

    if load is not None:
        lines = ["", "LOAD PROFILE", "-" * 40]
        lines.append(f"  Peak Demand:     {load.peak_demand:,.2f}")
        lines.append(f"  Average Demand:  {load.avg_demand:,.2f}")
        lines.append(f"  Minimum Demand:  {load.min_demand:,.2f}")
        lines.append(f"  Load Factor:     {load.load_factor:.1f}%")
        lines.append(f"  Peak Period:     {load.peak_period}")
        lines.append(f"  Off-Peak Period: {load.off_peak_period}")
        if load.periods:
            lines.append("")
            lines.append(f"  {'Period':<20} {'Demand':>10} {'%Peak':>8} {'Class':>10}")
            lines.append(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*10}")
            for p in load.periods:
                lines.append(
                    f"  {p.period:<20} {p.demand:>10,.2f} {p.pct_of_peak:>7.1f}% {p.classification:>10}"
                )
        sections.append("\n".join(lines))

    if pf is not None:
        lines = ["", "POWER FACTOR ANALYSIS", "-" * 40]
        lines.append(f"  Mean PF:            {pf.mean_pf:.4f}")
        lines.append(f"  Excellent (>0.95):  {pf.excellent_count}")
        lines.append(f"  Penalty Risk (<0.9): {pf.penalty_risk_count}")
        if pf.entities:
            lines.append("")
            lines.append(f"  {'Entity':<20} {'PF':>8} {'kW':>10} {'kVA':>10} {'Status':>10} {'Loss%':>8}")
            lines.append(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
            for e in pf.entities:
                lines.append(
                    f"  {e.entity:<20} {e.power_factor:>8.4f} {e.avg_kw:>10,.2f} "
                    f"{e.avg_kva:>10,.2f} {e.status:>10} {e.estimated_loss_pct:>7.2f}%"
                )
        sections.append("\n".join(lines))

    if sec is not None:
        lines = ["", "SPECIFIC ENERGY CONSUMPTION", "-" * 40]
        lines.append(f"  Mean SEC:       {sec.mean_sec:.2f} kWh/unit")
        lines.append(f"  Best SEC:       {sec.best_sec:.2f} ({sec.best_entity})")
        lines.append(f"  Worst SEC:      {sec.worst_sec:.2f} ({sec.worst_entity})")
        lines.append(f"  Savings Potential: {sec.potential_savings:.1f}%")
        if sec.entities:
            lines.append("")
            lines.append(
                f"  {'Entity':<20} {'SEC':>10} {'Energy':>12} {'Output':>12} {'Dev%':>8}"
            )
            lines.append(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
            for e in sec.entities:
                sec_str = f"{e.sec:.2f}" if e.sec != float("inf") else "inf"
                dev_str = f"{e.deviation_from_best_pct:.1f}%" if e.deviation_from_best_pct != float("inf") else "inf"
                lines.append(
                    f"  {e.entity:<20} {sec_str:>10} {e.total_energy:>12,.2f} "
                    f"{e.total_output:>12,.2f} {dev_str:>8}"
                )
        sections.append("\n".join(lines))

    if cost is not None:
        lines = ["", "ENERGY COST ANALYSIS", "-" * 40]
        lines.append(f"  Total Energy:   {cost.total_energy:,.2f} kWh")
        lines.append(f"  Total Cost:     {cost.total_cost:,.2f}")
        lines.append(f"  Peak Energy:    {cost.peak_energy:,.2f} kWh (cost: {cost.peak_cost:,.2f})")
        lines.append(f"  Off-Peak Energy: {cost.offpeak_energy:,.2f} kWh (cost: {cost.offpeak_cost:,.2f})")
        lines.append(f"  Average Rate:   {cost.avg_rate:.4f}/kWh")
        if cost.potential_shift_savings > 0:
            lines.append(f"  Shift Savings:  {cost.potential_shift_savings:,.2f}")
        sections.append("\n".join(lines))

    if load is None and pf is None and sec is None and cost is None:
        sections.append("")
        sections.append("No analysis data provided.")

    sections.append("")
    sections.append("=" * 60)

    return "\n".join(sections)
