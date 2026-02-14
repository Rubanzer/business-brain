"""Gate register and dispatch analytics for manufacturing.

Pure functions for analyzing gate traffic patterns, weighbridge data,
material movement tracking, and dispatch anomaly detection at
manufacturing plant gates.
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
# 1. Gate Traffic Analysis
# ---------------------------------------------------------------------------


@dataclass
class PeriodTraffic:
    """Traffic count for a single time period."""

    period: str
    vehicle_count: int
    pct_of_total: float


@dataclass
class GateTrafficResult:
    """Aggregated gate traffic analysis."""

    total_vehicles: int
    periods: list[PeriodTraffic]
    peak_period: str
    off_peak_period: str
    avg_per_period: float
    direction_split: dict[str, int] | None
    summary: str


def analyze_gate_traffic(
    rows: list[dict],
    time_column: str,
    vehicle_column: str | None = None,
    direction_column: str | None = None,
) -> GateTrafficResult | None:
    """Count vehicles per time period, identify peak hours and direction split.

    Each row represents one gate register entry.  When *vehicle_column* is
    provided the total vehicle count is the number of unique vehicle values;
    otherwise every row is counted as one vehicle passage.

    Args:
        rows: Data rows as dicts.
        time_column: Column identifying the time period (hour, day, shift, etc.).
        vehicle_column: Optional column identifying the vehicle (for unique
            counting and period attribution).
        direction_column: Optional column with direction (e.g. "in"/"out").

    Returns:
        GateTrafficResult or None if no valid data.
    """
    if not rows:
        return None

    period_counts: dict[str, int] = {}
    direction_counts: dict[str, int] | None = {} if direction_column is not None else None
    total_vehicles = 0

    for row in rows:
        period_val = row.get(time_column)
        if period_val is None:
            continue

        # If vehicle_column is specified, skip rows without a vehicle value
        if vehicle_column is not None and row.get(vehicle_column) is None:
            continue

        key = str(period_val)
        period_counts[key] = period_counts.get(key, 0) + 1
        total_vehicles += 1

        if direction_counts is not None:
            dir_val = row.get(direction_column)
            if dir_val is not None:
                dir_key = str(dir_val).strip().lower()
                direction_counts[dir_key] = direction_counts.get(dir_key, 0) + 1

    if not period_counts:
        return None

    periods: list[PeriodTraffic] = []
    for period_key, count in period_counts.items():
        pct = (count / total_vehicles * 100) if total_vehicles > 0 else 0.0
        periods.append(PeriodTraffic(
            period=period_key,
            vehicle_count=count,
            pct_of_total=round(pct, 2),
        ))

    peak = max(periods, key=lambda p: p.vehicle_count)
    off_peak = min(periods, key=lambda p: p.vehicle_count)
    avg = total_vehicles / len(periods)

    # Build direction split dict (or None)
    dir_split: dict[str, int] | None = None
    if direction_counts:
        dir_split = dict(sorted(direction_counts.items()))

    dir_summary = ""
    if dir_split:
        parts = [f"{k}={v}" for k, v in dir_split.items()]
        dir_summary = f" Direction split: {', '.join(parts)}."

    summary = (
        f"Gate traffic across {len(periods)} periods: "
        f"{total_vehicles} total vehicle passages, "
        f"avg {avg:.1f}/period. "
        f"Peak: {peak.period} ({peak.vehicle_count}), "
        f"Off-peak: {off_peak.period} ({off_peak.vehicle_count})."
        f"{dir_summary}"
    )

    return GateTrafficResult(
        total_vehicles=total_vehicles,
        periods=periods,
        peak_period=peak.period,
        off_peak_period=off_peak.period,
        avg_per_period=round(avg, 2),
        direction_split=dir_split,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. Weighbridge Analysis
# ---------------------------------------------------------------------------


@dataclass
class WeighEntry:
    """Weighbridge entry for a single vehicle pass."""

    vehicle: str
    gross_weight: float
    tare_weight: float
    net_weight: float
    material: str | None


@dataclass
class WeighbridgeResult:
    """Aggregated weighbridge analysis."""

    entries: list[WeighEntry]
    total_net_weight: float
    avg_net_weight: float
    total_vehicles: int
    by_material: dict[str, float] | None
    summary: str


def weighbridge_analysis(
    rows: list[dict],
    vehicle_column: str,
    gross_column: str,
    tare_column: str,
    material_column: str | None = None,
) -> WeighbridgeResult | None:
    """Analyze weighbridge data: net weight per vehicle and per material.

    Net weight is computed as ``gross_weight - tare_weight``.

    Args:
        rows: Data rows as dicts.
        vehicle_column: Column identifying the vehicle.
        gross_column: Column with gross weight.
        tare_column: Column with tare weight.
        material_column: Optional column identifying the material carried.

    Returns:
        WeighbridgeResult or None if no valid data.
    """
    if not rows:
        return None

    entries: list[WeighEntry] = []
    material_totals: dict[str, float] = {} if material_column is not None else None

    for row in rows:
        vehicle = row.get(vehicle_column)
        gross = _safe_float(row.get(gross_column))
        tare = _safe_float(row.get(tare_column))
        if vehicle is None or gross is None or tare is None:
            continue

        net = gross - tare
        mat: str | None = None
        if material_column is not None:
            mat_val = row.get(material_column)
            if mat_val is not None:
                mat = str(mat_val)

        entries.append(WeighEntry(
            vehicle=str(vehicle),
            gross_weight=round(gross, 4),
            tare_weight=round(tare, 4),
            net_weight=round(net, 4),
            material=mat,
        ))

        if material_totals is not None and mat is not None:
            material_totals[mat] = material_totals.get(mat, 0.0) + net

    if not entries:
        return None

    total_net = sum(e.net_weight for e in entries)
    avg_net = total_net / len(entries)
    total_vehicles = len(entries)

    by_material: dict[str, float] | None = None
    if material_totals:
        by_material = {k: round(v, 4) for k, v in sorted(material_totals.items())}

    mat_summary = ""
    if by_material:
        parts = [f"{k}={v:.1f}" for k, v in by_material.items()]
        mat_summary = f" By material: {', '.join(parts)}."

    summary = (
        f"Weighbridge analysis: {total_vehicles} entries, "
        f"total net weight {total_net:.1f}, "
        f"avg net weight {avg_net:.1f}."
        f"{mat_summary}"
    )

    return WeighbridgeResult(
        entries=entries,
        total_net_weight=round(total_net, 4),
        avg_net_weight=round(avg_net, 4),
        total_vehicles=total_vehicles,
        by_material=by_material,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. Material Movement Tracking
# ---------------------------------------------------------------------------


@dataclass
class MaterialMovement:
    """Movement summary for a single material."""

    material: str
    inward_qty: float
    outward_qty: float
    net_qty: float
    movement_count: int


@dataclass
class MovementResult:
    """Aggregated material movement analysis."""

    materials: list[MaterialMovement]
    total_inward: float
    total_outward: float
    net_movement: float
    summary: str


def track_material_movement(
    rows: list[dict],
    material_column: str,
    quantity_column: str,
    direction_column: str,
    time_column: str | None = None,
) -> MovementResult | None:
    """Track inward/outward material movement.

    Direction values are normalized to lower-case.  Recognized inward
    values: ``"in"``, ``"inward"``, ``"incoming"``, ``"receipt"``.
    Recognized outward values: ``"out"``, ``"outward"``, ``"outgoing"``,
    ``"dispatch"``.  Unrecognized directions are skipped.

    Args:
        rows: Data rows as dicts.
        material_column: Column identifying the material.
        quantity_column: Column with quantity moved.
        direction_column: Column with movement direction.
        time_column: Optional time column (reserved for future time-series).

    Returns:
        MovementResult or None if no valid data.
    """
    if not rows:
        return None

    inward_keys = {"in", "inward", "incoming", "receipt"}
    outward_keys = {"out", "outward", "outgoing", "dispatch"}

    # material -> {"inward": float, "outward": float, "count": int}
    acc: dict[str, dict[str, float]] = {}

    for row in rows:
        material = row.get(material_column)
        qty = _safe_float(row.get(quantity_column))
        direction = row.get(direction_column)
        if material is None or qty is None or direction is None:
            continue

        dir_lower = str(direction).strip().lower()
        if dir_lower not in inward_keys and dir_lower not in outward_keys:
            continue

        mat_key = str(material)
        if mat_key not in acc:
            acc[mat_key] = {"inward": 0.0, "outward": 0.0, "count": 0}

        if dir_lower in inward_keys:
            acc[mat_key]["inward"] += qty
        else:
            acc[mat_key]["outward"] += qty
        acc[mat_key]["count"] += 1

    if not acc:
        return None

    materials: list[MaterialMovement] = []
    for mat_name in sorted(acc.keys()):
        vals = acc[mat_name]
        inward = vals["inward"]
        outward = vals["outward"]
        net = inward - outward
        count = int(vals["count"])
        materials.append(MaterialMovement(
            material=mat_name,
            inward_qty=round(inward, 4),
            outward_qty=round(outward, 4),
            net_qty=round(net, 4),
            movement_count=count,
        ))

    total_inward = sum(m.inward_qty for m in materials)
    total_outward = sum(m.outward_qty for m in materials)
    net_movement = total_inward - total_outward

    summary = (
        f"Material movement across {len(materials)} materials: "
        f"total inward {total_inward:.1f}, total outward {total_outward:.1f}, "
        f"net {net_movement:.1f}."
    )

    return MovementResult(
        materials=materials,
        total_inward=round(total_inward, 4),
        total_outward=round(total_outward, 4),
        net_movement=round(net_movement, 4),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. Dispatch Anomaly Detection
# ---------------------------------------------------------------------------


@dataclass
class DispatchAnomaly:
    """A single anomalous dispatch entry."""

    vehicle: str
    weight: float
    expected_range: tuple[float, float]
    deviation_pct: float
    anomaly_type: str  # "overweight", "underweight", or "suspicious"


def detect_dispatch_anomalies(
    rows: list[dict],
    vehicle_column: str,
    weight_column: str,
    expected_min: float | None = None,
    expected_max: float | None = None,
) -> list[DispatchAnomaly]:
    """Flag vehicles with unusual weights.

    When *expected_min* and *expected_max* are not provided, they are
    auto-computed as ``mean +/- 2 * std`` from the data.

    Anomaly types:
    - ``"overweight"``: weight exceeds expected_max.
    - ``"underweight"``: weight is below expected_min.
    - ``"suspicious"``: weight is exactly zero or negative.

    Args:
        rows: Data rows as dicts.
        vehicle_column: Column identifying the vehicle.
        weight_column: Column with the weight value.
        expected_min: Optional lower bound for acceptable weight.
        expected_max: Optional upper bound for acceptable weight.

    Returns:
        List of DispatchAnomaly objects (may be empty).
    """
    if not rows:
        return []

    # Collect valid entries
    valid_entries: list[tuple[str, float]] = []
    for row in rows:
        vehicle = row.get(vehicle_column)
        weight = _safe_float(row.get(weight_column))
        if vehicle is None or weight is None:
            continue
        valid_entries.append((str(vehicle), weight))

    if not valid_entries:
        return []

    weights = [w for _, w in valid_entries]

    # Auto-compute bounds if not provided
    if expected_min is None or expected_max is None:
        mean_w = sum(weights) / len(weights)
        if len(weights) >= 2:
            variance = sum((w - mean_w) ** 2 for w in weights) / len(weights)
            std_w = variance ** 0.5
        else:
            std_w = 0.0

        if expected_min is None:
            expected_min = mean_w - 2 * std_w
        if expected_max is None:
            expected_max = mean_w + 2 * std_w

    midpoint = (expected_min + expected_max) / 2.0
    expected_range = (round(expected_min, 4), round(expected_max, 4))

    anomalies: list[DispatchAnomaly] = []

    for vehicle, weight in valid_entries:
        anomaly_type: str | None = None

        if weight <= 0:
            anomaly_type = "suspicious"
        elif weight > expected_max:
            anomaly_type = "overweight"
        elif weight < expected_min:
            anomaly_type = "underweight"

        if anomaly_type is None:
            continue

        # Deviation percentage from the midpoint
        if midpoint != 0:
            deviation_pct = abs(weight - midpoint) / abs(midpoint) * 100
        else:
            deviation_pct = 0.0

        anomalies.append(DispatchAnomaly(
            vehicle=vehicle,
            weight=round(weight, 4),
            expected_range=expected_range,
            deviation_pct=round(deviation_pct, 2),
            anomaly_type=anomaly_type,
        ))

    return anomalies


# ---------------------------------------------------------------------------
# 5. Combined Dispatch Report
# ---------------------------------------------------------------------------


def format_dispatch_report(
    traffic: GateTrafficResult | None = None,
    weighbridge: WeighbridgeResult | None = None,
    movement: MovementResult | None = None,
) -> str:
    """Generate a combined text report from available dispatch analyses.

    Args:
        traffic: Gate traffic analysis result.
        weighbridge: Weighbridge analysis result.
        movement: Material movement analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Dispatch & Gate Report")
    sections.append("=" * 50)

    if traffic is not None:
        lines = ["", "Gate Traffic", "-" * 48]
        for p in traffic.periods:
            lines.append(
                f"  {p.period}: {p.vehicle_count} vehicles "
                f"({p.pct_of_total:.1f}%)"
            )
        lines.append(
            f"  Total: {traffic.total_vehicles} | "
            f"Avg/period: {traffic.avg_per_period:.1f} | "
            f"Peak: {traffic.peak_period} | "
            f"Off-peak: {traffic.off_peak_period}"
        )
        if traffic.direction_split:
            parts = [f"{k}={v}" for k, v in traffic.direction_split.items()]
            lines.append(f"  Direction: {', '.join(parts)}")
        sections.append("\n".join(lines))

    if weighbridge is not None:
        lines = ["", "Weighbridge Analysis", "-" * 48]
        for e in weighbridge.entries:
            mat_str = f" [{e.material}]" if e.material else ""
            lines.append(
                f"  {e.vehicle}: gross={e.gross_weight:.1f} "
                f"tare={e.tare_weight:.1f} net={e.net_weight:.1f}"
                f"{mat_str}"
            )
        lines.append(
            f"  Total net: {weighbridge.total_net_weight:.1f} | "
            f"Avg net: {weighbridge.avg_net_weight:.1f} | "
            f"Vehicles: {weighbridge.total_vehicles}"
        )
        if weighbridge.by_material:
            parts = [f"{k}={v:.1f}" for k, v in weighbridge.by_material.items()]
            lines.append(f"  By material: {', '.join(parts)}")
        sections.append("\n".join(lines))

    if movement is not None:
        lines = ["", "Material Movement", "-" * 48]
        for m in movement.materials:
            lines.append(
                f"  {m.material}: in={m.inward_qty:.1f} "
                f"out={m.outward_qty:.1f} net={m.net_qty:.1f} "
                f"moves={m.movement_count}"
            )
        lines.append(
            f"  Total inward: {movement.total_inward:.1f} | "
            f"Total outward: {movement.total_outward:.1f} | "
            f"Net: {movement.net_movement:.1f}"
        )
        sections.append("\n".join(lines))

    if traffic is None and weighbridge is None and movement is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
