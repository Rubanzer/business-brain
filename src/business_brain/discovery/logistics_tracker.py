"""Logistics and dispatch tracking analytics for manufacturing.

Pure functions for computing delivery performance, truck/vehicle utilization,
dispatch frequency patterns, and transit time analysis across entities such
as routes, carriers, or distribution centres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


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
# 1. Delivery Performance
# ---------------------------------------------------------------------------


@dataclass
class EntityDelivery:
    """Delivery metrics for a single entity (carrier, route, warehouse)."""

    entity: str
    total_deliveries: int
    on_time_count: int
    late_count: int
    early_count: int
    on_time_rate: float
    avg_delay: float  # positive = late, negative = early


@dataclass
class DeliveryResult:
    """Aggregated delivery performance analysis."""

    entities: list[EntityDelivery]
    total_deliveries: int
    on_time_count: int
    on_time_rate: float
    avg_delay: float
    worst_entity: str
    best_entity: str
    summary: str


def analyze_delivery_performance(
    rows: list[dict],
    entity_column: str,
    promised_column: str,
    actual_column: str,
) -> DeliveryResult | None:
    """Analyse on-time delivery rate and average delay per entity.

    Delay is computed as ``actual - promised``.  A positive delay means
    the delivery was late; negative means early; zero means on time.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (carrier, route, etc.).
        promised_column: Column with the promised / scheduled delivery value.
        actual_column: Column with the actual delivery value.

    Returns:
        DeliveryResult or None if no valid data.
    """
    if not rows:
        return None

    # entity -> list of delays
    entity_delays: dict[str, list[float]] = {}

    for row in rows:
        ent = row.get(entity_column)
        promised = _safe_float(row.get(promised_column))
        actual = _safe_float(row.get(actual_column))
        if ent is None or promised is None or actual is None:
            continue
        key = str(ent)
        delay = actual - promised
        entity_delays.setdefault(key, []).append(delay)

    if not entity_delays:
        return None

    entities: list[EntityDelivery] = []

    for ent_name in sorted(entity_delays.keys()):
        delays = entity_delays[ent_name]
        total = len(delays)
        on_time = sum(1 for d in delays if d <= 0)
        late = sum(1 for d in delays if d > 0)
        early = sum(1 for d in delays if d < 0)
        avg_d = sum(delays) / total
        rate = (on_time / total * 100) if total > 0 else 0.0

        entities.append(EntityDelivery(
            entity=ent_name,
            total_deliveries=total,
            on_time_count=on_time,
            late_count=late,
            early_count=early,
            on_time_rate=round(rate, 2),
            avg_delay=round(avg_d, 4),
        ))

    # Aggregate totals
    total_deliveries = sum(e.total_deliveries for e in entities)
    total_on_time = sum(e.on_time_count for e in entities)
    overall_rate = (total_on_time / total_deliveries * 100) if total_deliveries > 0 else 0.0
    all_delays = [d for delays in entity_delays.values() for d in delays]
    overall_avg_delay = sum(all_delays) / len(all_delays) if all_delays else 0.0

    # Best = highest on-time rate, worst = lowest
    best = max(entities, key=lambda e: e.on_time_rate)
    worst = min(entities, key=lambda e: e.on_time_rate)

    summary = (
        f"Delivery performance across {len(entities)} entities: "
        f"{total_deliveries} total deliveries, "
        f"{total_on_time} on time ({overall_rate:.1f}%). "
        f"Avg delay: {overall_avg_delay:.2f}. "
        f"Best: {best.entity} ({best.on_time_rate:.1f}%), "
        f"Worst: {worst.entity} ({worst.on_time_rate:.1f}%)."
    )

    return DeliveryResult(
        entities=entities,
        total_deliveries=total_deliveries,
        on_time_count=total_on_time,
        on_time_rate=round(overall_rate, 2),
        avg_delay=round(overall_avg_delay, 4),
        worst_entity=worst.entity,
        best_entity=best.entity,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. Vehicle Utilization
# ---------------------------------------------------------------------------


@dataclass
class VehicleUtil:
    """Utilization metrics for a single vehicle."""

    vehicle: str
    total_trips: int
    avg_load: float
    avg_capacity: float
    utilization_pct: float
    status: str  # "underloaded", "optimal", "overloaded"


@dataclass
class VehicleUtilResult:
    """Aggregated vehicle utilization analysis."""

    vehicles: list[VehicleUtil]
    mean_utilization: float
    underloaded_count: int  # utilization < 60%
    overloaded_count: int   # utilization > 100%
    summary: str


def _vehicle_status(util_pct: float) -> str:
    """Classify vehicle utilization into a status bucket."""
    if util_pct < 60.0:
        return "underloaded"
    if util_pct > 100.0:
        return "overloaded"
    return "optimal"


def compute_vehicle_utilization(
    rows: list[dict],
    vehicle_column: str,
    capacity_column: str,
    load_column: str,
) -> VehicleUtilResult | None:
    """Compute load factor (actual_load / capacity * 100) per vehicle.

    Args:
        rows: Data rows as dicts.
        vehicle_column: Column identifying the vehicle.
        capacity_column: Column with vehicle capacity.
        load_column: Column with actual load carried.

    Returns:
        VehicleUtilResult or None if no valid data.
    """
    if not rows:
        return None

    # vehicle -> list of (load, capacity)
    vehicle_data: dict[str, list[tuple[float, float]]] = {}

    for row in rows:
        veh = row.get(vehicle_column)
        cap = _safe_float(row.get(capacity_column))
        load = _safe_float(row.get(load_column))
        if veh is None or cap is None or load is None:
            continue
        key = str(veh)
        vehicle_data.setdefault(key, []).append((load, cap))

    if not vehicle_data:
        return None

    vehicles: list[VehicleUtil] = []

    for veh_name in sorted(vehicle_data.keys()):
        records = vehicle_data[veh_name]
        total_trips = len(records)
        avg_load = sum(r[0] for r in records) / total_trips
        avg_cap = sum(r[1] for r in records) / total_trips
        util_pct = (avg_load / avg_cap * 100) if avg_cap > 0 else 0.0
        status = _vehicle_status(util_pct)

        vehicles.append(VehicleUtil(
            vehicle=veh_name,
            total_trips=total_trips,
            avg_load=round(avg_load, 4),
            avg_capacity=round(avg_cap, 4),
            utilization_pct=round(util_pct, 2),
            status=status,
        ))

    mean_util = sum(v.utilization_pct for v in vehicles) / len(vehicles)
    underloaded = sum(1 for v in vehicles if v.status == "underloaded")
    overloaded = sum(1 for v in vehicles if v.status == "overloaded")

    summary = (
        f"Vehicle utilization across {len(vehicles)} vehicles: "
        f"mean {mean_util:.1f}%. "
        f"{underloaded} underloaded (<60%), "
        f"{overloaded} overloaded (>100%)."
    )

    return VehicleUtilResult(
        vehicles=vehicles,
        mean_utilization=round(mean_util, 2),
        underloaded_count=underloaded,
        overloaded_count=overloaded,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. Dispatch Frequency
# ---------------------------------------------------------------------------


@dataclass
class PeriodDispatch:
    """Dispatch count for a single time period."""

    period: str
    dispatch_count: int
    pct_of_total: float


@dataclass
class DispatchResult:
    """Aggregated dispatch frequency analysis."""

    periods: list[PeriodDispatch]
    total_dispatches: int
    peak_period: str
    trough_period: str
    avg_per_period: float
    summary: str


def analyze_dispatch_frequency(
    rows: list[dict],
    time_column: str,
    entity_column: str | None = None,
) -> DispatchResult | None:
    """Count dispatches per time period and detect peak/trough patterns.

    Each row is counted as one dispatch.  When *entity_column* is provided,
    dispatches are still counted per period but the entity is used only
    for grouping context (the result is aggregated across all entities).

    Args:
        rows: Data rows as dicts.
        time_column: Column identifying the time period (day, week, month, etc.).
        entity_column: Optional column to filter / group by entity.

    Returns:
        DispatchResult or None if no valid data.
    """
    if not rows:
        return None

    # Count dispatches per period (preserve insertion order)
    period_counts: dict[str, int] = {}

    for row in rows:
        period_val = row.get(time_column)
        if period_val is None:
            continue
        # If entity_column specified, skip rows with missing entity
        if entity_column is not None and row.get(entity_column) is None:
            continue
        key = str(period_val)
        period_counts[key] = period_counts.get(key, 0) + 1

    if not period_counts:
        return None

    total = sum(period_counts.values())

    periods: list[PeriodDispatch] = []
    for period_key, count in period_counts.items():
        pct = (count / total * 100) if total > 0 else 0.0
        periods.append(PeriodDispatch(
            period=period_key,
            dispatch_count=count,
            pct_of_total=round(pct, 2),
        ))

    peak = max(periods, key=lambda p: p.dispatch_count)
    trough = min(periods, key=lambda p: p.dispatch_count)
    avg = total / len(periods)

    summary = (
        f"Dispatch frequency across {len(periods)} periods: "
        f"{total} total dispatches, avg {avg:.1f}/period. "
        f"Peak: {peak.period} ({peak.dispatch_count}), "
        f"Trough: {trough.period} ({trough.dispatch_count})."
    )

    return DispatchResult(
        periods=periods,
        total_dispatches=total,
        peak_period=peak.period,
        trough_period=trough.period,
        avg_per_period=round(avg, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. Transit Time
# ---------------------------------------------------------------------------


@dataclass
class EntityTransit:
    """Transit time metrics for a single entity (route, lane, carrier)."""

    entity: str
    trip_count: int
    avg_transit_time: float
    min_transit: float
    max_transit: float
    consistency: float  # coefficient of variation: std / mean (0 = perfectly consistent)


@dataclass
class TransitResult:
    """Aggregated transit time analysis."""

    entities: list[EntityTransit]
    avg_transit: float
    fastest_entity: str
    slowest_entity: str
    summary: str


def compute_transit_time(
    rows: list[dict],
    entity_column: str,
    departure_column: str,
    arrival_column: str,
) -> TransitResult | None:
    """Compute average transit time per entity.

    Transit time is ``arrival - departure``.  Both columns are expected to
    contain numeric values (e.g. epoch timestamps, day-of-year offsets, or
    hour values).

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (route, carrier, etc.).
        departure_column: Column with departure value.
        arrival_column: Column with arrival value.

    Returns:
        TransitResult or None if no valid data.
    """
    if not rows:
        return None

    # entity -> list of transit times
    entity_times: dict[str, list[float]] = {}

    for row in rows:
        ent = row.get(entity_column)
        dep = _safe_float(row.get(departure_column))
        arr = _safe_float(row.get(arrival_column))
        if ent is None or dep is None or arr is None:
            continue
        transit = arr - dep
        entity_times.setdefault(str(ent), []).append(transit)

    if not entity_times:
        return None

    entities: list[EntityTransit] = []

    for ent_name in sorted(entity_times.keys()):
        times = entity_times[ent_name]
        count = len(times)
        avg_t = sum(times) / count
        min_t = min(times)
        max_t = max(times)

        # Coefficient of variation (std / mean)
        if count < 2 or avg_t == 0:
            consistency = 0.0
        else:
            variance = sum((t - avg_t) ** 2 for t in times) / count
            std = math.sqrt(variance)
            consistency = std / abs(avg_t)

        entities.append(EntityTransit(
            entity=ent_name,
            trip_count=count,
            avg_transit_time=round(avg_t, 4),
            min_transit=round(min_t, 4),
            max_transit=round(max_t, 4),
            consistency=round(consistency, 4),
        ))

    # Aggregate
    all_times = [t for times in entity_times.values() for t in times]
    overall_avg = sum(all_times) / len(all_times) if all_times else 0.0

    fastest = min(entities, key=lambda e: e.avg_transit_time)
    slowest = max(entities, key=lambda e: e.avg_transit_time)

    summary = (
        f"Transit time across {len(entities)} entities: "
        f"avg {overall_avg:.2f}. "
        f"Fastest: {fastest.entity} ({fastest.avg_transit_time:.2f}), "
        f"Slowest: {slowest.entity} ({slowest.avg_transit_time:.2f})."
    )

    return TransitResult(
        entities=entities,
        avg_transit=round(overall_avg, 4),
        fastest_entity=fastest.entity,
        slowest_entity=slowest.entity,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. Combined Logistics Report
# ---------------------------------------------------------------------------


def format_logistics_report(
    delivery: DeliveryResult | None = None,
    vehicle: VehicleUtilResult | None = None,
    dispatch: DispatchResult | None = None,
    transit: TransitResult | None = None,
) -> str:
    """Generate a combined text report from available logistics analyses.

    Args:
        delivery: Delivery performance result.
        vehicle: Vehicle utilization result.
        dispatch: Dispatch frequency result.
        transit: Transit time result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Logistics & Dispatch Report")
    sections.append("=" * 50)

    if delivery is not None:
        lines = ["", "Delivery Performance", "-" * 48]
        for e in sorted(delivery.entities, key=lambda x: -x.on_time_rate):
            lines.append(
                f"  {e.entity}: on-time={e.on_time_rate:.1f}% "
                f"({e.on_time_count}/{e.total_deliveries}) "
                f"avg_delay={e.avg_delay:.2f}"
            )
        lines.append(
            f"  Overall: {delivery.on_time_rate:.1f}% on-time, "
            f"avg delay {delivery.avg_delay:.2f}"
        )
        sections.append("\n".join(lines))

    if vehicle is not None:
        lines = ["", "Vehicle Utilization", "-" * 48]
        for v in sorted(vehicle.vehicles, key=lambda x: -x.utilization_pct):
            lines.append(
                f"  {v.vehicle}: {v.utilization_pct:.1f}% "
                f"(load={v.avg_load:.1f}/{v.avg_capacity:.1f}) "
                f"[{v.status}] trips={v.total_trips}"
            )
        lines.append(
            f"  Mean utilization: {vehicle.mean_utilization:.1f}% | "
            f"Underloaded: {vehicle.underloaded_count}, "
            f"Overloaded: {vehicle.overloaded_count}"
        )
        sections.append("\n".join(lines))

    if dispatch is not None:
        lines = ["", "Dispatch Frequency", "-" * 48]
        for p in dispatch.periods:
            lines.append(
                f"  {p.period}: {p.dispatch_count} dispatches "
                f"({p.pct_of_total:.1f}%)"
            )
        lines.append(
            f"  Total: {dispatch.total_dispatches} | "
            f"Avg/period: {dispatch.avg_per_period:.1f} | "
            f"Peak: {dispatch.peak_period} | "
            f"Trough: {dispatch.trough_period}"
        )
        sections.append("\n".join(lines))

    if transit is not None:
        lines = ["", "Transit Time", "-" * 48]
        for e in sorted(transit.entities, key=lambda x: x.avg_transit_time):
            lines.append(
                f"  {e.entity}: avg={e.avg_transit_time:.2f} "
                f"(min={e.min_transit:.2f}, max={e.max_transit:.2f}) "
                f"consistency={e.consistency:.3f} trips={e.trip_count}"
            )
        lines.append(
            f"  Overall avg transit: {transit.avg_transit:.2f} | "
            f"Fastest: {transit.fastest_entity} | "
            f"Slowest: {transit.slowest_entity}"
        )
        sections.append("\n".join(lines))

    if delivery is None and vehicle is None and dispatch is None and transit is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
