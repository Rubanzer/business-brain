"""Rate and pricing analysis for procurement.

Pure functions for rate comparison across suppliers, rate trend tracking,
rate-volume correlation (bulk discount detection), anomaly detection,
and combined reporting.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data classes — Rate Comparison
# ---------------------------------------------------------------------------


@dataclass
class SupplierRate:
    """Rate statistics for a single supplier (optionally per item)."""

    supplier: str
    avg_rate: float
    min_rate: float
    max_rate: float
    volume: int
    total_value: float


@dataclass
class RateComparison:
    """Comparison of rates across suppliers for one item (or overall)."""

    item: str
    suppliers: list[SupplierRate]
    best_supplier: str
    worst_supplier: str
    spread: float
    spread_pct: float


@dataclass
class RateComparisonResult:
    """Complete rate comparison result."""

    comparisons: list[RateComparison]
    overall_savings_potential: float
    best_rate_supplier: str
    worst_rate_supplier: str
    rate_spread_pct: float
    summary: str


# ---------------------------------------------------------------------------
# Data classes — Rate Trend
# ---------------------------------------------------------------------------


@dataclass
class PeriodRate:
    """Rate statistics for a single time period."""

    period: str
    avg_rate: float
    min_rate: float
    max_rate: float
    volume: int
    change_pct: float


@dataclass
class RateTrendResult:
    """Rate trend analysis result."""

    periods: list[PeriodRate]
    trend_direction: str  # "increasing", "decreasing", "stable"
    total_change_pct: float
    avg_rate: float
    volatility: float
    summary: str


# ---------------------------------------------------------------------------
# Data classes — Rate-Volume
# ---------------------------------------------------------------------------


@dataclass
class SupplierRateVolume:
    """Rate-volume relationship for a single supplier."""

    supplier: str
    total_volume: float
    avg_rate: float
    rate_at_low_volume: float
    rate_at_high_volume: float
    volume_discount_pct: float


@dataclass
class RateVolumeResult:
    """Rate-volume correlation analysis result."""

    suppliers: list[SupplierRateVolume]
    correlation: float
    has_volume_discount: bool
    summary: str


# ---------------------------------------------------------------------------
# Data classes — Rate Anomalies
# ---------------------------------------------------------------------------


@dataclass
class RateAnomaly:
    """A single rate anomaly."""

    supplier: str
    item: str
    rate: float
    avg_rate: float
    deviation_pct: float
    anomaly_type: str  # "too_high" or "too_low"
    severity: str  # "low", "medium", "high"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Try to convert a value to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1. compare_rates
# ---------------------------------------------------------------------------


def compare_rates(
    rows: list[dict],
    supplier_column: str,
    rate_column: str,
    item_column: str | None = None,
) -> RateComparisonResult | None:
    """Compare rates across suppliers, optionally per item.

    Args:
        rows: Data rows as dicts.
        supplier_column: Column identifying the supplier.
        rate_column: Column with rate / price values.
        item_column: Optional column identifying the item. When absent,
            all rows are treated as a single "overall" group.

    Returns:
        RateComparisonResult or None if no valid data.
    """
    if not rows:
        return None

    # Group by (item, supplier) -> list of rates
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        supplier = row.get(supplier_column)
        rate = _safe_float(row.get(rate_column))
        if supplier is None or rate is None:
            continue
        item = str(row.get(item_column)) if item_column and row.get(item_column) is not None else "overall"
        supplier_key = str(supplier)
        grouped.setdefault(item, {}).setdefault(supplier_key, []).append(rate)

    if not grouped:
        return None

    comparisons: list[RateComparison] = []

    for item_name in sorted(grouped.keys()):
        supplier_data = grouped[item_name]
        supplier_rates: list[SupplierRate] = []

        for sup in sorted(supplier_data.keys()):
            rates = supplier_data[sup]
            avg_r = sum(rates) / len(rates)
            supplier_rates.append(SupplierRate(
                supplier=sup,
                avg_rate=round(avg_r, 4),
                min_rate=round(min(rates), 4),
                max_rate=round(max(rates), 4),
                volume=len(rates),
                total_value=round(sum(rates), 4),
            ))

        # Determine best (lowest avg rate) and worst (highest avg rate)
        supplier_rates.sort(key=lambda s: s.avg_rate)
        best = supplier_rates[0].supplier
        worst = supplier_rates[-1].supplier

        best_avg = supplier_rates[0].avg_rate
        worst_avg = supplier_rates[-1].avg_rate
        spread = round(worst_avg - best_avg, 4)
        spread_pct = round((spread / best_avg * 100) if best_avg != 0 else 0.0, 2)

        comparisons.append(RateComparison(
            item=item_name,
            suppliers=supplier_rates,
            best_supplier=best,
            worst_supplier=worst,
            spread=spread,
            spread_pct=spread_pct,
        ))

    # Overall aggregates
    all_supplier_avgs: dict[str, list[float]] = {}
    all_supplier_volumes: dict[str, int] = {}
    for comp in comparisons:
        for sr in comp.suppliers:
            all_supplier_avgs.setdefault(sr.supplier, []).append(sr.avg_rate)
            all_supplier_volumes[sr.supplier] = all_supplier_volumes.get(sr.supplier, 0) + sr.volume

    # Weighted average rate per supplier (weighted by volume)
    supplier_overall: dict[str, float] = {}
    for comp in comparisons:
        for sr in comp.suppliers:
            supplier_overall[sr.supplier] = supplier_overall.get(sr.supplier, 0.0) + sr.total_value
    supplier_wavg: dict[str, float] = {}
    for sup in supplier_overall:
        vol = all_supplier_volumes[sup]
        supplier_wavg[sup] = supplier_overall[sup] / vol if vol > 0 else 0.0

    if not supplier_wavg:
        return None

    best_overall = min(supplier_wavg, key=supplier_wavg.get)  # type: ignore[arg-type]
    worst_overall = max(supplier_wavg, key=supplier_wavg.get)  # type: ignore[arg-type]

    best_wavg = supplier_wavg[best_overall]
    worst_wavg = supplier_wavg[worst_overall]
    overall_spread_pct = round(
        ((worst_wavg - best_wavg) / best_wavg * 100) if best_wavg != 0 else 0.0,
        2,
    )

    # Savings potential: if everything went to the best-rate supplier
    total_current_value = sum(supplier_overall.values())
    total_volume = sum(all_supplier_volumes.values())
    savings_potential = round(total_current_value - (best_wavg * total_volume), 4)

    summary = (
        f"Rate comparison across {len(comparisons)} item(s) and "
        f"{len(supplier_wavg)} suppliers. "
        f"Best rate supplier: {best_overall} (avg {best_wavg:.2f}), "
        f"Worst: {worst_overall} (avg {worst_wavg:.2f}). "
        f"Spread: {overall_spread_pct:.1f}%. "
        f"Potential savings: {savings_potential:,.2f}."
    )

    return RateComparisonResult(
        comparisons=comparisons,
        overall_savings_potential=savings_potential,
        best_rate_supplier=best_overall,
        worst_rate_supplier=worst_overall,
        rate_spread_pct=overall_spread_pct,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. analyze_rate_trend
# ---------------------------------------------------------------------------


def analyze_rate_trend(
    rows: list[dict],
    time_column: str,
    rate_column: str,
    entity_column: str | None = None,
) -> RateTrendResult | None:
    """Track rate changes over time periods.

    Args:
        rows: Data rows as dicts.
        time_column: Column with time period identifier.
        rate_column: Column with rate value.
        entity_column: Optional column to filter/group by entity. When
            provided, rates are aggregated across entities per period.

    Returns:
        RateTrendResult or None if fewer than 2 periods.
    """
    if not rows:
        return None

    # Aggregate rates per period
    period_rates: dict[str, list[float]] = {}
    for row in rows:
        period = row.get(time_column)
        rate = _safe_float(row.get(rate_column))
        if period is None or rate is None:
            continue
        key = str(period)
        period_rates.setdefault(key, []).append(rate)

    if len(period_rates) < 2:
        return None

    # Sort periods by natural (string) order
    sorted_periods = sorted(period_rates.items(), key=lambda x: x[0])

    periods: list[PeriodRate] = []
    changes: list[float] = []
    prev_avg: float | None = None

    for period_name, rates in sorted_periods:
        avg_r = sum(rates) / len(rates)

        if prev_avg is not None and prev_avg != 0:
            change_pct = (avg_r - prev_avg) / abs(prev_avg) * 100
        elif prev_avg is not None:
            change_pct = 0.0
        else:
            change_pct = 0.0

        if prev_avg is not None:
            changes.append(change_pct)

        periods.append(PeriodRate(
            period=period_name,
            avg_rate=round(avg_r, 4),
            min_rate=round(min(rates), 4),
            max_rate=round(max(rates), 4),
            volume=len(rates),
            change_pct=round(change_pct, 2),
        ))
        prev_avg = avg_r

    # Overall statistics
    all_avgs = [p.avg_rate for p in periods]
    overall_avg = round(sum(all_avgs) / len(all_avgs), 4)

    first_avg = periods[0].avg_rate
    last_avg = periods[-1].avg_rate
    total_change_pct = round(
        ((last_avg - first_avg) / abs(first_avg) * 100) if first_avg != 0 else 0.0,
        2,
    )

    avg_change = sum(changes) / len(changes) if changes else 0.0
    if avg_change > 2.0:
        trend_direction = "increasing"
    elif avg_change < -2.0:
        trend_direction = "decreasing"
    else:
        trend_direction = "stable"

    # Volatility: standard deviation of period-over-period changes
    if len(changes) >= 2:
        volatility = round(statistics.stdev(changes), 2)
    else:
        volatility = 0.0

    summary = (
        f"Rate trend over {len(periods)} periods: "
        f"Direction = {trend_direction} ({avg_change:+.1f}% avg per period). "
        f"Total change = {total_change_pct:+.1f}%. "
        f"Avg rate = {overall_avg:.2f}. "
        f"Volatility = {volatility:.1f}%."
    )

    return RateTrendResult(
        periods=periods,
        trend_direction=trend_direction,
        total_change_pct=total_change_pct,
        avg_rate=overall_avg,
        volatility=volatility,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. rate_volume_analysis
# ---------------------------------------------------------------------------


def rate_volume_analysis(
    rows: list[dict],
    supplier_column: str,
    rate_column: str,
    volume_column: str,
) -> RateVolumeResult | None:
    """Analyze correlation between volume and rate (bulk discounts).

    For each supplier, splits transactions into low-volume and high-volume
    halves and compares the average rate in each half.

    Args:
        rows: Data rows as dicts.
        supplier_column: Column identifying the supplier.
        rate_column: Column with rate / price.
        volume_column: Column with volume / quantity.

    Returns:
        RateVolumeResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect (volume, rate) pairs per supplier
    supplier_pairs: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        supplier = row.get(supplier_column)
        rate = _safe_float(row.get(rate_column))
        volume = _safe_float(row.get(volume_column))
        if supplier is None or rate is None or volume is None:
            continue
        supplier_pairs.setdefault(str(supplier), []).append((volume, rate))

    if not supplier_pairs:
        return None

    supplier_results: list[SupplierRateVolume] = []
    all_volumes: list[float] = []
    all_rates: list[float] = []

    for sup in sorted(supplier_pairs.keys()):
        pairs = supplier_pairs[sup]
        # Sort by volume ascending
        pairs.sort(key=lambda p: p[0])
        volumes = [p[0] for p in pairs]
        rates = [p[1] for p in pairs]
        total_vol = sum(volumes)
        avg_rate = sum(rates) / len(rates)

        all_volumes.extend(volumes)
        all_rates.extend(rates)

        if len(pairs) >= 2:
            mid = len(pairs) // 2
            low_rates = [p[1] for p in pairs[:mid]]
            high_rates = [p[1] for p in pairs[mid:]]
            rate_low = sum(low_rates) / len(low_rates)
            rate_high = sum(high_rates) / len(high_rates)
        else:
            rate_low = avg_rate
            rate_high = avg_rate

        discount_pct = round(
            ((rate_low - rate_high) / rate_low * 100) if rate_low != 0 else 0.0,
            2,
        )

        supplier_results.append(SupplierRateVolume(
            supplier=sup,
            total_volume=round(total_vol, 4),
            avg_rate=round(avg_rate, 4),
            rate_at_low_volume=round(rate_low, 4),
            rate_at_high_volume=round(rate_high, 4),
            volume_discount_pct=discount_pct,
        ))

    # Pearson correlation between volume and rate across all data
    correlation = _pearson(all_volumes, all_rates)

    # Has volume discount if correlation is notably negative
    has_volume_discount = correlation < -0.3

    n_with_discount = sum(1 for s in supplier_results if s.volume_discount_pct > 0)
    summary = (
        f"Rate-volume analysis across {len(supplier_results)} suppliers. "
        f"Correlation = {correlation:.2f}. "
        f"Volume discount {'detected' if has_volume_discount else 'not detected'}. "
        f"{n_with_discount} supplier(s) show lower rates at higher volumes."
    )

    return RateVolumeResult(
        suppliers=supplier_results,
        correlation=round(correlation, 4),
        has_volume_discount=has_volume_discount,
        summary=summary,
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two lists.

    Returns 0.0 when the correlation is undefined (e.g. constant data,
    fewer than 2 points).
    """
    n = len(xs)
    if n < 2 or len(ys) < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return 0.0

    return cov / denom


# ---------------------------------------------------------------------------
# 4. detect_rate_anomalies
# ---------------------------------------------------------------------------


def detect_rate_anomalies(
    rows: list[dict],
    supplier_column: str,
    rate_column: str,
    item_column: str | None = None,
    threshold_pct: float = 20,
) -> list[RateAnomaly]:
    """Detect rates that deviate significantly from the group norm.

    For each (item) group, computes the mean rate and flags any
    (supplier, item) combination whose average rate deviates by more
    than ``threshold_pct`` percent.

    Args:
        rows: Data rows as dicts.
        supplier_column: Column identifying the supplier.
        rate_column: Column with rate / price.
        item_column: Optional column identifying the item. When absent,
            all rows are treated as a single group.
        threshold_pct: Percentage deviation to flag as anomalous.

    Returns:
        List of RateAnomaly (may be empty).
    """
    if not rows:
        return []

    # Group by item -> supplier -> list[float]
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        supplier = row.get(supplier_column)
        rate = _safe_float(row.get(rate_column))
        if supplier is None or rate is None:
            continue
        item = str(row.get(item_column)) if item_column and row.get(item_column) is not None else "overall"
        grouped.setdefault(item, {}).setdefault(str(supplier), []).append(rate)

    anomalies: list[RateAnomaly] = []

    for item_name in sorted(grouped.keys()):
        supplier_data = grouped[item_name]

        # Compute the overall group mean across all suppliers for this item
        all_rates: list[float] = []
        for rates in supplier_data.values():
            all_rates.extend(rates)

        if not all_rates:
            continue

        group_mean = sum(all_rates) / len(all_rates)

        if group_mean == 0:
            continue

        for sup in sorted(supplier_data.keys()):
            rates = supplier_data[sup]
            sup_avg = sum(rates) / len(rates)
            deviation_pct = (sup_avg - group_mean) / abs(group_mean) * 100

            if abs(deviation_pct) >= threshold_pct:
                anomaly_type = "too_high" if deviation_pct > 0 else "too_low"
                abs_dev = abs(deviation_pct)
                if abs_dev >= threshold_pct * 2:
                    severity = "high"
                elif abs_dev >= threshold_pct * 1.5:
                    severity = "medium"
                else:
                    severity = "low"

                anomalies.append(RateAnomaly(
                    supplier=sup,
                    item=item_name,
                    rate=round(sup_avg, 4),
                    avg_rate=round(group_mean, 4),
                    deviation_pct=round(deviation_pct, 2),
                    anomaly_type=anomaly_type,
                    severity=severity,
                ))

    return anomalies


# ---------------------------------------------------------------------------
# 5. format_rate_report
# ---------------------------------------------------------------------------


def format_rate_report(
    comparison: RateComparisonResult | None = None,
    trend: RateTrendResult | None = None,
    anomalies: list[RateAnomaly] | None = None,
) -> str:
    """Generate a combined text report from available rate analyses.

    Args:
        comparison: Rate comparison result.
        trend: Rate trend result.
        anomalies: List of rate anomalies.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Rate Analysis Report")
    sections.append("=" * 50)

    if comparison is not None:
        lines = ["", "Rate Comparison", "-" * 48]
        for comp in comparison.comparisons:
            lines.append(f"  Item: {comp.item}")
            for sr in comp.suppliers:
                lines.append(
                    f"    {sr.supplier}: avg={sr.avg_rate:.2f} "
                    f"min={sr.min_rate:.2f} max={sr.max_rate:.2f} "
                    f"vol={sr.volume}"
                )
            lines.append(
                f"    Best: {comp.best_supplier} | Worst: {comp.worst_supplier} "
                f"| Spread: {comp.spread:.2f} ({comp.spread_pct:.1f}%)"
            )
        lines.append(f"  Overall best: {comparison.best_rate_supplier}")
        lines.append(f"  Overall worst: {comparison.worst_rate_supplier}")
        lines.append(f"  Savings potential: {comparison.overall_savings_potential:,.2f}")
        sections.append("\n".join(lines))

    if trend is not None:
        lines = ["", "Rate Trend", "-" * 48]
        for p in trend.periods:
            change_str = f" ({p.change_pct:+.1f}%)" if p.change_pct != 0 else ""
            lines.append(
                f"  {p.period}: avg={p.avg_rate:.2f} "
                f"[{p.min_rate:.2f} - {p.max_rate:.2f}]{change_str}"
            )
        lines.append(f"  Direction: {trend.trend_direction}")
        lines.append(f"  Total change: {trend.total_change_pct:+.1f}%")
        lines.append(f"  Avg rate: {trend.avg_rate:.2f}")
        lines.append(f"  Volatility: {trend.volatility:.1f}%")
        sections.append("\n".join(lines))

    if anomalies is not None:
        lines = ["", "Rate Anomalies", "-" * 48]
        if anomalies:
            for a in anomalies:
                lines.append(
                    f"  {a.supplier} / {a.item}: rate={a.rate:.2f} "
                    f"vs avg={a.avg_rate:.2f} "
                    f"({a.deviation_pct:+.1f}%) [{a.anomaly_type}, {a.severity}]"
                )
        else:
            lines.append("  No anomalies detected.")
        sections.append("\n".join(lines))

    if comparison is None and trend is None and anomalies is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
