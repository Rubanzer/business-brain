"""Geographic / regional analytics -- distribution, comparison, growth, penetration.

Pure functions for regional distribution analysis, multi-metric region
comparison, geographic growth tracking, and market penetration assessment.
No DB, async, or LLM dependencies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, date


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(value) -> float | None:
    """Attempt to parse a value into a float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value) -> datetime | None:
    """Attempt to parse a value into a datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RegionMetrics:
    """Metrics for a single region in a distribution analysis."""

    region: str
    total_value: float
    share_pct: float
    count: int
    avg_value: float
    rank: int


@dataclass
class RegionalDistributionResult:
    """Complete regional distribution result."""

    regions: list[RegionMetrics]
    total_value: float
    total_count: int
    top_region: str
    concentration_ratio: float
    hhi_index: float
    summary: str


@dataclass
class MetricComparison:
    """Comparison of a single metric across regions."""

    metric: str
    best_region: str
    best_value: float
    worst_region: str
    worst_value: float
    avg_value: float
    std_dev: float


@dataclass
class RegionScore:
    """Aggregate score for a single region across multiple metrics."""

    region: str
    metrics: dict
    overall_score: float


@dataclass
class RegionComparisonResult:
    """Complete multi-metric region comparison result."""

    comparisons: list[MetricComparison]
    region_scores: list[RegionScore]
    best_overall: str
    summary: str


@dataclass
class RegionGrowth:
    """Growth metrics for a single region."""

    region: str
    first_period_value: float
    last_period_value: float
    growth_rate: float
    periods: int


@dataclass
class GeoGrowthResult:
    """Complete geographic growth analysis result."""

    regions: list[RegionGrowth]
    fastest_growing: str
    slowest_growing: str
    avg_growth: float
    summary: str


@dataclass
class RegionPenetration:
    """Market penetration metrics for a single region."""

    region: str
    customer_count: int
    potential: float | None
    penetration_pct: float | None
    rank: int


@dataclass
class MarketPenetrationResult:
    """Complete market penetration analysis result."""

    regions: list[RegionPenetration]
    total_customers: int
    total_regions: int
    best_penetration: str | None
    untapped_regions: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_regional_distribution(
    rows: list[dict],
    region_column: str,
    value_column: str,
    count_column: str | None = None,
) -> RegionalDistributionResult | None:
    """Breakdown of value (revenue, sales, etc.) by region.

    Computes each region's share of total, concentration analysis,
    and HHI index.

    Args:
        rows: Data as list of dicts.
        region_column: Column identifying the region.
        value_column: Column with the value to aggregate (e.g. revenue).
        count_column: Optional column for explicit counts; if None each
            row counts as 1.

    Returns:
        RegionalDistributionResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per region
    region_data: dict[str, dict] = {}
    for row in rows:
        region = row.get(region_column)
        val = _safe_float(row.get(value_column))
        if region is None or val is None:
            continue

        region_str = str(region)
        if region_str not in region_data:
            region_data[region_str] = {"total_value": 0.0, "count": 0}

        entry = region_data[region_str]
        entry["total_value"] += val

        if count_column is not None:
            cnt = _safe_float(row.get(count_column))
            if cnt is not None:
                entry["count"] += int(cnt)
            else:
                entry["count"] += 1
        else:
            entry["count"] += 1

    if not region_data:
        return None

    total_value = sum(e["total_value"] for e in region_data.values())
    total_count = sum(e["count"] for e in region_data.values())

    if total_value == 0:
        # All zero values -- still produce a result with 0 shares
        sorted_regions = sorted(region_data.keys())
        region_metrics: list[RegionMetrics] = []
        for rank, region_name in enumerate(sorted_regions, start=1):
            entry = region_data[region_name]
            region_metrics.append(RegionMetrics(
                region=region_name,
                total_value=0.0,
                share_pct=0.0,
                count=entry["count"],
                avg_value=0.0,
                rank=rank,
            ))
        return RegionalDistributionResult(
            regions=region_metrics,
            total_value=0.0,
            total_count=total_count,
            top_region=sorted_regions[0],
            concentration_ratio=0.0,
            hhi_index=0.0,
            summary=(
                f"Regional Distribution: {len(region_data)} regions, "
                f"total value $0.00. All regions have zero value."
            ),
        )

    # Sort regions by total_value descending for ranking
    sorted_items = sorted(
        region_data.items(), key=lambda x: x[1]["total_value"], reverse=True
    )

    region_metrics = []
    for rank, (region_name, entry) in enumerate(sorted_items, start=1):
        share = entry["total_value"] / total_value * 100
        avg_val = entry["total_value"] / entry["count"] if entry["count"] > 0 else 0.0
        region_metrics.append(RegionMetrics(
            region=region_name,
            total_value=round(entry["total_value"], 2),
            share_pct=round(share, 2),
            count=entry["count"],
            avg_value=round(avg_val, 2),
            rank=rank,
        ))

    top_region = sorted_items[0][0]
    concentration_ratio = round(
        sorted_items[0][1]["total_value"] / total_value * 100, 2
    )

    # HHI = sum of (share_pct / 100)^2 * 10000
    hhi = sum(
        (e["total_value"] / total_value) ** 2
        for e in region_data.values()
    ) * 10000
    hhi = round(hhi, 2)

    summary = (
        f"Regional Distribution: {len(region_data)} regions, "
        f"total value ${total_value:,.2f}. "
        f"Top region: {top_region} ({concentration_ratio:.1f}% share). "
        f"HHI: {hhi:.0f}."
    )

    return RegionalDistributionResult(
        regions=region_metrics,
        total_value=round(total_value, 2),
        total_count=total_count,
        top_region=top_region,
        concentration_ratio=concentration_ratio,
        hhi_index=hhi,
        summary=summary,
    )


def compare_regions(
    rows: list[dict],
    region_column: str,
    metric_columns: list[str],
) -> RegionComparisonResult | None:
    """Compare multiple metrics across regions.

    Computes z-scores for each metric per region relative to the
    all-region average, and identifies best/worst region per metric.

    Args:
        rows: Data as list of dicts.
        region_column: Column identifying the region.
        metric_columns: List of column names with numeric metrics.

    Returns:
        RegionComparisonResult or None if insufficient data.
    """
    if not rows or not metric_columns:
        return None

    # Aggregate per region: sum of each metric and count of valid rows
    region_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    region_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    has_any = False
    for row in rows:
        region = row.get(region_column)
        if region is None:
            continue
        region_str = str(region)
        for mc in metric_columns:
            val = _safe_float(row.get(mc))
            if val is not None:
                region_sums[region_str][mc] += val
                region_counts[region_str][mc] += 1
                has_any = True

    if not has_any or not region_sums:
        return None

    # Compute averages per region per metric
    region_avgs: dict[str, dict[str, float]] = {}
    all_regions = sorted(region_sums.keys())
    for r in all_regions:
        region_avgs[r] = {}
        for mc in metric_columns:
            cnt = region_counts[r].get(mc, 0)
            if cnt > 0:
                region_avgs[r][mc] = region_sums[r][mc] / cnt
            else:
                region_avgs[r][mc] = 0.0

    # For each metric: compute mean and std across regions
    comparisons: list[MetricComparison] = []
    valid_metrics: list[str] = []
    for mc in metric_columns:
        values = [region_avgs[r][mc] for r in all_regions]
        n = len(values)
        if n == 0:
            continue
        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / n if n > 0 else 0.0
        std_val = math.sqrt(variance)

        # Best / worst
        best_r = max(all_regions, key=lambda r: region_avgs[r][mc])
        worst_r = min(all_regions, key=lambda r: region_avgs[r][mc])

        comparisons.append(MetricComparison(
            metric=mc,
            best_region=best_r,
            best_value=round(region_avgs[best_r][mc], 2),
            worst_region=worst_r,
            worst_value=round(region_avgs[worst_r][mc], 2),
            avg_value=round(mean_val, 2),
            std_dev=round(std_val, 2),
        ))
        valid_metrics.append(mc)

    if not comparisons:
        return None

    # z-scores per region per metric, overall score = mean of z-scores
    region_scores: list[RegionScore] = []
    for r in all_regions:
        z_scores: dict[str, float] = {}
        for comp in comparisons:
            mc = comp.metric
            val = region_avgs[r][mc]
            if comp.std_dev > 0:
                z = (val - comp.avg_value) / comp.std_dev
            else:
                z = 0.0
            z_scores[mc] = round(z, 2)

        overall = sum(z_scores.values()) / len(z_scores) if z_scores else 0.0
        region_scores.append(RegionScore(
            region=r,
            metrics=z_scores,
            overall_score=round(overall, 2),
        ))

    # Sort region_scores descending by overall_score
    region_scores.sort(key=lambda rs: rs.overall_score, reverse=True)
    best_overall = region_scores[0].region

    summary = (
        f"Region Comparison: {len(all_regions)} regions across "
        f"{len(comparisons)} metrics. "
        f"Best overall: {best_overall} (score {region_scores[0].overall_score:.2f})."
    )

    return RegionComparisonResult(
        comparisons=comparisons,
        region_scores=region_scores,
        best_overall=best_overall,
        summary=summary,
    )


def analyze_geographic_growth(
    rows: list[dict],
    region_column: str,
    value_column: str,
    date_column: str,
) -> GeoGrowthResult | None:
    """Growth rates per region over time.

    Groups rows by region and date period, computes growth from the
    earliest to the latest period for each region.

    Args:
        rows: Data as list of dicts.
        region_column: Column identifying the region.
        value_column: Column with the value to track.
        date_column: Column with the date/period.

    Returns:
        GeoGrowthResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per region per period (YYYY-MM)
    region_periods: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    has_any = False

    for row in rows:
        region = row.get(region_column)
        val = _safe_float(row.get(value_column))
        dt = _parse_date(row.get(date_column))
        if region is None or val is None or dt is None:
            continue

        region_str = str(region)
        period_key = dt.strftime("%Y-%m")
        region_periods[region_str][period_key] += val
        has_any = True

    if not has_any or not region_periods:
        return None

    # Compute growth per region
    growths: list[RegionGrowth] = []
    for region_name in sorted(region_periods.keys()):
        periods = region_periods[region_name]
        if not periods:
            continue

        sorted_periods = sorted(periods.keys())
        n_periods = len(sorted_periods)
        first_val = periods[sorted_periods[0]]
        last_val = periods[sorted_periods[-1]]

        if first_val != 0:
            growth_rate = (last_val - first_val) / abs(first_val) * 100
        elif last_val != 0:
            # From zero to something: treat as infinite growth, cap at large value
            growth_rate = float("inf")
        else:
            growth_rate = 0.0

        growths.append(RegionGrowth(
            region=region_name,
            first_period_value=round(first_val, 2),
            last_period_value=round(last_val, 2),
            growth_rate=round(growth_rate, 2) if math.isfinite(growth_rate) else growth_rate,
            periods=n_periods,
        ))

    if not growths:
        return None

    # Filter finite growth rates for comparison
    finite_growths = [g for g in growths if math.isfinite(g.growth_rate)]

    if finite_growths:
        fastest = max(finite_growths, key=lambda g: g.growth_rate)
        slowest = min(finite_growths, key=lambda g: g.growth_rate)
        avg_growth = round(
            sum(g.growth_rate for g in finite_growths) / len(finite_growths), 2
        )
    else:
        # All infinite -- fallback
        fastest = growths[0]
        slowest = growths[0]
        avg_growth = 0.0

    summary = (
        f"Geographic Growth: {len(growths)} regions analyzed. "
        f"Fastest growing: {fastest.region} ({fastest.growth_rate:.1f}%). "
        f"Slowest growing: {slowest.region} ({slowest.growth_rate:.1f}%). "
        f"Average growth: {avg_growth:.1f}%."
    )

    return GeoGrowthResult(
        regions=growths,
        fastest_growing=fastest.region,
        slowest_growing=slowest.region,
        avg_growth=avg_growth,
        summary=summary,
    )


def compute_market_penetration(
    rows: list[dict],
    region_column: str,
    customer_column: str,
    potential_column: str | None = None,
) -> MarketPenetrationResult | None:
    """Unique customers per region and market penetration.

    If potential_column is provided, penetration = (actual / potential) * 100.
    Otherwise only customer density ranking is produced.

    Args:
        rows: Data as list of dicts.
        region_column: Column identifying the region.
        customer_column: Column identifying the customer.
        potential_column: Optional column with market potential (numeric).

    Returns:
        MarketPenetrationResult or None if insufficient data.
    """
    if not rows:
        return None

    # Collect unique customers per region + potential
    region_customers: dict[str, set] = defaultdict(set)
    region_potential: dict[str, list[float]] = defaultdict(list)
    has_any = False

    for row in rows:
        region = row.get(region_column)
        customer = row.get(customer_column)
        if region is None or customer is None:
            continue

        region_str = str(region)
        cust_str = str(customer)
        region_customers[region_str].add(cust_str)
        has_any = True

        if potential_column is not None:
            pot = _safe_float(row.get(potential_column))
            if pot is not None:
                region_potential[region_str].append(pot)

    if not has_any or not region_customers:
        return None

    # Compute penetration per region
    penetrations: list[dict] = []
    all_customers: set[str] = set()
    for region_name in sorted(region_customers.keys()):
        custs = region_customers[region_name]
        all_customers.update(custs)
        count = len(custs)

        potential: float | None = None
        pen_pct: float | None = None
        if potential_column is not None and region_potential.get(region_name):
            # Use the max potential value for the region (assumed to be region-level)
            potential = max(region_potential[region_name])
            if potential > 0:
                pen_pct = round(count / potential * 100, 2)
            else:
                pen_pct = 0.0

        penetrations.append({
            "region": region_name,
            "customer_count": count,
            "potential": round(potential, 2) if potential is not None else None,
            "penetration_pct": pen_pct,
        })

    # Rank: by penetration_pct if available, else by customer_count descending
    if potential_column is not None and any(p["penetration_pct"] is not None for p in penetrations):
        penetrations.sort(
            key=lambda p: p["penetration_pct"] if p["penetration_pct"] is not None else -1,
            reverse=True,
        )
    else:
        penetrations.sort(key=lambda p: p["customer_count"], reverse=True)

    region_pen_list: list[RegionPenetration] = []
    for rank, p in enumerate(penetrations, start=1):
        region_pen_list.append(RegionPenetration(
            region=p["region"],
            customer_count=p["customer_count"],
            potential=p["potential"],
            penetration_pct=p["penetration_pct"],
            rank=rank,
        ))

    total_customers = len(all_customers)
    total_regions = len(region_customers)

    # Best penetration
    best_pen: str | None = None
    if potential_column is not None:
        pen_regions = [rp for rp in region_pen_list if rp.penetration_pct is not None]
        if pen_regions:
            best_pen = max(pen_regions, key=lambda rp: rp.penetration_pct).region  # type: ignore[arg-type]

    # Untapped regions: regions with 0 customers (shouldn't normally happen
    # but can if rows exist with region but null customer after filtering)
    # OR regions with penetration < 10%
    untapped: list[str] = []
    if potential_column is not None:
        for rp in region_pen_list:
            if rp.penetration_pct is not None and rp.penetration_pct < 10.0:
                untapped.append(rp.region)
    else:
        # Without potential, consider regions with the fewest customers
        if len(region_pen_list) > 1:
            avg_count = total_customers / total_regions if total_regions > 0 else 0
            for rp in region_pen_list:
                if rp.customer_count < avg_count * 0.5:
                    untapped.append(rp.region)

    summary_parts = [
        f"Market Penetration: {total_regions} regions, "
        f"{total_customers} unique customers.",
    ]
    if best_pen is not None:
        summary_parts.append(f"Best penetration: {best_pen}.")
    if untapped:
        summary_parts.append(f"Untapped regions: {', '.join(untapped)}.")

    return MarketPenetrationResult(
        regions=region_pen_list,
        total_customers=total_customers,
        total_regions=total_regions,
        best_penetration=best_pen,
        untapped_regions=untapped,
        summary=" ".join(summary_parts),
    )


def format_geo_report(
    distribution: RegionalDistributionResult | None = None,
    comparison: RegionComparisonResult | None = None,
    growth: GeoGrowthResult | None = None,
    penetration: MarketPenetrationResult | None = None,
) -> str:
    """Format a combined geographic analytics report as plain text.

    Any parameter may be ``None``; only provided sections are rendered.

    Returns:
        Formatted multi-line report string.
    """
    lines: list[str] = []

    if distribution is not None:
        lines.append("Regional Distribution Report")
        lines.append("=" * 60)
        lines.append(f"Total Value:          ${distribution.total_value:,.2f}")
        lines.append(f"Total Count:          {distribution.total_count}")
        lines.append(f"Top Region:           {distribution.top_region}")
        lines.append(f"Concentration Ratio:  {distribution.concentration_ratio:.1f}%")
        lines.append(f"HHI Index:            {distribution.hhi_index:.0f}")
        lines.append("")
        lines.append(
            f"{'Region':<20}{'Value':>14}{'Share':>8}{'Count':>8}{'Avg':>12}{'Rank':>6}"
        )
        lines.append("-" * 68)
        for rm in distribution.regions:
            lines.append(
                f"{rm.region:<20}{rm.total_value:>14,.2f}"
                f"{rm.share_pct:>7.1f}%{rm.count:>8}"
                f"{rm.avg_value:>12,.2f}{rm.rank:>6}"
            )
        lines.append("")

    if comparison is not None:
        lines.append("Region Comparison Report")
        lines.append("=" * 60)
        lines.append(f"Best Overall Region:  {comparison.best_overall}")
        lines.append("")
        lines.append("Metric Comparisons:")
        lines.append(
            f"{'Metric':<20}{'Best Region':<16}{'Best':>10}"
            f"{'Worst Region':<16}{'Worst':>10}{'Avg':>10}{'StdDev':>8}"
        )
        lines.append("-" * 90)
        for mc in comparison.comparisons:
            lines.append(
                f"{mc.metric:<20}{mc.best_region:<16}{mc.best_value:>10,.2f}"
                f"{mc.worst_region:<16}{mc.worst_value:>10,.2f}"
                f"{mc.avg_value:>10,.2f}{mc.std_dev:>8.2f}"
            )
        lines.append("")
        lines.append("Region Scores:")
        lines.append(f"{'Region':<20}{'Overall Score':>14}")
        lines.append("-" * 34)
        for rs in comparison.region_scores:
            lines.append(f"{rs.region:<20}{rs.overall_score:>14.2f}")
        lines.append("")

    if growth is not None:
        lines.append("Geographic Growth Report")
        lines.append("=" * 60)
        lines.append(f"Fastest Growing:      {growth.fastest_growing}")
        lines.append(f"Slowest Growing:      {growth.slowest_growing}")
        lines.append(f"Average Growth:       {growth.avg_growth:.1f}%")
        lines.append("")
        lines.append(
            f"{'Region':<20}{'First':>12}{'Last':>12}{'Growth':>10}{'Periods':>8}"
        )
        lines.append("-" * 62)
        for rg in growth.regions:
            gr_str = f"{rg.growth_rate:.1f}%" if math.isfinite(rg.growth_rate) else "inf%"
            lines.append(
                f"{rg.region:<20}{rg.first_period_value:>12,.2f}"
                f"{rg.last_period_value:>12,.2f}{gr_str:>10}{rg.periods:>8}"
            )
        lines.append("")

    if penetration is not None:
        lines.append("Market Penetration Report")
        lines.append("=" * 60)
        lines.append(f"Total Customers:      {penetration.total_customers}")
        lines.append(f"Total Regions:        {penetration.total_regions}")
        if penetration.best_penetration is not None:
            lines.append(f"Best Penetration:     {penetration.best_penetration}")
        if penetration.untapped_regions:
            lines.append(
                f"Untapped Regions:     {', '.join(penetration.untapped_regions)}"
            )
        lines.append("")
        lines.append(
            f"{'Region':<20}{'Customers':>10}{'Potential':>12}{'Penetration':>12}{'Rank':>6}"
        )
        lines.append("-" * 60)
        for rp in penetration.regions:
            pot_str = f"{rp.potential:>12,.0f}" if rp.potential is not None else f"{'N/A':>12}"
            pen_str = f"{rp.penetration_pct:>10.1f}%" if rp.penetration_pct is not None else f"{'N/A':>11}"
            lines.append(
                f"{rp.region:<20}{rp.customer_count:>10}"
                f"{pot_str}{pen_str}{rp.rank:>6}"
            )
        lines.append("")

    if not lines:
        return "No data provided for geographic report."

    return "\n".join(lines)
