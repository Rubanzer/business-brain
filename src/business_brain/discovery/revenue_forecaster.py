"""Revenue forecasting and analysis.

Pure functions for forecasting revenue, analyzing revenue segments,
computing growth rates, and identifying revenue drivers from tabular
business data.  No DB, async, or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    if isinstance(val, bool):
        return float(val)
    try:
        return float(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_date(val) -> datetime | None:
    """Attempt to parse a date from string or datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(val), fmt)
        except (ValueError, TypeError):
            continue
    return None


def _month_key(dt: datetime) -> str:
    """Return 'YYYY-MM' string for a datetime."""
    return dt.strftime("%Y-%m")


def _quarter_key(dt: datetime) -> str:
    """Return 'YYYY-Q#' string for a datetime."""
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def _next_month(year: int, month: int) -> tuple[int, int]:
    """Return (year, month) for the next month."""
    month += 1
    if month > 12:
        month = 1
        year += 1
    return year, month


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PeriodRevenue:
    """Revenue for a single period."""
    period: str
    revenue: float
    growth_rate: float | None


@dataclass
class RevenueForecastResult:
    """Complete revenue forecast result."""
    periods: list[PeriodRevenue]
    forecasts: list[PeriodRevenue]
    avg_growth_rate: float
    trend: str
    total_historical: float
    total_forecast: float
    summary: str


@dataclass
class SegmentRevenue:
    """Revenue for a single segment."""
    segment: str
    revenue: float
    share_pct: float
    rank: int
    transaction_count: int


@dataclass
class RevenueSegmentResult:
    """Complete revenue segment analysis result."""
    segments: list[SegmentRevenue]
    total_revenue: float
    top_segment: str
    concentration_index: float
    summary: str


@dataclass
class GrowthPeriod:
    """Growth metrics for a single period."""
    period: str
    revenue: float
    growth_rate: float
    growth_absolute: float


@dataclass
class RevenueGrowthResult:
    """Complete revenue growth analysis result."""
    periods: list[GrowthPeriod]
    cagr: float | None
    avg_growth: float
    best_period: str
    worst_period: str
    volatility: float
    summary: str


@dataclass
class DriverCorrelation:
    """Correlation between a driver and revenue."""
    driver: str
    correlation: float
    direction: str
    avg_when_high_revenue: float
    avg_when_low_revenue: float


@dataclass
class RevenueDriverResult:
    """Complete revenue driver analysis result."""
    drivers: list[DriverCorrelation]
    top_driver: str
    summary: str


# ---------------------------------------------------------------------------
# 1. forecast_revenue
# ---------------------------------------------------------------------------


def forecast_revenue(
    rows: list[dict],
    revenue_column: str,
    date_column: str,
    periods_ahead: int = 3,
) -> RevenueForecastResult | None:
    """Forecast future revenue using linear extrapolation.

    Groups revenue by month, computes period-over-period growth rates,
    and projects future periods using a simple linear trend.

    Args:
        rows: Data rows as dicts.
        revenue_column: Column with revenue amounts.
        date_column: Column with date values.
        periods_ahead: Number of future periods to forecast.

    Returns:
        RevenueForecastResult or None if insufficient data.
    """
    if not rows or periods_ahead <= 0:
        return None

    # Aggregate revenue by month
    monthly_revenue: dict[str, float] = {}

    for row in rows:
        rev = _safe_float(row.get(revenue_column))
        dt = _parse_date(row.get(date_column))
        if rev is None or dt is None:
            continue
        mk = _month_key(dt)
        monthly_revenue[mk] = monthly_revenue.get(mk, 0.0) + rev

    if not monthly_revenue:
        return None

    sorted_months = sorted(monthly_revenue.keys())
    n = len(sorted_months)
    revenue_values = [monthly_revenue[m] for m in sorted_months]

    # Build historical periods with growth rates
    periods: list[PeriodRevenue] = []
    growth_rates: list[float] = []

    for i, month in enumerate(sorted_months):
        rev = revenue_values[i]
        if i == 0:
            gr = None
        else:
            prev_rev = revenue_values[i - 1]
            if prev_rev != 0:
                gr = round((rev - prev_rev) / abs(prev_rev) * 100, 2)
            else:
                gr = 0.0
            growth_rates.append(gr)
        periods.append(PeriodRevenue(
            period=month,
            revenue=round(rev, 2),
            growth_rate=gr,
        ))

    avg_growth_rate = round(sum(growth_rates) / len(growth_rates), 2) if growth_rates else 0.0
    total_historical = round(sum(revenue_values), 2)

    # Linear regression for forecasting: y = intercept + slope * x
    if n >= 2:
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(revenue_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, revenue_values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        else:
            slope = 0.0
            intercept = y_mean
    else:
        slope = 0.0
        intercept = revenue_values[0]

    # Trend direction
    if slope > 0.01:
        trend = "increasing"
    elif slope < -0.01:
        trend = "decreasing"
    else:
        trend = "stable"

    # Generate forecasts
    last_month_str = sorted_months[-1]
    last_dt = datetime.strptime(last_month_str + "-01", "%Y-%m-%d")
    cur_year, cur_month = last_dt.year, last_dt.month

    forecasts: list[PeriodRevenue] = []
    prev_forecast_rev = revenue_values[-1]

    for i in range(1, periods_ahead + 1):
        x_idx = n - 1 + i
        projected = intercept + slope * x_idx
        projected = round(max(projected, 0.0), 2)

        cur_year, cur_month = _next_month(cur_year, cur_month)
        period_label = f"{cur_year:04d}-{cur_month:02d}"

        if prev_forecast_rev != 0:
            gr = round((projected - prev_forecast_rev) / abs(prev_forecast_rev) * 100, 2)
        else:
            gr = 0.0

        forecasts.append(PeriodRevenue(
            period=period_label,
            revenue=projected,
            growth_rate=gr,
        ))
        prev_forecast_rev = projected

    total_forecast = round(sum(f.revenue for f in forecasts), 2)

    summary = (
        f"Revenue forecast: {n} historical periods, "
        f"{periods_ahead} forecasted. "
        f"Avg growth rate: {avg_growth_rate}%. "
        f"Trend: {trend}. "
        f"Total historical: {total_historical:,.2f}, "
        f"total forecast: {total_forecast:,.2f}."
    )

    return RevenueForecastResult(
        periods=periods,
        forecasts=forecasts,
        avg_growth_rate=avg_growth_rate,
        trend=trend,
        total_historical=total_historical,
        total_forecast=total_forecast,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. analyze_revenue_segments
# ---------------------------------------------------------------------------


def analyze_revenue_segments(
    rows: list[dict],
    revenue_column: str,
    segment_column: str,
    date_column: str | None = None,
) -> RevenueSegmentResult | None:
    """Analyze revenue breakdown by segment.

    Computes each segment's total revenue, share of total, rank, and
    transaction count.

    Args:
        rows: Data rows as dicts.
        revenue_column: Column with revenue amounts.
        segment_column: Column with segment identifiers.
        date_column: Optional date column (reserved for future period filtering).

    Returns:
        RevenueSegmentResult or None if insufficient data.
    """
    if not rows:
        return None

    segment_data: dict[str, dict] = {}
    total_revenue = 0.0
    valid_count = 0

    for row in rows:
        rev = _safe_float(row.get(revenue_column))
        seg = row.get(segment_column)
        if rev is None or seg is None:
            continue

        seg_key = str(seg)
        valid_count += 1
        total_revenue += rev

        if seg_key not in segment_data:
            segment_data[seg_key] = {"revenue": 0.0, "count": 0}
        segment_data[seg_key]["revenue"] += rev
        segment_data[seg_key]["count"] += 1

    if valid_count == 0 or not segment_data:
        return None

    # Sort by revenue descending for ranking
    sorted_segments = sorted(
        segment_data.items(),
        key=lambda item: item[1]["revenue"],
        reverse=True,
    )

    segments: list[SegmentRevenue] = []
    for rank, (seg_key, data) in enumerate(sorted_segments, start=1):
        share_pct = round(data["revenue"] / total_revenue * 100, 2) if total_revenue != 0 else 0.0
        segments.append(SegmentRevenue(
            segment=seg_key,
            revenue=round(data["revenue"], 2),
            share_pct=share_pct,
            rank=rank,
            transaction_count=data["count"],
        ))

    top_segment = segments[0].segment

    # Herfindahl-Hirschman Index (concentration)
    # Sum of squared market shares (as fractions)
    if total_revenue != 0:
        concentration_index = round(
            sum((d["revenue"] / total_revenue) ** 2 for d in segment_data.values()),
            4,
        )
    else:
        concentration_index = 0.0

    summary = (
        f"Revenue segment analysis: {len(segments)} segments, "
        f"total revenue={total_revenue:,.2f}. "
        f"Top segment: {top_segment} "
        f"({segments[0].share_pct}% share). "
        f"Concentration index (HHI): {concentration_index}."
    )

    return RevenueSegmentResult(
        segments=segments,
        total_revenue=round(total_revenue, 2),
        top_segment=top_segment,
        concentration_index=concentration_index,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. compute_revenue_growth
# ---------------------------------------------------------------------------


def compute_revenue_growth(
    rows: list[dict],
    revenue_column: str,
    date_column: str,
    comparison: str = "period_over_period",
) -> RevenueGrowthResult | None:
    """Compute period-over-period revenue growth rates.

    Groups revenue by month, computes growth rates between consecutive
    periods, CAGR if multiple years are present, and identifies best/worst
    performing periods.

    Args:
        rows: Data rows as dicts.
        revenue_column: Column with revenue amounts.
        date_column: Column with date values.
        comparison: Comparison mode (currently only "period_over_period").

    Returns:
        RevenueGrowthResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate revenue by month
    monthly_revenue: dict[str, float] = {}

    for row in rows:
        rev = _safe_float(row.get(revenue_column))
        dt = _parse_date(row.get(date_column))
        if rev is None or dt is None:
            continue
        mk = _month_key(dt)
        monthly_revenue[mk] = monthly_revenue.get(mk, 0.0) + rev

    if not monthly_revenue:
        return None

    sorted_months = sorted(monthly_revenue.keys())
    n = len(sorted_months)
    revenue_values = [monthly_revenue[m] for m in sorted_months]

    # Build growth periods (skip first period -- no prior to compare)
    growth_periods: list[GrowthPeriod] = []
    growth_rates: list[float] = []

    for i in range(1, n):
        prev_rev = revenue_values[i - 1]
        cur_rev = revenue_values[i]
        growth_abs = round(cur_rev - prev_rev, 2)

        if prev_rev != 0:
            gr = round((cur_rev - prev_rev) / abs(prev_rev) * 100, 2)
        else:
            gr = 0.0

        growth_rates.append(gr)
        growth_periods.append(GrowthPeriod(
            period=sorted_months[i],
            revenue=round(cur_rev, 2),
            growth_rate=gr,
            growth_absolute=growth_abs,
        ))

    if not growth_periods:
        # Only one period -- return minimal result
        return RevenueGrowthResult(
            periods=[],
            cagr=None,
            avg_growth=0.0,
            best_period=sorted_months[0],
            worst_period=sorted_months[0],
            volatility=0.0,
            summary=(
                f"Revenue growth: only 1 period ({sorted_months[0]}), "
                f"revenue={revenue_values[0]:,.2f}. "
                f"Insufficient data for growth calculation."
            ),
        )

    avg_growth = round(sum(growth_rates) / len(growth_rates), 2)

    # Best / worst periods
    best_gp = max(growth_periods, key=lambda gp: gp.growth_rate)
    worst_gp = min(growth_periods, key=lambda gp: gp.growth_rate)
    best_period = best_gp.period
    worst_period = worst_gp.period

    # Volatility (standard deviation of growth rates)
    if len(growth_rates) >= 2:
        mean_gr = sum(growth_rates) / len(growth_rates)
        variance = sum((g - mean_gr) ** 2 for g in growth_rates) / len(growth_rates)
        volatility = round(math.sqrt(variance), 2)
    else:
        volatility = 0.0

    # CAGR calculation
    # Parse years from first and last months
    first_dt = datetime.strptime(sorted_months[0] + "-01", "%Y-%m-%d")
    last_dt = datetime.strptime(sorted_months[-1] + "-01", "%Y-%m-%d")
    year_diff = (last_dt.year - first_dt.year) + (last_dt.month - first_dt.month) / 12.0

    cagr: float | None = None
    if year_diff >= 1.0 and revenue_values[0] > 0 and revenue_values[-1] > 0:
        try:
            cagr = round(
                (math.pow(revenue_values[-1] / revenue_values[0], 1.0 / year_diff) - 1) * 100,
                2,
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            cagr = None

    cagr_str = f"{cagr}%" if cagr is not None else "N/A"
    summary = (
        f"Revenue growth analysis: {len(growth_periods)} periods compared. "
        f"Avg growth: {avg_growth}%. CAGR: {cagr_str}. "
        f"Best period: {best_period} ({best_gp.growth_rate}%), "
        f"worst period: {worst_period} ({worst_gp.growth_rate}%). "
        f"Volatility: {volatility}%."
    )

    return RevenueGrowthResult(
        periods=growth_periods,
        cagr=cagr,
        avg_growth=avg_growth,
        best_period=best_period,
        worst_period=worst_period,
        volatility=volatility,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. analyze_revenue_drivers
# ---------------------------------------------------------------------------


def analyze_revenue_drivers(
    rows: list[dict],
    revenue_column: str,
    driver_columns: list[str],
) -> RevenueDriverResult | None:
    """Analyze which driver columns correlate with revenue.

    For each driver column, computes Pearson correlation with revenue,
    determines direction (positive/negative), and computes average driver
    value when revenue is above/below median.

    Args:
        rows: Data rows as dicts.
        revenue_column: Column with revenue amounts.
        driver_columns: List of column names to test as drivers.

    Returns:
        RevenueDriverResult or None if insufficient data.
    """
    if not rows or not driver_columns:
        return None

    # Collect valid revenue values
    rev_values: list[float] = []
    valid_rows: list[dict] = []

    for row in rows:
        rev = _safe_float(row.get(revenue_column))
        if rev is not None:
            rev_values.append(rev)
            valid_rows.append(row)

    if len(rev_values) < 2:
        return None

    rev_mean = sum(rev_values) / len(rev_values)

    # Compute median revenue for high/low split
    sorted_rev = sorted(rev_values)
    mid = len(sorted_rev) // 2
    if len(sorted_rev) % 2 == 0:
        rev_median = (sorted_rev[mid - 1] + sorted_rev[mid]) / 2.0
    else:
        rev_median = sorted_rev[mid]

    drivers: list[DriverCorrelation] = []

    for dcol in driver_columns:
        # Collect paired (revenue, driver) values
        paired_rev: list[float] = []
        paired_drv: list[float] = []

        for i, row in enumerate(valid_rows):
            drv = _safe_float(row.get(dcol))
            if drv is not None:
                paired_rev.append(rev_values[i])
                paired_drv.append(drv)

        if len(paired_rev) < 2:
            continue

        n = len(paired_rev)
        r_mean = sum(paired_rev) / n
        d_mean = sum(paired_drv) / n

        # Pearson correlation
        numerator = sum(
            (r - r_mean) * (d - d_mean)
            for r, d in zip(paired_rev, paired_drv)
        )
        denom_r = math.sqrt(sum((r - r_mean) ** 2 for r in paired_rev))
        denom_d = math.sqrt(sum((d - d_mean) ** 2 for d in paired_drv))

        if denom_r > 0 and denom_d > 0:
            correlation = round(numerator / (denom_r * denom_d), 4)
        else:
            correlation = 0.0

        # Direction
        if correlation > 0:
            direction = "positive"
        elif correlation < 0:
            direction = "negative"
        else:
            direction = "neutral"

        # Average driver value when revenue is high vs low
        high_drv: list[float] = []
        low_drv: list[float] = []

        for i, row in enumerate(valid_rows):
            drv = _safe_float(row.get(dcol))
            if drv is not None:
                if rev_values[i] >= rev_median:
                    high_drv.append(drv)
                else:
                    low_drv.append(drv)

        avg_high = round(sum(high_drv) / len(high_drv), 2) if high_drv else 0.0
        avg_low = round(sum(low_drv) / len(low_drv), 2) if low_drv else 0.0

        drivers.append(DriverCorrelation(
            driver=dcol,
            correlation=correlation,
            direction=direction,
            avg_when_high_revenue=avg_high,
            avg_when_low_revenue=avg_low,
        ))

    if not drivers:
        return None

    # Sort by absolute correlation descending
    drivers.sort(key=lambda d: abs(d.correlation), reverse=True)
    top_driver = drivers[0].driver

    summary = (
        f"Revenue driver analysis: {len(drivers)} drivers analyzed. "
        f"Top driver: {top_driver} "
        f"(correlation={drivers[0].correlation}, {drivers[0].direction}). "
        + ", ".join(
            f"{d.driver}={d.correlation}" for d in drivers
        )
        + "."
    )

    return RevenueDriverResult(
        drivers=drivers,
        top_driver=top_driver,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_revenue_report
# ---------------------------------------------------------------------------


def format_revenue_report(
    forecast: RevenueForecastResult | None = None,
    segments: RevenueSegmentResult | None = None,
    growth: RevenueGrowthResult | None = None,
    drivers: RevenueDriverResult | None = None,
) -> str:
    """Combine revenue analysis results into a text report.

    Args:
        forecast: Optional revenue forecast result.
        segments: Optional segment analysis result.
        growth: Optional growth analysis result.
        drivers: Optional driver analysis result.

    Returns:
        Combined text report string.
    """
    sections: list[str] = []

    if forecast is not None:
        lines = ["=== Revenue Forecast ==="]
        lines.append(f"Trend: {forecast.trend}")
        lines.append(f"Avg Growth Rate: {forecast.avg_growth_rate}%")
        lines.append(f"Total Historical Revenue: {forecast.total_historical:,.2f}")
        lines.append(f"Total Forecast Revenue: {forecast.total_forecast:,.2f}")
        if forecast.periods:
            lines.append("Historical Periods:")
            for p in forecast.periods:
                gr_str = f"{p.growth_rate}%" if p.growth_rate is not None else "N/A"
                lines.append(
                    f"  {p.period}: revenue={p.revenue:,.2f}, growth={gr_str}"
                )
        if forecast.forecasts:
            lines.append("Forecasted Periods:")
            for f_ in forecast.forecasts:
                gr_str = f"{f_.growth_rate}%" if f_.growth_rate is not None else "N/A"
                lines.append(
                    f"  {f_.period}: revenue={f_.revenue:,.2f}, growth={gr_str}"
                )
        sections.append("\n".join(lines))

    if segments is not None:
        lines = ["=== Revenue Segments ==="]
        lines.append(f"Total Revenue: {segments.total_revenue:,.2f}")
        lines.append(f"Top Segment: {segments.top_segment}")
        lines.append(f"Concentration Index (HHI): {segments.concentration_index}")
        if segments.segments:
            lines.append("Segment Breakdown:")
            for s in segments.segments:
                lines.append(
                    f"  #{s.rank} {s.segment}: revenue={s.revenue:,.2f}, "
                    f"share={s.share_pct}%, transactions={s.transaction_count}"
                )
        sections.append("\n".join(lines))

    if growth is not None:
        lines = ["=== Revenue Growth ==="]
        lines.append(f"Avg Growth: {growth.avg_growth}%")
        cagr_str = f"{growth.cagr}%" if growth.cagr is not None else "N/A"
        lines.append(f"CAGR: {cagr_str}")
        lines.append(f"Best Period: {growth.best_period}")
        lines.append(f"Worst Period: {growth.worst_period}")
        lines.append(f"Volatility: {growth.volatility}%")
        if growth.periods:
            lines.append("Period Growth:")
            for gp in growth.periods:
                lines.append(
                    f"  {gp.period}: revenue={gp.revenue:,.2f}, "
                    f"growth={gp.growth_rate}% ({gp.growth_absolute:+,.2f})"
                )
        sections.append("\n".join(lines))

    if drivers is not None:
        lines = ["=== Revenue Drivers ==="]
        lines.append(f"Top Driver: {drivers.top_driver}")
        if drivers.drivers:
            lines.append("Driver Correlations:")
            for d in drivers.drivers:
                lines.append(
                    f"  {d.driver}: correlation={d.correlation}, "
                    f"direction={d.direction}, "
                    f"avg_high_rev={d.avg_when_high_revenue:,.2f}, "
                    f"avg_low_rev={d.avg_when_low_revenue:,.2f}"
                )
        sections.append("\n".join(lines))

    if not sections:
        return "No revenue data available for report."

    return "\n\n".join(sections)
