"""Demand forecaster — demand pattern analysis, moving averages, smoothing, seasonality.

Pure functions for analyzing historical demand data to find patterns and
generate forecasts.  Computes demand variability classification, moving
averages, exponential smoothing (with optimal alpha search), and seasonal
indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
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
class ProductDemand:
    """Demand metrics for a single product."""
    product: str
    total_demand: float
    periods: int
    avg_per_period: float
    max_period_demand: float
    min_period_demand: float
    std_dev: float
    cv: float                # coefficient of variation (%)
    pattern: str             # "steady", "variable", "lumpy", "erratic"
    adi: float | None        # Average Demand Interval (None if no zero periods)
    trend: str               # "increasing", "decreasing", "stable"


@dataclass
class DemandPatternResult:
    """Aggregate demand pattern analysis across all products."""
    products: list[ProductDemand]
    overall_trend: str       # "increasing", "decreasing", "stable"
    summary: str


@dataclass
class MovingAvgPoint:
    """Single point in a moving average series."""
    period: str
    actual: float
    moving_avg: float | None
    error: float | None      # actual - moving_avg


@dataclass
class MovingAvgResult:
    """Result of moving average computation."""
    points: list[MovingAvgPoint]
    window: int
    mad: float | None        # Mean Absolute Deviation
    mape: float | None       # Mean Absolute Percentage Error (%)


@dataclass
class SmoothingPoint:
    """Single point in an exponential smoothing series."""
    period: str
    actual: float | None     # None for the one-step-ahead forecast
    forecast: float


@dataclass
class SmoothingResult:
    """Result of exponential smoothing."""
    points: list[SmoothingPoint]
    alpha: float
    mad: float | None
    mape: float | None
    optimal_alpha: float
    next_forecast: float


@dataclass
class SeasonalPeriod:
    """Seasonal index for a single period."""
    period: str
    avg_demand: float
    seasonal_index: float    # period_avg / overall_avg


@dataclass
class SeasonalityResult:
    """Result of seasonality detection."""
    periods: list[SeasonalPeriod]
    overall_avg: float
    peak_seasons: list[str]      # index > 1.2
    low_seasons: list[str]       # index < 0.8
    seasonal_strength: float     # max_index - min_index
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _group_by_period(
    rows: list[dict],
    quantity_column: str,
    date_column: str,
) -> list[tuple[str, float]]:
    """Group rows by date_column period and sum quantities.

    Returns sorted list of (period, total_qty) tuples.
    """
    period_sums: dict[str, float] = {}

    for row in rows:
        period_raw = row.get(date_column)
        qty_raw = row.get(quantity_column)
        if period_raw is None or qty_raw is None:
            continue
        qty = _safe_float(qty_raw)
        if qty is None:
            continue
        period = str(period_raw)
        period_sums[period] = period_sums.get(period, 0.0) + qty

    # Sort by period string (works for ISO dates, month names with prefix, etc.)
    return sorted(period_sums.items(), key=lambda t: t[0])


def _classify_pattern(cv: float) -> str:
    """Classify demand pattern based on coefficient of variation (%)."""
    if cv < 20:
        return "steady"
    elif cv < 50:
        return "variable"
    elif cv < 80:
        return "lumpy"
    else:
        return "erratic"


def _compute_trend(values: list[float]) -> str:
    """Determine overall trend from a list of periodic values.

    Uses simple linear regression slope direction.
    """
    n = len(values)
    if n < 2:
        return "stable"

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = 0.0
    denominator = 0.0
    for i, y in enumerate(values):
        numerator += (i - x_mean) * (y - y_mean)
        denominator += (i - x_mean) ** 2

    if denominator == 0:
        return "stable"

    slope = numerator / denominator

    # Normalise slope by mean to get relative trend strength
    if y_mean != 0:
        relative_slope = slope / abs(y_mean)
    else:
        relative_slope = slope

    if relative_slope > 0.05:
        return "increasing"
    elif relative_slope < -0.05:
        return "decreasing"
    else:
        return "stable"


def _compute_adi(period_demands: list[float]) -> float | None:
    """Compute Average Demand Interval.

    ADI = number of periods / number of non-zero demand periods.
    Returns None if all periods have non-zero demand.
    """
    non_zero = sum(1 for d in period_demands if d != 0)
    if non_zero == 0:
        return None
    if non_zero == len(period_demands):
        return None  # no intermittent demand
    return len(period_demands) / non_zero


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def analyze_demand_pattern(
    rows: list[dict],
    product_column: str,
    quantity_column: str,
    date_column: str,
) -> DemandPatternResult | None:
    """Analyze demand patterns across products and periods.

    Per product computes: total demand, avg per period, max/min period,
    variability (coefficient of variation), pattern classification
    (steady/variable/lumpy/erratic), ADI for intermittent demand,
    and trend direction.

    Args:
        rows: Data rows as dicts.
        product_column: Column identifying the product/SKU.
        quantity_column: Column with demand quantity.
        date_column: Column with period identifier.

    Returns:
        DemandPatternResult or None if no valid data.
    """
    if not rows:
        return None

    # Group by product, then by period
    product_periods: dict[str, dict[str, float]] = {}
    for row in rows:
        product_raw = row.get(product_column)
        period_raw = row.get(date_column)
        qty_raw = row.get(quantity_column)
        if product_raw is None or period_raw is None or qty_raw is None:
            continue
        qty = _safe_float(qty_raw)
        if qty is None:
            continue
        product = str(product_raw)
        period = str(period_raw)
        if product not in product_periods:
            product_periods[product] = {}
        product_periods[product][period] = (
            product_periods[product].get(period, 0.0) + qty
        )

    if not product_periods:
        return None

    products: list[ProductDemand] = []
    all_period_values: list[float] = []

    for product, periods_map in sorted(product_periods.items()):
        demands = [periods_map[p] for p in sorted(periods_map)]
        n = len(demands)
        if n == 0:
            continue

        total = sum(demands)
        avg = total / n
        max_d = max(demands)
        min_d = min(demands)

        if n >= 2:
            variance = sum((d - avg) ** 2 for d in demands) / n
            std = math.sqrt(variance)
        else:
            std = 0.0

        cv = (std / avg * 100) if avg != 0 else 0.0
        pattern = _classify_pattern(cv)
        adi = _compute_adi(demands)
        trend = _compute_trend(demands)

        products.append(ProductDemand(
            product=product,
            total_demand=round(total, 4),
            periods=n,
            avg_per_period=round(avg, 4),
            max_period_demand=round(max_d, 4),
            min_period_demand=round(min_d, 4),
            std_dev=round(std, 4),
            cv=round(cv, 2),
            pattern=pattern,
            adi=round(adi, 4) if adi is not None else None,
            trend=trend,
        ))
        all_period_values.extend(demands)

    if not products:
        return None

    # Determine overall trend using all period values
    # Aggregate by period across products
    all_periods: dict[str, float] = {}
    for _product, periods_map in product_periods.items():
        for p, v in periods_map.items():
            all_periods[p] = all_periods.get(p, 0.0) + v
    overall_values = [all_periods[p] for p in sorted(all_periods)]
    overall_trend = _compute_trend(overall_values)

    summary = (
        f"Demand pattern analysis: {len(products)} products across "
        f"{len(all_periods)} periods. "
        f"Overall trend: {overall_trend}. "
        f"Patterns: "
        + ", ".join(f"{p.product}={p.pattern}" for p in products)
        + "."
    )

    return DemandPatternResult(
        products=products,
        overall_trend=overall_trend,
        summary=summary,
    )


def compute_moving_average(
    rows: list[dict],
    quantity_column: str,
    date_column: str,
    window: int = 3,
) -> MovingAvgResult | None:
    """Compute simple moving average over period-aggregated demand.

    Groups rows by date_column, sums quantities per period, then computes
    a simple moving average with the given window size.

    Args:
        rows: Data rows as dicts.
        quantity_column: Column with demand quantity.
        date_column: Column with period identifier.
        window: Moving average window size (default 3).

    Returns:
        MovingAvgResult or None if insufficient data.
    """
    if not rows or window < 1:
        return None

    grouped = _group_by_period(rows, quantity_column, date_column)
    if len(grouped) < 1:
        return None

    points: list[MovingAvgPoint] = []

    for i, (period, actual) in enumerate(grouped):
        if i < window - 1:
            points.append(MovingAvgPoint(
                period=period,
                actual=round(actual, 4),
                moving_avg=None,
                error=None,
            ))
        else:
            window_vals = [grouped[j][1] for j in range(i - window + 1, i + 1)]
            ma = sum(window_vals) / window
            err = actual - ma
            points.append(MovingAvgPoint(
                period=period,
                actual=round(actual, 4),
                moving_avg=round(ma, 4),
                error=round(err, 4),
            ))

    # Compute MAD and MAPE over points that have a moving average
    errors = [p for p in points if p.error is not None]
    if not errors:
        return MovingAvgResult(points=points, window=window, mad=None, mape=None)

    mad = sum(abs(p.error) for p in errors) / len(errors)

    # MAPE: only for non-zero actuals
    mape_terms = [
        abs(p.error / p.actual) * 100
        for p in errors
        if p.actual != 0
    ]
    mape = sum(mape_terms) / len(mape_terms) if mape_terms else None

    return MovingAvgResult(
        points=points,
        window=window,
        mad=round(mad, 4),
        mape=round(mape, 2) if mape is not None else None,
    )


def exponential_smoothing(
    rows: list[dict],
    quantity_column: str,
    date_column: str,
    alpha: float = 0.3,
) -> SmoothingResult | None:
    """Simple exponential smoothing with optimal alpha search.

    Forecast[t] = alpha * actual[t-1] + (1-alpha) * forecast[t-1]
    First forecast = first actual value.

    Also searches alpha in {0.1, 0.2, ..., 0.9} to find the one with
    lowest MAD.

    Args:
        rows: Data rows as dicts.
        quantity_column: Column with demand quantity.
        date_column: Column with period identifier.
        alpha: Smoothing factor (default 0.3).

    Returns:
        SmoothingResult or None if insufficient data.
    """
    if not rows:
        return None

    grouped = _group_by_period(rows, quantity_column, date_column)
    if len(grouped) < 2:
        return None

    def _run_smoothing(
        periods: list[tuple[str, float]],
        a: float,
    ) -> tuple[list[SmoothingPoint], float]:
        """Run smoothing and return (points, MAD)."""
        pts: list[SmoothingPoint] = []
        first_period, first_actual = periods[0]
        forecast = first_actual
        pts.append(SmoothingPoint(
            period=first_period,
            actual=round(first_actual, 4),
            forecast=round(forecast, 4),
        ))

        abs_errors: list[float] = []
        for i in range(1, len(periods)):
            period, actual = periods[i]
            forecast = a * periods[i - 1][1] + (1 - a) * pts[i - 1].forecast
            err = abs(actual - forecast)
            abs_errors.append(err)
            pts.append(SmoothingPoint(
                period=period,
                actual=round(actual, 4),
                forecast=round(forecast, 4),
            ))

        # One step ahead forecast
        next_forecast = a * periods[-1][1] + (1 - a) * pts[-1].forecast
        pts.append(SmoothingPoint(
            period="next",
            actual=None,
            forecast=round(next_forecast, 4),
        ))

        mad = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        return pts, mad

    # Find optimal alpha
    best_alpha = alpha
    best_mad = float("inf")
    for candidate in [i / 10 for i in range(1, 10)]:
        _, candidate_mad = _run_smoothing(grouped, candidate)
        if candidate_mad < best_mad:
            best_mad = candidate_mad
            best_alpha = candidate

    # Run with the user-supplied alpha for reported results
    points, user_mad = _run_smoothing(grouped, alpha)

    # Compute MAPE for user alpha
    mape_terms = []
    for pt in points:
        if pt.actual is not None and pt.actual != 0:
            mape_terms.append(abs(pt.actual - pt.forecast) / abs(pt.actual) * 100)

    mape = sum(mape_terms) / len(mape_terms) if mape_terms else None

    return SmoothingResult(
        points=points,
        alpha=alpha,
        mad=round(user_mad, 4),
        mape=round(mape, 2) if mape is not None else None,
        optimal_alpha=round(best_alpha, 1),
        next_forecast=points[-1].forecast,
    )


def detect_demand_seasonality(
    rows: list[dict],
    quantity_column: str,
    date_column: str,
) -> SeasonalityResult | None:
    """Detect seasonal patterns in demand data.

    Groups by period, computes seasonal index (period_avg / overall_avg).
    Identifies peak seasons (index > 1.2) and low seasons (index < 0.8).

    Args:
        rows: Data rows as dicts.
        quantity_column: Column with demand quantity.
        date_column: Column with period identifier.

    Returns:
        SeasonalityResult or None if insufficient data.
    """
    if not rows:
        return None

    grouped = _group_by_period(rows, quantity_column, date_column)
    if len(grouped) < 2:
        return None

    demands = [qty for _, qty in grouped]
    overall_avg = sum(demands) / len(demands)

    if overall_avg == 0:
        return None

    periods: list[SeasonalPeriod] = []
    for period, qty in grouped:
        index = qty / overall_avg
        periods.append(SeasonalPeriod(
            period=period,
            avg_demand=round(qty, 4),
            seasonal_index=round(index, 4),
        ))

    peak = [sp.period for sp in periods if sp.seasonal_index > 1.2]
    low = [sp.period for sp in periods if sp.seasonal_index < 0.8]
    indices = [sp.seasonal_index for sp in periods]
    strength = max(indices) - min(indices) if indices else 0.0

    summary = (
        f"Seasonality analysis: {len(periods)} periods, "
        f"overall avg {overall_avg:.2f}. "
        f"Seasonal strength: {strength:.2f}. "
        f"{len(peak)} peak season(s), {len(low)} low season(s)."
    )

    return SeasonalityResult(
        periods=periods,
        overall_avg=round(overall_avg, 4),
        peak_seasons=peak,
        low_seasons=low,
        seasonal_strength=round(strength, 4),
        summary=summary,
    )


def format_demand_report(
    pattern: DemandPatternResult | None = None,
    moving_avg: MovingAvgResult | None = None,
    smoothing: SmoothingResult | None = None,
    seasonality: SeasonalityResult | None = None,
) -> str:
    """Format a combined demand analysis report.

    Args:
        pattern: Optional DemandPatternResult.
        moving_avg: Optional MovingAvgResult.
        smoothing: Optional SmoothingResult.
        seasonality: Optional SeasonalityResult.

    Returns:
        Human-readable text summary.
    """
    if pattern is None and moving_avg is None and smoothing is None and seasonality is None:
        return "No demand data available for report."

    sections: list[str] = []
    sections.append("=== Demand Forecast Report ===")
    sections.append("")

    if pattern is not None:
        sections.append("--- Demand Pattern Analysis ---")
        sections.append(pattern.summary)
        sections.append("")
        sections.append(f"  Overall Trend: {pattern.overall_trend}")
        sections.append(f"  Products:      {len(pattern.products)}")
        sections.append("")
        sections.append("  Product Details:")
        for p in pattern.products:
            sections.append(
                f"    {p.product}: total={p.total_demand:.0f}, "
                f"avg={p.avg_per_period:.1f}, cv={p.cv:.1f}%, "
                f"pattern={p.pattern}, trend={p.trend}"
            )
        sections.append("")

    if moving_avg is not None:
        sections.append("--- Moving Average ---")
        sections.append(f"  Window:  {moving_avg.window}")
        if moving_avg.mad is not None:
            sections.append(f"  MAD:     {moving_avg.mad:.2f}")
        if moving_avg.mape is not None:
            sections.append(f"  MAPE:    {moving_avg.mape:.2f}%")
        sections.append("")
        sections.append("  Period Data:")
        for pt in moving_avg.points:
            ma_str = f"{pt.moving_avg:.1f}" if pt.moving_avg is not None else "N/A"
            sections.append(
                f"    {pt.period}: actual={pt.actual:.1f}, MA={ma_str}"
            )
        sections.append("")

    if smoothing is not None:
        sections.append("--- Exponential Smoothing ---")
        sections.append(f"  Alpha:          {smoothing.alpha}")
        sections.append(f"  Optimal Alpha:  {smoothing.optimal_alpha}")
        if smoothing.mad is not None:
            sections.append(f"  MAD:            {smoothing.mad:.2f}")
        if smoothing.mape is not None:
            sections.append(f"  MAPE:           {smoothing.mape:.2f}%")
        sections.append(f"  Next Forecast:  {smoothing.next_forecast:.2f}")
        sections.append("")
        sections.append("  Period Data:")
        for pt in smoothing.points:
            actual_str = f"{pt.actual:.1f}" if pt.actual is not None else "N/A"
            sections.append(
                f"    {pt.period}: actual={actual_str}, forecast={pt.forecast:.1f}"
            )
        sections.append("")

    if seasonality is not None:
        sections.append("--- Seasonality ---")
        sections.append(seasonality.summary)
        sections.append("")
        sections.append(f"  Overall Avg:       {seasonality.overall_avg:.2f}")
        sections.append(f"  Seasonal Strength: {seasonality.seasonal_strength:.2f}")
        if seasonality.peak_seasons:
            sections.append(f"  Peak Seasons:      {', '.join(seasonality.peak_seasons)}")
        if seasonality.low_seasons:
            sections.append(f"  Low Seasons:       {', '.join(seasonality.low_seasons)}")
        sections.append("")
        sections.append("  Seasonal Indices:")
        for sp in seasonality.periods:
            sections.append(
                f"    {sp.period}: demand={sp.avg_demand:.1f}, index={sp.seasonal_index:.2f}"
            )
        sections.append("")

    return "\n".join(sections)
