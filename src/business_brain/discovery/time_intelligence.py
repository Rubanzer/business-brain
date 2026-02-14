"""Time intelligence — period-over-period calculations, trend detection, min/max periods.

Pure functions that take numeric series data and produce time-based insights.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class PeriodChange:
    """Result of a period-over-period comparison."""

    current: float
    previous: float
    absolute_change: float
    pct_change: float | None  # None if previous is 0


@dataclass
class TrendResult:
    """Result of trend direction analysis."""

    direction: str  # "increasing", "decreasing", "stable", "volatile"
    magnitude: float  # pct change per period (average)
    r_squared: float  # goodness of linear fit (0-1)


@dataclass
class MinMaxPeriod:
    """Identifies the min and max periods in a series."""

    max_index: int
    max_value: float
    min_index: int
    min_value: float


def compute_period_change(current: float, previous: float) -> PeriodChange:
    """Compute absolute and percentage change between two periods."""
    absolute = current - previous
    if previous != 0:
        pct = (absolute / abs(previous)) * 100
    else:
        pct = None
    return PeriodChange(
        current=current,
        previous=previous,
        absolute_change=absolute,
        pct_change=pct,
    )


def detect_trend(values: list[float]) -> TrendResult:
    """Detect trend direction and magnitude from a numeric series.

    Uses simple linear regression to determine direction and fit quality.
    """
    n = len(values)
    if n < 2:
        return TrendResult(direction="stable", magnitude=0.0, r_squared=0.0)

    # Simple linear regression: y = mx + b
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(values)

    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return TrendResult(direction="stable", magnitude=0.0, r_squared=1.0)

    slope = numerator / denominator

    # R-squared
    y_pred = [y_mean + slope * (i - x_mean) for i in range(n)]
    ss_res = sum((v - yp) ** 2 for v, yp in zip(values, y_pred))
    ss_tot = sum((v - y_mean) ** 2 for v in values)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    # Magnitude: average % change per period
    if y_mean != 0:
        magnitude = (slope / abs(y_mean)) * 100
    else:
        magnitude = 0.0

    # Direction — check stable first (tiny magnitude = stable regardless of fit)
    if abs(magnitude) < 1.0:
        direction = "stable"
    elif r_squared < 0.3:
        direction = "volatile"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return TrendResult(
        direction=direction,
        magnitude=round(magnitude, 2),
        r_squared=round(r_squared, 4),
    )


def find_min_max_periods(values: list[float]) -> MinMaxPeriod | None:
    """Find the indices and values of the min and max periods."""
    if not values:
        return None

    max_val = max(values)
    min_val = min(values)

    return MinMaxPeriod(
        max_index=values.index(max_val),
        max_value=max_val,
        min_index=values.index(min_val),
        min_value=min_val,
    )


def compute_moving_average(values: list[float], window: int = 3) -> list[float]:
    """Compute simple moving average with given window size."""
    if window < 1 or len(values) < window:
        return list(values)

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result


@dataclass
class ForecastResult:
    """Result of a time series forecast."""

    predicted_values: list[float]
    method: str  # "linear" or "exponential"
    confidence: str  # "high", "medium", "low"


def forecast_linear(values: list[float], periods_ahead: int = 3) -> ForecastResult:
    """Forecast future values using linear extrapolation.

    Uses the same linear regression as detect_trend, then extends the line.
    """
    n = len(values)
    if n < 2 or periods_ahead < 1:
        return ForecastResult(predicted_values=[], method="linear", confidence="low")

    # Linear regression
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(values)

    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return ForecastResult(
            predicted_values=[y_mean] * periods_ahead,
            method="linear",
            confidence="low",
        )

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # R-squared for confidence
    y_pred = [intercept + slope * i for i in range(n)]
    ss_res = sum((v - yp) ** 2 for v, yp in zip(values, y_pred))
    ss_tot = sum((v - y_mean) ** 2 for v in values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    if r_squared >= 0.8:
        confidence = "high"
    elif r_squared >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    predicted = [round(intercept + slope * (n + i), 2) for i in range(periods_ahead)]

    return ForecastResult(
        predicted_values=predicted,
        method="linear",
        confidence=confidence,
    )


def forecast_exponential(
    values: list[float],
    periods_ahead: int = 3,
    alpha: float = 0.3,
) -> ForecastResult:
    """Forecast using simple exponential smoothing.

    Alpha controls smoothing: closer to 1 = more weight on recent values.
    """
    n = len(values)
    if n < 2 or periods_ahead < 1:
        return ForecastResult(predicted_values=[], method="exponential", confidence="low")

    if not 0 < alpha < 1:
        alpha = 0.3

    # Compute smoothed values
    smoothed = [values[0]]
    for i in range(1, n):
        s = alpha * values[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(s)

    # Forecast: flat extension from the last smoothed value
    last_smoothed = smoothed[-1]
    predicted = [round(last_smoothed, 2)] * periods_ahead

    # Confidence based on how well smoothing fits
    errors = [abs(values[i] - smoothed[i]) for i in range(n)]
    mean_error = sum(errors) / n if n > 0 else 0
    mean_val = statistics.mean(values) if values else 1
    error_ratio = mean_error / abs(mean_val) if mean_val != 0 else 1

    if error_ratio < 0.1:
        confidence = "high"
    elif error_ratio < 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    return ForecastResult(
        predicted_values=predicted,
        method="exponential",
        confidence=confidence,
    )


def detect_changepoints(values: list[float], threshold: float = 2.0) -> list[int]:
    """Find indices where the series changes significantly.

    A changepoint is where the difference from the local mean exceeds
    threshold * stdev.
    """
    if len(values) < 4:
        return []

    stdev = statistics.stdev(values)
    if stdev == 0:
        return []

    mean = statistics.mean(values)
    changepoints = []

    for i in range(1, len(values)):
        diff = abs(values[i] - values[i - 1])
        if diff > threshold * stdev:
            changepoints.append(i)

    return changepoints
