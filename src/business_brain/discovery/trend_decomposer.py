"""Time-series decomposition — breaks data into trend + seasonal + residual.

Pure-Python implementation with no external dependencies.
Supports additive decomposition: value = trend + seasonal + residual.

Useful for:
- Identifying underlying growth or decline trends
- Detecting repeating seasonal patterns (weekly, monthly, quarterly)
- Isolating anomalous residuals after removing expected behaviour
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DecompositionResult:
    """Result of time-series decomposition."""

    original: list[float]
    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    period: int
    trend_direction: str  # "increasing", "decreasing", "flat"
    trend_strength: float  # 0-1
    seasonal_strength: float  # 0-1
    summary: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose(
    values: list[float],
    period: int | None = None,
    method: str = "additive",
) -> DecompositionResult | None:
    """Decompose time series into trend + seasonal + residual.

    If *period* is ``None``, auto-detect using autocorrelation.
    *method*: ``"additive"`` (value = trend + seasonal + residual).

    Returns ``None`` if fewer than ``2 * period`` values or fewer than 6 values.

    Algorithm
    ---------
    1. Estimate trend using centred moving average with window = period.
    2. Detrend: ``detrended = original - trend``.
    3. Estimate seasonal: average detrended values for each position in cycle.
    4. Residual = original - trend - seasonal.
    """
    if not values or len(values) < 6:
        return None

    if period is None:
        period = detect_period(values)

    # Period of 1 means no seasonality — still decompose with flat seasonal
    if period < 1:
        period = 1

    if len(values) < 2 * period:
        return None

    # 1. Trend via centred moving average
    trend = compute_trend_line(values, window=period)

    # 2. Detrend
    detrended = [v - t for v, t in zip(values, trend)]

    # 3. Seasonal component — average detrended value at each cycle position
    seasonal_indices: dict[int, list[float]] = {}
    for i, d in enumerate(detrended):
        pos = i % period
        seasonal_indices.setdefault(pos, []).append(d)

    seasonal_pattern = [
        _mean(seasonal_indices[pos]) for pos in range(period)
    ]

    # Normalise seasonal so it sums to zero over one full period
    pattern_mean = _mean(seasonal_pattern)
    seasonal_pattern = [s - pattern_mean for s in seasonal_pattern]

    # Tile the pattern across the full length
    seasonal = [seasonal_pattern[i % period] for i in range(len(values))]

    # 4. Residual
    residual = [v - t - s for v, t, s in zip(values, trend, seasonal)]

    # Classification & strength
    direction, t_strength = classify_trend(trend)

    # Seasonal strength: 1 - var(residual) / var(detrended), clamped [0, 1]
    var_resid = _variance(residual)
    var_detrend = _variance(detrended)
    if var_detrend > 0:
        s_strength = max(0.0, min(1.0, 1.0 - var_resid / var_detrend))
    else:
        s_strength = 0.0

    summary = _build_summary(direction, t_strength, s_strength, period, len(values))

    return DecompositionResult(
        original=list(values),
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        period=period,
        trend_direction=direction,
        trend_strength=t_strength,
        seasonal_strength=s_strength,
        summary=summary,
    )


def detect_period(values: list[float], max_period: int = 0) -> int:
    """Auto-detect seasonality period using autocorrelation.

    If *max_period* is 0, use ``len(values) // 3``.
    Compute autocorrelation at each lag, find first significant peak after
    lag 1.

    Returns the detected period (minimum 2).  Returns 1 if no seasonality is
    detected.
    """
    n = len(values)
    if n < 6:
        return 1

    if max_period <= 0:
        max_period = n // 3

    max_period = min(max_period, n // 2)
    if max_period < 2:
        return 1

    mean_val = _mean(values)
    # Variance at lag 0
    var0 = sum((v - mean_val) ** 2 for v in values)
    if var0 == 0:
        return 1

    acf: list[float] = []
    for lag in range(1, max_period + 1):
        cov = sum(
            (values[i] - mean_val) * (values[i + lag] - mean_val)
            for i in range(n - lag)
        )
        acf.append(cov / var0)

    # Find the first peak in ACF after lag 0
    # A peak is where acf[i] > acf[i-1] and acf[i] >= acf[i+1]
    best_lag = 1
    best_acf = -2.0

    for i in range(1, len(acf) - 1):
        lag = i + 1  # actual lag (since acf[0] corresponds to lag=1)
        if acf[i] > acf[i - 1] and acf[i] >= acf[i + 1]:
            # Significant peak — ACF should be meaningfully positive
            if acf[i] > 0.1 and acf[i] > best_acf:
                best_lag = lag
                best_acf = acf[i]
                break  # take the first significant peak

    if best_acf < 0.1:
        return 1

    return best_lag


def compute_trend_line(values: list[float], window: int = 5) -> list[float]:
    """Simple centred moving average for trend extraction.

    For edges where the full window doesn't fit, use available values
    (asymmetric/partial window).

    Returns a list of the same length as *values*.
    """
    n = len(values)
    if n == 0:
        return []
    if window < 1:
        window = 1

    half = window // 2
    result: list[float] = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(_mean(values[lo:hi]))

    return result


def compute_rate_of_change_trend(trend: list[float]) -> list[float]:
    """Compute first differences of trend to show acceleration / deceleration.

    Returns a list of ``len(trend) - 1`` values.
    """
    if len(trend) < 2:
        return []
    return [trend[i + 1] - trend[i] for i in range(len(trend) - 1)]


def classify_trend(trend: list[float]) -> tuple[str, float]:
    """Classify trend direction and strength.

    Direction: ``'increasing'``, ``'decreasing'``, ``'flat'``.
    Strength: 0--1 based on R-squared of a linear fit to the trend.

    Linear fit uses ordinary least squares (manual formula, no numpy).
    """
    n = len(trend)
    if n < 2:
        return ("flat", 0.0)

    # x = 0, 1, 2, ... n-1
    x_mean = (n - 1) / 2.0
    y_mean = _mean(trend)

    ss_xy = 0.0
    ss_xx = 0.0
    for i, y in enumerate(trend):
        dx = i - x_mean
        ss_xy += dx * (y - y_mean)
        ss_xx += dx * dx

    if ss_xx == 0:
        return ("flat", 0.0)

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R-squared
    ss_res = 0.0
    ss_tot = 0.0
    for i, y in enumerate(trend):
        y_hat = slope * i + intercept
        ss_res += (y - y_hat) ** 2
        ss_tot += (y - y_mean) ** 2

    if ss_tot == 0:
        r_squared = 1.0  # perfectly constant → perfect fit
    else:
        r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    # Determine direction using slope relative to data scale
    # Compare total change over the series to the mean absolute level.
    # If the absolute mean is near zero, fall back to data range.
    data_range = max(trend) - min(trend)
    abs_mean = abs(y_mean) if abs(y_mean) > 1e-12 else max(data_range, 1e-12)

    total_change = abs(slope * (n - 1))
    relative_change = total_change / abs_mean

    if data_range == 0 or relative_change < 0.01:
        direction = "flat"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return (direction, r_squared)


def find_anomalous_residuals(
    residual: list[float],
    threshold: float = 2.0,
) -> list[dict]:
    """Find residual values exceeding *threshold* standard deviations.

    Returns a list of dicts::

        [{"index": 5, "value": 3.2, "severity": "high"}, ...]

    Severity levels:
    - ``"high"``: |z| >= 3
    - ``"medium"``: |z| >= threshold (and < 3)
    """
    if len(residual) < 2:
        return []

    mean_r = _mean(residual)
    std_r = _std(residual)
    if std_r == 0:
        return []

    anomalies: list[dict] = []
    for i, r in enumerate(residual):
        z = abs(r - mean_r) / std_r
        if z >= threshold:
            severity = "high" if z >= 3.0 else "medium"
            anomalies.append({
                "index": i,
                "value": r,
                "z_score": round(z, 4),
                "severity": severity,
            })

    return anomalies


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """Arithmetic mean. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    """Population variance."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    return math.sqrt(_variance(values))


def _build_summary(
    direction: str,
    trend_strength: float,
    seasonal_strength: float,
    period: int,
    n: int,
) -> str:
    """Build a human-readable summary of the decomposition."""
    parts: list[str] = []

    # Trend description
    if trend_strength > 0.7:
        parts.append(f"Strong {direction} trend (R²={trend_strength:.2f})")
    elif trend_strength > 0.3:
        parts.append(f"Moderate {direction} trend (R²={trend_strength:.2f})")
    else:
        parts.append(f"Weak or no clear trend (R²={trend_strength:.2f})")

    # Seasonal description
    if seasonal_strength > 0.5 and period > 1:
        parts.append(f"with notable seasonality (period={period}, strength={seasonal_strength:.2f})")
    elif seasonal_strength > 0.2 and period > 1:
        parts.append(f"with mild seasonality (period={period}, strength={seasonal_strength:.2f})")
    else:
        parts.append("with little to no seasonality")

    parts.append(f"over {n} observations.")

    return " ".join(parts)
