"""Rolling/moving window statistics — compute statistics over sliding windows.

Pure functions for computing rolling means, volatility, z-scores,
and detecting regime changes in time series data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RollingResult:
    """Result of rolling window computation."""
    values: list[float]       # original values
    rolling_mean: list[float | None]
    rolling_std: list[float | None]
    rolling_min: list[float | None]
    rolling_max: list[float | None]
    z_scores: list[float | None]  # how many std from rolling mean
    window: int
    summary: str


def rolling_statistics(
    values: list[float],
    window: int = 5,
) -> RollingResult | None:
    """Compute rolling mean, std, min, max, and z-scores.

    Args:
        values: Time series values.
        window: Window size for rolling computation.

    Returns:
        RollingResult or None if insufficient data.
    """
    n = len(values)
    if n < window or window < 2:
        return None

    r_mean: list[float | None] = [None] * n
    r_std: list[float | None] = [None] * n
    r_min: list[float | None] = [None] * n
    r_max: list[float | None] = [None] * n
    z_scores: list[float | None] = [None] * n

    for i in range(window - 1, n):
        w = values[i - window + 1: i + 1]
        mean = sum(w) / len(w)
        var = sum((x - mean) ** 2 for x in w) / len(w)
        std = var ** 0.5

        r_mean[i] = round(mean, 4)
        r_std[i] = round(std, 4)
        r_min[i] = round(min(w), 4)
        r_max[i] = round(max(w), 4)
        z_scores[i] = round((values[i] - mean) / std, 4) if std > 0 else 0.0

    summary = (
        f"Rolling statistics (window={window}) over {n} values. "
        f"Final rolling mean: {r_mean[-1]}, final rolling std: {r_std[-1]}."
    )

    return RollingResult(
        values=values,
        rolling_mean=r_mean,
        rolling_std=r_std,
        rolling_min=r_min,
        rolling_max=r_max,
        z_scores=z_scores,
        window=window,
        summary=summary,
    )


def detect_regime_changes(
    values: list[float],
    window: int = 10,
    threshold: float = 2.0,
) -> list[dict]:
    """Detect points where the series behavior changes significantly.

    A regime change is when the value crosses threshold standard deviations
    from the rolling mean.

    Returns list of {"index": i, "value": v, "z_score": z, "direction": "up"|"down"}.
    """
    result = rolling_statistics(values, window)
    if result is None:
        return []

    changes = []
    for i in range(window - 1, len(values)):
        z = result.z_scores[i]
        if z is not None and abs(z) >= threshold:
            changes.append({
                "index": i,
                "value": values[i],
                "z_score": z,
                "rolling_mean": result.rolling_mean[i],
                "direction": "up" if z > 0 else "down",
            })

    return changes


def rolling_correlation(
    values_a: list[float],
    values_b: list[float],
    window: int = 10,
) -> list[float | None]:
    """Compute rolling Pearson correlation between two series.

    Returns list same length as inputs, with None for positions
    where window hasn't been filled yet.
    """
    n = min(len(values_a), len(values_b))
    if n < window or window < 3:
        return [None] * n

    result: list[float | None] = [None] * n

    for i in range(window - 1, n):
        wa = values_a[i - window + 1: i + 1]
        wb = values_b[i - window + 1: i + 1]

        mean_a = sum(wa) / len(wa)
        mean_b = sum(wb) / len(wb)

        cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(wa, wb)) / len(wa)
        std_a = (sum((a - mean_a) ** 2 for a in wa) / len(wa)) ** 0.5
        std_b = (sum((b - mean_b) ** 2 for b in wb) / len(wb)) ** 0.5

        if std_a > 0 and std_b > 0:
            result[i] = round(cov / (std_a * std_b), 4)
        else:
            result[i] = 0.0

    return result


def compute_volatility(
    values: list[float],
    window: int = 10,
) -> list[float | None]:
    """Compute rolling volatility (annualized standard deviation of returns).

    Returns list of volatility values.
    """
    if len(values) < window + 1 or window < 2:
        return [None] * len(values)

    # Compute returns
    returns = []
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            returns.append((values[i] - values[i - 1]) / abs(values[i - 1]))
        else:
            returns.append(0.0)

    # Rolling std of returns
    result: list[float | None] = [None] * len(values)  # offset by 1 for returns
    for i in range(window - 1, len(returns)):
        w = returns[i - window + 1: i + 1]
        mean = sum(w) / len(w)
        var = sum((x - mean) ** 2 for x in w) / len(w)
        result[i + 1] = round(var ** 0.5, 6)  # +1 because returns are offset

    return result


def find_outlier_windows(
    values: list[float],
    window: int = 10,
    threshold: float = 2.0,
) -> list[dict]:
    """Find windows where the mean significantly deviates from the global mean.

    Useful for finding "unusual periods" in time series.
    """
    n = len(values)
    if n < window:
        return []

    global_mean = sum(values) / n
    global_std = (sum((v - global_mean) ** 2 for v in values) / n) ** 0.5
    if global_std == 0:
        return []

    outliers = []
    for i in range(n - window + 1):
        w = values[i: i + window]
        w_mean = sum(w) / len(w)
        z = (w_mean - global_mean) / global_std
        if abs(z) >= threshold:
            outliers.append({
                "start_index": i,
                "end_index": i + window - 1,
                "window_mean": round(w_mean, 4),
                "z_score": round(z, 4),
                "direction": "above" if z > 0 else "below",
            })

    return outliers
