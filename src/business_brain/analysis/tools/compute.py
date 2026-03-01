"""Stateless Python compute functions for analysis operations.

All functions are pure: no DB, no LLM, no async.
Input: numpy arrays or plain lists. Output: dicts with results.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Single-column descriptors
# ---------------------------------------------------------------------------


def describe_numeric(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Compute descriptive statistics for a numeric series."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"count": 0, "error": "no valid values"}

    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    result: dict[str, Any] = {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(median),
        "stdev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }
    if len(arr) >= 8:
        result["skewness"] = float(sp_stats.skew(arr))
        result["kurtosis"] = float(sp_stats.kurtosis(arr))
    return result


def describe_categorical(values: list[str] | list[Any]) -> dict[str, Any]:
    """Compute descriptive statistics for a categorical series."""
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {"count": 0, "error": "no values"}

    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    sorted_counts = counts.most_common()
    top_value, top_count = sorted_counts[0]
    concentration = top_count / total  # Herfindahl-like

    return {
        "count": total,
        "unique": len(counts),
        "top_value": str(top_value),
        "top_count": int(top_count),
        "concentration": float(concentration),
        "entropy": float(entropy),
        "distribution": {str(k): int(v) for k, v in sorted_counts[:20]},
    }


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------


def detect_distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Fit candidate distributions and return the best match."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 20:
        return {"type": "unknown", "reason": "too few values"}

    candidates: list[dict[str, Any]] = []

    # Normal
    stat, p_normal = sp_stats.shapiro(arr[:5000])  # shapiro max ~5000
    candidates.append({"type": "normal", "p_value": float(p_normal), "params": {}})

    # Lognormal (only for positive values)
    if np.all(arr > 0):
        log_arr = np.log(arr)
        _, p_lognormal = sp_stats.shapiro(log_arr[:5000])
        candidates.append({"type": "lognormal", "p_value": float(p_lognormal), "params": {}})

    # Uniform
    _, p_uniform = sp_stats.kstest(arr, "uniform", args=(arr.min(), arr.ptp()))
    candidates.append({"type": "uniform", "p_value": float(p_uniform), "params": {}})

    # Bimodal check via Hartigan's dip test approximation
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    mid = n // 2
    lower_mode = np.median(sorted_arr[:mid])
    upper_mode = np.median(sorted_arr[mid:])
    gap = (upper_mode - lower_mode) / (arr.std() + 1e-10)
    if gap > 1.5:
        candidates.append({"type": "bimodal", "p_value": 0.5, "params": {"gap_ratio": float(gap)}})

    best = max(candidates, key=lambda c: c["p_value"])
    return {
        "type": best["type"],
        "fit_score": best["p_value"],
        "params": best["params"],
        "all_fits": candidates,
    }


# ---------------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------------


def compare_groups(groups: dict[str, list[float]]) -> dict[str, Any]:
    """Compare groups by effect size and significance.

    For 2 groups: Cohen's d + Welch's t-test.
    For N groups: ANOVA F-test + eta-squared.
    """
    group_names = list(groups.keys())
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[~np.isnan(a)] for a in arrays]

    if any(len(a) < 2 for a in arrays):
        return {"error": "each group needs at least 2 values"}

    if len(arrays) == 2:
        a, b = arrays
        pooled_std = math.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        cohens_d = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0
        t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
        return {
            "test": "welch_t",
            "groups": group_names,
            "group_means": {group_names[0]: float(a.mean()), group_names[1]: float(b.mean())},
            "cohens_d": cohens_d,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }

    # N groups: one-way ANOVA
    f_stat, p_value = sp_stats.f_oneway(*arrays)
    grand_mean = np.concatenate(arrays).mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_total = sum(((a - grand_mean) ** 2).sum() for a in arrays)
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return {
        "test": "anova_f",
        "groups": group_names,
        "group_means": {name: float(a.mean()) for name, a in zip(group_names, arrays)},
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": eta_squared,
        "significant": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def compute_correlation(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> dict[str, Any]:
    """Compute Pearson and Spearman correlations."""
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    mask = ~(np.isnan(ax) | np.isnan(ay))
    ax, ay = ax[mask], ay[mask]

    if len(ax) < 3:
        return {"error": "need at least 3 paired values"}

    pearson_r, pearson_p = sp_stats.pearsonr(ax, ay)
    spearman_r, spearman_p = sp_stats.spearmanr(ax, ay)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n": int(len(ax)),
        "significant": bool(pearson_p < 0.05),
    }


def compute_lag_correlation(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    max_lag: int = 10,
) -> dict[str, Any]:
    """Cross-correlation at each lag, returning best lag."""
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)

    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs, ys = ax[:len(ax) - lag] if lag > 0 else ax, ay[lag:]
        else:
            xs, ys = ax[-lag:], ay[:len(ay) + lag]
        if len(xs) < 3:
            continue
        r, p = sp_stats.pearsonr(xs, ys)
        results.append({"lag": lag, "r": float(r), "p": float(p)})

    if not results:
        return {"error": "not enough data for lag correlation"}

    best = max(results, key=lambda r: abs(r["r"]))
    return {
        "best_lag": best["lag"],
        "best_r": best["r"],
        "best_p": best["p"],
        "all_lags": results,
    }


def compute_partial_correlation(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    controls: list[list[float] | np.ndarray],
) -> dict[str, Any]:
    """Partial correlation between x and y controlling for confounders (Gap #1)."""
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    ctrl = np.column_stack([np.asarray(c, dtype=float) for c in controls]) if controls else np.empty((len(ax), 0))

    # Remove NaN rows across all variables
    combined = np.column_stack([ax, ay, ctrl])
    mask = ~np.any(np.isnan(combined), axis=1)
    combined = combined[mask]

    if len(combined) < ctrl.shape[1] + 3:
        return {"error": "not enough data after NaN removal"}

    ax = combined[:, 0]
    ay = combined[:, 1]
    ctrl = combined[:, 2:]

    if ctrl.shape[1] == 0:
        r, p = sp_stats.pearsonr(ax, ay)
        return {"partial_r": float(r), "p_value": float(p), "n": len(ax), "controls": 0}

    # Residualize x and y against controls via OLS
    def _residualize(target: np.ndarray, predictors: np.ndarray) -> np.ndarray:
        X = np.column_stack([predictors, np.ones(len(predictors))])
        beta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        return target - X @ beta

    rx = _residualize(ax, ctrl)
    ry = _residualize(ay, ctrl)

    r, p = sp_stats.pearsonr(rx, ry)
    return {
        "partial_r": float(r),
        "p_value": float(p),
        "n": int(len(ax)),
        "controls": int(ctrl.shape[1]),
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def find_anomalies_zscore(
    values: list[float] | np.ndarray,
    threshold: float = 3.0,
) -> dict[str, Any]:
    """Find anomalies by z-score (>threshold standard deviations from mean)."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 5:
        return {"anomalies": [], "error": "too few values"}

    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1))
    if std < 1e-10:
        return {"anomalies": [], "mean": mean, "std": 0.0}

    z_scores = (arr - mean) / std
    anomalies = []
    for i, (val, z) in enumerate(zip(arr, z_scores)):
        if not np.isnan(val) and abs(z) >= threshold:
            anomalies.append({"index": i, "value": float(val), "z_score": float(z)})

    return {
        "anomalies": anomalies,
        "count": len(anomalies),
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "total": int(len(valid)),
    }


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def decompose_series(
    values: list[float] | np.ndarray,
    period: int = 7,
) -> dict[str, Any]:
    """Additive decomposition into trend + seasonal + residual."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < period * 2:
        return {"error": f"need at least {period * 2} values for period={period}"}

    # Trend: centered moving average
    kernel = np.ones(period) / period
    trend = np.convolve(arr, kernel, mode="same")
    # Fix edges
    half = period // 2
    trend[:half] = trend[half]
    trend[-half:] = trend[-(half + 1)]

    # Seasonal: average by position within period
    detrended = arr - trend
    seasonal = np.zeros(n)
    for i in range(period):
        indices = list(range(i, n, period))
        seasonal[indices] = np.mean(detrended[indices])

    residual = arr - trend - seasonal

    return {
        "trend": trend.tolist(),
        "seasonal": seasonal.tolist(),
        "residual": residual.tolist(),
        "period": period,
        "trend_direction": "increasing" if trend[-1] > trend[0] else "decreasing",
        "seasonal_strength": float(np.std(seasonal) / (np.std(residual) + 1e-10)),
    }


def forecast_series(
    values: list[float] | np.ndarray,
    periods: int = 7,
    alpha: float = 0.3,
) -> dict[str, Any]:
    """Simple exponential smoothing forecast."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3:
        return {"error": "need at least 3 values"}

    # Exponential smoothing
    smoothed = np.zeros(len(arr))
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]

    # Forecast: flat from last smoothed value
    last = smoothed[-1]
    forecast = [float(last)] * periods

    # Prediction interval from residual std
    residuals = arr - smoothed
    std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    return {
        "forecast": forecast,
        "smoothed": smoothed.tolist(),
        "last_value": float(arr[-1]),
        "forecast_base": float(last),
        "residual_std": std,
        "upper_bound": [float(last + 1.96 * std) for _ in range(periods)],
        "lower_bound": [float(last - 1.96 * std) for _ in range(periods)],
    }


# ---------------------------------------------------------------------------
# Stability / bootstrap
# ---------------------------------------------------------------------------


def bootstrap_stability(
    data: list[float] | np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_samples: int = 1000,
) -> dict[str, Any]:
    """Bootstrap confidence interval and stability fraction."""
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return {"error": "need at least 5 values"}

    rng = np.random.default_rng(42)
    point_estimate = float(stat_fn(arr))

    boot_stats = []
    for _ in range(n_samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(float(stat_fn(sample)))

    boot_arr = np.array(boot_stats)
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))

    # Fraction of bootstraps within 10% of point estimate
    tolerance = abs(point_estimate) * 0.1 if point_estimate != 0 else 0.1
    fraction_stable = float(np.mean(np.abs(boot_arr - point_estimate) <= tolerance))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "fraction_stable": fraction_stable,
        "n_samples": n_samples,
    }
