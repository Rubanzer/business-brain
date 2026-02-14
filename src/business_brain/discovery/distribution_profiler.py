"""Distribution profiler — computes statistical distribution properties for numeric columns.

Pure functions that analyze value distributions: histogram bins, quartiles,
skewness, kurtosis, and normality assessment.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass
class DistributionProfile:
    """Complete distribution profile for a numeric column."""

    count: int
    mean: float
    median: float
    stdev: float
    min_val: float
    max_val: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # interquartile range
    skewness: float  # 0 = symmetric, >0 = right-skewed, <0 = left-skewed
    kurtosis: float  # 0 = normal, >0 = heavy-tailed, <0 = light-tailed
    histogram: list[dict]  # [{"bin_start", "bin_end", "count"}]
    shape: str  # "normal", "right_skewed", "left_skewed", "bimodal", "uniform"


def compute_quartiles(values: list[float]) -> tuple[float, float, float]:
    """Compute Q1, median (Q2), and Q3.

    Uses the exclusive percentile method (interpolation).
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    sorted_vals = sorted(values)

    if n == 1:
        return sorted_vals[0], sorted_vals[0], sorted_vals[0]

    median = statistics.median(sorted_vals)

    # Q1: median of lower half
    lower = sorted_vals[:n // 2]
    q1 = statistics.median(lower) if lower else sorted_vals[0]

    # Q3: median of upper half
    upper = sorted_vals[(n + 1) // 2:]
    q3 = statistics.median(upper) if upper else sorted_vals[-1]

    return q1, median, q3


def compute_skewness(values: list[float]) -> float:
    """Compute Fisher-Pearson skewness coefficient.

    Returns 0 for symmetric, >0 for right-skewed, <0 for left-skewed.
    """
    n = len(values)
    if n < 3:
        return 0.0

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    if stdev == 0:
        return 0.0

    m3 = sum((x - mean) ** 3 for x in values) / n
    return m3 / (stdev ** 3)


def compute_kurtosis(values: list[float]) -> float:
    """Compute excess kurtosis (Fisher definition).

    Returns 0 for normal distribution, >0 for heavy-tailed, <0 for light-tailed.
    """
    n = len(values)
    if n < 4:
        return 0.0

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    if stdev == 0:
        return 0.0

    m4 = sum((x - mean) ** 4 for x in values) / n
    return (m4 / (stdev ** 4)) - 3.0


def build_histogram(values: list[float], bins: int = 10) -> list[dict]:
    """Build a histogram with specified number of bins.

    Returns list of {"bin_start", "bin_end", "count"}.
    """
    if not values or bins < 1:
        return []

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [{"bin_start": min_val, "bin_end": max_val, "count": len(values)}]

    bin_width = (max_val - min_val) / bins
    histogram = []

    for i in range(bins):
        bin_start = min_val + i * bin_width
        bin_end = min_val + (i + 1) * bin_width
        count = sum(
            1 for v in values
            if (bin_start <= v < bin_end) or (i == bins - 1 and v == bin_end)
        )
        histogram.append({
            "bin_start": round(bin_start, 4),
            "bin_end": round(bin_end, 4),
            "count": count,
        })

    return histogram


def classify_shape(skewness: float, kurtosis: float, histogram: list[dict]) -> str:
    """Classify the distribution shape based on statistics.

    Returns one of: "normal", "right_skewed", "left_skewed", "bimodal", "uniform".
    """
    # Check for bimodal: two peaks in histogram
    if len(histogram) >= 5:
        counts = [h["count"] for h in histogram]
        peaks = 0
        for i in range(1, len(counts) - 1):
            if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
                peaks += 1
        if peaks >= 2:
            return "bimodal"

    # Check for uniform: low kurtosis and low skewness
    if abs(skewness) < 0.3 and kurtosis < -1.0:
        return "uniform"

    # Skewness-based classification
    if abs(skewness) < 0.5:
        return "normal"
    elif skewness > 0.5:
        return "right_skewed"
    else:
        return "left_skewed"


def profile_distribution(values: list[float], bins: int = 10) -> DistributionProfile | None:
    """Compute complete distribution profile for a list of numeric values.

    Returns None if insufficient data.
    """
    if len(values) < 3:
        return None

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    q1, median, q3 = compute_quartiles(values)
    iqr = q3 - q1
    skewness = compute_skewness(values)
    kurtosis = compute_kurtosis(values)
    histogram = build_histogram(values, bins)
    shape = classify_shape(skewness, kurtosis, histogram)

    return DistributionProfile(
        count=len(values),
        mean=round(mean, 4),
        median=round(median, 4),
        stdev=round(stdev, 4),
        min_val=min(values),
        max_val=max(values),
        q1=round(q1, 4),
        q3=round(q3, 4),
        iqr=round(iqr, 4),
        skewness=round(skewness, 4),
        kurtosis=round(kurtosis, 4),
        histogram=histogram,
        shape=shape,
    )
