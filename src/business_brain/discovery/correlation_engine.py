"""Correlation engine — computes pairwise correlations between numeric columns.

Pure functions that take numeric data and find statistically significant relationships.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass
class CorrelationPair:
    """A single pairwise correlation result."""

    column_a: str
    column_b: str
    correlation: float  # -1.0 to 1.0 (Pearson)
    strength: str  # "strong", "moderate", "weak", "none"
    direction: str  # "positive", "negative", "none"
    sample_size: int


def compute_pearson(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient between two series.

    Returns None if correlation cannot be computed (e.g. constant series).
    """
    n = len(x)
    if n != len(y) or n < 3:
        return None

    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return None

    return numerator / (denom_x * denom_y)


def classify_correlation(r: float | None) -> tuple[str, str]:
    """Classify correlation strength and direction.

    Returns (strength, direction).
    """
    if r is None:
        return "none", "none"

    abs_r = abs(r)

    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "none"

    if abs_r < 0.2:
        direction = "none"
    elif r > 0:
        direction = "positive"
    else:
        direction = "negative"

    return strength, direction


def compute_correlation_matrix(
    data: dict[str, list[float]],
) -> list[CorrelationPair]:
    """Compute pairwise correlations for all column combinations.

    Args:
        data: Dict mapping column names to numeric value lists.
              All lists must be the same length.

    Returns:
        List of CorrelationPair for all unique pairs.
    """
    columns = list(data.keys())
    pairs: list[CorrelationPair] = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]
            x = data[col_a]
            y = data[col_b]

            # Align lengths — use paired rows only
            min_len = min(len(x), len(y))
            if min_len < 3:
                continue

            x_trimmed = x[:min_len]
            y_trimmed = y[:min_len]

            r = compute_pearson(x_trimmed, y_trimmed)
            strength, direction = classify_correlation(r)

            pairs.append(CorrelationPair(
                column_a=col_a,
                column_b=col_b,
                correlation=round(r, 4) if r is not None else 0.0,
                strength=strength,
                direction=direction,
                sample_size=min_len,
            ))

    return pairs


def find_strong_correlations(
    pairs: list[CorrelationPair],
    threshold: float = 0.7,
) -> list[CorrelationPair]:
    """Filter to only strong correlations (abs(r) >= threshold)."""
    return [p for p in pairs if abs(p.correlation) >= threshold]


def find_surprising_correlations(
    pairs: list[CorrelationPair],
    threshold: float = 0.5,
) -> list[CorrelationPair]:
    """Find unexpected correlations — moderate-to-strong negative ones.

    Negative correlations between columns that wouldn't obviously be
    inversely related can signal interesting business dynamics.
    """
    return [
        p for p in pairs
        if p.correlation <= -threshold
    ]


def correlation_summary(pairs: list[CorrelationPair]) -> dict:
    """Summarize correlation analysis results.

    Returns count by strength category and top correlations.
    """
    if not pairs:
        return {
            "total_pairs": 0,
            "strong": 0,
            "moderate": 0,
            "weak": 0,
            "none": 0,
            "top_positive": None,
            "top_negative": None,
        }

    by_strength = {"strong": 0, "moderate": 0, "weak": 0, "none": 0}
    for p in pairs:
        by_strength[p.strength] = by_strength.get(p.strength, 0) + 1

    sorted_by_r = sorted(pairs, key=lambda p: p.correlation, reverse=True)
    top_positive = sorted_by_r[0] if sorted_by_r[0].correlation > 0.2 else None
    top_negative = sorted_by_r[-1] if sorted_by_r[-1].correlation < -0.2 else None

    return {
        "total_pairs": len(pairs),
        "strong": by_strength["strong"],
        "moderate": by_strength["moderate"],
        "weak": by_strength["weak"],
        "none": by_strength["none"],
        "top_positive": {
            "columns": [top_positive.column_a, top_positive.column_b],
            "correlation": top_positive.correlation,
        } if top_positive else None,
        "top_negative": {
            "columns": [top_negative.column_a, top_negative.column_b],
            "correlation": top_negative.correlation,
        } if top_negative else None,
    }
