"""Outlier explainer — generates natural language explanations for detected anomalies.

Pure functions that take column statistics and anomaly info to produce
human-readable explanations with severity and recommended actions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutlierExplanation:
    """Complete explanation for a detected outlier."""

    value: float
    column: str
    table: str
    severity: str  # "critical", "warning", "info"
    deviation_sigma: float  # how many standard deviations from mean
    explanation: str  # human-readable explanation
    context: str  # where this value sits relative to the distribution
    recommended_action: str  # what to do about it


def explain_outlier(
    value: float,
    column: str,
    table: str,
    mean: float,
    stdev: float,
    min_val: float,
    max_val: float,
    semantic_type: str = "",
) -> OutlierExplanation:
    """Generate a natural language explanation for a detected outlier.

    Pure function — no DB or LLM needed.
    """
    if stdev == 0:
        sigma = 0.0
    else:
        sigma = abs(value - mean) / stdev

    # Severity based on sigma distance
    if sigma >= 4:
        severity = "critical"
    elif sigma >= 3:
        severity = "warning"
    else:
        severity = "info"

    # Direction
    direction = "above" if value > mean else "below"

    # Build explanation
    explanation = (
        f"The value {_fmt(value)} in {column} is {_fmt(sigma)} standard deviations "
        f"{direction} the mean ({_fmt(mean)}). "
    )

    # Add semantic context
    if semantic_type == "numeric_currency" and value < 0:
        explanation += "Negative values in a currency column may indicate credits, refunds, or data entry errors. "
        severity = "critical"
    elif semantic_type == "numeric_percentage":
        if value > 100:
            explanation += f"This exceeds 100%, which is typically impossible for a percentage. "
            severity = "critical"
        elif value < 0:
            explanation += f"Negative percentages are typically invalid. "
            severity = "critical"
    elif semantic_type == "numeric_metric":
        if value > max_val * 0.95 and sigma >= 2:
            explanation += "This is near the maximum observed value and may indicate a measurement error or unusual event. "
        elif value < min_val * 1.05 and sigma >= 2:
            explanation += "This is near the minimum observed value and may indicate a measurement error or unusual event. "

    # Context
    range_span = max_val - min_val
    if range_span > 0:
        position = (value - min_val) / range_span * 100
        context = f"At {_fmt(position)}% of the observed range ({_fmt(min_val)} to {_fmt(max_val)})"
    else:
        context = "All values are identical"

    # Recommended action
    if severity == "critical":
        recommended_action = f"Investigate {column} in {table} immediately. Verify data source and check for data entry errors."
    elif severity == "warning":
        recommended_action = f"Review the value {_fmt(value)} in {column}. It may be a legitimate extreme or a data quality issue."
    else:
        recommended_action = f"Monitor {column} for recurring outlier patterns."

    return OutlierExplanation(
        value=value,
        column=column,
        table=table,
        severity=severity,
        deviation_sigma=round(sigma, 2),
        explanation=explanation.strip(),
        context=context,
        recommended_action=recommended_action,
    )


def explain_null_spike(
    column: str,
    table: str,
    null_count: int,
    total_count: int,
) -> OutlierExplanation:
    """Generate explanation for a null spike anomaly."""
    if total_count == 0:
        pct = 0
    else:
        pct = null_count / total_count * 100

    if pct >= 50:
        severity = "critical"
    elif pct >= 20:
        severity = "warning"
    else:
        severity = "info"

    explanation = (
        f"Column {column} in {table} has {null_count} null values out of {total_count} "
        f"rows ({_fmt(pct)}%). "
    )

    if pct >= 50:
        explanation += "More than half the data is missing, which severely limits analysis reliability."
    elif pct >= 20:
        explanation += "Significant missing data may bias analysis results."

    return OutlierExplanation(
        value=pct,
        column=column,
        table=table,
        severity=severity,
        deviation_sigma=0,
        explanation=explanation.strip(),
        context=f"{null_count}/{total_count} values are null",
        recommended_action=(
            f"Investigate why {column} has missing values. "
            "Check data pipelines, source systems, and ingestion processes."
        ),
    )


def explain_constant_column(column: str, table: str, value: float, count: int) -> OutlierExplanation:
    """Generate explanation for a constant (zero variance) column."""
    return OutlierExplanation(
        value=value,
        column=column,
        table=table,
        severity="info",
        deviation_sigma=0,
        explanation=(
            f"Column {column} in {table} has zero variance — all {count} values "
            f"are {_fmt(value)}. This column provides no analytical value."
        ),
        context="Zero variance",
        recommended_action=(
            f"Consider removing {column} from analysis or investigating "
            "why all values are identical."
        ),
    )


def _fmt(value: float) -> str:
    """Format a number for display."""
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e10:
            return str(int(value))
        return f"{value:.2f}"
    return str(value)
