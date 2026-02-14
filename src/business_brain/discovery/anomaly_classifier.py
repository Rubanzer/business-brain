"""Anomaly classifier — categorizes anomalies by pattern shape.

Given a time series of values, classifies anomalies into categories:
spike, dip, plateau, oscillation, step_change, gradual_drift.

Pure functions, no DB or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AnomalyClassification:
    """Classification result for an anomaly."""
    pattern: str  # spike, dip, plateau, oscillation, step_change, gradual_drift
    confidence: float  # 0.0-1.0
    description: str
    severity: str  # critical, warning, info
    affected_indices: list[int]
    metadata: dict[str, Any]


def classify_anomaly(
    values: list[float],
    anomaly_index: int,
    context_window: int = 5,
) -> AnomalyClassification:
    """Classify an anomaly at a specific index in a time series.

    Args:
        values: Full time series of values.
        anomaly_index: Index of the anomalous value.
        context_window: Number of points on each side to consider.

    Returns:
        AnomalyClassification with pattern type and details.
    """
    if not values or anomaly_index < 0 or anomaly_index >= len(values):
        return AnomalyClassification(
            pattern="unknown",
            confidence=0.0,
            description="Invalid index or empty values.",
            severity="info",
            affected_indices=[],
            metadata={},
        )

    n = len(values)
    idx = anomaly_index
    val = values[idx]

    # Get context window
    start = max(0, idx - context_window)
    end = min(n, idx + context_window + 1)
    window = values[start:end]
    window_mean = sum(window) / len(window)
    window_std = _std(window)

    # Get surrounding values (excluding anomaly)
    surrounding = [v for i, v in enumerate(values[start:end]) if i + start != idx]
    surr_mean = sum(surrounding) / len(surrounding) if surrounding else val

    deviation = abs(val - surr_mean) / window_std if window_std > 0 else 0

    # Check for spike or dip
    if deviation > 2:
        if val > surr_mean:
            return _make_spike(val, surr_mean, deviation, idx, window_std)
        else:
            return _make_dip(val, surr_mean, deviation, idx, window_std)

    # Check for step change
    if idx > 0 and idx < n - 1:
        before = values[max(0, idx - context_window):idx]
        after = values[idx + 1:min(n, idx + context_window + 1)]
        if before and after:
            step_result = _check_step_change(before, after, idx)
            if step_result:
                return step_result

    # Check for plateau
    if idx >= 2:
        plateau_result = _check_plateau(values, idx, context_window)
        if plateau_result:
            return plateau_result

    # Default: gradual drift
    return _check_gradual_drift(values, idx, context_window)


def classify_series(
    values: list[float],
    threshold_std: float = 2.0,
) -> list[AnomalyClassification]:
    """Scan a full series and classify all anomalies.

    Args:
        values: Time series values.
        threshold_std: Number of standard deviations to flag as anomaly.

    Returns:
        List of classifications for all detected anomalies.
    """
    if len(values) < 5:
        return []

    mean = sum(values) / len(values)
    std = _std(values)
    if std == 0:
        return []

    results = []
    for i, v in enumerate(values):
        if abs(v - mean) > threshold_std * std:
            classification = classify_anomaly(values, i)
            if classification.pattern != "unknown":
                results.append(classification)

    return _deduplicate_adjacent(results)


def compute_anomaly_score(classification: AnomalyClassification) -> float:
    """Compute a 0-100 anomaly score based on classification.

    Higher scores indicate more significant anomalies.
    """
    base_scores = {
        "spike": 70,
        "dip": 70,
        "step_change": 80,
        "oscillation": 50,
        "plateau": 40,
        "gradual_drift": 30,
        "unknown": 10,
    }
    base = base_scores.get(classification.pattern, 20)
    confidence_factor = classification.confidence
    severity_factor = {"critical": 1.3, "warning": 1.0, "info": 0.7}.get(classification.severity, 0.7)

    score = base * confidence_factor * severity_factor
    return min(round(score, 1), 100.0)


def summarize_anomalies(classifications: list[AnomalyClassification]) -> dict:
    """Summarize a list of anomaly classifications.

    Returns pattern counts, severity distribution, and overall summary.
    """
    if not classifications:
        return {"total": 0, "patterns": {}, "severities": {}, "summary": "No anomalies detected."}

    patterns: dict[str, int] = {}
    severities: dict[str, int] = {}
    scores = []

    for c in classifications:
        patterns[c.pattern] = patterns.get(c.pattern, 0) + 1
        severities[c.severity] = severities.get(c.severity, 0) + 1
        scores.append(compute_anomaly_score(c))

    avg_score = sum(scores) / len(scores) if scores else 0

    # Build summary text
    top_pattern = max(patterns, key=patterns.get)  # type: ignore
    summary = (
        f"{len(classifications)} anomalies detected. "
        f"Most common pattern: {top_pattern} ({patterns[top_pattern]}). "
        f"Average severity score: {avg_score:.0f}/100."
    )

    return {
        "total": len(classifications),
        "patterns": patterns,
        "severities": severities,
        "avg_score": round(avg_score, 1),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _std(values: list[float]) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


def _make_spike(val: float, surr_mean: float, deviation: float, idx: int, std: float) -> AnomalyClassification:
    """Create a spike classification."""
    confidence = min(deviation / 5, 1.0)
    severity = "critical" if deviation > 4 else "warning" if deviation > 3 else "info"
    return AnomalyClassification(
        pattern="spike",
        confidence=round(confidence, 3),
        description=f"Value {val:.2f} is {deviation:.1f} std devs above surrounding mean {surr_mean:.2f}.",
        severity=severity,
        affected_indices=[idx],
        metadata={"deviation": round(deviation, 2), "surrounding_mean": round(surr_mean, 4)},
    )


def _make_dip(val: float, surr_mean: float, deviation: float, idx: int, std: float) -> AnomalyClassification:
    """Create a dip classification."""
    confidence = min(deviation / 5, 1.0)
    severity = "critical" if deviation > 4 else "warning" if deviation > 3 else "info"
    return AnomalyClassification(
        pattern="dip",
        confidence=round(confidence, 3),
        description=f"Value {val:.2f} is {deviation:.1f} std devs below surrounding mean {surr_mean:.2f}.",
        severity=severity,
        affected_indices=[idx],
        metadata={"deviation": round(deviation, 2), "surrounding_mean": round(surr_mean, 4)},
    )


def _check_step_change(before: list[float], after: list[float], idx: int) -> AnomalyClassification | None:
    """Check if there's a step change at the boundary."""
    if not before or not after:
        return None

    before_mean = sum(before) / len(before)
    after_mean = sum(after) / len(after)
    before_std = _std(before)
    after_std = _std(after)

    # Both segments should be relatively stable (low internal variance)
    combined_std = max(before_std, after_std) if max(before_std, after_std) > 0 else 1
    shift = abs(after_mean - before_mean)

    if shift > combined_std * 1.5 and before_std < shift * 0.5 and after_std < shift * 0.5:
        confidence = min(shift / (combined_std * 3), 1.0)
        direction = "up" if after_mean > before_mean else "down"
        return AnomalyClassification(
            pattern="step_change",
            confidence=round(confidence, 3),
            description=f"Step {direction} at index {idx}: mean shifted from {before_mean:.2f} to {after_mean:.2f}.",
            severity="warning",
            affected_indices=[idx],
            metadata={
                "before_mean": round(before_mean, 4),
                "after_mean": round(after_mean, 4),
                "shift": round(shift, 4),
                "direction": direction,
            },
        )
    return None


def _check_plateau(values: list[float], idx: int, context_window: int) -> AnomalyClassification | None:
    """Check if the anomaly is part of a plateau (constant segment)."""
    n = len(values)
    val = values[idx]

    # Look for consecutive equal or near-equal values
    tolerance = abs(val) * 0.01 if val != 0 else 0.01
    plateau_start = idx
    plateau_end = idx

    while plateau_start > 0 and abs(values[plateau_start - 1] - val) <= tolerance:
        plateau_start -= 1
    while plateau_end < n - 1 and abs(values[plateau_end + 1] - val) <= tolerance:
        plateau_end += 1

    length = plateau_end - plateau_start + 1
    if length >= 3:
        confidence = min(length / 10, 1.0)
        return AnomalyClassification(
            pattern="plateau",
            confidence=round(confidence, 3),
            description=f"Constant value {val:.2f} from index {plateau_start} to {plateau_end} ({length} points).",
            severity="info",
            affected_indices=list(range(plateau_start, plateau_end + 1)),
            metadata={"value": val, "length": length},
        )
    return None


def _check_gradual_drift(values: list[float], idx: int, context_window: int) -> AnomalyClassification:
    """Default classification: gradual drift."""
    n = len(values)
    start = max(0, idx - context_window)
    end = min(n, idx + context_window + 1)
    window = values[start:end]

    # Simple trend detection
    if len(window) >= 3:
        diffs = [window[i+1] - window[i] for i in range(len(window) - 1)]
        pos = sum(1 for d in diffs if d > 0)
        direction = "upward" if pos > len(diffs) / 2 else "downward"
    else:
        direction = "uncertain"

    return AnomalyClassification(
        pattern="gradual_drift",
        confidence=0.4,
        description=f"Gradual {direction} drift detected around index {idx}.",
        severity="info",
        affected_indices=[idx],
        metadata={"direction": direction},
    )


def _deduplicate_adjacent(results: list[AnomalyClassification]) -> list[AnomalyClassification]:
    """Merge classifications that affect adjacent indices with same pattern."""
    if len(results) <= 1:
        return results

    merged = [results[0]]
    for curr in results[1:]:
        prev = merged[-1]
        # Merge if same pattern and adjacent
        if (curr.pattern == prev.pattern
                and curr.affected_indices
                and prev.affected_indices
                and abs(curr.affected_indices[0] - prev.affected_indices[-1]) <= 1):
            # Extend the previous result
            prev.affected_indices.extend(curr.affected_indices)
            prev.confidence = max(prev.confidence, curr.confidence)
            prev.severity = _max_severity(prev.severity, curr.severity)
        else:
            merged.append(curr)

    return merged


def _max_severity(a: str, b: str) -> str:
    """Return the higher severity."""
    order = {"critical": 3, "warning": 2, "info": 1}
    return a if order.get(a, 0) >= order.get(b, 0) else b
