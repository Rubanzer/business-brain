"""Interestingness Scorer — rates findings + spawns follow-ups.

5 sub-scores (all 0.0-1.0):
- surprise: how unexpected is this result?
- magnitude: how large is the effect?
- variance: how spread is the data?
- stability: how robust is the finding? (bootstrap)
- coverage: what fraction of data does this cover?

Plus follow-up spawning (Gap #7):
- DETECT_ANOMALY (high) → ATTRIBUTE candidates
- CORRELATE (high) → RANK candidates (does correlation hold across segments?)
"""

from __future__ import annotations

import math
from typing import Any

from business_brain.analysis.models import AnalysisResult
from business_brain.analysis.track1.enumerator import AnalysisCandidate, _make_dedup_key
from business_brain.analysis.track1.fingerprinter import TableFingerprint

# ---------------------------------------------------------------------------
# Default weights (tunable via LearningState)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "surprise": 0.30,
    "magnitude": 0.25,
    "variance": 0.15,
    "stability": 0.15,
    "coverage": 0.15,
}


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------


def _score_surprise(result: AnalysisResult) -> float:
    """How unexpected is this finding?"""
    data = result.result_data or {}

    if result.operation_type == "DETECT_ANOMALY":
        count = data.get("count", 0)
        total = data.get("total", 1)
        # More anomalies relative to total = more surprising
        rate = count / max(total, 1)
        if rate > 0.1:
            return 0.9  # lots of anomalies = very surprising
        if count > 0:
            return 0.5 + min(count / 10, 0.4)
        return 0.1

    if result.operation_type == "CORRELATE":
        r = abs(data.get("pearson_r", 0.0))
        # Very high or very low correlations are surprising
        if r > 0.8:
            return 0.8
        if r > 0.6:
            return 0.6
        if r < 0.1:
            return 0.3  # no correlation can be surprising too
        return 0.4

    if result.operation_type == "RANK":
        comparison = data.get("comparison", {})
        if comparison:
            p = comparison.get("p_value", 1.0)
            if p < 0.001:
                return 0.9
            if p < 0.01:
                return 0.7
            if p < 0.05:
                return 0.5
        return 0.3

    if result.operation_type == "DESCRIBE":
        stats = data.get("stats", {})
        skewness = abs(stats.get("skewness", 0.0))
        kurtosis = abs(stats.get("kurtosis", 0.0))
        if skewness > 2 or kurtosis > 7:
            return 0.7
        return 0.3

    return 0.3


def _score_magnitude(result: AnalysisResult) -> float:
    """How large is the effect?"""
    data = result.result_data or {}

    if result.operation_type == "CORRELATE":
        r = abs(data.get("pearson_r", 0.0))
        return min(r, 1.0)

    if result.operation_type == "RANK":
        comparison = data.get("comparison", {})
        d = abs(comparison.get("cohens_d", 0.0)) if comparison else 0.0
        eta = comparison.get("eta_squared", 0.0) if comparison else 0.0
        # Cohen's d: 0.2=small, 0.5=medium, 0.8=large
        if d > 0:
            return min(d / 1.0, 1.0)
        # eta-squared: 0.01=small, 0.06=medium, 0.14=large
        return min(eta / 0.14, 1.0)

    if result.operation_type == "DETECT_ANOMALY":
        anomalies = data.get("anomalies", [])
        if anomalies:
            max_z = max(abs(a.get("z_score", 0.0)) for a in anomalies)
            return min(max_z / 5.0, 1.0)
        return 0.1

    if result.operation_type == "DESCRIBE":
        stats = data.get("stats", {})
        cv = stats.get("stdev", 0.0) / abs(stats.get("mean", 1.0)) if stats.get("mean", 0) != 0 else 0.0
        return min(cv, 1.0)

    return 0.3


def _score_variance(result: AnalysisResult) -> float:
    """How much spread in the data? (Higher spread = more interesting patterns possible.)"""
    data = result.result_data or {}

    if result.operation_type == "DESCRIBE":
        stats = data.get("stats", {})
        iqr = stats.get("iqr", 0.0)
        range_val = stats.get("max", 0.0) - stats.get("min", 0.0)
        if range_val > 0:
            return min(iqr / range_val, 1.0)
        return 0.2

    if result.operation_type == "RANK":
        ranked = data.get("ranked", [])
        if len(ranked) >= 2:
            values = []
            for r in ranked:
                for k, v in r.items():
                    if k.startswith("avg_") and v is not None:
                        try:
                            values.append(float(v))
                        except (ValueError, TypeError):
                            pass
            if len(values) >= 2:
                spread = (max(values) - min(values)) / abs(max(values)) if max(values) != 0 else 0.0
                return min(abs(spread), 1.0)
        return 0.3

    return 0.4


def _score_stability(result: AnalysisResult) -> float:
    """How robust is the finding?"""
    data = result.result_data or {}

    if result.operation_type == "CORRELATE":
        p = data.get("pearson_p", 1.0)
        n = data.get("n", 0)
        # More data + lower p = more stable
        data_bonus = min(n / 1000, 0.3)
        p_score = 1.0 - min(p, 1.0)
        return min(p_score + data_bonus, 1.0)

    if result.operation_type == "RANK":
        comparison = data.get("comparison", {})
        p = comparison.get("p_value", 1.0) if comparison else 1.0
        return 1.0 - min(p, 1.0)

    if result.operation_type == "DETECT_ANOMALY":
        total = data.get("total", 0)
        return min(total / 500, 1.0)

    row_count = data.get("row_count", 0)
    return min(row_count / 500, 1.0)


def _score_coverage(result: AnalysisResult) -> float:
    """What fraction of the data does this cover?"""
    data = result.result_data or {}
    row_count = data.get("row_count", 0)
    stats = data.get("stats", {})
    total_count = stats.get("count", row_count)

    if total_count == 0:
        return 0.1

    # Penalize if too much data was filtered out by nulls
    null_rate = 1.0 - (total_count / max(row_count, 1)) if row_count > 0 else 0.0
    return max(1.0 - null_rate, 0.1)


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------


def score_result(
    result: AnalysisResult,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute interestingness score and set breakdown on the result."""
    w = weights or DEFAULT_WEIGHTS

    breakdown = {
        "surprise": _score_surprise(result),
        "magnitude": _score_magnitude(result),
        "variance": _score_variance(result),
        "stability": _score_stability(result),
        "coverage": _score_coverage(result),
    }

    total = sum(breakdown[k] * w.get(k, 0.2) for k in breakdown)
    result.interestingness_score = total
    result.interestingness_breakdown = breakdown
    return total


def score_batch(
    results: list[AnalysisResult],
    weights: dict[str, float] | None = None,
) -> list[AnalysisResult]:
    """Score all results and sort by interestingness (descending)."""
    for r in results:
        score_result(r, weights)
    results.sort(key=lambda r: r.interestingness_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Follow-up spawning (Gap #7)
# ---------------------------------------------------------------------------

_MAX_SOURCE_FINDINGS = 5
_MAX_FOLLOWUPS_PER_FINDING = 3
_FOLLOWUP_SCORE_THRESHOLD = 0.6


def spawn_followups(
    top_findings: list[AnalysisResult],
    fingerprints: dict[str, TableFingerprint],
) -> list[AnalysisCandidate]:
    """Generate follow-up candidates from high-scoring findings.

    - DETECT_ANOMALY (high) → ATTRIBUTE candidates (what explains this?)
    - CORRELATE (high) → RANK candidates (does this hold across segments?)
    - Any segmented finding (high) → same finding without segmenters (is this global?)
    """
    candidates: list[AnalysisCandidate] = []
    source_count = 0

    for finding in top_findings:
        if finding.interestingness_score < _FOLLOWUP_SCORE_THRESHOLD:
            continue
        if source_count >= _MAX_SOURCE_FINDINGS:
            break

        fp = fingerprints.get(finding.table_name)
        if not fp:
            continue

        spawned = 0
        source_count += 1

        # DETECT_ANOMALY → ATTRIBUTE (what dimension explains the anomaly?)
        if finding.operation_type == "DETECT_ANOMALY" and spawned < _MAX_FOLLOWUPS_PER_FINDING:
            for dim in fp.dimensions[:3]:
                if spawned >= _MAX_FOLLOWUPS_PER_FINDING:
                    break
                c = AnalysisCandidate(
                    operation="RANK",
                    table_name=finding.table_name,
                    target=list(finding.target),
                    segmenters=[dim],
                    tier=1,  # Run as Tier 1 (exhaustive-class since it's a follow-up)
                    priority_score=finding.interestingness_score,
                )
                c.dedup_key = _make_dedup_key("RANK", finding.table_name, finding.target, [dim], [])
                candidates.append(c)
                spawned += 1

        # CORRELATE → RANK per dimension (does this hold across segments?)
        if finding.operation_type == "CORRELATE" and spawned < _MAX_FOLLOWUPS_PER_FINDING:
            for dim in fp.dimensions[:3]:
                if spawned >= _MAX_FOLLOWUPS_PER_FINDING:
                    break
                c = AnalysisCandidate(
                    operation="RANK",
                    table_name=finding.table_name,
                    target=[finding.target[0]],
                    segmenters=[dim],
                    tier=1,
                    priority_score=finding.interestingness_score,
                )
                c.dedup_key = _make_dedup_key("RANK", finding.table_name, [finding.target[0]], [dim], [])
                candidates.append(c)
                spawned += 1

        # Segmented finding → un-segmented version (is this a global pattern?)
        if finding.segmenters and spawned < _MAX_FOLLOWUPS_PER_FINDING:
            c = AnalysisCandidate(
                operation=finding.operation_type,
                table_name=finding.table_name,
                target=list(finding.target),
                segmenters=[],
                tier=0,
                priority_score=finding.interestingness_score,
            )
            c.dedup_key = _make_dedup_key(finding.operation_type, finding.table_name, finding.target, [], [])
            candidates.append(c)
            spawned += 1

    return candidates
