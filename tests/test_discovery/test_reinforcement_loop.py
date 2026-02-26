"""Tests for the reinforcement loop — weight computation and application."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from business_brain.discovery.reinforcement_loop import (
    compute_weights,
    get_multiplier,
    _compute_dimension_multipliers,
    _apply_ema_smoothing,
    _clamp,
    MIN_MULTIPLIER,
    MAX_MULTIPLIER,
    EMA_ALPHA,
)
from business_brain.discovery.insight_recommender import (
    _time_priority,
    _benchmark_priority,
    _correlation_priority,
    _anomaly_priority,
    _cohort_priority,
    _forecast_priority,
)


# ---------------------------------------------------------------------------
# TestClamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self):
        assert _clamp(1.0) == 1.0
        assert _clamp(0.9) == 0.9
        assert _clamp(1.15) == 1.15

    def test_below_minimum(self):
        assert _clamp(0.5) == MIN_MULTIPLIER
        assert _clamp(0.0) == MIN_MULTIPLIER

    def test_above_maximum(self):
        assert _clamp(1.5) == MAX_MULTIPLIER
        assert _clamp(2.0) == MAX_MULTIPLIER


# ---------------------------------------------------------------------------
# TestComputeDimensionMultipliers
# ---------------------------------------------------------------------------


class TestComputeDimensionMultipliers:
    def test_empty_returns_empty(self):
        assert _compute_dimension_multipliers({}) == {}

    def test_insufficient_shown_skipped(self):
        result = _compute_dimension_multipliers({
            "anomaly": {"shown": 2, "deployed": 1, "dismissed": 0},
        })
        assert result == {}  # Below MIN_SHOWN_THRESHOLD (5)

    def test_above_average_boosted(self):
        data = {
            "anomaly":     {"shown": 10, "deployed": 5, "dismissed": 0},  # 50% rate
            "correlation": {"shown": 10, "deployed": 1, "dismissed": 0},  # 10% rate
        }
        # avg rate = 6/20 = 30%
        result = _compute_dimension_multipliers(data)
        assert result["anomaly"] > 1.0
        assert result["correlation"] < 1.0
        assert MIN_MULTIPLIER <= result["correlation"] <= MAX_MULTIPLIER
        assert MIN_MULTIPLIER <= result["anomaly"] <= MAX_MULTIPLIER

    def test_below_average_reduced(self):
        data = {
            "benchmark":   {"shown": 20, "deployed": 10, "dismissed": 1},  # 50% rate
            "anomaly":     {"shown": 20, "deployed": 1,  "dismissed": 5},  # 5% rate
        }
        result = _compute_dimension_multipliers(data)
        assert result["benchmark"] > 1.0
        assert result["anomaly"] < 1.0

    def test_zero_deployments_uses_dismiss_rate(self):
        data = {
            "anomaly": {"shown": 10, "deployed": 0, "dismissed": 8},
        }
        result = _compute_dimension_multipliers(data)
        # dismiss_rate = 0.8, multiplier = 1.0 - 0.8*0.3 = 0.76 → clamped to 0.8
        assert result["anomaly"] == MIN_MULTIPLIER

    def test_equal_rates_all_near_one(self):
        data = {
            "anomaly":     {"shown": 10, "deployed": 3, "dismissed": 0},
            "correlation": {"shown": 10, "deployed": 3, "dismissed": 0},
        }
        result = _compute_dimension_multipliers(data)
        assert abs(result["anomaly"] - 1.0) < 0.01
        assert abs(result["correlation"] - 1.0) < 0.01

    def test_mixed_sufficient_and_insufficient(self):
        data = {
            "anomaly":     {"shown": 10, "deployed": 5, "dismissed": 0},  # sufficient
            "correlation": {"shown": 2,  "deployed": 1, "dismissed": 0},  # insufficient
        }
        result = _compute_dimension_multipliers(data)
        # Only anomaly should appear (it's the only one with sufficient data)
        assert "anomaly" in result
        assert "correlation" not in result


# ---------------------------------------------------------------------------
# TestComputeWeights
# ---------------------------------------------------------------------------


class TestComputeWeights:
    def test_empty_summary_returns_empty_dicts(self):
        result = compute_weights({"by_analysis_type": {}, "by_severity": {}})
        assert result["analysis_type_multipliers"] == {}
        assert result["severity_multipliers"] == {}
        assert result["insight_type_multipliers"] == {}

    def test_full_summary_produces_multipliers(self):
        summary = {
            "total_events": 80,
            "by_analysis_type": {
                "benchmark": {"shown": 20, "deployed": 10, "dismissed": 2},
                "anomaly":   {"shown": 20, "deployed": 2,  "dismissed": 5},
            },
            "by_severity": {
                "critical": {"shown": 10, "deployed": 8, "dismissed": 0},
                "info":     {"shown": 10, "deployed": 1, "dismissed": 3},
            },
        }
        result = compute_weights(summary)
        # benchmark has higher deploy rate → boosted
        assert result["analysis_type_multipliers"]["benchmark"] > 1.0
        assert result["analysis_type_multipliers"]["anomaly"] < 1.0
        # critical has higher deploy rate → boosted
        assert result["severity_multipliers"]["critical"] > 1.0
        assert result["severity_multipliers"]["info"] < 1.0


# ---------------------------------------------------------------------------
# TestGetMultiplier
# ---------------------------------------------------------------------------


class TestGetMultiplier:
    def test_none_weights_returns_one(self):
        assert get_multiplier(None, "analysis_type_multipliers", "benchmark") == 1.0

    def test_missing_key_returns_one(self):
        weights = MagicMock()
        weights.analysis_type_multipliers = {"benchmark": 1.1}
        assert get_multiplier(weights, "analysis_type_multipliers", "correlation") == 1.0

    def test_existing_key_returns_value(self):
        weights = MagicMock()
        weights.analysis_type_multipliers = {"benchmark": 1.15}
        assert get_multiplier(weights, "analysis_type_multipliers", "benchmark") == 1.15

    def test_missing_dimension_returns_one(self):
        weights = MagicMock()
        weights.nonexistent_multipliers = None
        assert get_multiplier(weights, "nonexistent_multipliers", "benchmark") == 1.0


# ---------------------------------------------------------------------------
# TestEMASmoothing
# ---------------------------------------------------------------------------


class TestEMASmoothing:
    def test_no_previous_uses_raw(self):
        raw = {
            "analysis_type_multipliers": {"benchmark": 1.15},
            "severity_multipliers": {},
            "insight_type_multipliers": {},
        }
        result = _apply_ema_smoothing(raw, None)
        assert result["analysis_type_multipliers"]["benchmark"] == 1.15

    def test_blends_old_and_new(self):
        raw = {
            "analysis_type_multipliers": {"benchmark": 1.2},
            "severity_multipliers": {},
            "insight_type_multipliers": {},
        }
        prev = MagicMock()
        prev.analysis_type_multipliers = {"benchmark": 1.0}
        prev.severity_multipliers = {}
        prev.insight_type_multipliers = {}
        result = _apply_ema_smoothing(raw, prev)
        # EMA: 0.7 * 1.0 + 0.3 * 1.2 = 1.06
        expected = (1 - EMA_ALPHA) * 1.0 + EMA_ALPHA * 1.2
        assert abs(result["analysis_type_multipliers"]["benchmark"] - expected) < 0.001

    def test_handles_new_and_old_keys(self):
        raw = {
            "analysis_type_multipliers": {"new_type": 1.1},
            "severity_multipliers": {},
            "insight_type_multipliers": {},
        }
        prev = MagicMock()
        prev.analysis_type_multipliers = {"old_type": 1.05}
        prev.severity_multipliers = {}
        prev.insight_type_multipliers = {}
        result = _apply_ema_smoothing(raw, prev)
        # new_type: 0.7*1.0 + 0.3*1.1 = 1.03 (prev defaults to 1.0)
        assert "new_type" in result["analysis_type_multipliers"]
        # old_type: 0.7*1.05 + 0.3*1.0 = 1.035 (raw defaults to 1.0)
        assert "old_type" in result["analysis_type_multipliers"]
        expected_old = (1 - EMA_ALPHA) * 1.05 + EMA_ALPHA * 1.0
        assert abs(result["analysis_type_multipliers"]["old_type"] - expected_old) < 0.001


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_priority_functions_default_multiplier_unchanged(self):
        """multiplier=1.0 (default) should produce the same output as before."""
        assert _time_priority(500, 3) == _time_priority(500, 3, multiplier=1.0)
        assert _benchmark_priority(500, 5, 3) == _benchmark_priority(500, 5, 3, multiplier=1.0)
        assert _correlation_priority(6, 3) == _correlation_priority(6, 3, multiplier=1.0)
        assert _anomaly_priority(500, 3) == _anomaly_priority(500, 3, multiplier=1.0)
        assert _cohort_priority(500, 3) == _cohort_priority(500, 3, multiplier=1.0)
        assert _forecast_priority(500, 3) == _forecast_priority(500, 3, multiplier=1.0)

    def test_priority_functions_multiplier_affects_base(self):
        """A multiplier > 1 should increase base priority."""
        # time_priority base is 80. With multiplier=1.2, base=96.
        high = _time_priority(50, 0, multiplier=1.2)
        low = _time_priority(50, 0, multiplier=0.8)
        normal = _time_priority(50, 0, multiplier=1.0)
        assert high > normal > low

    def test_quality_gate_without_weights_unchanged(self):
        """reinforcement_weights=None should produce the same scoring."""
        from business_brain.db.discovery_models import Insight
        from business_brain.discovery.insight_quality_gate import _apply_business_scoring

        insight = Insight(
            insight_type="anomaly",
            severity="critical",
            title="Test outlier",
            description="An anomaly was detected",
            suggested_actions=["Check the data", "Investigate root cause"],
            source_tables=["orders"],
        )
        _apply_business_scoring(insight, reinforcement_weights=None)
        score_without = insight.impact_score

        insight2 = Insight(
            insight_type="anomaly",
            severity="critical",
            title="Test outlier",
            description="An anomaly was detected",
            suggested_actions=["Check the data", "Investigate root cause"],
            source_tables=["orders"],
        )
        _apply_business_scoring(insight2)
        score_default = insight2.impact_score

        assert score_without == score_default
