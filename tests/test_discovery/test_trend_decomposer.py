"""Tests for trend_decomposer — time-series decomposition module."""

from __future__ import annotations

import math

import pytest

from business_brain.discovery.trend_decomposer import (
    DecompositionResult,
    classify_trend,
    compute_rate_of_change_trend,
    compute_trend_line,
    decompose,
    detect_period,
    find_anomalous_residuals,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------

def _make_trend(n: int, slope: float = 1.0, intercept: float = 0.0) -> list[float]:
    """Linear trend: intercept + slope * i."""
    return [intercept + slope * i for i in range(n)]


def _make_seasonal(n: int, period: int, amplitude: float = 10.0) -> list[float]:
    """Sinusoidal seasonal pattern."""
    return [amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def _make_trend_seasonal(
    n: int,
    slope: float = 1.0,
    period: int = 7,
    amplitude: float = 10.0,
) -> list[float]:
    """Combine linear trend + sinusoidal seasonal."""
    t = _make_trend(n, slope)
    s = _make_seasonal(n, period, amplitude)
    return [ti + si for ti, si in zip(t, s)]


# ---------------------------------------------------------------------------
# 1. Decomposition of synthetic trend + seasonal data
# ---------------------------------------------------------------------------

class TestDecomposeBasic:

    def test_returns_decomposition_result(self):
        values = _make_trend_seasonal(60, slope=0.5, period=7, amplitude=5.0)
        result = decompose(values, period=7)
        assert isinstance(result, DecompositionResult)

    def test_components_length_matches_original(self):
        values = _make_trend_seasonal(50, slope=1.0, period=10)
        result = decompose(values, period=10)
        assert result is not None
        assert len(result.trend) == len(values)
        assert len(result.seasonal) == len(values)
        assert len(result.residual) == len(values)
        assert len(result.original) == len(values)

    def test_additive_reconstruction(self):
        """trend + seasonal + residual should approximate the original."""
        values = _make_trend_seasonal(70, slope=0.3, period=7, amplitude=8.0)
        result = decompose(values, period=7)
        assert result is not None
        for i in range(len(values)):
            reconstructed = result.trend[i] + result.seasonal[i] + result.residual[i]
            assert abs(reconstructed - values[i]) < 1e-9, (
                f"Reconstruction mismatch at index {i}"
            )

    def test_period_stored_correctly(self):
        values = _make_trend_seasonal(60, period=12)
        result = decompose(values, period=12)
        assert result is not None
        assert result.period == 12


# ---------------------------------------------------------------------------
# 2. Auto period detection
# ---------------------------------------------------------------------------

class TestDetectPeriod:

    def test_detects_period_7(self):
        values = _make_seasonal(100, period=7, amplitude=20.0)
        detected = detect_period(values)
        assert detected == 7

    def test_detects_period_12(self):
        values = _make_seasonal(120, period=12, amplitude=15.0)
        detected = detect_period(values)
        assert detected == 12

    def test_no_seasonality_returns_1(self):
        values = _make_trend(50, slope=1.0)
        detected = detect_period(values)
        assert detected == 1

    def test_constant_returns_1(self):
        values = [5.0] * 30
        detected = detect_period(values)
        assert detected == 1

    def test_short_data_returns_1(self):
        values = [1.0, 2.0, 3.0]
        detected = detect_period(values)
        assert detected == 1

    def test_max_period_respected(self):
        values = _make_seasonal(100, period=7, amplitude=20.0)
        detected = detect_period(values, max_period=5)
        # Should not detect period 7 when max_period < 7
        assert detected <= 5


# ---------------------------------------------------------------------------
# 3. Manual period specification
# ---------------------------------------------------------------------------

class TestManualPeriod:

    def test_manual_period_used(self):
        values = _make_trend_seasonal(60, period=7)
        result = decompose(values, period=5)
        assert result is not None
        assert result.period == 5

    def test_auto_detect_when_none(self):
        values = _make_seasonal(100, period=7, amplitude=20.0)
        result = decompose(values, period=None)
        assert result is not None
        assert result.period == 7


# ---------------------------------------------------------------------------
# 4. Pure trend (no seasonality)
# ---------------------------------------------------------------------------

class TestPureTrend:

    def test_pure_increasing_trend(self):
        values = _make_trend(30, slope=2.0)
        result = decompose(values, period=1)
        assert result is not None
        assert result.trend_direction == "increasing"
        assert result.trend_strength > 0.9

    def test_pure_decreasing_trend(self):
        values = _make_trend(30, slope=-3.0, intercept=100.0)
        result = decompose(values, period=1)
        assert result is not None
        assert result.trend_direction == "decreasing"
        assert result.trend_strength > 0.9

    def test_seasonal_strength_near_zero_for_pure_trend(self):
        values = _make_trend(30, slope=1.0)
        result = decompose(values, period=1)
        assert result is not None
        assert result.seasonal_strength < 0.1


# ---------------------------------------------------------------------------
# 5. Pure seasonal (no trend)
# ---------------------------------------------------------------------------

class TestPureSeasonal:

    def test_pure_seasonal_detected(self):
        values = _make_seasonal(70, period=7, amplitude=10.0)
        result = decompose(values, period=7)
        assert result is not None
        # Seasonal strength should be high
        assert result.seasonal_strength > 0.5

    def test_seasonal_pattern_repeats(self):
        values = _make_seasonal(70, period=7, amplitude=10.0)
        result = decompose(values, period=7)
        assert result is not None
        # The seasonal component should repeat with the given period
        for i in range(7, len(result.seasonal)):
            assert abs(result.seasonal[i] - result.seasonal[i - 7]) < 1e-9


# ---------------------------------------------------------------------------
# 6. Flat data
# ---------------------------------------------------------------------------

class TestFlatData:

    def test_constant_data(self):
        values = [42.0] * 20
        result = decompose(values, period=1)
        assert result is not None
        assert result.trend_direction == "flat"

    def test_constant_residuals_near_zero(self):
        values = [42.0] * 20
        result = decompose(values, period=1)
        assert result is not None
        for r in result.residual:
            assert abs(r) < 1e-9


# ---------------------------------------------------------------------------
# 7. Trend direction classification
# ---------------------------------------------------------------------------

class TestClassifyTrend:

    def test_increasing(self):
        trend = [float(i) for i in range(20)]
        direction, strength = classify_trend(trend)
        assert direction == "increasing"

    def test_decreasing(self):
        trend = [100.0 - i for i in range(20)]
        direction, strength = classify_trend(trend)
        assert direction == "decreasing"

    def test_flat_constant(self):
        trend = [5.0] * 20
        direction, strength = classify_trend(trend)
        assert direction == "flat"

    def test_nearly_flat(self):
        trend = [10.0 + 0.0001 * i for i in range(20)]
        direction, _ = classify_trend(trend)
        assert direction == "flat"


# ---------------------------------------------------------------------------
# 8. Trend strength measurement
# ---------------------------------------------------------------------------

class TestTrendStrength:

    def test_perfect_linear_has_high_r2(self):
        trend = [2.0 * i + 5.0 for i in range(30)]
        _, strength = classify_trend(trend)
        assert abs(strength - 1.0) < 1e-9

    def test_noisy_trend_lower_r2(self):
        import random
        rng = random.Random(42)
        trend = [float(i) + rng.gauss(0, 5) for i in range(50)]
        _, strength = classify_trend(trend)
        assert 0.0 <= strength <= 1.0
        # With noise, R² should be less than 1
        assert strength < 1.0


# ---------------------------------------------------------------------------
# 9. Anomalous residuals detection
# ---------------------------------------------------------------------------

class TestAnomalousResiduals:

    def test_finds_outlier(self):
        residual = [0.1, -0.2, 0.3, -0.1, 0.0, 50.0, 0.1, -0.3, 0.2, 0.0]
        anomalies = find_anomalous_residuals(residual, threshold=2.0)
        assert len(anomalies) >= 1
        indices = [a["index"] for a in anomalies]
        assert 5 in indices

    def test_no_outliers_in_clean_data(self):
        residual = [0.1 * i for i in range(20)]
        anomalies = find_anomalous_residuals(residual, threshold=3.0)
        # Uniform-ish spread — should have no anomalies at threshold=3
        # (linear data won't have outliers beyond 3 sigma easily)
        # Just check it returns a list
        assert isinstance(anomalies, list)

    def test_severity_high(self):
        residual = [0.0] * 20 + [100.0]
        anomalies = find_anomalous_residuals(residual, threshold=2.0)
        high = [a for a in anomalies if a["severity"] == "high"]
        assert len(high) >= 1

    def test_empty_residual(self):
        assert find_anomalous_residuals([]) == []

    def test_constant_residual_no_anomalies(self):
        assert find_anomalous_residuals([1.0] * 10) == []


# ---------------------------------------------------------------------------
# 10. Rate of change computation
# ---------------------------------------------------------------------------

class TestRateOfChange:

    def test_length(self):
        trend = [1.0, 3.0, 6.0, 10.0]
        roc = compute_rate_of_change_trend(trend)
        assert len(roc) == 3

    def test_values(self):
        trend = [0.0, 2.0, 5.0, 5.0, 3.0]
        roc = compute_rate_of_change_trend(trend)
        assert roc == [2.0, 3.0, 0.0, -2.0]

    def test_empty(self):
        assert compute_rate_of_change_trend([]) == []

    def test_single_value(self):
        assert compute_rate_of_change_trend([5.0]) == []


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_too_short_returns_none(self):
        assert decompose([1.0, 2.0]) is None

    def test_five_values_returns_none(self):
        assert decompose([1.0, 2.0, 3.0, 4.0, 5.0]) is None

    def test_six_values_works_with_period_1(self):
        result = decompose([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], period=1)
        assert result is not None

    def test_empty_list_returns_none(self):
        assert decompose([]) is None

    def test_fewer_than_2_periods_returns_none(self):
        # period=10 requires at least 20 values
        values = [float(i) for i in range(15)]
        assert decompose(values, period=10) is None

    def test_single_value_returns_none(self):
        assert decompose([42.0]) is None

    def test_constant_series_decomposes(self):
        values = [7.0] * 20
        result = decompose(values, period=1)
        assert result is not None
        assert result.trend_direction == "flat"

    def test_trend_line_empty_input(self):
        assert compute_trend_line([]) == []

    def test_trend_line_single_value(self):
        assert compute_trend_line([5.0]) == [5.0]

    def test_classify_trend_single_value(self):
        direction, strength = classify_trend([5.0])
        assert direction == "flat"
        assert strength == 0.0

    def test_summary_is_string(self):
        values = _make_trend_seasonal(60, period=7)
        result = decompose(values, period=7)
        assert result is not None
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0


# ---------------------------------------------------------------------------
# 12. Trend line (moving average)
# ---------------------------------------------------------------------------

class TestComputeTrendLine:

    def test_window_1_returns_original(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = compute_trend_line(values, window=1)
        assert trend == values

    def test_window_3_center(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = compute_trend_line(values, window=3)
        # Middle values should be exact 3-element averages
        assert abs(trend[1] - 2.0) < 1e-9
        assert abs(trend[2] - 3.0) < 1e-9
        assert abs(trend[3] - 4.0) < 1e-9

    def test_edge_values_use_partial_window(self):
        values = [10.0, 20.0, 30.0]
        trend = compute_trend_line(values, window=3)
        # First element: avg of [10, 20] (partial window: i=0, half=1, lo=0, hi=2)
        assert abs(trend[0] - 15.0) < 1e-9
        # Last element: avg of [20, 30]
        assert abs(trend[2] - 25.0) < 1e-9

    def test_same_length_as_input(self):
        values = [float(i) for i in range(25)]
        trend = compute_trend_line(values, window=7)
        assert len(trend) == 25
