"""Tests for the time intelligence module."""

from business_brain.discovery.time_intelligence import (
    compute_moving_average,
    compute_period_change,
    detect_changepoints,
    detect_trend,
    find_min_max_periods,
    forecast_exponential,
    forecast_linear,
)


class TestComputePeriodChange:
    """Test period-over-period change calculation."""

    def test_basic_increase(self):
        result = compute_period_change(120, 100)
        assert result.absolute_change == 20
        assert result.pct_change == 20.0

    def test_basic_decrease(self):
        result = compute_period_change(80, 100)
        assert result.absolute_change == -20
        assert result.pct_change == -20.0

    def test_no_change(self):
        result = compute_period_change(100, 100)
        assert result.absolute_change == 0
        assert result.pct_change == 0.0

    def test_previous_zero(self):
        result = compute_period_change(50, 0)
        assert result.absolute_change == 50
        assert result.pct_change is None

    def test_both_zero(self):
        result = compute_period_change(0, 0)
        assert result.absolute_change == 0
        assert result.pct_change is None

    def test_negative_values(self):
        result = compute_period_change(-50, -100)
        assert result.absolute_change == 50
        assert result.pct_change == 50.0

    def test_from_negative_to_positive(self):
        result = compute_period_change(10, -10)
        assert result.absolute_change == 20
        assert result.pct_change == 200.0

    def test_small_change(self):
        result = compute_period_change(100.5, 100.0)
        assert abs(result.absolute_change - 0.5) < 0.001
        assert abs(result.pct_change - 0.5) < 0.001


class TestDetectTrend:
    """Test trend direction detection."""

    def test_increasing_trend(self):
        values = [10, 20, 30, 40, 50]
        result = detect_trend(values)
        assert result.direction == "increasing"
        assert result.r_squared > 0.9

    def test_decreasing_trend(self):
        values = [50, 40, 30, 20, 10]
        result = detect_trend(values)
        assert result.direction == "decreasing"
        assert result.r_squared > 0.9

    def test_stable_trend(self):
        values = [100, 100.1, 99.9, 100.0, 100.1]
        result = detect_trend(values)
        assert result.direction == "stable"

    def test_volatile_trend(self):
        values = [10, 50, 5, 80, 3, 90]
        result = detect_trend(values)
        assert result.direction in ("volatile", "increasing")

    def test_single_value(self):
        result = detect_trend([42])
        assert result.direction == "stable"
        assert result.magnitude == 0.0

    def test_two_values_increasing(self):
        result = detect_trend([10, 20])
        assert result.direction == "increasing"

    def test_two_values_decreasing(self):
        result = detect_trend([20, 10])
        assert result.direction == "decreasing"

    def test_empty_values(self):
        result = detect_trend([])
        assert result.direction == "stable"
        assert result.r_squared == 0.0

    def test_all_same_values(self):
        result = detect_trend([42, 42, 42, 42])
        assert result.direction == "stable"
        assert result.r_squared == 1.0

    def test_magnitude_positive(self):
        values = [100, 110, 120, 130, 140]
        result = detect_trend(values)
        assert result.magnitude > 0

    def test_magnitude_negative(self):
        values = [140, 130, 120, 110, 100]
        result = detect_trend(values)
        assert result.magnitude < 0

    def test_perfect_linear_r_squared(self):
        values = [0, 5, 10, 15, 20]
        result = detect_trend(values)
        assert abs(result.r_squared - 1.0) < 0.001


class TestFindMinMaxPeriods:
    """Test min/max period identification."""

    def test_basic(self):
        values = [10, 20, 5, 30, 15]
        result = find_min_max_periods(values)
        assert result.max_value == 30
        assert result.max_index == 3
        assert result.min_value == 5
        assert result.min_index == 2

    def test_first_is_max(self):
        values = [100, 50, 75]
        result = find_min_max_periods(values)
        assert result.max_index == 0

    def test_last_is_max(self):
        values = [50, 75, 100]
        result = find_min_max_periods(values)
        assert result.max_index == 2

    def test_single_value(self):
        result = find_min_max_periods([42])
        assert result.max_value == 42
        assert result.min_value == 42
        assert result.max_index == 0
        assert result.min_index == 0

    def test_empty_returns_none(self):
        assert find_min_max_periods([]) is None

    def test_all_same(self):
        result = find_min_max_periods([10, 10, 10])
        assert result.max_value == 10
        assert result.min_value == 10

    def test_negative_values(self):
        values = [-10, -20, -5]
        result = find_min_max_periods(values)
        assert result.max_value == -5
        assert result.min_value == -20


class TestComputeMovingAverage:
    """Test moving average computation."""

    def test_window_3(self):
        values = [10, 20, 30, 40, 50]
        result = compute_moving_average(values, window=3)
        assert len(result) == 5
        # Last value: avg(30, 40, 50) = 40.0
        assert abs(result[4] - 40.0) < 0.001

    def test_window_1(self):
        values = [10, 20, 30]
        result = compute_moving_average(values, window=1)
        assert result == values

    def test_window_larger_than_series(self):
        values = [10, 20]
        result = compute_moving_average(values, window=5)
        assert result == values

    def test_empty_series(self):
        result = compute_moving_average([], window=3)
        assert result == []

    def test_single_value(self):
        result = compute_moving_average([42], window=3)
        assert result == [42]

    def test_window_0(self):
        values = [10, 20, 30]
        result = compute_moving_average(values, window=0)
        assert result == values


class TestDetectChangepoints:
    """Test changepoint detection."""

    def test_no_changepoints(self):
        values = [10, 11, 10, 11, 10]
        assert detect_changepoints(values) == []

    def test_single_spike(self):
        values = [10, 10, 10, 100, 10, 10]
        cps = detect_changepoints(values, threshold=2.0)
        assert 3 in cps  # Spike at index 3

    def test_too_few_values(self):
        assert detect_changepoints([10, 20]) == []
        assert detect_changepoints([]) == []

    def test_constant_values(self):
        values = [42, 42, 42, 42, 42]
        assert detect_changepoints(values) == []

    def test_step_change(self):
        values = [10, 10, 10, 50, 50, 50]
        cps = detect_changepoints(values, threshold=1.5)
        assert 3 in cps

    def test_high_threshold_fewer_changepoints(self):
        values = [10, 10, 30, 10, 10, 50, 10]
        cps_low = detect_changepoints(values, threshold=1.0)
        cps_high = detect_changepoints(values, threshold=3.0)
        assert len(cps_high) <= len(cps_low)

    def test_gradual_increase_no_changepoints(self):
        values = [10, 12, 14, 16, 18, 20]
        cps = detect_changepoints(values, threshold=2.0)
        assert cps == []


class TestForecastLinear:
    def test_perfect_upward(self):
        values = [10, 20, 30, 40, 50]
        result = forecast_linear(values, 3)
        assert len(result.predicted_values) == 3
        assert result.predicted_values[0] == 60
        assert result.predicted_values[1] == 70
        assert result.predicted_values[2] == 80
        assert result.method == "linear"
        assert result.confidence == "high"

    def test_flat_series(self):
        values = [100, 100, 100, 100]
        result = forecast_linear(values, 2)
        assert result.predicted_values == [100.0, 100.0]

    def test_downward_trend(self):
        values = [100, 90, 80, 70, 60]
        result = forecast_linear(values, 2)
        assert result.predicted_values[0] == 50
        assert result.predicted_values[1] == 40

    def test_too_few_values(self):
        result = forecast_linear([10], 3)
        assert result.predicted_values == []
        assert result.confidence == "low"

    def test_zero_periods_ahead(self):
        result = forecast_linear([10, 20, 30], 0)
        assert result.predicted_values == []

    def test_noisy_data_low_confidence(self):
        values = [10, 50, 5, 45, 8, 42]
        result = forecast_linear(values, 2)
        assert len(result.predicted_values) == 2
        # Should have low or medium confidence due to noise
        assert result.confidence in ("low", "medium")

    def test_two_values(self):
        result = forecast_linear([10, 20], 1)
        assert result.predicted_values == [30.0]

    def test_all_same_value(self):
        result = forecast_linear([42, 42, 42], 3)
        for v in result.predicted_values:
            assert abs(v - 42) < 0.01


class TestForecastExponential:
    def test_flat_series(self):
        values = [100, 100, 100, 100]
        result = forecast_exponential(values, 3)
        assert len(result.predicted_values) == 3
        for v in result.predicted_values:
            assert abs(v - 100) < 0.1
        assert result.method == "exponential"

    def test_returns_constant_forecast(self):
        """Exponential smoothing forecasts are always flat (last smoothed value)."""
        values = [10, 20, 30, 40, 50]
        result = forecast_exponential(values, 3)
        assert len(set(result.predicted_values)) == 1

    def test_too_few_values(self):
        result = forecast_exponential([10], 3)
        assert result.predicted_values == []

    def test_zero_periods_ahead(self):
        result = forecast_exponential([10, 20, 30], 0)
        assert result.predicted_values == []

    def test_alpha_near_one(self):
        """Alpha near 1 = last value dominates."""
        values = [10, 20, 30, 40, 50]
        result = forecast_exponential(values, 1, alpha=0.99)
        assert abs(result.predicted_values[0] - 50) < 1

    def test_alpha_near_zero(self):
        """Alpha near 0 = first value dominates."""
        values = [10, 20, 30, 40, 50]
        result = forecast_exponential(values, 1, alpha=0.01)
        # Should be much closer to the initial value
        assert result.predicted_values[0] < 20

    def test_invalid_alpha_clamped(self):
        """Alpha out of range should be clamped to 0.3."""
        values = [10, 20, 30]
        result1 = forecast_exponential(values, 1, alpha=0.3)
        result2 = forecast_exponential(values, 1, alpha=1.5)
        assert result1.predicted_values == result2.predicted_values

    def test_stable_data_high_confidence(self):
        values = [100, 101, 99, 100, 101, 99, 100]
        result = forecast_exponential(values, 2)
        assert result.confidence == "high"

    def test_noisy_data_lower_confidence(self):
        values = [10, 100, 5, 90, 15, 85]
        result = forecast_exponential(values, 2)
        assert result.confidence in ("low", "medium")
