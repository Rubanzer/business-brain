"""Tests for rolling statistics module."""

from business_brain.discovery.rolling_stats import (
    RollingResult,
    compute_volatility,
    detect_regime_changes,
    find_outlier_windows,
    rolling_correlation,
    rolling_statistics,
)


class TestRollingStatistics:
    def test_basic(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = rolling_statistics(values, window=3)
        assert result is not None
        assert len(result.rolling_mean) == 10
        # First 2 should be None (window=3)
        assert result.rolling_mean[0] is None
        assert result.rolling_mean[1] is None
        assert result.rolling_mean[2] is not None

    def test_rolling_mean_values(self):
        values = [10, 20, 30, 40, 50]
        result = rolling_statistics(values, window=3)
        assert result.rolling_mean[2] == 20.0  # (10+20+30)/3
        assert result.rolling_mean[3] == 30.0  # (20+30+40)/3
        assert result.rolling_mean[4] == 40.0  # (30+40+50)/3

    def test_returns_none_if_too_short(self):
        assert rolling_statistics([1, 2], window=3) is None

    def test_returns_none_if_window_too_small(self):
        assert rolling_statistics([1, 2, 3], window=1) is None

    def test_rolling_min_max(self):
        values = [5, 1, 3, 7, 2]
        result = rolling_statistics(values, window=3)
        assert result.rolling_min[2] == 1.0  # min(5,1,3)
        assert result.rolling_max[2] == 5.0  # max(5,1,3)

    def test_z_scores(self):
        values = [10, 10, 10, 10, 100]  # last value is outlier
        result = rolling_statistics(values, window=3)
        # Z-score of last value should be high
        assert result.z_scores[-1] is not None
        assert result.z_scores[-1] > 1.0

    def test_constant_values_z_score_zero(self):
        values = [5, 5, 5, 5, 5]
        result = rolling_statistics(values, window=3)
        for i in range(2, 5):
            assert result.z_scores[i] == 0.0

    def test_summary(self):
        result = rolling_statistics([1, 2, 3, 4, 5], window=3)
        assert "window=3" in result.summary

    def test_window_equals_length(self):
        values = [1, 2, 3]
        result = rolling_statistics(values, window=3)
        assert result is not None
        assert result.rolling_mean[2] is not None


class TestDetectRegimeChanges:
    def test_detects_spike(self):
        values = [10] * 20 + [100]
        changes = detect_regime_changes(values, window=5, threshold=2.0)
        assert len(changes) >= 1
        assert changes[-1]["direction"] == "up"

    def test_no_changes_constant(self):
        values = [5] * 20
        changes = detect_regime_changes(values, window=5)
        assert len(changes) == 0

    def test_detects_dip(self):
        values = [100] * 20 + [10]
        changes = detect_regime_changes(values, window=5, threshold=2.0)
        assert any(c["direction"] == "down" for c in changes)

    def test_empty_if_too_short(self):
        changes = detect_regime_changes([1, 2], window=5)
        assert len(changes) == 0


class TestRollingCorrelation:
    def test_perfect_positive(self):
        a = list(range(20))
        b = list(range(20))
        corrs = rolling_correlation(a, b, window=5)
        # Non-None values should be close to 1.0
        non_none = [c for c in corrs if c is not None]
        assert all(c > 0.95 for c in non_none)

    def test_perfect_negative(self):
        a = list(range(20))
        b = list(range(19, -1, -1))
        corrs = rolling_correlation(a, b, window=5)
        non_none = [c for c in corrs if c is not None]
        assert all(c < -0.95 for c in non_none)

    def test_returns_nones_at_start(self):
        a = [1, 2, 3, 4, 5]
        b = [5, 4, 3, 2, 1]
        corrs = rolling_correlation(a, b, window=3)
        assert corrs[0] is None
        assert corrs[1] is None
        assert corrs[2] is not None

    def test_too_short(self):
        corrs = rolling_correlation([1, 2], [3, 4], window=5)
        assert all(c is None for c in corrs)

    def test_same_length_output(self):
        a = list(range(10))
        b = list(range(10))
        corrs = rolling_correlation(a, b, window=5)
        assert len(corrs) == 10


class TestComputeVolatility:
    def test_basic(self):
        values = [100, 102, 98, 103, 97, 105, 95, 110, 90, 100, 105, 95]
        vol = compute_volatility(values, window=5)
        assert len(vol) == len(values)
        # Some non-None values
        non_none = [v for v in vol if v is not None]
        assert len(non_none) > 0
        assert all(v >= 0 for v in non_none)

    def test_constant_zero_volatility(self):
        values = [100] * 15
        vol = compute_volatility(values, window=5)
        non_none = [v for v in vol if v is not None]
        assert all(v == 0.0 for v in non_none)

    def test_too_short(self):
        vol = compute_volatility([1, 2, 3], window=10)
        assert all(v is None for v in vol)


class TestFindOutlierWindows:
    def test_finds_outlier_period(self):
        values = [10] * 20 + [100] * 5 + [10] * 20
        outliers = find_outlier_windows(values, window=5, threshold=2.0)
        assert len(outliers) >= 1
        assert any(o["direction"] == "above" for o in outliers)

    def test_no_outliers_constant(self):
        values = [5] * 30
        outliers = find_outlier_windows(values, window=5)
        assert len(outliers) == 0

    def test_empty_if_too_short(self):
        outliers = find_outlier_windows([1, 2], window=5)
        assert len(outliers) == 0

    def test_has_indices(self):
        values = [10] * 20 + [100] * 10
        outliers = find_outlier_windows(values, window=5)
        if outliers:
            assert "start_index" in outliers[0]
            assert "end_index" in outliers[0]
