"""Tests for distribution profiler pure functions."""

from business_brain.discovery.distribution_profiler import (
    build_histogram,
    classify_shape,
    compute_kurtosis,
    compute_quartiles,
    compute_skewness,
    profile_distribution,
)


class TestComputeQuartiles:
    def test_simple_odd(self):
        q1, median, q3 = compute_quartiles([1, 2, 3, 4, 5])
        assert median == 3
        assert q1 == 1.5
        assert q3 == 4.5

    def test_simple_even(self):
        q1, median, q3 = compute_quartiles([1, 2, 3, 4])
        assert median == 2.5

    def test_single_value(self):
        q1, median, q3 = compute_quartiles([42])
        assert q1 == 42
        assert median == 42
        assert q3 == 42

    def test_empty(self):
        q1, median, q3 = compute_quartiles([])
        assert q1 == 0
        assert median == 0
        assert q3 == 0

    def test_two_values(self):
        q1, median, q3 = compute_quartiles([10, 20])
        assert median == 15

    def test_q3_greater_than_q1(self):
        q1, _, q3 = compute_quartiles([1, 3, 5, 7, 9, 11, 13])
        assert q3 > q1


class TestComputeSkewness:
    def test_symmetric_near_zero(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = compute_skewness(values)
        assert abs(s) < 0.1

    def test_right_skewed(self):
        values = [1, 1, 1, 2, 2, 3, 10, 20, 50]
        s = compute_skewness(values)
        assert s > 0.5

    def test_left_skewed(self):
        values = [50, 20, 10, 3, 2, 2, 1, 1, 1]
        # Same data but described differently — still right-skewed
        # Let's make a truly left-skewed dataset
        values = [1, 50, 50, 50, 48, 49, 51, 47, 52]
        s = compute_skewness(values)
        assert s < -0.5

    def test_too_few_values(self):
        assert compute_skewness([1, 2]) == 0.0

    def test_constant_values(self):
        assert compute_skewness([5, 5, 5, 5]) == 0.0


class TestComputeKurtosis:
    def test_uniform_negative(self):
        """Uniform distribution has negative excess kurtosis."""
        values = list(range(1, 101))
        k = compute_kurtosis(values)
        assert k < 0

    def test_too_few_values(self):
        assert compute_kurtosis([1, 2, 3]) == 0.0

    def test_constant_values(self):
        assert compute_kurtosis([5, 5, 5, 5, 5]) == 0.0

    def test_heavy_tailed(self):
        """A distribution with outliers should have positive kurtosis."""
        values = [0] * 50 + [100] * 2 + [-100] * 2
        k = compute_kurtosis(values)
        assert k > 0


class TestBuildHistogram:
    def test_basic(self):
        values = list(range(10))
        hist = build_histogram(values, bins=5)
        assert len(hist) == 5
        total = sum(h["count"] for h in hist)
        assert total == 10

    def test_single_value(self):
        hist = build_histogram([5, 5, 5], bins=3)
        assert len(hist) == 1
        assert hist[0]["count"] == 3

    def test_empty(self):
        assert build_histogram([], 5) == []

    def test_bins_zero(self):
        assert build_histogram([1, 2, 3], 0) == []

    def test_all_values_counted(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        hist = build_histogram(values, bins=2)
        total = sum(h["count"] for h in hist)
        assert total == 10

    def test_bin_boundaries(self):
        values = [0, 5, 10]
        hist = build_histogram(values, bins=2)
        assert hist[0]["bin_start"] == 0
        assert hist[-1]["bin_end"] == 10


class TestClassifyShape:
    def test_normal(self):
        assert classify_shape(0.1, 0.0, []) == "normal"

    def test_right_skewed(self):
        assert classify_shape(1.5, 0.0, []) == "right_skewed"

    def test_left_skewed(self):
        assert classify_shape(-1.0, 0.0, []) == "left_skewed"

    def test_uniform(self):
        assert classify_shape(0.0, -1.5, []) == "uniform"

    def test_bimodal(self):
        # Create histogram with two peaks
        hist = [
            {"bin_start": 0, "bin_end": 2, "count": 5},
            {"bin_start": 2, "bin_end": 4, "count": 10},
            {"bin_start": 4, "bin_end": 6, "count": 3},
            {"bin_start": 6, "bin_end": 8, "count": 12},
            {"bin_start": 8, "bin_end": 10, "count": 4},
        ]
        assert classify_shape(0.0, 0.0, hist) == "bimodal"

    def test_bimodal_takes_precedence(self):
        """Bimodal check happens before skewness check."""
        hist = [
            {"bin_start": 0, "bin_end": 2, "count": 5},
            {"bin_start": 2, "bin_end": 4, "count": 10},
            {"bin_start": 4, "bin_end": 6, "count": 3},
            {"bin_start": 6, "bin_end": 8, "count": 12},
            {"bin_start": 8, "bin_end": 10, "count": 4},
        ]
        assert classify_shape(1.0, 2.0, hist) == "bimodal"


class TestProfileDistribution:
    def test_basic(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        profile = profile_distribution(values)
        assert profile is not None
        assert profile.count == 10
        assert profile.mean == 5.5
        assert profile.median == 5.5
        assert profile.min_val == 1
        assert profile.max_val == 10
        assert profile.q3 > profile.q1
        assert profile.iqr == profile.q3 - profile.q1
        assert len(profile.histogram) > 0
        assert profile.shape in ("normal", "right_skewed", "left_skewed", "bimodal", "uniform")

    def test_too_few_values(self):
        assert profile_distribution([1, 2]) is None

    def test_constant_values(self):
        profile = profile_distribution([5, 5, 5, 5, 5])
        assert profile is not None
        assert profile.stdev == 0
        assert profile.skewness == 0
        assert profile.shape == "normal"

    def test_custom_bins(self):
        values = list(range(100))
        profile = profile_distribution(values, bins=20)
        assert profile is not None
        assert len(profile.histogram) == 20

    def test_negative_values(self):
        values = [-10, -5, 0, 5, 10]
        profile = profile_distribution(values)
        assert profile is not None
        assert profile.min_val == -10
        assert profile.max_val == 10

    def test_right_skewed_data(self):
        values = [1] * 30 + [2] * 15 + [5] * 5 + [10] * 3 + [50] * 1
        profile = profile_distribution(values)
        assert profile is not None
        assert profile.skewness > 0
