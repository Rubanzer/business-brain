"""Tests for statistical summary generator."""

from business_brain.discovery.stat_summary import (
    StatSummary,
    compare_distributions,
    compute_stat_summary,
    format_stat_table,
)


class TestComputeStatSummary:
    def test_basic_summary(self):
        values = [10, 20, 30, 40, 50]
        result = compute_stat_summary(values, "test")
        assert result is not None
        assert result.column == "test"
        assert result.count == 5
        assert result.mean == 30.0
        assert result.median == 30.0
        assert result.min_val == 10.0
        assert result.max_val == 50.0
        assert result.range_val == 40.0

    def test_too_few_values(self):
        assert compute_stat_summary([1, 2]) is None
        assert compute_stat_summary([]) is None

    def test_std_and_variance(self):
        values = [10, 20, 30, 40, 50]
        result = compute_stat_summary(values)
        assert result.std > 0
        assert result.variance > 0
        assert abs(result.variance - result.std ** 2) < 0.01

    def test_quartiles(self):
        values = list(range(1, 101))  # 1 to 100
        result = compute_stat_summary(values)
        assert 24 <= result.q1 <= 26
        assert 74 <= result.q3 <= 76
        assert result.iqr > 0

    def test_percentiles(self):
        values = list(range(1, 101))
        result = compute_stat_summary(values)
        assert result.percentiles["p50"] == result.median
        assert result.percentiles["p5"] < result.percentiles["p95"]

    def test_confidence_interval(self):
        values = [100] * 50  # constant values
        result = compute_stat_summary(values)
        assert result.ci_95_lower <= result.mean <= result.ci_95_upper

    def test_normality_normal(self):
        # Roughly normal distribution
        import random
        random.seed(42)
        values = [random.gauss(100, 10) for _ in range(200)]
        result = compute_stat_summary(values)
        assert result.normality in ("normal", "approximately_normal")

    def test_normality_skewed(self):
        # Right-skewed distribution
        values = [1, 1, 1, 1, 2, 2, 3, 5, 10, 50, 100]
        result = compute_stat_summary(values)
        assert result.skewness > 0

    def test_outlier_count(self):
        values = [10, 11, 12, 13, 14, 15, 100]
        result = compute_stat_summary(values)
        assert result.outlier_count >= 1

    def test_no_outliers(self):
        values = [10, 11, 12, 13, 14]
        result = compute_stat_summary(values)
        assert result.outlier_count == 0

    def test_cv(self):
        values = [100, 100, 100, 100, 100]
        result = compute_stat_summary(values)
        assert result.cv == 0.0

    def test_mode(self):
        values = [1, 2, 2, 3, 4]
        result = compute_stat_summary(values)
        assert result.mode == 2.0

    def test_no_mode(self):
        values = [1, 2, 3, 4, 5]
        result = compute_stat_summary(values)
        assert result.mode is None

    def test_interpretation(self):
        values = [10, 20, 30, 40, 50]
        result = compute_stat_summary(values, "revenue")
        assert "revenue" in result.interpretation

    def test_constant_values(self):
        values = [42, 42, 42, 42, 42]
        result = compute_stat_summary(values)
        assert result.std == 0.0
        assert result.iqr == 0.0

    def test_negative_values(self):
        values = [-10, -5, 0, 5, 10]
        result = compute_stat_summary(values)
        assert result.mean == 0.0
        assert result.min_val == -10.0


class TestCompareDistributions:
    def test_basic_comparison(self):
        a = compute_stat_summary([10, 20, 30, 40, 50], "A")
        b = compute_stat_summary([50, 60, 70, 80, 90], "B")
        comp = compare_distributions(a, b)
        assert comp["mean_diff"] > 0
        assert comp["significance"] in ("large", "medium", "small", "negligible")
        assert comp["column_a"] == "A"
        assert comp["column_b"] == "B"

    def test_same_distributions(self):
        a = compute_stat_summary([10, 20, 30, 40, 50], "A")
        b = compute_stat_summary([10, 20, 30, 40, 50], "B")
        comp = compare_distributions(a, b)
        assert comp["mean_diff"] == 0
        assert comp["significance"] == "negligible"

    def test_large_difference(self):
        a = compute_stat_summary([1, 2, 3, 4, 5], "small")
        b = compute_stat_summary([100, 200, 300, 400, 500], "big")
        comp = compare_distributions(a, b)
        assert comp["significance"] == "large"


class TestFormatStatTable:
    def test_basic_format(self):
        summary = compute_stat_summary([10, 20, 30, 40, 50], "revenue")
        text = format_stat_table(summary)
        assert "Statistical Summary: revenue" in text
        assert "Count:" in text
        assert "Mean:" in text
        assert "Std Dev:" in text
        assert "Normality:" in text
