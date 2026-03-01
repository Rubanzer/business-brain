"""Tests for analysis/tools/compute.py — 11 stateless pure functions."""

import math

import numpy as np
import pytest

from business_brain.analysis.tools.compute import (
    bootstrap_stability,
    compare_groups,
    compute_correlation,
    compute_lag_correlation,
    compute_partial_correlation,
    decompose_series,
    describe_categorical,
    describe_numeric,
    detect_distribution,
    find_anomalies_zscore,
    forecast_series,
)


# ---------------------------------------------------------------------------
# describe_numeric
# ---------------------------------------------------------------------------


class TestDescribeNumeric:
    def test_basic_stats(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = describe_numeric(values)
        assert result["count"] == 10
        assert result["mean"] == pytest.approx(5.5)
        assert result["median"] == pytest.approx(5.5)
        assert result["min"] == 1.0
        assert result["max"] == 10.0
        assert result["q1"] == pytest.approx(3.25)
        assert result["q3"] == pytest.approx(7.75)

    def test_with_nans(self):
        values = [1.0, float("nan"), 3.0, float("nan"), 5.0]
        result = describe_numeric(values)
        assert result["count"] == 3
        assert result["mean"] == pytest.approx(3.0)

    def test_empty_list(self):
        result = describe_numeric([])
        assert result["count"] == 0
        assert "error" in result

    def test_skewness_kurtosis(self):
        values = list(range(1, 100))
        result = describe_numeric(values)
        assert "skewness" in result
        assert "kurtosis" in result

    def test_single_value(self):
        result = describe_numeric([42.0])
        assert result["count"] == 1
        assert result["mean"] == 42.0
        assert result["stdev"] == 0.0


# ---------------------------------------------------------------------------
# describe_categorical
# ---------------------------------------------------------------------------


class TestDescribeCategorical:
    def test_basic(self):
        values = ["A", "B", "A", "C", "A", "B"]
        result = describe_categorical(values)
        assert result["count"] == 6
        assert result["unique"] == 3
        assert result["top_value"] == "A"
        assert result["top_count"] == 3
        assert result["concentration"] == pytest.approx(0.5)

    def test_entropy(self):
        # Uniform distribution has max entropy
        values = ["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25
        result = describe_categorical(values)
        assert result["entropy"] == pytest.approx(2.0)  # log2(4) = 2

    def test_single_category(self):
        values = ["X"] * 10
        result = describe_categorical(values)
        assert result["concentration"] == 1.0
        assert result["entropy"] == 0.0

    def test_empty(self):
        result = describe_categorical([])
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# detect_distribution
# ---------------------------------------------------------------------------


class TestDetectDistribution:
    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        values = rng.normal(50, 10, 500).tolist()
        result = detect_distribution(values)
        assert result["type"] in ("normal", "lognormal", "uniform")
        assert "fit_score" in result

    def test_uniform_distribution(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(0, 100, 500).tolist()
        result = detect_distribution(values)
        assert "type" in result

    def test_too_few_values(self):
        result = detect_distribution([1, 2, 3])
        assert result["type"] == "unknown"

    def test_lognormal_positive_values(self):
        rng = np.random.default_rng(42)
        values = rng.lognormal(3, 1, 500).tolist()
        result = detect_distribution(values)
        assert "type" in result
        assert len(result.get("all_fits", [])) >= 3  # normal, lognormal, uniform


# ---------------------------------------------------------------------------
# compare_groups
# ---------------------------------------------------------------------------


class TestCompareGroups:
    def test_two_groups_significant(self):
        groups = {
            "control": [10, 12, 11, 13, 10, 12],
            "treatment": [20, 22, 21, 23, 20, 22],
        }
        result = compare_groups(groups)
        assert result["test"] == "welch_t"
        assert result["significant"] is True
        assert abs(result["cohens_d"]) > 1.0  # large effect

    def test_two_groups_not_significant(self):
        groups = {
            "A": [10.0, 10.1, 9.9, 10.2, 9.8],
            "B": [10.1, 9.9, 10.0, 10.1, 10.0],
        }
        result = compare_groups(groups)
        assert result["test"] == "welch_t"
        assert result["significant"] is False

    def test_three_groups_anova(self):
        groups = {
            "low": [1, 2, 3, 2, 1],
            "mid": [5, 6, 5, 6, 5],
            "high": [10, 11, 10, 11, 10],
        }
        result = compare_groups(groups)
        assert result["test"] == "anova_f"
        assert result["significant"] is True
        assert result["eta_squared"] > 0.5

    def test_insufficient_data(self):
        groups = {"A": [1], "B": [2]}
        result = compare_groups(groups)
        assert "error" in result


# ---------------------------------------------------------------------------
# compute_correlation
# ---------------------------------------------------------------------------


class TestComputeCorrelation:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = compute_correlation(x, y)
        assert result["pearson_r"] == pytest.approx(1.0)
        assert result["spearman_rho"] == pytest.approx(1.0)
        assert result["significant"] is True

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200).tolist()
        y = rng.normal(0, 1, 200).tolist()
        result = compute_correlation(x, y)
        assert abs(result["pearson_r"]) < 0.2

    def test_negative_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        result = compute_correlation(x, y)
        assert result["pearson_r"] == pytest.approx(-1.0)

    def test_insufficient_data(self):
        result = compute_correlation([1, 2], [3, 4])
        assert "error" in result

    def test_with_nans(self):
        x = [1.0, float("nan"), 3.0, 4.0, 5.0]
        y = [2.0, 4.0, float("nan"), 8.0, 10.0]
        result = compute_correlation(x, y)
        assert result["n"] == 3  # only 3 non-nan pairs


# ---------------------------------------------------------------------------
# compute_lag_correlation
# ---------------------------------------------------------------------------


class TestComputeLagCorrelation:
    def test_lag_zero(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        result = compute_lag_correlation(x, y, max_lag=3)
        assert "best_lag" in result
        assert abs(result["best_r"]) == pytest.approx(1.0, abs=0.01)

    def test_lagged_signal(self):
        x = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0]
        result = compute_lag_correlation(x, y, max_lag=5)
        assert "best_lag" in result
        assert "all_lags" in result

    def test_insufficient_data(self):
        result = compute_lag_correlation([1], [2], max_lag=1)
        assert "error" in result


# ---------------------------------------------------------------------------
# compute_partial_correlation
# ---------------------------------------------------------------------------


class TestComputePartialCorrelation:
    def test_no_controls(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = compute_partial_correlation(x, y, controls=[])
        assert result["partial_r"] == pytest.approx(1.0, abs=0.01)
        assert result["controls"] == 0

    def test_with_confounder(self):
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, 100)
        x = z + rng.normal(0, 0.1, 100)
        y = z + rng.normal(0, 0.1, 100)
        # x and y are correlated because of z (confounder)
        result = compute_correlation(x.tolist(), y.tolist())
        assert abs(result["pearson_r"]) > 0.5

        # After controlling for z, partial correlation should be low
        partial = compute_partial_correlation(x.tolist(), y.tolist(), [z.tolist()])
        assert abs(partial["partial_r"]) < abs(result["pearson_r"])
        assert partial["controls"] == 1

    def test_insufficient_data(self):
        result = compute_partial_correlation([1, 2], [3, 4], [[5, 6]])
        assert "error" in result


# ---------------------------------------------------------------------------
# find_anomalies_zscore
# ---------------------------------------------------------------------------


class TestFindAnomaliesZscore:
    def test_detects_outliers(self):
        values = [10] * 50 + [100]  # 100 is an outlier
        result = find_anomalies_zscore(values, threshold=2.0)
        assert result["count"] >= 1
        assert any(a["value"] == 100 for a in result["anomalies"])

    def test_no_anomalies(self):
        values = [10.0, 10.1, 9.9, 10.2, 9.8] * 10
        result = find_anomalies_zscore(values, threshold=3.0)
        assert result["count"] == 0

    def test_too_few_values(self):
        result = find_anomalies_zscore([1, 2, 3])
        assert result["anomalies"] == []
        assert "error" in result

    def test_constant_values(self):
        values = [5.0] * 20
        result = find_anomalies_zscore(values)
        assert result["std"] == 0.0


# ---------------------------------------------------------------------------
# decompose_series
# ---------------------------------------------------------------------------


class TestDecomposeSeries:
    def test_basic_decomposition(self):
        # Trend + seasonal signal
        trend = np.linspace(10, 20, 30)
        seasonal = np.tile([1, -1, 0, 1, -1, 0, 1], 5)[:30]
        values = (trend + seasonal).tolist()
        result = decompose_series(values, period=7)
        assert len(result["trend"]) == 30
        assert len(result["seasonal"]) == 30
        assert len(result["residual"]) == 30
        assert result["trend_direction"] == "increasing"

    def test_too_short(self):
        result = decompose_series([1, 2, 3], period=7)
        assert "error" in result


# ---------------------------------------------------------------------------
# forecast_series
# ---------------------------------------------------------------------------


class TestForecastSeries:
    def test_flat_forecast(self):
        values = [10.0] * 20
        result = forecast_series(values, periods=5)
        assert len(result["forecast"]) == 5
        assert result["forecast"][0] == pytest.approx(10.0, abs=0.5)

    def test_trending_forecast(self):
        values = list(range(1, 21))
        result = forecast_series(values, periods=3)
        assert len(result["forecast"]) == 3
        assert "upper_bound" in result
        assert "lower_bound" in result
        assert result["forecast_base"] > 10  # should track upward trend

    def test_insufficient_data(self):
        result = forecast_series([1, 2])
        assert "error" in result


# ---------------------------------------------------------------------------
# bootstrap_stability
# ---------------------------------------------------------------------------


class TestBootstrapStability:
    def test_stable_data(self):
        values = [100.0] * 50
        result = bootstrap_stability(values)
        assert result["fraction_stable"] == pytest.approx(1.0)
        assert result["ci_width"] == pytest.approx(0.0, abs=0.01)

    def test_noisy_data(self):
        rng = np.random.default_rng(42)
        values = rng.normal(50, 20, 100).tolist()
        result = bootstrap_stability(values)
        assert 0.0 < result["fraction_stable"] < 1.0
        assert result["ci_width"] > 0

    def test_custom_stat_fn(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = bootstrap_stability(values, stat_fn=np.median)
        assert result["point_estimate"] == pytest.approx(5.5)

    def test_insufficient_data(self):
        result = bootstrap_stability([1, 2])
        assert "error" in result
