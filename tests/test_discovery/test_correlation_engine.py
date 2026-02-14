"""Tests for correlation engine pure functions."""

import math

from business_brain.discovery.correlation_engine import (
    classify_correlation,
    compute_correlation_matrix,
    compute_pearson,
    correlation_summary,
    find_strong_correlations,
    find_surprising_correlations,
)


class TestComputePearson:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = compute_pearson(x, y)
        assert r is not None
        assert abs(r - 1.0) < 0.001

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        r = compute_pearson(x, y)
        assert r is not None
        assert abs(r - (-1.0)) < 0.001

    def test_no_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 1, 5, 3]
        r = compute_pearson(x, y)
        assert r is not None
        assert abs(r) < 0.5

    def test_constant_x_returns_none(self):
        x = [5, 5, 5, 5, 5]
        y = [1, 2, 3, 4, 5]
        assert compute_pearson(x, y) is None

    def test_constant_y_returns_none(self):
        x = [1, 2, 3, 4, 5]
        y = [7, 7, 7, 7, 7]
        assert compute_pearson(x, y) is None

    def test_too_few_values(self):
        assert compute_pearson([1, 2], [3, 4]) is None

    def test_mismatched_lengths(self):
        assert compute_pearson([1, 2, 3], [4, 5]) is None

    def test_empty_lists(self):
        assert compute_pearson([], []) is None

    def test_three_values_minimum(self):
        r = compute_pearson([1, 2, 3], [2, 4, 6])
        assert r is not None
        assert abs(r - 1.0) < 0.001

    def test_negative_values(self):
        x = [-5, -3, -1, 1, 3, 5]
        y = [-10, -6, -2, 2, 6, 10]
        r = compute_pearson(x, y)
        assert r is not None
        assert abs(r - 1.0) < 0.001


class TestClassifyCorrelation:
    def test_strong_positive(self):
        strength, direction = classify_correlation(0.85)
        assert strength == "strong"
        assert direction == "positive"

    def test_strong_negative(self):
        strength, direction = classify_correlation(-0.75)
        assert strength == "strong"
        assert direction == "negative"

    def test_moderate_positive(self):
        strength, direction = classify_correlation(0.55)
        assert strength == "moderate"
        assert direction == "positive"

    def test_weak_positive(self):
        strength, direction = classify_correlation(0.25)
        assert strength == "weak"
        assert direction == "positive"

    def test_none_strength(self):
        strength, direction = classify_correlation(0.1)
        assert strength == "none"
        assert direction == "none"

    def test_zero(self):
        strength, direction = classify_correlation(0.0)
        assert strength == "none"
        assert direction == "none"

    def test_exactly_0_7(self):
        strength, _ = classify_correlation(0.7)
        assert strength == "strong"

    def test_just_below_0_7(self):
        strength, _ = classify_correlation(0.69)
        assert strength == "moderate"

    def test_none_input(self):
        strength, direction = classify_correlation(None)
        assert strength == "none"
        assert direction == "none"

    def test_exactly_negative_1(self):
        strength, direction = classify_correlation(-1.0)
        assert strength == "strong"
        assert direction == "negative"


class TestComputeCorrelationMatrix:
    def test_two_columns(self):
        data = {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]}
        pairs = compute_correlation_matrix(data)
        assert len(pairs) == 1
        assert abs(pairs[0].correlation - 1.0) < 0.001

    def test_three_columns(self):
        data = {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "z": [10, 8, 6, 4, 2],
        }
        pairs = compute_correlation_matrix(data)
        assert len(pairs) == 3  # x-y, x-z, y-z

    def test_one_column(self):
        data = {"a": [1, 2, 3]}
        pairs = compute_correlation_matrix(data)
        assert len(pairs) == 0

    def test_empty_data(self):
        pairs = compute_correlation_matrix({})
        assert len(pairs) == 0

    def test_too_few_values(self):
        data = {"a": [1, 2], "b": [3, 4]}
        pairs = compute_correlation_matrix(data)
        assert len(pairs) == 0

    def test_different_lengths(self):
        data = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30]}
        pairs = compute_correlation_matrix(data)
        # Should use min length (3)
        assert len(pairs) == 1
        assert pairs[0].sample_size == 3

    def test_sample_size_correct(self):
        data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
        pairs = compute_correlation_matrix(data)
        assert pairs[0].sample_size == 4

    def test_strength_classification(self):
        data = {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]}
        pairs = compute_correlation_matrix(data)
        assert pairs[0].strength == "strong"
        assert pairs[0].direction == "positive"


class TestFindStrongCorrelations:
    def test_filters_strong(self):
        data = {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 3, 7, 1, 9],  # weakly correlated
        }
        pairs = compute_correlation_matrix(data)
        strong = find_strong_correlations(pairs, threshold=0.7)
        assert len(strong) >= 1
        for p in strong:
            assert abs(p.correlation) >= 0.7

    def test_custom_threshold(self):
        data = {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]}
        pairs = compute_correlation_matrix(data)
        strong = find_strong_correlations(pairs, threshold=0.99)
        assert len(strong) == 1

    def test_empty_input(self):
        assert find_strong_correlations([], 0.7) == []


class TestFindSurprisingCorrelations:
    def test_negative_correlation(self):
        data = {
            "price": [100, 200, 300, 400, 500],
            "demand": [50, 40, 30, 20, 10],
        }
        pairs = compute_correlation_matrix(data)
        surprising = find_surprising_correlations(pairs, threshold=0.5)
        assert len(surprising) == 1
        assert surprising[0].correlation < -0.5

    def test_no_negative(self):
        data = {"a": [1, 2, 3], "b": [2, 4, 6]}
        pairs = compute_correlation_matrix(data)
        assert find_surprising_correlations(pairs, 0.5) == []


class TestCorrelationSummary:
    def test_summary(self):
        data = {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [10, 8, 6, 4, 2],
            "d": [5, 3, 7, 1, 9],
        }
        pairs = compute_correlation_matrix(data)
        summary = correlation_summary(pairs)
        assert summary["total_pairs"] == 6  # C(4,2)
        assert summary["strong"] >= 2  # a-b and a-c at minimum
        assert summary["top_positive"] is not None
        assert summary["top_negative"] is not None

    def test_empty_summary(self):
        summary = correlation_summary([])
        assert summary["total_pairs"] == 0
        assert summary["top_positive"] is None
        assert summary["top_negative"] is None

    def test_all_weak(self):
        data = {"a": [1, 5, 2, 4, 3], "b": [3, 1, 4, 2, 5]}
        pairs = compute_correlation_matrix(data)
        summary = correlation_summary(pairs)
        assert summary["strong"] == 0
