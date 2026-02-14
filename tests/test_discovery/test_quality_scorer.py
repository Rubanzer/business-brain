"""Tests for the quality_scorer module.

Covers all dimension scorers, the full quality report, grade assignment,
critical issue detection, custom rules/weights, quality comparison, and
various edge cases.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from business_brain.discovery.quality_scorer import (
    DimensionScore,
    QualityReport,
    compare_quality,
    compute_quality_report,
    score_accuracy,
    score_completeness,
    score_consistency,
    score_freshness,
    score_uniqueness,
)


# -------------------------------------------------------------------------
# Helpers / fixtures
# -------------------------------------------------------------------------

def _perfect_rows(n: int = 20) -> list[dict]:
    """Generate *n* rows of perfectly clean data."""
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        {
            "id": i,
            "name": f"item_{i}",
            "price": round(10 + i * 0.5, 2),
            "status": "active",
            "created_date": today,
        }
        for i in range(1, n + 1)
    ]


def _poor_rows() -> list[dict]:
    """Generate rows with many quality problems."""
    return [
        {"id": 1, "name": "Alice", "price": 100, "status": "Active", "created_date": "2020-01-01"},
        {"id": 1, "name": "Alice", "price": 100, "status": "active", "created_date": "2020-01-01"},
        {"id": None, "name": None, "price": -50, "status": "ACTIVE", "created_date": None},
        {"id": 4, "name": "Bob", "price": 200, "status": 123, "created_date": "bad-date"},
        {"id": 5, "name": "Charlie", "price": 99999, "status": "active", "created_date": "2020-06-15"},
    ]


# =========================================================================
# 1. Completeness
# =========================================================================

class TestScoreCompleteness:

    def test_perfect_completeness(self):
        rows = _perfect_rows()
        cols = list(rows[0].keys())
        result = score_completeness(rows, cols)
        assert result.dimension == "completeness"
        assert result.score == 100.0
        assert result.issues == []

    def test_all_nulls(self):
        rows = [{"a": None, "b": None} for _ in range(5)]
        result = score_completeness(rows, ["a", "b"])
        assert result.score == 0.0
        assert len(result.issues) > 0

    def test_partial_nulls(self):
        rows = [{"a": 1, "b": None}, {"a": 2, "b": "x"}, {"a": None, "b": "y"}]
        result = score_completeness(rows, ["a", "b"])
        # 2 nulls out of 6 values => 33.3% null rate => score ~66.67
        assert 60 < result.score < 70

    def test_null_like_strings(self):
        rows = [{"a": "null"}, {"a": "N/A"}, {"a": "NA"}, {"a": ""}, {"a": "real"}]
        result = score_completeness(rows, ["a"])
        # 4 out of 5 are null-like => score = 20
        assert result.score == 20.0

    def test_empty_rows(self):
        result = score_completeness([], ["a"])
        assert result.score == 0.0
        assert "No data" in result.issues[0]

    def test_no_columns(self):
        result = score_completeness([{"a": 1}], [])
        assert result.score == 100.0

    def test_nan_treated_as_null(self):
        rows = [{"a": float("nan")}, {"a": 1.0}]
        result = score_completeness(rows, ["a"])
        assert result.score == 50.0


# =========================================================================
# 2. Uniqueness
# =========================================================================

class TestScoreUniqueness:

    def test_all_unique(self):
        rows = _perfect_rows()
        result = score_uniqueness(rows, ["id"])
        assert result.score == 100.0
        assert result.issues == []

    def test_exact_duplicates(self):
        rows = [{"id": 1}, {"id": 1}, {"id": 2}, {"id": 3}]
        result = score_uniqueness(rows, ["id"])
        # 3 unique out of 4 => 75
        assert result.score == 75.0
        assert any("exact duplicate" in i.lower() for i in result.issues)

    def test_near_duplicates_casing(self):
        rows = [{"name": "Alice"}, {"name": "alice"}, {"name": "Bob"}]
        result = score_uniqueness(rows, ["name"])
        # All 3 are exact-unique => score 100
        # But near-duplicate detection should flag case difference
        assert result.details["near_duplicate_rows"] > 0

    def test_empty_rows(self):
        result = score_uniqueness([], ["id"])
        assert result.score == 0.0
        assert "No data" in result.issues[0]

    def test_single_row(self):
        result = score_uniqueness([{"id": 1}], ["id"])
        assert result.score == 100.0

    def test_no_key_columns_uses_all(self):
        rows = [{"a": 1, "b": 2}, {"a": 1, "b": 2}]
        result = score_uniqueness(rows, [])
        # Falls back to all columns; exact dupe
        assert result.score == 50.0


# =========================================================================
# 3. Consistency
# =========================================================================

class TestScoreConsistency:

    def test_perfectly_consistent(self):
        rows = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        result = score_consistency(rows, ["a", "b"])
        assert result.score == 100.0

    def test_mixed_types(self):
        rows = [{"a": 1}, {"a": "hello"}, {"a": 3}]
        result = score_consistency(rows, ["a"])
        assert result.score < 100.0
        assert any("mixed types" in i.lower() for i in result.issues)

    def test_inconsistent_casing(self):
        rows = [{"s": "Active"}, {"s": "active"}, {"s": "ACTIVE"}, {"s": "active"}]
        result = score_consistency(rows, ["s"])
        assert result.score < 100.0
        assert any("casing" in i.lower() for i in result.issues)

    def test_empty_rows(self):
        result = score_consistency([], ["a"])
        assert result.score == 0.0

    def test_all_null_column(self):
        rows = [{"a": None}, {"a": None}]
        result = score_consistency(rows, ["a"])
        # No non-null values to check → 100 by default
        assert result.score == 100.0


# =========================================================================
# 4. Accuracy
# =========================================================================

class TestScoreAccuracy:

    def test_all_valid(self):
        rows = [{"price": 10}, {"price": 20}, {"price": 30}, {"price": 15}]
        result = score_accuracy(rows, ["price"])
        assert result.score == 100.0

    def test_negative_in_positive_column(self):
        rows = [{"price": 10}, {"price": -5}, {"price": 20}, {"price": 15}]
        result = score_accuracy(rows, ["price"])
        assert result.score < 100.0
        assert any("negative" in i.lower() for i in result.issues)

    def test_outliers_iqr(self):
        rows = [{"val": i} for i in range(20)] + [{"val": 10000}]
        result = score_accuracy(rows, ["val"])
        assert result.details["outlier_counts"].get("val", 0) > 0

    def test_custom_rule_min_max(self):
        rows = [{"age": 25}, {"age": 150}, {"age": -3}]
        rules = [{"column": "age", "min": 0, "max": 120}]
        result = score_accuracy(rows, ["age"], rules=rules)
        assert result.details["rule_violations"].get("age", 0) >= 1
        assert any("rule violation" in i.lower() for i in result.issues)

    def test_custom_rule_pattern(self):
        rows = [{"code": "ABC-123"}, {"code": "XYZ-456"}, {"code": "invalid"}]
        rules = [{"column": "code", "pattern": r"^[A-Z]{3}-\d{3}$"}]
        result = score_accuracy(rows, ["code"], rules=rules)
        assert result.details["rule_violations"].get("code", 0) == 1

    def test_empty_rows(self):
        result = score_accuracy([], ["price"])
        assert result.score == 0.0

    def test_no_numeric_columns(self):
        rows = [{"name": "Alice"}, {"name": "Bob"}]
        result = score_accuracy(rows, ["name"])
        # No numeric values → nothing to check → 100
        assert result.score == 100.0


# =========================================================================
# 5. Freshness
# =========================================================================

class TestScoreFreshness:

    def test_fresh_data(self):
        today = datetime.now().strftime("%Y-%m-%d")
        rows = [{"created_date": today}]
        result = score_freshness(rows, ["created_date"])
        assert result.score == 100.0

    def test_old_data(self):
        old = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
        rows = [{"created_date": old}]
        result = score_freshness(rows, ["created_date"])
        # 15 days old: 100 - 10*(15-1) = 100 - 140 → clamped to 0
        assert result.score == 0.0

    def test_moderately_old(self):
        dt = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        rows = [{"created_date": dt}]
        result = score_freshness(rows, ["created_date"])
        # 5 days old: 100 - 10*(5-1) = 60
        assert result.score == 60.0

    def test_no_date_columns(self):
        rows = [{"a": 1}]
        result = score_freshness(rows, [])
        assert result.score == 80.0  # neutral

    def test_unparseable_dates(self):
        rows = [{"created_date": "not-a-date"}]
        result = score_freshness(rows, ["created_date"])
        assert result.score == 80.0
        assert any("parse" in i.lower() for i in result.issues)

    def test_empty_rows(self):
        result = score_freshness([], ["created_date"])
        assert result.score == 0.0

    def test_reference_date(self):
        rows = [{"created_date": "2024-01-10"}]
        result = score_freshness(rows, ["created_date"], reference_date="2024-01-12")
        # 2 days old: 100 - 10*(2-1) = 90
        assert result.score == 90.0

    def test_dd_mm_yyyy_format(self):
        today = datetime.now()
        date_str = today.strftime("%d/%m/%Y")
        rows = [{"dt": date_str}]
        result = score_freshness(rows, ["dt"])
        assert result.score >= 90.0


# =========================================================================
# 6. Full quality report
# =========================================================================

class TestComputeQualityReport:

    def test_perfect_data_report(self):
        rows = _perfect_rows()
        report = compute_quality_report(rows)
        assert report.grade == "A"
        assert report.overall_score >= 90
        assert len(report.dimensions) == 5
        assert report.critical_issues == []

    def test_poor_data_report(self):
        rows = _poor_rows()
        report = compute_quality_report(
            rows,
            columns=["id", "name", "price", "status", "created_date"],
        )
        assert report.grade in ("C", "D", "F")
        assert len(report.critical_issues) > 0 or report.overall_score < 80

    def test_auto_detect_columns(self):
        rows = [{"id": 1, "name": "x", "created_date": "2024-01-01"}]
        report = compute_quality_report(rows)
        dim_names = {d.dimension for d in report.dimensions}
        assert dim_names == {"completeness", "uniqueness", "consistency", "accuracy", "freshness"}

    def test_auto_detect_key_columns(self):
        rows = [{"user_id": 1, "name": "a"}, {"user_id": 2, "name": "b"}]
        report = compute_quality_report(rows)
        uniqueness_dim = next(d for d in report.dimensions if d.dimension == "uniqueness")
        # user_id should be auto-detected as key column
        assert uniqueness_dim.score == 100.0

    def test_auto_detect_date_columns(self):
        today = datetime.now().strftime("%Y-%m-%d")
        rows = [{"id": 1, "created_date": today, "updated_time": today}]
        report = compute_quality_report(rows)
        freshness_dim = next(d for d in report.dimensions if d.dimension == "freshness")
        assert freshness_dim.score == 100.0


# =========================================================================
# 7. Grade assignment
# =========================================================================

class TestGradeAssignment:

    @pytest.mark.parametrize(
        "score,expected",
        [
            (95, "A"),
            (90, "A"),
            (89, "B"),
            (80, "B"),
            (79, "C"),
            (70, "C"),
            (69, "D"),
            (60, "D"),
            (59, "F"),
            (0, "F"),
        ],
    )
    def test_grade_thresholds(self, score, expected):
        from business_brain.discovery.quality_scorer import _assign_grade
        assert _assign_grade(score) == expected


# =========================================================================
# 8. Critical issues
# =========================================================================

class TestCriticalIssues:

    def test_critical_when_dimension_below_50(self):
        rows = [{"id": None, "name": None} for _ in range(10)]
        report = compute_quality_report(rows, columns=["id", "name"])
        assert len(report.critical_issues) > 0
        assert any("completeness" in ci.lower() for ci in report.critical_issues)

    def test_no_critical_for_good_data(self):
        rows = _perfect_rows()
        report = compute_quality_report(rows)
        assert report.critical_issues == []


# =========================================================================
# 9. Custom weights
# =========================================================================

class TestCustomWeights:

    def test_custom_weights_affect_score(self):
        rows = _poor_rows()
        cols = list(rows[0].keys())

        report_default = compute_quality_report(rows, columns=cols)
        report_custom = compute_quality_report(
            rows,
            columns=cols,
            weights={"completeness": 0.5, "uniqueness": 0.1, "consistency": 0.1, "accuracy": 0.1, "freshness": 0.2},
        )
        # Scores should differ because weights differ
        assert report_default.overall_score != report_custom.overall_score

    def test_weights_normalised(self):
        rows = _perfect_rows()
        report = compute_quality_report(rows, weights={"completeness": 10, "uniqueness": 10})
        # Should still work; weights get normalised
        assert 0 <= report.overall_score <= 100


# =========================================================================
# 10. Recommendations
# =========================================================================

class TestRecommendations:

    def test_recommendations_for_poor_data(self):
        rows = _poor_rows()
        report = compute_quality_report(rows, columns=list(rows[0].keys()))
        assert len(report.recommendations) > 0

    def test_no_recommendations_for_perfect_data(self):
        rows = _perfect_rows()
        report = compute_quality_report(rows)
        # Perfect data: all dimensions >= 90 → no recommendations
        assert report.recommendations == []


# =========================================================================
# 11. Quality comparison
# =========================================================================

class TestCompareQuality:

    def test_improvement(self):
        rows_bad = _poor_rows()
        rows_good = _perfect_rows()
        report_a = compute_quality_report(rows_bad, columns=list(rows_bad[0].keys()))
        report_b = compute_quality_report(rows_good)
        diff = compare_quality(report_a, report_b)
        assert diff["overall_change"] > 0
        assert diff["grade_change"][1] in ("A", "B")  # good data should be A or B

    def test_degradation(self):
        rows_good = _perfect_rows()
        rows_bad = _poor_rows()
        report_a = compute_quality_report(rows_good)
        report_b = compute_quality_report(rows_bad, columns=list(rows_bad[0].keys()))
        diff = compare_quality(report_a, report_b)
        assert diff["overall_change"] < 0

    def test_no_change(self):
        rows = _perfect_rows()
        report = compute_quality_report(rows)
        diff = compare_quality(report, report)
        assert diff["overall_change"] == 0
        for dim_info in diff["dimensions"].values():
            assert dim_info["status"] == "unchanged"

    def test_per_dimension_detail(self):
        report_a = compute_quality_report(_poor_rows(), columns=list(_poor_rows()[0].keys()))
        report_b = compute_quality_report(_perfect_rows())
        diff = compare_quality(report_a, report_b)
        assert "completeness" in diff["dimensions"]
        assert "uniqueness" in diff["dimensions"]
        assert isinstance(diff["dimensions"]["completeness"]["change"], float)


# =========================================================================
# 12. Edge cases
# =========================================================================

class TestEdgeCases:

    def test_single_row(self):
        rows = [{"id": 1, "name": "x"}]
        report = compute_quality_report(rows)
        assert 0 <= report.overall_score <= 100

    def test_missing_column_in_row(self):
        rows = [{"a": 1}, {"b": 2}]
        report = compute_quality_report(rows, columns=["a", "b"])
        # 'a' missing in second row, 'b' missing in first → 50% null rate
        completeness = next(d for d in report.dimensions if d.dimension == "completeness")
        assert completeness.score == 50.0

    def test_empty_dataset(self):
        report = compute_quality_report([], columns=["a"])
        assert report.overall_score == 0.0
        assert report.grade == "F"

    def test_summary_is_string(self):
        report = compute_quality_report(_perfect_rows())
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_dimension_weights_sum_close_to_one(self):
        report = compute_quality_report(_perfect_rows())
        total = sum(d.weight for d in report.dimensions)
        assert abs(total - 1.0) < 0.01
