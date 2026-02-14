"""Tests for outlier explainer pure functions."""

from business_brain.discovery.outlier_explainer import (
    explain_constant_column,
    explain_null_spike,
    explain_outlier,
)


class TestExplainOutlier:
    def test_basic_above_mean(self):
        exp = explain_outlier(
            value=150, column="temp", table="sensors",
            mean=100, stdev=10, min_val=80, max_val=120,
        )
        assert exp.severity == "critical"  # 5 sigma
        assert exp.deviation_sigma == 5.0
        assert "above" in exp.explanation
        assert exp.column == "temp"
        assert exp.table == "sensors"

    def test_basic_below_mean(self):
        exp = explain_outlier(
            value=50, column="temp", table="sensors",
            mean=100, stdev=10, min_val=80, max_val=120,
        )
        assert "below" in exp.explanation

    def test_warning_severity(self):
        exp = explain_outlier(
            value=130, column="val", table="t",
            mean=100, stdev=10, min_val=80, max_val=120,
        )
        assert exp.severity == "warning"  # 3 sigma

    def test_info_severity(self):
        exp = explain_outlier(
            value=115, column="val", table="t",
            mean=100, stdev=10, min_val=80, max_val=120,
        )
        assert exp.severity == "info"  # 1.5 sigma

    def test_zero_stdev(self):
        exp = explain_outlier(
            value=100, column="val", table="t",
            mean=100, stdev=0, min_val=100, max_val=100,
        )
        assert exp.deviation_sigma == 0

    def test_negative_currency(self):
        exp = explain_outlier(
            value=-50, column="amount", table="payments",
            mean=100, stdev=20, min_val=0, max_val=200,
            semantic_type="numeric_currency",
        )
        assert exp.severity == "critical"
        assert "credit" in exp.explanation.lower() or "refund" in exp.explanation.lower()

    def test_percentage_over_100(self):
        exp = explain_outlier(
            value=150, column="completion", table="tasks",
            mean=80, stdev=15, min_val=0, max_val=100,
            semantic_type="numeric_percentage",
        )
        assert exp.severity == "critical"
        assert "100%" in exp.explanation

    def test_negative_percentage(self):
        exp = explain_outlier(
            value=-5, column="rate", table="metrics",
            mean=50, stdev=10, min_val=0, max_val=100,
            semantic_type="numeric_percentage",
        )
        assert exp.severity == "critical"
        assert "negative" in exp.explanation.lower()

    def test_context_has_range(self):
        exp = explain_outlier(
            value=80, column="val", table="t",
            mean=50, stdev=10, min_val=0, max_val=100,
        )
        assert "0" in exp.context
        assert "100" in exp.context

    def test_critical_recommends_investigation(self):
        exp = explain_outlier(
            value=200, column="temp", table="furnace",
            mean=100, stdev=10, min_val=80, max_val=120,
        )
        assert "investigate" in exp.recommended_action.lower()

    def test_equal_range_context(self):
        exp = explain_outlier(
            value=5, column="val", table="t",
            mean=5, stdev=0, min_val=5, max_val=5,
        )
        assert "identical" in exp.context.lower()


class TestExplainNullSpike:
    def test_critical_over_50_pct(self):
        exp = explain_null_spike("col", "t", null_count=60, total_count=100)
        assert exp.severity == "critical"
        assert "60" in exp.explanation

    def test_warning_over_20_pct(self):
        exp = explain_null_spike("col", "t", null_count=25, total_count=100)
        assert exp.severity == "warning"

    def test_info_low_pct(self):
        exp = explain_null_spike("col", "t", null_count=5, total_count=100)
        assert exp.severity == "info"

    def test_zero_total(self):
        exp = explain_null_spike("col", "t", null_count=0, total_count=0)
        assert exp.severity == "info"

    def test_context_has_counts(self):
        exp = explain_null_spike("col", "t", null_count=10, total_count=100)
        assert "10/100" in exp.context

    def test_recommended_action(self):
        exp = explain_null_spike("col", "t", null_count=50, total_count=100)
        assert "investigate" in exp.recommended_action.lower()


class TestExplainConstantColumn:
    def test_basic(self):
        exp = explain_constant_column("status", "orders", 1.0, 500)
        assert exp.severity == "info"
        assert "zero variance" in exp.explanation.lower()
        assert "500" in exp.explanation
        assert "1" in exp.explanation

    def test_recommended_action(self):
        exp = explain_constant_column("flag", "t", 0, 100)
        assert "removing" in exp.recommended_action.lower() or "consider" in exp.recommended_action.lower()
