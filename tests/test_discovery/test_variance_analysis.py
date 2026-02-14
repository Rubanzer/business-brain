"""Tests for budget vs actual variance analysis."""

from __future__ import annotations

import math

import pytest

from business_brain.discovery.variance_analysis import (
    VarianceItem,
    VarianceReport,
    compute_variance,
    compute_variance_trend,
    find_root_causes,
    format_variance_table,
    waterfall_breakdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(data: list[tuple[str, float, float]]) -> list[dict]:
    """Shortcut: (category, planned, actual) -> list of row dicts."""
    return [
        {"category": cat, "planned": p, "actual": a}
        for cat, p, a in data
    ]


# ---------------------------------------------------------------------------
# 1. Basic variance computation
# ---------------------------------------------------------------------------

class TestComputeVarianceBasic:
    def test_basic_single_category(self):
        rows = _make_rows([("Sales", 100, 120)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert len(report.items) == 1
        item = report.items[0]
        assert item.category == "Sales"
        assert item.planned == 100
        assert item.actual == 120
        assert item.variance == 20
        assert item.variance_pct == 20.0

    def test_basic_multiple_categories(self):
        rows = _make_rows([
            ("Sales", 1000, 1100),
            ("Marketing", 500, 450),
        ])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert len(report.items) == 2
        assert report.total_planned == 1500
        assert report.total_actual == 1550
        assert report.total_variance == 50

    def test_returns_none_for_empty_rows(self):
        assert compute_variance([], "category", "planned", "actual") is None

    def test_returns_none_for_missing_columns(self):
        rows = [{"x": 1, "y": 2, "z": 3}]
        assert compute_variance(rows, "category", "planned", "actual") is None

    def test_skips_non_numeric_values(self):
        rows = [
            {"category": "A", "planned": 100, "actual": 120},
            {"category": "B", "planned": "bad", "actual": 50},
        ]
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert len(report.items) == 1
        assert report.items[0].category == "A"


# ---------------------------------------------------------------------------
# 2. Higher favorable direction (revenue)
# ---------------------------------------------------------------------------

class TestHigherFavorable:
    def test_positive_variance_is_favorable(self):
        rows = _make_rows([("Revenue", 1000, 1200)])
        report = compute_variance(rows, "category", "planned", "actual", "higher")
        assert report is not None
        assert report.items[0].is_favorable is True

    def test_negative_variance_is_unfavorable(self):
        rows = _make_rows([("Revenue", 1000, 800)])
        report = compute_variance(rows, "category", "planned", "actual", "higher")
        assert report is not None
        assert report.items[0].is_favorable is False

    def test_zero_variance_is_favorable(self):
        rows = _make_rows([("Revenue", 1000, 1000)])
        report = compute_variance(rows, "category", "planned", "actual", "higher")
        assert report is not None
        assert report.items[0].is_favorable is True


# ---------------------------------------------------------------------------
# 3. Lower favorable direction (cost)
# ---------------------------------------------------------------------------

class TestLowerFavorable:
    def test_negative_variance_is_favorable_for_cost(self):
        rows = _make_rows([("Rent", 1000, 900)])
        report = compute_variance(rows, "category", "planned", "actual", "lower")
        assert report is not None
        assert report.items[0].is_favorable is True

    def test_positive_variance_is_unfavorable_for_cost(self):
        rows = _make_rows([("Rent", 1000, 1200)])
        report = compute_variance(rows, "category", "planned", "actual", "lower")
        assert report is not None
        assert report.items[0].is_favorable is False


# ---------------------------------------------------------------------------
# 4. Severity classification
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_minor_severity(self):
        rows = _make_rows([("A", 1000, 1050)])  # 5%
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.items[0].severity == "minor"

    def test_warning_severity(self):
        rows = _make_rows([("A", 1000, 1150)])  # 15%
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.items[0].severity == "warning"

    def test_critical_severity(self):
        rows = _make_rows([("A", 1000, 1250)])  # 25%
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.items[0].severity == "critical"

    def test_exactly_ten_percent_is_minor(self):
        rows = _make_rows([("A", 1000, 1100)])  # exactly 10%
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.items[0].severity == "minor"

    def test_exactly_twenty_percent_is_warning(self):
        rows = _make_rows([("A", 1000, 1200)])  # exactly 20%
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.items[0].severity == "warning"


# ---------------------------------------------------------------------------
# 5. Waterfall breakdown
# ---------------------------------------------------------------------------

class TestWaterfall:
    def test_waterfall_cumulative(self):
        rows = _make_rows([
            ("A", 100, 150),
            ("B", 200, 180),
            ("C", 300, 350),
        ])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        wf = waterfall_breakdown(report)
        assert len(wf) == 3
        # First item starts at 0
        assert wf[0]["start"] == 0
        assert wf[0]["end"] == wf[0]["variance"]
        # Each subsequent item starts where the previous ended
        for i in range(1, len(wf)):
            assert wf[i]["start"] == wf[i - 1]["end"]

    def test_waterfall_types(self):
        rows = _make_rows([
            ("Rev", 100, 120),   # favorable (higher default)
            ("Cost", 100, 80),   # unfavorable (higher default)
        ])
        report = compute_variance(rows, "category", "planned", "actual", "higher")
        assert report is not None
        wf = waterfall_breakdown(report)
        # Cost item: variance = -20, unfavorable for "higher"
        cost_item = [w for w in wf if w["category"] == "Cost"][0]
        assert cost_item["type"] == "unfavorable"
        rev_item = [w for w in wf if w["category"] == "Rev"][0]
        assert rev_item["type"] == "favorable"

    def test_waterfall_empty_report(self):
        report = VarianceReport(
            items=[], total_planned=0, total_actual=0, total_variance=0,
            total_variance_pct=0, favorable_count=0, unfavorable_count=0,
            largest_variance=None, summary="",
        )
        assert waterfall_breakdown(report) == []


# ---------------------------------------------------------------------------
# 6. Root cause identification
# ---------------------------------------------------------------------------

class TestRootCauses:
    def test_filters_by_threshold(self):
        rows = _make_rows([
            ("A", 1000, 1050),   # 5% — below threshold
            ("B", 1000, 1200),   # 20% — at threshold for warning
            ("C", 1000, 1500),   # 50% — critical
        ])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        causes = find_root_causes(report, threshold_pct=10.0)
        categories = [c["category"] for c in causes]
        assert "A" not in categories
        assert "B" in categories
        assert "C" in categories

    def test_sorted_by_absolute_variance(self):
        rows = _make_rows([
            ("Small", 1000, 1150),   # +150
            ("Big", 1000, 1500),     # +500
        ])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        causes = find_root_causes(report, threshold_pct=10.0)
        assert causes[0]["category"] == "Big"
        assert causes[1]["category"] == "Small"

    def test_no_causes_below_threshold(self):
        rows = _make_rows([("A", 1000, 1050)])  # 5%
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        causes = find_root_causes(report, threshold_pct=10.0)
        assert causes == []


# ---------------------------------------------------------------------------
# 7. Variance trend over periods
# ---------------------------------------------------------------------------

class TestVarianceTrend:
    def test_basic_trend(self):
        periods = ["Q1", "Q2", "Q3"]
        planned = [100, 200, 300]
        actual = [110, 190, 350]
        trend = compute_variance_trend(periods, planned, actual)
        assert len(trend) == 3
        assert trend[0]["period"] == "Q1"
        assert trend[0]["variance"] == 10
        assert trend[0]["variance_pct"] == 10.0

    def test_trend_favorable_direction_lower(self):
        periods = ["Jan"]
        planned = [100.0]
        actual = [80.0]
        trend = compute_variance_trend(periods, planned, actual, "lower")
        assert trend[0]["is_favorable"] is True

    def test_trend_mismatched_lengths(self):
        periods = ["Q1", "Q2", "Q3"]
        planned = [100, 200]
        actual = [110, 190, 350]
        trend = compute_variance_trend(periods, planned, actual)
        assert len(trend) == 2  # limited to shortest


# ---------------------------------------------------------------------------
# 8. Text table formatting
# ---------------------------------------------------------------------------

class TestFormatTable:
    def test_table_has_header_and_total(self):
        rows = _make_rows([("Sales", 100, 120)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        table = format_variance_table(report)
        assert "Category" in table
        assert "Planned" in table
        assert "TOTAL" in table

    def test_table_favorable_markers(self):
        rows = _make_rows([
            ("Good", 100, 120),
            ("Bad", 100, 80),
        ])
        report = compute_variance(rows, "category", "planned", "actual", "higher")
        assert report is not None
        table = format_variance_table(report)
        lines = table.split("\n")
        # Find the line for "Good" — should have [+]
        good_line = [l for l in lines if "Good" in l][0]
        assert "[+]" in good_line
        # Find the line for "Bad" — should have [-]
        bad_line = [l for l in lines if "Bad" in l][0]
        assert "[-]" in bad_line


# ---------------------------------------------------------------------------
# 9. Zero planned values
# ---------------------------------------------------------------------------

class TestZeroPlanned:
    def test_zero_planned_positive_actual(self):
        rows = _make_rows([("New", 0, 100)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        item = report.items[0]
        assert item.variance == 100
        assert item.variance_pct == float("inf")

    def test_zero_planned_negative_actual(self):
        rows = _make_rows([("Loss", 0, -50)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.items[0].variance_pct == float("-inf")

    def test_zero_planned_zero_actual(self):
        rows = _make_rows([("Nothing", 0, 0)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.items[0].variance_pct == 0.0


# ---------------------------------------------------------------------------
# 10. All favorable variances
# ---------------------------------------------------------------------------

class TestAllFavorable:
    def test_all_favorable_higher(self):
        rows = _make_rows([
            ("A", 100, 150),
            ("B", 200, 250),
        ])
        report = compute_variance(rows, "category", "planned", "actual", "higher")
        assert report is not None
        assert report.favorable_count == 2
        assert report.unfavorable_count == 0
        assert all(it.is_favorable for it in report.items)


# ---------------------------------------------------------------------------
# 11. All unfavorable variances
# ---------------------------------------------------------------------------

class TestAllUnfavorable:
    def test_all_unfavorable_higher(self):
        rows = _make_rows([
            ("A", 100, 80),
            ("B", 200, 150),
        ])
        report = compute_variance(rows, "category", "planned", "actual", "higher")
        assert report is not None
        assert report.favorable_count == 0
        assert report.unfavorable_count == 2
        assert all(not it.is_favorable for it in report.items)


# ---------------------------------------------------------------------------
# 12. Aggregation of duplicate categories
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_duplicate_categories_are_summed(self):
        rows = _make_rows([
            ("Sales", 100, 120),
            ("Sales", 200, 210),
            ("Marketing", 50, 40),
        ])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        sales = [it for it in report.items if it.category == "Sales"][0]
        assert sales.planned == 300
        assert sales.actual == 330
        assert sales.variance == 30
        assert len(report.items) == 2


# ---------------------------------------------------------------------------
# 13. Single item
# ---------------------------------------------------------------------------

class TestSingleItem:
    def test_single_item_report(self):
        rows = _make_rows([("Only", 500, 600)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.total_planned == 500
        assert report.total_actual == 600
        assert report.largest_variance is not None
        assert report.largest_variance.category == "Only"


# ---------------------------------------------------------------------------
# 14. Negative values
# ---------------------------------------------------------------------------

class TestNegativeValues:
    def test_negative_planned_and_actual(self):
        rows = _make_rows([("Refund", -100, -80)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        item = report.items[0]
        # variance = -80 - (-100) = 20
        assert item.variance == 20
        # variance_pct = 20 / (-100) * 100 = -20%
        assert item.variance_pct == -20.0

    def test_negative_variance(self):
        rows = _make_rows([("Loss", 100, -50)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.items[0].variance == -150


# ---------------------------------------------------------------------------
# 15. Large variance percentages
# ---------------------------------------------------------------------------

class TestLargeVariance:
    def test_very_large_positive_variance(self):
        rows = _make_rows([("Boom", 10, 1000)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        item = report.items[0]
        assert item.variance == 990
        assert item.variance_pct == 9900.0
        assert item.severity == "critical"

    def test_very_large_negative_variance(self):
        rows = _make_rows([("Bust", 1000, 10)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        item = report.items[0]
        assert item.variance == -990
        assert item.variance_pct == -99.0
        assert item.severity == "critical"


# ---------------------------------------------------------------------------
# 16. Summary content
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_key_info(self):
        rows = _make_rows([
            ("A", 100, 130),
            ("B", 200, 180),
        ])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert "favorable" in report.summary
        assert "unfavorable" in report.summary
        assert "Largest outlier" in report.summary


# ---------------------------------------------------------------------------
# 17. Largest variance selection
# ---------------------------------------------------------------------------

class TestLargestVariance:
    def test_largest_variance_by_absolute_value(self):
        rows = _make_rows([
            ("Small", 100, 105),
            ("Big", 100, 200),
            ("Negative", 100, 50),
        ])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        assert report.largest_variance is not None
        assert report.largest_variance.category == "Big"


# ---------------------------------------------------------------------------
# 18. Root causes include inf variance_pct
# ---------------------------------------------------------------------------

class TestRootCausesInfPct:
    def test_inf_variance_pct_included(self):
        rows = _make_rows([("New", 0, 500)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        causes = find_root_causes(report, threshold_pct=10.0)
        assert len(causes) == 1
        assert causes[0]["variance_pct"] == float("inf")


# ---------------------------------------------------------------------------
# 19. Waterfall total matches report total
# ---------------------------------------------------------------------------

class TestWaterfallTotal:
    def test_waterfall_end_equals_total_variance(self):
        rows = _make_rows([
            ("A", 100, 150),
            ("B", 200, 180),
            ("C", 300, 350),
        ])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        wf = waterfall_breakdown(report)
        total_end = wf[-1]["end"]
        assert abs(total_end - report.total_variance) < 0.01


# ---------------------------------------------------------------------------
# 20. Format table with inf values
# ---------------------------------------------------------------------------

class TestFormatTableInf:
    def test_table_handles_inf_percentage(self):
        rows = _make_rows([("New", 0, 100)])
        report = compute_variance(rows, "category", "planned", "actual")
        assert report is not None
        table = format_variance_table(report)
        assert "+inf%" in table
