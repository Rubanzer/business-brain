"""Tests for pivot_engine — pivot table creation, formatting, and analysis."""

from __future__ import annotations

import math

import pytest

from business_brain.discovery.pivot_engine import (
    PivotCell,
    PivotTable,
    compute_pivot_percentages,
    create_pivot,
    find_pivot_outliers,
    format_pivot_csv,
    format_pivot_text,
    multi_pivot,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SALES_DATA = [
    {"region": "East", "quarter": "Q1", "revenue": 100, "cost": 40},
    {"region": "East", "quarter": "Q2", "revenue": 150, "cost": 60},
    {"region": "East", "quarter": "Q3", "revenue": 200, "cost": 80},
    {"region": "East", "quarter": "Q4", "revenue": 250, "cost": 100},
    {"region": "West", "quarter": "Q1", "revenue": 120, "cost": 50},
    {"region": "West", "quarter": "Q2", "revenue": 180, "cost": 70},
    {"region": "West", "quarter": "Q3", "revenue": 160, "cost": 65},
    {"region": "West", "quarter": "Q4", "revenue": 210, "cost": 85},
    {"region": "North", "quarter": "Q1", "revenue": 90, "cost": 35},
    {"region": "North", "quarter": "Q2", "revenue": 110, "cost": 45},
    {"region": "North", "quarter": "Q3", "revenue": 130, "cost": 55},
    {"region": "North", "quarter": "Q4", "revenue": 170, "cost": 70},
]


def _basic_pivot() -> PivotTable:
    pt = create_pivot(SALES_DATA, "region", "quarter", "revenue", "sum")
    assert pt is not None
    return pt


# ---------------------------------------------------------------------------
# 1. Basic pivot with sum
# ---------------------------------------------------------------------------


class TestBasicSum:
    def test_sum_values(self):
        pt = _basic_pivot()
        assert pt.cells[("East", "Q1")].value == 100
        assert pt.cells[("West", "Q2")].value == 180
        assert pt.cells[("North", "Q4")].value == 170

    def test_row_keys_sorted(self):
        pt = _basic_pivot()
        assert pt.row_keys == ["East", "North", "West"]

    def test_col_keys_sorted(self):
        pt = _basic_pivot()
        assert pt.col_keys == ["Q1", "Q2", "Q3", "Q4"]

    def test_agg_func_stored(self):
        pt = _basic_pivot()
        assert pt.agg_func == "sum"

    def test_fields_stored(self):
        pt = _basic_pivot()
        assert pt.row_field == "region"
        assert pt.col_field == "quarter"
        assert pt.value_field == "revenue"


# ---------------------------------------------------------------------------
# 2. Mean aggregation
# ---------------------------------------------------------------------------


class TestMeanAgg:
    def test_mean_single_value_per_cell(self):
        pt = create_pivot(SALES_DATA, "region", "quarter", "revenue", "mean")
        assert pt is not None
        # Each cell has exactly 1 row, so mean == the value itself
        assert pt.cells[("East", "Q1")].value == 100.0

    def test_mean_multiple_values(self):
        rows = [
            {"a": "X", "b": "Y", "v": 10},
            {"a": "X", "b": "Y", "v": 20},
            {"a": "X", "b": "Z", "v": 30},
        ]
        pt = create_pivot(rows, "a", "b", "v", "mean")
        assert pt is not None
        assert pt.cells[("X", "Y")].value == pytest.approx(15.0)
        assert pt.cells[("X", "Y")].count == 2
        assert pt.cells[("X", "Z")].value == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 3. Count aggregation
# ---------------------------------------------------------------------------


class TestCountAgg:
    def test_count_values(self):
        pt = create_pivot(SALES_DATA, "region", "quarter", "revenue", "count")
        assert pt is not None
        assert pt.cells[("East", "Q1")].value == 1.0
        assert pt.cells[("West", "Q3")].value == 1.0

    def test_count_multiple_entries(self):
        rows = [
            {"a": "X", "b": "Y", "v": 10},
            {"a": "X", "b": "Y", "v": 20},
            {"a": "X", "b": "Z", "v": 30},
        ]
        pt = create_pivot(rows, "a", "b", "v", "count")
        assert pt is not None
        assert pt.cells[("X", "Y")].value == 2.0
        assert pt.cells[("X", "Z")].value == 1.0


# ---------------------------------------------------------------------------
# 4. Min / Max aggregation
# ---------------------------------------------------------------------------


class TestMinMaxAgg:
    def test_min_agg(self):
        rows = [
            {"a": "X", "b": "Y", "v": 10},
            {"a": "X", "b": "Y", "v": 5},
            {"a": "X", "b": "Z", "v": 30},
        ]
        pt = create_pivot(rows, "a", "b", "v", "min")
        assert pt is not None
        assert pt.cells[("X", "Y")].value == 5.0

    def test_max_agg(self):
        rows = [
            {"a": "X", "b": "Y", "v": 10},
            {"a": "X", "b": "Y", "v": 25},
            {"a": "X", "b": "Z", "v": 3},
        ]
        pt = create_pivot(rows, "a", "b", "v", "max")
        assert pt is not None
        assert pt.cells[("X", "Y")].value == 25.0
        assert pt.cells[("X", "Z")].value == 3.0


# ---------------------------------------------------------------------------
# 5. Row and column totals
# ---------------------------------------------------------------------------


class TestTotals:
    def test_row_totals(self):
        pt = _basic_pivot()
        # East: 100 + 150 + 200 + 250 = 700
        assert pt.row_totals["East"] == pytest.approx(700.0)
        # West: 120 + 180 + 160 + 210 = 670
        assert pt.row_totals["West"] == pytest.approx(670.0)

    def test_col_totals(self):
        pt = _basic_pivot()
        # Q1: 100 + 120 + 90 = 310
        assert pt.col_totals["Q1"] == pytest.approx(310.0)
        # Q4: 250 + 210 + 170 = 630
        assert pt.col_totals["Q4"] == pytest.approx(630.0)

    def test_grand_total(self):
        pt = _basic_pivot()
        expected = sum(r["revenue"] for r in SALES_DATA)
        assert pt.grand_total == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6. Grand total consistency
# ---------------------------------------------------------------------------


class TestGrandTotal:
    def test_grand_total_equals_sum_of_row_totals(self):
        pt = _basic_pivot()
        assert pt.grand_total == pytest.approx(sum(pt.row_totals.values()))

    def test_grand_total_equals_sum_of_col_totals(self):
        pt = _basic_pivot()
        assert pt.grand_total == pytest.approx(sum(pt.col_totals.values()))


# ---------------------------------------------------------------------------
# 7. Text formatting
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_text_has_header(self):
        pt = _basic_pivot()
        txt = format_pivot_text(pt)
        assert "region" in txt
        assert "Q1" in txt
        assert "Total" in txt

    def test_text_has_data(self):
        pt = _basic_pivot()
        txt = format_pivot_text(pt)
        assert "East" in txt
        assert "100" in txt

    def test_text_truncation(self):
        # Create pivot with many columns
        rows = []
        for i in range(15):
            rows.append({"r": "A", "c": f"C{i:02d}", "v": i * 10})
            rows.append({"r": "B", "c": f"C{i:02d}", "v": i * 5})
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        txt = format_pivot_text(pt, max_cols=5)
        assert "..." in txt

    def test_text_no_truncation_when_within_limit(self):
        pt = _basic_pivot()
        txt = format_pivot_text(pt, max_cols=10)
        assert "..." not in txt


# ---------------------------------------------------------------------------
# 8. CSV formatting
# ---------------------------------------------------------------------------


class TestFormatCSV:
    def test_csv_header(self):
        pt = _basic_pivot()
        csv = format_pivot_csv(pt)
        first_line = csv.split("\n")[0]
        assert first_line.startswith("region,")
        assert "Q1" in first_line
        assert "Total" in first_line

    def test_csv_row_count(self):
        pt = _basic_pivot()
        csv = format_pivot_csv(pt)
        lines = csv.strip().split("\n")
        # header + 3 regions + totals = 5
        assert len(lines) == 5

    def test_csv_values(self):
        pt = _basic_pivot()
        csv = format_pivot_csv(pt)
        # Should contain East row with 100
        assert "East" in csv
        assert "100" in csv

    def test_csv_escaping(self):
        rows = [
            {"r": 'He said "hi"', "c": "A", "v": 10},
            {"r": 'He said "hi"', "c": "B", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        csv = format_pivot_csv(pt)
        # The row key should be escaped
        assert '""' in csv


# ---------------------------------------------------------------------------
# 9. Percentage computation (row, col, total modes)
# ---------------------------------------------------------------------------


class TestPercentages:
    def test_row_percentages_sum_to_100(self):
        pt = _basic_pivot()
        pcts = compute_pivot_percentages(pt, mode="row")
        for rk in pt.row_keys:
            total_pct = sum(pcts[(rk, ck)] for ck in pt.col_keys)
            assert total_pct == pytest.approx(100.0)

    def test_col_percentages_sum_to_100(self):
        pt = _basic_pivot()
        pcts = compute_pivot_percentages(pt, mode="col")
        for ck in pt.col_keys:
            total_pct = sum(pcts[(rk, ck)] for rk in pt.row_keys)
            assert total_pct == pytest.approx(100.0)

    def test_total_percentages_sum_to_100(self):
        pt = _basic_pivot()
        pcts = compute_pivot_percentages(pt, mode="total")
        total_pct = sum(pcts.values())
        assert total_pct == pytest.approx(100.0)

    def test_specific_row_percentage(self):
        pt = _basic_pivot()
        pcts = compute_pivot_percentages(pt, mode="row")
        # East Q1: 100 out of 700
        assert pcts[("East", "Q1")] == pytest.approx(100.0 / 700.0 * 100.0)


# ---------------------------------------------------------------------------
# 10. Outlier detection in pivot
# ---------------------------------------------------------------------------


class TestOutlierDetection:
    def test_no_outliers_in_uniform_data(self):
        rows = [
            {"r": "A", "c": "X", "v": 10},
            {"r": "A", "c": "Y", "v": 10},
            {"r": "A", "c": "Z", "v": 10},
            {"r": "B", "c": "X", "v": 20},
            {"r": "B", "c": "Y", "v": 20},
            {"r": "B", "c": "Z", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        outliers = find_pivot_outliers(pt, threshold=2.0)
        assert len(outliers) == 0

    def test_detects_obvious_outlier(self):
        rows = [
            {"r": "A", "c": "X", "v": 10},
            {"r": "A", "c": "Y", "v": 10},
            {"r": "A", "c": "Z", "v": 10},
            {"r": "A", "c": "W", "v": 1000},  # outlier!
            {"r": "B", "c": "X", "v": 20},
            {"r": "B", "c": "Y", "v": 20},
            {"r": "B", "c": "Z", "v": 20},
            {"r": "B", "c": "W", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        outliers = find_pivot_outliers(pt, threshold=1.5)
        assert len(outliers) >= 1
        outlier_vals = {o["value"] for o in outliers}
        assert 1000 in outlier_vals

    def test_outlier_structure(self):
        rows = [
            {"r": "A", "c": "X", "v": 10},
            {"r": "A", "c": "Y", "v": 10},
            {"r": "A", "c": "Z", "v": 10},
            {"r": "A", "c": "W", "v": 1000},
            {"r": "B", "c": "X", "v": 20},
            {"r": "B", "c": "Y", "v": 20},
            {"r": "B", "c": "Z", "v": 20},
            {"r": "B", "c": "W", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        outliers = find_pivot_outliers(pt, threshold=1.5)
        for o in outliers:
            assert "row" in o
            assert "col" in o
            assert "value" in o
            assert "expected" in o
            assert "deviation" in o


# ---------------------------------------------------------------------------
# 11. Multi-pivot (multiple value fields)
# ---------------------------------------------------------------------------


class TestMultiPivot:
    def test_multi_pivot_returns_two_tables(self):
        tables = multi_pivot(
            SALES_DATA, "region", "quarter", ["revenue", "cost"], "sum"
        )
        assert len(tables) == 2

    def test_multi_pivot_correct_value_fields(self):
        tables = multi_pivot(
            SALES_DATA, "region", "quarter", ["revenue", "cost"], "sum"
        )
        # Order is preserved from the input value_fields list
        assert tables[0].value_field == "revenue"
        assert tables[1].value_field == "cost"
        fields = {t.value_field for t in tables}
        assert fields == {"revenue", "cost"}

    def test_multi_pivot_skips_bad_field(self):
        tables = multi_pivot(
            SALES_DATA, "region", "quarter", ["revenue", "nonexistent"], "sum"
        )
        assert len(tables) == 1
        assert tables[0].value_field == "revenue"


# ---------------------------------------------------------------------------
# 12. Missing combinations
# ---------------------------------------------------------------------------


class TestMissingCombinations:
    def test_missing_combos_filled_with_zero(self):
        rows = [
            {"r": "A", "c": "X", "v": 10},
            {"r": "A", "c": "Y", "v": 20},
            {"r": "B", "c": "X", "v": 30},
            # B-Y missing
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        assert pt.cells[("B", "Y")].value == 0.0
        assert pt.cells[("B", "Y")].count == 0


# ---------------------------------------------------------------------------
# 13. Single row / single col
# ---------------------------------------------------------------------------


class TestSingleDimension:
    def test_single_row_key(self):
        rows = [
            {"r": "A", "c": "X", "v": 10},
            {"r": "A", "c": "Y", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        assert pt.row_keys == ["A"]
        assert pt.col_keys == ["X", "Y"]
        assert pt.grand_total == pytest.approx(30.0)

    def test_single_col_key(self):
        rows = [
            {"r": "A", "c": "X", "v": 10},
            {"r": "B", "c": "X", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        assert pt.col_keys == ["X"]
        assert pt.row_keys == ["A", "B"]


# ---------------------------------------------------------------------------
# 14. Empty / insufficient data
# ---------------------------------------------------------------------------


class TestEmptyData:
    def test_empty_list_returns_none(self):
        assert create_pivot([], "r", "c", "v", "sum") is None

    def test_single_row_returns_none(self):
        rows = [{"r": "A", "c": "X", "v": 10}]
        assert create_pivot(rows, "r", "c", "v", "sum") is None

    def test_invalid_agg_returns_none(self):
        assert create_pivot(SALES_DATA, "region", "quarter", "revenue", "median") is None

    def test_missing_fields_returns_none(self):
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert create_pivot(rows, "x", "y", "z", "sum") is None


# ---------------------------------------------------------------------------
# 15. Null / None handling
# ---------------------------------------------------------------------------


class TestNullHandling:
    def test_none_values_skipped(self):
        rows = [
            {"r": "A", "c": "X", "v": 10},
            {"r": "A", "c": "Y", "v": None},
            {"r": "B", "c": "X", "v": 20},
            {"r": "B", "c": "Y", "v": 30},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        # None value row is excluded; A-Y should still exist with value 0
        # because row A-Y had no valid numeric value_field
        assert pt.cells[("A", "Y")].value == 0.0

    def test_none_row_key_skipped(self):
        rows = [
            {"r": None, "c": "X", "v": 10},
            {"r": "A", "c": "X", "v": 20},
            {"r": "A", "c": "Y", "v": 30},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        assert "None" not in pt.row_keys  # None key should not appear
        assert pt.row_keys == ["A"]

    def test_nan_values_skipped(self):
        rows = [
            {"r": "A", "c": "X", "v": float("nan")},
            {"r": "A", "c": "Y", "v": 20},
            {"r": "B", "c": "X", "v": 30},
            {"r": "B", "c": "Y", "v": 40},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        # A-X had nan so should be 0
        assert pt.cells[("A", "X")].value == 0.0

    def test_all_none_returns_none(self):
        rows = [
            {"r": "A", "c": "X", "v": None},
            {"r": "B", "c": "Y", "v": None},
        ]
        assert create_pivot(rows, "r", "c", "v", "sum") is None


# ---------------------------------------------------------------------------
# 16. Large dataset
# ---------------------------------------------------------------------------


class TestLargeDataset:
    def test_large_pivot(self):
        rows = []
        for i in range(100):
            for j in range(50):
                rows.append({"r": f"R{i:03d}", "c": f"C{j:02d}", "v": float(i + j)})
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        assert len(pt.row_keys) == 100
        assert len(pt.col_keys) == 50
        # Check a specific cell
        assert pt.cells[("R000", "C00")].value == pytest.approx(0.0)
        assert pt.cells[("R099", "C49")].value == pytest.approx(148.0)
        # Grand total: sum of (i+j) for i in 0..99, j in 0..49
        expected = sum(i + j for i in range(100) for j in range(50))
        assert pt.grand_total == pytest.approx(float(expected))

    def test_large_text_format(self):
        rows = []
        for i in range(20):
            for j in range(20):
                rows.append({"r": f"R{i:02d}", "c": f"C{j:02d}", "v": float(i * j)})
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        txt = format_pivot_text(pt, max_cols=10)
        assert "..." in txt
        lines = txt.split("\n")
        assert len(lines) > 0


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_summary_field(self):
        pt = _basic_pivot()
        assert "revenue" in pt.summary
        assert "region" in pt.summary
        assert "quarter" in pt.summary
        assert "sum" in pt.summary

    def test_percentage_zero_total_row(self):
        """Row with all zeros should give 0% for all cells."""
        rows = [
            {"r": "A", "c": "X", "v": 0},
            {"r": "A", "c": "Y", "v": 0},
            {"r": "B", "c": "X", "v": 10},
            {"r": "B", "c": "Y", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        pcts = compute_pivot_percentages(pt, mode="row")
        assert pcts[("A", "X")] == 0.0
        assert pcts[("A", "Y")] == 0.0

    def test_count_ignores_none_value_field(self):
        """Count should count rows where row_field and col_field exist."""
        rows = [
            {"r": "A", "c": "X", "v": None},
            {"r": "A", "c": "X", "v": 10},
            {"r": "B", "c": "Y", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "count")
        assert pt is not None
        # count agg doesn't require numeric value_field, so both A-X rows count
        assert pt.cells[("A", "X")].value == 2.0

    def test_string_values_in_row_col(self):
        """Row/col keys should handle numeric types by converting to str."""
        rows = [
            {"r": 1, "c": 2, "v": 10},
            {"r": 1, "c": 3, "v": 20},
            {"r": 2, "c": 2, "v": 30},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        assert pt.row_keys == ["1", "2"]
        assert pt.col_keys == ["2", "3"]

    def test_outlier_with_single_column(self):
        """Outlier detection with 1 column per row should return nothing."""
        rows = [
            {"r": "A", "c": "X", "v": 10},
            {"r": "B", "c": "X", "v": 20},
        ]
        pt = create_pivot(rows, "r", "c", "v", "sum")
        assert pt is not None
        outliers = find_pivot_outliers(pt, threshold=2.0)
        assert len(outliers) == 0
