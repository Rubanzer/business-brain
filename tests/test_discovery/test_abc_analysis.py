"""Tests for ABC analysis module."""

from business_brain.discovery.abc_analysis import (
    ABCResult,
    abc_analysis,
    abc_matrix,
    format_abc_table,
    get_category_items,
)


class TestABCAnalysis:
    def test_basic(self):
        rows = [
            {"item": "A", "value": 800},
            {"item": "B", "value": 100},
            {"item": "C", "value": 50},
            {"item": "D", "value": 30},
            {"item": "E", "value": 20},
        ]
        result = abc_analysis(rows, "item", "value")
        assert result is not None
        assert result.total_value == 1000
        assert result.a_count >= 1
        assert result.items[0].category == "A"

    def test_empty_returns_none(self):
        assert abc_analysis([], "g", "v") is None

    def test_single_item_returns_none(self):
        rows = [{"g": "A", "v": 100}]
        assert abc_analysis(rows, "g", "v") is None

    def test_all_categories_assigned(self):
        rows = [{"g": f"item_{i}", "v": 100 - i} for i in range(20)]
        result = abc_analysis(rows, "g", "v")
        assert result is not None
        assert result.a_count + result.b_count + result.c_count == 20

    def test_a_threshold(self):
        rows = [
            {"g": "A", "v": 700},
            {"g": "B", "v": 150},
            {"g": "C", "v": 100},
            {"g": "D", "v": 50},
        ]
        result = abc_analysis(rows, "g", "v", a_threshold=70.0)
        a_items = [i for i in result.items if i.category == "A"]
        assert a_items[0].name == "A"

    def test_custom_thresholds(self):
        rows = [{"g": chr(65 + i), "v": 100 - i * 5} for i in range(10)]
        result = abc_analysis(rows, "g", "v", a_threshold=60.0, b_threshold=85.0)
        assert result is not None
        assert result.a_value_pct <= 65  # roughly 60% + first item overshoot

    def test_value_percentages_sum_to_100(self):
        rows = [{"g": f"x{i}", "v": i * 10 + 1} for i in range(15)]
        result = abc_analysis(rows, "g", "v")
        total_pct = result.a_value_pct + result.b_value_pct + result.c_value_pct
        assert abs(total_pct - 100.0) < 1.0

    def test_item_percentages_sum_to_100(self):
        rows = [{"g": f"x{i}", "v": i + 1} for i in range(10)]
        result = abc_analysis(rows, "g", "v")
        total_pct = result.a_item_pct + result.b_item_pct + result.c_item_pct
        assert abs(total_pct - 100.0) < 1.0

    def test_aggregates_duplicates(self):
        rows = [
            {"g": "A", "v": 50},
            {"g": "A", "v": 50},
            {"g": "B", "v": 100},
        ]
        result = abc_analysis(rows, "g", "v")
        assert result.total_value == 200

    def test_null_values_skipped(self):
        rows = [
            {"g": "A", "v": 100},
            {"g": None, "v": 50},
            {"g": "B", "v": 50},
        ]
        result = abc_analysis(rows, "g", "v")
        assert result is not None
        assert result.total_value == 150

    def test_negative_values_use_abs(self):
        rows = [
            {"g": "A", "v": -100},
            {"g": "B", "v": 200},
        ]
        result = abc_analysis(rows, "g", "v")
        assert result is not None

    def test_summary_text(self):
        rows = [{"g": "A", "v": 80}, {"g": "B", "v": 20}]
        result = abc_analysis(rows, "g", "v")
        assert "ABC" in result.summary

    def test_ranked_by_value(self):
        rows = [
            {"g": "Small", "v": 10},
            {"g": "Big", "v": 1000},
            {"g": "Mid", "v": 100},
        ]
        result = abc_analysis(rows, "g", "v")
        assert result.items[0].name == "Big"
        assert result.items[0].rank == 1

    def test_cumulative_pct_ends_at_100(self):
        rows = [{"g": f"x{i}", "v": 10} for i in range(5)]
        result = abc_analysis(rows, "g", "v")
        assert result.items[-1].cumulative_pct == 100.0


class TestGetCategoryItems:
    def test_get_a_items(self):
        rows = [
            {"g": "Big", "v": 800},
            {"g": "Mid", "v": 100},
            {"g": "Small", "v": 50},
            {"g": "Tiny", "v": 50},
        ]
        result = abc_analysis(rows, "g", "v")
        a_items = get_category_items(result, "A")
        assert len(a_items) >= 1
        assert all(i.category == "A" for i in a_items)

    def test_get_c_items(self):
        rows = [{"g": f"x{i}", "v": 100 - i * 5} for i in range(20)]
        result = abc_analysis(rows, "g", "v")
        c_items = get_category_items(result, "C")
        assert len(c_items) >= 1


class TestABCMatrix:
    def test_basic_matrix(self):
        rows = [
            {"item": "A", "value": 1000, "volume": 500},
            {"item": "B", "value": 500, "volume": 1000},
            {"item": "C", "value": 100, "volume": 100},
            {"item": "D", "value": 50, "volume": 50},
        ]
        matrix = abc_matrix(rows, "item", "value", "volume")
        assert len(matrix) >= 2
        assert all("priority" in m for m in matrix)

    def test_empty_returns_empty(self):
        matrix = abc_matrix([], "item", "value", "volume")
        assert matrix == []

    def test_combined_categories(self):
        rows = [
            {"item": "A", "value": 1000, "volume": 1000},
            {"item": "B", "value": 10, "volume": 10},
        ]
        matrix = abc_matrix(rows, "item", "value", "volume")
        aa = [m for m in matrix if m["combined"] == "AA"]
        assert len(aa) >= 1


class TestFormatABCTable:
    def test_basic_format(self):
        rows = [
            {"g": "A", "v": 80},
            {"g": "B", "v": 20},
        ]
        result = abc_analysis(rows, "g", "v")
        text = format_abc_table(result)
        assert "ABC Analysis" in text
        assert "A" in text
        assert "B" in text

    def test_contains_totals(self):
        rows = [{"g": f"x{i}", "v": 10} for i in range(5)]
        result = abc_analysis(rows, "g", "v")
        text = format_abc_table(result)
        assert "A:" in text
        assert "B:" in text
        assert "C:" in text
