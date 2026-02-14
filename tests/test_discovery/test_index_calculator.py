"""Tests for composite index calculator."""

from business_brain.discovery.index_calculator import (
    IndexResult,
    compare_entities,
    compute_index,
    format_index_table,
)


class TestComputeIndex:
    def test_basic(self):
        rows = [
            {"supplier": "A", "quality": 90, "cost": 50},
            {"supplier": "B", "quality": 70, "cost": 80},
            {"supplier": "C", "quality": 60, "cost": 30},
        ]
        metrics = [
            {"column": "quality", "weight": 0.5, "direction": "higher_is_better"},
            {"column": "cost", "weight": 0.5, "direction": "lower_is_better"},
        ]
        result = compute_index(rows, "supplier", metrics, "Supplier Score")
        assert result is not None
        assert result.entity_count == 3
        assert result.name == "Supplier Score"

    def test_ranking(self):
        rows = [
            {"e": "Good", "score": 100},
            {"e": "Bad", "score": 10},
            {"e": "Mid", "score": 50},
        ]
        metrics = [{"column": "score", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics)
        assert result.scores[0].entity == "Good"
        assert result.scores[0].rank == 1
        assert result.scores[-1].entity == "Bad"

    def test_grades(self):
        rows = [
            {"e": "A", "s": 100},
            {"e": "B", "s": 80},
            {"e": "C", "s": 60},
            {"e": "D", "s": 40},
            {"e": "F", "s": 0},
        ]
        metrics = [{"column": "s", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics)
        grade_map = {s.entity: s.grade for s in result.scores}
        assert grade_map["A"] == "A"
        assert grade_map["F"] == "F"

    def test_lower_is_better(self):
        rows = [
            {"e": "Cheap", "cost": 10},
            {"e": "Expensive", "cost": 100},
        ]
        metrics = [{"column": "cost", "weight": 1.0, "direction": "lower_is_better"}]
        result = compute_index(rows, "e", metrics)
        assert result.scores[0].entity == "Cheap"

    def test_empty_returns_none(self):
        assert compute_index([], "e", [{"column": "x", "weight": 1}]) is None

    def test_no_metrics_returns_none(self):
        assert compute_index([{"e": "A", "x": 1}], "e", []) is None

    def test_single_entity_returns_none(self):
        rows = [{"e": "A", "x": 100}]
        metrics = [{"column": "x", "weight": 1}]
        assert compute_index(rows, "e", metrics) is None

    def test_aggregates_multiple_rows(self):
        rows = [
            {"e": "A", "x": 80},
            {"e": "A", "x": 100},  # mean = 90
            {"e": "B", "x": 50},
        ]
        metrics = [{"column": "x", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics)
        assert result.scores[0].entity == "A"

    def test_multiple_metrics(self):
        rows = [
            {"e": "A", "quality": 90, "speed": 80, "cost": 20},
            {"e": "B", "quality": 70, "speed": 95, "cost": 50},
            {"e": "C", "quality": 60, "speed": 60, "cost": 10},
        ]
        metrics = [
            {"column": "quality", "weight": 0.4, "direction": "higher_is_better"},
            {"column": "speed", "weight": 0.3, "direction": "higher_is_better"},
            {"column": "cost", "weight": 0.3, "direction": "lower_is_better"},
        ]
        result = compute_index(rows, "e", metrics, "Performance Index")
        assert result is not None
        assert len(result.scores[0].components) == 3

    def test_statistics(self):
        rows = [
            {"e": "A", "x": 100},
            {"e": "B", "x": 50},
            {"e": "C", "x": 0},
        ]
        metrics = [{"column": "x", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics)
        assert result.mean_score == 50.0  # (100 + 50 + 0) / 3 = 50
        assert result.std_score >= 0

    def test_top_and_bottom(self):
        rows = [
            {"e": "Best", "x": 100},
            {"e": "Worst", "x": 10},
        ]
        metrics = [{"column": "x", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics)
        assert result.top_entity == "Best"
        assert result.bottom_entity == "Worst"

    def test_summary(self):
        rows = [{"e": "A", "x": 100}, {"e": "B", "x": 50}]
        metrics = [{"column": "x", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics, "Test Index")
        assert "Test Index" in result.summary
        assert "2 entities" in result.summary

    def test_null_values_skipped(self):
        rows = [
            {"e": "A", "x": 100},
            {"e": "B", "x": None},
            {"e": "C", "x": 50},
        ]
        metrics = [{"column": "x", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics)
        # B has no data, so only A and C should be ranked
        assert result is not None


class TestCompareEntities:
    def test_basic(self):
        rows = [
            {"e": "A", "x": 100},
            {"e": "B", "x": 50},
        ]
        metrics = [{"column": "x", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics)
        comp = compare_entities(result, "A", "B")
        assert comp["winner"] == "A"
        assert comp["score_a"] > comp["score_b"]

    def test_missing_entity(self):
        rows = [{"e": "A", "x": 100}, {"e": "B", "x": 50}]
        metrics = [{"column": "x", "weight": 1.0}]
        result = compute_index(rows, "e", metrics)
        comp = compare_entities(result, "A", "Z")
        assert "error" in comp


class TestFormatIndexTable:
    def test_basic(self):
        rows = [{"e": "A", "x": 100}, {"e": "B", "x": 50}]
        metrics = [{"column": "x", "weight": 1.0, "direction": "higher_is_better"}]
        result = compute_index(rows, "e", metrics, "Test")
        text = format_index_table(result)
        assert "Test" in text
        assert "A" in text
        assert "B" in text
        assert "Mean:" in text
