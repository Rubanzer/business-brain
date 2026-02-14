"""Tests for metric benchmarking module."""

from business_brain.discovery.benchmarking import (
    BenchmarkResult,
    benchmark_groups,
    compare_two_groups,
    rank_entities,
)


# ---------------------------------------------------------------------------
# benchmark_groups
# ---------------------------------------------------------------------------


class TestBenchmarkGroups:
    def test_basic_benchmark(self):
        rows = [
            {"supplier": "A", "cost": 100},
            {"supplier": "A", "cost": 120},
            {"supplier": "B", "cost": 200},
            {"supplier": "B", "cost": 180},
        ]
        result = benchmark_groups(rows, "supplier", "cost")
        assert result is not None
        assert result.best_group == "B"
        assert result.worst_group == "A"
        assert len(result.groups) == 2
        assert result.spread > 0

    def test_empty_rows(self):
        assert benchmark_groups([], "g", "m") is None

    def test_single_group(self):
        rows = [{"g": "A", "m": 10}, {"g": "A", "m": 20}]
        assert benchmark_groups(rows, "g", "m") is None

    def test_three_groups(self):
        rows = [
            {"dept": "HR", "score": 80},
            {"dept": "HR", "score": 90},
            {"dept": "IT", "score": 70},
            {"dept": "IT", "score": 60},
            {"dept": "Finance", "score": 85},
            {"dept": "Finance", "score": 95},
        ]
        result = benchmark_groups(rows, "dept", "score")
        assert result is not None
        assert len(result.groups) == 3
        assert len(result.ranking) == 3
        assert result.ranking[0]["rank"] == 1

    def test_ranking_ordered(self):
        rows = [
            {"g": "A", "m": 10},
            {"g": "B", "m": 50},
            {"g": "C", "m": 30},
        ]
        result = benchmark_groups(rows, "g", "m")
        assert result.ranking[0]["group"] == "B"
        assert result.ranking[1]["group"] == "C"
        assert result.ranking[2]["group"] == "A"

    def test_significant_gaps(self):
        rows = [
            {"g": "A", "m": 100},
            {"g": "B", "m": 500},
        ]
        result = benchmark_groups(rows, "g", "m")
        assert len(result.significant_gaps) >= 1

    def test_no_significant_gaps(self):
        rows = [
            {"g": "A", "m": 100},
            {"g": "B", "m": 105},
        ]
        result = benchmark_groups(rows, "g", "m")
        assert len(result.significant_gaps) == 0

    def test_null_values_skipped(self):
        rows = [
            {"g": "A", "m": 10},
            {"g": "A", "m": None},
            {"g": "B", "m": 20},
        ]
        result = benchmark_groups(rows, "g", "m")
        assert result is not None
        assert result.groups[0].count + result.groups[1].count == 2

    def test_non_numeric_skipped(self):
        rows = [
            {"g": "A", "m": "abc"},
            {"g": "B", "m": "def"},
        ]
        result = benchmark_groups(rows, "g", "m")
        assert result is None

    def test_summary_text(self):
        rows = [
            {"g": "X", "m": 100},
            {"g": "Y", "m": 200},
        ]
        result = benchmark_groups(rows, "g", "m")
        assert "X" in result.summary or "Y" in result.summary

    def test_custom_metric_name(self):
        rows = [
            {"g": "A", "m": 10},
            {"g": "B", "m": 20},
        ]
        result = benchmark_groups(rows, "g", "m", metric_name="Quality Score")
        assert "Quality Score" in result.summary

    def test_pct_of_best(self):
        rows = [
            {"g": "A", "m": 50},
            {"g": "B", "m": 100},
        ]
        result = benchmark_groups(rows, "g", "m")
        best_entry = result.ranking[0]
        assert best_entry["pct_of_best"] == 100.0
        worst_entry = result.ranking[1]
        assert worst_entry["pct_of_best"] == 50.0


# ---------------------------------------------------------------------------
# rank_entities
# ---------------------------------------------------------------------------


class TestRankEntities:
    def test_basic_ranking(self):
        rows = [
            {"name": "Alice", "score": 90},
            {"name": "Bob", "score": 80},
            {"name": "Charlie", "score": 95},
        ]
        result = rank_entities(rows, "name", "score")
        assert result[0]["entity"] == "Charlie"
        assert result[0]["rank"] == 1

    def test_ascending(self):
        rows = [
            {"name": "A", "cost": 100},
            {"name": "B", "cost": 50},
        ]
        result = rank_entities(rows, "name", "cost", ascending=True)
        assert result[0]["entity"] == "B"

    def test_limit(self):
        rows = [{"name": f"E{i}", "score": i} for i in range(20)]
        result = rank_entities(rows, "name", "score", limit=5)
        assert len(result) == 5

    def test_aggregates_multiple_rows(self):
        rows = [
            {"name": "A", "score": 80},
            {"name": "A", "score": 100},
            {"name": "B", "score": 70},
        ]
        result = rank_entities(rows, "name", "score")
        a_entry = next(e for e in result if e["entity"] == "A")
        assert a_entry["value"] == 90.0  # mean of 80 and 100
        assert a_entry["count"] == 2

    def test_empty(self):
        assert rank_entities([], "name", "score") == []


# ---------------------------------------------------------------------------
# compare_two_groups
# ---------------------------------------------------------------------------


class TestCompareTwoGroups:
    def test_basic_comparison(self):
        result = compare_two_groups([10, 20, 30], [40, 50, 60], "Low", "High")
        assert result["group_a"]["mean"] == 20.0
        assert result["group_b"]["mean"] == 50.0
        assert result["difference"] < 0
        assert result["winner"] == "High"

    def test_equal_groups(self):
        result = compare_two_groups([50, 50, 50], [50, 50, 50], "A", "B")
        assert result["difference"] == 0
        assert result["significance"] == "negligible"
        assert result["winner"] == "tie"

    def test_large_difference(self):
        result = compare_two_groups([10, 12, 11], [100, 110, 105], "Small", "Big")
        assert result["significance"] == "large"

    def test_empty_group_a(self):
        result = compare_two_groups([], [1, 2, 3], "A", "B")
        assert "error" in result

    def test_empty_group_b(self):
        result = compare_two_groups([1, 2, 3], [], "A", "B")
        assert "error" in result

    def test_effect_size(self):
        # Two groups with moderate separation
        result = compare_two_groups([10, 15, 12], [20, 25, 22])
        assert result["effect_size"] > 0

    def test_pct_difference(self):
        result = compare_two_groups([100, 100, 100], [50, 50, 50], "A", "B")
        assert result["pct_difference"] == 100.0  # A is 100% more than B
