"""Tests for Pareto analysis module."""

from business_brain.discovery.pareto_analysis import (
    compare_pareto,
    find_concentration_risk,
    pareto_analysis,
)


class TestParetoAnalysis:
    def test_basic_pareto(self):
        rows = [
            {"supplier": "A", "cost": 800},
            {"supplier": "B", "cost": 100},
            {"supplier": "C", "cost": 50},
            {"supplier": "D", "cost": 30},
            {"supplier": "E", "cost": 20},
        ]
        result = pareto_analysis(rows, "supplier", "cost")
        assert result is not None
        assert result.total == 1000
        assert result.items[0].name == "A"
        assert result.items[0].rank == 1
        assert result.is_pareto is True

    def test_empty_rows(self):
        assert pareto_analysis([], "g", "m") is None

    def test_single_group(self):
        rows = [{"g": "A", "m": 100}]
        assert pareto_analysis(rows, "g", "m") is None

    def test_even_distribution(self):
        rows = [
            {"g": "A", "m": 100},
            {"g": "B", "m": 100},
            {"g": "C", "m": 100},
            {"g": "D", "m": 100},
            {"g": "E", "m": 100},
        ]
        result = pareto_analysis(rows, "g", "m")
        assert result is not None
        assert result.is_pareto is False  # even distribution

    def test_cumulative_pct(self):
        rows = [
            {"g": "A", "m": 70},
            {"g": "B", "m": 20},
            {"g": "C", "m": 10},
        ]
        result = pareto_analysis(rows, "g", "m")
        assert result.items[0].cumulative_pct == 70.0
        assert result.items[-1].cumulative_pct == 100.0

    def test_vital_few(self):
        rows = [
            {"g": "A", "m": 800},
            {"g": "B", "m": 100},
            {"g": "C", "m": 50},
            {"g": "D", "m": 30},
            {"g": "E", "m": 20},
        ]
        result = pareto_analysis(rows, "g", "m")
        vital = [i for i in result.items if i.is_vital]
        assert len(vital) >= 1
        assert result.vital_few_contribution > 70

    def test_custom_threshold(self):
        rows = [
            {"g": "A", "m": 50},
            {"g": "B", "m": 30},
            {"g": "C", "m": 20},
        ]
        result = pareto_analysis(rows, "g", "m", threshold=60.0)
        vital = [i for i in result.items if i.is_vital]
        assert vital[0].name == "A"

    def test_aggregates_duplicates(self):
        rows = [
            {"g": "A", "m": 50},
            {"g": "A", "m": 50},
            {"g": "B", "m": 100},
        ]
        result = pareto_analysis(rows, "g", "m")
        assert result.total == 200

    def test_null_values_skipped(self):
        rows = [
            {"g": "A", "m": 100},
            {"g": None, "m": 50},
            {"g": "B", "m": 50},
        ]
        result = pareto_analysis(rows, "g", "m")
        assert result is not None
        assert result.total == 150

    def test_summary_text(self):
        rows = [
            {"g": "A", "m": 80},
            {"g": "B", "m": 20},
        ]
        result = pareto_analysis(rows, "g", "m")
        assert "Pareto" in result.summary

    def test_negative_values_use_abs(self):
        rows = [
            {"g": "A", "m": -100},
            {"g": "B", "m": 200},
        ]
        result = pareto_analysis(rows, "g", "m")
        assert result is not None


class TestFindConcentrationRisk:
    def test_high_concentration(self):
        rows = [
            {"g": "A", "m": 900},
            {"g": "B", "m": 50},
            {"g": "C", "m": 50},
        ]
        result = pareto_analysis(rows, "g", "m")
        risks = find_concentration_risk(result)
        assert len(risks) == 1
        assert risks[0]["name"] == "A"
        assert risks[0]["risk_level"] == "high"

    def test_no_concentration_risk(self):
        rows = [
            {"g": "A", "m": 30},
            {"g": "B", "m": 30},
            {"g": "C", "m": 40},
        ]
        result = pareto_analysis(rows, "g", "m")
        risks = find_concentration_risk(result)
        assert len(risks) == 0


class TestComparePareto:
    def test_basic_comparison(self):
        rows_a = [{"g": "A", "m": 80}, {"g": "B", "m": 20}]
        rows_b = [{"g": "A", "m": 60}, {"g": "B", "m": 40}]
        result_a = pareto_analysis(rows_a, "g", "m")
        result_b = pareto_analysis(rows_b, "g", "m")
        comp = compare_pareto(result_a, result_b)
        assert "total_change_pct" in comp
        assert "concentration_change" in comp
