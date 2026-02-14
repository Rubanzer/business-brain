"""Tests for what-if scenario engine."""

from business_brain.discovery.whatif_engine import (
    Scenario,
    breakeven_analysis,
    compare_scenarios,
    evaluate_scenario,
    sensitivity_table,
)


class TestEvaluateScenario:
    def test_simple_formula(self):
        s = Scenario("base", {"price": 100, "units": 50})
        result = evaluate_scenario(s, "price * units")
        assert result.result == 5000.0
        assert result.scenario_name == "base"

    def test_with_base_values(self):
        s = Scenario("override", {"price": 120})
        result = evaluate_scenario(s, "price * units", {"price": 100, "units": 50})
        assert result.result == 6000.0

    def test_subtraction(self):
        s = Scenario("profit", {"revenue": 1000, "cost": 600})
        result = evaluate_scenario(s, "revenue - cost")
        assert result.result == 400.0

    def test_division(self):
        s = Scenario("ratio", {"a": 100, "b": 50})
        result = evaluate_scenario(s, "a / b")
        assert result.result == 2.0

    def test_complex_formula(self):
        s = Scenario("complex", {"price": 100, "qty": 50, "discount": 10})
        result = evaluate_scenario(s, "(price * qty) - discount")
        assert result.result == 4990.0

    def test_invalid_formula(self):
        s = Scenario("bad", {"x": 10})
        result = evaluate_scenario(s, "import os")
        assert "Error" in result.interpretation

    def test_interpretation(self):
        s = Scenario("test", {"x": 42})
        result = evaluate_scenario(s, "x * 2")
        assert "84" in result.interpretation


class TestCompareScenarios:
    def test_basic_comparison(self):
        scenarios = [
            Scenario("low", {"price": 80}),
            Scenario("mid", {"price": 100}),
            Scenario("high", {"price": 120}),
        ]
        comp = compare_scenarios(scenarios, "price * units", {"price": 100, "units": 50})
        assert comp.best_scenario == "high"
        assert comp.worst_scenario == "low"
        assert comp.range_max > comp.range_min

    def test_empty_scenarios(self):
        comp = compare_scenarios([], "x * y")
        assert comp.best_scenario == ""
        assert "No scenarios" in comp.summary

    def test_single_scenario(self):
        scenarios = [Scenario("only", {"x": 10})]
        comp = compare_scenarios(scenarios, "x * 2")
        assert comp.best_scenario == "only"
        assert comp.worst_scenario == "only"

    def test_sensitivity(self):
        scenarios = [Scenario("base", {"price": 100, "qty": 50})]
        comp = compare_scenarios(scenarios, "price * qty", {"price": 100, "qty": 50})
        # Should have sensitivity data
        assert isinstance(comp.sensitivity, dict)

    def test_summary_text(self):
        scenarios = [
            Scenario("A", {"x": 10}),
            Scenario("B", {"x": 20}),
        ]
        comp = compare_scenarios(scenarios, "x * 5", {"x": 10})
        assert "2 scenarios" in comp.summary


class TestBreakevenAnalysis:
    def test_simple_breakeven(self):
        result = breakeven_analysis(
            "revenue - cost",
            "revenue",
            {"revenue": 0, "cost": 500},
            target=0.0,
            search_range=(0, 1000),
        )
        assert abs(result["breakeven_value"] - 500) < 20
        assert abs(result["achieved_result"]) < 20

    def test_custom_target(self):
        result = breakeven_analysis(
            "price * qty",
            "price",
            {"price": 0, "qty": 10},
            target=1000,
            search_range=(0, 200),
        )
        assert abs(result["breakeven_value"] - 100) < 5


class TestSensitivityTable:
    def test_basic_sensitivity(self):
        table = sensitivity_table("price * qty", "price", {"price": 100, "qty": 50})
        assert len(table) == 7  # default 7 variations
        # Find the 1.0 multiplier (base case)
        base = next(r for r in table if r["multiplier"] == 1.0)
        assert base["result"] == 5000.0
        assert base["change_pct"] == 0.0

    def test_custom_variations(self):
        table = sensitivity_table("x * 2", "x", {"x": 100}, variations=[0.5, 1.0, 1.5])
        assert len(table) == 3

    def test_linear_relationship(self):
        table = sensitivity_table("x * 10", "x", {"x": 100})
        # 10% increase in x should cause 10% increase in result
        up_10 = next(r for r in table if r["multiplier"] == 1.1)
        assert abs(up_10["change_pct"] - 10.0) < 0.1
