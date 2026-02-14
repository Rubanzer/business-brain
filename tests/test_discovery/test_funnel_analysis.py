"""Tests for funnel analysis module."""

from business_brain.discovery.funnel_analysis import (
    FunnelResult,
    analyze_funnel,
    compare_funnels,
    format_funnel_text,
    funnel_from_rows,
)


class TestAnalyzeFunnel:
    def test_basic_funnel(self):
        stages = [("Visits", 1000), ("Signups", 200), ("Purchases", 50)]
        result = analyze_funnel(stages)
        assert result is not None
        assert result.initial_count == 1000
        assert result.final_count == 50
        assert result.overall_conversion == 5.0

    def test_returns_none_for_empty(self):
        assert analyze_funnel([]) is None

    def test_returns_none_for_single_stage(self):
        assert analyze_funnel([("A", 100)]) is None

    def test_conversion_rates(self):
        stages = [("A", 100), ("B", 50), ("C", 25)]
        result = analyze_funnel(stages)
        assert result.stages[0].conversion_rate == 100.0
        assert result.stages[1].conversion_rate == 50.0
        assert result.stages[2].conversion_rate == 50.0

    def test_drop_off(self):
        stages = [("A", 100), ("B", 70), ("C", 30)]
        result = analyze_funnel(stages)
        assert result.stages[1].drop_off == 30
        assert result.stages[2].drop_off == 40

    def test_no_drop_off_first_stage(self):
        stages = [("A", 100), ("B", 80)]
        result = analyze_funnel(stages)
        assert result.stages[0].drop_off == 0
        assert result.stages[0].drop_off_pct == 0.0

    def test_biggest_drop(self):
        stages = [("A", 100), ("B", 90), ("C", 20)]
        result = analyze_funnel(stages)
        assert result.biggest_drop_stage == "C"

    def test_perfect_funnel(self):
        stages = [("A", 100), ("B", 100), ("C", 100)]
        result = analyze_funnel(stages)
        assert result.overall_conversion == 100.0
        assert all(s.drop_off == 0 for s in result.stages)

    def test_zero_final(self):
        stages = [("A", 100), ("B", 50), ("C", 0)]
        result = analyze_funnel(stages)
        assert result.overall_conversion == 0.0
        assert result.final_count == 0

    def test_pct_of_total(self):
        stages = [("A", 200), ("B", 100), ("C", 50)]
        result = analyze_funnel(stages)
        assert result.stages[0].pct_of_total == 100.0
        assert result.stages[1].pct_of_total == 50.0
        assert result.stages[2].pct_of_total == 25.0

    def test_summary(self):
        stages = [("A", 100), ("B", 80)]
        result = analyze_funnel(stages)
        assert "100" in result.summary
        assert "80" in result.summary

    def test_zero_initial_returns_none(self):
        assert analyze_funnel([("A", 0), ("B", 0)]) is None

    def test_total_stages(self):
        stages = [("A", 100), ("B", 80), ("C", 60), ("D", 40)]
        result = analyze_funnel(stages)
        assert result.total_stages == 4


class TestFunnelFromRows:
    def test_basic(self):
        rows = [
            {"customer": "C1", "stage": "visit"},
            {"customer": "C1", "stage": "signup"},
            {"customer": "C1", "stage": "purchase"},
            {"customer": "C2", "stage": "visit"},
            {"customer": "C2", "stage": "signup"},
            {"customer": "C3", "stage": "visit"},
        ]
        result = funnel_from_rows(rows, "customer", "stage", ["visit", "signup", "purchase"])
        assert result is not None
        assert result.initial_count == 3
        assert result.stages[1].count == 2
        assert result.stages[2].count == 1

    def test_returns_none_for_empty(self):
        assert funnel_from_rows([], "c", "s", ["a", "b"]) is None

    def test_returns_none_for_single_stage(self):
        assert funnel_from_rows([{"c": 1, "s": "a"}], "c", "s", ["a"]) is None

    def test_ignores_unknown_stages(self):
        rows = [
            {"c": "C1", "s": "visit"},
            {"c": "C1", "s": "unknown_stage"},
            {"c": "C2", "s": "visit"},
        ]
        result = funnel_from_rows(rows, "c", "s", ["visit", "signup"])
        assert result is not None
        assert result.stages[0].count == 2

    def test_highest_stage_used(self):
        rows = [
            {"c": "C1", "s": "visit"},
            {"c": "C1", "s": "signup"},
            {"c": "C1", "s": "visit"},  # duplicate lower stage
        ]
        result = funnel_from_rows(rows, "c", "s", ["visit", "signup", "purchase"])
        assert result.stages[0].count == 1  # 1 entity
        assert result.stages[1].count == 1  # entity reached signup


class TestCompareFunnels:
    def test_basic_comparison(self):
        a = analyze_funnel([("Visit", 100), ("Buy", 10)])
        b = analyze_funnel([("Visit", 100), ("Buy", 20)])
        comp = compare_funnels(a, b)
        assert comp["improved"] is True
        assert comp["conversion_diff"] == 10.0

    def test_degraded(self):
        a = analyze_funnel([("A", 100), ("B", 50)])
        b = analyze_funnel([("A", 100), ("B", 30)])
        comp = compare_funnels(a, b)
        assert comp["improved"] is False

    def test_stage_comparison(self):
        a = analyze_funnel([("A", 100), ("B", 50), ("C", 25)])
        b = analyze_funnel([("A", 100), ("B", 60), ("C", 30)])
        comp = compare_funnels(a, b)
        assert len(comp["stage_comparison"]) == 3


class TestFormatFunnelText:
    def test_basic_format(self):
        result = analyze_funnel([("Visit", 100), ("Signup", 50), ("Buy", 10)])
        text = format_funnel_text(result)
        assert "Funnel Analysis" in text
        assert "Visit" in text
        assert "Signup" in text
        assert "Buy" in text
        assert "conversion" in text.lower()

    def test_drop_indicators(self):
        result = analyze_funnel([("A", 100), ("B", 50)])
        text = format_funnel_text(result)
        assert "dropped" in text
