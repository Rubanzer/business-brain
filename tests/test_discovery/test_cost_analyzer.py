"""Tests for cost_analyzer module."""

from business_brain.discovery.cost_analyzer import (
    CostBreakdown,
    CostPerUnitResult,
    CostTrendResult,
    BreakevenResult,
    CostVarianceResult,
    cost_breakdown,
    cost_per_unit,
    cost_trend,
    breakeven_analysis,
    cost_variance,
    format_cost_report,
)


# ---------------------------------------------------------------------------
# cost_breakdown
# ---------------------------------------------------------------------------


class TestCostBreakdown:
    def test_basic(self):
        rows = [
            {"cat": "Materials", "amt": 500},
            {"cat": "Labour", "amt": 300},
            {"cat": "Overhead", "amt": 200},
        ]
        result = cost_breakdown(rows, "cat", "amt")
        assert result is not None
        assert result.total_cost == 1000
        assert result.top_category == "Materials"
        assert len(result.categories) == 3
        assert result.categories[0].rank == 1
        assert result.categories[0].share_pct == 50.0

    def test_shares_sum_to_100(self):
        rows = [
            {"cat": "A", "amt": 400},
            {"cat": "B", "amt": 350},
            {"cat": "C", "amt": 150},
            {"cat": "D", "amt": 100},
        ]
        result = cost_breakdown(rows, "cat", "amt")
        total_share = sum(c.share_pct for c in result.categories)
        assert abs(total_share - 100.0) < 0.5

    def test_cumulative_reaches_100(self):
        rows = [{"cat": f"c{i}", "amt": 10} for i in range(5)]
        result = cost_breakdown(rows, "cat", "amt")
        assert abs(result.categories[-1].cumulative_pct - 100.0) < 0.5

    def test_top_3_share(self):
        rows = [
            {"cat": "A", "amt": 500},
            {"cat": "B", "amt": 300},
            {"cat": "C", "amt": 100},
            {"cat": "D", "amt": 50},
            {"cat": "E", "amt": 50},
        ]
        result = cost_breakdown(rows, "cat", "amt")
        assert result.top_3_share_pct == 90.0

    def test_empty_rows(self):
        assert cost_breakdown([], "cat", "amt") is None

    def test_all_none_values(self):
        rows = [{"cat": None, "amt": None}]
        assert cost_breakdown(rows, "cat", "amt") is None

    def test_aggregates_duplicates(self):
        rows = [
            {"cat": "Materials", "amt": 200},
            {"cat": "Materials", "amt": 300},
            {"cat": "Labour", "amt": 500},
        ]
        result = cost_breakdown(rows, "cat", "amt")
        assert result.total_cost == 1000
        mat = [c for c in result.categories if c.name == "Materials"][0]
        assert mat.amount == 500

    def test_with_total_column(self):
        rows = [
            {"cat": "Materials", "amt": 400, "total": 1000},
            {"cat": "Labour", "amt": 300, "total": 1000},
        ]
        result = cost_breakdown(rows, "cat", "amt", total_column="total")
        assert result is not None
        # Share is computed against total_column sum (2000)
        mat = [c for c in result.categories if c.name == "Materials"][0]
        assert mat.share_pct == 20.0  # 400 / 2000

    def test_zero_costs(self):
        rows = [
            {"cat": "A", "amt": 0},
            {"cat": "B", "amt": 0},
        ]
        assert cost_breakdown(rows, "cat", "amt") is None

    def test_single_category(self):
        rows = [{"cat": "Only", "amt": 100}]
        result = cost_breakdown(rows, "cat", "amt")
        assert result is not None
        assert result.top_category == "Only"
        assert result.categories[0].share_pct == 100.0

    def test_summary_present(self):
        rows = [{"cat": "A", "amt": 100}]
        result = cost_breakdown(rows, "cat", "amt")
        assert "Cost breakdown" in result.summary

    def test_invalid_amount_skipped(self):
        rows = [
            {"cat": "A", "amt": "invalid"},
            {"cat": "B", "amt": 100},
        ]
        result = cost_breakdown(rows, "cat", "amt")
        assert result is not None
        assert result.total_cost == 100


# ---------------------------------------------------------------------------
# cost_per_unit
# ---------------------------------------------------------------------------


class TestCostPerUnit:
    def test_basic(self):
        rows = [
            {"entity": "Plant A", "cost": 1000, "qty": 100},
            {"entity": "Plant B", "cost": 1500, "qty": 100},
        ]
        result = cost_per_unit(rows, "entity", "cost", "qty")
        assert result is not None
        assert result.best_entity == "Plant A"
        assert result.worst_entity == "Plant B"
        assert result.mean_cpu == 12.5

    def test_median_cpu(self):
        rows = [
            {"entity": "A", "cost": 100, "qty": 10},
            {"entity": "B", "cost": 200, "qty": 10},
            {"entity": "C", "cost": 300, "qty": 10},
        ]
        result = cost_per_unit(rows, "entity", "cost", "qty")
        assert result.median_cpu == 20.0

    def test_spread_pct(self):
        rows = [
            {"entity": "A", "cost": 100, "qty": 10},  # CPU = 10
            {"entity": "B", "cost": 200, "qty": 10},  # CPU = 20
        ]
        result = cost_per_unit(rows, "entity", "cost", "qty")
        # mean = 15, spread = (20-10)/15 * 100 = 66.67
        assert abs(result.spread_pct - 66.67) < 0.1

    def test_deviation_from_mean(self):
        rows = [
            {"entity": "A", "cost": 100, "qty": 10},  # CPU = 10
            {"entity": "B", "cost": 200, "qty": 10},  # CPU = 20
        ]
        result = cost_per_unit(rows, "entity", "cost", "qty")
        a_ent = [e for e in result.entities if e.entity == "A"][0]
        # mean = 15, deviation = (10 - 15) / 15 * 100 = -33.33
        assert abs(a_ent.deviation_from_mean_pct - (-33.33)) < 0.1

    def test_empty_rows(self):
        assert cost_per_unit([], "entity", "cost", "qty") is None

    def test_zero_quantity(self):
        rows = [
            {"entity": "A", "cost": 100, "qty": 0},
            {"entity": "B", "cost": 100, "qty": 10},
        ]
        result = cost_per_unit(rows, "entity", "cost", "qty")
        assert result is not None
        # Entity A has inf CPU, B is best and worst among finite
        assert result.best_entity == "B"

    def test_single_entity(self):
        rows = [{"entity": "Solo", "cost": 500, "qty": 50}]
        result = cost_per_unit(rows, "entity", "cost", "qty")
        assert result is not None
        assert result.best_entity == "Solo"
        assert result.worst_entity == "Solo"
        assert result.spread_pct == 0.0

    def test_aggregates_rows(self):
        rows = [
            {"entity": "A", "cost": 100, "qty": 10},
            {"entity": "A", "cost": 100, "qty": 10},
        ]
        result = cost_per_unit(rows, "entity", "cost", "qty")
        assert result.entities[0].total_cost == 200
        assert result.entities[0].total_quantity == 20
        assert result.entities[0].cost_per_unit == 10.0

    def test_none_values_skipped(self):
        rows = [
            {"entity": "A", "cost": None, "qty": 10},
            {"entity": "B", "cost": 100, "qty": 10},
        ]
        result = cost_per_unit(rows, "entity", "cost", "qty")
        assert result is not None
        assert len(result.entities) == 1

    def test_all_zero_quantity_returns_none(self):
        rows = [
            {"entity": "A", "cost": 100, "qty": 0},
            {"entity": "B", "cost": 200, "qty": 0},
        ]
        # All CPUs are inf => no finite CPUs => None
        assert cost_per_unit(rows, "entity", "cost", "qty") is None


# ---------------------------------------------------------------------------
# cost_trend
# ---------------------------------------------------------------------------


class TestCostTrend:
    def test_increasing_trend(self):
        rows = [
            {"period": "2024-Q1", "cost": 100},
            {"period": "2024-Q2", "cost": 120},
            {"period": "2024-Q3", "cost": 150},
        ]
        result = cost_trend(rows, "period", "cost")
        assert result is not None
        assert result.trend_direction == "increasing"
        assert result.total_change_pct == 50.0

    def test_decreasing_trend(self):
        rows = [
            {"period": "2024-Q1", "cost": 200},
            {"period": "2024-Q2", "cost": 150},
            {"period": "2024-Q3", "cost": 100},
        ]
        result = cost_trend(rows, "period", "cost")
        assert result.trend_direction == "decreasing"
        assert result.total_change_pct == -50.0

    def test_stable_trend(self):
        rows = [
            {"period": "2024-Q1", "cost": 100},
            {"period": "2024-Q2", "cost": 101},
            {"period": "2024-Q3", "cost": 100},
        ]
        result = cost_trend(rows, "period", "cost")
        assert result.trend_direction == "stable"

    def test_period_change_pct(self):
        rows = [
            {"period": "Q1", "cost": 100},
            {"period": "Q2", "cost": 110},
        ]
        result = cost_trend(rows, "period", "cost")
        assert result.periods[1].change_pct == 10.0

    def test_first_period_zero_change(self):
        rows = [
            {"period": "Q1", "cost": 100},
            {"period": "Q2", "cost": 200},
        ]
        result = cost_trend(rows, "period", "cost")
        assert result.periods[0].change_from_prev == 0.0
        assert result.periods[0].change_pct == 0.0

    def test_empty_rows(self):
        assert cost_trend([], "period", "cost") is None

    def test_single_period(self):
        rows = [{"period": "Q1", "cost": 100}]
        assert cost_trend(rows, "period", "cost") is None

    def test_volatility(self):
        rows = [
            {"period": "Q1", "cost": 100},
            {"period": "Q2", "cost": 200},
            {"period": "Q3", "cost": 100},
            {"period": "Q4", "cost": 200},
        ]
        result = cost_trend(rows, "period", "cost")
        assert result.volatility > 0

    def test_aggregates_same_period(self):
        rows = [
            {"period": "Q1", "cost": 50},
            {"period": "Q1", "cost": 50},
            {"period": "Q2", "cost": 120},
        ]
        result = cost_trend(rows, "period", "cost")
        assert result.periods[0].total_cost == 100
        assert result.periods[1].total_cost == 120

    def test_summary_present(self):
        rows = [
            {"period": "Q1", "cost": 100},
            {"period": "Q2", "cost": 110},
        ]
        result = cost_trend(rows, "period", "cost")
        assert "Cost trend" in result.summary


# ---------------------------------------------------------------------------
# breakeven_analysis
# ---------------------------------------------------------------------------


class TestBreakevenAnalysis:
    def test_basic(self):
        # FC=10000, VC=5, P=15 => BEP = 10000/10 = 1000
        result = breakeven_analysis(10000, 5, 15)
        assert result is not None
        assert result.breakeven_units == 1000.0
        assert result.breakeven_revenue == 15000.0
        assert result.contribution_margin == 10.0

    def test_contribution_margin_ratio(self):
        result = breakeven_analysis(10000, 5, 20)
        # CM = 15, ratio = 15/20 = 0.75
        assert result.contribution_margin_ratio == 0.75

    def test_zero_price(self):
        # Price = 0 => CM = 0 - VC which is negative => None
        result = breakeven_analysis(10000, 5, 0)
        assert result is None

    def test_price_equals_variable_cost(self):
        # CM = 0 => division by zero => None
        result = breakeven_analysis(10000, 10, 10)
        assert result is None

    def test_price_less_than_variable_cost(self):
        # CM negative => None
        result = breakeven_analysis(10000, 15, 10)
        assert result is None

    def test_zero_fixed_costs(self):
        result = breakeven_analysis(0, 5, 15)
        assert result is not None
        assert result.breakeven_units == 0.0
        assert result.breakeven_revenue == 0.0

    def test_summary_present(self):
        result = breakeven_analysis(10000, 5, 15)
        assert "Breakeven" in result.summary

    def test_large_fixed_costs(self):
        result = breakeven_analysis(1_000_000, 2, 10)
        assert result is not None
        assert result.breakeven_units == 125000.0


# ---------------------------------------------------------------------------
# cost_variance
# ---------------------------------------------------------------------------


class TestCostVariance:
    def test_basic(self):
        rows = [
            {"dept": "Mfg", "actual": 1000, "budget": 1200},
            {"dept": "R&D", "actual": 500, "budget": 400},
        ]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert result is not None
        assert result.total_actual == 1500
        assert result.total_budget == 1600

    def test_favorable_status(self):
        rows = [{"dept": "A", "actual": 80, "budget": 100}]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert result.entities[0].status == "favorable"
        assert result.favorable_count == 1

    def test_unfavorable_status(self):
        rows = [{"dept": "A", "actual": 120, "budget": 100}]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert result.entities[0].status == "unfavorable"
        assert result.unfavorable_count == 1

    def test_on_budget_within_2_pct(self):
        rows = [{"dept": "A", "actual": 101, "budget": 100}]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert result.entities[0].status == "on_budget"

    def test_empty_rows(self):
        assert cost_variance([], "dept", "actual", "budget") is None

    def test_total_variance_pct(self):
        rows = [
            {"dept": "A", "actual": 110, "budget": 100},
            {"dept": "B", "actual": 90, "budget": 100},
        ]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert result.total_variance == 0.0
        assert result.total_variance_pct == 0.0

    def test_aggregates_rows(self):
        rows = [
            {"dept": "A", "actual": 50, "budget": 60},
            {"dept": "A", "actual": 50, "budget": 60},
        ]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert result.entities[0].actual == 100
        assert result.entities[0].budget == 120

    def test_none_values_skipped(self):
        rows = [
            {"dept": "A", "actual": None, "budget": 100},
            {"dept": "B", "actual": 90, "budget": 100},
        ]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert len(result.entities) == 1

    def test_sorted_by_absolute_variance(self):
        rows = [
            {"dept": "Small", "actual": 101, "budget": 100},
            {"dept": "Big", "actual": 200, "budget": 100},
        ]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert result.entities[0].entity == "Big"

    def test_summary_present(self):
        rows = [{"dept": "A", "actual": 100, "budget": 100}]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert "Cost variance" in result.summary

    def test_zero_budget(self):
        rows = [{"dept": "A", "actual": 100, "budget": 0}]
        result = cost_variance(rows, "dept", "actual", "budget")
        assert result is not None
        # variance_pct is 0 when budget is 0
        assert result.entities[0].variance_pct == 0.0


# ---------------------------------------------------------------------------
# format_cost_report
# ---------------------------------------------------------------------------


class TestFormatCostReport:
    def test_no_data(self):
        report = format_cost_report()
        assert "No analysis data provided" in report

    def test_with_breakdown(self):
        rows = [
            {"cat": "A", "amt": 700},
            {"cat": "B", "amt": 300},
        ]
        bd = cost_breakdown(rows, "cat", "amt")
        report = format_cost_report(breakdown=bd)
        assert "Cost Breakdown" in report
        assert "A" in report

    def test_with_all_sections(self):
        bd_rows = [{"cat": "A", "amt": 100}]
        cpu_rows = [{"e": "X", "c": 100, "q": 10}]
        trend_rows = [
            {"t": "Q1", "c": 100},
            {"t": "Q2", "c": 120},
        ]
        var_rows = [{"e": "X", "a": 100, "b": 110}]

        bd = cost_breakdown(bd_rows, "cat", "amt")
        cp = cost_per_unit(cpu_rows, "e", "c", "q")
        tr = cost_trend(trend_rows, "t", "c")
        va = cost_variance(var_rows, "e", "a", "b")

        report = format_cost_report(breakdown=bd, cpu=cp, trend=tr, variance=va)
        assert "Cost Analysis Report" in report
        assert "Cost Breakdown" in report
        assert "Cost Per Unit" in report
        assert "Cost Trend" in report
        assert "Cost Variance" in report

    def test_report_includes_title(self):
        report = format_cost_report()
        assert "Cost Analysis Report" in report
