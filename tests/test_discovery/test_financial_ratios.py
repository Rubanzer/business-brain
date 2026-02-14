"""Tests for financial_ratios module."""

from business_brain.discovery.financial_ratios import (
    EntityLiquidity,
    EntityProfit,
    EntityTrend,
    PeriodMetric,
    ProfitabilityResult,
    LiquidityResult,
    EfficiencyResult,
    EfficiencyRatios,
    FinancialTrendResult,
    _safe_float,
    compute_profitability_ratios,
    compute_liquidity_ratios,
    compute_efficiency_ratios,
    analyze_financial_trends,
    format_financial_report,
)


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("123.45") == 123.45

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_negative(self):
        assert _safe_float(-5.5) == -5.5


# ---------------------------------------------------------------------------
# compute_profitability_ratios
# ---------------------------------------------------------------------------


class TestComputeProfitabilityRatios:
    def test_empty_rows(self):
        assert compute_profitability_ratios([], "rev", "cost") is None

    def test_no_valid_data(self):
        rows = [{"rev": "abc", "cost": "def"}]
        assert compute_profitability_ratios(rows, "rev", "cost") is None

    def test_missing_columns(self):
        rows = [{"x": 100}]
        assert compute_profitability_ratios(rows, "rev", "cost") is None

    def test_single_row_no_entity(self):
        rows = [{"rev": 1000, "cost": 600}]
        result = compute_profitability_ratios(rows, "rev", "cost")
        assert result is not None
        assert len(result.entities) == 1
        assert result.entities[0].entity == "Overall"
        assert result.entities[0].revenue == 1000.0
        assert result.entities[0].cost == 600.0
        assert result.entities[0].gross_profit == 400.0
        assert result.entities[0].gross_margin_pct == 40.0
        assert result.entities[0].cost_ratio == 60.0
        assert result.overall_margin == 40.0

    def test_multiple_rows_no_entity(self):
        rows = [
            {"rev": 1000, "cost": 600},
            {"rev": 2000, "cost": 1000},
        ]
        result = compute_profitability_ratios(rows, "rev", "cost")
        assert result is not None
        assert result.total_revenue == 3000.0
        assert result.total_cost == 1600.0
        # margin = (3000-1600)/3000 * 100 = 46.67
        assert result.overall_margin == 46.67

    def test_with_entity_column(self):
        rows = [
            {"entity": "A", "rev": 1000, "cost": 400},
            {"entity": "A", "rev": 500, "cost": 200},
            {"entity": "B", "rev": 2000, "cost": 1500},
        ]
        result = compute_profitability_ratios(rows, "rev", "cost", entity_column="entity")
        assert result is not None
        assert len(result.entities) == 2

        entity_a = next(e for e in result.entities if e.entity == "A")
        assert entity_a.revenue == 1500.0
        assert entity_a.cost == 600.0
        assert entity_a.gross_profit == 900.0
        assert entity_a.gross_margin_pct == 60.0

        entity_b = next(e for e in result.entities if e.entity == "B")
        assert entity_b.revenue == 2000.0
        assert entity_b.cost == 1500.0
        assert entity_b.gross_margin_pct == 25.0

    def test_zero_revenue(self):
        rows = [{"rev": 0, "cost": 100}]
        result = compute_profitability_ratios(rows, "rev", "cost")
        assert result is not None
        assert result.entities[0].gross_margin_pct == 0.0
        assert result.entities[0].cost_ratio == 0.0
        assert result.overall_margin == 0.0

    def test_string_numeric_values(self):
        rows = [{"rev": "1000", "cost": "400"}]
        result = compute_profitability_ratios(rows, "rev", "cost")
        assert result is not None
        assert result.entities[0].revenue == 1000.0
        assert result.entities[0].cost == 400.0

    def test_mixed_valid_invalid(self):
        rows = [
            {"rev": 1000, "cost": 400},
            {"rev": "bad", "cost": 200},
            {"rev": 500, "cost": 100},
        ]
        result = compute_profitability_ratios(rows, "rev", "cost")
        assert result is not None
        assert result.total_revenue == 1500.0
        assert result.total_cost == 500.0

    def test_summary_contains_margin(self):
        rows = [{"rev": 1000, "cost": 600}]
        result = compute_profitability_ratios(rows, "rev", "cost")
        assert "40.0%" in result.summary

    def test_negative_margin(self):
        rows = [{"rev": 100, "cost": 200}]
        result = compute_profitability_ratios(rows, "rev", "cost")
        assert result is not None
        assert result.entities[0].gross_profit == -100.0
        assert result.entities[0].gross_margin_pct == -100.0
        assert result.overall_margin == -100.0

    def test_multiple_entities_best_margin_in_summary(self):
        rows = [
            {"e": "Low", "rev": 100, "cost": 90},
            {"e": "High", "rev": 100, "cost": 10},
        ]
        result = compute_profitability_ratios(rows, "rev", "cost", entity_column="e")
        assert "High" in result.summary


# ---------------------------------------------------------------------------
# compute_liquidity_ratios
# ---------------------------------------------------------------------------


class TestComputeLiquidityRatios:
    def test_empty_rows(self):
        assert compute_liquidity_ratios([], "ca", "cl") is None

    def test_no_valid_data(self):
        rows = [{"ca": "abc", "cl": "def"}]
        assert compute_liquidity_ratios(rows, "ca", "cl") is None

    def test_basic_current_ratio(self):
        rows = [{"ca": 2000, "cl": 1000}]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        assert result is not None
        assert len(result.entities) == 1
        assert result.entities[0].current_ratio == 2.0
        assert result.entities[0].rating == "Adequate"  # > 1, not > 2

    def test_strong_rating(self):
        rows = [{"ca": 3000, "cl": 1000}]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        assert result.entities[0].current_ratio == 3.0
        assert result.entities[0].rating == "Strong"

    def test_weak_rating(self):
        rows = [{"ca": 500, "cl": 1000}]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        assert result.entities[0].current_ratio == 0.5
        assert result.entities[0].rating == "Weak"

    def test_exactly_two_adequate(self):
        rows = [{"ca": 2000, "cl": 1000}]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        # current_ratio == 2.0, which is NOT > 2, so "Adequate"
        assert result.entities[0].rating == "Adequate"

    def test_exactly_one_weak(self):
        rows = [{"ca": 1000, "cl": 1000}]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        # current_ratio == 1.0, which is NOT > 1, so "Weak"
        assert result.entities[0].rating == "Weak"

    def test_with_cash_column(self):
        rows = [{"ca": 2000, "cl": 1000, "cash": 500}]
        result = compute_liquidity_ratios(rows, "ca", "cl", cash_column="cash")
        assert result is not None
        assert result.entities[0].cash_ratio == 0.5

    def test_with_inventory_column(self):
        rows = [{"ca": 2000, "cl": 1000, "inv": 300}]
        result = compute_liquidity_ratios(rows, "ca", "cl", inventory_column="inv")
        assert result is not None
        # quick_ratio = (2000 - 300) / 1000 = 1.7
        assert result.entities[0].quick_ratio == 1.7

    def test_no_optional_columns(self):
        rows = [{"ca": 2000, "cl": 1000}]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        assert result.entities[0].quick_ratio is None
        assert result.entities[0].cash_ratio is None

    def test_zero_liabilities(self):
        rows = [{"ca": 2000, "cl": 0}]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        assert result is not None
        assert result.entities[0].current_ratio == 0.0

    def test_zero_liabilities_with_optional_columns(self):
        rows = [{"ca": 2000, "cl": 0, "cash": 500, "inv": 300}]
        result = compute_liquidity_ratios(
            rows, "ca", "cl", cash_column="cash", inventory_column="inv"
        )
        assert result.entities[0].quick_ratio == 0.0
        assert result.entities[0].cash_ratio == 0.0

    def test_multiple_entities(self):
        rows = [
            {"e": "X", "ca": 3000, "cl": 1000},
            {"e": "Y", "ca": 500, "cl": 1000},
        ]
        result = compute_liquidity_ratios(rows, "ca", "cl", entity_column="e")
        assert result is not None
        assert len(result.entities) == 2
        x = next(e for e in result.entities if e.entity == "X")
        y = next(e for e in result.entities if e.entity == "Y")
        assert x.rating == "Strong"
        assert y.rating == "Weak"

    def test_avg_current_ratio(self):
        rows = [
            {"e": "X", "ca": 3000, "cl": 1000},  # 3.0
            {"e": "Y", "ca": 1000, "cl": 1000},   # 1.0
        ]
        result = compute_liquidity_ratios(rows, "ca", "cl", entity_column="e")
        assert result.avg_current_ratio == 2.0

    def test_summary_contains_ratings(self):
        rows = [{"ca": 3000, "cl": 1000}]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        assert "Strong" in result.summary

    def test_multiple_rows_same_entity(self):
        rows = [
            {"ca": 1000, "cl": 500},
            {"ca": 1500, "cl": 500},
        ]
        result = compute_liquidity_ratios(rows, "ca", "cl")
        # totals: ca=2500, cl=1000, ratio=2.5
        assert result.entities[0].current_ratio == 2.5
        assert result.entities[0].rating == "Strong"


# ---------------------------------------------------------------------------
# compute_efficiency_ratios
# ---------------------------------------------------------------------------


class TestComputeEfficiencyRatios:
    def test_empty_rows(self):
        assert compute_efficiency_ratios([], "rev", "assets") is None

    def test_no_valid_data(self):
        rows = [{"rev": "bad", "assets": "bad"}]
        assert compute_efficiency_ratios(rows, "rev", "assets") is None

    def test_zero_assets(self):
        rows = [{"rev": 1000, "assets": 0}]
        assert compute_efficiency_ratios(rows, "rev", "assets") is None

    def test_basic_asset_turnover(self):
        rows = [{"rev": 5000, "assets": 2500}]
        result = compute_efficiency_ratios(rows, "rev", "assets")
        assert result is not None
        assert result.ratios.asset_turnover == 2.0

    def test_no_optional_columns(self):
        rows = [{"rev": 1000, "assets": 500}]
        result = compute_efficiency_ratios(rows, "rev", "assets")
        assert result.ratios.receivables_turnover is None
        assert result.ratios.dso is None
        assert result.ratios.payables_turnover is None
        assert result.ratios.dpo is None
        assert result.ratios.cash_conversion_cycle is None

    def test_with_receivables(self):
        rows = [{"rev": 3650, "assets": 1000, "rec": 365}]
        result = compute_efficiency_ratios(
            rows, "rev", "assets", receivables_column="rec"
        )
        assert result is not None
        assert result.ratios.receivables_turnover == 10.0
        assert result.ratios.dso == 36.5

    def test_with_payables_and_cogs(self):
        rows = [{"rev": 1000, "assets": 500, "pay": 365, "cogs": 730}]
        result = compute_efficiency_ratios(
            rows, "rev", "assets", payables_column="pay", cogs_column="cogs"
        )
        assert result is not None
        assert result.ratios.payables_turnover == 2.0
        assert result.ratios.dpo == 182.5

    def test_cash_conversion_cycle(self):
        rows = [{"rev": 3650, "assets": 1000, "rec": 365, "pay": 365, "cogs": 730}]
        result = compute_efficiency_ratios(
            rows, "rev", "assets",
            receivables_column="rec",
            payables_column="pay",
            cogs_column="cogs",
        )
        assert result is not None
        # DSO = 365/10 = 36.5, DPO = 365/2 = 182.5
        # CCC = 36.5 - 182.5 = -146.0
        assert result.ratios.cash_conversion_cycle == -146.0

    def test_zero_receivables(self):
        rows = [{"rev": 1000, "assets": 500, "rec": 0}]
        result = compute_efficiency_ratios(
            rows, "rev", "assets", receivables_column="rec"
        )
        assert result.ratios.receivables_turnover is None
        assert result.ratios.dso is None

    def test_zero_payables(self):
        rows = [{"rev": 1000, "assets": 500, "pay": 0, "cogs": 500}]
        result = compute_efficiency_ratios(
            rows, "rev", "assets", payables_column="pay", cogs_column="cogs"
        )
        assert result.ratios.payables_turnover is None
        assert result.ratios.dpo is None

    def test_payables_without_cogs(self):
        rows = [{"rev": 1000, "assets": 500, "pay": 100}]
        result = compute_efficiency_ratios(
            rows, "rev", "assets", payables_column="pay"
        )
        # cogs_column is None, so has_payables is False
        assert result.ratios.payables_turnover is None

    def test_summary_contains_asset_turnover(self):
        rows = [{"rev": 1000, "assets": 500}]
        result = compute_efficiency_ratios(rows, "rev", "assets")
        assert "Asset turnover" in result.summary

    def test_summary_contains_dso(self):
        rows = [{"rev": 3650, "assets": 1000, "rec": 365}]
        result = compute_efficiency_ratios(
            rows, "rev", "assets", receivables_column="rec"
        )
        assert "DSO" in result.summary

    def test_multiple_rows(self):
        rows = [
            {"rev": 500, "assets": 250},
            {"rev": 500, "assets": 250},
        ]
        result = compute_efficiency_ratios(rows, "rev", "assets")
        # total_rev=1000, total_assets=500 -> 2.0
        assert result.ratios.asset_turnover == 2.0


# ---------------------------------------------------------------------------
# analyze_financial_trends
# ---------------------------------------------------------------------------


class TestAnalyzeFinancialTrends:
    def test_empty_rows(self):
        assert analyze_financial_trends([], "metric", "period") is None

    def test_no_valid_data(self):
        rows = [{"metric": "bad", "period": "Q1"}]
        assert analyze_financial_trends(rows, "metric", "period") is None

    def test_single_period(self):
        rows = [
            {"metric": 100, "period": "Q1"},
            {"metric": 200, "period": "Q1"},
        ]
        # Only 1 period => entity filtered out => None
        assert analyze_financial_trends(rows, "metric", "period") is None

    def test_two_periods_improving(self):
        rows = [
            {"metric": 100, "period": "Q1"},
            {"metric": 150, "period": "Q2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        assert result is not None
        assert len(result.entities) == 1
        et = result.entities[0]
        assert et.trend == "improving"
        assert et.best_period == "Q2"
        assert et.worst_period == "Q1"
        assert et.cagr is None  # < 3 periods

    def test_two_periods_declining(self):
        rows = [
            {"metric": 200, "period": "Q1"},
            {"metric": 100, "period": "Q2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        assert result.entities[0].trend == "declining"

    def test_three_periods_with_cagr(self):
        rows = [
            {"metric": 100, "period": "P1"},
            {"metric": 110, "period": "P2"},
            {"metric": 121, "period": "P3"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        assert result is not None
        et = result.entities[0]
        assert et.cagr is not None
        # CAGR = ((121/100)^(1/2) - 1)*100 = 10%
        assert et.cagr == 10.0

    def test_cagr_not_computed_for_two_periods(self):
        rows = [
            {"metric": 100, "period": "P1"},
            {"metric": 200, "period": "P2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        assert result.entities[0].cagr is None

    def test_cagr_negative_start_value(self):
        rows = [
            {"metric": -100, "period": "P1"},
            {"metric": 50, "period": "P2"},
            {"metric": 200, "period": "P3"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        # first_val < 0, so CAGR should be None
        assert result.entities[0].cagr is None

    def test_period_over_period_change(self):
        rows = [
            {"metric": 100, "period": "Q1"},
            {"metric": 120, "period": "Q2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        periods = result.entities[0].periods
        assert periods[0].change is None
        assert periods[1].change == 20.0
        assert periods[1].change_pct == 20.0

    def test_change_pct_zero_prev(self):
        rows = [
            {"metric": 0, "period": "Q1"},
            {"metric": 100, "period": "Q2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        periods = result.entities[0].periods
        # prev is 0, so change_pct should be None
        assert periods[1].change == 100.0
        assert periods[1].change_pct is None

    def test_with_entity_column(self):
        rows = [
            {"e": "A", "metric": 100, "period": "Q1"},
            {"e": "A", "metric": 200, "period": "Q2"},
            {"e": "B", "metric": 300, "period": "Q1"},
            {"e": "B", "metric": 250, "period": "Q2"},
        ]
        result = analyze_financial_trends(
            rows, "metric", "period", entity_column="e"
        )
        assert result is not None
        assert len(result.entities) == 2
        a = next(e for e in result.entities if e.entity == "A")
        b = next(e for e in result.entities if e.entity == "B")
        assert a.trend == "improving"
        assert b.trend == "declining"

    def test_stable_trend(self):
        rows = [
            {"metric": 100, "period": "Q1"},
            {"metric": 110, "period": "Q2"},
            {"metric": 100, "period": "Q3"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        et = result.entities[0]
        # changes: +10, -10 => 1 positive, 1 negative => stable
        assert et.trend == "stable"

    def test_overall_trend(self):
        rows = [
            {"e": "A", "metric": 100, "period": "Q1"},
            {"e": "A", "metric": 200, "period": "Q2"},
            {"e": "B", "metric": 100, "period": "Q1"},
            {"e": "B", "metric": 200, "period": "Q2"},
            {"e": "C", "metric": 200, "period": "Q1"},
            {"e": "C", "metric": 100, "period": "Q2"},
        ]
        result = analyze_financial_trends(
            rows, "metric", "period", entity_column="e"
        )
        # 2 improving, 1 declining => overall improving
        assert result.overall_trend == "improving"

    def test_aggregation_within_period(self):
        rows = [
            {"metric": 50, "period": "Q1"},
            {"metric": 50, "period": "Q1"},
            {"metric": 200, "period": "Q2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        periods = result.entities[0].periods
        # Q1 aggregated to 100
        assert periods[0].value == 100.0

    def test_summary_content(self):
        rows = [
            {"metric": 100, "period": "Q1"},
            {"metric": 200, "period": "Q2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        assert "1 entity" in result.summary
        assert "improving" in result.summary

    def test_best_worst_periods(self):
        rows = [
            {"metric": 50, "period": "Q1"},
            {"metric": 300, "period": "Q2"},
            {"metric": 100, "period": "Q3"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        et = result.entities[0]
        assert et.best_period == "Q2"
        assert et.worst_period == "Q1"

    def test_null_period_skipped(self):
        rows = [
            {"metric": 100, "period": "Q1"},
            {"metric": 200, "period": None},
            {"metric": 300, "period": "Q2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        assert result is not None
        assert len(result.entities[0].periods) == 2

    def test_null_metric_skipped(self):
        rows = [
            {"metric": None, "period": "Q1"},
            {"metric": 100, "period": "Q1"},
            {"metric": 200, "period": "Q2"},
        ]
        result = analyze_financial_trends(rows, "metric", "period")
        assert result is not None


# ---------------------------------------------------------------------------
# format_financial_report
# ---------------------------------------------------------------------------


class TestFormatFinancialReport:
    def test_no_data(self):
        report = format_financial_report()
        assert report == "No financial data available for report."

    def test_profitability_only(self):
        rows = [{"rev": 1000, "cost": 400}]
        prof = compute_profitability_ratios(rows, "rev", "cost")
        report = format_financial_report(profitability=prof)
        assert "Profitability" in report
        assert "1,000.00" in report
        assert "Liquidity" not in report

    def test_liquidity_only(self):
        rows = [{"ca": 3000, "cl": 1000}]
        liq = compute_liquidity_ratios(rows, "ca", "cl")
        report = format_financial_report(liquidity=liq)
        assert "Liquidity" in report
        assert "Current Ratio" in report

    def test_efficiency_only(self):
        rows = [{"rev": 1000, "assets": 500}]
        eff = compute_efficiency_ratios(rows, "rev", "assets")
        report = format_financial_report(efficiency=eff)
        assert "Efficiency" in report
        assert "Asset Turnover" in report

    def test_trends_only(self):
        rows = [
            {"metric": 100, "period": "Q1"},
            {"metric": 200, "period": "Q2"},
        ]
        trends = analyze_financial_trends(rows, "metric", "period")
        report = format_financial_report(trends=trends)
        assert "Financial Trends" in report
        assert "improving" in report

    def test_combined_report(self):
        prof_rows = [{"rev": 1000, "cost": 400}]
        liq_rows = [{"ca": 3000, "cl": 1000}]
        eff_rows = [{"rev": 1000, "assets": 500}]
        trend_rows = [
            {"metric": 100, "period": "Q1"},
            {"metric": 200, "period": "Q2"},
        ]
        prof = compute_profitability_ratios(prof_rows, "rev", "cost")
        liq = compute_liquidity_ratios(liq_rows, "ca", "cl")
        eff = compute_efficiency_ratios(eff_rows, "rev", "assets")
        trends = analyze_financial_trends(trend_rows, "metric", "period")

        report = format_financial_report(
            profitability=prof,
            liquidity=liq,
            efficiency=eff,
            trends=trends,
        )
        assert "Profitability" in report
        assert "Liquidity" in report
        assert "Efficiency" in report
        assert "Financial Trends" in report

    def test_efficiency_with_all_ratios_in_report(self):
        rows = [{"rev": 3650, "assets": 1000, "rec": 365, "pay": 365, "cogs": 730}]
        eff = compute_efficiency_ratios(
            rows, "rev", "assets",
            receivables_column="rec",
            payables_column="pay",
            cogs_column="cogs",
        )
        report = format_financial_report(efficiency=eff)
        assert "Receivables Turnover" in report
        assert "Days Sales Outstanding" in report
        assert "Payables Turnover" in report
        assert "Days Payable Outstanding" in report
        assert "Cash Conversion Cycle" in report

    def test_liquidity_with_optional_ratios_in_report(self):
        rows = [{"ca": 2000, "cl": 1000, "cash": 500, "inv": 300}]
        liq = compute_liquidity_ratios(
            rows, "ca", "cl", cash_column="cash", inventory_column="inv"
        )
        report = format_financial_report(liquidity=liq)
        assert "quick_ratio" in report
        assert "cash_ratio" in report

    def test_trends_with_cagr_in_report(self):
        rows = [
            {"metric": 100, "period": "P1"},
            {"metric": 110, "period": "P2"},
            {"metric": 121, "period": "P3"},
        ]
        trends = analyze_financial_trends(rows, "metric", "period")
        report = format_financial_report(trends=trends)
        assert "CAGR" in report

    def test_trends_without_cagr_in_report(self):
        rows = [
            {"metric": 100, "period": "P1"},
            {"metric": 200, "period": "P2"},
        ]
        trends = analyze_financial_trends(rows, "metric", "period")
        report = format_financial_report(trends=trends)
        assert "CAGR" not in report

    def test_profitability_multiple_entities_in_report(self):
        rows = [
            {"e": "Alpha", "rev": 1000, "cost": 400},
            {"e": "Beta", "rev": 2000, "cost": 1800},
        ]
        prof = compute_profitability_ratios(rows, "rev", "cost", entity_column="e")
        report = format_financial_report(profitability=prof)
        assert "Alpha" in report
        assert "Beta" in report
