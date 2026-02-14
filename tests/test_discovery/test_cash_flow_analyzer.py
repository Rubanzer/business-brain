"""Tests for cash_flow_analyzer module."""

from business_brain.discovery.cash_flow_analyzer import (
    PeriodFlow,
    CategoryFlow,
    CashFlowResult,
    PeriodWC,
    WorkingCapitalResult,
    MonthlyExpense,
    BurnRateResult,
    ProjectedPeriod,
    CashForecastResult,
    _safe_float,
    _parse_date,
    analyze_cash_flow,
    compute_working_capital,
    analyze_burn_rate,
    forecast_cash_position,
    format_cash_flow_report,
)

from datetime import datetime


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

    def test_boolean(self):
        # bool is subclass of int in Python
        assert _safe_float(True) == 1.0

    def test_list_returns_none(self):
        assert _safe_float([1, 2]) is None


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_none(self):
        assert _parse_date(None) is None

    def test_datetime_passthrough(self):
        dt = datetime(2024, 1, 15)
        assert _parse_date(dt) == dt

    def test_yyyy_mm_dd(self):
        result = _parse_date("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_yyyy_slash(self):
        result = _parse_date("2024/03/20")
        assert result == datetime(2024, 3, 20)

    def test_mm_dd_yyyy(self):
        result = _parse_date("03/20/2024")
        assert result == datetime(2024, 3, 20)

    def test_datetime_with_time(self):
        result = _parse_date("2024-01-15 10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30, 0)

    def test_invalid_string(self):
        assert _parse_date("not-a-date") is None

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_integer_returns_none(self):
        assert _parse_date(12345) is None


# ---------------------------------------------------------------------------
# analyze_cash_flow
# ---------------------------------------------------------------------------


class TestAnalyzeCashFlow:
    def test_empty_rows(self):
        assert analyze_cash_flow([], "inflow", "outflow") is None

    def test_all_null_data(self):
        rows = [{"inflow": None, "outflow": None}]
        assert analyze_cash_flow(rows, "inflow", "outflow") is None

    def test_all_invalid_data(self):
        rows = [{"inflow": "abc", "outflow": "xyz"}]
        assert analyze_cash_flow(rows, "inflow", "outflow") is None

    def test_missing_columns(self):
        rows = [{"x": 100, "y": 50}]
        assert analyze_cash_flow(rows, "inflow", "outflow") is None

    def test_single_row(self):
        rows = [{"inflow": 1000, "outflow": 600}]
        result = analyze_cash_flow(rows, "inflow", "outflow")
        assert result is not None
        assert result.total_inflow == 1000.0
        assert result.total_outflow == 600.0
        assert result.net_flow == 400.0
        assert result.cash_flow_ratio == round(1000 / 600, 4)
        assert result.period_flows is None
        assert result.category_flows is None

    def test_multiple_rows(self):
        rows = [
            {"inflow": 500, "outflow": 200},
            {"inflow": 300, "outflow": 400},
        ]
        result = analyze_cash_flow(rows, "inflow", "outflow")
        assert result is not None
        assert result.total_inflow == 800.0
        assert result.total_outflow == 600.0
        assert result.net_flow == 200.0

    def test_negative_net_flow(self):
        rows = [{"inflow": 100, "outflow": 500}]
        result = analyze_cash_flow(rows, "inflow", "outflow")
        assert result is not None
        assert result.net_flow == -400.0
        assert result.negative_periods_count == 1

    def test_zero_outflow_ratio(self):
        rows = [{"inflow": 500, "outflow": 0}]
        result = analyze_cash_flow(rows, "inflow", "outflow")
        assert result is not None
        assert result.cash_flow_ratio == 0.0

    def test_string_numeric_values(self):
        rows = [{"inflow": "1000", "outflow": "400"}]
        result = analyze_cash_flow(rows, "inflow", "outflow")
        assert result is not None
        assert result.total_inflow == 1000.0
        assert result.total_outflow == 400.0

    def test_mixed_valid_invalid_rows(self):
        rows = [
            {"inflow": 1000, "outflow": 500},
            {"inflow": "bad", "outflow": 300},
            {"inflow": 500, "outflow": 200},
        ]
        result = analyze_cash_flow(rows, "inflow", "outflow")
        assert result is not None
        assert result.total_inflow == 1500.0
        assert result.total_outflow == 700.0

    def test_with_date_column(self):
        rows = [
            {"inflow": 1000, "outflow": 400, "date": "2024-01-15"},
            {"inflow": 800, "outflow": 600, "date": "2024-01-20"},
            {"inflow": 500, "outflow": 700, "date": "2024-02-10"},
        ]
        result = analyze_cash_flow(rows, "inflow", "outflow", date_column="date")
        assert result is not None
        assert result.period_flows is not None
        assert len(result.period_flows) == 2

        jan = result.period_flows[0]
        assert jan.period == "2024-01"
        assert jan.inflow == 1800.0
        assert jan.outflow == 1000.0
        assert jan.net_flow == 800.0

        feb = result.period_flows[1]
        assert feb.period == "2024-02"
        assert feb.net_flow == -200.0

    def test_negative_periods_count_with_dates(self):
        rows = [
            {"inflow": 100, "outflow": 400, "date": "2024-01-15"},
            {"inflow": 800, "outflow": 200, "date": "2024-02-10"},
            {"inflow": 50, "outflow": 300, "date": "2024-03-05"},
        ]
        result = analyze_cash_flow(rows, "inflow", "outflow", date_column="date")
        assert result is not None
        assert result.negative_periods_count == 2

    def test_with_category_column(self):
        rows = [
            {"inflow": 500, "outflow": 200, "cat": "Sales"},
            {"inflow": 300, "outflow": 100, "cat": "Sales"},
            {"inflow": 200, "outflow": 400, "cat": "Services"},
        ]
        result = analyze_cash_flow(rows, "inflow", "outflow", category_column="cat")
        assert result is not None
        assert result.category_flows is not None
        assert len(result.category_flows) == 2

        sales = next(c for c in result.category_flows if c.category == "Sales")
        assert sales.total_inflow == 800.0
        assert sales.total_outflow == 300.0
        assert sales.net_flow == 500.0

    def test_with_date_and_category(self):
        rows = [
            {"inflow": 100, "outflow": 50, "date": "2024-01-15", "cat": "A"},
            {"inflow": 200, "outflow": 100, "date": "2024-02-10", "cat": "B"},
        ]
        result = analyze_cash_flow(
            rows, "inflow", "outflow", date_column="date", category_column="cat"
        )
        assert result is not None
        assert result.period_flows is not None
        assert result.category_flows is not None

    def test_summary_content(self):
        rows = [{"inflow": 1000, "outflow": 600}]
        result = analyze_cash_flow(rows, "inflow", "outflow")
        assert "1,000.00" in result.summary
        assert "600.00" in result.summary
        assert "400.00" in result.summary

    def test_date_column_with_invalid_dates(self):
        rows = [
            {"inflow": 100, "outflow": 50, "date": "not-a-date"},
            {"inflow": 200, "outflow": 100, "date": "2024-01-15"},
        ]
        result = analyze_cash_flow(rows, "inflow", "outflow", date_column="date")
        assert result is not None
        assert result.period_flows is not None
        assert len(result.period_flows) == 1

    def test_category_with_none_category_values(self):
        rows = [
            {"inflow": 100, "outflow": 50, "cat": None},
            {"inflow": 200, "outflow": 100, "cat": "Sales"},
        ]
        result = analyze_cash_flow(rows, "inflow", "outflow", category_column="cat")
        assert result is not None
        assert result.category_flows is not None
        assert len(result.category_flows) == 1

    def test_positive_net_no_negative_periods_without_date(self):
        rows = [{"inflow": 500, "outflow": 200}]
        result = analyze_cash_flow(rows, "inflow", "outflow")
        assert result.negative_periods_count == 0


# ---------------------------------------------------------------------------
# compute_working_capital
# ---------------------------------------------------------------------------


class TestComputeWorkingCapital:
    def test_empty_rows(self):
        assert compute_working_capital([], "rec", "pay") is None

    def test_all_null_data(self):
        rows = [{"rec": None, "pay": None}]
        assert compute_working_capital(rows, "rec", "pay") is None

    def test_all_invalid_data(self):
        rows = [{"rec": "abc", "pay": "xyz"}]
        assert compute_working_capital(rows, "rec", "pay") is None

    def test_missing_columns(self):
        rows = [{"x": 100}]
        assert compute_working_capital(rows, "rec", "pay") is None

    def test_single_row_basic(self):
        rows = [{"rec": 1000, "pay": 600}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result is not None
        assert result.avg_working_capital == 400.0
        assert result.min_wc == 400.0
        assert result.max_wc == 400.0
        assert result.health == "Healthy"
        assert len(result.periods) == 1

    def test_with_inventory(self):
        rows = [{"rec": 1000, "pay": 600, "inv": 200}]
        result = compute_working_capital(rows, "rec", "pay", inventory_column="inv")
        assert result is not None
        # WC = 1000 - 600 + 200 = 600
        assert result.avg_working_capital == 600.0

    def test_without_inventory_no_column(self):
        rows = [{"rec": 1000, "pay": 600}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result is not None
        # WC = 1000 - 600 = 400
        assert result.avg_working_capital == 400.0
        assert result.periods[0].inventory == 0.0

    def test_negative_working_capital_strained(self):
        # avg WC between -10% of receivables and 0
        # receivables=1000, payables=1050 => WC = -50
        # -50 >= -100 (which is -10% of 1000) => Strained
        rows = [{"rec": 1000, "pay": 1050}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result is not None
        assert result.avg_working_capital == -50.0
        assert result.health == "Strained"

    def test_negative_working_capital_critical(self):
        # avg WC worse than -10% of receivables
        # receivables=1000, payables=1200 => WC = -200
        # -200 < -100 (which is -10% of 1000) => Critical
        rows = [{"rec": 1000, "pay": 1200}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result is not None
        assert result.avg_working_capital == -200.0
        assert result.health == "Critical"

    def test_healthy_boundary(self):
        # WC exactly 0 => should be Strained (not > 0)
        # But -10% of receivables = -10, so 0 >= -10 => Strained
        rows = [{"rec": 100, "pay": 100}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result.health == "Strained"

    def test_strained_boundary_exact(self):
        # WC = -10% of receivables exactly
        # receivables=100, payables=110 => WC=-10
        # -10 >= -10 => Strained
        rows = [{"rec": 100, "pay": 110}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result.health == "Strained"

    def test_critical_boundary_just_below(self):
        # WC just below -10% of receivables
        # receivables=100, payables=110.01 => WC=-10.01
        # -10.01 < -10 => Critical
        rows = [{"rec": 100, "pay": 110.01}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result.health == "Critical"

    def test_with_period_column(self):
        rows = [
            {"rec": 1000, "pay": 500, "period": "Q1"},
            {"rec": 800, "pay": 900, "period": "Q2"},
        ]
        result = compute_working_capital(rows, "rec", "pay", period_column="period")
        assert result is not None
        assert len(result.periods) == 2

        q1 = next(p for p in result.periods if p.period == "Q1")
        assert q1.working_capital == 500.0

        q2 = next(p for p in result.periods if p.period == "Q2")
        assert q2.working_capital == -100.0

    def test_min_max_across_periods(self):
        rows = [
            {"rec": 1000, "pay": 500, "period": "Q1"},
            {"rec": 500, "pay": 800, "period": "Q2"},
            {"rec": 2000, "pay": 400, "period": "Q3"},
        ]
        result = compute_working_capital(rows, "rec", "pay", period_column="period")
        assert result.min_wc == -300.0
        assert result.max_wc == 1600.0

    def test_average_across_periods(self):
        rows = [
            {"rec": 1000, "pay": 500, "period": "Q1"},   # WC=500
            {"rec": 500, "pay": 800, "period": "Q2"},     # WC=-300
            {"rec": 2000, "pay": 400, "period": "Q3"},    # WC=1600
        ]
        result = compute_working_capital(rows, "rec", "pay", period_column="period")
        # avg = (500 + -300 + 1600) / 3 = 600
        assert result.avg_working_capital == 600.0

    def test_inventory_with_null_values(self):
        rows = [{"rec": 1000, "pay": 500, "inv": None}]
        result = compute_working_capital(rows, "rec", "pay", inventory_column="inv")
        assert result is not None
        # inventory is None, defaults to 0
        assert result.avg_working_capital == 500.0

    def test_string_numeric_values(self):
        rows = [{"rec": "1000", "pay": "400"}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result is not None
        assert result.avg_working_capital == 600.0

    def test_summary_content(self):
        rows = [{"rec": 1000, "pay": 600}]
        result = compute_working_capital(rows, "rec", "pay")
        assert "Healthy" in result.summary

    def test_zero_receivables_health(self):
        # receivables=0, payables=100 => WC=-100
        # avg_receivables=0 => set to 1 to avoid div by zero
        # -100 < -0.1 * 1 => Critical
        rows = [{"rec": 0, "pay": 100}]
        result = compute_working_capital(rows, "rec", "pay")
        assert result.health == "Critical"

    def test_multiple_rows_same_period(self):
        rows = [
            {"rec": 500, "pay": 200, "period": "Q1"},
            {"rec": 300, "pay": 100, "period": "Q1"},
        ]
        result = compute_working_capital(rows, "rec", "pay", period_column="period")
        assert len(result.periods) == 1
        # Aggregated: rec=800, pay=300 => WC=500
        assert result.periods[0].working_capital == 500.0


# ---------------------------------------------------------------------------
# analyze_burn_rate
# ---------------------------------------------------------------------------


class TestAnalyzeBurnRate:
    def test_empty_rows(self):
        assert analyze_burn_rate([], "expense", "date") is None

    def test_all_null_data(self):
        rows = [{"expense": None, "date": None}]
        assert analyze_burn_rate(rows, "expense", "date") is None

    def test_all_invalid_data(self):
        rows = [{"expense": "abc", "date": "not-a-date"}]
        assert analyze_burn_rate(rows, "expense", "date") is None

    def test_missing_columns(self):
        rows = [{"x": 100, "y": "2024-01-15"}]
        assert analyze_burn_rate(rows, "expense", "date") is None

    def test_single_month(self):
        rows = [
            {"expense": 500, "date": "2024-01-15"},
            {"expense": 300, "date": "2024-01-20"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        assert result.months_analyzed == 1
        assert result.total_expenses == 800.0
        assert result.gross_burn_rate == 800.0
        assert result.trend == "stable"

    def test_multiple_months(self):
        rows = [
            {"expense": 1000, "date": "2024-01-15"},
            {"expense": 1200, "date": "2024-02-15"},
            {"expense": 900, "date": "2024-03-15"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        assert result.months_analyzed == 3
        assert result.total_expenses == 3100.0
        # gross_burn = 3100 / 3 = 1033.33
        assert abs(result.gross_burn_rate - 1033.33) < 0.01

    def test_net_burn_rate_with_revenue(self):
        rows = [
            {"expense": 1000, "date": "2024-01-15", "revenue": 500},
            {"expense": 1000, "date": "2024-02-15", "revenue": 600},
        ]
        result = analyze_burn_rate(rows, "expense", "date", revenue_column="revenue")
        assert result is not None
        assert result.gross_burn_rate == 1000.0
        # net = (2000 - 1100) / 2 = 450
        assert result.net_burn_rate == 450.0

    def test_no_net_burn_without_revenue(self):
        rows = [{"expense": 1000, "date": "2024-01-15"}]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        assert result.net_burn_rate is None

    def test_highest_month_flagged(self):
        rows = [
            {"expense": 500, "date": "2024-01-15"},
            {"expense": 1500, "date": "2024-02-15"},
            {"expense": 800, "date": "2024-03-15"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        highest = [me for me in result.monthly_expenses if me.is_highest]
        assert len(highest) == 1
        assert highest[0].month == "2024-02"
        assert highest[0].amount == 1500.0

    def test_trend_increasing(self):
        rows = [
            {"expense": 100, "date": "2024-01-15"},
            {"expense": 100, "date": "2024-02-15"},
            {"expense": 500, "date": "2024-03-15"},
            {"expense": 500, "date": "2024-04-15"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        assert result.trend == "increasing"

    def test_trend_decreasing(self):
        rows = [
            {"expense": 500, "date": "2024-01-15"},
            {"expense": 500, "date": "2024-02-15"},
            {"expense": 100, "date": "2024-03-15"},
            {"expense": 100, "date": "2024-04-15"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        assert result.trend == "decreasing"

    def test_trend_stable(self):
        rows = [
            {"expense": 1000, "date": "2024-01-15"},
            {"expense": 1000, "date": "2024-02-15"},
            {"expense": 1000, "date": "2024-03-15"},
            {"expense": 1000, "date": "2024-04-15"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        assert result.trend == "stable"

    def test_monthly_expenses_sorted(self):
        rows = [
            {"expense": 300, "date": "2024-03-15"},
            {"expense": 100, "date": "2024-01-15"},
            {"expense": 200, "date": "2024-02-15"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        months = [me.month for me in result.monthly_expenses]
        assert months == ["2024-01", "2024-02", "2024-03"]

    def test_aggregation_within_month(self):
        rows = [
            {"expense": 200, "date": "2024-01-10"},
            {"expense": 300, "date": "2024-01-20"},
            {"expense": 500, "date": "2024-02-15"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result.months_analyzed == 2
        jan = next(me for me in result.monthly_expenses if me.month == "2024-01")
        assert jan.amount == 500.0

    def test_string_numeric_expense(self):
        rows = [{"expense": "1000", "date": "2024-01-15"}]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        assert result.total_expenses == 1000.0

    def test_mixed_valid_invalid(self):
        rows = [
            {"expense": 500, "date": "2024-01-15"},
            {"expense": "bad", "date": "2024-02-15"},
            {"expense": 300, "date": "2024-03-15"},
        ]
        result = analyze_burn_rate(rows, "expense", "date")
        assert result is not None
        assert result.months_analyzed == 2
        assert result.total_expenses == 800.0

    def test_summary_content(self):
        rows = [{"expense": 1000, "date": "2024-01-15"}]
        result = analyze_burn_rate(rows, "expense", "date")
        assert "1,000.00" in result.summary
        assert "1 months" in result.summary or "1 month" in result.summary

    def test_revenue_with_null_values(self):
        rows = [
            {"expense": 1000, "date": "2024-01-15", "revenue": None},
            {"expense": 800, "date": "2024-02-15", "revenue": 500},
        ]
        result = analyze_burn_rate(rows, "expense", "date", revenue_column="revenue")
        assert result is not None
        # Only Feb has valid revenue=500
        # net = (1800 - 500) / 2 = 650
        assert result.net_burn_rate == 650.0


# ---------------------------------------------------------------------------
# forecast_cash_position
# ---------------------------------------------------------------------------


class TestForecastCashPosition:
    def test_empty_rows(self):
        assert forecast_cash_position([], "amount", "date") is None

    def test_zero_periods_ahead(self):
        rows = [{"amount": 100, "date": "2024-01-15"}]
        assert forecast_cash_position(rows, "amount", "date", periods_ahead=0) is None

    def test_negative_periods_ahead(self):
        rows = [{"amount": 100, "date": "2024-01-15"}]
        assert forecast_cash_position(rows, "amount", "date", periods_ahead=-1) is None

    def test_all_null_data(self):
        rows = [{"amount": None, "date": None}]
        assert forecast_cash_position(rows, "amount", "date") is None

    def test_all_invalid_data(self):
        rows = [{"amount": "abc", "date": "not-a-date"}]
        assert forecast_cash_position(rows, "amount", "date") is None

    def test_single_month_positive(self):
        rows = [{"amount": 500, "date": "2024-01-15"}]
        result = forecast_cash_position(rows, "amount", "date", periods_ahead=2)
        assert result is not None
        assert result.current_net_monthly == 500.0
        assert len(result.projections) == 2
        assert result.projections[0].period_label == "2024-02"
        assert result.projections[1].period_label == "2024-03"

    def test_single_month_negative(self):
        rows = [{"amount": -200, "date": "2024-01-15"}]
        result = forecast_cash_position(rows, "amount", "date", periods_ahead=1)
        assert result is not None
        assert result.current_net_monthly == -200.0

    def test_positive_negative_classification(self):
        rows = [
            {"amount": 1000, "date": "2024-01-15"},
            {"amount": -400, "date": "2024-01-20"},
        ]
        result = forecast_cash_position(rows, "amount", "date", periods_ahead=1)
        assert result is not None
        assert result.current_net_monthly == 600.0

    def test_with_type_column_inflow(self):
        rows = [
            {"amount": 1000, "date": "2024-01-15", "type": "revenue"},
            {"amount": 400, "date": "2024-01-20", "type": "expense"},
        ]
        result = forecast_cash_position(
            rows, "amount", "date", type_column="type", periods_ahead=1
        )
        assert result is not None
        assert result.current_net_monthly == 600.0

    def test_type_column_keywords(self):
        # Test multiple inflow keywords
        for keyword in ["income", "receipt", "revenue", "credit", "incoming"]:
            rows = [
                {"amount": 500, "date": "2024-01-15", "type": keyword},
                {"amount": 200, "date": "2024-01-20", "type": "expense"},
            ]
            result = forecast_cash_position(
                rows, "amount", "date", type_column="type", periods_ahead=1
            )
            assert result is not None
            assert result.current_net_monthly == 300.0, f"Failed for keyword: {keyword}"

    def test_trend_improving(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
            {"amount": 300, "date": "2024-03-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date")
        assert result is not None
        assert result.trend_direction == "improving"

    def test_trend_declining(self):
        rows = [
            {"amount": -100, "date": "2024-01-15"},
            {"amount": -200, "date": "2024-02-15"},
            {"amount": -300, "date": "2024-03-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date")
        assert result is not None
        assert result.trend_direction == "declining"

    def test_trend_stable(self):
        rows = [
            {"amount": 500, "date": "2024-01-15"},
            {"amount": 500, "date": "2024-02-15"},
            {"amount": 500, "date": "2024-03-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date")
        assert result is not None
        assert result.trend_direction == "stable"

    def test_confidence_high(self):
        # Perfect linear trend => high R^2
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
            {"amount": 300, "date": "2024-03-15"},
            {"amount": 400, "date": "2024-04-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date")
        assert result is not None
        assert result.confidence == "High"

    def test_confidence_low(self):
        # Highly volatile data => low R^2
        rows = [
            {"amount": 1000, "date": "2024-01-15"},
            {"amount": -500, "date": "2024-02-15"},
            {"amount": 800, "date": "2024-03-15"},
            {"amount": -300, "date": "2024-04-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date")
        assert result is not None
        assert result.confidence == "Low"

    def test_projections_count(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date", periods_ahead=5)
        assert result is not None
        assert len(result.projections) == 5

    def test_projection_period_labels(self):
        rows = [
            {"amount": 100, "date": "2024-10-15"},
            {"amount": 200, "date": "2024-11-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date", periods_ahead=3)
        labels = [p.period_label for p in result.projections]
        assert labels == ["2024-12", "2025-01", "2025-02"]

    def test_projection_year_wrap(self):
        rows = [
            {"amount": 100, "date": "2024-11-15"},
            {"amount": 200, "date": "2024-12-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date", periods_ahead=2)
        labels = [p.period_label for p in result.projections]
        assert labels == ["2025-01", "2025-02"]

    def test_projections_have_inflow_outflow_net(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date", periods_ahead=1)
        proj = result.projections[0]
        assert hasattr(proj, "projected_inflow")
        assert hasattr(proj, "projected_outflow")
        assert hasattr(proj, "projected_net")

    def test_summary_content(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date")
        assert "Trend" in result.summary
        assert "Confidence" in result.summary

    def test_single_month_confidence_low(self):
        rows = [{"amount": 100, "date": "2024-01-15"}]
        result = forecast_cash_position(rows, "amount", "date")
        assert result is not None
        # n=1, r_squared=0 => Low
        assert result.confidence == "Low"

    def test_all_same_values_confidence_high(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 100, "date": "2024-02-15"},
            {"amount": 100, "date": "2024-03-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date")
        assert result is not None
        # All same => ss_tot=0 => r_squared=1.0 => High
        assert result.confidence == "High"

    def test_trend_magnitude_positive(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
        ]
        result = forecast_cash_position(rows, "amount", "date")
        assert result.trend_magnitude >= 0


# ---------------------------------------------------------------------------
# format_cash_flow_report
# ---------------------------------------------------------------------------


class TestFormatCashFlowReport:
    def test_no_data(self):
        report = format_cash_flow_report()
        assert report == "No cash flow data available for report."

    def test_cash_flow_only(self):
        rows = [{"inflow": 1000, "outflow": 400}]
        cf = analyze_cash_flow(rows, "inflow", "outflow")
        report = format_cash_flow_report(cash_flow=cf)
        assert "Cash Flow Analysis" in report
        assert "1,000.00" in report
        assert "Working Capital" not in report

    def test_working_capital_only(self):
        rows = [{"rec": 1000, "pay": 400}]
        wc = compute_working_capital(rows, "rec", "pay")
        report = format_cash_flow_report(working_capital=wc)
        assert "Working Capital" in report
        assert "Healthy" in report
        assert "Cash Flow Analysis" not in report

    def test_burn_rate_only(self):
        rows = [{"expense": 1000, "date": "2024-01-15"}]
        br = analyze_burn_rate(rows, "expense", "date")
        report = format_cash_flow_report(burn_rate=br)
        assert "Burn Rate" in report
        assert "1,000.00" in report
        assert "Cash Flow Analysis" not in report

    def test_forecast_only(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
        ]
        fc = forecast_cash_position(rows, "amount", "date")
        report = format_cash_flow_report(forecast=fc)
        assert "Cash Forecast" in report
        assert "Confidence" in report
        assert "Cash Flow Analysis" not in report

    def test_all_sections_combined(self):
        cf_rows = [{"inflow": 1000, "outflow": 400}]
        wc_rows = [{"rec": 1000, "pay": 400}]
        br_rows = [{"expense": 500, "date": "2024-01-15"}]
        fc_rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
        ]

        cf = analyze_cash_flow(cf_rows, "inflow", "outflow")
        wc = compute_working_capital(wc_rows, "rec", "pay")
        br = analyze_burn_rate(br_rows, "expense", "date")
        fc = forecast_cash_position(fc_rows, "amount", "date")

        report = format_cash_flow_report(
            cash_flow=cf, working_capital=wc, burn_rate=br, forecast=fc
        )
        assert "Cash Flow Analysis" in report
        assert "Working Capital" in report
        assert "Burn Rate" in report
        assert "Cash Forecast" in report

    def test_cash_flow_with_periods(self):
        rows = [
            {"inflow": 1000, "outflow": 400, "date": "2024-01-15"},
            {"inflow": 800, "outflow": 600, "date": "2024-02-15"},
        ]
        cf = analyze_cash_flow(rows, "inflow", "outflow", date_column="date")
        report = format_cash_flow_report(cash_flow=cf)
        assert "Period Breakdown" in report
        assert "2024-01" in report
        assert "2024-02" in report

    def test_cash_flow_with_categories(self):
        rows = [
            {"inflow": 1000, "outflow": 400, "cat": "Sales"},
            {"inflow": 500, "outflow": 600, "cat": "Services"},
        ]
        cf = analyze_cash_flow(rows, "inflow", "outflow", category_column="cat")
        report = format_cash_flow_report(cash_flow=cf)
        assert "Category Breakdown" in report
        assert "Sales" in report
        assert "Services" in report

    def test_burn_rate_with_net(self):
        rows = [
            {"expense": 1000, "date": "2024-01-15", "revenue": 500},
        ]
        br = analyze_burn_rate(rows, "expense", "date", revenue_column="revenue")
        report = format_cash_flow_report(burn_rate=br)
        assert "Net Burn Rate" in report

    def test_burn_rate_without_net(self):
        rows = [{"expense": 1000, "date": "2024-01-15"}]
        br = analyze_burn_rate(rows, "expense", "date")
        report = format_cash_flow_report(burn_rate=br)
        assert "Gross Burn Rate" in report
        assert "Net Burn Rate" not in report

    def test_burn_rate_highest_month_flag(self):
        rows = [
            {"expense": 500, "date": "2024-01-15"},
            {"expense": 1500, "date": "2024-02-15"},
        ]
        br = analyze_burn_rate(rows, "expense", "date")
        report = format_cash_flow_report(burn_rate=br)
        assert "[HIGHEST]" in report

    def test_working_capital_periods_in_report(self):
        rows = [
            {"rec": 1000, "pay": 400, "inv": 100, "period": "Q1"},
            {"rec": 800, "pay": 900, "inv": 50, "period": "Q2"},
        ]
        wc = compute_working_capital(
            rows, "rec", "pay", inventory_column="inv", period_column="period"
        )
        report = format_cash_flow_report(working_capital=wc)
        assert "Q1" in report
        assert "Q2" in report
        assert "WC=" in report

    def test_forecast_projections_in_report(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
        ]
        fc = forecast_cash_position(rows, "amount", "date", periods_ahead=2)
        report = format_cash_flow_report(forecast=fc)
        assert "Projections" in report
        assert "2024-03" in report
        assert "2024-04" in report


# ---------------------------------------------------------------------------
# Dataclass instantiation tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_period_flow(self):
        pf = PeriodFlow(period="2024-01", inflow=100.0, outflow=50.0, net_flow=50.0)
        assert pf.period == "2024-01"
        assert pf.inflow == 100.0
        assert pf.outflow == 50.0
        assert pf.net_flow == 50.0

    def test_category_flow(self):
        cf = CategoryFlow(category="Sales", total_inflow=1000.0, total_outflow=400.0, net_flow=600.0)
        assert cf.category == "Sales"
        assert cf.net_flow == 600.0

    def test_cash_flow_result(self):
        cfr = CashFlowResult(
            total_inflow=1000.0, total_outflow=400.0, net_flow=600.0,
            cash_flow_ratio=2.5, negative_periods_count=0,
            period_flows=None, category_flows=None, summary="test"
        )
        assert cfr.total_inflow == 1000.0
        assert cfr.cash_flow_ratio == 2.5

    def test_period_wc(self):
        pw = PeriodWC(period="Q1", receivables=1000.0, payables=400.0, inventory=100.0, working_capital=700.0)
        assert pw.working_capital == 700.0

    def test_working_capital_result(self):
        wcr = WorkingCapitalResult(
            avg_working_capital=500.0, min_wc=200.0, max_wc=800.0,
            health="Healthy", periods=[], summary="test"
        )
        assert wcr.health == "Healthy"

    def test_monthly_expense(self):
        me = MonthlyExpense(month="2024-01", amount=1000.0, is_highest=True)
        assert me.is_highest is True

    def test_burn_rate_result(self):
        brr = BurnRateResult(
            gross_burn_rate=1000.0, net_burn_rate=500.0, months_analyzed=3,
            total_expenses=3000.0, monthly_expenses=[], trend="stable", summary="test"
        )
        assert brr.gross_burn_rate == 1000.0

    def test_projected_period(self):
        pp = ProjectedPeriod(
            period_label="2024-04", projected_inflow=500.0,
            projected_outflow=300.0, projected_net=200.0
        )
        assert pp.projected_net == 200.0

    def test_cash_forecast_result(self):
        cfr = CashForecastResult(
            current_net_monthly=200.0, trend_direction="improving",
            trend_magnitude=50.0, projections=[], confidence="High", summary="test"
        )
        assert cfr.confidence == "High"
