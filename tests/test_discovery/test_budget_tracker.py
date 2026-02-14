"""Tests for budget tracking and financial planning module."""

from datetime import datetime

from business_brain.discovery.budget_tracker import (
    BudgetResult,
    BurnRateResult,
    CategorySpend,
    CategoryVariance,
    ForecastEntry,
    MonthChange,
    PeriodVariance,
    SpendingResult,
    VendorSpend,
    analyze_spending_patterns,
    budget_vs_actual,
    compute_burn_rate,
    forecast_budget,
    format_budget_report,
)


# ---------------------------------------------------------------------------
# budget_vs_actual
# ---------------------------------------------------------------------------


class TestBudgetVsActual:
    def test_basic_categories(self):
        rows = [
            {"cat": "Marketing", "budget": 1000, "actual": 1200},
            {"cat": "Engineering", "budget": 2000, "actual": 1800},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert len(result.categories) == 2
        assert result.total_budget == 3000.0
        assert result.total_actual == 3000.0

    def test_over_budget_flagged(self):
        rows = [
            {"cat": "Marketing", "budget": 1000, "actual": 1200},
            {"cat": "Engineering", "budget": 2000, "actual": 1800},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        mkt = [c for c in result.categories if c.category == "Marketing"][0]
        eng = [c for c in result.categories if c.category == "Engineering"][0]
        assert mkt.over_budget is True
        assert eng.over_budget is False

    def test_variance_calculation(self):
        rows = [
            {"cat": "Ops", "budget": 500, "actual": 600},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        cat = result.categories[0]
        assert cat.variance == 100.0
        assert cat.variance_pct == 20.0

    def test_negative_variance(self):
        rows = [
            {"cat": "Ops", "budget": 500, "actual": 400},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        cat = result.categories[0]
        assert cat.variance == -100.0
        assert cat.variance_pct == -20.0
        assert cat.over_budget is False

    def test_overall_variance_pct(self):
        rows = [
            {"cat": "A", "budget": 1000, "actual": 1100},
            {"cat": "B", "budget": 1000, "actual": 900},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert result.overall_variance == 0.0
        assert result.overall_variance_pct == 0.0

    def test_over_under_counts(self):
        rows = [
            {"cat": "A", "budget": 100, "actual": 150},
            {"cat": "B", "budget": 100, "actual": 80},
            {"cat": "C", "budget": 100, "actual": 120},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert result.over_budget_count == 2
        assert result.under_budget_count == 1

    def test_empty_rows(self):
        result = budget_vs_actual([], "cat", "budget", "actual")
        assert result is None

    def test_all_invalid_data(self):
        rows = [
            {"cat": "A", "budget": "abc", "actual": "xyz"},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is None

    def test_missing_columns(self):
        rows = [
            {"other": "val"},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is None

    def test_with_period_column(self):
        rows = [
            {"cat": "A", "budget": 100, "actual": 110, "period": "Q1"},
            {"cat": "A", "budget": 100, "actual": 90, "period": "Q2"},
            {"cat": "B", "budget": 200, "actual": 250, "period": "Q1"},
            {"cat": "B", "budget": 200, "actual": 180, "period": "Q2"},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual", "period")
        assert result is not None
        assert result.periods is not None
        assert len(result.periods) == 2
        q1 = [p for p in result.periods if p.period == "Q1"][0]
        assert q1.budget == 300.0
        assert q1.actual == 360.0
        assert q1.variance == 60.0

    def test_period_variance_pct(self):
        rows = [
            {"cat": "A", "budget": 100, "actual": 120, "q": "Jan"},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual", "q")
        assert result is not None
        assert result.periods is not None
        assert result.periods[0].variance_pct == 20.0

    def test_aggregates_same_category(self):
        rows = [
            {"cat": "A", "budget": 100, "actual": 50},
            {"cat": "A", "budget": 100, "actual": 80},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert len(result.categories) == 1
        assert result.categories[0].budget == 200.0
        assert result.categories[0].actual == 130.0

    def test_summary_contains_key_info(self):
        rows = [{"cat": "X", "budget": 500, "actual": 600}]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert "500" in result.summary
        assert "600" in result.summary

    def test_zero_budget_no_division_error(self):
        rows = [{"cat": "A", "budget": 0, "actual": 100}]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert result.categories[0].variance_pct == 0.0

    def test_string_numeric_values(self):
        rows = [{"cat": "A", "budget": "1000", "actual": "900"}]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert result.categories[0].budget == 1000.0
        assert result.categories[0].actual == 900.0

    def test_mixed_valid_invalid_rows(self):
        rows = [
            {"cat": "A", "budget": 100, "actual": 120},
            {"cat": "B", "budget": "bad", "actual": 50},
            {"cat": "C", "budget": 200, "actual": 180},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert len(result.categories) == 2  # B skipped


# ---------------------------------------------------------------------------
# compute_burn_rate
# ---------------------------------------------------------------------------


class TestComputeBurnRate:
    def test_basic_burn_rate(self):
        rows = [
            {"amount": 100, "date": "2024-01-01"},
            {"amount": 100, "date": "2024-01-11"},
            {"amount": 100, "date": "2024-01-21"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert result.total_spend == 300.0
        assert result.num_days == 20
        assert result.daily_burn_rate == 15.0
        assert result.monthly_burn_rate == 450.0

    def test_with_total_budget(self):
        rows = [
            {"amount": 100, "date": "2024-01-01"},
            {"amount": 100, "date": "2024-01-11"},
        ]
        result = compute_burn_rate(rows, "amount", "date", total_budget=1000)
        assert result is not None
        assert result.remaining_budget == 800.0
        assert result.days_until_exhaustion is not None
        assert result.days_until_exhaustion > 0
        assert result.projected_end_date is not None

    def test_budget_exhausted(self):
        rows = [
            {"amount": 500, "date": "2024-01-01"},
            {"amount": 600, "date": "2024-01-11"},
        ]
        result = compute_burn_rate(rows, "amount", "date", total_budget=1000)
        assert result is not None
        assert result.remaining_budget is not None
        assert result.remaining_budget < 0
        assert result.days_until_exhaustion == 0.0

    def test_no_budget_projections_omitted(self):
        rows = [
            {"amount": 100, "date": "2024-01-01"},
            {"amount": 100, "date": "2024-01-11"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert result.remaining_budget is None
        assert result.days_until_exhaustion is None
        assert result.projected_end_date is None

    def test_trend_accelerating(self):
        # More spending in second half
        rows = [
            {"amount": 10, "date": "2024-01-01"},
            {"amount": 10, "date": "2024-01-05"},
            {"amount": 10, "date": "2024-01-10"},
            {"amount": 10, "date": "2024-01-15"},
            {"amount": 100, "date": "2024-01-20"},
            {"amount": 100, "date": "2024-01-25"},
            {"amount": 100, "date": "2024-01-30"},
            {"amount": 100, "date": "2024-02-04"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert result.trend == "accelerating"
        assert result.second_half_rate > result.first_half_rate

    def test_trend_decelerating(self):
        # More spending in first half
        rows = [
            {"amount": 100, "date": "2024-01-01"},
            {"amount": 100, "date": "2024-01-05"},
            {"amount": 100, "date": "2024-01-10"},
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 10, "date": "2024-01-20"},
            {"amount": 10, "date": "2024-01-25"},
            {"amount": 10, "date": "2024-01-30"},
            {"amount": 10, "date": "2024-02-04"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert result.trend == "decelerating"

    def test_trend_stable(self):
        rows = [
            {"amount": 100, "date": "2024-01-01"},
            {"amount": 100, "date": "2024-01-06"},
            {"amount": 100, "date": "2024-01-11"},
            {"amount": 100, "date": "2024-01-16"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert result.trend == "stable"

    def test_empty_rows(self):
        result = compute_burn_rate([], "amount", "date")
        assert result is None

    def test_invalid_dates(self):
        rows = [
            {"amount": 100, "date": "not-a-date"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is None

    def test_invalid_amounts(self):
        rows = [
            {"amount": "bad", "date": "2024-01-01"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is None

    def test_single_entry(self):
        rows = [
            {"amount": 500, "date": "2024-01-15"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert result.total_spend == 500.0
        assert result.num_days == 1  # min 1 day

    def test_datetime_objects(self):
        rows = [
            {"amount": 50, "date": datetime(2024, 3, 1)},
            {"amount": 50, "date": datetime(2024, 3, 11)},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert result.total_spend == 100.0
        assert result.num_days == 10

    def test_summary_text(self):
        rows = [
            {"amount": 100, "date": "2024-01-01"},
            {"amount": 100, "date": "2024-01-11"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert "Burn rate" in result.summary
        assert "200" in result.summary


# ---------------------------------------------------------------------------
# analyze_spending_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeSpendingPatterns:
    def test_basic_category_breakdown(self):
        rows = [
            {"cat": "Food", "amount": 300},
            {"cat": "Rent", "amount": 500},
            {"cat": "Transport", "amount": 200},
        ]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is not None
        assert result.total_spend == 1000.0
        assert len(result.category_breakdown) == 3

    def test_pct_of_total(self):
        rows = [
            {"cat": "A", "amount": 250},
            {"cat": "B", "amount": 750},
        ]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is not None
        b = [c for c in result.category_breakdown if c.category == "B"][0]
        assert b.pct_of_total == 75.0

    def test_top_5_categories(self):
        rows = [
            {"cat": f"Cat{i}", "amount": i * 10}
            for i in range(1, 8)
        ]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is not None
        assert len(result.top_categories) == 5
        # Top should be Cat7 (70)
        assert result.top_categories[0].category == "Cat7"

    def test_fewer_than_5_categories(self):
        rows = [
            {"cat": "A", "amount": 100},
            {"cat": "B", "amount": 200},
        ]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is not None
        assert len(result.top_categories) == 2

    def test_month_over_month(self):
        rows = [
            {"cat": "A", "amount": 100, "date": "2024-01-15"},
            {"cat": "A", "amount": 120, "date": "2024-02-15"},
            {"cat": "A", "amount": 150, "date": "2024-03-15"},
        ]
        result = analyze_spending_patterns(rows, "cat", "amount", date_column="date")
        assert result is not None
        assert result.month_over_month is not None
        # First month should have no change, subsequent months should
        first_entry = result.month_over_month[0]
        assert first_entry.change is None
        second_entry = result.month_over_month[1]
        assert second_entry.change == 20.0

    def test_month_over_month_pct(self):
        rows = [
            {"cat": "A", "amount": 100, "date": "2024-01-15"},
            {"cat": "A", "amount": 150, "date": "2024-02-15"},
        ]
        result = analyze_spending_patterns(rows, "cat", "amount", date_column="date")
        assert result is not None
        assert result.month_over_month is not None
        mom = result.month_over_month[1]
        assert mom.change_pct == 50.0

    def test_vendor_analysis(self):
        rows = [
            {"cat": "A", "amount": 100, "vendor": "VendorX"},
            {"cat": "A", "amount": 200, "vendor": "VendorY"},
            {"cat": "B", "amount": 150, "vendor": "VendorX"},
        ]
        result = analyze_spending_patterns(
            rows, "cat", "amount", vendor_column="vendor"
        )
        assert result is not None
        assert result.top_vendors is not None
        assert len(result.top_vendors) == 2
        assert result.top_vendors[0].vendor == "VendorX"
        assert result.top_vendors[0].amount == 250.0

    def test_top_5_vendors_limit(self):
        rows = [
            {"cat": "A", "amount": i * 10, "vendor": f"V{i}"}
            for i in range(1, 8)
        ]
        result = analyze_spending_patterns(
            rows, "cat", "amount", vendor_column="vendor"
        )
        assert result is not None
        assert result.top_vendors is not None
        assert len(result.top_vendors) == 5

    def test_no_date_no_vendor(self):
        rows = [{"cat": "A", "amount": 100}]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is not None
        assert result.month_over_month is None
        assert result.top_vendors is None

    def test_empty_rows(self):
        result = analyze_spending_patterns([], "cat", "amount")
        assert result is None

    def test_all_invalid(self):
        rows = [{"cat": "A", "amount": "bad"}]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is None

    def test_zero_total_spend(self):
        rows = [{"cat": "A", "amount": 0}]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is None

    def test_summary_text(self):
        rows = [
            {"cat": "Food", "amount": 300},
            {"cat": "Rent", "amount": 500},
        ]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is not None
        assert "Spending analysis" in result.summary
        assert "800" in result.summary

    def test_aggregates_same_category(self):
        rows = [
            {"cat": "A", "amount": 100},
            {"cat": "A", "amount": 200},
            {"cat": "B", "amount": 50},
        ]
        result = analyze_spending_patterns(rows, "cat", "amount")
        assert result is not None
        a = [c for c in result.category_breakdown if c.category == "A"][0]
        assert a.amount == 300.0


# ---------------------------------------------------------------------------
# forecast_budget
# ---------------------------------------------------------------------------


class TestForecastBudget:
    def test_basic_forecast(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 120, "date": "2024-02-15"},
            {"amount": 140, "date": "2024-03-15"},
        ]
        result = forecast_budget(rows, "amount", "date", periods_ahead=2)
        assert len(result) == 2
        # Linear trend: should project increasing amounts
        assert result[0].projected_amount > 0
        assert result[1].projected_amount >= result[0].projected_amount

    def test_cumulative(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 100, "date": "2024-02-15"},
            {"amount": 100, "date": "2024-03-15"},
        ]
        result = forecast_budget(rows, "amount", "date", periods_ahead=3)
        assert len(result) == 3
        assert result[0].cumulative == result[0].projected_amount
        assert result[1].cumulative == round(
            result[0].projected_amount + result[1].projected_amount, 2
        )

    def test_period_names(self):
        rows = [
            {"amount": 100, "date": "2024-10-15"},
            {"amount": 100, "date": "2024-11-15"},
            {"amount": 100, "date": "2024-12-15"},
        ]
        result = forecast_budget(rows, "amount", "date", periods_ahead=3)
        assert result[0].period == "2025-01"
        assert result[1].period == "2025-02"
        assert result[2].period == "2025-03"

    def test_year_rollover(self):
        rows = [
            {"amount": 100, "date": "2024-11-15"},
            {"amount": 100, "date": "2024-12-15"},
        ]
        result = forecast_budget(rows, "amount", "date", periods_ahead=2)
        assert result[0].period == "2025-01"
        assert result[1].period == "2025-02"

    def test_high_confidence(self):
        # Very consistent data -> low CV -> high confidence
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 102, "date": "2024-02-15"},
            {"amount": 99, "date": "2024-03-15"},
            {"amount": 101, "date": "2024-04-15"},
        ]
        result = forecast_budget(rows, "amount", "date")
        assert all(f.confidence == "high" for f in result)

    def test_low_confidence(self):
        # Very volatile data -> high CV -> low confidence
        rows = [
            {"amount": 10, "date": "2024-01-15"},
            {"amount": 500, "date": "2024-02-15"},
            {"amount": 20, "date": "2024-03-15"},
            {"amount": 600, "date": "2024-04-15"},
        ]
        result = forecast_budget(rows, "amount", "date")
        assert all(f.confidence == "low" for f in result)

    def test_empty_rows(self):
        result = forecast_budget([], "amount", "date")
        assert result == []

    def test_zero_periods(self):
        rows = [{"amount": 100, "date": "2024-01-15"}]
        result = forecast_budget(rows, "amount", "date", periods_ahead=0)
        assert result == []

    def test_negative_periods(self):
        rows = [{"amount": 100, "date": "2024-01-15"}]
        result = forecast_budget(rows, "amount", "date", periods_ahead=-1)
        assert result == []

    def test_single_month_flat_projection(self):
        rows = [
            {"amount": 200, "date": "2024-06-10"},
            {"amount": 100, "date": "2024-06-20"},
        ]
        result = forecast_budget(rows, "amount", "date", periods_ahead=2)
        assert len(result) == 2
        # Single month aggregated to 300, flat projection
        assert result[0].projected_amount == 300.0

    def test_all_invalid_data(self):
        rows = [
            {"amount": "bad", "date": "not-a-date"},
        ]
        result = forecast_budget(rows, "amount", "date")
        assert result == []

    def test_no_negative_forecast(self):
        # Strongly decreasing trend should still not go negative
        rows = [
            {"amount": 500, "date": "2024-01-15"},
            {"amount": 300, "date": "2024-02-15"},
            {"amount": 100, "date": "2024-03-15"},
        ]
        result = forecast_budget(rows, "amount", "date", periods_ahead=5)
        for entry in result:
            assert entry.projected_amount >= 0

    def test_default_periods_ahead(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 100, "date": "2024-02-15"},
        ]
        result = forecast_budget(rows, "amount", "date")
        assert len(result) == 3  # default is 3


# ---------------------------------------------------------------------------
# format_budget_report
# ---------------------------------------------------------------------------


class TestFormatBudgetReport:
    def test_no_data(self):
        report = format_budget_report()
        assert "No analysis data provided" in report

    def test_with_budget_result(self):
        br = BudgetResult(
            categories=[
                CategoryVariance("A", 100, 120, 20, 20.0, True),
            ],
            total_budget=100,
            total_actual=120,
            overall_variance=20,
            overall_variance_pct=20.0,
            over_budget_count=1,
            under_budget_count=0,
            periods=None,
            summary="test",
        )
        report = format_budget_report(budget_result=br)
        assert "Budget vs Actual" in report
        assert "[OVER]" in report
        assert "100" in report

    def test_with_budget_periods(self):
        br = BudgetResult(
            categories=[],
            total_budget=100,
            total_actual=120,
            overall_variance=20,
            overall_variance_pct=20.0,
            over_budget_count=0,
            under_budget_count=0,
            periods=[
                PeriodVariance("Q1", 50, 60, 10, 20.0),
            ],
            summary="test",
        )
        report = format_budget_report(budget_result=br)
        assert "Period Breakdown" in report
        assert "Q1" in report

    def test_with_burn_rate(self):
        br = BurnRateResult(
            total_spend=5000,
            num_days=30,
            daily_burn_rate=166.67,
            monthly_burn_rate=5000.0,
            remaining_budget=5000.0,
            days_until_exhaustion=30.0,
            projected_end_date="2024-03-01",
            trend="stable",
            first_half_rate=166.0,
            second_half_rate=167.0,
            summary="test",
        )
        report = format_budget_report(burn_rate=br)
        assert "Burn Rate" in report
        assert "166.67" in report
        assert "Remaining budget" in report
        assert "2024-03-01" in report

    def test_with_spending(self):
        sp = SpendingResult(
            category_breakdown=[
                CategorySpend("Food", 300, 60.0),
                CategorySpend("Transport", 200, 40.0),
            ],
            total_spend=500,
            top_categories=[
                CategorySpend("Food", 300, 60.0),
                CategorySpend("Transport", 200, 40.0),
            ],
            month_over_month=None,
            top_vendors=[
                VendorSpend("VendorA", 300, 60.0),
            ],
            summary="test",
        )
        report = format_budget_report(spending=sp)
        assert "Spending Patterns" in report
        assert "Food" in report
        assert "VendorA" in report

    def test_with_forecast(self):
        fc = [
            ForecastEntry("2024-04", 100.0, 100.0, "high"),
            ForecastEntry("2024-05", 110.0, 210.0, "high"),
        ]
        report = format_budget_report(forecast=fc)
        assert "Budget Forecast" in report
        assert "2024-04" in report
        assert "high confidence" in report

    def test_full_report(self):
        br = BudgetResult(
            categories=[CategoryVariance("A", 100, 90, -10, -10.0, False)],
            total_budget=100,
            total_actual=90,
            overall_variance=-10,
            overall_variance_pct=-10.0,
            over_budget_count=0,
            under_budget_count=1,
            periods=None,
            summary="test",
        )
        burn = BurnRateResult(
            total_spend=90,
            num_days=30,
            daily_burn_rate=3.0,
            monthly_burn_rate=90.0,
            remaining_budget=None,
            days_until_exhaustion=None,
            projected_end_date=None,
            trend="stable",
            first_half_rate=3.0,
            second_half_rate=3.0,
            summary="test",
        )
        sp = SpendingResult(
            category_breakdown=[CategorySpend("A", 90, 100.0)],
            total_spend=90,
            top_categories=[CategorySpend("A", 90, 100.0)],
            month_over_month=None,
            top_vendors=None,
            summary="test",
        )
        fc = [ForecastEntry("2024-04", 95.0, 95.0, "moderate")]
        report = format_budget_report(br, burn, sp, fc)
        assert "Budget Tracking Report" in report
        assert "Budget vs Actual" in report
        assert "Burn Rate" in report
        assert "Spending Patterns" in report
        assert "Budget Forecast" in report

    def test_report_header(self):
        report = format_budget_report()
        assert report.startswith("Budget Tracking Report")
        assert "=" * 40 in report

    def test_burn_rate_without_remaining(self):
        burn = BurnRateResult(
            total_spend=500,
            num_days=10,
            daily_burn_rate=50.0,
            monthly_burn_rate=1500.0,
            remaining_budget=None,
            days_until_exhaustion=None,
            projected_end_date=None,
            trend="accelerating",
            first_half_rate=40.0,
            second_half_rate=60.0,
            summary="test",
        )
        report = format_budget_report(burn_rate=burn)
        assert "Remaining budget" not in report
        assert "Days until exhaustion" not in report

    def test_spending_without_vendors(self):
        sp = SpendingResult(
            category_breakdown=[CategorySpend("A", 100, 100.0)],
            total_spend=100,
            top_categories=[CategorySpend("A", 100, 100.0)],
            month_over_month=None,
            top_vendors=None,
            summary="test",
        )
        report = format_budget_report(spending=sp)
        assert "Top vendors" not in report


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_budget_vs_actual_with_float_strings(self):
        rows = [
            {"cat": "A", "budget": "1500.50", "actual": "1600.75"},
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        assert result.categories[0].budget == 1500.5
        assert result.categories[0].actual == 1600.75

    def test_burn_rate_same_day(self):
        rows = [
            {"amount": 100, "date": "2024-01-01"},
            {"amount": 200, "date": "2024-01-01"},
        ]
        result = compute_burn_rate(rows, "amount", "date")
        assert result is not None
        assert result.num_days == 1
        assert result.total_spend == 300.0

    def test_spending_with_all_options(self):
        rows = [
            {"cat": "A", "amount": 100, "date": "2024-01-15", "vendor": "V1"},
            {"cat": "A", "amount": 150, "date": "2024-02-15", "vendor": "V2"},
            {"cat": "B", "amount": 200, "date": "2024-01-20", "vendor": "V1"},
        ]
        result = analyze_spending_patterns(
            rows, "cat", "amount", date_column="date", vendor_column="vendor"
        )
        assert result is not None
        assert result.month_over_month is not None
        assert result.top_vendors is not None
        assert result.total_spend == 450.0

    def test_forecast_increasing_trend(self):
        rows = [
            {"amount": 100, "date": "2024-01-15"},
            {"amount": 200, "date": "2024-02-15"},
            {"amount": 300, "date": "2024-03-15"},
            {"amount": 400, "date": "2024-04-15"},
        ]
        result = forecast_budget(rows, "amount", "date", periods_ahead=2)
        assert len(result) == 2
        # Should project continued increase
        assert result[0].projected_amount > 400

    def test_forecast_decreasing_trend(self):
        rows = [
            {"amount": 400, "date": "2024-01-15"},
            {"amount": 300, "date": "2024-02-15"},
            {"amount": 200, "date": "2024-03-15"},
            {"amount": 100, "date": "2024-04-15"},
        ]
        result = forecast_budget(rows, "amount", "date", periods_ahead=1)
        assert len(result) == 1
        # Should project decrease, but not below zero
        assert result[0].projected_amount >= 0

    def test_burn_rate_projected_end_date_format(self):
        rows = [
            {"amount": 100, "date": "2024-01-01"},
            {"amount": 100, "date": "2024-01-31"},
        ]
        result = compute_burn_rate(rows, "amount", "date", total_budget=500)
        assert result is not None
        assert result.projected_end_date is not None
        # Should be a valid date string
        datetime.strptime(result.projected_end_date, "%Y-%m-%d")

    def test_month_over_month_zero_prev(self):
        rows = [
            {"cat": "A", "amount": 0, "date": "2024-01-15"},
            {"cat": "A", "amount": 100, "date": "2024-02-15"},
        ]
        result = analyze_spending_patterns(rows, "cat", "amount", date_column="date")
        assert result is not None
        assert result.month_over_month is not None
        # Change from 0 should give 0% to avoid division error
        second = result.month_over_month[1]
        assert second.change_pct == 0.0

    def test_budget_vs_actual_sorted_by_variance(self):
        rows = [
            {"cat": "A", "budget": 100, "actual": 110},  # variance +10
            {"cat": "B", "budget": 100, "actual": 150},  # variance +50
            {"cat": "C", "budget": 100, "actual": 80},   # variance -20
        ]
        result = budget_vs_actual(rows, "cat", "budget", "actual")
        assert result is not None
        # Sorted by variance descending (most over-budget first)
        assert result.categories[0].category == "B"
        assert result.categories[1].category == "A"
        assert result.categories[2].category == "C"
