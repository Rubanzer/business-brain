"""Tests for revenue_forecaster module."""

from business_brain.discovery.revenue_forecaster import (
    PeriodRevenue,
    RevenueForecastResult,
    SegmentRevenue,
    RevenueSegmentResult,
    GrowthPeriod,
    RevenueGrowthResult,
    DriverCorrelation,
    RevenueDriverResult,
    _safe_float,
    _parse_date,
    forecast_revenue,
    analyze_revenue_segments,
    compute_revenue_growth,
    analyze_revenue_drivers,
    format_revenue_report,
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
        assert _safe_float(True) == 1.0

    def test_list_returns_none(self):
        assert _safe_float([1, 2]) is None

    def test_comma_separated(self):
        assert _safe_float("1,000.50") == 1000.50

    def test_whitespace(self):
        assert _safe_float("  42  ") == 42.0


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
# forecast_revenue
# ---------------------------------------------------------------------------


class TestForecastRevenue:
    def test_empty_rows(self):
        assert forecast_revenue([], "revenue", "date") is None

    def test_zero_periods_ahead(self):
        rows = [{"revenue": 100, "date": "2024-01-15"}]
        assert forecast_revenue(rows, "revenue", "date", periods_ahead=0) is None

    def test_negative_periods_ahead(self):
        rows = [{"revenue": 100, "date": "2024-01-15"}]
        assert forecast_revenue(rows, "revenue", "date", periods_ahead=-1) is None

    def test_all_null_data(self):
        rows = [{"revenue": None, "date": None}]
        assert forecast_revenue(rows, "revenue", "date") is None

    def test_all_invalid_data(self):
        rows = [{"revenue": "abc", "date": "not-a-date"}]
        assert forecast_revenue(rows, "revenue", "date") is None

    def test_missing_columns(self):
        rows = [{"x": 100, "y": "2024-01-15"}]
        assert forecast_revenue(rows, "revenue", "date") is None

    def test_single_period(self):
        rows = [{"revenue": 1000, "date": "2024-01-15"}]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=2)
        assert result is not None
        assert len(result.periods) == 1
        assert result.periods[0].growth_rate is None
        assert len(result.forecasts) == 2
        assert result.avg_growth_rate == 0.0
        assert result.total_historical == 1000.0

    def test_two_periods_growth_rate(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=1)
        assert result is not None
        assert len(result.periods) == 2
        assert result.periods[0].growth_rate is None
        assert result.periods[1].growth_rate == 20.0  # (1200-1000)/1000*100

    def test_multiple_periods(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1100, "date": "2024-02-15"},
            {"revenue": 1200, "date": "2024-03-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=2)
        assert result is not None
        assert len(result.periods) == 3
        assert len(result.forecasts) == 2
        assert result.total_historical == 3300.0

    def test_trend_increasing(self):
        rows = [
            {"revenue": 100, "date": "2024-01-15"},
            {"revenue": 200, "date": "2024-02-15"},
            {"revenue": 300, "date": "2024-03-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date")
        assert result is not None
        assert result.trend == "increasing"

    def test_trend_decreasing(self):
        rows = [
            {"revenue": 300, "date": "2024-01-15"},
            {"revenue": 200, "date": "2024-02-15"},
            {"revenue": 100, "date": "2024-03-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date")
        assert result is not None
        assert result.trend == "decreasing"

    def test_trend_stable(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1000, "date": "2024-02-15"},
            {"revenue": 1000, "date": "2024-03-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date")
        assert result is not None
        assert result.trend == "stable"

    def test_forecast_period_labels(self):
        rows = [
            {"revenue": 100, "date": "2024-10-15"},
            {"revenue": 200, "date": "2024-11-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=3)
        labels = [f.period for f in result.forecasts]
        assert labels == ["2024-12", "2025-01", "2025-02"]

    def test_forecast_year_wrap(self):
        rows = [
            {"revenue": 100, "date": "2024-11-15"},
            {"revenue": 200, "date": "2024-12-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=2)
        labels = [f.period for f in result.forecasts]
        assert labels == ["2025-01", "2025-02"]

    def test_aggregation_within_month(self):
        rows = [
            {"revenue": 500, "date": "2024-01-10"},
            {"revenue": 300, "date": "2024-01-20"},
            {"revenue": 1000, "date": "2024-02-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=1)
        assert result is not None
        assert len(result.periods) == 2
        assert result.periods[0].revenue == 800.0  # 500+300
        assert result.periods[1].revenue == 1000.0

    def test_string_numeric_values(self):
        rows = [
            {"revenue": "1000", "date": "2024-01-15"},
            {"revenue": "1200", "date": "2024-02-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=1)
        assert result is not None
        assert result.total_historical == 2200.0

    def test_comma_separated_values(self):
        rows = [
            {"revenue": "1,000", "date": "2024-01-15"},
            {"revenue": "2,500", "date": "2024-02-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=1)
        assert result is not None
        assert result.total_historical == 3500.0

    def test_mixed_valid_invalid_rows(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": "bad", "date": "2024-02-15"},
            {"revenue": 1200, "date": "2024-03-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=1)
        assert result is not None
        assert len(result.periods) == 2
        assert result.total_historical == 2200.0

    def test_forecast_non_negative(self):
        # Decreasing trend should not produce negative forecasts
        rows = [
            {"revenue": 300, "date": "2024-01-15"},
            {"revenue": 200, "date": "2024-02-15"},
            {"revenue": 100, "date": "2024-03-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=5)
        assert result is not None
        for f in result.forecasts:
            assert f.revenue >= 0.0

    def test_avg_growth_rate(self):
        rows = [
            {"revenue": 100, "date": "2024-01-15"},
            {"revenue": 150, "date": "2024-02-15"},  # +50%
            {"revenue": 225, "date": "2024-03-15"},  # +50%
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=1)
        assert result is not None
        assert result.avg_growth_rate == 50.0

    def test_total_forecast(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1000, "date": "2024-02-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=2)
        assert result is not None
        assert result.total_forecast == sum(f.revenue for f in result.forecasts)

    def test_summary_content(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date")
        assert "Trend" in result.summary
        assert "forecast" in result.summary.lower()

    def test_zero_revenue_growth_rate(self):
        rows = [
            {"revenue": 0, "date": "2024-01-15"},
            {"revenue": 100, "date": "2024-02-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=1)
        assert result is not None
        # growth rate from 0 -> 100 should be 0.0 (division guard)
        assert result.periods[1].growth_rate == 0.0

    def test_all_same_revenues(self):
        rows = [
            {"revenue": 500, "date": "2024-01-15"},
            {"revenue": 500, "date": "2024-02-15"},
            {"revenue": 500, "date": "2024-03-15"},
        ]
        result = forecast_revenue(rows, "revenue", "date", periods_ahead=2)
        assert result is not None
        assert result.avg_growth_rate == 0.0
        assert result.trend == "stable"


# ---------------------------------------------------------------------------
# analyze_revenue_segments
# ---------------------------------------------------------------------------


class TestAnalyzeRevenueSegments:
    def test_empty_rows(self):
        assert analyze_revenue_segments([], "revenue", "segment") is None

    def test_all_null_data(self):
        rows = [{"revenue": None, "segment": None}]
        assert analyze_revenue_segments(rows, "revenue", "segment") is None

    def test_all_invalid_data(self):
        rows = [{"revenue": "abc", "segment": "A"}]
        assert analyze_revenue_segments(rows, "revenue", "segment") is None

    def test_missing_revenue_column(self):
        rows = [{"x": 100, "segment": "A"}]
        assert analyze_revenue_segments(rows, "revenue", "segment") is None

    def test_missing_segment_column(self):
        rows = [{"revenue": 100, "x": "A"}]
        assert analyze_revenue_segments(rows, "revenue", "segment") is None

    def test_null_segment_values(self):
        rows = [
            {"revenue": 100, "segment": None},
            {"revenue": 200, "segment": "A"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        assert result is not None
        assert len(result.segments) == 1
        assert result.segments[0].segment == "A"

    def test_single_segment(self):
        rows = [
            {"revenue": 500, "segment": "A"},
            {"revenue": 300, "segment": "A"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        assert result is not None
        assert len(result.segments) == 1
        assert result.segments[0].revenue == 800.0
        assert result.segments[0].share_pct == 100.0
        assert result.segments[0].rank == 1
        assert result.segments[0].transaction_count == 2
        assert result.top_segment == "A"

    def test_multiple_segments_ranking(self):
        rows = [
            {"revenue": 500, "segment": "A"},
            {"revenue": 300, "segment": "B"},
            {"revenue": 200, "segment": "C"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        assert result is not None
        assert len(result.segments) == 3
        assert result.segments[0].segment == "A"
        assert result.segments[0].rank == 1
        assert result.segments[1].segment == "B"
        assert result.segments[1].rank == 2
        assert result.segments[2].segment == "C"
        assert result.segments[2].rank == 3

    def test_share_pct_sum(self):
        rows = [
            {"revenue": 600, "segment": "A"},
            {"revenue": 300, "segment": "B"},
            {"revenue": 100, "segment": "C"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        total_share = sum(s.share_pct for s in result.segments)
        assert abs(total_share - 100.0) < 0.01

    def test_total_revenue(self):
        rows = [
            {"revenue": 500, "segment": "A"},
            {"revenue": 300, "segment": "B"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        assert result.total_revenue == 800.0

    def test_concentration_index_single_segment(self):
        rows = [{"revenue": 1000, "segment": "A"}]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        # HHI = 1.0^2 = 1.0 for single segment
        assert result.concentration_index == 1.0

    def test_concentration_index_equal_segments(self):
        rows = [
            {"revenue": 250, "segment": "A"},
            {"revenue": 250, "segment": "B"},
            {"revenue": 250, "segment": "C"},
            {"revenue": 250, "segment": "D"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        # HHI = 4 * (0.25)^2 = 0.25
        assert result.concentration_index == 0.25

    def test_string_numeric_revenue(self):
        rows = [{"revenue": "1000", "segment": "A"}]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        assert result is not None
        assert result.total_revenue == 1000.0

    def test_top_segment_correct(self):
        rows = [
            {"revenue": 100, "segment": "Small"},
            {"revenue": 5000, "segment": "Big"},
            {"revenue": 500, "segment": "Medium"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        assert result.top_segment == "Big"

    def test_summary_content(self):
        rows = [
            {"revenue": 500, "segment": "A"},
            {"revenue": 300, "segment": "B"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        assert "segment" in result.summary.lower()
        assert result.top_segment in result.summary

    def test_with_date_column(self):
        rows = [
            {"revenue": 500, "segment": "A", "date": "2024-01-15"},
            {"revenue": 300, "segment": "B", "date": "2024-02-15"},
        ]
        result = analyze_revenue_segments(
            rows, "revenue", "segment", date_column="date"
        )
        assert result is not None
        assert result.total_revenue == 800.0

    def test_mixed_valid_invalid_revenue(self):
        rows = [
            {"revenue": 500, "segment": "A"},
            {"revenue": "bad", "segment": "B"},
            {"revenue": 300, "segment": "C"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        assert result is not None
        assert result.total_revenue == 800.0
        assert len(result.segments) == 2

    def test_transaction_count(self):
        rows = [
            {"revenue": 100, "segment": "A"},
            {"revenue": 200, "segment": "A"},
            {"revenue": 300, "segment": "A"},
            {"revenue": 400, "segment": "B"},
        ]
        result = analyze_revenue_segments(rows, "revenue", "segment")
        seg_a = next(s for s in result.segments if s.segment == "A")
        seg_b = next(s for s in result.segments if s.segment == "B")
        assert seg_a.transaction_count == 3
        assert seg_b.transaction_count == 1


# ---------------------------------------------------------------------------
# compute_revenue_growth
# ---------------------------------------------------------------------------


class TestComputeRevenueGrowth:
    def test_empty_rows(self):
        assert compute_revenue_growth([], "revenue", "date") is None

    def test_all_null_data(self):
        rows = [{"revenue": None, "date": None}]
        assert compute_revenue_growth(rows, "revenue", "date") is None

    def test_all_invalid_data(self):
        rows = [{"revenue": "abc", "date": "not-a-date"}]
        assert compute_revenue_growth(rows, "revenue", "date") is None

    def test_missing_columns(self):
        rows = [{"x": 100, "y": "2024-01-15"}]
        assert compute_revenue_growth(rows, "revenue", "date") is None

    def test_single_period(self):
        rows = [{"revenue": 1000, "date": "2024-01-15"}]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert len(result.periods) == 0
        assert result.cagr is None
        assert result.avg_growth == 0.0
        assert result.volatility == 0.0

    def test_two_periods_positive_growth(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert len(result.periods) == 1
        assert result.periods[0].growth_rate == 20.0
        assert result.periods[0].growth_absolute == 200.0
        assert result.avg_growth == 20.0

    def test_two_periods_negative_growth(self):
        rows = [
            {"revenue": 1200, "date": "2024-01-15"},
            {"revenue": 1000, "date": "2024-02-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert result.periods[0].growth_rate < 0
        assert result.periods[0].growth_absolute == -200.0

    def test_best_worst_period(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1500, "date": "2024-02-15"},  # +50%
            {"revenue": 1200, "date": "2024-03-15"},  # -20%
            {"revenue": 1800, "date": "2024-04-15"},  # +50%
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert result.best_period == "2024-02" or result.best_period == "2024-04"
        assert result.worst_period == "2024-03"

    def test_cagr_with_multiple_years(self):
        rows = [
            {"revenue": 1000, "date": "2023-01-15"},
            {"revenue": 1100, "date": "2023-06-15"},
            {"revenue": 1200, "date": "2024-01-15"},
            {"revenue": 1300, "date": "2024-06-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert result.cagr is not None

    def test_cagr_none_for_short_span(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-06-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        # Less than 1 year span => cagr should be None
        assert result.cagr is None

    def test_volatility_zero_constant_growth(self):
        rows = [
            {"revenue": 100, "date": "2024-01-15"},
            {"revenue": 200, "date": "2024-02-15"},  # +100%
            {"revenue": 400, "date": "2024-03-15"},  # +100%
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert result.volatility == 0.0

    def test_volatility_positive(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1500, "date": "2024-02-15"},  # +50%
            {"revenue": 1200, "date": "2024-03-15"},  # -20%
            {"revenue": 1800, "date": "2024-04-15"},  # +50%
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert result.volatility > 0

    def test_aggregation_within_month(self):
        rows = [
            {"revenue": 500, "date": "2024-01-10"},
            {"revenue": 500, "date": "2024-01-20"},
            {"revenue": 1500, "date": "2024-02-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert len(result.periods) == 1
        # growth from 1000 (Jan) to 1500 (Feb) = 50%
        assert result.periods[0].growth_rate == 50.0

    def test_string_numeric_values(self):
        rows = [
            {"revenue": "1000", "date": "2024-01-15"},
            {"revenue": "1200", "date": "2024-02-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert result.periods[0].revenue == 1200.0

    def test_zero_revenue_period(self):
        rows = [
            {"revenue": 0, "date": "2024-01-15"},
            {"revenue": 500, "date": "2024-02-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        # Division by zero guard: growth_rate = 0.0
        assert result.periods[0].growth_rate == 0.0

    def test_summary_content(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert "growth" in result.summary.lower()
        assert "CAGR" in result.summary

    def test_all_same_revenues(self):
        rows = [
            {"revenue": 500, "date": "2024-01-15"},
            {"revenue": 500, "date": "2024-02-15"},
            {"revenue": 500, "date": "2024-03-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert result.avg_growth == 0.0
        assert result.volatility == 0.0

    def test_mixed_valid_invalid_rows(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": "bad", "date": "2024-02-15"},
            {"revenue": 1500, "date": "2024-03-15"},
        ]
        result = compute_revenue_growth(rows, "revenue", "date")
        assert result is not None
        assert len(result.periods) == 1


# ---------------------------------------------------------------------------
# analyze_revenue_drivers
# ---------------------------------------------------------------------------


class TestAnalyzeRevenueDrivers:
    def test_empty_rows(self):
        assert analyze_revenue_drivers([], "revenue", ["driver1"]) is None

    def test_empty_driver_columns(self):
        rows = [{"revenue": 100, "driver1": 10}]
        assert analyze_revenue_drivers(rows, "revenue", []) is None

    def test_all_null_revenue(self):
        rows = [
            {"revenue": None, "driver1": 10},
            {"revenue": None, "driver1": 20},
        ]
        assert analyze_revenue_drivers(rows, "revenue", ["driver1"]) is None

    def test_single_row(self):
        rows = [{"revenue": 100, "driver1": 10}]
        assert analyze_revenue_drivers(rows, "revenue", ["driver1"]) is None

    def test_missing_revenue_column(self):
        rows = [
            {"x": 100, "driver1": 10},
            {"x": 200, "driver1": 20},
        ]
        assert analyze_revenue_drivers(rows, "revenue", ["driver1"]) is None

    def test_all_null_driver(self):
        rows = [
            {"revenue": 100, "driver1": None},
            {"revenue": 200, "driver1": None},
        ]
        assert analyze_revenue_drivers(rows, "revenue", ["driver1"]) is None

    def test_positive_correlation(self):
        rows = [
            {"revenue": 100, "driver1": 10},
            {"revenue": 200, "driver1": 20},
            {"revenue": 300, "driver1": 30},
            {"revenue": 400, "driver1": 40},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert result is not None
        assert len(result.drivers) == 1
        assert result.drivers[0].correlation > 0.9
        assert result.drivers[0].direction == "positive"

    def test_negative_correlation(self):
        rows = [
            {"revenue": 100, "driver1": 40},
            {"revenue": 200, "driver1": 30},
            {"revenue": 300, "driver1": 20},
            {"revenue": 400, "driver1": 10},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert result is not None
        assert result.drivers[0].correlation < -0.9
        assert result.drivers[0].direction == "negative"

    def test_no_correlation(self):
        rows = [
            {"revenue": 100, "driver1": 50},
            {"revenue": 200, "driver1": 50},
            {"revenue": 300, "driver1": 50},
            {"revenue": 400, "driver1": 50},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert result is not None
        assert result.drivers[0].correlation == 0.0
        assert result.drivers[0].direction == "neutral"

    def test_multiple_drivers_ranking(self):
        rows = [
            {"revenue": 100, "d1": 10, "d2": 50, "d3": 30},
            {"revenue": 200, "d1": 20, "d2": 45, "d3": 25},
            {"revenue": 300, "d1": 30, "d2": 55, "d3": 20},
            {"revenue": 400, "d1": 40, "d2": 60, "d3": 15},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["d1", "d2", "d3"])
        assert result is not None
        assert len(result.drivers) == 3
        # d1 has perfect positive correlation => should be top
        assert result.top_driver == "d1"
        # Drivers sorted by abs(correlation) descending
        assert abs(result.drivers[0].correlation) >= abs(result.drivers[1].correlation)
        assert abs(result.drivers[1].correlation) >= abs(result.drivers[2].correlation)

    def test_avg_when_high_low_revenue(self):
        rows = [
            {"revenue": 100, "driver1": 5},
            {"revenue": 200, "driver1": 10},
            {"revenue": 300, "driver1": 15},
            {"revenue": 400, "driver1": 20},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert result is not None
        # median = (200+300)/2 = 250
        # high revenue rows (>=250): rev=300 (d=15), rev=400 (d=20) -> avg=17.5
        # low revenue rows (<250): rev=100 (d=5), rev=200 (d=10) -> avg=7.5
        assert result.drivers[0].avg_when_high_revenue == 17.5
        assert result.drivers[0].avg_when_low_revenue == 7.5

    def test_string_numeric_driver(self):
        rows = [
            {"revenue": 100, "driver1": "10"},
            {"revenue": 200, "driver1": "20"},
            {"revenue": 300, "driver1": "30"},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert result is not None
        assert result.drivers[0].correlation > 0.9

    def test_mixed_valid_invalid_driver(self):
        rows = [
            {"revenue": 100, "driver1": 10},
            {"revenue": 200, "driver1": "bad"},
            {"revenue": 300, "driver1": 30},
            {"revenue": 400, "driver1": 40},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert result is not None
        # Should still compute correlation on valid pairs

    def test_summary_content(self):
        rows = [
            {"revenue": 100, "driver1": 10},
            {"revenue": 200, "driver1": 20},
            {"revenue": 300, "driver1": 30},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert "driver" in result.summary.lower()
        assert result.top_driver in result.summary

    def test_driver_with_partial_data(self):
        # One driver has data, another doesn't
        rows = [
            {"revenue": 100, "d1": 10},
            {"revenue": 200, "d1": 20},
            {"revenue": 300, "d1": 30},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["d1", "d2"])
        assert result is not None
        assert len(result.drivers) == 1
        assert result.drivers[0].driver == "d1"

    def test_constant_revenue_zero_correlation(self):
        rows = [
            {"revenue": 100, "driver1": 10},
            {"revenue": 100, "driver1": 20},
            {"revenue": 100, "driver1": 30},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert result is not None
        assert result.drivers[0].correlation == 0.0

    def test_constant_driver_zero_correlation(self):
        rows = [
            {"revenue": 100, "driver1": 50},
            {"revenue": 200, "driver1": 50},
            {"revenue": 300, "driver1": 50},
        ]
        result = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        assert result is not None
        assert result.drivers[0].correlation == 0.0


# ---------------------------------------------------------------------------
# format_revenue_report
# ---------------------------------------------------------------------------


class TestFormatRevenueReport:
    def test_no_data(self):
        report = format_revenue_report()
        assert report == "No revenue data available for report."

    def test_forecast_only(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        fc = forecast_revenue(rows, "revenue", "date", periods_ahead=2)
        report = format_revenue_report(forecast=fc)
        assert "Revenue Forecast" in report
        assert "Revenue Segments" not in report
        assert "Revenue Growth" not in report
        assert "Revenue Drivers" not in report

    def test_segments_only(self):
        rows = [
            {"revenue": 500, "segment": "A"},
            {"revenue": 300, "segment": "B"},
        ]
        seg = analyze_revenue_segments(rows, "revenue", "segment")
        report = format_revenue_report(segments=seg)
        assert "Revenue Segments" in report
        assert "Revenue Forecast" not in report

    def test_growth_only(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        gr = compute_revenue_growth(rows, "revenue", "date")
        report = format_revenue_report(growth=gr)
        assert "Revenue Growth" in report
        assert "Revenue Forecast" not in report

    def test_drivers_only(self):
        rows = [
            {"revenue": 100, "driver1": 10},
            {"revenue": 200, "driver1": 20},
            {"revenue": 300, "driver1": 30},
        ]
        drv = analyze_revenue_drivers(rows, "revenue", ["driver1"])
        report = format_revenue_report(drivers=drv)
        assert "Revenue Drivers" in report
        assert "Revenue Forecast" not in report

    def test_all_sections_combined(self):
        fc_rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        seg_rows = [
            {"revenue": 500, "segment": "A"},
            {"revenue": 300, "segment": "B"},
        ]
        gr_rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        drv_rows = [
            {"revenue": 100, "d1": 10},
            {"revenue": 200, "d1": 20},
            {"revenue": 300, "d1": 30},
        ]

        fc = forecast_revenue(fc_rows, "revenue", "date", periods_ahead=2)
        seg = analyze_revenue_segments(seg_rows, "revenue", "segment")
        gr = compute_revenue_growth(gr_rows, "revenue", "date")
        drv = analyze_revenue_drivers(drv_rows, "revenue", ["d1"])

        report = format_revenue_report(
            forecast=fc, segments=seg, growth=gr, drivers=drv
        )
        assert "Revenue Forecast" in report
        assert "Revenue Segments" in report
        assert "Revenue Growth" in report
        assert "Revenue Drivers" in report

    def test_forecast_section_has_historical_and_forecasted(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        fc = forecast_revenue(rows, "revenue", "date", periods_ahead=2)
        report = format_revenue_report(forecast=fc)
        assert "Historical Periods" in report
        assert "Forecasted Periods" in report
        assert "2024-01" in report
        assert "2024-02" in report

    def test_segments_section_has_breakdown(self):
        rows = [
            {"revenue": 500, "segment": "Alpha"},
            {"revenue": 300, "segment": "Beta"},
        ]
        seg = analyze_revenue_segments(rows, "revenue", "segment")
        report = format_revenue_report(segments=seg)
        assert "Segment Breakdown" in report
        assert "Alpha" in report
        assert "Beta" in report
        assert "share=" in report

    def test_growth_section_has_cagr(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        gr = compute_revenue_growth(rows, "revenue", "date")
        report = format_revenue_report(growth=gr)
        assert "CAGR" in report
        assert "Volatility" in report
        assert "Period Growth" in report

    def test_drivers_section_has_correlations(self):
        rows = [
            {"revenue": 100, "d1": 10},
            {"revenue": 200, "d1": 20},
            {"revenue": 300, "d1": 30},
        ]
        drv = analyze_revenue_drivers(rows, "revenue", ["d1"])
        report = format_revenue_report(drivers=drv)
        assert "Driver Correlations" in report
        assert "correlation=" in report
        assert "direction=" in report

    def test_growth_section_absolute_values(self):
        rows = [
            {"revenue": 1000, "date": "2024-01-15"},
            {"revenue": 1200, "date": "2024-02-15"},
        ]
        gr = compute_revenue_growth(rows, "revenue", "date")
        report = format_revenue_report(growth=gr)
        assert "+200.00" in report


# ---------------------------------------------------------------------------
# Dataclass instantiation tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_period_revenue(self):
        pr = PeriodRevenue(period="2024-01", revenue=1000.0, growth_rate=5.0)
        assert pr.period == "2024-01"
        assert pr.revenue == 1000.0
        assert pr.growth_rate == 5.0

    def test_period_revenue_none_growth(self):
        pr = PeriodRevenue(period="2024-01", revenue=1000.0, growth_rate=None)
        assert pr.growth_rate is None

    def test_revenue_forecast_result(self):
        rfr = RevenueForecastResult(
            periods=[], forecasts=[], avg_growth_rate=5.0,
            trend="increasing", total_historical=10000.0,
            total_forecast=3000.0, summary="test"
        )
        assert rfr.trend == "increasing"
        assert rfr.total_historical == 10000.0

    def test_segment_revenue(self):
        sr = SegmentRevenue(
            segment="A", revenue=5000.0, share_pct=50.0,
            rank=1, transaction_count=10
        )
        assert sr.segment == "A"
        assert sr.share_pct == 50.0

    def test_revenue_segment_result(self):
        rsr = RevenueSegmentResult(
            segments=[], total_revenue=10000.0, top_segment="A",
            concentration_index=0.5, summary="test"
        )
        assert rsr.top_segment == "A"
        assert rsr.concentration_index == 0.5

    def test_growth_period(self):
        gp = GrowthPeriod(
            period="2024-02", revenue=1200.0,
            growth_rate=20.0, growth_absolute=200.0
        )
        assert gp.growth_rate == 20.0
        assert gp.growth_absolute == 200.0

    def test_revenue_growth_result(self):
        rgr = RevenueGrowthResult(
            periods=[], cagr=15.0, avg_growth=10.0,
            best_period="2024-03", worst_period="2024-01",
            volatility=5.0, summary="test"
        )
        assert rgr.cagr == 15.0
        assert rgr.best_period == "2024-03"

    def test_revenue_growth_result_none_cagr(self):
        rgr = RevenueGrowthResult(
            periods=[], cagr=None, avg_growth=0.0,
            best_period="2024-01", worst_period="2024-01",
            volatility=0.0, summary="test"
        )
        assert rgr.cagr is None

    def test_driver_correlation(self):
        dc = DriverCorrelation(
            driver="marketing_spend", correlation=0.85,
            direction="positive", avg_when_high_revenue=1000.0,
            avg_when_low_revenue=500.0
        )
        assert dc.driver == "marketing_spend"
        assert dc.correlation == 0.85
        assert dc.direction == "positive"

    def test_revenue_driver_result(self):
        rdr = RevenueDriverResult(
            drivers=[], top_driver="marketing_spend",
            summary="test"
        )
        assert rdr.top_driver == "marketing_spend"
