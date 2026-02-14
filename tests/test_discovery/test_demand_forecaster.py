"""Tests for demand forecaster pure functions."""

from business_brain.discovery.demand_forecaster import (
    DemandPatternResult,
    MovingAvgPoint,
    MovingAvgResult,
    ProductDemand,
    SeasonalPeriod,
    SeasonalityResult,
    SmoothingPoint,
    SmoothingResult,
    analyze_demand_pattern,
    compute_moving_average,
    detect_demand_seasonality,
    exponential_smoothing,
    format_demand_report,
)


# ---------------------------------------------------------------------------
# Helper data builders
# ---------------------------------------------------------------------------

def _make_rows(product, periods_qty):
    """Build rows from a list of (period, qty) tuples for a single product."""
    return [
        {"product": product, "period": p, "qty": q}
        for p, q in periods_qty
    ]


def _make_multi_product_rows(product_data):
    """Build rows for multiple products.
    product_data: dict of {product: [(period, qty), ...]}
    """
    rows = []
    for product, pq_list in product_data.items():
        rows.extend(_make_rows(product, pq_list))
    return rows


# ---------------------------------------------------------------------------
# analyze_demand_pattern
# ---------------------------------------------------------------------------

class TestAnalyzeDemandPattern:
    def test_empty_rows_returns_none(self):
        result = analyze_demand_pattern([], "product", "qty", "period")
        assert result is None

    def test_single_product_steady(self):
        # cv < 20% -> steady
        rows = _make_rows("A", [
            ("2024-01", 100), ("2024-02", 102), ("2024-03", 98),
            ("2024-04", 101), ("2024-05", 99),
        ])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert len(result.products) == 1
        p = result.products[0]
        assert p.product == "A"
        assert p.pattern == "steady"
        assert p.cv < 20

    def test_variable_pattern(self):
        # cv between 20% and 50%
        rows = _make_rows("B", [
            ("2024-01", 100), ("2024-02", 70), ("2024-03", 130),
            ("2024-04", 80), ("2024-05", 120),
        ])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        p = result.products[0]
        assert p.pattern == "variable"
        assert 20 <= p.cv < 50

    def test_lumpy_pattern(self):
        # cv between 50% and 80%
        rows = _make_rows("C", [
            ("2024-01", 200), ("2024-02", 50), ("2024-03", 180),
            ("2024-04", 30), ("2024-05", 150),
        ])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        p = result.products[0]
        assert p.pattern == "lumpy"
        assert 50 <= p.cv < 80

    def test_erratic_pattern(self):
        # cv >= 80%
        rows = _make_rows("D", [
            ("2024-01", 500), ("2024-02", 10), ("2024-03", 400),
            ("2024-04", 5), ("2024-05", 300),
        ])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        p = result.products[0]
        assert p.pattern == "erratic"
        assert p.cv >= 80

    def test_total_demand_computed(self):
        rows = _make_rows("A", [("Q1", 100), ("Q2", 200), ("Q3", 300)])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].total_demand == 600

    def test_avg_per_period(self):
        rows = _make_rows("A", [("Q1", 100), ("Q2", 200), ("Q3", 300)])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].avg_per_period == 200.0

    def test_max_min_period(self):
        rows = _make_rows("A", [("Q1", 50), ("Q2", 200), ("Q3", 100)])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].max_period_demand == 200
        assert result.products[0].min_period_demand == 50

    def test_increasing_trend(self):
        rows = _make_rows("A", [
            ("M01", 100), ("M02", 120), ("M03", 140),
            ("M04", 160), ("M05", 180),
        ])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].trend == "increasing"

    def test_decreasing_trend(self):
        rows = _make_rows("A", [
            ("M01", 200), ("M02", 180), ("M03", 160),
            ("M04", 140), ("M05", 120),
        ])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].trend == "decreasing"

    def test_stable_trend(self):
        rows = _make_rows("A", [
            ("M01", 100), ("M02", 100), ("M03", 100),
            ("M04", 100), ("M05", 100),
        ])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].trend == "stable"

    def test_adi_for_intermittent_demand(self):
        # 2 out of 5 periods have zero demand -> ADI = 5/3
        rows = _make_rows("A", [
            ("M01", 100), ("M02", 0), ("M03", 50),
            ("M04", 0), ("M05", 80),
        ])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].adi is not None
        assert abs(result.products[0].adi - 5.0 / 3.0) < 0.01

    def test_adi_none_when_no_zeros(self):
        rows = _make_rows("A", [("M01", 100), ("M02", 200), ("M03", 150)])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].adi is None

    def test_multi_product(self):
        rows = _make_multi_product_rows({
            "Widget": [("Q1", 100), ("Q2", 110), ("Q3", 105)],
            "Gadget": [("Q1", 50), ("Q2", 200), ("Q3", 30)],
        })
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert len(result.products) == 2
        names = {p.product for p in result.products}
        assert "Widget" in names
        assert "Gadget" in names

    def test_overall_trend(self):
        rows = _make_multi_product_rows({
            "A": [("M1", 100), ("M2", 120), ("M3", 140)],
            "B": [("M1", 50), ("M2", 70), ("M3", 90)],
        })
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.overall_trend == "increasing"

    def test_none_values_skipped(self):
        rows = [
            {"product": "A", "period": "Q1", "qty": 100},
            {"product": None, "period": "Q2", "qty": 200},
            {"product": "A", "period": None, "qty": 300},
            {"product": "A", "period": "Q3", "qty": None},
        ]
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].periods == 1  # only Q1 is valid for A

    def test_all_invalid_returns_none(self):
        rows = [{"product": None, "period": "Q1", "qty": 100}]
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is None

    def test_non_numeric_qty_skipped(self):
        rows = [
            {"product": "A", "period": "Q1", "qty": "abc"},
            {"product": "A", "period": "Q2", "qty": 100},
        ]
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert result.products[0].periods == 1

    def test_summary_contains_product_names(self):
        rows = _make_rows("TestProd", [("Q1", 100), ("Q2", 200)])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        assert "TestProd" in result.summary

    def test_single_period_product(self):
        rows = _make_rows("A", [("Q1", 100)])
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        p = result.products[0]
        assert p.periods == 1
        assert p.std_dev == 0.0
        assert p.cv == 0.0
        assert p.pattern == "steady"

    def test_duplicate_period_rows_summed(self):
        rows = [
            {"product": "A", "period": "Q1", "qty": 50},
            {"product": "A", "period": "Q1", "qty": 60},
            {"product": "A", "period": "Q2", "qty": 100},
        ]
        result = analyze_demand_pattern(rows, "product", "qty", "period")
        assert result is not None
        p = result.products[0]
        assert p.periods == 2
        assert p.total_demand == 210  # 50+60+100


# ---------------------------------------------------------------------------
# compute_moving_average
# ---------------------------------------------------------------------------

class TestComputeMovingAverage:
    def test_empty_rows_returns_none(self):
        result = compute_moving_average([], "qty", "period")
        assert result is None

    def test_basic_window_3(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
            {"period": "M4", "qty": 130},
        ]
        result = compute_moving_average(rows, "qty", "period", window=3)
        assert result is not None
        assert len(result.points) == 4
        # First 2 have no MA
        assert result.points[0].moving_avg is None
        assert result.points[1].moving_avg is None
        # M3: avg(100,120,110) = 110
        assert abs(result.points[2].moving_avg - 110.0) < 0.01
        # M4: avg(120,110,130) = 120
        assert abs(result.points[3].moving_avg - 120.0) < 0.01

    def test_error_computed(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
        ]
        result = compute_moving_average(rows, "qty", "period", window=3)
        assert result is not None
        pt = result.points[2]
        assert pt.error is not None
        assert abs(pt.error - (110 - 110)) < 0.01

    def test_mad_computation(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
            {"period": "M4", "qty": 150},
        ]
        result = compute_moving_average(rows, "qty", "period", window=3)
        assert result is not None
        assert result.mad is not None
        # M3: MA=avg(100,120,110)=110, error=0
        # M4: MA=avg(120,110,150)=126.667, error=23.333
        # MAD = (0+23.333)/2 = 11.667
        assert abs(result.mad - 11.6667) < 0.01

    def test_mape_computation(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
            {"period": "M4", "qty": 150},
        ]
        result = compute_moving_average(rows, "qty", "period", window=3)
        assert result is not None
        assert result.mape is not None
        # M3: error=0, actual=110, pct=0%
        # M4: error=23.333, actual=150, pct=15.56%
        # MAPE = (0+15.56)/2 = 7.78%
        assert abs(result.mape - 7.78) < 0.01

    def test_window_1(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 200},
        ]
        result = compute_moving_average(rows, "qty", "period", window=1)
        assert result is not None
        # Every point should have MA = actual
        for pt in result.points:
            assert pt.moving_avg == pt.actual

    def test_window_equals_data_length(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 200},
            {"period": "M3", "qty": 300},
        ]
        result = compute_moving_average(rows, "qty", "period", window=3)
        assert result is not None
        assert result.points[0].moving_avg is None
        assert result.points[1].moving_avg is None
        assert abs(result.points[2].moving_avg - 200.0) < 0.01

    def test_invalid_window_returns_none(self):
        rows = [{"period": "M1", "qty": 100}]
        result = compute_moving_average(rows, "qty", "period", window=0)
        assert result is None

    def test_grouped_by_period(self):
        # Two rows in same period should be summed
        rows = [
            {"period": "M1", "qty": 50},
            {"period": "M1", "qty": 50},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
        ]
        result = compute_moving_average(rows, "qty", "period", window=3)
        assert result is not None
        assert len(result.points) == 3
        assert result.points[0].actual == 100.0  # 50+50

    def test_no_valid_data_returns_none(self):
        rows = [{"period": "M1", "qty": "bad"}]
        result = compute_moving_average(rows, "qty", "period")
        assert result is None


# ---------------------------------------------------------------------------
# exponential_smoothing
# ---------------------------------------------------------------------------

class TestExponentialSmoothing:
    def test_empty_rows_returns_none(self):
        result = exponential_smoothing([], "qty", "period")
        assert result is None

    def test_single_period_returns_none(self):
        rows = [{"period": "M1", "qty": 100}]
        result = exponential_smoothing(rows, "qty", "period")
        assert result is None

    def test_basic_smoothing(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
        ]
        result = exponential_smoothing(rows, "qty", "period", alpha=0.5)
        assert result is not None
        assert len(result.points) == 4  # 3 actual + 1 next
        # First forecast = first actual = 100
        assert result.points[0].forecast == 100.0
        # M2 forecast: 0.5*100 + 0.5*100 = 100
        assert abs(result.points[1].forecast - 100.0) < 0.01
        # M3 forecast: 0.5*120 + 0.5*100 = 110
        assert abs(result.points[2].forecast - 110.0) < 0.01

    def test_next_forecast_included(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
        ]
        result = exponential_smoothing(rows, "qty", "period", alpha=0.3)
        assert result is not None
        last = result.points[-1]
        assert last.period == "next"
        assert last.actual is None
        assert last.forecast is not None

    def test_alpha_stored(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
        ]
        result = exponential_smoothing(rows, "qty", "period", alpha=0.7)
        assert result is not None
        assert result.alpha == 0.7

    def test_optimal_alpha_found(self):
        # With strictly increasing data, higher alpha should be better
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 200},
            {"period": "M3", "qty": 300},
            {"period": "M4", "qty": 400},
            {"period": "M5", "qty": 500},
        ]
        result = exponential_smoothing(rows, "qty", "period", alpha=0.3)
        assert result is not None
        assert 0.1 <= result.optimal_alpha <= 0.9

    def test_mad_computed(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 100},
            {"period": "M3", "qty": 100},
        ]
        result = exponential_smoothing(rows, "qty", "period", alpha=0.3)
        assert result is not None
        # Constant demand -> forecast converges, MAD should be small
        assert result.mad is not None
        assert result.mad < 1.0

    def test_mape_computed(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
        ]
        result = exponential_smoothing(rows, "qty", "period", alpha=0.3)
        assert result is not None
        assert result.mape is not None
        assert result.mape >= 0

    def test_next_forecast_value(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 100},
        ]
        result = exponential_smoothing(rows, "qty", "period", alpha=0.5)
        assert result is not None
        # M2 forecast: 0.5*100 + 0.5*100 = 100
        # Next forecast: 0.5*100 + 0.5*100 = 100
        assert abs(result.next_forecast - 100.0) < 0.01

    def test_smoothing_with_alpha_1(self):
        # alpha=0.9 (closest to 1.0 in search) -> forecast = previous actual
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 200},
            {"period": "M3", "qty": 300},
        ]
        result = exponential_smoothing(rows, "qty", "period", alpha=0.9)
        assert result is not None
        # M2 forecast: 0.9*100 + 0.1*100 = 100
        assert abs(result.points[1].forecast - 100.0) < 0.01
        # M3 forecast: 0.9*200 + 0.1*100 = 190
        assert abs(result.points[2].forecast - 190.0) < 0.01


# ---------------------------------------------------------------------------
# detect_demand_seasonality
# ---------------------------------------------------------------------------

class TestDetectDemandSeasonality:
    def test_empty_rows_returns_none(self):
        result = detect_demand_seasonality([], "qty", "period")
        assert result is None

    def test_single_period_returns_none(self):
        rows = [{"period": "M1", "qty": 100}]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is None

    def test_basic_seasonality(self):
        rows = [
            {"period": "Q1", "qty": 100},
            {"period": "Q2", "qty": 200},
            {"period": "Q3", "qty": 100},
            {"period": "Q4", "qty": 200},
        ]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is not None
        assert len(result.periods) == 4
        assert result.overall_avg == 150.0

    def test_peak_seasons(self):
        # Avg = 150, threshold = 1.2 * 150 = 180
        rows = [
            {"period": "Q1", "qty": 100},
            {"period": "Q2", "qty": 250},
            {"period": "Q3", "qty": 100},
            {"period": "Q4", "qty": 150},
        ]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is not None
        assert "Q2" in result.peak_seasons

    def test_low_seasons(self):
        # Avg = 150, threshold = 0.8 * 150 = 120
        rows = [
            {"period": "Q1", "qty": 50},
            {"period": "Q2", "qty": 200},
            {"period": "Q3", "qty": 200},
            {"period": "Q4", "qty": 150},
        ]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is not None
        assert "Q1" in result.low_seasons

    def test_seasonal_strength(self):
        rows = [
            {"period": "Q1", "qty": 50},
            {"period": "Q2", "qty": 150},
        ]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is not None
        # Avg = 100, Q1 index = 0.5, Q2 index = 1.5
        # Strength = 1.5 - 0.5 = 1.0
        assert abs(result.seasonal_strength - 1.0) < 0.01

    def test_seasonal_index(self):
        rows = [
            {"period": "Q1", "qty": 100},
            {"period": "Q2", "qty": 200},
        ]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is not None
        # Avg = 150
        q1 = next(p for p in result.periods if p.period == "Q1")
        q2 = next(p for p in result.periods if p.period == "Q2")
        assert abs(q1.seasonal_index - 100 / 150) < 0.01
        assert abs(q2.seasonal_index - 200 / 150) < 0.01

    def test_uniform_demand_no_peaks_or_lows(self):
        rows = [
            {"period": "Q1", "qty": 100},
            {"period": "Q2", "qty": 100},
            {"period": "Q3", "qty": 100},
            {"period": "Q4", "qty": 100},
        ]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is not None
        assert len(result.peak_seasons) == 0
        assert len(result.low_seasons) == 0
        assert result.seasonal_strength == 0.0

    def test_zero_overall_avg_returns_none(self):
        rows = [
            {"period": "Q1", "qty": 0},
            {"period": "Q2", "qty": 0},
        ]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is None

    def test_summary_present(self):
        rows = [
            {"period": "Q1", "qty": 100},
            {"period": "Q2", "qty": 200},
        ]
        result = detect_demand_seasonality(rows, "qty", "period")
        assert result is not None
        assert "Seasonality analysis" in result.summary
        assert "2 periods" in result.summary


# ---------------------------------------------------------------------------
# format_demand_report
# ---------------------------------------------------------------------------

class TestFormatDemandReport:
    def test_no_data(self):
        report = format_demand_report()
        assert "No demand data" in report

    def test_pattern_only(self):
        rows = _make_rows("A", [("Q1", 100), ("Q2", 120)])
        pattern = analyze_demand_pattern(rows, "product", "qty", "period")
        report = format_demand_report(pattern=pattern)
        assert "Demand Pattern Analysis" in report
        assert "A" in report
        assert "Moving Average" not in report

    def test_moving_avg_only(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
        ]
        ma = compute_moving_average(rows, "qty", "period", window=3)
        report = format_demand_report(moving_avg=ma)
        assert "Moving Average" in report
        assert "Window" in report

    def test_smoothing_only(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
        ]
        sm = exponential_smoothing(rows, "qty", "period", alpha=0.3)
        report = format_demand_report(smoothing=sm)
        assert "Exponential Smoothing" in report
        assert "Alpha" in report
        assert "Optimal Alpha" in report

    def test_seasonality_only(self):
        rows = [
            {"period": "Q1", "qty": 50},
            {"period": "Q2", "qty": 200},
        ]
        seas = detect_demand_seasonality(rows, "qty", "period")
        report = format_demand_report(seasonality=seas)
        assert "Seasonality" in report
        assert "Seasonal Strength" in report

    def test_combined_report(self):
        base_rows = _make_rows("A", [
            ("Q1", 100), ("Q2", 200), ("Q3", 150), ("Q4", 250),
        ])
        flat_rows = [
            {"period": "Q1", "qty": 100},
            {"period": "Q2", "qty": 200},
            {"period": "Q3", "qty": 150},
            {"period": "Q4", "qty": 250},
        ]
        pattern = analyze_demand_pattern(base_rows, "product", "qty", "period")
        ma = compute_moving_average(flat_rows, "qty", "period", window=3)
        sm = exponential_smoothing(flat_rows, "qty", "period")
        seas = detect_demand_seasonality(flat_rows, "qty", "period")
        report = format_demand_report(
            pattern=pattern, moving_avg=ma, smoothing=sm, seasonality=seas,
        )
        assert "Demand Forecast Report" in report
        assert "Demand Pattern Analysis" in report
        assert "Moving Average" in report
        assert "Exponential Smoothing" in report
        assert "Seasonality" in report

    def test_report_includes_mad_mape(self):
        rows = [
            {"period": "M1", "qty": 100},
            {"period": "M2", "qty": 120},
            {"period": "M3", "qty": 110},
            {"period": "M4", "qty": 130},
        ]
        ma = compute_moving_average(rows, "qty", "period", window=3)
        report = format_demand_report(moving_avg=ma)
        assert "MAD" in report
        assert "MAPE" in report

    def test_report_peak_and_low_seasons(self):
        rows = [
            {"period": "Q1", "qty": 50},
            {"period": "Q2", "qty": 300},
            {"period": "Q3", "qty": 100},
            {"period": "Q4", "qty": 150},
        ]
        seas = detect_demand_seasonality(rows, "qty", "period")
        report = format_demand_report(seasonality=seas)
        assert "Peak Seasons" in report
        assert "Low Seasons" in report
