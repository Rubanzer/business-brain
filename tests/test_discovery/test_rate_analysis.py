"""Tests for rate analysis module."""

from business_brain.discovery.rate_analysis import (
    RateComparisonResult,
    RateComparison,
    SupplierRate,
    RateTrendResult,
    PeriodRate,
    RateVolumeResult,
    SupplierRateVolume,
    RateAnomaly,
    compare_rates,
    analyze_rate_trend,
    rate_volume_analysis,
    detect_rate_anomalies,
    format_rate_report,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _multi_supplier_rows():
    """Three suppliers with varying rates for two items."""
    return [
        {"supplier": "Alpha", "rate": 100, "item": "Steel", "volume": 50},
        {"supplier": "Alpha", "rate": 105, "item": "Steel", "volume": 60},
        {"supplier": "Beta",  "rate": 120, "item": "Steel", "volume": 40},
        {"supplier": "Beta",  "rate": 125, "item": "Steel", "volume": 45},
        {"supplier": "Gamma", "rate": 90,  "item": "Steel", "volume": 80},
        {"supplier": "Gamma", "rate": 95,  "item": "Steel", "volume": 70},
        {"supplier": "Alpha", "rate": 200, "item": "Copper", "volume": 20},
        {"supplier": "Alpha", "rate": 210, "item": "Copper", "volume": 25},
        {"supplier": "Beta",  "rate": 180, "item": "Copper", "volume": 30},
        {"supplier": "Beta",  "rate": 190, "item": "Copper", "volume": 35},
    ]


def _trend_rows():
    """Rate data over 4 time periods with increasing rates."""
    return [
        {"period": "2024-Q1", "rate": 100},
        {"period": "2024-Q1", "rate": 110},
        {"period": "2024-Q2", "rate": 115},
        {"period": "2024-Q2", "rate": 120},
        {"period": "2024-Q3", "rate": 130},
        {"period": "2024-Q3", "rate": 135},
        {"period": "2024-Q4", "rate": 145},
        {"period": "2024-Q4", "rate": 150},
    ]


# ---------------------------------------------------------------------------
# 1. compare_rates
# ---------------------------------------------------------------------------


class TestCompareRates:
    def test_basic_returns_result(self):
        result = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        assert result is not None
        assert isinstance(result, RateComparisonResult)

    def test_comparison_count_matches_items(self):
        result = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        # Two items: Steel and Copper
        assert len(result.comparisons) == 2

    def test_best_supplier_has_lowest_avg(self):
        result = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        # For Steel: Gamma avg = 92.5, Alpha avg = 102.5, Beta avg = 122.5
        steel = [c for c in result.comparisons if c.item == "Steel"][0]
        assert steel.best_supplier == "Gamma"

    def test_worst_supplier_has_highest_avg(self):
        result = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        steel = [c for c in result.comparisons if c.item == "Steel"][0]
        assert steel.worst_supplier == "Beta"

    def test_spread_is_positive(self):
        result = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        for comp in result.comparisons:
            assert comp.spread >= 0
            assert comp.spread_pct >= 0

    def test_without_item_column_uses_overall(self):
        rows = [
            {"supplier": "A", "rate": 10},
            {"supplier": "B", "rate": 20},
        ]
        result = compare_rates(rows, "supplier", "rate")
        assert result is not None
        assert len(result.comparisons) == 1
        assert result.comparisons[0].item == "overall"

    def test_empty_rows_returns_none(self):
        assert compare_rates([], "supplier", "rate") is None

    def test_none_rate_values_skipped(self):
        rows = [
            {"supplier": "A", "rate": 10},
            {"supplier": "A", "rate": None},
            {"supplier": "B", "rate": 20},
        ]
        result = compare_rates(rows, "supplier", "rate")
        assert result is not None
        a_rate = [
            sr for c in result.comparisons for sr in c.suppliers if sr.supplier == "A"
        ][0]
        assert a_rate.volume == 1  # None row skipped

    def test_single_supplier_spread_zero(self):
        rows = [
            {"supplier": "Solo", "rate": 50},
            {"supplier": "Solo", "rate": 55},
        ]
        result = compare_rates(rows, "supplier", "rate")
        assert result is not None
        assert result.comparisons[0].spread == 0
        assert result.comparisons[0].spread_pct == 0

    def test_savings_potential_nonnegative(self):
        result = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        assert result.overall_savings_potential >= 0

    def test_supplier_rate_fields_populated(self):
        result = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        for comp in result.comparisons:
            for sr in comp.suppliers:
                assert isinstance(sr, SupplierRate)
                assert sr.min_rate <= sr.avg_rate <= sr.max_rate
                assert sr.volume > 0
                assert sr.total_value > 0

    def test_summary_contains_supplier_names(self):
        result = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        assert result.best_rate_supplier in result.summary
        assert result.worst_rate_supplier in result.summary


# ---------------------------------------------------------------------------
# 2. analyze_rate_trend
# ---------------------------------------------------------------------------


class TestAnalyzeRateTrend:
    def test_basic_returns_result(self):
        result = analyze_rate_trend(_trend_rows(), "period", "rate")
        assert result is not None
        assert isinstance(result, RateTrendResult)

    def test_period_count(self):
        result = analyze_rate_trend(_trend_rows(), "period", "rate")
        assert len(result.periods) == 4

    def test_increasing_trend_detected(self):
        result = analyze_rate_trend(_trend_rows(), "period", "rate")
        assert result.trend_direction == "increasing"

    def test_total_change_positive_for_increasing(self):
        result = analyze_rate_trend(_trend_rows(), "period", "rate")
        assert result.total_change_pct > 0

    def test_decreasing_trend(self):
        rows = [
            {"period": "Q1", "rate": 200},
            {"period": "Q2", "rate": 150},
            {"period": "Q3", "rate": 100},
        ]
        result = analyze_rate_trend(rows, "period", "rate")
        assert result is not None
        assert result.trend_direction == "decreasing"
        assert result.total_change_pct < 0

    def test_stable_trend(self):
        rows = [
            {"period": "Q1", "rate": 100},
            {"period": "Q2", "rate": 101},
            {"period": "Q3", "rate": 100},
        ]
        result = analyze_rate_trend(rows, "period", "rate")
        assert result is not None
        assert result.trend_direction == "stable"

    def test_empty_rows_returns_none(self):
        assert analyze_rate_trend([], "period", "rate") is None

    def test_single_period_returns_none(self):
        rows = [{"period": "Q1", "rate": 100}]
        assert analyze_rate_trend(rows, "period", "rate") is None

    def test_none_rate_values_skipped(self):
        rows = [
            {"period": "Q1", "rate": 100},
            {"period": "Q1", "rate": None},
            {"period": "Q2", "rate": 120},
        ]
        result = analyze_rate_trend(rows, "period", "rate")
        assert result is not None
        # Q1 should only have 1 observation
        assert result.periods[0].volume == 1

    def test_period_rate_fields(self):
        result = analyze_rate_trend(_trend_rows(), "period", "rate")
        for p in result.periods:
            assert isinstance(p, PeriodRate)
            assert p.min_rate <= p.avg_rate <= p.max_rate

    def test_summary_contains_direction(self):
        result = analyze_rate_trend(_trend_rows(), "period", "rate")
        assert result.trend_direction in result.summary


# ---------------------------------------------------------------------------
# 3. rate_volume_analysis
# ---------------------------------------------------------------------------


class TestRateVolumeAnalysis:
    def test_basic_returns_result(self):
        rows = [
            {"supplier": "A", "rate": 100, "volume": 10},
            {"supplier": "A", "rate": 90,  "volume": 50},
            {"supplier": "A", "rate": 80,  "volume": 100},
            {"supplier": "A", "rate": 70,  "volume": 200},
        ]
        result = rate_volume_analysis(rows, "supplier", "rate", "volume")
        assert result is not None
        assert isinstance(result, RateVolumeResult)

    def test_negative_correlation_for_bulk_discount(self):
        """When rates decrease as volume increases, correlation should be negative."""
        rows = [
            {"supplier": "A", "rate": 100, "volume": 10},
            {"supplier": "A", "rate": 90,  "volume": 50},
            {"supplier": "A", "rate": 80,  "volume": 100},
            {"supplier": "A", "rate": 70,  "volume": 200},
        ]
        result = rate_volume_analysis(rows, "supplier", "rate", "volume")
        assert result.correlation < 0
        assert result.has_volume_discount is True

    def test_no_volume_discount_detected(self):
        """When rates don't correlate with volume."""
        rows = [
            {"supplier": "A", "rate": 100, "volume": 10},
            {"supplier": "A", "rate": 100, "volume": 50},
            {"supplier": "A", "rate": 100, "volume": 100},
            {"supplier": "A", "rate": 100, "volume": 200},
        ]
        result = rate_volume_analysis(rows, "supplier", "rate", "volume")
        assert result is not None
        assert result.has_volume_discount is False

    def test_empty_rows_returns_none(self):
        assert rate_volume_analysis([], "supplier", "rate", "volume") is None

    def test_none_values_skipped(self):
        rows = [
            {"supplier": "A", "rate": 100, "volume": 10},
            {"supplier": "A", "rate": None, "volume": 50},
            {"supplier": "A", "rate": 80,  "volume": None},
            {"supplier": "A", "rate": 70,  "volume": 200},
        ]
        result = rate_volume_analysis(rows, "supplier", "rate", "volume")
        assert result is not None
        # Only two valid rows for supplier A
        a_data = [s for s in result.suppliers if s.supplier == "A"][0]
        assert a_data.total_volume == 210.0  # 10 + 200

    def test_zero_volume_rows_included(self):
        rows = [
            {"supplier": "A", "rate": 100, "volume": 0},
            {"supplier": "A", "rate": 90,  "volume": 50},
        ]
        result = rate_volume_analysis(rows, "supplier", "rate", "volume")
        assert result is not None
        a_data = [s for s in result.suppliers if s.supplier == "A"][0]
        assert a_data.total_volume == 50.0

    def test_single_row_per_supplier(self):
        rows = [
            {"supplier": "A", "rate": 100, "volume": 10},
        ]
        result = rate_volume_analysis(rows, "supplier", "rate", "volume")
        assert result is not None
        a_data = result.suppliers[0]
        # With a single row, low and high volume rates equal avg
        assert a_data.rate_at_low_volume == a_data.avg_rate
        assert a_data.rate_at_high_volume == a_data.avg_rate

    def test_volume_discount_pct_positive_when_discount(self):
        """volume_discount_pct should be positive when higher volume has lower rate."""
        rows = [
            {"supplier": "A", "rate": 100, "volume": 10},
            {"supplier": "A", "rate": 90,  "volume": 50},
            {"supplier": "A", "rate": 80,  "volume": 100},
            {"supplier": "A", "rate": 70,  "volume": 200},
        ]
        result = rate_volume_analysis(rows, "supplier", "rate", "volume")
        a_data = result.suppliers[0]
        assert a_data.volume_discount_pct > 0


# ---------------------------------------------------------------------------
# 4. detect_rate_anomalies
# ---------------------------------------------------------------------------


class TestDetectRateAnomalies:
    def test_detects_high_anomaly(self):
        rows = [
            {"supplier": "Normal1", "rate": 100},
            {"supplier": "Normal2", "rate": 105},
            {"supplier": "Normal3", "rate": 95},
            {"supplier": "Expensive", "rate": 200},  # way above average
        ]
        anomalies = detect_rate_anomalies(rows, "supplier", "rate")
        high = [a for a in anomalies if a.anomaly_type == "too_high"]
        assert len(high) >= 1
        assert high[0].supplier == "Expensive"

    def test_detects_low_anomaly(self):
        rows = [
            {"supplier": "Normal1", "rate": 100},
            {"supplier": "Normal2", "rate": 105},
            {"supplier": "Normal3", "rate": 95},
            {"supplier": "Cheap", "rate": 20},  # way below average
        ]
        anomalies = detect_rate_anomalies(rows, "supplier", "rate")
        low = [a for a in anomalies if a.anomaly_type == "too_low"]
        assert len(low) >= 1
        assert low[0].supplier == "Cheap"

    def test_no_anomalies_when_similar_rates(self):
        rows = [
            {"supplier": "A", "rate": 100},
            {"supplier": "B", "rate": 102},
            {"supplier": "C", "rate": 98},
        ]
        anomalies = detect_rate_anomalies(rows, "supplier", "rate")
        assert len(anomalies) == 0

    def test_empty_rows_returns_empty(self):
        assert detect_rate_anomalies([], "supplier", "rate") == []

    def test_custom_threshold(self):
        rows = [
            {"supplier": "A", "rate": 100},
            {"supplier": "B", "rate": 115},  # 15% above mean ~107.5 -> ~7%
        ]
        # With default 20% threshold, this should not be anomalous
        anomalies_default = detect_rate_anomalies(rows, "supplier", "rate", threshold_pct=20)
        # With 5% threshold, it should be
        anomalies_low = detect_rate_anomalies(rows, "supplier", "rate", threshold_pct=5)
        assert len(anomalies_low) >= len(anomalies_default)

    def test_per_item_anomalies(self):
        rows = [
            {"supplier": "A", "rate": 100, "item": "Steel"},
            {"supplier": "B", "rate": 100, "item": "Steel"},
            {"supplier": "C", "rate": 200, "item": "Steel"},  # anomaly for Steel
            {"supplier": "A", "rate": 50,  "item": "Copper"},
            {"supplier": "B", "rate": 50,  "item": "Copper"},
        ]
        anomalies = detect_rate_anomalies(rows, "supplier", "rate", "item")
        steel_anomalies = [a for a in anomalies if a.item == "Steel"]
        copper_anomalies = [a for a in anomalies if a.item == "Copper"]
        assert len(steel_anomalies) >= 1
        assert len(copper_anomalies) == 0

    def test_severity_scales_with_deviation(self):
        rows = [
            {"supplier": "A", "rate": 100},
            {"supplier": "B", "rate": 100},
            {"supplier": "Extreme", "rate": 500},  # very high deviation
        ]
        anomalies = detect_rate_anomalies(rows, "supplier", "rate", threshold_pct=20)
        extreme = [a for a in anomalies if a.supplier == "Extreme"]
        assert len(extreme) == 1
        assert extreme[0].severity == "high"

    def test_anomaly_fields_populated(self):
        rows = [
            {"supplier": "A", "rate": 100},
            {"supplier": "B", "rate": 200},
        ]
        anomalies = detect_rate_anomalies(rows, "supplier", "rate", threshold_pct=10)
        for a in anomalies:
            assert isinstance(a, RateAnomaly)
            assert a.anomaly_type in ("too_high", "too_low")
            assert a.severity in ("low", "medium", "high")
            assert a.avg_rate > 0


# ---------------------------------------------------------------------------
# 5. format_rate_report
# ---------------------------------------------------------------------------


class TestFormatRateReport:
    def test_no_data_message(self):
        report = format_rate_report()
        assert "No analysis data provided" in report

    def test_contains_header(self):
        report = format_rate_report()
        assert "Rate Analysis Report" in report

    def test_with_comparison_section(self):
        comp = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        report = format_rate_report(comparison=comp)
        assert "Rate Comparison" in report
        assert "Savings potential" in report

    def test_with_trend_section(self):
        trend = analyze_rate_trend(_trend_rows(), "period", "rate")
        report = format_rate_report(trend=trend)
        assert "Rate Trend" in report
        assert "Direction" in report

    def test_with_anomalies_section(self):
        rows = [
            {"supplier": "A", "rate": 100},
            {"supplier": "B", "rate": 300},
        ]
        anomalies = detect_rate_anomalies(rows, "supplier", "rate", threshold_pct=10)
        report = format_rate_report(anomalies=anomalies)
        assert "Rate Anomalies" in report

    def test_with_empty_anomalies_list(self):
        report = format_rate_report(anomalies=[])
        assert "No anomalies detected" in report

    def test_combined_report(self):
        comp = compare_rates(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        trend = analyze_rate_trend(_trend_rows(), "period", "rate")
        anomalies = detect_rate_anomalies(
            _multi_supplier_rows(), "supplier", "rate", "item"
        )
        report = format_rate_report(
            comparison=comp, trend=trend, anomalies=anomalies
        )
        assert "Rate Comparison" in report
        assert "Rate Trend" in report
        assert "Rate Anomalies" in report
