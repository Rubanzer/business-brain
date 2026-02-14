"""Tests for capacity planning module."""

from business_brain.discovery.capacity_planning import (
    Bottleneck,
    EntityUtilization,
    ExhaustionForecast,
    UtilizationResult,
    capacity_summary,
    compute_utilization,
    detect_bottlenecks,
    forecast_capacity_exhaustion,
)


# ---------------------------------------------------------------------------
# compute_utilization
# ---------------------------------------------------------------------------


class TestComputeUtilization:
    def test_basic_utilization(self):
        rows = [
            {"plant": "A", "actual": 80, "capacity": 100},
            {"plant": "B", "actual": 90, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert result.entity_count == 2
        assert len(result.entities) == 2
        a = next(e for e in result.entities if e.entity == "A")
        assert a.utilization_pct == 80.0
        b = next(e for e in result.entities if e.entity == "B")
        assert b.utilization_pct == 90.0

    def test_mean_utilization(self):
        rows = [
            {"plant": "A", "actual": 80, "capacity": 100},
            {"plant": "B", "actual": 60, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert result.mean_utilization == 70.0

    def test_over_utilized_detection(self):
        rows = [
            {"plant": "A", "actual": 97, "capacity": 100},
            {"plant": "B", "actual": 50, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert "A" in result.over_utilized
        assert "B" not in result.over_utilized
        assert "A" in result.bottlenecks

    def test_under_utilized_detection(self):
        rows = [
            {"plant": "A", "actual": 30, "capacity": 100},
            {"plant": "B", "actual": 80, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert "A" in result.under_utilized
        assert "B" not in result.under_utilized

    def test_with_time_column(self):
        rows = [
            {"plant": "A", "month": "Jan", "actual": 80, "capacity": 100},
            {"plant": "A", "month": "Feb", "actual": 90, "capacity": 100},
            {"plant": "B", "month": "Jan", "actual": 50, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity", "month")
        assert result is not None
        a = next(e for e in result.entities if e.entity == "A")
        assert a.periods == 2
        assert len(a.per_period) == 2
        jan = next(p for p in a.per_period if p["period"] == "Jan")
        assert jan["utilization_pct"] == 80.0
        feb = next(p for p in a.per_period if p["period"] == "Feb")
        assert feb["utilization_pct"] == 90.0

    def test_empty_rows(self):
        assert compute_utilization([], "plant", "actual", "capacity") is None

    def test_single_entity(self):
        rows = [{"plant": "A", "actual": 75, "capacity": 100}]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert result.entity_count == 1
        assert result.mean_utilization == 75.0

    def test_none_values_skipped(self):
        rows = [
            {"plant": "A", "actual": 80, "capacity": 100},
            {"plant": "A", "actual": None, "capacity": 100},
            {"plant": "B", "actual": 60, "capacity": None},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        # Only plant A's first row should be valid; B is fully skipped
        assert result.entity_count == 1
        a = result.entities[0]
        assert a.entity == "A"
        assert a.utilization_pct == 80.0

    def test_all_same_values(self):
        rows = [
            {"plant": "A", "actual": 50, "capacity": 100},
            {"plant": "B", "actual": 50, "capacity": 100},
            {"plant": "C", "actual": 50, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert result.mean_utilization == 50.0
        # 50% is not < 50, so none under-utilized
        assert result.under_utilized == []

    def test_zero_capacity(self):
        rows = [
            {"plant": "A", "actual": 80, "capacity": 0},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert result.entities[0].utilization_pct == 0.0

    def test_multiple_rows_same_entity_aggregates(self):
        rows = [
            {"plant": "A", "actual": 40, "capacity": 50},
            {"plant": "A", "actual": 40, "capacity": 50},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        a = result.entities[0]
        assert a.total_actual == 80.0
        assert a.total_capacity == 100.0
        assert a.utilization_pct == 80.0

    def test_summary_contains_entity_count(self):
        rows = [
            {"plant": "A", "actual": 80, "capacity": 100},
            {"plant": "B", "actual": 60, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert "2 entities" in result.summary

    def test_all_invalid_data_returns_none(self):
        rows = [
            {"plant": "A", "actual": "bad", "capacity": 100},
            {"plant": "B", "actual": 60, "capacity": "nope"},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is None

    def test_boundary_95_not_over(self):
        """Exactly 95% should NOT be flagged as over-utilized (>95 required)."""
        rows = [{"plant": "A", "actual": 95, "capacity": 100}]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert result.over_utilized == []

    def test_boundary_50_not_under(self):
        """Exactly 50% should NOT be flagged as under-utilized (<50 required)."""
        rows = [{"plant": "A", "actual": 50, "capacity": 100}]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        assert result is not None
        assert result.under_utilized == []


# ---------------------------------------------------------------------------
# detect_bottlenecks
# ---------------------------------------------------------------------------


class TestDetectBottlenecks:
    def test_basic_bottleneck(self):
        rows = [
            {"stage": "Cutting", "throughput": 100},
            {"stage": "Welding", "throughput": 50},
            {"stage": "Assembly", "throughput": 90},
        ]
        results = detect_bottlenecks(rows, "stage", "throughput")
        assert len(results) == 3
        welding = next(b for b in results if b.stage == "Welding")
        assert welding.is_bottleneck is True
        assert welding.throughput_pct_of_max == 50.0

    def test_no_bottleneck_all_equal(self):
        rows = [
            {"stage": "A", "throughput": 100},
            {"stage": "B", "throughput": 100},
            {"stage": "C", "throughput": 100},
        ]
        results = detect_bottlenecks(rows, "stage", "throughput")
        assert all(not b.is_bottleneck for b in results)
        assert all(b.throughput_pct_of_max == 100.0 for b in results)

    def test_multiple_bottlenecks(self):
        rows = [
            {"stage": "A", "throughput": 100},
            {"stage": "B", "throughput": 40},
            {"stage": "C", "throughput": 30},
        ]
        results = detect_bottlenecks(rows, "stage", "throughput")
        bottlenecks = [b for b in results if b.is_bottleneck]
        assert len(bottlenecks) == 2  # B at 40% and C at 30%

    def test_empty_rows(self):
        assert detect_bottlenecks([], "stage", "throughput") == []

    def test_with_time_column_averages(self):
        rows = [
            {"stage": "A", "period": "Jan", "throughput": 100},
            {"stage": "A", "period": "Feb", "throughput": 120},
            {"stage": "B", "period": "Jan", "throughput": 50},
            {"stage": "B", "period": "Feb", "throughput": 70},
        ]
        results = detect_bottlenecks(rows, "stage", "throughput", "period")
        a = next(b for b in results if b.stage == "A")
        b = next(bn for bn in results if bn.stage == "B")
        # A mean = 110, B mean = 60, so B at ~54.5% of max
        assert a.throughput == 110.0
        assert b.throughput == 60.0
        assert b.is_bottleneck is True

    def test_constraint_ratio(self):
        rows = [
            {"stage": "A", "throughput": 200},
            {"stage": "B", "throughput": 100},
        ]
        results = detect_bottlenecks(rows, "stage", "throughput")
        b = next(bn for bn in results if bn.stage == "B")
        assert b.constraint_ratio == 0.5

    def test_sorted_by_throughput_ascending(self):
        rows = [
            {"stage": "A", "throughput": 100},
            {"stage": "B", "throughput": 30},
            {"stage": "C", "throughput": 60},
        ]
        results = detect_bottlenecks(rows, "stage", "throughput")
        throughputs = [b.throughput for b in results]
        assert throughputs == sorted(throughputs)

    def test_none_throughput_skipped(self):
        rows = [
            {"stage": "A", "throughput": 100},
            {"stage": "B", "throughput": None},
        ]
        results = detect_bottlenecks(rows, "stage", "throughput")
        assert len(results) == 1
        assert results[0].stage == "A"


# ---------------------------------------------------------------------------
# forecast_capacity_exhaustion
# ---------------------------------------------------------------------------


class TestForecastCapacityExhaustion:
    def test_basic_forecast(self):
        rows = [
            {"plant": "A", "month": "1", "actual": 70, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 80, "capacity": 100},
            {"plant": "A", "month": "3", "actual": 90, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        assert len(forecasts) == 1
        f = forecasts[0]
        assert f.entity == "A"
        assert f.current_utilization == 90.0
        assert f.trend_per_period > 0
        assert f.periods_to_exhaustion is not None
        assert f.periods_to_exhaustion > 0

    def test_declining_trend_no_exhaustion(self):
        rows = [
            {"plant": "A", "month": "1", "actual": 90, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 80, "capacity": 100},
            {"plant": "A", "month": "3", "actual": 70, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        f = forecasts[0]
        assert f.trend_per_period < 0
        assert f.periods_to_exhaustion is None
        assert f.urgency == "ok"

    def test_already_over_capacity(self):
        rows = [
            {"plant": "A", "month": "1", "actual": 90, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 100, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        f = forecasts[0]
        assert f.current_utilization == 100.0
        assert f.periods_to_exhaustion == 0.0
        assert f.urgency == "critical"

    def test_urgency_critical(self):
        # Trend is steep: +10 per period, current at 80 => 2 periods to 100
        rows = [
            {"plant": "A", "month": "1", "actual": 60, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 70, "capacity": 100},
            {"plant": "A", "month": "3", "actual": 80, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        f = forecasts[0]
        assert f.urgency == "critical"
        assert f.periods_to_exhaustion <= 3

    def test_urgency_warning(self):
        # +5 per period, current at 70 => 6 periods to 100 => warning
        rows = [
            {"plant": "A", "month": "1", "actual": 60, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 65, "capacity": 100},
            {"plant": "A", "month": "3", "actual": 70, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        f = forecasts[0]
        assert f.urgency == "warning"

    def test_urgency_ok(self):
        # +1 per period, current at 50 => 50 periods to 100 => ok
        rows = [
            {"plant": "A", "month": "1", "actual": 48, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 49, "capacity": 100},
            {"plant": "A", "month": "3", "actual": 50, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        f = forecasts[0]
        assert f.urgency == "ok"

    def test_empty_rows(self):
        assert forecast_capacity_exhaustion([], "p", "a", "c", "t") == []

    def test_multiple_entities_sorted_by_urgency(self):
        rows = [
            # Entity A: rising steeply
            {"plant": "A", "month": "1", "actual": 80, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 90, "capacity": 100},
            {"plant": "A", "month": "3", "actual": 95, "capacity": 100},
            # Entity B: declining
            {"plant": "B", "month": "1", "actual": 70, "capacity": 100},
            {"plant": "B", "month": "2", "actual": 60, "capacity": 100},
            {"plant": "B", "month": "3", "actual": 50, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        assert len(forecasts) == 2
        # A should come first (critical), B is ok
        assert forecasts[0].entity == "A"
        assert forecasts[1].entity == "B"
        assert forecasts[0].urgency in ("critical", "warning")
        assert forecasts[1].urgency == "ok"

    def test_none_values_skipped(self):
        rows = [
            {"plant": "A", "month": "1", "actual": 70, "capacity": 100},
            {"plant": "A", "month": None, "actual": 80, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 80, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        assert len(forecasts) == 1
        # Only 2 valid periods
        assert forecasts[0].entity == "A"

    def test_flat_trend(self):
        rows = [
            {"plant": "A", "month": "1", "actual": 50, "capacity": 100},
            {"plant": "A", "month": "2", "actual": 50, "capacity": 100},
            {"plant": "A", "month": "3", "actual": 50, "capacity": 100},
        ]
        forecasts = forecast_capacity_exhaustion(rows, "plant", "actual", "capacity", "month")
        f = forecasts[0]
        assert f.trend_per_period == 0.0
        assert f.periods_to_exhaustion is None
        assert f.urgency == "ok"


# ---------------------------------------------------------------------------
# capacity_summary
# ---------------------------------------------------------------------------


class TestCapacitySummary:
    def test_basic_summary(self):
        rows = [
            {"plant": "A", "actual": 97, "capacity": 100},
            {"plant": "B", "actual": 30, "capacity": 100},
            {"plant": "C", "actual": 75, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        text = capacity_summary(result)
        assert "Capacity Utilization Report" in text
        assert "3" in text  # entity count
        assert "OVER-UTILIZED" in text
        assert "UNDER-UTILIZED" in text
        assert "A" in text
        assert "B" in text

    def test_summary_no_alerts(self):
        rows = [
            {"plant": "A", "actual": 70, "capacity": 100},
            {"plant": "B", "actual": 80, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        text = capacity_summary(result)
        assert "OVER-UTILIZED" not in text
        assert "UNDER-UTILIZED" not in text

    def test_summary_includes_mean(self):
        rows = [
            {"plant": "A", "actual": 60, "capacity": 100},
            {"plant": "B", "actual": 80, "capacity": 100},
        ]
        result = compute_utilization(rows, "plant", "actual", "capacity")
        text = capacity_summary(result)
        assert "70.0%" in text
