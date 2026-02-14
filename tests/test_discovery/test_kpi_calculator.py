"""Tests for KPI calculator module."""

from business_brain.discovery.kpi_calculator import (
    KPIResult,
    compound_growth_rate,
    compute_all_kpis,
    efficiency_ratio,
    exponential_moving_average,
    growth_rate,
    inventory_turnover,
    moving_average,
    rate_of_change,
    utilization_rate,
    variance_from_target,
    yield_rate,
)


# ---------------------------------------------------------------------------
# growth_rate
# ---------------------------------------------------------------------------


class TestGrowthRate:
    def test_positive_growth(self):
        r = growth_rate(120, 100)
        assert r.value == 20.0
        assert r.trend == "up"
        assert r.status == "good"

    def test_negative_growth(self):
        r = growth_rate(80, 100)
        assert r.value == -20.0
        assert r.trend == "down"
        assert r.status == "bad"

    def test_zero_change(self):
        r = growth_rate(100, 100)
        assert r.value == 0.0
        assert r.trend == "stable"

    def test_from_zero(self):
        r = growth_rate(50, 0)
        assert r.value == 100.0

    def test_to_zero(self):
        r = growth_rate(0, 100)
        assert r.value == -100.0

    def test_unit_is_percent(self):
        assert growth_rate(110, 100).unit == "%"


# ---------------------------------------------------------------------------
# compound_growth_rate
# ---------------------------------------------------------------------------


class TestCompoundGrowthRate:
    def test_positive_cagr(self):
        r = compound_growth_rate(100, 200, 5)
        assert r.value > 0
        assert "CAGR" in r.name

    def test_negative_cagr(self):
        r = compound_growth_rate(200, 100, 5)
        assert r.value < 0

    def test_one_period(self):
        r = compound_growth_rate(100, 150, 1)
        assert r.value == 50.0

    def test_zero_first(self):
        r = compound_growth_rate(0, 100, 5)
        assert r.value == 0.0

    def test_zero_periods(self):
        r = compound_growth_rate(100, 200, 0)
        assert r.value == 0.0


# ---------------------------------------------------------------------------
# utilization_rate
# ---------------------------------------------------------------------------


class TestUtilizationRate:
    def test_normal_utilization(self):
        r = utilization_rate(75, 100)
        assert r.value == 75.0
        assert r.status == "good"

    def test_over_utilization(self):
        r = utilization_rate(98, 100)
        assert r.value == 98.0
        assert r.status == "bad"

    def test_under_utilization(self):
        r = utilization_rate(20, 100)
        assert r.value == 20.0
        assert r.status == "bad"

    def test_zero_capacity(self):
        r = utilization_rate(50, 0)
        assert r.value == 0.0


# ---------------------------------------------------------------------------
# efficiency_ratio
# ---------------------------------------------------------------------------


class TestEfficiencyRatio:
    def test_good_efficiency(self):
        r = efficiency_ratio(150, 100)
        assert r.value == 1.5
        assert r.status == "good"

    def test_poor_efficiency(self):
        r = efficiency_ratio(30, 100)
        assert r.value == 0.3
        assert r.status == "bad"

    def test_zero_input(self):
        r = efficiency_ratio(100, 0)
        assert r.value == 0.0


# ---------------------------------------------------------------------------
# variance_from_target
# ---------------------------------------------------------------------------


class TestVarianceFromTarget:
    def test_on_target(self):
        r = variance_from_target(100, 100)
        assert r.value == 0.0
        assert r.status == "good"

    def test_above_target(self):
        r = variance_from_target(110, 100)
        assert r.value == 10.0
        assert r.trend == "up"

    def test_below_target(self):
        r = variance_from_target(80, 100)
        assert r.value == -20.0
        assert r.status == "bad"

    def test_zero_target(self):
        r = variance_from_target(10, 0)
        assert r.value == 0.0


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------


class TestMovingAverage:
    def test_basic_ma(self):
        result = moving_average([10, 20, 30, 40, 50], window=3)
        assert len(result) == 5
        assert result[2] == 20.0  # (10+20+30)/3
        assert result[4] == 40.0  # (30+40+50)/3

    def test_window_1(self):
        result = moving_average([10, 20, 30], window=1)
        assert result == [10, 20, 30]

    def test_empty(self):
        assert moving_average([], 3) == []

    def test_window_zero(self):
        assert moving_average([1, 2, 3], 0) == []


# ---------------------------------------------------------------------------
# exponential_moving_average
# ---------------------------------------------------------------------------


class TestExponentialMovingAverage:
    def test_basic_ema(self):
        result = exponential_moving_average([10, 20, 30, 40, 50])
        assert len(result) == 5
        assert result[0] == 10  # first value unchanged

    def test_high_alpha(self):
        result = exponential_moving_average([10, 20], alpha=0.9)
        # With high alpha, EMA should be close to latest value
        assert result[1] > 15

    def test_low_alpha(self):
        result = exponential_moving_average([10, 20], alpha=0.1)
        # With low alpha, EMA changes slowly
        assert result[1] < 15

    def test_empty(self):
        assert exponential_moving_average([]) == []


# ---------------------------------------------------------------------------
# rate_of_change
# ---------------------------------------------------------------------------


class TestRateOfChange:
    def test_basic(self):
        result = rate_of_change([100, 120, 90])
        assert len(result) == 2
        assert result[0] == 20.0
        assert result[1] == -25.0

    def test_from_zero(self):
        result = rate_of_change([0, 100])
        assert result[0] == 100.0

    def test_too_few(self):
        assert rate_of_change([100]) == []

    def test_empty(self):
        assert rate_of_change([]) == []


# ---------------------------------------------------------------------------
# yield_rate
# ---------------------------------------------------------------------------


class TestYieldRate:
    def test_high_yield(self):
        r = yield_rate(98, 100)
        assert r.value == 98.0
        assert r.status == "good"

    def test_low_yield(self):
        r = yield_rate(70, 100)
        assert r.value == 70.0
        assert r.status == "bad"

    def test_zero_total(self):
        r = yield_rate(0, 0)
        assert r.value == 0.0


# ---------------------------------------------------------------------------
# inventory_turnover
# ---------------------------------------------------------------------------


class TestInventoryTurnover:
    def test_high_turnover(self):
        r = inventory_turnover(600, 50)
        assert r.value == 12.0
        assert r.status == "good"

    def test_low_turnover(self):
        r = inventory_turnover(100, 100)
        assert r.value == 1.0
        assert r.status == "bad"

    def test_zero_inventory(self):
        r = inventory_turnover(100, 0)
        assert r.value == 0.0


# ---------------------------------------------------------------------------
# compute_all_kpis
# ---------------------------------------------------------------------------


class TestComputeAllKPIs:
    def test_with_values_only(self):
        results = compute_all_kpis([100, 110, 120])
        assert len(results) >= 2  # growth_rate + cagr

    def test_with_target(self):
        results = compute_all_kpis([100, 110], target=120)
        assert any(r.name == "Variance from Target" for r in results)

    def test_with_capacity(self):
        results = compute_all_kpis([80], capacity=100)
        assert any(r.name == "Utilization Rate" for r in results)

    def test_single_value(self):
        results = compute_all_kpis([100])
        assert len(results) == 0  # not enough for growth rate

    def test_empty(self):
        assert compute_all_kpis([]) == []
