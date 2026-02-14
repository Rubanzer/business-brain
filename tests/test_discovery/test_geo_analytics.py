"""Tests for geo_analytics module -- 80+ tests covering all functions and dataclasses."""

from __future__ import annotations

import math
from datetime import datetime, date

from business_brain.discovery.geo_analytics import (
    RegionMetrics,
    RegionalDistributionResult,
    MetricComparison,
    RegionScore,
    RegionComparisonResult,
    RegionGrowth,
    GeoGrowthResult,
    RegionPenetration,
    MarketPenetrationResult,
    analyze_regional_distribution,
    compare_regions,
    analyze_geographic_growth,
    compute_market_penetration,
    format_geo_report,
    _safe_float,
    _parse_date,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_region_row(region: str, value: float, count: int = 1) -> dict:
    """Build a single distribution row."""
    return {"region": region, "value": value, "count": count}


def _make_distribution_rows() -> list[dict]:
    """Generate distribution rows across several regions."""
    return [
        _make_region_row("North", 1000),
        _make_region_row("North", 1500),
        _make_region_row("South", 800),
        _make_region_row("South", 600),
        _make_region_row("East", 2000),
        _make_region_row("West", 500),
        _make_region_row("West", 300),
        _make_region_row("West", 200),
    ]


def _make_comparison_rows() -> list[dict]:
    """Generate rows for multi-metric region comparison."""
    return [
        {"region": "North", "revenue": 1000, "orders": 50, "satisfaction": 4.5},
        {"region": "North", "revenue": 1200, "orders": 60, "satisfaction": 4.2},
        {"region": "South", "revenue": 800, "orders": 40, "satisfaction": 3.8},
        {"region": "South", "revenue": 900, "orders": 45, "satisfaction": 3.5},
        {"region": "East", "revenue": 2000, "orders": 80, "satisfaction": 4.0},
        {"region": "West", "revenue": 500, "orders": 30, "satisfaction": 4.8},
    ]


def _make_growth_rows() -> list[dict]:
    """Generate rows for geographic growth analysis."""
    return [
        {"region": "North", "revenue": 1000, "date": "2024-01-15"},
        {"region": "North", "revenue": 1200, "date": "2024-02-15"},
        {"region": "North", "revenue": 1500, "date": "2024-03-15"},
        {"region": "South", "revenue": 800, "date": "2024-01-10"},
        {"region": "South", "revenue": 750, "date": "2024-02-10"},
        {"region": "South", "revenue": 700, "date": "2024-03-10"},
        {"region": "East", "revenue": 500, "date": "2024-01-20"},
        {"region": "East", "revenue": 1000, "date": "2024-03-20"},
    ]


def _make_penetration_rows() -> list[dict]:
    """Generate rows for market penetration analysis."""
    return [
        {"region": "North", "customer": "Alice", "potential": 1000},
        {"region": "North", "customer": "Bob", "potential": 1000},
        {"region": "North", "customer": "Charlie", "potential": 1000},
        {"region": "South", "customer": "Diana", "potential": 500},
        {"region": "South", "customer": "Eve", "potential": 500},
        {"region": "East", "customer": "Frank", "potential": 2000},
        {"region": "West", "customer": "Grace", "potential": 200},
        {"region": "West", "customer": "Hank", "potential": 200},
        {"region": "West", "customer": "Ivy", "potential": 200},
        {"region": "West", "customer": "Jack", "potential": 200},
    ]


# ---------------------------------------------------------------------------
# Tests: _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("99.5") == 99.5

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_invalid_string_returns_none(self):
        assert _safe_float("abc") is None

    def test_empty_string_returns_none(self):
        assert _safe_float("") is None

    def test_negative_number(self):
        assert _safe_float(-7.3) == -7.3

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_string_zero(self):
        assert _safe_float("0") == 0.0


# ---------------------------------------------------------------------------
# Tests: _parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_iso_format(self):
        dt = _parse_date("2024-06-15")
        assert dt == datetime(2024, 6, 15)

    def test_datetime_passthrough(self):
        dt = datetime(2024, 1, 1, 12, 30)
        assert _parse_date(dt) is dt

    def test_date_object_converted(self):
        d = date(2024, 3, 20)
        result = _parse_date(d)
        assert result == datetime(2024, 3, 20)

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_date("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_date("   ") is None

    def test_iso_with_time(self):
        dt = _parse_date("2024-06-15T10:30:00")
        assert dt == datetime(2024, 6, 15, 10, 30, 0)

    def test_iso_with_space_time(self):
        dt = _parse_date("2024-06-15 10:30:00")
        assert dt == datetime(2024, 6, 15, 10, 30, 0)

    def test_us_slash_format(self):
        dt = _parse_date("06/15/2024")
        assert dt is not None

    def test_slash_ymd_format(self):
        dt = _parse_date("2024/06/15")
        assert dt is not None
        assert dt.year == 2024

    def test_unparseable_returns_none(self):
        assert _parse_date("not-a-date") is None


# ---------------------------------------------------------------------------
# Tests: dataclass fields
# ---------------------------------------------------------------------------


class TestDataclassFields:
    def test_region_metrics_fields(self):
        rm = RegionMetrics(region="North", total_value=1000.0, share_pct=50.0,
                           count=10, avg_value=100.0, rank=1)
        assert rm.region == "North"
        assert rm.total_value == 1000.0
        assert rm.share_pct == 50.0
        assert rm.count == 10
        assert rm.avg_value == 100.0
        assert rm.rank == 1

    def test_regional_distribution_result_fields(self):
        rm = RegionMetrics(region="X", total_value=100.0, share_pct=100.0,
                           count=1, avg_value=100.0, rank=1)
        result = RegionalDistributionResult(
            regions=[rm], total_value=100.0, total_count=1,
            top_region="X", concentration_ratio=100.0, hhi_index=10000.0,
            summary="test",
        )
        assert result.top_region == "X"
        assert result.hhi_index == 10000.0

    def test_metric_comparison_fields(self):
        mc = MetricComparison(metric="revenue", best_region="A", best_value=100.0,
                              worst_region="B", worst_value=10.0, avg_value=55.0,
                              std_dev=45.0)
        assert mc.metric == "revenue"
        assert mc.best_region == "A"
        assert mc.worst_value == 10.0

    def test_region_score_fields(self):
        rs = RegionScore(region="Z", metrics={"m1": 1.0}, overall_score=0.5)
        assert rs.region == "Z"
        assert rs.metrics == {"m1": 1.0}
        assert rs.overall_score == 0.5

    def test_region_comparison_result_fields(self):
        mc = MetricComparison(metric="x", best_region="A", best_value=1.0,
                              worst_region="B", worst_value=0.0, avg_value=0.5,
                              std_dev=0.5)
        rs = RegionScore(region="A", metrics={"x": 1.0}, overall_score=1.0)
        result = RegionComparisonResult(
            comparisons=[mc], region_scores=[rs],
            best_overall="A", summary="test",
        )
        assert result.best_overall == "A"

    def test_region_growth_fields(self):
        rg = RegionGrowth(region="North", first_period_value=100.0,
                          last_period_value=200.0, growth_rate=100.0, periods=3)
        assert rg.growth_rate == 100.0
        assert rg.periods == 3

    def test_geo_growth_result_fields(self):
        rg = RegionGrowth(region="A", first_period_value=10.0,
                          last_period_value=20.0, growth_rate=100.0, periods=2)
        result = GeoGrowthResult(regions=[rg], fastest_growing="A",
                                 slowest_growing="A", avg_growth=100.0,
                                 summary="test")
        assert result.fastest_growing == "A"

    def test_region_penetration_fields(self):
        rp = RegionPenetration(region="West", customer_count=5,
                               potential=100.0, penetration_pct=5.0, rank=1)
        assert rp.potential == 100.0
        assert rp.penetration_pct == 5.0

    def test_market_penetration_result_fields(self):
        rp = RegionPenetration(region="West", customer_count=5,
                               potential=None, penetration_pct=None, rank=1)
        result = MarketPenetrationResult(
            regions=[rp], total_customers=5, total_regions=1,
            best_penetration=None, untapped_regions=[], summary="test",
        )
        assert result.total_regions == 1
        assert result.best_penetration is None


# ---------------------------------------------------------------------------
# Tests: analyze_regional_distribution
# ---------------------------------------------------------------------------


class TestAnalyzeRegionalDistribution:
    def test_basic_distribution(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert isinstance(result, RegionalDistributionResult)

    def test_empty_rows_returns_none(self):
        assert analyze_regional_distribution([], "region", "value") is None

    def test_all_null_region_returns_none(self):
        rows = [{"region": None, "value": 100}]
        assert analyze_regional_distribution(rows, "region", "value") is None

    def test_all_null_value_returns_none(self):
        rows = [{"region": "North", "value": None}]
        assert analyze_regional_distribution(rows, "region", "value") is None

    def test_missing_columns_returns_none(self):
        rows = [{"foo": "bar"}]
        assert analyze_regional_distribution(rows, "region", "value") is None

    def test_region_count(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert len(result.regions) == 4

    def test_total_value_sums_correctly(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        expected = 1000 + 1500 + 800 + 600 + 2000 + 500 + 300 + 200
        assert result.total_value == expected

    def test_total_count(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert result.total_count == 8

    def test_top_region(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        # North=2500, South=1400, East=2000, West=1000
        assert result.top_region == "North"

    def test_rank_ordering(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        ranks = [rm.rank for rm in result.regions]
        assert ranks == sorted(ranks)
        assert result.regions[0].rank == 1

    def test_share_pct_sums_to_100(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        total_share = sum(rm.share_pct for rm in result.regions)
        assert abs(total_share - 100.0) < 0.1

    def test_concentration_ratio(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        # Top region North=2500, total=6900
        expected = round(2500 / 6900 * 100, 2)
        assert result.concentration_ratio == expected

    def test_hhi_single_region(self):
        rows = [_make_region_row("Only", 500)]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert result.hhi_index == 10000.0

    def test_hhi_two_equal_regions(self):
        rows = [
            _make_region_row("A", 500),
            _make_region_row("B", 500),
        ]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert result.hhi_index == 5000.0

    def test_hhi_many_equal_regions(self):
        rows = [_make_region_row(f"R{i}", 100) for i in range(10)]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert result.hhi_index == 1000.0

    def test_avg_value_per_region(self):
        rows = [
            _make_region_row("North", 100),
            _make_region_row("North", 200),
        ]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        north = next(rm for rm in result.regions if rm.region == "North")
        assert north.avg_value == 150.0

    def test_with_count_column(self):
        rows = [
            {"region": "North", "value": 1000, "count": 10},
            {"region": "South", "value": 800, "count": 8},
        ]
        result = analyze_regional_distribution(rows, "region", "value", count_column="count")
        assert result is not None
        north = next(rm for rm in result.regions if rm.region == "North")
        assert north.count == 10

    def test_count_column_with_invalid_count(self):
        """If count column has non-numeric, fall back to 1 per row."""
        rows = [
            {"region": "North", "value": 1000, "count": "abc"},
        ]
        result = analyze_regional_distribution(rows, "region", "value", count_column="count")
        assert result is not None
        north = next(rm for rm in result.regions if rm.region == "North")
        assert north.count == 1

    def test_single_row(self):
        rows = [_make_region_row("Solo", 999)]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert result.total_value == 999
        assert result.total_count == 1
        assert result.regions[0].share_pct == 100.0

    def test_non_numeric_value_skipped(self):
        rows = [
            {"region": "North", "value": "abc"},
            {"region": "South", "value": 100},
        ]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert len(result.regions) == 1
        assert result.regions[0].region == "South"

    def test_zero_values_all_regions(self):
        rows = [
            _make_region_row("A", 0),
            _make_region_row("B", 0),
        ]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert result.total_value == 0.0
        assert result.hhi_index == 0.0
        for rm in result.regions:
            assert rm.share_pct == 0.0
            assert rm.avg_value == 0.0

    def test_zero_value_summary_message(self):
        rows = [_make_region_row("A", 0)]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert "zero value" in result.summary.lower()

    def test_string_value_parsed(self):
        rows = [{"region": "North", "value": "500.50"}]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert result.total_value == 500.50

    def test_integer_region_name(self):
        """Region column with integer value is converted to string."""
        rows = [{"region": 1, "value": 100}, {"region": 2, "value": 200}]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        region_names = {rm.region for rm in result.regions}
        assert "1" in region_names
        assert "2" in region_names

    def test_summary_present(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert "Regional Distribution" in result.summary
        assert "HHI" in result.summary

    def test_regions_sorted_by_value_descending(self):
        rows = _make_distribution_rows()
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        values = [rm.total_value for rm in result.regions]
        assert values == sorted(values, reverse=True)

    def test_all_same_region(self):
        rows = [_make_region_row("Same", v) for v in [100, 200, 300]]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert len(result.regions) == 1
        assert result.regions[0].share_pct == 100.0
        assert result.hhi_index == 10000.0

    def test_mixed_valid_and_invalid_rows(self):
        rows = [
            {"region": "A", "value": 100},
            {"region": None, "value": 200},
            {"region": "B", "value": None},
            {"region": "A", "value": "xyz"},
            {"region": "C", "value": 300},
        ]
        result = analyze_regional_distribution(rows, "region", "value")
        assert result is not None
        assert len(result.regions) == 2  # A and C
        assert result.total_value == 400.0


# ---------------------------------------------------------------------------
# Tests: compare_regions
# ---------------------------------------------------------------------------


class TestCompareRegions:
    def test_basic_comparison(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue", "orders"])
        assert result is not None
        assert isinstance(result, RegionComparisonResult)

    def test_empty_rows_returns_none(self):
        assert compare_regions([], "region", ["revenue"]) is None

    def test_empty_metrics_returns_none(self):
        rows = [{"region": "A", "revenue": 100}]
        assert compare_regions(rows, "region", []) is None

    def test_all_null_region_returns_none(self):
        rows = [{"region": None, "revenue": 100}]
        assert compare_regions(rows, "region", ["revenue"]) is None

    def test_all_null_values_returns_none(self):
        rows = [{"region": "A", "revenue": None}]
        assert compare_regions(rows, "region", ["revenue"]) is None

    def test_comparison_count_matches_metrics(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue", "orders", "satisfaction"])
        assert result is not None
        assert len(result.comparisons) == 3

    def test_best_region_per_metric(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue"])
        assert result is not None
        rev_comp = result.comparisons[0]
        # East has 2000 avg, highest single region
        assert rev_comp.best_region == "East"

    def test_worst_region_per_metric(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue"])
        assert result is not None
        rev_comp = result.comparisons[0]
        # West has 500 avg, lowest
        assert rev_comp.worst_region == "West"

    def test_best_overall_region(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue", "orders"])
        assert result is not None
        assert result.best_overall is not None
        assert len(result.best_overall) > 0

    def test_region_scores_count(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue", "orders"])
        assert result is not None
        assert len(result.region_scores) == 4  # North, South, East, West

    def test_region_scores_sorted_descending(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue", "orders"])
        assert result is not None
        scores = [rs.overall_score for rs in result.region_scores]
        assert scores == sorted(scores, reverse=True)

    def test_single_metric(self):
        rows = [
            {"region": "A", "value": 100},
            {"region": "B", "value": 200},
        ]
        result = compare_regions(rows, "region", ["value"])
        assert result is not None
        assert len(result.comparisons) == 1

    def test_single_region(self):
        rows = [{"region": "Only", "revenue": 100}]
        result = compare_regions(rows, "region", ["revenue"])
        assert result is not None
        assert len(result.region_scores) == 1
        # With single region, std_dev is 0 -> z-score is 0
        assert result.region_scores[0].overall_score == 0.0

    def test_std_dev_zero_gives_zero_zscore(self):
        """All regions have same value -> std_dev = 0, z-score = 0."""
        rows = [
            {"region": "A", "revenue": 100},
            {"region": "B", "revenue": 100},
        ]
        result = compare_regions(rows, "region", ["revenue"])
        assert result is not None
        assert result.comparisons[0].std_dev == 0.0
        for rs in result.region_scores:
            assert rs.overall_score == 0.0

    def test_non_numeric_metric_skipped(self):
        rows = [
            {"region": "A", "revenue": "abc"},
            {"region": "B", "revenue": "def"},
        ]
        result = compare_regions(rows, "region", ["revenue"])
        assert result is None

    def test_mixed_valid_invalid_metrics(self):
        rows = [
            {"region": "A", "revenue": 100, "orders": "abc"},
            {"region": "B", "revenue": 200, "orders": 50},
        ]
        result = compare_regions(rows, "region", ["revenue", "orders"])
        assert result is not None
        # Both metrics should produce comparisons
        assert len(result.comparisons) == 2

    def test_missing_metric_column(self):
        """Missing metric column defaults to 0.0 per region."""
        rows = [{"region": "A", "revenue": 100}]
        result = compare_regions(rows, "region", ["revenue", "nonexistent"])
        assert result is not None
        # Both metrics produce comparisons; nonexistent gets 0.0
        assert len(result.comparisons) == 2
        nonexistent_comp = next(c for c in result.comparisons if c.metric == "nonexistent")
        assert nonexistent_comp.avg_value == 0.0

    def test_summary_contains_regions_and_metrics(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue", "orders"])
        assert result is not None
        assert "Region Comparison" in result.summary
        assert "4 regions" in result.summary
        assert "2 metrics" in result.summary

    def test_z_scores_in_metrics_dict(self):
        rows = _make_comparison_rows()
        result = compare_regions(rows, "region", ["revenue", "orders"])
        assert result is not None
        for rs in result.region_scores:
            assert "revenue" in rs.metrics
            assert "orders" in rs.metrics

    def test_avg_value_computed_correctly(self):
        rows = [
            {"region": "A", "m": 100},
            {"region": "B", "m": 200},
            {"region": "C", "m": 300},
        ]
        result = compare_regions(rows, "region", ["m"])
        assert result is not None
        assert result.comparisons[0].avg_value == 200.0


# ---------------------------------------------------------------------------
# Tests: analyze_geographic_growth
# ---------------------------------------------------------------------------


class TestAnalyzeGeographicGrowth:
    def test_basic_growth(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert isinstance(result, GeoGrowthResult)

    def test_empty_rows_returns_none(self):
        assert analyze_geographic_growth([], "region", "revenue", "date") is None

    def test_all_null_returns_none(self):
        rows = [{"region": None, "revenue": None, "date": None}]
        assert analyze_geographic_growth(rows, "region", "revenue", "date") is None

    def test_missing_date_column_returns_none(self):
        rows = [{"region": "A", "revenue": 100}]
        assert analyze_geographic_growth(rows, "region", "revenue", "date") is None

    def test_region_count(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert len(result.regions) == 3  # North, South, East

    def test_growth_rate_positive(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        north = next(rg for rg in result.regions if rg.region == "North")
        # North: first=1000, last=1500 -> growth = 50%
        assert north.growth_rate == 50.0

    def test_growth_rate_negative(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        south = next(rg for rg in result.regions if rg.region == "South")
        # South: first=800, last=700 -> growth = -12.5%
        assert south.growth_rate == -12.5

    def test_first_and_last_period_values(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        east = next(rg for rg in result.regions if rg.region == "East")
        assert east.first_period_value == 500.0
        assert east.last_period_value == 1000.0

    def test_periods_count(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        north = next(rg for rg in result.regions if rg.region == "North")
        assert north.periods == 3  # Jan, Feb, Mar

    def test_fastest_growing(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        # East: 500->1000 = 100%, North: 50%, South: -12.5%
        assert result.fastest_growing == "East"

    def test_slowest_growing(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert result.slowest_growing == "South"

    def test_avg_growth(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        # (50 + (-12.5) + 100) / 3 = 45.83
        assert abs(result.avg_growth - 45.83) < 0.1

    def test_single_period_zero_growth(self):
        rows = [{"region": "A", "revenue": 100, "date": "2024-01-15"}]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert result.regions[0].growth_rate == 0.0
        assert result.regions[0].periods == 1

    def test_growth_from_zero_is_inf(self):
        rows = [
            {"region": "A", "revenue": 0, "date": "2024-01-15"},
            {"region": "A", "revenue": 100, "date": "2024-02-15"},
        ]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert math.isinf(result.regions[0].growth_rate)

    def test_growth_zero_to_zero_is_zero(self):
        rows = [
            {"region": "A", "revenue": 0, "date": "2024-01-15"},
            {"region": "A", "revenue": 0, "date": "2024-02-15"},
        ]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert result.regions[0].growth_rate == 0.0

    def test_all_infinite_growth_fallback(self):
        rows = [
            {"region": "A", "revenue": 0, "date": "2024-01-15"},
            {"region": "A", "revenue": 100, "date": "2024-02-15"},
            {"region": "B", "revenue": 0, "date": "2024-01-15"},
            {"region": "B", "revenue": 200, "date": "2024-02-15"},
        ]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        # All infinite: fallback to first region
        assert result.avg_growth == 0.0

    def test_non_numeric_value_skipped(self):
        rows = [
            {"region": "A", "revenue": "abc", "date": "2024-01-15"},
            {"region": "B", "revenue": 100, "date": "2024-01-15"},
        ]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert len(result.regions) == 1
        assert result.regions[0].region == "B"

    def test_invalid_date_skipped(self):
        rows = [
            {"region": "A", "revenue": 100, "date": "not-a-date"},
            {"region": "B", "revenue": 200, "date": "2024-01-15"},
        ]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert len(result.regions) == 1

    def test_multiple_rows_same_period_aggregated(self):
        rows = [
            {"region": "A", "revenue": 100, "date": "2024-01-10"},
            {"region": "A", "revenue": 200, "date": "2024-01-20"},
            {"region": "A", "revenue": 500, "date": "2024-02-15"},
        ]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        a = result.regions[0]
        # Jan: 100+200=300, Feb: 500
        assert a.first_period_value == 300.0
        assert a.last_period_value == 500.0

    def test_summary_present(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert "Geographic Growth" in result.summary
        assert "Fastest" in result.summary

    def test_datetime_objects_accepted(self):
        rows = [
            {"region": "A", "revenue": 100, "date": datetime(2024, 1, 15)},
            {"region": "A", "revenue": 200, "date": datetime(2024, 2, 15)},
        ]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert result.regions[0].growth_rate == 100.0

    def test_date_objects_accepted(self):
        rows = [
            {"region": "A", "revenue": 100, "date": date(2024, 1, 15)},
            {"region": "A", "revenue": 200, "date": date(2024, 2, 15)},
        ]
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        assert len(result.regions) == 1

    def test_regions_sorted_alphabetically(self):
        rows = _make_growth_rows()
        result = analyze_geographic_growth(rows, "region", "revenue", "date")
        assert result is not None
        names = [rg.region for rg in result.regions]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Tests: compute_market_penetration
# ---------------------------------------------------------------------------


class TestComputeMarketPenetration:
    def test_basic_penetration(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        assert isinstance(result, MarketPenetrationResult)

    def test_empty_rows_returns_none(self):
        assert compute_market_penetration([], "region", "customer") is None

    def test_all_null_region_returns_none(self):
        rows = [{"region": None, "customer": "A"}]
        assert compute_market_penetration(rows, "region", "customer") is None

    def test_all_null_customer_returns_none(self):
        rows = [{"region": "North", "customer": None}]
        assert compute_market_penetration(rows, "region", "customer") is None

    def test_total_customers_unique(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        assert result.total_customers == 10  # All unique customers

    def test_total_regions(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        assert result.total_regions == 4

    def test_customer_count_per_region(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        west = next(rp for rp in result.regions if rp.region == "West")
        assert west.customer_count == 4

    def test_penetration_pct_with_potential(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        north = next(rp for rp in result.regions if rp.region == "North")
        # 3 customers / 1000 potential * 100 = 0.3%
        assert north.penetration_pct == 0.3

    def test_penetration_pct_none_without_potential(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        for rp in result.regions:
            assert rp.penetration_pct is None
            assert rp.potential is None

    def test_best_penetration_with_potential(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        # West: 4/200=2%, South: 2/500=0.4%, North: 3/1000=0.3%, East: 1/2000=0.05%
        assert result.best_penetration == "West"

    def test_best_penetration_none_without_potential(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        assert result.best_penetration is None

    def test_rank_ordering(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        ranks = [rp.rank for rp in result.regions]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_rank_by_customer_count_without_potential(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        # Sorted by customer_count descending: West(4), North(3), South(2), East(1)
        assert result.regions[0].region == "West"
        assert result.regions[-1].region == "East"

    def test_untapped_regions_with_potential(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        # All penetrations < 10%, so all should be untapped
        assert len(result.untapped_regions) == 4

    def test_untapped_regions_without_potential(self):
        """Without potential, untapped = regions with customer_count < avg * 0.5."""
        rows = [
            {"region": "A", "customer": "c1"},
            {"region": "A", "customer": "c2"},
            {"region": "A", "customer": "c3"},
            {"region": "A", "customer": "c4"},
            {"region": "A", "customer": "c5"},
            {"region": "A", "customer": "c6"},
            {"region": "A", "customer": "c7"},
            {"region": "A", "customer": "c8"},
            {"region": "B", "customer": "c9"},
        ]
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        # Total=9 unique, 2 regions -> avg=4.5, 50% threshold=2.25
        # B has 1 customer < 2.25 -> untapped
        assert "B" in result.untapped_regions

    def test_single_region(self):
        rows = [{"region": "Only", "customer": "A"}]
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        assert result.total_regions == 1
        assert result.total_customers == 1

    def test_duplicate_customer_in_region_counted_once(self):
        rows = [
            {"region": "A", "customer": "Alice"},
            {"region": "A", "customer": "Alice"},
            {"region": "A", "customer": "Alice"},
        ]
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        assert result.regions[0].customer_count == 1

    def test_same_customer_different_regions(self):
        rows = [
            {"region": "A", "customer": "Alice"},
            {"region": "B", "customer": "Alice"},
        ]
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        # Alice appears in both regions; each region gets count=1
        assert result.regions[0].customer_count == 1
        assert result.regions[1].customer_count == 1
        # But total_customers is unique across all regions = 1
        assert result.total_customers == 1

    def test_zero_potential_gives_zero_penetration(self):
        rows = [{"region": "A", "customer": "Alice", "potential": 0}]
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        assert result.regions[0].penetration_pct == 0.0

    def test_potential_uses_max_per_region(self):
        rows = [
            {"region": "A", "customer": "c1", "potential": 100},
            {"region": "A", "customer": "c2", "potential": 200},
        ]
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        a = result.regions[0]
        # Uses max potential: 200
        assert a.potential == 200.0

    def test_summary_present(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        assert "Market Penetration" in result.summary

    def test_summary_with_best_penetration(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        assert "Best penetration" in result.summary

    def test_summary_with_untapped(self):
        rows = _make_penetration_rows()
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        assert "Untapped regions" in result.summary

    def test_non_numeric_potential_ignored(self):
        rows = [{"region": "A", "customer": "c1", "potential": "abc"}]
        result = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        assert result is not None
        assert result.regions[0].potential is None
        assert result.regions[0].penetration_pct is None

    def test_no_untapped_without_potential_single_region(self):
        """Single region cannot be untapped without potential."""
        rows = [{"region": "A", "customer": "c1"}]
        result = compute_market_penetration(rows, "region", "customer")
        assert result is not None
        assert result.untapped_regions == []


# ---------------------------------------------------------------------------
# Tests: format_geo_report
# ---------------------------------------------------------------------------


class TestFormatGeoReport:
    def test_no_data(self):
        report = format_geo_report()
        assert "No data" in report

    def test_distribution_section(self):
        rows = _make_distribution_rows()
        dist = analyze_regional_distribution(rows, "region", "value")
        report = format_geo_report(distribution=dist)
        assert "Regional Distribution Report" in report
        assert "Total Value" in report
        assert "HHI" in report

    def test_comparison_section(self):
        rows = _make_comparison_rows()
        comp = compare_regions(rows, "region", ["revenue", "orders"])
        report = format_geo_report(comparison=comp)
        assert "Region Comparison Report" in report
        assert "Best Overall Region" in report

    def test_growth_section(self):
        rows = _make_growth_rows()
        growth = analyze_geographic_growth(rows, "region", "revenue", "date")
        report = format_geo_report(growth=growth)
        assert "Geographic Growth Report" in report
        assert "Fastest Growing" in report
        assert "Slowest Growing" in report

    def test_penetration_section(self):
        rows = _make_penetration_rows()
        pen = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        report = format_geo_report(penetration=pen)
        assert "Market Penetration Report" in report
        assert "Total Customers" in report

    def test_all_sections(self):
        dist = analyze_regional_distribution(
            _make_distribution_rows(), "region", "value"
        )
        comp = compare_regions(
            _make_comparison_rows(), "region", ["revenue", "orders"]
        )
        growth = analyze_geographic_growth(
            _make_growth_rows(), "region", "revenue", "date"
        )
        pen = compute_market_penetration(
            _make_penetration_rows(), "region", "customer",
            potential_column="potential",
        )
        report = format_geo_report(
            distribution=dist, comparison=comp,
            growth=growth, penetration=pen,
        )
        assert "Regional Distribution Report" in report
        assert "Region Comparison Report" in report
        assert "Geographic Growth Report" in report
        assert "Market Penetration Report" in report

    def test_distribution_contains_region_names(self):
        rows = _make_distribution_rows()
        dist = analyze_regional_distribution(rows, "region", "value")
        report = format_geo_report(distribution=dist)
        assert "North" in report
        assert "South" in report

    def test_growth_infinite_rate_displayed(self):
        rows = [
            {"region": "A", "revenue": 0, "date": "2024-01-15"},
            {"region": "A", "revenue": 100, "date": "2024-02-15"},
        ]
        growth = analyze_geographic_growth(rows, "region", "revenue", "date")
        report = format_geo_report(growth=growth)
        assert "inf%" in report

    def test_penetration_without_potential(self):
        rows = _make_penetration_rows()
        pen = compute_market_penetration(rows, "region", "customer")
        report = format_geo_report(penetration=pen)
        assert "N/A" in report

    def test_penetration_with_untapped(self):
        rows = _make_penetration_rows()
        pen = compute_market_penetration(
            rows, "region", "customer", potential_column="potential"
        )
        report = format_geo_report(penetration=pen)
        assert "Untapped" in report

    def test_none_sections_excluded(self):
        """Passing None for some sections does not produce their headers."""
        rows = _make_distribution_rows()
        dist = analyze_regional_distribution(rows, "region", "value")
        report = format_geo_report(distribution=dist, comparison=None)
        assert "Regional Distribution Report" in report
        assert "Region Comparison Report" not in report
