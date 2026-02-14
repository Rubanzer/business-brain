"""Tests for asset depreciation and lifecycle analysis module."""

from business_brain.discovery.asset_depreciation import (
    AssetAgeResult,
    AssetBookValue,
    AssetMaintRatio,
    AssetSchedule,
    BookValueResult,
    CategoryAge,
    DepreciationScheduleResult,
    LifecycleStage,
    MaintenanceCostResult,
    YearDepreciation,
    analyze_asset_age,
    analyze_maintenance_cost_ratio,
    compute_book_values,
    compute_depreciation_schedule,
    format_asset_report,
)


# ---------------------------------------------------------------------------
# compute_depreciation_schedule
# ---------------------------------------------------------------------------


class TestDepreciationScheduleEmpty:
    def test_empty_rows(self):
        assert compute_depreciation_schedule([], "a", "c", "l") is None

    def test_all_null_data(self):
        rows = [
            {"asset": None, "cost": None, "life": None},
            {"asset": "A", "cost": None, "life": 5},
        ]
        assert compute_depreciation_schedule(rows, "asset", "cost", "life") is None

    def test_zero_useful_life(self):
        rows = [{"asset": "A", "cost": 10000, "life": 0}]
        assert compute_depreciation_schedule(rows, "asset", "cost", "life") is None

    def test_negative_useful_life(self):
        rows = [{"asset": "A", "cost": 10000, "life": -3}]
        assert compute_depreciation_schedule(rows, "asset", "cost", "life") is None

    def test_non_numeric_cost(self):
        rows = [{"asset": "A", "cost": "bad", "life": 5}]
        assert compute_depreciation_schedule(rows, "asset", "cost", "life") is None


class TestStraightLineDepreciation:
    def test_basic_straight_line(self):
        rows = [{"asset": "Truck", "cost": 50000, "life": 5}]
        result = compute_depreciation_schedule(rows, "asset", "cost", "life")
        assert result is not None
        assert result.method == "straight_line"
        assert len(result.assets) == 1
        asset = result.assets[0]
        assert asset.asset == "Truck"
        assert asset.cost == 50000
        assert asset.salvage == 0
        assert asset.useful_life == 5
        assert asset.annual_depreciation == 10000
        assert len(asset.schedule) == 5

    def test_straight_line_year_values(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5}]
        result = compute_depreciation_schedule(rows, "asset", "cost", "life")
        sched = result.assets[0].schedule
        assert sched[0].year == 1
        assert sched[0].depreciation_amount == 2000
        assert sched[0].book_value == 8000
        assert sched[4].year == 5
        assert sched[4].book_value == 0

    def test_straight_line_with_salvage(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5, "salvage": 2000}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", salvage_column="salvage"
        )
        asset = result.assets[0]
        assert asset.salvage == 2000
        assert asset.annual_depreciation == 1600  # (10000-2000)/5
        assert asset.schedule[-1].book_value == 2000

    def test_straight_line_multiple_assets(self):
        rows = [
            {"asset": "A", "cost": 10000, "life": 5},
            {"asset": "B", "cost": 20000, "life": 10},
        ]
        result = compute_depreciation_schedule(rows, "asset", "cost", "life")
        assert len(result.assets) == 2
        assert result.total_cost == 30000
        # A: 2000/yr + B: 2000/yr = 4000
        assert result.total_annual_depreciation == 4000

    def test_single_year_life(self):
        rows = [{"asset": "A", "cost": 5000, "life": 1}]
        result = compute_depreciation_schedule(rows, "asset", "cost", "life")
        asset = result.assets[0]
        assert asset.annual_depreciation == 5000
        assert len(asset.schedule) == 1
        assert asset.schedule[0].book_value == 0

    def test_duplicate_asset_uses_first(self):
        rows = [
            {"asset": "A", "cost": 10000, "life": 5},
            {"asset": "A", "cost": 99999, "life": 3},
        ]
        result = compute_depreciation_schedule(rows, "asset", "cost", "life")
        assert len(result.assets) == 1
        assert result.assets[0].cost == 10000

    def test_summary_text(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5}]
        result = compute_depreciation_schedule(rows, "asset", "cost", "life")
        assert "straight_line" in result.summary
        assert "1 assets" in result.summary

    def test_string_numeric_values(self):
        rows = [{"asset": "A", "cost": "10000", "life": "5"}]
        result = compute_depreciation_schedule(rows, "asset", "cost", "life")
        assert result is not None
        assert result.assets[0].cost == 10000


class TestDecliningBalanceDepreciation:
    def test_basic_declining_balance(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", method="declining_balance"
        )
        assert result is not None
        assert result.method == "declining_balance"
        asset = result.assets[0]
        # Rate = 2/5 = 0.4; first year = 10000 * 0.4 = 4000
        assert asset.annual_depreciation == 4000
        assert asset.schedule[0].depreciation_amount == 4000
        assert asset.schedule[0].book_value == 6000

    def test_declining_balance_decreasing_depreciation(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", method="declining_balance"
        )
        deps = [y.depreciation_amount for y in result.assets[0].schedule]
        # Each year's depreciation should be <= previous year
        for i in range(1, len(deps)):
            assert deps[i] <= deps[i - 1]

    def test_declining_balance_with_salvage(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5, "salvage": 1000}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life",
            method="declining_balance", salvage_column="salvage",
        )
        asset = result.assets[0]
        # Book value should never go below salvage
        for yd in asset.schedule:
            assert yd.book_value >= 1000 - 0.01

    def test_declining_balance_book_values_decreasing(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", method="declining_balance"
        )
        books = [y.book_value for y in result.assets[0].schedule]
        for i in range(1, len(books)):
            assert books[i] <= books[i - 1]

    def test_declining_balance_multiple_assets(self):
        rows = [
            {"asset": "A", "cost": 10000, "life": 5},
            {"asset": "B", "cost": 20000, "life": 4},
        ]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", method="declining_balance"
        )
        assert len(result.assets) == 2
        # A: 10000 * 0.4 = 4000; B: 20000 * 0.5 = 10000
        assert result.total_annual_depreciation == 14000


class TestSumOfYearsDigitsDepreciation:
    def test_basic_syd(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", method="sum_of_years_digits"
        )
        assert result is not None
        assert result.method == "sum_of_years_digits"
        asset = result.assets[0]
        # SoY = 15; year 1 = 10000 * 5/15 = 3333.3333
        assert abs(asset.annual_depreciation - 3333.3333) < 0.01

    def test_syd_decreasing_depreciation(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", method="sum_of_years_digits"
        )
        deps = [y.depreciation_amount for y in result.assets[0].schedule]
        for i in range(1, len(deps)):
            assert deps[i] < deps[i - 1]

    def test_syd_with_salvage(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5, "salvage": 2000}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life",
            method="sum_of_years_digits", salvage_column="salvage",
        )
        asset = result.assets[0]
        # SoY = 15; depreciable = 8000; year 1 = 8000 * 5/15 = 2666.67
        assert abs(asset.annual_depreciation - 2666.6667) < 0.01
        # Final book value should be at salvage
        assert abs(asset.schedule[-1].book_value - 2000) < 0.01

    def test_syd_total_depreciation_equals_depreciable_amount(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5, "salvage": 1500}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life",
            method="sum_of_years_digits", salvage_column="salvage",
        )
        total_dep = sum(y.depreciation_amount for y in result.assets[0].schedule)
        assert abs(total_dep - 8500) < 0.1

    def test_syd_single_year_life(self):
        rows = [{"asset": "A", "cost": 5000, "life": 1}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", method="sum_of_years_digits"
        )
        asset = result.assets[0]
        assert asset.annual_depreciation == 5000
        assert asset.schedule[0].book_value == 0


# ---------------------------------------------------------------------------
# analyze_asset_age
# ---------------------------------------------------------------------------


class TestAssetAgeEmpty:
    def test_empty_rows(self):
        assert analyze_asset_age([], "asset", "date") is None

    def test_all_null_dates(self):
        rows = [
            {"asset": "A", "date": None},
            {"asset": "B", "date": "not-a-date"},
        ]
        assert analyze_asset_age(rows, "asset", "date") is None

    def test_null_asset_names(self):
        rows = [{"asset": None, "date": "2020-01-01"}]
        assert analyze_asset_age(rows, "asset", "date") is None


class TestAssetAgeAnalysis:
    def test_basic_age_analysis(self):
        rows = [
            {"asset": "A", "date": "2020-01-01"},
            {"asset": "B", "date": "2023-01-01"},
            {"asset": "C", "date": "2024-06-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        assert result is not None
        assert result.total_assets == 3
        assert result.avg_age > 0

    def test_single_asset(self):
        rows = [{"asset": "A", "date": "2020-01-01"}]
        result = analyze_asset_age(rows, "asset", "date")
        assert result is not None
        assert result.total_assets == 1
        # Single asset relative to itself = 0 age
        assert result.avg_age == 0.0

    def test_lifecycle_stage_new(self):
        # All assets purchased within 2 years of the max date
        rows = [
            {"asset": "A", "date": "2024-01-01"},
            {"asset": "B", "date": "2024-06-01"},
            {"asset": "C", "date": "2024-12-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        new_stage = next(s for s in result.by_lifecycle_stage if s.stage == "New")
        assert new_stage.count == 3
        assert new_stage.pct == 100.0

    def test_lifecycle_stage_midlife(self):
        # Max date = 2024-01-01; asset from 2020-06-01 = ~3.5 years => Mid-life
        rows = [
            {"asset": "A", "date": "2020-06-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        midlife = next(s for s in result.by_lifecycle_stage if s.stage == "Mid-life")
        assert midlife.count == 1

    def test_lifecycle_stage_aging(self):
        # Max date = 2024-01-01; asset from 2017-01-01 = 7 years => Aging
        rows = [
            {"asset": "A", "date": "2017-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        aging = next(s for s in result.by_lifecycle_stage if s.stage == "Aging")
        assert aging.count == 1

    def test_lifecycle_stage_end_of_life(self):
        # Max date = 2024-01-01; asset from 2010-01-01 = 14 years => End-of-life
        rows = [
            {"asset": "A", "date": "2010-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        eol = next(s for s in result.by_lifecycle_stage if s.stage == "End-of-life")
        assert eol.count == 1

    def test_lifecycle_boundary_exactly_2_years(self):
        # Exactly 2 years => "New" (<=2)
        rows = [
            {"asset": "A", "date": "2022-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        # A is ~2.0 years old relative to B
        new_stage = next(s for s in result.by_lifecycle_stage if s.stage == "New")
        assert new_stage.count == 2  # both are <=2 (B is 0, A is ~2)

    def test_lifecycle_boundary_exactly_5_years(self):
        # Exactly 5 years => "Mid-life" (<=5)
        rows = [
            {"asset": "A", "date": "2019-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        midlife = next(s for s in result.by_lifecycle_stage if s.stage == "Mid-life")
        assert midlife.count == 1  # A is ~5 years

    def test_lifecycle_boundary_exactly_10_years(self):
        # ~10 years => "Aging" (<=10)
        rows = [
            {"asset": "A", "date": "2014-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        aging = next(s for s in result.by_lifecycle_stage if s.stage == "Aging")
        assert aging.count == 1  # A is ~10 years

    def test_with_category(self):
        rows = [
            {"asset": "A", "date": "2020-01-01", "cat": "Vehicle"},
            {"asset": "B", "date": "2022-01-01", "cat": "Vehicle"},
            {"asset": "C", "date": "2023-01-01", "cat": "Equipment"},
            {"asset": "D", "date": "2024-01-01", "cat": "Equipment"},
        ]
        result = analyze_asset_age(rows, "asset", "date", category_column="cat")
        assert result.by_category is not None
        assert len(result.by_category) == 2
        eq = next(c for c in result.by_category if c.category == "Equipment")
        veh = next(c for c in result.by_category if c.category == "Vehicle")
        assert eq.count == 2
        assert veh.count == 2
        assert veh.avg_age > eq.avg_age

    def test_weighted_avg_age(self):
        # Two assets: A expensive + old, B cheap + new
        rows = [
            {"asset": "A", "date": "2018-01-01", "cost": 100000},
            {"asset": "B", "date": "2024-01-01", "cost": 10000},
        ]
        result = analyze_asset_age(rows, "asset", "date", cost_column="cost")
        assert result.weighted_avg_age is not None
        # Weighted should be closer to A's age since A is much more expensive
        assert result.weighted_avg_age > result.avg_age

    def test_weighted_avg_age_not_returned_without_cost(self):
        rows = [
            {"asset": "A", "date": "2020-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        assert result.weighted_avg_age is None

    def test_category_not_returned_without_category(self):
        rows = [
            {"asset": "A", "date": "2020-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        assert result.by_category is None

    def test_duplicate_asset_uses_first(self):
        rows = [
            {"asset": "A", "date": "2020-01-01"},
            {"asset": "A", "date": "2024-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        assert result.total_assets == 2

    def test_summary_text(self):
        rows = [
            {"asset": "A", "date": "2020-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        assert "2 assets" in result.summary
        assert "Average age" in result.summary

    def test_datetime_objects_as_dates(self):
        from datetime import datetime
        rows = [
            {"asset": "A", "date": datetime(2020, 1, 1)},
            {"asset": "B", "date": datetime(2024, 1, 1)},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        assert result is not None
        assert result.total_assets == 2

    def test_mixed_date_formats(self):
        rows = [
            {"asset": "A", "date": "01/01/2020"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        assert result is not None
        assert result.total_assets == 2


# ---------------------------------------------------------------------------
# compute_book_values
# ---------------------------------------------------------------------------


class TestBookValuesEmpty:
    def test_empty_rows(self):
        assert compute_book_values([], "a", "c", "d", "l") is None

    def test_all_null_data(self):
        rows = [{"asset": None, "cost": None, "date": None, "life": None}]
        assert compute_book_values(rows, "asset", "cost", "date", "life") is None

    def test_no_valid_dates(self):
        rows = [{"asset": "A", "cost": 10000, "date": "baddate", "life": 5}]
        assert compute_book_values(rows, "asset", "cost", "date", "life") is None


class TestBookValues:
    def test_basic_book_value(self):
        # Asset purchased at ref date => age=0, book = cost
        rows = [
            {"asset": "A", "cost": 10000, "date": "2024-01-01", "life": 5},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        assert result is not None
        assert len(result.assets) == 1
        assert result.assets[0].book_value == 10000
        assert result.assets[0].depreciation_pct == 0.0

    def test_half_depreciated(self):
        # A is 2.5 years old with 5-year life => 50% depreciated
        rows = [
            {"asset": "A", "cost": 10000, "date": "2021-07-01", "life": 5},
            {"asset": "B", "cost": 5000, "date": "2024-01-01", "life": 5},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        assert result is not None
        a = next(x for x in result.assets if x.asset == "A")
        # Age ~ 2.5 years; dep = 10000/5 * 2.5 = 5000; book = 5000
        assert abs(a.book_value - 5000) < 200  # approximate due to 365.25

    def test_fully_depreciated(self):
        # Asset 10 years old, life = 5 => fully depreciated
        rows = [
            {"asset": "A", "cost": 10000, "date": "2014-01-01", "life": 5},
            {"asset": "B", "cost": 5000, "date": "2024-01-01", "life": 5},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        a = next(x for x in result.assets if x.asset == "A")
        assert a.book_value == 0
        assert a.depreciation_pct == 100.0
        assert result.fully_depreciated_count >= 1

    def test_with_salvage_value(self):
        # Fully depreciated with salvage = 2000
        rows = [
            {"asset": "A", "cost": 10000, "date": "2014-01-01", "life": 5, "salvage": 2000},
            {"asset": "B", "cost": 5000, "date": "2024-01-01", "life": 5, "salvage": 500},
        ]
        result = compute_book_values(
            rows, "asset", "cost", "date", "life", salvage_column="salvage"
        )
        a = next(x for x in result.assets if x.asset == "A")
        assert a.book_value == 2000  # min is salvage
        assert a.salvage_value == 2000

    def test_total_values(self):
        rows = [
            {"asset": "A", "cost": 10000, "date": "2024-01-01", "life": 5},
            {"asset": "B", "cost": 20000, "date": "2024-01-01", "life": 10},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        assert result.total_original_cost == 30000
        assert result.total_book_value == 30000  # both age = 0

    def test_depreciation_percentage(self):
        # All at ref date => 0% depreciated
        rows = [
            {"asset": "A", "cost": 10000, "date": "2024-01-01", "life": 5},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        assert result.depreciation_pct == 0.0

    def test_fully_depreciated_count(self):
        rows = [
            {"asset": "A", "cost": 10000, "date": "2010-01-01", "life": 5},
            {"asset": "B", "cost": 10000, "date": "2012-01-01", "life": 5},
            {"asset": "C", "cost": 10000, "date": "2024-01-01", "life": 5},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        assert result.fully_depreciated_count == 2  # A and B are fully depreciated

    def test_summary_text(self):
        rows = [
            {"asset": "A", "cost": 10000, "date": "2024-01-01", "life": 5},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        assert "1 assets" in result.summary
        assert "book value" in result.summary.lower()

    def test_duplicate_asset_uses_first(self):
        rows = [
            {"asset": "A", "cost": 10000, "date": "2024-01-01", "life": 5},
            {"asset": "A", "cost": 99999, "date": "2020-01-01", "life": 3},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        assert len(result.assets) == 1
        assert result.assets[0].original_cost == 10000

    def test_string_cost_values(self):
        rows = [
            {"asset": "A", "cost": "10000", "date": "2024-01-01", "life": "5"},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        assert result is not None
        assert result.assets[0].original_cost == 10000

    def test_age_years_computed(self):
        rows = [
            {"asset": "A", "cost": 10000, "date": "2020-01-01", "life": 10},
            {"asset": "B", "cost": 5000, "date": "2024-01-01", "life": 5},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        a = next(x for x in result.assets if x.asset == "A")
        b = next(x for x in result.assets if x.asset == "B")
        assert a.age_years > 3.5  # ~4 years
        assert b.age_years == 0.0


# ---------------------------------------------------------------------------
# analyze_maintenance_cost_ratio
# ---------------------------------------------------------------------------


class TestMaintenanceCostEmpty:
    def test_empty_rows(self):
        assert analyze_maintenance_cost_ratio([], "a", "m", "v") is None

    def test_all_null_data(self):
        rows = [{"asset": None, "maint": None, "value": None}]
        assert analyze_maintenance_cost_ratio(rows, "asset", "maint", "value") is None

    def test_non_numeric_values(self):
        rows = [{"asset": "A", "maint": "bad", "value": "bad"}]
        assert analyze_maintenance_cost_ratio(rows, "asset", "maint", "value") is None


class TestMaintenanceCostRatio:
    def test_basic_ratio(self):
        rows = [
            {"asset": "A", "maint": 5000, "value": 10000},
            {"asset": "B", "maint": 2000, "value": 20000},
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert result is not None
        assert len(result.assets) == 2
        a = next(x for x in result.assets if x.asset == "A")
        assert a.ratio_pct == 50.0
        assert a.is_replacement_candidate is False  # 50% is not > 50%

    def test_replacement_candidate(self):
        rows = [
            {"asset": "A", "maint": 6000, "value": 10000},
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        a = result.assets[0]
        assert a.ratio_pct == 60.0
        assert a.is_replacement_candidate is True
        assert result.replacement_candidates == 1

    def test_no_replacement_candidates(self):
        rows = [
            {"asset": "A", "maint": 1000, "value": 10000},
            {"asset": "B", "maint": 500, "value": 10000},
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert result.replacement_candidates == 0

    def test_average_ratio(self):
        rows = [
            {"asset": "A", "maint": 3000, "value": 10000},  # 30%
            {"asset": "B", "maint": 5000, "value": 10000},  # 50%
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert result.avg_ratio == 40.0

    def test_total_values(self):
        rows = [
            {"asset": "A", "maint": 3000, "value": 10000},
            {"asset": "B", "maint": 5000, "value": 20000},
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert result.total_maintenance == 8000

    def test_multiple_rows_same_asset(self):
        rows = [
            {"asset": "A", "maint": 1000, "value": 10000},
            {"asset": "A", "maint": 2000, "value": 10000},
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert len(result.assets) == 1
        a = result.assets[0]
        assert a.maintenance_cost == 3000  # summed
        # value is averaged: (10000+10000)/2 = 10000
        assert a.asset_value == 10000
        assert a.ratio_pct == 30.0

    def test_zero_asset_value(self):
        rows = [
            {"asset": "A", "maint": 5000, "value": 0},
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        a = result.assets[0]
        assert a.ratio_pct == 0.0
        assert a.is_replacement_candidate is False

    def test_single_asset(self):
        rows = [{"asset": "A", "maint": 200, "value": 1000}]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert result is not None
        assert result.assets[0].ratio_pct == 20.0

    def test_replacement_boundary_exactly_50(self):
        rows = [{"asset": "A", "maint": 5000, "value": 10000}]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        # 50% is NOT > 50%, so not a replacement candidate
        assert result.assets[0].is_replacement_candidate is False

    def test_replacement_boundary_just_over_50(self):
        rows = [{"asset": "A", "maint": 5001, "value": 10000}]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert result.assets[0].is_replacement_candidate is True

    def test_summary_text(self):
        rows = [
            {"asset": "A", "maint": 5000, "value": 10000},
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert "1 assets" in result.summary
        assert "replacement" in result.summary.lower()

    def test_string_numeric_values(self):
        rows = [{"asset": "A", "maint": "3000", "value": "10000"}]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert result is not None
        assert result.assets[0].ratio_pct == 30.0


# ---------------------------------------------------------------------------
# format_asset_report
# ---------------------------------------------------------------------------


class TestFormatAssetReport:
    def test_no_sections(self):
        report = format_asset_report()
        assert "Asset Depreciation & Lifecycle Report" in report
        assert "No analysis data provided." in report

    def test_depreciation_section_only(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5}]
        dep = compute_depreciation_schedule(rows, "asset", "cost", "life")
        report = format_asset_report(depreciation=dep)
        assert "Depreciation Schedule" in report
        assert "straight_line" in report
        assert "No analysis data provided." not in report

    def test_age_section_only(self):
        rows = [
            {"asset": "A", "date": "2020-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        age = analyze_asset_age(rows, "asset", "date")
        report = format_asset_report(age=age)
        assert "Asset Age Analysis" in report
        assert "Total assets: 2" in report

    def test_book_values_section_only(self):
        rows = [
            {"asset": "A", "cost": 10000, "date": "2024-01-01", "life": 5},
        ]
        bv = compute_book_values(rows, "asset", "cost", "date", "life")
        report = format_asset_report(book_values=bv)
        assert "Book Values" in report
        assert "10,000.00" in report

    def test_maintenance_section_only(self):
        rows = [
            {"asset": "A", "maint": 6000, "value": 10000},
        ]
        maint = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        report = format_asset_report(maintenance=maint)
        assert "Maintenance Cost Analysis" in report
        assert "[REPLACE]" in report

    def test_all_sections(self):
        dep_rows = [{"asset": "A", "cost": 10000, "life": 5}]
        dep = compute_depreciation_schedule(dep_rows, "asset", "cost", "life")

        age_rows = [
            {"asset": "A", "date": "2020-01-01"},
            {"asset": "B", "date": "2024-01-01"},
        ]
        age = analyze_asset_age(age_rows, "asset", "date")

        bv_rows = [
            {"asset": "A", "cost": 10000, "date": "2024-01-01", "life": 5},
        ]
        bv = compute_book_values(bv_rows, "asset", "cost", "date", "life")

        maint_rows = [{"asset": "A", "maint": 3000, "value": 10000}]
        maint = analyze_maintenance_cost_ratio(maint_rows, "asset", "maint", "value")

        report = format_asset_report(
            depreciation=dep, age=age, book_values=bv, maintenance=maint
        )
        assert "Depreciation Schedule" in report
        assert "Asset Age Analysis" in report
        assert "Book Values" in report
        assert "Maintenance Cost Analysis" in report
        assert "No analysis data provided." not in report

    def test_report_with_category_age(self):
        rows = [
            {"asset": "A", "date": "2020-01-01", "cat": "Vehicle"},
            {"asset": "B", "date": "2024-01-01", "cat": "Equipment"},
        ]
        age = analyze_asset_age(rows, "asset", "date", category_column="cat")
        report = format_asset_report(age=age)
        assert "By category:" in report
        assert "Vehicle" in report
        assert "Equipment" in report

    def test_report_with_weighted_age(self):
        rows = [
            {"asset": "A", "date": "2020-01-01", "cost": 50000},
            {"asset": "B", "date": "2024-01-01", "cost": 10000},
        ]
        age = analyze_asset_age(rows, "asset", "date", cost_column="cost")
        report = format_asset_report(age=age)
        assert "Weighted avg age" in report

    def test_report_no_replace_flag(self):
        rows = [{"asset": "A", "maint": 1000, "value": 10000}]
        maint = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        report = format_asset_report(maintenance=maint)
        assert "[REPLACE]" not in report


# ---------------------------------------------------------------------------
# Dataclass construction tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_year_depreciation(self):
        yd = YearDepreciation(year=1, depreciation_amount=2000.0, book_value=8000.0)
        assert yd.year == 1
        assert yd.depreciation_amount == 2000.0
        assert yd.book_value == 8000.0

    def test_asset_schedule(self):
        sched = AssetSchedule(
            asset="Truck",
            cost=50000.0,
            salvage=5000.0,
            useful_life=5,
            annual_depreciation=9000.0,
            schedule=[],
        )
        assert sched.asset == "Truck"
        assert sched.useful_life == 5

    def test_depreciation_schedule_result(self):
        result = DepreciationScheduleResult(
            assets=[], total_cost=0.0, total_annual_depreciation=0.0,
            method="straight_line", summary="test",
        )
        assert result.method == "straight_line"

    def test_lifecycle_stage(self):
        ls = LifecycleStage(stage="New", count=5, pct=50.0)
        assert ls.stage == "New"

    def test_category_age(self):
        ca = CategoryAge(category="Vehicle", count=3, avg_age=4.5)
        assert ca.category == "Vehicle"

    def test_asset_age_result(self):
        result = AssetAgeResult(
            total_assets=10, avg_age=5.0, by_lifecycle_stage=[],
            by_category=None, weighted_avg_age=None, summary="test",
        )
        assert result.total_assets == 10

    def test_asset_book_value(self):
        abv = AssetBookValue(
            asset="A", original_cost=10000.0, salvage_value=1000.0,
            age_years=3.0, book_value=7300.0, depreciation_pct=27.0,
        )
        assert abv.asset == "A"

    def test_book_value_result(self):
        result = BookValueResult(
            assets=[], total_original_cost=0.0, total_book_value=0.0,
            depreciation_pct=0.0, fully_depreciated_count=0, summary="test",
        )
        assert result.fully_depreciated_count == 0

    def test_asset_maint_ratio(self):
        amr = AssetMaintRatio(
            asset="A", maintenance_cost=5000.0, asset_value=10000.0,
            ratio_pct=50.0, is_replacement_candidate=False,
        )
        assert amr.ratio_pct == 50.0

    def test_maintenance_cost_result(self):
        result = MaintenanceCostResult(
            assets=[], avg_ratio=0.0, replacement_candidates=0,
            total_maintenance=0.0, total_asset_value=0.0, summary="test",
        )
        assert result.replacement_candidates == 0


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_depreciation_large_salvage_equals_cost(self):
        rows = [{"asset": "A", "cost": 10000, "life": 5, "salvage": 10000}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", salvage_column="salvage"
        )
        assert result is not None
        asset = result.assets[0]
        assert asset.annual_depreciation == 0.0
        for yd in asset.schedule:
            assert yd.book_value == 10000

    def test_depreciation_very_long_life(self):
        rows = [{"asset": "A", "cost": 100000, "life": 50}]
        result = compute_depreciation_schedule(rows, "asset", "cost", "life")
        assert result is not None
        assert len(result.assets[0].schedule) == 50
        assert result.assets[0].annual_depreciation == 2000

    def test_book_value_new_asset_at_ref_date(self):
        rows = [
            {"asset": "A", "cost": 50000, "date": "2024-06-01", "life": 10},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        # Single asset, ref_date = its own date, age = 0
        assert result.assets[0].book_value == 50000

    def test_maintenance_many_assets(self):
        rows = [
            {"asset": f"Asset_{i}", "maint": 1000 * i, "value": 10000}
            for i in range(1, 11)
        ]
        result = analyze_maintenance_cost_ratio(rows, "asset", "maint", "value")
        assert len(result.assets) == 10
        # Assets 6-10 have maint > 50% of value
        assert result.replacement_candidates == 5

    def test_age_analysis_all_same_date(self):
        rows = [
            {"asset": "A", "date": "2024-01-01"},
            {"asset": "B", "date": "2024-01-01"},
            {"asset": "C", "date": "2024-01-01"},
        ]
        result = analyze_asset_age(rows, "asset", "date")
        assert result.avg_age == 0.0
        new_stage = next(s for s in result.by_lifecycle_stage if s.stage == "New")
        assert new_stage.count == 3

    def test_declining_balance_salvage_floor(self):
        # With high salvage, declining balance should not go below salvage
        rows = [{"asset": "A", "cost": 10000, "life": 3, "salvage": 5000}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life",
            method="declining_balance", salvage_column="salvage",
        )
        for yd in result.assets[0].schedule:
            assert yd.book_value >= 5000 - 0.01

    def test_syd_three_year_life(self):
        rows = [{"asset": "A", "cost": 6000, "life": 3}]
        result = compute_depreciation_schedule(
            rows, "asset", "cost", "life", method="sum_of_years_digits"
        )
        # SoY = 6; year1=6000*3/6=3000, year2=6000*2/6=2000, year3=6000*1/6=1000
        sched = result.assets[0].schedule
        assert sched[0].depreciation_amount == 3000
        assert sched[1].depreciation_amount == 2000
        assert sched[2].depreciation_amount == 1000
        assert abs(sched[2].book_value) < 0.01

    def test_book_value_partial_year_depreciation(self):
        # Asset 1 year old in a 10-year life
        rows = [
            {"asset": "A", "cost": 10000, "date": "2023-01-01", "life": 10},
            {"asset": "B", "cost": 10000, "date": "2024-01-01", "life": 10},
        ]
        result = compute_book_values(rows, "asset", "cost", "date", "life")
        a = next(x for x in result.assets if x.asset == "A")
        # ~1 year old, dep = 1000/yr, book ~ 9000
        assert 8800 < a.book_value < 9200

    def test_maintenance_with_date_column_accepted(self):
        rows = [
            {"asset": "A", "maint": 1000, "value": 10000, "date": "2024-01"},
            {"asset": "A", "maint": 1500, "value": 10000, "date": "2024-02"},
        ]
        result = analyze_maintenance_cost_ratio(
            rows, "asset", "maint", "value", date_column="date"
        )
        assert result is not None
        assert result.assets[0].maintenance_cost == 2500
