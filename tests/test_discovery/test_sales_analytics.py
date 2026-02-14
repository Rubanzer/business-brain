"""Tests for sales analytics module."""

import math

from business_brain.discovery.sales_analytics import (
    DiscountBucket,
    DiscountResult,
    PeriodSales,
    ProductDiscount,
    ProductMixResult,
    ProductRevenue,
    RegionSales,
    RepSales,
    SalesResult,
    VelocityResult,
    _parse_date,
    _safe_float,
    analyze_discount_impact,
    analyze_product_mix,
    analyze_sales_performance,
    compute_sales_velocity,
    format_sales_report,
)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("100.5") == 100.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None


class TestParseDate:
    def test_iso_format(self):
        dt = _parse_date("2024-01-15")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_slash_format(self):
        dt = _parse_date("2024/03/20")
        assert dt is not None
        assert dt.month == 3

    def test_us_format(self):
        dt = _parse_date("12/25/2024")
        assert dt is not None
        assert dt.month == 12
        assert dt.day == 25

    def test_datetime_passthrough(self):
        from datetime import datetime
        original = datetime(2024, 6, 1)
        assert _parse_date(original) is original

    def test_none(self):
        assert _parse_date(None) is None

    def test_invalid(self):
        assert _parse_date("not-a-date") is None

    def test_datetime_with_time(self):
        dt = _parse_date("2024-01-15 10:30:00")
        assert dt is not None
        assert dt.hour == 10


# ---------------------------------------------------------------------------
# analyze_sales_performance
# ---------------------------------------------------------------------------


class TestAnalyzeSalesPerformance:
    def test_empty_rows(self):
        assert analyze_sales_performance([], "amount", "date") is None

    def test_all_invalid_data(self):
        rows = [{"amount": "bad", "date": "Q1"}]
        assert analyze_sales_performance(rows, "amount", "date") is None

    def test_single_transaction(self):
        rows = [{"amount": 1000, "date": "Q1"}]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.total_sales == 1000.0
        assert result.period_count == 1
        assert result.avg_per_period == 1000.0
        assert result.growth_rate == 0.0

    def test_basic_performance(self):
        rows = [
            {"amount": 100, "date": "Q1"},
            {"amount": 200, "date": "Q2"},
            {"amount": 300, "date": "Q3"},
            {"amount": 400, "date": "Q4"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.total_sales == 1000.0
        assert result.period_count == 4
        assert result.avg_per_period == 250.0

    def test_growth_rate_positive(self):
        rows = [
            {"amount": 100, "date": "Q1"},
            {"amount": 100, "date": "Q2"},
            {"amount": 200, "date": "Q3"},
            {"amount": 200, "date": "Q4"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        # first half = Q1+Q2 = 200, second half = Q3+Q4 = 400
        assert result.growth_rate == 100.0

    def test_growth_rate_negative(self):
        rows = [
            {"amount": 400, "date": "Q1"},
            {"amount": 400, "date": "Q2"},
            {"amount": 100, "date": "Q3"},
            {"amount": 100, "date": "Q4"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.growth_rate == -75.0

    def test_best_worst_period(self):
        rows = [
            {"amount": 100, "date": "Jan"},
            {"amount": 500, "date": "Feb"},
            {"amount": 50, "date": "Mar"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.best_period.period == "Feb"
        assert result.best_period.amount == 500.0
        assert result.worst_period.period == "Mar"
        assert result.worst_period.amount == 50.0

    def test_multiple_transactions_per_period(self):
        rows = [
            {"amount": 100, "date": "Q1"},
            {"amount": 200, "date": "Q1"},
            {"amount": 300, "date": "Q2"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.period_count == 2
        assert result.total_sales == 600.0

    def test_with_rep_column(self):
        rows = [
            {"amount": 100, "date": "Q1", "rep": "Alice"},
            {"amount": 200, "date": "Q1", "rep": "Bob"},
            {"amount": 300, "date": "Q2", "rep": "Alice"},
        ]
        result = analyze_sales_performance(rows, "amount", "date", rep_column="rep")
        assert result is not None
        assert len(result.by_rep) == 2
        # Alice total=400, Bob total=200 => Alice rank 1
        assert result.by_rep[0].rep == "Alice"
        assert result.by_rep[0].rank == 1
        assert result.by_rep[0].total == 400.0
        assert result.by_rep[1].rep == "Bob"
        assert result.by_rep[1].rank == 2

    def test_with_region_column(self):
        rows = [
            {"amount": 100, "date": "Q1", "region": "East"},
            {"amount": 200, "date": "Q1", "region": "West"},
            {"amount": 300, "date": "Q2", "region": "East"},
        ]
        result = analyze_sales_performance(rows, "amount", "date", region_column="region")
        assert result is not None
        assert len(result.by_region) == 2
        east = next(r for r in result.by_region if r.region == "East")
        assert east.total == 400.0
        west = next(r for r in result.by_region if r.region == "West")
        assert west.total == 200.0
        # Percentages should sum to 100
        total_pct = sum(r.pct for r in result.by_region)
        assert abs(total_pct - 100.0) < 0.2

    def test_with_both_rep_and_region(self):
        rows = [
            {"amount": 100, "date": "Q1", "rep": "Alice", "region": "East"},
            {"amount": 200, "date": "Q2", "rep": "Bob", "region": "West"},
        ]
        result = analyze_sales_performance(
            rows, "amount", "date", rep_column="rep", region_column="region"
        )
        assert result is not None
        assert len(result.by_rep) == 2
        assert len(result.by_region) == 2

    def test_missing_amount_skipped(self):
        rows = [
            {"amount": 100, "date": "Q1"},
            {"amount": None, "date": "Q2"},
            {"amount": 200, "date": "Q3"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.total_sales == 300.0
        assert result.period_count == 2

    def test_missing_date_skipped(self):
        rows = [
            {"amount": 100, "date": "Q1"},
            {"amount": 200, "date": None},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.total_sales == 100.0

    def test_summary_includes_key_info(self):
        rows = [
            {"amount": 500, "date": "Q1"},
            {"amount": 500, "date": "Q2"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert "1,000.00" in result.summary
        assert "2 periods" in result.summary

    def test_string_amounts(self):
        rows = [
            {"amount": "100.50", "date": "Q1"},
            {"amount": "200.25", "date": "Q2"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.total_sales == 300.75

    def test_zero_amounts(self):
        rows = [
            {"amount": 0, "date": "Q1"},
            {"amount": 0, "date": "Q2"},
        ]
        result = analyze_sales_performance(rows, "amount", "date")
        assert result is not None
        assert result.total_sales == 0.0
        assert result.growth_rate == 0.0


# ---------------------------------------------------------------------------
# analyze_product_mix
# ---------------------------------------------------------------------------


class TestAnalyzeProductMix:
    def test_empty_rows(self):
        assert analyze_product_mix([], "product", "amount") is None

    def test_all_invalid_data(self):
        rows = [{"product": "A", "amount": "bad"}]
        assert analyze_product_mix(rows, "product", "amount") is None

    def test_single_product(self):
        rows = [{"product": "Widget", "amount": 1000}]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert len(result.products) == 1
        assert result.products[0].pct_of_total == 100.0
        assert result.hhi == 10000.0  # single product = max concentration

    def test_equal_products(self):
        rows = [
            {"product": "A", "amount": 250},
            {"product": "B", "amount": 250},
            {"product": "C", "amount": 250},
            {"product": "D", "amount": 250},
        ]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert result.total_revenue == 1000.0
        assert len(result.products) == 4
        # Each product is 25%, HHI = 4 * 25^2 = 2500
        assert abs(result.hhi - 2500.0) < 1.0

    def test_concentration_risk_low(self):
        rows = [{"product": f"P{i}", "amount": 100} for i in range(10)]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert result.concentration_risk == "low"

    def test_concentration_risk_high(self):
        rows = [{"product": "Dominant", "amount": 10000}]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert result.concentration_risk == "high"

    def test_concentration_risk_moderate(self):
        # HHI between 1500 and 2500
        # 3 products: 50%, 30%, 20% => HHI = 2500+900+400 = 3800 (high)
        # 4 products: 40%, 25%, 20%, 15% => HHI = 1600+625+400+225 = 2850 (high)
        # Let's aim for moderate: 5 products, mild imbalance
        # 30%, 25%, 20%, 15%, 10% => HHI = 900+625+400+225+100 = 2250
        rows = [
            {"product": "A", "amount": 300},
            {"product": "B", "amount": 250},
            {"product": "C", "amount": 200},
            {"product": "D", "amount": 150},
            {"product": "E", "amount": 100},
        ]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert result.concentration_risk == "moderate"

    def test_top_and_bottom_products(self):
        rows = [{"product": f"P{i}", "amount": (i + 1) * 100} for i in range(8)]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert len(result.top_products) == 5
        assert len(result.bottom_products) == 5
        # Top product should be P7 (800)
        assert result.top_products[0].product == "P7"

    def test_fewer_than_five_products(self):
        rows = [
            {"product": "A", "amount": 100},
            {"product": "B", "amount": 200},
        ]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert len(result.top_products) == 2
        assert len(result.bottom_products) == 2

    def test_with_quantity_column(self):
        rows = [
            {"product": "Widget", "amount": 1000, "qty": 50},
            {"product": "Widget", "amount": 500, "qty": 25},
        ]
        result = analyze_product_mix(rows, "product", "amount", quantity_column="qty")
        assert result is not None
        assert result.products[0].avg_price == 20.0  # 1500 / 75

    def test_without_quantity_column(self):
        rows = [{"product": "Widget", "amount": 1000}]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert result.products[0].avg_price is None

    def test_transaction_count(self):
        rows = [
            {"product": "A", "amount": 100},
            {"product": "A", "amount": 200},
            {"product": "A", "amount": 300},
            {"product": "B", "amount": 400},
        ]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        prod_a = next(p for p in result.products if p.product == "A")
        assert prod_a.transaction_count == 3
        prod_b = next(p for p in result.products if p.product == "B")
        assert prod_b.transaction_count == 1

    def test_pct_of_total_sums_to_100(self):
        rows = [
            {"product": "A", "amount": 300},
            {"product": "B", "amount": 500},
            {"product": "C", "amount": 200},
        ]
        result = analyze_product_mix(rows, "product", "amount")
        total_pct = sum(p.pct_of_total for p in result.products)
        assert abs(total_pct - 100.0) < 0.1

    def test_summary_includes_top_product(self):
        rows = [
            {"product": "Alpha", "amount": 1000},
            {"product": "Beta", "amount": 200},
        ]
        result = analyze_product_mix(rows, "product", "amount")
        assert "Alpha" in result.summary

    def test_missing_product_skipped(self):
        rows = [
            {"product": "A", "amount": 100},
            {"product": None, "amount": 200},
        ]
        result = analyze_product_mix(rows, "product", "amount")
        assert result is not None
        assert len(result.products) == 1


# ---------------------------------------------------------------------------
# compute_sales_velocity
# ---------------------------------------------------------------------------


class TestComputeSalesVelocity:
    def test_empty_rows(self):
        assert compute_sales_velocity([], "deal", "amount", "date") is None

    def test_all_invalid(self):
        rows = [{"deal": "D1", "amount": "bad", "date": "2024-01-01"}]
        assert compute_sales_velocity(rows, "deal", "amount", "date") is None

    def test_single_deal(self):
        rows = [{"deal": "D1", "amount": 5000, "date": "2024-01-01"}]
        result = compute_sales_velocity(rows, "deal", "amount", "date")
        assert result is not None
        assert result.total_deals == 1
        assert result.total_value == 5000.0
        assert result.avg_deal_size == 5000.0
        assert result.win_rate is None  # no stage column

    def test_multiple_deals(self):
        rows = [
            {"deal": "D1", "amount": 1000, "date": "2024-01-01"},
            {"deal": "D2", "amount": 2000, "date": "2024-02-01"},
            {"deal": "D3", "amount": 3000, "date": "2024-03-01"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date")
        assert result is not None
        assert result.total_deals == 3
        assert result.total_value == 6000.0
        assert result.avg_deal_size == 2000.0

    def test_win_rate_with_stages(self):
        rows = [
            {"deal": "D1", "amount": 1000, "date": "2024-01-01", "stage": "Won"},
            {"deal": "D2", "amount": 2000, "date": "2024-02-01", "stage": "Lost"},
            {"deal": "D3", "amount": 3000, "date": "2024-03-01", "stage": "Won"},
            {"deal": "D4", "amount": 500, "date": "2024-04-01", "stage": "Prospect"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date", stage_column="stage")
        assert result is not None
        assert result.win_rate == 50.0  # 2 out of 4

    def test_win_rate_closed_won(self):
        rows = [
            {"deal": "D1", "amount": 100, "date": "2024-01-01", "stage": "Closed Won"},
            {"deal": "D2", "amount": 200, "date": "2024-01-01", "stage": "Open"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date", stage_column="stage")
        assert result is not None
        assert result.win_rate == 50.0

    def test_win_rate_closed(self):
        rows = [
            {"deal": "D1", "amount": 100, "date": "2024-01-01", "stage": "Closed"},
            {"deal": "D2", "amount": 200, "date": "2024-01-01", "stage": "Open"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date", stage_column="stage")
        assert result is not None
        assert result.win_rate == 50.0

    def test_avg_days_to_close(self):
        rows = [
            {"deal": "D1", "amount": 1000, "date": "2024-01-01"},
            {"deal": "D1", "amount": 1000, "date": "2024-01-31"},  # 30 days
            {"deal": "D2", "amount": 2000, "date": "2024-02-01"},
            {"deal": "D2", "amount": 2000, "date": "2024-02-11"},  # 10 days
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date")
        assert result is not None
        assert result.avg_days_to_close == 20.0  # (30 + 10) / 2

    def test_pipeline_velocity(self):
        rows = [
            {"deal": "D1", "amount": 1000, "date": "2024-01-01", "stage": "Won"},
            {"deal": "D1", "amount": 1000, "date": "2024-01-11", "stage": "Won"},  # 10 days
            {"deal": "D2", "amount": 1000, "date": "2024-02-01", "stage": "Won"},
            {"deal": "D2", "amount": 1000, "date": "2024-02-11", "stage": "Won"},  # 10 days
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date", stage_column="stage")
        assert result is not None
        # deals=2, avg_deal=1000, win_rate=100%, avg_cycle=10 days
        # velocity = (2 * 1000 * 1.0) / 10 = 200
        assert result.pipeline_velocity == 200.0

    def test_no_pipeline_velocity_without_stages(self):
        rows = [
            {"deal": "D1", "amount": 1000, "date": "2024-01-01"},
            {"deal": "D1", "amount": 1000, "date": "2024-01-31"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date")
        assert result is not None
        assert result.pipeline_velocity is None

    def test_no_velocity_without_dates(self):
        rows = [
            {"deal": "D1", "amount": 1000, "date": "bad-date", "stage": "Won"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date", stage_column="stage")
        assert result is not None
        assert result.avg_days_to_close is None
        assert result.pipeline_velocity is None

    def test_funnel_stages(self):
        rows = [
            {"deal": "D1", "amount": 100, "date": "2024-01-01", "stage": "Lead"},
            {"deal": "D1", "amount": 100, "date": "2024-01-01", "stage": "Prospect"},
            {"deal": "D2", "amount": 200, "date": "2024-01-01", "stage": "Lead"},
            {"deal": "D3", "amount": 300, "date": "2024-01-01", "stage": "Lead"},
            {"deal": "D3", "amount": 300, "date": "2024-01-01", "stage": "Won"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date", stage_column="stage")
        assert result is not None
        assert len(result.funnel) > 0
        # Lead should have highest count (3 deals have lead stage)
        assert result.funnel[0][0] == "lead"
        assert result.funnel[0][1] == 3

    def test_deal_takes_max_amount(self):
        rows = [
            {"deal": "D1", "amount": 500, "date": "2024-01-01"},
            {"deal": "D1", "amount": 1000, "date": "2024-02-01"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date")
        assert result is not None
        assert result.total_value == 1000.0

    def test_missing_deal_skipped(self):
        rows = [
            {"deal": None, "amount": 100, "date": "2024-01-01"},
            {"deal": "D1", "amount": 200, "date": "2024-01-01"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date")
        assert result is not None
        assert result.total_deals == 1

    def test_summary_includes_deal_count(self):
        rows = [
            {"deal": "D1", "amount": 100, "date": "2024-01-01"},
            {"deal": "D2", "amount": 200, "date": "2024-02-01"},
        ]
        result = compute_sales_velocity(rows, "deal", "amount", "date")
        assert "2 deals" in result.summary


# ---------------------------------------------------------------------------
# analyze_discount_impact
# ---------------------------------------------------------------------------


class TestAnalyzeDiscountImpact:
    def test_empty_rows(self):
        assert analyze_discount_impact([], "amount", "discount") is None

    def test_all_invalid(self):
        rows = [{"amount": "bad", "discount": 0.1}]
        assert analyze_discount_impact(rows, "amount", "discount") is None

    def test_no_discount(self):
        rows = [
            {"amount": 1000, "discount": 0},
            {"amount": 2000, "discount": 0},
        ]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.avg_discount == 0.0
        assert result.max_discount == 0.0
        assert result.revenue_impact == 0.0

    def test_basic_discount(self):
        rows = [
            {"amount": 900, "discount": 0.10},  # 10% discount
        ]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.avg_discount == 0.10
        assert result.max_discount == 0.10
        # Revenue impact: 900 * 0.10 / 0.90 = 100
        assert abs(result.revenue_impact - 100.0) < 0.01

    def test_percentage_normalization(self):
        """Values > 1 should be treated as percentages."""
        rows = [{"amount": 900, "discount": 10}]  # 10 means 10%
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.avg_discount == 0.10

    def test_high_discount_clamped(self):
        """Discount >= 100% should be clamped to 99.99%."""
        rows = [{"amount": 100, "discount": 100}]  # 100%
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.max_discount <= 0.9999

    def test_negative_discount_clamped(self):
        """Negative discount should be clamped to 0."""
        rows = [{"amount": 100, "discount": -5}]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.avg_discount == 0.0

    def test_distribution_buckets(self):
        rows = [
            {"amount": 100, "discount": 0},     # 0%
            {"amount": 100, "discount": 0.03},   # 1-5%
            {"amount": 100, "discount": 0.07},   # 5-10%
            {"amount": 100, "discount": 0.15},   # 10-20%
            {"amount": 100, "discount": 0.25},   # 20%+
        ]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert len(result.distribution) == 5
        # Each bucket should have exactly 1
        for bucket in result.distribution:
            assert bucket.count == 1
            assert bucket.pct == 20.0

    def test_distribution_zero_bucket(self):
        rows = [
            {"amount": 100, "discount": 0},
            {"amount": 100, "discount": 0},
        ]
        result = analyze_discount_impact(rows, "amount", "discount")
        zero_bucket = next(b for b in result.distribution if b.range_label == "0%")
        assert zero_bucket.count == 2

    def test_with_product_column(self):
        rows = [
            {"amount": 100, "discount": 0.05, "product": "Widget"},
            {"amount": 200, "discount": 0.10, "product": "Widget"},
            {"amount": 150, "discount": 0.15, "product": "Gadget"},
        ]
        result = analyze_discount_impact(
            rows, "amount", "discount", product_column="product"
        )
        assert result is not None
        assert len(result.by_product) == 2
        widget = next(p for p in result.by_product if p.product == "Widget")
        assert widget.deal_count == 2
        assert abs(widget.avg_discount - 0.075) < 0.001

    def test_with_quantity_column_correlation(self):
        rows = [
            {"amount": 100, "discount": 0.05, "qty": 10},
            {"amount": 200, "discount": 0.10, "qty": 20},
            {"amount": 300, "discount": 0.15, "qty": 30},
            {"amount": 400, "discount": 0.20, "qty": 40},
        ]
        result = analyze_discount_impact(
            rows, "amount", "discount", quantity_column="qty"
        )
        assert result is not None
        assert result.discount_volume_correlation is not None
        # Perfect positive correlation
        assert result.discount_volume_correlation == 1.0

    def test_no_correlation_without_quantity(self):
        rows = [{"amount": 100, "discount": 0.05}]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.discount_volume_correlation is None

    def test_multiple_discounts_avg(self):
        rows = [
            {"amount": 100, "discount": 0.10},
            {"amount": 100, "discount": 0.20},
        ]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.avg_discount == 0.15

    def test_revenue_impact_multiple(self):
        rows = [
            {"amount": 900, "discount": 0.10},   # lost = 100
            {"amount": 800, "discount": 0.20},   # lost = 200
        ]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert abs(result.revenue_impact - 300.0) < 0.01

    def test_summary_content(self):
        rows = [{"amount": 100, "discount": 0.10}]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert "10.0%" in result.summary
        assert "1 transaction" in result.summary

    def test_missing_discount_skipped(self):
        rows = [
            {"amount": 100, "discount": 0.10},
            {"amount": 200, "discount": None},
        ]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.avg_discount == 0.10

    def test_all_zero_discount(self):
        rows = [
            {"amount": 100, "discount": 0},
            {"amount": 200, "discount": 0},
            {"amount": 300, "discount": 0},
        ]
        result = analyze_discount_impact(rows, "amount", "discount")
        assert result is not None
        assert result.revenue_impact == 0.0
        zero_bucket = next(b for b in result.distribution if b.range_label == "0%")
        assert zero_bucket.count == 3

    def test_single_quantity_no_correlation(self):
        """Need >= 2 pairs for correlation."""
        rows = [{"amount": 100, "discount": 0.05, "qty": 10}]
        result = analyze_discount_impact(
            rows, "amount", "discount", quantity_column="qty"
        )
        assert result is not None
        assert result.discount_volume_correlation is None


# ---------------------------------------------------------------------------
# format_sales_report
# ---------------------------------------------------------------------------


class TestFormatSalesReport:
    def test_empty_report(self):
        report = format_sales_report()
        assert "SALES ANALYTICS REPORT" in report
        assert "No data available" in report

    def test_performance_only(self):
        rows = [
            {"amount": 100, "date": "Q1"},
            {"amount": 200, "date": "Q2"},
        ]
        perf = analyze_sales_performance(rows, "amount", "date")
        report = format_sales_report(performance=perf)
        assert "SALES PERFORMANCE" in report
        assert "300.00" in report

    def test_product_mix_only(self):
        rows = [
            {"product": "A", "amount": 500},
            {"product": "B", "amount": 300},
        ]
        mix = analyze_product_mix(rows, "product", "amount")
        report = format_sales_report(product_mix=mix)
        assert "PRODUCT MIX" in report
        assert "800.00" in report

    def test_velocity_only(self):
        rows = [
            {"deal": "D1", "amount": 1000, "date": "2024-01-01"},
        ]
        vel = compute_sales_velocity(rows, "deal", "amount", "date")
        report = format_sales_report(velocity=vel)
        assert "SALES VELOCITY" in report
        assert "1,000.00" in report

    def test_discount_only(self):
        rows = [
            {"amount": 900, "discount": 0.10},
        ]
        disc = analyze_discount_impact(rows, "amount", "discount")
        report = format_sales_report(discounts=disc)
        assert "DISCOUNT IMPACT" in report
        assert "10.0%" in report

    def test_combined_report(self):
        perf_rows = [{"amount": 100, "date": "Q1"}, {"amount": 200, "date": "Q2"}]
        mix_rows = [{"product": "X", "amount": 300}]
        vel_rows = [{"deal": "D1", "amount": 500, "date": "2024-01-01"}]
        disc_rows = [{"amount": 100, "discount": 0.05}]

        perf = analyze_sales_performance(perf_rows, "amount", "date")
        mix = analyze_product_mix(mix_rows, "product", "amount")
        vel = compute_sales_velocity(vel_rows, "deal", "amount", "date")
        disc = analyze_discount_impact(disc_rows, "amount", "discount")

        report = format_sales_report(
            performance=perf, product_mix=mix, velocity=vel, discounts=disc
        )
        assert "SALES PERFORMANCE" in report
        assert "PRODUCT MIX" in report
        assert "SALES VELOCITY" in report
        assert "DISCOUNT IMPACT" in report

    def test_report_with_reps(self):
        rows = [
            {"amount": 100, "date": "Q1", "rep": "Alice"},
            {"amount": 200, "date": "Q2", "rep": "Bob"},
        ]
        perf = analyze_sales_performance(rows, "amount", "date", rep_column="rep")
        report = format_sales_report(performance=perf)
        assert "Top Reps" in report
        assert "Bob" in report

    def test_report_with_regions(self):
        rows = [
            {"amount": 100, "date": "Q1", "region": "North"},
            {"amount": 200, "date": "Q2", "region": "South"},
        ]
        perf = analyze_sales_performance(rows, "amount", "date", region_column="region")
        report = format_sales_report(performance=perf)
        assert "Regions" in report
        assert "North" in report

    def test_report_with_funnel(self):
        rows = [
            {"deal": "D1", "amount": 100, "date": "2024-01-01", "stage": "Lead"},
            {"deal": "D2", "amount": 200, "date": "2024-01-01", "stage": "Won"},
        ]
        vel = compute_sales_velocity(rows, "deal", "amount", "date", stage_column="stage")
        report = format_sales_report(velocity=vel)
        assert "Stage Funnel" in report

    def test_report_with_product_discounts(self):
        rows = [
            {"amount": 100, "discount": 0.05, "product": "Gizmo"},
        ]
        disc = analyze_discount_impact(
            rows, "amount", "discount", product_column="product"
        )
        report = format_sales_report(discounts=disc)
        assert "By Product" in report
        assert "Gizmo" in report
