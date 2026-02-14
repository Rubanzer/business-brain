"""Tests for pricing_optimizer module."""

from business_brain.discovery.pricing_optimizer import (
    ProductPrice,
    CategoryPrice,
    PriceDistributionResult,
    ProductElasticity,
    PriceElasticityResult,
    ProductGap,
    CompetitorGap,
    CompetitivePricingResult,
    ProductMargin,
    MarginBucket,
    PriceMarginResult,
    _safe_float,
    analyze_price_distribution,
    compute_price_elasticity,
    analyze_competitive_pricing,
    compute_price_margin_analysis,
    format_pricing_report,
)


# ---------------------------------------------------------------------------
# _safe_float helper
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(10) == 10.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("42.5") == 42.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_bool(self):
        # bool is a subclass of int in Python
        assert _safe_float(True) == 1.0

    def test_negative(self):
        assert _safe_float(-5.5) == -5.5


# ---------------------------------------------------------------------------
# analyze_price_distribution
# ---------------------------------------------------------------------------


class TestAnalyzePriceDistribution:
    def test_empty_rows(self):
        assert analyze_price_distribution([], "price") is None

    def test_all_none_prices(self):
        rows = [{"price": None}, {"price": None}]
        assert analyze_price_distribution(rows, "price") is None

    def test_all_invalid_prices(self):
        rows = [{"price": "abc"}, {"price": "def"}]
        assert analyze_price_distribution(rows, "price") is None

    def test_single_row(self):
        rows = [{"price": 100}]
        result = analyze_price_distribution(rows, "price")
        assert result is not None
        assert result.mean_price == 100.0
        assert result.median_price == 100.0
        assert result.std_price == 0.0
        assert result.min_price == 100.0
        assert result.max_price == 100.0

    def test_basic_stats(self):
        rows = [{"price": p} for p in [10, 20, 30, 40, 50]]
        result = analyze_price_distribution(rows, "price")
        assert result is not None
        assert result.mean_price == 30.0
        assert result.median_price == 30.0
        assert result.min_price == 10.0
        assert result.max_price == 50.0

    def test_std_deviation(self):
        rows = [{"price": p} for p in [10, 10, 10, 10]]
        result = analyze_price_distribution(rows, "price")
        assert result.std_price == 0.0

    def test_outlier_detection(self):
        # Create data with a clear outlier
        prices = [10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 100]
        rows = [{"price": p} for p in prices]
        result = analyze_price_distribution(rows, "price")
        assert result is not None
        assert result.outlier_count >= 1

    def test_no_outliers(self):
        # Very tight distribution
        rows = [{"price": p} for p in [10, 10, 10, 10, 10]]
        result = analyze_price_distribution(rows, "price")
        assert result.outlier_count == 0

    def test_outlier_pct(self):
        # 10 normal, 2 outliers
        prices = [10] * 10 + [1000, 1001]
        rows = [{"price": p} for p in prices]
        result = analyze_price_distribution(rows, "price")
        assert result.outlier_pct > 0

    def test_by_product(self):
        rows = [
            {"price": 10, "product": "A"},
            {"price": 20, "product": "A"},
            {"price": 30, "product": "B"},
        ]
        result = analyze_price_distribution(rows, "price", product_column="product")
        assert len(result.by_product) == 2
        a = [p for p in result.by_product if p.product == "A"][0]
        assert a.avg_price == 15.0
        assert a.min_price == 10.0
        assert a.max_price == 20.0
        assert a.count == 2

    def test_by_category(self):
        rows = [
            {"price": 10, "cat": "Electronics"},
            {"price": 20, "cat": "Electronics"},
            {"price": 50, "cat": "Clothing"},
        ]
        result = analyze_price_distribution(rows, "price", category_column="cat")
        assert len(result.by_category) == 2
        elec = [c for c in result.by_category if c.category == "Electronics"][0]
        assert elec.avg_price == 15.0
        assert elec.count == 2

    def test_by_product_and_category(self):
        rows = [
            {"price": 10, "product": "A", "cat": "X"},
            {"price": 20, "product": "B", "cat": "Y"},
        ]
        result = analyze_price_distribution(
            rows, "price", product_column="product", category_column="cat"
        )
        assert len(result.by_product) == 2
        assert len(result.by_category) == 2

    def test_product_none_skipped(self):
        rows = [
            {"price": 10, "product": None},
            {"price": 20, "product": "B"},
        ]
        result = analyze_price_distribution(rows, "price", product_column="product")
        assert len(result.by_product) == 1

    def test_summary_present(self):
        rows = [{"price": 100}]
        result = analyze_price_distribution(rows, "price")
        assert "Price distribution" in result.summary

    def test_mixed_valid_invalid(self):
        rows = [
            {"price": 10},
            {"price": "abc"},
            {"price": 30},
        ]
        result = analyze_price_distribution(rows, "price")
        assert result is not None
        assert result.mean_price == 20.0

    def test_missing_column(self):
        rows = [{"other": 10}]
        assert analyze_price_distribution(rows, "price") is None

    def test_string_prices(self):
        rows = [{"price": "10"}, {"price": "20"}]
        result = analyze_price_distribution(rows, "price")
        assert result is not None
        assert result.mean_price == 15.0


# ---------------------------------------------------------------------------
# compute_price_elasticity
# ---------------------------------------------------------------------------


class TestComputePriceElasticity:
    def test_empty_rows(self):
        assert compute_price_elasticity([], "price", "qty") is None

    def test_all_none_data(self):
        rows = [{"price": None, "qty": None}]
        assert compute_price_elasticity(rows, "price", "qty") is None

    def test_single_row(self):
        rows = [{"price": 10, "qty": 100}]
        # Need at least 2 rows for elasticity
        assert compute_price_elasticity(rows, "price", "qty") is None

    def test_inelastic_demand(self):
        # Price goes up 100%, quantity drops only a little => inelastic
        rows = [
            {"price": 10, "qty": 100},
            {"price": 20, "qty": 95},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert result is not None
        assert result.elasticity_type == "Inelastic"
        assert abs(result.overall_elasticity) < 1.0

    def test_elastic_demand(self):
        # Price goes up a little, quantity drops a lot => elastic
        rows = [
            {"price": 10, "qty": 100},
            {"price": 11, "qty": 50},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert result is not None
        assert result.elasticity_type == "Elastic"
        assert abs(result.overall_elasticity) > 1.0

    def test_unit_elastic(self):
        # Elasticity close to -1.0
        # midpoint: P_avg = 10, Q_avg = 100
        # We need (DQ/Q_avg) / (DP/P_avg) ~ -1
        # Let DP=2, so DP/P_avg = 2/10 = 0.2
        # DQ/Q_avg = -0.2 => DQ = -20
        rows = [
            {"price": 9, "qty": 110},
            {"price": 11, "qty": 90},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert result is not None
        assert result.elasticity_type == "Unit Elastic"

    def test_by_product(self):
        rows = [
            {"price": 10, "qty": 100, "product": "A"},
            {"price": 20, "qty": 50, "product": "A"},
            {"price": 5, "qty": 200, "product": "B"},
            {"price": 10, "qty": 190, "product": "B"},
        ]
        result = compute_price_elasticity(rows, "price", "qty", product_column="product")
        assert result is not None
        assert len(result.by_product) == 2

    def test_product_with_single_row_skipped(self):
        rows = [
            {"price": 10, "qty": 100, "product": "A"},
            {"price": 20, "qty": 50, "product": "A"},
            {"price": 5, "qty": 200, "product": "B"},  # only 1 row for B
        ]
        result = compute_price_elasticity(rows, "price", "qty", product_column="product")
        assert result is not None
        assert len(result.by_product) == 1
        assert result.by_product[0].product == "A"

    def test_zero_price_change_skipped(self):
        rows = [
            {"price": 10, "qty": 100},
            {"price": 10, "qty": 90},
            {"price": 20, "qty": 50},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert result is not None
        # First pair has delta_p = 0, should be skipped

    def test_summary_present(self):
        rows = [
            {"price": 10, "qty": 100},
            {"price": 20, "qty": 50},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert "elasticity" in result.summary.lower()

    def test_elasticity_boundary_high(self):
        # |e| = 1.1 => Unit Elastic
        # midpoint: P_avg=10, Q_avg=100
        # DQ/Q_avg / (DP/P_avg) = -1.1
        # DP=2, DP/P_avg=0.2, DQ/Q_avg=-0.22, DQ=-22
        rows = [
            {"price": 9, "qty": 111},
            {"price": 11, "qty": 89},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert result is not None
        assert result.elasticity_type == "Unit Elastic"

    def test_elasticity_boundary_elastic(self):
        # |e| = 1.11 => Elastic (just beyond 1.0 + 0.1)
        # midpoint: P_avg=100, Q_avg=100
        # DP=20 => DP/P_avg = 0.2
        # DQ/Q_avg = -1.11 * 0.2 = -0.222 => DQ = -22.2
        rows = [
            {"price": 90, "qty": 111.1},
            {"price": 110, "qty": 88.9},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert result is not None
        assert result.elasticity_type == "Unit Elastic" or result.elasticity_type == "Elastic"

    def test_positive_elasticity(self):
        # Giffen/Veblen goods: price up, quantity up
        rows = [
            {"price": 10, "qty": 100},
            {"price": 20, "qty": 200},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert result is not None
        assert result.overall_elasticity > 0

    def test_multiple_pairs(self):
        rows = [
            {"price": 10, "qty": 100},
            {"price": 15, "qty": 80},
            {"price": 20, "qty": 60},
        ]
        result = compute_price_elasticity(rows, "price", "qty")
        assert result is not None

    def test_invalid_qty_skipped(self):
        rows = [
            {"price": 10, "qty": "abc"},
            {"price": 20, "qty": 50},
        ]
        # Only one valid row => not enough
        assert compute_price_elasticity(rows, "price", "qty") is None


# ---------------------------------------------------------------------------
# analyze_competitive_pricing
# ---------------------------------------------------------------------------


class TestAnalyzeCompetitivePricing:
    def test_empty_rows(self):
        assert analyze_competitive_pricing([], "product", "our", "comp") is None

    def test_all_none_data(self):
        rows = [{"product": None, "our": None, "comp": None}]
        assert analyze_competitive_pricing(rows, "product", "our", "comp") is None

    def test_premium_position(self):
        rows = [{"product": "A", "our": 120, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result is not None
        assert result.premium_count == 1
        assert result.by_product[0].position == "Premium"
        assert result.by_product[0].price_gap_pct == 20.0

    def test_competitive_position(self):
        rows = [{"product": "A", "our": 105, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.competitive_count == 1
        assert result.by_product[0].position == "Competitive"

    def test_discount_position(self):
        rows = [{"product": "A", "our": 80, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.discount_count == 1
        assert result.by_product[0].position == "Discount"
        assert result.by_product[0].price_gap_pct == -20.0

    def test_boundary_premium(self):
        # Exactly 10% is Competitive (<=10%)
        rows = [{"product": "A", "our": 110, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.by_product[0].position == "Competitive"

    def test_boundary_just_above_premium(self):
        # 10.01% is Premium
        rows = [{"product": "A", "our": 110.01, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.by_product[0].position == "Premium"

    def test_boundary_discount(self):
        # Exactly -10% is Competitive
        rows = [{"product": "A", "our": 90, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.by_product[0].position == "Competitive"

    def test_boundary_just_below_discount(self):
        # -10.01% is Discount
        rows = [{"product": "A", "our": 89.99, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.by_product[0].position == "Discount"

    def test_multiple_products_mixed(self):
        rows = [
            {"product": "A", "our": 130, "comp": 100},  # Premium
            {"product": "B", "our": 100, "comp": 100},  # Competitive
            {"product": "C", "our": 70, "comp": 100},   # Discount
        ]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.premium_count == 1
        assert result.competitive_count == 1
        assert result.discount_count == 1

    def test_avg_price_gap(self):
        rows = [
            {"product": "A", "our": 120, "comp": 100},  # +20%
            {"product": "B", "our": 80, "comp": 100},   # -20%
        ]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.avg_price_gap == 0.0

    def test_by_competitor(self):
        rows = [
            {"product": "A", "our": 120, "comp": 100, "competitor": "X"},
            {"product": "A", "our": 120, "comp": 110, "competitor": "Y"},
        ]
        result = analyze_competitive_pricing(
            rows, "product", "our", "comp", competitor_column="competitor"
        )
        assert len(result.by_competitor) == 2
        x = [c for c in result.by_competitor if c.competitor == "X"][0]
        assert x.product_count == 1

    def test_no_competitor_column(self):
        rows = [{"product": "A", "our": 120, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result.by_competitor == []

    def test_zero_competitor_price(self):
        rows = [{"product": "A", "our": 100, "comp": 0}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result is not None
        assert result.by_product[0].price_gap_pct == 0.0

    def test_summary_present(self):
        rows = [{"product": "A", "our": 100, "comp": 100}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert "Competitive pricing" in result.summary

    def test_aggregate_multiple_competitor_rows(self):
        rows = [
            {"product": "A", "our": 100, "comp": 80, "competitor": "X"},
            {"product": "A", "our": 100, "comp": 120, "competitor": "Y"},
        ]
        result = analyze_competitive_pricing(
            rows, "product", "our", "comp", competitor_column="competitor"
        )
        # Product A: our_avg=100, comp_avg=(80+120)/2=100, gap=0%
        assert result.by_product[0].price_gap_pct == 0.0
        assert result.by_product[0].position == "Competitive"

    def test_missing_product(self):
        rows = [{"product": None, "our": 100, "comp": 80}]
        assert analyze_competitive_pricing(rows, "product", "our", "comp") is None

    def test_string_prices(self):
        rows = [{"product": "A", "our": "120", "comp": "100"}]
        result = analyze_competitive_pricing(rows, "product", "our", "comp")
        assert result is not None
        assert result.by_product[0].price_gap_pct == 20.0


# ---------------------------------------------------------------------------
# compute_price_margin_analysis
# ---------------------------------------------------------------------------


class TestComputePriceMarginAnalysis:
    def test_empty_rows(self):
        assert compute_price_margin_analysis([], "price", "cost") is None

    def test_all_none_data(self):
        rows = [{"price": None, "cost": None}]
        assert compute_price_margin_analysis(rows, "price", "cost") is None

    def test_single_item(self):
        rows = [{"price": 100, "cost": 60}]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert result is not None
        assert result.avg_margin == 40.0
        assert result.min_margin == 40.0
        assert result.max_margin == 40.0

    def test_negative_margin(self):
        rows = [{"price": 80, "cost": 100}]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert result.avg_margin == -25.0
        assert result.negative_margin_count == 1

    def test_zero_price(self):
        rows = [{"price": 0, "cost": 50}]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert result is not None
        assert result.avg_margin == 0.0

    def test_zero_cost(self):
        rows = [{"price": 100, "cost": 0}]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert result.avg_margin == 100.0

    def test_by_product(self):
        rows = [
            {"price": 100, "cost": 60, "product": "A"},
            {"price": 200, "cost": 100, "product": "B"},
        ]
        result = compute_price_margin_analysis(
            rows, "price", "cost", product_column="product"
        )
        assert len(result.by_product) == 2
        a = [p for p in result.by_product if p.product == "A"][0]
        assert a.margin_pct == 40.0

    def test_weighted_margin(self):
        rows = [
            {"price": 100, "cost": 50, "qty": 10},   # margin=50%, revenue=1000
            {"price": 200, "cost": 180, "qty": 5},    # margin=10%, revenue=1000
        ]
        result = compute_price_margin_analysis(
            rows, "price", "cost", quantity_column="qty"
        )
        assert result.weighted_margin is not None
        # weighted = (50*1000 + 10*1000) / (1000+1000) = 60000/2000 = 30
        assert result.weighted_margin == 30.0

    def test_no_quantity_weighted_margin_none(self):
        rows = [{"price": 100, "cost": 50}]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert result.weighted_margin is None

    def test_margin_distribution_buckets(self):
        rows = [
            {"price": 100, "cost": 110},  # margin = -10% => <0%
            {"price": 100, "cost": 95},   # margin = 5% => 0-10%
            {"price": 100, "cost": 85},   # margin = 15% => 10-20%
            {"price": 100, "cost": 75},   # margin = 25% => 20-30%
            {"price": 100, "cost": 60},   # margin = 40% => 30-50%
            {"price": 100, "cost": 30},   # margin = 70% => >50%
        ]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert len(result.margin_distribution) == 6

        dist = {b.range_label: b.count for b in result.margin_distribution}
        assert dist["<0%"] == 1
        assert dist["0-10%"] == 1
        assert dist["10-20%"] == 1
        assert dist["20-30%"] == 1
        assert dist["30-50%"] == 1
        assert dist[">50%"] == 1

    def test_margin_distribution_pct(self):
        rows = [{"price": 100, "cost": 50}] * 4  # all 50% margin => >50%
        result = compute_price_margin_analysis(rows, "price", "cost")
        gt50 = [b for b in result.margin_distribution if b.range_label == ">50%"][0]
        assert gt50.pct == 100.0

    def test_by_product_volume(self):
        rows = [
            {"price": 100, "cost": 60, "product": "A", "qty": 5},
            {"price": 100, "cost": 60, "product": "A", "qty": 3},
        ]
        result = compute_price_margin_analysis(
            rows, "price", "cost",
            product_column="product", quantity_column="qty",
        )
        a = result.by_product[0]
        assert a.volume == 8

    def test_summary_present(self):
        rows = [{"price": 100, "cost": 50}]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert "Margin analysis" in result.summary

    def test_summary_weighted_present(self):
        rows = [{"price": 100, "cost": 50, "qty": 10}]
        result = compute_price_margin_analysis(
            rows, "price", "cost", quantity_column="qty"
        )
        assert "weighted" in result.summary.lower()

    def test_invalid_cost_skipped(self):
        rows = [
            {"price": 100, "cost": "bad"},
            {"price": 100, "cost": 50},
        ]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert result is not None
        assert result.avg_margin == 50.0

    def test_multiple_items_avg(self):
        rows = [
            {"price": 100, "cost": 50},  # margin = 50%
            {"price": 100, "cost": 80},  # margin = 20%
        ]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert result.avg_margin == 35.0

    def test_mixed_negative_positive(self):
        rows = [
            {"price": 100, "cost": 50},   # +50%
            {"price": 100, "cost": 120},   # -20%
        ]
        result = compute_price_margin_analysis(rows, "price", "cost")
        assert result.negative_margin_count == 1
        assert result.min_margin == -20.0
        assert result.max_margin == 50.0

    def test_product_none_still_counts_overall(self):
        rows = [
            {"price": 100, "cost": 60, "product": None},
            {"price": 200, "cost": 100, "product": "B"},
        ]
        result = compute_price_margin_analysis(
            rows, "price", "cost", product_column="product"
        )
        assert result is not None
        # Overall should include both rows
        assert len(result.margin_distribution) == 6
        # by_product should only have B
        assert len(result.by_product) == 1


# ---------------------------------------------------------------------------
# format_pricing_report
# ---------------------------------------------------------------------------


class TestFormatPricingReport:
    def test_no_data(self):
        report = format_pricing_report()
        assert "No analysis data provided" in report

    def test_title_always_present(self):
        report = format_pricing_report()
        assert "Pricing Analysis Report" in report

    def test_with_distribution_only(self):
        rows = [{"price": p} for p in [10, 20, 30]]
        dist = analyze_price_distribution(rows, "price")
        report = format_pricing_report(distribution=dist)
        assert "Price Distribution" in report
        assert "Mean" in report
        assert "No analysis data provided" not in report

    def test_with_elasticity_only(self):
        rows = [
            {"price": 10, "qty": 100},
            {"price": 20, "qty": 50},
        ]
        elast = compute_price_elasticity(rows, "price", "qty")
        report = format_pricing_report(elasticity=elast)
        assert "Price Elasticity" in report
        assert "No analysis data provided" not in report

    def test_with_competitive_only(self):
        rows = [{"product": "A", "our": 120, "comp": 100}]
        comp = analyze_competitive_pricing(rows, "product", "our", "comp")
        report = format_pricing_report(competitive=comp)
        assert "Competitive Pricing" in report
        assert "No analysis data provided" not in report

    def test_with_margins_only(self):
        rows = [{"price": 100, "cost": 60}]
        marg = compute_price_margin_analysis(rows, "price", "cost")
        report = format_pricing_report(margins=marg)
        assert "Price Margins" in report
        assert "No analysis data provided" not in report

    def test_all_sections(self):
        dist_rows = [{"price": p} for p in [10, 20, 30]]
        elast_rows = [
            {"price": 10, "qty": 100},
            {"price": 20, "qty": 50},
        ]
        comp_rows = [{"product": "A", "our": 120, "comp": 100}]
        margin_rows = [{"price": 100, "cost": 60}]

        dist = analyze_price_distribution(dist_rows, "price")
        elast = compute_price_elasticity(elast_rows, "price", "qty")
        comp = analyze_competitive_pricing(comp_rows, "product", "our", "comp")
        marg = compute_price_margin_analysis(margin_rows, "price", "cost")

        report = format_pricing_report(
            distribution=dist,
            elasticity=elast,
            competitive=comp,
            margins=marg,
        )
        assert "Price Distribution" in report
        assert "Price Elasticity" in report
        assert "Competitive Pricing" in report
        assert "Price Margins" in report
        assert "No analysis data provided" not in report

    def test_distribution_with_products(self):
        rows = [
            {"price": 10, "product": "A"},
            {"price": 20, "product": "B"},
        ]
        dist = analyze_price_distribution(rows, "price", product_column="product")
        report = format_pricing_report(distribution=dist)
        assert "By Product" in report

    def test_distribution_with_categories(self):
        rows = [
            {"price": 10, "cat": "X"},
            {"price": 20, "cat": "Y"},
        ]
        dist = analyze_price_distribution(rows, "price", category_column="cat")
        report = format_pricing_report(distribution=dist)
        assert "By Category" in report

    def test_competitive_with_competitor(self):
        rows = [
            {"product": "A", "our": 120, "comp": 100, "competitor": "X"},
        ]
        comp = analyze_competitive_pricing(
            rows, "product", "our", "comp", competitor_column="competitor"
        )
        report = format_pricing_report(competitive=comp)
        assert "By Competitor" in report

    def test_margins_with_products(self):
        rows = [{"price": 100, "cost": 60, "product": "A"}]
        marg = compute_price_margin_analysis(
            rows, "price", "cost", product_column="product"
        )
        report = format_pricing_report(margins=marg)
        assert "By Product" in report

    def test_margins_with_weighted(self):
        rows = [{"price": 100, "cost": 50, "qty": 10}]
        marg = compute_price_margin_analysis(
            rows, "price", "cost", quantity_column="qty"
        )
        report = format_pricing_report(margins=marg)
        assert "Volume-Weighted" in report

    def test_margins_distribution_in_report(self):
        rows = [{"price": 100, "cost": 50}]
        marg = compute_price_margin_analysis(rows, "price", "cost")
        report = format_pricing_report(margins=marg)
        assert "Margin Distribution" in report

    def test_elasticity_with_products(self):
        rows = [
            {"price": 10, "qty": 100, "product": "A"},
            {"price": 20, "qty": 50, "product": "A"},
            {"price": 5, "qty": 200, "product": "B"},
            {"price": 10, "qty": 100, "product": "B"},
        ]
        elast = compute_price_elasticity(rows, "price", "qty", product_column="product")
        report = format_pricing_report(elasticity=elast)
        assert "By Product" in report


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_product_price(self):
        pp = ProductPrice(product="A", min_price=5.0, max_price=15.0, avg_price=10.0, count=3)
        assert pp.product == "A"
        assert pp.count == 3

    def test_category_price(self):
        cp = CategoryPrice(category="Electronics", avg_price=50.0, count=10)
        assert cp.category == "Electronics"

    def test_product_elasticity(self):
        pe = ProductElasticity(product="Widget", elasticity=-1.5, elasticity_type="Elastic")
        assert pe.elasticity == -1.5

    def test_product_gap(self):
        pg = ProductGap(product="A", our_price=120, competitor_avg_price=100, price_gap_pct=20.0, position="Premium")
        assert pg.position == "Premium"

    def test_competitor_gap(self):
        cg = CompetitorGap(competitor="X", avg_gap_pct=15.0, product_count=5)
        assert cg.competitor == "X"

    def test_product_margin(self):
        pm = ProductMargin(product="A", avg_price=100, avg_cost=60, margin_pct=40.0, volume=50)
        assert pm.margin_pct == 40.0

    def test_margin_bucket(self):
        mb = MarginBucket(range_label="20-30%", count=5, pct=25.0)
        assert mb.range_label == "20-30%"

    def test_price_distribution_result(self):
        r = PriceDistributionResult(
            mean_price=50, median_price=45, std_price=10,
            min_price=20, max_price=80, outlier_count=2, outlier_pct=5.0,
            by_product=[], by_category=[], summary="test",
        )
        assert r.mean_price == 50

    def test_price_elasticity_result(self):
        r = PriceElasticityResult(
            overall_elasticity=-0.5, elasticity_type="Inelastic",
            by_product=[], summary="test",
        )
        assert r.elasticity_type == "Inelastic"

    def test_competitive_pricing_result(self):
        r = CompetitivePricingResult(
            avg_price_gap=5.0, premium_count=1, competitive_count=2,
            discount_count=0, by_product=[], by_competitor=[], summary="test",
        )
        assert r.competitive_count == 2

    def test_price_margin_result(self):
        r = PriceMarginResult(
            avg_margin=30.0, weighted_margin=28.0, min_margin=10.0,
            max_margin=50.0, negative_margin_count=0, by_product=[],
            margin_distribution=[], summary="test",
        )
        assert r.weighted_margin == 28.0
