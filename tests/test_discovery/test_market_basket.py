"""Tests for market basket analysis module."""

import math

from business_brain.discovery.market_basket import (
    AssociationResult,
    BasketSizeResult,
    CrossSellProduct,
    CrossSellResult,
    ProductFreq,
    ProductFrequencyResult,
    ProductPair,
    SizeBucket,
    _build_baskets,
    _median,
    _safe_float,
    analyze_basket_size,
    analyze_product_frequency,
    find_cross_sell_opportunities,
    find_product_associations,
    format_basket_report,
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


class TestMedian:
    def test_empty(self):
        assert _median([]) == 0.0

    def test_single(self):
        assert _median([5]) == 5.0

    def test_odd_count(self):
        assert _median([1, 3, 5]) == 3.0

    def test_even_count(self):
        assert _median([1, 2, 3, 4]) == 2.5

    def test_unsorted_input(self):
        assert _median([5, 1, 3]) == 3.0

    def test_all_same(self):
        assert _median([7, 7, 7]) == 7.0


class TestBuildBaskets:
    def test_empty_rows(self):
        assert _build_baskets([], "txn", "product") is None

    def test_missing_columns(self):
        rows = [{"other": "value"}]
        assert _build_baskets(rows, "txn", "product") is None

    def test_basic(self):
        rows = [
            {"txn": "T1", "product": "A"},
            {"txn": "T1", "product": "B"},
            {"txn": "T2", "product": "C"},
        ]
        baskets = _build_baskets(rows, "txn", "product")
        assert baskets is not None
        assert baskets["T1"] == {"A", "B"}
        assert baskets["T2"] == {"C"}

    def test_deduplication(self):
        rows = [
            {"txn": "T1", "product": "A"},
            {"txn": "T1", "product": "A"},
        ]
        baskets = _build_baskets(rows, "txn", "product")
        assert baskets is not None
        assert baskets["T1"] == {"A"}

    def test_skips_none_txn(self):
        rows = [
            {"txn": None, "product": "A"},
            {"txn": "T1", "product": "B"},
        ]
        baskets = _build_baskets(rows, "txn", "product")
        assert baskets is not None
        assert len(baskets) == 1

    def test_skips_none_product(self):
        rows = [
            {"txn": "T1", "product": None},
            {"txn": "T1", "product": "B"},
        ]
        baskets = _build_baskets(rows, "txn", "product")
        assert baskets is not None
        assert baskets["T1"] == {"B"}


# ---------------------------------------------------------------------------
# find_product_associations
# ---------------------------------------------------------------------------


def _make_basket_rows(baskets: dict[str, list[str]]) -> list[dict]:
    """Helper to create rows from a dict of {txn_id: [products]}."""
    rows = []
    for txn, products in baskets.items():
        for p in products:
            rows.append({"txn": txn, "product": p})
    return rows


class TestFindProductAssociations:
    def test_empty_rows(self):
        assert find_product_associations([], "txn", "product") is None

    def test_missing_columns(self):
        rows = [{"other": "val"}]
        assert find_product_associations(rows, "txn", "product") is None

    def test_single_product_no_pairs(self):
        rows = [
            {"txn": "T1", "product": "A"},
            {"txn": "T2", "product": "A"},
        ]
        result = find_product_associations(rows, "txn", "product")
        assert result is not None
        assert len(result.pairs) == 0
        assert result.unique_products == 1

    def test_two_products_always_together(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
            "T3": ["A", "B"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.support == 1.0
        assert pair.confidence_a_to_b == 1.0
        assert pair.confidence_b_to_a == 1.0
        assert pair.lift == 1.0
        assert pair.co_occurrence_count == 3

    def test_support_calculation(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "C"],
            "T3": ["B", "C"],
            "T4": ["A", "B", "C"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert result.total_transactions == 4
        # A&B appear together in T1, T4 => support = 2/4 = 0.5
        ab_pair = next(
            (p for p in result.pairs if
             {p.product_a, p.product_b} == {"A", "B"}),
            None,
        )
        assert ab_pair is not None
        assert ab_pair.support == 0.5
        assert ab_pair.co_occurrence_count == 2

    def test_confidence_asymmetric(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A"],
            "T3": ["B"],
            "T4": ["A", "B"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        pair = result.pairs[0]
        # A appears in T1, T2, T4 (3 times). A&B in T1, T4 (2 times).
        # confidence(A->B) = 2/3
        # B appears in T1, T3, T4 (3 times). confidence(B->A) = 2/3
        assert abs(pair.confidence_a_to_b - 2 / 3) < 0.01
        assert abs(pair.confidence_b_to_a - 2 / 3) < 0.01

    def test_lift_greater_than_one(self):
        # A and B appear together more than expected by chance
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
            "T3": ["C", "D"],
            "T4": ["C", "D"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        ab_pair = next(
            (p for p in result.pairs if
             {p.product_a, p.product_b} == {"A", "B"}),
            None,
        )
        assert ab_pair is not None
        assert ab_pair.lift > 1.0

    def test_lift_equals_one_independent(self):
        # Products appear in every transaction => lift = 1
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert result.pairs[0].lift == 1.0

    def test_min_support_filtering(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["C", "D"],
            "T3": ["E", "F"],
            "T4": ["G", "H"],
        })
        # Each pair has support = 1/4 = 0.25
        result_low = find_product_associations(rows, "txn", "product", min_support=0.2)
        assert result_low is not None
        assert len(result_low.pairs) == 4

        result_high = find_product_associations(rows, "txn", "product", min_support=0.3)
        assert result_high is not None
        assert len(result_high.pairs) == 0

    def test_top_20_limit(self):
        # Create 25 distinct pairs
        baskets = {}
        for i in range(25):
            baskets[f"T{i}"] = [f"P{i}", f"Q{i}"]
        # Add a universal pair so the 25 pairs all have low support
        # Instead, put all pairs in each transaction
        baskets_data = {}
        for i in range(25):
            baskets_data[f"T{i}"] = [f"P{2*i}", f"P{2*i+1}"]
        rows = _make_basket_rows(baskets_data)
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert len(result.pairs) <= 20

    def test_avg_basket_size(self):
        rows = _make_basket_rows({
            "T1": ["A", "B", "C"],
            "T2": ["A"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert result.avg_basket_size == 2.0  # (3+1)/2

    def test_unique_products_count(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["B", "C"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert result.unique_products == 3

    def test_total_transactions(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["C", "D"],
            "T3": ["E"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert result.total_transactions == 3

    def test_summary_not_empty(self):
        rows = _make_basket_rows({"T1": ["A", "B"]})
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert len(result.summary) > 0

    def test_summary_mentions_strongest_pair(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert "A" in result.summary
        assert "B" in result.summary

    def test_sorted_by_lift_descending(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
            "T3": ["C", "D"],
            "T4": ["A", "C"],
            "T5": ["B", "D"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        if len(result.pairs) >= 2:
            for i in range(len(result.pairs) - 1):
                assert result.pairs[i].lift >= result.pairs[i + 1].lift

    def test_single_item_transactions_no_pairs(self):
        rows = _make_basket_rows({
            "T1": ["A"],
            "T2": ["B"],
            "T3": ["C"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert len(result.pairs) == 0

    def test_large_basket(self):
        rows = _make_basket_rows({
            "T1": ["A", "B", "C", "D", "E"],
        })
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        # 5 choose 2 = 10 pairs
        assert len(result.pairs) == 10

    def test_duplicate_product_in_row(self):
        rows = [
            {"txn": "T1", "product": "A"},
            {"txn": "T1", "product": "A"},
            {"txn": "T1", "product": "B"},
        ]
        result = find_product_associations(rows, "txn", "product", min_support=0.0)
        assert result is not None
        assert result.pairs[0].co_occurrence_count == 1


# ---------------------------------------------------------------------------
# analyze_basket_size
# ---------------------------------------------------------------------------


class TestAnalyzeBasketSize:
    def test_empty_rows(self):
        assert analyze_basket_size([], "txn", "product") is None

    def test_missing_columns(self):
        rows = [{"other": "val"}]
        assert analyze_basket_size(rows, "txn", "product") is None

    def test_single_item_basket(self):
        rows = [{"txn": "T1", "product": "A"}]
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        assert result.avg_size == 1.0
        assert result.median_size == 1.0
        assert result.max_size == 1
        assert result.min_size == 1

    def test_multiple_baskets(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["C"],
            "T3": ["D", "E", "F"],
        })
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        assert result.min_size == 1
        assert result.max_size == 3
        assert result.avg_size == 2.0  # (2+1+3)/3
        assert result.median_size == 2.0

    def test_distribution(self):
        rows = _make_basket_rows({
            "T1": ["A"],
            "T2": ["B", "C"],
            "T3": ["D"],
            "T4": ["E", "F"],
        })
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        assert len(result.distribution) == 2  # size 1 and size 2
        size_1 = next(b for b in result.distribution if b.size == 1)
        assert size_1.count == 2
        assert size_1.pct == 50.0
        size_2 = next(b for b in result.distribution if b.size == 2)
        assert size_2.count == 2
        assert size_2.pct == 50.0

    def test_distribution_sorted_by_size(self):
        rows = _make_basket_rows({
            "T1": ["A", "B", "C"],
            "T2": ["D"],
            "T3": ["E", "F"],
        })
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        sizes = [b.size for b in result.distribution]
        assert sizes == sorted(sizes)

    def test_without_value_column(self):
        rows = _make_basket_rows({"T1": ["A", "B"]})
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        assert result.avg_value is None

    def test_with_value_column(self):
        rows = [
            {"txn": "T1", "product": "A", "price": 10},
            {"txn": "T1", "product": "B", "price": 20},
            {"txn": "T2", "product": "C", "price": 30},
        ]
        result = analyze_basket_size(rows, "txn", "product", value_column="price")
        assert result is not None
        # T1 value = 30, T2 value = 30, avg = 30
        assert result.avg_value == 30.0

    def test_with_value_column_different_values(self):
        rows = [
            {"txn": "T1", "product": "A", "price": 100},
            {"txn": "T2", "product": "B", "price": 200},
        ]
        result = analyze_basket_size(rows, "txn", "product", value_column="price")
        assert result is not None
        assert result.avg_value == 150.0  # (100+200)/2

    def test_value_column_missing_values(self):
        rows = [
            {"txn": "T1", "product": "A", "price": 100},
            {"txn": "T2", "product": "B"},  # missing price
        ]
        result = analyze_basket_size(rows, "txn", "product", value_column="price")
        assert result is not None
        # Only T1 has a value => avg_value = 100
        assert result.avg_value == 100.0

    def test_all_value_column_missing(self):
        rows = [
            {"txn": "T1", "product": "A"},
            {"txn": "T2", "product": "B"},
        ]
        result = analyze_basket_size(rows, "txn", "product", value_column="price")
        assert result is not None
        assert result.avg_value is None

    def test_summary_not_empty(self):
        rows = _make_basket_rows({"T1": ["A"]})
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        assert len(result.summary) > 0

    def test_summary_includes_avg(self):
        rows = _make_basket_rows({"T1": ["A", "B"], "T2": ["C"]})
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        assert "1.5" in result.summary

    def test_summary_with_value(self):
        rows = [
            {"txn": "T1", "product": "A", "price": 50},
        ]
        result = analyze_basket_size(rows, "txn", "product", value_column="price")
        assert result is not None
        assert "50.00" in result.summary

    def test_pct_sums_to_100(self):
        rows = _make_basket_rows({
            "T1": ["A"],
            "T2": ["B", "C"],
            "T3": ["D", "E", "F"],
        })
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        total_pct = sum(b.pct for b in result.distribution)
        assert abs(total_pct - 100.0) < 0.5

    def test_even_median(self):
        rows = _make_basket_rows({
            "T1": ["A"],
            "T2": ["B", "C"],
            "T3": ["D", "E", "F"],
            "T4": ["G", "H", "I", "J"],
        })
        result = analyze_basket_size(rows, "txn", "product")
        assert result is not None
        # Sizes: 1, 2, 3, 4 => median = 2.5
        assert result.median_size == 2.5


# ---------------------------------------------------------------------------
# find_cross_sell_opportunities
# ---------------------------------------------------------------------------


class TestFindCrossSellOpportunities:
    def test_empty_rows(self):
        assert find_cross_sell_opportunities([], "txn", "product", "A") is None

    def test_missing_columns(self):
        rows = [{"other": "val"}]
        assert find_cross_sell_opportunities(rows, "txn", "product", "A") is None

    def test_target_not_found(self):
        rows = [{"txn": "T1", "product": "B"}]
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is None

    def test_target_always_alone(self):
        rows = [
            {"txn": "T1", "product": "A"},
            {"txn": "T2", "product": "A"},
        ]
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        assert result.target_product == "A"
        assert result.target_transactions == 2
        assert len(result.recommendations) == 0

    def test_basic_cross_sell(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "C"],
            "T3": ["A", "B", "C"],
        })
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        assert result.target_product == "A"
        assert result.target_transactions == 3

        # B appears in T1, T3 => co_purchase_rate = 2/3
        b_rec = next(r for r in result.recommendations if r.product == "B")
        assert abs(b_rec.co_purchase_rate - 2 / 3) < 0.01
        assert b_rec.co_occurrence_count == 2

        # C appears in T2, T3 => co_purchase_rate = 2/3
        c_rec = next(r for r in result.recommendations if r.product == "C")
        assert abs(c_rec.co_purchase_rate - 2 / 3) < 0.01

    def test_co_purchase_rate_one(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
        })
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        assert result.recommendations[0].co_purchase_rate == 1.0

    def test_lift_calculation(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
            "T3": ["C"],
            "T4": ["C"],
        })
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        b_rec = next(r for r in result.recommendations if r.product == "B")
        # P(B|A) = 1.0, P(B) = 2/4 = 0.5 => lift = 2.0
        assert b_rec.lift == 2.0

    def test_sorted_by_co_purchase_rate(self):
        rows = _make_basket_rows({
            "T1": ["A", "B", "C"],
            "T2": ["A", "B"],
            "T3": ["A", "C"],
            "T4": ["A", "D"],
        })
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        rates = [r.co_purchase_rate for r in result.recommendations]
        assert rates == sorted(rates, reverse=True)

    def test_summary_includes_target(self):
        rows = _make_basket_rows({"T1": ["Widget", "Gadget"]})
        result = find_cross_sell_opportunities(rows, "txn", "product", "Widget")
        assert result is not None
        assert "Widget" in result.summary

    def test_summary_includes_top_recommendation(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
        })
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        assert "B" in result.summary

    def test_target_not_in_recommendations(self):
        rows = _make_basket_rows({"T1": ["A", "B", "C"]})
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        products = [r.product for r in result.recommendations]
        assert "A" not in products

    def test_multiple_recommendations_ordering(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
            "T3": ["A", "B"],
            "T4": ["A", "C"],
        })
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        assert len(result.recommendations) == 2
        # B has higher co_purchase_rate than C
        assert result.recommendations[0].product == "B"
        assert result.recommendations[1].product == "C"

    def test_alone_summary_message(self):
        rows = [
            {"txn": "T1", "product": "A"},
            {"txn": "T2", "product": "A"},
        ]
        result = find_cross_sell_opportunities(rows, "txn", "product", "A")
        assert result is not None
        assert "alone" in result.summary.lower() or "no cross-sell" in result.summary.lower()


# ---------------------------------------------------------------------------
# analyze_product_frequency
# ---------------------------------------------------------------------------


class TestAnalyzeProductFrequency:
    def test_empty_rows(self):
        assert analyze_product_frequency([], "txn", "product") is None

    def test_missing_columns(self):
        rows = [{"other": "val"}]
        assert analyze_product_frequency(rows, "txn", "product") is None

    def test_single_product(self):
        rows = [{"txn": "T1", "product": "A"}]
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert len(result.products) == 1
        assert result.products[0].product == "A"
        assert result.products[0].frequency == 1
        assert result.products[0].pct_of_transactions == 100.0
        assert result.products[0].rank == 1

    def test_frequency_ranking(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "C"],
            "T3": ["A"],
            "T4": ["B"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        # A in 3 txns, B in 2 txns, C in 1 txn
        assert result.products[0].product == "A"
        assert result.products[0].frequency == 3
        assert result.products[0].rank == 1
        assert result.products[1].product == "B"
        assert result.products[1].frequency == 2
        assert result.products[1].rank == 2
        assert result.products[2].product == "C"
        assert result.products[2].frequency == 1
        assert result.products[2].rank == 3

    def test_pct_of_transactions(self):
        rows = _make_basket_rows({
            "T1": ["A"],
            "T2": ["A"],
            "T3": ["B"],
            "T4": ["B"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        # Each product in 2 out of 4 transactions = 50%
        for pf in result.products:
            assert pf.pct_of_transactions == 50.0

    def test_most_popular(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert result.most_popular == "A"

    def test_least_popular(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert result.least_popular == "B"

    def test_total_transactions(self):
        rows = _make_basket_rows({
            "T1": ["A"],
            "T2": ["B"],
            "T3": ["C"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert result.total_transactions == 3

    def test_total_products(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["C"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert result.total_products == 3

    def test_without_customer_column(self):
        rows = [{"txn": "T1", "product": "A"}]
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert result.products[0].unique_customers is None

    def test_with_customer_column(self):
        rows = [
            {"txn": "T1", "product": "A", "customer": "C1"},
            {"txn": "T2", "product": "A", "customer": "C2"},
            {"txn": "T3", "product": "A", "customer": "C1"},
            {"txn": "T4", "product": "B", "customer": "C3"},
        ]
        result = analyze_product_frequency(rows, "txn", "product", customer_column="customer")
        assert result is not None
        a_prod = next(p for p in result.products if p.product == "A")
        assert a_prod.unique_customers == 2  # C1, C2
        b_prod = next(p for p in result.products if p.product == "B")
        assert b_prod.unique_customers == 1

    def test_customer_column_missing_values(self):
        rows = [
            {"txn": "T1", "product": "A", "customer": "C1"},
            {"txn": "T2", "product": "A"},  # no customer
        ]
        result = analyze_product_frequency(rows, "txn", "product", customer_column="customer")
        assert result is not None
        a_prod = next(p for p in result.products if p.product == "A")
        assert a_prod.unique_customers == 1

    def test_summary_not_empty(self):
        rows = [{"txn": "T1", "product": "A"}]
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert len(result.summary) > 0

    def test_summary_mentions_most_popular(self):
        rows = _make_basket_rows({
            "T1": ["Widget"],
            "T2": ["Widget"],
            "T3": ["Gadget"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert "Widget" in result.summary

    def test_product_in_every_transaction(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "C"],
            "T3": ["A", "D"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        a_prod = next(p for p in result.products if p.product == "A")
        assert a_prod.pct_of_transactions == 100.0
        assert a_prod.frequency == 3

    def test_tied_frequency(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
        })
        result = analyze_product_frequency(rows, "txn", "product")
        assert result is not None
        assert len(result.products) == 2
        assert result.products[0].frequency == 2
        assert result.products[1].frequency == 2


# ---------------------------------------------------------------------------
# format_basket_report
# ---------------------------------------------------------------------------


class TestFormatBasketReport:
    def test_empty_report(self):
        report = format_basket_report()
        assert "MARKET BASKET ANALYSIS REPORT" in report
        assert "No data available" in report

    def test_associations_only(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "B"],
        })
        assoc = find_product_associations(rows, "txn", "product", min_support=0.0)
        report = format_basket_report(associations=assoc)
        assert "PRODUCT ASSOCIATIONS" in report
        assert "A" in report
        assert "B" in report

    def test_basket_size_only(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["C"],
        })
        bs = analyze_basket_size(rows, "txn", "product")
        report = format_basket_report(basket_size=bs)
        assert "BASKET SIZE DISTRIBUTION" in report

    def test_cross_sell_only(self):
        rows = _make_basket_rows({"T1": ["A", "B"]})
        cs = find_cross_sell_opportunities(rows, "txn", "product", "A")
        report = format_basket_report(cross_sell=cs)
        assert "CROSS-SELL OPPORTUNITIES" in report
        assert "A" in report

    def test_frequency_only(self):
        rows = _make_basket_rows({"T1": ["X", "Y"]})
        freq = analyze_product_frequency(rows, "txn", "product")
        report = format_basket_report(frequency=freq)
        assert "PRODUCT FREQUENCY" in report

    def test_combined_report(self):
        rows = _make_basket_rows({
            "T1": ["A", "B"],
            "T2": ["A", "C"],
            "T3": ["B", "C"],
        })
        assoc = find_product_associations(rows, "txn", "product", min_support=0.0)
        bs = analyze_basket_size(rows, "txn", "product")
        cs = find_cross_sell_opportunities(rows, "txn", "product", "A")
        freq = analyze_product_frequency(rows, "txn", "product")

        report = format_basket_report(
            associations=assoc,
            basket_size=bs,
            cross_sell=cs,
            frequency=freq,
        )
        assert "PRODUCT ASSOCIATIONS" in report
        assert "BASKET SIZE DISTRIBUTION" in report
        assert "CROSS-SELL OPPORTUNITIES" in report
        assert "PRODUCT FREQUENCY" in report

    def test_report_has_separator_lines(self):
        report = format_basket_report()
        assert "=" * 60 in report

    def test_report_with_value_column(self):
        rows = [
            {"txn": "T1", "product": "A", "price": 50},
            {"txn": "T1", "product": "B", "price": 30},
        ]
        bs = analyze_basket_size(rows, "txn", "product", value_column="price")
        report = format_basket_report(basket_size=bs)
        assert "Avg Basket Value" in report

    def test_report_with_customer_data(self):
        rows = [
            {"txn": "T1", "product": "A", "customer": "C1"},
            {"txn": "T2", "product": "A", "customer": "C2"},
        ]
        freq = analyze_product_frequency(rows, "txn", "product", customer_column="customer")
        report = format_basket_report(frequency=freq)
        assert "customers=" in report

    def test_report_without_customer_data(self):
        rows = [{"txn": "T1", "product": "A"}]
        freq = analyze_product_frequency(rows, "txn", "product")
        report = format_basket_report(frequency=freq)
        assert "customers=" not in report

    def test_report_no_pairs(self):
        rows = [
            {"txn": "T1", "product": "A"},
            {"txn": "T2", "product": "B"},
        ]
        assoc = find_product_associations(rows, "txn", "product", min_support=0.0)
        report = format_basket_report(associations=assoc)
        assert "PRODUCT ASSOCIATIONS" in report
        assert "Pairs Found" in report


# ---------------------------------------------------------------------------
# Dataclass field tests
# ---------------------------------------------------------------------------


class TestDataclassFields:
    def test_product_pair_fields(self):
        pp = ProductPair(
            product_a="A", product_b="B",
            support=0.5, confidence_a_to_b=0.8,
            confidence_b_to_a=0.6, lift=1.5,
            co_occurrence_count=10,
        )
        assert pp.product_a == "A"
        assert pp.product_b == "B"
        assert pp.support == 0.5
        assert pp.confidence_a_to_b == 0.8
        assert pp.confidence_b_to_a == 0.6
        assert pp.lift == 1.5
        assert pp.co_occurrence_count == 10

    def test_association_result_fields(self):
        ar = AssociationResult(
            pairs=[], total_transactions=100,
            unique_products=10, avg_basket_size=3.5,
            summary="test",
        )
        assert ar.total_transactions == 100
        assert ar.unique_products == 10
        assert ar.avg_basket_size == 3.5

    def test_size_bucket_fields(self):
        sb = SizeBucket(size=3, count=5, pct=25.0)
        assert sb.size == 3
        assert sb.count == 5
        assert sb.pct == 25.0

    def test_basket_size_result_fields(self):
        bsr = BasketSizeResult(
            avg_size=2.5, median_size=2.0,
            max_size=5, min_size=1,
            distribution=[], avg_value=100.0,
            summary="test",
        )
        assert bsr.avg_size == 2.5
        assert bsr.median_size == 2.0
        assert bsr.max_size == 5
        assert bsr.min_size == 1
        assert bsr.avg_value == 100.0

    def test_cross_sell_product_fields(self):
        csp = CrossSellProduct(
            product="B", co_purchase_rate=0.75,
            lift=2.0, co_occurrence_count=15,
        )
        assert csp.product == "B"
        assert csp.co_purchase_rate == 0.75
        assert csp.lift == 2.0
        assert csp.co_occurrence_count == 15

    def test_cross_sell_result_fields(self):
        csr = CrossSellResult(
            target_product="A", target_transactions=20,
            recommendations=[], summary="test",
        )
        assert csr.target_product == "A"
        assert csr.target_transactions == 20

    def test_product_freq_fields(self):
        pf = ProductFreq(
            product="X", frequency=42,
            pct_of_transactions=84.0,
            unique_customers=10, rank=1,
        )
        assert pf.product == "X"
        assert pf.frequency == 42
        assert pf.pct_of_transactions == 84.0
        assert pf.unique_customers == 10
        assert pf.rank == 1

    def test_product_freq_no_customers(self):
        pf = ProductFreq(
            product="Y", frequency=5,
            pct_of_transactions=50.0,
            unique_customers=None, rank=2,
        )
        assert pf.unique_customers is None

    def test_product_frequency_result_fields(self):
        pfr = ProductFrequencyResult(
            products=[], total_transactions=50,
            total_products=5, most_popular="A",
            least_popular="E", summary="test",
        )
        assert pfr.total_transactions == 50
        assert pfr.total_products == 5
        assert pfr.most_popular == "A"
        assert pfr.least_popular == "E"
