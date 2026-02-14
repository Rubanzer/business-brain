"""Tests for customer_analytics module -- 50+ tests covering all functions."""

from datetime import datetime, timedelta

from business_brain.discovery.customer_analytics import (
    BehaviorResult,
    ChurnResult,
    ChurnStatus,
    ConcentrationResult,
    CustomerBehavior,
    CustomerSegmentResult,
    CustomerShare,
    CustomerTier,
    analyze_churn_risk,
    analyze_customer_segments,
    analyze_purchase_behavior,
    compute_customer_concentration,
    format_customer_report,
    _safe_float,
    _parse_date,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_segment_rows(n=20):
    """Generate rows with varied revenue for segmentation testing."""
    rows = []
    for i in range(n):
        rows.append({
            "customer": f"cust_{i:03d}",
            "revenue": (i + 1) * 100,
            "frequency": (i % 5) + 1,
            "category": ["Electronics", "Clothing", "Food", "Books", "Sports"][i % 5],
        })
    return rows


def _make_transaction_rows():
    """Generate transaction rows with realistic date spread."""
    base = datetime(2024, 6, 30)
    rows = []
    customers = {
        "Alice": {"days_ago": [1, 5, 10, 15], "amounts": [500, 400, 300, 200]},
        "Bob": {"days_ago": [2, 8], "amounts": [100, 150]},
        "Charlie": {"days_ago": [60, 70], "amounts": [200, 250]},
        "Diana": {"days_ago": [100, 120, 140], "amounts": [300, 350, 400]},
        "Eve": {"days_ago": [200, 220, 240], "amounts": [50, 60, 70]},
        "Frank": {"days_ago": [350, 365], "amounts": [1000, 1200]},
    }
    for cust, data in customers.items():
        for d, a in zip(data["days_ago"], data["amounts"]):
            rows.append({
                "customer": cust,
                "date": (base - timedelta(days=d)).strftime("%Y-%m-%d"),
                "amount": a,
            })
    return rows


def _make_concentrated_rows():
    """Revenue heavily concentrated in one customer."""
    return [
        {"customer": "BigCo", "amount": 10000},
        {"customer": "SmallA", "amount": 100},
        {"customer": "SmallB", "amount": 100},
        {"customer": "SmallC", "amount": 100},
        {"customer": "SmallD", "amount": 100},
        {"customer": "SmallE", "amount": 50},
        {"customer": "SmallF", "amount": 50},
        {"customer": "SmallG", "amount": 50},
        {"customer": "SmallH", "amount": 50},
        {"customer": "SmallI", "amount": 50},
    ]


def _make_behavior_rows():
    """Rows for purchase behavior analysis with products."""
    return [
        {"customer": "Alice", "amount": 100, "date": "2024-01-01", "product": "Widget"},
        {"customer": "Alice", "amount": 200, "date": "2024-01-01", "product": "Gadget"},
        {"customer": "Alice", "amount": 150, "date": "2024-03-15", "product": "Widget"},
        {"customer": "Alice", "amount": 300, "date": "2024-06-01", "product": "Gizmo"},
        {"customer": "Bob", "amount": 50, "date": "2024-02-01", "product": "Widget"},
        {"customer": "Charlie", "amount": 75, "date": "2024-04-01", "product": "Gadget"},
        {"customer": "Charlie", "amount": 80, "date": "2024-05-01", "product": "Gadget"},
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


# ---------------------------------------------------------------------------
# Tests: _parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_iso_format(self):
        dt = _parse_date("2024-06-15")
        assert dt == datetime(2024, 6, 15)

    def test_datetime_passthrough(self):
        dt = datetime(2024, 1, 1)
        assert _parse_date(dt) is dt

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_date("") is None

    def test_slash_format(self):
        dt = _parse_date("06/15/2024")
        assert dt is not None

    def test_unparseable_returns_none(self):
        assert _parse_date("not-a-date") is None


# ---------------------------------------------------------------------------
# Tests: analyze_customer_segments
# ---------------------------------------------------------------------------


class TestAnalyzeCustomerSegments:
    def test_basic_segmentation(self):
        rows = _make_segment_rows()
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        assert isinstance(result, CustomerSegmentResult)
        assert result.total_customers == 20

    def test_empty_rows_returns_none(self):
        assert analyze_customer_segments([], "customer", "revenue") is None

    def test_all_null_returns_none(self):
        rows = [{"customer": None, "revenue": None}]
        assert analyze_customer_segments(rows, "customer", "revenue") is None

    def test_three_tiers_exist(self):
        rows = _make_segment_rows(30)
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        tier_names = {t.tier for t in result.tiers}
        assert "Premium" in tier_names
        assert "Standard" in tier_names
        assert "Basic" in tier_names

    def test_premium_has_highest_avg_revenue(self):
        rows = _make_segment_rows(30)
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        premium = next(t for t in result.tiers if t.tier == "Premium")
        basic = next(t for t in result.tiers if t.tier == "Basic")
        assert premium.avg_revenue > basic.avg_revenue

    def test_total_revenue_sums_correctly(self):
        rows = _make_segment_rows(10)
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        tier_sum = sum(t.total_revenue for t in result.tiers)
        assert abs(tier_sum - result.total_revenue) < 0.01

    def test_pct_of_total_sums_to_100(self):
        rows = _make_segment_rows(20)
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        pct_sum = sum(t.pct_of_total for t in result.tiers)
        assert abs(pct_sum - 100.0) < 0.1

    def test_customer_counts_sum_to_total(self):
        rows = _make_segment_rows(20)
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        count_sum = sum(t.customer_count for t in result.tiers)
        assert count_sum == result.total_customers

    def test_with_frequency_column(self):
        rows = _make_segment_rows(20)
        result = analyze_customer_segments(
            rows, "customer", "revenue", frequency_column="frequency"
        )
        assert result is not None
        for t in result.tiers:
            assert t.avg_frequency is not None
            assert t.avg_frequency > 0

    def test_without_frequency_column(self):
        rows = _make_segment_rows(20)
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        for t in result.tiers:
            assert t.avg_frequency is None

    def test_with_category_column(self):
        rows = _make_segment_rows(20)
        result = analyze_customer_segments(
            rows, "customer", "revenue", category_column="category"
        )
        assert result is not None
        # Just verify it doesn't crash and returns valid result
        assert result.total_customers == 20

    def test_single_customer(self):
        rows = [{"customer": "Solo", "revenue": 500}]
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        assert result.total_customers == 1

    def test_two_customers(self):
        rows = [
            {"customer": "A", "revenue": 1000},
            {"customer": "B", "revenue": 100},
        ]
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        assert result.total_customers == 2

    def test_string_revenue_parsed(self):
        rows = [
            {"customer": "A", "revenue": "500.50"},
            {"customer": "B", "revenue": "200"},
            {"customer": "C", "revenue": "300"},
        ]
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        assert result.total_revenue == 1000.50

    def test_summary_present(self):
        rows = _make_segment_rows()
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        assert "Segmentation" in result.summary

    def test_duplicate_customers_aggregated(self):
        rows = [
            {"customer": "A", "revenue": 100},
            {"customer": "A", "revenue": 200},
            {"customer": "B", "revenue": 50},
        ]
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        assert result.total_customers == 2
        assert result.total_revenue == 350.0

    def test_zero_revenue(self):
        rows = [
            {"customer": "A", "revenue": 0},
            {"customer": "B", "revenue": 0},
            {"customer": "C", "revenue": 0},
        ]
        result = analyze_customer_segments(rows, "customer", "revenue")
        assert result is not None
        assert result.total_revenue == 0.0


# ---------------------------------------------------------------------------
# Tests: analyze_churn_risk
# ---------------------------------------------------------------------------


class TestAnalyzeChurnRisk:
    def test_basic_churn_analysis(self):
        rows = _make_transaction_rows()
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        assert isinstance(result, ChurnResult)

    def test_empty_rows_returns_none(self):
        assert analyze_churn_risk([], "customer", "date") is None

    def test_all_null_returns_none(self):
        rows = [{"customer": None, "date": None}]
        assert analyze_churn_risk(rows, "customer", "date") is None

    def test_active_classification(self):
        """Customers with activity within 30 days are Active."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=10)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        active = next(s for s in result.statuses if s.status == "Active")
        assert active.customer_count == 2

    def test_at_risk_classification(self):
        """Customers with last activity 31-90 days ago are At-Risk."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=60)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        at_risk = next(s for s in result.statuses if s.status == "At-Risk")
        assert at_risk.customer_count == 1

    def test_dormant_classification(self):
        """Customers with last activity 91-180 days ago are Dormant."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=150)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        dormant = next(s for s in result.statuses if s.status == "Dormant")
        assert dormant.customer_count == 1

    def test_churned_classification(self):
        """Customers with last activity >180 days ago are Churned."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=200)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        churned = next(s for s in result.statuses if s.status == "Churned")
        assert churned.customer_count == 1

    def test_churn_rate_calculation(self):
        """Churn rate = churned / total * 100."""
        rows = _make_transaction_rows()
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        churned = next(s for s in result.statuses if s.status == "Churned")
        expected_rate = churned.customer_count / result.total_customers * 100
        assert abs(result.churn_rate - round(expected_rate, 2)) < 0.01

    def test_at_risk_rate_calculation(self):
        rows = _make_transaction_rows()
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        at_risk = next(s for s in result.statuses if s.status == "At-Risk")
        expected = at_risk.customer_count / result.total_customers * 100
        assert abs(result.at_risk_rate - round(expected, 2)) < 0.01

    def test_status_pcts_sum_to_100(self):
        rows = _make_transaction_rows()
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        pct_sum = sum(s.pct for s in result.statuses)
        assert abs(pct_sum - 100.0) < 0.1

    def test_with_amount_column(self):
        rows = _make_transaction_rows()
        result = analyze_churn_risk(rows, "customer", "date", amount_column="amount")
        assert result is not None
        for s in result.statuses:
            if s.customer_count > 0:
                assert s.avg_spend is not None

    def test_without_amount_column(self):
        rows = _make_transaction_rows()
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        for s in result.statuses:
            assert s.avg_spend is None

    def test_single_customer(self):
        rows = [{"customer": "Solo", "date": "2024-06-30"}]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        assert result.total_customers == 1
        active = next(s for s in result.statuses if s.status == "Active")
        assert active.customer_count == 1

    def test_all_same_date(self):
        """All customers active on same date -> all Active, 0% churn."""
        rows = [
            {"customer": "A", "date": "2024-06-30"},
            {"customer": "B", "date": "2024-06-30"},
            {"customer": "C", "date": "2024-06-30"},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        assert result.churn_rate == 0.0
        active = next(s for s in result.statuses if s.status == "Active")
        assert active.customer_count == 3

    def test_boundary_30_days(self):
        """Customer exactly 30 days ago is Active."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=30)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        active = next(s for s in result.statuses if s.status == "Active")
        assert active.customer_count == 2

    def test_boundary_90_days(self):
        """Customer exactly 90 days ago is At-Risk."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=90)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        at_risk = next(s for s in result.statuses if s.status == "At-Risk")
        assert at_risk.customer_count == 1

    def test_boundary_180_days(self):
        """Customer exactly 180 days ago is Dormant."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=180)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        dormant = next(s for s in result.statuses if s.status == "Dormant")
        assert dormant.customer_count == 1

    def test_boundary_181_days(self):
        """Customer exactly 181 days ago is Churned."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=181)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        churned = next(s for s in result.statuses if s.status == "Churned")
        assert churned.customer_count == 1

    def test_summary_contains_key_info(self):
        rows = _make_transaction_rows()
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        assert "Churn" in result.summary
        assert str(result.total_customers) in result.summary

    def test_multiple_transactions_uses_last_date(self):
        """Multiple transactions for same customer -> recency from latest."""
        base = datetime(2024, 6, 30)
        rows = [
            {"customer": "A", "date": base.strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=200)).strftime("%Y-%m-%d")},
            {"customer": "B", "date": (base - timedelta(days=10)).strftime("%Y-%m-%d")},
        ]
        result = analyze_churn_risk(rows, "customer", "date")
        assert result is not None
        # B's latest is 10 days ago -> Active
        active = next(s for s in result.statuses if s.status == "Active")
        assert active.customer_count == 2


# ---------------------------------------------------------------------------
# Tests: compute_customer_concentration
# ---------------------------------------------------------------------------


class TestComputeCustomerConcentration:
    def test_basic_concentration(self):
        rows = _make_concentrated_rows()
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert isinstance(result, ConcentrationResult)

    def test_empty_rows_returns_none(self):
        assert compute_customer_concentration([], "customer", "amount") is None

    def test_all_null_returns_none(self):
        rows = [{"customer": None, "amount": None}]
        assert compute_customer_concentration(rows, "customer", "amount") is None

    def test_high_concentration_risk(self):
        """Single customer > 25% share -> High risk."""
        rows = _make_concentrated_rows()
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert result.concentration_risk == "High"

    def test_low_concentration_risk(self):
        """Equal distribution -> Low risk."""
        rows = [{"customer": f"cust_{i}", "amount": 100} for i in range(20)]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert result.concentration_risk == "Low"

    def test_moderate_concentration_risk(self):
        """Top 5 > 60% but no single > 25% -> Moderate."""
        rows = [
            {"customer": "A", "amount": 240},
            {"customer": "B", "amount": 230},
            {"customer": "C", "amount": 220},
            {"customer": "D", "amount": 210},
            {"customer": "E", "amount": 200},
            {"customer": "F", "amount": 30},
            {"customer": "G", "amount": 25},
            {"customer": "H", "amount": 20},
            {"customer": "I", "amount": 15},
            {"customer": "J", "amount": 10},
        ]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        # Top 5 share: (240+230+220+210+200) / 1200 = 91.7% > 60%
        # Top 1 share: 240/1200 = 20% < 25%
        assert result.concentration_risk == "Moderate"

    def test_hhi_single_customer(self):
        """Single customer = HHI of 10000."""
        rows = [{"customer": "Monopoly", "amount": 1000}]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert result.hhi == 10000.0

    def test_hhi_equal_two(self):
        """Two equal customers -> HHI = 5000."""
        rows = [
            {"customer": "A", "amount": 500},
            {"customer": "B", "amount": 500},
        ]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert result.hhi == 5000.0

    def test_hhi_equal_ten(self):
        """Ten equal customers -> HHI = 1000."""
        rows = [{"customer": f"c{i}", "amount": 100} for i in range(10)]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert result.hhi == 1000.0

    def test_pareto_80_pct(self):
        rows = _make_concentrated_rows()
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        # BigCo has 10000 / 10650 = 93.9% -> 1 customer for 80%
        assert result.customers_for_80pct == 1

    def test_top_customers_limited_to_10(self):
        rows = [{"customer": f"cust_{i}", "amount": 100 + i} for i in range(20)]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert len(result.top_customers) <= 10

    def test_top_customers_sorted_descending(self):
        rows = _make_concentrated_rows()
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        spends = [c.total_spend for c in result.top_customers]
        assert spends == sorted(spends, reverse=True)

    def test_cumulative_pct_monotonic(self):
        rows = _make_concentrated_rows()
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        for i in range(1, len(result.top_customers)):
            assert result.top_customers[i].cumulative_pct >= result.top_customers[i - 1].cumulative_pct

    def test_cumulative_pct_ends_at_100(self):
        """If all customers are in top list, cumulative should end near 100."""
        rows = [{"customer": f"c{i}", "amount": 100} for i in range(5)]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert abs(result.top_customers[-1].cumulative_pct - 100.0) < 0.1

    def test_share_pct_sum_to_100_for_full_list(self):
        rows = [{"customer": f"c{i}", "amount": 100} for i in range(5)]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        total_share = sum(c.share_pct for c in result.top_customers)
        assert abs(total_share - 100.0) < 0.1

    def test_zero_total_revenue_returns_none(self):
        rows = [
            {"customer": "A", "amount": 0},
            {"customer": "B", "amount": 0},
        ]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is None

    def test_summary_contains_risk(self):
        rows = _make_concentrated_rows()
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        assert "Concentration" in result.summary
        assert result.concentration_risk in result.summary

    def test_aggregates_multiple_transactions(self):
        rows = [
            {"customer": "A", "amount": 100},
            {"customer": "A", "amount": 200},
            {"customer": "B", "amount": 50},
        ]
        result = compute_customer_concentration(rows, "customer", "amount")
        assert result is not None
        a_entry = next(c for c in result.top_customers if c.customer == "A")
        assert a_entry.total_spend == 300.0


# ---------------------------------------------------------------------------
# Tests: analyze_purchase_behavior
# ---------------------------------------------------------------------------


class TestAnalyzePurchaseBehavior:
    def test_basic_behavior(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        assert isinstance(result, BehaviorResult)

    def test_empty_rows_returns_none(self):
        assert analyze_purchase_behavior([], "customer", "amount", "date") is None

    def test_all_null_returns_none(self):
        rows = [{"customer": None, "amount": None, "date": None}]
        assert analyze_purchase_behavior(rows, "customer", "amount", "date") is None

    def test_total_orders_correct(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        alice = next(c for c in result.customers if c.customer == "Alice")
        assert alice.total_orders == 4

    def test_total_spend_correct(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        alice = next(c for c in result.customers if c.customer == "Alice")
        assert alice.total_spend == 750.0

    def test_avg_order_value(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        alice = next(c for c in result.customers if c.customer == "Alice")
        assert alice.avg_order_value == round(750 / 4, 2)

    def test_first_last_purchase(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        alice = next(c for c in result.customers if c.customer == "Alice")
        assert alice.first_purchase == "2024-01-01"
        assert alice.last_purchase == "2024-06-01"

    def test_lifespan_days(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        alice = next(c for c in result.customers if c.customer == "Alice")
        # 2024-06-01 - 2024-01-01 = 152 days
        expected = (datetime(2024, 6, 1) - datetime(2024, 1, 1)).days
        assert alice.lifespan_days == expected

    def test_single_order_lifespan_zero(self):
        rows = [{"customer": "Solo", "amount": 100, "date": "2024-01-01"}]
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        solo = result.customers[0]
        assert solo.lifespan_days == 0

    def test_repeat_purchase_rate(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        # Alice: 4 orders, Bob: 1, Charlie: 2 -> 2/3 repeat = 66.67%
        assert abs(result.repeat_purchase_rate - 66.67) < 0.1

    def test_all_single_orders_zero_repeat(self):
        rows = [
            {"customer": "A", "amount": 100, "date": "2024-01-01"},
            {"customer": "B", "amount": 200, "date": "2024-01-02"},
        ]
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        assert result.repeat_purchase_rate == 0.0

    def test_all_repeat_100_pct(self):
        rows = [
            {"customer": "A", "amount": 100, "date": "2024-01-01"},
            {"customer": "A", "amount": 200, "date": "2024-01-02"},
            {"customer": "B", "amount": 50, "date": "2024-01-01"},
            {"customer": "B", "amount": 60, "date": "2024-01-03"},
        ]
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        assert result.repeat_purchase_rate == 100.0

    def test_avg_orders(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        # Alice:4, Bob:1, Charlie:2 -> avg = 7/3
        expected = round(7 / 3, 2)
        assert result.avg_orders == expected

    def test_with_product_column(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(
            rows, "customer", "amount", "date", product_column="product"
        )
        assert result is not None
        assert "basket" in result.summary.lower()

    def test_without_product_column(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        assert "basket" not in result.summary.lower()

    def test_customers_sorted_alphabetically(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        names = [c.customer for c in result.customers]
        assert names == sorted(names)

    def test_summary_present(self):
        rows = _make_behavior_rows()
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        assert "Purchase Behavior" in result.summary

    def test_zero_amount_orders(self):
        rows = [
            {"customer": "A", "amount": 0, "date": "2024-01-01"},
            {"customer": "A", "amount": 0, "date": "2024-01-02"},
        ]
        result = analyze_purchase_behavior(rows, "customer", "amount", "date")
        assert result is not None
        assert result.customers[0].total_spend == 0.0
        assert result.customers[0].avg_order_value == 0.0


# ---------------------------------------------------------------------------
# Tests: format_customer_report
# ---------------------------------------------------------------------------


class TestFormatCustomerReport:
    def test_no_data(self):
        report = format_customer_report()
        assert "No data" in report

    def test_segments_section(self):
        rows = _make_segment_rows()
        seg = analyze_customer_segments(rows, "customer", "revenue")
        report = format_customer_report(segments=seg)
        assert "Customer Segmentation Report" in report
        assert "Premium" in report

    def test_churn_section(self):
        rows = _make_transaction_rows()
        churn = analyze_churn_risk(rows, "customer", "date")
        report = format_customer_report(churn=churn)
        assert "Churn Risk Analysis" in report

    def test_concentration_section(self):
        rows = _make_concentrated_rows()
        conc = compute_customer_concentration(rows, "customer", "amount")
        report = format_customer_report(concentration=conc)
        assert "Revenue Concentration Analysis" in report
        assert "HHI" in report

    def test_behavior_section(self):
        rows = _make_behavior_rows()
        beh = analyze_purchase_behavior(rows, "customer", "amount", "date")
        report = format_customer_report(behavior=beh)
        assert "Purchase Behavior Analysis" in report

    def test_all_sections(self):
        seg_rows = _make_segment_rows()
        txn_rows = _make_transaction_rows()
        conc_rows = _make_concentrated_rows()
        beh_rows = _make_behavior_rows()

        seg = analyze_customer_segments(seg_rows, "customer", "revenue")
        churn = analyze_churn_risk(txn_rows, "customer", "date")
        conc = compute_customer_concentration(conc_rows, "customer", "amount")
        beh = analyze_purchase_behavior(beh_rows, "customer", "amount", "date")

        report = format_customer_report(
            segments=seg, churn=churn, concentration=conc, behavior=beh
        )
        assert "Customer Segmentation Report" in report
        assert "Churn Risk Analysis" in report
        assert "Revenue Concentration Analysis" in report
        assert "Purchase Behavior Analysis" in report

    def test_report_contains_customer_names(self):
        rows = _make_concentrated_rows()
        conc = compute_customer_concentration(rows, "customer", "amount")
        report = format_customer_report(concentration=conc)
        assert "BigCo" in report

    def test_churn_with_spend(self):
        rows = _make_transaction_rows()
        churn = analyze_churn_risk(rows, "customer", "date", amount_column="amount")
        report = format_customer_report(churn=churn)
        assert "avg spend" in report

    def test_segments_with_frequency(self):
        rows = _make_segment_rows()
        seg = analyze_customer_segments(
            rows, "customer", "revenue", frequency_column="frequency"
        )
        report = format_customer_report(segments=seg)
        assert "avg freq" in report
