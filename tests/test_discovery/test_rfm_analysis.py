"""Tests for RFM analysis module."""

from datetime import datetime

from business_brain.discovery.rfm_analysis import (
    CLVResult,
    CustomerCLV,
    CustomerRFM,
    RFMResult,
    compute_rfm,
    customer_lifetime_value,
    format_rfm_report,
    segment_customers,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_rows(n_customers=10, txns_per_customer=5):
    """Generate sample transaction rows for *n_customers* with varied dates/amounts."""
    rows = []
    base = datetime(2024, 1, 1)
    for i in range(n_customers):
        for j in range(txns_per_customer):
            rows.append({
                "customer": f"cust_{i:02d}",
                "date": (base + __import__("datetime").timedelta(days=i * 10 + j * 3)).strftime("%Y-%m-%d"),
                "amount": round(50 + i * 20 + j * 5, 2),
            })
    return rows


def _make_diverse_rows():
    """Build a dataset with clearly separable RFM characteristics."""
    return [
        # Champions: recent, frequent, high spend
        {"customer": "Alice", "date": "2024-06-28", "amount": 500},
        {"customer": "Alice", "date": "2024-06-25", "amount": 600},
        {"customer": "Alice", "date": "2024-06-20", "amount": 550},
        {"customer": "Alice", "date": "2024-06-15", "amount": 700},
        {"customer": "Alice", "date": "2024-06-10", "amount": 650},
        # New customer: single recent purchase
        {"customer": "Bob", "date": "2024-06-29", "amount": 100},
        # Lost: old single purchase
        {"customer": "Charlie", "date": "2024-01-01", "amount": 30},
        # At Risk: old but was frequent
        {"customer": "Diana", "date": "2024-02-01", "amount": 200},
        {"customer": "Diana", "date": "2024-01-15", "amount": 180},
        {"customer": "Diana", "date": "2024-01-10", "amount": 190},
        {"customer": "Diana", "date": "2024-01-05", "amount": 210},
        # Moderate customer
        {"customer": "Eve", "date": "2024-04-15", "amount": 120},
        {"customer": "Eve", "date": "2024-03-10", "amount": 130},
        # Another moderate
        {"customer": "Frank", "date": "2024-05-20", "amount": 300},
        {"customer": "Frank", "date": "2024-04-20", "amount": 310},
        {"customer": "Frank", "date": "2024-03-20", "amount": 290},
        # Low value recent
        {"customer": "Grace", "date": "2024-06-27", "amount": 10},
        {"customer": "Grace", "date": "2024-06-22", "amount": 15},
        # Old moderate
        {"customer": "Hank", "date": "2024-02-15", "amount": 400},
        {"customer": "Hank", "date": "2024-02-01", "amount": 350},
        {"customer": "Hank", "date": "2024-01-20", "amount": 380},
        # Another filler
        {"customer": "Iris", "date": "2024-05-01", "amount": 90},
        {"customer": "Iris", "date": "2024-04-01", "amount": 85},
        # One more
        {"customer": "Jack", "date": "2024-06-01", "amount": 250},
        {"customer": "Jack", "date": "2024-05-01", "amount": 240},
        {"customer": "Jack", "date": "2024-04-01", "amount": 260},
        {"customer": "Jack", "date": "2024-03-01", "amount": 255},
    ]


# ---------------------------------------------------------------------------
# Tests: compute_rfm
# ---------------------------------------------------------------------------


class TestComputeRFM:
    def test_basic_computation(self):
        rows = _make_rows()
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        assert isinstance(result, RFMResult)
        assert result.total_customers == 10

    def test_empty_rows_returns_none(self):
        assert compute_rfm([], "customer", "date", "amount") is None

    def test_single_customer_returns_none(self):
        rows = [{"customer": "A", "date": "2024-01-01", "amount": 100}]
        assert compute_rfm(rows, "customer", "date", "amount") is None

    def test_scores_are_1_to_5(self):
        rows = _make_rows(n_customers=20)
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        for c in result.customers:
            assert 1 <= c.r_score <= 5, f"r_score {c.r_score} out of range"
            assert 1 <= c.f_score <= 5, f"f_score {c.f_score} out of range"
            assert 1 <= c.m_score <= 5, f"m_score {c.m_score} out of range"

    def test_rfm_score_is_weighted_average(self):
        rows = _make_rows(n_customers=10)
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        for c in result.customers:
            expected = round((c.r_score + c.f_score + c.m_score) / 3.0, 2)
            assert c.rfm_score == expected

    def test_recency_is_positive_days(self):
        rows = _make_rows()
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        for c in result.customers:
            assert c.recency_days >= 0

    def test_frequency_matches_transaction_count(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": 10},
            {"customer": "A", "date": "2024-01-05", "amount": 20},
            {"customer": "A", "date": "2024-01-10", "amount": 30},
            {"customer": "B", "date": "2024-01-01", "amount": 50},
            {"customer": "B", "date": "2024-01-15", "amount": 60},
        ]
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        cust_b = next(c for c in result.customers if c.customer == "B")
        assert cust_a.frequency == 3
        assert cust_b.frequency == 2

    def test_monetary_is_total_spend(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": 100},
            {"customer": "A", "date": "2024-01-05", "amount": 200},
            {"customer": "B", "date": "2024-01-01", "amount": 50},
        ]
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        assert cust_a.monetary == 300.0

    def test_custom_reference_date(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": 100},
            {"customer": "B", "date": "2024-01-15", "amount": 200},
        ]
        ref = datetime(2024, 2, 1)
        result = compute_rfm(rows, "customer", "date", "amount", reference_date=ref)
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        assert cust_a.recency_days == 31  # 2024-02-01 - 2024-01-01

    def test_null_values_skipped(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": 100},
            {"customer": None, "date": "2024-01-01", "amount": 50},
            {"customer": "B", "date": None, "amount": 50},
            {"customer": "C", "date": "2024-01-01", "amount": None},
            {"customer": "D", "date": "2024-01-10", "amount": 200},
        ]
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        names = {c.customer for c in result.customers}
        assert "A" in names
        assert "D" in names
        assert result.total_customers == 2

    def test_segment_distribution_counts_match(self):
        rows = _make_diverse_rows()
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        total_from_segments = sum(result.segment_distribution.values())
        assert total_from_segments == result.total_customers

    def test_all_customers_have_segment(self):
        rows = _make_diverse_rows()
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        for c in result.customers:
            assert c.segment != ""

    def test_summary_contains_key_info(self):
        rows = _make_rows()
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        assert "RFM" in result.summary
        assert str(result.total_customers) in result.summary

    def test_avg_metrics_computed(self):
        rows = _make_rows()
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        assert result.avg_recency > 0
        assert result.avg_frequency > 0
        assert result.avg_monetary > 0

    def test_string_amounts_parsed(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": "100.50"},
            {"customer": "A", "date": "2024-01-10", "amount": "200"},
            {"customer": "B", "date": "2024-01-05", "amount": "75"},
        ]
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        assert cust_a.monetary == 300.50

    def test_non_parseable_amount_skipped(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": 100},
            {"customer": "A", "date": "2024-01-05", "amount": "not_a_number"},
            {"customer": "B", "date": "2024-01-01", "amount": 200},
        ]
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        assert cust_a.frequency == 1  # only one valid transaction

    def test_date_format_slash(self):
        rows = [
            {"customer": "A", "date": "01/15/2024", "amount": 100},
            {"customer": "B", "date": "01/20/2024", "amount": 200},
        ]
        result = compute_rfm(rows, "customer", "date", "amount")
        assert result is not None
        assert result.total_customers == 2


# ---------------------------------------------------------------------------
# Tests: segment_customers
# ---------------------------------------------------------------------------


class TestSegmentCustomers:
    def test_basic_segmentation(self):
        rows = _make_diverse_rows()
        rfm = compute_rfm(rows, "customer", "date", "amount")
        assert rfm is not None
        segs = segment_customers(rfm)
        assert isinstance(segs, dict)
        assert len(segs) > 0

    def test_all_customers_accounted_for(self):
        rows = _make_diverse_rows()
        rfm = compute_rfm(rows, "customer", "date", "amount")
        assert rfm is not None
        segs = segment_customers(rfm)
        total = sum(info["count"] for info in segs.values())
        assert total == rfm.total_customers

    def test_segment_has_customers_and_count(self):
        rows = _make_diverse_rows()
        rfm = compute_rfm(rows, "customer", "date", "amount")
        assert rfm is not None
        segs = segment_customers(rfm)
        for seg_name, info in segs.items():
            assert "customers" in info
            assert "count" in info
            assert info["count"] == len(info["customers"])

    def test_empty_rfm_returns_empty(self):
        result = segment_customers(RFMResult(
            customers=[], segment_distribution={}, total_customers=0,
            avg_recency=0, avg_frequency=0, avg_monetary=0, summary="",
        ))
        assert result == {}

    def test_customer_names_sorted(self):
        rows = _make_diverse_rows()
        rfm = compute_rfm(rows, "customer", "date", "amount")
        assert rfm is not None
        segs = segment_customers(rfm)
        for info in segs.values():
            assert info["customers"] == sorted(info["customers"])


# ---------------------------------------------------------------------------
# Tests: customer_lifetime_value
# ---------------------------------------------------------------------------


class TestCustomerLifetimeValue:
    def test_basic_clv(self):
        rows = _make_diverse_rows()
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        assert isinstance(result, CLVResult)
        assert len(result.customers) > 0

    def test_empty_rows_returns_none(self):
        assert customer_lifetime_value([], "customer", "amount", "date") is None

    def test_all_null_returns_none(self):
        rows = [
            {"customer": None, "amount": None, "date": None},
        ]
        assert customer_lifetime_value(rows, "customer", "amount", "date") is None

    def test_total_spent_correct(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": 100},
            {"customer": "A", "date": "2024-06-01", "amount": 200},
            {"customer": "B", "date": "2024-03-01", "amount": 50},
        ]
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        assert cust_a.total_spent == 300.0
        assert cust_a.purchase_count == 2
        assert cust_a.avg_purchase == 150.0

    def test_lifespan_days(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": 100},
            {"customer": "A", "date": "2024-04-01", "amount": 200},
        ]
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        # 2024-04-01 - 2024-01-01 = 91 days
        assert cust_a.lifespan_days == 91

    def test_single_purchase_lifespan_zero(self):
        rows = [
            {"customer": "A", "date": "2024-01-01", "amount": 100},
            {"customer": "B", "date": "2024-01-01", "amount": 200},
        ]
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        assert cust_a.lifespan_days == 0

    def test_top_customers_by_clv(self):
        rows = _make_diverse_rows()
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        assert len(result.top_customers) > 0
        assert len(result.top_customers) <= 5

    def test_avg_clv_computed(self):
        rows = _make_diverse_rows()
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        assert result.avg_clv > 0

    def test_clv_summary_present(self):
        rows = _make_diverse_rows()
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        assert "CLV" in result.summary

    def test_first_last_purchase_dates(self):
        rows = [
            {"customer": "A", "date": "2024-01-15", "amount": 100},
            {"customer": "A", "date": "2024-06-20", "amount": 200},
        ]
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        cust_a = next(c for c in result.customers if c.customer == "A")
        assert cust_a.first_purchase == "2024-01-15"
        assert cust_a.last_purchase == "2024-06-20"

    def test_estimated_clv_nonnegative(self):
        rows = _make_diverse_rows()
        result = customer_lifetime_value(rows, "customer", "amount", "date")
        assert result is not None
        for c in result.customers:
            assert c.estimated_clv >= 0


# ---------------------------------------------------------------------------
# Tests: format_rfm_report
# ---------------------------------------------------------------------------


class TestFormatRFMReport:
    def test_all_sections(self):
        rows = _make_diverse_rows()
        rfm = compute_rfm(rows, "customer", "date", "amount")
        segs = segment_customers(rfm)
        clv = customer_lifetime_value(rows, "customer", "amount", "date")
        report = format_rfm_report(rfm=rfm, segments=segs, clv=clv)
        assert "RFM Analysis Report" in report
        assert "Customer Segments" in report
        assert "Customer Lifetime Value" in report

    def test_rfm_only(self):
        rows = _make_diverse_rows()
        rfm = compute_rfm(rows, "customer", "date", "amount")
        report = format_rfm_report(rfm=rfm)
        assert "RFM Analysis Report" in report
        assert "Customer Segments" not in report

    def test_clv_only(self):
        rows = _make_diverse_rows()
        clv = customer_lifetime_value(rows, "customer", "amount", "date")
        report = format_rfm_report(clv=clv)
        assert "Customer Lifetime Value" in report
        assert "RFM Analysis Report" not in report

    def test_no_data(self):
        report = format_rfm_report()
        assert "No data" in report

    def test_segments_only(self):
        rows = _make_diverse_rows()
        rfm = compute_rfm(rows, "customer", "date", "amount")
        segs = segment_customers(rfm)
        report = format_rfm_report(segments=segs)
        assert "Customer Segments" in report

    def test_report_contains_customer_names(self):
        rows = _make_diverse_rows()
        rfm = compute_rfm(rows, "customer", "date", "amount")
        report = format_rfm_report(rfm=rfm)
        assert "Alice" in report
