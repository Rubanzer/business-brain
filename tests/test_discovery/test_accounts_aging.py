"""Tests for accounts_aging module."""

from business_brain.discovery.accounts_aging import (
    AgingBucket,
    CustomerAging,
    ReceivablesAgingResult,
    VendorAging,
    PayablesAgingResult,
    PeriodDSO,
    DSOResult,
    CustomerCollection,
    CollectionResult,
    _safe_float,
    _parse_date,
    analyze_receivables_aging,
    analyze_payables_aging,
    compute_dso,
    analyze_collection_effectiveness,
    format_aging_report,
)

from datetime import datetime, date, timedelta


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("123.45") == 123.45

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_negative(self):
        assert _safe_float(-5.5) == -5.5

    def test_boolean(self):
        assert _safe_float(True) == 1.0

    def test_list_returns_none(self):
        assert _safe_float([1, 2]) is None


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_none(self):
        assert _parse_date(None) is None

    def test_datetime_passthrough(self):
        dt = datetime(2024, 1, 15)
        assert _parse_date(dt) == dt

    def test_date_object(self):
        d = date(2024, 6, 15)
        result = _parse_date(d)
        assert result == datetime(2024, 6, 15)

    def test_yyyy_mm_dd(self):
        result = _parse_date("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_yyyy_slash(self):
        result = _parse_date("2024/03/20")
        assert result == datetime(2024, 3, 20)

    def test_mm_dd_yyyy(self):
        result = _parse_date("03/20/2024")
        assert result == datetime(2024, 3, 20)

    def test_datetime_with_time(self):
        result = _parse_date("2024-01-15 10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30, 0)

    def test_invalid_string(self):
        assert _parse_date("not-a-date") is None

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_integer_returns_none(self):
        assert _parse_date(12345) is None


# ---------------------------------------------------------------------------
# analyze_receivables_aging
# ---------------------------------------------------------------------------


class TestAnalyzeReceivablesAging:
    def test_empty_rows(self):
        assert analyze_receivables_aging([], "customer", "amount", "date") is None

    def test_all_null_data(self):
        rows = [{"customer": None, "amount": None, "date": None}]
        assert analyze_receivables_aging(rows, "customer", "amount", "date") is None

    def test_all_invalid_data(self):
        rows = [{"customer": "Acme", "amount": "abc", "date": "not-a-date"}]
        assert analyze_receivables_aging(rows, "customer", "amount", "date") is None

    def test_missing_columns(self):
        rows = [{"x": 100, "y": "2024-01-15"}]
        assert analyze_receivables_aging(rows, "customer", "amount", "date") is None

    def test_missing_amount_column(self):
        rows = [{"customer": "Acme", "date": "2024-01-01"}]
        assert analyze_receivables_aging(rows, "customer", "amount", "date") is None

    def test_missing_date_column(self):
        rows = [{"customer": "Acme", "amount": 1000}]
        assert analyze_receivables_aging(rows, "customer", "amount", "date") is None

    def test_single_row_current(self):
        # Invoice due today (reference_date = due_date), 0 days past due => Current
        ref = datetime(2024, 2, 14)
        rows = [{"customer": "Acme", "amount": 1000, "inv_date": "2024-01-15", "due": "2024-02-14"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.total_outstanding == 1000.0
        assert result.total_overdue == 0.0
        assert result.buckets[0].count == 1  # Current bucket
        assert result.buckets[0].amount == 1000.0

    def test_single_row_overdue_31_60(self):
        ref = datetime(2024, 3, 20)
        rows = [{"customer": "Acme", "amount": 500, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        # days_past = (Mar 20 - Feb 1) = 48 days => 31-60 bucket
        assert result.buckets[1].count == 1
        assert result.buckets[1].amount == 500.0
        assert result.total_overdue == 500.0

    def test_single_row_overdue_61_90(self):
        ref = datetime(2024, 4, 15)
        rows = [{"customer": "Acme", "amount": 750, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # days_past = (Apr 15 - Feb 1) = 74 days => 61-90 bucket
        assert result.buckets[2].count == 1
        assert result.buckets[2].amount == 750.0

    def test_single_row_overdue_91_120(self):
        ref = datetime(2024, 5, 10)
        rows = [{"customer": "Acme", "amount": 300, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # days_past = (May 10 - Feb 1) = 99 days => 91-120 bucket
        assert result.buckets[3].count == 1

    def test_single_row_overdue_120_plus(self):
        ref = datetime(2024, 7, 1)
        rows = [{"customer": "Acme", "amount": 200, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # days_past = (Jul 1 - Feb 1) = 151 days => 120+ bucket
        assert result.buckets[4].count == 1
        assert result.buckets[4].amount == 200.0

    def test_no_due_date_defaults_to_invoice_plus_30(self):
        # invoice_date = Jan 1, no due_date => due = Jan 31
        # reference_date = Feb 15 => days_past = 15 => Current
        ref = datetime(2024, 2, 15)
        rows = [{"customer": "Acme", "amount": 1000, "inv_date": "2024-01-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            reference_date=ref,
        )
        assert result is not None
        assert result.buckets[0].count == 1  # Current

    def test_default_reference_date_uses_max_date(self):
        rows = [
            {"customer": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-01-31"},
            {"customer": "B", "amount": 200, "inv_date": "2024-03-01", "due": "2024-03-31"},
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date", due_date_column="due",
        )
        assert result is not None
        # Max due date is Mar 31; A is 60 days past due (31-60 bucket)
        assert result.total_outstanding == 300.0

    def test_multiple_customers(self):
        ref = datetime(2024, 3, 1)
        rows = [
            {"customer": "Acme", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-02-01"},
            {"customer": "Beta", "amount": 500, "inv_date": "2024-02-01", "due": "2024-03-01"},
            {"customer": "Acme", "amount": 200, "inv_date": "2024-02-15", "due": "2024-03-01"},
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert len(result.by_customer) == 2
        acme = next(c for c in result.by_customer if c.customer == "Acme")
        assert acme.total == 1200.0

    def test_customer_aging_breakdown(self):
        ref = datetime(2024, 4, 1)
        rows = [
            {"customer": "Acme", "amount": 100, "inv_date": "2024-01-01", "due": "2024-03-15"},  # 17 days past
            {"customer": "Acme", "amount": 200, "inv_date": "2024-01-01", "due": "2024-02-01"},  # 60 days past
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        acme = next(c for c in result.by_customer if c.customer == "Acme")
        assert acme.current == 100.0
        assert acme.days_31_60 == 200.0

    def test_worst_customers_ordering(self):
        ref = datetime(2024, 6, 1)
        rows = [
            {"customer": "A", "amount": 5000, "inv_date": "2024-01-01", "due": "2024-02-01"},
            {"customer": "B", "amount": 100, "inv_date": "2024-01-01", "due": "2024-02-01"},
            {"customer": "C", "amount": 3000, "inv_date": "2024-01-01", "due": "2024-02-01"},
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # A has highest overdue, then C, then B
        assert result.worst_customers[0] == "A"
        assert result.worst_customers[1] == "C"

    def test_worst_customers_max_five(self):
        ref = datetime(2024, 6, 1)
        rows = [
            {"customer": f"C{i}", "amount": 100 * i, "inv_date": "2024-01-01", "due": "2024-02-01"}
            for i in range(1, 8)
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert len(result.worst_customers) <= 5

    def test_pct_of_total(self):
        ref = datetime(2024, 3, 15)
        rows = [
            {"customer": "A", "amount": 750, "inv_date": "2024-01-01", "due": "2024-03-01"},  # 14 days past => current
            {"customer": "B", "amount": 250, "inv_date": "2024-01-01", "due": "2024-02-01"},  # 43 days past => 31-60
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result.buckets[0].pct_of_total == 75.0
        assert result.buckets[1].pct_of_total == 25.0

    def test_all_buckets_structure(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-03-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert len(result.buckets) == 5
        labels = [b.label for b in result.buckets]
        assert "Current (0-30)" in labels
        assert "31-60 days" in labels
        assert "61-90 days" in labels
        assert "91-120 days" in labels
        assert "120+ days" in labels

    def test_future_due_date_treated_as_current(self):
        ref = datetime(2024, 2, 1)
        rows = [{"customer": "A", "amount": 500, "inv_date": "2024-01-15", "due": "2024-03-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # Due date is in future, so days_past = 0 => Current
        assert result.buckets[0].count == 1
        assert result.total_overdue == 0.0

    def test_avg_days_outstanding(self):
        ref = datetime(2024, 3, 1)
        rows = [
            {"customer": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-02-01"},  # 29 days
            {"customer": "B", "amount": 100, "inv_date": "2024-01-01", "due": "2024-01-31"},  # 30 days
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # avg = (29 + 30) / 2 = 29.5
        assert result.avg_days_outstanding == 29.5

    def test_summary_contains_totals(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": "A", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert "1,000.00" in result.summary
        assert "Receivables aging" in result.summary

    def test_string_amount_values(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": "A", "amount": "500.50", "inv_date": "2024-01-01", "due": "2024-03-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.total_outstanding == 500.5

    def test_mixed_valid_invalid_rows(self):
        ref = datetime(2024, 3, 1)
        rows = [
            {"customer": "A", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-03-01"},
            {"customer": "B", "amount": "bad", "inv_date": "2024-01-01", "due": "2024-02-01"},
            {"customer": "C", "amount": 500, "inv_date": "not-a-date", "due": "2024-02-01"},
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.total_outstanding == 1000.0

    def test_invalid_due_date_falls_back_to_invoice_plus_30(self):
        ref = datetime(2024, 3, 15)
        rows = [{"customer": "A", "amount": 100, "inv_date": "2024-01-01", "due": "bad-date"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        # Fallback due = Jan 31. Days past = (Mar 15 - Jan 31) = 44 => 31-60 bucket
        assert result.buckets[1].count == 1

    def test_all_overdue(self):
        ref = datetime(2024, 6, 1)
        rows = [
            {"customer": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-02-01"},
            {"customer": "B", "amount": 200, "inv_date": "2024-01-01", "due": "2024-03-01"},
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result.total_overdue == 300.0
        assert result.buckets[0].count == 0

    def test_no_overdue(self):
        ref = datetime(2024, 2, 1)
        rows = [
            {"customer": "A", "amount": 100, "inv_date": "2024-01-15", "due": "2024-02-15"},
            {"customer": "B", "amount": 200, "inv_date": "2024-01-20", "due": "2024-02-20"},
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result.total_overdue == 0.0
        assert result.buckets[0].count == 2

    def test_customers_sorted_alphabetically(self):
        ref = datetime(2024, 3, 1)
        rows = [
            {"customer": "Zeta", "amount": 100, "inv_date": "2024-01-01", "due": "2024-03-01"},
            {"customer": "Alpha", "amount": 200, "inv_date": "2024-01-01", "due": "2024-03-01"},
            {"customer": "Mid", "amount": 150, "inv_date": "2024-01-01", "due": "2024-03-01"},
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        names = [c.customer for c in result.by_customer]
        assert names == ["Alpha", "Mid", "Zeta"]


# ---------------------------------------------------------------------------
# analyze_payables_aging
# ---------------------------------------------------------------------------


class TestAnalyzePayablesAging:
    def test_empty_rows(self):
        assert analyze_payables_aging([], "vendor", "amount", "date") is None

    def test_all_null_data(self):
        rows = [{"vendor": None, "amount": None, "date": None}]
        assert analyze_payables_aging(rows, "vendor", "amount", "date") is None

    def test_all_invalid_data(self):
        rows = [{"vendor": "Sup", "amount": "bad", "date": "not-a-date"}]
        assert analyze_payables_aging(rows, "vendor", "amount", "date") is None

    def test_missing_columns(self):
        rows = [{"x": 100}]
        assert analyze_payables_aging(rows, "vendor", "amount", "date") is None

    def test_single_row_current(self):
        ref = datetime(2024, 2, 14)
        rows = [{"vendor": "SupA", "amount": 800, "inv_date": "2024-01-15", "due": "2024-02-14"}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.total_outstanding == 800.0
        assert result.total_overdue == 0.0
        assert result.buckets[0].count == 1

    def test_single_row_overdue(self):
        ref = datetime(2024, 4, 1)
        rows = [{"vendor": "SupA", "amount": 500, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # 60 days past => 31-60 bucket
        assert result.buckets[1].count == 1
        assert result.total_overdue == 500.0

    def test_no_due_date_defaults_to_invoice_plus_30(self):
        ref = datetime(2024, 2, 15)
        rows = [{"vendor": "SupA", "amount": 1000, "inv_date": "2024-01-01"}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            reference_date=ref,
        )
        assert result is not None
        # due = Jan 31, days_past = 15 => Current
        assert result.buckets[0].count == 1

    def test_multiple_vendors(self):
        ref = datetime(2024, 3, 1)
        rows = [
            {"vendor": "SupA", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-02-01"},
            {"vendor": "SupB", "amount": 500, "inv_date": "2024-02-01", "due": "2024-03-01"},
        ]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert len(result.by_vendor) == 2
        assert result.total_outstanding == 1500.0

    def test_vendor_aging_breakdown(self):
        ref = datetime(2024, 4, 1)
        rows = [
            {"vendor": "SupA", "amount": 300, "inv_date": "2024-01-01", "due": "2024-03-15"},  # 17 days
            {"vendor": "SupA", "amount": 700, "inv_date": "2024-01-01", "due": "2024-02-01"},  # 60 days
        ]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        sup = next(v for v in result.by_vendor if v.vendor == "SupA")
        assert sup.current == 300.0
        assert sup.days_31_60 == 700.0

    def test_all_five_buckets(self):
        ref = datetime(2024, 3, 1)
        rows = [{"vendor": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-03-01"}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert len(result.buckets) == 5

    def test_summary_content(self):
        ref = datetime(2024, 3, 1)
        rows = [{"vendor": "A", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert "Payables aging" in result.summary
        assert "1,000.00" in result.summary

    def test_string_amount(self):
        ref = datetime(2024, 3, 1)
        rows = [{"vendor": "A", "amount": "500", "inv_date": "2024-01-01", "due": "2024-03-01"}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.total_outstanding == 500.0

    def test_future_due_date(self):
        ref = datetime(2024, 2, 1)
        rows = [{"vendor": "A", "amount": 100, "inv_date": "2024-01-15", "due": "2024-03-01"}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result.buckets[0].count == 1
        assert result.total_overdue == 0.0

    def test_vendors_sorted_alphabetically(self):
        ref = datetime(2024, 3, 1)
        rows = [
            {"vendor": "Zeta", "amount": 100, "inv_date": "2024-01-01", "due": "2024-03-01"},
            {"vendor": "Alpha", "amount": 200, "inv_date": "2024-01-01", "due": "2024-03-01"},
        ]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        names = [v.vendor for v in result.by_vendor]
        assert names == ["Alpha", "Zeta"]

    def test_avg_days_outstanding(self):
        ref = datetime(2024, 3, 1)
        rows = [
            {"vendor": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-02-01"},  # 29 days
            {"vendor": "B", "amount": 100, "inv_date": "2024-01-01", "due": "2024-02-10"},  # 20 days
        ]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result.avg_days_outstanding == 24.5  # (29+20)/2

    def test_default_reference_date(self):
        rows = [
            {"vendor": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-02-01"},
            {"vendor": "B", "amount": 200, "inv_date": "2024-03-01", "due": "2024-04-01"},
        ]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due",
        )
        assert result is not None
        # reference_date = max(Feb 1, Apr 1) = Apr 1
        # A: 60 days past => 31-60 bucket
        # B: 0 days past => Current
        assert result.total_outstanding == 300.0


# ---------------------------------------------------------------------------
# compute_dso
# ---------------------------------------------------------------------------


class TestComputeDSO:
    def test_empty_rows(self):
        assert compute_dso([], "revenue", "receivables") is None

    def test_all_null_data(self):
        rows = [{"revenue": None, "receivables": None}]
        assert compute_dso(rows, "revenue", "receivables") is None

    def test_all_invalid_data(self):
        rows = [{"revenue": "abc", "receivables": "xyz"}]
        assert compute_dso(rows, "revenue", "receivables") is None

    def test_missing_columns(self):
        rows = [{"x": 100}]
        assert compute_dso(rows, "revenue", "receivables") is None

    def test_zero_revenue_returns_none(self):
        rows = [{"revenue": 0, "receivables": 1000}]
        assert compute_dso(rows, "revenue", "receivables") is None

    def test_single_row_no_date(self):
        rows = [{"revenue": 10000, "receivables": 5000}]
        result = compute_dso(rows, "revenue", "receivables")
        assert result is not None
        # avg_rec=5000, total_rev=10000, days=30
        # DSO = (5000/10000)*30 = 15
        assert result.overall_dso == 15.0
        assert result.benchmark_status == "excellent"

    def test_multiple_rows_no_date(self):
        rows = [
            {"revenue": 5000, "receivables": 2000},
            {"revenue": 5000, "receivables": 3000},
        ]
        result = compute_dso(rows, "revenue", "receivables")
        assert result is not None
        # avg_rec = (2000+3000)/2 = 2500, total_rev = 10000, days = 60
        # DSO = (2500/10000)*60 = 15
        assert result.overall_dso == 15.0

    def test_benchmark_excellent(self):
        rows = [{"revenue": 10000, "receivables": 2000}]
        result = compute_dso(rows, "revenue", "receivables")
        # DSO = (2000/10000)*30 = 6
        assert result.benchmark_status == "excellent"

    def test_benchmark_good(self):
        rows = [{"revenue": 1000, "receivables": 1200}]
        result = compute_dso(rows, "revenue", "receivables")
        # DSO = (1200/1000)*30 = 36
        assert result.benchmark_status == "good"

    def test_benchmark_fair(self):
        rows = [{"revenue": 1000, "receivables": 1800}]
        result = compute_dso(rows, "revenue", "receivables")
        # DSO = (1800/1000)*30 = 54
        assert result.benchmark_status == "fair"

    def test_benchmark_needs_attention(self):
        rows = [{"revenue": 1000, "receivables": 3000}]
        result = compute_dso(rows, "revenue", "receivables")
        # DSO = (3000/1000)*30 = 90
        assert result.benchmark_status == "needs attention"

    def test_with_date_column(self):
        rows = [
            {"revenue": 5000, "receivables": 1500, "date": "2024-01-15"},
            {"revenue": 6000, "receivables": 2000, "date": "2024-02-15"},
            {"revenue": 4000, "receivables": 1000, "date": "2024-03-15"},
        ]
        result = compute_dso(rows, "revenue", "receivables", date_column="date")
        assert result is not None
        assert len(result.periods) == 3

    def test_period_dso_computation(self):
        rows = [
            {"revenue": 10000, "receivables": 3000, "date": "2024-01-15"},
        ]
        result = compute_dso(rows, "revenue", "receivables", date_column="date")
        assert result is not None
        assert len(result.periods) == 1
        p = result.periods[0]
        assert p.period == "2024-01"
        # p_dso = (3000/10000)*30 = 9
        assert p.dso == 9.0

    def test_trend_stable(self):
        rows = [
            {"revenue": 10000, "receivables": 3000, "date": "2024-01-15"},
            {"revenue": 10000, "receivables": 3000, "date": "2024-02-15"},
            {"revenue": 10000, "receivables": 3000, "date": "2024-03-15"},
            {"revenue": 10000, "receivables": 3000, "date": "2024-04-15"},
        ]
        result = compute_dso(rows, "revenue", "receivables", date_column="date")
        assert result.trend == "stable"

    def test_trend_increasing(self):
        rows = [
            {"revenue": 10000, "receivables": 1000, "date": "2024-01-15"},
            {"revenue": 10000, "receivables": 1000, "date": "2024-02-15"},
            {"revenue": 10000, "receivables": 5000, "date": "2024-03-15"},
            {"revenue": 10000, "receivables": 5000, "date": "2024-04-15"},
        ]
        result = compute_dso(rows, "revenue", "receivables", date_column="date")
        assert result.trend == "increasing"

    def test_trend_decreasing(self):
        rows = [
            {"revenue": 10000, "receivables": 5000, "date": "2024-01-15"},
            {"revenue": 10000, "receivables": 5000, "date": "2024-02-15"},
            {"revenue": 10000, "receivables": 1000, "date": "2024-03-15"},
            {"revenue": 10000, "receivables": 1000, "date": "2024-04-15"},
        ]
        result = compute_dso(rows, "revenue", "receivables", date_column="date")
        assert result.trend == "decreasing"

    def test_trend_stable_without_dates(self):
        rows = [{"revenue": 10000, "receivables": 3000}]
        result = compute_dso(rows, "revenue", "receivables")
        assert result.trend == "stable"

    def test_string_numeric_values(self):
        rows = [{"revenue": "10000", "receivables": "3000"}]
        result = compute_dso(rows, "revenue", "receivables")
        assert result is not None
        assert result.overall_dso == 9.0

    def test_mixed_valid_invalid(self):
        rows = [
            {"revenue": 10000, "receivables": 3000},
            {"revenue": "bad", "receivables": 2000},
        ]
        result = compute_dso(rows, "revenue", "receivables")
        assert result is not None
        # Only first row valid
        assert result.overall_dso == 9.0

    def test_summary_content(self):
        rows = [{"revenue": 10000, "receivables": 3000}]
        result = compute_dso(rows, "revenue", "receivables")
        assert "DSO analysis" in result.summary
        assert "excellent" in result.summary

    def test_periods_sorted(self):
        rows = [
            {"revenue": 5000, "receivables": 1000, "date": "2024-03-15"},
            {"revenue": 5000, "receivables": 1000, "date": "2024-01-15"},
            {"revenue": 5000, "receivables": 1000, "date": "2024-02-15"},
        ]
        result = compute_dso(rows, "revenue", "receivables", date_column="date")
        period_names = [p.period for p in result.periods]
        assert period_names == ["2024-01", "2024-02", "2024-03"]

    def test_period_zero_revenue(self):
        rows = [
            {"revenue": 0, "receivables": 1000, "date": "2024-01-15"},
            {"revenue": 5000, "receivables": 2000, "date": "2024-02-15"},
        ]
        # Total revenue = 5000 (non-zero), so overall result is not None
        result = compute_dso(rows, "revenue", "receivables", date_column="date")
        assert result is not None
        # Jan period has 0 revenue => p_dso = 0
        jan = next(p for p in result.periods if p.period == "2024-01")
        assert jan.dso == 0.0


# ---------------------------------------------------------------------------
# analyze_collection_effectiveness
# ---------------------------------------------------------------------------


class TestAnalyzeCollectionEffectiveness:
    def test_empty_rows(self):
        assert analyze_collection_effectiveness([], "customer", "amount", "paid") is None

    def test_all_null_data(self):
        rows = [{"customer": None, "amount": None, "paid": None}]
        assert analyze_collection_effectiveness(rows, "customer", "amount", "paid") is None

    def test_all_invalid_data(self):
        rows = [{"customer": "A", "amount": "bad", "paid": "bad"}]
        assert analyze_collection_effectiveness(rows, "customer", "amount", "paid") is None

    def test_missing_columns(self):
        rows = [{"x": 100}]
        assert analyze_collection_effectiveness(rows, "customer", "amount", "paid") is None

    def test_missing_customer(self):
        rows = [{"amount": 1000, "paid": 500}]
        assert analyze_collection_effectiveness(rows, "customer", "amount", "paid") is None

    def test_zero_invoiced_returns_none(self):
        rows = [{"customer": "A", "amount": 0, "paid": 0}]
        assert analyze_collection_effectiveness(rows, "customer", "amount", "paid") is None

    def test_single_row_full_collection(self):
        rows = [{"customer": "Acme", "amount": 1000, "paid": 1000}]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result is not None
        assert result.overall_rate == 100.0
        assert result.total_outstanding == 0.0

    def test_single_row_partial_collection(self):
        rows = [{"customer": "Acme", "amount": 1000, "paid": 600}]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result is not None
        assert result.overall_rate == 60.0
        assert result.total_outstanding == 400.0

    def test_single_row_no_collection(self):
        rows = [{"customer": "Acme", "amount": 1000, "paid": 0}]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result is not None
        assert result.overall_rate == 0.0
        assert result.total_outstanding == 1000.0

    def test_multiple_customers(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 1000},
            {"customer": "B", "amount": 500, "paid": 200},
            {"customer": "C", "amount": 500, "paid": 300},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result is not None
        assert result.total_invoiced == 2000.0
        assert result.total_collected == 1500.0
        assert result.overall_rate == 75.0

    def test_customer_collection_breakdown(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 900},
            {"customer": "B", "amount": 500, "paid": 100},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        cust_a = next(c for c in result.by_customer if c.customer == "A")
        assert cust_a.collection_rate == 90.0
        assert cust_a.outstanding == 100.0

        cust_b = next(c for c in result.by_customer if c.customer == "B")
        assert cust_b.collection_rate == 20.0
        assert cust_b.outstanding == 400.0

    def test_best_collectors(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 1000},
            {"customer": "B", "amount": 1000, "paid": 900},
            {"customer": "C", "amount": 1000, "paid": 100},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert "A" in result.best_collectors
        assert result.best_collectors[0] == "A"

    def test_worst_collectors(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 1000},
            {"customer": "B", "amount": 1000, "paid": 500},
            {"customer": "C", "amount": 1000, "paid": 100},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert "C" in result.worst_collectors

    def test_all_fully_paid_no_worst(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 1000},
            {"customer": "B", "amount": 500, "paid": 500},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result.overall_rate == 100.0
        # All at 100% => worst_collectors should exclude those at 100%
        assert len(result.worst_collectors) == 0

    def test_aggregation_same_customer(self):
        rows = [
            {"customer": "A", "amount": 500, "paid": 400},
            {"customer": "A", "amount": 500, "paid": 500},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert len(result.by_customer) == 1
        cust = result.by_customer[0]
        assert cust.invoiced == 1000.0
        assert cust.collected == 900.0
        assert cust.collection_rate == 90.0

    def test_string_numeric_values(self):
        rows = [{"customer": "A", "amount": "1000", "paid": "800"}]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result is not None
        assert result.overall_rate == 80.0

    def test_mixed_valid_invalid_rows(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 800},
            {"customer": "B", "amount": "bad", "paid": 500},
            {"customer": "C", "amount": 500, "paid": 500},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result is not None
        assert result.total_invoiced == 1500.0
        assert result.total_collected == 1300.0

    def test_summary_content(self):
        rows = [{"customer": "A", "amount": 1000, "paid": 800}]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert "Collection analysis" in result.summary
        assert "80.0%" in result.summary

    def test_customers_sorted_alphabetically(self):
        rows = [
            {"customer": "Zeta", "amount": 100, "paid": 50},
            {"customer": "Alpha", "amount": 200, "paid": 100},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        names = [c.customer for c in result.by_customer]
        assert names == ["Alpha", "Zeta"]

    def test_overcollection(self):
        # Paid more than invoiced (credits, overpayment)
        rows = [{"customer": "A", "amount": 1000, "paid": 1200}]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result is not None
        assert result.overall_rate == 120.0
        assert result.total_outstanding == -200.0

    def test_best_worst_no_overlap(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 900},
            {"customer": "B", "amount": 1000, "paid": 500},
        ]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        for c in result.best_collectors:
            assert c not in result.worst_collectors

    def test_with_date_column(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 800, "date": "2024-01-15"},
        ]
        result = analyze_collection_effectiveness(
            rows, "customer", "amount", "paid", date_column="date"
        )
        assert result is not None
        assert result.overall_rate == 80.0


# ---------------------------------------------------------------------------
# format_aging_report
# ---------------------------------------------------------------------------


class TestFormatAgingReport:
    def test_no_data(self):
        report = format_aging_report()
        assert report == "No aging data available for report."

    def test_receivables_only(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": "A", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        rec = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        report = format_aging_report(receivables=rec)
        assert "Receivables Aging" in report
        assert "1,000.00" in report
        assert "Payables Aging" not in report

    def test_payables_only(self):
        ref = datetime(2024, 3, 1)
        rows = [{"vendor": "SupA", "amount": 800, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        pay = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        report = format_aging_report(payables=pay)
        assert "Payables Aging" in report
        assert "800.00" in report
        assert "Receivables Aging" not in report

    def test_dso_only(self):
        rows = [{"revenue": 10000, "receivables": 3000}]
        d = compute_dso(rows, "revenue", "receivables")
        report = format_aging_report(dso=d)
        assert "Days Sales Outstanding" in report
        assert "excellent" in report
        assert "Receivables Aging" not in report

    def test_collection_only(self):
        rows = [{"customer": "A", "amount": 1000, "paid": 800}]
        c = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        report = format_aging_report(collection=c)
        assert "Collection Effectiveness" in report
        assert "80.00%" in report
        assert "Receivables Aging" not in report

    def test_all_sections_combined(self):
        ref = datetime(2024, 3, 1)

        rec_rows = [{"customer": "A", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        rec = analyze_receivables_aging(
            rec_rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )

        pay_rows = [{"vendor": "SupA", "amount": 800, "inv_date": "2024-01-01", "due": "2024-02-01"}]
        pay = analyze_payables_aging(
            pay_rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )

        dso_rows = [{"revenue": 10000, "receivables": 3000}]
        d = compute_dso(dso_rows, "revenue", "receivables")

        col_rows = [{"customer": "A", "amount": 1000, "paid": 800}]
        c = analyze_collection_effectiveness(col_rows, "customer", "amount", "paid")

        report = format_aging_report(receivables=rec, payables=pay, dso=d, collection=c)
        assert "Receivables Aging" in report
        assert "Payables Aging" in report
        assert "Days Sales Outstanding" in report
        assert "Collection Effectiveness" in report

    def test_receivables_with_buckets_in_report(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": "A", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-03-01"}]
        rec = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        report = format_aging_report(receivables=rec)
        assert "Aging Buckets" in report
        assert "Current (0-30)" in report

    def test_receivables_customer_in_report(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": "Acme Corp", "amount": 1000, "inv_date": "2024-01-01", "due": "2024-03-01"}]
        rec = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        report = format_aging_report(receivables=rec)
        assert "By Customer" in report
        assert "Acme Corp" in report

    def test_payables_vendor_in_report(self):
        ref = datetime(2024, 3, 1)
        rows = [{"vendor": "Widget Inc", "amount": 500, "inv_date": "2024-01-01", "due": "2024-03-01"}]
        pay = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        report = format_aging_report(payables=pay)
        assert "By Vendor" in report
        assert "Widget Inc" in report

    def test_dso_with_periods_in_report(self):
        rows = [
            {"revenue": 5000, "receivables": 1500, "date": "2024-01-15"},
            {"revenue": 6000, "receivables": 2000, "date": "2024-02-15"},
        ]
        d = compute_dso(rows, "revenue", "receivables", date_column="date")
        report = format_aging_report(dso=d)
        assert "Period Details" in report
        assert "2024-01" in report
        assert "2024-02" in report

    def test_collection_best_worst_in_report(self):
        rows = [
            {"customer": "A", "amount": 1000, "paid": 1000},
            {"customer": "B", "amount": 1000, "paid": 100},
        ]
        c = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        report = format_aging_report(collection=c)
        assert "Best Collectors" in report
        assert "Worst Collectors" in report

    def test_receivables_worst_customers_in_report(self):
        ref = datetime(2024, 6, 1)
        rows = [
            {"customer": "BadPayer", "amount": 5000, "inv_date": "2024-01-01", "due": "2024-02-01"},
        ]
        rec = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        report = format_aging_report(receivables=rec)
        assert "Worst Customers" in report
        assert "BadPayer" in report


# ---------------------------------------------------------------------------
# Dataclass instantiation tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_aging_bucket(self):
        ab = AgingBucket(label="Current (0-30)", min_days=0, max_days=30, count=5, amount=5000.0, pct_of_total=50.0)
        assert ab.label == "Current (0-30)"
        assert ab.min_days == 0
        assert ab.max_days == 30
        assert ab.count == 5
        assert ab.amount == 5000.0
        assert ab.pct_of_total == 50.0

    def test_aging_bucket_none_max(self):
        ab = AgingBucket(label="120+ days", min_days=121, max_days=None, count=1, amount=100.0, pct_of_total=10.0)
        assert ab.max_days is None

    def test_customer_aging(self):
        ca = CustomerAging(
            customer="Acme", total=1000.0, current=500.0,
            days_31_60=200.0, days_61_90=150.0, days_91_120=100.0, over_120=50.0,
        )
        assert ca.customer == "Acme"
        assert ca.total == 1000.0
        assert ca.current == 500.0
        assert ca.days_31_60 == 200.0
        assert ca.days_61_90 == 150.0
        assert ca.days_91_120 == 100.0
        assert ca.over_120 == 50.0

    def test_receivables_aging_result(self):
        rar = ReceivablesAgingResult(
            buckets=[], by_customer=[], total_outstanding=1000.0,
            total_overdue=500.0, avg_days_outstanding=45.0,
            worst_customers=["Acme"], summary="test",
        )
        assert rar.total_outstanding == 1000.0
        assert rar.worst_customers == ["Acme"]

    def test_vendor_aging(self):
        va = VendorAging(
            vendor="SupA", total=800.0, current=300.0,
            days_31_60=200.0, days_61_90=150.0, days_91_120=100.0, over_120=50.0,
        )
        assert va.vendor == "SupA"
        assert va.total == 800.0

    def test_payables_aging_result(self):
        par = PayablesAgingResult(
            buckets=[], by_vendor=[], total_outstanding=800.0,
            total_overdue=300.0, avg_days_outstanding=35.0, summary="test",
        )
        assert par.total_outstanding == 800.0

    def test_period_dso(self):
        pd = PeriodDSO(period="2024-01", dso=25.0, revenue=10000.0, receivables=8333.0)
        assert pd.period == "2024-01"
        assert pd.dso == 25.0

    def test_dso_result(self):
        dr = DSOResult(
            overall_dso=30.0, periods=[], trend="stable",
            benchmark_status="good", summary="test",
        )
        assert dr.overall_dso == 30.0
        assert dr.benchmark_status == "good"

    def test_customer_collection(self):
        cc = CustomerCollection(
            customer="Acme", invoiced=1000.0, collected=800.0,
            collection_rate=80.0, outstanding=200.0,
        )
        assert cc.collection_rate == 80.0
        assert cc.outstanding == 200.0

    def test_collection_result(self):
        cr = CollectionResult(
            overall_rate=85.0, total_invoiced=10000.0,
            total_collected=8500.0, total_outstanding=1500.0,
            by_customer=[], best_collectors=["A"],
            worst_collectors=["C"], summary="test",
        )
        assert cr.overall_rate == 85.0
        assert cr.best_collectors == ["A"]
        assert cr.worst_collectors == ["C"]


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_receivables_with_date_object(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": "A", "amount": 100, "inv_date": date(2024, 1, 1), "due": date(2024, 2, 1)}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.total_outstanding == 100.0

    def test_payables_with_date_object(self):
        ref = datetime(2024, 3, 1)
        rows = [{"vendor": "A", "amount": 100, "inv_date": date(2024, 1, 1), "due": date(2024, 2, 1)}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.total_outstanding == 100.0

    def test_receivables_large_dataset(self):
        ref = datetime(2024, 6, 1)
        rows = [
            {"customer": f"C{i}", "amount": 100, "inv_date": "2024-01-01", "due": "2024-02-01"}
            for i in range(100)
        ]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.total_outstanding == 10000.0
        assert len(result.by_customer) == 100

    def test_collection_single_customer_is_best_and_not_worst(self):
        rows = [{"customer": "A", "amount": 1000, "paid": 1000}]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert "A" in result.best_collectors
        assert "A" not in result.worst_collectors

    def test_dso_multiple_rows_same_month(self):
        rows = [
            {"revenue": 3000, "receivables": 1000, "date": "2024-01-10"},
            {"revenue": 2000, "receivables": 2000, "date": "2024-01-20"},
        ]
        result = compute_dso(rows, "revenue", "receivables", date_column="date")
        assert result is not None
        assert len(result.periods) == 1
        p = result.periods[0]
        # Revenue = 5000, avg receivables = (1000+2000)/2 = 1500
        assert p.revenue == 5000.0
        assert p.receivables == 1500.0

    def test_receivables_boundary_30_days(self):
        ref = datetime(2024, 3, 2)
        rows = [{"customer": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-01-31"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # (Mar 2 - Jan 31) = 31 days => 31-60 bucket
        assert result.buckets[1].count == 1

    def test_receivables_exactly_30_days(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": "A", "amount": 100, "inv_date": "2024-01-01", "due": "2024-01-31"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        # (Mar 1 - Jan 31) = 30 days => Current bucket
        assert result.buckets[0].count == 1

    def test_format_report_with_none_sections(self):
        rows = [{"customer": "A", "amount": 1000, "paid": 800}]
        c = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        report = format_aging_report(receivables=None, payables=None, dso=None, collection=c)
        assert "Collection Effectiveness" in report
        assert "Receivables" not in report

    def test_receivables_numeric_customer_name(self):
        ref = datetime(2024, 3, 1)
        rows = [{"customer": 12345, "amount": 500, "inv_date": "2024-01-01", "due": "2024-03-01"}]
        result = analyze_receivables_aging(
            rows, "customer", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.by_customer[0].customer == "12345"

    def test_payables_numeric_vendor_name(self):
        ref = datetime(2024, 3, 1)
        rows = [{"vendor": 99, "amount": 500, "inv_date": "2024-01-01", "due": "2024-03-01"}]
        result = analyze_payables_aging(
            rows, "vendor", "amount", "inv_date",
            due_date_column="due", reference_date=ref,
        )
        assert result is not None
        assert result.by_vendor[0].vendor == "99"

    def test_collection_negative_paid(self):
        # Edge case: negative paid amounts (refunds)
        rows = [{"customer": "A", "amount": 1000, "paid": -100}]
        result = analyze_collection_effectiveness(rows, "customer", "amount", "paid")
        assert result is not None
        assert result.overall_rate == -10.0
        assert result.total_outstanding == 1100.0
