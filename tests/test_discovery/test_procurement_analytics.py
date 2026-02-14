"""Tests for procurement_analytics module."""

from __future__ import annotations

from datetime import datetime

import pytest

from business_brain.discovery.procurement_analytics import (
    ItemVariance,
    MonthlyPOTrend,
    PriceVarianceResult,
    PurchaseOrderResult,
    SpendCategory,
    SpendCategoryResult,
    VendorDelivery,
    VendorOrderSummary,
    VendorPerfResult,
    _parse_date,
    _safe_float,
    analyze_purchase_orders,
    analyze_spend_by_category,
    analyze_vendor_performance,
    compute_purchase_price_variance,
    format_procurement_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_str_numeric(self):
        assert _safe_float("100.5") == 100.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_non_numeric_str(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_bool(self):
        # bool is subclass of int in Python
        assert _safe_float(True) == 1.0

    def test_negative(self):
        assert _safe_float("-10.5") == -10.5

    def test_zero(self):
        assert _safe_float(0) == 0.0


class TestParseDate:
    def test_datetime_object(self):
        dt = datetime(2024, 6, 15)
        assert _parse_date(dt) == dt

    def test_iso_string(self):
        result = _parse_date("2024-06-15")
        assert result == datetime(2024, 6, 15)

    def test_iso_with_time(self):
        result = _parse_date("2024-06-15T10:30:00")
        assert result == datetime(2024, 6, 15, 10, 30, 0)

    def test_none(self):
        assert _parse_date(None) is None

    def test_invalid_string(self):
        assert _parse_date("not-a-date") is None

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_numeric_value(self):
        # numbers cannot be parsed as ISO date
        assert _parse_date(12345) is None


# ---------------------------------------------------------------------------
# 1. analyze_purchase_orders
# ---------------------------------------------------------------------------


def _po_rows():
    """Standard PO test data."""
    return [
        {"po": "PO-001", "vendor": "Acme", "amount": 1000, "date": "2024-01-15", "status": "closed"},
        {"po": "PO-002", "vendor": "Acme", "amount": 2000, "date": "2024-01-20", "status": "closed"},
        {"po": "PO-003", "vendor": "Beta", "amount": 3000, "date": "2024-02-10", "status": "open"},
        {"po": "PO-004", "vendor": "Gamma", "amount": 500, "date": "2024-02-15", "status": "cancelled"},
        {"po": "PO-005", "vendor": "Beta", "amount": 1500, "date": "2024-03-01", "status": "open"},
    ]


class TestAnalyzePurchaseOrders:
    def test_basic(self):
        result = analyze_purchase_orders(
            _po_rows(), "po", "vendor", "amount",
        )
        assert result is not None
        assert isinstance(result, PurchaseOrderResult)
        assert result.total_orders == 5
        assert result.total_value == 8000.0
        assert len(result.vendors) == 3

    def test_vendors_sorted_by_value(self):
        result = analyze_purchase_orders(_po_rows(), "po", "vendor", "amount")
        assert result is not None
        assert result.vendors[0].vendor == "Beta"
        assert result.vendors[0].total_value == 4500.0
        assert result.vendors[1].vendor == "Acme"
        assert result.vendors[1].total_value == 3000.0
        assert result.vendors[2].vendor == "Gamma"
        assert result.vendors[2].total_value == 500.0

    def test_vendor_order_count(self):
        result = analyze_purchase_orders(_po_rows(), "po", "vendor", "amount")
        assert result is not None
        acme = [v for v in result.vendors if v.vendor == "Acme"][0]
        assert acme.order_count == 2
        assert acme.avg_value == 1500.0

    def test_with_date_column(self):
        result = analyze_purchase_orders(
            _po_rows(), "po", "vendor", "amount", date_column="date",
        )
        assert result is not None
        assert len(result.monthly_trends) == 3
        assert result.monthly_trends[0].month == "2024-01"
        assert result.monthly_trends[0].order_count == 2
        assert result.monthly_trends[0].total_value == 3000.0

    def test_with_status_column(self):
        result = analyze_purchase_orders(
            _po_rows(), "po", "vendor", "amount", status_column="status",
        )
        assert result is not None
        assert result.status_breakdown["closed"] == 2
        assert result.status_breakdown["open"] == 2
        assert result.status_breakdown["cancelled"] == 1

    def test_empty_rows(self):
        assert analyze_purchase_orders([], "po", "vendor", "amount") is None

    def test_missing_columns(self):
        rows = [{"foo": "bar", "baz": 100}]
        result = analyze_purchase_orders(rows, "po", "vendor", "amount")
        assert result is None

    def test_non_numeric_amount(self):
        rows = [
            {"po": "PO-001", "vendor": "Acme", "amount": "not_a_number"},
        ]
        result = analyze_purchase_orders(rows, "po", "vendor", "amount")
        assert result is None

    def test_single_row(self):
        rows = [{"po": "PO-001", "vendor": "X", "amount": 100}]
        result = analyze_purchase_orders(rows, "po", "vendor", "amount")
        assert result is not None
        assert result.total_orders == 1
        assert len(result.vendors) == 1
        assert result.vendors[0].order_count == 1
        assert result.vendors[0].avg_value == 100.0

    def test_all_same_vendor(self):
        rows = [
            {"po": f"PO-{i}", "vendor": "OnlyVendor", "amount": 100}
            for i in range(10)
        ]
        result = analyze_purchase_orders(rows, "po", "vendor", "amount")
        assert result is not None
        assert len(result.vendors) == 1
        assert result.vendors[0].order_count == 10
        assert result.vendors[0].total_value == 1000.0

    def test_zero_amount(self):
        rows = [{"po": "PO-001", "vendor": "V", "amount": 0}]
        result = analyze_purchase_orders(rows, "po", "vendor", "amount")
        assert result is not None
        assert result.total_value == 0.0
        assert result.vendors[0].avg_value == 0.0

    def test_summary_contains_vendors(self):
        result = analyze_purchase_orders(_po_rows(), "po", "vendor", "amount")
        assert result is not None
        assert "Beta" in result.summary
        assert "Acme" in result.summary

    def test_no_monthly_trends_without_date(self):
        result = analyze_purchase_orders(_po_rows(), "po", "vendor", "amount")
        assert result is not None
        assert result.monthly_trends == []

    def test_no_status_without_column(self):
        result = analyze_purchase_orders(_po_rows(), "po", "vendor", "amount")
        assert result is not None
        assert result.status_breakdown == {}

    def test_vendor_summary_dataclass_fields(self):
        result = analyze_purchase_orders(_po_rows(), "po", "vendor", "amount")
        assert result is not None
        v = result.vendors[0]
        assert isinstance(v, VendorOrderSummary)
        assert hasattr(v, "vendor")
        assert hasattr(v, "order_count")
        assert hasattr(v, "total_value")
        assert hasattr(v, "avg_value")

    def test_monthly_trend_dataclass_fields(self):
        result = analyze_purchase_orders(
            _po_rows(), "po", "vendor", "amount", date_column="date",
        )
        assert result is not None
        mt = result.monthly_trends[0]
        assert isinstance(mt, MonthlyPOTrend)
        assert hasattr(mt, "month")
        assert hasattr(mt, "order_count")
        assert hasattr(mt, "total_value")


# ---------------------------------------------------------------------------
# 2. compute_purchase_price_variance
# ---------------------------------------------------------------------------


def _ppv_rows():
    """Standard PPV test data."""
    return [
        {"item": "Steel", "qty": 100, "actual_price": 10, "std_price": 9},
        {"item": "Steel", "qty": 200, "actual_price": 11, "std_price": 9},
        {"item": "Copper", "qty": 50, "actual_price": 8, "std_price": 10},
        {"item": "Copper", "qty": 150, "actual_price": 9, "std_price": 10},
    ]


class TestComputePurchasePriceVariance:
    def test_basic(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        assert isinstance(result, PriceVarianceResult)
        assert len(result.items) == 2

    def test_steel_unfavorable(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        steel = [i for i in result.items if i.item == "Steel"][0]
        # Steel PPV = (10-9)*100 + (11-9)*200 = 100 + 400 = 500 (unfavorable)
        assert steel.total_variance == 500.0
        assert steel.variance_type == "unfavorable"
        assert steel.quantity == 300.0

    def test_copper_favorable(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        copper = [i for i in result.items if i.item == "Copper"][0]
        # Copper PPV = (8-10)*50 + (9-10)*150 = -100 + -150 = -250 (favorable)
        assert copper.total_variance == -250.0
        assert copper.variance_type == "favorable"

    def test_total_variance(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        # Total = 500 + (-250) = 250
        assert result.total_variance == 250.0

    def test_favorable_unfavorable_counts(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        assert result.favorable_count == 1
        assert result.unfavorable_count == 1

    def test_sorted_by_absolute_variance(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        # Steel (500) before Copper (250 abs)
        assert result.items[0].item == "Steel"
        assert result.items[1].item == "Copper"

    def test_without_standard_price(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price",
        )
        assert result is not None
        assert result.total_variance == 0.0
        for it in result.items:
            assert it.standard_price is None
            assert it.total_variance is None
            assert it.variance_type is None

    def test_empty_rows(self):
        assert compute_purchase_price_variance([], "item", "qty", "price") is None

    def test_missing_columns(self):
        rows = [{"foo": "bar"}]
        result = compute_purchase_price_variance(rows, "item", "qty", "price")
        assert result is None

    def test_single_row(self):
        rows = [{"item": "X", "qty": 10, "actual_price": 5, "std_price": 4}]
        result = compute_purchase_price_variance(
            rows, "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        assert len(result.items) == 1
        assert result.items[0].total_variance == 10.0  # (5-4)*10

    def test_zero_variance(self):
        rows = [{"item": "X", "qty": 10, "actual_price": 5, "std_price": 5}]
        result = compute_purchase_price_variance(
            rows, "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        assert result.items[0].total_variance == 0.0
        assert result.items[0].variance_type == "neutral"

    def test_summary_content(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        assert "2 items" in result.summary
        assert "Favorable" in result.summary
        assert "Unfavorable" in result.summary

    def test_item_variance_dataclass_fields(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        it = result.items[0]
        assert isinstance(it, ItemVariance)
        assert hasattr(it, "item")
        assert hasattr(it, "quantity")
        assert hasattr(it, "avg_actual_price")
        assert hasattr(it, "standard_price")
        assert hasattr(it, "total_variance")
        assert hasattr(it, "variance_type")

    def test_avg_actual_price(self):
        result = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        assert result is not None
        steel = [i for i in result.items if i.item == "Steel"][0]
        # avg actual = (10*100 + 11*200) / 300 = 3200/300 ~ 10.6667
        assert abs(steel.avg_actual_price - 10.6667) < 0.01


# ---------------------------------------------------------------------------
# 3. analyze_vendor_performance
# ---------------------------------------------------------------------------


def _vendor_perf_rows():
    """Standard vendor performance test data."""
    return [
        {"vendor": "Acme", "delivery": "2024-01-10", "promised": "2024-01-10", "quality": 95, "qty": 100},
        {"vendor": "Acme", "delivery": "2024-02-15", "promised": "2024-02-10", "quality": 90, "qty": 200},
        {"vendor": "Acme", "delivery": "2024-03-05", "promised": "2024-03-10", "quality": 92, "qty": 150},
        {"vendor": "Beta", "delivery": "2024-01-20", "promised": "2024-01-15", "quality": 80, "qty": 300},
        {"vendor": "Beta", "delivery": "2024-02-25", "promised": "2024-02-20", "quality": 85, "qty": 250},
    ]


class TestAnalyzeVendorPerformance:
    def test_basic(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
            quality_column="quality", quantity_column="qty",
        )
        assert result is not None
        assert isinstance(result, VendorPerfResult)
        assert len(result.vendors) == 2

    def test_acme_performance(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
        )
        assert result is not None
        acme = [v for v in result.vendors if v.vendor == "Acme"][0]
        assert acme.total_deliveries == 3
        # on_time: 2024-01-10 <= 2024-01-10 (yes), 2024-02-15 > 2024-02-10 (no, 5d late),
        # 2024-03-05 <= 2024-03-10 (yes)
        assert acme.on_time_count == 2
        assert acme.late_count == 1
        assert abs(acme.on_time_pct - 66.67) < 0.01

    def test_beta_all_late(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
        )
        assert result is not None
        beta = [v for v in result.vendors if v.vendor == "Beta"][0]
        assert beta.total_deliveries == 2
        assert beta.on_time_count == 0
        assert beta.late_count == 2
        assert beta.on_time_pct == 0.0

    def test_sorted_by_on_time_pct(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
        )
        assert result is not None
        assert result.vendors[0].vendor == "Acme"
        assert result.vendors[1].vendor == "Beta"

    def test_avg_days_late(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
        )
        assert result is not None
        acme = [v for v in result.vendors if v.vendor == "Acme"][0]
        assert acme.avg_days_late == 5.0  # only 1 late delivery, 5 days
        beta = [v for v in result.vendors if v.vendor == "Beta"][0]
        assert beta.avg_days_late == 5.0  # (5 + 5) / 2

    def test_quality_column(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
            quality_column="quality",
        )
        assert result is not None
        acme = [v for v in result.vendors if v.vendor == "Acme"][0]
        # (95 + 90 + 92) / 3 = 92.3333
        assert acme.avg_quality is not None
        assert abs(acme.avg_quality - 92.3333) < 0.01

    def test_quantity_column(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
            quantity_column="qty",
        )
        assert result is not None
        acme = [v for v in result.vendors if v.vendor == "Acme"][0]
        assert acme.total_quantity == 450.0

    def test_overall_on_time(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
        )
        assert result is not None
        # 2 on-time out of 5
        assert result.overall_on_time_pct == 40.0

    def test_overall_avg_quality(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
            quality_column="quality",
        )
        assert result is not None
        # (95 + 90 + 92 + 80 + 85) / 5 = 88.4
        assert result.avg_quality is not None
        assert abs(result.avg_quality - 88.4) < 0.01

    def test_no_quality_column(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
        )
        assert result is not None
        assert result.avg_quality is None
        for v in result.vendors:
            assert v.avg_quality is None

    def test_no_quantity_column(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
        )
        assert result is not None
        for v in result.vendors:
            assert v.total_quantity is None

    def test_empty_rows(self):
        assert analyze_vendor_performance([], "vendor", "del", "prom") is None

    def test_missing_date_columns(self):
        rows = [{"vendor": "X", "foo": "bar"}]
        result = analyze_vendor_performance(rows, "vendor", "delivery", "promised")
        assert result is None

    def test_single_row_on_time(self):
        rows = [{"vendor": "V", "delivery": "2024-01-05", "promised": "2024-01-10"}]
        result = analyze_vendor_performance(rows, "vendor", "delivery", "promised")
        assert result is not None
        assert result.vendors[0].on_time_count == 1
        assert result.vendors[0].on_time_pct == 100.0

    def test_summary_content(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
            quality_column="quality",
        )
        assert result is not None
        assert "2 vendors" in result.summary
        assert "5 deliveries" in result.summary
        assert "40.0%" in result.summary
        assert "quality" in result.summary.lower()

    def test_vendor_delivery_dataclass_fields(self):
        result = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
            quality_column="quality", quantity_column="qty",
        )
        assert result is not None
        vd = result.vendors[0]
        assert isinstance(vd, VendorDelivery)
        assert hasattr(vd, "vendor")
        assert hasattr(vd, "total_deliveries")
        assert hasattr(vd, "on_time_count")
        assert hasattr(vd, "late_count")
        assert hasattr(vd, "on_time_pct")
        assert hasattr(vd, "avg_days_late")
        assert hasattr(vd, "avg_quality")
        assert hasattr(vd, "total_quantity")


# ---------------------------------------------------------------------------
# 4. analyze_spend_by_category
# ---------------------------------------------------------------------------


def _spend_rows():
    """Standard spend test data."""
    return [
        {"category": "Raw Materials", "amount": 5000, "vendor": "Acme"},
        {"category": "Raw Materials", "amount": 3000, "vendor": "Beta"},
        {"category": "Packaging", "amount": 2000, "vendor": "Gamma"},
        {"category": "Packaging", "amount": 1000, "vendor": "Gamma"},
        {"category": "Tools", "amount": 500, "vendor": "Delta"},
        {"category": "Services", "amount": 1500, "vendor": "Epsilon"},
    ]


class TestAnalyzeSpendByCategory:
    def test_basic(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        assert isinstance(result, SpendCategoryResult)
        assert len(result.categories) == 4

    def test_total_spend(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        assert result.total_spend == 13000.0

    def test_sorted_by_spend(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        assert result.categories[0].category == "Raw Materials"
        assert result.categories[0].total_spend == 8000.0
        assert result.categories[1].category == "Packaging"
        assert result.categories[1].total_spend == 3000.0

    def test_pct_of_total(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        raw = result.categories[0]
        # 8000 / 13000 * 100 ~ 61.54
        assert abs(raw.pct_of_total - 61.54) < 0.1

    def test_transaction_count(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        raw = [c for c in result.categories if c.category == "Raw Materials"][0]
        assert raw.transaction_count == 2
        pkg = [c for c in result.categories if c.category == "Packaging"][0]
        assert pkg.transaction_count == 2

    def test_vendor_count(self):
        result = analyze_spend_by_category(
            _spend_rows(), "category", "amount", vendor_column="vendor",
        )
        assert result is not None
        raw = [c for c in result.categories if c.category == "Raw Materials"][0]
        assert raw.vendor_count == 2  # Acme, Beta
        pkg = [c for c in result.categories if c.category == "Packaging"][0]
        assert pkg.vendor_count == 1  # Gamma only

    def test_no_vendor_column(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        for c in result.categories:
            assert c.vendor_count is None

    def test_hhi(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        # raw_pct ~61.54, pkg ~23.08, services ~11.54, tools ~3.85
        # HHI = 61.54^2 + 23.08^2 + 11.54^2 + 3.85^2
        # ~3787.2 + 532.7 + 133.2 + 14.8 = ~4467.9
        assert result.hhi > 4000
        assert result.concentration_risk == "high"

    def test_low_concentration(self):
        rows = [
            {"category": f"Cat{i}", "amount": 100}
            for i in range(20)
        ]
        result = analyze_spend_by_category(rows, "category", "amount")
        assert result is not None
        # Equal shares: 20 * (5%)^2 = 20 * 25 = 500
        assert result.hhi == 500.0
        assert result.concentration_risk == "low"

    def test_top_categories(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        assert len(result.top_categories) == 3
        assert result.top_categories[0] == "Raw Materials"

    def test_empty_rows(self):
        assert analyze_spend_by_category([], "category", "amount") is None

    def test_missing_columns(self):
        rows = [{"foo": "bar"}]
        result = analyze_spend_by_category(rows, "category", "amount")
        assert result is None

    def test_all_zero_amounts(self):
        rows = [{"category": "X", "amount": 0}]
        result = analyze_spend_by_category(rows, "category", "amount")
        # total_spend == 0, returns None
        assert result is None

    def test_single_category(self):
        rows = [{"category": "Only", "amount": 1000}]
        result = analyze_spend_by_category(rows, "category", "amount")
        assert result is not None
        assert len(result.categories) == 1
        assert result.categories[0].pct_of_total == 100.0
        # HHI = 10000
        assert result.hhi == 10000.0
        assert result.concentration_risk == "high"

    def test_summary_content(self):
        result = analyze_spend_by_category(_spend_rows(), "category", "amount")
        assert result is not None
        assert "4 categories" in result.summary
        assert "13,000" in result.summary
        assert "concentration" in result.summary.lower()

    def test_spend_category_dataclass_fields(self):
        result = analyze_spend_by_category(
            _spend_rows(), "category", "amount", vendor_column="vendor",
        )
        assert result is not None
        sc = result.categories[0]
        assert isinstance(sc, SpendCategory)
        assert hasattr(sc, "category")
        assert hasattr(sc, "total_spend")
        assert hasattr(sc, "pct_of_total")
        assert hasattr(sc, "transaction_count")
        assert hasattr(sc, "vendor_count")


# ---------------------------------------------------------------------------
# 5. format_procurement_report
# ---------------------------------------------------------------------------


class TestFormatProcurementReport:
    def test_no_data(self):
        report = format_procurement_report()
        assert "Procurement Analytics Report" in report
        assert "No analysis data provided" in report

    def test_with_purchase_orders(self):
        po = analyze_purchase_orders(
            _po_rows(), "po", "vendor", "amount",
            date_column="date", status_column="status",
        )
        report = format_procurement_report(purchase_orders=po)
        assert "Purchase Orders" in report
        assert "Total orders" in report
        assert "Top Vendors" in report
        assert "Monthly Trends" in report
        assert "Status Breakdown" in report

    def test_with_price_variance(self):
        pv = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        report = format_procurement_report(price_variance=pv)
        assert "Purchase Price Variance" in report
        assert "Total variance" in report
        assert "Items (by impact)" in report

    def test_with_vendor_perf(self):
        vp = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
            quality_column="quality", quantity_column="qty",
        )
        report = format_procurement_report(vendor_perf=vp)
        assert "Vendor Performance" in report
        assert "Overall on-time" in report
        assert "quality" in report.lower()

    def test_with_spend_category(self):
        sc = analyze_spend_by_category(
            _spend_rows(), "category", "amount", vendor_column="vendor",
        )
        report = format_procurement_report(spend_category=sc)
        assert "Spend by Category" in report
        assert "HHI" in report
        assert "Categories" in report

    def test_combined_report(self):
        po = analyze_purchase_orders(
            _po_rows(), "po", "vendor", "amount",
        )
        pv = compute_purchase_price_variance(
            _ppv_rows(), "item", "qty", "actual_price", "std_price",
        )
        vp = analyze_vendor_performance(
            _vendor_perf_rows(), "vendor", "delivery", "promised",
        )
        sc = analyze_spend_by_category(
            _spend_rows(), "category", "amount",
        )
        report = format_procurement_report(
            purchase_orders=po,
            price_variance=pv,
            vendor_perf=vp,
            spend_category=sc,
        )
        assert "Purchase Orders" in report
        assert "Purchase Price Variance" in report
        assert "Vendor Performance" in report
        assert "Spend by Category" in report
        assert "No analysis data provided" not in report

    def test_report_is_string(self):
        report = format_procurement_report()
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# Edge cases across multiple functions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_string_amounts(self):
        rows = [
            {"po": "PO-1", "vendor": "V", "amount": "1000.50"},
            {"po": "PO-2", "vendor": "V", "amount": "2000.75"},
        ]
        result = analyze_purchase_orders(rows, "po", "vendor", "amount")
        assert result is not None
        assert abs(result.total_value - 3001.25) < 0.01

    def test_mixed_valid_invalid_amounts(self):
        rows = [
            {"po": "PO-1", "vendor": "V", "amount": 1000},
            {"po": "PO-2", "vendor": "V", "amount": "bad"},
            {"po": "PO-3", "vendor": "V", "amount": 2000},
        ]
        result = analyze_purchase_orders(rows, "po", "vendor", "amount")
        assert result is not None
        assert result.total_orders == 2
        assert result.total_value == 3000.0

    def test_vendor_perf_same_date_on_time(self):
        rows = [
            {"vendor": "V", "delivery": "2024-01-01", "promised": "2024-01-01"},
        ]
        result = analyze_vendor_performance(rows, "vendor", "delivery", "promised")
        assert result is not None
        assert result.vendors[0].on_time_count == 1
        assert result.vendors[0].late_count == 0

    def test_vendor_perf_early_delivery(self):
        rows = [
            {"vendor": "V", "delivery": "2024-01-05", "promised": "2024-01-10"},
        ]
        result = analyze_vendor_performance(rows, "vendor", "delivery", "promised")
        assert result is not None
        assert result.vendors[0].on_time_count == 1
        assert result.vendors[0].avg_days_late == 0.0

    def test_ppv_non_numeric_quantity(self):
        rows = [{"item": "X", "qty": "lots", "actual_price": 10}]
        result = compute_purchase_price_variance(rows, "item", "qty", "actual_price")
        assert result is None

    def test_many_vendors_top5(self):
        rows = [
            {"po": f"PO-{i}", "vendor": f"V{i}", "amount": (10 - i) * 100}
            for i in range(10)
        ]
        result = analyze_purchase_orders(rows, "po", "vendor", "amount")
        assert result is not None
        assert len(result.vendors) == 10
        # Summary should mention top vendors
        assert "V0" in result.summary

    def test_spend_moderate_concentration(self):
        # 3 categories: 60%, 30%, 10% => HHI = 3600 + 900 + 100 = 4600
        # Actually let's aim for moderate: 40%, 35%, 25% => 1600 + 1225 + 625 = 3450 (high)
        # For moderate (1500-2500): 35%, 30%, 20%, 15% => 1225+900+400+225 = 2750 (high)
        # Let's try 25%, 25%, 25%, 25% => 4 * 625 = 2500 (high boundary)
        # 20%, 20%, 20%, 20%, 20% => 5 * 400 = 2000 (moderate)
        rows = [{"category": f"C{i}", "amount": 200} for i in range(5)]
        result = analyze_spend_by_category(rows, "category", "amount")
        assert result is not None
        assert result.hhi == 2000.0
        assert result.concentration_risk == "moderate"
