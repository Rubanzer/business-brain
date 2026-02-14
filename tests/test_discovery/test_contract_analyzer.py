"""Tests for contract_analyzer module."""

from datetime import datetime, timedelta

from business_brain.discovery.contract_analyzer import (
    ContractResult,
    ConcentrationResult,
    ExpiringContract,
    RenewalResult,
    VendorRenewal,
    VendorShare,
    VendorSummary,
    analyze_contracts,
    detect_expiring_contracts,
    vendor_contract_concentration,
    analyze_renewal_patterns,
    format_contract_report,
)


# ---------------------------------------------------------------------------
# Helper — sample data builders
# ---------------------------------------------------------------------------

def _base_rows():
    """Minimal valid rows with contract, vendor, value."""
    return [
        {"id": "C001", "vendor": "Acme", "value": 10000},
        {"id": "C002", "vendor": "Acme", "value": 15000},
        {"id": "C003", "vendor": "Globex", "value": 8000},
        {"id": "C004", "vendor": "Initech", "value": 12000},
        {"id": "C005", "vendor": "Initech", "value": 5000},
    ]


def _dated_rows():
    """Rows with start/end dates."""
    return [
        {"id": "C001", "vendor": "Acme", "value": 10000,
         "start": "2024-01-01", "end": "2024-12-31"},
        {"id": "C002", "vendor": "Acme", "value": 15000,
         "start": "2024-06-01", "end": "2025-05-31"},
        {"id": "C003", "vendor": "Globex", "value": 8000,
         "start": "2024-03-01", "end": "2025-02-28"},
        {"id": "C004", "vendor": "Initech", "value": 12000,
         "start": "2024-01-15", "end": "2024-07-14"},
        {"id": "C005", "vendor": "Initech", "value": 5000,
         "start": "2024-07-15", "end": "2025-01-14"},
    ]


def _status_rows():
    """Rows with status column."""
    return [
        {"id": "C001", "vendor": "Acme", "value": 10000, "status": "active"},
        {"id": "C002", "vendor": "Acme", "value": 15000, "status": "active"},
        {"id": "C003", "vendor": "Globex", "value": 8000, "status": "expired"},
        {"id": "C004", "vendor": "Initech", "value": 12000, "status": "pending"},
        {"id": "C005", "vendor": "Initech", "value": 5000, "status": "active"},
    ]


# ---------------------------------------------------------------------------
# 1. analyze_contracts
# ---------------------------------------------------------------------------


class TestAnalyzeContracts:

    def test_basic(self):
        result = analyze_contracts(_base_rows(), "id", "vendor", "value")
        assert result is not None
        assert result.total_contracts == 5
        assert result.total_value == 50000
        assert result.avg_value == 10000

    def test_vendor_aggregation(self):
        result = analyze_contracts(_base_rows(), "id", "vendor", "value")
        assert len(result.vendors) == 3
        # Sorted by total value descending
        assert result.vendors[0].vendor == "Acme"
        assert result.vendors[0].total_value == 25000
        assert result.vendors[0].contract_count == 2

    def test_vendor_avg_value(self):
        result = analyze_contracts(_base_rows(), "id", "vendor", "value")
        acme = result.vendors[0]
        assert acme.avg_value == 12500  # 25000 / 2

    def test_empty_rows(self):
        assert analyze_contracts([], "id", "vendor", "value") is None

    def test_missing_columns(self):
        rows = [{"id": "C001", "vendor": "X"}]  # no value
        assert analyze_contracts(rows, "id", "vendor", "value") is None

    def test_invalid_value(self):
        rows = [{"id": "C001", "vendor": "X", "value": "not_a_number"}]
        assert analyze_contracts(rows, "id", "vendor", "value") is None

    def test_status_counts(self):
        result = analyze_contracts(
            _status_rows(), "id", "vendor", "value", status_column="status"
        )
        assert result.status_counts is not None
        assert result.status_counts["active"] == 3
        assert result.status_counts["expired"] == 1
        assert result.status_counts["pending"] == 1

    def test_no_status_column(self):
        result = analyze_contracts(_base_rows(), "id", "vendor", "value")
        assert result.status_counts is None

    def test_avg_duration(self):
        result = analyze_contracts(
            _dated_rows(), "id", "vendor", "value",
            start_column="start", end_column="end",
        )
        assert result.avg_duration_days is not None
        assert result.avg_duration_days > 0

    def test_no_dates_no_duration(self):
        result = analyze_contracts(_base_rows(), "id", "vendor", "value")
        assert result.avg_duration_days is None

    def test_summary_non_empty(self):
        result = analyze_contracts(_base_rows(), "id", "vendor", "value")
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_partial_invalid_rows_skipped(self):
        rows = [
            {"id": "C001", "vendor": "Acme", "value": 10000},
            {"id": "C002", "vendor": "Acme", "value": "bad"},
            {"id": "C003", "vendor": None, "value": 5000},
        ]
        result = analyze_contracts(rows, "id", "vendor", "value")
        assert result is not None
        assert result.total_contracts == 1

    def test_single_contract(self):
        rows = [{"id": "C001", "vendor": "Solo", "value": 7500}]
        result = analyze_contracts(rows, "id", "vendor", "value")
        assert result.total_contracts == 1
        assert result.avg_value == 7500

    def test_datetime_objects_in_dates(self):
        rows = [{
            "id": "C001", "vendor": "X", "value": 1000,
            "start": datetime(2024, 1, 1), "end": datetime(2024, 12, 31),
        }]
        result = analyze_contracts(
            rows, "id", "vendor", "value",
            start_column="start", end_column="end",
        )
        assert result.avg_duration_days == 365.0


# ---------------------------------------------------------------------------
# 2. detect_expiring_contracts
# ---------------------------------------------------------------------------


class TestDetectExpiringContracts:

    def test_basic(self):
        ref = datetime(2025, 1, 1)
        rows = [
            {"id": "C1", "end": "2025-01-15"},
            {"id": "C2", "end": "2025-02-15"},
            {"id": "C3", "end": "2025-06-01"},  # beyond 90d
        ]
        result = detect_expiring_contracts(rows, "id", "end", reference_date=ref)
        assert len(result) == 2

    def test_urgency_critical(self):
        ref = datetime(2025, 1, 1)
        rows = [{"id": "C1", "end": "2025-01-20"}]
        result = detect_expiring_contracts(rows, "id", "end", reference_date=ref)
        assert result[0].urgency == "critical"
        assert result[0].days_until_expiry == 19

    def test_urgency_warning(self):
        ref = datetime(2025, 1, 1)
        rows = [{"id": "C1", "end": "2025-02-15"}]
        result = detect_expiring_contracts(rows, "id", "end", reference_date=ref)
        assert result[0].urgency == "warning"

    def test_urgency_upcoming(self):
        ref = datetime(2025, 1, 1)
        rows = [{"id": "C1", "end": "2025-03-15"}]
        result = detect_expiring_contracts(rows, "id", "end", reference_date=ref)
        assert result[0].urgency == "upcoming"

    def test_sorted_by_date_ascending(self):
        ref = datetime(2025, 1, 1)
        rows = [
            {"id": "C2", "end": "2025-03-01"},
            {"id": "C1", "end": "2025-01-10"},
            {"id": "C3", "end": "2025-02-01"},
        ]
        result = detect_expiring_contracts(rows, "id", "end", reference_date=ref)
        dates = [e.end_date for e in result]
        assert dates == sorted(dates)

    def test_default_reference_date_is_max(self):
        rows = [
            {"id": "C1", "end": "2025-01-01"},
            {"id": "C2", "end": "2025-02-01"},
            {"id": "C3", "end": "2025-03-01"},
        ]
        # default ref = max = 2025-03-01; nothing expires *after* max within horizon
        # Only C3 end == ref, which is 0 days, so it is "critical"
        result = detect_expiring_contracts(rows, "id", "end")
        # C3 expiry == ref date, days_until_expiry = 0, includes it
        assert any(e.contract == "C3" for e in result)

    def test_vendor_and_value_included(self):
        ref = datetime(2025, 1, 1)
        rows = [{"id": "C1", "vendor": "Acme", "value": 5000, "end": "2025-01-20"}]
        result = detect_expiring_contracts(
            rows, "id", "end",
            vendor_column="vendor", value_column="value",
            reference_date=ref,
        )
        assert result[0].vendor == "Acme"
        assert result[0].value == 5000

    def test_empty_rows(self):
        assert detect_expiring_contracts([], "id", "end") == []

    def test_no_expiring(self):
        ref = datetime(2025, 1, 1)
        rows = [{"id": "C1", "end": "2026-01-01"}]
        result = detect_expiring_contracts(rows, "id", "end", reference_date=ref)
        assert result == []

    def test_custom_horizon(self):
        ref = datetime(2025, 1, 1)
        rows = [{"id": "C1", "end": "2025-02-15"}]
        # 30-day horizon: 2025-01-31 is the cutoff; C1 at Feb 15 is outside
        result = detect_expiring_contracts(
            rows, "id", "end", reference_date=ref, horizon_days=30
        )
        assert len(result) == 0
        # 60-day horizon includes it
        result = detect_expiring_contracts(
            rows, "id", "end", reference_date=ref, horizon_days=60
        )
        assert len(result) == 1

    def test_invalid_dates_skipped(self):
        ref = datetime(2025, 1, 1)
        rows = [
            {"id": "C1", "end": "not-a-date"},
            {"id": "C2", "end": "2025-01-15"},
        ]
        result = detect_expiring_contracts(rows, "id", "end", reference_date=ref)
        assert len(result) == 1
        assert result[0].contract == "C2"

    def test_expiry_on_boundary(self):
        ref = datetime(2025, 1, 1)
        # Exactly 90 days out
        rows = [{"id": "C1", "end": "2025-04-01"}]
        result = detect_expiring_contracts(
            rows, "id", "end", reference_date=ref, horizon_days=90
        )
        assert len(result) == 1
        assert result[0].days_until_expiry == 90
        assert result[0].urgency == "upcoming"


# ---------------------------------------------------------------------------
# 3. vendor_contract_concentration
# ---------------------------------------------------------------------------


class TestVendorContractConcentration:

    def test_basic(self):
        rows = [
            {"vendor": "A", "value": 5000},
            {"vendor": "B", "value": 3000},
            {"vendor": "C", "value": 2000},
        ]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result is not None
        assert result.hhi > 0

    def test_hhi_monopoly(self):
        rows = [{"vendor": "Solo", "value": 100000}]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result.hhi == 10000.0  # 100^2
        assert result.risk_rating == "High"
        assert result.single_vendor_dependency is True

    def test_hhi_equal_split(self):
        # 4 vendors each 25% => HHI = 4 * 25^2 = 2500
        rows = [
            {"vendor": "A", "value": 250},
            {"vendor": "B", "value": 250},
            {"vendor": "C", "value": 250},
            {"vendor": "D", "value": 250},
        ]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result.hhi == 2500.0
        assert result.risk_rating == "Moderate"  # > 1500 but not > 2500

    def test_low_concentration(self):
        # 10 vendors each 10% => HHI = 10 * 100 = 1000
        rows = [{"vendor": f"V{i}", "value": 100} for i in range(10)]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result.hhi == 1000.0
        assert result.risk_rating == "Low"

    def test_dependency_flag(self):
        rows = [
            {"vendor": "Big", "value": 8000},
            {"vendor": "Small", "value": 2000},
        ]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result.single_vendor_dependency is True
        assert "Big" in result.dependency_vendors

    def test_no_dependency(self):
        rows = [
            {"vendor": "A", "value": 400},
            {"vendor": "B", "value": 300},
            {"vendor": "C", "value": 300},
        ]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result.single_vendor_dependency is False
        assert result.dependency_vendors == []

    def test_sorted_by_share_desc(self):
        rows = [
            {"vendor": "Small", "value": 100},
            {"vendor": "Big", "value": 900},
        ]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result.top_vendors[0].vendor == "Big"

    def test_empty_rows(self):
        assert vendor_contract_concentration([], "vendor", "value") is None

    def test_zero_values(self):
        rows = [{"vendor": "A", "value": 0}, {"vendor": "B", "value": 0}]
        assert vendor_contract_concentration(rows, "vendor", "value") is None

    def test_invalid_values_skipped(self):
        rows = [
            {"vendor": "A", "value": "bad"},
            {"vendor": "B", "value": 1000},
        ]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result is not None
        assert len(result.top_vendors) == 1

    def test_shares_sum_to_100(self):
        rows = [
            {"vendor": "A", "value": 5000},
            {"vendor": "B", "value": 3000},
            {"vendor": "C", "value": 2000},
        ]
        result = vendor_contract_concentration(rows, "vendor", "value")
        total_share = sum(v.share_pct for v in result.top_vendors)
        assert abs(total_share - 100.0) < 0.1

    def test_hhi_high_boundary(self):
        # HHI exactly at 2500 is Moderate (not > 2500)
        rows = [
            {"vendor": "A", "value": 250},
            {"vendor": "B", "value": 250},
            {"vendor": "C", "value": 250},
            {"vendor": "D", "value": 250},
        ]
        result = vendor_contract_concentration(rows, "vendor", "value")
        assert result.hhi == 2500.0
        assert result.risk_rating == "Moderate"


# ---------------------------------------------------------------------------
# 4. analyze_renewal_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeRenewalPatterns:

    def _renewal_rows(self):
        """Vendor Acme has consecutive contracts, Globex has one."""
        return [
            {"id": "C001", "vendor": "Acme", "value": 10000,
             "start": "2024-01-01", "end": "2024-06-30"},
            {"id": "C002", "vendor": "Acme", "value": 11000,
             "start": "2024-07-01", "end": "2024-12-31"},
            {"id": "C003", "vendor": "Globex", "value": 8000,
             "start": "2024-03-01", "end": "2025-02-28"},
        ]

    def test_basic(self):
        result = analyze_renewal_patterns(
            self._renewal_rows(), "id", "vendor", "start", "end",
        )
        assert result is not None
        assert result.total_vendors == 2

    def test_renewal_detected(self):
        result = analyze_renewal_patterns(
            self._renewal_rows(), "id", "vendor", "start", "end",
        )
        assert result.vendors_with_renewals == 1  # only Acme
        acme = [v for v in result.vendor_details if v.vendor == "Acme"][0]
        assert acme.has_renewals is True
        assert acme.renewal_count == 1

    def test_no_renewal_for_single_contract(self):
        result = analyze_renewal_patterns(
            self._renewal_rows(), "id", "vendor", "start", "end",
        )
        globex = [v for v in result.vendor_details if v.vendor == "Globex"][0]
        assert globex.has_renewals is False
        assert globex.renewal_count == 0

    def test_renewal_rate(self):
        result = analyze_renewal_patterns(
            self._renewal_rows(), "id", "vendor", "start", "end",
        )
        assert result.renewal_rate == 50.0  # 1 of 2 vendors

    def test_price_escalation(self):
        result = analyze_renewal_patterns(
            self._renewal_rows(), "id", "vendor", "start", "end",
            value_column="value",
        )
        assert result.avg_price_escalation_pct is not None
        # Acme: 10000 -> 11000 = +10%
        assert result.avg_price_escalation_pct == 10.0

    def test_no_value_column(self):
        result = analyze_renewal_patterns(
            self._renewal_rows(), "id", "vendor", "start", "end",
        )
        assert result.avg_price_escalation_pct is None

    def test_empty_rows(self):
        assert analyze_renewal_patterns([], "id", "vendor", "start", "end") is None

    def test_invalid_dates_skipped(self):
        rows = [
            {"id": "C1", "vendor": "X", "start": "bad", "end": "2025-01-01"},
            {"id": "C2", "vendor": "X", "start": "2024-01-01", "end": "2024-12-31"},
        ]
        result = analyze_renewal_patterns(rows, "id", "vendor", "start", "end")
        assert result is not None
        assert result.total_vendors == 1

    def test_gap_beyond_30_days_no_renewal(self):
        rows = [
            {"id": "C1", "vendor": "X",
             "start": "2024-01-01", "end": "2024-06-01"},
            {"id": "C2", "vendor": "X",
             "start": "2024-08-01", "end": "2024-12-31"},  # 61 day gap
        ]
        result = analyze_renewal_patterns(rows, "id", "vendor", "start", "end")
        x = [v for v in result.vendor_details if v.vendor == "X"][0]
        assert x.has_renewals is False

    def test_overlapping_contracts_counted_as_renewal(self):
        rows = [
            {"id": "C1", "vendor": "X",
             "start": "2024-01-01", "end": "2024-07-15"},
            {"id": "C2", "vendor": "X",
             "start": "2024-06-01", "end": "2024-12-31"},  # overlaps
        ]
        result = analyze_renewal_patterns(rows, "id", "vendor", "start", "end")
        x = [v for v in result.vendor_details if v.vendor == "X"][0]
        assert x.has_renewals is True

    def test_multiple_renewals(self):
        rows = [
            {"id": "C1", "vendor": "Y", "value": 1000,
             "start": "2024-01-01", "end": "2024-03-31"},
            {"id": "C2", "vendor": "Y", "value": 1100,
             "start": "2024-04-01", "end": "2024-06-30"},
            {"id": "C3", "vendor": "Y", "value": 1200,
             "start": "2024-07-01", "end": "2024-09-30"},
        ]
        result = analyze_renewal_patterns(
            rows, "id", "vendor", "start", "end", value_column="value",
        )
        y = [v for v in result.vendor_details if v.vendor == "Y"][0]
        assert y.renewal_count == 2

    def test_summary_non_empty(self):
        result = analyze_renewal_patterns(
            self._renewal_rows(), "id", "vendor", "start", "end",
        )
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0


# ---------------------------------------------------------------------------
# 5. format_contract_report
# ---------------------------------------------------------------------------


class TestFormatContractReport:

    def test_all_none(self):
        report = format_contract_report()
        assert "No analysis data provided" in report

    def test_contracts_section(self):
        result = analyze_contracts(_base_rows(), "id", "vendor", "value")
        report = format_contract_report(contracts=result)
        assert "Portfolio Overview" in report
        assert "Total contracts: 5" in report
        assert "Acme" in report

    def test_expiring_section(self):
        ref = datetime(2025, 1, 1)
        rows = [{"id": "C1", "vendor": "V", "value": 5000, "end": "2025-01-15"}]
        expiring = detect_expiring_contracts(
            rows, "id", "end",
            vendor_column="vendor", value_column="value",
            reference_date=ref,
        )
        report = format_contract_report(expiring=expiring)
        assert "Expiring Contracts" in report
        assert "CRITICAL" in report

    def test_concentration_section(self):
        rows = [
            {"vendor": "A", "value": 8000},
            {"vendor": "B", "value": 2000},
        ]
        conc = vendor_contract_concentration(rows, "vendor", "value")
        report = format_contract_report(concentration=conc)
        assert "Vendor Concentration" in report
        assert "HHI" in report

    def test_renewals_section(self):
        rows = [
            {"id": "C1", "vendor": "X", "value": 1000,
             "start": "2024-01-01", "end": "2024-06-30"},
            {"id": "C2", "vendor": "X", "value": 1100,
             "start": "2024-07-01", "end": "2024-12-31"},
        ]
        ren = analyze_renewal_patterns(
            rows, "id", "vendor", "start", "end", value_column="value",
        )
        report = format_contract_report(renewals=ren)
        assert "Renewal Patterns" in report
        assert "renewal(s)" in report

    def test_full_report(self):
        rows = _dated_rows()
        contracts = analyze_contracts(
            rows, "id", "vendor", "value",
            start_column="start", end_column="end",
        )
        ref = datetime(2025, 1, 1)
        expiring = detect_expiring_contracts(
            rows, "id", "end",
            vendor_column="vendor", value_column="value",
            reference_date=ref,
        )
        conc = vendor_contract_concentration(rows, "vendor", "value")
        ren = analyze_renewal_patterns(
            rows, "id", "vendor", "start", "end", value_column="value",
        )
        report = format_contract_report(contracts, expiring, conc, ren)
        assert "Contract Analysis Report" in report
        assert "Portfolio Overview" in report

    def test_report_header(self):
        report = format_contract_report()
        assert report.startswith("Contract Analysis Report")

    def test_status_in_report(self):
        result = analyze_contracts(
            _status_rows(), "id", "vendor", "value", status_column="status",
        )
        report = format_contract_report(contracts=result)
        assert "Status:" in report
        assert "active" in report
