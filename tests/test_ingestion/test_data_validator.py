"""Tests for the data validation module (hard checks + z-score outlier detection)."""

import math
from datetime import datetime, timezone, timedelta

import pytest

from business_brain.ingestion.data_validator import (
    ValidationResult,
    _is_quantity_column,
    _is_date_column,
    _is_identifier_column,
    _check_negative,
    _check_future_date,
    _check_missing_identifier,
    _check_duplicate_pks,
    _compute_column_stats,
    detect_outliers,
    validate_rows,
)


# ---------------------------------------------------------------------------
# Column classification heuristics
# ---------------------------------------------------------------------------


class TestColumnClassification:
    """Tests for heuristic column classifiers."""

    def test_quantity_columns_detected(self):
        """Keywords like 'weight', 'qty', 'amount' in numeric columns are detected."""
        assert _is_quantity_column("total_weight", "DOUBLE PRECISION") is True
        assert _is_quantity_column("qty_ordered", "BIGINT") is True
        assert _is_quantity_column("amount_usd", "DOUBLE PRECISION") is True
        assert _is_quantity_column("production_mt", "BIGINT") is True

    def test_non_quantity_columns_not_flagged(self):
        """Non-quantity columns or text columns should not be flagged."""
        assert _is_quantity_column("name", "TEXT") is False
        assert _is_quantity_column("weight", "TEXT") is False  # text type
        assert _is_quantity_column("category", "BIGINT") is False

    def test_date_columns_detected(self):
        """Date-related columns are detected."""
        assert _is_date_column("order_date", "TIMESTAMP") is True
        assert _is_date_column("created_at", "TEXT") is True
        assert _is_date_column("timestamp", "TEXT") is True

    def test_non_date_columns_not_flagged(self):
        """Non-date columns should not be flagged as date."""
        assert _is_date_column("name", "TEXT") is False
        assert _is_date_column("total", "BIGINT") is False

    def test_identifier_columns_detected(self):
        """First column or columns with 'id'/'key' in name are identifiers."""
        assert _is_identifier_column("heat_id", 5) is True
        assert _is_identifier_column("product_code", 3) is True
        assert _is_identifier_column("any_column", 0) is True  # first column

    def test_non_identifier_columns(self):
        """Regular data columns should not be flagged as identifiers."""
        assert _is_identifier_column("weight", 3) is False
        assert _is_identifier_column("temperature", 5) is False


# ---------------------------------------------------------------------------
# Hard checks - negative values
# ---------------------------------------------------------------------------


class TestNegativeCheck:
    """Tests for negative value detection in quantity columns."""

    def test_negative_value_flagged(self):
        row = {"total_amount": -500}
        issue = _check_negative(row, "total_amount", 0)
        assert issue is not None
        assert issue["check"] == "negative_value"
        assert issue["severity"] == "critical"

    def test_positive_value_passes(self):
        row = {"total_amount": 100.5}
        issue = _check_negative(row, "total_amount", 0)
        assert issue is None

    def test_zero_passes(self):
        row = {"total_amount": 0}
        issue = _check_negative(row, "total_amount", 0)
        assert issue is None

    def test_none_passes(self):
        row = {"total_amount": None}
        issue = _check_negative(row, "total_amount", 0)
        assert issue is None

    def test_string_negative_detected(self):
        row = {"total_amount": "-250.50"}
        issue = _check_negative(row, "total_amount", 0)
        assert issue is not None

    def test_comma_separated_number(self):
        row = {"total_amount": "-1,234.56"}
        issue = _check_negative(row, "total_amount", 0)
        assert issue is not None

    def test_non_numeric_passes(self):
        row = {"total_amount": "not a number"}
        issue = _check_negative(row, "total_amount", 0)
        assert issue is None


# ---------------------------------------------------------------------------
# Hard checks - future dates
# ---------------------------------------------------------------------------


class TestFutureDateCheck:
    """Tests for future date detection."""

    def test_future_date_flagged(self):
        now = datetime(2025, 1, 15, tzinfo=timezone.utc)
        row = {"order_date": "2026-06-01"}
        issue = _check_future_date(row, "order_date", 0, now)
        assert issue is not None
        assert issue["check"] == "future_date"
        assert issue["severity"] == "warning"

    def test_past_date_passes(self):
        now = datetime(2025, 1, 15, tzinfo=timezone.utc)
        row = {"order_date": "2024-12-31"}
        issue = _check_future_date(row, "order_date", 0, now)
        assert issue is None

    def test_today_date_passes(self):
        now = datetime(2025, 1, 15, tzinfo=timezone.utc)
        row = {"order_date": "2025-01-15"}
        issue = _check_future_date(row, "order_date", 0, now)
        assert issue is None

    def test_empty_date_passes(self):
        now = datetime.now(timezone.utc)
        row = {"order_date": ""}
        issue = _check_future_date(row, "order_date", 0, now)
        assert issue is None

    def test_none_date_passes(self):
        now = datetime.now(timezone.utc)
        row = {"order_date": None}
        issue = _check_future_date(row, "order_date", 0, now)
        assert issue is None

    def test_slash_format_future(self):
        now = datetime(2025, 1, 15, tzinfo=timezone.utc)
        row = {"order_date": "06/01/2026"}  # MM/DD/YYYY
        issue = _check_future_date(row, "order_date", 0, now)
        assert issue is not None


# ---------------------------------------------------------------------------
# Hard checks - missing identifiers
# ---------------------------------------------------------------------------


class TestMissingIdentifier:
    """Tests for missing identifier detection."""

    def test_empty_id_flagged(self):
        row = {"heat_id": ""}
        issue = _check_missing_identifier(row, "heat_id", 0)
        assert issue is not None
        assert issue["check"] == "missing_identifier"
        assert issue["severity"] == "critical"

    def test_none_id_flagged(self):
        row = {"heat_id": None}
        issue = _check_missing_identifier(row, "heat_id", 0)
        assert issue is not None

    def test_whitespace_only_flagged(self):
        row = {"heat_id": "   "}
        issue = _check_missing_identifier(row, "heat_id", 0)
        assert issue is not None

    def test_valid_id_passes(self):
        row = {"heat_id": "H-12345"}
        issue = _check_missing_identifier(row, "heat_id", 0)
        assert issue is None


# ---------------------------------------------------------------------------
# Hard checks - duplicate PKs
# ---------------------------------------------------------------------------


class TestDuplicatePKs:
    """Tests for duplicate primary key detection."""

    def test_duplicates_detected(self):
        rows = [
            {"id": "A1", "value": 10},
            {"id": "A2", "value": 20},
            {"id": "A1", "value": 30},  # duplicate
            {"id": "A3", "value": 40},
            {"id": "A2", "value": 50},  # duplicate
        ]
        dups = _check_duplicate_pks(rows, "id")
        assert 2 in dups  # row 2 (A1 duplicate)
        assert 4 in dups  # row 4 (A2 duplicate)
        assert len(dups) == 2

    def test_no_duplicates(self):
        rows = [
            {"id": "A1", "value": 10},
            {"id": "A2", "value": 20},
            {"id": "A3", "value": 30},
        ]
        dups = _check_duplicate_pks(rows, "id")
        assert len(dups) == 0

    def test_empty_rows(self):
        dups = _check_duplicate_pks([], "id")
        assert len(dups) == 0

    def test_empty_pk_values_ignored(self):
        rows = [
            {"id": "", "value": 10},
            {"id": "", "value": 20},
        ]
        dups = _check_duplicate_pks(rows, "id")
        assert len(dups) == 0  # empty values not considered duplicates


# ---------------------------------------------------------------------------
# Soft checks - z-score outlier detection
# ---------------------------------------------------------------------------


class TestOutlierDetection:
    """Tests for z-score outlier detection."""

    def test_basic_stats_computation(self):
        rows = [{"val": str(i)} for i in range(100)]
        mean, stdev, count = _compute_column_stats(rows, "val")
        assert count == 100
        assert abs(mean - 49.5) < 0.01
        assert stdev > 0

    def test_too_few_rows_returns_zero(self):
        rows = [{"val": "1"}, {"val": "2"}]
        mean, stdev, count = _compute_column_stats(rows, "val")
        assert count == 0  # < 5 minimum sample

    def test_outlier_detected(self):
        # 98 rows with values 1-98, then add a massive outlier
        rows = [{"val": str(i)} for i in range(1, 99)]
        rows.append({"val": "10000"})  # z-score will be huge
        rows.append({"val": "50"})  # normal

        col_types = {"val": "DOUBLE PRECISION"}
        outliers = detect_outliers(rows, col_types, z_threshold=3.0)
        assert len(outliers) >= 1
        # The 10000 value should be flagged
        outlier_values = [o["value"] for o in outliers]
        assert 10000.0 in outlier_values

    def test_no_outliers_in_uniform_data(self):
        rows = [{"val": str(50 + i)} for i in range(100)]
        col_types = {"val": "DOUBLE PRECISION"}
        outliers = detect_outliers(rows, col_types, z_threshold=3.0)
        assert len(outliers) == 0

    def test_text_columns_skipped(self):
        rows = [{"name": f"item_{i}"} for i in range(100)]
        col_types = {"name": "TEXT"}
        outliers = detect_outliers(rows, col_types)
        assert len(outliers) == 0

    def test_none_values_handled(self):
        rows = [{"val": str(i) if i % 2 == 0 else None} for i in range(100)]
        col_types = {"val": "DOUBLE PRECISION"}
        # Should not crash
        outliers = detect_outliers(rows, col_types)
        assert isinstance(outliers, list)


# ---------------------------------------------------------------------------
# Integration: validate_rows
# ---------------------------------------------------------------------------


class TestValidateRows:
    """Integration tests for the full validate_rows function."""

    def test_all_clean_rows(self):
        """Rows with no issues should all pass."""
        rows = [
            {"id": f"H{i}", "weight": str(100 + i), "name": f"Product {i}"}
            for i in range(50)
        ]
        col_types = {"id": "TEXT", "weight": "DOUBLE PRECISION", "name": "TEXT"}

        result = validate_rows(rows, col_types)
        assert len(result.clean_rows) == 50
        assert len(result.quarantined) == 0

    def test_negative_quantity_quarantined(self):
        """Rows with negative values in quantity columns are quarantined."""
        rows = [
            {"id": "H1", "total_weight": "100"},
            {"id": "H2", "total_weight": "-50"},  # should be quarantined
            {"id": "H3", "total_weight": "200"},
        ]
        col_types = {"id": "TEXT", "total_weight": "DOUBLE PRECISION"}

        result = validate_rows(rows, col_types)
        assert len(result.quarantined) == 1
        assert result.quarantined[0]["row_index"] == 1
        assert len(result.clean_rows) == 2

    def test_future_date_warning_not_quarantined(self):
        """Future dates produce warnings but don't quarantine (severity=warning)."""
        rows = [
            {"id": "1", "order_date": "2020-01-01"},
            {"id": "2", "order_date": "2090-12-31"},  # future date - warning
            {"id": "3", "order_date": "2024-06-15"},
        ]
        col_types = {"id": "TEXT", "order_date": "TIMESTAMP"}

        result = validate_rows(rows, col_types)
        # Future date is severity=warning, not critical, so row is NOT quarantined
        assert len(result.clean_rows) == 3
        assert len(result.quarantined) == 0

    def test_missing_pk_quarantined(self):
        """Row with missing primary key is quarantined."""
        rows = [
            {"id": "H1", "value": "100"},
            {"id": "", "value": "200"},  # missing PK - critical
            {"id": "H3", "value": "300"},
        ]
        col_types = {"id": "TEXT", "value": "DOUBLE PRECISION"}

        result = validate_rows(rows, col_types)
        assert len(result.quarantined) == 1
        assert result.quarantined[0]["row_data"]["id"] == ""
        assert len(result.clean_rows) == 2

    def test_serial_pk_skips_pk_checks(self):
        """With serial PK, duplicate and missing PK checks are skipped."""
        rows = [
            {"name": "A", "value": "100"},
            {"name": "", "value": "200"},  # no PK check
            {"name": "A", "value": "300"},  # no duplicate check
        ]
        col_types = {"name": "TEXT", "value": "DOUBLE PRECISION"}

        result = validate_rows(rows, col_types, use_serial_pk=True)
        assert len(result.clean_rows) == 3
        assert len(result.quarantined) == 0

    def test_multiple_issues_on_one_row(self):
        """A row can have multiple issues; critical ones trigger quarantine."""
        rows = [
            {"id": "", "total_amount": "-500", "order_date": "2024-01-01"},
        ]
        col_types = {
            "id": "TEXT",
            "total_amount": "DOUBLE PRECISION",
            "order_date": "TIMESTAMP",
        }

        result = validate_rows(rows, col_types)
        assert len(result.quarantined) == 1
        # Should have at least 2 issues: missing PK + negative amount
        assert len(result.quarantined[0]["issues"]) >= 2

    def test_outliers_detected_in_clean_rows(self):
        """Z-score outliers are flagged but rows remain in clean_rows."""
        rows = [{"id": str(i), "measurement": str(50)} for i in range(100)]
        rows.append({"id": "999", "measurement": "10000"})  # outlier

        col_types = {"id": "TEXT", "measurement": "DOUBLE PRECISION"}
        result = validate_rows(rows, col_types)

        # Outlier row should still be in clean_rows (soft check)
        assert len(result.clean_rows) == 101
        assert len(result.quarantined) == 0
        # But should be flagged as an outlier
        assert len(result.outliers) >= 1

    def test_empty_rows_handled(self):
        """Empty row list returns empty result."""
        result = validate_rows([], {})
        assert len(result.clean_rows) == 0
        assert len(result.quarantined) == 0
        assert len(result.outliers) == 0

    def test_validation_summary(self):
        """summary() returns expected keys."""
        rows = [
            {"id": "H1", "total_weight": "100"},
            {"id": "H2", "total_weight": "-50"},
        ]
        col_types = {"id": "TEXT", "total_weight": "DOUBLE PRECISION"}

        result = validate_rows(rows, col_types)
        summary = result.summary()
        assert "rows_clean" in summary
        assert "rows_quarantined" in summary
        assert "outliers_flagged" in summary
        assert summary["rows_clean"] == 1
        assert summary["rows_quarantined"] == 1

    def test_batch_id_in_quarantine(self):
        """Quarantined rows should have a batch_id for grouping."""
        rows = [{"id": "", "value": "100"}]
        col_types = {"id": "TEXT", "value": "DOUBLE PRECISION"}

        result = validate_rows(rows, col_types)
        assert len(result.quarantined) == 1
        assert "batch_id" in result.quarantined[0]
        assert len(result.quarantined[0]["batch_id"]) > 0
