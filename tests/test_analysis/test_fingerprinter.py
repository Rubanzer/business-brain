"""Tests for analysis/track1/fingerprinter.py — role detection + fingerprinting."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from business_brain.analysis.track1.fingerprinter import (
    ColumnFingerprint,
    TableFingerprint,
    _infer_role,
    _safe,
    fingerprint_table,
)


# ---------------------------------------------------------------------------
# _safe (SQL injection prevention)
# ---------------------------------------------------------------------------


class TestSafe:
    def test_normal_name(self):
        assert _safe("production") == "production"

    def test_strips_special_chars(self):
        assert _safe("table; DROP TABLE") == "tableDROPTABLE"

    def test_underscores_preserved(self):
        assert _safe("my_table_123") == "my_table_123"

    def test_empty_string(self):
        assert _safe("") == ""


# ---------------------------------------------------------------------------
# Role inference
# ---------------------------------------------------------------------------


class TestInferRole:
    def test_foreign_key_takes_priority(self):
        assert _infer_role("order_id", "identifier", 0.9, is_foreign_key=True) == "FOREIGN_KEY"

    def test_temporal_by_type(self):
        assert _infer_role("created", "temporal", 0.5, is_foreign_key=False) == "TIME_INDEX"

    def test_temporal_by_name(self):
        assert _infer_role("created_at", "text", 0.1, is_foreign_key=False) == "TIME_INDEX"

    def test_temporal_date_suffix(self):
        assert _infer_role("order_date", "text", 0.1, is_foreign_key=False) == "TIME_INDEX"

    def test_grain_key_by_type(self):
        assert _infer_role("invoice_no", "identifier", 0.95, is_foreign_key=False) == "GRAIN_KEY"

    def test_grain_key_by_name_and_ratio(self):
        assert _infer_role("customer_id", "text", 0.95, is_foreign_key=False) == "GRAIN_KEY"

    def test_measure(self):
        assert _infer_role("revenue", "numeric_metric", 0.5, is_foreign_key=False) == "MEASURE"

    def test_measure_currency(self):
        assert _infer_role("total_amount", "numeric_currency", 0.5, is_foreign_key=False) == "MEASURE"

    def test_measure_percentage(self):
        assert _infer_role("efficiency", "numeric_percentage", 0.5, is_foreign_key=False) == "MEASURE"

    def test_dimension_categorical(self):
        assert _infer_role("region", "categorical", 0.1, is_foreign_key=False) == "DIMENSION"

    def test_dimension_boolean(self):
        assert _infer_role("is_active", "boolean", 0.01, is_foreign_key=False) == "DIMENSION"

    def test_free_text(self):
        assert _infer_role("notes", "text", 0.9, is_foreign_key=False) == "FREE_TEXT"

    def test_unknown_defaults_to_dimension(self):
        assert _infer_role("something", "custom_type", 0.1, is_foreign_key=False) == "DIMENSION"


# ---------------------------------------------------------------------------
# TableFingerprint dataclass
# ---------------------------------------------------------------------------


class TestTableFingerprint:
    def test_basic_construction(self):
        fp = TableFingerprint(
            table_name="test",
            row_count=100,
            data_hash="abc",
            domain_hint="general",
            time_index=None,
            measures=["m1"],
            dimensions=["d1"],
        )
        assert fp.table_name == "test"
        assert fp.row_count == 100
        assert fp.columns == {}

    def test_columns_default_empty(self):
        fp = TableFingerprint(
            table_name="t", row_count=0, data_hash="x",
            domain_hint="g", time_index=None, measures=[], dimensions=[],
        )
        assert isinstance(fp.columns, dict)
        assert len(fp.columns) == 0


# ---------------------------------------------------------------------------
# ColumnFingerprint dataclass
# ---------------------------------------------------------------------------


class TestColumnFingerprint:
    def test_basic_construction(self):
        fp = ColumnFingerprint(
            name="revenue",
            semantic_type="numeric_metric",
            role="MEASURE",
            cardinality=100,
            null_rate=0.05,
        )
        assert fp.name == "revenue"
        assert fp.joinable_to == []

    def test_joinable_to_default(self):
        fp = ColumnFingerprint(name="x", semantic_type="t", role="r")
        assert fp.joinable_to == []
        assert fp.distribution is None


# ---------------------------------------------------------------------------
# fingerprint_table (async, requires mocking)
# ---------------------------------------------------------------------------


class TestFingerprintTable:
    @pytest.mark.asyncio
    async def test_with_profile(self):
        """fingerprint_table with a pre-built TableProfile."""
        session = AsyncMock()

        # Mock profile
        profile = MagicMock()
        profile.row_count = 500
        profile.data_hash = "hash123"
        profile.domain_hint = "manufacturing"
        profile.column_classification = {
            "output_kg": {"type": "numeric_metric", "stats": {"cardinality": 200, "null_pct": 2}},
            "shift": {"type": "categorical", "stats": {"cardinality": 3, "null_pct": 0}},
            "date": {"type": "temporal", "stats": {"cardinality": 100, "null_pct": 0}},
        }

        # Mock relationship query - no relationships
        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=rel_result)

        fp = await fingerprint_table(session, "production", profile=profile)

        assert fp.table_name == "production"
        assert fp.row_count == 500
        assert fp.data_hash == "hash123"
        assert "output_kg" in fp.measures
        assert "shift" in fp.dimensions
        assert fp.time_index == "date"

    @pytest.mark.asyncio
    async def test_measures_and_dimensions_separated(self):
        """Verify measures and dimensions are correctly categorized."""
        session = AsyncMock()

        profile = MagicMock()
        profile.row_count = 1000
        profile.data_hash = "xyz"
        profile.domain_hint = "sales"
        profile.column_classification = {
            "revenue": "numeric_metric",
            "cost": "numeric_currency",
            "margin_pct": "numeric_percentage",
            "region": "categorical",
            "is_premium": "boolean",
            "notes": "text",
        }

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=rel_result)

        fp = await fingerprint_table(session, "sales", profile=profile)

        assert set(fp.measures) == {"revenue", "cost", "margin_pct"}
        assert set(fp.dimensions) == {"region", "is_premium"}
        assert fp.columns["notes"].role == "FREE_TEXT"

    @pytest.mark.asyncio
    async def test_relationships_populate_joinable_to(self):
        """Cross-table relationships should appear in joinable_to (Gap #2)."""
        session = AsyncMock()

        profile = MagicMock()
        profile.row_count = 100
        profile.data_hash = "h"
        profile.domain_hint = "general"
        profile.column_classification = {
            "product_id": "identifier",
            "amount": "numeric_metric",
        }

        # Mock relationship
        rel = MagicMock()
        rel.table_a = "orders"
        rel.column_a = "product_id"
        rel.table_b = "products"
        rel.column_b = "id"
        rel.confidence = 0.9

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = [rel]
        session.execute = AsyncMock(return_value=rel_result)

        fp = await fingerprint_table(session, "orders", profile=profile)

        assert len(fp.columns["product_id"].joinable_to) == 1
        assert fp.columns["product_id"].joinable_to[0]["table"] == "products"
        assert fp.columns["product_id"].joinable_to[0]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_null_rate_normalization(self):
        """null_pct > 1 should be divided by 100 to get rate."""
        session = AsyncMock()

        profile = MagicMock()
        profile.row_count = 100
        profile.data_hash = "h"
        profile.domain_hint = "g"
        profile.column_classification = {
            "col1": {"type": "numeric_metric", "stats": {"cardinality": 50, "null_pct": 25}},
        }

        rel_result = MagicMock()
        rel_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=rel_result)

        fp = await fingerprint_table(session, "test", profile=profile)

        assert fp.columns["col1"].null_rate == pytest.approx(0.25)
