"""Tests for suggest metrics endpoint — suggests trackable metrics from numeric columns."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.action.api import suggest_metrics


def _make_metadata_entry(table_name, columns):
    """Create a mock metadata entry.

    columns: list of dicts like [{"name": "power_kwh", "type": "float"}]
    """
    entry = MagicMock()
    entry.table_name = table_name
    entry.columns_metadata = columns
    return entry


class TestSuggestMetrics:
    """Tests for the POST /setup/suggest-metrics endpoint."""

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_suggest_from_numeric_columns(self, mock_store):
        """Finds numeric columns as metric suggestions."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("production", [
                {"name": "output_tons", "type": "float"},
                {"name": "power_kwh", "type": "numeric"},
                {"name": "batch_name", "type": "varchar"},
                {"name": "created_at", "type": "timestamp"},
            ])
        ])

        # No existing metrics configured
        existing_result = MagicMock()
        existing_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=existing_result)

        result = await suggest_metrics(session)

        assert len(result["suggestions"]) == 1
        table_sugg = result["suggestions"][0]
        assert table_sugg["table_name"] == "production"
        col_names = [c["column_name"] for c in table_sugg["columns"]]
        assert "output_tons" in col_names
        assert "power_kwh" in col_names
        # Non-numeric columns should not be suggested
        assert "batch_name" not in col_names
        assert "created_at" not in col_names

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_no_data_empty_suggestions(self, mock_store):
        """No tables uploaded returns empty suggestions with message."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[])

        result = await suggest_metrics(session)

        assert result["suggestions"] == []
        assert "No data uploaded" in result.get("message", "")

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_deduplicates_existing_metrics(self, mock_store):
        """Already-configured metrics are excluded from suggestions."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", [
                {"name": "power_kwh", "type": "float"},
                {"name": "temperature", "type": "decimal"},
                {"name": "voltage", "type": "double"},
            ])
        ])

        # "power_kwh" is already configured as a metric
        existing_result = MagicMock()
        existing_result.fetchall.return_value = [("power_kwh",)]
        session.execute = AsyncMock(return_value=existing_result)

        result = await suggest_metrics(session)

        assert len(result["suggestions"]) == 1
        col_names = [c["column_name"] for c in result["suggestions"][0]["columns"]]
        assert "power_kwh" not in col_names
        assert "temperature" in col_names
        assert "voltage" in col_names

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_skips_id_columns(self, mock_store):
        """Columns ending with _id or named 'id' are skipped."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("orders", [
                {"name": "id", "type": "integer"},
                {"name": "customer_id", "type": "integer"},
                {"name": "order_total", "type": "float"},
                {"name": "quantity", "type": "int"},
            ])
        ])

        existing_result = MagicMock()
        existing_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=existing_result)

        result = await suggest_metrics(session)

        assert len(result["suggestions"]) == 1
        col_names = [c["column_name"] for c in result["suggestions"][0]["columns"]]
        assert "id" not in col_names
        assert "customer_id" not in col_names
        assert "order_total" in col_names
        assert "quantity" in col_names

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_suggested_metric_name_formatting(self, mock_store):
        """Suggested metric name replaces underscores with spaces and title-cases."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("metrics", [
                {"name": "total_power_kwh", "type": "float"},
            ])
        ])

        existing_result = MagicMock()
        existing_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=existing_result)

        result = await suggest_metrics(session)

        col = result["suggestions"][0]["columns"][0]
        assert col["suggested_metric_name"] == "Total Power Kwh"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_multiple_tables(self, mock_store):
        """Suggestions come from all tables with numeric columns."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", [
                {"name": "kwh", "type": "float"},
            ]),
            _make_metadata_entry("production", [
                {"name": "output_mt", "type": "real"},
            ]),
        ])

        existing_result = MagicMock()
        existing_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=existing_result)

        result = await suggest_metrics(session)

        table_names = [s["table_name"] for s in result["suggestions"]]
        assert "energy" in table_names
        assert "production" in table_names

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_table_with_no_numeric_columns(self, mock_store):
        """Table with only non-numeric columns produces no suggestions."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("logs", [
                {"name": "message", "type": "text"},
                {"name": "severity", "type": "varchar"},
                {"name": "timestamp", "type": "datetime"},
            ])
        ])

        existing_result = MagicMock()
        existing_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=existing_result)

        result = await suggest_metrics(session)

        assert result["suggestions"] == []
