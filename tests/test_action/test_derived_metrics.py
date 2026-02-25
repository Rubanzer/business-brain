"""Tests for derived metric creation — formula-based calculated metrics."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from business_brain.action.api import create_derived_metric


def _make_metadata_entry(table_name, columns):
    """Create a mock metadata entry with columns_metadata."""
    entry = MagicMock()
    entry.table_name = table_name
    entry.columns_metadata = [{"name": c} for c in columns]
    return entry


class TestCreateDerivedMetric:
    """Tests for the POST /metrics/derived endpoint."""

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_create_derived_valid_formula(self, mock_store):
        """Creates a derived metric with a valid formula referencing existing columns."""
        session = AsyncMock()
        session.add = MagicMock()

        async def fake_refresh(obj):
            obj.id = 100

        session.refresh = AsyncMock(side_effect=fake_refresh)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", ["total_kwh", "output_mt"]),
        ])

        body = {
            "metric_name": "Specific Energy Consumption",
            "formula": "energy.total_kwh / energy.output_mt",
            "unit": "kWh/MT",
            "normal_min": 400,
            "normal_max": 600,
        }

        result = await create_derived_metric(body, session)

        assert result["status"] == "created"
        assert result["id"] == 100
        assert result["metric_name"] == "Specific Energy Consumption"
        assert result["is_derived"] is True

        added_metric = session.add.call_args[0][0]
        assert added_metric.is_derived is True
        assert added_metric.formula == "energy.total_kwh / energy.output_mt"
        assert added_metric.source_columns == ["energy.total_kwh", "energy.output_mt"]
        assert added_metric.unit == "kWh/MT"
        assert added_metric.normal_min == 400
        assert added_metric.normal_max == 600
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_derived_invalid_column_rejected(self, mock_store):
        """Referencing a nonexistent column raises HTTPException 400."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", ["total_kwh"]),
        ])

        body = {
            "metric_name": "Bad Metric",
            "formula": "energy.total_kwh / energy.nonexistent_col",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "nonexistent_col" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_derived_invalid_table_rejected(self, mock_store):
        """Referencing a nonexistent table raises HTTPException 400."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", ["total_kwh"]),
        ])

        body = {
            "metric_name": "Bad Table Metric",
            "formula": "nonexistent_table.some_col * 2",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "nonexistent_table" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_derived_stored_correctly(self, mock_store):
        """Derived metric is stored with is_derived=True, formula, and source_columns."""
        session = AsyncMock()
        session.add = MagicMock()

        async def fake_refresh(obj):
            obj.id = 200

        session.refresh = AsyncMock(side_effect=fake_refresh)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("production", ["heats_done", "total_hours"]),
        ])

        body = {
            "metric_name": "Heats Per Hour",
            "formula": "production.heats_done / production.total_hours",
        }

        result = await create_derived_metric(body, session)

        added = session.add.call_args[0][0]
        assert added.is_derived is True
        assert "production.heats_done" in added.source_columns
        assert "production.total_hours" in added.source_columns
        assert added.formula == "production.heats_done / production.total_hours"
        assert added.metric_name == "Heats Per Hour"

    @pytest.mark.asyncio
    async def test_derived_missing_metric_name(self):
        """Missing metric_name raises HTTPException 400."""
        session = AsyncMock()

        body = {
            "formula": "energy.total_kwh / energy.output_mt",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "metric_name" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_derived_missing_formula(self):
        """Missing formula raises HTTPException 400."""
        session = AsyncMock()

        body = {
            "metric_name": "Some Metric",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "formula" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_derived_empty_metric_name(self):
        """Empty/whitespace-only metric_name raises HTTPException 400."""
        session = AsyncMock()

        body = {
            "metric_name": "   ",
            "formula": "something",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "metric_name" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_derived_formula_no_table_refs(self, mock_store):
        """Formula with no table.column references stores empty source_columns."""
        session = AsyncMock()
        session.add = MagicMock()

        async def fake_refresh(obj):
            obj.id = 300

        session.refresh = AsyncMock(side_effect=fake_refresh)

        # No table.column refs in formula, so no validation needed
        mock_store.get_all = AsyncMock(return_value=[])

        body = {
            "metric_name": "Constant Metric",
            "formula": "100 + 200",
        }

        result = await create_derived_metric(body, session)

        assert result["status"] == "created"
        added = session.add.call_args[0][0]
        assert added.source_columns == []
        assert added.is_derived is True

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_derived_in_thresholds_list(self, mock_store):
        """Derived metrics appear correctly alongside normal metrics
        when stored as MetricThreshold."""
        session = AsyncMock()
        session.add = MagicMock()

        async def fake_refresh(obj):
            obj.id = 400

        session.refresh = AsyncMock(side_effect=fake_refresh)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", ["kwh", "output"]),
        ])

        body = {
            "metric_name": "Energy Efficiency",
            "formula": "energy.kwh / energy.output",
            "unit": "kWh/unit",
            "normal_min": 50,
            "normal_max": 100,
            "warning_min": 40,
            "warning_max": 120,
            "critical_min": 30,
            "critical_max": 150,
        }

        result = await create_derived_metric(body, session)

        assert result["is_derived"] is True
        added = session.add.call_args[0][0]
        assert added.normal_min == 50
        assert added.normal_max == 100
        assert added.warning_min == 40
        assert added.warning_max == 120
        assert added.critical_min == 30
        assert added.critical_max == 150
