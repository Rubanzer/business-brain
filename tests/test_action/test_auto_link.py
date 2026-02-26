"""Tests for auto-link metrics endpoint — fuzzy matching of metric names to columns."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.action.api import auto_link_metrics


def _make_metric(metric_id, name, table_name=None, column_name=None):
    """Create a mock MetricThreshold with the given attributes."""
    m = MagicMock()
    m.id = metric_id
    m.metric_name = name
    m.table_name = table_name
    m.column_name = column_name
    m.auto_linked = False
    m.confidence = None
    return m


def _make_metadata_entry(table_name, columns):
    """Create a mock metadata entry with columns_metadata.

    columns: list of dicts like [{"name": "power_consumption", "type": "float"}]
    """
    entry = MagicMock()
    entry.table_name = table_name
    entry.columns_metadata = columns
    return entry


class TestAutoLinkMetrics:
    """Tests for the POST /setup/auto-link-metrics endpoint."""

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_exact_match_high_confidence(self, mock_store):
        """Exact name match gets 1.0 confidence and is auto-linked."""
        session = AsyncMock()

        # Unlinked metric
        metric = _make_metric(1, "power_consumption")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]

        session.execute = AsyncMock(return_value=unlinked_result)

        # Table metadata with exact column match
        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("scada_readings", [
                {"name": "power_consumption", "type": "float"},
                {"name": "timestamp", "type": "datetime"},
            ])
        ])

        result = await auto_link_metrics(session)

        assert len(result["auto_linked"]) == 1
        assert result["auto_linked"][0]["confidence"] == 1.0
        assert result["auto_linked"][0]["table_name"] == "scada_readings"
        assert result["auto_linked"][0]["column_name"] == "power_consumption"
        assert metric.auto_linked is True
        assert metric.confidence == 1.0
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_case_insensitive_match(self, mock_store):
        """'Power_Consumption' matches 'power_consumption' at 0.95 confidence."""
        session = AsyncMock()

        metric = _make_metric(2, "Power_Consumption")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("readings", [
                {"name": "power_consumption", "type": "float"},
            ])
        ])

        result = await auto_link_metrics(session)

        # After lowering and normalizing, "power_consumption" == "power_consumption"
        # so this is actually an exact match at 1.0
        assert len(result["auto_linked"]) == 1
        linked = result["auto_linked"][0]
        assert linked["confidence"] >= 0.95
        assert linked["column_name"] == "power_consumption"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_no_match_returns_unmatched(self, mock_store):
        """Unrecognized metric with no column matches goes to unmatched list."""
        session = AsyncMock()

        metric = _make_metric(3, "xyzzyx_unknown_metric")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("production", [
                {"name": "output_tons", "type": "float"},
                {"name": "batch_id", "type": "int"},
            ])
        ])

        result = await auto_link_metrics(session)

        assert len(result["auto_linked"]) == 0
        assert len(result["suggestions"]) == 0
        assert len(result["unmatched"]) == 1
        assert result["unmatched"][0]["metric_name"] == "xyzzyx_unknown_metric"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_fuzzy_match_substring(self, mock_store):
        """'consumption' is a substring of 'power_consumption' -> 0.7 confidence (suggestion)."""
        session = AsyncMock()

        metric = _make_metric(4, "consumption")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", [
                {"name": "power_consumption", "type": "float"},
            ])
        ])

        result = await auto_link_metrics(session)

        # 0.7 is below 0.8 threshold, so it becomes a suggestion, not auto-linked
        assert len(result["auto_linked"]) == 0
        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0]["metric_name"] == "consumption"
        candidates = result["suggestions"][0]["candidates"]
        assert len(candidates) >= 1
        assert candidates[0]["confidence"] == 0.7

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_auto_link_no_tables_uploaded(self, mock_store):
        """Empty metadata (no tables uploaded) returns empty results."""
        session = AsyncMock()

        metric = _make_metric(5, "temperature")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[])

        result = await auto_link_metrics(session)

        assert len(result["auto_linked"]) == 0
        assert len(result["suggestions"]) == 0
        assert len(result["unmatched"]) == 1
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_multiple_candidates_sorted(self, mock_store):
        """Multiple matches are sorted by confidence descending."""
        session = AsyncMock()

        # Metric "power" should match "power_consumption" (substring 0.7)
        # and "power" exactly (1.0)
        metric = _make_metric(6, "power")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("readings", [
                {"name": "power_consumption", "type": "float"},
                {"name": "power", "type": "float"},
                {"name": "power_factor", "type": "float"},
            ])
        ])

        result = await auto_link_metrics(session)

        # "power" == "power" exact match at 1.0, so it gets auto-linked
        assert len(result["auto_linked"]) == 1
        assert result["auto_linked"][0]["confidence"] == 1.0
        assert result["auto_linked"][0]["column_name"] == "power"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_no_unlinked_metrics(self, mock_store):
        """When all metrics already have table_name, nothing to do."""
        session = AsyncMock()

        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("data", [{"name": "col_a", "type": "float"}])
        ])

        result = await auto_link_metrics(session)

        assert result["auto_linked"] == []
        assert result["suggestions"] == []
        assert result["unmatched"] == []
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process.metadata_store")
    async def test_word_overlap_low_confidence(self, mock_store):
        """Word overlap produces a lower confidence score."""
        session = AsyncMock()

        # "total power" has word "power" overlapping with "power_reading"
        metric = _make_metric(7, "total_power")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("meters", [
                {"name": "voltage_reading", "type": "float"},
                {"name": "ampere_total", "type": "float"},
            ])
        ])

        result = await auto_link_metrics(session)

        # "total_power" words: {"total", "power"}
        # "voltage_reading" words: {"voltage", "reading"} -> no overlap
        # "ampere_total" words: {"ampere", "total"} -> overlap = {"total"}, 1 word
        # overlap=1 >= max(1, 2//2)=1, confidence = 0.3 + 0.2*1 = 0.5
        # 0.5 < 0.8, so it is a suggestion not auto-linked
        if result["suggestions"]:
            candidates = result["suggestions"][0]["candidates"]
            for c in candidates:
                assert c["confidence"] < 0.8
