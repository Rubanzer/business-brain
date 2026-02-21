"""Tests for profiler.profile_all_tables — Focus Mode table_filter support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.discovery.profiler import profile_all_tables


@patch("business_brain.discovery.profiler.classify_columns")
@patch("business_brain.discovery.profiler.metadata_store")
class TestProfileAllTablesFiltering:
    @pytest.mark.asyncio
    async def test_profile_with_filter_calls_get_filtered(
        self, mock_metadata_store, mock_classify
    ):
        """Verify get_filtered is called with the table_filter argument."""
        session = AsyncMock()

        mock_metadata_store.get_filtered = AsyncMock(return_value=[])

        await profile_all_tables(session, table_filter=["sales", "orders"])

        mock_metadata_store.get_filtered.assert_called_once_with(
            session, ["sales", "orders"]
        )

    @pytest.mark.asyncio
    async def test_profile_none_filter_calls_get_filtered_none(
        self, mock_metadata_store, mock_classify
    ):
        """When table_filter is None, get_filtered is called with None (all tables)."""
        session = AsyncMock()

        mock_metadata_store.get_filtered = AsyncMock(return_value=[])

        await profile_all_tables(session, table_filter=None)

        mock_metadata_store.get_filtered.assert_called_once_with(session, None)

    @pytest.mark.asyncio
    async def test_profile_empty_filter_returns_empty(
        self, mock_metadata_store, mock_classify
    ):
        """Empty table_filter list means get_filtered returns [], so no profiling done."""
        session = AsyncMock()

        mock_metadata_store.get_filtered = AsyncMock(return_value=[])

        result = await profile_all_tables(session, table_filter=[])

        mock_metadata_store.get_filtered.assert_called_once_with(session, [])
        assert result == []
        # classify_columns should never be called when there are no entries
        mock_classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_profile_handles_single_table(
        self, mock_metadata_store, mock_classify
    ):
        """Profiles a single table correctly when filter contains one table."""
        session = AsyncMock()

        entry = MagicMock(
            table_name="sales",
            columns_metadata=[
                {"name": "id", "type": "integer"},
                {"name": "amount", "type": "numeric"},
            ],
        )
        mock_metadata_store.get_filtered = AsyncMock(return_value=[entry])

        # Mock the SQL queries that _profile_table executes
        count_result = MagicMock()
        count_result.scalar.return_value = 42

        sample_row = MagicMock()
        sample_row._mapping = {"id": 1, "amount": 100.0}
        sample_result = MagicMock()
        sample_result.fetchall.return_value = [sample_row]

        # Mock existing profile lookup (for upsert check)
        existing_profile_result = MagicMock()
        existing_profile_result.scalar_one_or_none.return_value = None

        # session.execute is called multiple times: COUNT, SELECT sample, profile lookup
        session.execute = AsyncMock(
            side_effect=[count_result, sample_result, existing_profile_result]
        )

        mock_classify.return_value = {
            "id": {"role": "identifier", "confidence": 0.95},
            "amount": {"role": "measure", "confidence": 0.90},
        }

        result = await profile_all_tables(session, table_filter=["sales"])

        mock_metadata_store.get_filtered.assert_called_once_with(session, ["sales"])
        # The function should have attempted to profile the single entry
        assert session.execute.call_count >= 1
