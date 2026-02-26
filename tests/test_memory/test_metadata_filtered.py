"""Tests for metadata_store.get_filtered — Focus Mode table filtering."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.memory.metadata_store import get_filtered


class TestGetFiltered:
    @pytest.mark.asyncio
    @patch("business_brain.memory.metadata_store.get_all")
    async def test_get_filtered_none_returns_all(self, mock_get_all):
        """When table_names is None, delegates to get_all."""
        session = AsyncMock()
        entry_a = MagicMock(table_name="sales")
        entry_b = MagicMock(table_name="customers")
        mock_get_all.return_value = [entry_a, entry_b]

        result = await get_filtered(session, table_names=None)

        mock_get_all.assert_called_once_with(session)
        assert result == [entry_a, entry_b]
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_filtered_with_names(self):
        """Returns only entries matching the provided table names."""
        session = AsyncMock()
        entry_sales = MagicMock(table_name="sales")
        entry_orders = MagicMock(table_name="orders")

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [entry_sales, entry_orders]
        session.execute = AsyncMock(return_value=result_mock)

        result = await get_filtered(session, table_names=["sales", "orders"])

        assert len(result) == 2
        assert entry_sales in result
        assert entry_orders in result
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_filtered_empty_list(self):
        """Empty table_names list returns empty list immediately without querying."""
        session = AsyncMock()

        result = await get_filtered(session, table_names=[])

        assert result == []
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_filtered_nonexistent_table(self):
        """Returns empty list when queried table names don't match any entries."""
        session = AsyncMock()

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result_mock)

        result = await get_filtered(session, table_names=["nonexistent_table"])

        assert result == []
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_filtered_db_error_returns_empty(self):
        """On database exception, performs rollback and returns empty list."""
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB connection lost"))

        result = await get_filtered(session, table_names=["sales"])

        assert result == []
        session.rollback.assert_called_once()
