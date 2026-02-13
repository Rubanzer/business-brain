"""Tests for metadata store — including validate_tables."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.memory.metadata_store import delete, get_all, get_by_table, upsert, validate_tables


class TestMetadataStoreCRUD:
    @pytest.mark.asyncio
    async def test_get_all(self):
        session = AsyncMock()
        entry = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [entry]
        session.execute = AsyncMock(return_value=result)

        entries = await get_all(session)
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_get_by_table(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = MagicMock(table_name="sales")
        session.execute = AsyncMock(return_value=result)

        entry = await get_by_table(session, "sales")
        assert entry.table_name == "sales"

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        session = AsyncMock()
        entry = MagicMock(table_name="sales")
        result = MagicMock()
        result.scalar_one_or_none.return_value = entry
        session.execute = AsyncMock(return_value=result)

        deleted = await delete(session, "sales")
        assert deleted is True
        session.delete.assert_called_once_with(entry)

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        deleted = await delete(session, "nonexistent")
        assert deleted is False


class TestValidateTables:
    @pytest.mark.asyncio
    @patch("business_brain.memory.metadata_store.get_all")
    async def test_removes_stale_metadata(self, mock_get_all):
        session = AsyncMock()

        # Simulate pg_tables returning only "sales"
        pg_result = MagicMock()
        pg_result.fetchall.return_value = [("sales",), ("customers",)]

        # Metadata has "sales", "customers", and "old_table" (stale)
        entry_sales = MagicMock(table_name="sales")
        entry_customers = MagicMock(table_name="customers")
        entry_old = MagicMock(table_name="old_table")
        mock_get_all.return_value = [entry_sales, entry_customers, entry_old]

        session.execute = AsyncMock(return_value=pg_result)

        removed = await validate_tables(session)
        assert removed == ["old_table"]
        session.delete.assert_called_once_with(entry_old)
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.memory.metadata_store.get_all")
    async def test_no_stale_tables(self, mock_get_all):
        session = AsyncMock()

        pg_result = MagicMock()
        pg_result.fetchall.return_value = [("sales",)]

        entry = MagicMock(table_name="sales")
        mock_get_all.return_value = [entry]
        session.execute = AsyncMock(return_value=pg_result)

        removed = await validate_tables(session)
        assert removed == []
        session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_pg_tables_error(self):
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB error"))

        removed = await validate_tables(session)
        assert removed == []
