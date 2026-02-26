"""Tests for Focus Mode API — _get_focus_tables helper and focus endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.action.api import (
    _get_focus_tables,
    clear_focus,
    get_focus,
    update_focus,
)


# ---------------------------------------------------------------------------
# _get_focus_tables helper
# ---------------------------------------------------------------------------


class TestGetFocusTables:
    @pytest.mark.asyncio
    async def test_get_focus_tables_no_scopes_returns_none(self):
        """When no FocusScope rows exist, returns None (= all tables)."""
        session = AsyncMock()

        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        session.execute = AsyncMock(return_value=result_mock)

        result = await _get_focus_tables(session)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_focus_tables_with_scopes(self):
        """When FocusScope rows exist with is_included=True, returns their table names."""
        session = AsyncMock()

        result_mock = MagicMock()
        result_mock.fetchall.return_value = [("sales",), ("orders",)]
        session.execute = AsyncMock(return_value=result_mock)

        result = await _get_focus_tables(session)

        assert result == ["sales", "orders"]
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_focus_tables_db_error_returns_none(self):
        """On exception, returns None as a safe fallback (all tables)."""
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB connection failed"))

        result = await _get_focus_tables(session)

        assert result is None

    @pytest.mark.asyncio
    async def test_focus_all_tables_selected_returns_none(self):
        """When query returns empty list (no included rows), returns None (optimization)."""
        session = AsyncMock()

        # Simulate the case where FocusScope rows exist but fetchall returns
        # no included tables (empty list after filtering is_included == True).
        # The function does: tables = [row[0] for row in result.fetchall()]
        # then: return tables if tables else None
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        session.execute = AsyncMock(return_value=result_mock)

        result = await _get_focus_tables(session)

        # Empty tables list => returns None (focus mode effectively off)
        assert result is None


# ---------------------------------------------------------------------------
# Focus endpoints (direct function calls, not HTTP)
# ---------------------------------------------------------------------------


class TestGetFocusEndpoint:
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.focus.metadata_store")
    async def test_get_focus_empty(self, mock_metadata_store):
        """No focus scopes returns {active: False}."""
        session = AsyncMock()

        # No FocusScope rows
        focus_result = MagicMock()
        focus_result.scalars.return_value.all.return_value = []

        # Metadata entries (all tables in the system)
        entry_a = MagicMock(table_name="sales")
        entry_b = MagicMock(table_name="customers")
        mock_metadata_store.get_all = AsyncMock(return_value=[entry_a, entry_b])

        session.execute = AsyncMock(return_value=focus_result)

        result = await get_focus(session)

        assert result["active"] is False
        assert result["total"] == 2


class TestUpdateFocusEndpoint:
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.focus.metadata_store")
    async def test_put_focus_saves(self, mock_metadata_store):
        """Saves focus scope entries and returns update confirmation."""
        session = AsyncMock()

        # Metadata entries for validation
        entry_a = MagicMock(table_name="sales")
        entry_b = MagicMock(table_name="customers")
        mock_metadata_store.get_all = AsyncMock(return_value=[entry_a, entry_b])

        # Mock the session.execute calls (select for init + delete)
        select_result = MagicMock()
        session.execute = AsyncMock(return_value=select_result)

        body = {
            "tables": [
                {"table_name": "sales", "is_included": True},
                {"table_name": "customers", "is_included": False},
            ]
        }

        result = await update_focus(body, session)

        assert result["status"] == "updated"
        assert result["total"] == 2
        assert result["included"] == 1
        session.commit.assert_called_once()
        # Two scope objects should have been added
        assert session.add.call_count == 2


class TestClearFocusEndpoint:
    @pytest.mark.asyncio
    async def test_delete_focus_clears(self):
        """Clears all focus scopes and returns confirmation."""
        session = AsyncMock()

        delete_result = MagicMock()
        delete_result.rowcount = 3
        session.execute = AsyncMock(return_value=delete_result)

        result = await clear_focus(session)

        assert result["status"] == "cleared"
        assert result["rows_removed"] == 3
        session.commit.assert_called_once()
