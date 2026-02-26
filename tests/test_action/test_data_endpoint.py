"""Comprehensive tests for GET /data/{table} and PUT /data/{table} endpoints.

Covers paginated read access, inline cell editing, access control,
input validation, sorting, and error handling.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.db.connection import get_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session_override():
    session = AsyncMock()
    yield session


def _make_row(mapping: dict) -> MagicMock:
    """Create a mock row with a _mapping attribute for fetchall results."""
    row = MagicMock()
    row._mapping = mapping
    return row


def _setup_session_for_data(
    session: AsyncMock,
    total: int = 5,
    rows: list[dict] | None = None,
):
    """Configure an AsyncMock session to return pg_class + count + data for get_table_data.

    session.execute is called three times:
      1. pg_class estimate -> fetchone() returns (total,) (small table triggers exact count)
      2. COUNT(*) -> scalar() returns total
      3. SELECT * -> fetchall() returns mock rows
    """
    if rows is None:
        rows = [{"id": 1, "name": "test", "value": 10}]

    # pg_class estimated count (returns small number so exact count is used)
    mock_pg_class_result = MagicMock()
    mock_pg_class_result.fetchone.return_value = (total,)

    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = total

    mock_data_result = MagicMock()
    mock_data_result.fetchall.return_value = [_make_row(r) for r in rows]

    session.execute = AsyncMock(side_effect=[mock_pg_class_result, mock_count_result, mock_data_result])


@pytest.fixture()
def client():
    """Create a TestClient that skips real DB startup events and background discovery."""
    with patch(
        "business_brain.action.api._run_discovery_background",
        new_callable=AsyncMock,
    ):
        from business_brain.action.api import app

        original_startup = list(app.router.on_startup)
        original_shutdown = list(app.router.on_shutdown)
        app.router.on_startup.clear()
        app.router.on_shutdown.clear()

        app.dependency_overrides[get_session] = _mock_session_override

        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            yield c

        app.dependency_overrides.clear()
        app.router.on_startup = original_startup
        app.router.on_shutdown = original_shutdown


# ===================================================================
# GET /data/{table} tests
# ===================================================================


class TestGetTableData:
    """Tests for the GET /data/{table} endpoint."""

    # 1. No auth (user=None) -> no access check, data returned
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_no_auth_returns_data(self, mock_accessible):
        """When user is None, _get_accessible_tables returns None (no filtering)
        and data is returned successfully."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        _setup_session_for_data(
            session,
            total=3,
            rows=[
                {"id": 1, "name": "Alice", "value": 100},
                {"id": 2, "name": "Bob", "value": 200},
                {"id": 3, "name": "Carol", "value": 300},
            ],
        )
        mock_accessible.return_value = None  # no access check

        result = await get_table_data(
            table="sales",
            page=1,
            page_size=50,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=None,
        )

        assert "rows" in result
        assert len(result["rows"]) == 3
        assert result["total"] == 3
        mock_accessible.assert_called_once_with(session, None)

    # 2. Admin user -> full access
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_admin_returns_data(self, mock_accessible):
        """Admin role gets None from _get_accessible_tables (full access),
        so data is returned without restriction."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        _setup_session_for_data(session, total=1, rows=[{"id": 1, "metric": "revenue"}])
        mock_accessible.return_value = None  # admin = full access

        admin_user = {"sub": "admin-1", "role": "admin", "email": "admin@test.com"}
        result = await get_table_data(
            table="financials",
            page=1,
            page_size=50,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=admin_user,
        )

        assert "rows" in result
        assert result["total"] == 1
        assert result["rows"][0]["metric"] == "revenue"

    # 3. Viewer accessing admin's table -> "Access denied"
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_viewer_blocked(self, mock_accessible):
        """Viewer trying to access a table not in their accessible list gets
        an 'Access denied' error."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        # Viewer can only see "viewer_uploads" — not "admin_secrets"
        mock_accessible.return_value = ["viewer_uploads"]

        viewer = {"sub": "v1", "role": "viewer", "email": "viewer@test.com"}
        result = await get_table_data(
            table="admin_secrets",
            page=1,
            page_size=50,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=viewer,
        )

        assert "error" in result
        assert "Access denied" in result["error"]
        assert "admin_secrets" in result["error"]

    # 4. Viewer accessing own upload -> data returned
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_viewer_own_table(self, mock_accessible):
        """Viewer accessing their own uploaded table should get data back."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        _setup_session_for_data(session, total=2, rows=[
            {"id": 1, "product": "Widget"},
            {"id": 2, "product": "Gadget"},
        ])
        mock_accessible.return_value = ["my_products"]  # viewer's own table

        viewer = {"sub": "v1", "role": "viewer"}
        result = await get_table_data(
            table="my_products",
            page=1,
            page_size=50,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=viewer,
        )

        assert "rows" in result
        assert len(result["rows"]) == 2
        assert result["total"] == 2

    # 5. Legacy table (no uploader) visible to all roles
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_legacy_table_visible(self, mock_accessible):
        """A legacy table (no uploader) appears in the accessible list for
        any authenticated user, so data is returned."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        _setup_session_for_data(session, total=1, rows=[{"id": 1, "old_col": "legacy_data"}])
        # Accessible list includes the legacy table
        mock_accessible.return_value = ["legacy_report", "own_stuff"]

        viewer = {"sub": "v1", "role": "viewer"}
        result = await get_table_data(
            table="legacy_report",
            page=1,
            page_size=50,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=viewer,
        )

        assert "rows" in result
        assert result["rows"][0]["old_col"] == "legacy_data"

    # 6. Invalid table name (special chars only) -> "Invalid table name"
    @pytest.mark.asyncio
    async def test_data_invalid_table_name(self):
        """Table name with only special characters is sanitized to empty string,
        returning an 'Invalid table name' error."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        result = await get_table_data(
            table="!@#$%^&*()",
            page=1,
            page_size=50,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=None,
        )

        assert result == {"error": "Invalid table name"}

    # 7. Pagination params passed correctly
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_pagination_params(self, mock_accessible):
        """page=3, page_size=10 should produce offset=20 and correct response metadata."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        _setup_session_for_data(session, total=50, rows=[
            {"id": 21, "col": "row21"},
        ])
        mock_accessible.return_value = None

        result = await get_table_data(
            table="big_table",
            page=3,
            page_size=10,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=None,
        )

        assert result["page"] == 3
        assert result["page_size"] == 10
        assert result["total"] == 50

        # Verify the session.execute call used correct LIMIT/OFFSET
        # Third call is the SELECT * (after pg_class + COUNT)
        select_call = session.execute.call_args_list[2]
        params = select_call[1] if select_call[1] else select_call[0][1]
        # The params are passed as a dict: {"limit": 10, "offset": 20}
        assert params["limit"] == 10
        assert params["offset"] == 20

    # 8. Sort params work
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_sort_params(self, mock_accessible):
        """sort_by='revenue' and sort_dir='desc' produce ORDER BY clause in the query."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        _setup_session_for_data(session, total=2, rows=[
            {"id": 2, "revenue": 500},
            {"id": 1, "revenue": 100},
        ])
        mock_accessible.return_value = None

        result = await get_table_data(
            table="sales",
            page=1,
            page_size=50,
            sort_by="revenue",
            sort_dir="desc",
            session=session,
            user=None,
        )

        assert "rows" in result
        assert len(result["rows"]) == 2

        # Verify the SQL query includes ORDER BY
        # Third call is the SELECT * (after pg_class + COUNT)
        select_call = session.execute.call_args_list[2]
        query_text = str(select_call[0][0].text)
        assert 'ORDER BY "revenue" DESC' in query_text

    # 9. Response format has all expected keys
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_response_format(self, mock_accessible):
        """Response dict must contain rows, total, page, page_size, and columns."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        _setup_session_for_data(session, total=1, rows=[
            {"id": 1, "name": "test", "value": 42},
        ])
        mock_accessible.return_value = None

        result = await get_table_data(
            table="metrics",
            page=1,
            page_size=50,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=None,
        )

        expected_keys = {"rows", "total", "total_exact", "page", "page_size", "columns"}
        assert set(result.keys()) == expected_keys
        assert isinstance(result["rows"], list)
        assert isinstance(result["total"], int)
        assert isinstance(result["page"], int)
        assert isinstance(result["page_size"], int)
        assert isinstance(result["columns"], list)
        assert result["columns"] == ["id", "name", "value"]

    # 10. DB error returns error dict
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.get_accessible_tables")
    async def test_data_db_error_returns_error(self, mock_accessible):
        """When session.execute raises an exception, the endpoint returns
        {'error': '...'} instead of crashing."""
        from business_brain.action.api import get_table_data

        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=Exception('relation "nonexistent" does not exist')
        )
        session.rollback = AsyncMock()
        mock_accessible.return_value = None

        result = await get_table_data(
            table="nonexistent",
            page=1,
            page_size=50,
            sort_by=None,
            sort_dir="asc",
            session=session,
            user=None,
        )

        assert "error" in result
        assert "nonexistent" in result["error"]
        session.rollback.assert_called_once()


# ===================================================================
# PUT /data/{table} tests
# ===================================================================


class TestUpdateCell:
    """Tests for the PUT /data/{table} endpoint."""

    # 11. Valid update returns {"status": "updated"}
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.metadata_store")
    async def test_update_cell_success(self, mock_metadata_store):
        """A valid cell update with all required fields returns success."""
        from business_brain.action.api import update_cell

        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()

        entry = MagicMock()
        entry.columns_metadata = [{"name": "order_id"}, {"name": "amount"}]
        mock_metadata_store.get_by_table = AsyncMock(return_value=entry)

        body = {"row_id": 42, "column": "amount", "value": 999.99}
        result = await update_cell(table="orders", body=body, session=session)

        assert result["status"] == "updated"
        assert result["table"] == "orders"
        assert result["row_id"] == 42
        assert result["column"] == "amount"
        session.commit.assert_called_once()

    # 12. Missing required fields -> error
    @pytest.mark.asyncio
    async def test_update_cell_missing_fields(self):
        """Missing row_id, column, or empty table should return an error."""
        from business_brain.action.api import update_cell

        session = AsyncMock()

        # Missing row_id
        body_no_rowid = {"column": "name", "value": "Alice"}
        result = await update_cell(table="users", body=body_no_rowid, session=session)
        assert "error" in result
        assert "Missing required fields" in result["error"]

        # Missing column
        body_no_col = {"row_id": 1, "value": "Alice"}
        result = await update_cell(table="users", body=body_no_col, session=session)
        assert "error" in result
        assert "Missing required fields" in result["error"]

        # Invalid table name (all special chars -> empty after sanitization)
        body_valid = {"row_id": 1, "column": "name", "value": "Alice"}
        result = await update_cell(table="!!!", body=body_valid, session=session)
        assert "error" in result
        assert "Missing required fields" in result["error"]

    # 13. PK column detected from database or metadata fallback
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.metadata_store")
    async def test_update_cell_uses_pk_from_metadata(self, mock_metadata_store):
        """When pg_index query fails (mock), falls back to metadata_store's
        first column as PK for the WHERE clause."""
        from business_brain.action.api import update_cell

        session = AsyncMock()
        session.add = MagicMock()

        # pg_index query fails (first call), then old-value SELECT, then UPDATE
        pg_index_error = Exception("pg_index not available in mock")
        mock_old_value = MagicMock()
        mock_old_value.fetchone.return_value = ("old_price",)
        mock_update_result = MagicMock()

        session.execute = AsyncMock(
            side_effect=[pg_index_error, mock_old_value, mock_update_result]
        )
        session.commit = AsyncMock()

        entry = MagicMock()
        entry.columns_metadata = [{"name": "product_sku"}, {"name": "price"}]
        mock_metadata_store.get_by_table = AsyncMock(return_value=entry)

        body = {"row_id": "SKU-001", "column": "price", "value": 29.99}
        result = await update_cell(table="products", body=body, session=session)

        assert result["status"] == "updated"

        # The UPDATE query (3rd execute call) should use product_sku as PK
        update_call = session.execute.call_args_list[2]
        query_text = str(update_call[0][0].text)
        assert '"product_sku"' in query_text
        params = update_call[0][1] if len(update_call[0]) > 1 else update_call[1]
        assert params["pk"] == "SKU-001"
        assert params["val"] == 29.99

    # 14. No metadata -> uses "id" as PK fallback
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.metadata_store")
    async def test_update_cell_fallback_pk_id(self, mock_metadata_store):
        """When pg_index and metadata_store both fail, uses 'id' as PK fallback."""
        from business_brain.action.api import update_cell

        session = AsyncMock()
        session.add = MagicMock()

        # pg_index query fails, old-value SELECT, then UPDATE
        pg_index_error = Exception("pg_index not available")
        mock_old_value = MagicMock()
        mock_old_value.fetchone.return_value = None
        mock_update_result = MagicMock()

        session.execute = AsyncMock(
            side_effect=[pg_index_error, mock_old_value, mock_update_result]
        )
        session.commit = AsyncMock()

        mock_metadata_store.get_by_table = AsyncMock(return_value=None)

        body = {"row_id": 7, "column": "status", "value": "active"}
        result = await update_cell(table="accounts", body=body, session=session)

        assert result["status"] == "updated"

        # The UPDATE query should use "id" as the PK column
        update_call = session.execute.call_args_list[2]
        query_text = str(update_call[0][0].text)
        assert '"id"' in query_text
        assert 'WHERE "id"' in query_text

    # 15. DB error during update -> {"error": "..."}
    @pytest.mark.asyncio
    @patch("business_brain.action.routers.data.metadata_store")
    async def test_update_cell_db_error(self, mock_metadata_store):
        """When session.execute raises during UPDATE, the endpoint returns
        an error dict and rolls back the session."""
        from business_brain.action.api import update_cell

        session = AsyncMock()
        # All execute calls raise (pg_index, old value, and update all fail)
        session.execute = AsyncMock(
            side_effect=Exception('column "nonexistent_col" does not exist')
        )
        session.rollback = AsyncMock()

        entry = MagicMock()
        entry.columns_metadata = [{"name": "id"}]
        mock_metadata_store.get_by_table = AsyncMock(return_value=entry)

        body = {"row_id": 1, "column": "nonexistent_col", "value": "oops"}
        result = await update_cell(table="sales", body=body, session=session)

        assert "error" in result
        assert "nonexistent_col" in result["error"]
        session.rollback.assert_called_once()
