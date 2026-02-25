"""Comprehensive tests for the /metadata, /metadata/{table}, and DELETE /metadata/{table} endpoints.

Covers access control based on role hierarchy, response format validation,
error handling, and store interaction verification.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from business_brain.db.connection import get_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session_override():
    """Yield a mock async session for dependency injection."""
    session = AsyncMock()
    yield session


def _make_entry(table_name, uploaded_by=None, uploaded_by_role=None):
    """Create a mock MetadataEntry with the given attributes."""
    entry = MagicMock()
    entry.table_name = table_name
    entry.uploaded_by = uploaded_by
    entry.uploaded_by_role = uploaded_by_role
    entry.description = f"Description of {table_name}"
    entry.columns_metadata = [{"name": "id", "type": "int64"}]
    return entry


def _user(sub, role):
    """Build a user dict matching JWT payload shape."""
    return {"sub": sub, "role": role}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Create a TestClient that skips real DB startup events and background discovery."""
    with patch("business_brain.action.routers.data.run_discovery_background", new_callable=AsyncMock):
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


# ---------------------------------------------------------------------------
# GET /metadata — List metadata
# ---------------------------------------------------------------------------


class TestListMetadata:
    """Tests for the GET /metadata endpoint."""

    @patch("business_brain.action.routers.data.metadata_store")
    def test_list_metadata_no_auth_returns_all(self, mock_store, client):
        """No auth (user=None) returns all metadata entries via get_all."""
        entries = [
            _make_entry("sales"),
            _make_entry("orders"),
            _make_entry("inventory"),
        ]
        mock_store.get_all = AsyncMock(return_value=entries)

        resp = client.get("/metadata")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        table_names = [d["table_name"] for d in data]
        assert "sales" in table_names
        assert "orders" in table_names
        assert "inventory" in table_names
        # When user=None, _get_accessible_tables returns None, so get_all is used
        mock_store.get_all.assert_called_once()

    @patch("business_brain.action.routers.data.metadata_store")
    def test_list_metadata_admin_returns_all(self, mock_store, client):
        """Admin role gets full access (None from _get_accessible_tables) -> get_all."""
        from business_brain.action.api import app, get_current_user

        entries = [
            _make_entry("sales", uploaded_by="v1", uploaded_by_role="viewer"),
            _make_entry("secrets", uploaded_by="admin1", uploaded_by_role="admin"),
        ]
        mock_store.get_all = AsyncMock(return_value=entries)

        async def _admin_user():
            return _user("admin1", "admin")

        app.dependency_overrides[get_current_user] = _admin_user
        try:
            resp = client.get("/metadata")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            table_names = [d["table_name"] for d in data]
            assert "sales" in table_names
            assert "secrets" in table_names
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @patch("business_brain.action.routers.data.metadata_store")
    def test_list_metadata_viewer_filters(self, mock_store, client):
        """Viewer gets filtered results: only own uploads and legacy tables."""
        from business_brain.action.api import app, get_current_user

        all_entries = [
            _make_entry("legacy_data", uploaded_by=None, uploaded_by_role=None),
            _make_entry("viewer_upload", uploaded_by="v1", uploaded_by_role="viewer"),
            _make_entry("admin_data", uploaded_by="admin1", uploaded_by_role="admin"),
        ]
        # get_all is called by _get_accessible_tables for access control check
        mock_store.get_all = AsyncMock(return_value=all_entries)

        filtered_entries = [
            _make_entry("legacy_data", uploaded_by=None, uploaded_by_role=None),
            _make_entry("viewer_upload", uploaded_by="v1", uploaded_by_role="viewer"),
        ]
        mock_store.get_filtered = AsyncMock(return_value=filtered_entries)

        async def _viewer_user():
            return _user("v1", "viewer")

        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            resp = client.get("/metadata")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            table_names = [d["table_name"] for d in data]
            assert "legacy_data" in table_names
            assert "viewer_upload" in table_names
            assert "admin_data" not in table_names
            # get_filtered is called with the accessible table list
            mock_store.get_filtered.assert_called_once()
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @patch("business_brain.action.routers.data.metadata_store")
    def test_list_metadata_operator_sees_lower_roles(self, mock_store, client):
        """Operator sees own + viewer uploads + legacy, but not admin/manager."""
        from business_brain.action.api import app, get_current_user

        all_entries = [
            _make_entry("legacy_tbl", uploaded_by=None, uploaded_by_role=None),
            _make_entry("op_own", uploaded_by="op1", uploaded_by_role="operator"),
            _make_entry("viewer_tbl", uploaded_by="v1", uploaded_by_role="viewer"),
            _make_entry("other_op", uploaded_by="op2", uploaded_by_role="operator"),
            _make_entry("mgr_tbl", uploaded_by="mgr1", uploaded_by_role="manager"),
            _make_entry("admin_tbl", uploaded_by="admin1", uploaded_by_role="admin"),
        ]
        mock_store.get_all = AsyncMock(return_value=all_entries)

        # get_filtered should be called with ["legacy_tbl", "op_own", "viewer_tbl", "other_op"]
        accessible_entries = [
            _make_entry("legacy_tbl"),
            _make_entry("op_own"),
            _make_entry("viewer_tbl"),
            _make_entry("other_op"),
        ]
        mock_store.get_filtered = AsyncMock(return_value=accessible_entries)

        async def _operator_user():
            return _user("op1", "operator")

        app.dependency_overrides[get_current_user] = _operator_user
        try:
            resp = client.get("/metadata")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 4
            table_names = [d["table_name"] for d in data]
            assert "legacy_tbl" in table_names
            assert "op_own" in table_names
            assert "viewer_tbl" in table_names
            assert "other_op" in table_names
            assert "mgr_tbl" not in table_names
            assert "admin_tbl" not in table_names

            # Verify get_filtered was called with the correct accessible list
            call_args = mock_store.get_filtered.call_args
            accessible_arg = call_args[0][1]  # second positional arg
            assert "legacy_tbl" in accessible_arg
            assert "op_own" in accessible_arg
            assert "viewer_tbl" in accessible_arg
            assert "other_op" in accessible_arg
            assert "mgr_tbl" not in accessible_arg
            assert "admin_tbl" not in accessible_arg
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @patch("business_brain.action.routers.data.metadata_store")
    def test_list_metadata_empty_store(self, mock_store, client):
        """No metadata entries in store returns empty list."""
        mock_store.get_all = AsyncMock(return_value=[])

        resp = client.get("/metadata")

        assert resp.status_code == 200
        data = resp.json()
        assert data == []

    @patch("business_brain.action.routers.data.metadata_store")
    def test_list_metadata_response_format(self, mock_store, client):
        """Each entry in the response has exactly table_name, description, columns keys."""
        entries = [
            _make_entry("sales"),
            _make_entry("orders"),
        ]
        mock_store.get_all = AsyncMock(return_value=entries)

        resp = client.get("/metadata")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        for entry in data:
            assert set(entry.keys()) == {"table_name", "description", "columns"}
            assert isinstance(entry["table_name"], str)
            assert isinstance(entry["description"], str)
            assert isinstance(entry["columns"], list)

    @patch("business_brain.action.routers.data.metadata_store")
    def test_list_metadata_db_error_returns_empty(self, mock_store, client):
        """Database exception in list_metadata returns empty list (graceful degradation)."""
        mock_store.get_all = AsyncMock(side_effect=RuntimeError("DB connection failed"))

        resp = client.get("/metadata")

        assert resp.status_code == 200
        data = resp.json()
        assert data == []


# ---------------------------------------------------------------------------
# GET /metadata/{table} — Get single table metadata
# ---------------------------------------------------------------------------


class TestGetTableMetadata:
    """Tests for the GET /metadata/{table} endpoint."""

    @patch("business_brain.action.routers.data.metadata_store")
    def test_get_metadata_no_auth_succeeds(self, mock_store, client):
        """No auth (user=None) can access any table metadata."""
        entry = _make_entry("sales")
        mock_store.get_by_table = AsyncMock(return_value=entry)

        resp = client.get("/metadata/sales")

        assert resp.status_code == 200
        data = resp.json()
        assert data["table_name"] == "sales"
        assert data["description"] == "Description of sales"
        mock_store.get_by_table.assert_called_once()

    @patch("business_brain.action.routers.data.metadata_store")
    def test_get_metadata_admin_succeeds(self, mock_store, client):
        """Admin can access any table metadata regardless of uploader."""
        from business_brain.action.api import app, get_current_user

        entry = _make_entry("viewer_data", uploaded_by="v1", uploaded_by_role="viewer")
        mock_store.get_by_table = AsyncMock(return_value=entry)

        async def _admin_user():
            return _user("admin1", "admin")

        app.dependency_overrides[get_current_user] = _admin_user
        try:
            resp = client.get("/metadata/viewer_data")
            assert resp.status_code == 200
            data = resp.json()
            assert data["table_name"] == "viewer_data"
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @patch("business_brain.action.routers.data.metadata_store")
    def test_get_metadata_viewer_blocked(self, mock_store, client):
        """Viewer cannot see admin-uploaded table -> 'Table not found'."""
        from business_brain.action.api import app, get_current_user

        all_entries = [
            _make_entry("admin_secrets", uploaded_by="admin1", uploaded_by_role="admin"),
        ]
        mock_store.get_all = AsyncMock(return_value=all_entries)

        async def _viewer_user():
            return _user("v1", "viewer")

        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            resp = client.get("/metadata/admin_secrets")
            assert resp.status_code == 200
            data = resp.json()
            assert data["error"] == "Table not found"
            # get_by_table should NOT be called since access check failed first
            mock_store.get_by_table.assert_not_called()
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @patch("business_brain.action.routers.data.metadata_store")
    def test_get_metadata_viewer_own_table(self, mock_store, client):
        """Viewer can see their own uploaded table."""
        from business_brain.action.api import app, get_current_user

        all_entries = [
            _make_entry("my_report", uploaded_by="v1", uploaded_by_role="viewer"),
        ]
        mock_store.get_all = AsyncMock(return_value=all_entries)

        entry = _make_entry("my_report", uploaded_by="v1", uploaded_by_role="viewer")
        mock_store.get_by_table = AsyncMock(return_value=entry)

        async def _viewer_user():
            return _user("v1", "viewer")

        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            resp = client.get("/metadata/my_report")
            assert resp.status_code == 200
            data = resp.json()
            assert data["table_name"] == "my_report"
            assert "error" not in data
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @patch("business_brain.action.routers.data.metadata_store")
    def test_get_metadata_viewer_sees_legacy(self, mock_store, client):
        """Viewer can see legacy table (no uploader recorded)."""
        from business_brain.action.api import app, get_current_user

        all_entries = [
            _make_entry("old_data", uploaded_by=None, uploaded_by_role=None),
        ]
        mock_store.get_all = AsyncMock(return_value=all_entries)

        entry = _make_entry("old_data", uploaded_by=None, uploaded_by_role=None)
        mock_store.get_by_table = AsyncMock(return_value=entry)

        async def _viewer_user():
            return _user("v1", "viewer")

        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            resp = client.get("/metadata/old_data")
            assert resp.status_code == 200
            data = resp.json()
            assert data["table_name"] == "old_data"
            assert "error" not in data
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @patch("business_brain.action.routers.data.metadata_store")
    def test_get_metadata_nonexistent(self, mock_store, client):
        """Requesting a nonexistent table returns 'Table not found'."""
        mock_store.get_by_table = AsyncMock(return_value=None)

        resp = client.get("/metadata/nonexistent_table")

        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "Table not found"

    @patch("business_brain.action.routers.data.metadata_store")
    def test_get_metadata_response_format(self, mock_store, client):
        """Successful response has exactly table_name, description, columns keys."""
        entry = _make_entry("formatted_table")
        mock_store.get_by_table = AsyncMock(return_value=entry)

        resp = client.get("/metadata/formatted_table")

        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"table_name", "description", "columns"}
        assert data["table_name"] == "formatted_table"
        assert data["description"] == "Description of formatted_table"
        assert data["columns"] == [{"name": "id", "type": "int64"}]


# ---------------------------------------------------------------------------
# DELETE /metadata/{table} — Delete table metadata
# ---------------------------------------------------------------------------


class TestDeleteTableMetadata:
    """Tests for the DELETE /metadata/{table} endpoint."""

    @patch("business_brain.action.routers.data.metadata_store")
    def test_delete_metadata_existing(self, mock_store, client):
        """Deleting an existing table returns status=deleted."""
        mock_store.delete = AsyncMock(return_value=True)

        resp = client.delete("/metadata/old_report")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "deleted", "table": "old_report"}

    @patch("business_brain.action.routers.data.metadata_store")
    def test_delete_metadata_nonexistent(self, mock_store, client):
        """Deleting a nonexistent table returns 'Table not found' error."""
        mock_store.delete = AsyncMock(return_value=False)

        resp = client.delete("/metadata/ghost_table")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"error": "Table not found"}

    @patch("business_brain.action.routers.data.metadata_store")
    def test_delete_metadata_db_error(self, mock_store, client):
        """Database exception during delete returns error dict."""
        mock_store.delete = AsyncMock(side_effect=RuntimeError("Connection lost"))

        resp = client.delete("/metadata/broken_table")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"error": "Failed to delete table metadata"}

    @patch("business_brain.action.routers.data.metadata_store")
    def test_delete_metadata_calls_store_delete(self, mock_store, client):
        """Verify metadata_store.delete is called with session and correct table name."""
        mock_store.delete = AsyncMock(return_value=True)

        resp = client.delete("/metadata/target_table")

        assert resp.status_code == 200
        mock_store.delete.assert_called_once()
        call_args = mock_store.delete.call_args
        # First positional arg is the session, second is the table name
        assert call_args[0][1] == "target_table"
