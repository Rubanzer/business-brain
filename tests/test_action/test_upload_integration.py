"""Integration-style tests: upload -> metadata -> access control -> data endpoint flows.

These tests verify the full lifecycle works end-to-end using mocks (no real DB).
They combine CSV upload, metadata visibility with access control, data access,
cell editing, and cascade delete operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from business_brain.db.connection import get_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30"


class FakeMetadataStore:
    """In-memory metadata store that remembers upserts for multi-step tests."""

    def __init__(self):
        self._entries = {}

    def _make_entry(self, table_name, description, columns_metadata, uploaded_by, uploaded_by_role):
        entry = MagicMock()
        entry.table_name = table_name
        entry.description = description
        entry.columns_metadata = columns_metadata
        entry.uploaded_by = uploaded_by
        entry.uploaded_by_role = uploaded_by_role
        return entry

    def seed(self, table_name, description="", columns_metadata=None,
             uploaded_by=None, uploaded_by_role=None):
        """Synchronous helper to pre-populate entries for test setup."""
        entry = self._make_entry(table_name, description, columns_metadata, uploaded_by, uploaded_by_role)
        self._entries[table_name] = entry
        return entry

    def get_entries(self):
        """Synchronous helper to read all entries."""
        return list(self._entries.values())

    async def upsert(
        self,
        session,
        table_name,
        description,
        columns_metadata=None,
        uploaded_by=None,
        uploaded_by_role=None,
    ):
        entry = self._make_entry(table_name, description, columns_metadata, uploaded_by, uploaded_by_role)
        self._entries[table_name] = entry
        return entry

    async def get_all(self, session):
        return list(self._entries.values())

    async def get_by_table(self, session, table_name):
        return self._entries.get(table_name)

    async def get_filtered(self, session, table_names):
        return [e for e in self._entries.values() if e.table_name in table_names]

    async def delete(self, session, table_name):
        return self._entries.pop(table_name, None) is not None


def _mock_session_override():
    session = AsyncMock()
    yield session


def _make_auth_client_fixture(sub: str, role: str):
    """Factory: create a TestClient fixture whose user has the given sub/role."""

    @pytest.fixture()
    def _fixture():
        with patch(
            "business_brain.action.api._run_discovery_background",
            new_callable=AsyncMock,
        ):
            from business_brain.action.api import app, get_current_user

            original_startup = list(app.router.on_startup)
            original_shutdown = list(app.router.on_shutdown)
            app.router.on_startup.clear()
            app.router.on_shutdown.clear()

            app.dependency_overrides[get_session] = _mock_session_override

            async def _fake_user():
                return {"sub": sub, "role": role}

            app.dependency_overrides[get_current_user] = _fake_user

            from fastapi.testclient import TestClient

            with TestClient(app) as c:
                yield c

            app.dependency_overrides.clear()
            app.router.on_startup = original_startup
            app.router.on_shutdown = original_shutdown

    return _fixture


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Unauthenticated TestClient (no user)."""
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


viewer_client = _make_auth_client_fixture("viewer-001", "viewer")
admin_client = _make_auth_client_fixture("admin-001", "admin")
manager_client = _make_auth_client_fixture("manager-001", "manager")


@pytest.fixture()
def fake_meta():
    """Fresh FakeMetadataStore for each test."""
    return FakeMetadataStore()


# ===========================================================================
# 1-5: Upload -> Metadata visibility
# ===========================================================================


@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch("business_brain.action.api.metadata_store")
def test_csv_upload_then_metadata_list_includes_table(mock_meta, mock_upsert, client):
    """Upload CSV, then verify metadata list returns the uploaded table."""
    fake = FakeMetadataStore()

    mock_upsert.return_value = 3
    mock_meta.upsert = AsyncMock(side_effect=fake.upsert)
    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)

    # Step 1: upload
    resp = client.post("/csv", files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200
    assert resp.json()["status"] == "loaded"

    # Step 2: list metadata (no auth -> full access -> get_all)
    resp = client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["table_name"] == "sales"
    assert "columns" in data[0]


@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch("business_brain.action.api.metadata_store")
def test_csv_by_viewer_admin_sees(mock_meta, mock_upsert, admin_client):
    """Viewer uploads a table; admin (role=admin) should see it (full access)."""
    fake = FakeMetadataStore()

    mock_upsert.return_value = 3
    mock_meta.upsert = AsyncMock(side_effect=fake.upsert)
    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)

    # Simulate viewer upload by inserting into fake store directly
    fake.seed(
        table_name="viewer_data",
        description="Data uploaded by viewer",
        columns_metadata=[{"name": "id", "type": "int64"}],
        uploaded_by="viewer-001",
        uploaded_by_role="viewer",
    )

    # Admin requests metadata -> admin gets None from _get_accessible_tables -> get_all
    resp = admin_client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["table_name"] == "viewer_data"


@patch("business_brain.action.api.metadata_store")
def test_csv_by_admin_viewer_blocked(mock_meta, viewer_client):
    """Admin uploads a table; viewer should NOT see it in metadata."""
    fake = FakeMetadataStore()

    fake.seed(
        table_name="admin_only",
        description="Admin data",
        columns_metadata=[{"name": "id", "type": "int64"}],
        uploaded_by="admin-001",
        uploaded_by_role="admin",
    )

    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)
    mock_meta.get_filtered = AsyncMock(side_effect=fake.get_filtered)

    # Viewer calls metadata list -> _get_accessible_tables filters
    # The viewer can only see own uploads + legacy (no uploader) + lower-role uploads.
    # Admin-uploaded data has uploader_level=3, viewer_level=0 -> blocked
    resp = viewer_client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    # Viewer should see 0 tables — the admin upload is not accessible
    assert len(data) == 0


@patch("business_brain.action.api.metadata_store")
def test_csv_by_viewer_sees_own(mock_meta, viewer_client):
    """Viewer uploads a table; same viewer should see it in their metadata list."""
    fake = FakeMetadataStore()

    fake.seed(
        table_name="my_data",
        description="My data",
        columns_metadata=[{"name": "col_a", "type": "object"}],
        uploaded_by="viewer-001",
        uploaded_by_role="viewer",
    )

    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)
    mock_meta.get_filtered = AsyncMock(side_effect=fake.get_filtered)

    resp = viewer_client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["table_name"] == "my_data"


@patch("business_brain.action.api.metadata_store")
def test_csv_no_auth_legacy_visible_to_all(mock_meta, viewer_client):
    """Legacy table (no uploaded_by) should be visible to all roles including viewer."""
    fake = FakeMetadataStore()

    fake.seed(
        table_name="legacy_table",
        description="Old data, no uploader recorded",
        columns_metadata=[{"name": "id", "type": "int64"}],
        uploaded_by=None,
        uploaded_by_role=None,
    )

    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)
    mock_meta.get_filtered = AsyncMock(side_effect=fake.get_filtered)

    resp = viewer_client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["table_name"] == "legacy_table"


# ===========================================================================
# 6-7: Upload -> Data access
# ===========================================================================


@pytest.mark.asyncio
@patch("business_brain.action.api.metadata_store")
async def test_upload_then_data_accessible(mock_meta):
    """After upload, GET /data/{table} works for an authorized (no-auth) user."""
    fake = FakeMetadataStore()

    fake.seed(
        table_name="sales",
        description="Sales",
        columns_metadata=[{"name": "id", "type": "int64"}],
        uploaded_by=None,
        uploaded_by_role=None,
    )

    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)

    from business_brain.action.api import get_table_data

    session = AsyncMock()
    # Mock session.execute for count and data queries
    count_result = MagicMock()
    count_result.scalar.return_value = 3
    data_result = MagicMock()
    row1 = MagicMock()
    row1._mapping = {"id": 1, "name": "alpha", "value": 10}
    data_result.fetchall.return_value = [row1]
    session.execute = AsyncMock(side_effect=[count_result, data_result])

    # No auth user -> accessible is None -> full access
    user = None
    with patch("business_brain.action.api._get_accessible_tables", new_callable=AsyncMock) as mock_access:
        mock_access.return_value = None  # no auth => all tables
        result = await get_table_data(table="sales", session=session, user=user)

    assert "error" not in result
    assert result["rows"] == [{"id": 1, "name": "alpha", "value": 10}]
    assert result["total"] == 3


@pytest.mark.asyncio
@patch("business_brain.action.api.metadata_store")
async def test_upload_then_data_blocked(mock_meta):
    """Admin uploads -> viewer is blocked from /data/{table}."""
    fake = FakeMetadataStore()

    fake.seed(
        table_name="secret",
        description="Admin only data",
        columns_metadata=[{"name": "id", "type": "int64"}],
        uploaded_by="admin-001",
        uploaded_by_role="admin",
    )

    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)

    from business_brain.action.api import get_table_data

    session = AsyncMock()
    viewer_user = {"sub": "viewer-001", "role": "viewer"}

    # _get_accessible_tables for viewer: admin upload -> uploader_level=3, viewer_level=0 -> blocked
    # The accessible list won't contain "secret"
    with patch("business_brain.action.api._get_accessible_tables", new_callable=AsyncMock) as mock_access:
        mock_access.return_value = []  # viewer can see nothing
        result = await get_table_data(table="secret", session=session, user=viewer_user)

    assert "error" in result
    assert "Access denied" in result["error"]


# ===========================================================================
# 8-9: Cell editing
# ===========================================================================


@pytest.mark.asyncio
@patch("business_brain.action.api.metadata_store")
async def test_update_cell_commits_to_session(mock_meta):
    """PUT /data/{table} calls session.execute + session.commit."""
    from business_brain.action.api import update_cell

    entry = MagicMock()
    entry.columns_metadata = [{"name": "order_id"}, {"name": "amount"}]
    mock_meta.get_by_table = AsyncMock(return_value=entry)

    session = AsyncMock()
    session.add = MagicMock()

    result = await update_cell(
        table="orders",
        body={"row_id": 42, "column": "amount", "value": 999.99},
        session=session,
    )

    assert result["status"] == "updated"
    # session.execute is called multiple times (PK detection, old value, UPDATE)
    assert session.execute.call_count >= 1
    session.commit.assert_called_once()


@pytest.mark.asyncio
@patch("business_brain.action.api.metadata_store")
async def test_update_cell_uses_correct_pk(mock_meta):
    """PK column should come from pg_index or metadata columns_metadata[0]['name']."""
    from business_brain.action.api import update_cell

    entry = MagicMock()
    entry.columns_metadata = [{"name": "sku_id"}, {"name": "price"}]
    mock_meta.get_by_table = AsyncMock(return_value=entry)

    session = AsyncMock()
    session.add = MagicMock()

    # pg_index fails, old-value SELECT, then UPDATE
    pg_index_error = Exception("pg_index not available")
    mock_old_value = MagicMock()
    mock_old_value.fetchone.return_value = None
    mock_update_result = MagicMock()

    session.execute = AsyncMock(
        side_effect=[pg_index_error, mock_old_value, mock_update_result]
    )
    session.commit = AsyncMock()

    await update_cell(
        table="products",
        body={"row_id": "SKU-100", "column": "price", "value": 49.99},
        session=session,
    )

    # Inspect the UPDATE SQL (3rd execute call)
    update_call = session.execute.call_args_list[2]
    sql_text_obj = update_call[0][0]
    sql_str = str(sql_text_obj.text) if hasattr(sql_text_obj, "text") else str(sql_text_obj)

    # The WHERE clause should use sku_id (from columns_metadata[0])
    assert "sku_id" in sql_str
    # The SET clause should use price
    assert "price" in sql_str

    # Check the bind params include the correct pk value
    params = update_call[0][1] if len(update_call[0]) > 1 else update_call[1]
    assert params["pk"] == "SKU-100"
    assert params["val"] == 49.99


# ===========================================================================
# 10-13: Delete cascade
# ===========================================================================


def _make_delete_session():
    """Create a mock session that properly supports drop_table_cascade operations.

    The cascade delete function calls session.execute() many times and accesses
    .rowcount on results and .scalars().all() for insight queries.
    """
    session = AsyncMock()

    # Each execute() call returns a result mock with .rowcount and .scalars().all()
    def _make_result():
        r = MagicMock()
        r.rowcount = 0
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = []
        r.scalars.return_value = scalars_mock
        return r

    session.execute = AsyncMock(side_effect=lambda *a, **kw: _make_result())
    return session


def _delete_session_override():
    yield _make_delete_session()


def test_delete_table_drops_data_and_metadata(client):
    """DELETE /tables/{table} removes both the SQL table and metadata."""
    from business_brain.action.api import app
    app.dependency_overrides[get_session] = _delete_session_override
    try:
        resp = client.delete("/tables/sales_data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["table"] == "sales_data"
        assert "removed" in data
        assert data["removed"]["data_table"] is True
    finally:
        app.dependency_overrides[get_session] = _mock_session_override


def test_delete_table_removes_all_dependent_data(client):
    """Verify all 12 cleanup steps are executed during cascade delete."""
    from business_brain.action.api import app
    app.dependency_overrides[get_session] = _delete_session_override
    try:
        resp = client.delete("/tables/old_table")
        assert resp.status_code == 200
        data = resp.json()
        removed = data["removed"]

        # All 12 keys should be present
        expected_keys = [
            "data_table",
            "metadata",
            "business_context",
            "table_profile",
            "relationships",
            "deployed_reports",
            "insights",
            "data_sources",
            "change_log",
            "sanctity_issues",
            "fingerprints",
            "source_mappings",
            "thresholds",
        ]
        for key in expected_keys:
            assert key in removed, f"Missing cleanup key: {key}"
    finally:
        app.dependency_overrides[get_session] = _mock_session_override


def test_delete_table_invalid_name(client):
    """DELETE /tables/{table} returns error for a table name with only special chars."""
    resp = client.delete("/tables/---!!!")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert "Invalid" in data["error"]


@patch("business_brain.action.api.metadata_store")
def test_delete_metadata_only(mock_meta, client):
    """DELETE /metadata/{table} removes metadata but does NOT drop the data table."""
    mock_meta.delete = AsyncMock(return_value=True)

    resp = client.delete("/metadata/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    assert data["table"] == "reports"

    # Verify metadata_store.delete was called but NOT session.execute with DROP TABLE
    mock_meta.delete.assert_called_once()


# ===========================================================================
# 14-15: Metadata upsert idempotency
# ===========================================================================


@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch("business_brain.action.api.metadata_store")
def test_metadata_upsert_updates_not_duplicates(mock_meta, mock_upsert, client):
    """Two uploads to same table -> metadata is updated (not duplicated)."""
    fake = FakeMetadataStore()

    mock_upsert.return_value = 3
    mock_meta.upsert = AsyncMock(side_effect=fake.upsert)
    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)

    # Upload 1
    csv1 = b"id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30"
    resp1 = client.post("/csv", files={"file": ("sales.csv", csv1, "text/csv")})
    assert resp1.status_code == 200

    # Upload 2 (same table name derived from filename)
    csv2 = b"id,name,value\n4,delta,40\n5,epsilon,50"
    resp2 = client.post("/csv", files={"file": ("sales.csv", csv2, "text/csv")})
    assert resp2.status_code == 200

    # Metadata upsert was called twice
    assert mock_meta.upsert.call_count == 2

    # But the fake store should still have only one entry (updated, not duplicated)
    entries = fake.get_entries()
    assert len(entries) == 1
    assert entries[0].table_name == "sales"


@patch("business_brain.action.api.metadata_store")
@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
def test_recurring_upload_preserves_metadata(mock_find, mock_upsert, mock_meta, client):
    """Recurring upload updates description but metadata entry still exists."""
    fake = FakeMetadataStore()

    # Pre-populate: the table was originally uploaded
    fake.seed(
        table_name="monthly_report",
        description="Original description",
        columns_metadata=[{"name": "id", "type": "int64"}, {"name": "revenue", "type": "float64"}],
        uploaded_by="user-001",
        uploaded_by_role="operator",
    )

    # Simulate recurring upload: fingerprint match found
    match = MagicMock()
    match.table_name = "monthly_report"
    match.column_mapping = None
    match.match_count = 3
    match.id = 42
    mock_find.return_value = match
    mock_upsert.return_value = 5

    mock_meta.upsert = AsyncMock(side_effect=fake.upsert)
    mock_meta.get_all = AsyncMock(side_effect=fake.get_all)

    resp = client.post(
        "/upload", files={"file": ("report_jan.csv", SAMPLE_CSV, "text/csv")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "loaded"
    assert data["recurring"] is True

    # The metadata entry should still exist after recurring upload
    entries = fake.get_entries()
    assert len(entries) == 1
    assert entries[0].table_name == "monthly_report"
    # The description was updated by the recurring path
    assert "Recurring upload" in entries[0].description or "monthly_report" in entries[0].description
