"""Tests for the /upload smart-upload endpoint (recurring + normal paths)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from business_brain.db.connection import get_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_session_override():
    session = AsyncMock()
    yield session


def _make_fingerprint_match(
    table_name: str = "existing_table",
    column_mapping: dict | None = None,
    match_count: int = 5,
    fp_id: int = 42,
) -> MagicMock:
    """Create a fake FormatFingerprint returned by find_matching_fingerprint."""
    match = MagicMock()
    match.table_name = table_name
    match.column_mapping = column_mapping
    match.match_count = match_count
    match.id = fp_id
    return match


SAMPLE_CSV = b"id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30"


@pytest.fixture()
def client():
    """Create a TestClient that skips real DB startup events and background discovery."""
    with patch("business_brain.action.api._run_discovery_background", new_callable=AsyncMock):
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


@pytest.fixture()
def auth_client():
    """TestClient with an authenticated user override."""
    with patch("business_brain.action.api._run_discovery_background", new_callable=AsyncMock):
        from business_brain.action.api import app, get_current_user

        original_startup = list(app.router.on_startup)
        original_shutdown = list(app.router.on_shutdown)
        app.router.on_startup.clear()
        app.router.on_shutdown.clear()

        app.dependency_overrides[get_session] = _mock_session_override

        async def _fake_user():
            return {"sub": "user-123", "role": "admin"}

        app.dependency_overrides[get_current_user] = _fake_user

        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            yield c

        app.dependency_overrides.clear()
        app.router.on_startup = original_startup
        app.router.on_shutdown = original_shutdown


# ---------------------------------------------------------------------------
# RECURRING PATH tests (fingerprint match found)
# ---------------------------------------------------------------------------


@patch("business_brain.action.api.metadata_store")
@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
def test_recurring_calls_metadata_upsert(mock_find, mock_upsert, mock_meta, client):
    """Recurring path must call metadata_store.upsert after appending rows."""
    match = _make_fingerprint_match()
    mock_find.return_value = match
    mock_upsert.return_value = 3
    mock_meta.upsert = AsyncMock(return_value=MagicMock())

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200

    mock_meta.upsert.assert_called_once()
    call_kwargs = mock_meta.upsert.call_args
    assert call_kwargs.kwargs.get("table_name") or call_kwargs[1].get("table_name") or (
        len(call_kwargs.args) >= 2 and call_kwargs.args[1]
    )


@patch("business_brain.action.api.metadata_store")
@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
def test_recurring_table_name_from_match(mock_find, mock_upsert, mock_meta, client):
    """Recurring path must use match.table_name (not the uploaded filename)."""
    match = _make_fingerprint_match(table_name="monthly_sales")
    mock_find.return_value = match
    mock_upsert.return_value = 3
    mock_meta.upsert = AsyncMock(return_value=MagicMock())

    resp = client.post(
        "/upload", files={"file": ("random_name.csv", SAMPLE_CSV, "text/csv")}
    )
    data = resp.json()
    assert data["table_name"] == "monthly_sales"
    # upsert_dataframe should have been called with the match table name
    args, kwargs = mock_upsert.call_args
    # Third positional arg is table_name (df, session, table_name)
    assert args[2] == "monthly_sales" or kwargs.get("table_name") == "monthly_sales"


@patch("business_brain.action.api.metadata_store")
@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
def test_recurring_with_auth_passes_user(mock_find, mock_upsert, mock_meta, auth_client):
    """Recurring path must forward uploaded_by and uploaded_by_role from the user."""
    match = _make_fingerprint_match()
    mock_find.return_value = match
    mock_upsert.return_value = 3
    mock_meta.upsert = AsyncMock(return_value=MagicMock())

    resp = auth_client.post(
        "/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")}
    )
    assert resp.status_code == 200

    mock_meta.upsert.assert_called_once()
    call_kwargs = mock_meta.upsert.call_args
    # Check keyword args (positional or keyword)
    kw = call_kwargs.kwargs if call_kwargs.kwargs else {}
    assert kw.get("uploaded_by") == "user-123"
    assert kw.get("uploaded_by_role") == "admin"


@patch("business_brain.action.api.metadata_store")
@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
def test_recurring_without_auth(mock_find, mock_upsert, mock_meta, client):
    """Recurring path with no auth should pass uploaded_by=None."""
    match = _make_fingerprint_match()
    mock_find.return_value = match
    mock_upsert.return_value = 3
    mock_meta.upsert = AsyncMock(return_value=MagicMock())

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200

    mock_meta.upsert.assert_called_once()
    kw = mock_meta.upsert.call_args.kwargs
    assert kw.get("uploaded_by") is None
    assert kw.get("uploaded_by_role") is None


@patch("business_brain.action.api.metadata_store")
@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
def test_recurring_metadata_failure_noncritical(mock_find, mock_upsert, mock_meta, client):
    """If metadata_store.upsert raises, the endpoint should still return success."""
    match = _make_fingerprint_match()
    mock_find.return_value = match
    mock_upsert.return_value = 5
    mock_meta.upsert = AsyncMock(side_effect=RuntimeError("db error"))

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "loaded"
    assert data["rows"] == 5
    assert data["recurring"] is True


@patch("business_brain.action.api.metadata_store")
@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
def test_recurring_increments_match_count(mock_find, mock_upsert, mock_meta, client):
    """Recurring path must increment match.match_count by 1."""
    match = _make_fingerprint_match(match_count=7)
    mock_find.return_value = match
    mock_upsert.return_value = 3
    mock_meta.upsert = AsyncMock(return_value=MagicMock())

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200
    assert match.match_count == 8


@patch("business_brain.action.api.metadata_store")
@patch("business_brain.ingestion.csv_loader.upsert_dataframe", new_callable=AsyncMock)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
def test_recurring_returns_recurring_flag(mock_find, mock_upsert, mock_meta, client):
    """Recurring path must return 'recurring': True and fingerprint_id in response."""
    match = _make_fingerprint_match(fp_id=99)
    mock_find.return_value = match
    mock_upsert.return_value = 10
    mock_meta.upsert = AsyncMock(return_value=MagicMock())

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    data = resp.json()
    assert data["recurring"] is True
    assert data["fingerprint_id"] == 99
    assert data["status"] == "loaded"
    assert data["rows"] == 10


# ---------------------------------------------------------------------------
# NORMAL PATH tests (no fingerprint match — DataEngineerAgent)
# ---------------------------------------------------------------------------


@patch(
    "business_brain.ingestion.format_matcher.register_fingerprint",
    new_callable=AsyncMock,
)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_normal_calls_data_engineer_agent(mock_invoke, mock_find, mock_register, client):
    """When no fingerprint match is found, DataEngineerAgent.invoke must be called."""
    mock_find.return_value = None
    mock_invoke.return_value = {
        "table_name": "sales",
        "file_type": "csv",
        "rows_total": 3,
        "rows_inserted": 3,
        "rows_dropped": 0,
        "duplicates_removed": 0,
        "issues": [],
        "metadata": {"description": "Sales data", "columns": []},
        "context_generated": "Sales tracking.",
    }

    resp = client.post("/upload", files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200
    mock_invoke.assert_called_once()


@patch(
    "business_brain.ingestion.format_matcher.register_fingerprint",
    new_callable=AsyncMock,
)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_normal_passes_user_to_agent(mock_invoke, mock_find, mock_register, auth_client):
    """Normal path must forward uploaded_by and uploaded_by_role to the agent."""
    mock_find.return_value = None
    mock_invoke.return_value = {"table_name": "t", "rows_total": 1}

    resp = auth_client.post(
        "/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")}
    )
    assert resp.status_code == 200

    call_args = mock_invoke.call_args[0][0]  # first positional arg (the dict)
    assert call_args["uploaded_by"] == "user-123"
    assert call_args["uploaded_by_role"] == "admin"


@patch(
    "business_brain.ingestion.format_matcher.register_fingerprint",
    new_callable=AsyncMock,
)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_normal_without_auth_passes_none(mock_invoke, mock_find, mock_register, client):
    """Normal path with no auth should pass uploaded_by=None to the agent."""
    mock_find.return_value = None
    mock_invoke.return_value = {"table_name": "t", "rows_total": 1}

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200

    call_args = mock_invoke.call_args[0][0]
    assert call_args["uploaded_by"] is None
    assert call_args["uploaded_by_role"] is None


@patch(
    "business_brain.ingestion.format_matcher.register_fingerprint",
    new_callable=AsyncMock,
)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_normal_registers_fingerprint(mock_invoke, mock_find, mock_register, client):
    """Normal path must call register_fingerprint after successful agent upload."""
    mock_find.return_value = None
    mock_invoke.return_value = {"table_name": "new_table", "rows_total": 3}

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200

    mock_register.assert_called_once()
    args, kwargs = mock_register.call_args
    # register_fingerprint(session, columns, table_name)
    registered_table = args[2] if len(args) >= 3 else kwargs.get("table_name")
    assert registered_table == "new_table"


@patch(
    "business_brain.ingestion.format_matcher.register_fingerprint",
    new_callable=AsyncMock,
)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_normal_error_returns_error_dict(mock_invoke, mock_find, mock_register, client):
    """If DataEngineerAgent.invoke raises, the response should contain an error key."""
    mock_find.return_value = None
    mock_invoke.side_effect = RuntimeError("Agent exploded")

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert "Agent exploded" in data["error"]


@patch(
    "business_brain.ingestion.format_matcher.register_fingerprint",
    new_callable=AsyncMock,
)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_normal_response_from_agent(mock_invoke, mock_find, mock_register, client):
    """Normal path should return the agent report directly."""
    mock_find.return_value = None
    expected_report = {
        "table_name": "inventory",
        "file_type": "csv",
        "rows_total": 100,
        "rows_inserted": 95,
        "rows_dropped": 5,
        "duplicates_removed": 2,
        "issues": ["5 rows had missing values"],
        "metadata": {"description": "Inventory data", "columns": [{"name": "sku"}]},
        "context_generated": "Product inventory tracking.",
    }
    mock_invoke.return_value = expected_report

    resp = client.post(
        "/upload", files={"file": ("inventory.csv", SAMPLE_CSV, "text/csv")}
    )
    data = resp.json()
    assert data == expected_report


# ---------------------------------------------------------------------------
# EDGE-CASE / FALLBACK tests
# ---------------------------------------------------------------------------


@patch(
    "business_brain.ingestion.format_matcher.register_fingerprint",
    new_callable=AsyncMock,
)
@patch(
    "business_brain.ingestion.format_matcher.find_matching_fingerprint",
    new_callable=AsyncMock,
)
@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_fingerprint_failure_falls_to_normal(
    mock_invoke, mock_find, mock_register, client
):
    """If find_matching_fingerprint raises, upload should fall through to the normal agent path."""
    mock_find.side_effect = RuntimeError("DB connection lost")
    mock_invoke.return_value = {"table_name": "fallback", "rows_total": 3}

    resp = client.post("/upload", files={"file": ("data.csv", SAMPLE_CSV, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    # Should have used the agent path
    assert data["table_name"] == "fallback"
    mock_invoke.assert_called_once()


@patch(
    "business_brain.ingestion.format_matcher.register_fingerprint",
    new_callable=AsyncMock,
)
@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_upload_no_extension_uses_agent(mock_invoke, mock_register, client):
    """A file without a recognised extension should skip fingerprint check and use the agent."""
    mock_invoke.return_value = {"table_name": "mystery", "rows_total": 1}

    resp = client.post(
        "/upload",
        files={"file": ("datafile", b"some raw bytes", "application/octet-stream")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["table_name"] == "mystery"
    mock_invoke.assert_called_once()
    # register_fingerprint should NOT have been called (ext not in csv/xlsx/xls)
    mock_register.assert_not_called()
