"""Tests for the FastAPI API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from business_brain.db.connection import get_session


# Override DB dependency with a mock session
def _mock_session_override():
    session = AsyncMock()
    yield session


@pytest.fixture()
def client():
    """Create a TestClient that skips real DB startup events and background discovery."""
    with patch("business_brain.action.api._run_discovery_background", new_callable=AsyncMock):
        from business_brain.action.api import app

        # Neutralise startup/shutdown handlers that need a live database
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


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "3.0.0"


@patch("business_brain.action.api.build_graph")
def test_analyze(mock_build, client):
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "question": "revenue trends?",
        "plan": [],
        "sql_result": {"query": "SELECT 1", "rows": []},
        "analysis": {"findings": [], "summary": "No data."},
        "approved": True,
        "cfo_notes": "Approved.",
        "db_session": "should_be_stripped",
    })
    mock_build.return_value = mock_graph

    resp = client.post("/analyze", json={"question": "revenue trends?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "revenue trends?"
    assert "db_session" not in data


@patch("business_brain.action.api.ingest_context", new_callable=AsyncMock)
def test_context(mock_ingest, client):
    mock_ingest.return_value = [7]

    resp = client.post("/context", json={"text": "We sell widgets", "source": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"
    assert data["ids"] == [7]
    assert data["chunks"] == 1


@patch("business_brain.ingestion.csv_loader.upsert_dataframe")
def test_upload_csv(mock_upsert, client):
    mock_upsert.return_value = 3

    csv_content = b"id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30"
    resp = client.post("/csv", files={"file": ("sales.csv", csv_content, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "loaded"
    assert data["table"] == "sales"
    assert data["rows"] == 3
    mock_upsert.assert_called_once()


@patch("business_brain.cognitive.data_engineer_agent.DataEngineerAgent.invoke")
def test_upload_file(mock_invoke, client):
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

    csv_content = b"id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30"
    resp = client.post("/upload", files={"file": ("sales.csv", csv_content, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["table_name"] == "sales"
    assert data["rows_total"] == 3
    assert data["file_type"] == "csv"
    mock_invoke.assert_called_once()


@patch("business_brain.action.api.metadata_store")
def test_list_metadata(mock_store, client):
    entry = MagicMock()
    entry.table_name = "sales"
    entry.description = "Sales data"
    entry.columns_metadata = [{"name": "id", "type": "integer"}]
    mock_store.get_all = AsyncMock(return_value=[entry])

    resp = client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["table_name"] == "sales"


@patch("business_brain.action.api.metadata_store")
def test_get_table_metadata(mock_store, client):
    entry = MagicMock()
    entry.table_name = "orders"
    entry.description = "Order records"
    entry.columns_metadata = []
    mock_store.get_by_table = AsyncMock(return_value=entry)

    resp = client.get("/metadata/orders")
    assert resp.status_code == 200
    assert resp.json()["table_name"] == "orders"


@patch("business_brain.action.api.metadata_store")
def test_get_table_metadata_not_found(mock_store, client):
    mock_store.get_by_table = AsyncMock(return_value=None)

    resp = client.get("/metadata/nonexistent")
    assert resp.status_code == 200
    assert resp.json()["error"] == "Table not found"
