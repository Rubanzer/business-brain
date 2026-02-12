"""Tests for the FastAPI API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from business_brain.action.api import app
from business_brain.db.connection import get_session


# Override DB dependency with a mock session
def _mock_session_override():
    session = AsyncMock()
    yield session


app.dependency_overrides[get_session] = _mock_session_override

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("business_brain.action.api.build_graph")
def test_analyze(mock_build):
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


@patch("business_brain.action.api.ingest_context")
def test_context(mock_ingest):
    mock_ingest.return_value = 7

    resp = client.post("/context", json={"text": "We sell widgets", "source": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"
    assert data["id"] == 7


@patch("business_brain.action.api.metadata_store")
def test_list_metadata(mock_store):
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
def test_get_table_metadata(mock_store):
    entry = MagicMock()
    entry.table_name = "orders"
    entry.description = "Order records"
    entry.columns_metadata = []
    mock_store.get_by_table = AsyncMock(return_value=entry)

    resp = client.get("/metadata/orders")
    assert resp.status_code == 200
    assert resp.json()["table_name"] == "orders"


@patch("business_brain.action.api.metadata_store")
def test_get_table_metadata_not_found(mock_store):
    mock_store.get_by_table = AsyncMock(return_value=None)

    resp = client.get("/metadata/nonexistent")
    assert resp.status_code == 200
    assert resp.json()["error"] == "Table not found"
