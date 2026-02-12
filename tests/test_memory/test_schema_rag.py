"""Tests for schema RAG retrieval logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.memory.schema_rag import retrieve_relevant_tables


def _make_entry(table_name, description, columns=None):
    entry = MagicMock()
    entry.table_name = table_name
    entry.description = description
    entry.columns_metadata = columns or [{"name": "id", "type": "integer"}]
    return entry


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.mark.asyncio
@patch("business_brain.memory.schema_rag.vector_store")
@patch("business_brain.memory.schema_rag.metadata_store")
@patch("business_brain.memory.schema_rag.embed_text")
async def test_keyword_match_on_table_name(mock_embed, mock_meta, mock_vs, mock_session):
    """Tables whose names appear in the query should rank highest."""
    mock_embed.return_value = [0.1] * 768
    mock_vs.search = AsyncMock(return_value=[])
    mock_meta.get_all = AsyncMock(return_value=[
        _make_entry("sales_orders", "Records of all sales orders"),
        _make_entry("customers", "Customer information"),
    ])

    results = await retrieve_relevant_tables(mock_session, "show me sales orders")

    assert results[0]["table_name"] == "sales_orders"


@pytest.mark.asyncio
@patch("business_brain.memory.schema_rag.vector_store")
@patch("business_brain.memory.schema_rag.metadata_store")
@patch("business_brain.memory.schema_rag.embed_text")
async def test_fallback_returns_all(mock_embed, mock_meta, mock_vs, mock_session):
    """When no keyword matches, all entries are returned as fallback."""
    mock_embed.return_value = [0.1] * 768
    mock_vs.search = AsyncMock(return_value=[])
    mock_meta.get_all = AsyncMock(return_value=[
        _make_entry("products", "Product catalog"),
    ])

    results = await retrieve_relevant_tables(mock_session, "xyz unrelated query")

    assert len(results) == 1
    assert results[0]["table_name"] == "products"


@pytest.mark.asyncio
@patch("business_brain.memory.schema_rag.vector_store")
@patch("business_brain.memory.schema_rag.metadata_store")
@patch("business_brain.memory.schema_rag.embed_text")
async def test_description_keyword_match(mock_embed, mock_meta, mock_vs, mock_session):
    """Keywords from the query matching descriptions should boost score."""
    mock_embed.return_value = [0.1] * 768
    mock_vs.search = AsyncMock(return_value=[])
    mock_meta.get_all = AsyncMock(return_value=[
        _make_entry("tbl_a", "Contains revenue and profit data"),
        _make_entry("tbl_b", "Employee directory"),
    ])

    results = await retrieve_relevant_tables(mock_session, "what is total revenue?")

    assert results[0]["table_name"] == "tbl_a"


@pytest.mark.asyncio
@patch("business_brain.memory.schema_rag.vector_store")
@patch("business_brain.memory.schema_rag.metadata_store")
@patch("business_brain.memory.schema_rag.embed_text")
async def test_context_boost(mock_embed, mock_meta, mock_vs, mock_session):
    """Tables mentioned in semantically similar contexts get a score boost."""
    mock_embed.return_value = [0.1] * 768

    ctx_hit = MagicMock()
    ctx_hit.content = "The customers table tracks customer lifetime value"
    mock_vs.search = AsyncMock(return_value=[ctx_hit])

    mock_meta.get_all = AsyncMock(return_value=[
        _make_entry("customers", "Customer info"),
        _make_entry("orders", "Order records"),
    ])

    results = await retrieve_relevant_tables(mock_session, "lifetime value analysis")

    assert results[0]["table_name"] == "customers"
