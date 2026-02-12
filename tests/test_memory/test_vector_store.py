"""Tests for the vector store module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.memory import vector_store


@pytest.fixture
def mock_session():
    session = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_search_returns_results(mock_session):
    """search() should execute a query and return scalar results."""
    fake_ctx = MagicMock()
    fake_ctx.content = "test context"
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [fake_ctx]
    mock_session.execute.return_value = mock_result

    results = await vector_store.search(mock_session, [0.1] * 768, top_k=3)

    assert len(results) == 1
    assert results[0].content == "test context"
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_search_empty(mock_session):
    """search() should return empty list when no matches."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute.return_value = mock_result

    results = await vector_store.search(mock_session, [0.0] * 768)
    assert results == []


@pytest.mark.asyncio
async def test_insert_commits_and_returns_id(mock_session):
    """insert() should add entry, commit, refresh, and return ID."""
    mock_session.refresh = AsyncMock(side_effect=lambda e: setattr(e, "id", 42))

    row_id = await vector_store.insert(mock_session, "hello world", [0.1] * 768, source="test")

    assert row_id == 42
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once()


@pytest.mark.asyncio
@patch("business_brain.memory.vector_store.embed_text")
async def test_search_by_text(mock_embed, mock_session):
    """search_by_text() should embed then delegate to search()."""
    mock_embed.return_value = [0.5] * 768

    fake_ctx = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [fake_ctx]
    mock_session.execute.return_value = mock_result

    results = await vector_store.search_by_text(mock_session, "revenue trends", top_k=2)

    mock_embed.assert_called_once_with("revenue trends")
    assert len(results) == 1
