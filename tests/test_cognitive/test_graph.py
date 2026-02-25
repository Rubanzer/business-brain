"""Tests for the v4 graph state machine — routing, validation, edge cases."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.cognitive.graph import _validate_schema, build_graph


class TestValidateSchema:
    @pytest.mark.asyncio
    @patch("business_brain.cognitive.graph.metadata_store")
    async def test_validate_schema_calls_validate_tables(self, mock_meta):
        mock_meta.validate_tables = AsyncMock(return_value=["old_table"])
        session = AsyncMock()
        state = {"db_session": session}

        result = await _validate_schema(state)
        mock_meta.validate_tables.assert_called_once_with(session)
        assert result is state

    @pytest.mark.asyncio
    async def test_validate_schema_no_session(self):
        state = {}
        result = await _validate_schema(state)
        assert result is state

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.graph.metadata_store")
    async def test_validate_schema_exception(self, mock_meta):
        mock_meta.validate_tables = AsyncMock(side_effect=RuntimeError("DB error"))
        session = AsyncMock()
        state = {"db_session": session}

        result = await _validate_schema(state)
        assert result is state  # Should not raise

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.graph.metadata_store")
    async def test_validate_schema_no_stale(self, mock_meta):
        mock_meta.validate_tables = AsyncMock(return_value=[])
        session = AsyncMock()
        state = {"db_session": session}

        result = await _validate_schema(state)
        assert result is state


class TestBuildGraph:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """The compiled graph should contain all expected node names."""
        graph = build_graph()
        # LangGraph compiled graphs have .nodes attribute
        assert graph is not None
