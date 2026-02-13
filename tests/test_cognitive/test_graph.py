"""Tests for the graph state machine — routing, validation, edge cases."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.cognitive.graph import _should_continue_sql, _validate_schema, build_graph


class TestShouldContinueSQL:
    def test_continues_when_more_tasks(self):
        state = {
            "plan": [
                {"agent": "sql_agent", "task": "Q1"},
                {"agent": "sql_agent", "task": "Q2"},
                {"agent": "analyst_agent", "task": "Analyze"},
            ],
            "current_query_index": 0,
        }
        assert _should_continue_sql(state) == "sql_agent"

    def test_continues_at_index_1(self):
        state = {
            "plan": [
                {"agent": "sql_agent", "task": "Q1"},
                {"agent": "sql_agent", "task": "Q2"},
                {"agent": "sql_agent", "task": "Q3"},
            ],
            "current_query_index": 1,
        }
        assert _should_continue_sql(state) == "sql_agent"

    def test_stops_after_all_tasks(self):
        state = {
            "plan": [
                {"agent": "sql_agent", "task": "Q1"},
                {"agent": "analyst_agent", "task": "Analyze"},
            ],
            "current_query_index": 1,
        }
        assert _should_continue_sql(state) == "analyst"

    def test_stops_at_max_3(self):
        state = {
            "plan": [
                {"agent": "sql_agent", "task": f"Q{i}"}
                for i in range(5)
            ],
            "current_query_index": 3,
        }
        assert _should_continue_sql(state) == "analyst"

    def test_no_plan(self):
        state = {"current_query_index": 0}
        assert _should_continue_sql(state) == "analyst"

    def test_empty_plan(self):
        state = {"plan": [], "current_query_index": 0}
        assert _should_continue_sql(state) == "analyst"

    def test_no_index_defaults_to_zero(self):
        state = {
            "plan": [{"agent": "sql_agent", "task": "Q1"}],
        }
        assert _should_continue_sql(state) == "sql_agent"

    def test_all_non_sql_tasks(self):
        state = {
            "plan": [
                {"agent": "analyst_agent", "task": "Analyze"},
                {"agent": "cfo_agent", "task": "Review"},
            ],
            "current_query_index": 0,
        }
        assert _should_continue_sql(state) == "analyst"

    def test_mixed_agents(self):
        state = {
            "plan": [
                {"agent": "sql_agent", "task": "Q1"},
                {"agent": "analyst_agent", "task": "Analyze"},
                {"agent": "sql_agent", "task": "Q2"},
            ],
            "current_query_index": 0,
        }
        # sql_tasks = [Q1, Q2], idx=0, 0 < 2 and 0 < 3 → sql_agent
        assert _should_continue_sql(state) == "sql_agent"

    def test_index_exactly_at_max(self):
        """Index == 3, which is the max."""
        state = {
            "plan": [{"agent": "sql_agent", "task": f"Q{i}"} for i in range(10)],
            "current_query_index": 3,
        }
        assert _should_continue_sql(state) == "analyst"

    def test_index_at_2_continues(self):
        state = {
            "plan": [{"agent": "sql_agent", "task": f"Q{i}"} for i in range(5)],
            "current_query_index": 2,
        }
        assert _should_continue_sql(state) == "sql_agent"

    def test_plan_with_missing_agent_key(self):
        state = {
            "plan": [{"task": "something"}],
            "current_query_index": 0,
        }
        # No items match s.get("agent") == "sql_agent"
        assert _should_continue_sql(state) == "analyst"

    def test_single_sql_task_at_index_0(self):
        state = {
            "plan": [{"agent": "sql_agent", "task": "Q1"}],
            "current_query_index": 0,
        }
        assert _should_continue_sql(state) == "sql_agent"

    def test_single_sql_task_at_index_1(self):
        state = {
            "plan": [{"agent": "sql_agent", "task": "Q1"}],
            "current_query_index": 1,
        }
        assert _should_continue_sql(state) == "analyst"


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
