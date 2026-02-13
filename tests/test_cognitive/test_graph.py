"""Tests for the graph state machine."""

from business_brain.cognitive.graph import _should_continue_sql, build_graph


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


class TestBuildGraph:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None
