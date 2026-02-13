"""Tests for the SQL agent — multi-query, retry, and business context."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.cognitive.sql_agent import (
    SQLAgent,
    _format_business_context,
    _format_schema_context,
    _strip_sql_fences,
)


class TestFormatHelpers:
    def test_format_schema_context(self):
        tables = [
            {"table_name": "sales", "description": "Sales data", "columns": [{"name": "id", "type": "BIGINT"}]},
        ]
        result = _format_schema_context(tables)
        assert "sales" in result
        assert "id (BIGINT)" in result

    def test_format_business_context(self):
        contexts = [
            {"content": "Revenue is tracked quarterly", "source": "manual"},
            {"content": "Enterprise customers", "source": "upload:doc.txt"},
        ]
        result = _format_business_context(contexts)
        assert "[manual]" in result
        assert "Revenue is tracked quarterly" in result
        assert "[upload:doc.txt]" in result

    def test_format_business_context_empty(self):
        assert _format_business_context([]) == ""

    def test_strip_sql_fences(self):
        assert _strip_sql_fences("SELECT 1") == "SELECT 1"
        assert _strip_sql_fences("```sql\nSELECT 1\n```") == "SELECT 1"
        assert _strip_sql_fences("```\nSELECT 1\n```") == "SELECT 1"


class TestSQLAgentMultiQuery:
    @pytest.mark.asyncio
    @patch("business_brain.cognitive.sql_agent.retrieve_relevant_tables")
    @patch("business_brain.cognitive.sql_agent._get_client")
    async def test_multi_query_increments_index(self, mock_client, mock_rag):
        mock_rag.return_value = ([], [])
        mock_response = MagicMock()
        mock_response.text = "SELECT 1"
        mock_client.return_value.models.generate_content.return_value = mock_response

        session = AsyncMock()
        result_obj = MagicMock()
        result_obj.fetchall.return_value = []
        session.execute = AsyncMock(return_value=result_obj)

        agent = SQLAgent()
        state = {
            "question": "test",
            "db_session": session,
            "plan": [
                {"agent": "sql_agent", "task": "Get sales"},
                {"agent": "sql_agent", "task": "Get inventory"},
                {"agent": "analyst_agent", "task": "Analyze"},
            ],
            "current_query_index": 0,
        }

        result = await agent.invoke(state)
        assert result["current_query_index"] == 1
        assert len(result["sql_results"]) == 1

        # Second invocation
        result = await agent.invoke(result)
        assert result["current_query_index"] == 2
        assert len(result["sql_results"]) == 2

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.sql_agent.retrieve_relevant_tables")
    @patch("business_brain.cognitive.sql_agent._get_client")
    async def test_no_db_session(self, mock_client, mock_rag):
        agent = SQLAgent()
        state = {"question": "test"}
        result = await agent.invoke(state)
        assert result["sql_result"]["error"] == "No database session"


class TestSQLAgentRetry:
    @pytest.mark.asyncio
    @patch("business_brain.cognitive.sql_agent.retrieve_relevant_tables")
    @patch("business_brain.cognitive.sql_agent._get_client")
    async def test_retry_on_sql_error(self, mock_client, mock_rag):
        mock_rag.return_value = ([], [])

        # First call: gen SQL, second call: fix SQL
        gen_response = MagicMock()
        gen_response.text = "SELECT bad_col FROM t"
        fix_response = MagicMock()
        fix_response.text = "SELECT 1"
        mock_client.return_value.models.generate_content.side_effect = [gen_response, fix_response]

        session = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("column bad_col does not exist")
            result = MagicMock()
            result.fetchall.return_value = [MagicMock(_mapping={"val": 1})]
            return result

        session.execute = mock_execute
        session.rollback = AsyncMock()

        agent = SQLAgent()
        state = {
            "question": "test",
            "db_session": session,
            "plan": [{"agent": "sql_agent", "task": "Get data"}],
        }

        result = await agent.invoke(state)
        # Should have retried and succeeded
        assert "error" not in result["sql_result"] or result["sql_result"].get("error") is None
        assert len(result["sql_result"]["rows"]) == 1
