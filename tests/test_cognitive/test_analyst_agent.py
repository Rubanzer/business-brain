"""Tests for the Analyst Agent — multi-query support, retry, JSON parsing edge cases."""

import json
from unittest.mock import MagicMock, patch

import pytest

from business_brain.cognitive.analyst_agent import AnalystAgent, _build_data_section


# ---------------------------------------------------------------------------
# _build_data_section tests
# ---------------------------------------------------------------------------


class TestBuildDataSection:
    def test_single_result(self):
        state = {
            "sql_result": {
                "query": "SELECT 1",
                "rows": [{"a": 1}, {"a": 2}],
                "task": "Get data",
            }
        }
        text, count = _build_data_section(state)
        assert count == 2
        assert "Get data" in text
        assert "SELECT 1" in text

    def test_multi_results(self):
        state = {
            "sql_results": [
                {"query": "Q1", "rows": [{"a": 1}], "task": "Task 1"},
                {"query": "Q2", "rows": [{"b": 2}, {"b": 3}], "task": "Task 2"},
            ]
        }
        text, count = _build_data_section(state)
        assert count == 3
        assert "Task 1" in text
        assert "Task 2" in text

    def test_empty_state(self):
        text, count = _build_data_section({})
        assert count == 0
        assert text == ""

    def test_empty_sql_result(self):
        state = {"sql_result": {}}
        text, count = _build_data_section(state)
        assert count == 0
        assert text == ""

    def test_empty_sql_results_list(self):
        state = {"sql_results": []}
        text, count = _build_data_section(state)
        assert count == 0
        assert text == ""

    def test_sql_results_with_empty_rows(self):
        state = {
            "sql_results": [
                {"query": "Q1", "rows": [], "task": "Task 1"},
                {"query": "Q2", "rows": [{"x": 1}], "task": "Task 2"},
            ]
        }
        text, count = _build_data_section(state)
        assert count == 1
        assert "Task 1" in text
        assert "Task 2" in text

    def test_missing_task_key(self):
        """Result without 'task' key should use default label."""
        state = {"sql_results": [{"query": "Q1", "rows": [{"a": 1}]}]}
        text, count = _build_data_section(state)
        assert count == 1
        assert "Query 1" in text

    def test_missing_query_key(self):
        state = {"sql_results": [{"rows": [{"a": 1}], "task": "T1"}]}
        text, count = _build_data_section(state)
        assert count == 1
        assert "T1" in text

    def test_rows_truncated_to_20(self):
        rows = [{"x": i} for i in range(50)]
        state = {"sql_result": {"query": "Q", "rows": rows, "task": "T"}}
        text, count = _build_data_section(state)
        assert count == 50
        assert "50 total, showing 20" in text

    def test_non_serializable_values(self):
        """Rows with datetime-like objects should be handled by json default=str."""
        from datetime import datetime
        state = {
            "sql_result": {
                "query": "Q",
                "rows": [{"ts": datetime(2024, 1, 1)}],
                "task": "T",
            }
        }
        text, count = _build_data_section(state)
        assert count == 1
        assert "2024" in text

    def test_both_sql_result_and_sql_results(self):
        """sql_results should take priority over sql_result."""
        state = {
            "sql_results": [{"query": "Q1", "rows": [{"a": 1}], "task": "Multi"}],
            "sql_result": {"query": "Q2", "rows": [{"b": 2}], "task": "Single"},
        }
        text, count = _build_data_section(state)
        assert "Multi" in text
        assert "Single" not in text


# ---------------------------------------------------------------------------
# AnalystAgent.invoke tests
# ---------------------------------------------------------------------------


class TestAnalystAgent:
    def test_no_rows(self):
        agent = AnalystAgent()
        state = {"sql_result": {"query": "", "rows": []}, "question": "test"}
        result = agent.invoke(state)
        assert "No data" in result["analysis"]["summary"]

    def test_no_sql_result_at_all(self):
        agent = AnalystAgent()
        state = {"question": "test"}
        result = agent.invoke(state)
        assert "No data" in result["analysis"]["summary"]

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_analysis_with_chart_suggestions(self, mock_client):
        response = MagicMock()
        response.text = '{"findings": [{"type": "trend", "description": "Rising sales", "confidence": 0.9}], "summary": "Sales are rising.", "chart_suggestions": [{"type": "bar", "x": "month", "y": ["sales"], "title": "Monthly Sales"}]}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {
            "sql_result": {"query": "Q", "rows": [{"month": "Jan", "sales": 100}]},
            "question": "sales trends",
        }
        result = agent.invoke(state)
        assert result["analysis"]["chart_suggestions"][0]["type"] == "bar"
        assert result["analysis"]["chart_suggestions"][0]["x"] == "month"

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_retry_on_empty_findings(self, mock_client):
        empty_response = MagicMock()
        empty_response.text = '{"findings": [], "summary": ""}'
        valid_response = MagicMock()
        valid_response.text = '{"findings": [{"type": "insight", "description": "Found data", "confidence": 0.8}], "summary": "Data found."}'

        mock_client.return_value.models.generate_content.side_effect = [
            empty_response,
            valid_response,
        ]

        agent = AnalystAgent()
        state = {
            "sql_result": {"query": "Q", "rows": [{"x": 1}]},
            "question": "test",
        }
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "Data found."
        assert mock_client.return_value.models.generate_content.call_count == 2

    # --- JSON parsing edge cases ---

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_with_markdown_json_fence(self, mock_client):
        """LLM wraps response in ```json ... ```."""
        response = MagicMock()
        response.text = '```json\n{"findings": [{"type": "trend", "description": "Up", "confidence": 0.7}], "summary": "Going up."}\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "Going up."

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_with_plain_backtick_fence(self, mock_client):
        """LLM wraps response in ``` ... ``` without json tag."""
        response = MagicMock()
        response.text = '```\n{"findings": [{"type": "insight", "description": "X", "confidence": 0.5}], "summary": "Summary."}\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "Summary."

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_with_invalid_json(self, mock_client):
        """LLM returns invalid JSON — should fallback gracefully."""
        response = MagicMock()
        response.text = '{"findings": [{"type": "trend", BROKEN}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        # Should fall back to default analysis
        assert "failed" in result["analysis"]["summary"].lower() or result["analysis"]["findings"]

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_with_unclosed_backticks(self, mock_client):
        """LLM returns unclosed markdown fence — should handle gracefully."""
        response = MagicMock()
        response.text = '```json\n{"findings": [], "summary": "test"}'
        # With unclosed backticks, split("```") gives 2 parts, so parts[1] is
        # 'json\n{"findings": [], "summary": "test"}' — json tag stripping should work
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        # Even with empty findings, it should handle without crashing
        assert "analysis" in result

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_with_text_before_json(self, mock_client):
        """LLM includes explanation text before the JSON."""
        response = MagicMock()
        response.text = 'Here is my analysis:\n```json\n{"findings": [{"type": "insight", "description": "D", "confidence": 0.9}], "summary": "S"}\n```\nLet me know if you need more.'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "S"

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_empty_string(self, mock_client):
        """LLM returns empty string."""
        response = MagicMock()
        response.text = ""
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        # Should fall back to default
        assert "analysis" in result
        assert result["analysis"]["findings"]

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_just_backticks(self, mock_client):
        """LLM returns only backticks with no content."""
        response = MagicMock()
        response.text = "```\n```"
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert "analysis" in result

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_json_with_trailing_comma(self, mock_client):
        """LLM returns JSON with trailing comma (common LLM mistake)."""
        response = MagicMock()
        response.text = '{"findings": [{"type": "trend", "description": "X", "confidence": 0.5},], "summary": "S",}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        # Trailing commas are invalid JSON — should fallback
        assert "analysis" in result

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_with_newline_after_json_tag(self, mock_client):
        """LLM puts newline between json tag and content."""
        response = MagicMock()
        response.text = '```json\n\n{"findings": [{"type": "insight", "description": "D", "confidence": 0.8}], "summary": "Works"}\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "Works"

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_llm_api_exception(self, mock_client):
        """LLM API throws exception on both attempts."""
        mock_client.return_value.models.generate_content.side_effect = RuntimeError("API down")

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert "failed" in result["analysis"]["summary"].lower()
        assert len(result["analysis"]["findings"]) == 1

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_retry_both_attempts_empty(self, mock_client):
        """Both retry attempts return empty findings."""
        empty = MagicMock()
        empty.text = '{"findings": [], "summary": ""}'
        mock_client.return_value.models.generate_content.return_value = empty

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        # Second attempt's result is used even if empty
        assert "analysis" in result

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_first_attempt_exception_second_succeeds(self, mock_client):
        """First LLM call fails, second succeeds."""
        good = MagicMock()
        good.text = '{"findings": [{"type": "insight", "description": "OK", "confidence": 0.7}], "summary": "Recovered."}'
        mock_client.return_value.models.generate_content.side_effect = [
            RuntimeError("timeout"),
            good,
        ]

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "Recovered."

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_non_object_json(self, mock_client):
        """LLM returns valid JSON but as an array, not object."""
        response = MagicMock()
        response.text = '[{"type": "trend"}]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        # parsed.get("findings") on a list raises no error but returns None
        # Should fallback after both attempts
        assert "analysis" in result

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_findings_without_summary(self, mock_client):
        """LLM returns findings but no summary — triggers retry."""
        no_summary = MagicMock()
        no_summary.text = '{"findings": [{"type": "insight", "description": "X", "confidence": 0.5}]}'
        with_summary = MagicMock()
        with_summary.text = '{"findings": [{"type": "insight", "description": "Y", "confidence": 0.6}], "summary": "Done."}'

        mock_client.return_value.models.generate_content.side_effect = [
            no_summary,
            with_summary,
        ]

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "Done."

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_multi_query_analysis(self, mock_client):
        """Analyst processes multiple SQL results."""
        response = MagicMock()
        response.text = '{"findings": [{"type": "insight", "description": "Combined", "confidence": 0.8}], "summary": "Multi OK."}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {
            "sql_results": [
                {"query": "Q1", "rows": [{"a": 1}], "task": "T1"},
                {"query": "Q2", "rows": [{"b": 2}], "task": "T2"},
            ],
            "question": "compare data",
        }
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "Multi OK."

    @patch("business_brain.cognitive.analyst_agent._get_client")
    def test_response_with_multiple_backtick_blocks(self, mock_client):
        """LLM returns multiple markdown blocks — should use first one."""
        response = MagicMock()
        response.text = '```json\n{"findings": [{"type": "insight", "description": "First", "confidence": 0.9}], "summary": "First block"}\n```\n\nHere is another:\n```json\n{"other": true}\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = AnalystAgent()
        state = {"sql_result": {"query": "Q", "rows": [{"x": 1}]}, "question": "test"}
        result = agent.invoke(state)
        assert result["analysis"]["summary"] == "First block"
