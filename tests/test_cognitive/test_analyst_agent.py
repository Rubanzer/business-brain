"""Tests for the Analyst Agent — multi-query support and retry."""

from unittest.mock import MagicMock, patch

import pytest

from business_brain.cognitive.analyst_agent import AnalystAgent, _build_data_section


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


class TestAnalystAgent:
    def test_no_rows(self):
        agent = AnalystAgent()
        state = {"sql_result": {"query": "", "rows": []}, "question": "test"}
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
        # First response: empty findings, second: valid
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
