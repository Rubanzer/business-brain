"""Tests for the Python Analyst Agent (two-phase architecture)."""

from unittest.mock import MagicMock, patch

import pytest

from business_brain.cognitive.python_analyst_agent import (
    PythonAnalystAgent,
    _extract_code,
    execute_sandboxed,
)


# ---------------------------------------------------------------------------
# Sandbox execution tests
# ---------------------------------------------------------------------------


class TestExecuteSandboxed:
    def test_captures_print_output(self):
        code = "print('hello')\nprint('world')"
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert "hello" in out["stdout"]
        assert "world" in out["stdout"]

    def test_captures_variables(self):
        code = "total = sum(r['x'] for r in rows)\navg = total / len(rows)"
        rows = [{"x": 10}, {"x": 20}, {"x": 30}]
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert out["variables"]["total"] == 60
        assert out["variables"]["avg"] == 20.0

    def test_statistics_module(self):
        code = """
import statistics
values = [r['amount'] for r in rows]
mean_val = statistics.mean(values)
print(f'Mean: {mean_val}')
"""
        rows = [{"amount": 10}, {"amount": 20}, {"amount": 30}]
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert "Mean: 20" in out["stdout"]

    def test_collections_module(self):
        code = """
from collections import Counter
counts = Counter(r['cat'] for r in rows)
print(counts.most_common(1))
"""
        rows = [{"cat": "A"}, {"cat": "B"}, {"cat": "A"}, {"cat": "A"}]
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert "A" in out["stdout"]

    def test_blocked_import(self):
        code = "import pandas"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "not allowed" in out["error"]

    def test_blocked_open(self):
        code = "f = open('/etc/passwd')"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None

    def test_runtime_error(self):
        code = "x = 1 / 0"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "ZeroDivision" in out["error"]

    def test_partial_output_on_error(self):
        code = "print('before')\nx = 1 / 0\nprint('after')"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "before" in out["stdout"]

    def test_none_values_filtered(self):
        code = """
values = [r['v'] for r in rows if r['v'] is not None]
print(f'Count: {len(values)}')
"""
        rows = [{"v": 1}, {"v": None}, {"v": 3}]
        out = execute_sandboxed(code, [])
        # code references `rows` but we passed [] — let's pass the real rows
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert "Count: 2" in out["stdout"]

    def test_function_wrapped_code(self):
        """Functions are fine now — we capture print output regardless."""
        code = """
def analyze():
    total = sum(r['x'] for r in rows)
    print(f'Total: {total}')
analyze()
"""
        rows = [{"x": 5}, {"x": 15}]
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert "Total: 20" in out["stdout"]

    def test_math_module(self):
        code = """
import math
print(f'sqrt2: {round(math.sqrt(2), 4)}')
"""
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert "1.4142" in out["stdout"]

    def test_skips_non_serializable_variables(self):
        code = """
import statistics
good_var = 42
"""
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert out["variables"]["good_var"] == 42
        # statistics module should be skipped (callable/module)
        assert "statistics" not in out["variables"]


# ---------------------------------------------------------------------------
# Code extraction tests
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_plain_code(self):
        assert _extract_code("x = 1") == "x = 1"

    def test_markdown_fenced(self):
        raw = "```python\nx = 1\n```"
        assert _extract_code(raw) == "x = 1"

    def test_markdown_no_language(self):
        raw = "```\nx = 1\n```"
        assert _extract_code(raw) == "x = 1"

    def test_with_surrounding_text(self):
        raw = "Here is the code:\n```python\nx = 1\n```\nDone."
        assert _extract_code(raw) == "x = 1"


# ---------------------------------------------------------------------------
# Agent invoke tests
# ---------------------------------------------------------------------------


class TestPythonAnalystAgent:
    def test_no_rows(self):
        agent = PythonAnalystAgent()
        state = {"sql_result": {"rows": [], "query": ""}, "question": "test"}
        result = agent.invoke(state)
        assert result["python_analysis"]["error"] is None
        assert "No data" in result["python_analysis"]["narrative"]

    def test_no_sql_result(self):
        agent = PythonAnalystAgent()
        state = {"question": "test"}
        result = agent.invoke(state)
        assert result["python_analysis"]["error"] is None

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_successful_two_phase(self, mock_client):
        # Phase 1: code generation response
        code_response = MagicMock()
        code_response.text = """
import statistics
values = [r['amount'] for r in rows if r['amount'] is not None]
print(f'Mean: {statistics.mean(values)}')
print(f'Median: {statistics.median(values)}')
"""
        # Phase 2: interpretation response
        interpret_response = MagicMock()
        interpret_response.text = '{"computations": [{"label": "Mean", "value": "20"}, {"label": "Median", "value": "20"}], "narrative": "The average amount is 20."}'

        mock_client.return_value.models.generate_content.side_effect = [
            code_response,
            interpret_response,
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {
                "rows": [{"amount": 10}, {"amount": 20}, {"amount": 30}],
                "query": "SELECT amount FROM sales",
            },
            "question": "What is the average?",
        }
        result = agent.invoke(state)

        pa = result["python_analysis"]
        assert pa["error"] is None
        assert len(pa["computations"]) == 2
        assert pa["narrative"] == "The average amount is 20."
        assert pa["code"] != ""
        # Two LLM calls: code gen + interpretation
        assert mock_client.return_value.models.generate_content.call_count == 2

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_code_gen_failure(self, mock_client):
        mock_client.return_value.models.generate_content.side_effect = RuntimeError("API down")

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "SELECT x"},
            "question": "test",
        }
        result = agent.invoke(state)
        assert result["python_analysis"]["error"] is not None
        assert "Failed" in result["python_analysis"]["error"]

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_execution_error_captured(self, mock_client):
        code_response = MagicMock()
        code_response.text = "x = 1 / 0"
        mock_client.return_value.models.generate_content.return_value = code_response

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "SELECT x"},
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is not None
        assert "ZeroDivision" in pa["error"]
        assert pa["code"] == "x = 1 / 0"

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_interpret_failure_falls_back_to_stdout(self, mock_client):
        code_response = MagicMock()
        code_response.text = "print('Revenue is 100')"

        mock_client.return_value.models.generate_content.side_effect = [
            code_response,
            RuntimeError("interpret failed"),
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "SELECT x"},
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is None
        assert "Revenue is 100" in pa["narrative"]
