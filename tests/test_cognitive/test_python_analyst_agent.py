"""Tests for the Python Analyst Agent."""

from unittest.mock import MagicMock, patch

import pytest

from business_brain.cognitive.python_analyst_agent import (
    PythonAnalystAgent,
    execute_sandboxed,
)


# ---------------------------------------------------------------------------
# Sandbox execution tests
# ---------------------------------------------------------------------------


class TestExecuteSandboxed:
    def test_basic_computation(self):
        code = """
import statistics
values = [r["amount"] for r in rows]
result = {
    "computations": [{"label": "mean", "value": str(statistics.mean(values))}],
    "narrative": "Computed the mean."
}
"""
        rows = [{"amount": 10}, {"amount": 20}, {"amount": 30}]
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert len(out["computations"]) == 1
        assert out["computations"][0]["label"] == "mean"
        assert out["computations"][0]["value"] == "20"
        assert "mean" in out["narrative"]

    def test_collections_module(self):
        code = """
from collections import Counter
counts = Counter(r["category"] for r in rows)
most_common = counts.most_common(1)[0]
result = {
    "computations": [{"label": "top_category", "value": most_common[0]}],
    "narrative": f"Most common category is {most_common[0]} with {most_common[1]} entries."
}
"""
        rows = [{"category": "A"}, {"category": "B"}, {"category": "A"}, {"category": "A"}]
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert out["computations"][0]["value"] == "A"

    def test_blocked_import(self):
        code = "import pandas as pd\nresult = {}"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "not allowed" in out["error"]

    def test_blocked_open(self):
        code = "f = open('/etc/passwd')\nresult = {}"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None

    def test_no_result_variable(self):
        code = "x = 42"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "result" in out["error"].lower()

    def test_runtime_error(self):
        code = "result = 1 / 0"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "ZeroDivision" in out["error"]

    def test_result_string_becomes_narrative(self):
        code = "result = 'some analysis text'"
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert out["narrative"] == "some analysis text"

    def test_result_not_dict_or_string(self):
        code = "result = 42"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None

    def test_math_module(self):
        code = """
import math
result = {
    "computations": [{"label": "sqrt2", "value": str(round(math.sqrt(2), 4))}],
    "narrative": "Computed sqrt of 2."
}
"""
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert out["computations"][0]["value"] == "1.4142"

    def test_itertools_module(self):
        code = """
import itertools
groups = {}
for key, group in itertools.groupby(sorted(rows, key=lambda r: r["cat"]), key=lambda r: r["cat"]):
    groups[key] = list(group)
result = {
    "computations": [{"label": "groups", "value": str(len(groups))}],
    "narrative": f"Found {len(groups)} groups."
}
"""
        rows = [{"cat": "A"}, {"cat": "B"}, {"cat": "A"}]
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert out["computations"][0]["value"] == "2"


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
    def test_successful_invocation(self, mock_client):
        mock_response = MagicMock()
        mock_response.text = """
import statistics
values = [r["amount"] for r in rows]
result = {
    "computations": [{"label": "mean", "value": str(statistics.mean(values))}],
    "narrative": "The mean amount is computed."
}
"""
        mock_client.return_value.models.generate_content.return_value = mock_response

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {
                "rows": [{"amount": 10}, {"amount": 20}, {"amount": 30}],
                "query": "SELECT amount FROM sales",
            },
            "question": "What is the average amount?",
        }
        result = agent.invoke(state)

        pa = result["python_analysis"]
        assert pa["error"] is None
        assert len(pa["computations"]) == 1
        assert pa["computations"][0]["value"] == "20"
        assert pa["code"] != ""

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_llm_failure(self, mock_client):
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
    def test_strips_markdown_fences(self, mock_client):
        mock_response = MagicMock()
        mock_response.text = '```python\nresult = {"computations": [], "narrative": "done"}\n```'
        mock_client.return_value.models.generate_content.return_value = mock_response

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "SELECT x"},
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is None
        assert pa["narrative"] == "done"
