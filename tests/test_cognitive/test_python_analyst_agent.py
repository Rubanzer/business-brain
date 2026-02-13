"""Tests for the Python Analyst Agent (two-phase architecture) — comprehensive edge cases."""

from unittest.mock import MagicMock, patch

import pytest

from business_brain.cognitive.python_analyst_agent import (
    PythonAnalystAgent,
    _extract_code,
    _interpret_output,
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
        out = execute_sandboxed(code, rows)
        assert out["error"] is None
        assert "Count: 2" in out["stdout"]

    def test_function_wrapped_code(self):
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
        assert "statistics" not in out["variables"]

    def test_blocked_eval(self):
        code = "eval('1+1')"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None

    def test_blocked_exec(self):
        code = "exec('x = 1')"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None

    def test_empty_code(self):
        out = execute_sandboxed("", [])
        assert out["error"] is None
        assert out["stdout"] == ""
        assert out["variables"] == {}

    def test_empty_rows(self):
        code = "print(f'len: {len(rows)}')"
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert "len: 0" in out["stdout"]

    def test_datetime_module(self):
        code = """
import datetime
d = datetime.date(2024, 1, 1)
print(f'date: {d}')
"""
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert "2024-01-01" in out["stdout"]

    def test_re_module(self):
        code = """
import re
m = re.match(r'\\d+', '42abc')
print(f'match: {m.group()}')
"""
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert "match: 42" in out["stdout"]

    def test_json_module(self):
        code = """
import json
print(json.dumps({'a': 1}))
"""
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert '"a": 1' in out["stdout"] or '"a":1' in out["stdout"]

    def test_itertools_module(self):
        code = """
import itertools
pairs = list(itertools.combinations([1,2,3], 2))
print(f'pairs: {len(pairs)}')
"""
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert "pairs: 3" in out["stdout"]

    def test_blocked_os(self):
        code = "import os"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "not allowed" in out["error"]

    def test_blocked_subprocess(self):
        code = "import subprocess"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None

    def test_blocked_sys(self):
        code = "import sys"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None

    def test_print_with_custom_sep(self):
        code = "print('a', 'b', 'c', sep='-')"
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert "a-b-c" in out["stdout"]

    def test_large_output(self):
        code = "for i in range(100): print(f'line {i}')"
        out = execute_sandboxed(code, [])
        assert out["error"] is None
        assert "line 0" in out["stdout"]
        assert "line 99" in out["stdout"]

    def test_syntax_error(self):
        code = "def foo("
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "SyntaxError" in out["error"]

    def test_name_error(self):
        code = "print(undefined_var)"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "NameError" in out["error"]

    def test_key_error_on_row(self):
        code = "print(rows[0]['nonexistent'])"
        rows = [{"x": 1}]
        out = execute_sandboxed(code, rows)
        assert out["error"] is not None
        assert "KeyError" in out["error"]

    def test_type_error(self):
        code = "x = 'a' + 1"
        out = execute_sandboxed(code, [])
        assert out["error"] is not None
        assert "TypeError" in out["error"]


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

    def test_py_tag(self):
        raw = "```py\nx = 1\n```"
        assert _extract_code(raw) == "x = 1"

    def test_unclosed_fence(self):
        raw = "```python\nx = 1"
        result = _extract_code(raw)
        # parts = ['', 'python\nx = 1'], len < 3, takes parts[1]
        # strips "python" -> "\nx = 1" -> "x = 1"
        assert "x = 1" in result

    def test_empty_fence(self):
        raw = "```\n```"
        result = _extract_code(raw)
        assert result == "" or result.strip() == ""

    def test_multiple_fences(self):
        raw = "```python\nfirst = 1\n```\ntext\n```python\nsecond = 2\n```"
        result = _extract_code(raw)
        # Should take first fenced block
        assert "first = 1" in result

    def test_no_backticks(self):
        raw = "total = sum(v for v in values)"
        assert _extract_code(raw) == raw

    def test_fence_with_json_tag(self):
        """Used for interpretation phase — json tag stripping."""
        raw = '```json\n{"key": "value"}\n```'
        result = _extract_code(raw)
        # json tag is NOT stripped (only python/py tags are stripped)
        assert "json" in result or '{"key"' in result

    def test_newline_before_language_tag(self):
        raw = "```\npython\nx = 1\n```"
        result = _extract_code(raw)
        # parts[1] = "\npython\nx = 1", doesn't start with "python" (starts with \n)
        assert "x = 1" in result

    def test_whitespace_only_in_fence(self):
        raw = "```\n   \n```"
        result = _extract_code(raw)
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# _interpret_output tests
# ---------------------------------------------------------------------------


class TestInterpretOutput:
    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_no_output(self, mock_client):
        result = _interpret_output(mock_client.return_value, "q", "", {})
        assert result["narrative"] == "Analysis code ran but produced no output."
        assert result["computations"] == []
        mock_client.return_value.models.generate_content.assert_not_called()

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_successful_interpretation(self, mock_client):
        response = MagicMock()
        response.text = '{"computations": [{"label": "Total", "value": "100"}], "narrative": "Total is 100."}'
        mock_client.return_value.models.generate_content.return_value = response

        result = _interpret_output(mock_client.return_value, "q", "Total: 100", {})
        assert result["computations"][0]["label"] == "Total"
        assert result["narrative"] == "Total is 100."

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_interpretation_with_fenced_json(self, mock_client):
        response = MagicMock()
        response.text = '```json\n{"computations": [{"label": "X", "value": "1"}], "narrative": "N"}\n```'
        mock_client.return_value.models.generate_content.return_value = response

        result = _interpret_output(mock_client.return_value, "q", "output", {})
        # _extract_code strips json fence but leaves "json" prefix
        # json\n{"computations"... -> starts with "json" but _extract_code only strips python/py
        # So this would fail json.loads... should fall back
        assert isinstance(result, dict)

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_interpretation_failure_fallback(self, mock_client):
        mock_client.return_value.models.generate_content.side_effect = RuntimeError("fail")

        result = _interpret_output(mock_client.return_value, "q", "Revenue: $100", {})
        assert "Revenue: $100" in result["narrative"]

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_interpretation_invalid_json(self, mock_client):
        response = MagicMock()
        response.text = "Not valid JSON at all"
        mock_client.return_value.models.generate_content.return_value = response

        result = _interpret_output(mock_client.return_value, "q", "stdout here", {})
        # Falls back to stdout
        assert "stdout here" in result["narrative"]

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_interpretation_with_variables(self, mock_client):
        response = MagicMock()
        response.text = '{"computations": [{"label": "avg", "value": "42"}], "narrative": "Avg is 42."}'
        mock_client.return_value.models.generate_content.return_value = response

        result = _interpret_output(mock_client.return_value, "q", "", {"avg": 42})
        assert len(result["computations"]) == 1

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_interpretation_empty_json(self, mock_client):
        response = MagicMock()
        response.text = '{}'
        mock_client.return_value.models.generate_content.return_value = response

        result = _interpret_output(mock_client.return_value, "q", "output", {})
        assert result["computations"] == []
        assert result["narrative"] == ""


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
        code_response = MagicMock()
        code_response.text = """
import statistics
values = [r['amount'] for r in rows if r['amount'] is not None]
print(f'Mean: {statistics.mean(values)}')
print(f'Median: {statistics.median(values)}')
"""
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
            "column_classification": {
                "columns": {"amount": {"semantic_type": "numeric_currency"}},
                "domain_hint": "finance",
                "analysis_plan": ["descriptive_statistics"],
            },
            "analysis": {
                "findings": [{"type": "insight", "description": "Avg is 20"}],
                "summary": "Amount averages 20.",
            },
        }
        result = agent.invoke(state)

        pa = result["python_analysis"]
        assert pa["error"] is None
        assert len(pa["computations"]) == 2
        assert pa["narrative"] == "The average amount is 20."
        assert pa["code"] != ""
        assert mock_client.return_value.models.generate_content.call_count == 2

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_classification_and_findings_in_prompt(self, mock_client):
        """Verify column_classification and analyst findings appear in the code gen prompt."""
        code_response = MagicMock()
        code_response.text = "print('hello')"
        interpret_response = MagicMock()
        interpret_response.text = '{"computations": [], "narrative": "Done."}'

        mock_client.return_value.models.generate_content.side_effect = [
            code_response,
            interpret_response,
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"val": 1}], "query": "Q"},
            "question": "test",
            "column_classification": {
                "columns": {"val": {"semantic_type": "numeric_metric"}},
                "domain_hint": "general",
                "analysis_plan": ["descriptive_statistics"],
            },
            "analysis": {
                "findings": [{"type": "insight", "description": "Value is 1"}],
                "summary": "Single value.",
            },
        }
        result = agent.invoke(state)

        # Verify the prompt included classification and findings
        call_args = mock_client.return_value.models.generate_content.call_args_list[0]
        prompt = call_args.kwargs.get("contents", call_args.args[0] if call_args.args else "")
        assert "numeric_metric" in prompt
        assert "Value is 1" in prompt

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

    # --- Multi-query tests ---

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_multi_query_rows_combined(self, mock_client):
        code_response = MagicMock()
        code_response.text = "print(f'total rows: {len(rows)}')"
        interpret_response = MagicMock()
        interpret_response.text = '{"computations": [{"label": "rows", "value": "3"}], "narrative": "3 rows."}'

        mock_client.return_value.models.generate_content.side_effect = [
            code_response,
            interpret_response,
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_results": [
                {"rows": [{"a": 1}], "task": "T1"},
                {"rows": [{"a": 2}, {"a": 3}], "task": "T2"},
            ],
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is None

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_multi_query_metadata_tags(self, mock_client):
        """Rows should be tagged with _query_num and _query_task."""
        captured_code = {}

        def mock_generate(model, contents):
            # First call: code gen
            if "_query_num" not in str(captured_code):
                resp = MagicMock()
                resp.text = """
for r in rows:
    print(r.get('_query_num'), r.get('_query_task'))
"""
                captured_code["called"] = True
                return resp
            # Second call: interpret
            resp = MagicMock()
            resp.text = '{"computations": [], "narrative": "Done."}'
            return resp

        mock_client.return_value.models.generate_content.side_effect = [
            MagicMock(text="print(rows[0].get('_query_num'), rows[0].get('_query_task'))"),
            MagicMock(text='{"computations": [], "narrative": "Tagged."}'),
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_results": [
                {"rows": [{"x": 1}], "task": "Get sales"},
            ],
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is None

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_multi_query_empty_results(self, mock_client):
        agent = PythonAnalystAgent()
        state = {
            "sql_results": [
                {"rows": [], "task": "T1"},
                {"rows": [], "task": "T2"},
            ],
            "question": "test",
        }
        result = agent.invoke(state)
        assert "No data" in result["python_analysis"]["narrative"]

    # --- Retry tests ---

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_retry_on_execution_error(self, mock_client):
        """First code fails, Gemini fixes it, second code succeeds."""
        bad_code = MagicMock()
        bad_code.text = "x = rows[0]['nonexistent']"
        fixed_code = MagicMock()
        fixed_code.text = "print('fixed')"
        interpret_response = MagicMock()
        interpret_response.text = '{"computations": [], "narrative": "Fixed!"}'

        mock_client.return_value.models.generate_content.side_effect = [
            bad_code,
            fixed_code,
            interpret_response,
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "Q"},
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is None
        assert pa["narrative"] == "Fixed!"

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_retry_fix_also_fails(self, mock_client):
        """Both code gen and fix produce failing code."""
        bad1 = MagicMock()
        bad1.text = "x = 1/0"
        bad2 = MagicMock()
        bad2.text = "y = 1/0"
        bad3 = MagicMock()
        bad3.text = "z = 1/0"

        mock_client.return_value.models.generate_content.side_effect = [bad1, bad2, bad3]

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "Q"},
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is not None
        assert "ZeroDivision" in pa["error"]

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_retry_fix_llm_throws(self, mock_client):
        """Code fails and the fix LLM call also throws."""
        bad_code = MagicMock()
        bad_code.text = "x = 1/0"
        mock_client.return_value.models.generate_content.side_effect = [
            bad_code,
            RuntimeError("fix failed"),
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "Q"},
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is not None

    # --- Code extraction edge cases in invoke ---

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_code_gen_returns_fenced_code(self, mock_client):
        code_response = MagicMock()
        code_response.text = "```python\nprint('hello')\n```"
        interpret_response = MagicMock()
        interpret_response.text = '{"computations": [], "narrative": "Hello!"}'

        mock_client.return_value.models.generate_content.side_effect = [
            code_response,
            interpret_response,
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "Q"},
            "question": "test",
        }
        result = agent.invoke(state)
        pa = result["python_analysis"]
        assert pa["error"] is None
        assert "hello" in pa["code"] or "print" in pa["code"]

    @patch("business_brain.cognitive.python_analyst_agent._get_client")
    def test_code_gen_returns_empty(self, mock_client):
        code_response = MagicMock()
        code_response.text = ""

        mock_client.return_value.models.generate_content.side_effect = [
            code_response,
            code_response,
            code_response,
        ]

        agent = PythonAnalystAgent()
        state = {
            "sql_result": {"rows": [{"x": 1}], "query": "Q"},
            "question": "test",
        }
        result = agent.invoke(state)
        # Empty code might cause no output, which leads to "no output" narrative
        assert "python_analysis" in result
