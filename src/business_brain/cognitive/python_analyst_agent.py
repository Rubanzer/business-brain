"""Python Analyst Agent — executes generated Python code for deep data analysis."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)

SAFE_MODULES = frozenset({
    "statistics", "collections", "math", "itertools", "datetime", "re", "json",
})

SYSTEM_PROMPT = """\
You are a Python data analyst. Given tabular data as a list of dicts, write
Python code to perform deep analysis. The data is available as `rows` (list[dict]).

Rules:
- Only use stdlib: statistics, collections, math, itertools, datetime, re
- No pandas, numpy, scipy, or any external libraries
- IMPORTANT: Values in rows may be None. Always filter out None values before doing math, e.g.: values = [r["col"] for r in rows if r["col"] is not None]
- Keep code concise (under 50 lines)
- Return ONLY valid Python code. No markdown fences, no explanations, no comments before/after.
- CRITICAL: Your code MUST end by assigning a variable called `result` as a dict with exactly this structure:

result = {
    "computations": [{"label": "metric name", "value": "metric value"}, ...],
    "narrative": "2-3 sentence interpretation of the data"
}

Example:
import statistics
values = [r["revenue"] for r in rows if r["revenue"] is not None]
result = {
    "computations": [
        {"label": "Mean Revenue", "value": str(round(statistics.mean(values), 2))},
        {"label": "Median Revenue", "value": str(round(statistics.median(values), 2))},
    ],
    "narrative": "Revenue averages $X with a median of $Y, suggesting a right-skewed distribution."
}
"""

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Custom __import__ that only allows safe stdlib modules."""
    if name not in SAFE_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed. Only {sorted(SAFE_MODULES)} are permitted.")
    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)


def _make_restricted_builtins() -> dict[str, Any]:
    """Build a restricted __builtins__ dict for sandboxed exec."""
    import builtins as _builtins

    BLOCKED = {"open", "exec", "eval", "compile", "globals", "locals",
               "breakpoint", "exit", "quit", "input", "memoryview",
               "__import__"}

    safe = {k: getattr(_builtins, k) for k in dir(_builtins)
            if not k.startswith("_") and k not in BLOCKED}
    safe["__import__"] = _safe_import
    safe["__build_class__"] = _builtins.__build_class__
    return safe


def execute_sandboxed(code: str, rows: list[dict]) -> dict[str, Any]:
    """Execute generated Python code in a restricted sandbox.

    Returns:
        dict with keys: computations, narrative, error
    """
    namespace: dict[str, Any] = {
        "__builtins__": _make_restricted_builtins(),
        "rows": rows,
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:
        return {
            "computations": [],
            "narrative": "",
            "error": f"Execution error: {type(exc).__name__}: {exc}",
        }

    result = namespace.get("result")

    # If result is a string, wrap it as a narrative
    if isinstance(result, str):
        return {
            "computations": [],
            "narrative": result,
            "error": None,
        }

    if not isinstance(result, dict):
        # Try to find any dict in the namespace that looks like output
        for val in namespace.values():
            if isinstance(val, dict) and ("computations" in val or "narrative" in val):
                result = val
                break
        else:
            return {
                "computations": [],
                "narrative": "",
                "error": "Code did not produce a `result` dict.",
            }

    return {
        "computations": result.get("computations", []),
        "narrative": result.get("narrative", ""),
        "error": None,
    }


class PythonAnalystAgent:
    """Generates and executes Python analysis code on SQL result data."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate Python analysis code via Gemini and execute it."""
        sql_result = state.get("sql_result", {})
        rows = sql_result.get("rows", [])
        question = state.get("question", "")

        if not rows:
            state["python_analysis"] = {
                "code": "",
                "computations": [],
                "narrative": "No data available for Python analysis.",
                "error": None,
            }
            return state

        # Build prompt with column names and sample rows
        columns = list(rows[0].keys()) if rows else []
        sample = rows[:10]

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {question}\n"
            f"Columns: {columns}\n"
            f"Total rows: {len(rows)}\n"
            f"Sample data ({len(sample)} rows):\n"
            f"{json.dumps(sample, default=str, indent=2)}"
        )

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            code = response.text.strip()
            # Strip markdown fences if present
            if "```" in code:
                code = code.split("```")[1]
                if code.startswith("python"):
                    code = code[6:]
                code = code.strip()
        except Exception:
            logger.exception("Python analyst LLM call failed")
            state["python_analysis"] = {
                "code": "",
                "computations": [],
                "narrative": "",
                "error": "Failed to generate analysis code.",
            }
            return state

        # Execute in sandbox with ALL rows
        result = execute_sandboxed(code, rows)
        result["code"] = code

        state["python_analysis"] = result
        return state
