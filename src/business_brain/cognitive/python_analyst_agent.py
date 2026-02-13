"""Python Analyst Agent — two-phase: execute code then interpret results."""

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

# Phase 1: Ask Gemini to write analysis code. No format constraints — just print.
CODE_GEN_PROMPT = """\
You are a Python data analyst. Write Python code to analyze the data in `rows` (a list of dicts).

When analyzing multi-query data, each row may include `_query_num` (int) and
`_query_task` (str) metadata fields to indicate which query produced it.
Use these to separate or compare datasets when present.

Rules:
- Only use stdlib: statistics, collections, math, itertools, datetime, re
- No pandas, numpy, scipy, or any external libraries
- Values may be None — always filter before math: [v for v in vals if v is not None]
- Use print() to output every metric, insight, or finding you compute
- Write flat top-level code, no functions or classes
- Keep it under 50 lines
- Return ONLY Python code, no markdown fences or explanation
"""

# Phase 2: Ask Gemini to structure the raw output into our format.
INTERPRET_PROMPT = """\
You are formatting raw analysis output into structured JSON.

Given the printed output from a Python analysis script, return ONLY a JSON object:
{
  "computations": [{"label": "metric name", "value": "metric value"}, ...],
  "narrative": "2-3 sentence interpretation of the findings"
}

Rules:
- Extract every numeric finding into computations with a clear label
- Write a concise narrative summarizing the key insights
- Return ONLY valid JSON, no markdown fences or explanation
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
        raise ImportError(
            f"Import of '{name}' is not allowed. "
            f"Only {sorted(SAFE_MODULES)} are permitted."
        )
    return __import__(name, *args, **kwargs)


def _make_restricted_builtins() -> dict[str, Any]:
    """Build a restricted __builtins__ dict for sandboxed exec."""
    import builtins as _builtins

    BLOCKED = {
        "open", "exec", "eval", "compile", "globals", "locals",
        "breakpoint", "exit", "quit", "input", "memoryview",
        "__import__",
    }

    safe = {
        k: getattr(_builtins, k)
        for k in dir(_builtins)
        if not k.startswith("_") and k not in BLOCKED
    }
    safe["__import__"] = _safe_import
    safe["__build_class__"] = _builtins.__build_class__
    return safe


def execute_sandboxed(code: str, rows: list[dict]) -> dict[str, Any]:
    """Execute code in sandbox, capture print output and all namespace variables.

    Returns dict with: stdout, variables, error
    """
    captured_lines: list[str] = []

    def _capture_print(*args: Any, **kwargs: Any) -> None:
        line = kwargs.get("sep", " ").join(str(a) for a in args)
        captured_lines.append(line)

    builtins = _make_restricted_builtins()
    builtins["print"] = _capture_print

    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "rows": rows,
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:
        return {
            "stdout": "\n".join(captured_lines),
            "variables": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    # Collect user-defined variables (skip builtins, modules, callables, rows)
    import types

    skip = {"__builtins__", "rows"}
    variables = {}
    for k, v in namespace.items():
        if k in skip or k.startswith("_"):
            continue
        # Skip modules, functions, types — keep data only
        if isinstance(v, types.ModuleType):
            continue
        if callable(v) and not isinstance(v, (list, dict, tuple, set)):
            continue
        try:
            json.dumps(v, default=str)  # verify serializable
            variables[k] = v
        except (TypeError, ValueError):
            continue

    return {
        "stdout": "\n".join(captured_lines),
        "variables": variables,
        "error": None,
    }


def _extract_code(raw: str) -> str:
    """Strip markdown fences and language tags from LLM response."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        # Take the first fenced block
        if len(parts) >= 3:
            block = parts[1]
        else:
            block = parts[1] if len(parts) > 1 else text
        # Strip language tag
        for tag in ("python", "py"):
            if block.startswith(tag):
                block = block[len(tag):]
        return block.strip()
    return text


def _interpret_output(
    client: genai.Client,
    question: str,
    stdout: str,
    variables: dict,
) -> dict[str, Any]:
    """Phase 2: Have Gemini interpret raw execution output into structured format."""
    # Build a summary of what the code produced
    output_parts = []
    if stdout.strip():
        output_parts.append(f"Printed output:\n{stdout}")
    if variables:
        var_summary = json.dumps(variables, default=str, indent=2)
        # Truncate if too long
        if len(var_summary) > 2000:
            var_summary = var_summary[:2000] + "\n... (truncated)"
        output_parts.append(f"Variables:\n{var_summary}")

    if not output_parts:
        return {
            "computations": [],
            "narrative": "Analysis code ran but produced no output.",
        }

    raw_output = "\n\n".join(output_parts)

    prompt = (
        f"{INTERPRET_PROMPT}\n\n"
        f"Original question: {question}\n\n"
        f"Raw analysis output:\n{raw_output}"
    )

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        text = _extract_code(response.text)  # reuse fence stripper for JSON
        parsed = json.loads(text)
        return {
            "computations": parsed.get("computations", []),
            "narrative": parsed.get("narrative", ""),
        }
    except Exception:
        logger.exception("Interpretation LLM call failed")
        # Fall back: use raw stdout as narrative
        return {
            "computations": [],
            "narrative": stdout.strip()[:500] if stdout.strip() else "Analysis completed but interpretation failed.",
        }


class PythonAnalystAgent:
    """Two-phase Python analysis: generate code → execute → interpret results."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        # Combine rows from multi-query results, tagging each with metadata
        sql_results = state.get("sql_results")
        if sql_results:
            rows = []
            for i, res in enumerate(sql_results):
                task = res.get("task", f"Query {i + 1}")
                for row in res.get("rows", []):
                    tagged = dict(row)
                    tagged["_query_num"] = i + 1
                    tagged["_query_task"] = task
                    rows.append(tagged)
        else:
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

        # --- Phase 1: Generate code ---
        columns = list(rows[0].keys()) if rows else []
        sample = rows[:10]

        prompt = (
            f"{CODE_GEN_PROMPT}\n\n"
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
            code = _extract_code(response.text)
        except Exception:
            logger.exception("Code generation LLM call failed")
            state["python_analysis"] = {
                "code": "",
                "computations": [],
                "narrative": "",
                "error": "Failed to generate analysis code.",
            }
            return state

        # --- Phase 2: Execute with retry ---
        max_retries = 2
        exec_result = execute_sandboxed(code, rows)

        for retry in range(max_retries):
            if not exec_result["error"]:
                break
            logger.warning("Python exec failed (attempt %d): %s", retry, exec_result["error"])
            # Ask Gemini to fix the code
            fix_prompt = (
                f"{CODE_GEN_PROMPT}\n\n"
                f"Question: {question}\n"
                f"Columns: {columns}\n"
                f"Total rows: {len(rows)}\n\n"
                f"The following code failed:\n```python\n{code}\n```\n\n"
                f"Error: {exec_result['error']}\n\n"
                f"Fix the code to resolve the error. Return ONLY corrected Python code."
            )
            try:
                fix_response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=fix_prompt,
                )
                code = _extract_code(fix_response.text)
                exec_result = execute_sandboxed(code, rows)
            except Exception:
                logger.exception("Code fix LLM call failed")
                break

        if exec_result["error"]:
            state["python_analysis"] = {
                "code": code,
                "computations": [],
                "narrative": "",
                "error": f"Execution error: {exec_result['error']}",
            }
            return state

        # --- Phase 3: Interpret ---
        interpreted = _interpret_output(
            client, question, exec_result["stdout"], exec_result["variables"]
        )

        state["python_analysis"] = {
            "code": code,
            "computations": interpreted["computations"],
            "narrative": interpreted["narrative"],
            "error": None,
        }
        return state
