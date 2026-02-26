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
You are an expert Python data analyst. Write Python code to analyze the data in `rows`
(a list of dicts).

CRITICAL — the column names in the data are EXACTLY: {columns}
Use ONLY these exact names (case-sensitive). Do NOT invent or rename columns.

When analyzing multi-query data, each row may include `_query_num` (int) and
`_query_task` (str) metadata fields to indicate which query produced it.
Use these to separate or compare datasets when present.

COLUMN CLASSIFICATION (auto-detected):
{column_classification}

ANALYST FINDINGS TO VERIFY/DEEPEN:
{analyst_findings}

ANALYSIS REQUIREMENTS — ALWAYS show the FULL picture first, then highlight issues:

1. FULL ENTITY BREAKDOWN (categorical + numeric columns) — MOST IMPORTANT:
   For each categorical column, compute per-entity: count, sum, mean, min, max.
   Print a COMPLETE RANKED TABLE of ALL entities (not just top/bottom).
   Format as: "1. EntityName: avg=X, total=Y, count=Z (vs group avg: +/-N%)"
   NEVER truncate — show every entity. This is the primary deliverable.

2. PERFORMANCE GAPS (categorical + numeric):
   After the full table, flag: best performer, worst performer, gap between them.
   Print: "BEST: X at Y (+Z% vs avg) | WORST: A at B (-C% vs avg) | GAP: D%"

3. STATISTICAL SUMMARY (numeric columns):
   For each numeric column: mean, median, stdev, P25/P75, min, max.
   Use the `statistics` module.

4. CORRELATIONS (2+ numeric columns):
   Pearson r = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x²)-sum(x)²)*(n*sum(y²)-sum(y)²))
   Print pairs with |r| > 0.3.

5. OUTLIER DETECTION (numeric columns):
   Flag values > 2 std deviations from mean. Print count and examples.
   But this is SECONDARY to the full breakdown — show ALL data first.

6. TIME ANALYSIS (temporal + numeric columns):
   Group by time period, compute period-over-period changes.
   Show the FULL time series, not just the trend summary.

KEY PRINCIPLE: The user wants COMPLETE VISIBILITY — every entity, every period,
every group. Show the full ranked data first, THEN highlight what's unusual.
Never say "only showing top 5" — show everything.

ADVANCED FUNCTIONS (pre-imported, call directly — no import needed):

FORECASTING:
  result = forecast_linear(values: list[float], periods_ahead: int = 3)
  result = forecast_exponential(values: list[float], periods_ahead: int = 3, alpha: float = 0.3)
  # result.predicted_values (list[float]), result.method (str), result.confidence ("high"/"medium"/"low")

  trend = detect_trend(values: list[float])
  # trend.direction ("increasing"/"decreasing"/"stable"/"volatile"), trend.magnitude (% per period), trend.r_squared

  smoothed = compute_moving_average(values: list[float], window: int = 3)

SEGMENTATION:
  result = segment_data(rows: list[dict], features: list[str], n_segments: int = 3)
  # result.segments (list with .label, .size, .center dict, .members list), result.quality_score, result.summary

Use these when the question involves predicting, forecasting, projecting, segmenting, clustering, or grouping.

Rules:
- Only use stdlib (statistics, collections, math, itertools, datetime, re, json) PLUS the pre-imported advanced functions above
- No pandas, numpy, scipy, or external libraries
- Values may be None — ALWAYS filter: [v for v in vals if v is not None]
- Use print() for every metric and finding
- Flat top-level code, no functions or classes
- Keep under 80 lines
- Return ONLY Python code, no markdown fences
"""

# Phase 2: Ask Gemini to structure the raw output into our format.
INTERPRET_PROMPT = """\
You are formatting raw analysis output into structured JSON for a business dashboard.
The user wants COMPLETE VISIBILITY — the full picture, not just highlights.

Given the printed output from a Python analysis script, return ONLY a JSON object:
{{
  "computations": [
    {{
      "label": "descriptive metric name",
      "value": "formatted value with appropriate precision",
      "unit": "%, Rs, Rs/ton, count, etc. or empty string",
      "format": "number|currency|percentage|text",
      "priority": 1-10
    }}
  ],
  "full_breakdown": [
    {{
      "entity": "name of entity (sales rep, shift, supplier, machine, etc.)",
      "metrics": {{"metric_name": "formatted_value", ...}},
      "vs_avg": "+12% or -8%",
      "rank": 1
    }}
  ],
  "narrative": "3-5 sentence executive interpretation. Start with the OVERALL picture
                 (totals, averages), then call out the BEST and WORST performers by name
                 with specific numbers, then state the GAP and what to do about it.
                 Be specific: 'Rajesh sold ₹45L vs team avg ₹58L' not 'one rep underperformed'."
}}

IMPORTANT: Include EVERY entity in full_breakdown — never truncate or say "and X more".
The full_breakdown is the primary deliverable. Computations are the summary stats on top.

Priority guide: full entity breakdowns=10, per-group comparisons=9, outliers=8,
key averages=7, correlations=6, distributions=5, time trends=9.
Format guide: use "currency" for monetary values, "percentage" for rates/ratios/yields,
"number" for counts/quantities.
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

    # Inject advanced analysis functions into sandbox
    try:
        from business_brain.discovery.time_intelligence import (
            forecast_linear,
            forecast_exponential,
            detect_trend,
            compute_moving_average,
        )
        namespace["forecast_linear"] = forecast_linear
        namespace["forecast_exponential"] = forecast_exponential
        namespace["detect_trend"] = detect_trend
        namespace["compute_moving_average"] = compute_moving_average
    except ImportError:
        pass  # Functions not available — sandbox runs without them

    try:
        from business_brain.discovery.segmentation_engine import segment_data
        namespace["segment_data"] = segment_data
    except ImportError:
        pass

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
        for tag in ("python", "py", "json"):
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
        raw_text = response.text or ""
        text = _extract_code(raw_text)  # reuse fence stripper for JSON
        if not text:
            raise ValueError("LLM returned empty response")
        parsed = json.loads(text)
        return {
            "computations": parsed.get("computations", []),
            "full_breakdown": parsed.get("full_breakdown", []),
            "narrative": parsed.get("narrative", ""),
        }
    except Exception:
        logger.exception("Interpretation LLM call failed")
        # Fall back: use raw stdout as narrative
        return {
            "computations": [],
            "full_breakdown": [],
            "narrative": stdout.strip()[:500] if stdout.strip() else "Analysis completed but interpretation failed.",
        }


def _format_classification_summary(classification: dict) -> str:
    """Build a short classification summary for the code gen prompt."""
    if not classification:
        return "No classification available."
    cols = classification.get("columns", {})
    lines = []
    for name, info in cols.items():
        sem = info.get("semantic_type", "unknown")
        lines.append(f"  {name}: {sem}")
    domain = classification.get("domain_hint", "general")
    return f"Domain: {domain}\n" + "\n".join(lines)


def _format_analyst_findings(analysis: dict) -> str:
    """Build a summary of analyst findings for the code gen prompt."""
    if not analysis:
        return "No prior findings."
    parts = []
    summary = analysis.get("summary", "")
    if summary:
        parts.append(f"Summary: {summary}")
    for f in analysis.get("findings", [])[:5]:
        parts.append(f"- [{f.get('type', '?')}] {f.get('description', '')}")
    return "\n".join(parts) if parts else "No prior findings."


class PythonAnalystAgent:
    """Two-phase Python analysis: generate code -> execute -> interpret results."""

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

        # Get classification and analyst findings from state
        classification = state.get("column_classification", {})
        analysis = state.get("analysis", {})
        classification_summary = _format_classification_summary(classification)
        analyst_findings = _format_analyst_findings(analysis)

        # --- Phase 1: Generate code ---
        columns = list(rows[0].keys()) if rows else []
        sample = rows[:10]

        prompt = (
            CODE_GEN_PROMPT.format(
                columns=columns,
                column_classification=classification_summary,
                analyst_findings=analyst_findings,
            ) + "\n\n"
            f"Question: {question}\n"
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
                CODE_GEN_PROMPT.format(
                    columns=columns,
                    column_classification=classification_summary,
                    analyst_findings=analyst_findings,
                ) + "\n\n"
                f"Question: {question}\n"
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
            "full_breakdown": interpreted.get("full_breakdown", []),
            "narrative": interpreted["narrative"],
            "error": None,
        }
        return state
