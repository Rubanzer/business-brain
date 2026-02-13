"""Data Scientist agent — statistical analysis with Gemini."""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a data scientist. Given SQL query results, perform analysis to identify:
1. Key trends
2. Anomalies or outliers
3. Actionable insights

Return ONLY a JSON object with this structure:
{
  "findings": [
    {"type": "trend|anomaly|insight", "description": "...", "confidence": 0.0-1.0}
  ],
  "summary": "One paragraph executive summary",
  "chart_suggestions": [
    {"type": "bar|line|pie|scatter", "x": "column_name", "y": ["column_name"], "title": "Chart Title"}
  ]
}

For chart_suggestions:
- Only suggest charts when the data has clear x/y relationships (numeric y values)
- Use "line" for time series, "bar" for categorical comparisons, "pie" for proportions, "scatter" for correlations
- Suggest 1-3 charts maximum. Omit chart_suggestions if the data is not suitable for visualization.
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _build_data_section(state: dict[str, Any]) -> tuple[str, int]:
    """Build the data section for the analyst prompt from single or multi-query results.

    Returns (data_text, total_row_count).
    """
    # Try multi-query results first, fall back to single sql_result
    sql_results = state.get("sql_results")
    if not sql_results:
        single = state.get("sql_result", {})
        if single:
            sql_results = [single]
        else:
            return "", 0

    parts = []
    total_rows = 0
    for i, res in enumerate(sql_results):
        rows = res.get("rows", [])
        total_rows += len(rows)
        query = res.get("query", "")
        task = res.get("task", f"Query {i + 1}")
        sample = rows[:20]
        header = f"--- Query {i + 1}: {task} ---"
        parts.append(
            f"{header}\n"
            f"SQL: {query}\n"
            f"Result rows ({len(rows)} total, showing {len(sample)}):\n"
            f"{json.dumps(sample, default=str, indent=2)}"
        )

    return "\n\n".join(parts), total_rows


class AnalystAgent:
    """Performs analysis on query results using Gemini."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Analyse the SQL result set(s)."""
        question = state.get("question", "")
        data_section, total_rows = _build_data_section(state)

        if not total_rows:
            logger.warning("No rows to analyse")
            state["analysis"] = {
                "findings": [],
                "summary": "No data was returned by the query.",
            }
            return state

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Original question: {question}\n\n"
            f"{data_section}"
        )

        analysis = None
        for attempt in range(2):
            try:
                client = _get_client()
                use_prompt = prompt if attempt == 0 else (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Original question: {question}\n\n"
                    f"Previous analysis attempt returned invalid or empty results. "
                    f"Please provide a simpler analysis focusing on basic statistics.\n\n"
                    f"{data_section}"
                )
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=use_prompt,
                )
                raw = response.text.strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw)
                # Validate: must have findings list and summary
                if parsed.get("findings") and parsed.get("summary"):
                    analysis = parsed
                    break
                elif attempt == 0:
                    logger.warning("Analysis returned empty findings, retrying with simplified prompt")
                    continue
                else:
                    analysis = parsed
            except Exception:
                logger.exception("Analysis LLM call failed (attempt %d)", attempt)
                if attempt == 0:
                    continue

        if analysis is None:
            analysis = {
                "findings": [{"type": "insight", "description": f"Retrieved {total_rows} rows.", "confidence": 0.5}],
                "summary": f"Query returned {total_rows} rows but automated analysis failed.",
            }

        state["analysis"] = analysis
        return state
