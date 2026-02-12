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
  "summary": "One paragraph executive summary"
}
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


class AnalystAgent:
    """Performs analysis on query results using Gemini."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Analyse the SQL result set."""
        sql_result = state.get("sql_result", {})
        rows = sql_result.get("rows", [])
        query = sql_result.get("query", "")
        question = state.get("question", "")

        if not rows:
            logger.warning("No rows to analyse")
            state["analysis"] = {
                "findings": [],
                "summary": "No data was returned by the query.",
            }
            return state

        # Truncate rows for prompt (max ~50 rows to avoid token limits)
        sample = rows[:50]
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Original question: {question}\n"
            f"SQL query: {query}\n"
            f"Result rows ({len(rows)} total, showing {len(sample)}):\n"
            f"{json.dumps(sample, default=str, indent=2)}"
        )

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            raw = response.text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            analysis = json.loads(raw)
        except Exception:
            logger.exception("Analysis LLM call failed")
            analysis = {
                "findings": [{"type": "insight", "description": f"Retrieved {len(rows)} rows.", "confidence": 0.5}],
                "summary": f"Query returned {len(rows)} rows but automated analysis failed.",
            }

        state["analysis"] = analysis
        return state
