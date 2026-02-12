"""Supervisor agent — decomposes business questions into analysis tasks."""

import json
import logging
from typing import Any

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Supervisor of a business analytics team. Given a business question,
you decompose it into sub-tasks and delegate to specialist agents:
- sql_agent: for data retrieval via SQL
- analyst_agent: for statistical analysis and insight extraction
- cfo_agent: for economic viability assessment

Return ONLY a JSON array of steps. Each step must have "agent" and "task" keys.
Example:
[
  {"agent": "sql_agent", "task": "Retrieve monthly revenue for the last 12 months"},
  {"agent": "analyst_agent", "task": "Identify revenue trends and anomalies"},
  {"agent": "cfo_agent", "task": "Assess viability of scaling the top revenue channel"}
]
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


class SupervisorAgent:
    """Plans analysis tasks and routes to specialist agents."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Produce an analysis plan from the user's business question."""
        question = state.get("question", "")
        logger.info("Planning analysis for: %s", question)

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=f"{SYSTEM_PROMPT}\n\nQuestion: {question}",
            )
            raw = response.text.strip()
            # Extract JSON from possible markdown fences
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            plan = json.loads(raw)
        except Exception:
            logger.exception("LLM planning failed, using default plan")
            plan = [
                {"agent": "sql_agent", "task": f"Retrieve data relevant to: {question}"},
                {"agent": "analyst_agent", "task": "Analyse retrieved data"},
                {"agent": "cfo_agent", "task": "Evaluate economic viability"},
            ]

        state["plan"] = plan
        return state
