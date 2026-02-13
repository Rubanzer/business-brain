"""Supervisor agent — decomposes business questions into analysis tasks."""
from __future__ import annotations

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
- python_analyst: for deep computational analysis (correlations, percentiles, aggregations)
- cfo_agent: for economic viability assessment

Return ONLY a JSON array of steps. Each step must have "agent" and "task" keys.
Example:
[
  {"agent": "sql_agent", "task": "Retrieve monthly revenue for the last 12 months"},
  {"agent": "analyst_agent", "task": "Identify revenue trends and anomalies"},
  {"agent": "cfo_agent", "task": "Assess viability of scaling the top revenue channel"}
]
"""

DRILL_DOWN_PROMPT = """\
You are the Supervisor of a business analytics team. The user is drilling deeper
into a specific insight from a previous analysis. Your job is to design SQL queries
and analysis tasks that INVESTIGATE this insight thoroughly.

INSIGHT BEING INVESTIGATED:
Type: {finding_type}
Finding: {finding_description}
Business Impact: {finding_impact}

The user's follow-up question is: {question}

Design SQL queries that:
1. Retrieve ALL relevant data supporting or contradicting this insight
2. Break down the insight by additional dimensions (time, category, sub-group)
3. Look for root causes, patterns, or exceptions related to this insight
4. Compare the subject of the insight against benchmarks or peers

Return ONLY a JSON array of steps. Each step must have "agent" and "task" keys.
Focus the sql_agent tasks on granular data retrieval for this specific insight.
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
        chat_history = state.get("chat_history", [])
        parent_finding = state.get("parent_finding")
        logger.info("Planning analysis for: %s", question)

        # Build prompt — use drill-down prompt if investigating a specific finding
        if parent_finding:
            system = DRILL_DOWN_PROMPT.format(
                finding_type=parent_finding.get("type", "insight"),
                finding_description=parent_finding.get("description", ""),
                finding_impact=parent_finding.get("business_impact", ""),
                question=question,
            )
            prompt_parts = [system]
        else:
            prompt_parts = [SYSTEM_PROMPT]

        if chat_history:
            prompt_parts.append("\nRecent conversation history (for context):")
            for msg in chat_history[-10:]:  # last 5 Q&A pairs = 10 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"  {role}: {content}")
            prompt_parts.append("")
        prompt_parts.append(f"Question: {question}")

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents="\n".join(prompt_parts),
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
