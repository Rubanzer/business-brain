"""Supervisor agent — decomposes business questions into analysis tasks."""

from typing import Any

SYSTEM_PROMPT = """\
You are the Supervisor of a business analytics team. Given a business question,
you decompose it into sub-tasks and delegate to specialist agents:
- SQL Specialist: for data retrieval
- Data Analyst: for statistical analysis
- CFO Filter: for economic viability assessment

Return a plan as a list of steps with agent assignments.
"""


class SupervisorAgent:
    """Plans analysis tasks and routes to specialist agents."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Produce an analysis plan from the user's business question.

        TODO: call LLM with SYSTEM_PROMPT + state["question"]
        """
        question = state.get("question", "")
        print(f"[supervisor] Planning analysis for: {question}")
        state["plan"] = [
            {"agent": "sql_agent", "task": f"Retrieve data relevant to: {question}"},
            {"agent": "analyst_agent", "task": "Analyse retrieved data"},
            {"agent": "cfo_agent", "task": "Evaluate economic viability"},
        ]
        return state
