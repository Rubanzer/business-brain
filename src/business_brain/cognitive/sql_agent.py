"""SQL Specialist agent — converts natural language to SQL and executes."""

from typing import Any

SYSTEM_PROMPT = """\
You are an expert SQL analyst. Given a business question and schema context,
generate a precise PostgreSQL query to retrieve the required data.
Always use CTEs for clarity. Never use DELETE/DROP/ALTER.
"""


class SQLAgent:
    """Translates natural language questions into SQL and returns results."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate and execute SQL for the current task.

        TODO: call LLM with schema context, generate SQL, execute against DB.
        """
        task = state.get("plan", [{}])[0].get("task", "")
        print(f"[sql_agent] Generating SQL for: {task}")
        state["sql_result"] = {"query": "SELECT 1", "rows": []}  # placeholder
        return state
