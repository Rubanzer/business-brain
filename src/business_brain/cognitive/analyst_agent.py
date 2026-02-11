"""Data Scientist agent — statistical analysis with pandas/scipy."""

from typing import Any

SYSTEM_PROMPT = """\
You are a data scientist. Given query results, perform statistical analysis
using pandas and scipy. Identify trends, anomalies, and actionable insights.
Return structured findings with confidence levels.
"""


class AnalystAgent:
    """Performs statistical analysis on query results."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Analyse the SQL result set.

        TODO: use pandas/scipy for trend detection, anomaly finding, etc.
        """
        sql_result = state.get("sql_result", {})
        print(f"[analyst_agent] Analysing {len(sql_result.get('rows', []))} rows")
        state["analysis"] = {"findings": [], "confidence": 0.0}  # placeholder
        return state
