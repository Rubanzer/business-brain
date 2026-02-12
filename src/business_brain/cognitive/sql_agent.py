"""SQL Specialist agent — converts natural language to SQL and executes."""

import logging
from typing import Any

from google import genai
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.memory.schema_rag import retrieve_relevant_tables
from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert SQL analyst. Given a business question and schema context,
generate a precise PostgreSQL query to retrieve the required data.
Rules:
- Use CTEs for clarity when appropriate.
- NEVER use DELETE, DROP, ALTER, INSERT, UPDATE, or TRUNCATE.
- Return ONLY the SQL query, no explanation or markdown fences.
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _format_schema_context(tables: list[dict]) -> str:
    """Format table schemas into a string for the LLM prompt."""
    parts = []
    for t in tables:
        cols = ", ".join(
            f"{c['name']} ({c['type']})" for c in (t.get("columns") or [])
        )
        desc = t.get("description") or "No description"
        parts.append(f"Table: {t['table_name']} — {desc}\n  Columns: {cols}")
    return "\n\n".join(parts)


class SQLAgent:
    """Translates natural language questions into SQL and returns results."""

    async def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate and execute SQL for the current task."""
        task = state.get("plan", [{}])[0].get("task", "")
        question = state.get("question", task)
        db_session: AsyncSession | None = state.get("db_session")

        if db_session is None:
            logger.error("No db_session in state — cannot execute SQL")
            state["sql_result"] = {"query": "", "rows": [], "error": "No database session"}
            return state

        # Retrieve relevant schemas
        tables = await retrieve_relevant_tables(db_session, question)
        schema_context = _format_schema_context(tables)

        # Generate SQL via Gemini
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Schema:\n{schema_context}\n\n"
            f"Question: {question}"
        )

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            sql = response.text.strip()
            # Strip markdown fences if present
            if sql.startswith("```"):
                sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        except Exception:
            logger.exception("SQL generation failed")
            state["sql_result"] = {"query": "", "rows": [], "error": "SQL generation failed"}
            return state

        # Execute the generated SQL (read-only)
        logger.info("Executing SQL: %s", sql[:200])
        try:
            result = await db_session.execute(text(sql))
            rows = [dict(row._mapping) for row in result.fetchall()]
            state["sql_result"] = {"query": sql, "rows": rows}
        except Exception as exc:
            logger.exception("SQL execution failed")
            state["sql_result"] = {"query": sql, "rows": [], "error": str(exc)}

        return state
