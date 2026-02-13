"""SQL Specialist agent — converts natural language to SQL and executes."""
from __future__ import annotations

import logging
from typing import Any

from google import genai
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.memory.schema_rag import retrieve_relevant_tables
from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert SQL analyst. Given a business question, schema context, and
business context (domain definitions, business rules, KPIs), generate a precise
PostgreSQL query to retrieve the required data.

Use the business context to understand domain-specific terms, acronyms, and
business rules that inform correct query logic.

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


def _format_business_context(contexts: list[dict]) -> str:
    """Format business context snippets into a prompt section."""
    if not contexts:
        return ""
    parts = []
    for ctx in contexts:
        source = ctx.get("source", "unknown")
        content = ctx.get("content", "")
        parts.append(f"[{source}] {content}")
    return "\n".join(parts)


def _strip_sql_fences(raw: str) -> str:
    """Strip markdown fences from an LLM SQL response."""
    sql = raw.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return sql


class SQLAgent:
    """Translates natural language questions into SQL and returns results."""

    async def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate and execute SQL for the current task.

        Supports multi-query mode: reads ``current_query_index`` from state,
        executes the matching SQL task from the plan, appends to ``sql_results``,
        and increments the index for the next pass.
        """
        db_session: AsyncSession | None = state.get("db_session")

        if db_session is None:
            logger.error("No db_session in state — cannot execute SQL")
            state["sql_result"] = {"query": "", "rows": [], "error": "No database session"}
            return state

        # Multi-query: determine which SQL task to execute
        idx = state.get("current_query_index", 0)
        sql_tasks = [s for s in state.get("plan", []) if s.get("agent") == "sql_agent"]

        if sql_tasks and idx < len(sql_tasks):
            task = sql_tasks[idx].get("task", "")
        else:
            task = state.get("plan", [{}])[0].get("task", "")

        question = state.get("question", task)

        # Retrieve relevant schemas and business context
        tables, contexts = await retrieve_relevant_tables(db_session, question)
        schema_context = _format_schema_context(tables)
        biz_context = _format_business_context(contexts)

        # Generate SQL via Gemini
        prompt_parts = [SYSTEM_PROMPT, ""]

        # Include last question's context for continuity
        chat_history = state.get("chat_history", [])
        if chat_history:
            last_msgs = chat_history[-2:]  # last Q&A pair
            prompt_parts.append("Recent conversation context:")
            for msg in last_msgs:
                prompt_parts.append(f"  {msg.get('role', 'user')}: {msg.get('content', '')}")
            prompt_parts.append("")

        if biz_context:
            prompt_parts.append(f"Business Context:\n{biz_context}\n")
        prompt_parts.append(f"Schema:\n{schema_context}\n")
        prompt_parts.append(f"Task: {task}")
        prompt_parts.append(f"Question: {question}")
        prompt = "\n".join(prompt_parts)

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            sql = _strip_sql_fences(response.text)
        except Exception:
            logger.exception("SQL generation failed")
            result = {"query": "", "rows": [], "error": "SQL generation failed"}
            state["sql_result"] = result
            state.setdefault("sql_results", []).append(result)
            state["current_query_index"] = idx + 1
            return state

        # Execute the generated SQL (read-only) with retry on failure
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            logger.info("Executing SQL [task %d, attempt %d]: %s", idx, attempt, sql[:200])
            try:
                db_result = await db_session.execute(text(sql))
                rows = [dict(row._mapping) for row in db_result.fetchall()]
                result = {"query": sql, "rows": rows, "task": task}
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning("SQL execution failed (attempt %d): %s", attempt, last_error)
                await db_session.rollback()
                if attempt < max_retries:
                    # Ask Gemini to fix the query
                    retry_prompt = (
                        f"{SYSTEM_PROMPT}\n\n"
                        f"Schema:\n{schema_context}\n\n"
                        f"The following SQL query failed:\n{sql}\n\n"
                        f"Error: {last_error}\n\n"
                        f"Fix the query to resolve the error. Return ONLY the corrected SQL."
                    )
                    try:
                        retry_response = client.models.generate_content(
                            model=settings.gemini_model,
                            contents=retry_prompt,
                        )
                        sql = _strip_sql_fences(retry_response.text)
                    except Exception:
                        logger.exception("SQL retry generation failed")
                        break

        if last_error:
            result = {"query": sql, "rows": [], "error": last_error, "task": task}

        # Update state: latest result + accumulate
        state["sql_result"] = result
        state.setdefault("sql_results", []).append(result)
        state["current_query_index"] = idx + 1

        return state
