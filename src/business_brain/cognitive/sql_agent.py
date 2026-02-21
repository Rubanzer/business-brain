"""SQL Specialist agent — converts natural language to SQL and executes.

Now reuses RAG context from state (populated by Supervisor) to avoid duplicate
embedding calls. Falls back to fresh retrieval if state context is missing.
"""
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
You are an expert SQL analyst working for a secondary steel manufacturing company
(induction furnace). You have DEEP knowledge of steel plant data structures and
can interpret domain-specific column names correctly.

Given a business question, schema context, business context (domain definitions,
business rules, KPIs), and metric thresholds, generate a precise PostgreSQL query
to retrieve the required data.

Use the business context to understand domain-specific terms, acronyms, and
business rules that inform correct query logic.

DOMAIN-AWARE COLUMN INTERPRETATION:
- "fe_t", "fe_mz", "fe_content" → Iron content (different measurement points)
- "kva" → Apparent power (KVA), "kwh" → Real energy consumption
- "sec" → Specific Energy Consumption (kWh/ton) — a KEY efficiency metric
- "pf" → Power Factor (ratio, 0-1 scale)
- "heat_no" → Unique identifier for a single production batch
- "yield_pct" → Output/Input weight ratio × 100
- "tap_to_tap" → Cycle time between furnace taps (minutes)
- "shift" → Work period (typically 'A', 'B', 'C' for morning/afternoon/night)

IMPORTANT RULES:
- Use CTEs for clarity when appropriate.
- NEVER use DELETE, DROP, ALTER, INSERT, UPDATE, or TRUNCATE.
- Return ONLY the SQL query, no explanation or markdown fences.
- When table relationships/joins are provided, use them for cross-table queries.
- When metric thresholds are available, use CASE WHEN to flag values:
  e.g., CASE WHEN power_kwh > 750 THEN 'critical'
             WHEN power_kwh > 625 THEN 'warning'
             ELSE 'normal' END AS power_status
- Use meaningful column aliases that reflect business meaning.
- LIMIT results to 500 rows maximum unless the task explicitly needs more.
- For aggregations, always include COUNT(*) so analysts know sample sizes.
- When querying for "efficiency", include SEC, yield, and power factor together.
- When querying for "cost", include both scrap cost and energy cost components.
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
        rels = t.get("relationships", [])

        table_str = f"Table: {t['table_name']} — {desc}\n  Columns: {cols}"
        if rels:
            table_str += f"\n  Joins: {'; '.join(rels)}"
        parts.append(table_str)
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

        Reuses RAG context from state if available (populated by Supervisor),
        falling back to fresh retrieval.
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
            plan = state.get("plan", [])
            task = plan[0].get("task", "") if plan else ""

        question = state.get("question", task)

        # Reuse RAG context from state (populated by Supervisor) or fetch fresh
        tables = state.get("_rag_tables")
        contexts = state.get("_rag_contexts")

        if tables is None or contexts is None:
            tables, contexts = await retrieve_relevant_tables(db_session, question)
            state["_rag_tables"] = tables
            state["_rag_contexts"] = contexts

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
                    # Ask Gemini to fix the query — include business context for better fixes
                    retry_prompt = (
                        f"{SYSTEM_PROMPT}\n\n"
                        f"Schema:\n{schema_context}\n\n"
                    )
                    if biz_context:
                        retry_prompt += f"Business Context:\n{biz_context}\n\n"
                    retry_prompt += (
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
