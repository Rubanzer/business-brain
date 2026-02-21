"""Supervisor agent — decomposes business questions into analysis tasks.

The Supervisor is the orchestrator. It receives the user's question along with
schema context, business context, and metric thresholds so it can create
informed, data-aware analysis plans instead of generic ones.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Supervisor of a business analytics team at a manufacturing company.
Given a business question, available data schemas, business context, and metric
thresholds, you decompose the question into sub-tasks and delegate to specialists:

- sql_agent: for data retrieval via SQL queries (can run up to 3 queries)
- analyst_agent: for statistical analysis and insight extraction
- python_analyst: for deep computational analysis (correlations, percentiles, aggregations)
- cfo_agent: for economic viability assessment and business impact evaluation

CRITICAL RULES FOR PLANNING:
1. Always start with sql_agent tasks to retrieve data. Design SQL tasks that are
   SPECIFIC — reference actual table names and columns from the schema below.
2. If the question involves multiple data dimensions (e.g., cost AND quality AND time),
   plan separate sql_agent tasks for each dimension.
3. If metric thresholds are available, include them in the sql_agent task description
   so the SQL agent can use CASE WHEN to flag values as normal/warning/critical.
4. For comparisons (supplier vs supplier, month vs month), ensure the sql_agent
   retrieves ALL comparison groups in one query using GROUP BY.
5. If relationships between tables are available, mention JOIN conditions in the task.

AVAILABLE DATA:
{schema_context}

BUSINESS CONTEXT:
{business_context}

Return ONLY a JSON array of steps. Each step must have "agent" and "task" keys.
The task description should be detailed and reference specific tables/columns.

Example:
[
  {{"agent": "sql_agent", "task": "Retrieve monthly power_consumption and melting_loss from production_log grouped by month, including average, min, max. Flag values where power_consumption > 750 kWh/ton as 'critical' per threshold."}},
  {{"agent": "sql_agent", "task": "Retrieve supplier-wise scrap_grade, average_yield, and total_cost from procurement_data joined with quality_results on batch_id."}},
  {{"agent": "analyst_agent", "task": "Analyze trends in power consumption, identify worst-performing months, and correlate with melting loss."}},
  {{"agent": "cfo_agent", "task": "Assess cost impact of power overconsumption and recommend optimization strategy."}}
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

AVAILABLE DATA:
{schema_context}

BUSINESS CONTEXT:
{business_context}

The user's follow-up question is: {question}

Design SQL queries that:
1. Retrieve ALL relevant data supporting or contradicting this insight
2. Break down the insight by additional dimensions (time, category, sub-group)
3. Look for root causes, patterns, or exceptions related to this insight
4. Compare the subject of the insight against benchmarks or peers

Return ONLY a JSON array of steps. Each step must have "agent" and "task" keys.
Focus the sql_agent tasks on granular data retrieval for this specific insight.
Reference specific table names and columns from the schema above.
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _build_schema_summary(tables: list[dict]) -> str:
    """Build a concise schema summary for the supervisor prompt."""
    if not tables:
        return "No tables available yet."

    parts = []
    for t in tables:
        cols = t.get("columns") or []
        col_strs = []
        for c in cols:
            name = c.get("name", "?")
            ctype = c.get("type", "?")
            desc = c.get("description", "")
            col_str = f"{name} ({ctype})"
            if desc:
                col_str += f" — {desc}"
            col_strs.append(col_str)

        desc = t.get("description", "")
        rels = t.get("relationships", [])

        header = f"Table: {t['table_name']}"
        if desc:
            header += f" — {desc}"
        parts.append(header)
        parts.append(f"  Columns: {', '.join(col_strs)}")
        if rels:
            parts.append(f"  Joins: {'; '.join(rels)}")
        parts.append("")

    return "\n".join(parts)


def _build_context_summary(contexts: list[dict]) -> str:
    """Build a business context summary for the supervisor prompt."""
    if not contexts:
        return "No business context available yet."

    parts = []
    for ctx in contexts:
        source = ctx.get("source", "unknown")
        content = ctx.get("content", "")
        if content:
            parts.append(f"[{source}] {content}")

    return "\n".join(parts) if parts else "No business context available yet."


class SupervisorAgent:
    """Plans analysis tasks and routes to specialist agents.

    Now receives schema and business context from the RAG layer so it can
    create data-aware plans instead of flying blind.
    """

    async def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Produce an analysis plan from the user's business question.

        This is now async so it can fetch schema and context from the RAG layer.
        """
        question = state.get("question", "")
        chat_history = state.get("chat_history", [])
        parent_finding = state.get("parent_finding")
        db_session = state.get("db_session")
        logger.info("Planning analysis for: %s", question)

        # Fetch schema and context from RAG layer for informed planning
        schema_context = "No tables available."
        business_context = "No business context available."

        if db_session:
            try:
                from business_brain.memory.schema_rag import retrieve_relevant_tables
                tables, contexts = await retrieve_relevant_tables(
                    db_session, question, top_k=8,
                    allowed_tables=state.get("allowed_tables"),
                )
                schema_context = _build_schema_summary(tables)
                business_context = _build_context_summary(contexts)

                # Store in state so downstream agents can reuse (avoid duplicate RAG calls)
                state["_rag_tables"] = tables
                state["_rag_contexts"] = contexts
            except Exception:
                logger.exception("Schema/context retrieval failed for supervisor")

        # Build prompt — use drill-down prompt if investigating a specific finding
        if parent_finding:
            system = DRILL_DOWN_PROMPT.format(
                finding_type=parent_finding.get("type", "insight"),
                finding_description=parent_finding.get("description", ""),
                finding_impact=parent_finding.get("business_impact", ""),
                question=question,
                schema_context=schema_context,
                business_context=business_context,
            )
            prompt_parts = [system]
        else:
            system = SYSTEM_PROMPT.format(
                schema_context=schema_context,
                business_context=business_context,
            )
            prompt_parts = [system]

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
