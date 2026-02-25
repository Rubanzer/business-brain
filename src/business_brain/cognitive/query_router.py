"""Query Router — classifies questions and prepares SQL tasks.

Replaces the Supervisor agent. Instead of complex multi-task plan decomposition,
the router classifies the query type, fetches RAG context, and generates a
single focused SQL task. This reduces LLM calls from 1 (plan) + 1-3 (SQL loop)
to 1 (route) + 1 (SQL).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)


ROUTER_PROMPT = """\
You are a query router for a business intelligence system at a manufacturing company.
Given a business question and available data schemas, your job is to:

1. CLASSIFY the query type (one of: metric_lookup, trend, comparison, anomaly, drill_down, custom)
2. ESTIMATE your confidence (0.0-1.0) in answering this from available data
3. GENERATE a single, focused SQL task description referencing specific tables and columns

AVAILABLE DATA:
{schema_context}

BUSINESS CONTEXT:
{business_context}

CLASSIFICATION GUIDE:
- metric_lookup: "What is the current X?" / "Show me Y" — simple aggregation
- trend: "How has X changed over time?" — needs temporal column + GROUP BY period
- comparison: "Compare X across Y" — needs GROUP BY categorical column
- anomaly: "Any unusual patterns?" / "Find outliers" — statistical deviation
- drill_down: Investigating a specific previous finding deeper
- custom: Complex multi-dimensional question not fitting above categories

CONFIDENCE SCORING:
- 1.0: Exact table+column match, simple query
- 0.7-0.9: Good table match, may need joins or aggregations
- 0.4-0.6: Partial match, uncertain column mapping
- 0.1-0.3: Weak match, question may not be answerable from available data

Return ONLY a JSON object:
{{
  "query_type": "metric_lookup|trend|comparison|anomaly|drill_down|custom",
  "confidence": 0.0-1.0,
  "sql_task": "Detailed SQL task description referencing specific tables, columns, and any JOIN conditions needed",
  "reasoning": "Brief explanation of why this classification and task were chosen"
}}
"""

DRILL_DOWN_ROUTER_PROMPT = """\
You are a query router investigating a specific insight from a previous analysis.

INSIGHT BEING INVESTIGATED:
Type: {finding_type}
Finding: {finding_description}
Business Impact: {finding_impact}

AVAILABLE DATA:
{schema_context}

BUSINESS CONTEXT:
{business_context}

The user's follow-up question is: {question}

Generate a focused SQL task that digs deeper into this specific insight.
The task should retrieve granular data to validate, explain, or contradict the finding.

Return ONLY a JSON object:
{{
  "query_type": "drill_down",
  "confidence": 0.0-1.0,
  "sql_task": "Detailed SQL task for investigating this finding — reference specific tables/columns",
  "reasoning": "Brief explanation of the investigation strategy"
}}
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _extract_json(raw: str) -> dict | None:
    """Robustly extract a JSON object from an LLM response."""
    text = raw.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            block = part.strip()
            for tag in ("json", "JSON"):
                if block.startswith(tag):
                    block = block[len(tag):].strip()
            try:
                return json.loads(block)
            except (json.JSONDecodeError, ValueError):
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _build_schema_summary(tables: list[dict]) -> str:
    """Build a concise schema summary for the router prompt."""
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
    """Build a business context summary for the router prompt."""
    if not contexts:
        return "No business context available yet."
    parts = []
    for ctx in contexts:
        source = ctx.get("source", "unknown")
        content = ctx.get("content", "")
        if content:
            parts.append(f"[{source}] {content}")
    return "\n".join(parts) if parts else "No business context available yet."


class QueryRouter:
    """Classifies queries and prepares focused SQL tasks.

    Replaces SupervisorAgent with a simpler, faster approach:
    - 1 LLM call instead of complex plan decomposition
    - Single SQL task instead of multi-task array
    - Confidence scoring for Deep Tier escalation
    """

    async def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Classify question, fetch RAG context, generate SQL task."""
        question = state.get("question", "")
        chat_history = state.get("chat_history", [])
        parent_finding = state.get("parent_finding")
        db_session = state.get("db_session")
        logger.info("Routing query: %s", question)

        # Fetch schema and context from RAG layer
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

                # Store in state for downstream agents
                state["_rag_tables"] = tables
                state["_rag_contexts"] = contexts
            except Exception:
                logger.exception("RAG retrieval failed in query router")

        # Build prompt
        if parent_finding:
            prompt = DRILL_DOWN_ROUTER_PROMPT.format(
                finding_type=parent_finding.get("type", "insight"),
                finding_description=parent_finding.get("description", ""),
                finding_impact=parent_finding.get("business_impact", ""),
                question=question,
                schema_context=schema_context,
                business_context=business_context,
            )
        else:
            prompt = ROUTER_PROMPT.format(
                schema_context=schema_context,
                business_context=business_context,
            )

        prompt_parts = [prompt]
        if chat_history:
            prompt_parts.append("\nRecent conversation:")
            for msg in chat_history[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"  {role}: {content}")
        prompt_parts.append(f"\nQuestion: {question}")

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents="\n".join(prompt_parts),
            )
            raw = response.text.strip()
            result = _extract_json(raw)
            if result is None:
                raise ValueError(f"Could not parse router JSON: {raw[:200]}")

            state["query_type"] = result.get("query_type", "custom")
            state["query_confidence"] = result.get("confidence", 0.5)
            # Build a plan with single SQL task (backward-compatible with SQL agent)
            sql_task = result.get("sql_task", f"Retrieve data relevant to: {question}")
            state["plan"] = [
                {"agent": "sql_agent", "task": sql_task},
            ]
            state["router_reasoning"] = result.get("reasoning", "")
        except Exception:
            logger.exception("Query router LLM call failed, using fallback")
            state["query_type"] = "custom"
            state["query_confidence"] = 0.3
            state["plan"] = [
                {"agent": "sql_agent", "task": f"Retrieve data relevant to: {question}"},
            ]
            state["router_reasoning"] = "Router failed — using fallback"

        return state
