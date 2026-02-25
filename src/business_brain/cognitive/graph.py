"""LangGraph state machine — v4 streamlined 3-step pipeline.

Context flow (v4):
  validate_schema → query_router (classifies query + fetches RAG context) →
  sql_agent (generates + executes SQL with retry) →
  insight_formatter (unified statistical analysis + economic assessment)

Reduced from 5 nodes to 3 active nodes (+ validate_schema pre-step).
- Query Router replaces Supervisor (simpler, single task instead of plan array)
- Insight Formatter merges Analyst + CFO (one LLM call instead of two)
- Python Analyst preserved as optional enhancement (not in main path)
- SQL Agent kept with single-query focus (no multi-query loop)

Every stage populates `_diagnostics` so the API can report what happened.
"""

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from business_brain.cognitive.insight_formatter import InsightFormatter
from business_brain.cognitive.query_router import QueryRouter
from business_brain.cognitive.sql_agent import SQLAgent
from business_brain.memory import metadata_store

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    plan: list[dict]                 # single SQL task from query router
    sql_result: dict                 # single SQL result
    analysis: dict
    approved: bool
    cfo_notes: str
    recommendations: list[str]
    chat_history: list[dict]         # conversation memory
    session_id: str                  # chat session identifier
    db_session: Any                  # AsyncSession (typed as Any to avoid serialization issues)
    column_classification: dict      # column classifier output
    cfo_key_metrics: list[dict]      # key metrics with verdicts
    cfo_chart_suggestions: list[dict]  # chart suggestions from insight formatter
    parent_finding: dict             # drill-down: the finding being investigated
    allowed_tables: list[str]        # focus mode: only analyze these tables
    # v4 fields
    query_type: str                  # metric_lookup|trend|comparison|anomaly|drill_down|custom
    query_confidence: float          # 0.0-1.0 confidence from router
    router_reasoning: str            # why the router chose this classification
    key_metrics: list[dict]          # unified key metrics (from insight formatter)
    leakage_patterns: list[str]      # detected financial leakage patterns
    # RAG context — populated by query_router, reused by all downstream
    _rag_tables: list[dict]
    _rag_contexts: list[dict]
    # Deep Tier
    deep_tier_task_id: str               # set when auto-escalated to Claude
    # Pipeline diagnostics
    _diagnostics: list[dict]


# Instantiate agents
_query_router = QueryRouter()
_sql_agent = SQLAgent()
_insight_formatter = InsightFormatter()


def _add_diagnostic(state: dict, stage: str, status: str, detail: str = "",
                    duration_ms: int = 0) -> None:
    """Append a diagnostic entry to state._diagnostics."""
    diags = state.setdefault("_diagnostics", [])
    diags.append({
        "stage": stage,
        "status": status,
        "detail": detail,
        "duration_ms": duration_ms,
    })


def _elapsed(t0: float) -> int:
    """Milliseconds since t0."""
    return int((time.monotonic() - t0) * 1000)


# ---------------------------------------------------------------------------
# Node wrappers with diagnostic harness
# ---------------------------------------------------------------------------


async def _validate_schema(state: dict) -> dict:
    """Pre-step: clean up stale metadata for dropped tables."""
    t0 = time.monotonic()
    db_session = state.get("db_session")
    if db_session:
        try:
            removed = await metadata_store.validate_tables(db_session)
            if removed:
                logger.info("Cleaned up stale metadata for: %s", removed)
                _add_diagnostic(state, "validate_schema", "ok",
                                f"Removed stale: {removed}", _elapsed(t0))
            else:
                _add_diagnostic(state, "validate_schema", "ok", "", _elapsed(t0))
        except Exception:
            logger.exception("Schema validation failed")
            _add_diagnostic(state, "validate_schema", "warn",
                            "Schema validation failed, continuing", _elapsed(t0))
    else:
        _add_diagnostic(state, "validate_schema", "skip", "No db_session")
    return state


async def _query_router_with_diagnostics(state: dict) -> dict:
    """Query router wrapper: classifies query, fetches RAG context, generates SQL task."""
    t0 = time.monotonic()
    try:
        state = await _query_router.invoke(state)
    except Exception as exc:
        logger.exception("Query router failed")
        state["query_type"] = "custom"
        state["query_confidence"] = 0.3
        state["plan"] = [
            {"agent": "sql_agent", "task": f"Retrieve data relevant to: {state.get('question', '')}"},
        ]
        _add_diagnostic(state, "query_router", "error",
                        f"Router failed ({exc}), using fallback", _elapsed(t0))
        return state

    query_type = state.get("query_type", "custom")
    confidence = state.get("query_confidence", 0.5)
    rag_tables = state.get("_rag_tables", [])
    table_names = [t.get("table_name", "?") for t in rag_tables]
    reasoning = state.get("router_reasoning", "")

    _add_diagnostic(state, "query_router", "ok",
                    f"Type={query_type}, Confidence={confidence:.2f}. "
                    f"Tables: {', '.join(table_names) or 'none found'}. "
                    f"{reasoning[:60]}",
                    _elapsed(t0))
    return state


async def _sql_with_diagnostics(state: dict) -> dict:
    """SQL agent wrapper: generates and executes single SQL query with retry."""
    t0 = time.monotonic()

    try:
        state = await _sql_agent.invoke(state)
    except Exception as exc:
        logger.exception("SQL agent failed")
        state["sql_result"] = {"query": "", "rows": [], "error": str(exc)}
        _add_diagnostic(state, "sql_agent", "error",
                        f"SQL agent exception: {exc}", _elapsed(t0))
        return state

    sql_result = state.get("sql_result", {})
    rows = sql_result.get("rows", [])
    query = sql_result.get("query", "")
    error = sql_result.get("error", "")

    if error:
        _add_diagnostic(state, "sql_agent", "error",
                        f"SQL error: {error}. Query: {query[:100]}", _elapsed(t0))
    elif not rows:
        _add_diagnostic(state, "sql_agent", "warn",
                        f"SQL returned 0 rows. Query: {query[:100]}", _elapsed(t0))
    else:
        _add_diagnostic(state, "sql_agent", "ok",
                        f"SQL returned {len(rows)} rows. Query: {query[:80]}...",
                        _elapsed(t0))
    return state


def _insight_formatter_with_diagnostics(state: dict) -> dict:
    """Insight formatter wrapper: unified analysis + economic assessment."""
    t0 = time.monotonic()

    # Check if we have any data to analyze
    sql_result = state.get("sql_result", {})
    total_rows = len(sql_result.get("rows", []))

    if not total_rows:
        state["analysis"] = {
            "findings": [],
            "summary": "No data was returned by the SQL query. The question may not "
                       "match any uploaded data, or the table might be empty.",
            "chart_suggestions": [],
        }
        state["approved"] = True
        state["cfo_notes"] = "No data to assess."
        state["recommendations"] = []
        state["cfo_key_metrics"] = []
        state["cfo_chart_suggestions"] = []
        state["key_metrics"] = []
        state["leakage_patterns"] = []
        _add_diagnostic(state, "insight_formatter", "skip",
                        "No data to analyse (0 rows)", _elapsed(t0))
        return state

    try:
        state = _insight_formatter.invoke(state)
    except Exception as exc:
        logger.exception("Insight formatter failed")
        state["analysis"] = {
            "findings": [
                {"type": "insight", "description": f"Analysis failed: {exc}", "confidence": 0}
            ],
            "summary": f"Analysis failed: {exc}. Raw SQL data is available.",
            "chart_suggestions": [],
        }
        state["approved"] = False
        state["cfo_notes"] = f"Insight formatter failed: {exc}"
        state["recommendations"] = []
        state["cfo_key_metrics"] = []
        state["cfo_chart_suggestions"] = []
        state["key_metrics"] = []
        state["leakage_patterns"] = []
        _add_diagnostic(state, "insight_formatter", "error",
                        f"Formatter exception: {exc}", _elapsed(t0))
        return state

    analysis = state.get("analysis", {})
    findings = analysis.get("findings", [])
    summary = analysis.get("summary", "")
    metrics = state.get("key_metrics", [])
    approved = state.get("approved", True)
    leakage = state.get("leakage_patterns", [])

    detail = (
        f"{len(findings)} findings, {len(metrics)} key metrics, "
        f"approved={approved}"
    )
    if leakage:
        detail += f", {len(leakage)} leakage pattern(s) flagged"
    if summary:
        detail += f". {summary[:60]}..."

    _add_diagnostic(state, "insight_formatter", "ok", detail, _elapsed(t0))
    return state


async def _deep_tier_escalation(state: dict) -> dict:
    """Post-analysis: auto-create a Deep Tier task if confidence is low.

    Runs after insight_formatter. If query_confidence < auto threshold and
    Claude API is configured, creates a background Deep Tier task.
    """
    t0 = time.monotonic()

    confidence = state.get("query_confidence", 1.0)
    db_session = state.get("db_session")

    try:
        from config.settings import settings
        from business_brain.cognitive.deep_tier import is_available, create_task

        threshold = settings.deep_tier_auto_threshold

        if confidence >= threshold or not is_available() or not db_session:
            reason = (
                f"Confidence {confidence:.2f} >= {threshold}" if confidence >= threshold
                else "Deep Tier not configured" if not is_available()
                else "No db_session"
            )
            _add_diagnostic(state, "deep_tier_check", "skip", reason, _elapsed(t0))
            return state

        # Build fast tier summary for the task
        analysis = state.get("analysis", {})
        sql_result = state.get("sql_result", {})

        task_info = await create_task(
            db_session,
            question=state.get("question", ""),
            fast_tier_result={
                "findings": analysis.get("findings", []),
                "summary": analysis.get("summary", ""),
                "key_metrics": state.get("key_metrics", []),
                "query_type": state.get("query_type", "custom"),
            },
            sql_query=sql_result.get("query", ""),
            sql_rows=sql_result.get("rows", []),
            tables_used=[
                t.get("table_name", "") for t in state.get("_rag_tables", [])
            ],
            fast_confidence=confidence,
            session_id=state.get("session_id", ""),
            source_tier="fast",
            requested_by="auto",
        )

        state["deep_tier_task_id"] = task_info.get("task_id")

        _add_diagnostic(
            state, "deep_tier_check", "ok",
            f"Auto-escalated (confidence={confidence:.2f} < {threshold}). "
            f"Task: {task_info.get('task_id', '?')}",
            _elapsed(t0),
        )

    except Exception as exc:
        logger.exception("Deep Tier auto-escalation failed")
        _add_diagnostic(state, "deep_tier_check", "warn",
                        f"Auto-escalation failed: {exc}", _elapsed(t0))

    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construct and return the compiled v4 LangGraph state machine.

    4-step pipeline (Deep Tier check is a lightweight post-step):
      validate_schema → query_router → sql_agent →
      insight_formatter → deep_tier_check → END

    If confidence < threshold and Claude API is configured, deep_tier_check
    creates a background analysis task. The Fast Tier response is returned
    immediately; Deep Tier results are polled via /deep-tier/task/{id}.

    Every node is wrapped in a diagnostic harness.
    """
    graph = StateGraph(AgentState)

    graph.add_node("validate_schema", _validate_schema)
    graph.add_node("query_router", _query_router_with_diagnostics)
    graph.add_node("sql_agent", _sql_with_diagnostics)
    graph.add_node("insight_formatter", _insight_formatter_with_diagnostics)
    graph.add_node("deep_tier_check", _deep_tier_escalation)

    graph.set_entry_point("validate_schema")
    graph.add_edge("validate_schema", "query_router")
    graph.add_edge("query_router", "sql_agent")
    graph.add_edge("sql_agent", "insight_formatter")
    graph.add_edge("insight_formatter", "deep_tier_check")
    graph.add_edge("deep_tier_check", END)

    return graph.compile()
