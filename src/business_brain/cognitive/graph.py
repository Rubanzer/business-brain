"""LangGraph state machine wiring the agent swarm.

Context flow:
  validate_schema → supervisor (fetches RAG context + schema) →
  sql_agent (reuses RAG context from state, generates SQL) →  [loop up to 3x]
  validate_sql (check we got data) →
  analyst (receives SQL results + business context) →
  python_analyst (receives analyst findings + data) →
  cfo (receives everything: analysis + python analysis + business context + thresholds)

Every stage now populates `_diagnostics` so the API can report what happened.
"""

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from business_brain.cognitive.analyst_agent import AnalystAgent
from business_brain.cognitive.cfo_agent import CFOAgent
from business_brain.cognitive.python_analyst_agent import PythonAnalystAgent
from business_brain.cognitive.sql_agent import SQLAgent
from business_brain.cognitive.supervisor import SupervisorAgent
from business_brain.memory import metadata_store

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    plan: list[dict]
    sql_result: dict
    sql_results: list[dict]          # accumulates results from each SQL pass
    current_query_index: int         # tracks which SQL task we're on
    analysis: dict
    python_analysis: dict
    approved: bool
    cfo_notes: str
    recommendations: list[str]
    chat_history: list[dict]         # conversation memory
    session_id: str                  # chat session identifier
    db_session: Any  # AsyncSession — typed as Any to avoid serialization issues
    column_classification: dict      # column classifier output
    cfo_key_metrics: list[dict]      # CFO's top metrics with verdicts
    cfo_chart_suggestions: list[dict]  # CFO's chart suggestions
    parent_finding: dict             # drill-down: the finding being investigated
    allowed_tables: list[str]        # focus mode: only analyze these tables (None = all)
    # RAG context — populated by supervisor, reused by all downstream agents
    _rag_tables: list[dict]          # relevant table schemas from schema_rag
    _rag_contexts: list[dict]        # business context snippets from vector store
    # Pipeline diagnostics — tracks what happened at each stage
    _diagnostics: list[dict]


# Instantiate agents
_supervisor = SupervisorAgent()
_sql_agent = SQLAgent()
_analyst = AnalystAgent()
_python_analyst = PythonAnalystAgent()
_cfo = CFOAgent()


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


async def _supervisor_with_diagnostics(state: dict) -> dict:
    """Supervisor wrapper: validates plan output and logs diagnostics."""
    t0 = time.monotonic()
    try:
        state = await _supervisor.invoke(state)
    except Exception as exc:
        logger.exception("Supervisor failed")
        state["plan"] = [
            {"agent": "sql_agent", "task": f"Retrieve data relevant to: {state.get('question', '')}"},
            {"agent": "analyst_agent", "task": "Analyse retrieved data"},
            {"agent": "cfo_agent", "task": "Evaluate economic viability"},
        ]
        _add_diagnostic(state, "supervisor", "error",
                        f"Supervisor failed ({exc}), using default plan", _elapsed(t0))
        return state

    plan = state.get("plan", [])
    sql_tasks = [s for s in plan if s.get("agent") == "sql_agent"]
    rag_tables = state.get("_rag_tables", [])
    table_names = [t.get("table_name", "?") for t in rag_tables]

    _add_diagnostic(state, "supervisor", "ok",
                    f"Plan: {len(plan)} steps ({len(sql_tasks)} SQL tasks). "
                    f"Tables: {', '.join(table_names) or 'none found'}",
                    _elapsed(t0))

    # Validate: plan must have at least one sql_agent task
    if not sql_tasks:
        logger.warning("Supervisor plan has no SQL tasks — adding default")
        plan.insert(0, {
            "agent": "sql_agent",
            "task": f"Retrieve data relevant to: {state.get('question', '')}",
        })
        state["plan"] = plan

    return state


async def _sql_with_diagnostics(state: dict) -> dict:
    """SQL agent wrapper: validates query results and logs diagnostics."""
    t0 = time.monotonic()
    idx = state.get("current_query_index", 0)

    try:
        state = await _sql_agent.invoke(state)
    except Exception as exc:
        logger.exception("SQL agent failed")
        result = {"query": "", "rows": [], "error": str(exc)}
        state["sql_result"] = result
        state.setdefault("sql_results", []).append(result)
        state["current_query_index"] = idx + 1
        _add_diagnostic(state, f"sql_agent[{idx}]", "error",
                        f"SQL agent exception: {exc}", _elapsed(t0))
        return state

    sql_result = state.get("sql_result", {})
    rows = sql_result.get("rows", [])
    query = sql_result.get("query", "")
    error = sql_result.get("error", "")

    if error:
        _add_diagnostic(state, f"sql_agent[{idx}]", "error",
                        f"SQL error: {error}. Query: {query[:100]}", _elapsed(t0))
    elif not rows:
        _add_diagnostic(state, f"sql_agent[{idx}]", "warn",
                        f"SQL returned 0 rows. Query: {query[:100]}", _elapsed(t0))
    else:
        _add_diagnostic(state, f"sql_agent[{idx}]", "ok",
                        f"SQL returned {len(rows)} rows. Query: {query[:80]}...",
                        _elapsed(t0))

    return state


def _analyst_with_diagnostics(state: dict) -> dict:
    """Analyst wrapper: validates analysis output and logs diagnostics."""
    t0 = time.monotonic()

    # Check total data available before analysis
    total_rows = 0
    for res in state.get("sql_results", []):
        total_rows += len(res.get("rows", []))
    if not total_rows:
        single = state.get("sql_result", {})
        total_rows = len(single.get("rows", []))

    if not total_rows:
        state["analysis"] = {
            "findings": [],
            "summary": "No data was returned by any SQL query. The question may not match any uploaded data, or the table might be empty.",
            "chart_suggestions": [],
        }
        _add_diagnostic(state, "analyst", "skip",
                        "No data to analyse (0 rows from all SQL queries)", _elapsed(t0))
        return state

    try:
        state = _analyst.invoke(state)
    except Exception as exc:
        logger.exception("Analyst failed")
        state["analysis"] = {
            "findings": [{"type": "insight", "description": f"Analysis failed: {exc}", "confidence": 0}],
            "summary": f"Automated analysis failed: {exc}. Raw SQL data is available in the details section.",
            "chart_suggestions": [],
        }
        _add_diagnostic(state, "analyst", "error",
                        f"Analyst exception: {exc}", _elapsed(t0))
        return state

    analysis = state.get("analysis", {})
    findings = analysis.get("findings", [])
    summary = analysis.get("summary", "")

    if findings and summary:
        _add_diagnostic(state, "analyst", "ok",
                        f"{len(findings)} findings. Summary: {summary[:80]}...",
                        _elapsed(t0))
    else:
        _add_diagnostic(state, "analyst", "warn",
                        "Analysis produced empty findings or summary",
                        _elapsed(t0))

    return state


def _python_analyst_with_diagnostics(state: dict) -> dict:
    """Python analyst wrapper: validates computation output and logs diagnostics."""
    t0 = time.monotonic()
    try:
        state = _python_analyst.invoke(state)
    except Exception as exc:
        logger.exception("Python analyst failed")
        state["python_analysis"] = {
            "code": "",
            "computations": [],
            "narrative": f"Computational analysis failed: {exc}. See analyst findings above for the main analysis.",
            "error": str(exc),
        }
        _add_diagnostic(state, "python_analyst", "error",
                        f"Python analyst exception: {exc}", _elapsed(t0))
        return state

    py = state.get("python_analysis", {})
    computations = py.get("computations", [])
    error = py.get("error")
    narrative = py.get("narrative", "")

    if error:
        _add_diagnostic(state, "python_analyst", "warn",
                        f"Code execution error: {error}", _elapsed(t0))
    elif computations:
        _add_diagnostic(state, "python_analyst", "ok",
                        f"{len(computations)} computations. Narrative: {narrative[:60]}...",
                        _elapsed(t0))
    else:
        _add_diagnostic(state, "python_analyst", "warn",
                        "No computations produced", _elapsed(t0))

    return state


def _cfo_with_diagnostics(state: dict) -> dict:
    """CFO wrapper: validates assessment output and logs diagnostics."""
    t0 = time.monotonic()
    try:
        state = _cfo.invoke(state)
    except Exception as exc:
        logger.exception("CFO agent failed")
        state["approved"] = False
        state["cfo_notes"] = f"CFO evaluation failed: {exc}. Manual review required."
        state["recommendations"] = []
        state["cfo_key_metrics"] = []
        state["cfo_chart_suggestions"] = []
        _add_diagnostic(state, "cfo", "error",
                        f"CFO exception: {exc}", _elapsed(t0))
        return state

    metrics = state.get("cfo_key_metrics", [])
    approved = state.get("approved", False)
    notes = state.get("cfo_notes", "")

    _add_diagnostic(state, "cfo", "ok",
                    f"Approved={approved}, {len(metrics)} key metrics. Notes: {notes[:60]}...",
                    _elapsed(t0))
    return state


def _should_continue_sql(state: dict) -> str:
    """Decide whether to loop back to sql_agent or proceed to analyst."""
    sql_tasks = [s for s in state.get("plan", []) if s.get("agent") == "sql_agent"]
    idx = state.get("current_query_index", 0)
    if idx < len(sql_tasks) and idx < 3:  # max 3 SQL queries
        return "sql_agent"
    return "analyst"


def _elapsed(t0: float) -> int:
    """Milliseconds since t0."""
    return int((time.monotonic() - t0) * 1000)


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph state machine.

    Every node is now wrapped in a diagnostic harness that:
    - Catches and handles exceptions gracefully
    - Logs timing information
    - Validates output before passing to the next stage
    - Records diagnostics in state['_diagnostics']
    """
    graph = StateGraph(AgentState)

    graph.add_node("validate_schema", _validate_schema)
    graph.add_node("supervisor", _supervisor_with_diagnostics)
    graph.add_node("sql_agent", _sql_with_diagnostics)
    graph.add_node("analyst", _analyst_with_diagnostics)
    graph.add_node("python_analyst", _python_analyst_with_diagnostics)
    graph.add_node("cfo", _cfo_with_diagnostics)

    graph.set_entry_point("validate_schema")
    graph.add_edge("validate_schema", "supervisor")
    graph.add_edge("supervisor", "sql_agent")
    # Conditional edge: loop sql_agent or proceed to analyst
    graph.add_conditional_edges("sql_agent", _should_continue_sql)
    graph.add_edge("analyst", "python_analyst")
    graph.add_edge("python_analyst", "cfo")
    graph.add_edge("cfo", END)

    return graph.compile()
