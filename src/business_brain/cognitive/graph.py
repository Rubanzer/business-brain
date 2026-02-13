"""LangGraph state machine wiring the agent swarm."""

import logging
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


# Instantiate agents
_supervisor = SupervisorAgent()
_sql_agent = SQLAgent()
_analyst = AnalystAgent()
_python_analyst = PythonAnalystAgent()
_cfo = CFOAgent()


async def _validate_schema(state: dict) -> dict:
    """Pre-step: clean up stale metadata for dropped tables."""
    db_session = state.get("db_session")
    if db_session:
        try:
            removed = await metadata_store.validate_tables(db_session)
            if removed:
                logger.info("Cleaned up stale metadata for: %s", removed)
        except Exception:
            logger.exception("Schema validation failed")
    return state


def _should_continue_sql(state: dict) -> str:
    """Decide whether to loop back to sql_agent or proceed to analyst."""
    sql_tasks = [s for s in state.get("plan", []) if s.get("agent") == "sql_agent"]
    idx = state.get("current_query_index", 0)
    if idx < len(sql_tasks) and idx < 3:  # max 3 SQL queries
        return "sql_agent"
    return "analyst"


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("validate_schema", _validate_schema)
    graph.add_node("supervisor", _supervisor.invoke)
    graph.add_node("sql_agent", _sql_agent.invoke)  # async invoke
    graph.add_node("analyst", _analyst.invoke)
    graph.add_node("python_analyst", _python_analyst.invoke)
    graph.add_node("cfo", _cfo.invoke)

    graph.set_entry_point("validate_schema")
    graph.add_edge("validate_schema", "supervisor")
    graph.add_edge("supervisor", "sql_agent")
    # Conditional edge: loop sql_agent or proceed to analyst
    graph.add_conditional_edges("sql_agent", _should_continue_sql)
    graph.add_edge("analyst", "python_analyst")
    graph.add_edge("python_analyst", "cfo")
    graph.add_edge("cfo", END)

    return graph.compile()
