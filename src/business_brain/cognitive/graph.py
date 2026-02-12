"""LangGraph state machine wiring the agent swarm."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from business_brain.cognitive.analyst_agent import AnalystAgent
from business_brain.cognitive.cfo_agent import CFOAgent
from business_brain.cognitive.sql_agent import SQLAgent
from business_brain.cognitive.supervisor import SupervisorAgent


class AgentState(TypedDict, total=False):
    question: str
    plan: list[dict]
    sql_result: dict
    analysis: dict
    approved: bool
    cfo_notes: str
    recommendations: list[str]
    db_session: Any  # AsyncSession — typed as Any to avoid serialization issues


# Instantiate agents
_supervisor = SupervisorAgent()
_sql_agent = SQLAgent()
_analyst = AnalystAgent()
_cfo = CFOAgent()


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", _supervisor.invoke)
    graph.add_node("sql_agent", _sql_agent.invoke)  # async invoke
    graph.add_node("analyst", _analyst.invoke)
    graph.add_node("cfo", _cfo.invoke)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "sql_agent")
    graph.add_edge("sql_agent", "analyst")
    graph.add_edge("analyst", "cfo")
    graph.add_edge("cfo", END)

    return graph.compile()
