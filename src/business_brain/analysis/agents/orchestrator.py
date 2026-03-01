"""Orchestrator — LangGraph state machine for the three-track analysis engine.

Pipeline:
  run_track1 → QUALITY_FIRST → [if VETO: exclude] → [parallel: DOMAIN + TEMPORAL] → RESOLVE → DELTA → OUTPUT

Situation types:
- EXPLORATORY: full analysis (feed)
- DIAGNOSTIC: question-driven (targeted)
- MONITORING: quick delta check (post-sync)
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.agents.base import AgentContext
from business_brain.analysis.agents.domain_agent import DomainAgent
from business_brain.analysis.agents.quality_agent import QualityAgent
from business_brain.analysis.agents.temporal_agent import TemporalAgent
from business_brain.analysis.models import AnalysisResult, AnalysisRun
from business_brain.analysis.track1 import run_track1
from business_brain.analysis.track1.enumerator import EnumerationBudget
from business_brain.analysis.track1.executor import TimeScope

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent singletons
# ---------------------------------------------------------------------------

_quality_agent = QualityAgent()
_domain_agent = DomainAgent()
_temporal_agent = TemporalAgent()


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class OrchestratorState(TypedDict, total=False):
    # Input
    session: Any  # AsyncSession
    run: Any  # AnalysisRun
    table_names: list[str]
    situation_type: str  # EXPLORATORY / DIAGNOSTIC / MONITORING
    time_scope: dict | None
    budget: dict | None
    question: str | None  # for DIAGNOSTIC

    # Pipeline state
    track1_results: list[Any]  # list[AnalysisResult]
    quality_outputs: dict  # result_id -> AgentOutput
    vetoed_ids: set
    domain_outputs: dict
    temporal_outputs: dict
    final_results: list[Any]

    # Diagnostics
    timings: dict[str, int]  # step -> ms


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------


async def _run_track1_node(state: dict) -> dict:
    """Step 1: Run algorithmic analysis (Track 1)."""
    t0 = time.monotonic()
    session: AsyncSession = state["session"]
    run: AnalysisRun = state["run"]

    time_scope = None
    ts_dict = state.get("time_scope")
    if ts_dict:
        time_scope = TimeScope(
            column=ts_dict.get("column"),
            window=ts_dict.get("window", "all"),
            compare_to=ts_dict.get("compare_to"),
        )

    budget = None
    budget_dict = state.get("budget")
    if budget_dict:
        budget = EnumerationBudget(
            budgeted_tier_limits=budget_dict.get("budgeted_tier_limits", {2: 100, 3: 50, 4: 50})
        )

    results = await run_track1(
        session=session,
        table_names=state["table_names"],
        run_id=run.id,
        time_scope=time_scope,
        budget=budget,
        top_n=50,  # more than final top_n — agents will filter further
    )

    ms = int((time.monotonic() - t0) * 1000)
    logger.info("Orchestrator: Track 1 produced %d results in %dms", len(results), ms)

    return {
        "track1_results": results,
        "timings": {**state.get("timings", {}), "track1": ms},
    }


async def _quality_gate_node(state: dict) -> dict:
    """Step 2: Quality agent evaluates all findings. VETO power."""
    t0 = time.monotonic()
    session: AsyncSession = state["session"]
    run: AnalysisRun = state["run"]
    results: list[AnalysisResult] = state.get("track1_results", [])

    quality_outputs = {}
    vetoed_ids: set = set()

    for result in results:
        ctx = AgentContext(
            session=session,
            result=result,
            run_id=run.id,
            time_scope=state.get("time_scope"),
        )
        output = await _quality_agent.run(ctx)
        quality_outputs[result.id] = output

        # Apply verdict to result
        verdict = (output.output or {}).get("verdict", "RELIABLE")
        result.quality_verdict = verdict

        if verdict == "UNRELIABLE":
            vetoed_ids.add(result.id)
            logger.debug("Quality VETO on result %s", result.id)

    ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Orchestrator: Quality gate — %d passed, %d vetoed in %dms",
        len(results) - len(vetoed_ids), len(vetoed_ids), ms,
    )

    return {
        "quality_outputs": quality_outputs,
        "vetoed_ids": vetoed_ids,
        "timings": {**state.get("timings", {}), "quality": ms},
    }


async def _enrichment_node(state: dict) -> dict:
    """Step 3: Domain + Temporal agents run in parallel on non-vetoed results."""
    t0 = time.monotonic()
    session: AsyncSession = state["session"]
    run: AnalysisRun = state["run"]
    results: list[AnalysisResult] = state.get("track1_results", [])
    vetoed_ids: set = state.get("vetoed_ids", set())

    domain_outputs = {}
    temporal_outputs = {}

    for result in results:
        if result.id in vetoed_ids:
            continue

        ctx = AgentContext(
            session=session,
            result=result,
            run_id=run.id,
            time_scope=state.get("time_scope"),
        )

        # Run domain and temporal agents
        domain_out = await _domain_agent.run(ctx)
        domain_outputs[result.id] = domain_out

        # Apply domain relevance
        result.domain_relevance = (domain_out.output or {}).get("relevance_score", 0.5)

        temporal_out = await _temporal_agent.run(ctx)
        temporal_outputs[result.id] = temporal_out

        # Apply temporal context
        result.temporal_context = temporal_out.output

    ms = int((time.monotonic() - t0) * 1000)
    logger.info("Orchestrator: Enrichment complete in %dms", ms)

    return {
        "domain_outputs": domain_outputs,
        "temporal_outputs": temporal_outputs,
        "timings": {**state.get("timings", {}), "enrichment": ms},
    }


async def _resolve_node(state: dict) -> dict:
    """Step 4: Compute final scores incorporating agent signals."""
    t0 = time.monotonic()
    results: list[AnalysisResult] = state.get("track1_results", [])
    vetoed_ids: set = state.get("vetoed_ids", set())

    final_results = []
    for result in results:
        if result.id in vetoed_ids:
            continue

        # Final score = algorithmic × quality_boost × domain_relevance
        base = result.interestingness_score
        quality_boost = 1.0 if result.quality_verdict == "RELIABLE" else 0.7
        domain_factor = result.domain_relevance or 0.5

        result.final_score = base * quality_boost * (0.5 + 0.5 * domain_factor)
        final_results.append(result)

    # Sort by final score
    final_results.sort(key=lambda r: r.final_score, reverse=True)

    ms = int((time.monotonic() - t0) * 1000)
    return {
        "final_results": final_results,
        "timings": {**state.get("timings", {}), "resolve": ms},
    }


async def _output_node(state: dict) -> dict:
    """Step 5: Finalize — update run record, flush."""
    t0 = time.monotonic()
    session: AsyncSession = state["session"]
    run: AnalysisRun = state["run"]
    final_results: list[AnalysisResult] = state.get("final_results", [])
    timings = state.get("timings", {})

    from sqlalchemy import func

    run.status = "completed"
    run.completed_at = func.now()
    run.summary = {
        "total_findings": len(final_results),
        "vetoed": len(state.get("vetoed_ids", set())),
        "top_score": round(final_results[0].final_score, 3) if final_results else 0,
        "timings": timings,
    }

    await session.flush()

    ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Orchestrator: Complete — %d findings, top score %.3f",
        len(final_results),
        final_results[0].final_score if final_results else 0,
    )

    return {
        "final_results": final_results,
        "timings": {**timings, "output": ms},
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_analysis_graph() -> Any:
    """Construct the analysis orchestrator state machine."""
    graph = StateGraph(OrchestratorState)

    graph.add_node("track1", _run_track1_node)
    graph.add_node("quality_gate", _quality_gate_node)
    graph.add_node("enrichment", _enrichment_node)
    graph.add_node("resolve", _resolve_node)
    graph.add_node("output", _output_node)

    graph.set_entry_point("track1")
    graph.add_edge("track1", "quality_gate")
    graph.add_edge("quality_gate", "enrichment")
    graph.add_edge("enrichment", "resolve")
    graph.add_edge("resolve", "output")
    graph.add_edge("output", END)

    return graph.compile()


# Module-level compiled graph
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_analysis_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_analysis(
    session: AsyncSession,
    table_names: list[str],
    situation_type: str = "EXPLORATORY",
    time_scope: dict | None = None,
    budget: dict | None = None,
    question: str | None = None,
) -> tuple[AnalysisRun, list[AnalysisResult]]:
    """Run the full three-track analysis pipeline.

    Returns (run, final_results).
    """
    # Create run record
    run = AnalysisRun(
        situation_type=situation_type.lower(),
        trigger="manual",
        config={"table_names": table_names, "question": question},
        time_scope=time_scope,
    )
    session.add(run)
    await session.flush()

    logger.info("Starting analysis run %s (%s) for %d tables", run.id, situation_type, len(table_names))

    # Invoke the graph
    graph = _get_graph()
    initial_state = {
        "session": session,
        "run": run,
        "table_names": table_names,
        "situation_type": situation_type,
        "time_scope": time_scope,
        "budget": budget,
        "question": question,
        "timings": {},
    }

    final_state = await graph.ainvoke(initial_state)
    final_results = final_state.get("final_results", [])

    return run, final_results
