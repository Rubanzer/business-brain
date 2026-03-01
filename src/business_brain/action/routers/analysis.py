"""API router for the three-track analysis engine.

9 endpoints:
- POST /analysis/explore — full three-track analysis
- POST /analysis/diagnose — question-driven analysis
- POST /analysis/monitor — quick delta check
- GET  /analysis/findings/{result_id} — single finding + agent outputs
- GET  /analysis/findings/{result_id}/followups — child findings (Gap #7)
- GET  /analysis/runs — recent runs
- GET  /analysis/runs/{run_id} — run detail + findings
- POST /analysis/findings/{result_id}/feedback — submit feedback
- GET  /analysis/learning/state — current learning parameters
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.agents.orchestrator import run_analysis
from business_brain.analysis.models import (
    AgentOutput,
    AnalysisDelta,
    AnalysisFeedback,
    AnalysisResult,
    AnalysisRun,
    LearningState,
)
from business_brain.db.connection import get_session

router = APIRouter(tags=["analysis"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ExploreRequest(BaseModel):
    table_names: list[str]
    time_scope: Optional[dict] = None  # {column?, window?, compare_to?}
    budget: Optional[dict] = None  # {budgeted_tier_limits?}


class DiagnoseRequest(BaseModel):
    question: str
    table_names: Optional[list[str]] = None


class MonitorRequest(BaseModel):
    table_names: Optional[list[str]] = None


class FeedbackRequest(BaseModel):
    feedback_type: str  # useful / not_useful / wrong / expected
    comment: Optional[str] = None
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/analysis/explore")
async def explore(
    body: ExploreRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Run full three-track exploratory analysis."""
    run, results = await run_analysis(
        session=session,
        table_names=body.table_names,
        situation_type="EXPLORATORY",
        time_scope=body.time_scope,
        budget=body.budget,
    )
    await session.commit()
    return {
        "run_id": run.id,
        "status": run.status,
        "finding_count": len(results),
        "top_findings": [_serialize_result(r) for r in results[:10]],
        "summary": run.summary,
    }


@router.post("/analysis/diagnose")
async def diagnose(
    body: DiagnoseRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Run question-driven diagnostic analysis."""
    table_names = body.table_names or []
    # If no tables specified, use all profiled tables
    if not table_names:
        from business_brain.db.discovery_models import TableProfile

        result = await session.execute(select(TableProfile.table_name))
        table_names = [r[0] for r in result.all()]

    run, results = await run_analysis(
        session=session,
        table_names=table_names,
        situation_type="DIAGNOSTIC",
        question=body.question,
    )
    await session.commit()
    return {
        "run_id": run.id,
        "question": body.question,
        "finding_count": len(results),
        "top_findings": [_serialize_result(r) for r in results[:10]],
        "summary": run.summary,
    }


@router.post("/analysis/monitor")
async def monitor(
    body: MonitorRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Quick delta check for monitoring."""
    table_names = body.table_names or []
    if not table_names:
        from business_brain.db.discovery_models import TableProfile

        result = await session.execute(select(TableProfile.table_name))
        table_names = [r[0] for r in result.all()]

    run, results = await run_analysis(
        session=session,
        table_names=table_names,
        situation_type="MONITORING",
        budget={"budgeted_tier_limits": {2: 20, 3: 10, 4: 10}},  # reduced budget
    )
    await session.commit()
    return {
        "run_id": run.id,
        "finding_count": len(results),
        "top_findings": [_serialize_result(r) for r in results[:5]],
        "summary": run.summary,
    }


@router.get("/analysis/findings/{result_id}")
async def get_finding(
    result_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Get a single finding with agent outputs."""
    result = await session.execute(
        select(AnalysisResult).where(AnalysisResult.id == result_id)
    )
    finding = result.scalar_one_or_none()
    if not finding:
        return {"error": "not found"}

    # Fetch agent outputs
    agent_result = await session.execute(
        select(AgentOutput).where(AgentOutput.result_id == result_id)
    )
    agents = agent_result.scalars().all()

    # Fetch deltas
    delta_result = await session.execute(
        select(AnalysisDelta).where(AnalysisDelta.result_id == result_id)
    )
    deltas = delta_result.scalars().all()

    return {
        "finding": _serialize_result(finding),
        "agent_outputs": [
            {
                "agent_id": a.agent_id,
                "output": a.output,
                "confidence": a.confidence,
                "duration_ms": a.duration_ms,
                "error": a.error,
            }
            for a in agents
        ],
        "deltas": [
            {
                "delta_type": d.delta_type,
                "description": d.description,
                "algorithmic_view": d.algorithmic_view,
                "contextual_view": d.contextual_view,
            }
            for d in deltas
        ],
    }


@router.get("/analysis/findings/{result_id}/followups")
async def get_followups(
    result_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Get follow-up findings spawned from a parent finding (Gap #7)."""
    result = await session.execute(
        select(AnalysisResult)
        .where(AnalysisResult.parent_result_id == result_id)
        .order_by(AnalysisResult.final_score.desc())
    )
    followups = result.scalars().all()
    return {
        "parent_id": result_id,
        "followups": [_serialize_result(f) for f in followups],
    }


@router.get("/analysis/runs")
async def list_runs(
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """List recent analysis runs."""
    result = await session.execute(
        select(AnalysisRun)
        .order_by(AnalysisRun.started_at.desc())
        .limit(limit)
    )
    runs = result.scalars().all()
    return {
        "runs": [
            {
                "id": r.id,
                "situation_type": r.situation_type,
                "trigger": r.trigger,
                "status": r.status,
                "summary": r.summary,
                "started_at": str(r.started_at) if r.started_at else None,
                "completed_at": str(r.completed_at) if r.completed_at else None,
            }
            for r in runs
        ],
    }


@router.get("/analysis/runs/{run_id}")
async def get_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Get run detail with all findings."""
    run_result = await session.execute(
        select(AnalysisRun).where(AnalysisRun.id == run_id)
    )
    run = run_result.scalar_one_or_none()
    if not run:
        return {"error": "not found"}

    findings_result = await session.execute(
        select(AnalysisResult)
        .where(AnalysisResult.run_id == run_id)
        .order_by(AnalysisResult.final_score.desc())
    )
    findings = findings_result.scalars().all()

    return {
        "run": {
            "id": run.id,
            "situation_type": run.situation_type,
            "trigger": run.trigger,
            "status": run.status,
            "config": run.config,
            "time_scope": run.time_scope,
            "summary": run.summary,
            "started_at": str(run.started_at) if run.started_at else None,
            "completed_at": str(run.completed_at) if run.completed_at else None,
            "error": run.error,
        },
        "findings": [_serialize_result(f) for f in findings],
    }


@router.post("/analysis/findings/{result_id}/feedback")
async def submit_feedback(
    result_id: str,
    body: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Submit feedback on a finding."""
    feedback = AnalysisFeedback(
        result_id=result_id,
        feedback_type=body.feedback_type,
        comment=body.comment,
        session_id=body.session_id,
    )
    session.add(feedback)
    await session.commit()
    return {"id": feedback.id, "status": "recorded"}


@router.get("/analysis/learning/state")
async def get_learning_state(
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Get current learning parameters."""
    result = await session.execute(
        select(LearningState)
        .order_by(LearningState.version.desc())
        .limit(1)
    )
    state = result.scalar_one_or_none()
    if not state:
        return {"version": 0, "message": "no learning state yet"}

    return {
        "version": state.version,
        "interestingness_weights": state.interestingness_weights,
        "agent_calibration": state.agent_calibration,
        "operation_preferences": state.operation_preferences,
        "tier_budgets": state.tier_budgets,
        "feedback_count": state.feedback_count,
        "computed_at": str(state.computed_at) if state.computed_at else None,
    }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_result(r: AnalysisResult) -> dict[str, Any]:
    return {
        "id": r.id,
        "operation_type": r.operation_type,
        "table_name": r.table_name,
        "tier": r.tier,
        "target": r.target,
        "segmenters": r.segmenters,
        "controls": r.controls,
        "join_spec": r.join_spec,
        "interestingness_score": r.interestingness_score,
        "interestingness_breakdown": r.interestingness_breakdown,
        "quality_verdict": r.quality_verdict,
        "domain_relevance": r.domain_relevance,
        "temporal_context": r.temporal_context,
        "delta_type": r.delta_type,
        "final_score": r.final_score,
        "parent_result_id": r.parent_result_id,
        "result_data": r.result_data,
        "created_at": str(r.created_at) if r.created_at else None,
    }
