"""Deep Tier (Claude API) routes — trigger, status, results, and task management."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.action.dependencies import get_current_user
from business_brain.db.connection import async_session, get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["deep-tier"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class DeepAnalysisRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    # Optional: pass fast tier context for manual trigger
    fast_findings: Optional[list] = None
    sql_query: Optional[str] = None
    tables_used: Optional[list] = None
    fast_confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/deep-tier/status")
async def deep_tier_status() -> dict:
    """Check if Deep Tier (Claude API) is configured and available."""
    from business_brain.cognitive.deep_tier import is_available
    from config.settings import settings

    return {
        "available": is_available(),
        "model": settings.claude_model if is_available() else None,
        "auto_threshold": settings.deep_tier_auto_threshold,
    }


@router.post("/deep-tier/analyze")
async def trigger_deep_analysis(
    req: DeepAnalysisRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    authorization: str = Header(default=""),
) -> dict:
    """Manually trigger a Deep Tier analysis (the 'Investigate Deeper' button).

    Creates a task in the queue and starts execution in the background.
    """
    from business_brain.cognitive.deep_tier import is_available, create_task

    if not is_available():
        return {
            "error": "Deep Tier not configured — set ANTHROPIC_API_KEY environment variable",
            "status": "unavailable",
        }

    # Get user if authenticated
    user = await get_current_user(authorization)
    requested_by = user.get("sub", "anonymous") if user else "anonymous"

    # Create the task
    fast_result = {}
    if req.fast_findings:
        fast_result["findings"] = req.fast_findings

    task_info = await create_task(
        session,
        question=req.question,
        fast_tier_result=fast_result,
        sql_query=req.sql_query or "",
        tables_used=req.tables_used or [],
        fast_confidence=req.fast_confidence or 0.5,
        session_id=req.session_id or "",
        source_tier="manual",
        requested_by=requested_by,
        priority=1,  # manual requests get higher priority
    )

    # Execute in background
    task_id = task_info["task_id"]
    background_tasks.add_task(_execute_task_background, task_id)

    return {
        "status": "queued",
        "task_id": task_id,
        "message": "Deep analysis started. Poll /deep-tier/task/{task_id} for results.",
    }


@router.get("/deep-tier/task/{task_id}")
async def get_task(
    task_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get the status and result of a Deep Tier analysis task."""
    from business_brain.cognitive.deep_tier import get_task_status

    result = await get_task_status(session, task_id)
    if result is None:
        return {"error": "Task not found"}
    return result


@router.get("/deep-tier/tasks")
async def list_deep_tasks(
    status: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
    session: AsyncSession = Depends(get_session),
) -> list:
    """List Deep Tier analysis tasks."""
    from business_brain.cognitive.deep_tier import list_tasks

    return await list_tasks(session, status=status, limit=limit)


@router.post("/deep-tier/task/{task_id}/retry")
async def retry_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Retry a failed Deep Tier task."""
    from business_brain.cognitive.deep_tier import get_task_status

    task = await get_task_status(session, task_id)
    if task is None:
        return {"error": "Task not found"}
    if task["status"] != "failed":
        return {"error": f"Task is {task['status']}, only failed tasks can be retried"}

    background_tasks.add_task(_execute_task_background, task_id)
    return {"status": "queued", "task_id": task_id, "message": "Retry started."}


# ---------------------------------------------------------------------------
# Background execution helper
# ---------------------------------------------------------------------------


async def _execute_task_background(task_id: str):
    """Execute a Deep Tier task in the background with its own session."""
    try:
        from business_brain.cognitive.deep_tier import execute_task

        async with async_session() as session:
            result = await execute_task(session, task_id)
            if result.get("status") == "failed":
                logger.warning("Deep Tier task %s failed: %s", task_id, result.get("error"))
            else:
                logger.info("Deep Tier task %s completed", task_id)
    except Exception:
        logger.exception("Background Deep Tier execution failed: %s", task_id)
