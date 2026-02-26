"""Pattern Memory routes."""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["patterns"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class PatternCreateRequest(BaseModel):
    name: str
    source_table: str
    conditions: list[dict]
    time_window_minutes: int = 15
    description: str = ""


class PatternFeedbackRequest(BaseModel):
    outcome: str  # confirmed_breakdown / false_positive


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/patterns")
async def create_pattern_endpoint(
    req: PatternCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Create a new pattern from user labeling."""
    from business_brain.discovery.pattern_memory import learn_pattern
    pattern = await learn_pattern(
        session, req.name, req.source_table, req.conditions,
        req.time_window_minutes, req.description,
    )
    return {"status": "created", "pattern_id": pattern.id, "name": pattern.name}


@router.get("/patterns")
async def list_patterns(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all patterns."""
    from business_brain.discovery.pattern_memory import get_all_patterns
    patterns = await get_all_patterns(session)
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "source_tables": p.source_tables,
            "conditions": p.conditions,
            "time_window_minutes": p.time_window_minutes,
            "confidence": p.confidence,
            "match_count": p.match_count,
            "false_positive_count": p.false_positive_count,
            "active": p.active,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "last_matched_at": p.last_matched_at.isoformat() if p.last_matched_at else None,
        }
        for p in patterns
    ]


@router.get("/patterns/{pattern_id}")
async def get_pattern_detail(pattern_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get pattern details including match history."""
    from business_brain.discovery.pattern_memory import get_pattern, get_pattern_matches
    pattern = await get_pattern(session, pattern_id)
    if not pattern:
        return {"error": "Pattern not found"}
    matches = await get_pattern_matches(session, pattern_id)
    return {
        "id": pattern.id,
        "name": pattern.name,
        "description": pattern.description,
        "source_tables": pattern.source_tables,
        "conditions": pattern.conditions,
        "confidence": pattern.confidence,
        "historical_occurrences": pattern.historical_occurrences,
        "match_count": pattern.match_count,
        "matches": [
            {
                "id": m.id,
                "matched_at": m.matched_at.isoformat() if m.matched_at else None,
                "similarity_score": m.similarity_score,
                "outcome": m.outcome,
            }
            for m in matches
        ],
    }


@router.post("/patterns/matches/{match_id}/feedback")
async def pattern_feedback_endpoint(
    match_id: int,
    req: PatternFeedbackRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Confirm or reject a pattern match."""
    from business_brain.discovery.pattern_memory import confirm_match
    match = await confirm_match(session, match_id, req.outcome)
    if not match:
        return {"error": "Match not found"}
    return {"status": "updated", "outcome": match.outcome}


@router.delete("/patterns/{pattern_id}")
async def delete_pattern_endpoint(pattern_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete a pattern."""
    from business_brain.discovery.pattern_memory import delete_pattern
    deleted = await delete_pattern(session, pattern_id)
    return {"status": "deleted"} if deleted else {"error": "Pattern not found"}
