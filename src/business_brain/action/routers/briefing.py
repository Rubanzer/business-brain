"""Daily Briefing route."""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["briefing"])


@router.get("/briefing")
async def get_briefing(
    hours: int = 24,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get a daily briefing summarizing recent insights and alerts."""
    from business_brain.discovery.daily_briefing import generate_daily_briefing

    try:
        return await generate_daily_briefing(session, since_hours=hours)
    except Exception:
        logger.exception("Daily briefing generation failed")
        return {
            "error": "Failed to generate briefing",
            "overall_status": "green",
            "summary": "Unable to generate briefing at this time.",
            "sections": [],
            "top_actions": [],
            "alert_summary": {"total": 0, "events": []},
            "metrics": {
                "total_insights": 0,
                "critical_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "alert_count": 0,
                "tables_analyzed": 0,
            },
        }
