"""Company Onboarding & Metric Thresholds routes."""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["company"])


# ---------------------------------------------------------------------------
# Company Profile Routes
# ---------------------------------------------------------------------------


@router.get("/company")
async def get_company(session: AsyncSession = Depends(get_session)) -> dict:
    """Get company profile."""
    from business_brain.action.onboarding import compute_profile_completeness, get_company_profile
    profile = await get_company_profile(session)
    if not profile:
        return {"exists": False, "completeness": 0}
    return {
        "exists": True,
        "id": profile.id,
        "name": profile.name,
        "industry": profile.industry,
        "products": profile.products,
        "departments": profile.departments,
        "process_flow": profile.process_flow,
        "systems": profile.systems,
        "known_relationships": profile.known_relationships,
        "completeness": compute_profile_completeness(profile),
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


@router.put("/company")
async def update_company(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Update company profile."""
    from business_brain.action.onboarding import save_company_profile
    profile = await save_company_profile(session, body)
    return {"status": "updated", "id": profile.id}


@router.post("/company/onboard")
async def full_onboard(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Submit full onboarding data."""
    from business_brain.action.onboarding import compute_profile_completeness, save_company_profile
    profile = await save_company_profile(session, body)
    return {
        "status": "onboarded",
        "id": profile.id,
        "completeness": compute_profile_completeness(profile),
    }


# ---------------------------------------------------------------------------
# Metric Thresholds Routes
# ---------------------------------------------------------------------------


@router.get("/thresholds")
async def list_thresholds(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all metric thresholds."""
    from business_brain.action.onboarding import get_all_thresholds
    thresholds = await get_all_thresholds(session)
    return [
        {
            "id": t.id,
            "metric_name": t.metric_name,
            "table_name": t.table_name,
            "column_name": t.column_name,
            "unit": t.unit,
            "normal_min": t.normal_min,
            "normal_max": t.normal_max,
            "warning_min": t.warning_min,
            "warning_max": t.warning_max,
            "critical_min": t.critical_min,
            "critical_max": t.critical_max,
        }
        for t in thresholds
    ]


@router.post("/thresholds")
async def create_threshold_endpoint(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Create a metric threshold."""
    from business_brain.action.onboarding import create_threshold
    threshold = await create_threshold(session, body)
    return {"status": "created", "id": threshold.id}


@router.put("/thresholds/{threshold_id}")
async def update_threshold_endpoint(
    threshold_id: int,
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update a metric threshold."""
    from business_brain.action.onboarding import update_threshold
    t = await update_threshold(session, threshold_id, body)
    if not t:
        return {"error": "Threshold not found"}
    return {"status": "updated", "id": t.id}


@router.delete("/thresholds/{threshold_id}")
async def delete_threshold_endpoint(threshold_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete a metric threshold."""
    from business_brain.action.onboarding import delete_threshold
    deleted = await delete_threshold(session, threshold_id)
    return {"status": "deleted"} if deleted else {"error": "Threshold not found"}
