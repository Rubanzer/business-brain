"""Alert System routes."""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["alerts"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class AlertParseRequest(BaseModel):
    text: str


class AlertDeployRequest(BaseModel):
    text: str
    parsed_rule: Optional[dict] = None
    notification_config: Optional[dict] = None


class AlertUpdateRequest(BaseModel):
    name: Optional[str] = None
    threshold: Optional[float] = None
    notification_channel: Optional[str] = None
    message_template: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/alerts/parse")
async def parse_alert_endpoint(
    req: AlertParseRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Parse natural language into a structured alert rule (without deploying)."""
    from business_brain.action.alert_parser import build_confirmation_message, parse_alert_natural_language

    try:
        parsed = await parse_alert_natural_language(session, req.text)
        confirmation = build_confirmation_message(parsed)
        return {
            "status": "parsed",
            "parsed_rule": parsed,
            "confirmation": confirmation,
        }
    except Exception as exc:
        logger.exception("Failed to parse alert")
        return {"error": str(exc)}


@router.post("/alerts/deploy")
async def deploy_alert_endpoint(
    req: AlertDeployRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Deploy an alert rule."""
    from business_brain.action.alert_parser import build_confirmation_message, parse_alert_natural_language
    from business_brain.action.alert_engine import deploy_alert

    try:
        if req.parsed_rule:
            parsed = req.parsed_rule
        else:
            parsed = await parse_alert_natural_language(session, req.text)

        rule = await deploy_alert(session, parsed, req.text, req.notification_config)
        confirmation = build_confirmation_message(parsed)
        return {
            "status": "deployed",
            "alert_id": rule.id,
            "parsed_rule": parsed,
            "confirmation": confirmation,
        }
    except Exception as exc:
        logger.exception("Failed to deploy alert")
        return {"error": str(exc)}


@router.get("/alerts")
async def list_alerts(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all alert rules."""
    from business_brain.action.alert_engine import get_all_alerts

    rules = await get_all_alerts(session)
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "rule_type": r.rule_type,
            "rule_config": r.rule_config,
            "notification_channel": r.notification_channel,
            "active": r.active,
            "paused_until": r.paused_until.isoformat() if r.paused_until else None,
            "last_triggered_at": r.last_triggered_at.isoformat() if r.last_triggered_at else None,
            "trigger_count": r.trigger_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rules
    ]


@router.get("/alerts/{alert_id}")
async def get_alert_detail(alert_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get alert rule details."""
    from business_brain.action.alert_engine import get_alert
    rule = await get_alert(session, alert_id)
    if not rule:
        return {"error": "Alert not found"}
    return {
        "id": rule.id,
        "name": rule.name,
        "description": rule.description,
        "rule_type": rule.rule_type,
        "rule_config": rule.rule_config,
        "notification_channel": rule.notification_channel,
        "message_template": rule.message_template,
        "active": rule.active,
        "paused_until": rule.paused_until.isoformat() if rule.paused_until else None,
        "last_triggered_at": rule.last_triggered_at.isoformat() if rule.last_triggered_at else None,
        "trigger_count": rule.trigger_count,
    }


@router.put("/alerts/{alert_id}")
async def update_alert_endpoint(
    alert_id: str,
    req: AlertUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update an alert rule."""
    from business_brain.action.alert_engine import update_alert
    updates = req.model_dump(exclude_none=True)
    rule = await update_alert(session, alert_id, updates)
    if not rule:
        return {"error": "Alert not found"}
    return {"status": "updated", "alert_id": rule.id}


@router.post("/alerts/{alert_id}/pause")
async def pause_alert_endpoint(alert_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Pause an alert rule."""
    from business_brain.action.alert_engine import pause_alert
    rule = await pause_alert(session, alert_id)
    return {"status": "paused"} if rule else {"error": "Alert not found"}


@router.post("/alerts/{alert_id}/resume")
async def resume_alert_endpoint(alert_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Resume a paused alert rule."""
    from business_brain.action.alert_engine import resume_alert
    rule = await resume_alert(session, alert_id)
    return {"status": "resumed"} if rule else {"error": "Alert not found"}


@router.delete("/alerts/{alert_id}")
async def delete_alert_endpoint(alert_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete an alert rule."""
    from business_brain.action.alert_engine import delete_alert
    deleted = await delete_alert(session, alert_id)
    return {"status": "deleted"} if deleted else {"error": "Alert not found"}


@router.get("/alerts/{alert_id}/events")
async def get_alert_events_endpoint(alert_id: str, session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get trigger history for an alert rule."""
    from business_brain.action.alert_engine import get_alert_events
    events = await get_alert_events(session, alert_id)
    return [
        {
            "id": e.id,
            "triggered_at": e.triggered_at.isoformat() if e.triggered_at else None,
            "trigger_value": e.trigger_value,
            "threshold_value": e.threshold_value,
            "notification_sent": e.notification_sent,
            "notification_error": e.notification_error,
        }
        for e in events
    ]


@router.post("/alerts/evaluate")
async def evaluate_alerts_endpoint(
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Manually trigger evaluation of all alert rules."""
    from business_brain.action.alert_engine import evaluate_all_alerts
    events = await evaluate_all_alerts(session)
    return {"status": "evaluated", "triggered": len(events)}


@router.get("/alerts/{alert_id}/preview")
async def preview_alert(
    alert_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Dry-run an alert rule — shows what it would match right now without triggering."""
    from business_brain.action.alert_engine import get_alert
    from sqlalchemy import text as sql_text
    import re

    rule = await get_alert(session, alert_id)
    if not rule:
        return {"error": "Alert not found"}

    config = rule.rule_config or {}
    table = config.get("table", "")
    column = config.get("column", "")
    condition = config.get("condition", "")
    threshold = config.get("threshold")

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table)
    safe_col = re.sub(r"[^a-zA-Z0-9_]", "", column)

    if not safe_table or not safe_col:
        return {"error": "Alert rule missing table or column", "rule_config": config}

    try:
        query = f'SELECT AVG("{safe_col}") as avg_val, MIN("{safe_col}") as min_val, MAX("{safe_col}") as max_val, COUNT(*) as row_count FROM "{safe_table}"'
        result = await session.execute(sql_text(query))
        row = dict(result.fetchone()._mapping)

        op_map = {
            "greater_than": ">", "less_than": "<",
            "equals": "=", "not_equals": "!=",
        }
        op = op_map.get(condition, ">")
        match_query = f'SELECT COUNT(*) as matches FROM "{safe_table}" WHERE "{safe_col}" {op} :threshold'
        match_result = await session.execute(sql_text(match_query), {"threshold": threshold})
        match_count = match_result.scalar() or 0

        would_trigger = match_count > 0

        return {
            "alert_id": alert_id,
            "alert_name": rule.name,
            "would_trigger": would_trigger,
            "current_stats": {
                "avg": row.get("avg_val"),
                "min": row.get("min_val"),
                "max": row.get("max_val"),
                "row_count": row.get("row_count"),
            },
            "threshold": threshold,
            "condition": condition,
            "matching_rows": match_count,
            "source": f"{safe_table}.{safe_col}",
        }
    except Exception as exc:
        logger.exception("Alert preview failed")
        return {"error": str(exc), "alert_id": alert_id}
