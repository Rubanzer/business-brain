"""Alert evaluation engine — checks all active alert rules against current data."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.v3_models import AlertEvent, AlertRule

logger = logging.getLogger(__name__)


async def evaluate_all_alerts(session: AsyncSession) -> list[AlertEvent]:
    """Evaluate all active alert rules and return triggered events.

    Called after every data change (sync/upload) or manually.
    """
    now = datetime.now(timezone.utc)

    result = await session.execute(
        select(AlertRule).where(
            AlertRule.active == True,  # noqa: E712
        )
    )
    rules = list(result.scalars().all())
    events: list[AlertEvent] = []

    for rule in rules:
        # Skip paused rules
        if rule.paused_until and rule.paused_until > now:
            continue

        try:
            event = await _evaluate_rule(session, rule)
            if event:
                events.append(event)
        except Exception:
            logger.exception("Failed to evaluate alert rule %s", rule.name)

    return events


async def _evaluate_rule(session: AsyncSession, rule: AlertRule) -> AlertEvent | None:
    """Evaluate a single alert rule against current data.

    Returns:
        An AlertEvent if the rule triggered, None otherwise.
    """
    config = rule.rule_config or {}
    rule_type = rule.rule_type

    if rule_type == "threshold":
        return await _eval_threshold(session, rule, config)
    elif rule_type == "trend":
        return await _eval_trend(session, rule, config)
    elif rule_type == "absence":
        return await _eval_absence(session, rule, config)
    elif rule_type == "composite":
        return await _eval_threshold(session, rule, config)  # same logic, different source
    elif rule_type == "cross_source":
        return await _eval_cross_source(session, rule, config)
    else:
        logger.warning("Unknown rule type: %s", rule_type)
        return None


async def _eval_threshold(
    session: AsyncSession,
    rule: AlertRule,
    config: dict,
) -> AlertEvent | None:
    """Evaluate a threshold-based alert rule."""
    table = _safe_name(config.get("table", ""))
    column = _safe_name(config.get("column", ""))
    condition = config.get("condition", "greater_than")
    threshold = config.get("threshold")

    if not table or not column or threshold is None:
        return None

    try:
        # Get the latest value (most recent row)
        query = f'SELECT "{column}" FROM "{table}" ORDER BY ctid DESC LIMIT 1'
        result = await session.execute(sql_text(query))
        row = result.fetchone()
        if not row:
            return None

        current_value = row[0]
        try:
            current_num = float(str(current_value).replace(",", ""))
        except (ValueError, TypeError):
            return None

        threshold_num = float(threshold)
        triggered = False

        if condition == "greater_than":
            triggered = current_num > threshold_num
        elif condition == "less_than":
            triggered = current_num < threshold_num
        elif condition == "equals":
            triggered = current_num == threshold_num
        elif condition == "not_equals":
            triggered = current_num != threshold_num
        elif condition == "between":
            upper = float(config.get("threshold_upper", threshold))
            triggered = threshold_num <= current_num <= upper

        if triggered:
            return await _create_event(session, rule, str(current_value), str(threshold))

    except Exception:
        logger.exception("Threshold evaluation failed for rule %s", rule.name)

    return None


async def _eval_trend(
    session: AsyncSession,
    rule: AlertRule,
    config: dict,
) -> AlertEvent | None:
    """Evaluate a trend-based alert rule (N consecutive increases/decreases)."""
    table = _safe_name(config.get("table", ""))
    column = _safe_name(config.get("column", ""))
    condition = config.get("condition", "trend_down")
    consecutive = int(config.get("threshold", 3))

    if not table or not column:
        return None

    try:
        query = f'SELECT "{column}" FROM "{table}" ORDER BY ctid DESC LIMIT {consecutive + 1}'
        result = await session.execute(sql_text(query))
        rows = result.fetchall()

        if len(rows) < consecutive + 1:
            return None

        values = []
        for row in rows:
            try:
                values.append(float(str(row[0]).replace(",", "")))
            except (ValueError, TypeError):
                return None

        # Values are in reverse chronological order
        values.reverse()

        if condition == "trend_down":
            all_decreasing = all(values[i] > values[i + 1] for i in range(len(values) - 1))
            if all_decreasing:
                return await _create_event(
                    session, rule,
                    f"Decreased {consecutive} times: {values}",
                    f"{consecutive} consecutive decreases",
                )
        elif condition == "trend_up":
            all_increasing = all(values[i] < values[i + 1] for i in range(len(values) - 1))
            if all_increasing:
                return await _create_event(
                    session, rule,
                    f"Increased {consecutive} times: {values}",
                    f"{consecutive} consecutive increases",
                )

    except Exception:
        logger.exception("Trend evaluation failed for rule %s", rule.name)

    return None


async def _eval_absence(
    session: AsyncSession,
    rule: AlertRule,
    config: dict,
) -> AlertEvent | None:
    """Evaluate a data absence alert (no new data for N minutes)."""
    table = _safe_name(config.get("table", ""))
    minutes = int(config.get("threshold", 30))

    if not table:
        return None

    try:
        # Check if the table has a temporal column with recent data
        query = f"SELECT MAX(ctid) FROM \"{table}\""
        result = await session.execute(sql_text(query))
        # For absence detection, we'd ideally check timestamps
        # For now, flag if the table exists but no data change was recorded recently
        from business_brain.db.v3_models import DataSource
        src_result = await session.execute(
            select(DataSource).where(DataSource.table_name == table)
        )
        source = src_result.scalar_one_or_none()

        if source and source.last_sync_at:
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            last = source.last_sync_at
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            elapsed = now - last
            if elapsed > timedelta(minutes=minutes):
                return await _create_event(
                    session, rule,
                    f"No data for {int(elapsed.total_seconds() / 60)} minutes",
                    f"{minutes} minute threshold",
                )

    except Exception:
        logger.exception("Absence evaluation failed for rule %s", rule.name)

    return None


async def _eval_cross_source(
    session: AsyncSession,
    rule: AlertRule,
    config: dict,
) -> AlertEvent | None:
    """Evaluate a cross-source comparison alert."""
    table_a = _safe_name(config.get("table", ""))
    table_b = _safe_name(config.get("table_b", ""))
    column_a = _safe_name(config.get("column", ""))
    column_b = _safe_name(config.get("column_b", column_a))
    threshold = float(config.get("threshold", 5))

    if not all([table_a, table_b, column_a, column_b]):
        return None

    try:
        query_a = f'SELECT "{column_a}" FROM "{table_a}" ORDER BY ctid DESC LIMIT 1'
        query_b = f'SELECT "{column_b}" FROM "{table_b}" ORDER BY ctid DESC LIMIT 1'

        result_a = await session.execute(sql_text(query_a))
        result_b = await session.execute(sql_text(query_b))

        row_a = result_a.fetchone()
        row_b = result_b.fetchone()

        if not row_a or not row_b:
            return None

        val_a = float(str(row_a[0]).replace(",", ""))
        val_b = float(str(row_b[0]).replace(",", ""))
        diff = abs(val_a - val_b)

        if diff > threshold:
            return await _create_event(
                session, rule,
                f"{table_a}.{column_a}={val_a} vs {table_b}.{column_b}={val_b} (diff={diff})",
                str(threshold),
            )

    except Exception:
        logger.exception("Cross-source evaluation failed for rule %s", rule.name)

    return None


async def _create_event(
    session: AsyncSession,
    rule: AlertRule,
    trigger_value: str,
    threshold_value: str,
) -> AlertEvent:
    """Create an alert event and update the rule's trigger info."""
    event = AlertEvent(
        alert_rule_id=rule.id,
        trigger_value=trigger_value,
        threshold_value=threshold_value,
        context={"rule_name": rule.name, "rule_type": rule.rule_type},
    )
    session.add(event)

    rule.last_triggered_at = datetime.now(timezone.utc)
    rule.trigger_count = (rule.trigger_count or 0) + 1
    await session.flush()

    # Send notification
    await _send_notification(session, rule, event)

    return event


async def _send_notification(
    session: AsyncSession,
    rule: AlertRule,
    event: AlertEvent,
) -> None:
    """Send alert notification via the configured channel."""
    if rule.notification_channel == "telegram":
        try:
            from business_brain.action.telegram_bot import send_alert
            config = rule.notification_config or {}
            chat_id = config.get("chat_id")
            if chat_id:
                message = _format_alert_message(rule, event)
                await send_alert(chat_id, message)
                event.notification_sent = True
            else:
                event.notification_error = "No chat_id configured"
        except Exception as exc:
            event.notification_error = str(exc)
            logger.exception("Failed to send Telegram notification for rule %s", rule.name)
    else:
        # Feed-only notification — event creation is sufficient
        event.notification_sent = True

    await session.flush()


def _format_alert_message(rule: AlertRule, event: AlertEvent) -> str:
    """Format an alert message using the rule's template or default format."""
    if rule.message_template:
        # Simple template substitution
        msg = rule.message_template
        msg = msg.replace("{{value}}", event.trigger_value or "")
        msg = msg.replace("{{threshold}}", event.threshold_value or "")
        return msg

    config = rule.rule_config or {}
    return (
        f"ALERT: {rule.name}\n\n"
        f"Current: {event.trigger_value}\n"
        f"Threshold: {event.threshold_value}\n"
        f"Source: {config.get('table', 'unknown')}.{config.get('column', 'unknown')}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


def _safe_name(name: str) -> str:
    """Sanitize a table or column name."""
    return re.sub(r"[^a-zA-Z0-9_]", "", name or "")


# ---------------------------------------------------------------------------
# Alert CRUD helpers
# ---------------------------------------------------------------------------


async def deploy_alert(
    session: AsyncSession,
    parsed_rule: dict,
    original_text: str,
    notification_config: dict | None = None,
) -> AlertRule:
    """Create a new alert rule from a parsed configuration."""
    rule = AlertRule(
        name=parsed_rule.get("name", "Unnamed Alert"),
        description=original_text,
        rule_config={
            "table": parsed_rule.get("table"),
            "column": parsed_rule.get("column"),
            "condition": parsed_rule.get("condition"),
            "threshold": parsed_rule.get("threshold"),
            "threshold_upper": parsed_rule.get("threshold_upper"),
            "table_b": parsed_rule.get("table_b"),
            "column_b": parsed_rule.get("column_b"),
        },
        rule_type=parsed_rule.get("rule_type", "threshold"),
        check_trigger=parsed_rule.get("check_trigger", "on_data_change"),
        notification_channel=parsed_rule.get("notification_channel", "feed"),
        notification_config=notification_config or {},
        message_template=parsed_rule.get("message_template"),
        active=True,
    )
    session.add(rule)
    await session.commit()
    await session.refresh(rule)
    return rule


async def get_all_alerts(session: AsyncSession) -> list[AlertRule]:
    """Get all alert rules."""
    result = await session.execute(select(AlertRule).order_by(AlertRule.created_at.desc()))
    return list(result.scalars().all())


async def get_alert(session: AsyncSession, alert_id: str) -> AlertRule | None:
    """Get a single alert rule."""
    result = await session.execute(select(AlertRule).where(AlertRule.id == alert_id))
    return result.scalar_one_or_none()


async def get_alert_events(session: AsyncSession, alert_id: str, limit: int = 50) -> list[AlertEvent]:
    """Get trigger history for an alert rule."""
    result = await session.execute(
        select(AlertEvent)
        .where(AlertEvent.alert_rule_id == alert_id)
        .order_by(AlertEvent.triggered_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def pause_alert(
    session: AsyncSession,
    alert_id: str,
    until: datetime | None = None,
) -> AlertRule | None:
    """Pause an alert rule, optionally until a specific time."""
    rule = await get_alert(session, alert_id)
    if not rule:
        return None
    rule.active = False
    rule.paused_until = until
    await session.commit()
    return rule


async def resume_alert(session: AsyncSession, alert_id: str) -> AlertRule | None:
    """Resume a paused alert rule."""
    rule = await get_alert(session, alert_id)
    if not rule:
        return None
    rule.active = True
    rule.paused_until = None
    await session.commit()
    return rule


async def update_alert(session: AsyncSession, alert_id: str, updates: dict) -> AlertRule | None:
    """Update an alert rule's configuration."""
    rule = await get_alert(session, alert_id)
    if not rule:
        return None

    if "name" in updates:
        rule.name = updates["name"]
    if "threshold" in updates and rule.rule_config:
        config = dict(rule.rule_config)
        config["threshold"] = updates["threshold"]
        rule.rule_config = config
    if "notification_channel" in updates:
        rule.notification_channel = updates["notification_channel"]
    if "message_template" in updates:
        rule.message_template = updates["message_template"]

    await session.commit()
    return rule


async def delete_alert(session: AsyncSession, alert_id: str) -> bool:
    """Delete an alert rule."""
    rule = await get_alert(session, alert_id)
    if not rule:
        return False
    await session.delete(rule)
    await session.commit()
    return True


# ---------------------------------------------------------------------------
# Shift digest — batched alert summary
# ---------------------------------------------------------------------------

async def send_shift_digest(
    session: AsyncSession,
    since_hours: int = 8,
) -> dict:
    """Generate and send a batched shift digest instead of individual alerts.

    Aggregates all alert events from the last N hours into a single
    structured message. Sends via Telegram if configured.

    Args:
        session: Database session.
        since_hours: Look-back window (default 8 = one shift).

    Returns:
        Dict with digest stats and message.
    """
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    # Get all events since cutoff
    result = await session.execute(
        select(AlertEvent)
        .where(AlertEvent.triggered_at >= cutoff)
        .order_by(AlertEvent.triggered_at.desc())
    )
    events = list(result.scalars().all())

    if not events:
        return {"sent": False, "reason": "no_events", "event_count": 0}

    # Categorize events by severity (inferred from rule config)
    critical_events = []
    warning_events = []
    info_events = []

    for event in events:
        ctx = event.context if isinstance(event.context, dict) else {}
        # Try to determine severity from the rule
        rule_id = event.alert_rule_id
        rule_result = await session.execute(
            select(AlertRule).where(AlertRule.id == rule_id)
        )
        rule = rule_result.scalar_one_or_none()

        if rule:
            # Infer severity from rule config
            config = rule.rule_config or {}
            rule_name = rule.name
        else:
            config = {}
            rule_name = ctx.get("rule_name", "Unknown")

        event_info = {
            "rule_name": rule_name,
            "trigger_value": event.trigger_value,
            "threshold_value": event.threshold_value,
            "triggered_at": event.triggered_at,
        }

        # Classify: critical if "critical" in name or config, otherwise warning
        combined = f"{rule_name} {str(config)}".lower()
        if "critical" in combined or "emergency" in combined:
            critical_events.append(event_info)
        elif "warning" in combined or "alert" in combined:
            warning_events.append(event_info)
        else:
            info_events.append(event_info)

    # Build digest message
    message = _format_digest_message(
        critical_events, warning_events, info_events, since_hours,
    )

    # Try to send via Telegram
    telegram_sent = False
    try:
        # Look for any rule with telegram config to get the chat_id
        tg_result = await session.execute(
            select(AlertRule).where(
                AlertRule.notification_channel == "telegram",
                AlertRule.active == True,  # noqa: E712
            ).limit(1)
        )
        tg_rule = tg_result.scalar_one_or_none()

        if tg_rule:
            chat_id = (tg_rule.notification_config or {}).get("chat_id")
            if chat_id:
                from business_brain.action.telegram_bot import send_alert
                await send_alert(chat_id, message)
                telegram_sent = True
    except Exception:
        logger.exception("Failed to send shift digest via Telegram")

    return {
        "sent": True,
        "telegram_sent": telegram_sent,
        "event_count": len(events),
        "critical_count": len(critical_events),
        "warning_count": len(warning_events),
        "info_count": len(info_events),
        "message": message,
    }


def _format_digest_message(
    critical: list[dict],
    warnings: list[dict],
    info: list[dict],
    hours: int,
) -> str:
    """Format a digest message for shift summary."""
    total = len(critical) + len(warnings) + len(info)

    lines = [
        f"\U0001f3ed Shift Summary ({hours}h)",
        "",
        f"\U0001f534 {len(critical)} critical alerts",
        f"\U0001f7e1 {len(warnings)} warnings",
        f"\U0001f7e2 {len(info)} info",
        f"\U0001f4ca Total: {total} events",
        "",
    ]

    if critical:
        lines.append("CRITICAL:")
        for e in critical[:5]:
            lines.append(f"  \u2022 {e['rule_name']}: {e['trigger_value']}")

    if warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for e in warnings[:5]:
            lines.append(f"  \u2022 {e['rule_name']}: {e['trigger_value']}")

    if info and not critical and not warnings:
        lines.append("INFO:")
        for e in info[:3]:
            lines.append(f"  \u2022 {e['rule_name']}: {e['trigger_value']}")

    return "\n".join(lines)
