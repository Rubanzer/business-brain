"""Natural language → structured alert rule parser using LLM."""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import TableProfile

logger = logging.getLogger(__name__)

_PARSE_PROMPT = """You are an alert configuration assistant for a manufacturing intelligence platform.

The user wants to create an alert rule. Parse their natural language description into a structured JSON alert configuration.

Available tables and columns:
{schema_context}

Parse the user's request into this JSON format:
{{
  "name": "Short alert name",
  "table": "target_table_name",
  "column": "column_to_monitor",
  "condition": "greater_than | less_than | equals | not_equals | between | trend_down | trend_up | absent",
  "threshold": <number or null>,
  "threshold_upper": <number or null (for 'between')>,
  "check_trigger": "on_data_change",
  "rule_type": "threshold | trend | absence | cross_source | composite",
  "notification_channel": "telegram | feed",
  "message_template": "Alert message with {{column_name}} placeholder for the actual value"
}}

Rules:
- "greater_than" means value > threshold
- "less_than" means value < threshold
- "trend_down" means the column value has been decreasing (specify consecutive data points as threshold)
- "trend_up" means increasing
- "absent" means no new data for threshold minutes
- If user mentions "telegram", set notification_channel to "telegram", otherwise "feed"
- Use the exact table and column names from the schema
- If you can't determine the table/column, use your best guess based on the context

User's request: {user_input}

Respond with ONLY the JSON object, no explanation."""


async def parse_alert_natural_language(
    session: AsyncSession,
    user_input: str,
) -> dict[str, Any]:
    """Parse natural language alert description into a structured rule config.

    Args:
        session: DB session for loading schema context.
        user_input: The natural language alert description.

    Returns:
        Parsed alert configuration dict.
    """
    # Build schema context from profiled tables
    result = await session.execute(select(TableProfile))
    profiles = list(result.scalars().all())

    schema_lines = []
    for profile in profiles:
        cls = profile.column_classification
        if not cls or "columns" not in cls:
            continue
        cols = cls["columns"]
        col_descs = []
        for col_name, info in cols.items():
            sem_type = info.get("semantic_type", "unknown")
            col_descs.append(f"  - {col_name} ({sem_type})")
        schema_lines.append(f"Table: {profile.table_name}\n" + "\n".join(col_descs))

    schema_context = "\n\n".join(schema_lines) if schema_lines else "No tables profiled yet."

    # Call LLM
    from business_brain.analysis.tools.llm_gateway import reason as _llm_reason

    prompt = _PARSE_PROMPT.format(schema_context=schema_context, user_input=user_input)
    text = await _llm_reason(prompt)
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    if text.startswith("json"):
        text = text[4:]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON: %s", text)
        raise ValueError(f"Could not parse alert rule from: {user_input}")

    return parsed


def build_confirmation_message(parsed_rule: dict) -> str:
    """Build a human-readable confirmation message for the parsed alert rule."""
    name = parsed_rule.get("name", "Unnamed Alert")
    table = parsed_rule.get("table", "unknown")
    column = parsed_rule.get("column", "unknown")
    condition = parsed_rule.get("condition", "unknown")
    threshold = parsed_rule.get("threshold")
    channel = parsed_rule.get("notification_channel", "feed")

    condition_text = {
        "greater_than": f"exceeds {threshold}",
        "less_than": f"drops below {threshold}",
        "equals": f"equals {threshold}",
        "not_equals": f"is not {threshold}",
        "between": f"is between {threshold} and {parsed_rule.get('threshold_upper')}",
        "trend_down": f"decreases for {threshold} consecutive readings",
        "trend_up": f"increases for {threshold} consecutive readings",
        "absent": f"has no new data for {threshold} minutes",
    }.get(condition, f"{condition} {threshold}")

    delivery = "Telegram" if channel == "telegram" else "the Feed"

    return (
        f"I'll monitor **{column}** in **{table}**. "
        f"When it {condition_text}, I'll send you an alert on {delivery}. "
        f"Alert name: \"{name}\""
    )
