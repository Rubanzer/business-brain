"""Daily briefing generator — proactive intelligence summary for plant owners.

Aggregates recent insights and alerts into a structured briefing with
status indicators (green/yellow/red) and prioritized action items.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)


async def generate_daily_briefing(
    session: AsyncSession,
    since_hours: int = 24,
) -> dict:
    """Generate a daily briefing summarizing recent insights and alerts.

    Args:
        session: Database session.
        since_hours: Look-back window in hours (default 24).

    Returns:
        Structured JSON with sections, top_actions, overall_status, and summary.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    # 1. Fetch recent insights
    result = await session.execute(
        select(Insight)
        .where(Insight.discovered_at >= cutoff)
        .order_by(Insight.impact_score.desc())
    )
    recent_insights = list(result.scalars().all())

    # 2. Fetch recent alert events
    alert_events = []
    try:
        from business_brain.db.v3_models import AlertEvent
        ae_result = await session.execute(
            select(AlertEvent)
            .where(AlertEvent.triggered_at >= cutoff)
            .order_by(AlertEvent.triggered_at.desc())
        )
        alert_events = list(ae_result.scalars().all())
    except Exception:
        logger.debug("Could not fetch alert events for briefing")

    # 3. Get table profiles for context
    tp_result = await session.execute(select(TableProfile))
    profiles = list(tp_result.scalars().all())
    domain_map = {p.table_name: (p.domain_hint or "general") for p in profiles}

    # 4. Group insights by domain
    sections = _build_sections(recent_insights, domain_map)

    # 5. Extract top actions
    top_actions = _extract_top_actions(recent_insights)

    # 6. Determine overall status
    overall_status = _determine_status(recent_insights, alert_events)

    # 7. Generate executive summary
    summary = await _generate_summary(recent_insights, alert_events, sections)

    # 8. Alert summary
    alert_summary = _build_alert_summary(alert_events)

    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "period_hours": since_hours,
        "overall_status": overall_status,
        "summary": summary,
        "sections": sections,
        "top_actions": top_actions,
        "alert_summary": alert_summary,
        "metrics": {
            "total_insights": len(recent_insights),
            "critical_count": sum(1 for i in recent_insights if i.severity == "critical"),
            "warning_count": sum(1 for i in recent_insights if i.severity == "warning"),
            "info_count": sum(1 for i in recent_insights if i.severity == "info"),
            "alert_count": len(alert_events),
            "tables_analyzed": len({t for i in recent_insights for t in (i.source_tables or [])}),
        },
    }


def _build_sections(
    insights: list[Insight],
    domain_map: dict[str, str],
) -> list[dict]:
    """Group insights into domain sections."""
    domain_insights: dict[str, list[Insight]] = {}

    for insight in insights:
        tables = insight.source_tables or []
        # Find the most specific domain for this insight
        domains = set()
        for t in tables:
            d = domain_map.get(t, "general")
            domains.add(d)

        domain = next(iter(domains)) if domains else "general"
        # Map internal domain to display name
        domain_insights.setdefault(domain, []).append(insight)

    _DOMAIN_LABELS = {
        "manufacturing": "Manufacturing & Production",
        "quality": "Quality Control",
        "energy": "Energy & Power",
        "procurement": "Procurement & Suppliers",
        "logistics": "Logistics & Dispatch",
        "sales": "Sales & Revenue",
        "finance": "Finance & Costs",
        "hr": "Human Resources",
        "inventory": "Inventory & Stock",
        "marketing": "Marketing",
        "general": "General",
    }

    sections = []
    for domain, domain_ins in sorted(
        domain_insights.items(),
        key=lambda x: max((i.impact_score or 0) for i in x[1]),
        reverse=True,
    ):
        # Determine section status
        has_critical = any(i.severity == "critical" for i in domain_ins)
        has_warning = any(i.severity == "warning" for i in domain_ins)
        status = "red" if has_critical else "yellow" if has_warning else "green"

        # Top 3 insights for this section
        top_3 = sorted(domain_ins, key=lambda i: -(i.impact_score or 0))[:3]

        sections.append({
            "domain": domain,
            "label": _DOMAIN_LABELS.get(domain, domain.title()),
            "status": status,
            "insight_count": len(domain_ins),
            "top_insights": [
                {
                    "title": i.title,
                    "severity": i.severity,
                    "impact_score": i.impact_score,
                    "type": i.insight_type,
                }
                for i in top_3
            ],
        })

    return sections


def _extract_top_actions(insights: list[Insight]) -> list[str]:
    """Extract the most important actions from high-impact insights."""
    actions = []
    seen = set()

    for insight in sorted(insights, key=lambda i: -(i.impact_score or 0)):
        for action in (insight.suggested_actions or []):
            # Deduplicate similar actions
            action_key = action[:50].lower()
            if action_key not in seen:
                seen.add(action_key)
                actions.append(action)

        if len(actions) >= 5:
            break

    return actions[:5]


def _determine_status(
    insights: list[Insight],
    alert_events: list,
) -> str:
    """Determine overall briefing status: green/yellow/red."""
    # Red: any critical insight or alert
    if any(i.severity == "critical" for i in insights):
        return "red"
    if any(
        getattr(e, "context", {}).get("severity") == "critical"
        for e in alert_events
        if hasattr(e, "context") and isinstance(getattr(e, "context", None), dict)
    ):
        return "red"

    # Yellow: any warnings
    if any(i.severity == "warning" for i in insights):
        return "yellow"
    if alert_events:
        return "yellow"

    return "green"


async def _generate_summary(
    insights: list[Insight],
    alert_events: list,
    sections: list[dict],
) -> str:
    """Generate a 2-3 sentence executive summary.

    Uses LLM if available, otherwise builds a template-based summary.
    """
    if not insights:
        return "No new findings in this period. All systems operating normally."

    # Try LLM-powered summary
    try:
        from google import genai
        from config.settings import settings

        if settings.gemini_api_key:
            # Build insight summaries
            top_insights = sorted(insights, key=lambda i: -(i.impact_score or 0))[:5]
            summaries = [
                f"- [{i.severity}] {i.title}: {i.description[:100]}"
                for i in top_insights
            ]

            prompt = (
                "Write a 2-3 sentence executive briefing for a factory owner. "
                "Be direct, use specific numbers, state what needs attention.\n\n"
                f"Findings:\n" + "\n".join(summaries) + "\n\n"
                f"Alerts: {len(alert_events)} triggered\n"
                "Write the briefing (2-3 sentences, no bullet points):"
            )

            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            return response.text.strip()
    except Exception:
        logger.debug("LLM summary generation failed, using template")

    # Template-based fallback
    critical = sum(1 for i in insights if i.severity == "critical")
    warnings = sum(1 for i in insights if i.severity == "warning")
    total = len(insights)

    parts = [f"{total} findings from last {len(sections)} area{'s' if len(sections) != 1 else ''}."]
    if critical:
        parts.append(f"{critical} critical issue{'s' if critical != 1 else ''} requiring immediate attention.")
    if warnings:
        parts.append(f"{warnings} warning{'s' if warnings != 1 else ''} to review.")
    if alert_events:
        parts.append(f"{len(alert_events)} alert{'s' if len(alert_events) != 1 else ''} triggered.")

    return " ".join(parts)


def _build_alert_summary(alert_events: list) -> dict:
    """Build alert summary for the briefing."""
    if not alert_events:
        return {"total": 0, "events": []}

    events = []
    for e in alert_events[:10]:
        events.append({
            "rule_name": getattr(e, "context", {}).get("rule_name", "Unknown") if isinstance(getattr(e, "context", None), dict) else "Unknown",
            "trigger_value": getattr(e, "trigger_value", ""),
            "triggered_at": str(getattr(e, "triggered_at", "")),
        })

    return {
        "total": len(alert_events),
        "events": events,
    }
