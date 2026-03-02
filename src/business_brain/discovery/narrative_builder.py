"""LLM-powered story synthesis connecting related insights."""

from __future__ import annotations

import json
import logging
import uuid

from business_brain.db.discovery_models import Insight

logger = logging.getLogger(__name__)


async def build_narratives(insights: list[Insight]) -> list[Insight]:
    """Use Gemini LLM to connect related insights into narrative stories.

    Only runs when there are 2+ non-story insights.
    Groups insights by shared tables/entities before sending to LLM.
    """
    # Filter out existing stories
    non_stories = [i for i in insights if i.insight_type != "story"]
    if len(non_stories) < 2:
        return []

    # Group by shared tables
    groups = _group_by_tables(non_stories)

    story_insights: list[Insight] = []
    for group in groups:
        if len(group) < 2:
            continue
        try:
            stories = await _generate_stories(group)
            story_insights.extend(stories)
        except Exception:
            logger.exception("Narrative generation failed for insight group")

    return story_insights


def _group_by_tables(insights: list[Insight]) -> list[list[Insight]]:
    """Group insights by shared source tables."""
    # Simple approach: group insights that share at least one table
    groups: list[list[Insight]] = []
    used: set[str] = set()

    for insight in insights:
        if insight.id in used:
            continue

        tables = set(insight.source_tables or [])
        group = [insight]
        used.add(insight.id)

        for other in insights:
            if other.id in used:
                continue
            other_tables = set(other.source_tables or [])
            if tables & other_tables:
                group.append(other)
                used.add(other.id)
                tables |= other_tables

        groups.append(group)

    return groups


async def _generate_stories(insights: list[Insight]) -> list[Insight]:
    """Call LLM to generate narrative stories from a group of insights."""
    from business_brain.analysis.tools.llm_gateway import reason as _llm_reason

    # Build insight summaries
    summaries = []
    insight_ids = []
    source_tables: set[str] = set()

    for i, ins in enumerate(insights):
        summaries.append(f"{i+1}. [{ins.insight_type}] {ins.title}: {ins.description}")
        insight_ids.append(ins.id)
        source_tables.update(ins.source_tables or [])

    prompt = (
        "You are a CFO reviewing data findings. Your job is to synthesize these raw "
        "findings into 1-2 ACTIONABLE business stories with SPECIFIC numbers.\n\n"
        "RAW FINDINGS:\n" + "\n".join(summaries) + "\n\n"
        "RULES — FOLLOW STRICTLY:\n"
        "- Every story MUST contain specific numbers, percentages, or monetary values from the findings\n"
        "- State the BUSINESS IMPACT: how much money is at risk, what efficiency is lost, what cost can be saved\n"
        "- State a CONCRETE ACTION: not 'investigate further' but 'renegotiate with supplier X' or 'audit shift B staffing'\n"
        "- NEVER say 'warrants further investigation', 'suggests', 'may indicate', or 'could be'\n"
        "- NEVER describe what a table contains — only state what the DATA REVEALS\n"
        "- If the findings don't support a meaningful business story, return an EMPTY array []\n\n"
        "Return ONLY valid JSON (no markdown fences): "
        '[{"title": "short action-oriented title with key number", '
        '"narrative": "2-3 sentences with specific numbers connecting cause to effect to business impact", '
        '"connected_insight_ids": [0,1,...], '
        '"suggested_actions": ["specific action with expected outcome"]}]'
    )

    try:
        raw = await _llm_reason(prompt)

        # Clean markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        stories = json.loads(raw)
    except Exception:
        logger.exception("Failed to parse LLM narrative response")
        return []

    results: list[Insight] = []
    for story in stories:
        if not isinstance(story, dict):
            continue

        connected_ids = []
        for idx in story.get("connected_insight_ids", []):
            if isinstance(idx, int) and 0 <= idx < len(insight_ids):
                connected_ids.append(insight_ids[idx])

        results.append(Insight(
            id=str(uuid.uuid4()),
            insight_type="story",
            severity="info",
            impact_score=75,
            title=story.get("title", "Connected Insight"),
            description=story.get("narrative", ""),
            narrative=story.get("narrative", ""),
            source_tables=list(source_tables),
            source_columns=[],
            related_insights=connected_ids,
            suggested_actions=story.get("suggested_actions", []),
        ))

    return results
