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
    """Call Gemini to generate narrative stories from a group of insights."""
    from google import genai
    from config.settings import settings

    if not settings.gemini_api_key:
        logger.warning("No Gemini API key — skipping narrative generation")
        return []

    # Build insight summaries
    summaries = []
    insight_ids = []
    source_tables: set[str] = set()

    for i, ins in enumerate(insights):
        summaries.append(f"{i+1}. [{ins.insight_type}] {ins.title}: {ins.description}")
        insight_ids.append(ins.id)
        source_tables.update(ins.source_tables or [])

    prompt = (
        "You are a senior business analyst. Given these discovered insights, connect them into "
        "1-2 narrative stories that explain cause-and-effect relationships.\n\n"
        "INSIGHTS:\n" + "\n".join(summaries) + "\n\n"
        "Return ONLY valid JSON (no markdown fences): "
        '[{"title": "...", "narrative": "...", "connected_insight_ids": [0,1,...], "suggested_actions": ["..."]}]'
    )

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        text = response.text.strip()

        # Clean markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        stories = json.loads(text)
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
