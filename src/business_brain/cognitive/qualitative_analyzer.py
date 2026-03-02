"""Qualitative document analysis via LLM.

Performs: summary, sentiment analysis, theme extraction, key findings,
entity extraction, and cross-referencing with quantitative data.
Reuses the existing llm_gateway (rate limiting, caching, retries).
"""

from __future__ import annotations

import logging
from typing import Any

from business_brain.analysis.tools.llm_gateway import (
    extract,
    generate_narrative,
)

logger = logging.getLogger(__name__)

# Maximum text length to send to LLM (avoid token limits)
_MAX_TEXT_LEN = 30_000


def _truncate(text: str, max_len: int = _MAX_TEXT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n\n[... truncated, {len(text) - max_len} chars omitted ...]"


class QualitativeAnalyzer:
    """Full qualitative analysis pipeline for text documents."""

    async def analyze(self, text: str, file_name: str) -> dict[str, Any]:
        """Run the complete analysis pipeline.

        Returns dict with: summary, sentiment, themes, key_findings, entities.
        Each step is independent and fault-tolerant (one failure doesn't block others).
        """
        truncated = _truncate(text)
        results: dict[str, Any] = {}

        # 1. Summary
        try:
            results["summary"] = await self._summarize(truncated, file_name)
        except Exception:
            logger.exception("Qualitative summary failed for %s", file_name)
            results["summary"] = None

        # 2. Sentiment analysis
        try:
            results["sentiment"] = await self._sentiment(truncated)
        except Exception:
            logger.exception("Qualitative sentiment failed for %s", file_name)
            results["sentiment"] = None

        # 3. Theme extraction
        try:
            results["themes"] = await self._themes(truncated)
        except Exception:
            logger.exception("Qualitative themes failed for %s", file_name)
            results["themes"] = None

        # 4. Key findings
        try:
            results["key_findings"] = await self._findings(truncated)
        except Exception:
            logger.exception("Qualitative findings failed for %s", file_name)
            results["key_findings"] = None

        # 5. Entity extraction
        try:
            results["entities"] = await self._entities(truncated)
        except Exception:
            logger.exception("Qualitative entities failed for %s", file_name)
            results["entities"] = None

        return results

    async def _summarize(self, text: str, file_name: str) -> str:
        prompt = f"""Summarize the following document in 3-5 sentences. Focus on the key business implications, decisions, or findings.

Document: {file_name}

---
{text}
---

Provide a concise, business-focused summary:"""
        return await generate_narrative(prompt)

    async def _sentiment(self, text: str) -> dict:
        prompt = f"""Analyze the sentiment of the following document. Provide:
1. Overall sentiment: positive, negative, neutral, or mixed
2. Overall sentiment score: -1.0 (very negative) to +1.0 (very positive)
3. Breakdown by section/topic if the document covers multiple topics

Return JSON:
{{
  "overall": "positive|negative|neutral|mixed",
  "score": 0.0,
  "breakdown": [
    {{"section": "topic/section name", "sentiment": "positive|negative|neutral", "score": 0.0, "reason": "brief explanation"}}
  ]
}}

Document:
---
{text}
---"""
        return await extract(prompt)

    async def _themes(self, text: str) -> list[dict]:
        prompt = f"""Extract the main themes/topics from this document. For each theme:
1. Name the theme concisely
2. Count how many times it appears or is referenced
3. Provide 1-3 direct quotes as evidence

Return JSON array:
[
  {{
    "theme": "theme name",
    "count": 5,
    "evidence": ["direct quote 1", "direct quote 2"]
  }}
]

Limit to the top 8 most important themes. Order by importance.

Document:
---
{text}
---"""
        result = await extract(prompt)
        # extract() returns a dict; themes should be a list
        if isinstance(result, dict):
            return result.get("themes", result.get("data", []))
        if isinstance(result, list):
            return result
        return []

    async def _findings(self, text: str) -> list[dict]:
        prompt = f"""Extract the key business findings, conclusions, and action items from this document.

For each finding:
1. State the finding clearly
2. Assign severity: critical, warning, or info
3. Provide supporting evidence (quote or data point)

Return JSON array:
[
  {{
    "finding": "clear statement of the finding",
    "severity": "critical|warning|info",
    "evidence": "supporting quote or data"
  }}
]

Limit to the top 10 most important findings. Order by severity (critical first).

Document:
---
{text}
---"""
        result = await extract(prompt)
        if isinstance(result, dict):
            return result.get("findings", result.get("key_findings", result.get("data", [])))
        if isinstance(result, list):
            return result
        return []

    async def _entities(self, text: str) -> list[dict]:
        prompt = f"""Extract named entities from this document that are relevant to business analysis.

Categories: person, organization, product, process, metric, location, date, system

Return JSON array:
[
  {{
    "name": "entity name",
    "type": "person|organization|product|process|metric|location|date|system",
    "mentions": 3
  }}
]

Limit to the top 20 most frequently mentioned entities.

Document:
---
{text}
---"""
        result = await extract(prompt)
        if isinstance(result, dict):
            return result.get("entities", result.get("data", []))
        if isinstance(result, list):
            return result
        return []


async def find_linked_tables(
    entities: list[dict],
    table_names: list[str],
) -> list[str]:
    """Match extracted entities against existing table names.

    Simple heuristic: if an entity name (lowered) appears in or matches
    a table name, they're linked.
    """
    linked = set()
    entity_names = {e.get("name", "").lower() for e in (entities or [])}

    for table in table_names:
        tl = table.lower()
        for ename in entity_names:
            if not ename:
                continue
            # Match: entity appears in table name, or table name appears in entity
            if ename in tl or tl in ename:
                linked.add(table)
                break
            # Match: entity words overlap with table name parts
            entity_words = set(ename.split())
            table_words = set(tl.replace("_", " ").split())
            if entity_words & table_words:
                linked.add(table)
                break

    return sorted(linked)
