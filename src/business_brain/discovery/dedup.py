"""Duplicate insight detection — prevents re-inserting semantically equivalent insights.

Uses a hash of (insight_type, source_tables, source_columns, composite_template)
to identify duplicates across discovery runs.
"""

from __future__ import annotations

import hashlib
import json

from business_brain.db.discovery_models import Insight


def compute_insight_key(insight: Insight) -> str:
    """Compute a deterministic key for deduplication.

    Two insights are considered duplicates if they have the same type,
    source tables, source columns, and composite template.
    """
    tables = sorted(insight.source_tables or [])
    columns = sorted(insight.source_columns or [])
    template = insight.composite_template or ""

    payload = json.dumps({
        "type": insight.insight_type,
        "tables": tables,
        "columns": columns,
        "template": template,
    }, sort_keys=True)

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def deduplicate_insights(
    new_insights: list[Insight],
    existing_keys: set[str],
) -> list[Insight]:
    """Filter out insights whose keys already exist.

    Args:
        new_insights: Insights from the current discovery run.
        existing_keys: Set of insight keys already in the DB.

    Returns:
        List of new_insights that are not duplicates.
    """
    unique: list[Insight] = []
    seen: set[str] = set(existing_keys)

    for insight in new_insights:
        key = compute_insight_key(insight)
        if key not in seen:
            unique.append(insight)
            seen.add(key)

    return unique
