"""Per-table column classification + stats profiling."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.cognitive.column_classifier import classify_columns
from business_brain.db.discovery_models import TableProfile
from business_brain.memory import metadata_store

logger = logging.getLogger(__name__)


async def profile_all_tables(session: AsyncSession) -> list[TableProfile]:
    """Profile every table that has metadata, returning updated TableProfile objects."""
    entries = await metadata_store.get_all(session)
    profiles: list[TableProfile] = []

    for entry in entries:
        try:
            profile = await _profile_table(session, entry.table_name, entry.columns_metadata)
            if profile:
                profiles.append(profile)
        except Exception:
            logger.exception("Failed to profile table %s", entry.table_name)
            await session.rollback()

    return profiles


async def _profile_table(
    session: AsyncSession,
    table_name: str,
    columns_metadata: list[dict] | None,
) -> TableProfile | None:
    """Profile a single table: sample rows, classify columns, hash data."""
    import re

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table_name)
    if not safe_table:
        return None

    # Get row count
    count_res = await session.execute(sql_text(f'SELECT COUNT(*) FROM "{safe_table}"'))
    row_count = count_res.scalar() or 0

    # Sample first 100 rows
    sample_res = await session.execute(sql_text(f'SELECT * FROM "{safe_table}" LIMIT 100'))
    sample_rows = [dict(row._mapping) for row in sample_res.fetchall()]

    if not sample_rows:
        return None

    columns = list(sample_rows[0].keys())

    # Build col_types from metadata
    col_types: dict[str, str] = {}
    if columns_metadata:
        for cm in columns_metadata:
            col_types[cm["name"]] = cm.get("type", "")

    # Classify columns
    classification = classify_columns(columns, sample_rows, col_types)

    # Hash first 100 rows for change detection
    data_str = json.dumps(sample_rows, default=str, sort_keys=True)
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

    # Upsert TableProfile
    from sqlalchemy import select

    existing = (
        await session.execute(
            select(TableProfile).where(TableProfile.table_name == safe_table)
        )
    ).scalar_one_or_none()

    if existing:
        existing.row_count = row_count
        existing.column_classification = classification
        existing.domain_hint = classification.get("domain_hint", "general")
        existing.profiled_at = datetime.now(timezone.utc)
        existing.data_hash = data_hash
        profile = existing
    else:
        profile = TableProfile(
            table_name=safe_table,
            row_count=row_count,
            column_classification=classification,
            domain_hint=classification.get("domain_hint", "general"),
            data_hash=data_hash,
        )
        session.add(profile)

    await session.flush()
    return profile


def generate_suggestions(profiles: list[TableProfile]) -> list[str]:
    """Generate smart question suggestions from profiled tables."""
    suggestions: list[str] = []

    for profile in profiles:
        cls = profile.column_classification
        if not cls or "columns" not in cls:
            continue

        cols = cls["columns"]
        table = profile.table_name

        cat_cols = [c for c, info in cols.items() if info.get("semantic_type") == "categorical"]
        num_cols = [
            c for c, info in cols.items()
            if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
        ]
        temp_cols = [c for c, info in cols.items() if info.get("semantic_type") == "temporal"]
        cur_cols = [c for c, info in cols.items() if info.get("semantic_type") == "numeric_currency"]

        # categorical + numeric -> comparison
        if cat_cols and num_cols:
            suggestions.append(f"Compare {num_cols[0]} across {cat_cols[0]}s")

        # temporal + numeric -> trend
        if temp_cols and num_cols:
            suggestions.append(f"Show {num_cols[0]} trend over {temp_cols[0]}")

        # 2+ numeric -> correlation
        if len(num_cols) >= 2:
            suggestions.append(f"Correlation between {num_cols[0]} and {num_cols[1]}?")

        # domain-specific
        domain = profile.domain_hint or "general"
        if domain == "procurement":
            suggestions.append("Which supplier offers the best value?")
        elif domain == "sales":
            suggestions.append("What are the top-performing products by revenue?")
        elif domain == "hr":
            suggestions.append("What is the salary distribution by department?")
        elif domain == "finance":
            suggestions.append("What are the biggest expense categories?")

        # currency -> total
        if cur_cols:
            suggestions.append(f"What is the total {cur_cols[0]} by category?")

    # Deduplicate while preserving order, limit to 5
    seen: set[str] = set()
    unique: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= 5:
            break

    return unique
