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


async def profile_all_tables(
    session: AsyncSession,
    table_filter: list[str] | None = None,
) -> list[TableProfile]:
    """Profile tables that have metadata, returning updated TableProfile objects.

    If table_filter is provided, only profile the specified tables.
    If table_filter is None, profile all tables (backward compatible).
    """
    entries = await metadata_store.get_filtered(session, table_filter)
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

    # Auto-generate column descriptions and persist to metadata_store
    # IMPORTANT: Preserve existing Gemini-generated descriptions — only fill in
    # blanks using pattern matching (auto_describe_column).
    try:
        from business_brain.discovery.data_dictionary import auto_describe_column, infer_column_type

        existing_meta = await metadata_store.get_by_table(session, safe_table)
        # Build a lookup of existing descriptions (from Gemini or previous enrichment)
        existing_descs: dict[str, str] = {}
        if existing_meta and existing_meta.columns_metadata:
            for c in existing_meta.columns_metadata:
                name = c.get("name", "")
                desc = c.get("description", "")
                if name and desc:
                    existing_descs[name] = desc

        updated_columns = []
        changed = False
        for col_name in columns:
            col_values = [row.get(col_name) for row in sample_rows]
            col_type = infer_column_type(col_values)
            non_null = [v for v in col_values if v is not None]
            original_type = col_types.get(col_name, col_type)

            # Prefer existing description (from Gemini), only generate if missing
            existing_desc = existing_descs.get(col_name, "")
            if existing_desc:
                desc = existing_desc
            else:
                desc = auto_describe_column(col_name, col_type, {
                    "unique_pct": len(set(non_null)) / max(len(non_null), 1) * 100,
                    "null_pct": (len(col_values) - len(non_null)) / max(len(col_values), 1) * 100,
                })
                changed = True

            updated_columns.append({
                "name": col_name,
                "type": original_type,
                "description": desc,
            })

        # Update metadata with enriched column descriptions
        if existing_meta and (changed or not existing_meta.columns_metadata):
            existing_meta.columns_metadata = updated_columns
            await session.flush()
            logger.info("Updated column descriptions for table: %s (%d columns)", safe_table, len(updated_columns))
    except Exception:
        logger.debug("Column description generation failed for %s, non-critical", safe_table)

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

        # Shift-specific suggestions (shift column + numeric)
        shift_cols = [c for c in cat_cols if "shift" in c.lower()]
        if shift_cols and num_cols:
            suggestions.append(f"Compare {num_cols[0]} across {shift_cols[0]}s")
            if len(num_cols) >= 2:
                suggestions.append(f"Which {shift_cols[0]} has the best {num_cols[0]} to {num_cols[1]} ratio?")

        # domain-specific
        domain = profile.domain_hint or "general"
        if domain == "manufacturing":
            if shift_cols:
                suggestions.append(f"What is the shift-wise production output by {shift_cols[0]}?")
            else:
                suggestions.append("What is the shift-wise production output?")
            if any("power" in c.lower() or "kva" in c.lower() for c in num_cols):
                suggestions.append("Show power consumption trend and anomalies")
            if any("heat" in c.lower() for c in cols):
                suggestions.append("What is the average heat cycle time?")
        elif domain == "quality":
            suggestions.append("What is the rejection rate by grade?")
        elif domain == "logistics":
            suggestions.append("How many trucks arrived today vs yesterday?")
        elif domain == "energy":
            suggestions.append("What is the unit power consumption trend?")
        elif domain == "procurement":
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


def compute_data_quality_score(profile: TableProfile) -> dict:
    """Compute a data quality score (0-100) for a profiled table.

    Returns dict with:
        score: int (0-100)
        breakdown: dict of individual metric scores
        issues: list of quality issue descriptions
    """
    cls = profile.column_classification
    if not cls or "columns" not in cls:
        return {"score": 0, "breakdown": {}, "issues": ["No column classification available"]}

    cols = cls["columns"]
    row_count = profile.row_count or 0

    if not cols or row_count == 0:
        return {"score": 0, "breakdown": {}, "issues": ["Empty table"]}

    issues: list[str] = []
    n_cols = len(cols)

    # 1. Completeness score (based on null rates) — 30% weight
    null_penalties = []
    for col_name, info in cols.items():
        null_count = info.get("null_count", 0)
        if row_count > 0:
            null_pct = null_count / row_count
            null_penalties.append(null_pct)
            if null_pct > 0.20:
                issues.append(f"{col_name} has {null_pct*100:.0f}% missing values")

    avg_null_rate = sum(null_penalties) / len(null_penalties) if null_penalties else 0
    completeness_score = max(0, 100 - int(avg_null_rate * 200))  # 50% nulls → 0 score

    # 2. Uniqueness score (no constant columns) — 20% weight
    constant_cols = sum(1 for info in cols.values() if info.get("cardinality", 0) == 1)
    if constant_cols and row_count > 1:
        uniqueness_score = max(0, 100 - int(constant_cols / n_cols * 100))
        if constant_cols > 0:
            issues.append(f"{constant_cols} constant column(s) provide no analytical value")
    else:
        uniqueness_score = 100

    # 3. Validity score (no impossible values) — 30% weight
    validity_deductions = 0
    for col_name, info in cols.items():
        sem_type = info.get("semantic_type", "")
        stats = info.get("stats")
        if not stats:
            continue

        if sem_type == "numeric_currency" and stats.get("min", 0) < 0:
            validity_deductions += 1
            issues.append(f"{col_name}: negative values in currency column")
        if sem_type == "numeric_percentage":
            if stats.get("max", 0) > 100 or stats.get("min", 0) < 0:
                validity_deductions += 1
                issues.append(f"{col_name}: percentage values outside 0-100")

    validity_score = max(0, 100 - validity_deductions * 30)

    # 4. Diversity score (mix of column types) — 20% weight
    types_present = {info.get("semantic_type") for info in cols.values()}
    diversity_score = min(100, len(types_present) * 25)

    # Weighted total
    total = int(
        completeness_score * 0.30
        + uniqueness_score * 0.20
        + validity_score * 0.30
        + diversity_score * 0.20
    )

    return {
        "score": min(total, 100),
        "breakdown": {
            "completeness": completeness_score,
            "uniqueness": uniqueness_score,
            "validity": validity_score,
            "diversity": diversity_score,
        },
        "issues": issues,
    }
