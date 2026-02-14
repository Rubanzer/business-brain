"""Same-data-different-format detection and semantic column matching.

Detects when two data sources contain the same data in different formats,
e.g., SCADA export vs manual Google Sheet vs ERP export.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import TableProfile
from business_brain.db.v3_models import SourceMapping

logger = logging.getLogger(__name__)


async def detect_duplicate_sources(
    session: AsyncSession,
    profiles: list[TableProfile] | None = None,
) -> list[dict]:
    """Scan all profiled tables for potential duplicate data sources.

    Returns:
        List of potential duplicate pairs with similarity scores and column mappings.
    """
    if profiles is None:
        result = await session.execute(select(TableProfile))
        profiles = list(result.scalars().all())

    if len(profiles) < 2:
        return []

    # Load existing confirmed mappings to skip
    existing_result = await session.execute(select(SourceMapping))
    existing = {
        (m.source_a_table, m.source_b_table)
        for m in existing_result.scalars().all()
    }

    duplicates: list[dict] = []

    for i, profile_a in enumerate(profiles):
        for j, profile_b in enumerate(profiles):
            if j <= i:
                continue

            pair = (profile_a.table_name, profile_b.table_name)
            reverse_pair = (profile_b.table_name, profile_a.table_name)
            if pair in existing or reverse_pair in existing:
                continue

            score, mapping = _compare_profiles(profile_a, profile_b)
            if score >= 0.6:
                duplicates.append({
                    "table_a": profile_a.table_name,
                    "table_b": profile_b.table_name,
                    "similarity_score": round(score, 2),
                    "column_mapping": mapping,
                    "domain_a": profile_a.domain_hint,
                    "domain_b": profile_b.domain_hint,
                })

    return sorted(duplicates, key=lambda d: d["similarity_score"], reverse=True)


def _compare_profiles(
    profile_a: TableProfile,
    profile_b: TableProfile,
) -> tuple[float, list[dict]]:
    """Compare two table profiles for semantic similarity.

    Returns:
        (similarity_score 0-1, column_mapping list)
    """
    cls_a = profile_a.column_classification
    cls_b = profile_b.column_classification

    if not cls_a or not cls_b:
        return 0.0, []
    if "columns" not in cls_a or "columns" not in cls_b:
        return 0.0, []

    cols_a = cls_a["columns"]
    cols_b = cls_b["columns"]

    if not cols_a or not cols_b:
        return 0.0, []

    # Three scoring components
    name_score, name_mapping = _match_by_name(cols_a, cols_b)
    type_score = _match_by_type(cols_a, cols_b)
    domain_score = 1.0 if profile_a.domain_hint == profile_b.domain_hint and profile_a.domain_hint != "general" else 0.0

    # Weighted score
    total_score = name_score * 0.5 + type_score * 0.3 + domain_score * 0.2

    return total_score, name_mapping


def _match_by_name(
    cols_a: dict,
    cols_b: dict,
) -> tuple[float, list[dict]]:
    """Match columns by name similarity (exact + fuzzy).

    Returns:
        (score 0-1, list of column mappings)
    """
    mapping: list[dict] = []
    used_b: set[str] = set()

    names_a = list(cols_a.keys())
    names_b = list(cols_b.keys())

    for col_a in names_a:
        norm_a = _normalize(col_a)
        best_match = None
        best_score = 0.0

        for col_b in names_b:
            if col_b in used_b:
                continue
            norm_b = _normalize(col_b)

            # Exact normalized match
            if norm_a == norm_b:
                best_match = col_b
                best_score = 1.0
                break

            # Fuzzy match
            score = SequenceMatcher(None, norm_a, norm_b).ratio()
            if score > best_score and score >= 0.6:
                best_score = score
                best_match = col_b

        if best_match:
            canonical = _normalize(col_a)
            mapping.append({
                "a": col_a,
                "b": best_match,
                "canonical": canonical,
                "similarity": round(best_score, 2),
            })
            used_b.add(best_match)

    max_cols = max(len(names_a), len(names_b))
    score = len(mapping) / max_cols if max_cols > 0 else 0.0
    return score, mapping


def _match_by_type(cols_a: dict, cols_b: dict) -> float:
    """Score how similar the semantic type distributions are between two tables."""
    types_a = _type_distribution(cols_a)
    types_b = _type_distribution(cols_b)

    all_types = set(types_a.keys()) | set(types_b.keys())
    if not all_types:
        return 0.0

    similarity = 0.0
    for t in all_types:
        a = types_a.get(t, 0)
        b = types_b.get(t, 0)
        similarity += min(a, b) / max(a, b) if max(a, b) > 0 else 0

    return similarity / len(all_types)


def _type_distribution(cols: dict) -> dict[str, int]:
    """Count semantic types in a column dict."""
    dist: dict[str, int] = {}
    for info in cols.values():
        sem_type = info.get("semantic_type", "unknown")
        dist[sem_type] = dist.get(sem_type, 0) + 1
    return dist


def _normalize(name: str) -> str:
    """Normalize a column name for comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())


# ---------------------------------------------------------------------------
# Value Overlap Check
# ---------------------------------------------------------------------------


async def check_value_overlap(
    session: AsyncSession,
    table_a: str,
    column_a: str,
    table_b: str,
    column_b: str,
) -> float:
    """Check what fraction of values in column_a also appear in column_b.

    Returns overlap ratio (0-1).
    """
    safe_ta = re.sub(r"[^a-zA-Z0-9_]", "", table_a)
    safe_tb = re.sub(r"[^a-zA-Z0-9_]", "", table_b)
    safe_ca = re.sub(r"[^a-zA-Z0-9_]", "", column_a)
    safe_cb = re.sub(r"[^a-zA-Z0-9_]", "", column_b)

    try:
        query = (
            f'SELECT COUNT(*) FROM '
            f'(SELECT DISTINCT "{safe_ca}" FROM "{safe_ta}") a '
            f'JOIN (SELECT DISTINCT "{safe_cb}" FROM "{safe_tb}") b '
            f'ON a."{safe_ca}"::text = b."{safe_cb}"::text'
        )
        overlap_result = await session.execute(sql_text(query))
        overlap_count = overlap_result.scalar() or 0

        # Get min cardinality
        card_a_result = await session.execute(
            sql_text(f'SELECT COUNT(DISTINCT "{safe_ca}") FROM "{safe_ta}"')
        )
        card_a = card_a_result.scalar() or 0

        card_b_result = await session.execute(
            sql_text(f'SELECT COUNT(DISTINCT "{safe_cb}") FROM "{safe_tb}"')
        )
        card_b = card_b_result.scalar() or 0

        min_card = min(card_a, card_b)
        if min_card == 0:
            return 0.0

        return overlap_count / min_card

    except Exception:
        logger.exception("Value overlap check failed for %s.%s vs %s.%s", table_a, column_a, table_b, column_b)
        return 0.0


# ---------------------------------------------------------------------------
# Source Mapping CRUD
# ---------------------------------------------------------------------------


async def confirm_source_mapping(
    session: AsyncSession,
    table_a: str,
    table_b: str,
    column_mappings: list[dict],
    entity_type: str = "",
    authoritative_source: str = "",
) -> SourceMapping:
    """Confirm a detected source mapping (user-verified)."""
    mapping = SourceMapping(
        source_a_table=table_a,
        source_b_table=table_b,
        column_mappings=column_mappings,
        entity_type=entity_type,
        authoritative_source=authoritative_source,
        confirmed_by_user=True,
    )
    session.add(mapping)
    await session.commit()
    await session.refresh(mapping)

    logger.info("Confirmed source mapping: %s ↔ %s (%s)", table_a, table_b, entity_type)
    return mapping


async def get_source_mappings(session: AsyncSession) -> list[SourceMapping]:
    """Get all source mappings."""
    result = await session.execute(select(SourceMapping).order_by(SourceMapping.created_at.desc()))
    return list(result.scalars().all())
