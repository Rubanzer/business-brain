"""Cross-table join detection using name matching, value overlap, and semantic matching."""

from __future__ import annotations

import logging
import re
from itertools import combinations

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import DiscoveredRelationship, TableProfile

logger = logging.getLogger(__name__)


async def find_relationships(
    session: AsyncSession,
    profiles: list[TableProfile],
) -> list[DiscoveredRelationship]:
    """Detect cross-table relationships from profiled tables."""
    relationships: list[DiscoveredRelationship] = []

    if len(profiles) < 2:
        return relationships

    # Clear existing relationships for profiled tables so we don't accumulate
    # duplicates across scans.
    table_names = [p.table_name for p in profiles]
    try:
        from sqlalchemy import delete
        await session.execute(
            delete(DiscoveredRelationship).where(
                DiscoveredRelationship.table_a.in_(table_names)
                | DiscoveredRelationship.table_b.in_(table_names)
            )
        )
        await session.flush()
    except Exception:
        logger.debug("Failed to clear old relationships, continuing")

    for prof_a, prof_b in combinations(profiles, 2):
        try:
            # Use savepoint so a single pair failure doesn't rollback everything
            async with session.begin_nested():
                rels = await _find_between(session, prof_a, prof_b)
                relationships.extend(rels)
        except Exception:
            logger.exception(
                "Failed to find relationships between %s and %s",
                prof_a.table_name,
                prof_b.table_name,
            )

    # Bulk insert
    for rel in relationships:
        session.add(rel)
    await session.flush()

    return relationships


async def _find_between(
    session: AsyncSession,
    prof_a: TableProfile,
    prof_b: TableProfile,
) -> list[DiscoveredRelationship]:
    """Find relationships between two specific tables."""
    results: list[DiscoveredRelationship] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    cls_a = prof_a.column_classification or {}
    cls_b = prof_b.column_classification or {}
    cols_a = cls_a.get("columns", {})
    cols_b = cls_b.get("columns", {})

    # Method A: Name matching
    for col_a, info_a in cols_a.items():
        for col_b, info_b in cols_b.items():
            pair_key = (prof_a.table_name, col_a, prof_b.table_name, col_b)
            if pair_key in seen_pairs:
                continue

            if _names_match(col_a, col_b, prof_a.table_name, prof_b.table_name):
                results.append(DiscoveredRelationship(
                    table_a=prof_a.table_name,
                    column_a=col_a,
                    table_b=prof_b.table_name,
                    column_b=col_b,
                    relationship_type="join_key",
                    confidence=0.9,
                    overlap_count=0,
                ))
                seen_pairs.add(pair_key)

    # Method B: Value overlap for identifier/categorical columns
    id_cat_types = {"identifier", "categorical"}
    id_cols_a = [c for c, i in cols_a.items() if i.get("semantic_type") in id_cat_types]
    id_cols_b = [c for c, i in cols_b.items() if i.get("semantic_type") in id_cat_types]

    for col_a in id_cols_a:
        for col_b in id_cols_b:
            pair_key = (prof_a.table_name, col_a, prof_b.table_name, col_b)
            if pair_key in seen_pairs:
                continue

            try:
                async with session.begin_nested():
                    overlap = await _check_value_overlap(
                        session, prof_a.table_name, col_a, prof_b.table_name, col_b
                    )
                if overlap["confidence"] > 0.3:
                    results.append(DiscoveredRelationship(
                        table_a=prof_a.table_name,
                        column_a=col_a,
                        table_b=prof_b.table_name,
                        column_b=col_b,
                        relationship_type="value_overlap",
                        confidence=overlap["confidence"],
                        overlap_count=overlap["count"],
                    ))
                    seen_pairs.add(pair_key)
            except Exception:
                logger.debug(
                    "Value overlap check failed for %s.%s <-> %s.%s",
                    prof_a.table_name, col_a, prof_b.table_name, col_b,
                )

    # Method C: Semantic matching — both identifier with similar cardinality
    for col_a, info_a in cols_a.items():
        for col_b, info_b in cols_b.items():
            pair_key = (prof_a.table_name, col_a, prof_b.table_name, col_b)
            if pair_key in seen_pairs:
                continue

            if _semantic_match(info_a, info_b):
                results.append(DiscoveredRelationship(
                    table_a=prof_a.table_name,
                    column_a=col_a,
                    table_b=prof_b.table_name,
                    column_b=col_b,
                    relationship_type="semantic_match",
                    confidence=0.5,
                    overlap_count=0,
                ))
                seen_pairs.add(pair_key)

    return results


def _names_match(col_a: str, col_b: str, table_a: str, table_b: str) -> bool:
    """Check if column names suggest a join relationship."""
    a_lower = col_a.lower()
    b_lower = col_b.lower()

    # Exact match
    if a_lower == b_lower:
        return True

    # customer_id <-> id where table is "customers"
    table_a_singular = re.sub(r"s$", "", table_a.lower())
    table_b_singular = re.sub(r"s$", "", table_b.lower())

    if a_lower == f"{table_b_singular}_id" and b_lower == "id":
        return True
    if b_lower == f"{table_a_singular}_id" and a_lower == "id":
        return True

    # Strip common prefixes: fk_, pk_
    stripped_a = re.sub(r"^(fk_|pk_)", "", a_lower)
    stripped_b = re.sub(r"^(fk_|pk_)", "", b_lower)
    if stripped_a == stripped_b and stripped_a != a_lower:
        return True

    return False


async def _check_value_overlap(
    session: AsyncSession,
    table_a: str,
    col_a: str,
    table_b: str,
    col_b: str,
) -> dict:
    """Check value overlap between two columns across tables."""
    safe_ta = re.sub(r"[^a-zA-Z0-9_]", "", table_a)
    safe_ca = re.sub(r"[^a-zA-Z0-9_]", "", col_a)
    safe_tb = re.sub(r"[^a-zA-Z0-9_]", "", table_b)
    safe_cb = re.sub(r"[^a-zA-Z0-9_]", "", col_b)

    query = f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT "{safe_ca}" FROM "{safe_ta}" WHERE "{safe_ca}" IS NOT NULL
        ) a
        JOIN (
            SELECT DISTINCT "{safe_cb}" FROM "{safe_tb}" WHERE "{safe_cb}" IS NOT NULL
        ) b ON a."{safe_ca}"::text = b."{safe_cb}"::text
    """
    overlap_res = await session.execute(sql_text(query))
    overlap_count = overlap_res.scalar() or 0

    # Get cardinalities
    card_a_res = await session.execute(
        sql_text(f'SELECT COUNT(DISTINCT "{safe_ca}") FROM "{safe_ta}" WHERE "{safe_ca}" IS NOT NULL')
    )
    card_a = card_a_res.scalar() or 1

    card_b_res = await session.execute(
        sql_text(f'SELECT COUNT(DISTINCT "{safe_cb}") FROM "{safe_tb}" WHERE "{safe_cb}" IS NOT NULL')
    )
    card_b = card_b_res.scalar() or 1

    confidence = overlap_count / min(card_a, card_b) if min(card_a, card_b) > 0 else 0

    return {"count": overlap_count, "confidence": round(confidence, 3)}


def _semantic_match(info_a: dict, info_b: dict) -> bool:
    """Check if two columns semantically match (both identifier with similar cardinality)."""
    type_a = info_a.get("semantic_type", "")
    type_b = info_b.get("semantic_type", "")

    # Both identifiers with similar cardinality
    if type_a == "identifier" and type_b == "identifier":
        card_a = info_a.get("cardinality", 0)
        card_b = info_b.get("cardinality", 0)
        if card_a and card_b:
            ratio = min(card_a, card_b) / max(card_a, card_b)
            return ratio > 0.5
    return False
