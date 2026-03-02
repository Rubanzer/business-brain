"""Cross-table join detection using name matching, value overlap, and semantic matching.

Performance: pre-fetches sampled distinct values for all identifier/categorical
columns upfront (1 query per column, capped at 500 values), then compares value
sets in Python.  This turns O(T^2 * C^2) SQL queries into O(T * C) queries.
"""

from __future__ import annotations

import asyncio
import logging
import re
from itertools import combinations

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import DiscoveredRelationship, TableProfile

logger = logging.getLogger(__name__)

# Max distinct values to sample per column (keeps memory & query time bounded)
_SAMPLE_LIMIT = 500
# Max concurrent DB queries when pre-fetching column values
_MAX_CONCURRENT = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def find_relationships(
    session: AsyncSession,
    profiles: list[TableProfile],
) -> list[DiscoveredRelationship]:
    """Detect cross-table relationships from profiled tables.

    Algorithm:
    1. Pre-fetch sampled distinct values for all id/categorical columns (parallel)
    2. For each table pair, try name matching (no SQL)
    3. For unmatched pairs, compare pre-fetched value sets in Python (no SQL)
    4. Fallback: semantic matching via profile metadata (no SQL)
    """
    relationships: list[DiscoveredRelationship] = []

    if len(profiles) < 2:
        return relationships

    # Clear existing relationships for profiled tables
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

    # ---- Phase 1: Pre-fetch distinct values for all id/cat columns ----
    value_cache = await _prefetch_column_values(session, profiles)
    logger.info(
        "Pre-fetched distinct values for %d columns across %d tables",
        len(value_cache),
        len(profiles),
    )

    # ---- Phase 2: Compare all table pairs ----
    for prof_a, prof_b in combinations(profiles, 2):
        try:
            rels = _find_between(prof_a, prof_b, value_cache)
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


# ---------------------------------------------------------------------------
# Pre-fetching: 1 query per column, bounded concurrency
# ---------------------------------------------------------------------------


async def _prefetch_column_values(
    session: AsyncSession,
    profiles: list[TableProfile],
) -> dict[tuple[str, str], set[str]]:
    """Fetch sampled distinct values for all identifier/categorical columns.

    Returns {(table_name, column_name): set_of_string_values}.
    Each query samples up to _SAMPLE_LIMIT distinct non-null values.
    """
    id_cat_types = {"identifier", "categorical"}
    tasks: list[tuple[str, str]] = []  # (table, column) pairs to fetch

    for prof in profiles:
        cls = (prof.column_classification or {}).get("columns", {})
        for col_name, info in cls.items():
            if info.get("semantic_type") in id_cat_types:
                tasks.append((prof.table_name, col_name))

    if not tasks:
        return {}

    # Run queries with bounded concurrency using a semaphore
    sem = asyncio.Semaphore(_MAX_CONCURRENT)
    cache: dict[tuple[str, str], set[str]] = {}

    async def _fetch_one(table: str, col: str) -> None:
        safe_t = re.sub(r"[^a-zA-Z0-9_]", "", table)
        safe_c = re.sub(r"[^a-zA-Z0-9_]", "", col)
        query = (
            f'SELECT DISTINCT "{safe_c}"::text '
            f'FROM "{safe_t}" '
            f'WHERE "{safe_c}" IS NOT NULL '
            f"LIMIT {_SAMPLE_LIMIT}"
        )
        async with sem:
            try:
                result = await session.execute(sql_text(query))
                values = {row[0] for row in result.fetchall()}
                cache[(table, col)] = values
            except Exception:
                logger.debug("Failed to fetch values for %s.%s", table, col)
                cache[(table, col)] = set()

    await asyncio.gather(*[_fetch_one(t, c) for t, c in tasks])
    return cache


# ---------------------------------------------------------------------------
# Per-table-pair comparison (pure Python, no SQL)
# ---------------------------------------------------------------------------


def _find_between(
    prof_a: TableProfile,
    prof_b: TableProfile,
    value_cache: dict[tuple[str, str], set[str]],
) -> list[DiscoveredRelationship]:
    """Find relationships between two specific tables (no DB queries)."""
    results: list[DiscoveredRelationship] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    cls_a = (prof_a.column_classification or {}).get("columns", {})
    cls_b = (prof_b.column_classification or {}).get("columns", {})

    # ------ Method A: Name matching (fast, pure string comparison) ------
    for col_a in cls_a:
        for col_b in cls_b:
            pair_key = (prof_a.table_name, col_a, prof_b.table_name, col_b)
            if pair_key in seen_pairs:
                continue
            if _names_match(col_a, col_b, prof_a.table_name, prof_b.table_name):
                results.append(
                    DiscoveredRelationship(
                        table_a=prof_a.table_name,
                        column_a=col_a,
                        table_b=prof_b.table_name,
                        column_b=col_b,
                        relationship_type="join_key",
                        confidence=0.9,
                        overlap_count=0,
                    )
                )
                seen_pairs.add(pair_key)

    # ------ Method B: Value overlap via pre-fetched sets ------
    id_cat_types = {"identifier", "categorical"}
    id_cols_a = [c for c, i in cls_a.items() if i.get("semantic_type") in id_cat_types]
    id_cols_b = [c for c, i in cls_b.items() if i.get("semantic_type") in id_cat_types]

    for col_a in id_cols_a:
        vals_a = value_cache.get((prof_a.table_name, col_a))
        if not vals_a:
            continue
        for col_b in id_cols_b:
            pair_key = (prof_a.table_name, col_a, prof_b.table_name, col_b)
            if pair_key in seen_pairs:
                continue

            vals_b = value_cache.get((prof_b.table_name, col_b))
            if not vals_b:
                continue

            # Set intersection — instant compared to SQL JOIN
            overlap = vals_a & vals_b
            overlap_count = len(overlap)
            min_card = min(len(vals_a), len(vals_b))
            confidence = round(overlap_count / min_card, 3) if min_card > 0 else 0

            if confidence > 0.3:
                results.append(
                    DiscoveredRelationship(
                        table_a=prof_a.table_name,
                        column_a=col_a,
                        table_b=prof_b.table_name,
                        column_b=col_b,
                        relationship_type="value_overlap",
                        confidence=confidence,
                        overlap_count=overlap_count,
                    )
                )
                seen_pairs.add(pair_key)

    # ------ Method C: Semantic matching (profile metadata, no SQL) ------
    for col_a, info_a in cls_a.items():
        for col_b, info_b in cls_b.items():
            pair_key = (prof_a.table_name, col_a, prof_b.table_name, col_b)
            if pair_key in seen_pairs:
                continue
            if _semantic_match(info_a, info_b):
                results.append(
                    DiscoveredRelationship(
                        table_a=prof_a.table_name,
                        column_a=col_a,
                        table_b=prof_b.table_name,
                        column_b=col_b,
                        relationship_type="semantic_match",
                        confidence=0.5,
                        overlap_count=0,
                    )
                )
                seen_pairs.add(pair_key)

    return results


# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------


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


def _semantic_match(info_a: dict, info_b: dict) -> bool:
    """Check if two columns semantically match (both identifier with similar cardinality)."""
    type_a = info_a.get("semantic_type", "")
    type_b = info_b.get("semantic_type", "")

    if type_a == "identifier" and type_b == "identifier":
        card_a = info_a.get("cardinality", 0)
        card_b = info_b.get("cardinality", 0)
        if card_a and card_b:
            ratio = min(card_a, card_b) / max(card_a, card_b)
            return ratio > 0.5
    return False
