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
    """Find relationships between two specific tables (no DB queries).

    All methods now require value overlap validation when values are available.
    A column name match alone is NOT sufficient — columns with the same name
    in different tables can contain completely different value domains.
    """
    results: list[DiscoveredRelationship] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    cls_a = (prof_a.column_classification or {}).get("columns", {})
    cls_b = (prof_b.column_classification or {}).get("columns", {})

    # ------ Method A: Name matching + value validation ------
    # Name match is a HINT, not proof. Validate with value overlap when
    # values are available. Without validation, use low confidence.
    for col_a in cls_a:
        for col_b in cls_b:
            pair_key = (prof_a.table_name, col_a, prof_b.table_name, col_b)
            if pair_key in seen_pairs:
                continue
            if not _names_match(col_a, col_b, prof_a.table_name, prof_b.table_name):
                continue

            # Validate name match with actual value overlap
            vals_a = value_cache.get((prof_a.table_name, col_a))
            vals_b = value_cache.get((prof_b.table_name, col_b))

            if vals_a and vals_b:
                overlap = vals_a & vals_b
                overlap_count = len(overlap)
                min_card = min(len(vals_a), len(vals_b))
                overlap_pct = overlap_count / min_card if min_card > 0 else 0

                if overlap_pct >= 0.5 and overlap_count >= 3:
                    # Name match + strong value overlap → high confidence
                    results.append(
                        DiscoveredRelationship(
                            table_a=prof_a.table_name,
                            column_a=col_a,
                            table_b=prof_b.table_name,
                            column_b=col_b,
                            relationship_type="join_key",
                            confidence=round(min(0.95, 0.7 + overlap_pct * 0.25), 3),
                            overlap_count=overlap_count,
                        )
                    )
                    seen_pairs.add(pair_key)
                else:
                    # Name match but values DON'T overlap → reject
                    logger.debug(
                        "Name match %s.%s ↔ %s.%s rejected: overlap=%d/%d (%.0f%%)",
                        prof_a.table_name, col_a, prof_b.table_name, col_b,
                        overlap_count, min_card, overlap_pct * 100,
                    )
            else:
                # No values to validate — mark as unverified with low confidence
                # so it doesn't get used for SQL JOINs but is still visible
                results.append(
                    DiscoveredRelationship(
                        table_a=prof_a.table_name,
                        column_a=col_a,
                        table_b=prof_b.table_name,
                        column_b=col_b,
                        relationship_type="name_match_unverified",
                        confidence=0.4,
                        overlap_count=0,
                    )
                )
                seen_pairs.add(pair_key)

    # ------ Method B: Value overlap via pre-fetched sets ------
    # Require strong overlap: ≥50% AND at least 3 common values.
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

            if confidence >= 0.5 and overlap_count >= 3:
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

    # ------ Method C: Semantic matching — ONLY with value validation ------
    # Two identifier columns with similar cardinality is NOT proof of a
    # relationship. Require actual value overlap to confirm.
    for col_a, info_a in cls_a.items():
        for col_b, info_b in cls_b.items():
            pair_key = (prof_a.table_name, col_a, prof_b.table_name, col_b)
            if pair_key in seen_pairs:
                continue
            if not _semantic_match(info_a, info_b):
                continue

            # Semantic hint found — require value overlap to confirm
            vals_a = value_cache.get((prof_a.table_name, col_a))
            vals_b = value_cache.get((prof_b.table_name, col_b))
            if vals_a and vals_b:
                overlap = vals_a & vals_b
                overlap_count = len(overlap)
                min_card = min(len(vals_a), len(vals_b))
                overlap_pct = overlap_count / min_card if min_card > 0 else 0

                if overlap_pct >= 0.5 and overlap_count >= 3:
                    results.append(
                        DiscoveredRelationship(
                            table_a=prof_a.table_name,
                            column_a=col_a,
                            table_b=prof_b.table_name,
                            column_b=col_b,
                            relationship_type="semantic_match",
                            confidence=round(min(0.85, 0.4 + overlap_pct * 0.45), 3),
                            overlap_count=overlap_count,
                        )
                    )
                    seen_pairs.add(pair_key)
            # No values available → skip entirely (don't create unverified semantic matches)

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
