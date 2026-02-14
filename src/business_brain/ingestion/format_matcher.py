"""Format fingerprinting and recurring file detection."""

from __future__ import annotations

import hashlib
import logging
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.v3_models import FormatFingerprint

logger = logging.getLogger(__name__)


def compute_fingerprint(columns: list[str]) -> str:
    """Create a fingerprint hash from a list of column names.

    Normalizes column names (lowercase, strip whitespace, remove special chars)
    then sorts and hashes them.
    """
    normalized = sorted(_normalize_col(c) for c in columns)
    signature = "|".join(normalized)
    return hashlib.sha256(signature.encode()).hexdigest()[:16]


def _normalize_col(name: str) -> str:
    """Normalize a column name for comparison."""
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())


def fuzzy_match_columns(cols_a: list[str], cols_b: list[str], threshold: float = 0.7) -> dict[str, str]:
    """Fuzzy-match column names between two schemas.

    Returns:
        Dict mapping cols_a names to their best match in cols_b.
    """
    mapping: dict[str, str] = {}
    used_b: set[str] = set()

    # First pass: exact normalized matches
    norm_b = {_normalize_col(c): c for c in cols_b}
    for col_a in cols_a:
        norm_a = _normalize_col(col_a)
        if norm_a in norm_b and norm_b[norm_a] not in used_b:
            mapping[col_a] = norm_b[norm_a]
            used_b.add(norm_b[norm_a])

    # Second pass: fuzzy matches for unmatched columns
    remaining_a = [c for c in cols_a if c not in mapping]
    remaining_b = [c for c in cols_b if c not in used_b]

    for col_a in remaining_a:
        best_match = None
        best_score = 0.0
        norm_a = _normalize_col(col_a)

        for col_b in remaining_b:
            if col_b in used_b:
                continue
            norm_b_val = _normalize_col(col_b)
            score = SequenceMatcher(None, norm_a, norm_b_val).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = col_b

        if best_match:
            mapping[col_a] = best_match
            used_b.add(best_match)

    return mapping


async def find_matching_fingerprint(
    session: AsyncSession,
    columns: list[str],
) -> FormatFingerprint | None:
    """Find a known fingerprint that matches the given columns.

    First tries exact hash match, then falls back to fuzzy matching
    against stored variations.
    """
    fp_hash = compute_fingerprint(columns)

    # Exact match
    result = await session.execute(
        select(FormatFingerprint).where(FormatFingerprint.fingerprint_hash == fp_hash)
    )
    exact = result.scalar_one_or_none()
    if exact:
        return exact

    # Fuzzy match against all known fingerprints
    all_fps = await session.execute(select(FormatFingerprint))
    fingerprints = list(all_fps.scalars().all())

    for fp in fingerprints:
        variations = fp.source_variations or []
        for variation in variations:
            mapping = fuzzy_match_columns(columns, variation)
            # If >80% of columns match, consider it a match
            if len(mapping) >= len(columns) * 0.8:
                return fp

    return None


async def register_fingerprint(
    session: AsyncSession,
    columns: list[str],
    table_name: str,
    column_mapping: dict[str, str] | None = None,
) -> FormatFingerprint:
    """Register a new format fingerprint or update an existing one."""
    fp_hash = compute_fingerprint(columns)

    result = await session.execute(
        select(FormatFingerprint).where(FormatFingerprint.fingerprint_hash == fp_hash)
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.match_count += 1
        variations = existing.source_variations or []
        if columns not in variations:
            variations.append(columns)
            existing.source_variations = variations
        await session.commit()
        return existing

    fp = FormatFingerprint(
        fingerprint_hash=fp_hash,
        table_name=table_name,
        column_mapping=column_mapping or {c: c for c in columns},
        source_variations=[columns],
        match_count=1,
    )
    session.add(fp)
    await session.commit()
    await session.refresh(fp)
    return fp
