"""Operation Enumerator — tiered candidate generation.

Tier 0+1: EXHAUSTIVE — all valid single-column and pairwise combos.
Tiers 2-4: BUDGETED — scored and capped by tier budget.

Supports:
- N-ary analysis via target/segmenters/controls (Gap #1)
- Cross-table candidates via join_spec (Gap #2)
- Canonical deduplication (Gap #6)
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field

from business_brain.analysis.track1.fingerprinter import ColumnFingerprint, TableFingerprint

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnalysisCandidate:
    operation: str  # DESCRIBE/DESCRIBE_CATEGORICAL/COMPARE/CORRELATE/RANK/DETECT_ANOMALY/FORECAST/ATTRIBUTE
    table_name: str
    target: list[str]  # measure(s) being analyzed
    segmenters: list[str] = field(default_factory=list)  # GROUP BY dimensions
    controls: list[str] = field(default_factory=list)  # WHERE conditions
    join_spec: dict | None = None  # {table, local_col, remote_col}
    tier: int = 0
    priority_score: float = 0.0  # for budgeted tiers (2-4) only
    dedup_key: str = ""


@dataclass
class EnumerationBudget:
    # Tiers 0+1 are EXHAUSTIVE — no limit, all valid combos always run.
    # Only Tiers 2-4 have budget caps.
    budgeted_tier_limits: dict[int, int] = field(default_factory=lambda: {
        2: 100,   # +1 segmenter
        3: 50,    # cross-table
        4: 50,    # multi-segment
    })


# ---------------------------------------------------------------------------
# Quality filters (the ONLY pruning for Tier 0+1)
# ---------------------------------------------------------------------------

_MIN_ROWS = 10
_MAX_NULL_RATE = 0.5  # >50% null → skip
_MIN_CARDINALITY_FOR_DIM = 2
_MAX_CARDINALITY_FOR_DIM = 200


def _is_valid_measure(col: ColumnFingerprint, row_count: int) -> bool:
    """A measure is valid if it has enough non-null data."""
    return col.null_rate < _MAX_NULL_RATE and row_count >= _MIN_ROWS


def _is_valid_dimension(col: ColumnFingerprint) -> bool:
    """A dimension is valid if cardinality is in a useful range."""
    return (
        _MIN_CARDINALITY_FOR_DIM <= col.cardinality <= _MAX_CARDINALITY_FOR_DIM
        and col.null_rate < _MAX_NULL_RATE
    )


# ---------------------------------------------------------------------------
# Canonical dedup key (Gap #6)
# ---------------------------------------------------------------------------


def _make_dedup_key(
    operation: str,
    table: str,
    target: list[str],
    segmenters: list[str],
    controls: list[str],
    join_spec: dict | None = None,
) -> str:
    """Canonical key for deduplication."""
    # Symmetric operations: sort target alphabetically
    if operation in ("CORRELATE",):
        sorted_target = sorted(target)
    else:
        sorted_target = list(target)

    parts = [
        operation,
        table,
        "|".join(sorted_target),
        "|".join(sorted(segmenters)),
        "|".join(sorted(controls)),
    ]
    if join_spec:
        parts.append(f"join:{join_spec.get('table', '')}:{join_spec.get('local_col', '')}:{join_spec.get('remote_col', '')}")

    raw = "::".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Priority scoring (for budgeted tiers only)
# ---------------------------------------------------------------------------


def _score_candidate(
    candidate: AnalysisCandidate,
    fingerprints: dict[str, TableFingerprint],
) -> float:
    """Score a budgeted candidate by expected value."""
    score = 0.5  # base

    fp = fingerprints.get(candidate.table_name)
    if not fp:
        return score

    # Boost for low-null target columns
    for col_name in candidate.target:
        col = fp.columns.get(col_name)
        if col:
            score += (1.0 - col.null_rate) * 0.2

    # Boost for good-cardinality segmenters
    for col_name in candidate.segmenters:
        col = fp.columns.get(col_name)
        if col and 3 <= col.cardinality <= 20:
            score += 0.15

    # Boost cross-table candidates with high confidence joins
    if candidate.join_spec:
        confidence = candidate.join_spec.get("confidence", 0.0)
        score += confidence * 0.3

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Enumeration — Tier 0 (single-column, exhaustive)
# ---------------------------------------------------------------------------


def _enumerate_tier0(fp: TableFingerprint) -> list[AnalysisCandidate]:
    """Tier 0: DESCRIBE + DETECT_ANOMALY for every valid measure. Exhaustive."""
    candidates = []
    for col_name in fp.measures:
        col = fp.columns.get(col_name)
        if not col or not _is_valid_measure(col, fp.row_count):
            continue

        # DESCRIBE for numeric
        c = AnalysisCandidate(
            operation="DESCRIBE",
            table_name=fp.table_name,
            target=[col_name],
            tier=0,
        )
        c.dedup_key = _make_dedup_key("DESCRIBE", fp.table_name, [col_name], [], [])
        candidates.append(c)

        # DETECT_ANOMALY for numeric with enough rows
        if fp.row_count >= 20:
            c = AnalysisCandidate(
                operation="DETECT_ANOMALY",
                table_name=fp.table_name,
                target=[col_name],
                tier=0,
            )
            c.dedup_key = _make_dedup_key("DETECT_ANOMALY", fp.table_name, [col_name], [], [])
            candidates.append(c)

    # DESCRIBE_CATEGORICAL for dimensions
    for col_name in fp.dimensions:
        col = fp.columns.get(col_name)
        if not col or col.null_rate >= _MAX_NULL_RATE:
            continue

        c = AnalysisCandidate(
            operation="DESCRIBE_CATEGORICAL",
            table_name=fp.table_name,
            target=[col_name],
            tier=0,
        )
        c.dedup_key = _make_dedup_key("DESCRIBE_CATEGORICAL", fp.table_name, [col_name], [], [])
        candidates.append(c)

    return candidates


# ---------------------------------------------------------------------------
# Enumeration — Tier 1 (pairwise, exhaustive)
# ---------------------------------------------------------------------------


def _enumerate_tier1(fp: TableFingerprint) -> list[AnalysisCandidate]:
    """Tier 1: COMPARE, CORRELATE, RANK for every valid pair. Exhaustive."""
    candidates = []
    valid_measures = [
        m for m in fp.measures
        if fp.columns.get(m) and _is_valid_measure(fp.columns[m], fp.row_count)
    ]
    valid_dims = [
        d for d in fp.dimensions
        if fp.columns.get(d) and _is_valid_dimension(fp.columns[d])
    ]

    # CORRELATE: measure × measure (symmetric → canonical order)
    for m1, m2 in itertools.combinations(valid_measures, 2):
        c = AnalysisCandidate(
            operation="CORRELATE",
            table_name=fp.table_name,
            target=sorted([m1, m2]),  # canonical order for symmetric op
            tier=1,
        )
        c.dedup_key = _make_dedup_key("CORRELATE", fp.table_name, sorted([m1, m2]), [], [])
        candidates.append(c)

    # COMPARE + RANK: measure × dimension
    # RANK subsumes COMPARE (richer output) → keep RANK, skip COMPARE for same pair
    for measure in valid_measures:
        for dim in valid_dims:
            # RANK (subsumes COMPARE — Gap #6)
            c = AnalysisCandidate(
                operation="RANK",
                table_name=fp.table_name,
                target=[measure],
                segmenters=[dim],
                tier=1,
            )
            c.dedup_key = _make_dedup_key("RANK", fp.table_name, [measure], [dim], [])
            candidates.append(c)

    return candidates


# ---------------------------------------------------------------------------
# Enumeration — Tier 2 (+1 segmenter, budgeted)
# ---------------------------------------------------------------------------


def _enumerate_tier2(
    fp: TableFingerprint,
    tier1_candidates: list[AnalysisCandidate],
) -> list[AnalysisCandidate]:
    """Tier 2: Add 1 segmenter to Tier 1 RANK candidates. Budgeted."""
    candidates = []
    valid_dims = [
        d for d in fp.dimensions
        if fp.columns.get(d) and _is_valid_dimension(fp.columns[d])
    ]

    for base in tier1_candidates:
        if base.operation != "RANK":
            continue
        for dim in valid_dims:
            if dim in base.segmenters:
                continue
            c = AnalysisCandidate(
                operation="RANK",
                table_name=fp.table_name,
                target=list(base.target),
                segmenters=list(base.segmenters) + [dim],
                tier=2,
            )
            c.dedup_key = _make_dedup_key(
                "RANK", fp.table_name, c.target, c.segmenters, []
            )
            candidates.append(c)

    return candidates


# ---------------------------------------------------------------------------
# Enumeration — Tier 3 (cross-table, budgeted)
# ---------------------------------------------------------------------------


def _enumerate_tier3(
    fingerprints: dict[str, TableFingerprint],
    relationships: list[dict],
) -> list[AnalysisCandidate]:
    """Tier 3: Cross-table analysis via discovered relationships. Budgeted."""
    candidates = []

    for rel in relationships:
        confidence = rel.get("confidence", 0.0)
        if confidence < 0.7:
            continue

        table_a = rel.get("table_a", "")
        table_b = rel.get("table_b", "")
        col_a = rel.get("column_a", "")
        col_b = rel.get("column_b", "")

        fp_a = fingerprints.get(table_a)
        fp_b = fingerprints.get(table_b)
        if not fp_a or not fp_b:
            continue

        # Measures from table A, dimensions from table B
        for measure in fp_a.measures[:5]:
            col = fp_a.columns.get(measure)
            if not col or not _is_valid_measure(col, fp_a.row_count):
                continue
            for dim in fp_b.dimensions[:5]:
                dim_col = fp_b.columns.get(dim)
                if not dim_col or not _is_valid_dimension(dim_col):
                    continue
                join = {"table": table_b, "local_col": col_a, "remote_col": col_b, "confidence": confidence}
                c = AnalysisCandidate(
                    operation="RANK",
                    table_name=table_a,
                    target=[measure],
                    segmenters=[f"{table_b}.{dim}"],
                    join_spec=join,
                    tier=3,
                )
                c.dedup_key = _make_dedup_key("RANK", table_a, [measure], [f"{table_b}.{dim}"], [], join)
                candidates.append(c)

        # Correlate: measures from A vs measures from B
        for m_a in fp_a.measures[:5]:
            col_ma = fp_a.columns.get(m_a)
            if not col_ma or not _is_valid_measure(col_ma, fp_a.row_count):
                continue
            for m_b in fp_b.measures[:5]:
                col_mb = fp_b.columns.get(m_b)
                if not col_mb or not _is_valid_measure(col_mb, fp_b.row_count):
                    continue
                join = {"table": table_b, "local_col": col_a, "remote_col": col_b, "confidence": confidence}
                c = AnalysisCandidate(
                    operation="CORRELATE",
                    table_name=table_a,
                    target=sorted([m_a, f"{table_b}.{m_b}"]),
                    join_spec=join,
                    tier=3,
                )
                c.dedup_key = _make_dedup_key(
                    "CORRELATE", table_a, sorted([m_a, f"{table_b}.{m_b}"]), [], [], join
                )
                candidates.append(c)

    return candidates


# ---------------------------------------------------------------------------
# Enumeration — Tier 4 (multi-segment, budgeted)
# ---------------------------------------------------------------------------


def _enumerate_tier4(
    fp: TableFingerprint,
    tier2_candidates: list[AnalysisCandidate],
) -> list[AnalysisCandidate]:
    """Tier 4: 2+ segmenters. Budgeted."""
    candidates = []
    valid_dims = [
        d for d in fp.dimensions
        if fp.columns.get(d) and _is_valid_dimension(fp.columns[d])
    ]

    for base in tier2_candidates:
        if len(base.segmenters) < 2:
            continue
        for dim in valid_dims:
            if dim in base.segmenters:
                continue
            if len(base.segmenters) >= 3:
                continue
            c = AnalysisCandidate(
                operation=base.operation,
                table_name=fp.table_name,
                target=list(base.target),
                segmenters=list(base.segmenters) + [dim],
                tier=4,
            )
            c.dedup_key = _make_dedup_key(
                c.operation, fp.table_name, c.target, c.segmenters, []
            )
            candidates.append(c)

    return candidates


# ---------------------------------------------------------------------------
# Main enumeration
# ---------------------------------------------------------------------------


def enumerate_operations(
    fingerprints: dict[str, TableFingerprint],
    relationships: list[dict],
    budget: EnumerationBudget | None = None,
) -> list[AnalysisCandidate]:
    """Generate all analysis candidates across all tiers.

    Tier 0+1: Exhaustive (all valid combos, no cap).
    Tiers 2-4: Budgeted (scored, capped by tier limit).
    """
    if budget is None:
        budget = EnumerationBudget()

    all_candidates: list[AnalysisCandidate] = []
    seen_keys: set[str] = set()

    def _add(candidates: list[AnalysisCandidate]) -> list[AnalysisCandidate]:
        """Add candidates, deduplicating by canonical key."""
        added = []
        for c in candidates:
            if c.dedup_key and c.dedup_key in seen_keys:
                continue
            seen_keys.add(c.dedup_key)
            added.append(c)
        return added

    # EXHAUSTIVE: Tier 0 + Tier 1 — all valid combos, no limit
    for fp in fingerprints.values():
        tier0 = _enumerate_tier0(fp)
        all_candidates.extend(_add(tier0))

        tier1 = _enumerate_tier1(fp)
        all_candidates.extend(_add(tier1))

    # BUDGETED: Tier 2 — score, sort, cap
    tier2_all: list[AnalysisCandidate] = []
    for fp in fingerprints.values():
        tier1_for_table = [c for c in all_candidates if c.tier == 1 and c.table_name == fp.table_name]
        tier2_all.extend(_enumerate_tier2(fp, tier1_for_table))
    tier2_deduped = _add(tier2_all)
    for c in tier2_deduped:
        c.priority_score = _score_candidate(c, fingerprints)
    tier2_deduped.sort(key=lambda c: c.priority_score, reverse=True)
    tier2_limit = budget.budgeted_tier_limits.get(2, 100)
    all_candidates.extend(tier2_deduped[:tier2_limit])

    # BUDGETED: Tier 3 — cross-table
    tier3_all = _enumerate_tier3(fingerprints, relationships)
    tier3_deduped = _add(tier3_all)
    for c in tier3_deduped:
        c.priority_score = _score_candidate(c, fingerprints)
    tier3_deduped.sort(key=lambda c: c.priority_score, reverse=True)
    tier3_limit = budget.budgeted_tier_limits.get(3, 50)
    all_candidates.extend(tier3_deduped[:tier3_limit])

    # BUDGETED: Tier 4 — multi-segment
    tier4_all: list[AnalysisCandidate] = []
    tier2_used = [c for c in all_candidates if c.tier == 2]
    for fp in fingerprints.values():
        tier4_all.extend(_enumerate_tier4(fp, tier2_used))
    tier4_deduped = _add(tier4_all)
    for c in tier4_deduped:
        c.priority_score = _score_candidate(c, fingerprints)
    tier4_deduped.sort(key=lambda c: c.priority_score, reverse=True)
    tier4_limit = budget.budgeted_tier_limits.get(4, 50)
    all_candidates.extend(tier4_deduped[:tier4_limit])

    return all_candidates
