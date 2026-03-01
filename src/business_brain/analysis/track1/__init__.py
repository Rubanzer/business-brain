"""Track 1 — Algorithmic Analysis Engine.

Public API: run_track1()

Pipeline:
1. Fingerprint tables
2. Enumerate: exhaustive Tier 0+1 (all valid combos) + budgeted Tiers 2-4
3. Execute: Tier 0+1 unconditionally, Tiers 2-4 up to budget (incremental skip via data_hash)
4. Score all results
5. Spawn follow-ups from top findings (Gap #7)
6. Execute follow-ups (second pass)
7. Merge + re-score + rank
8. Return top N
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.models import AnalysisResult
from business_brain.analysis.track1.enumerator import (
    AnalysisCandidate,
    EnumerationBudget,
    enumerate_operations,
)
from business_brain.analysis.track1.executor import TimeScope, execute_batch, execute_one
from business_brain.analysis.track1.fingerprinter import TableFingerprint, fingerprint_table
from business_brain.analysis.track1.scorer import score_batch, spawn_followups
from business_brain.db.discovery_models import DiscoveredRelationship

logger = logging.getLogger(__name__)


async def run_track1(
    session: AsyncSession,
    table_names: list[str],
    run_id: str,
    time_scope: TimeScope | None = None,
    budget: EnumerationBudget | None = None,
    weights: dict[str, float] | None = None,
    top_n: int = 30,
) -> list[AnalysisResult]:
    """Run the full Track 1 algorithmic analysis pipeline.

    Returns the top N findings sorted by interestingness score.
    """

    # 1. Fingerprint all tables
    logger.info("Track 1: Fingerprinting %d tables", len(table_names))
    fingerprints: dict[str, TableFingerprint] = {}
    for name in table_names:
        try:
            fp = await fingerprint_table(session, name)
            fingerprints[name] = fp
            logger.info(
                "  %s: %d rows, %d measures, %d dimensions, time_index=%s",
                name, fp.row_count, len(fp.measures), len(fp.dimensions), fp.time_index,
            )
        except Exception:
            logger.warning("Failed to fingerprint %s", name, exc_info=True)

    if not fingerprints:
        logger.warning("No tables fingerprinted, aborting Track 1")
        return []

    # 2. Load discovered relationships for cross-table enumeration
    rel_result = await session.execute(select(DiscoveredRelationship))
    raw_rels = rel_result.scalars().all()
    relationships = [
        {
            "table_a": r.table_a,
            "column_a": r.column_a,
            "table_b": r.table_b,
            "column_b": r.column_b,
            "confidence": r.confidence,
        }
        for r in raw_rels
        if r.table_a in fingerprints or r.table_b in fingerprints
    ]

    # 3. Enumerate candidates (exhaustive Tier 0+1 + budgeted 2-4)
    candidates = enumerate_operations(fingerprints, relationships, budget)
    tier_counts = {}
    for c in candidates:
        tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
    logger.info(
        "Track 1: Enumerated %d candidates — %s",
        len(candidates),
        ", ".join(f"T{t}:{n}" for t, n in sorted(tier_counts.items())),
    )

    # 4. Execute candidates
    results = await execute_batch(
        session, candidates, fingerprints, run_id, time_scope, budget,
    )
    logger.info("Track 1: Executed %d results (from %d candidates)", len(results), len(candidates))

    # 5. Score all results
    scored = score_batch(results, weights)

    # 6. Spawn follow-ups from top findings (Gap #7)
    followup_candidates = spawn_followups(scored[:10], fingerprints)
    if followup_candidates:
        logger.info("Track 1: Spawned %d follow-up candidates", len(followup_candidates))
        for fc in followup_candidates:
            ar = await execute_one(session, fc, fingerprints, run_id, time_scope)
            if ar:
                scored.append(ar)

    # 7. Re-score everything (including follow-ups) and rank
    scored = score_batch(scored, weights)

    # 8. Flush and return top N
    await session.flush()
    return scored[:top_n]
