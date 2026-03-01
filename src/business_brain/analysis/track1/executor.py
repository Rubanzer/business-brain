"""Analysis Executor — runs candidates against the database.

Supports:
- Exhaustive execution for Tier 0+1 (no budget cap)
- Budget enforcement for Tiers 2-4
- Incremental skip via data_hash (Gap #4)
- Time scoping (Gap #5)
- Cross-table JOINs (Gap #2)
- N-ary GROUP BY (Gap #1)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.models import AnalysisResult
from business_brain.analysis.tools import compute
from business_brain.analysis.tools.sql_executor import (
    AggregationSpec,
    FilterSpec,
    JoinSpec,
    QueryIntent,
    QueryResult,
    TimeRange,
    execute,
)
from business_brain.analysis.track1.enumerator import AnalysisCandidate, EnumerationBudget
from business_brain.analysis.track1.fingerprinter import TableFingerprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Time scope
# ---------------------------------------------------------------------------


@dataclass
class TimeScope:
    """First-class time scoping for analysis windows (Gap #5)."""

    column: str | None = None  # time column (auto-detected from TIME_INDEX)
    window: str = "all"  # "7d", "30d", "90d", "ytd", "all"
    compare_to: str | None = None  # "previous_period" for period-over-period


# ---------------------------------------------------------------------------
# Incremental cache check
# ---------------------------------------------------------------------------


async def _check_cache(
    session: AsyncSession,
    table_name: str,
    data_hash: str,
    dedup_key: str,
) -> AnalysisResult | None:
    """Check if a valid cached result exists for this candidate."""
    if not data_hash or not dedup_key:
        return None
    result = await session.execute(
        select(AnalysisResult).where(
            AnalysisResult.table_name == table_name,
            AnalysisResult.data_hash == data_hash,
            AnalysisResult.dedup_key == dedup_key,
        ).order_by(AnalysisResult.created_at.desc()).limit(1)
    )
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Operation executors
# ---------------------------------------------------------------------------


async def _execute_describe(
    session: AsyncSession,
    candidate: AnalysisCandidate,
    fp: TableFingerprint,
    time_scope: TimeScope | None,
) -> dict[str, Any]:
    """Execute a DESCRIBE operation."""
    col = candidate.target[0]

    intent = QueryIntent(
        tables=[candidate.table_name],
        select_columns=[col],
        filters=[FilterSpec(column=col, operator="IS NOT NULL")],
    )
    if time_scope and time_scope.column and time_scope.window != "all":
        intent.time_range = TimeRange(column=time_scope.column, window=time_scope.window)
    if candidate.join_spec:
        j = candidate.join_spec
        intent.join = JoinSpec(table=j["table"], local_col=j["local_col"], remote_col=j["remote_col"])

    result = await execute(session, intent)
    values = result.to_series(col).tolist()

    if not values:
        return {"error": "no data", "query": result.query}

    stats = compute.describe_numeric(values)
    dist = compute.detect_distribution(values) if len(values) >= 20 else None

    return {
        "stats": stats,
        "distribution": dist,
        "row_count": result.row_count,
        "query": result.query,
    }


async def _execute_describe_categorical(
    session: AsyncSession,
    candidate: AnalysisCandidate,
    fp: TableFingerprint,
    time_scope: TimeScope | None,
) -> dict[str, Any]:
    """Execute a DESCRIBE_CATEGORICAL operation."""
    col = candidate.target[0]

    intent = QueryIntent(
        tables=[candidate.table_name],
        select_columns=[col],
        filters=[FilterSpec(column=col, operator="IS NOT NULL")],
    )
    if time_scope and time_scope.column and time_scope.window != "all":
        intent.time_range = TimeRange(column=time_scope.column, window=time_scope.window)

    result = await execute(session, intent)
    values = [r[col] for r in result.rows]
    stats = compute.describe_categorical(values)
    return {"stats": stats, "row_count": result.row_count, "query": result.query}


async def _execute_correlate(
    session: AsyncSession,
    candidate: AnalysisCandidate,
    fp: TableFingerprint,
    time_scope: TimeScope | None,
) -> dict[str, Any]:
    """Execute a CORRELATE operation."""
    col1, col2 = candidate.target[0], candidate.target[1]

    intent = QueryIntent(
        tables=[candidate.table_name],
        select_columns=[col1, col2],
        filters=[
            FilterSpec(column=col1, operator="IS NOT NULL"),
            FilterSpec(column=col2, operator="IS NOT NULL"),
        ],
    )
    if time_scope and time_scope.column and time_scope.window != "all":
        intent.time_range = TimeRange(column=time_scope.column, window=time_scope.window)
    if candidate.join_spec:
        j = candidate.join_spec
        intent.join = JoinSpec(table=j["table"], local_col=j["local_col"], remote_col=j["remote_col"])

    result = await execute(session, intent)
    x = result.to_series(col1).tolist()
    y = result.to_series(col2).tolist()

    if len(x) < 3:
        return {"error": "not enough data", "query": result.query}

    corr = compute.compute_correlation(x, y)
    corr["query"] = result.query
    corr["row_count"] = result.row_count
    return corr


async def _execute_rank(
    session: AsyncSession,
    candidate: AnalysisCandidate,
    fp: TableFingerprint,
    time_scope: TimeScope | None,
) -> dict[str, Any]:
    """Execute a RANK operation (subsumes COMPARE)."""
    measure = candidate.target[0]
    dims = candidate.segmenters

    intent = QueryIntent(
        tables=[candidate.table_name],
        select_columns=list(dims),
        aggregations=[
            AggregationSpec(column=measure, function="AVG", alias=f"avg_{measure}"),
            AggregationSpec(column=measure, function="COUNT", alias="cnt"),
            AggregationSpec(column=measure, function="STDDEV", alias=f"std_{measure}"),
        ],
        group_by=list(dims),
        filters=[FilterSpec(column=measure, operator="IS NOT NULL")],
        order_by=[f"avg_{measure} DESC"],
    )
    if time_scope and time_scope.column and time_scope.window != "all":
        intent.time_range = TimeRange(column=time_scope.column, window=time_scope.window)
    if candidate.join_spec:
        j = candidate.join_spec
        intent.join = JoinSpec(table=j["table"], local_col=j["local_col"], remote_col=j["remote_col"])

    result = await execute(session, intent)

    if result.row_count < 2:
        return {"error": "not enough groups", "query": result.query}

    # Group comparison for effect size
    groups: dict[str, list[float]] = {}
    # Fetch raw values for group comparison
    raw_intent = QueryIntent(
        tables=[candidate.table_name],
        select_columns=list(dims) + [measure],
        filters=[FilterSpec(column=measure, operator="IS NOT NULL")],
    )
    if time_scope and time_scope.column and time_scope.window != "all":
        raw_intent.time_range = TimeRange(column=time_scope.column, window=time_scope.window)
    if candidate.join_spec:
        j = candidate.join_spec
        raw_intent.join = JoinSpec(table=j["table"], local_col=j["local_col"], remote_col=j["remote_col"])

    raw_result = await execute(session, raw_intent)
    primary_dim = dims[0]
    for row in raw_result.rows:
        key = str(row.get(primary_dim, ""))
        val = row.get(measure)
        if val is not None:
            try:
                groups.setdefault(key, []).append(float(val))
            except (ValueError, TypeError):
                continue

    comparison = compute.compare_groups(groups) if len(groups) >= 2 else None

    return {
        "ranked": result.rows,
        "group_count": result.row_count,
        "comparison": comparison,
        "query": result.query,
    }


async def _execute_detect_anomaly(
    session: AsyncSession,
    candidate: AnalysisCandidate,
    fp: TableFingerprint,
    time_scope: TimeScope | None,
) -> dict[str, Any]:
    """Execute a DETECT_ANOMALY operation."""
    col = candidate.target[0]

    intent = QueryIntent(
        tables=[candidate.table_name],
        select_columns=[col],
        filters=[FilterSpec(column=col, operator="IS NOT NULL")],
    )
    if time_scope and time_scope.column and time_scope.window != "all":
        intent.time_range = TimeRange(column=time_scope.column, window=time_scope.window)

    result = await execute(session, intent)
    values = result.to_series(col).tolist()

    if len(values) < 10:
        return {"error": "not enough data", "query": result.query}

    anomalies = compute.find_anomalies_zscore(values)
    anomalies["query"] = result.query
    anomalies["row_count"] = result.row_count
    return anomalies


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "DESCRIBE": _execute_describe,
    "DESCRIBE_CATEGORICAL": _execute_describe_categorical,
    "CORRELATE": _execute_correlate,
    "RANK": _execute_rank,
    "DETECT_ANOMALY": _execute_detect_anomaly,
}


# ---------------------------------------------------------------------------
# Main batch executor
# ---------------------------------------------------------------------------


async def execute_one(
    session: AsyncSession,
    candidate: AnalysisCandidate,
    fingerprints: dict[str, TableFingerprint],
    run_id: str,
    time_scope: TimeScope | None = None,
) -> AnalysisResult | None:
    """Execute a single candidate and return an AnalysisResult (or None on skip/error)."""
    fp = fingerprints.get(candidate.table_name)
    if not fp:
        return None

    # Incremental cache check (Gap #4)
    cached = await _check_cache(session, candidate.table_name, fp.data_hash, candidate.dedup_key)
    if cached:
        logger.debug("Cache hit for %s on %s", candidate.operation, candidate.dedup_key[:8])
        return cached

    # Auto-detect time column from fingerprint
    effective_scope = time_scope
    if effective_scope and not effective_scope.column and fp.time_index:
        effective_scope = TimeScope(
            column=fp.time_index,
            window=effective_scope.window,
            compare_to=effective_scope.compare_to,
        )

    # Execute
    dispatch_fn = _DISPATCH.get(candidate.operation)
    if not dispatch_fn:
        logger.warning("Unknown operation: %s", candidate.operation)
        return None

    try:
        result_data = await asyncio.wait_for(
            dispatch_fn(session, candidate, fp, effective_scope),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Timeout executing %s for %s", candidate.operation, candidate.dedup_key[:8])
        return None
    except Exception:
        logger.warning("Error executing %s", candidate.operation, exc_info=True)
        return None

    # Build AnalysisResult
    ar = AnalysisResult(
        run_id=run_id,
        operation_type=candidate.operation,
        table_name=candidate.table_name,
        data_hash=fp.data_hash,
        tier=candidate.tier,
        target=candidate.target,
        segmenters=candidate.segmenters or [],
        controls=candidate.controls or [],
        join_spec=candidate.join_spec,
        dedup_key=candidate.dedup_key,
        result_data=result_data,
    )
    session.add(ar)
    return ar


async def execute_batch(
    session: AsyncSession,
    candidates: list[AnalysisCandidate],
    fingerprints: dict[str, TableFingerprint],
    run_id: str,
    time_scope: TimeScope | None = None,
    budget: EnumerationBudget | None = None,
) -> list[AnalysisResult]:
    """Execute a batch of candidates.

    Tier 0+1: ALL execute unconditionally (exhaustive).
    Tiers 2-4: Execute up to budget limits, in priority order.
    """
    if budget is None:
        budget = EnumerationBudget()

    results: list[AnalysisResult] = []

    # Split by tier class
    exhaustive = [c for c in candidates if c.tier <= 1]
    budgeted = [c for c in candidates if c.tier >= 2]

    # Execute ALL exhaustive candidates (Tier 0+1)
    for candidate in exhaustive:
        ar = await execute_one(session, candidate, fingerprints, run_id, time_scope)
        if ar:
            results.append(ar)

    # Execute budgeted candidates by tier, up to limits
    tier_counts: dict[int, int] = {}
    budgeted.sort(key=lambda c: (c.tier, -c.priority_score))

    for candidate in budgeted:
        tier = candidate.tier
        limit = budget.budgeted_tier_limits.get(tier, 50)
        count = tier_counts.get(tier, 0)
        if count >= limit:
            continue

        ar = await execute_one(session, candidate, fingerprints, run_id, time_scope)
        if ar:
            results.append(ar)
            tier_counts[tier] = count + 1

    await session.flush()
    return results
