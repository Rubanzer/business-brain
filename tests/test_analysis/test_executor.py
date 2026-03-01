"""Tests for analysis/track1/executor.py — execution + caching + budget."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from business_brain.analysis.models import AnalysisResult
from business_brain.analysis.track1.enumerator import AnalysisCandidate, EnumerationBudget
from business_brain.analysis.track1.executor import (
    TimeScope,
    _check_cache,
    execute_batch,
    execute_one,
)
from business_brain.analysis.track1.fingerprinter import ColumnFingerprint, TableFingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fp(
    table: str = "sales",
    measures: list[str] | None = None,
    dimensions: list[str] | None = None,
) -> TableFingerprint:
    measures = measures or ["revenue"]
    dimensions = dimensions or ["region"]
    columns = {}
    for m in measures:
        columns[m] = ColumnFingerprint(
            name=m, semantic_type="numeric_metric", role="MEASURE",
            cardinality=100, null_rate=0.02,
        )
    for d in dimensions:
        columns[d] = ColumnFingerprint(
            name=d, semantic_type="categorical", role="DIMENSION",
            cardinality=5, null_rate=0.01,
        )
    return TableFingerprint(
        table_name=table, row_count=1000, data_hash="hash123",
        domain_hint="sales", time_index="date",
        measures=measures, dimensions=dimensions, columns=columns,
    )


def _make_candidate(
    operation: str = "DESCRIBE",
    table: str = "sales",
    target: list[str] | None = None,
    segmenters: list[str] | None = None,
    tier: int = 0,
    priority: float = 0.5,
) -> AnalysisCandidate:
    target = target or ["revenue"]
    segmenters = segmenters or []
    return AnalysisCandidate(
        operation=operation,
        table_name=table,
        target=target,
        segmenters=segmenters,
        controls=[],
        join_spec=None,
        tier=tier,
        priority_score=priority,
        dedup_key=f"{operation}:{table}:{':'.join(sorted(target))}",
    )


# ---------------------------------------------------------------------------
# TimeScope
# ---------------------------------------------------------------------------


class TestTimeScope:
    def test_defaults(self):
        ts = TimeScope()
        assert ts.column is None
        assert ts.window == "all"
        assert ts.compare_to is None

    def test_custom(self):
        ts = TimeScope(column="date", window="30d", compare_to="previous_period")
        assert ts.column == "date"
        assert ts.window == "30d"


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------


class TestCheckCache:
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_hash(self):
        session = AsyncMock()
        result = await _check_cache(session, "t", "", "key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_key(self):
        session = AsyncMock()
        result = await _check_cache(session, "t", "hash", "")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        session = AsyncMock()
        cached_result = MagicMock(spec=AnalysisResult)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cached_result
        session.execute = AsyncMock(return_value=mock_result)

        result = await _check_cache(session, "sales", "hash123", "key456")
        assert result is cached_result

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        result = await _check_cache(session, "sales", "hash123", "key456")
        assert result is None


# ---------------------------------------------------------------------------
# execute_one
# ---------------------------------------------------------------------------


class TestExecuteOne:
    @pytest.mark.asyncio
    async def test_returns_none_for_missing_fingerprint(self):
        session = AsyncMock()
        candidate = _make_candidate()
        result = await execute_one(session, candidate, {}, "run-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_skips_execution(self):
        """Incremental skip via data_hash (Gap #4)."""
        session = AsyncMock()
        fp = _make_fp()
        candidate = _make_candidate()

        cached = MagicMock(spec=AnalysisResult)
        with patch("business_brain.analysis.track1.executor._check_cache", return_value=cached):
            result = await execute_one(session, candidate, {"sales": fp}, "run-1")
            assert result is cached

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_none(self):
        session = AsyncMock()
        fp = _make_fp()
        candidate = _make_candidate(operation="UNKNOWN_OP")

        with patch("business_brain.analysis.track1.executor._check_cache", return_value=None):
            result = await execute_one(session, candidate, {"sales": fp}, "run-1")
            assert result is None

    @pytest.mark.asyncio
    async def test_auto_detects_time_column(self):
        """When time_scope has no column, should use fingerprint's time_index."""
        session = AsyncMock()
        fp = _make_fp()
        fp.time_index = "order_date"
        candidate = _make_candidate()
        time_scope = TimeScope(column=None, window="30d")

        mock_result = MagicMock()
        mock_result.to_series.return_value = MagicMock(tolist=MagicMock(return_value=[1.0, 2.0, 3.0]))
        mock_result.row_count = 3
        mock_result.query = "SELECT ..."
        mock_result.rows = []

        with patch("business_brain.analysis.track1.executor._check_cache", return_value=None), \
             patch("business_brain.analysis.track1.executor.execute", return_value=mock_result) as mock_execute, \
             patch("business_brain.analysis.track1.executor.compute") as mock_compute:
            mock_compute.describe_numeric.return_value = {"mean": 2.0, "count": 3}
            result = await execute_one(session, candidate, {"sales": fp}, "run-1", time_scope)
            # Should have called execute with intent containing time filter
            assert result is not None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        """10s timeout should return None on slow operations."""
        import asyncio
        session = AsyncMock()
        fp = _make_fp()
        candidate = _make_candidate()

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(100)

        with patch("business_brain.analysis.track1.executor._check_cache", return_value=None), \
             patch("business_brain.analysis.track1.executor._execute_describe", side_effect=slow_execute):
            result = await execute_one(session, candidate, {"sales": fp}, "run-1")
            assert result is None


# ---------------------------------------------------------------------------
# execute_batch
# ---------------------------------------------------------------------------


class TestExecuteBatch:
    @pytest.mark.asyncio
    async def test_exhaustive_tier01(self):
        """All Tier 0+1 candidates execute regardless of budget."""
        session = AsyncMock()
        session.flush = AsyncMock()
        fp = _make_fp()

        t0 = _make_candidate(operation="DESCRIBE", tier=0)
        t1 = _make_candidate(operation="CORRELATE", tier=1, target=["revenue", "cost"])

        mock_ar = MagicMock(spec=AnalysisResult)
        with patch("business_brain.analysis.track1.executor.execute_one", return_value=mock_ar):
            results = await execute_batch(
                session, [t0, t1], {"sales": fp}, "run-1",
                budget=EnumerationBudget(budgeted_tier_limits={2: 0, 3: 0, 4: 0}),
            )
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_budget_enforcement_tier2(self):
        """Tier 2 candidates should be capped at budget."""
        session = AsyncMock()
        session.flush = AsyncMock()
        fp = _make_fp()

        candidates = [
            _make_candidate(operation="RANK", tier=2, priority=0.9 - i * 0.1,
                            target=[f"m{i}"], segmenters=["d1", "d2"])
            for i in range(5)
        ]

        mock_ar = MagicMock(spec=AnalysisResult)
        with patch("business_brain.analysis.track1.executor.execute_one", return_value=mock_ar):
            results = await execute_batch(
                session, candidates, {"sales": fp}, "run-1",
                budget=EnumerationBudget(budgeted_tier_limits={2: 2, 3: 0, 4: 0}),
            )
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_priority_ordering_for_budgeted(self):
        """Higher priority_score candidates should be executed first."""
        session = AsyncMock()
        session.flush = AsyncMock()
        fp = _make_fp()

        executed_order = []

        async def track_execution(sess, candidate, fps, run_id, ts=None):
            executed_order.append(candidate.priority_score)
            return MagicMock(spec=AnalysisResult)

        candidates = [
            _make_candidate(tier=2, priority=0.3, target=["m1"], segmenters=["d1", "d2"]),
            _make_candidate(tier=2, priority=0.9, target=["m2"], segmenters=["d1", "d2"]),
            _make_candidate(tier=2, priority=0.6, target=["m3"], segmenters=["d1", "d2"]),
        ]
        # Give unique dedup keys
        for i, c in enumerate(candidates):
            c.dedup_key = f"key-{i}"

        with patch("business_brain.analysis.track1.executor.execute_one", side_effect=track_execution):
            await execute_batch(
                session, candidates, {"sales": fp}, "run-1",
                budget=EnumerationBudget(budgeted_tier_limits={2: 10, 3: 0, 4: 0}),
            )
            # Should be sorted by priority descending
            assert executed_order == [0.9, 0.6, 0.3]

    @pytest.mark.asyncio
    async def test_mixed_tiers(self):
        """Mix of exhaustive and budgeted candidates."""
        session = AsyncMock()
        session.flush = AsyncMock()
        fp = _make_fp()

        candidates = [
            _make_candidate(tier=0, target=["m1"]),
            _make_candidate(tier=1, target=["m1", "m2"]),
            _make_candidate(tier=2, priority=0.8, target=["m1"], segmenters=["d1", "d2"]),
            _make_candidate(tier=2, priority=0.5, target=["m2"], segmenters=["d1", "d2"]),
            _make_candidate(tier=3, priority=0.7, target=["m1"]),
        ]
        for i, c in enumerate(candidates):
            c.dedup_key = f"key-{i}"

        mock_ar = MagicMock(spec=AnalysisResult)
        with patch("business_brain.analysis.track1.executor.execute_one", return_value=mock_ar):
            results = await execute_batch(
                session, candidates, {"sales": fp}, "run-1",
                budget=EnumerationBudget(budgeted_tier_limits={2: 1, 3: 1, 4: 0}),
            )
            # 2 exhaustive + 1 tier2 (budget=1) + 1 tier3 (budget=1)
            assert len(results) == 4

    @pytest.mark.asyncio
    async def test_skips_failed_executions(self):
        """execute_one returning None should not count toward results."""
        session = AsyncMock()
        session.flush = AsyncMock()
        fp = _make_fp()

        candidates = [
            _make_candidate(tier=0, target=["m1"]),
            _make_candidate(tier=0, target=["m2"]),
        ]
        for i, c in enumerate(candidates):
            c.dedup_key = f"key-{i}"

        call_count = [0]

        async def alternate_result(*args, **kwargs):
            call_count[0] += 1
            return MagicMock(spec=AnalysisResult) if call_count[0] % 2 == 0 else None

        with patch("business_brain.analysis.track1.executor.execute_one", side_effect=alternate_result):
            results = await execute_batch(session, candidates, {"sales": fp}, "run-1")
            assert len(results) == 1  # only one succeeded
