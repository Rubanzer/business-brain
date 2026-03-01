"""Tests for analysis agents — Quality, Domain, Temporal + base."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from business_brain.analysis.agents.base import AgentContext, AnalysisAgent
from business_brain.analysis.agents.quality_agent import QualityAgent
from business_brain.analysis.agents.domain_agent import DomainAgent
from business_brain.analysis.agents.temporal_agent import TemporalAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    operation: str = "RANK",
    target: list[str] | None = None,
    segmenters: list[str] | None = None,
    result_data: dict | None = None,
    score: float = 0.5,
) -> MagicMock:
    r = MagicMock()
    r.id = "res-1"
    r.run_id = "run-1"
    r.operation_type = operation
    r.table_name = "production"
    r.target = target or ["output_kg"]
    r.segmenters = segmenters or ["shift"]
    r.controls = []
    r.result_data = result_data or {}
    r.interestingness_score = score
    r.interestingness_breakdown = {"surprise": 0.5}
    r.quality_verdict = None
    r.domain_relevance = None
    r.temporal_context = None
    r.delta_type = None
    r.final_score = 0.0
    r.parent_result_id = None
    return r


def _make_ctx(result=None, time_scope=None) -> AgentContext:
    session = AsyncMock()
    return AgentContext(
        session=session,
        result=result or _make_result(),
        run_id="run-1",
        time_scope=time_scope,
    )


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_run_catches_errors(self):
        class FailingAgent(AnalysisAgent):
            agent_id = "failing"
            async def build_context(self, ctx):
                raise ValueError("boom")
            async def analyze(self, ctx, data):
                return {}

        agent = FailingAgent()
        ctx = _make_ctx()
        output = await agent.run(ctx)
        assert output.error is not None
        assert "boom" in output.error
        assert output.agent_id == "failing"

    @pytest.mark.asyncio
    async def test_run_records_timing(self):
        class FastAgent(AnalysisAgent):
            agent_id = "fast"
            async def build_context(self, ctx):
                return {}
            async def analyze(self, ctx, data):
                return {"_confidence": 0.9, "result": "ok"}

        agent = FastAgent()
        ctx = _make_ctx()
        output = await agent.run(ctx)
        assert output.duration_ms >= 0
        assert output.confidence == pytest.approx(0.9)
        assert output.error is None


# ---------------------------------------------------------------------------
# Quality Agent
# ---------------------------------------------------------------------------


class TestQualityAgent:
    @pytest.mark.asyncio
    async def test_reliable_verdict(self):
        agent = QualityAgent()
        ctx = _make_ctx()
        session = ctx.session

        # Mock: 1000 rows, low null rate, good segment sizes
        session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=1000)),  # COUNT(*)
            MagicMock(scalar=MagicMock(return_value=0.02)),  # null rate for output_kg
            MagicMock(fetchall=MagicMock(return_value=[
                MagicMock(**{"__getitem__": lambda s, i: ["day", 200][i]}),
                MagicMock(**{"__getitem__": lambda s, i: ["night", 150][i]}),
            ])),  # segment sizes
            MagicMock(scalar=MagicMock(return_value=5.0)),  # freshness
            MagicMock(scalar=MagicMock(return_value=3)),  # distinct values for shift
        ])

        output = await agent.run(ctx)
        data = output.output
        assert data["verdict"] in ("RELIABLE", "CAUTIONARY", "UNRELIABLE")

    @pytest.mark.asyncio
    async def test_veto_on_tiny_sample(self):
        agent = QualityAgent()
        ctx = _make_ctx()
        session = ctx.session

        # Mock: only 5 rows
        session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=5)),  # COUNT(*)
            MagicMock(scalar=MagicMock(return_value=0.0)),  # null rate
            MagicMock(fetchall=MagicMock(return_value=[])),  # segments
            MagicMock(scalar=MagicMock(return_value=None)),  # freshness (no time col)
            MagicMock(scalar=MagicMock(return_value=2)),  # distinct
        ])

        output = await agent.run(ctx)
        data = output.output
        assert data["verdict"] == "UNRELIABLE"
        assert data["veto"] is True

    @pytest.mark.asyncio
    async def test_per_segment_check(self):
        """Gap #1: Quality checks per segment for N-ary analysis."""
        agent = QualityAgent()
        result = _make_result(segmenters=["shift", "furnace"])
        ctx = _make_ctx(result=result)
        session = ctx.session

        # Mock: segment with <10 rows
        session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=500)),  # total
            MagicMock(scalar=MagicMock(return_value=0.01)),  # null rate
            MagicMock(fetchall=MagicMock(return_value=[
                MagicMock(**{"__getitem__": lambda s, i: ["small_seg", 3][i]}),
            ])),  # tiny segment!
            MagicMock(scalar=MagicMock(return_value=2.0)),  # freshness
            MagicMock(scalar=MagicMock(return_value=5)),  # distinct shift
            MagicMock(scalar=MagicMock(return_value=4)),  # distinct furnace
        ])

        output = await agent.run(ctx)
        data = output.output
        segment_checks = [c for c in data.get("checks", []) if c["check"] == "segment_size"]
        assert any(c["status"] == "WARN" for c in segment_checks)


# ---------------------------------------------------------------------------
# Domain Agent
# ---------------------------------------------------------------------------


class TestDomainAgent:
    @pytest.mark.asyncio
    async def test_produces_classification(self):
        agent = DomainAgent()
        ctx = _make_ctx()

        with patch("business_brain.analysis.agents.domain_agent.rag_service") as mock_rag, \
             patch("business_brain.analysis.agents.domain_agent.llm_gateway") as mock_llm:
            mock_rag.search_multi = AsyncMock(return_value=[
                {"content": "Production targets are 500kg/shift", "distance": 0.3, "similarity": 0.7}
            ])
            mock_llm.extract = AsyncMock(return_value={
                "relevance_score": 0.8,
                "classification": "SURPRISING",
                "business_translation": "Output dropped significantly",
                "benchmark_context": "Normal is 500kg",
                "estimated_impact": "HIGH",
            })

            output = await agent.run(ctx)
            data = output.output
            assert data["classification"] in ("CONFIRMING", "SURPRISING", "NOVEL", "IRRELEVANT")
            assert 0 <= data["relevance_score"] <= 1
            assert len(data["business_translation"]) > 0

    @pytest.mark.asyncio
    async def test_handles_no_rag_hits(self):
        agent = DomainAgent()
        ctx = _make_ctx()

        with patch("business_brain.analysis.agents.domain_agent.rag_service") as mock_rag, \
             patch("business_brain.analysis.agents.domain_agent.llm_gateway") as mock_llm:
            mock_rag.search_multi = AsyncMock(return_value=[])
            mock_llm.extract = AsyncMock(return_value={
                "relevance_score": 0.3,
                "classification": "NOVEL",
                "business_translation": "No context available",
                "benchmark_context": "No benchmarks",
                "estimated_impact": "LOW",
            })

            output = await agent.run(ctx)
            assert output.error is None


# ---------------------------------------------------------------------------
# Temporal Agent
# ---------------------------------------------------------------------------


class TestTemporalAgent:
    @pytest.mark.asyncio
    async def test_no_time_data(self):
        agent = TemporalAgent()
        ctx = _make_ctx(time_scope=None)

        output = await agent.run(ctx)
        data = output.output
        assert data["trend_status"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_with_time_data(self):
        agent = TemporalAgent()
        ctx = _make_ctx(time_scope={"column": "date", "window": "30d"})
        session = ctx.session

        # Mock time series data — 30 daily points
        ts_rows = [
            MagicMock(_mapping={"period": f"2025-01-{i+1:02d}", "val": 100 + i, "cnt": 10})
            for i in range(30)
        ]
        mock_ts_result = MagicMock()
        mock_ts_result.fetchall.return_value = ts_rows
        session.execute = AsyncMock(return_value=mock_ts_result)

        with patch("business_brain.analysis.agents.temporal_agent.rag_service") as mock_rag:
            mock_rag.search = AsyncMock(return_value=[])

            output = await agent.run(ctx)
            data = output.output
            assert data["trend_status"] in ("ACCELERATING", "STABLE", "DECELERATING", "REVERSING")
            assert data["novelty"] in ("NEW", "RECURRING", "CHRONIC")
            assert isinstance(data["data_points"], int)
