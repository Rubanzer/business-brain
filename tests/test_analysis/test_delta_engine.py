"""Tests for analysis/track3/delta_engine.py — disagreement detection."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from business_brain.analysis.models import AnalysisDelta
from business_brain.analysis.track3.delta_engine import (
    _detect_filtered_by_context,
    _detect_magnitude_disagreement,
    _detect_segment_reversal,
    _detect_unexplained_signal,
    compute_deltas,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    operation: str = "RANK",
    result_data: dict | None = None,
    target: list[str] | None = None,
    segmenters: list[str] | None = None,
    score: float = 0.7,
) -> MagicMock:
    r = MagicMock()
    r.id = "res-1"
    r.run_id = "run-1"
    r.operation_type = operation
    r.table_name = "production"
    r.target = target or ["output_kg"]
    r.segmenters = segmenters or []
    r.controls = []
    r.result_data = result_data or {}
    r.interestingness_score = score
    r.interestingness_breakdown = {"surprise": 0.7}
    r.quality_verdict = None
    r.domain_relevance = None
    r.temporal_context = None
    r.delta_type = None
    r.final_score = 0.0
    r.parent_result_id = None
    return r


def _make_agent_output(output: dict, agent_id: str = "domain") -> MagicMock:
    ao = MagicMock()
    ao.agent_id = agent_id
    ao.output = output
    ao.confidence = 0.8
    ao.duration_ms = 100
    ao.error = None
    return ao


# ---------------------------------------------------------------------------
# FILTERED_BY_CONTEXT
# ---------------------------------------------------------------------------


class TestFilteredByContext:
    def test_detects_filtered(self):
        result = _make_result(score=0.8)
        domain = {"classification": "IRRELEVANT", "relevance_score": 0.1, "business_translation": "Not relevant"}
        delta = _detect_filtered_by_context(result, domain)
        assert delta is not None
        assert delta.delta_type == "FILTERED_BY_CONTEXT"

    def test_no_filter_when_relevant(self):
        result = _make_result(score=0.8)
        domain = {"classification": "SURPRISING", "relevance_score": 0.9}
        delta = _detect_filtered_by_context(result, domain)
        assert delta is None

    def test_no_filter_when_low_score(self):
        result = _make_result(score=0.3)
        domain = {"classification": "IRRELEVANT", "relevance_score": 0.1}
        delta = _detect_filtered_by_context(result, domain)
        assert delta is None

    def test_no_filter_when_no_domain(self):
        result = _make_result(score=0.8)
        delta = _detect_filtered_by_context(result, None)
        assert delta is None

    def test_filtered_by_low_relevance_score(self):
        result = _make_result(score=0.7)
        domain = {"classification": "CONFIRMING", "relevance_score": 0.2}
        delta = _detect_filtered_by_context(result, domain)
        assert delta is not None
        assert delta.delta_type == "FILTERED_BY_CONTEXT"


# ---------------------------------------------------------------------------
# UNEXPLAINED_SIGNAL
# ---------------------------------------------------------------------------


class TestUnexplainedSignal:
    def test_no_domain_output(self):
        result = _make_result(score=0.8)
        delta = _detect_unexplained_signal(result, None)
        assert delta is not None
        assert delta.delta_type == "UNEXPLAINED_SIGNAL"

    def test_novel_and_relevant(self):
        result = _make_result(score=0.8)
        domain = {"classification": "NOVEL", "relevance_score": 0.9, "business_translation": "New pattern"}
        delta = _detect_unexplained_signal(result, domain)
        assert delta is not None
        assert delta.delta_type == "UNEXPLAINED_SIGNAL"

    def test_no_signal_when_confirming(self):
        result = _make_result(score=0.8)
        domain = {"classification": "CONFIRMING", "relevance_score": 0.9}
        delta = _detect_unexplained_signal(result, domain)
        assert delta is None

    def test_no_signal_when_low_score(self):
        result = _make_result(score=0.3)
        delta = _detect_unexplained_signal(result, None)
        assert delta is None

    def test_novel_but_low_relevance(self):
        result = _make_result(score=0.8)
        domain = {"classification": "NOVEL", "relevance_score": 0.3}
        delta = _detect_unexplained_signal(result, domain)
        assert delta is None


# ---------------------------------------------------------------------------
# MAGNITUDE_DISAGREEMENT
# ---------------------------------------------------------------------------


class TestMagnitudeDisagreement:
    def test_surprising_with_benchmark(self):
        result = _make_result(score=0.7)
        domain = {
            "classification": "SURPRISING",
            "benchmark_context": "Normal output is 500kg/shift",
            "business_translation": "Output dropped to 300kg",
            "estimated_impact": "HIGH",
        }
        delta = _detect_magnitude_disagreement(result, domain)
        assert delta is not None
        assert delta.delta_type == "MAGNITUDE_DISAGREEMENT"

    def test_no_disagreement_without_benchmark(self):
        result = _make_result(score=0.7)
        domain = {
            "classification": "SURPRISING",
            "benchmark_context": "No benchmarks available",
        }
        delta = _detect_magnitude_disagreement(result, domain)
        assert delta is None

    def test_no_disagreement_when_confirming(self):
        result = _make_result(score=0.7)
        domain = {
            "classification": "CONFIRMING",
            "benchmark_context": "Normal is 500kg",
        }
        delta = _detect_magnitude_disagreement(result, domain)
        assert delta is None

    def test_no_domain(self):
        result = _make_result()
        delta = _detect_magnitude_disagreement(result, None)
        assert delta is None

    def test_empty_benchmark(self):
        result = _make_result(score=0.7)
        domain = {
            "classification": "SURPRISING",
            "benchmark_context": "",
        }
        delta = _detect_magnitude_disagreement(result, domain)
        assert delta is None


# ---------------------------------------------------------------------------
# SEGMENT_REVERSAL (Simpson's paradox)
# ---------------------------------------------------------------------------


class TestSegmentReversal:
    def test_correlate_sign_flip(self):
        """Simpson's paradox: positive correlation globally, negative in segment."""
        global_result = _make_result(
            operation="CORRELATE",
            result_data={"pearson_r": 0.7},
            segmenters=[],
        )
        global_result.id = "global"

        segmented_result = _make_result(
            operation="CORRELATE",
            result_data={"pearson_r": -0.6},
            segmenters=["shift"],
        )
        segmented_result.id = "segmented"

        delta = _detect_segment_reversal(segmented_result, [global_result, segmented_result])
        assert delta is not None
        assert delta.delta_type == "SEGMENT_REVERSAL"
        assert "Simpson's paradox" in delta.description

    def test_no_reversal_same_sign(self):
        global_result = _make_result(
            operation="CORRELATE",
            result_data={"pearson_r": 0.7},
            segmenters=[],
        )
        global_result.id = "global"

        segmented_result = _make_result(
            operation="CORRELATE",
            result_data={"pearson_r": 0.6},
            segmenters=["shift"],
        )
        segmented_result.id = "segmented"

        delta = _detect_segment_reversal(segmented_result, [global_result, segmented_result])
        assert delta is None

    def test_no_reversal_without_global_match(self):
        segmented_result = _make_result(
            operation="CORRELATE",
            result_data={"pearson_r": -0.6},
            segmenters=["shift"],
        )
        delta = _detect_segment_reversal(segmented_result, [segmented_result])
        assert delta is None

    def test_no_reversal_without_segmenters(self):
        result = _make_result(operation="CORRELATE", segmenters=[])
        delta = _detect_segment_reversal(result, [result])
        assert delta is None

    def test_rank_reversal(self):
        """Top globally but bottom in segment."""
        global_result = _make_result(
            operation="RANK",
            result_data={"ranked": [{"shift": "day"}, {"shift": "night"}]},
            segmenters=[],
        )
        global_result.id = "global"

        segmented_result = _make_result(
            operation="RANK",
            result_data={"ranked": [{"shift": "night"}, {"shift": "day"}]},
            segmenters=["furnace"],
        )
        segmented_result.id = "segmented"

        delta = _detect_segment_reversal(segmented_result, [global_result, segmented_result])
        assert delta is not None
        assert delta.delta_type == "SEGMENT_REVERSAL"

    def test_weak_correlation_no_reversal(self):
        """Weak correlations should not trigger reversal."""
        global_result = _make_result(
            operation="CORRELATE",
            result_data={"pearson_r": 0.3},
            segmenters=[],
        )
        global_result.id = "global"

        segmented_result = _make_result(
            operation="CORRELATE",
            result_data={"pearson_r": -0.2},
            segmenters=["shift"],
        )
        segmented_result.id = "segmented"

        delta = _detect_segment_reversal(segmented_result, [global_result, segmented_result])
        assert delta is None  # both below threshold


# ---------------------------------------------------------------------------
# compute_deltas (full pipeline)
# ---------------------------------------------------------------------------


class TestComputeDeltas:
    @pytest.mark.asyncio
    async def test_finds_multiple_delta_types(self):
        session = AsyncMock()
        session.flush = AsyncMock()

        # Result 1: high score, no domain context → UNEXPLAINED_SIGNAL
        r1 = _make_result(score=0.8)
        r1.id = "r1"

        # Result 2: high score, irrelevant domain → FILTERED_BY_CONTEXT
        r2 = _make_result(score=0.75)
        r2.id = "r2"

        domain_outputs = {
            "r2": _make_agent_output({
                "classification": "IRRELEVANT",
                "relevance_score": 0.1,
                "business_translation": "Not relevant",
            }),
        }

        deltas = await compute_deltas(session, [r1, r2], domain_outputs, "run-1")
        delta_types = {d.delta_type for d in deltas}
        assert "UNEXPLAINED_SIGNAL" in delta_types
        assert "FILTERED_BY_CONTEXT" in delta_types

    @pytest.mark.asyncio
    async def test_tags_results(self):
        """compute_deltas should set result.delta_type."""
        session = AsyncMock()
        session.flush = AsyncMock()

        r = _make_result(score=0.8)
        r.id = "r1"

        deltas = await compute_deltas(session, [r], {}, "run-1")
        assert r.delta_type == "UNEXPLAINED_SIGNAL"

    @pytest.mark.asyncio
    async def test_no_deltas_for_low_scoring(self):
        session = AsyncMock()
        session.flush = AsyncMock()

        r = _make_result(score=0.3)
        r.id = "r1"

        deltas = await compute_deltas(session, [r], {}, "run-1")
        assert len(deltas) == 0

    @pytest.mark.asyncio
    async def test_empty_results(self):
        session = AsyncMock()
        session.flush = AsyncMock()

        deltas = await compute_deltas(session, [], {}, "run-1")
        assert deltas == []
