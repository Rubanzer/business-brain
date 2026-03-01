"""Tests for analysis/integration.py — feed bridge + discovery trigger."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from business_brain.analysis.integration import (
    _build_insight_description,
    _build_insight_title,
    _score_to_severity,
    persist_to_feed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    operation: str = "RANK",
    target: list[str] | None = None,
    segmenters: list[str] | None = None,
    result_data: dict | None = None,
    final_score: float = 0.7,
    quality_verdict: str | None = None,
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
    r.interestingness_score = 0.7
    r.interestingness_breakdown = {}
    r.quality_verdict = quality_verdict
    r.domain_relevance = None
    r.temporal_context = None
    r.delta_type = None
    r.final_score = final_score
    r.parent_result_id = None
    r.tier = 1
    return r


# ---------------------------------------------------------------------------
# _score_to_severity
# ---------------------------------------------------------------------------


class TestScoreToSeverity:
    def test_critical(self):
        assert _score_to_severity(0.9) == "critical"
        assert _score_to_severity(0.8) == "critical"

    def test_warning(self):
        assert _score_to_severity(0.7) == "warning"
        assert _score_to_severity(0.6) == "warning"

    def test_info(self):
        assert _score_to_severity(0.5) == "info"
        assert _score_to_severity(0.0) == "info"


# ---------------------------------------------------------------------------
# _build_insight_title
# ---------------------------------------------------------------------------


class TestBuildInsightTitle:
    def test_rank_with_segmenters(self):
        r = _make_result(operation="RANK", target=["output_kg"], segmenters=["shift"])
        title = _build_insight_title(r)
        assert "output_kg" in title
        assert "shift" in title

    def test_rank_without_segmenters(self):
        r = _make_result(operation="RANK", target=["output_kg"], segmenters=[])
        r.segmenters = []  # ensure it's a real empty list, not MagicMock
        title = _build_insight_title(r)
        assert "Ranking" in title

    def test_correlate(self):
        r = _make_result(operation="CORRELATE", target=["temp", "pressure"])
        title = _build_insight_title(r)
        assert "Correlation" in title
        assert "temp, pressure" in title

    def test_detect_anomaly(self):
        r = _make_result(operation="DETECT_ANOMALY", target=["output_kg"])
        title = _build_insight_title(r)
        assert "Anomaly" in title

    def test_describe(self):
        r = _make_result(operation="DESCRIBE", target=["output_kg"])
        title = _build_insight_title(r)
        assert "Distribution" in title

    def test_describe_categorical(self):
        r = _make_result(operation="DESCRIBE_CATEGORICAL", target=["region"])
        title = _build_insight_title(r)
        assert "Categories" in title

    def test_unknown_operation(self):
        r = _make_result(operation="CUSTOM_OP", target=["x"])
        title = _build_insight_title(r)
        assert "CUSTOM_OP" in title


# ---------------------------------------------------------------------------
# _build_insight_description
# ---------------------------------------------------------------------------


class TestBuildInsightDescription:
    def test_correlate_description(self):
        r = _make_result(
            operation="CORRELATE",
            result_data={"pearson_r": 0.85, "pearson_p": 0.001},
        )
        desc = _build_insight_description(r)
        assert "r=0.850" in desc
        assert "p=0.0010" in desc

    def test_rank_with_comparison(self):
        r = _make_result(
            operation="RANK",
            result_data={"comparison": {"p_value": 0.01}, "ranked": [{"shift": "day", "avg": 100}]},
        )
        desc = _build_insight_description(r)
        assert "p=" in desc

    def test_detect_anomaly_description(self):
        r = _make_result(
            operation="DETECT_ANOMALY",
            result_data={"count": 5, "total": 100},
        )
        desc = _build_insight_description(r)
        assert "5 anomalies" in desc
        assert "100 values" in desc

    def test_describe_with_stats(self):
        r = _make_result(
            operation="DESCRIBE",
            result_data={"stats": {"mean": 42.5, "stdev": 8.3}},
        )
        desc = _build_insight_description(r)
        assert "Mean=42.50" in desc
        assert "Stdev=8.30" in desc

    def test_describe_categorical_unique(self):
        r = _make_result(
            operation="DESCRIBE_CATEGORICAL",
            result_data={"stats": {"unique": 15}},
        )
        desc = _build_insight_description(r)
        assert "15 unique" in desc

    def test_segmenters_shown(self):
        r = _make_result(segmenters=["shift", "furnace"])
        desc = _build_insight_description(r)
        assert "shift" in desc
        assert "furnace" in desc

    def test_fallback_shows_score(self):
        r = _make_result(operation="UNKNOWN", result_data={}, segmenters=[])
        r.segmenters = []  # ensure it's a real empty list, not MagicMock
        desc = _build_insight_description(r)
        assert "Interestingness" in desc


# ---------------------------------------------------------------------------
# persist_to_feed
# ---------------------------------------------------------------------------


class TestPersistToFeed:
    @pytest.mark.asyncio
    async def test_creates_insights_for_high_scores(self):
        session = AsyncMock()
        session.flush = AsyncMock()

        results = [
            _make_result(final_score=0.8),
            _make_result(final_score=0.6),
        ]

        insights = await persist_to_feed(session, results, "run-1", min_score=0.4)
        assert len(insights) == 2
        assert session.add.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_low_scores(self):
        session = AsyncMock()
        session.flush = AsyncMock()

        results = [
            _make_result(final_score=0.3),
            _make_result(final_score=0.2),
        ]

        insights = await persist_to_feed(session, results, "run-1", min_score=0.4)
        assert len(insights) == 0

    @pytest.mark.asyncio
    async def test_skips_unreliable(self):
        session = AsyncMock()
        session.flush = AsyncMock()

        r = _make_result(final_score=0.9, quality_verdict="UNRELIABLE")
        insights = await persist_to_feed(session, [r], "run-1")
        assert len(insights) == 0

    @pytest.mark.asyncio
    async def test_empty_results(self):
        session = AsyncMock()
        insights = await persist_to_feed(session, [], "run-1")
        assert insights == []

    @pytest.mark.asyncio
    async def test_insight_fields(self):
        session = AsyncMock()
        session.flush = AsyncMock()

        r = _make_result(operation="CORRELATE", final_score=0.85)
        insights = await persist_to_feed(session, [r], "run-1")
        assert len(insights) == 1

        insight = insights[0]
        assert insight.insight_type == "correlation"
        assert insight.severity == "critical"
        assert insight.impact_score == 85
        assert insight.source_tables == ["production"]
        assert insight.discovery_run_id == "run-1"
        assert insight.evidence["operation"] == "CORRELATE"
