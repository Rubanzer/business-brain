"""Tests for analysis/track1/scorer.py — scoring + follow-up spawning."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from business_brain.analysis.track1.fingerprinter import ColumnFingerprint, TableFingerprint
from business_brain.analysis.track1.scorer import (
    DEFAULT_WEIGHTS,
    _score_coverage,
    _score_magnitude,
    _score_stability,
    _score_surprise,
    _score_variance,
    score_batch,
    score_result,
    spawn_followups,
)


def _make_result(
    operation: str = "CORRELATE",
    result_data: dict | None = None,
    target: list[str] | None = None,
    segmenters: list[str] | None = None,
    score: float = 0.0,
) -> MagicMock:
    r = MagicMock()
    r.id = "test-id"
    r.run_id = "run-1"
    r.operation_type = operation
    r.table_name = "sales"
    r.result_data = result_data or {}
    r.target = target or ["revenue"]
    r.segmenters = segmenters or []
    r.controls = []
    r.interestingness_score = score
    r.interestingness_breakdown = None
    r.quality_verdict = None
    r.domain_relevance = None
    r.temporal_context = None
    r.delta_type = None
    r.final_score = 0.0
    r.parent_result_id = None
    return r


def _make_fp(measures=None, dimensions=None) -> TableFingerprint:
    measures = measures or ["revenue", "quantity"]
    dimensions = dimensions or ["region", "category", "channel"]
    columns = {}
    for m in measures:
        columns[m] = ColumnFingerprint(name=m, semantic_type="numeric_metric", role="MEASURE",
                                       cardinality=100, null_rate=0.01)
    for d in dimensions:
        columns[d] = ColumnFingerprint(name=d, semantic_type="categorical", role="DIMENSION",
                                       cardinality=10, null_rate=0.01)
    return TableFingerprint(table_name="sales", row_count=1000, data_hash="x",
                           domain_hint="sales", time_index=None,
                           measures=measures, dimensions=dimensions, columns=columns)


# ---------------------------------------------------------------------------
# Sub-score tests
# ---------------------------------------------------------------------------


class TestSurpriseScore:
    def test_anomaly_with_findings(self):
        r = _make_result("DETECT_ANOMALY", {"count": 5, "total": 100})
        assert _score_surprise(r) > 0.5

    def test_anomaly_no_findings(self):
        r = _make_result("DETECT_ANOMALY", {"count": 0, "total": 100})
        assert _score_surprise(r) < 0.3

    def test_high_correlation(self):
        r = _make_result("CORRELATE", {"pearson_r": 0.95})
        assert _score_surprise(r) > 0.7

    def test_rank_significant(self):
        r = _make_result("RANK", {"comparison": {"p_value": 0.001}})
        assert _score_surprise(r) >= 0.7

    def test_describe_skewed(self):
        r = _make_result("DESCRIBE", {"stats": {"skewness": 3.0, "kurtosis": 10.0}})
        assert _score_surprise(r) > 0.5


class TestMagnitudeScore:
    def test_high_correlation(self):
        r = _make_result("CORRELATE", {"pearson_r": 0.9})
        assert _score_magnitude(r) > 0.8

    def test_large_cohens_d(self):
        r = _make_result("RANK", {"comparison": {"cohens_d": 1.2}})
        assert _score_magnitude(r) > 0.8

    def test_anomaly_extreme_z(self):
        r = _make_result("DETECT_ANOMALY", {"anomalies": [{"z_score": 5.0}]})
        assert _score_magnitude(r) == pytest.approx(1.0)


class TestStabilityScore:
    def test_low_p_value(self):
        r = _make_result("CORRELATE", {"pearson_p": 0.001, "n": 100})
        assert _score_stability(r) > 0.9

    def test_high_p_value(self):
        r = _make_result("CORRELATE", {"pearson_p": 0.8, "n": 10})
        assert _score_stability(r) < 0.5


class TestCoverageScore:
    def test_full_coverage(self):
        r = _make_result("DESCRIBE", {"row_count": 100, "stats": {"count": 100}})
        assert _score_coverage(r) > 0.9

    def test_low_coverage(self):
        r = _make_result("DESCRIBE", {"row_count": 100, "stats": {"count": 10}})
        assert _score_coverage(r) < 0.5


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


class TestScoreResult:
    def test_sets_breakdown(self):
        r = _make_result("CORRELATE", {"pearson_r": 0.8, "pearson_p": 0.01, "n": 50})
        score = score_result(r)
        assert r.interestingness_breakdown is not None
        assert "surprise" in r.interestingness_breakdown
        assert "magnitude" in r.interestingness_breakdown
        assert r.interestingness_score == score

    def test_score_in_range(self):
        r = _make_result("DESCRIBE", {"stats": {"mean": 10, "stdev": 2, "count": 50}, "row_count": 50})
        score = score_result(r)
        assert 0.0 <= score <= 1.0


class TestScoreBatch:
    def test_sorts_descending(self):
        results = [
            _make_result("CORRELATE", {"pearson_r": 0.3, "pearson_p": 0.1, "n": 20}),
            _make_result("CORRELATE", {"pearson_r": 0.95, "pearson_p": 0.001, "n": 100}),
            _make_result("CORRELATE", {"pearson_r": 0.6, "pearson_p": 0.01, "n": 50}),
        ]
        scored = score_batch(results)
        assert scored[0].interestingness_score >= scored[1].interestingness_score
        assert scored[1].interestingness_score >= scored[2].interestingness_score

    def test_custom_weights(self):
        r = _make_result("CORRELATE", {"pearson_r": 0.8, "pearson_p": 0.01, "n": 50})
        # All weight on surprise
        score1 = score_result(r, weights={"surprise": 1.0, "magnitude": 0.0, "variance": 0.0, "stability": 0.0, "coverage": 0.0})
        surprise_val = r.interestingness_breakdown["surprise"]
        assert score1 == pytest.approx(surprise_val)


# ---------------------------------------------------------------------------
# Follow-up spawning (Gap #7)
# ---------------------------------------------------------------------------


class TestSpawnFollowups:
    def test_anomaly_spawns_rank(self):
        r = _make_result("DETECT_ANOMALY", {"count": 5, "total": 100}, target=["revenue"], score=0.8)
        r.interestingness_score = 0.8
        fp = _make_fp()
        followups = spawn_followups([r], {"sales": fp})
        rank_followups = [c for c in followups if c.operation == "RANK"]
        assert len(rank_followups) > 0
        # Each should use a different dimension as segmenter
        dims_used = {c.segmenters[0] for c in rank_followups}
        assert len(dims_used) == len(rank_followups)

    def test_correlate_spawns_rank(self):
        r = _make_result("CORRELATE", {"pearson_r": 0.9, "pearson_p": 0.001}, target=["revenue", "cost"], score=0.8)
        r.interestingness_score = 0.8
        fp = _make_fp()
        followups = spawn_followups([r], {"sales": fp})
        assert len(followups) > 0

    def test_segmented_spawns_unsegmented(self):
        r = _make_result("RANK", {}, target=["revenue"], segmenters=["region"], score=0.8)
        r.interestingness_score = 0.8
        fp = _make_fp()
        followups = spawn_followups([r], {"sales": fp})
        unsegmented = [c for c in followups if c.segmenters == []]
        assert len(unsegmented) == 1

    def test_low_score_no_followups(self):
        r = _make_result("DETECT_ANOMALY", {"count": 1, "total": 100}, score=0.3)
        r.interestingness_score = 0.3
        fp = _make_fp()
        followups = spawn_followups([r], {"sales": fp})
        assert len(followups) == 0

    def test_max_followups_per_finding(self):
        r = _make_result("DETECT_ANOMALY", {"count": 10, "total": 100}, score=0.9)
        r.interestingness_score = 0.9
        fp = _make_fp(dimensions=[f"d{i}" for i in range(10)])
        followups = spawn_followups([r], {"sales": fp})
        assert len(followups) <= 3 + 1  # 3 RANK + possibly 1 unsegmented

    def test_max_source_findings(self):
        results = []
        for i in range(10):
            r = _make_result("DETECT_ANOMALY", {"count": 5, "total": 100}, score=0.9)
            r.id = f"r-{i}"
            r.interestingness_score = 0.9
            results.append(r)
        fp = _make_fp()
        followups = spawn_followups(results, {"sales": fp})
        # Max 5 source findings × max 3+1 follow-ups = max ~20
        assert len(followups) <= 5 * 4
