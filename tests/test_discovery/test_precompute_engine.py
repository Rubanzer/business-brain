"""Tests for the pre-compute engine — background analysis intelligence."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from business_brain.discovery.precompute_engine import (
    generate_candidates,
    _precompute_benchmark,
    _precompute_correlation,
    _precompute_anomaly,
    _precompute_trend,
    _safe,
)
from business_brain.discovery.insight_recommender import (
    Recommendation,
    _enrich_with_precomputed,
)


# ---------------------------------------------------------------------------
# Helpers — mock TableProfile objects
# ---------------------------------------------------------------------------


def _mock_profile(
    table_name="test_table",
    row_count=500,
    columns=None,
    domain_hint="general",
    data_hash="abc123",
):
    """Create a mock TableProfile for testing."""
    if columns is None:
        columns = {
            "supplier": {
                "semantic_type": "categorical",
                "cardinality": 10,
                "null_count": 5,
                "sample_values": ["A", "B", "C", "D", "E"],
                "stats": {},
            },
            "rate": {
                "semantic_type": "numeric_metric",
                "cardinality": 200,
                "null_count": 0,
                "sample_values": [100, 200, 300],
                "stats": {"mean": 200.0, "stdev": 80.0, "min": 50, "max": 400},
            },
            "amount": {
                "semantic_type": "numeric_metric",
                "cardinality": 150,
                "null_count": 10,
                "sample_values": [1000, 2000, 3000],
                "stats": {"mean": 2000.0, "stdev": 500.0, "min": 100, "max": 5000},
            },
            "order_date": {
                "semantic_type": "temporal",
                "cardinality": 100,
                "null_count": 0,
                "sample_values": ["2024-01-01", "2024-02-01"],
            },
        }
    profile = MagicMock()
    profile.table_name = table_name
    profile.row_count = row_count
    profile.column_classification = {
        "columns": columns,
        "domain_hint": domain_hint,
    }
    profile.data_hash = data_hash
    return profile


# ---------------------------------------------------------------------------
# TestCandidateGeneration
# ---------------------------------------------------------------------------


class TestCandidateGeneration:
    """Tests for generate_candidates()."""

    def test_generates_benchmark_candidates(self):
        """Table with cat + num columns produces benchmark candidates."""
        profile = _mock_profile()
        candidates = generate_candidates([profile])
        bench = [c for c in candidates if c["analysis_type"] == "benchmark"]
        assert len(bench) > 0
        # Should use supplier (cat) and rate or amount (num)
        for c in bench:
            assert len(c["columns"]) == 2
            assert c["columns"][0] == "supplier"  # Only categorical
            assert c["columns"][1] in ("rate", "amount")
            assert c["priority_score"] > 0

    def test_generates_correlation_candidates(self):
        """Table with 2+ numeric columns produces correlation candidates."""
        profile = _mock_profile()
        candidates = generate_candidates([profile])
        corr = [c for c in candidates if c["analysis_type"] == "correlation"]
        assert len(corr) > 0
        for c in corr:
            assert len(c["columns"]) == 2
            assert all(col in ("rate", "amount") for col in c["columns"])

    def test_generates_anomaly_candidates(self):
        """Table with numeric columns produces anomaly candidates."""
        profile = _mock_profile()
        candidates = generate_candidates([profile])
        anom = [c for c in candidates if c["analysis_type"] == "anomaly"]
        assert len(anom) > 0
        for c in anom:
            assert len(c["columns"]) == 1
            assert c["columns"][0] in ("rate", "amount")

    def test_generates_trend_candidates(self):
        """Table with temporal + numeric columns produces trend candidates."""
        profile = _mock_profile()
        candidates = generate_candidates([profile])
        trend = [c for c in candidates if c["analysis_type"] == "trend"]
        assert len(trend) > 0
        for c in trend:
            assert len(c["columns"]) == 2
            assert c["columns"][0] == "order_date"

    def test_candidate_ranking_respects_column_scores(self):
        """High-CV columns rank higher in candidates."""
        columns = {
            "supplier": {
                "semantic_type": "categorical",
                "cardinality": 10,
                "null_count": 0,
                "sample_values": ["A", "B", "C"],
            },
            "boring_metric": {
                "semantic_type": "numeric_metric",
                "cardinality": 5,
                "null_count": 0,
                "sample_values": [100, 101, 102],
                "stats": {"mean": 100.5, "stdev": 0.5, "min": 100, "max": 102},
            },
            "interesting_metric": {
                "semantic_type": "numeric_metric",
                "cardinality": 200,
                "null_count": 0,
                "sample_values": [50, 300, 150],
                "stats": {"mean": 175.0, "stdev": 120.0, "min": 10, "max": 500},
            },
        }
        profile = _mock_profile(columns=columns)
        candidates = generate_candidates([profile])
        bench = [c for c in candidates if c["analysis_type"] == "benchmark"]

        # interesting_metric should appear first (higher CV)
        assert len(bench) >= 1
        first_num = bench[0]["columns"][1]
        assert first_num == "interesting_metric"

    def test_max_per_table_respected(self):
        """Candidate count per table doesn't exceed max_per_table."""
        profile = _mock_profile()
        candidates = generate_candidates([profile], max_per_table=3)
        table_counts = {}
        for c in candidates:
            table_counts[c["table_name"]] = table_counts.get(c["table_name"], 0) + 1
        for count in table_counts.values():
            assert count <= 3

    def test_no_columns_no_crash(self):
        """Profile with no columns produces no candidates."""
        profile = _mock_profile(columns={})
        candidates = generate_candidates([profile])
        assert candidates == []

    def test_only_numeric_no_benchmark(self):
        """Table with only numeric columns generates no benchmark candidates."""
        columns = {
            "col_a": {
                "semantic_type": "numeric_metric",
                "cardinality": 100,
                "null_count": 0,
                "stats": {"mean": 50, "stdev": 10, "min": 0, "max": 100},
            },
            "col_b": {
                "semantic_type": "numeric_metric",
                "cardinality": 80,
                "null_count": 0,
                "stats": {"mean": 30, "stdev": 15, "min": 0, "max": 80},
            },
        }
        profile = _mock_profile(columns=columns)
        candidates = generate_candidates([profile])
        bench = [c for c in candidates if c["analysis_type"] == "benchmark"]
        assert len(bench) == 0
        # But should have correlation candidates
        corr = [c for c in candidates if c["analysis_type"] == "correlation"]
        assert len(corr) > 0


# ---------------------------------------------------------------------------
# TestSQLPrecomputation (mocked DB)
# ---------------------------------------------------------------------------


class TestSQLPrecomputation:
    """Tests for the individual SQL precomputation functions."""

    @pytest.mark.asyncio
    async def test_benchmark_computes_group_stats(self):
        """Benchmark pre-computation returns correct spread and groups."""
        mock_rows = [
            {"supplier": "A", "cnt": 50, "avg_val": 300.0},
            {"supplier": "B", "cnt": 30, "avg_val": 200.0},
            {"supplier": "C", "cnt": 20, "avg_val": 150.0},
        ]
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(_mapping=r) for r in mock_rows
        ]
        session.execute = AsyncMock(return_value=mock_result)

        summary, detail, quality = await _precompute_benchmark(
            session, "orders", "supplier", "rate"
        )

        assert summary["group_count"] == 3
        assert summary["top_group"] == "A"
        assert summary["top_value"] == 300.0
        assert summary["bottom_group"] == "C"
        assert summary["bottom_value"] == 150.0
        assert summary["spread_pct"] == 100.0  # (300-150)/150 * 100
        assert quality > 0.5  # High spread + 3 groups → interesting

    @pytest.mark.asyncio
    async def test_benchmark_empty_result(self):
        """Empty GROUP BY result returns empty summary."""
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=mock_result)

        summary, detail, quality = await _precompute_benchmark(
            session, "orders", "supplier", "rate"
        )
        assert summary == {}
        assert quality == 0.0

    @pytest.mark.asyncio
    async def test_correlation_computes_r_value(self):
        """Correlation pre-computation returns correct r-value."""
        session = AsyncMock()

        # First call: CORR() query
        corr_result = MagicMock()
        corr_row = MagicMock()
        corr_row.r_val = 0.83
        corr_row.sample_size = 450
        corr_result.fetchone.return_value = corr_row

        # Second call: scatter sample
        scatter_result = MagicMock()
        scatter_result.fetchall.return_value = [
            MagicMock(x=1.0, y=2.0),
            MagicMock(x=3.0, y=4.0),
        ]

        session.execute = AsyncMock(side_effect=[corr_result, scatter_result])

        summary, detail, quality = await _precompute_correlation(
            session, "orders", "rate", "amount"
        )

        assert summary["r_value"] == 0.83
        assert summary["direction"] == "positive"
        assert summary["sample_size"] == 450
        assert summary["p_approx"] == "significant"
        assert quality >= 0.8  # |r| >= 0.8 → high quality

    @pytest.mark.asyncio
    async def test_correlation_null_r_value(self):
        """Null correlation result returns empty summary."""
        session = AsyncMock()
        corr_result = MagicMock()
        corr_row = MagicMock()
        corr_row.r_val = None
        corr_row.sample_size = 0
        corr_result.fetchone.return_value = corr_row
        session.execute = AsyncMock(return_value=corr_result)

        summary, detail, quality = await _precompute_correlation(
            session, "orders", "rate", "amount"
        )
        assert summary == {}
        assert quality == 0.0

    @pytest.mark.asyncio
    async def test_anomaly_detects_outliers(self):
        """Anomaly pre-computation finds known outliers."""
        session = AsyncMock()

        # Stats query result
        stats_result = MagicMock()
        stats_row = MagicMock()
        stats_row.mean_val = 100.0
        stats_row.stdev_val = 10.0
        stats_row.cnt = 500
        stats_result.fetchone.return_value = stats_row

        # Outlier query result
        outlier_result = MagicMock()
        outlier_result.fetchall.return_value = [
            MagicMock(val=180.0),  # z = 8
            MagicMock(val=160.0),  # z = 6
        ]

        session.execute = AsyncMock(side_effect=[stats_result, outlier_result])

        summary, detail, quality = await _precompute_anomaly(
            session, "orders", "rate"
        )

        assert summary["outlier_count"] == 2
        assert summary["max_z_score"] == 8.0
        assert summary["most_anomalous_value"] == 180.0
        assert summary["mean"] == 100.0
        assert summary["stdev"] == 10.0
        assert quality >= 0.5  # 2 outliers → interesting

    @pytest.mark.asyncio
    async def test_anomaly_no_outliers(self):
        """No outliers found → low quality score."""
        session = AsyncMock()

        stats_result = MagicMock()
        stats_row = MagicMock()
        stats_row.mean_val = 100.0
        stats_row.stdev_val = 10.0
        stats_row.cnt = 500
        stats_result.fetchone.return_value = stats_row

        outlier_result = MagicMock()
        outlier_result.fetchall.return_value = []

        session.execute = AsyncMock(side_effect=[stats_result, outlier_result])

        summary, detail, quality = await _precompute_anomaly(
            session, "orders", "rate"
        )

        assert summary["outlier_count"] == 0
        assert quality <= 0.2

    @pytest.mark.asyncio
    async def test_trend_detects_direction(self):
        """Trend pre-computation detects increasing direction."""
        session = AsyncMock()

        # Time series: clearly increasing
        rows = [
            MagicMock(period="2024-01", avg_val=100.0, cnt=30),
            MagicMock(period="2024-02", avg_val=120.0, cnt=28),
            MagicMock(period="2024-03", avg_val=140.0, cnt=32),
            MagicMock(period="2024-04", avg_val=160.0, cnt=29),
            MagicMock(period="2024-05", avg_val=180.0, cnt=31),
        ]
        mock_result = MagicMock()
        mock_result.fetchall.return_value = rows
        session.execute = AsyncMock(return_value=mock_result)

        summary, detail, quality = await _precompute_trend(
            session, "orders", "order_date", "rate"
        )

        assert summary["direction"] == "increasing"
        assert summary["r_squared"] > 0.9  # Perfect linear trend
        assert summary["periods"] == 5
        assert quality > 0.5  # Strong trend → interesting

    @pytest.mark.asyncio
    async def test_trend_too_few_periods(self):
        """Fewer than 3 periods → empty result."""
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(period="2024-01", avg_val=100.0, cnt=30),
            MagicMock(period="2024-02", avg_val=120.0, cnt=28),
        ]
        session.execute = AsyncMock(return_value=mock_result)

        summary, detail, quality = await _precompute_trend(
            session, "orders", "order_date", "rate"
        )
        assert summary == {}
        assert quality == 0.0


# ---------------------------------------------------------------------------
# TestOrchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator:
    """Tests for run_precomputation orchestrator."""

    def test_max_total_limits_candidates(self):
        """generate_candidates respects global max when called by orchestrator."""
        # Create profiles that would generate many candidates
        profiles = [
            _mock_profile(table_name=f"table_{i}")
            for i in range(5)
        ]
        candidates = generate_candidates(profiles, max_per_table=8)
        # With 5 tables × 8 per table, could be up to 40
        # Orchestrator truncates to max_total, but generate_candidates
        # just caps per-table
        for table_name in [f"table_{i}" for i in range(5)]:
            table_cands = [c for c in candidates if c["table_name"] == table_name]
            assert len(table_cands) <= 8

    def test_candidates_sorted_by_priority(self):
        """Candidates returned in descending priority order."""
        profile = _mock_profile()
        candidates = generate_candidates([profile])
        for i in range(len(candidates) - 1):
            assert candidates[i]["priority_score"] >= candidates[i + 1]["priority_score"]

    def test_candidate_has_required_fields(self):
        """Each candidate has all required fields."""
        profile = _mock_profile()
        candidates = generate_candidates([profile])
        required = {"table_name", "analysis_type", "columns", "column_scores",
                     "priority_score", "data_hash"}
        for c in candidates:
            assert required.issubset(c.keys()), f"Missing fields: {required - c.keys()}"


# ---------------------------------------------------------------------------
# TestStaleness
# ---------------------------------------------------------------------------


class TestStaleness:
    """Tests for staleness management."""

    def test_safe_name_sanitization(self):
        """_safe removes special characters."""
        assert _safe("normal_table") == "normal_table"
        assert _safe("table; DROP TABLE--") == "tableDROPTABLE"
        assert _safe("my-table.name") == "mytablename"


# ---------------------------------------------------------------------------
# TestRecommendationEnrichment
# ---------------------------------------------------------------------------


class TestRecommendationEnrichment:
    """Tests for _enrich_with_precomputed."""

    def test_matching_precomputed_sets_confidence(self):
        """Recommendation matched to completed pre-computation gets pre-computed confidence."""
        recs = [
            Recommendation(
                title="Benchmark Rate by Supplier",
                description="",
                analysis_type="benchmark",
                target_table="orders",
                columns=["supplier", "rate"],
                priority=70,
                reason="",
            )
        ]
        precomputed = [
            {
                "table_name": "orders",
                "analysis_type": "benchmark",
                "columns": ["supplier", "rate"],
                "status": "completed",
                "result_summary": {"spread_pct": 45.0, "top_group": "A"},
                "quality_score": 0.8,
                "precomputed_id": "pc-123",
            }
        ]

        enriched = _enrich_with_precomputed(recs, precomputed)
        assert enriched[0].confidence == "pre-computed"
        assert enriched[0].precomputed_summary == {"spread_pct": 45.0, "top_group": "A"}
        assert enriched[0].precomputed_id == "pc-123"

    def test_high_quality_boosts_priority(self):
        """Pre-computed result with quality > 0.5 boosts priority by 15."""
        recs = [
            Recommendation(
                title="Benchmark",
                description="",
                analysis_type="benchmark",
                target_table="orders",
                columns=["supplier", "rate"],
                priority=70,
                reason="",
            )
        ]
        precomputed = [
            {
                "table_name": "orders",
                "analysis_type": "benchmark",
                "columns": ["supplier", "rate"],
                "status": "completed",
                "result_summary": {},
                "quality_score": 0.8,
                "precomputed_id": "pc-123",
            }
        ]

        enriched = _enrich_with_precomputed(recs, precomputed)
        assert enriched[0].priority == 85  # 70 + 15

    def test_low_quality_deprioritizes(self):
        """Pre-computed result with quality < 0.2 deprioritizes by 10."""
        recs = [
            Recommendation(
                title="Benchmark",
                description="",
                analysis_type="benchmark",
                target_table="orders",
                columns=["supplier", "rate"],
                priority=70,
                reason="",
            )
        ]
        precomputed = [
            {
                "table_name": "orders",
                "analysis_type": "benchmark",
                "columns": ["supplier", "rate"],
                "status": "completed",
                "result_summary": {},
                "quality_score": 0.1,
                "precomputed_id": "pc-123",
            }
        ]

        enriched = _enrich_with_precomputed(recs, precomputed)
        assert enriched[0].priority == 60  # 70 - 10

    def test_failed_precomputation_deprioritizes(self):
        """Failed pre-computation deprioritizes by 20."""
        recs = [
            Recommendation(
                title="Benchmark",
                description="",
                analysis_type="benchmark",
                target_table="orders",
                columns=["supplier", "rate"],
                priority=70,
                reason="",
            )
        ]
        precomputed = [
            {
                "table_name": "orders",
                "analysis_type": "benchmark",
                "columns": ["supplier", "rate"],
                "status": "failed",
                "result_summary": None,
                "quality_score": 0.0,
                "precomputed_id": "pc-123",
            }
        ]

        enriched = _enrich_with_precomputed(recs, precomputed)
        assert enriched[0].priority == 50  # 70 - 20

    def test_no_match_leaves_heuristic(self):
        """Unmatched recommendation stays heuristic."""
        recs = [
            Recommendation(
                title="Benchmark",
                description="",
                analysis_type="benchmark",
                target_table="orders",
                columns=["supplier", "rate"],
                priority=70,
                reason="",
            )
        ]
        precomputed = [
            {
                "table_name": "other_table",
                "analysis_type": "benchmark",
                "columns": ["a", "b"],
                "status": "completed",
                "result_summary": {},
                "quality_score": 0.9,
                "precomputed_id": "pc-456",
            }
        ]

        enriched = _enrich_with_precomputed(recs, precomputed)
        assert enriched[0].confidence == "heuristic"
        assert enriched[0].priority == 70

    def test_time_trend_matches_trend_type(self):
        """Recommendation with analysis_type='time_trend' matches precomputed 'trend'."""
        recs = [
            Recommendation(
                title="Trend",
                description="",
                analysis_type="time_trend",
                target_table="orders",
                columns=["order_date", "rate"],
                priority=65,
                reason="",
            )
        ]
        precomputed = [
            {
                "table_name": "orders",
                "analysis_type": "trend",
                "columns": ["order_date", "rate"],
                "status": "completed",
                "result_summary": {"direction": "increasing"},
                "quality_score": 0.7,
                "precomputed_id": "pc-789",
            }
        ]

        enriched = _enrich_with_precomputed(recs, precomputed)
        assert enriched[0].confidence == "pre-computed"
        assert enriched[0].priority == 80  # 65 + 15


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure pre-computed changes don't break existing behavior."""

    def test_recommend_without_precomputed_works(self):
        """recommend_analyses() works fine with precomputed=None (default)."""
        from business_brain.discovery.insight_recommender import recommend_analyses

        profiles = [
            {
                "table_name": "orders",
                "row_count": 100,
                "column_classification": {
                    "columns": {
                        "supplier": {"semantic_type": "categorical", "cardinality": 5},
                        "rate": {"semantic_type": "numeric_metric", "cardinality": 80,
                                 "stats": {"mean": 100, "stdev": 30, "min": 10, "max": 200}},
                    }
                },
            }
        ]
        recs = recommend_analyses(profiles, [], [])
        assert isinstance(recs, list)
        for r in recs:
            assert r.confidence == "heuristic"
            assert r.precomputed_summary is None

    def test_enrich_with_empty_precomputed(self):
        """_enrich_with_precomputed with empty list doesn't change anything."""
        recs = [
            Recommendation(
                title="Test",
                description="",
                analysis_type="benchmark",
                target_table="t",
                columns=["a", "b"],
                priority=50,
                reason="",
            )
        ]
        enriched = _enrich_with_precomputed(recs, [])
        assert enriched[0].confidence == "heuristic"
        assert enriched[0].priority == 50

    def test_recommendation_dataclass_defaults(self):
        """New Recommendation fields have correct defaults."""
        rec = Recommendation(
            title="Test",
            description="desc",
            analysis_type="benchmark",
            target_table="t",
            columns=["a"],
            priority=50,
            reason="reason",
        )
        assert rec.precomputed_id is None
        assert rec.precomputed_summary is None
        assert rec.confidence == "heuristic"
