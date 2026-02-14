"""Tests for insight recommender module."""

from business_brain.discovery.insight_recommender import (
    Recommendation,
    compute_coverage,
    recommend_analyses,
)


def _profile(name, row_count=100, columns=None):
    """Helper to create a profile dict."""
    return {
        "table_name": name,
        "row_count": row_count,
        "column_classification": {"columns": columns or {}},
    }


# ---------------------------------------------------------------------------
# recommend_analyses
# ---------------------------------------------------------------------------


class TestRecommendAnalyses:
    def test_empty_input(self):
        assert recommend_analyses([], [], []) == []

    def test_tiny_table_skipped(self):
        profiles = [_profile("tiny", row_count=5)]
        recs = recommend_analyses(profiles, [], [])
        assert all(r.target_table != "tiny" for r in recs)

    def test_temporal_numeric_suggests_time_trend(self):
        cols = {
            "date": {"semantic_type": "temporal"},
            "revenue": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("sales", 200, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "time_trend" for r in recs)

    def test_categorical_numeric_suggests_benchmark(self):
        cols = {
            "dept": {"semantic_type": "categorical"},
            "score": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("employees", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "benchmark" for r in recs)

    def test_multiple_numerics_suggest_correlation(self):
        cols = {
            "a": {"semantic_type": "numeric_metric"},
            "b": {"semantic_type": "numeric_metric"},
            "c": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("metrics", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "correlation" for r in recs)

    def test_numeric_suggests_anomaly(self):
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profiles = [_profile("data", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "anomaly" for r in recs)

    def test_cohort_analysis_suggested(self):
        cols = {
            "supplier": {"semantic_type": "categorical"},
            "month": {"semantic_type": "temporal"},
            "cost": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("purchases", 200, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "cohort" for r in recs)

    def test_forecast_suggested_with_enough_data(self):
        cols = {
            "date": {"semantic_type": "temporal"},
            "val": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("time_series", 50, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "forecast" for r in recs)

    def test_existing_insights_reduce_priority(self):
        cols = {
            "val": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("data", 100, cols)]
        # No insights
        recs_empty = recommend_analyses(profiles, [], [])
        # With existing anomaly insight
        insights = [{"insight_type": "anomaly", "source_tables": ["data"]}] * 10
        recs_full = recommend_analyses(profiles, insights, [])
        # Should have fewer anomaly recommendations when already covered
        anomaly_empty = [r for r in recs_empty if r.analysis_type == "anomaly"]
        anomaly_full = [r for r in recs_full if r.analysis_type == "anomaly"]
        assert len(anomaly_full) <= len(anomaly_empty)

    def test_max_recommendations(self):
        cols = {
            "cat": {"semantic_type": "categorical"},
            "date": {"semantic_type": "temporal"},
            "num1": {"semantic_type": "numeric_metric"},
            "num2": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile(f"table_{i}", 100, cols) for i in range(10)]
        recs = recommend_analyses(profiles, [], [], max_recommendations=5)
        assert len(recs) <= 5

    def test_cross_table_recommendation(self):
        profiles = [_profile("A", 100), _profile("B", 100)]
        relationships = [
            {"table_a": "A", "table_b": "B"},
            {"table_a": "A", "table_b": "C"},
        ]
        recs = recommend_analyses(profiles, [], relationships)
        # Should suggest cross-table analysis for A (has 2 relationships)
        assert any(r.target_table == "A" for r in recs)

    def test_deduplication(self):
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profiles = [_profile("data", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        # Should not have duplicate (table, analysis_type) pairs
        keys = [(r.target_table, r.analysis_type) for r in recs]
        assert len(keys) == len(set(keys))

    def test_recommendation_fields(self):
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profiles = [_profile("data", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        if recs:
            r = recs[0]
            assert r.title
            assert r.description
            assert r.analysis_type
            assert r.target_table == "data"
            assert 1 <= r.priority <= 100


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_full_coverage(self):
        profiles = [_profile("A"), _profile("B")]
        insights = [
            {"source_tables": ["A"]},
            {"source_tables": ["B"]},
        ]
        cov = compute_coverage(profiles, insights)
        assert cov["coverage_pct"] == 100.0
        assert cov["uncovered_tables"] == []

    def test_partial_coverage(self):
        profiles = [_profile("A"), _profile("B"), _profile("C")]
        insights = [{"source_tables": ["A"]}]
        cov = compute_coverage(profiles, insights)
        assert cov["covered_tables"] == 1
        assert len(cov["uncovered_tables"]) == 2

    def test_no_insights(self):
        profiles = [_profile("A")]
        cov = compute_coverage(profiles, [])
        assert cov["coverage_pct"] == 0.0

    def test_empty(self):
        cov = compute_coverage([], [])
        assert cov["total_tables"] == 0
        assert cov["coverage_pct"] == 0.0
