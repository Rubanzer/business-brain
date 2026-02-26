"""Tests for correlation discoverer pure functions."""

from business_brain.discovery.correlation_discoverer import (
    _estimate_sample_correlations,
    _make_opportunity_insight,
    _parse_samples,
    _quick_pearson,
    discover_correlations_from_profiles,
)


class _Prof:
    def __init__(self, table_name, row_count=100, columns=None):
        self.table_name = table_name
        self.row_count = row_count
        self.column_classification = {"columns": columns or {}}


class TestParseSamples:
    def test_numeric_strings(self):
        assert _parse_samples(["1", "2.5", "3"]) == [1.0, 2.5, 3.0]

    def test_comma_numbers(self):
        assert _parse_samples(["1,000", "2,500"]) == [1000.0, 2500.0]

    def test_non_numeric_skipped(self):
        assert _parse_samples(["abc", "1", None, "2"]) == [1.0, 2.0]

    def test_empty(self):
        assert _parse_samples([]) == []


class TestQuickPearson:
    def test_perfect_positive(self):
        r = _quick_pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r is not None
        assert abs(r - 1.0) < 0.001

    def test_perfect_negative(self):
        r = _quick_pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r is not None
        assert abs(r - (-1.0)) < 0.001

    def test_constant_returns_none(self):
        assert _quick_pearson([5, 5, 5], [1, 2, 3]) is None

    def test_too_few_values(self):
        assert _quick_pearson([1, 2], [3, 4]) is None


class TestEstimateSampleCorrelations:
    def test_correlated_pair(self):
        cols = [
            ("a", {"sample_values": [1, 2, 3, 4, 5, 6, 7]}),
            ("b", {"sample_values": [2, 4, 6, 8, 10, 12, 14]}),
        ]
        results = _estimate_sample_correlations(cols)
        assert len(results) == 1
        assert results[0][0] == "a"
        assert results[0][1] == "b"
        assert results[0][3] == "positive"

    def test_uncorrelated_pair(self):
        cols = [
            ("a", {"sample_values": [1, 5, 2, 4, 3, 6, 7]}),
            ("b", {"sample_values": [3, 1, 4, 2, 5, 1, 3]}),
        ]
        results = _estimate_sample_correlations(cols)
        # Weak correlation should not be included (threshold raised to |r| >= 0.7)
        for r in results:
            assert abs(r[2]) >= 0.7

    def test_too_few_samples(self):
        cols = [
            ("a", {"sample_values": [1, 2]}),
            ("b", {"sample_values": [3, 4]}),
        ]
        assert _estimate_sample_correlations(cols) == []

    def test_missing_samples(self):
        cols = [
            ("a", {"sample_values": [1, 2, 3, 4, 5]}),
            ("b", {}),
        ]
        assert _estimate_sample_correlations(cols) == []


class TestMakeOpportunityInsight:
    def test_basic(self):
        prof = _Prof("test_table")
        cols = [("col1", {}), ("col2", {}), ("col3", {})]
        insight = _make_opportunity_insight(prof, cols)
        assert insight.insight_type == "correlation"
        assert insight.severity == "info"
        assert "test_table" in insight.title
        assert "3 numeric columns" in insight.description


class TestDiscoverCorrelationsFromProfiles:
    def test_empty_profiles(self):
        assert discover_correlations_from_profiles([]) == []

    def test_no_numeric_columns(self):
        p = _Prof("t1", columns={
            "name": {"semantic_type": "categorical"},
            "id": {"semantic_type": "identifier"},
        })
        assert discover_correlations_from_profiles([p]) == []

    def test_single_numeric_column(self):
        p = _Prof("t1", columns={
            "val": {"semantic_type": "numeric_metric", "stats": {"stdev": 5}},
        })
        assert discover_correlations_from_profiles([p]) == []

    def test_two_numeric_no_samples(self):
        """Two numeric cols without samples → no insight (opportunity insights suppressed)."""
        p = _Prof("t1", columns={
            "a": {"semantic_type": "numeric_metric", "stats": {"stdev": 5}},
            "b": {"semantic_type": "numeric_metric", "stats": {"stdev": 10}},
        })
        insights = discover_correlations_from_profiles([p])
        assert len(insights) == 0  # Opportunity meta-insights suppressed

    def test_correlated_samples_detected(self):
        p = _Prof("t1", columns={
            "a": {
                "semantic_type": "numeric_metric",
                "stats": {"stdev": 5},
                "sample_values": [1, 2, 3, 4, 5, 6, 7, 8],
            },
            "b": {
                "semantic_type": "numeric_metric",
                "stats": {"stdev": 10},
                "sample_values": [2, 4, 6, 8, 10, 12, 14, 16],
            },
        })
        insights = discover_correlations_from_profiles([p])
        assert len(insights) >= 1
        corr_insights = [i for i in insights if "Potential correlation" in i.title]
        assert len(corr_insights) == 1
        assert "positive" in corr_insights[0].description.lower()

    def test_zero_stdev_skipped(self):
        p = _Prof("t1", columns={
            "a": {"semantic_type": "numeric_metric", "stats": {"stdev": 0}},
            "b": {"semantic_type": "numeric_metric", "stats": {"stdev": 10}},
        })
        assert discover_correlations_from_profiles([p]) == []

    def test_too_few_rows(self):
        p = _Prof("t1", row_count=5, columns={
            "a": {"semantic_type": "numeric_metric", "stats": {"stdev": 5}},
            "b": {"semantic_type": "numeric_metric", "stats": {"stdev": 10}},
        })
        assert discover_correlations_from_profiles([p]) == []

    def test_no_classification(self):
        p = _Prof("t1")
        p.column_classification = None
        assert discover_correlations_from_profiles([p]) == []

    def test_multiple_profiles_no_samples(self):
        """Multiple profiles without samples → no insights (opportunity suppressed)."""
        p1 = _Prof("t1", columns={
            "x": {"semantic_type": "numeric_metric", "stats": {"stdev": 5}},
            "y": {"semantic_type": "numeric_metric", "stats": {"stdev": 10}},
        })
        p2 = _Prof("t2", columns={
            "a": {"semantic_type": "numeric_currency", "stats": {"stdev": 100}},
            "b": {"semantic_type": "numeric_currency", "stats": {"stdev": 200}},
        })
        insights = discover_correlations_from_profiles([p1, p2])
        assert len(insights) == 0  # Opportunity meta-insights suppressed

    def test_multiple_profiles_with_correlations(self):
        """Multiple profiles WITH correlated samples → actual correlation insights."""
        p1 = _Prof("t1", columns={
            "x": {"semantic_type": "numeric_metric", "stats": {"stdev": 5},
                   "sample_values": [1, 2, 3, 4, 5, 6, 7, 8]},
            "y": {"semantic_type": "numeric_metric", "stats": {"stdev": 10},
                   "sample_values": [2, 4, 6, 8, 10, 12, 14, 16]},
        })
        p2 = _Prof("t2", columns={
            "a": {"semantic_type": "numeric_currency", "stats": {"stdev": 100},
                   "sample_values": [100, 200, 300, 400, 500, 600, 700, 800]},
            "b": {"semantic_type": "numeric_currency", "stats": {"stdev": 200},
                   "sample_values": [200, 400, 600, 800, 1000, 1200, 1400, 1600]},
        })
        insights = discover_correlations_from_profiles([p1, p2])
        assert len(insights) == 2  # One correlation per table
        tables = {i.source_tables[0] for i in insights}
        assert "t1" in tables
        assert "t2" in tables

    def test_all_insights_have_ids(self):
        """Requires sample values to produce actual correlation insights."""
        p = _Prof("t1", columns={
            "a": {"semantic_type": "numeric_metric", "stats": {"stdev": 5},
                   "sample_values": [1, 2, 3, 4, 5, 6, 7, 8]},
            "b": {"semantic_type": "numeric_metric", "stats": {"stdev": 10},
                   "sample_values": [2, 4, 6, 8, 10, 12, 14, 16]},
        })
        insights = discover_correlations_from_profiles([p])
        assert len(insights) >= 1
        for insight in insights:
            assert insight.id is not None
            assert insight.insight_type == "correlation"
