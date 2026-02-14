"""Tests for dashboard summary aggregator."""

from business_brain.discovery.dashboard_summary import (
    _compute_table_quality,
    compute_dashboard_summary,
)


class _Prof:
    def __init__(self, table_name, row_count=100, columns=None, domain="general", data_hash=None):
        self.table_name = table_name
        self.row_count = row_count
        self.domain_hint = domain
        self.data_hash = data_hash
        self.column_classification = {"columns": columns or {}} if columns is not None else None


class _Insight:
    def __init__(self, insight_type="anomaly", severity="info"):
        self.insight_type = insight_type
        self.severity = severity


class _Report:
    pass


class _Run:
    def __init__(self, status="completed", completed_at="2024-01-15T10:00:00"):
        self.status = status
        self.completed_at = completed_at


class TestComputeTableQuality:
    def test_perfect_quality(self):
        cols = {
            "id": {"semantic_type": "identifier", "cardinality": 100, "null_count": 0},
            "name": {"semantic_type": "categorical", "cardinality": 50, "null_count": 0},
            "amount": {"semantic_type": "numeric_currency", "cardinality": 80, "null_count": 0, "stats": {"min": 0, "max": 1000}},
            "date": {"semantic_type": "temporal", "cardinality": 90, "null_count": 0},
            "pct": {"semantic_type": "numeric_percentage", "cardinality": 20, "null_count": 0, "stats": {"min": 0, "max": 100}},
        }
        score = _compute_table_quality(cols, 100)
        assert score > 70

    def test_high_nulls_lower_quality(self):
        cols = {
            "id": {"semantic_type": "identifier", "cardinality": 100, "null_count": 50},
            "name": {"semantic_type": "categorical", "cardinality": 10, "null_count": 80},
        }
        score = _compute_table_quality(cols, 100)
        assert score < 70

    def test_empty_columns(self):
        assert _compute_table_quality({}, 100) == 0.0

    def test_zero_rows(self):
        assert _compute_table_quality({"a": {}}, 0) == 0.0

    def test_invalid_currency_lowers_validity(self):
        cols = {
            "amount": {"semantic_type": "numeric_currency", "cardinality": 50, "null_count": 0, "stats": {"min": -100, "max": 1000}},
        }
        score_invalid = _compute_table_quality(cols, 100)

        cols2 = {
            "amount": {"semantic_type": "numeric_currency", "cardinality": 50, "null_count": 0, "stats": {"min": 0, "max": 1000}},
        }
        score_valid = _compute_table_quality(cols2, 100)
        assert score_invalid < score_valid

    def test_more_types_higher_diversity(self):
        cols_single = {
            "a": {"semantic_type": "numeric_metric", "cardinality": 50, "null_count": 0},
            "b": {"semantic_type": "numeric_metric", "cardinality": 50, "null_count": 0},
        }
        cols_diverse = {
            "a": {"semantic_type": "identifier", "cardinality": 50, "null_count": 0},
            "b": {"semantic_type": "numeric_metric", "cardinality": 50, "null_count": 0},
            "c": {"semantic_type": "categorical", "cardinality": 10, "null_count": 0},
            "d": {"semantic_type": "temporal", "cardinality": 30, "null_count": 0},
        }
        score_single = _compute_table_quality(cols_single, 100)
        score_diverse = _compute_table_quality(cols_diverse, 100)
        assert score_diverse > score_single


class TestComputeDashboardSummary:
    def test_empty(self):
        summary = compute_dashboard_summary([], [], [])
        assert summary.total_tables == 0
        assert summary.total_rows == 0
        assert summary.total_insights == 0
        assert summary.total_reports == 0

    def test_basic_profiles(self):
        profiles = [
            _Prof("orders", 500, {"id": {"semantic_type": "identifier", "cardinality": 500}}),
            _Prof("products", 100, {"id": {"semantic_type": "identifier", "cardinality": 100}}),
        ]
        summary = compute_dashboard_summary(profiles, [], [])
        assert summary.total_tables == 2
        assert summary.total_rows == 600
        assert summary.total_columns == 2

    def test_top_tables_sorted(self):
        profiles = [
            _Prof("small", 10),
            _Prof("medium", 100),
            _Prof("large", 1000),
        ]
        summary = compute_dashboard_summary(profiles, [], [])
        assert summary.top_tables[0]["table"] == "large"
        assert summary.top_tables[0]["rows"] == 1000

    def test_insight_breakdown(self):
        insights = [
            _Insight("anomaly", "critical"),
            _Insight("anomaly", "warning"),
            _Insight("correlation", "info"),
            _Insight("seasonality", "info"),
        ]
        summary = compute_dashboard_summary([], insights, [])
        assert summary.total_insights == 4
        assert summary.insight_breakdown["anomaly"] == 2
        assert summary.insight_breakdown["correlation"] == 1
        assert summary.severity_breakdown["critical"] == 1
        assert summary.severity_breakdown["warning"] == 1
        assert summary.severity_breakdown["info"] == 2

    def test_reports_counted(self):
        reports = [_Report(), _Report(), _Report()]
        summary = compute_dashboard_summary([], [], reports)
        assert summary.total_reports == 3

    def test_freshness(self):
        profiles = [
            _Prof("t1", data_hash="abc123"),
            _Prof("t2", data_hash="def456"),
            _Prof("t3", data_hash=None),
        ]
        summary = compute_dashboard_summary(profiles, [], [])
        assert abs(summary.data_freshness_pct - 66.7) < 0.1

    def test_last_discovery(self):
        runs = [
            _Run("completed", "2024-01-15T10:00:00"),
            _Run("completed", "2024-01-20T10:00:00"),
            _Run("failed", "2024-01-25T10:00:00"),
        ]
        summary = compute_dashboard_summary([], [], [], runs)
        assert summary.last_discovery_at == "2024-01-20T10:00:00"

    def test_no_discovery_runs(self):
        summary = compute_dashboard_summary([], [], [], [])
        assert summary.last_discovery_at is None

    def test_dict_inputs(self):
        profiles = [{"table_name": "t1", "row_count": 50, "domain_hint": "hr", "data_hash": "x", "column_classification": {"columns": {"a": {"semantic_type": "identifier", "cardinality": 50}}}}]
        insights = [{"insight_type": "anomaly", "severity": "warning"}]
        reports = [{}]
        summary = compute_dashboard_summary(profiles, insights, reports)
        assert summary.total_tables == 1
        assert summary.total_insights == 1
        assert summary.total_reports == 1

    def test_avg_quality_score(self):
        profiles = [
            _Prof("t1", 100, {
                "id": {"semantic_type": "identifier", "cardinality": 100, "null_count": 0},
                "val": {"semantic_type": "numeric_metric", "cardinality": 80, "null_count": 0},
            }),
        ]
        summary = compute_dashboard_summary(profiles, [], [])
        assert summary.avg_quality_score > 0
        assert summary.avg_quality_score <= 100

    def test_none_classification_handled(self):
        p = _Prof("t1")
        p.column_classification = None
        summary = compute_dashboard_summary([p], [], [])
        assert summary.total_tables == 1
        assert summary.total_columns == 0
