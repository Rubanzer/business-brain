"""Tests for cohort analysis module."""

from business_brain.discovery.cohort_analysis import (
    CohortResult,
    build_cohorts,
    compute_cohort_health,
    find_declining_cohorts,
    pivot_cohort_table,
)


# ---------------------------------------------------------------------------
# build_cohorts
# ---------------------------------------------------------------------------


class TestBuildCohorts:
    def test_basic_cohorts(self):
        rows = [
            {"supplier": "A", "month": "Jan", "revenue": 100},
            {"supplier": "A", "month": "Feb", "revenue": 120},
            {"supplier": "B", "month": "Jan", "revenue": 200},
            {"supplier": "B", "month": "Feb", "revenue": 180},
        ]
        result = build_cohorts(rows, "supplier", "month", "revenue")
        assert result is not None
        assert len(result.cohorts) == 2
        assert len(result.periods) == 2
        assert len(result.buckets) == 4

    def test_empty_rows(self):
        assert build_cohorts([], "c", "t", "m") is None

    def test_single_bucket(self):
        rows = [{"c": "A", "t": "Jan", "m": 100}]
        assert build_cohorts(rows, "c", "t", "m") is None

    def test_multiple_rows_per_bucket(self):
        rows = [
            {"c": "A", "t": "Jan", "m": 10},
            {"c": "A", "t": "Jan", "m": 20},
            {"c": "A", "t": "Feb", "m": 30},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        assert result is not None
        jan_bucket = next(b for b in result.buckets if b.period == "Jan")
        assert jan_bucket.count == 2
        assert jan_bucket.metric_mean == 15.0

    def test_null_values_skipped(self):
        rows = [
            {"c": "A", "t": "Jan", "m": 100},
            {"c": "A", "t": None, "m": 200},
            {"c": "A", "t": "Feb", "m": 300},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        assert result is not None
        assert len(result.buckets) == 2

    def test_retention_matrix(self):
        rows = [
            {"c": "A", "t": "P1", "m": 100},
            {"c": "A", "t": "P1", "m": 100},
            {"c": "A", "t": "P2", "m": 100},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        assert result is not None
        assert "A" in result.retention_matrix
        # P1 has 2, P2 has 1 → 50% retention
        assert result.retention_matrix["A"]["P2"] == 50.0

    def test_growth_matrix(self):
        rows = [
            {"c": "A", "t": "P1", "m": 100},
            {"c": "A", "t": "P2", "m": 150},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        assert result is not None
        assert "A" in result.growth_matrix
        assert result.growth_matrix["A"]["P2"] == 50.0  # 50% growth

    def test_summary(self):
        rows = [
            {"c": "X", "t": "Jan", "m": 10},
            {"c": "Y", "t": "Jan", "m": 20},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        assert "2 cohorts" in result.summary

    def test_non_numeric_metric_skipped(self):
        rows = [
            {"c": "A", "t": "Jan", "m": "abc"},
            {"c": "B", "t": "Jan", "m": "def"},
        ]
        assert build_cohorts(rows, "c", "t", "m") is None


# ---------------------------------------------------------------------------
# compute_cohort_health
# ---------------------------------------------------------------------------


class TestComputeCohortHealth:
    def test_basic_health(self):
        rows = [
            {"c": "A", "t": "P1", "m": 100},
            {"c": "A", "t": "P2", "m": 120},
            {"c": "B", "t": "P1", "m": 200},
            {"c": "B", "t": "P2", "m": 180},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        health = compute_cohort_health(result)
        assert health["total_cohorts"] == 2
        assert health["total_periods"] == 2
        assert health["best_cohort"] is not None
        assert health["worst_cohort"] is not None

    def test_no_data(self):
        health = compute_cohort_health(None)
        assert health["status"] == "no_data"

    def test_best_worst_in_last_period(self):
        rows = [
            {"c": "A", "t": "P1", "m": 10},
            {"c": "A", "t": "P2", "m": 50},
            {"c": "B", "t": "P1", "m": 100},
            {"c": "B", "t": "P2", "m": 20},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        health = compute_cohort_health(result)
        assert health["best_cohort"] == "A"
        assert health["worst_cohort"] == "B"


# ---------------------------------------------------------------------------
# pivot_cohort_table
# ---------------------------------------------------------------------------


class TestPivotCohortTable:
    def test_basic_pivot(self):
        rows = [
            {"c": "A", "t": "Jan", "m": 100},
            {"c": "A", "t": "Feb", "m": 120},
            {"c": "B", "t": "Jan", "m": 200},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        table = pivot_cohort_table(result)
        assert len(table) == 2
        a_row = next(r for r in table if r["cohort"] == "A")
        assert a_row["Jan"] == 100.0
        assert a_row["Feb"] == 120.0

    def test_missing_values(self):
        rows = [
            {"c": "A", "t": "Jan", "m": 100},
            {"c": "B", "t": "Feb", "m": 200},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        table = pivot_cohort_table(result)
        a_row = next(r for r in table if r["cohort"] == "A")
        assert a_row.get("Feb") is None

    def test_none_result(self):
        assert pivot_cohort_table(None) == []


# ---------------------------------------------------------------------------
# find_declining_cohorts
# ---------------------------------------------------------------------------


class TestFindDecliningCohorts:
    def test_declining_cohort(self):
        rows = [
            {"c": "A", "t": "P1", "m": 100},
            {"c": "A", "t": "P2", "m": 50},
            {"c": "B", "t": "P1", "m": 100},
            {"c": "B", "t": "P2", "m": 120},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        declining = find_declining_cohorts(result)
        assert len(declining) == 1
        assert declining[0]["cohort"] == "A"
        assert declining[0]["decline_pct"] < 0

    def test_no_declining(self):
        rows = [
            {"c": "A", "t": "P1", "m": 100},
            {"c": "A", "t": "P2", "m": 120},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        assert find_declining_cohorts(result) == []

    def test_custom_threshold(self):
        rows = [
            {"c": "A", "t": "P1", "m": 100},
            {"c": "A", "t": "P2", "m": 95},
        ]
        result = build_cohorts(rows, "c", "t", "m")
        # -5% decline, default threshold is -10%
        assert find_declining_cohorts(result) == []
        # With -3% threshold, it should be flagged
        declining = find_declining_cohorts(result, threshold=-3.0)
        assert len(declining) == 1

    def test_none_result(self):
        assert find_declining_cohorts(None) == []
