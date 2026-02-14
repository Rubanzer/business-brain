"""Tests for the data freshness tracking module."""

from business_brain.db.discovery_models import TableProfile
from business_brain.discovery.data_freshness import (
    compute_freshness_score,
    detect_stale_tables,
)


def _make_profile(table_name, data_hash=None, row_count=100):
    p = TableProfile()
    p.table_name = table_name
    p.row_count = row_count
    p.data_hash = data_hash
    p.column_classification = {}
    return p


class TestDetectStaleTables:
    def test_same_hash_flagged(self):
        curr = [_make_profile("sales", data_hash="abc123")]
        prev = [_make_profile("sales", data_hash="abc123")]
        insights = detect_stale_tables(curr, prev)
        assert len(insights) == 1
        assert insights[0].evidence["pattern_type"] == "stale_data"
        assert insights[0].source_tables == ["sales"]

    def test_different_hash_not_flagged(self):
        curr = [_make_profile("sales", data_hash="abc123")]
        prev = [_make_profile("sales", data_hash="def456")]
        insights = detect_stale_tables(curr, prev)
        assert len(insights) == 0

    def test_new_table_not_flagged(self):
        curr = [_make_profile("new_table", data_hash="abc123")]
        prev = []
        insights = detect_stale_tables(curr, prev)
        assert len(insights) == 0

    def test_no_hash_skipped(self):
        curr = [_make_profile("sales", data_hash=None)]
        prev = [_make_profile("sales", data_hash="abc123")]
        insights = detect_stale_tables(curr, prev)
        assert len(insights) == 0

    def test_prev_no_hash_skipped(self):
        curr = [_make_profile("sales", data_hash="abc123")]
        prev = [_make_profile("sales", data_hash=None)]
        insights = detect_stale_tables(curr, prev)
        assert len(insights) == 0

    def test_multiple_tables(self):
        curr = [
            _make_profile("sales", data_hash="aaa"),
            _make_profile("orders", data_hash="bbb"),
            _make_profile("hr", data_hash="ccc"),
        ]
        prev = [
            _make_profile("sales", data_hash="aaa"),  # stale
            _make_profile("orders", data_hash="xxx"),  # fresh
            _make_profile("hr", data_hash="ccc"),  # stale
        ]
        insights = detect_stale_tables(curr, prev)
        assert len(insights) == 2
        stale_tables = {i.source_tables[0] for i in insights}
        assert "sales" in stale_tables
        assert "hr" in stale_tables

    def test_empty_profiles(self):
        assert detect_stale_tables([], []) == []

    def test_insight_fields(self):
        curr = [_make_profile("t", data_hash="x")]
        prev = [_make_profile("t", data_hash="x")]
        ins = detect_stale_tables(curr, prev)[0]
        assert ins.insight_type == "data_quality"
        assert ins.severity == "info"
        assert ins.id is not None
        assert len(ins.suggested_actions) > 0


class TestComputeFreshnessScore:
    def test_all_fresh(self):
        curr = [_make_profile("a", "h1"), _make_profile("b", "h2")]
        prev = [_make_profile("a", "x1"), _make_profile("b", "x2")]
        result = compute_freshness_score(curr, prev)
        assert result["score"] == 100
        assert result["fresh_count"] == 2
        assert result["stale_count"] == 0

    def test_all_stale(self):
        curr = [_make_profile("a", "h1"), _make_profile("b", "h2")]
        prev = [_make_profile("a", "h1"), _make_profile("b", "h2")]
        result = compute_freshness_score(curr, prev)
        assert result["score"] == 0
        assert result["stale_count"] == 2

    def test_mixed(self):
        curr = [_make_profile("a", "h1"), _make_profile("b", "h2")]
        prev = [_make_profile("a", "h1"), _make_profile("b", "x2")]
        result = compute_freshness_score(curr, prev)
        assert result["score"] == 50
        assert result["fresh_count"] == 1
        assert result["stale_count"] == 1

    def test_unknown_hashes(self):
        curr = [_make_profile("a", None)]
        prev = [_make_profile("a", "h1")]
        result = compute_freshness_score(curr, prev)
        assert result["unknown_count"] == 1
        assert result["score"] == 100  # No known stale

    def test_no_tables(self):
        result = compute_freshness_score([], [])
        assert result["score"] == 100
        assert result["total_tables"] == 0

    def test_total_tables_count(self):
        curr = [_make_profile("a", "h1"), _make_profile("b", "h2"), _make_profile("c", None)]
        prev = [_make_profile("a", "h1"), _make_profile("b", "x2")]
        result = compute_freshness_score(curr, prev)
        assert result["total_tables"] == 3
