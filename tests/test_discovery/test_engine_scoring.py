"""Tests for the discovery engine scoring function."""

from business_brain.db.discovery_models import Insight
from business_brain.discovery.engine import _apply_scoring


def _make_insight(severity="info", impact_score=50, source_tables=None):
    ins = Insight()
    ins.id = "test-1"
    ins.insight_type = "anomaly"
    ins.severity = severity
    ins.impact_score = impact_score
    ins.title = "Test"
    ins.description = "Test"
    ins.source_tables = source_tables
    ins.source_columns = []
    ins.evidence = {}
    ins.suggested_actions = []
    return ins


class TestApplyScoring:
    """Test the _apply_scoring formula."""

    def test_info_severity_single_table(self):
        ins = _make_insight(severity="info", impact_score=50, source_tables=["sales"])
        _apply_scoring(ins)
        # severity=info → 0.3*40=12, cross=0 (1 table), magnitude=50/100*30=15
        assert ins.impact_score == 27

    def test_critical_severity_single_table(self):
        ins = _make_insight(severity="critical", impact_score=50, source_tables=["sales"])
        _apply_scoring(ins)
        # severity=critical → 1.0*40=40, cross=0, magnitude=50/100*30=15
        assert ins.impact_score == 55

    def test_warning_severity_single_table(self):
        ins = _make_insight(severity="warning", impact_score=50, source_tables=["sales"])
        _apply_scoring(ins)
        # severity=warning → 0.6*40=24, cross=0, magnitude=50/100*30=15
        assert ins.impact_score == 39

    def test_cross_table_bonus(self):
        ins = _make_insight(severity="info", impact_score=50, source_tables=["sales", "orders"])
        _apply_scoring(ins)
        # severity=12, cross=30 (2 tables), magnitude=15
        assert ins.impact_score == 57

    def test_critical_cross_table_high_impact(self):
        ins = _make_insight(severity="critical", impact_score=100, source_tables=["a", "b"])
        _apply_scoring(ins)
        # severity=40, cross=30, magnitude=100/100*30=30 → total=100
        assert ins.impact_score == 100

    def test_capped_at_100(self):
        ins = _make_insight(severity="critical", impact_score=200, source_tables=["a", "b"])
        _apply_scoring(ins)
        assert ins.impact_score <= 100

    def test_none_source_tables(self):
        ins = _make_insight(severity="info", impact_score=50, source_tables=None)
        _apply_scoring(ins)
        # Should not crash; cross_table_bonus = 0
        assert ins.impact_score == 27

    def test_empty_source_tables(self):
        ins = _make_insight(severity="info", impact_score=50, source_tables=[])
        _apply_scoring(ins)
        assert ins.impact_score == 27

    def test_none_impact_score(self):
        ins = _make_insight(severity="info", impact_score=None, source_tables=["x"])
        _apply_scoring(ins)
        # magnitude = 0/100*30 = 0, severity=12
        assert ins.impact_score == 12

    def test_zero_impact_score(self):
        ins = _make_insight(severity="info", impact_score=0, source_tables=["x"])
        _apply_scoring(ins)
        # severity=12, cross=0, magnitude=0
        assert ins.impact_score == 12

    def test_unknown_severity_uses_default(self):
        ins = _make_insight(severity="unknown", impact_score=50, source_tables=["x"])
        _apply_scoring(ins)
        # default weight 0.3*40=12, magnitude=15
        assert ins.impact_score == 27

    def test_three_tables_still_30_bonus(self):
        ins = _make_insight(severity="info", impact_score=50, source_tables=["a", "b", "c"])
        _apply_scoring(ins)
        # cross_table_bonus is 30 whether 2 or more tables
        assert ins.impact_score == 57
