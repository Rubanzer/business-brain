"""Tests for the cross-event correlator confidence filtering."""

from business_brain.db.discovery_models import DiscoveredRelationship
from business_brain.discovery.cross_event_correlator import filter_by_confidence


def _make_rel(confidence=0.8):
    r = DiscoveredRelationship()
    r.table_a = "a"
    r.column_a = "id"
    r.table_b = "b"
    r.column_b = "id"
    r.relationship_type = "join_key"
    r.confidence = confidence
    r.overlap_count = 10
    return r


class TestFilterByConfidence:
    def test_all_above_threshold(self):
        rels = [_make_rel(0.9), _make_rel(0.8)]
        result = filter_by_confidence(rels, min_confidence=0.5)
        assert len(result) == 2

    def test_all_below_threshold(self):
        rels = [_make_rel(0.3), _make_rel(0.2)]
        result = filter_by_confidence(rels, min_confidence=0.5)
        assert len(result) == 0

    def test_mixed(self):
        rels = [_make_rel(0.9), _make_rel(0.3), _make_rel(0.6)]
        result = filter_by_confidence(rels, min_confidence=0.5)
        assert len(result) == 2

    def test_exact_threshold_included(self):
        rels = [_make_rel(0.5)]
        result = filter_by_confidence(rels, min_confidence=0.5)
        assert len(result) == 1

    def test_just_below_threshold(self):
        rels = [_make_rel(0.49)]
        result = filter_by_confidence(rels, min_confidence=0.5)
        assert len(result) == 0

    def test_empty_list(self):
        assert filter_by_confidence([], min_confidence=0.5) == []

    def test_none_confidence_treated_as_zero(self):
        r = _make_rel()
        r.confidence = None
        result = filter_by_confidence([r], min_confidence=0.5)
        assert len(result) == 0

    def test_zero_threshold_includes_all(self):
        rels = [_make_rel(0.0), _make_rel(0.1)]
        result = filter_by_confidence(rels, min_confidence=0.0)
        assert len(result) == 2

    def test_high_threshold(self):
        rels = [_make_rel(0.9), _make_rel(0.95)]
        result = filter_by_confidence(rels, min_confidence=0.9)
        assert len(result) == 2
