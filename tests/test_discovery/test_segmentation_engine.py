"""Tests for segmentation_engine -- pure-Python k-means clustering."""

from __future__ import annotations

import math

import pytest

from business_brain.discovery.segmentation_engine import (
    Segment,
    SegmentationResult,
    compare_segments,
    find_segment_drivers,
    label_segments,
    segment_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(groups: list[tuple[int, dict[str, float]]]) -> list[dict]:
    """Build rows from (count, template) pairs with small jitter so rows
    within a group are close but not identical."""
    rows: list[dict] = []
    for count, template in groups:
        for i in range(count):
            row = {}
            for k, v in template.items():
                row[k] = v + i * 0.001  # tiny jitter keeps rows unique
            rows.append(row)
    return rows


# ===================================================================
# 1-7  Basic clustering with clear groups
# ===================================================================

class TestBasicClustering:
    """segment_data on well-separated clusters."""

    def test_two_clear_clusters(self):
        """Two tight groups far apart should yield 2 segments."""
        rows = _make_rows([
            (10, {"x": 0.0, "y": 0.0}),
            (10, {"x": 100.0, "y": 100.0}),
        ])
        result = segment_data(rows, ["x", "y"], n_segments=2)
        assert result is not None
        assert result.n_segments == 2
        assert result.total_rows == 20
        sizes = sorted(s.size for s in result.segments)
        assert sizes == [10, 10]

    def test_three_clear_clusters(self):
        rows = _make_rows([
            (10, {"a": 0.0}),
            (10, {"a": 50.0}),
            (10, {"a": 100.0}),
        ])
        result = segment_data(rows, ["a"], n_segments=3)
        assert result is not None
        assert result.n_segments == 3
        assert sum(s.size for s in result.segments) == 30

    def test_centroids_near_group_means(self):
        rows = _make_rows([
            (20, {"v": 10.0}),
            (20, {"v": 90.0}),
        ])
        result = segment_data(rows, ["v"], n_segments=2)
        assert result is not None
        centres = sorted(s.center["v"] for s in result.segments)
        assert centres[0] < 20
        assert centres[1] > 80

    def test_quality_score_high_for_clear_clusters(self):
        rows = _make_rows([
            (15, {"x": 0.0, "y": 0.0}),
            (15, {"x": 100.0, "y": 100.0}),
        ])
        result = segment_data(rows, ["x", "y"], n_segments=2)
        assert result is not None
        assert result.quality_score > 0.5

    def test_members_indices_correct(self):
        rows = _make_rows([
            (10, {"x": 0.0}),
            (10, {"x": 1000.0}),
        ])
        result = segment_data(rows, ["x"], n_segments=2)
        assert result is not None
        all_members = []
        for s in result.segments:
            all_members.extend(s.members)
        assert sorted(all_members) == list(range(20))

    def test_features_list_returned(self):
        rows = _make_rows([(10, {"a": 1.0, "b": 2.0})])
        result = segment_data(rows, ["a", "b"], n_segments=1)
        assert result is not None
        assert set(result.features) == {"a", "b"}

    def test_summary_is_non_empty(self):
        rows = _make_rows([
            (10, {"x": 0.0}),
            (10, {"x": 100.0}),
        ])
        result = segment_data(rows, ["x"], n_segments=2)
        assert result is not None
        assert len(result.summary) > 0


# ===================================================================
# 8  Single segment
# ===================================================================

class TestSingleSegment:
    def test_single_segment(self):
        rows = _make_rows([(20, {"x": 5.0, "y": 5.0})])
        result = segment_data(rows, ["x", "y"], n_segments=1)
        assert result is not None
        assert result.n_segments == 1
        assert result.segments[0].size == 20


# ===================================================================
# 9  Many features
# ===================================================================

class TestManyFeatures:
    def test_ten_features(self):
        feats = [f"f{i}" for i in range(10)]
        low = {f: 0.0 for f in feats}
        high = {f: 100.0 for f in feats}
        rows = _make_rows([(10, low), (10, high)])
        result = segment_data(rows, feats, n_segments=2)
        assert result is not None
        assert result.n_segments == 2
        assert len(result.features) == 10


# ===================================================================
# 10-11  All same values
# ===================================================================

class TestAllSameValues:
    def test_all_identical_rows(self):
        """When all values are the same, clustering should still work."""
        rows = [{"x": 5.0, "y": 5.0} for _ in range(20)]
        result = segment_data(rows, ["x", "y"], n_segments=2)
        assert result is not None
        # All rows are identical so quality should be low
        assert result.quality_score <= 0.5

    def test_constant_single_feature(self):
        rows = [{"x": 3.0} for _ in range(10)]
        result = segment_data(rows, ["x"], n_segments=1)
        assert result is not None
        assert result.segments[0].center["x"] == pytest.approx(3.0, abs=0.01)


# ===================================================================
# 12-14  Insufficient data / edge cases
# ===================================================================

class TestInsufficientData:
    def test_empty_rows(self):
        assert segment_data([], ["x"], n_segments=2) is None

    def test_too_few_rows(self):
        rows = [{"x": 1.0}, {"x": 2.0}]
        assert segment_data(rows, ["x"], n_segments=3) is None  # 2 < 3*2

    def test_no_features(self):
        rows = [{"x": 1.0}]
        assert segment_data(rows, [], n_segments=1) is None


# ===================================================================
# 15-17  Label generation
# ===================================================================

class TestLabelGeneration:
    def test_labels_assigned(self):
        rows = _make_rows([
            (10, {"revenue": 100.0, "cost": 10.0}),
            (10, {"revenue": 10.0, "cost": 100.0}),
        ])
        result = segment_data(rows, ["revenue", "cost"], n_segments=2)
        assert result is not None
        for seg in result.segments:
            assert len(seg.label) > 0

    def test_label_contains_feature_names(self):
        rows = _make_rows([
            (10, {"revenue": 200.0, "cost": 5.0}),
            (10, {"revenue": 5.0, "cost": 200.0}),
        ])
        result = segment_data(rows, ["revenue", "cost"], n_segments=2)
        assert result is not None
        all_labels = " ".join(s.label for s in result.segments)
        assert "revenue" in all_labels
        assert "cost" in all_labels

    def test_label_segments_standalone(self):
        """Calling label_segments directly."""
        seg_a = Segment(0, "", 10, {"x": 100.0}, {"x": 5.0}, list(range(10)))
        seg_b = Segment(1, "", 10, {"x": 0.0}, {"x": 5.0}, list(range(10, 20)))
        labelled = label_segments([seg_a, seg_b], ["x"])
        labels = [s.label for s in labelled]
        assert any("High" in l for l in labels)
        assert any("Low" in l for l in labels)


# ===================================================================
# 18-20  Segment drivers
# ===================================================================

class TestSegmentDrivers:
    def test_drivers_sorted_by_importance(self):
        seg_a = Segment(0, "A", 10, {"revenue": 100.0, "clicks": 50.0}, {}, [])
        seg_b = Segment(1, "B", 10, {"revenue": 10.0, "clicks": 48.0}, {}, [])
        drivers = find_segment_drivers([seg_a, seg_b], ["revenue", "clicks"])
        assert drivers[0]["feature"] == "revenue"
        assert drivers[0]["importance"] == pytest.approx(1.0)

    def test_drivers_range_values(self):
        seg_a = Segment(0, "A", 10, {"x": 0.0, "y": 50.0}, {}, [])
        seg_b = Segment(1, "B", 10, {"x": 100.0, "y": 50.0}, {}, [])
        drivers = find_segment_drivers([seg_a, seg_b], ["x", "y"])
        driver_map = {d["feature"]: d for d in drivers}
        assert driver_map["x"]["range_across_segments"] == pytest.approx(100.0)
        assert driver_map["y"]["range_across_segments"] == pytest.approx(0.0)

    def test_drivers_empty_with_single_segment(self):
        seg = Segment(0, "A", 10, {"x": 5.0}, {}, [])
        assert find_segment_drivers([seg], ["x"]) == []


# ===================================================================
# 21-23  Segment comparison
# ===================================================================

class TestSegmentComparison:
    def test_compare_dominant_features(self):
        seg_a = Segment(0, "A", 10, {"x": 100.0, "y": 10.0}, {}, [])
        seg_b = Segment(1, "B", 10, {"x": 10.0, "y": 100.0}, {}, [])
        cmp = compare_segments(seg_a, seg_b, ["x", "y"])
        assert "x" in cmp["dominant_a"]
        assert "y" in cmp["dominant_b"]

    def test_compare_similar_features(self):
        seg_a = Segment(0, "A", 10, {"x": 50.0, "y": 50.0}, {}, [])
        seg_b = Segment(1, "B", 10, {"x": 50.0, "y": 50.0}, {}, [])
        cmp = compare_segments(seg_a, seg_b, ["x", "y"])
        assert "x" in cmp["similar"]
        assert "y" in cmp["similar"]
        assert cmp["distance"] == pytest.approx(0.0, abs=0.01)

    def test_compare_distance(self):
        seg_a = Segment(0, "A", 10, {"x": 0.0}, {}, [])
        seg_b = Segment(1, "B", 10, {"x": 100.0}, {}, [])
        cmp = compare_segments(seg_a, seg_b, ["x"])
        assert cmp["distance"] == pytest.approx(100.0, abs=0.1)


# ===================================================================
# 24-25  Edge cases (nulls, non-numeric) & determinism
# ===================================================================

class TestEdgeCases:
    def test_rows_with_none_values(self):
        """None values should be imputed; clustering should still succeed."""
        rows = [
            {"x": 0.0, "y": 0.0},
            {"x": None, "y": 0.0},
            {"x": 0.0, "y": None},
        ] * 4  # 12 rows total
        rows += [{"x": 100.0, "y": 100.0}] * 12
        result = segment_data(rows, ["x", "y"], n_segments=2)
        assert result is not None
        assert result.n_segments == 2

    def test_non_numeric_feature_skipped(self):
        rows = [{"x": float(i), "name": f"row_{i}"} for i in range(20)]
        result = segment_data(rows, ["x", "name"], n_segments=2)
        assert result is not None
        assert "name" not in result.features
        assert "x" in result.features

    def test_all_features_non_numeric(self):
        rows = [{"a": "foo", "b": "bar"} for _ in range(20)]
        assert segment_data(rows, ["a", "b"], n_segments=2) is None


class TestDeterminism:
    def test_same_input_same_output(self):
        rows = _make_rows([
            (15, {"x": 0.0, "y": 0.0}),
            (15, {"x": 100.0, "y": 100.0}),
        ])
        r1 = segment_data(rows, ["x", "y"], n_segments=2)
        r2 = segment_data(rows, ["x", "y"], n_segments=2)
        assert r1 is not None and r2 is not None
        # Same assignments
        for s1, s2 in zip(r1.segments, r2.segments):
            assert s1.members == s2.members
            assert s1.center == s2.center

    def test_deterministic_quality_score(self):
        rows = _make_rows([
            (10, {"a": 0.0}),
            (10, {"a": 50.0}),
            (10, {"a": 100.0}),
        ])
        r1 = segment_data(rows, ["a"], n_segments=3)
        r2 = segment_data(rows, ["a"], n_segments=3)
        assert r1 is not None and r2 is not None
        assert r1.quality_score == r2.quality_score


# ===================================================================
# 28-29  Extra edge-case coverage
# ===================================================================

class TestExtraEdgeCases:
    def test_single_row_per_segment_insufficient(self):
        """n_segments=3 requires >= 6 rows."""
        rows = [{"x": float(i)} for i in range(5)]
        assert segment_data(rows, ["x"], n_segments=3) is None

    def test_spread_non_negative(self):
        rows = _make_rows([
            (10, {"x": 10.0, "y": 20.0}),
            (10, {"x": 90.0, "y": 80.0}),
        ])
        result = segment_data(rows, ["x", "y"], n_segments=2)
        assert result is not None
        for seg in result.segments:
            for feat in result.features:
                assert seg.spread[feat] >= 0.0

    def test_missing_key_in_row(self):
        """Rows that lack a feature entirely should be imputed."""
        rows = [{"x": 1.0, "y": 2.0}] * 10 + [{"x": 100.0}] * 10
        result = segment_data(rows, ["x", "y"], n_segments=2)
        assert result is not None
        assert "y" in result.features
