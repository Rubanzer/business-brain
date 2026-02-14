"""Tests for pattern memory module — behavior scoring."""

from business_brain.discovery.pattern_memory import _parse_magnitude, _score_behavior


class TestScoreBehavior:
    """Test the behavior scoring function."""

    def test_decreasing_values(self):
        values = [100, 90, 80, 70, 60]
        score = _score_behavior(values, "decreasing", "")
        assert score > 0.5

    def test_increasing_values(self):
        values = [60, 70, 80, 90, 100]
        score = _score_behavior(values, "increasing", "")
        assert score > 0.5

    def test_decreasing_when_increasing(self):
        values = [60, 70, 80, 90, 100]
        score = _score_behavior(values, "decreasing", "")
        assert score == 0.0

    def test_increasing_when_decreasing(self):
        values = [100, 90, 80, 70, 60]
        score = _score_behavior(values, "increasing", "")
        assert score == 0.0

    def test_stable_values(self):
        values = [100, 101, 99, 100, 101]
        score = _score_behavior(values, "stable", "")
        assert score > 0.7

    def test_stable_or_increasing(self):
        values = [100, 101, 102, 103, 104]
        score = _score_behavior(values, "stable_or_increasing", "")
        assert score > 0.4

    def test_empty_values(self):
        score = _score_behavior([], "decreasing", "")
        assert score == 0.0

    def test_single_value(self):
        score = _score_behavior([100], "decreasing", "")
        assert score == 0.0

    def test_decreasing_with_magnitude(self):
        values = [100, 95, 90, 85, 80]  # 20% decrease
        score = _score_behavior(values, "decreasing", ">10%")
        assert score > 0.5

    def test_spike_detection(self):
        values = [10, 10, 10, 50, 10]  # spike at index 3
        score = _score_behavior(values, "spike", "")
        assert score > 0.0

    def test_no_spike_stable(self):
        values = [10, 11, 10, 11, 10]
        score = _score_behavior(values, "spike", "")
        assert score == 0.0


class TestParseMagnitude:
    """Test magnitude string parsing."""

    def test_percentage(self):
        assert _parse_magnitude(">10%") == 10.0

    def test_decimal(self):
        assert _parse_magnitude(">5.5%") == 5.5

    def test_no_number(self):
        assert _parse_magnitude("large") == 0.0

    def test_just_number(self):
        assert _parse_magnitude("20") == 20.0

    def test_empty_string(self):
        assert _parse_magnitude("") == 0.0

    def test_less_than(self):
        assert _parse_magnitude("<15%") == 15.0

    def test_equals(self):
        assert _parse_magnitude("=100") == 100.0


class TestScoreBehaviorEdgeCases:
    """Additional edge cases for _score_behavior."""

    def test_two_values_decreasing(self):
        score = _score_behavior([100, 80], "decreasing", "")
        assert score > 0.0

    def test_two_values_increasing(self):
        score = _score_behavior([80, 100], "increasing", "")
        assert score > 0.0

    def test_unknown_behavior_returns_zero(self):
        score = _score_behavior([10, 20, 30], "unknown_behavior", "")
        assert score == 0.0

    def test_decreasing_below_magnitude_halved(self):
        """If decrease is < required magnitude, score is halved."""
        values = [100, 98, 96, 94, 92]  # 8% decrease
        score_full = _score_behavior(values, "decreasing", "")
        score_with_mag = _score_behavior(values, "decreasing", ">20%")
        assert score_with_mag < score_full

    def test_stable_with_high_variance(self):
        values = [100, 200, 50, 300, 10]
        score = _score_behavior(values, "stable", "")
        assert score < 0.3

    def test_spike_with_zero_median(self):
        """Edge case: median is 0, should not crash."""
        values = [0, 0, 0, 100, 0]
        score = _score_behavior(values, "spike", "")
        assert score > 0.0

    def test_decreasing_from_zero(self):
        """First value is 0 → uses 1 as denominator."""
        values = [0, -1, -2, -3, -4]
        score = _score_behavior(values, "decreasing", "")
        assert score > 0.0

    def test_all_same_values_stable(self):
        values = [50, 50, 50, 50, 50]
        score = _score_behavior(values, "stable", "")
        assert score == 1.0

    def test_increasing_with_magnitude(self):
        values = [100, 110, 120, 130, 140]  # 40% increase
        score = _score_behavior(values, "increasing", ">30%")
        assert score > 0.5

    def test_increasing_below_magnitude(self):
        values = [100, 102, 104, 106, 108]  # 8% increase
        score_no_mag = _score_behavior(values, "increasing", "")
        score_with_mag = _score_behavior(values, "increasing", ">20%")
        assert score_with_mag < score_no_mag
