"""Tests for contribution analysis module."""

from business_brain.discovery.contribution_analysis import (
    analyze_contributions,
    contribution_from_rows,
    waterfall_data,
)


class TestAnalyzeContributions:
    def test_basic(self):
        before = {"Sales": 100, "Services": 50}
        after = {"Sales": 120, "Services": 60}
        result = analyze_contributions(before, after)
        assert result is not None
        assert result.total_before == 150
        assert result.total_after == 180
        assert result.total_change == 30

    def test_empty_returns_none(self):
        assert analyze_contributions({}, {}) is None

    def test_positive_driver(self):
        before = {"A": 100, "B": 50}
        after = {"A": 200, "B": 50}
        result = analyze_contributions(before, after)
        assert result.top_positive_driver == "A"
        assert result.items[0].name == "A"

    def test_negative_driver(self):
        before = {"A": 100, "B": 50}
        after = {"A": 100, "B": 20}
        result = analyze_contributions(before, after)
        assert result.top_negative_driver == "B"

    def test_new_component(self):
        before = {"A": 100}
        after = {"A": 100, "B": 50}
        result = analyze_contributions(before, after)
        b_item = next(i for i in result.items if i.name == "B")
        assert b_item.value_before == 0
        assert b_item.value_after == 50

    def test_removed_component(self):
        before = {"A": 100, "B": 50}
        after = {"A": 100}
        result = analyze_contributions(before, after)
        b_item = next(i for i in result.items if i.name == "B")
        assert b_item.value_after == 0
        assert b_item.direction == "negative"

    def test_no_change(self):
        values = {"A": 100, "B": 50}
        result = analyze_contributions(values, values)
        assert result.total_change == 0
        assert all(i.direction == "unchanged" for i in result.items)

    def test_contribution_pct(self):
        before = {"A": 100, "B": 100}
        after = {"A": 200, "B": 100}
        result = analyze_contributions(before, after)
        a_item = next(i for i in result.items if i.name == "A")
        assert a_item.contribution_pct == 100.0

    def test_mixed_directions(self):
        before = {"A": 100, "B": 50, "C": 80}
        after = {"A": 150, "B": 30, "C": 80}
        result = analyze_contributions(before, after)
        a_item = next(i for i in result.items if i.name == "A")
        b_item = next(i for i in result.items if i.name == "B")
        c_item = next(i for i in result.items if i.name == "C")
        assert a_item.direction == "positive"
        assert b_item.direction == "negative"
        assert c_item.direction == "unchanged"

    def test_total_change_pct(self):
        before = {"A": 100}
        after = {"A": 150}
        result = analyze_contributions(before, after)
        assert result.total_change_pct == 50.0

    def test_summary_text(self):
        before = {"A": 100}
        after = {"A": 120}
        result = analyze_contributions(before, after)
        assert "100" in result.summary
        assert "120" in result.summary

    def test_sorted_by_absolute_change(self):
        before = {"A": 100, "B": 100, "C": 100}
        after = {"A": 110, "B": 150, "C": 90}
        result = analyze_contributions(before, after)
        assert abs(result.items[0].absolute_change) >= abs(result.items[1].absolute_change)

    def test_concentration(self):
        before = {"A": 100, "B": 100}
        after = {"A": 200, "B": 110}
        result = analyze_contributions(before, after)
        assert result.concentration > 0


class TestContributionFromRows:
    def test_basic(self):
        rows_before = [
            {"dept": "Sales", "revenue": 100},
            {"dept": "Eng", "revenue": 80},
        ]
        rows_after = [
            {"dept": "Sales", "revenue": 120},
            {"dept": "Eng", "revenue": 90},
        ]
        result = contribution_from_rows(rows_before, rows_after, "dept", "revenue")
        assert result is not None
        assert result.total_change > 0

    def test_aggregates_duplicates(self):
        rows_before = [
            {"d": "A", "v": 50},
            {"d": "A", "v": 50},
        ]
        rows_after = [
            {"d": "A", "v": 120},
        ]
        result = contribution_from_rows(rows_before, rows_after, "d", "v")
        assert result is not None
        a_item = next(i for i in result.items if i.name == "A")
        assert a_item.value_before == 100

    def test_empty_returns_none(self):
        assert contribution_from_rows([], [], "d", "v") is None


class TestWaterfall:
    def test_basic_waterfall(self):
        before = {"A": 100, "B": 50}
        after = {"A": 130, "B": 40}
        result = analyze_contributions(before, after)
        wf = waterfall_data(result)
        assert wf[0]["label"] == "Start"
        assert wf[-1]["label"] == "End"
        assert wf[0]["value"] == 150
        assert wf[-1]["value"] == 170

    def test_waterfall_types(self):
        before = {"A": 100, "B": 50}
        after = {"A": 120, "B": 30}
        result = analyze_contributions(before, after)
        wf = waterfall_data(result)
        types = {w["type"] for w in wf}
        assert "total" in types

    def test_empty_change(self):
        result = analyze_contributions({"A": 100}, {"A": 100})
        wf = waterfall_data(result)
        assert len(wf) == 2  # just Start and End
