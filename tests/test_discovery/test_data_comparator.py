"""Tests for data comparator — snapshot diffing and change detection."""

from business_brain.discovery.data_comparator import (
    ColumnChange,
    DataDiff,
    RowChange,
    classify_change,
    compare_snapshots,
    compute_change_rate,
    summarize_column_drift,
)


# ---------------------------------------------------------------------------
# compare_snapshots
# ---------------------------------------------------------------------------


class TestCompareSnapshots:
    def test_empty_both(self):
        diff = compare_snapshots([], [], ["id"])
        assert diff.added_rows == 0
        assert diff.removed_rows == 0
        assert diff.changed_rows == 0
        assert diff.unchanged_rows == 0

    def test_all_added(self):
        new = [{"id": 1, "val": 10}, {"id": 2, "val": 20}]
        diff = compare_snapshots([], new, ["id"])
        assert diff.added_rows == 2
        assert diff.removed_rows == 0
        assert diff.total_new == 2

    def test_all_removed(self):
        old = [{"id": 1, "val": 10}, {"id": 2, "val": 20}]
        diff = compare_snapshots(old, [], ["id"])
        assert diff.removed_rows == 2
        assert diff.added_rows == 0

    def test_no_changes(self):
        data = [{"id": 1, "val": 10}, {"id": 2, "val": 20}]
        diff = compare_snapshots(data, data, ["id"])
        assert diff.added_rows == 0
        assert diff.removed_rows == 0
        assert diff.changed_rows == 0
        assert diff.unchanged_rows == 2

    def test_value_change(self):
        old = [{"id": 1, "val": 10}]
        new = [{"id": 1, "val": 20}]
        diff = compare_snapshots(old, new, ["id"])
        assert diff.changed_rows == 1
        assert diff.unchanged_rows == 0
        assert "val" in diff.column_changes
        assert diff.column_changes["val"] == 1

    def test_mixed_changes(self):
        old = [{"id": 1, "val": 10}, {"id": 2, "val": 20}, {"id": 3, "val": 30}]
        new = [{"id": 1, "val": 10}, {"id": 2, "val": 25}, {"id": 4, "val": 40}]
        diff = compare_snapshots(old, new, ["id"])
        assert diff.added_rows == 1  # id=4
        assert diff.removed_rows == 1  # id=3
        assert diff.changed_rows == 1  # id=2
        assert diff.unchanged_rows == 1  # id=1

    def test_composite_key(self):
        old = [{"a": 1, "b": "x", "val": 10}]
        new = [{"a": 1, "b": "x", "val": 20}]
        diff = compare_snapshots(old, new, ["a", "b"])
        assert diff.changed_rows == 1

    def test_no_key_columns(self):
        old = [{"x": 1}]
        new = [{"x": 2}]
        diff = compare_snapshots(old, new, [])
        assert diff.added_rows == 1
        assert diff.removed_rows == 1

    def test_sample_limit(self):
        old = []
        new = [{"id": i, "val": i} for i in range(20)]
        diff = compare_snapshots(old, new, ["id"], sample_limit=3)
        assert len(diff.sample_additions) == 3
        assert diff.added_rows == 20

    def test_table_name(self):
        diff = compare_snapshots([], [], ["id"], table_name="orders")
        assert diff.table_name == "orders"

    def test_summary_text(self):
        old = [{"id": 1, "val": 10}]
        new = [{"id": 1, "val": 20}, {"id": 2, "val": 30}]
        diff = compare_snapshots(old, new, ["id"], table_name="test")
        assert "1 rows added" in diff.summary
        assert "1 rows changed" in diff.summary
        assert "test:" in diff.summary

    def test_column_changes_top(self):
        old = [
            {"id": 1, "a": 1, "b": 1, "c": 1},
            {"id": 2, "a": 2, "b": 2, "c": 2},
        ]
        new = [
            {"id": 1, "a": 99, "b": 99, "c": 1},
            {"id": 2, "a": 88, "b": 2, "c": 2},
        ]
        diff = compare_snapshots(old, new, ["id"])
        assert diff.column_changes["a"] == 2
        assert diff.column_changes["b"] == 1

    def test_sample_changes_have_details(self):
        old = [{"id": 1, "val": 10, "name": "old"}]
        new = [{"id": 1, "val": 20, "name": "new"}]
        diff = compare_snapshots(old, new, ["id"])
        assert len(diff.sample_changes) == 1
        change = diff.sample_changes[0]
        assert change.key_values == {"id": 1}
        assert len(change.changes) == 2  # val and name changed


# ---------------------------------------------------------------------------
# compute_change_rate
# ---------------------------------------------------------------------------


class TestComputeChangeRate:
    def test_no_data(self):
        diff = DataDiff("t", ["id"], 0, 0, 0, 0, 0, 0, {}, [], [], [], "")
        assert compute_change_rate(diff) == 0.0

    def test_all_added(self):
        diff = DataDiff("t", ["id"], 10, 0, 0, 0, 0, 10, {}, [], [], [], "")
        rate = compute_change_rate(diff)
        assert rate == 100.0

    def test_no_changes(self):
        diff = DataDiff("t", ["id"], 0, 0, 0, 10, 10, 10, {}, [], [], [], "")
        rate = compute_change_rate(diff)
        assert rate == 0.0

    def test_partial_changes(self):
        diff = DataDiff("t", ["id"], 2, 1, 3, 5, 10, 11, {}, [], [], [], "")
        rate = compute_change_rate(diff)
        assert 0 < rate < 100


# ---------------------------------------------------------------------------
# classify_change
# ---------------------------------------------------------------------------


class TestClassifyChange:
    def test_no_change(self):
        diff = DataDiff("t", ["id"], 0, 0, 0, 10, 10, 10, {}, [], [], [], "")
        assert classify_change(diff) == "no_change"

    def test_minor_update(self):
        # 1 change out of 50
        diff = DataDiff("t", ["id"], 0, 0, 1, 49, 50, 50, {}, [], [], [], "")
        assert classify_change(diff) == "minor_update"

    def test_major_update(self):
        # All rows changed
        diff = DataDiff("t", ["id"], 0, 0, 50, 0, 50, 50, {}, [], [], [], "")
        assert classify_change(diff) == "major_update"

    def test_data_refresh(self):
        # 90% removed and 90% added
        diff = DataDiff("t", ["id"], 45, 45, 0, 5, 50, 50, {}, [], [], [], "")
        assert classify_change(diff) == "data_refresh"

    def test_moderate_update(self):
        # About 10% changed
        diff = DataDiff("t", ["id"], 5, 5, 0, 90, 100, 100, {}, [], [], [], "")
        assert classify_change(diff) == "minor_update" or classify_change(diff) == "moderate_update"


# ---------------------------------------------------------------------------
# summarize_column_drift
# ---------------------------------------------------------------------------


class TestSummarizeColumnDrift:
    def test_numeric_column(self):
        old = [{"val": 10}, {"val": 20}, {"val": 30}]
        new = [{"val": 15}, {"val": 25}, {"val": 35}]
        result = summarize_column_drift(old, new, "val")
        assert result["old_mean"] == 20.0
        assert result["new_mean"] == 25.0
        assert result["mean_change"] == 5.0
        assert result["mean_change_pct"] == 25.0

    def test_categorical_column(self):
        old = [{"cat": "A"}, {"cat": "B"}, {"cat": "C"}]
        new = [{"cat": "A"}, {"cat": "D"}, {"cat": "E"}]
        result = summarize_column_drift(old, new, "cat")
        assert result["old_distinct"] == 3
        assert result["new_distinct"] == 3
        assert "D" in result["added_values"]
        assert "B" in result["removed_values"]

    def test_null_values_excluded(self):
        old = [{"val": 10}, {"val": None}, {"val": 30}]
        new = [{"val": 20}, {"val": 40}]
        result = summarize_column_drift(old, new, "val")
        assert result["old_count"] == 2
        assert result["new_count"] == 2

    def test_empty_old(self):
        result = summarize_column_drift([], [{"val": 10}], "val")
        assert result["old_count"] == 0
        assert result["new_count"] == 1

    def test_zero_mean_no_pct(self):
        old = [{"val": 0}, {"val": 0}]
        new = [{"val": 5}, {"val": 5}]
        result = summarize_column_drift(old, new, "val")
        assert "mean_change_pct" not in result  # division by zero skipped
