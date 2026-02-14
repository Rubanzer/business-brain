"""Data comparator — compares two snapshots of the same table to find changes.

Pure functions that detect added rows, removed rows, changed values,
and summarize the overall diff between two data snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnChange:
    """A single column-level change in a row."""
    column: str
    old_value: Any
    new_value: Any


@dataclass
class RowChange:
    """A changed row with its column-level diffs."""
    key_values: dict[str, Any]
    changes: list[ColumnChange]


@dataclass
class DataDiff:
    """Complete diff between two data snapshots."""
    table_name: str
    key_columns: list[str]
    added_rows: int
    removed_rows: int
    changed_rows: int
    unchanged_rows: int
    total_old: int
    total_new: int
    column_changes: dict[str, int]  # column -> count of changes
    sample_additions: list[dict]  # first N added rows
    sample_removals: list[dict]  # first N removed rows
    sample_changes: list[RowChange]  # first N changed rows
    summary: str


def compare_snapshots(
    old_rows: list[dict],
    new_rows: list[dict],
    key_columns: list[str],
    table_name: str = "unknown",
    sample_limit: int = 5,
) -> DataDiff:
    """Compare two snapshots and produce a structured diff.

    Args:
        old_rows: Previous data snapshot as list of dicts.
        new_rows: Current data snapshot as list of dicts.
        key_columns: Columns that uniquely identify a row.
        table_name: Name of the table being compared.
        sample_limit: Max samples to include for each change type.

    Returns:
        DataDiff with detailed comparison results.
    """
    if not key_columns:
        return DataDiff(
            table_name=table_name,
            key_columns=[],
            added_rows=len(new_rows),
            removed_rows=len(old_rows),
            changed_rows=0,
            unchanged_rows=0,
            total_old=len(old_rows),
            total_new=len(new_rows),
            column_changes={},
            sample_additions=new_rows[:sample_limit],
            sample_removals=old_rows[:sample_limit],
            sample_changes=[],
            summary=f"No key columns specified; treating all {len(new_rows)} new rows as additions and {len(old_rows)} old rows as removals.",
        )

    # Index old and new by key
    old_index = _build_index(old_rows, key_columns)
    new_index = _build_index(new_rows, key_columns)

    old_keys = set(old_index.keys())
    new_keys = set(new_index.keys())

    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    common_keys = old_keys & new_keys

    # Detect changes in common rows
    changed_rows_list: list[RowChange] = []
    column_change_counts: dict[str, int] = {}
    unchanged_count = 0

    for key in common_keys:
        old_row = old_index[key]
        new_row = new_index[key]
        changes = _diff_row(old_row, new_row, key_columns)
        if changes:
            key_values = {k: old_row.get(k) for k in key_columns}
            changed_rows_list.append(RowChange(key_values=key_values, changes=changes))
            for c in changes:
                column_change_counts[c.column] = column_change_counts.get(c.column, 0) + 1
        else:
            unchanged_count += 1

    # Build samples
    sample_additions = [new_index[k] for k in list(added_keys)[:sample_limit]]
    sample_removals = [old_index[k] for k in list(removed_keys)[:sample_limit]]
    sample_changes = changed_rows_list[:sample_limit]

    # Build summary
    parts = []
    if added_keys:
        parts.append(f"{len(added_keys)} rows added")
    if removed_keys:
        parts.append(f"{len(removed_keys)} rows removed")
    if changed_rows_list:
        parts.append(f"{len(changed_rows_list)} rows changed")
    if unchanged_count:
        parts.append(f"{unchanged_count} rows unchanged")
    if not parts:
        parts.append("No data")
    summary = f"{table_name}: " + ", ".join(parts) + "."

    if column_change_counts:
        top_cols = sorted(column_change_counts.items(), key=lambda x: -x[1])[:3]
        col_summary = ", ".join(f"{col} ({cnt})" for col, cnt in top_cols)
        summary += f" Most changed columns: {col_summary}."

    return DataDiff(
        table_name=table_name,
        key_columns=key_columns,
        added_rows=len(added_keys),
        removed_rows=len(removed_keys),
        changed_rows=len(changed_rows_list),
        unchanged_rows=unchanged_count,
        total_old=len(old_rows),
        total_new=len(new_rows),
        column_changes=column_change_counts,
        sample_additions=sample_additions,
        sample_removals=sample_removals,
        sample_changes=sample_changes,
        summary=summary,
    )


def compute_change_rate(diff: DataDiff) -> float:
    """Compute overall change rate as percentage.

    Returns 0-100 representing how much of the data changed.
    """
    total = diff.total_old + diff.total_new
    if total == 0:
        return 0.0
    changes = diff.added_rows + diff.removed_rows + diff.changed_rows
    # Normalize: max possible changes = total_old (all removed) + total_new (all added)
    max_changes = diff.total_old + diff.total_new
    return min(changes / max_changes * 100, 100.0)


def classify_change(diff: DataDiff) -> str:
    """Classify the nature of the change.

    Returns one of: 'no_change', 'minor_update', 'moderate_update',
    'major_update', 'data_refresh', 'schema_shift'.
    """
    if diff.added_rows == 0 and diff.removed_rows == 0 and diff.changed_rows == 0:
        return "no_change"

    rate = compute_change_rate(diff)

    # Data refresh: all rows replaced
    if diff.removed_rows > 0 and diff.added_rows > 0:
        if diff.removed_rows >= diff.total_old * 0.8 and diff.added_rows >= diff.total_new * 0.8:
            return "data_refresh"

    if rate < 5:
        return "minor_update"
    elif rate < 25:
        return "moderate_update"
    else:
        return "major_update"


def summarize_column_drift(
    old_rows: list[dict],
    new_rows: list[dict],
    column: str,
) -> dict:
    """Summarize how a specific column's values changed between snapshots.

    Returns stats about the column's value distribution change.
    """
    old_values = [r.get(column) for r in old_rows if r.get(column) is not None]
    new_values = [r.get(column) for r in new_rows if r.get(column) is not None]

    result: dict = {
        "column": column,
        "old_count": len(old_values),
        "new_count": len(new_values),
    }

    # Check if numeric
    old_nums = _to_floats(old_values)
    new_nums = _to_floats(new_values)

    if old_nums and new_nums:
        old_mean = sum(old_nums) / len(old_nums)
        new_mean = sum(new_nums) / len(new_nums)
        result["old_mean"] = round(old_mean, 4)
        result["new_mean"] = round(new_mean, 4)
        result["mean_change"] = round(new_mean - old_mean, 4)
        if old_mean != 0:
            result["mean_change_pct"] = round((new_mean - old_mean) / abs(old_mean) * 100, 2)
        result["old_min"] = min(old_nums)
        result["new_min"] = min(new_nums)
        result["old_max"] = max(old_nums)
        result["new_max"] = max(new_nums)
    else:
        # Categorical: count distinct values
        old_distinct = set(str(v) for v in old_values)
        new_distinct = set(str(v) for v in new_values)
        result["old_distinct"] = len(old_distinct)
        result["new_distinct"] = len(new_distinct)
        result["added_values"] = list(new_distinct - old_distinct)[:10]
        result["removed_values"] = list(old_distinct - new_distinct)[:10]

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_index(rows: list[dict], key_columns: list[str]) -> dict[tuple, dict]:
    """Index rows by composite key."""
    index: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row.get(k) for k in key_columns)
        index[key] = row
    return index


def _diff_row(
    old_row: dict,
    new_row: dict,
    key_columns: list[str],
) -> list[ColumnChange]:
    """Find column-level differences between two rows, excluding key columns."""
    changes = []
    all_cols = set(old_row.keys()) | set(new_row.keys())
    for col in sorted(all_cols):
        if col in key_columns:
            continue
        old_val = old_row.get(col)
        new_val = new_row.get(col)
        if old_val != new_val:
            changes.append(ColumnChange(column=col, old_value=old_val, new_value=new_val))
    return changes


def _to_floats(values: list) -> list[float]:
    """Try to convert values to floats. Returns empty if not numeric."""
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            return []
    return nums
