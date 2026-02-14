"""Schema change detection — compares current profile columns against previous snapshot.

Detects:
- New columns added to a table
- Columns removed from a table
- Semantic type changes for existing columns
"""

from __future__ import annotations

import logging
import uuid

from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)


def detect_schema_changes(
    current_profiles: list[TableProfile],
    previous_profiles: list[TableProfile],
) -> list[Insight]:
    """Compare current and previous profiles to find schema changes.

    Args:
        current_profiles: Profiles from the current discovery run.
        previous_profiles: Profiles from the prior run (or stored snapshots).

    Returns:
        List of Insight objects describing schema changes.
    """
    insights: list[Insight] = []

    prev_map = {p.table_name: p for p in previous_profiles}

    for profile in current_profiles:
        prev = prev_map.get(profile.table_name)
        if not prev:
            continue  # New table — not a schema *change*

        try:
            insights.extend(_compare_schemas(profile, prev))
        except Exception:
            logger.exception("Schema comparison failed for %s", profile.table_name)

    return insights


def _get_columns(profile: TableProfile) -> dict[str, dict]:
    """Extract column dict from a profile's classification."""
    cls = profile.column_classification or {}
    return cls.get("columns", {})


def _compare_schemas(
    current: TableProfile,
    previous: TableProfile,
) -> list[Insight]:
    """Compare two profiles of the same table for schema changes."""
    results: list[Insight] = []
    table = current.table_name

    cur_cols = _get_columns(current)
    prev_cols = _get_columns(previous)

    cur_names = set(cur_cols.keys())
    prev_names = set(prev_cols.keys())

    # New columns
    added = cur_names - prev_names
    if added:
        results.append(Insight(
            id=str(uuid.uuid4()),
            insight_type="schema_change",
            severity="info",
            impact_score=30,
            title=f"New columns in {table}",
            description=(
                f"{len(added)} new column(s) detected: {', '.join(sorted(added))}. "
                f"These may contain new data worth analyzing."
            ),
            source_tables=[table],
            source_columns=sorted(added),
            evidence={
                "change_type": "columns_added",
                "columns": sorted(added),
                "types": {c: cur_cols[c].get("semantic_type") for c in added},
            },
            suggested_actions=[
                f"Review new columns {', '.join(sorted(added))} for analytical value",
                "Update dashboards and reports to include new data",
            ],
        ))

    # Removed columns
    removed = prev_names - cur_names
    if removed:
        results.append(Insight(
            id=str(uuid.uuid4()),
            insight_type="schema_change",
            severity="warning",
            impact_score=50,
            title=f"Columns removed from {table}",
            description=(
                f"{len(removed)} column(s) no longer present: {', '.join(sorted(removed))}. "
                f"This may break existing reports or analyses."
            ),
            source_tables=[table],
            source_columns=sorted(removed),
            evidence={
                "change_type": "columns_removed",
                "columns": sorted(removed),
                "previous_types": {c: prev_cols[c].get("semantic_type") for c in removed},
            },
            suggested_actions=[
                "Check if column removal was intentional",
                "Update affected reports and queries",
            ],
        ))

    # Type changes
    common = cur_names & prev_names
    type_changes = []
    for col in sorted(common):
        cur_type = cur_cols[col].get("semantic_type")
        prev_type = prev_cols[col].get("semantic_type")
        if cur_type and prev_type and cur_type != prev_type:
            type_changes.append({
                "column": col,
                "from": prev_type,
                "to": cur_type,
            })

    if type_changes:
        cols_changed = [tc["column"] for tc in type_changes]
        results.append(Insight(
            id=str(uuid.uuid4()),
            insight_type="schema_change",
            severity="info",
            impact_score=25,
            title=f"Column type changes in {table}",
            description=(
                f"{len(type_changes)} column(s) changed semantic type: "
                + ", ".join(f"{tc['column']} ({tc['from']} → {tc['to']})" for tc in type_changes)
            ),
            source_tables=[table],
            source_columns=cols_changed,
            evidence={
                "change_type": "type_changed",
                "changes": type_changes,
            },
            suggested_actions=[
                "Verify that type changes reflect actual data changes",
                "Update column-specific analysis logic if needed",
            ],
        ))

    return results
