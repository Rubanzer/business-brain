"""Sanctity engine — data integrity validation across three layers.

Layer 1: Change tracking (data diffs across syncs)
Layer 2: Impossible value detection (column-type-aware validation)
Layer 3: Cross-source conflict detection (same metric, different sources)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import Insight, TableProfile
from business_brain.db.v3_models import (
    DataChangeLog,
    MetricThreshold,
    SanctityIssue,
    SourceMapping,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: Change Tracking
# ---------------------------------------------------------------------------


async def get_recent_changes(
    session: AsyncSession,
    table_name: str | None = None,
    limit: int = 50,
) -> list[DataChangeLog]:
    """Get recent data changes, optionally filtered by table."""
    query = select(DataChangeLog).order_by(DataChangeLog.detected_at.desc()).limit(limit)
    if table_name:
        query = query.where(DataChangeLog.table_name == table_name)
    result = await session.execute(query)
    return list(result.scalars().all())


async def detect_critical_changes(
    session: AsyncSession,
    table_name: str,
    threshold_pct: float = 20.0,
) -> list[SanctityIssue]:
    """Detect changes that exceed a significance threshold.

    A change is critical when the numeric value changed by more than threshold_pct%.
    """
    changes = await get_recent_changes(session, table_name, limit=200)
    issues: list[SanctityIssue] = []

    for change in changes:
        if change.change_type != "row_modified":
            continue
        if change.old_value is None or change.new_value is None:
            continue

        try:
            old_num = float(change.old_value)
            new_num = float(change.new_value)
            if old_num == 0:
                continue
            pct_change = abs((new_num - old_num) / old_num) * 100

            if pct_change >= threshold_pct:
                issue = SanctityIssue(
                    table_name=change.table_name,
                    column_name=change.column_name,
                    row_identifier=change.row_identifier,
                    issue_type="unauthorized_change",
                    severity="critical" if pct_change >= 50 else "warning",
                    description=(
                        f"Value in {change.column_name} changed by {pct_change:.1f}% "
                        f"(from {change.old_value} to {change.new_value})"
                    ),
                    current_value=change.new_value,
                    expected_range=f"~{change.old_value} (±{threshold_pct}%)",
                )
                issues.append(issue)
                session.add(issue)
        except (ValueError, TypeError):
            pass

    if issues:
        await session.flush()
    return issues


# ---------------------------------------------------------------------------
# Layer 2: Impossible Value Detection
# ---------------------------------------------------------------------------


async def check_impossible_values(
    session: AsyncSession,
    profiles: list[TableProfile],
) -> list[SanctityIssue]:
    """Check all profiled tables for impossible values using column classification
    and metric thresholds."""
    issues: list[SanctityIssue] = []

    # Load custom thresholds
    thresh_result = await session.execute(select(MetricThreshold))
    thresholds = {
        (t.table_name, t.column_name): t
        for t in thresh_result.scalars().all()
        if t.table_name and t.column_name
    }

    for profile in profiles:
        cls = profile.column_classification
        if not cls or "columns" not in cls:
            continue

        cols = cls["columns"]
        row_count = profile.row_count or 0

        for col_name, info in cols.items():
            sem_type = info.get("semantic_type", "")
            stats = info.get("stats")
            null_count = info.get("null_count", 0)
            samples = info.get("sample_values", [])

            # Custom threshold check
            threshold = thresholds.get((profile.table_name, col_name))
            if threshold and stats:
                _issues = _check_threshold(profile.table_name, col_name, stats, threshold)
                issues.extend(_issues)
                for issue in _issues:
                    session.add(issue)

            # Negative currency
            if sem_type == "numeric_currency" and stats and stats.get("min", 0) < 0:
                issue = SanctityIssue(
                    table_name=profile.table_name,
                    column_name=col_name,
                    issue_type="impossible_value",
                    severity="critical",
                    description=f"Negative values in currency column {col_name} (min={stats['min']})",
                    current_value=str(stats["min"]),
                    expected_range="≥ 0",
                )
                issues.append(issue)
                session.add(issue)

            # Percentage out of range
            if sem_type == "numeric_percentage" and stats:
                if stats.get("max", 0) > 100 or stats.get("min", 0) < 0:
                    issue = SanctityIssue(
                        table_name=profile.table_name,
                        column_name=col_name,
                        issue_type="impossible_value",
                        severity="critical",
                        description=(
                            f"Percentage column {col_name} has values outside 0-100 "
                            f"(min={stats.get('min')}, max={stats.get('max')})"
                        ),
                        current_value=f"min={stats.get('min')}, max={stats.get('max')}",
                        expected_range="0-100",
                    )
                    issues.append(issue)
                    session.add(issue)

            # Statistical outlier (> 4 sigma)
            if stats and "stdev" in stats and stats["stdev"] > 0 and "mean" in stats:
                mean = stats["mean"]
                stdev = stats["stdev"]
                for s in samples:
                    try:
                        v = float(str(s).replace(",", ""))
                        if abs(v - mean) > 4 * stdev:
                            issue = SanctityIssue(
                                table_name=profile.table_name,
                                column_name=col_name,
                                issue_type="statistical_outlier",
                                severity="warning",
                                description=(
                                    f"Value {v} in {col_name} is > 4σ from mean "
                                    f"(mean={mean:.2f}, σ={stdev:.2f})"
                                ),
                                current_value=str(v),
                                expected_range=f"{mean - 4*stdev:.2f} to {mean + 4*stdev:.2f}",
                            )
                            issues.append(issue)
                            session.add(issue)
                            break  # one per column
                    except (ValueError, TypeError):
                        pass

            # Null spike — historically 0% nulls but now has nulls
            if row_count > 0 and null_count > 0:
                null_pct = null_count / row_count * 100
                if null_pct > 20:
                    issue = SanctityIssue(
                        table_name=profile.table_name,
                        column_name=col_name,
                        issue_type="null_spike",
                        severity="warning",
                        description=f"{col_name} has {null_pct:.1f}% null values ({null_count}/{row_count})",
                        current_value=f"{null_pct:.1f}% null",
                    )
                    issues.append(issue)
                    session.add(issue)

    if issues:
        await session.flush()
    return issues


def _check_threshold(
    table_name: str,
    col_name: str,
    stats: dict,
    threshold: MetricThreshold,
) -> list[SanctityIssue]:
    """Check a column's stats against its configured threshold."""
    issues: list[SanctityIssue] = []

    val_min = stats.get("min")
    val_max = stats.get("max")

    # Critical range
    if threshold.critical_min is not None and val_min is not None and val_min < threshold.critical_min:
        issues.append(SanctityIssue(
            table_name=table_name,
            column_name=col_name,
            issue_type="impossible_value",
            severity="critical",
            description=(
                f"{col_name} min value {val_min} is below critical minimum "
                f"{threshold.critical_min} {threshold.unit or ''}"
            ),
            current_value=str(val_min),
            expected_range=f"≥ {threshold.critical_min} {threshold.unit or ''}",
        ))

    if threshold.critical_max is not None and val_max is not None and val_max > threshold.critical_max:
        issues.append(SanctityIssue(
            table_name=table_name,
            column_name=col_name,
            issue_type="impossible_value",
            severity="critical",
            description=(
                f"{col_name} max value {val_max} exceeds critical maximum "
                f"{threshold.critical_max} {threshold.unit or ''}"
            ),
            current_value=str(val_max),
            expected_range=f"≤ {threshold.critical_max} {threshold.unit or ''}",
        ))

    # Warning range
    if threshold.warning_min is not None and val_min is not None and val_min < threshold.warning_min:
        if not (threshold.critical_min is not None and val_min < threshold.critical_min):
            issues.append(SanctityIssue(
                table_name=table_name,
                column_name=col_name,
                issue_type="impossible_value",
                severity="warning",
                description=(
                    f"{col_name} min value {val_min} is below warning threshold "
                    f"{threshold.warning_min} {threshold.unit or ''}"
                ),
                current_value=str(val_min),
                expected_range=f"{threshold.normal_min}-{threshold.normal_max} {threshold.unit or ''}",
            ))

    if threshold.warning_max is not None and val_max is not None and val_max > threshold.warning_max:
        if not (threshold.critical_max is not None and val_max > threshold.critical_max):
            issues.append(SanctityIssue(
                table_name=table_name,
                column_name=col_name,
                issue_type="impossible_value",
                severity="warning",
                description=(
                    f"{col_name} max value {val_max} exceeds warning threshold "
                    f"{threshold.warning_max} {threshold.unit or ''}"
                ),
                current_value=str(val_max),
                expected_range=f"{threshold.normal_min}-{threshold.normal_max} {threshold.unit or ''}",
            ))

    return issues


# ---------------------------------------------------------------------------
# Layer 3: Cross-Source Conflict Detection
# ---------------------------------------------------------------------------


async def detect_cross_source_conflicts(
    session: AsyncSession,
) -> list[SanctityIssue]:
    """Check for conflicts between mapped data sources.

    Uses SourceMapping records to know which tables contain the same data,
    then compares values for common entities.
    """
    result = await session.execute(
        select(SourceMapping).where(SourceMapping.confirmed_by_user == True)  # noqa: E712
    )
    mappings = list(result.scalars().all())
    issues: list[SanctityIssue] = []

    for mapping in mappings:
        try:
            conflicts = await _check_mapping_conflicts(session, mapping)
            issues.extend(conflicts)
            for issue in conflicts:
                session.add(issue)
        except Exception:
            logger.exception(
                "Failed to check conflicts between %s and %s",
                mapping.source_a_table, mapping.source_b_table,
            )

    if issues:
        await session.flush()
    return issues


async def _check_mapping_conflicts(
    session: AsyncSession,
    mapping: SourceMapping,
) -> list[SanctityIssue]:
    """Check for value conflicts between two mapped sources."""
    import re

    issues: list[SanctityIssue] = []
    col_maps = mapping.column_mappings or []

    safe_a = re.sub(r"[^a-zA-Z0-9_]", "", mapping.source_a_table)
    safe_b = re.sub(r"[^a-zA-Z0-9_]", "", mapping.source_b_table)

    # Find identifier columns for joining
    id_mappings = [m for m in col_maps if "id" in m.get("canonical", "").lower()]
    if not id_mappings:
        return issues

    # Build a simple comparison query for numeric columns
    numeric_maps = [m for m in col_maps if m != id_mappings[0]]

    for num_map in numeric_maps[:3]:  # limit to 3 columns
        col_a = re.sub(r"[^a-zA-Z0-9_]", "", num_map.get("a", ""))
        col_b = re.sub(r"[^a-zA-Z0-9_]", "", num_map.get("b", ""))
        join_a = re.sub(r"[^a-zA-Z0-9_]", "", id_mappings[0].get("a", ""))
        join_b = re.sub(r"[^a-zA-Z0-9_]", "", id_mappings[0].get("b", ""))

        if not all([col_a, col_b, join_a, join_b]):
            continue

        try:
            query = (
                f'SELECT a."{join_a}" as entity, '
                f'a."{col_a}" as val_a, b."{col_b}" as val_b '
                f'FROM "{safe_a}" a '
                f'JOIN "{safe_b}" b ON a."{join_a}"::text = b."{join_b}"::text '
                f'WHERE a."{col_a}"::text != b."{col_b}"::text '
                f'LIMIT 10'
            )
            result = await session.execute(sql_text(query))
            conflicts = [dict(r._mapping) for r in result.fetchall()]

            for conflict in conflicts:
                issues.append(SanctityIssue(
                    table_name=safe_a,
                    column_name=col_a,
                    row_identifier=str(conflict.get("entity")),
                    issue_type="cross_source_conflict",
                    severity="warning",
                    description=(
                        f"{col_a} in {safe_a} shows '{conflict['val_a']}' "
                        f"but {col_b} in {safe_b} shows '{conflict['val_b']}' "
                        f"for entity '{conflict['entity']}'"
                    ),
                    current_value=str(conflict["val_a"]),
                    conflicting_source=safe_b,
                    conflicting_value=str(conflict["val_b"]),
                ))

        except Exception:
            logger.exception("Failed to compare %s.%s vs %s.%s", safe_a, col_a, safe_b, col_b)

    return issues


# ---------------------------------------------------------------------------
# Full sanctity check
# ---------------------------------------------------------------------------


async def run_sanctity_check(
    session: AsyncSession,
    profiles: list[TableProfile] | None = None,
) -> dict:
    """Run all sanctity checks and return a summary.

    Returns:
        Dict with issue counts by type and severity.
    """
    if profiles is None:
        from business_brain.db.discovery_models import TableProfile
        result = await session.execute(select(TableProfile))
        profiles = list(result.scalars().all())

    all_issues: list[SanctityIssue] = []

    # Layer 2: Impossible values
    impossible = await check_impossible_values(session, profiles)
    all_issues.extend(impossible)

    # Layer 3: Cross-source conflicts
    conflicts = await detect_cross_source_conflicts(session)
    all_issues.extend(conflicts)

    # Summarize
    summary = {
        "total": len(all_issues),
        "by_type": {},
        "by_severity": {"critical": 0, "warning": 0, "info": 0},
    }
    for issue in all_issues:
        summary["by_type"][issue.issue_type] = summary["by_type"].get(issue.issue_type, 0) + 1
        summary["by_severity"][issue.severity] = summary["by_severity"].get(issue.severity, 0) + 1

    return summary


async def get_open_issues(
    session: AsyncSession,
    limit: int = 50,
) -> list[SanctityIssue]:
    """Get all unresolved sanctity issues."""
    result = await session.execute(
        select(SanctityIssue)
        .where(SanctityIssue.resolved == False)  # noqa: E712
        .order_by(SanctityIssue.detected_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def resolve_issue(
    session: AsyncSession,
    issue_id: int,
    resolved_by: str,
    note: str | None = None,
) -> SanctityIssue | None:
    """Mark a sanctity issue as resolved."""
    result = await session.execute(
        select(SanctityIssue).where(SanctityIssue.id == issue_id)
    )
    issue = result.scalar_one_or_none()
    if not issue:
        return None

    issue.resolved = True
    issue.resolved_by = resolved_by
    issue.resolution_note = note
    await session.commit()
    return issue
