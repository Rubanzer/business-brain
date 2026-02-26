"""Data freshness tracking — detects stale tables that haven't changed between runs.

Pure function that compares data_hash fields between current and previous profiles
to identify tables with unchanged data.
"""

from __future__ import annotations

import logging
import uuid

from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)


def detect_stale_tables(
    current_profiles: list[TableProfile],
    previous_profiles: list[TableProfile],
    stale_threshold: int = 1,
) -> list[Insight]:
    """Compare data hashes to find tables that haven't changed.

    Args:
        current_profiles: Profiles from the current discovery run.
        previous_profiles: Profiles from the prior run.
        stale_threshold: Minimum number of unchanged runs to consider stale.
            Default 1 means flag on the first run where data is unchanged.

    Returns:
        List of staleness warning insights.
    """
    insights: list[Insight] = []
    prev_map = {p.table_name: p for p in previous_profiles}

    for profile in current_profiles:
        prev = prev_map.get(profile.table_name)
        if not prev:
            continue  # New table, can't compare

        cur_hash = profile.data_hash
        prev_hash = prev.data_hash

        if not cur_hash or not prev_hash:
            continue  # No hashes to compare

        if cur_hash == prev_hash:
            insights.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="data_quality",
                severity="info",
                impact_score=15,
                title=f"Stale data in {profile.table_name}",
                description=(
                    f"Table {profile.table_name} data is unchanged since last scan "
                    f"(hash: {cur_hash[:8]}..., {profile.row_count or 0} rows). "
                    f"Data pipeline for this table is not delivering new data."
                ),
                source_tables=[profile.table_name],
                source_columns=[],
                evidence={
                    "data_hash": cur_hash,
                    "row_count": profile.row_count,
                    "pattern_type": "stale_data",
                },
                suggested_actions=[
                    f"Check if {profile.table_name} data pipeline is running",
                    "Verify the source system is active",
                ],
            ))

    return insights


def compute_freshness_score(
    current_profiles: list[TableProfile],
    previous_profiles: list[TableProfile],
) -> dict:
    """Compute an overall freshness score across all tables.

    Returns:
        Dict with score (0-100), stale_count, fresh_count, unknown_count.
    """
    prev_map = {p.table_name: p for p in previous_profiles}

    stale = 0
    fresh = 0
    unknown = 0

    for profile in current_profiles:
        prev = prev_map.get(profile.table_name)
        if not prev or not profile.data_hash or not prev.data_hash:
            unknown += 1
            continue

        if profile.data_hash == prev.data_hash:
            stale += 1
        else:
            fresh += 1

    total = stale + fresh
    score = int((fresh / total) * 100) if total > 0 else 100

    return {
        "score": score,
        "stale_count": stale,
        "fresh_count": fresh,
        "unknown_count": unknown,
        "total_tables": len(current_profiles),
    }
