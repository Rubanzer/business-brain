"""Pattern memory — learns, stores, and matches data patterns for predictive alerting.

Example: SCADA shows KVA dropping while unit consumption stays high →
this was the signature before the last 3 furnace breakdowns → alert!
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.v3_models import Pattern, PatternMatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern Learning
# ---------------------------------------------------------------------------


async def learn_pattern(
    session: AsyncSession,
    name: str,
    source_table: str,
    conditions: list[dict],
    time_window_minutes: int = 15,
    description: str = "",
    occurrence: dict | None = None,
    created_by: str = "user",
) -> Pattern:
    """Store a new pattern learned from user labeling or auto-detection.

    Args:
        session: DB session.
        name: Human-readable name (e.g., "Pre-Breakdown SCADA Signature").
        source_table: Table to monitor.
        conditions: List of condition dicts:
            [{"column": "kva", "behavior": "decreasing", "magnitude": ">10%"},
             {"column": "unit_consumption", "behavior": "stable_or_increasing"}]
        time_window_minutes: Time window for pattern matching.
        description: Detailed description.
        occurrence: Optional initial occurrence dict: {start, end, outcome}.
        created_by: "user" or "auto_detected".

    Returns:
        The created Pattern record.
    """
    occurrences = [occurrence] if occurrence else []

    pattern = Pattern(
        name=name,
        description=description,
        source_tables=[source_table],
        conditions=conditions,
        time_window_minutes=time_window_minutes,
        similarity_threshold=0.75,
        historical_occurrences=occurrences,
        confidence=0.5 if created_by == "auto_detected" else 0.7,
        created_by=created_by,
        active=True,
    )
    session.add(pattern)
    await session.commit()
    await session.refresh(pattern)

    logger.info("Learned new pattern: %s (table: %s, %d conditions)", name, source_table, len(conditions))
    return pattern


async def add_occurrence(
    session: AsyncSession,
    pattern_id: str,
    start: str,
    end: str,
    outcome: str,
) -> Pattern | None:
    """Add a historical occurrence to an existing pattern."""
    result = await session.execute(select(Pattern).where(Pattern.id == pattern_id))
    pattern = result.scalar_one_or_none()
    if not pattern:
        return None

    occurrences = list(pattern.historical_occurrences or [])
    occurrences.append({"start": start, "end": end, "outcome": outcome})
    pattern.historical_occurrences = occurrences

    # Boost confidence with more occurrences
    pattern.confidence = min(0.95, pattern.confidence + 0.05)
    await session.commit()
    return pattern


# ---------------------------------------------------------------------------
# Pattern Matching
# ---------------------------------------------------------------------------


async def check_patterns(
    session: AsyncSession,
    table_name: str,
) -> list[PatternMatch]:
    """Check all active patterns for the given table against current data.

    Returns list of PatternMatch records for patterns that matched.
    """
    result = await session.execute(
        select(Pattern).where(
            Pattern.active == True,  # noqa: E712
        )
    )
    patterns = list(result.scalars().all())

    # Filter to patterns that monitor this table
    relevant = [p for p in patterns if table_name in (p.source_tables or [])]
    if not relevant:
        return []

    matches: list[PatternMatch] = []

    for pattern in relevant:
        try:
            score = await _match_pattern(session, pattern, table_name)
            if score >= pattern.similarity_threshold:
                match = PatternMatch(
                    pattern_id=pattern.id,
                    similarity_score=score,
                    data_snapshot=await _get_data_snapshot(session, table_name, pattern),
                )
                session.add(match)

                # Update pattern stats
                pattern.last_matched_at = datetime.now(timezone.utc)
                pattern.match_count = (pattern.match_count or 0) + 1

                matches.append(match)
                logger.info(
                    "Pattern '%s' matched in %s (score: %.2f)",
                    pattern.name, table_name, score,
                )
        except Exception:
            logger.exception("Failed to check pattern '%s' on table '%s'", pattern.name, table_name)

    if matches:
        await session.flush()

    return matches


async def _match_pattern(
    session: AsyncSession,
    pattern: Pattern,
    table_name: str,
) -> float:
    """Compute similarity score between current data and a stored pattern.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    conditions = pattern.conditions or []
    if not conditions:
        return 0.0

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table_name)

    # Get recent data points for pattern comparison
    # Use a window of N rows based on time_window_minutes and data density
    window_size = max(5, pattern.time_window_minutes)

    columns_needed = [c["column"] for c in conditions if "column" in c]
    if not columns_needed:
        return 0.0

    safe_cols = [re.sub(r"[^a-zA-Z0-9_]", "", c) for c in columns_needed]
    col_list = ", ".join(f'"{c}"' for c in safe_cols)

    try:
        query = f'SELECT {col_list} FROM "{safe_table}" ORDER BY ctid DESC LIMIT {window_size}'
        result = await session.execute(sql_text(query))
        rows = [dict(r._mapping) for r in result.fetchall()]
    except Exception:
        return 0.0

    if len(rows) < 3:
        return 0.0

    # Reverse to chronological order
    rows.reverse()

    # Score each condition
    condition_scores: list[float] = []

    for condition in conditions:
        col = condition.get("column", "")
        behavior = condition.get("behavior", "")
        magnitude = condition.get("magnitude", "")

        safe_col = re.sub(r"[^a-zA-Z0-9_]", "", col)
        values = []
        for row in rows:
            v = row.get(safe_col)
            if v is not None:
                try:
                    values.append(float(str(v).replace(",", "")))
                except (ValueError, TypeError):
                    pass

        if len(values) < 3:
            condition_scores.append(0.0)
            continue

        score = _score_behavior(values, behavior, magnitude)
        condition_scores.append(score)

    if not condition_scores:
        return 0.0

    # All conditions must match simultaneously (AND logic)
    # Average score, but penalize if any condition scores < 0.3
    min_score = min(condition_scores)
    if min_score < 0.3:
        return min_score  # Fails if any condition doesn't match at all

    return sum(condition_scores) / len(condition_scores)


def _score_behavior(values: list[float], behavior: str, magnitude: str) -> float:
    """Score how well a series of values matches an expected behavior.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not values or len(values) < 2:
        return 0.0

    # Calculate trend
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    avg_diff = sum(diffs) / len(diffs)
    first_val = values[0] if values[0] != 0 else 1
    pct_change = (values[-1] - values[0]) / abs(first_val) * 100

    if behavior == "decreasing":
        if avg_diff >= 0:
            return 0.0
        score = min(1.0, abs(pct_change) / 20)  # Scale: 20% decrease = full score
        # Check magnitude requirement
        if magnitude:
            required_pct = _parse_magnitude(magnitude)
            if abs(pct_change) < required_pct:
                score *= 0.5
        return score

    elif behavior == "increasing":
        if avg_diff <= 0:
            return 0.0
        score = min(1.0, abs(pct_change) / 20)
        if magnitude:
            required_pct = _parse_magnitude(magnitude)
            if abs(pct_change) < required_pct:
                score *= 0.5
        return score

    elif behavior in ("stable", "stable_or_increasing"):
        # Stable means low variance
        if len(values) >= 2:
            import statistics
            cv = statistics.stdev(values) / (statistics.mean(values) or 1) * 100
            if behavior == "stable":
                return max(0.0, 1.0 - cv / 20)  # CV < 5% = good, > 20% = bad
            else:  # stable_or_increasing
                if avg_diff >= 0:
                    return max(0.5, 1.0 - cv / 30)
                else:
                    return max(0.0, 1.0 - cv / 20 - abs(pct_change) / 30)
        return 0.5

    elif behavior == "spike":
        # Check if any value deviates significantly from the median of all values
        import statistics
        median_val = statistics.median(values)
        if median_val == 0:
            median_val = 1
        max_deviation = max(abs(v - median_val) for v in values)
        deviation_pct = max_deviation / abs(median_val) * 100
        # A spike means at least one value is >50% away from the median
        if deviation_pct > 50:
            return min(1.0, deviation_pct / 200)
        return 0.0

    return 0.0


def _parse_magnitude(magnitude: str) -> float:
    """Parse a magnitude string like '>10%' into a float percentage."""
    import re
    match = re.search(r"(\d+(?:\.\d+)?)", magnitude)
    if match:
        return float(match.group(1))
    return 0.0


async def _get_data_snapshot(
    session: AsyncSession,
    table_name: str,
    pattern: Pattern,
) -> dict:
    """Get a snapshot of current data for the matched pattern."""
    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table_name)
    conditions = pattern.conditions or []
    columns_needed = [c["column"] for c in conditions if "column" in c]

    if not columns_needed:
        return {}

    safe_cols = [re.sub(r"[^a-zA-Z0-9_]", "", c) for c in columns_needed]
    col_list = ", ".join(f'"{c}"' for c in safe_cols)

    try:
        query = f'SELECT {col_list} FROM "{safe_table}" ORDER BY ctid DESC LIMIT 5'
        result = await session.execute(sql_text(query))
        rows = [dict(r._mapping) for r in result.fetchall()]
        return {"recent_rows": rows, "matched_at": datetime.now(timezone.utc).isoformat()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Pattern Feedback
# ---------------------------------------------------------------------------


async def confirm_match(
    session: AsyncSession,
    match_id: int,
    outcome: str,
) -> PatternMatch | None:
    """Confirm or reject a pattern match, adjusting pattern confidence.

    Args:
        match_id: The PatternMatch ID.
        outcome: "confirmed_breakdown" / "false_positive" / other.
    """
    result = await session.execute(select(PatternMatch).where(PatternMatch.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        return None

    match.outcome = outcome

    # Adjust pattern confidence
    pattern_result = await session.execute(select(Pattern).where(Pattern.id == match.pattern_id))
    pattern = pattern_result.scalar_one_or_none()
    if pattern:
        if "confirmed" in outcome or "true" in outcome:
            # Positive feedback: boost confidence
            pattern.confidence = min(0.99, pattern.confidence + 0.05)
            # Add to historical occurrences
            occurrences = list(pattern.historical_occurrences or [])
            occurrences.append({
                "outcome": outcome,
                "matched_at": match.matched_at.isoformat() if match.matched_at else None,
                "similarity_score": match.similarity_score,
            })
            pattern.historical_occurrences = occurrences
        elif "false" in outcome:
            # Negative feedback: decrease confidence
            pattern.confidence = max(0.1, pattern.confidence - 0.1)
            pattern.false_positive_count = (pattern.false_positive_count or 0) + 1

    await session.commit()
    return match


# ---------------------------------------------------------------------------
# Pattern CRUD
# ---------------------------------------------------------------------------


async def get_all_patterns(session: AsyncSession) -> list[Pattern]:
    """Get all patterns."""
    result = await session.execute(select(Pattern).order_by(Pattern.created_at.desc()))
    return list(result.scalars().all())


async def get_pattern(session: AsyncSession, pattern_id: str) -> Pattern | None:
    """Get a single pattern."""
    result = await session.execute(select(Pattern).where(Pattern.id == pattern_id))
    return result.scalar_one_or_none()


async def get_pattern_matches(session: AsyncSession, pattern_id: str, limit: int = 20) -> list[PatternMatch]:
    """Get match history for a pattern."""
    result = await session.execute(
        select(PatternMatch)
        .where(PatternMatch.pattern_id == pattern_id)
        .order_by(PatternMatch.matched_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def delete_pattern(session: AsyncSession, pattern_id: str) -> bool:
    """Delete a pattern."""
    pattern = await get_pattern(session, pattern_id)
    if not pattern:
        return False
    await session.delete(pattern)
    await session.commit()
    return True
