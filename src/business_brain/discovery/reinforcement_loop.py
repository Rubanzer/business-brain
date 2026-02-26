"""Reinforcement loop — computes weight adjustments from engagement data.

Reads engagement data (shown/deployed/dismissed counts) and produces
multiplier dicts that modulate the hardcoded scoring constants in the
quality gate and recommendation engine.

All multipliers are clamped to [MIN_MULTIPLIER, MAX_MULTIPLIER] and
use exponential moving average (EMA) smoothing against prior weights
to prevent wild swings.

Triggered automatically at the end of each discovery run.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import ReinforcementWeights
from business_brain.discovery.engagement_tracker import get_engagement_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

MIN_MULTIPLIER = 0.8
MAX_MULTIPLIER = 1.2
EMA_ALPHA = 0.3           # new_weight = (1-alpha)*old + alpha*computed
MIN_SHOWN_THRESHOLD = 5   # need at least this many "shown" events per key
MIN_TOTAL_EVENTS = 10     # need at least this many total events to run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    """Clamp a multiplier to the safe range."""
    return max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, value))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_dimension_multipliers(by_dimension: dict[str, dict]) -> dict[str, float]:
    """Compute multipliers for one dimension (analysis_type or severity).

    Algorithm:
    1. Compute engagement_rate per key: deployed / shown
    2. Compute weighted-average rate across all keys
    3. ratio = key_rate / avg_rate
    4. multiplier = 1.0 + (ratio - 1.0) * 0.2  (gentle slope)
    5. Clamp to [0.8, 1.2]

    Keys with < MIN_SHOWN_THRESHOLD shown events are skipped.
    If zero deployments exist, use dismiss_rate as a penalty signal instead.
    """
    if not by_dimension:
        return {}

    # Collect rates for keys with sufficient data
    rates: dict[str, float] = {}
    total_shown = 0
    total_deployed = 0

    for key, stats in by_dimension.items():
        shown = stats.get("shown", 0)
        deployed = stats.get("deployed", 0)
        if shown >= MIN_SHOWN_THRESHOLD:
            rates[key] = deployed / shown
            total_shown += shown
            total_deployed += deployed

    if not rates or total_shown == 0:
        return {}

    # Weighted average engagement rate
    avg_rate = total_deployed / total_shown

    if avg_rate == 0:
        # No deployments at all — use dismiss rate as penalty signal
        result = {}
        for key, stats in by_dimension.items():
            shown = stats.get("shown", 0)
            dismissed = stats.get("dismissed", 0)
            if shown >= MIN_SHOWN_THRESHOLD and dismissed > 0:
                dismiss_rate = dismissed / shown
                result[key] = _clamp(1.0 - dismiss_rate * 0.3)
        return result

    # Compute multiplier per key
    result: dict[str, float] = {}
    for key, rate in rates.items():
        ratio = rate / avg_rate if avg_rate > 0 else 1.0
        # Linear map: ratio=1 → 1.0, ratio=2 → 1.2, ratio=0 → 0.8
        multiplier = 1.0 + (ratio - 1.0) * 0.2
        result[key] = _clamp(multiplier)

    return result


def compute_weights(engagement_summary: dict) -> dict:
    """Compute raw weight adjustments from engagement data.

    Pure function (no DB access).

    Returns:
        {
            "analysis_type_multipliers": {"benchmark": 1.1, ...},
            "severity_multipliers": {"critical": 1.05, ...},
            "insight_type_multipliers": {"anomaly": 1.1, ...},
        }

    All values default to 1.0 if insufficient data.
    """
    by_analysis = engagement_summary.get("by_analysis_type", {})
    by_severity = engagement_summary.get("by_severity", {})

    return {
        "analysis_type_multipliers": _compute_dimension_multipliers(by_analysis),
        "severity_multipliers": _compute_dimension_multipliers(by_severity),
        # Insight types map to analysis types in engagement data
        "insight_type_multipliers": _compute_dimension_multipliers(by_analysis),
    }


# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------


def _apply_ema_smoothing(
    raw: dict,
    previous: ReinforcementWeights | None,
) -> dict:
    """Apply exponential moving average: new = (1-alpha)*old + alpha*raw.

    If no previous weights exist, raw values are used directly.
    """
    if previous is None:
        return raw

    smoothed = {}
    for dimension in ("analysis_type_multipliers", "severity_multipliers",
                      "insight_type_multipliers"):
        raw_dict = raw.get(dimension, {})
        prev_dict = getattr(previous, dimension, None) or {}

        merged: dict[str, float] = {}
        all_keys = set(raw_dict) | set(prev_dict)
        for key in all_keys:
            raw_val = raw_dict.get(key, 1.0)
            prev_val = prev_dict.get(key, 1.0)
            merged[key] = _clamp((1 - EMA_ALPHA) * prev_val + EMA_ALPHA * raw_val)

        smoothed[dimension] = merged

    return smoothed


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------


async def get_latest_weights(
    session: AsyncSession,
) -> ReinforcementWeights | None:
    """Fetch the most recent weight snapshot from DB.

    Returns None if no weights have been computed yet (safe default = all 1.0).
    """
    result = await session.execute(
        select(ReinforcementWeights)
        .order_by(ReinforcementWeights.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def get_multiplier(
    weights: ReinforcementWeights | None,
    dimension: str,
    key: str,
) -> float:
    """Get a specific multiplier from a weights record.

    Returns 1.0 if weights is None, dimension is missing, or key is missing.

    Args:
        weights: Latest ReinforcementWeights record, or None.
        dimension: "analysis_type_multipliers", "severity_multipliers",
                   or "insight_type_multipliers".
        key: The specific key (e.g., "benchmark", "critical").
    """
    if weights is None:
        return 1.0
    multipliers = getattr(weights, dimension, None) or {}
    return multipliers.get(key, 1.0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def update_weights(
    session: AsyncSession,
    discovery_run_id: str | None = None,
    period_days: int = 30,
) -> ReinforcementWeights | None:
    """Recompute reinforcement weights from recent engagement data.

    Steps:
    1. Get engagement summary for the last period_days
    2. If insufficient data (< MIN_TOTAL_EVENTS), skip
    3. Compute raw multipliers
    4. Apply EMA smoothing against previous weights
    5. Store new versioned record

    Fire-and-forget: logs errors, never raises.
    """
    try:
        summary = await get_engagement_summary(session, days=period_days)

        if summary.get("total_events", 0) < MIN_TOTAL_EVENTS:
            logger.info(
                "Reinforcement loop: skipping, only %d events (need %d+)",
                summary.get("total_events", 0),
                MIN_TOTAL_EVENTS,
            )
            return None

        raw = compute_weights(summary)

        previous = await get_latest_weights(session)
        smoothed = _apply_ema_smoothing(raw, previous)

        new_version = (previous.version + 1) if previous else 1

        record = ReinforcementWeights(
            version=new_version,
            analysis_type_multipliers=smoothed["analysis_type_multipliers"],
            severity_multipliers=smoothed["severity_multipliers"],
            insight_type_multipliers=smoothed["insight_type_multipliers"],
            engagement_summary=summary,
            discovery_run_id=discovery_run_id,
            period_days=period_days,
            total_events=summary.get("total_events", 0),
        )
        session.add(record)
        await session.flush()

        logger.info(
            "Reinforcement loop: computed v%d weights from %d events "
            "(analysis=%s, severity=%s)",
            new_version,
            summary["total_events"],
            smoothed["analysis_type_multipliers"],
            smoothed["severity_multipliers"],
        )
        return record

    except Exception:
        logger.exception("Reinforcement loop failed, continuing with previous weights")
        return None
