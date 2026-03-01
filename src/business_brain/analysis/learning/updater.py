"""Learning Updater — EMA-based parameter adjustment from feedback.

Adjusts:
- Interestingness weights (surprise, magnitude, variance, stability, coverage)
- Agent calibration (per-agent confidence adjustments)
- Operation preferences (per-operation boost/penalty)
- Tier budget allocation (learn which tiers produce useful findings)

Uses EMA smoothing from discovery/reinforcement_loop.py pattern.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.models import AnalysisFeedback, AnalysisResult, LearningState
from business_brain.analysis.track1.scorer import DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EMA config
# ---------------------------------------------------------------------------

_EMA_ALPHA = 0.3  # Smoothing factor: higher = faster adaptation
_MIN_FEEDBACK = 5  # Don't update until we have this many feedbacks
_CLAMP_LOW = 0.5
_CLAMP_HIGH = 1.5


def _clamp(value: float, lo: float = _CLAMP_LOW, hi: float = _CLAMP_HIGH) -> float:
    return max(lo, min(hi, value))


def _ema(old: float, new: float, alpha: float = _EMA_ALPHA) -> float:
    return (1 - alpha) * old + alpha * new


# ---------------------------------------------------------------------------
# Feedback aggregation
# ---------------------------------------------------------------------------


async def _aggregate_feedback(session: AsyncSession) -> dict[str, Any]:
    """Aggregate recent feedback signals."""
    # Count feedback by type
    result = await session.execute(
        select(
            AnalysisFeedback.feedback_type,
            func.count(AnalysisFeedback.id),
        ).group_by(AnalysisFeedback.feedback_type)
    )
    type_counts = {row[0]: row[1] for row in result.all()}
    total = sum(type_counts.values())

    if total < _MIN_FEEDBACK:
        return {"total": total, "ready": False}

    # Join feedback with results to get operation-level signals
    result = await session.execute(
        select(
            AnalysisResult.operation_type,
            AnalysisFeedback.feedback_type,
            func.count(AnalysisFeedback.id),
        )
        .join(AnalysisResult, AnalysisResult.id == AnalysisFeedback.result_id)
        .group_by(AnalysisResult.operation_type, AnalysisFeedback.feedback_type)
    )
    operation_feedback: dict[str, dict[str, int]] = {}
    for op, fb_type, count in result.all():
        operation_feedback.setdefault(op, {})
        operation_feedback[op][fb_type] = count

    # Tier-level aggregation
    result = await session.execute(
        select(
            AnalysisResult.tier,
            AnalysisFeedback.feedback_type,
            func.count(AnalysisFeedback.id),
        )
        .join(AnalysisResult, AnalysisResult.id == AnalysisFeedback.result_id)
        .group_by(AnalysisResult.tier, AnalysisFeedback.feedback_type)
    )
    tier_feedback: dict[int, dict[str, int]] = {}
    for tier, fb_type, count in result.all():
        tier_feedback.setdefault(tier, {})
        tier_feedback[tier][fb_type] = count

    return {
        "total": total,
        "ready": True,
        "type_counts": type_counts,
        "operation_feedback": operation_feedback,
        "tier_feedback": tier_feedback,
    }


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def _compute_operation_preferences(
    operation_feedback: dict[str, dict[str, int]],
    prev_prefs: dict[str, float] | None,
) -> dict[str, float]:
    """Compute per-operation multipliers from feedback."""
    prefs = dict(prev_prefs) if prev_prefs else {}

    for op, fb in operation_feedback.items():
        useful = fb.get("useful", 0)
        not_useful = fb.get("not_useful", 0) + fb.get("wrong", 0)
        total = useful + not_useful
        if total == 0:
            continue

        # Raw signal: useful fraction → 0.5-1.5 range
        raw = 0.5 + (useful / total)

        old = prefs.get(op, 1.0)
        prefs[op] = _clamp(_ema(old, raw))

    return prefs


def _compute_tier_budgets(
    tier_feedback: dict[int, dict[str, int]],
    prev_budgets: dict[str, int] | None,
) -> dict[str, int]:
    """Adjust tier budgets based on which tiers produce useful findings."""
    defaults = {"2": 100, "3": 50, "4": 50}
    budgets = dict(prev_budgets) if prev_budgets else defaults

    for tier, fb in tier_feedback.items():
        if tier <= 1:
            continue  # Tier 0+1 are exhaustive, no budget
        useful = fb.get("useful", 0)
        not_useful = fb.get("not_useful", 0) + fb.get("wrong", 0)
        total = useful + not_useful
        if total < 3:
            continue

        useful_rate = useful / total
        key = str(tier)
        old = budgets.get(key, defaults.get(key, 50))

        # Scale budget: high useful rate → increase, low → decrease
        scale = 0.8 + 0.4 * useful_rate  # 0.8-1.2
        budgets[key] = max(10, min(200, int(_ema(old, old * scale))))

    return budgets


def _compute_interestingness_weights(
    type_counts: dict[str, int],
    prev_weights: dict[str, float] | None,
) -> dict[str, float]:
    """Adjust interestingness sub-score weights based on feedback patterns."""
    weights = dict(prev_weights) if prev_weights else dict(DEFAULT_WEIGHTS)

    total = sum(type_counts.values())
    if total < _MIN_FEEDBACK:
        return weights

    useful_rate = type_counts.get("useful", 0) / total
    wrong_rate = type_counts.get("wrong", 0) / total

    # If too many wrong predictions, boost stability weight
    if wrong_rate > 0.3:
        weights["stability"] = _clamp(_ema(weights.get("stability", 0.15), 0.25), 0.1, 0.4)

    # If most are useful, lean toward surprise (more novel findings)
    if useful_rate > 0.7:
        weights["surprise"] = _clamp(_ema(weights.get("surprise", 0.3), 0.35), 0.1, 0.5)

    # Normalize to sum=1
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    return weights


# ---------------------------------------------------------------------------
# Main update
# ---------------------------------------------------------------------------


async def update_learning_state(session: AsyncSession) -> LearningState | None:
    """Aggregate feedback and compute new learning parameters.

    Returns the new LearningState, or None if not enough feedback.
    """
    aggregated = await _aggregate_feedback(session)
    if not aggregated.get("ready"):
        logger.info("Learning: not enough feedback (%d), skipping", aggregated.get("total", 0))
        return None

    # Load previous state
    result = await session.execute(
        select(LearningState).order_by(LearningState.version.desc()).limit(1)
    )
    prev = result.scalar_one_or_none()
    prev_version = prev.version if prev else 0

    # Compute new parameters
    new_weights = _compute_interestingness_weights(
        aggregated["type_counts"],
        prev.interestingness_weights if prev else None,
    )
    new_op_prefs = _compute_operation_preferences(
        aggregated["operation_feedback"],
        prev.operation_preferences if prev else None,
    )
    new_tier_budgets = _compute_tier_budgets(
        aggregated["tier_feedback"],
        prev.tier_budgets if prev else None,
    )

    state = LearningState(
        version=prev_version + 1,
        interestingness_weights=new_weights,
        agent_calibration=prev.agent_calibration if prev else None,
        operation_preferences=new_op_prefs,
        tier_budgets=new_tier_budgets,
        feedback_count=aggregated["total"],
    )
    session.add(state)
    await session.flush()

    logger.info(
        "Learning: v%d computed from %d feedbacks — weights=%s, tier_budgets=%s",
        state.version, aggregated["total"], new_weights, new_tier_budgets,
    )
    return state
