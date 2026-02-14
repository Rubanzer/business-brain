"""Metric goal tracker — set targets and track progress toward them.

Pure functions for defining goals, evaluating progress, and
classifying status. No DB or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Goal:
    """A metric goal definition."""
    metric_name: str
    target_value: float
    direction: str  # "above", "below", "between"
    target_min: float | None = None  # for "between" direction
    deadline: str | None = None  # ISO date string
    baseline: float | None = None  # starting value


@dataclass
class GoalProgress:
    """Progress toward a goal."""
    metric_name: str
    current_value: float
    target_value: float
    direction: str
    progress_pct: float  # 0-100 (can exceed 100)
    status: str  # "on_track", "at_risk", "behind", "achieved", "exceeded"
    remaining: float
    summary: str


def evaluate_goal(goal: Goal, current_value: float) -> GoalProgress:
    """Evaluate progress toward a goal.

    Args:
        goal: The goal definition.
        current_value: Current metric value.

    Returns:
        GoalProgress with status and details.
    """
    if goal.direction == "above":
        return _evaluate_above(goal, current_value)
    elif goal.direction == "below":
        return _evaluate_below(goal, current_value)
    elif goal.direction == "between":
        return _evaluate_between(goal, current_value)
    else:
        return GoalProgress(
            metric_name=goal.metric_name,
            current_value=current_value,
            target_value=goal.target_value,
            direction=goal.direction,
            progress_pct=0.0,
            status="behind",
            remaining=goal.target_value - current_value,
            summary=f"Unknown goal direction: {goal.direction}",
        )


def evaluate_goals(goals: list[Goal], current_values: dict[str, float]) -> list[GoalProgress]:
    """Evaluate multiple goals at once.

    Args:
        goals: List of goal definitions.
        current_values: Dict mapping metric_name to current value.

    Returns:
        List of GoalProgress, one per goal.
    """
    results = []
    for goal in goals:
        value = current_values.get(goal.metric_name)
        if value is not None:
            results.append(evaluate_goal(goal, value))
        else:
            results.append(GoalProgress(
                metric_name=goal.metric_name,
                current_value=0.0,
                target_value=goal.target_value,
                direction=goal.direction,
                progress_pct=0.0,
                status="behind",
                remaining=goal.target_value,
                summary=f"No current value available for {goal.metric_name}.",
            ))
    return results


def compute_overall_health(progress_list: list[GoalProgress]) -> dict:
    """Compute overall goal health summary.

    Returns dict with counts by status and overall health score.
    """
    if not progress_list:
        return {"total": 0, "health_score": 0.0, "status_counts": {}}

    status_counts: dict[str, int] = {}
    for p in progress_list:
        status_counts[p.status] = status_counts.get(p.status, 0) + 1

    # Health score: achieved/exceeded = 100, on_track = 75, at_risk = 40, behind = 10
    weights = {"achieved": 100, "exceeded": 100, "on_track": 75, "at_risk": 40, "behind": 10}
    total_weight = sum(weights.get(p.status, 0) for p in progress_list)
    health_score = total_weight / len(progress_list)

    return {
        "total": len(progress_list),
        "health_score": round(health_score, 1),
        "status_counts": status_counts,
        "achieved_pct": round(
            (status_counts.get("achieved", 0) + status_counts.get("exceeded", 0))
            / len(progress_list) * 100,
            1,
        ),
    }


def classify_trend(values: list[float], target: float, direction: str) -> str:
    """Classify the trend of metric values toward the goal.

    Args:
        values: Historical values (oldest first).
        target: The target value.
        direction: "above" or "below".

    Returns:
        One of: "improving", "stable", "declining", "volatile".
    """
    if len(values) < 3:
        return "stable"

    # Compute direction of change for consecutive pairs
    diffs = [values[i+1] - values[i] for i in range(len(values) - 1)]

    # Count positive and negative diffs
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    total = len(diffs)

    # Check volatility: many direction changes
    direction_changes = sum(
        1 for i in range(len(diffs) - 1)
        if (diffs[i] > 0) != (diffs[i+1] > 0)
    )
    if direction_changes >= total * 0.6:
        return "volatile"

    # For "above" goals, positive movement is improving
    if direction == "above":
        if pos >= total * 0.6:
            return "improving"
        elif neg >= total * 0.6:
            return "declining"
    elif direction == "below":
        if neg >= total * 0.6:
            return "improving"
        elif pos >= total * 0.6:
            return "declining"

    return "stable"


def format_goal_summary(progress: GoalProgress) -> str:
    """Format a human-readable summary for a goal's progress."""
    status_emoji = {
        "achieved": "[OK]",
        "exceeded": "[++]",
        "on_track": "[->]",
        "at_risk": "[!]",
        "behind": "[X]",
    }
    icon = status_emoji.get(progress.status, "[-]")
    return (
        f"{icon} {progress.metric_name}: "
        f"{progress.current_value:.1f} / {progress.target_value:.1f} "
        f"({progress.progress_pct:.0f}%) — {progress.status}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _evaluate_above(goal: Goal, current: float) -> GoalProgress:
    """Evaluate goal where target is to be above a value."""
    baseline = goal.baseline if goal.baseline is not None else 0.0
    target = goal.target_value
    remaining = target - current

    if target == baseline:
        progress = 100.0 if current >= target else 0.0
    else:
        progress = (current - baseline) / (target - baseline) * 100

    if current >= target * 1.1:
        status = "exceeded"
    elif current >= target:
        status = "achieved"
    elif progress >= 70:
        status = "on_track"
    elif progress >= 40:
        status = "at_risk"
    else:
        status = "behind"

    summary = (
        f"{goal.metric_name} is at {current:.1f}, target is {target:.1f} "
        f"({progress:.0f}% progress). Status: {status}."
    )

    return GoalProgress(
        metric_name=goal.metric_name,
        current_value=current,
        target_value=target,
        direction="above",
        progress_pct=round(progress, 1),
        status=status,
        remaining=round(remaining, 4),
        summary=summary,
    )


def _evaluate_below(goal: Goal, current: float) -> GoalProgress:
    """Evaluate goal where target is to be below a value."""
    baseline = goal.baseline if goal.baseline is not None else goal.target_value * 2
    target = goal.target_value
    remaining = current - target

    if baseline == target:
        progress = 100.0 if current <= target else 0.0
    else:
        # Progress is reduction from baseline toward target
        progress = (baseline - current) / (baseline - target) * 100

    if current <= target * 0.9:
        status = "exceeded"
    elif current <= target:
        status = "achieved"
    elif progress >= 70:
        status = "on_track"
    elif progress >= 40:
        status = "at_risk"
    else:
        status = "behind"

    summary = (
        f"{goal.metric_name} is at {current:.1f}, target is below {target:.1f} "
        f"({progress:.0f}% progress). Status: {status}."
    )

    return GoalProgress(
        metric_name=goal.metric_name,
        current_value=current,
        target_value=target,
        direction="below",
        progress_pct=round(progress, 1),
        status=status,
        remaining=round(remaining, 4),
        summary=summary,
    )


def _evaluate_between(goal: Goal, current: float) -> GoalProgress:
    """Evaluate goal where target is to be between min and max."""
    target_min = goal.target_min if goal.target_min is not None else 0.0
    target_max = goal.target_value
    midpoint = (target_min + target_max) / 2
    remaining = 0.0

    if target_min <= current <= target_max:
        # In range — compute how close to center
        half_range = (target_max - target_min) / 2
        if half_range > 0:
            dist_from_center = abs(current - midpoint) / half_range
            progress = 100.0 - dist_from_center * 20  # slight penalty for edge proximity
        else:
            progress = 100.0
        status = "achieved"
    else:
        # Out of range
        if current < target_min:
            remaining = target_min - current
            range_size = target_max - target_min
            if range_size > 0:
                progress = max(0, (1 - remaining / range_size) * 80)
            else:
                progress = 0.0
        else:
            remaining = current - target_max
            range_size = target_max - target_min
            if range_size > 0:
                progress = max(0, (1 - remaining / range_size) * 80)
            else:
                progress = 0.0

        if progress >= 70:
            status = "at_risk"  # close to range
        elif progress >= 40:
            status = "at_risk"
        else:
            status = "behind"

    summary = (
        f"{goal.metric_name} is at {current:.1f}, target range is "
        f"{target_min:.1f}–{target_max:.1f}. Status: {status}."
    )

    return GoalProgress(
        metric_name=goal.metric_name,
        current_value=current,
        target_value=target_max,
        direction="between",
        progress_pct=round(progress, 1),
        status=status,
        remaining=round(remaining, 4),
        summary=summary,
    )
