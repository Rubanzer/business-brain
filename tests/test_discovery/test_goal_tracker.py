"""Tests for metric goal tracker — goal evaluation and health scoring."""

from business_brain.discovery.goal_tracker import (
    Goal,
    GoalProgress,
    classify_trend,
    compute_overall_health,
    evaluate_goal,
    evaluate_goals,
    format_goal_summary,
)


# ---------------------------------------------------------------------------
# evaluate_goal — "above" direction
# ---------------------------------------------------------------------------


class TestEvaluateGoalAbove:
    def test_achieved(self):
        goal = Goal("revenue", target_value=1000, direction="above", baseline=0)
        result = evaluate_goal(goal, 1000)
        assert result.status == "achieved"
        assert result.progress_pct == 100.0

    def test_exceeded(self):
        goal = Goal("revenue", target_value=1000, direction="above", baseline=0)
        result = evaluate_goal(goal, 1200)
        assert result.status == "exceeded"

    def test_on_track(self):
        goal = Goal("revenue", target_value=1000, direction="above", baseline=0)
        result = evaluate_goal(goal, 750)
        assert result.status == "on_track"
        assert result.progress_pct == 75.0

    def test_at_risk(self):
        goal = Goal("revenue", target_value=1000, direction="above", baseline=0)
        result = evaluate_goal(goal, 450)
        assert result.status == "at_risk"

    def test_behind(self):
        goal = Goal("revenue", target_value=1000, direction="above", baseline=0)
        result = evaluate_goal(goal, 100)
        assert result.status == "behind"

    def test_no_baseline_defaults_to_zero(self):
        goal = Goal("metric", target_value=100, direction="above")
        result = evaluate_goal(goal, 50)
        assert result.progress_pct == 50.0

    def test_remaining(self):
        goal = Goal("metric", target_value=100, direction="above", baseline=0)
        result = evaluate_goal(goal, 75)
        assert result.remaining == 25.0

    def test_summary_text(self):
        goal = Goal("revenue", target_value=1000, direction="above", baseline=0)
        result = evaluate_goal(goal, 500)
        assert "revenue" in result.summary
        assert "500" in result.summary


# ---------------------------------------------------------------------------
# evaluate_goal — "below" direction
# ---------------------------------------------------------------------------


class TestEvaluateGoalBelow:
    def test_achieved(self):
        goal = Goal("defects", target_value=10, direction="below", baseline=50)
        result = evaluate_goal(goal, 10)
        assert result.status == "achieved"

    def test_exceeded(self):
        goal = Goal("defects", target_value=10, direction="below", baseline=50)
        result = evaluate_goal(goal, 5)
        assert result.status == "exceeded"

    def test_behind(self):
        goal = Goal("defects", target_value=10, direction="below", baseline=50)
        result = evaluate_goal(goal, 45)
        assert result.status == "behind"

    def test_remaining_below(self):
        goal = Goal("defects", target_value=10, direction="below", baseline=50)
        result = evaluate_goal(goal, 30)
        assert result.remaining == 20.0

    def test_no_baseline_uses_double_target(self):
        goal = Goal("cost", target_value=100, direction="below")
        result = evaluate_goal(goal, 150)
        assert result.progress_pct == 50.0


# ---------------------------------------------------------------------------
# evaluate_goal — "between" direction
# ---------------------------------------------------------------------------


class TestEvaluateGoalBetween:
    def test_in_range_center(self):
        goal = Goal("pH", target_value=8.0, direction="between", target_min=6.0)
        result = evaluate_goal(goal, 7.0)
        assert result.status == "achieved"
        assert result.progress_pct > 80

    def test_in_range_edge(self):
        goal = Goal("pH", target_value=8.0, direction="between", target_min=6.0)
        result = evaluate_goal(goal, 6.0)
        assert result.status == "achieved"

    def test_below_range(self):
        goal = Goal("pH", target_value=8.0, direction="between", target_min=6.0)
        result = evaluate_goal(goal, 4.0)
        assert result.status in ("at_risk", "behind")

    def test_above_range(self):
        goal = Goal("pH", target_value=8.0, direction="between", target_min=6.0)
        result = evaluate_goal(goal, 12.0)
        assert result.status in ("at_risk", "behind")

    def test_remaining_when_below(self):
        goal = Goal("pH", target_value=8.0, direction="between", target_min=6.0)
        result = evaluate_goal(goal, 4.0)
        assert result.remaining == 2.0  # 6.0 - 4.0


# ---------------------------------------------------------------------------
# evaluate_goal — unknown direction
# ---------------------------------------------------------------------------


class TestEvaluateGoalUnknown:
    def test_unknown_direction(self):
        goal = Goal("x", target_value=100, direction="sideways")
        result = evaluate_goal(goal, 50)
        assert result.status == "behind"
        assert "Unknown" in result.summary


# ---------------------------------------------------------------------------
# evaluate_goals (batch)
# ---------------------------------------------------------------------------


class TestEvaluateGoals:
    def test_all_present(self):
        goals = [
            Goal("revenue", 1000, "above", baseline=0),
            Goal("defects", 10, "below", baseline=50),
        ]
        values = {"revenue": 800, "defects": 15}
        results = evaluate_goals(goals, values)
        assert len(results) == 2
        assert results[0].metric_name == "revenue"
        assert results[1].metric_name == "defects"

    def test_missing_value(self):
        goals = [Goal("revenue", 1000, "above")]
        results = evaluate_goals(goals, {})
        assert len(results) == 1
        assert results[0].status == "behind"
        assert "No current value" in results[0].summary

    def test_empty_goals(self):
        results = evaluate_goals([], {})
        assert results == []


# ---------------------------------------------------------------------------
# compute_overall_health
# ---------------------------------------------------------------------------


class TestComputeOverallHealth:
    def test_empty(self):
        h = compute_overall_health([])
        assert h["total"] == 0
        assert h["health_score"] == 0.0

    def test_all_achieved(self):
        progress = [
            GoalProgress("a", 100, 100, "above", 100, "achieved", 0, ""),
            GoalProgress("b", 5, 10, "below", 100, "achieved", 0, ""),
        ]
        h = compute_overall_health(progress)
        assert h["health_score"] == 100.0
        assert h["achieved_pct"] == 100.0

    def test_mixed_statuses(self):
        progress = [
            GoalProgress("a", 100, 100, "above", 100, "achieved", 0, ""),
            GoalProgress("b", 10, 100, "above", 10, "behind", 90, ""),
        ]
        h = compute_overall_health(progress)
        assert 40 < h["health_score"] < 70
        assert h["status_counts"]["achieved"] == 1
        assert h["status_counts"]["behind"] == 1

    def test_all_behind(self):
        progress = [
            GoalProgress("a", 0, 100, "above", 0, "behind", 100, ""),
        ]
        h = compute_overall_health(progress)
        assert h["health_score"] == 10.0


# ---------------------------------------------------------------------------
# classify_trend
# ---------------------------------------------------------------------------


class TestClassifyTrend:
    def test_improving_above(self):
        values = [10, 20, 30, 40, 50]
        assert classify_trend(values, 100, "above") == "improving"

    def test_declining_above(self):
        values = [50, 40, 30, 20, 10]
        assert classify_trend(values, 100, "above") == "declining"

    def test_improving_below(self):
        values = [50, 40, 30, 20, 10]
        assert classify_trend(values, 5, "below") == "improving"

    def test_volatile(self):
        values = [10, 50, 10, 50, 10, 50]
        assert classify_trend(values, 100, "above") == "volatile"

    def test_stable(self):
        values = [50, 51, 50, 51, 50]
        result = classify_trend(values, 100, "above")
        assert result in ("stable", "volatile")  # small fluctuations

    def test_too_few_values(self):
        assert classify_trend([10, 20], 100, "above") == "stable"


# ---------------------------------------------------------------------------
# format_goal_summary
# ---------------------------------------------------------------------------


class TestFormatGoalSummary:
    def test_achieved(self):
        p = GoalProgress("revenue", 1000, 1000, "above", 100, "achieved", 0, "")
        text = format_goal_summary(p)
        assert "[OK]" in text
        assert "revenue" in text
        assert "100%" in text

    def test_behind(self):
        p = GoalProgress("defects", 50, 10, "below", 20, "behind", 40, "")
        text = format_goal_summary(p)
        assert "[X]" in text
        assert "behind" in text

    def test_exceeded(self):
        p = GoalProgress("output", 1200, 1000, "above", 120, "exceeded", -200, "")
        text = format_goal_summary(p)
        assert "[++]" in text
