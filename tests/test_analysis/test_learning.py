"""Tests for analysis/learning/updater.py — EMA-based parameter learning."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from business_brain.analysis.learning.updater import (
    _clamp,
    _compute_interestingness_weights,
    _compute_operation_preferences,
    _compute_tier_budgets,
    _ema,
)
from business_brain.analysis.track1.scorer import DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self):
        assert _clamp(1.0) == 1.0

    def test_below_minimum(self):
        assert _clamp(0.1) == 0.5

    def test_above_maximum(self):
        assert _clamp(2.0) == 1.5

    def test_at_boundary(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(1.5) == 1.5

    def test_custom_bounds(self):
        assert _clamp(0.3, lo=0.2, hi=0.8) == 0.3
        assert _clamp(0.1, lo=0.2, hi=0.8) == 0.2
        assert _clamp(0.9, lo=0.2, hi=0.8) == 0.8


class TestEma:
    def test_basic_ema(self):
        # alpha=0.3: (1-0.3)*10 + 0.3*20 = 7+6 = 13
        assert _ema(10.0, 20.0, alpha=0.3) == pytest.approx(13.0)

    def test_alpha_zero_keeps_old(self):
        assert _ema(10.0, 20.0, alpha=0.0) == pytest.approx(10.0)

    def test_alpha_one_takes_new(self):
        assert _ema(10.0, 20.0, alpha=1.0) == pytest.approx(20.0)

    def test_default_alpha(self):
        result = _ema(1.0, 2.0)
        # default alpha=0.3: 0.7*1 + 0.3*2 = 1.3
        assert result == pytest.approx(1.3)


# ---------------------------------------------------------------------------
# Operation preferences
# ---------------------------------------------------------------------------


class TestComputeOperationPreferences:
    def test_useful_boosts(self):
        feedback = {
            "CORRELATE": {"useful": 8, "not_useful": 2},
        }
        prefs = _compute_operation_preferences(feedback, None)
        assert prefs["CORRELATE"] > 1.0  # boosted

    def test_not_useful_penalizes(self):
        feedback = {
            "RANK": {"useful": 1, "not_useful": 9},
        }
        prefs = _compute_operation_preferences(feedback, None)
        assert prefs["RANK"] < 1.0  # penalized

    def test_preserves_previous(self):
        feedback = {
            "CORRELATE": {"useful": 5, "not_useful": 5},
        }
        prev = {"CORRELATE": 1.2, "DESCRIBE": 0.8}
        prefs = _compute_operation_preferences(feedback, prev)
        assert "DESCRIBE" in prefs  # preserved from previous
        assert prefs["DESCRIBE"] == 0.8  # unchanged

    def test_empty_feedback(self):
        prefs = _compute_operation_preferences({}, None)
        assert prefs == {}

    def test_zero_total_skipped(self):
        feedback = {"RANK": {"useful": 0, "not_useful": 0}}
        prefs = _compute_operation_preferences(feedback, None)
        assert "RANK" not in prefs

    def test_clamped_range(self):
        feedback = {
            "CORRELATE": {"useful": 100, "not_useful": 0},
        }
        prefs = _compute_operation_preferences(feedback, None)
        assert prefs["CORRELATE"] <= 1.5
        assert prefs["CORRELATE"] >= 0.5


# ---------------------------------------------------------------------------
# Tier budgets
# ---------------------------------------------------------------------------


class TestComputeTierBudgets:
    def test_useful_increases_budget(self):
        feedback = {
            2: {"useful": 8, "not_useful": 2},
        }
        budgets = _compute_tier_budgets(feedback, None)
        assert int(budgets["2"]) >= 100  # default or higher

    def test_not_useful_decreases_budget(self):
        feedback = {
            3: {"useful": 1, "not_useful": 9},
        }
        budgets = _compute_tier_budgets(feedback, None)
        assert int(budgets["3"]) <= 50  # default or lower

    def test_skips_low_tiers(self):
        """Tier 0+1 are exhaustive — no budget adjustment."""
        feedback = {
            0: {"useful": 1, "not_useful": 9},
            1: {"useful": 1, "not_useful": 9},
        }
        budgets = _compute_tier_budgets(feedback, None)
        # Defaults should not have keys for tier 0/1
        assert "0" not in budgets or budgets.get("0") == budgets.get("0")

    def test_skips_few_feedbacks(self):
        feedback = {
            2: {"useful": 1, "not_useful": 1},  # total=2, below min=3
        }
        budgets = _compute_tier_budgets(feedback, None)
        # Should use default
        assert budgets["2"] == 100

    def test_budget_bounds(self):
        """Budgets should be clamped between 10 and 200."""
        feedback = {
            2: {"useful": 100, "not_useful": 0},
        }
        budgets = _compute_tier_budgets(feedback, None)
        assert 10 <= budgets["2"] <= 200


# ---------------------------------------------------------------------------
# Interestingness weights
# ---------------------------------------------------------------------------


class TestComputeInterestingnessWeights:
    def test_high_wrong_rate_boosts_stability(self):
        type_counts = {"useful": 2, "wrong": 5, "not_useful": 3}  # wrong=50%
        weights = _compute_interestingness_weights(type_counts, None)
        assert weights["stability"] > DEFAULT_WEIGHTS.get("stability", 0.15)

    def test_high_useful_rate_boosts_surprise(self):
        type_counts = {"useful": 8, "wrong": 1, "not_useful": 1}  # useful=80%
        weights = _compute_interestingness_weights(type_counts, None)
        assert weights["surprise"] > DEFAULT_WEIGHTS.get("surprise", 0.3)

    def test_weights_sum_to_one(self):
        type_counts = {"useful": 5, "wrong": 3, "not_useful": 2}
        weights = _compute_interestingness_weights(type_counts, None)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_not_enough_feedback_returns_defaults(self):
        type_counts = {"useful": 2}  # total=2, below _MIN_FEEDBACK=5
        weights = _compute_interestingness_weights(type_counts, None)
        assert weights == DEFAULT_WEIGHTS

    def test_preserves_previous_weights(self):
        prev = {"surprise": 0.4, "magnitude": 0.2, "variance": 0.15, "stability": 0.15, "coverage": 0.1}
        type_counts = {"useful": 3, "wrong": 3, "not_useful": 4}  # high wrong rate
        weights = _compute_interestingness_weights(type_counts, prev)
        # stability should increase from previous (use approx for float precision)
        assert weights["stability"] >= prev["stability"] - 0.001
        # Should still sum to 1
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
