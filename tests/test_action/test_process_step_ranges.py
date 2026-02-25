"""Tests for per-metric target_ranges in ProcessStep create/update/list.

Covers:
 - target_ranges dict storage on create
 - backward compatibility: target_range (singular) auto-maps to target_ranges
 - target_ranges overrides legacy target_range when both provided
 - update replaces target_ranges
 - list_process_steps serialisation of target_ranges
 - edge cases (extra keys, unicode, empty values)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from business_brain.action.api import (
    create_process_step,
    update_process_step,
    list_process_steps,
    ProcessStepRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_for_create():
    """Return an AsyncMock session wired for create_process_step.

    - session.add captures the ORM object
    - session.refresh assigns id = 1
    - session.commit is a no-op coroutine
    """
    session = AsyncMock()
    captured = {}

    def capture_add(obj):
        captured["step"] = obj

    session.add = capture_add

    async def fake_refresh(obj):
        obj.id = 1

    session.refresh = fake_refresh

    return session, captured


def _make_step_mock(**overrides):
    """Build a MagicMock that looks like a ProcessStep ORM row."""
    defaults = dict(
        id=1,
        step_order=0,
        process_name="Melting",
        inputs="Scrap",
        outputs="Billet",
        key_metric="SEC",
        key_metrics=["SEC"],
        target_range="500-625",
        target_ranges={"SEC": "500-625"},
        linked_table="",
    )
    defaults.update(overrides)
    step = MagicMock(**defaults)
    # MagicMock auto-generates attributes, but direct attribute access is
    # preferred for ORM-style reads in the list endpoint.
    for k, v in defaults.items():
        setattr(step, k, v)
    return step


def _make_session_for_update(existing_step):
    """Return an AsyncMock session wired for update_process_step.

    session.execute returns a result whose scalar_one_or_none gives
    *existing_step* (or None to simulate not-found).
    """
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing_step
    session.execute = AsyncMock(return_value=result)
    return session


def _make_session_for_list(steps):
    """Return an AsyncMock session wired for list_process_steps."""
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = steps
    session.execute = AsyncMock(return_value=result)
    return session


# ===================================================================
# TestCreateProcessStepRanges
# ===================================================================


class TestCreateProcessStepRanges:
    """Tests for POST /process-steps — target_ranges behaviour."""

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_create_with_target_ranges_dict(self, mock_regen):
        """Send key_metrics + target_ranges, verify both are stored on the ORM object."""
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(
            process_name="Melting",
            key_metrics=["SEC", "Yield"],
            target_ranges={"SEC": "500-625 kWh/ton", "Yield": "85-95%"},
        )
        result = await create_process_step(req, session)

        assert result["status"] == "created"
        assert result["id"] == 1

        step = captured["step"]
        assert step.target_ranges == {"SEC": "500-625 kWh/ton", "Yield": "85-95%"}
        assert step.key_metrics == ["SEC", "Yield"]
        # target_range (singular) auto-set from first metric
        assert step.target_range == "500-625 kWh/ton"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_create_backward_compat_single_range(self, mock_regen):
        """Send key_metrics + target_range (singular, no target_ranges dict).

        The code should auto-populate target_ranges = {first_metric: target_range}.
        """
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(
            process_name="Casting",
            key_metrics=["SEC"],
            target_range="500-625",
        )
        result = await create_process_step(req, session)

        assert result["status"] == "created"
        step = captured["step"]
        assert step.target_ranges == {"SEC": "500-625"}
        assert step.target_range == "500-625"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_create_target_ranges_overrides_target_range(self, mock_regen):
        """When both target_range and target_ranges provided, target_ranges wins."""
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(
            process_name="Rolling",
            key_metrics=["SEC", "Yield"],
            target_range="old-value",
            target_ranges={"SEC": "500-625", "Yield": "85-95%"},
        )
        result = await create_process_step(req, session)

        step = captured["step"]
        # target_ranges dict is the authoritative source
        assert step.target_ranges == {"SEC": "500-625", "Yield": "85-95%"}
        # target_range (singular) preserves the explicit value from the request
        assert step.target_range == "old-value"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_create_no_metrics_no_ranges(self, mock_regen):
        """Empty key_metrics and no ranges => target_ranges should be {}."""
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(process_name="Inspection")
        result = await create_process_step(req, session)

        step = captured["step"]
        assert step.target_ranges == {}
        assert step.target_range == ""

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_create_target_range_auto_set_from_first_metric(self, mock_regen):
        """target_range (singular) is auto-set from the first metric's range in target_ranges."""
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(
            process_name="Reheating",
            key_metrics=["Temperature", "Duration"],
            target_ranges={"Temperature": "1100-1200 C", "Duration": "45-60 min"},
        )
        result = await create_process_step(req, session)

        step = captured["step"]
        # When no explicit target_range is given, it falls back to first metric's range
        assert step.target_range == "1100-1200 C"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_create_three_metrics_three_ranges(self, mock_regen):
        """Three metrics with three individual ranges — all stored correctly."""
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(
            process_name="EAF",
            key_metrics=["SEC", "Yield %", "Tap-to-Tap"],
            target_ranges={
                "SEC": "500-625 kWh/ton",
                "Yield %": "85-95%",
                "Tap-to-Tap": "55-70 min",
            },
        )
        result = await create_process_step(req, session)

        step = captured["step"]
        assert step.key_metrics == ["SEC", "Yield %", "Tap-to-Tap"]
        assert step.target_ranges == {
            "SEC": "500-625 kWh/ton",
            "Yield %": "85-95%",
            "Tap-to-Tap": "55-70 min",
        }
        # singular target_range from first metric
        assert step.target_range == "500-625 kWh/ton"


# ===================================================================
# TestUpdateProcessStepRanges
# ===================================================================


class TestUpdateProcessStepRanges:
    """Tests for PUT /process-steps/{step_id} — target_ranges behaviour."""

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_update_replaces_target_ranges(self, mock_regen):
        """Update step with new target_ranges, verify old replaced."""
        existing = _make_step_mock(
            target_ranges={"SEC": "500-625"},
            key_metrics=["SEC"],
        )
        session = _make_session_for_update(existing)

        req = ProcessStepRequest(
            process_name="Melting",
            key_metrics=["SEC", "Yield"],
            target_ranges={"SEC": "550-650", "Yield": "88-96%"},
        )
        result = await update_process_step(1, req, session)

        assert result["status"] == "updated"
        assert existing.target_ranges == {"SEC": "550-650", "Yield": "88-96%"}
        assert existing.key_metrics == ["SEC", "Yield"]

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_update_backward_compat(self, mock_regen):
        """Update with single target_range, auto-maps to first metric."""
        existing = _make_step_mock(
            target_ranges={"SEC": "500-625"},
            key_metrics=["SEC"],
        )
        session = _make_session_for_update(existing)

        req = ProcessStepRequest(
            process_name="Casting",
            key_metrics=["SEC"],
            target_range="600-700",
        )
        result = await update_process_step(1, req, session)

        assert result["status"] == "updated"
        # Backward compat: target_range auto-populates target_ranges
        assert existing.target_ranges == {"SEC": "600-700"}
        assert existing.target_range == "600-700"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_update_step_not_found(self, mock_regen):
        """Returns error dict when step doesn't exist."""
        session = _make_session_for_update(None)

        req = ProcessStepRequest(process_name="Ghost")
        result = await update_process_step(999, req, session)

        assert result == {"error": "Process step not found"}


# ===================================================================
# TestListProcessStepRanges
# ===================================================================


class TestListProcessStepRanges:
    """Tests for GET /process-steps — target_ranges serialisation."""

    @pytest.mark.asyncio
    async def test_list_includes_target_ranges(self):
        """Mock ProcessStep objects with target_ranges, verify response includes them."""
        step_a = _make_step_mock(
            id=1,
            process_name="Melting",
            key_metrics=["SEC", "Yield"],
            target_ranges={"SEC": "500-625", "Yield": "85-95%"},
            target_range="500-625",
        )
        step_b = _make_step_mock(
            id=2,
            process_name="Rolling",
            key_metrics=["Elongation"],
            target_ranges={"Elongation": "14-18%"},
            target_range="14-18%",
        )
        session = _make_session_for_list([step_a, step_b])

        rows = await list_process_steps(session)

        assert len(rows) == 2
        assert rows[0]["target_ranges"] == {"SEC": "500-625", "Yield": "85-95%"}
        assert rows[1]["target_ranges"] == {"Elongation": "14-18%"}

    @pytest.mark.asyncio
    async def test_list_empty_target_ranges_returns_empty_dict(self):
        """Step with target_ranges=None returns {} in the response."""
        step = _make_step_mock(
            target_ranges=None,
            key_metrics=["SEC"],
            key_metric="SEC",
        )
        session = _make_session_for_list([step])

        rows = await list_process_steps(session)

        assert rows[0]["target_ranges"] == {}

    @pytest.mark.asyncio
    async def test_list_backward_compat_no_target_ranges(self):
        """Step with only target_range (no target_ranges column yet) returns {} for target_ranges."""
        step = _make_step_mock(
            target_range="500-625",
            target_ranges=None,
            key_metric="SEC",
            key_metrics=None,
        )
        session = _make_session_for_list([step])

        rows = await list_process_steps(session)

        # target_ranges coalesces to {} when None
        assert rows[0]["target_ranges"] == {}
        # target_range singular is preserved
        assert rows[0]["target_range"] == "500-625"
        # key_metrics falls back from key_metric singular
        assert rows[0]["key_metrics"] == ["SEC"]


# ===================================================================
# TestTargetRangesEdgeCases
# ===================================================================


class TestTargetRangesEdgeCases:
    """Edge cases for target_ranges handling."""

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_extra_keys_in_target_ranges(self, mock_regen):
        """target_ranges with keys not in key_metrics should be stored, not rejected."""
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(
            process_name="Melting",
            key_metrics=["SEC"],
            target_ranges={
                "SEC": "500-625",
                "Phantom Metric": "0-100",  # not in key_metrics
            },
        )
        result = await create_process_step(req, session)

        step = captured["step"]
        # Extra key is stored verbatim
        assert "Phantom Metric" in step.target_ranges
        assert step.target_ranges["Phantom Metric"] == "0-100"
        assert step.target_ranges["SEC"] == "500-625"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_unicode_metric_names_in_ranges(self, mock_regen):
        """Metric names with special/unicode characters work without error."""
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(
            process_name="Furnace",
            key_metrics=["Temperature (C)", "CO2 Emission"],
            target_ranges={
                "Temperature (C)": "1100-1200",
                "CO2 Emission": "50-80 kg/ton",
            },
        )
        result = await create_process_step(req, session)

        assert result["status"] == "created"
        step = captured["step"]
        assert step.target_ranges["Temperature (C)"] == "1100-1200"
        assert step.target_ranges["CO2 Emission"] == "50-80 kg/ton"

    @pytest.mark.asyncio
    @patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock)
    async def test_empty_range_values(self, mock_regen):
        """target_ranges={"SEC": ""} is valid — empty range string accepted."""
        session, captured = _make_session_for_create()

        req = ProcessStepRequest(
            process_name="Melting",
            key_metrics=["SEC"],
            target_ranges={"SEC": ""},
        )
        result = await create_process_step(req, session)

        step = captured["step"]
        assert step.target_ranges == {"SEC": ""}
        # target_range auto-derived from first metric's range (empty string)
        assert step.target_range == ""
