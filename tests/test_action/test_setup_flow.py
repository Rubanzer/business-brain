"""Tests for multi-metric process steps — create and update with key_metrics array."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.action.api import create_process_step, update_process_step, ProcessStepRequest


class TestProcessStepMultipleMetrics:
    """Tests for process step creation/update with multi-metric support."""

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_process_step_multiple_metrics(self, mock_regen):
        """POST with key_metrics array stores all metrics properly."""
        session = AsyncMock()
        session.add = MagicMock()

        # After commit + refresh, the step gets an id
        async def fake_refresh(obj):
            obj.id = 42

        session.refresh = AsyncMock(side_effect=fake_refresh)

        req = ProcessStepRequest(
            step_order=1,
            process_name="Smelting",
            inputs="Iron Ore",
            outputs="Pig Iron",
            key_metrics=["temperature", "power_consumption", "yield_pct"],
            target_range="85-95%",
            linked_table="smelting_data",
        )

        result = await create_process_step(req, session)

        assert result["status"] == "created"
        assert result["id"] == 42

        # Verify the step object passed to session.add
        added_step = session.add.call_args[0][0]
        assert added_step.key_metrics == ["temperature", "power_consumption", "yield_pct"]
        assert added_step.key_metric == "temperature"  # first element becomes key_metric
        assert added_step.process_name == "Smelting"

        session.commit.assert_called_once()
        mock_regen.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_process_step_backward_compat_single(self, mock_regen):
        """key_metric (singular) alone still works for backward compatibility."""
        session = AsyncMock()
        session.add = MagicMock()

        async def fake_refresh(obj):
            obj.id = 10

        session.refresh = AsyncMock(side_effect=fake_refresh)

        req = ProcessStepRequest(
            step_order=2,
            process_name="Rolling",
            key_metric="throughput",
        )

        result = await create_process_step(req, session)

        assert result["status"] == "created"
        added_step = session.add.call_args[0][0]
        # When only key_metric is provided, key_metrics becomes [key_metric]
        assert added_step.key_metrics == ["throughput"]
        assert added_step.key_metric == "throughput"

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_key_metrics_populates_key_metric(self, mock_regen):
        """First element of key_metrics array becomes key_metric (singular)."""
        session = AsyncMock()
        session.add = MagicMock()

        async def fake_refresh(obj):
            obj.id = 99

        session.refresh = AsyncMock(side_effect=fake_refresh)

        req = ProcessStepRequest(
            step_order=3,
            process_name="Casting",
            key_metrics=["casting_speed", "mold_temperature"],
        )

        result = await create_process_step(req, session)

        added_step = session.add.call_args[0][0]
        assert added_step.key_metric == "casting_speed"
        assert added_step.key_metrics == ["casting_speed", "mold_temperature"]

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_update_step_metrics(self, mock_regen):
        """PUT updates key_metrics on an existing step."""
        session = AsyncMock()

        # Simulate finding an existing step
        existing_step = MagicMock()
        existing_step.id = 5
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = existing_step
        session.execute = AsyncMock(return_value=result_mock)

        req = ProcessStepRequest(
            step_order=1,
            process_name="Updated Smelting",
            key_metrics=["new_metric_a", "new_metric_b"],
        )

        result = await update_process_step(5, req, session)

        assert result["status"] == "updated"
        assert result["id"] == 5
        assert existing_step.key_metrics == ["new_metric_a", "new_metric_b"]
        assert existing_step.key_metric == "new_metric_a"
        assert existing_step.process_name == "Updated Smelting"
        session.commit.assert_called_once()
        mock_regen.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_empty_key_metrics(self, mock_regen):
        """Empty key_metrics array means no key_metric."""
        session = AsyncMock()
        session.add = MagicMock()

        async def fake_refresh(obj):
            obj.id = 77

        session.refresh = AsyncMock(side_effect=fake_refresh)

        req = ProcessStepRequest(
            step_order=4,
            process_name="Inspection",
            key_metrics=[],
            key_metric="",
        )

        result = await create_process_step(req, session)

        added_step = session.add.call_args[0][0]
        assert added_step.key_metrics == []
        # With empty key_metrics and empty key_metric, key_metric stays empty
        assert added_step.key_metric == ""

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_update_step_not_found(self, mock_regen):
        """PUT on nonexistent step returns error."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        req = ProcessStepRequest(
            step_order=1,
            process_name="Ghost Step",
        )

        result = await update_process_step(999, req, session)

        assert result.get("error") == "Process step not found"
        session.commit.assert_not_called()
        mock_regen.assert_not_called()
