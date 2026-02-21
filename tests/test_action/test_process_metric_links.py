"""Tests for process-metric linking — link, get, and unlink metrics from process steps."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import HTTPException

from business_brain.action.api import (
    link_metrics_to_step,
    get_step_metrics,
    unlink_metric_from_step,
)


def _make_step_found_result():
    """Return a MagicMock result where scalar_one_or_none returns a step object."""
    step = MagicMock()
    step.id = 1
    result = MagicMock()
    result.scalar_one_or_none.return_value = step
    return result, step


def _make_not_found_result():
    """Return a MagicMock result where scalar_one_or_none returns None."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    return result


def _make_metric_mock(metric_id, name="test_metric", table_name="data", column_name="col"):
    """Create a mock MetricThreshold object."""
    m = MagicMock()
    m.id = metric_id
    m.metric_name = name
    m.table_name = table_name
    m.column_name = column_name
    m.unit = "units"
    m.is_derived = False
    m.auto_linked = False
    m.confidence = None
    return m


class TestLinkMetricsToStep:
    """Tests for POST /process-steps/{step_id}/metrics."""

    @pytest.mark.asyncio
    async def test_link_metric_to_step(self):
        """Successfully creates ProcessMetricLink records."""
        session = AsyncMock()
        session.add = MagicMock()

        step_result, step = _make_step_found_result()

        metric = _make_metric_mock(10, "temperature")
        metric_result = MagicMock()
        metric_result.scalar_one_or_none.return_value = metric

        # No existing link
        no_link_result = _make_not_found_result()

        # session.execute is called:
        #   1) select ProcessStep where id == step_id -> found
        #   2) select MetricThreshold where id == 10 -> found
        #   3) select ProcessMetricLink (duplicate check) -> not found
        session.execute = AsyncMock(
            side_effect=[step_result, metric_result, no_link_result]
        )

        body = {"metric_ids": [10]}
        result = await link_metrics_to_step(1, body, session)

        assert result["status"] == "linked"
        assert result["step_id"] == 1
        assert result["metrics_linked"] == [10]
        session.add.assert_called_once()
        session.commit.assert_called_once()

        # Verify the link object was created correctly
        added_link = session.add.call_args[0][0]
        assert added_link.process_step_id == 1
        assert added_link.metric_id == 10
        assert added_link.is_primary is True  # first link is primary

    @pytest.mark.asyncio
    async def test_link_multiple_metrics(self):
        """Linking multiple metrics sets first as primary, rest as not."""
        session = AsyncMock()
        session.add = MagicMock()

        step_result, step = _make_step_found_result()

        metric_a = _make_metric_mock(10, "temperature")
        metric_a_result = MagicMock()
        metric_a_result.scalar_one_or_none.return_value = metric_a

        metric_b = _make_metric_mock(20, "pressure")
        metric_b_result = MagicMock()
        metric_b_result.scalar_one_or_none.return_value = metric_b

        no_link_result_1 = _make_not_found_result()
        no_link_result_2 = _make_not_found_result()

        # Execute calls: step check, metric_a, dup check, metric_b, dup check
        session.execute = AsyncMock(
            side_effect=[
                step_result,
                metric_a_result, no_link_result_1,
                metric_b_result, no_link_result_2,
            ]
        )

        body = {"metric_ids": [10, 20]}
        result = await link_metrics_to_step(1, body, session)

        assert result["metrics_linked"] == [10, 20]
        assert session.add.call_count == 2

        # First link is primary
        first_link = session.add.call_args_list[0][0][0]
        assert first_link.is_primary is True

        # Second link is not primary
        second_link = session.add.call_args_list[1][0][0]
        assert second_link.is_primary is False

    @pytest.mark.asyncio
    async def test_link_nonexistent_step(self):
        """404 for bad step_id."""
        session = AsyncMock()

        not_found = _make_not_found_result()
        session.execute = AsyncMock(return_value=not_found)

        body = {"metric_ids": [1]}

        with pytest.raises(HTTPException) as exc_info:
            await link_metrics_to_step(999, body, session)

        assert exc_info.value.status_code == 404
        assert "Process step not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_link_nonexistent_metric(self):
        """Nonexistent metric_id is silently skipped (not linked)."""
        session = AsyncMock()
        session.add = MagicMock()

        step_result, step = _make_step_found_result()

        # Metric not found
        metric_not_found = _make_not_found_result()

        session.execute = AsyncMock(
            side_effect=[step_result, metric_not_found]
        )

        body = {"metric_ids": [999]}
        result = await link_metrics_to_step(1, body, session)

        assert result["metrics_linked"] == []
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_link_prevented(self):
        """Same link twice is idempotent — duplicate is silently skipped."""
        session = AsyncMock()
        session.add = MagicMock()

        step_result, step = _make_step_found_result()

        metric = _make_metric_mock(10, "temperature")
        metric_result = MagicMock()
        metric_result.scalar_one_or_none.return_value = metric

        # Duplicate link already exists
        existing_link = MagicMock()
        existing_link_result = MagicMock()
        existing_link_result.scalar_one_or_none.return_value = existing_link

        session.execute = AsyncMock(
            side_effect=[step_result, metric_result, existing_link_result]
        )

        body = {"metric_ids": [10]}
        result = await link_metrics_to_step(1, body, session)

        # Duplicate is skipped
        assert result["metrics_linked"] == []
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_link_empty_metric_ids(self):
        """Empty metric_ids array raises HTTPException 400."""
        session = AsyncMock()

        step_result, step = _make_step_found_result()
        session.execute = AsyncMock(return_value=step_result)

        body = {"metric_ids": []}

        with pytest.raises(HTTPException) as exc_info:
            await link_metrics_to_step(1, body, session)

        assert exc_info.value.status_code == 400
        assert "metric_ids" in exc_info.value.detail


class TestGetStepMetrics:
    """Tests for GET /process-steps/{step_id}/metrics."""

    @pytest.mark.asyncio
    async def test_get_step_metrics(self):
        """Returns linked metrics with full detail."""
        session = AsyncMock()

        # Simulate two links for step_id=1
        link_a = MagicMock()
        link_a.id = 100
        link_a.metric_id = 10
        link_a.is_primary = True

        link_b = MagicMock()
        link_b.id = 101
        link_b.metric_id = 20
        link_b.is_primary = False

        links_result = MagicMock()
        links_result.scalars.return_value.all.return_value = [link_a, link_b]

        metric_a = _make_metric_mock(10, "temperature", "scada", "temp_c")
        metric_a_result = MagicMock()
        metric_a_result.scalar_one_or_none.return_value = metric_a

        metric_b = _make_metric_mock(20, "pressure", "scada", "psi")
        metric_b_result = MagicMock()
        metric_b_result.scalar_one_or_none.return_value = metric_b

        # Execute calls: get links, get metric_a, get metric_b
        session.execute = AsyncMock(
            side_effect=[links_result, metric_a_result, metric_b_result]
        )

        result = await get_step_metrics(1, session)

        assert len(result) == 2
        assert result[0]["metric_id"] == 10
        assert result[0]["metric_name"] == "temperature"
        assert result[0]["is_primary"] is True
        assert result[0]["link_id"] == 100
        assert result[1]["metric_id"] == 20
        assert result[1]["is_primary"] is False

    @pytest.mark.asyncio
    async def test_get_step_metrics_empty(self):
        """Step with no linked metrics returns empty list."""
        session = AsyncMock()

        links_result = MagicMock()
        links_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=links_result)

        result = await get_step_metrics(99, session)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_step_metrics_includes_derived(self):
        """Derived metrics show is_derived=True and auto_linked flag."""
        session = AsyncMock()

        link = MagicMock()
        link.id = 200
        link.metric_id = 30
        link.is_primary = True

        links_result = MagicMock()
        links_result.scalars.return_value.all.return_value = [link]

        derived_metric = _make_metric_mock(30, "energy_per_ton")
        derived_metric.is_derived = True
        derived_metric.auto_linked = True
        derived_metric.confidence = 0.95

        metric_result = MagicMock()
        metric_result.scalar_one_or_none.return_value = derived_metric

        session.execute = AsyncMock(
            side_effect=[links_result, metric_result]
        )

        result = await get_step_metrics(1, session)

        assert len(result) == 1
        assert result[0]["is_derived"] is True
        assert result[0]["auto_linked"] is True
        assert result[0]["confidence"] == 0.95


class TestUnlinkMetricFromStep:
    """Tests for DELETE /process-steps/{step_id}/metrics/{metric_id}."""

    @pytest.mark.asyncio
    async def test_unlink_metric(self):
        """Successfully deletes the link."""
        session = AsyncMock()

        delete_result = MagicMock()
        delete_result.rowcount = 1
        session.execute = AsyncMock(return_value=delete_result)

        result = await unlink_metric_from_step(1, 10, session)

        assert result["status"] == "unlinked"
        assert result["step_id"] == 1
        assert result["metric_id"] == 10
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_unlink_nonexistent_link(self):
        """404 when no matching link found (rowcount == 0)."""
        session = AsyncMock()

        delete_result = MagicMock()
        delete_result.rowcount = 0
        session.execute = AsyncMock(return_value=delete_result)

        with pytest.raises(HTTPException) as exc_info:
            await unlink_metric_from_step(999, 999, session)

        assert exc_info.value.status_code == 404
        assert "Link not found" in exc_info.value.detail
