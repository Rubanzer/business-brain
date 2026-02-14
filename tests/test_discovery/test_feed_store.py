"""Tests for feed_store CRUD functions using mock AsyncSession."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.db.discovery_models import DeployedReport, DiscoveryRun, Insight
from business_brain.discovery.feed_store import (
    delete_report,
    deploy_insight,
    get_feed,
    get_last_run,
    get_report,
    get_reports,
    refresh_report,
    update_status,
)


def _make_insight(**kwargs) -> MagicMock:
    """Create a mock Insight."""
    ins = MagicMock(spec=Insight)
    ins.id = kwargs.get("id", "test-id")
    ins.insight_type = kwargs.get("insight_type", "anomaly")
    ins.severity = kwargs.get("severity", "warning")
    ins.impact_score = kwargs.get("impact_score", 50)
    ins.title = kwargs.get("title", "Test Insight")
    ins.description = kwargs.get("description", "Test description")
    ins.status = kwargs.get("status", "new")
    ins.evidence = kwargs.get("evidence", {"query": "SELECT 1", "chart_spec": {"type": "bar"}})
    ins.session_id = kwargs.get("session_id", "sess-1")
    ins.source_tables = kwargs.get("source_tables", ["table_a"])
    return ins


def _make_report(**kwargs) -> MagicMock:
    """Create a mock DeployedReport."""
    report = MagicMock(spec=DeployedReport)
    report.id = kwargs.get("id", "report-1")
    report.name = kwargs.get("name", "Test Report")
    report.insight_id = kwargs.get("insight_id", "test-id")
    report.query = kwargs.get("query", "SELECT * FROM sales")
    report.chart_spec = kwargs.get("chart_spec", {"type": "bar"})
    report.active = kwargs.get("active", True)
    report.last_result = kwargs.get("last_result", None)
    report.last_run_at = kwargs.get("last_run_at", None)
    return report


class TestGetFeed:
    """Test insight feed retrieval."""

    @pytest.mark.asyncio
    async def test_returns_insights(self):
        session = AsyncMock()
        ins = _make_insight()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [ins]
        session.execute = AsyncMock(return_value=result)

        feed = await get_feed(session)
        assert len(feed) == 1
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_status_filter(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result)

        feed = await get_feed(session, status="new")
        assert feed == []
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_limit(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result)

        await get_feed(session, limit=10)
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_feed(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result)

        feed = await get_feed(session)
        assert feed == []


class TestUpdateStatus:
    """Test insight status updates."""

    @pytest.mark.asyncio
    async def test_updates_existing_insight(self):
        session = AsyncMock()
        ins = _make_insight(status="new")
        result = MagicMock()
        result.scalar_one_or_none.return_value = ins
        session.execute = AsyncMock(return_value=result)

        await update_status(session, "test-id", "seen")
        assert ins.status == "seen"
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_nonexistent_insight_no_commit(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        await update_status(session, "missing-id", "seen")
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_dismiss_status(self):
        session = AsyncMock()
        ins = _make_insight(status="new")
        result = MagicMock()
        result.scalar_one_or_none.return_value = ins
        session.execute = AsyncMock(return_value=result)

        await update_status(session, "test-id", "dismissed")
        assert ins.status == "dismissed"


class TestDeployInsight:
    """Test deploying an insight as a report."""

    @pytest.mark.asyncio
    async def test_deploy_creates_report(self):
        session = AsyncMock()
        ins = _make_insight(evidence={"query": "SELECT 1", "chart_spec": {"type": "line"}, "sample_rows": [{"a": 1}]})
        result = MagicMock()
        result.scalar_one_or_none.return_value = ins
        session.execute = AsyncMock(return_value=result)

        report = await deploy_insight(session, "test-id", "My Report")
        session.add.assert_called_once()
        session.commit.assert_called_once()
        assert ins.status == "deployed"

    @pytest.mark.asyncio
    async def test_deploy_nonexistent_raises(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        with pytest.raises(ValueError, match="not found"):
            await deploy_insight(session, "missing", "Report")

    @pytest.mark.asyncio
    async def test_deploy_with_empty_evidence(self):
        session = AsyncMock()
        ins = _make_insight(evidence={})
        result = MagicMock()
        result.scalar_one_or_none.return_value = ins
        session.execute = AsyncMock(return_value=result)

        report = await deploy_insight(session, "test-id", "Report")
        session.add.assert_called_once()


class TestGetReports:
    """Test fetching deployed reports."""

    @pytest.mark.asyncio
    async def test_returns_active_reports(self):
        session = AsyncMock()
        report = _make_report()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [report]
        session.execute = AsyncMock(return_value=result)

        reports = await get_reports(session)
        assert len(reports) == 1

    @pytest.mark.asyncio
    async def test_empty_reports(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result)

        reports = await get_reports(session)
        assert reports == []


class TestGetReport:
    """Test fetching single report."""

    @pytest.mark.asyncio
    async def test_returns_report(self):
        session = AsyncMock()
        report = _make_report()
        result = MagicMock()
        result.scalar_one_or_none.return_value = report
        session.execute = AsyncMock(return_value=result)

        r = await get_report(session, "report-1")
        assert r == report

    @pytest.mark.asyncio
    async def test_returns_none_for_missing(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        r = await get_report(session, "missing")
        assert r is None


class TestDeleteReport:
    """Test soft-deleting a report."""

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        session = AsyncMock()
        report = _make_report(active=True)
        # get_report is called internally; mock the chain
        result = MagicMock()
        result.scalar_one_or_none.return_value = report
        session.execute = AsyncMock(return_value=result)

        deleted = await delete_report(session, "report-1")
        assert deleted is True
        assert report.active is False
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        deleted = await delete_report(session, "missing")
        assert deleted is False
        session.commit.assert_not_called()


class TestRefreshReport:
    """Test report query refresh."""

    @pytest.mark.asyncio
    async def test_refresh_with_select_query(self):
        session = AsyncMock()
        report = _make_report(query="SELECT * FROM sales")

        # First call: get_report, second call: execute query
        result_get = MagicMock()
        result_get.scalar_one_or_none.return_value = report

        row = MagicMock()
        row._mapping = {"a": 1, "b": 2}
        result_query = MagicMock()
        result_query.fetchall.return_value = [row]

        session.execute = AsyncMock(side_effect=[result_get, result_query])

        refreshed = await refresh_report(session, "report-1")
        assert refreshed == report
        assert report.last_result == [{"a": 1, "b": 2}]

    @pytest.mark.asyncio
    async def test_refresh_no_query_returns_report(self):
        session = AsyncMock()
        report = _make_report(query=None)
        result = MagicMock()
        result.scalar_one_or_none.return_value = report
        session.execute = AsyncMock(return_value=result)

        refreshed = await refresh_report(session, "report-1")
        assert refreshed == report
        # execute called only once (for get_report), not for query
        assert session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_refresh_nonexistent_returns_none(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        refreshed = await refresh_report(session, "missing")
        assert refreshed is None

    @pytest.mark.asyncio
    async def test_refresh_non_select_skipped(self):
        session = AsyncMock()
        report = _make_report(query="DELETE FROM sales")
        result = MagicMock()
        result.scalar_one_or_none.return_value = report
        session.execute = AsyncMock(return_value=result)

        refreshed = await refresh_report(session, "report-1")
        assert refreshed == report
        # Only one execute call (get_report), not the DELETE
        assert session.execute.call_count == 1


class TestGetLastRun:
    """Test getting the most recent discovery run."""

    @pytest.mark.asyncio
    async def test_returns_latest_run(self):
        session = AsyncMock()
        run = MagicMock(spec=DiscoveryRun)
        run.status = "completed"
        result = MagicMock()
        result.scalar_one_or_none.return_value = run
        session.execute = AsyncMock(return_value=result)

        last = await get_last_run(session)
        assert last == run

    @pytest.mark.asyncio
    async def test_returns_none_when_no_runs(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        last = await get_last_run(session)
        assert last is None
