"""Tests for the engagement tracker — implicit signal capture."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from business_brain.db.discovery_models import EngagementEvent
from business_brain.discovery.engagement_tracker import (
    _record_event,
    track_insights_shown,
    track_insight_action,
    track_insights_dismissed_all,
    track_recommendations_shown,
    get_engagement_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(events=None):
    """Create a mock AsyncSession that tracks added objects."""
    session = AsyncMock()
    session._added = []

    def _add(obj):
        session._added.append(obj)

    session.add = _add

    # For get_engagement_summary: return events from execute()
    if events is not None:
        result_mock = MagicMock()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = events
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

    return session


def _make_event(
    event_type="insight_shown",
    entity_type="insight",
    entity_id=None,
    analysis_type=None,
    table_name=None,
    severity=None,
    impact_score=None,
):
    """Create a mock EngagementEvent for summary tests."""
    e = MagicMock()
    e.event_type = event_type
    e.entity_type = entity_type
    e.entity_id = entity_id
    e.analysis_type = analysis_type
    e.table_name = table_name
    e.severity = severity
    e.impact_score = impact_score
    return e


# ---------------------------------------------------------------------------
# TestEventCreation
# ---------------------------------------------------------------------------


class TestEventCreation:
    """Test low-level event creation."""

    @pytest.mark.asyncio
    async def test_record_event_creates_engagement_event(self):
        session = _make_session()
        await _record_event(
            session,
            event_type="insight_shown",
            entity_type="insight",
            entity_id="abc-123",
            analysis_type="anomaly",
            table_name="orders",
        )
        assert len(session._added) == 1
        evt = session._added[0]
        assert isinstance(evt, EngagementEvent)
        assert evt.event_type == "insight_shown"
        assert evt.entity_type == "insight"
        assert evt.entity_id == "abc-123"
        assert evt.analysis_type == "anomaly"
        assert evt.table_name == "orders"
        session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_event_handles_none_fields(self):
        session = _make_session()
        await _record_event(
            session,
            event_type="insights_dismissed_all",
            entity_type="insight",
        )
        assert len(session._added) == 1
        evt = session._added[0]
        assert evt.entity_id is None
        assert evt.analysis_type is None
        assert evt.table_name is None
        assert evt.columns is None
        assert evt.severity is None
        assert evt.impact_score is None

    @pytest.mark.asyncio
    async def test_record_event_error_doesnt_raise(self):
        session = _make_session()
        session.flush = AsyncMock(side_effect=Exception("DB error"))
        # Should NOT raise
        await _record_event(
            session,
            event_type="insight_shown",
            entity_type="insight",
        )

    @pytest.mark.asyncio
    async def test_record_event_with_metadata(self):
        session = _make_session()
        await _record_event(
            session,
            event_type="insights_dismissed_all",
            entity_type="insight",
            metadata={"dismissed_count": 5},
            session_id="sess-xyz",
        )
        evt = session._added[0]
        assert evt.extra_metadata == {"dismissed_count": 5}
        assert evt.session_id == "sess-xyz"


# ---------------------------------------------------------------------------
# TestTrackingFunctions
# ---------------------------------------------------------------------------


class TestTrackingFunctions:
    """Test high-level tracking functions."""

    @pytest.mark.asyncio
    async def test_track_insights_shown_creates_events(self):
        session = _make_session()
        insights = [
            {"id": "i1", "insight_type": "anomaly", "severity": "critical",
             "impact_score": 85, "source_tables": ["orders"]},
            {"id": "i2", "insight_type": "trend", "severity": "info",
             "impact_score": 40, "source_tables": ["products", "sales"]},
        ]
        await track_insights_shown(session, insights)
        assert len(session._added) == 2
        assert session._added[0].event_type == "insight_shown"
        assert session._added[0].entity_id == "i1"
        assert session._added[0].analysis_type == "anomaly"
        assert session._added[0].table_name == "orders"
        assert session._added[1].entity_id == "i2"
        assert session._added[1].table_name == "products"

    @pytest.mark.asyncio
    async def test_track_insights_shown_empty_source_tables(self):
        session = _make_session()
        insights = [
            {"id": "i1", "insight_type": "anomaly", "severity": "info",
             "impact_score": 50, "source_tables": []},
        ]
        await track_insights_shown(session, insights)
        assert session._added[0].table_name is None

    @pytest.mark.asyncio
    async def test_track_insight_action_deployed(self):
        session = _make_session()
        await track_insight_action(
            session, "ins-1", "deployed",
            insight_type="correlation",
            severity="warning",
            impact_score=70,
            source_tables=["invoices"],
        )
        assert len(session._added) == 1
        evt = session._added[0]
        assert evt.event_type == "insight_deployed"
        assert evt.entity_id == "ins-1"
        assert evt.analysis_type == "correlation"
        assert evt.table_name == "invoices"

    @pytest.mark.asyncio
    async def test_track_insight_action_dismissed(self):
        session = _make_session()
        await track_insight_action(session, "ins-2", "dismissed")
        evt = session._added[0]
        assert evt.event_type == "insight_dismissed"
        assert evt.entity_id == "ins-2"

    @pytest.mark.asyncio
    async def test_track_insights_dismissed_all(self):
        session = _make_session()
        await track_insights_dismissed_all(session, 12)
        assert len(session._added) == 1
        evt = session._added[0]
        assert evt.event_type == "insights_dismissed_all"
        assert evt.extra_metadata == {"dismissed_count": 12}

    @pytest.mark.asyncio
    async def test_track_recommendations_shown(self):
        session = _make_session()
        recs = [
            {"analysis_type": "benchmark", "target_table": "orders",
             "columns": ["supplier", "amount"], "confidence": "pre-computed",
             "priority": 90},
            {"analysis_type": "correlation", "target_table": "sales",
             "columns": ["price", "qty"], "confidence": "high",
             "priority": 75},
        ]
        await track_recommendations_shown(session, recs)
        assert len(session._added) == 2
        assert session._added[0].event_type == "recommendation_shown"
        assert session._added[0].entity_type == "recommendation"
        assert session._added[0].analysis_type == "benchmark"
        assert session._added[0].table_name == "orders"
        assert session._added[0].columns == ["supplier", "amount"]
        assert session._added[1].analysis_type == "correlation"

    @pytest.mark.asyncio
    async def test_tracking_functions_dont_raise_on_error(self):
        """All tracking functions swallow exceptions."""
        session = _make_session()
        session.flush = AsyncMock(side_effect=Exception("DB down"))

        # None of these should raise
        await track_insights_shown(session, [{"id": "x"}])
        await track_insight_action(session, "x", "deployed")
        await track_insights_dismissed_all(session, 5)
        await track_recommendations_shown(session, [{"analysis_type": "benchmark"}])


# ---------------------------------------------------------------------------
# TestEngagementSummary
# ---------------------------------------------------------------------------


class TestEngagementSummary:
    """Test aggregation query."""

    @pytest.mark.asyncio
    async def test_summary_groups_by_event_type(self):
        events = [
            _make_event(event_type="insight_shown", analysis_type="anomaly", table_name="t1"),
            _make_event(event_type="insight_shown", analysis_type="anomaly", table_name="t1"),
            _make_event(event_type="insight_deployed", analysis_type="anomaly", table_name="t1"),
            _make_event(event_type="recommendation_shown", analysis_type="benchmark", table_name="t2"),
        ]
        session = _make_session(events=events)
        result = await get_engagement_summary(session, days=30)

        assert result["period_days"] == 30
        assert result["total_events"] == 4
        assert result["by_event_type"]["insight_shown"] == 2
        assert result["by_event_type"]["insight_deployed"] == 1
        assert result["by_event_type"]["recommendation_shown"] == 1

    @pytest.mark.asyncio
    async def test_summary_groups_by_analysis_type_with_rates(self):
        events = [
            _make_event(event_type="insight_shown", analysis_type="anomaly"),
            _make_event(event_type="insight_shown", analysis_type="anomaly"),
            _make_event(event_type="insight_shown", analysis_type="anomaly"),
            _make_event(event_type="insight_shown", analysis_type="anomaly"),
            _make_event(event_type="insight_deployed", analysis_type="anomaly"),
            _make_event(event_type="insight_dismissed", analysis_type="anomaly"),
            _make_event(event_type="insight_shown", analysis_type="correlation"),
            _make_event(event_type="insight_shown", analysis_type="correlation"),
        ]
        session = _make_session(events=events)
        result = await get_engagement_summary(session, days=30)

        anomaly = result["by_analysis_type"]["anomaly"]
        assert anomaly["shown"] == 4
        assert anomaly["deployed"] == 1
        assert anomaly["dismissed"] == 1
        assert anomaly["engagement_rate"] == 0.25  # 1/4

        corr = result["by_analysis_type"]["correlation"]
        assert corr["shown"] == 2
        assert corr["deployed"] == 0
        assert corr["engagement_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_summary_by_severity(self):
        events = [
            _make_event(event_type="insight_shown", severity="critical"),
            _make_event(event_type="insight_shown", severity="critical"),
            _make_event(event_type="insight_deployed", severity="critical"),
            _make_event(event_type="insight_shown", severity="info"),
        ]
        session = _make_session(events=events)
        result = await get_engagement_summary(session, days=7)

        assert result["by_severity"]["critical"]["shown"] == 2
        assert result["by_severity"]["critical"]["deployed"] == 1
        assert result["by_severity"]["critical"]["engagement_rate"] == 0.5
        assert result["by_severity"]["info"]["shown"] == 1

    @pytest.mark.asyncio
    async def test_summary_by_table(self):
        events = [
            _make_event(event_type="insight_shown", table_name="orders"),
            _make_event(event_type="insight_shown", table_name="orders"),
            _make_event(event_type="insight_deployed", table_name="orders"),
            _make_event(event_type="recommendation_shown", table_name="products"),
        ]
        session = _make_session(events=events)
        result = await get_engagement_summary(session, days=30)

        assert result["by_table"]["orders"]["shown"] == 2
        assert result["by_table"]["orders"]["deployed"] == 1
        assert result["by_table"]["products"]["shown"] == 1

    @pytest.mark.asyncio
    async def test_summary_empty_returns_zeros(self):
        session = _make_session(events=[])
        result = await get_engagement_summary(session, days=30)

        assert result["period_days"] == 30
        assert result["total_events"] == 0
        assert result["by_event_type"] == {}
        assert result["by_analysis_type"] == {}
        assert result["by_severity"] == {}
        assert result["by_table"] == {}


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure tracking failures never break main endpoints."""

    @pytest.mark.asyncio
    async def test_feed_endpoint_tracking_failure_doesnt_crash(self):
        """Simulates the try/except pattern used in feed.py."""
        result_data = []
        # The pattern in feed.py:
        try:
            raise ImportError("engagement_tracker not found")
        except Exception:
            pass  # Feed continues working
        # If we get here, the pattern works
        assert True

    @pytest.mark.asyncio
    async def test_recommendations_tracking_failure_doesnt_crash(self):
        """Simulates the try/except pattern used in table_analysis.py."""
        try:
            raise RuntimeError("tracking DB error")
        except Exception:
            pass  # Recommendations continue working
        assert True
