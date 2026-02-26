"""Tests for the quarantine API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest

from business_brain.action.routers.quarantine import (
    list_quarantined,
    quarantine_summary,
    get_batch,
    approve_quarantined,
    reject_quarantined,
    bulk_review,
    ReviewRequest,
    BulkReviewRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_quarantine_item(
    id: int = 1,
    table_name: str = "sales",
    batch_id: str = "batch-abc",
    row_index: int = 0,
    row_data: dict = None,
    issues: list = None,
    status: str = "quarantined",
):
    """Create a mock DataQuarantine ORM object."""
    item = MagicMock()
    item.id = id
    item.table_name = table_name
    item.upload_batch_id = batch_id
    item.row_index = row_index
    item.row_data = row_data or {"id": "", "value": "100"}
    item.issues = issues or [{"check": "missing_identifier", "column": "id", "severity": "critical"}]
    item.validation_status = status
    item.quarantined_at = datetime(2025, 1, 15, tzinfo=timezone.utc)
    item.reviewed_by = None
    item.reviewed_at = None
    item.review_note = None
    return item


# ---------------------------------------------------------------------------
# Tests: list_quarantined
# ---------------------------------------------------------------------------


class TestListQuarantined:
    @pytest.mark.asyncio
    async def test_returns_quarantined_rows(self):
        """GET /quarantine returns quarantined rows."""
        session = AsyncMock()

        item = _mock_quarantine_item()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [item]
        session.execute = AsyncMock(side_effect=[
            result_mock,  # for the main query
            MagicMock(all=MagicMock(return_value=[])),  # for the count query
        ])

        result = await list_quarantined(
            table_name=None, status="quarantined", limit=50, session=session,
        )
        assert "rows" in result
        assert len(result["rows"]) == 1
        assert result["rows"][0]["table_name"] == "sales"

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(self):
        """DB error during list_quarantined returns empty result."""
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB down"))

        result = await list_quarantined(
            table_name=None, status="quarantined", limit=50, session=session,
        )
        assert result["rows"] == []
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# Tests: quarantine_summary
# ---------------------------------------------------------------------------


class TestQuarantineSummary:
    @pytest.mark.asyncio
    async def test_returns_summary(self):
        """GET /quarantine/summary returns counts by table and status."""
        session = AsyncMock()

        row1 = MagicMock()
        row1.table_name = "sales"
        row1.validation_status = "quarantined"
        row1.count = 5

        row2 = MagicMock()
        row2.table_name = "sales"
        row2.validation_status = "approved"
        row2.count = 2

        result_mock = MagicMock()
        result_mock.all.return_value = [row1, row2]
        session.execute = AsyncMock(return_value=result_mock)

        result = await quarantine_summary(session=session)
        assert result["total_quarantined"] == 5
        assert result["total_approved"] == 2
        assert "sales" in result["tables"]

    @pytest.mark.asyncio
    async def test_db_error_returns_zeros(self):
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB down"))

        result = await quarantine_summary(session=session)
        assert result["total_quarantined"] == 0


# ---------------------------------------------------------------------------
# Tests: approve_quarantined
# ---------------------------------------------------------------------------


class TestApproveQuarantined:
    @pytest.mark.asyncio
    async def test_approve_inserts_and_updates_status(self):
        """Approving a quarantined item inserts it into the table."""
        session = AsyncMock()
        item = _mock_quarantine_item(
            row_data={"id": "H1", "value": "100"}
        )

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = item
        session.execute = AsyncMock(return_value=result_mock)
        session.commit = AsyncMock()

        req = ReviewRequest(reviewed_by="admin-1", note="Looks ok")
        result = await approve_quarantined(item_id=1, req=req, session=session)

        assert result["status"] == "approved"
        assert result["id"] == 1
        assert item.validation_status == "approved"
        assert item.reviewed_by == "admin-1"

    @pytest.mark.asyncio
    async def test_approve_not_found(self):
        """Approving non-existent item returns error."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        req = ReviewRequest(reviewed_by="admin")
        result = await approve_quarantined(item_id=999, req=req, session=session)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_approve_already_reviewed(self):
        """Approving an already-approved item returns error."""
        session = AsyncMock()
        item = _mock_quarantine_item(status="approved")
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = item
        session.execute = AsyncMock(return_value=result_mock)

        req = ReviewRequest(reviewed_by="admin")
        result = await approve_quarantined(item_id=1, req=req, session=session)
        assert "error" in result
        assert "already" in result["error"]


# ---------------------------------------------------------------------------
# Tests: reject_quarantined
# ---------------------------------------------------------------------------


class TestRejectQuarantined:
    @pytest.mark.asyncio
    async def test_reject_updates_status(self):
        """Rejecting a quarantined item marks it as rejected."""
        session = AsyncMock()
        item = _mock_quarantine_item()

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = item
        session.execute = AsyncMock(return_value=result_mock)
        session.commit = AsyncMock()

        req = ReviewRequest(reviewed_by="admin-1", note="Bad data")
        result = await reject_quarantined(item_id=1, req=req, session=session)

        assert result["status"] == "rejected"
        assert item.validation_status == "rejected"

    @pytest.mark.asyncio
    async def test_reject_not_found(self):
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        req = ReviewRequest(reviewed_by="admin")
        result = await reject_quarantined(item_id=999, req=req, session=session)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: bulk_review
# ---------------------------------------------------------------------------


class TestBulkReview:
    @pytest.mark.asyncio
    async def test_bulk_reject(self):
        """Bulk reject multiple quarantined items."""
        session = AsyncMock()
        items = [_mock_quarantine_item(id=i) for i in range(3)]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = items
        session.execute = AsyncMock(return_value=result_mock)
        session.commit = AsyncMock()

        req = BulkReviewRequest(
            ids=[0, 1, 2],
            action="reject",
            reviewed_by="admin",
            note="Batch rejected",
        )
        result = await bulk_review(req=req, session=session)

        assert result["status"] == "completed"
        assert result["rejected"] == 3

    @pytest.mark.asyncio
    async def test_invalid_action_returns_error(self):
        session = AsyncMock()
        req = BulkReviewRequest(
            ids=[1, 2],
            action="invalid",
            reviewed_by="admin",
        )
        result = await bulk_review(req=req, session=session)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: get_batch
# ---------------------------------------------------------------------------


class TestGetBatch:
    @pytest.mark.asyncio
    async def test_returns_batch_rows(self):
        session = AsyncMock()
        items = [_mock_quarantine_item(id=i, batch_id="batch-xyz", row_index=i) for i in range(3)]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = items
        session.execute = AsyncMock(return_value=result_mock)

        result = await get_batch(batch_id="batch-xyz", session=session)
        assert result["batch_id"] == "batch-xyz"
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_batch_not_found(self):
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result_mock)

        result = await get_batch(batch_id="nonexistent", session=session)
        assert "error" in result
