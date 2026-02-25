"""Comprehensive error-handling tests for Business Brain API endpoints.

Verifies that endpoints degrade gracefully when DB operations fail,
returning safe defaults (empty lists, error dicts) rather than
unhandled exceptions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from business_brain.action.api import (
    app,
    get_focus,
    get_table_metadata,
    list_invites,
    list_metadata,
    list_process_steps,
    create_process_step,
    revoke_invite,
    update_focus,
    update_process_step,
    upload_csv,
    ProcessStepRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _failing_session(error_msg: str = "DB connection lost") -> AsyncMock:
    """Return an AsyncMock session whose execute() always raises."""
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception(error_msg))
    session.rollback = AsyncMock()
    session.commit = AsyncMock(side_effect=Exception(error_msg))
    return session


def _admin_user() -> dict:
    """Return a minimal admin user dict for endpoints requiring auth."""
    return {"sub": "admin-1", "role": "admin", "email": "admin@test.com"}


# ---------------------------------------------------------------------------
# TestMetadataErrorHandling
# ---------------------------------------------------------------------------


class TestMetadataErrorHandling:
    """Error handling for /metadata endpoints."""

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_list_metadata_db_error_returns_empty(self, mock_metadata_store):
        """When _get_accessible_tables or metadata_store raises, list_metadata returns []."""
        session = _failing_session()
        # The function calls _get_accessible_tables first which calls metadata_store,
        # but session.execute failing will be caught by the try/except.
        mock_metadata_store.get_all = AsyncMock(side_effect=Exception("DB down"))

        result = await list_metadata(session, user=None)

        assert result == []

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_get_table_metadata_db_error_returns_error_dict(self, mock_metadata_store):
        """When DB fails, get_table_metadata returns {'error': 'Failed to fetch table metadata'}."""
        session = _failing_session()
        mock_metadata_store.get_by_table = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        # _get_accessible_tables with user=None returns None (no access check),
        # so it proceeds to metadata_store.get_by_table which raises.
        result = await get_table_metadata("sales", session, user=None)

        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "Failed to fetch table metadata"

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_list_metadata_with_none_user_still_works(self, mock_metadata_store):
        """user=None should not cause any additional errors — it means no auth."""
        entry = MagicMock(
            table_name="orders",
            description="Order data",
            columns_metadata=["id", "total"],
        )
        mock_metadata_store.get_all = AsyncMock(return_value=[entry])

        session = AsyncMock()
        # _get_accessible_tables returns None for user=None (no filtering)
        result = await list_metadata(session, user=None)

        assert len(result) == 1
        assert result[0]["table_name"] == "orders"


# ---------------------------------------------------------------------------
# TestProcessStepErrorHandling
# ---------------------------------------------------------------------------


class TestProcessStepErrorHandling:
    """Error handling for /process-steps endpoints."""

    @pytest.mark.asyncio
    async def test_list_process_steps_db_error(self):
        """DB error during list_process_steps should propagate (no try/except guard)."""
        session = _failing_session("Connection timed out")

        # list_process_steps has no try/except, so the exception propagates
        # and gets caught by the global exception handler.
        with pytest.raises(Exception, match="Connection timed out"):
            await list_process_steps(session)

    @pytest.mark.asyncio
    async def test_create_process_step_db_error(self):
        """session.commit raising should propagate from create_process_step."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock(side_effect=Exception("Disk full"))

        req = ProcessStepRequest(
            step_order=1,
            process_name="Mixing",
            inputs="raw material",
            outputs="mixture",
            key_metric="throughput",
        )

        with pytest.raises(Exception, match="Disk full"):
            await create_process_step(req, session)

    @pytest.mark.asyncio
    async def test_create_step_missing_name(self):
        """ProcessStepRequest requires process_name; empty string is accepted by Pydantic
        but creates a valid step with empty name."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        req = ProcessStepRequest(
            step_order=0,
            process_name="",
        )

        with patch("business_brain.action.routers.process._regenerate_process_context", new_callable=AsyncMock):
            # Mock the step object returned after refresh
            result = await create_process_step(req, session)

        # The endpoint should still succeed — empty string is valid
        assert result["status"] == "created"

    @pytest.mark.asyncio
    async def test_update_nonexistent_step(self):
        """Updating a step_id that doesn't exist returns {'error': 'Process step not found'}."""
        session = AsyncMock()

        # scalar_one_or_none returns None → step not found
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        req = ProcessStepRequest(
            step_order=1,
            process_name="Ghost Step",
        )

        result = await update_process_step(999, req, session)

        assert isinstance(result, dict)
        assert result["error"] == "Process step not found"


# ---------------------------------------------------------------------------
# TestInviteErrorHandling
# ---------------------------------------------------------------------------


class TestInviteErrorHandling:
    """Error handling for invite-related endpoints."""

    @pytest.mark.asyncio
    async def test_list_invites_db_error_returns_empty(self):
        """When session.execute raises in list_invites, it returns []."""
        session = _failing_session("Connection reset by peer")
        user = _admin_user()

        result = await list_invites(session, user)

        assert result == []

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_invite_returns_404(self):
        """Revoking an invite that doesn't exist raises HTTPException(404)."""
        session = AsyncMock()

        # scalars().first() returns None → invite not found
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        user = _admin_user()

        with pytest.raises(HTTPException) as exc_info:
            await revoke_invite("nonexistent-id", session, user)

        assert exc_info.value.status_code == 404
        assert "Invite not found" in exc_info.value.detail


# ---------------------------------------------------------------------------
# TestFocusErrorHandling
# ---------------------------------------------------------------------------


class TestFocusErrorHandling:
    """Error handling for /focus endpoints."""

    @pytest.mark.asyncio
    async def test_get_focus_db_error(self):
        """DB error in get_focus returns safe default: {active: False, tables: [], total: 0}."""
        session = _failing_session("Relation does not exist")

        result = await get_focus(session)

        assert isinstance(result, dict)
        assert result["active"] is False
        assert result["tables"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_put_focus_db_error(self, mock_metadata_store):
        """DB error when validating tables in update_focus returns 500 JSON error."""
        mock_metadata_store.get_all = AsyncMock(
            side_effect=Exception("Cannot reach database")
        )
        session = AsyncMock()

        body = {
            "tables": [
                {"table_name": "sales", "is_included": True},
            ]
        }

        # update_focus catches DB errors and returns a JSONResponse with 500 status
        result = await update_focus(body, session)
        assert result.status_code == 500


# ---------------------------------------------------------------------------
# TestUploadErrorHandling
# ---------------------------------------------------------------------------


class TestUploadErrorHandling:
    """Error handling for /csv upload endpoint."""

    @pytest.mark.asyncio
    async def test_csv_upload_corrupt_file(self):
        """Non-CSV binary data returns {'error': '...'}."""
        session = AsyncMock()
        user = None

        # Simulate UploadFile with garbage binary content
        file = AsyncMock()
        file.read = AsyncMock(return_value=b"\x00\x01\x02\xff\xfe\xfd")
        file.filename = "corrupt.csv"

        result = await upload_csv(file, session, user)

        assert isinstance(result, dict)
        assert "error" in result
        # The error string should mention a decode/parse problem
        assert len(result["error"]) > 0

    @pytest.mark.asyncio
    async def test_csv_upload_empty_file(self):
        """Empty file should return an error dict."""
        session = AsyncMock()
        user = None

        file = AsyncMock()
        file.read = AsyncMock(return_value=b"")
        file.filename = "empty.csv"

        result = await upload_csv(file, session, user)

        assert isinstance(result, dict)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestGlobalExceptionHandler
# ---------------------------------------------------------------------------


class TestGlobalExceptionHandler:
    """Tests for the app-level global exception handler."""

    def test_global_handler_registered_on_app(self):
        """The app should have a catch-all Exception handler registered."""
        # FastAPI stores exception handlers keyed by exception class
        assert Exception in app.exception_handlers
        handler = app.exception_handlers[Exception]
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_unhandled_exception_format(self):
        """The global handler returns JSON with status 500 and a 'detail' key."""
        handler = app.exception_handlers[Exception]

        # Build a minimal mock Request
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url = MagicMock()
        mock_request.url.path = "/some/endpoint"

        exc = RuntimeError("Unexpected kaboom")

        response = await handler(mock_request, exc)

        assert response.status_code == 500
        # JSONResponse stores body as bytes; decode and check
        import json

        body = json.loads(response.body.decode("utf-8"))
        assert "detail" in body
        assert "Unexpected kaboom" in body["detail"]
        assert "error" in body
        assert body["status"] == "failed"
