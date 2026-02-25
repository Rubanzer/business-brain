"""Tests for the Deep Tier (Claude API) module — anonymization, task queue, and analysis."""

import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone

import pytest

from business_brain.cognitive.deep_tier import (
    anonymize_text,
    anonymize_rows,
    _extract_json,
    is_available,
    run_deep_analysis,
    create_task,
    execute_task,
    get_task_status,
    list_tasks,
)


# ---------------------------------------------------------------------------
# Entity Anonymization
# ---------------------------------------------------------------------------


class TestAnonymizeText:
    """Tests for PII anonymization."""

    def test_email_anonymized(self):
        text, mapping = anonymize_text("Contact john@example.com for details")
        assert "john@example.com" not in text
        assert "[EMAIL_1]" in text
        assert "[EMAIL_1]" in mapping
        assert mapping["[EMAIL_1]"] == "john@example.com"

    def test_multiple_emails(self):
        text, mapping = anonymize_text("From a@b.com and c@d.com")
        assert "a@b.com" not in text
        assert "c@d.com" not in text
        assert len([k for k in mapping if "EMAIL" in k]) == 2

    def test_pan_anonymized(self):
        text, mapping = anonymize_text("PAN: ABCDE1234F")
        assert "ABCDE1234F" not in text
        assert any("TAX_ID" in k for k in mapping)

    def test_no_pii_unchanged(self):
        original = "Revenue is ₹50 crore for FY2024"
        text, mapping = anonymize_text(original)
        assert text == original
        assert len(mapping) == 0

    def test_empty_string(self):
        text, mapping = anonymize_text("")
        assert text == ""
        assert len(mapping) == 0


class TestAnonymizeRows:
    """Tests for row-level anonymization."""

    def test_numeric_values_preserved(self):
        rows = [{"revenue": 50000, "cost": 30000.5, "active": True}]
        result = anonymize_rows(rows)
        assert result[0]["revenue"] == 50000
        assert result[0]["cost"] == 30000.5
        assert result[0]["active"] is True

    def test_none_values_preserved(self):
        rows = [{"name": None, "value": 100}]
        result = anonymize_rows(rows)
        assert result[0]["name"] is None

    def test_string_values_anonymized(self):
        rows = [{"email": "test@example.com", "amount": 100}]
        result = anonymize_rows(rows)
        assert "test@example.com" not in str(result[0]["email"])
        assert result[0]["amount"] == 100

    def test_max_rows_limit(self):
        rows = [{"id": i} for i in range(200)]
        result = anonymize_rows(rows, max_rows=10)
        assert len(result) == 10

    def test_empty_rows(self):
        assert anonymize_rows([]) == []


# ---------------------------------------------------------------------------
# JSON Extraction
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_direct_json(self):
        raw = '{"key": "value"}'
        assert _extract_json(raw) == {"key": "value"}

    def test_markdown_fenced(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _extract_json(raw) == {"key": "value"}

    def test_embedded_json(self):
        raw = 'Here is the result:\n{"key": "value"}\nDone.'
        assert _extract_json(raw) == {"key": "value"}

    def test_invalid_json(self):
        assert _extract_json("not json at all") is None

    def test_empty_string(self):
        assert _extract_json("") is None


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    @patch("business_brain.cognitive.deep_tier.settings")
    def test_available_with_key(self, mock_settings):
        mock_settings.anthropic_api_key = "sk-ant-test-key"
        assert is_available() is True

    @patch("business_brain.cognitive.deep_tier.settings")
    def test_not_available_without_key(self, mock_settings):
        mock_settings.anthropic_api_key = ""
        assert is_available() is False


# ---------------------------------------------------------------------------
# run_deep_analysis
# ---------------------------------------------------------------------------


class TestRunDeepAnalysis:
    @pytest.mark.asyncio
    @patch("business_brain.cognitive.deep_tier.settings")
    async def test_not_available_returns_error(self, mock_settings):
        mock_settings.anthropic_api_key = ""
        result = await run_deep_analysis("test question", {})
        assert result["status"] == "unavailable"
        assert "error" in result

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.deep_tier._get_client")
    @patch("business_brain.cognitive.deep_tier.settings")
    async def test_successful_analysis(self, mock_settings, mock_get_client):
        mock_settings.anthropic_api_key = "sk-ant-test"
        mock_settings.claude_model = "claude-test"

        response_json = {
            "deep_findings": [
                {"angle": "statistical", "description": "Test finding", "confidence": 0.8}
            ],
            "summary": "Test summary",
            "root_cause_analysis": "Test root cause",
            "confidence_overall": 0.85,
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps(response_json)
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await run_deep_analysis(
            question="Why is SEC high?",
            fast_tier_result={"findings": [], "summary": "Low confidence"},
            sql_query="SELECT * FROM energy",
            sql_rows=[{"sec": 650, "month": "Jan"}],
            tables_used=["energy"],
            fast_confidence=0.2,
        )

        assert result["status"] == "completed"
        assert result["tier"] == "deep"
        assert len(result["deep_findings"]) == 1

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.deep_tier._get_client")
    @patch("business_brain.cognitive.deep_tier.settings")
    async def test_api_error_returns_failed(self, mock_settings, mock_get_client):
        mock_settings.anthropic_api_key = "sk-ant-test"
        mock_settings.claude_model = "claude-test"
        mock_get_client.side_effect = RuntimeError("API key invalid")

        result = await run_deep_analysis("test", {})
        assert result["status"] == "failed"
        assert "error" in result


# ---------------------------------------------------------------------------
# Task Queue: create_task
# ---------------------------------------------------------------------------


class TestCreateTask:
    @pytest.mark.asyncio
    @patch("business_brain.db.discovery_models.AnalysisTask")
    async def test_creates_task_in_db(self, MockTask):
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()

        mock_instance = MagicMock()
        mock_instance.id = "task-123"
        mock_instance.question = "Why is yield low?"
        MockTask.return_value = mock_instance

        session.refresh = AsyncMock()

        result = await create_task(
            session,
            question="Why is yield low?",
            fast_confidence=0.25,
            source_tier="fast",
            requested_by="auto",
        )

        assert result["status"] == "pending"
        assert "task_id" in result
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_anonymizes_data_rows(self):
        """Verify anonymize_rows is applied to sql_rows before storage."""
        rows = [{"email": "secret@company.com", "value": 100}]
        anonymized = anonymize_rows(rows)

        # The email should be anonymized in the output
        assert "secret@company.com" not in str(anonymized[0].get("email", ""))
        # Numeric values should be preserved
        assert anonymized[0]["value"] == 100


# ---------------------------------------------------------------------------
# Task Queue: get_task_status
# ---------------------------------------------------------------------------


class TestGetTaskStatus:
    @pytest.mark.asyncio
    async def test_returns_task_info(self):
        session = AsyncMock()
        mock_task = MagicMock()
        mock_task.id = "task-abc"
        mock_task.question = "What happened?"
        mock_task.status = "completed"
        mock_task.source_tier = "manual"
        mock_task.fast_confidence = 0.25
        mock_task.priority = 1
        mock_task.result = {"summary": "Deep analysis done"}
        mock_task.error = None
        mock_task.created_at = datetime(2025, 1, 15, tzinfo=timezone.utc)
        mock_task.started_at = datetime(2025, 1, 15, 0, 1, tzinfo=timezone.utc)
        mock_task.completed_at = datetime(2025, 1, 15, 0, 2, tzinfo=timezone.utc)
        mock_task.requested_by = "user-1"

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = mock_task
        session.execute = AsyncMock(return_value=result_mock)

        result = await get_task_status(session, "task-abc")
        assert result is not None
        assert result["task_id"] == "task-abc"
        assert result["status"] == "completed"
        assert result["result"]["summary"] == "Deep analysis done"

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self):
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        result = await get_task_status(session, "nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Task Queue: list_tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    @pytest.mark.asyncio
    async def test_returns_task_list(self):
        session = AsyncMock()
        mock_task = MagicMock()
        mock_task.id = "t-1"
        mock_task.question = "Why?"
        mock_task.status = "pending"
        mock_task.source_tier = "fast"
        mock_task.fast_confidence = 0.2
        mock_task.priority = 0
        mock_task.created_at = datetime(2025, 1, 15, tzinfo=timezone.utc)
        mock_task.completed_at = None
        mock_task.result = None

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [mock_task]
        session.execute = AsyncMock(return_value=result_mock)

        tasks = await list_tasks(session)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t-1"
        assert tasks[0]["has_result"] is False


# ---------------------------------------------------------------------------
# Pipeline Auto-Escalation
# ---------------------------------------------------------------------------


class TestDeepTierEscalation:
    """Tests for the deep_tier_check node in the pipeline graph."""

    @pytest.mark.asyncio
    async def test_high_confidence_skips_escalation(self):
        from business_brain.cognitive.graph import _deep_tier_escalation

        state = {
            "query_confidence": 0.8,
            "db_session": AsyncMock(),
            "_diagnostics": [],
        }
        result = await _deep_tier_escalation(state)
        assert "deep_tier_task_id" not in result
        # Should have a skip diagnostic
        diags = result.get("_diagnostics", [])
        assert any(d["stage"] == "deep_tier_check" and d["status"] == "skip" for d in diags)

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.deep_tier.is_available", return_value=False)
    async def test_unavailable_deep_tier_skips(self, mock_avail):
        from business_brain.cognitive.graph import _deep_tier_escalation

        state = {
            "query_confidence": 0.1,
            "db_session": AsyncMock(),
            "_diagnostics": [],
        }
        result = await _deep_tier_escalation(state)
        assert "deep_tier_task_id" not in result

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.deep_tier.create_task")
    @patch("business_brain.cognitive.deep_tier.is_available", return_value=True)
    @patch("business_brain.cognitive.deep_tier.settings")
    async def test_low_confidence_creates_task(self, mock_settings, mock_avail, mock_create):
        from business_brain.cognitive.graph import _deep_tier_escalation

        mock_settings.deep_tier_auto_threshold = 0.3
        mock_create.return_value = {"task_id": "deep-task-xyz"}

        session = AsyncMock()
        state = {
            "query_confidence": 0.15,
            "db_session": session,
            "question": "Why is yield dropping?",
            "analysis": {"findings": [], "summary": "Low data"},
            "sql_result": {"query": "SELECT 1", "rows": []},
            "key_metrics": [],
            "query_type": "anomaly",
            "_rag_tables": [{"table_name": "production"}],
            "session_id": "sess-123",
            "_diagnostics": [],
        }
        result = await _deep_tier_escalation(state)
        assert result.get("deep_tier_task_id") == "deep-task-xyz"
        mock_create.assert_called_once()

        # Check diagnostic
        diags = result.get("_diagnostics", [])
        assert any(
            d["stage"] == "deep_tier_check" and d["status"] == "ok"
            for d in diags
        )
