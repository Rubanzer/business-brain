"""Tests for action/routers/analysis.py — API serialization + request handling."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helper to build mock results for serialization
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> MagicMock:
    r = MagicMock()
    defaults = dict(
        id="res-1",
        run_id="run-1",
        operation_type="RANK",
        table_name="production",
        tier=1,
        target=["output_kg"],
        segmenters=["shift"],
        controls=[],
        join_spec=None,
        interestingness_score=0.75,
        interestingness_breakdown={"surprise": 0.8},
        quality_verdict="RELIABLE",
        domain_relevance=0.7,
        temporal_context=None,
        delta_type=None,
        final_score=0.72,
        parent_result_id=None,
        result_data={"ranked": [{"shift": "day", "avg": 100}]},
        created_at=None,
        data_hash="abc",
        dedup_key="dk",
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(r, k, v)
    return r


# ---------------------------------------------------------------------------
# _serialize_result
# ---------------------------------------------------------------------------


class TestSerializeResult:
    def test_all_fields_present(self):
        from business_brain.action.routers.analysis import _serialize_result

        r = _make_result()
        data = _serialize_result(r)

        assert data["id"] == "res-1"
        assert data["operation_type"] == "RANK"
        assert data["table_name"] == "production"
        assert data["tier"] == 1
        assert data["target"] == ["output_kg"]
        assert data["segmenters"] == ["shift"]
        assert data["controls"] == []
        assert data["interestingness_score"] == 0.75
        assert data["quality_verdict"] == "RELIABLE"
        assert data["final_score"] == 0.72
        assert data["parent_result_id"] is None
        assert "result_data" in data

    def test_created_at_string_conversion(self):
        from business_brain.action.routers.analysis import _serialize_result
        from datetime import datetime

        r = _make_result(created_at=datetime(2025, 1, 15, 10, 30, 0))
        data = _serialize_result(r)
        assert "2025" in data["created_at"]

    def test_none_created_at(self):
        from business_brain.action.routers.analysis import _serialize_result

        r = _make_result(created_at=None)
        data = _serialize_result(r)
        assert data["created_at"] is None


# ---------------------------------------------------------------------------
# Request model validation
# ---------------------------------------------------------------------------


class TestRequestModels:
    def test_explore_request_minimal(self):
        from business_brain.action.routers.analysis import ExploreRequest

        req = ExploreRequest(table_names=["sales"])
        assert req.table_names == ["sales"]
        assert req.time_scope is None
        assert req.budget is None

    def test_explore_request_full(self):
        from business_brain.action.routers.analysis import ExploreRequest

        req = ExploreRequest(
            table_names=["sales", "orders"],
            time_scope={"column": "date", "window": "30d"},
            budget={"budgeted_tier_limits": {2: 50}},
        )
        assert len(req.table_names) == 2
        assert req.time_scope["window"] == "30d"

    def test_diagnose_request(self):
        from business_brain.action.routers.analysis import DiagnoseRequest

        req = DiagnoseRequest(question="Why did output drop?")
        assert req.question == "Why did output drop?"
        assert req.table_names is None

    def test_monitor_request(self):
        from business_brain.action.routers.analysis import MonitorRequest

        req = MonitorRequest()
        assert req.table_names is None

    def test_feedback_request(self):
        from business_brain.action.routers.analysis import FeedbackRequest

        req = FeedbackRequest(feedback_type="useful", comment="Good insight")
        assert req.feedback_type == "useful"
        assert req.comment == "Good insight"
        assert req.session_id is None


# ---------------------------------------------------------------------------
# Endpoint imports (verify router can be imported)
# ---------------------------------------------------------------------------


class TestRouterImport:
    def test_router_exists(self):
        from business_brain.action.routers.analysis import router
        assert router is not None

    def test_router_has_routes(self):
        from business_brain.action.routers.analysis import router
        routes = [r.path for r in router.routes]
        assert "/analysis/explore" in routes
        assert "/analysis/diagnose" in routes
        assert "/analysis/monitor" in routes
        assert "/analysis/runs" in routes
        assert "/analysis/learning/state" in routes
