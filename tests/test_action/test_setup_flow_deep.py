"""Deep edge-case tests for setup flow features — process steps, auto-link,
derived metrics, suggest metrics, and process-metric linking.

Targets edge cases not covered by the existing test suites:
- Stress and boundary conditions for multi-metric process steps
- Normalization quirks in auto-link fuzzy matching
- Formula parsing edge cases in derived metrics
- Column type filtering in suggest metrics
- Cross-step metric linking scenarios
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from business_brain.action.api import (
    create_process_step,
    update_process_step,
    auto_link_metrics,
    create_derived_metric,
    suggest_metrics,
    link_metrics_to_step,
    ProcessStepRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metric(metric_id, name, table_name=None, column_name=None):
    """Create a mock MetricThreshold with the given attributes."""
    m = MagicMock()
    m.id = metric_id
    m.metric_name = name
    m.table_name = table_name
    m.column_name = column_name
    m.auto_linked = False
    m.confidence = None
    return m


def _make_metadata_entry(table_name, columns):
    """Create a mock metadata entry.

    columns: list of dicts like [{"name": "power_consumption", "type": "float"}]
    or list of strings (auto-wrapped as {"name": s}).
    """
    entry = MagicMock()
    entry.table_name = table_name
    if columns and isinstance(columns[0], str):
        entry.columns_metadata = [{"name": c} for c in columns]
    else:
        entry.columns_metadata = columns
    return entry


def _session_with_refresh():
    """Return an AsyncMock session whose refresh assigns a sequential id."""
    session = AsyncMock()
    session.add = MagicMock()
    counter = {"next_id": 1}

    async def fake_refresh(obj):
        obj.id = counter["next_id"]
        counter["next_id"] += 1

    session.refresh = AsyncMock(side_effect=fake_refresh)
    return session


def _make_step_found_result(step_id=1):
    """Return (mock result, mock step) where scalar_one_or_none returns the step."""
    step = MagicMock()
    step.id = step_id
    result = MagicMock()
    result.scalar_one_or_none.return_value = step
    return result, step


def _make_not_found_result():
    """Return a mock result where scalar_one_or_none returns None."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    return result


def _make_metric_mock(metric_id, name="test_metric", table_name="data", column_name="col"):
    """Create a mock MetricThreshold for link tests."""
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


# ===========================================================================
# TestProcessStepEdgeCases
# ===========================================================================


class TestProcessStepEdgeCases:
    """Edge cases for create_process_step and update_process_step with multi-metric handling."""

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_ten_metrics_stress(self, mock_regen):
        """Process step with 10 key_metrics stores all of them correctly."""
        session = _session_with_refresh()

        metrics = [f"metric_{i}" for i in range(10)]
        ranges = {m: f"{i*100}-{i*100+99}" for i, m in enumerate(metrics)}

        req = ProcessStepRequest(
            step_order=1,
            process_name="Complex Smelting",
            inputs="Raw Ore, Flux",
            outputs="Refined Metal",
            key_metrics=metrics,
            target_ranges=ranges,
            linked_table="complex_data",
        )

        result = await create_process_step(req, session)

        assert result["status"] == "created"
        added_step = session.add.call_args[0][0]
        assert len(added_step.key_metrics) == 10
        assert added_step.key_metrics == metrics
        assert added_step.key_metric == "metric_0"  # first element
        assert added_step.target_ranges == ranges
        session.commit.assert_called_once()
        mock_regen.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_metric_with_special_characters(self, mock_regen):
        """Metric names with special characters like parentheses, slashes, hyphens work."""
        session = _session_with_refresh()

        special_metrics = [
            "Yield (%)",
            "kWh/ton",
            "Tap-to-tap (min)",
            "Temperature [C]",
            "Power & Energy",
        ]

        req = ProcessStepRequest(
            step_order=2,
            process_name="EAF Operation",
            key_metrics=special_metrics,
        )

        result = await create_process_step(req, session)

        assert result["status"] == "created"
        added_step = session.add.call_args[0][0]
        assert added_step.key_metrics == special_metrics
        assert added_step.key_metric == "Yield (%)"

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_duplicate_metric_names_stored(self, mock_regen):
        """Duplicate metric names in key_metrics are stored as-is with no deduplication."""
        session = _session_with_refresh()

        req = ProcessStepRequest(
            step_order=3,
            process_name="Refining",
            key_metrics=["SEC", "SEC", "Yield"],
        )

        result = await create_process_step(req, session)

        assert result["status"] == "created"
        added_step = session.add.call_args[0][0]
        assert added_step.key_metrics == ["SEC", "SEC", "Yield"]
        assert len(added_step.key_metrics) == 3

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_empty_string_metrics_in_array(self, mock_regen):
        """Empty strings in key_metrics array are stored as-is — no filtering."""
        session = _session_with_refresh()

        req = ProcessStepRequest(
            step_order=4,
            process_name="Casting",
            key_metrics=["SEC", "", "Yield"],
        )

        result = await create_process_step(req, session)

        assert result["status"] == "created"
        added_step = session.add.call_args[0][0]
        assert added_step.key_metrics == ["SEC", "", "Yield"]
        # First element "SEC" becomes key_metric
        assert added_step.key_metric == "SEC"

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_process_step_no_metrics_allowed(self, mock_regen):
        """Empty key_metrics=[] is valid — some steps are informational only."""
        session = _session_with_refresh()

        req = ProcessStepRequest(
            step_order=5,
            process_name="Inspection",
            key_metrics=[],
            key_metric="",
        )

        result = await create_process_step(req, session)

        assert result["status"] == "created"
        added_step = session.add.call_args[0][0]
        assert added_step.key_metrics == []
        assert added_step.key_metric == ""
        assert added_step.target_ranges == {}

    @pytest.mark.asyncio
    @patch("business_brain.action.api._regenerate_process_context", new_callable=AsyncMock)
    async def test_very_long_metric_name(self, mock_regen):
        """A 500-character metric name is stored — key_metrics is JSON, not varchar-limited."""
        session = _session_with_refresh()

        long_name = "A" * 500

        req = ProcessStepRequest(
            step_order=6,
            process_name="Data Collection",
            key_metrics=[long_name],
        )

        result = await create_process_step(req, session)

        assert result["status"] == "created"
        added_step = session.add.call_args[0][0]
        assert added_step.key_metrics == [long_name]
        assert len(added_step.key_metrics[0]) == 500
        assert added_step.key_metric == long_name


# ===========================================================================
# TestAutoLinkEdgeCases
# ===========================================================================


class TestAutoLinkEdgeCases:
    """Edge cases for auto_link_metrics fuzzy matching — normalization, empty states,
    multi-table disambiguation."""

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_auto_link_underscore_vs_space(self, mock_store):
        """Metric 'power consumption' (with space) matches column 'power_consumption'
        (with underscore) because both are normalized to 'power_consumption'."""
        session = AsyncMock()

        metric = _make_metric(1, "power consumption")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("scada", [
                {"name": "power_consumption", "type": "float"},
            ])
        ])

        result = await auto_link_metrics(session)

        # "power consumption" -> normalized "power_consumption"
        # "power_consumption" -> normalized "power_consumption"
        # exact match after normalization -> 1.0 confidence -> auto-linked
        assert len(result["auto_linked"]) == 1
        assert result["auto_linked"][0]["confidence"] == 1.0
        assert result["auto_linked"][0]["column_name"] == "power_consumption"
        assert metric.auto_linked is True

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_auto_link_hyphen_normalization(self, mock_store):
        """Metric 'power-consumption' (with hyphen) matches column 'power_consumption'
        because hyphens are normalized to underscores."""
        session = AsyncMock()

        metric = _make_metric(2, "power-consumption")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("readings", [
                {"name": "power_consumption", "type": "float"},
            ])
        ])

        result = await auto_link_metrics(session)

        # "power-consumption" -> normalize: lower().replace(" ","_").replace("-","_") = "power_consumption"
        # "power_consumption" -> normalize: same = "power_consumption"
        # exact match -> 1.0
        assert len(result["auto_linked"]) == 1
        assert result["auto_linked"][0]["confidence"] == 1.0
        assert result["auto_linked"][0]["column_name"] == "power_consumption"

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_auto_link_case_and_underscore_combined(self, mock_store):
        """'Power Consumption' (mixed case + spaces) matches 'power_consumption'
        with at least 0.95 confidence after normalization."""
        session = AsyncMock()

        metric = _make_metric(3, "Power Consumption")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", [
                {"name": "power_consumption", "type": "float"},
            ])
        ])

        result = await auto_link_metrics(session)

        # "Power Consumption" -> lower() = "power consumption"
        # replace(" ","_") = "power_consumption", replace("-","_") = "power_consumption"
        # Column "power_consumption" -> same normalization = "power_consumption"
        # Exact match after normalization -> 1.0 confidence -> auto-linked
        assert len(result["auto_linked"]) == 1
        assert result["auto_linked"][0]["confidence"] >= 0.95
        assert result["auto_linked"][0]["column_name"] == "power_consumption"
        assert metric.table_name == "energy"

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_auto_link_no_unlinked_metrics_empty_result(self, mock_store):
        """When all metrics are already linked, result has empty auto_linked,
        suggestions, and unmatched lists."""
        session = AsyncMock()

        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("data", [
                {"name": "col_a", "type": "float"},
                {"name": "col_b", "type": "int"},
            ])
        ])

        result = await auto_link_metrics(session)

        assert result["auto_linked"] == []
        assert result["suggestions"] == []
        assert result["unmatched"] == []
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_auto_link_multiple_tables_best_match(self, mock_store):
        """Metric matching columns in two tables picks the highest-confidence match.
        Exact match in table2 beats substring match in table1."""
        session = AsyncMock()

        metric = _make_metric(4, "temperature")
        unlinked_result = MagicMock()
        unlinked_result.scalars.return_value.all.return_value = [metric]
        session.execute = AsyncMock(return_value=unlinked_result)

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("furnace_logs", [
                # "temperature" is substring of "temperature_celsius" -> 0.7
                {"name": "temperature_celsius", "type": "float"},
            ]),
            _make_metadata_entry("scada_readings", [
                # exact match "temperature" -> 1.0
                {"name": "temperature", "type": "float"},
            ]),
        ])

        result = await auto_link_metrics(session)

        assert len(result["auto_linked"]) == 1
        linked = result["auto_linked"][0]
        assert linked["confidence"] == 1.0
        assert linked["table_name"] == "scada_readings"
        assert linked["column_name"] == "temperature"


# ===========================================================================
# TestDerivedMetricEdgeCases
# ===========================================================================


class TestDerivedMetricEdgeCases:
    """Edge cases for create_derived_metric — formula parsing, validation, multi-table refs."""

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_derived_formula_with_constants(self, mock_store):
        """Formula with only constants and no table.column references is valid
        and stores source_columns=[]."""
        session = _session_with_refresh()

        mock_store.get_all = AsyncMock(return_value=[])

        body = {
            "metric_name": "Fixed Overhead",
            "formula": "100 + 200",
        }

        result = await create_derived_metric(body, session)

        assert result["status"] == "created"
        assert result["is_derived"] is True
        added = session.add.call_args[0][0]
        assert added.source_columns == []
        assert added.formula == "100 + 200"
        assert added.is_derived is True

    @pytest.mark.asyncio
    async def test_derived_whitespace_metric_name_rejected(self):
        """Whitespace-only metric_name raises HTTPException(400)."""
        session = AsyncMock()

        body = {
            "metric_name": "   ",
            "formula": "energy.kwh / production.output",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "metric_name" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_derived_formula_multiple_tables(self, mock_store):
        """Formula referencing columns from two different tables parses
        source_columns from both."""
        session = _session_with_refresh()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", ["kwh"]),
            _make_metadata_entry("production", ["output"]),
        ])

        body = {
            "metric_name": "Specific Energy Consumption",
            "formula": "energy.kwh / production.output",
            "unit": "kWh/MT",
        }

        result = await create_derived_metric(body, session)

        assert result["status"] == "created"
        added = session.add.call_args[0][0]
        assert "energy.kwh" in added.source_columns
        assert "production.output" in added.source_columns
        assert len(added.source_columns) == 2

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_derived_formula_nonexistent_table_raises(self, mock_store):
        """Formula referencing a table that does not exist raises 400."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", ["kwh"]),
        ])

        body = {
            "metric_name": "Bad Metric",
            "formula": "ghost_table.col / energy.kwh",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "ghost_table" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_derived_formula_nonexistent_column_raises(self, mock_store):
        """Formula referencing a nonexistent column in a valid table raises 400."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("energy", ["kwh", "total_cost"]),
        ])

        body = {
            "metric_name": "Bad Column Metric",
            "formula": "energy.kwh / energy.nonexistent",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "nonexistent" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_derived_empty_formula_rejected(self):
        """Empty formula raises HTTPException(400)."""
        session = AsyncMock()

        body = {
            "metric_name": "Valid Name",
            "formula": "   ",
        }

        with pytest.raises(HTTPException) as exc_info:
            await create_derived_metric(body, session)

        assert exc_info.value.status_code == 400
        assert "formula" in exc_info.value.detail


# ===========================================================================
# TestSuggestMetricsEdgeCases
# ===========================================================================


class TestSuggestMetricsEdgeCases:
    """Edge cases for suggest_metrics — ID column exclusion, empty tables, text-only tables."""

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_suggest_skips_id_columns(self, mock_store):
        """Columns named 'id', 'user_id', 'order_id' are excluded from suggestions
        even though they are numeric."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("orders", [
                {"name": "id", "type": "integer"},
                {"name": "user_id", "type": "integer"},
                {"name": "order_id", "type": "bigint"},
                {"name": "amount", "type": "float"},
                {"name": "quantity", "type": "int"},
            ])
        ])

        existing_result = MagicMock()
        existing_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=existing_result)

        result = await suggest_metrics(session)

        assert len(result["suggestions"]) == 1
        col_names = [c["column_name"] for c in result["suggestions"][0]["columns"]]
        assert "id" not in col_names
        assert "user_id" not in col_names
        assert "order_id" not in col_names
        assert "amount" in col_names
        assert "quantity" in col_names

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_suggest_no_tables_uploaded(self, mock_store):
        """Empty metadata returns suggestions=[] with a helpful message."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[])

        result = await suggest_metrics(session)

        assert result["suggestions"] == []
        assert result["message"] == "No data uploaded yet"

    @pytest.mark.asyncio
    @patch("business_brain.action.api.metadata_store")
    async def test_suggest_only_text_columns(self, mock_store):
        """Table with only varchar/text/date columns produces no suggestions."""
        session = AsyncMock()

        mock_store.get_all = AsyncMock(return_value=[
            _make_metadata_entry("logs", [
                {"name": "message", "type": "text"},
                {"name": "severity", "type": "varchar"},
                {"name": "created_at", "type": "timestamp"},
                {"name": "category", "type": "character varying"},
            ])
        ])

        existing_result = MagicMock()
        existing_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=existing_result)

        result = await suggest_metrics(session)

        # No numeric columns -> no suggestions for this table -> empty list
        assert result["suggestions"] == []


# ===========================================================================
# TestLinkMetricsEdgeCases
# ===========================================================================


class TestLinkMetricsEdgeCases:
    """Edge cases for link_metrics_to_step — cross-step linking, empty metric_ids."""

    @pytest.mark.asyncio
    async def test_link_same_metric_to_multiple_steps(self):
        """The same metric can be linked to two different process steps independently."""
        # ---- Step 1 linkage ----
        session1 = AsyncMock()
        session1.add = MagicMock()

        step1_result, step1 = _make_step_found_result(step_id=1)
        metric = _make_metric_mock(10, "temperature")
        metric_result = MagicMock()
        metric_result.scalar_one_or_none.return_value = metric
        no_link = _make_not_found_result()

        session1.execute = AsyncMock(
            side_effect=[step1_result, metric_result, no_link]
        )

        result1 = await link_metrics_to_step(1, {"metric_ids": [10]}, session1)
        assert result1["status"] == "linked"
        assert result1["step_id"] == 1
        assert result1["metrics_linked"] == [10]

        # ---- Step 2 linkage (same metric) ----
        session2 = AsyncMock()
        session2.add = MagicMock()

        step2_result, step2 = _make_step_found_result(step_id=2)
        metric2 = _make_metric_mock(10, "temperature")
        metric2_result = MagicMock()
        metric2_result.scalar_one_or_none.return_value = metric2
        no_link2 = _make_not_found_result()

        session2.execute = AsyncMock(
            side_effect=[step2_result, metric2_result, no_link2]
        )

        result2 = await link_metrics_to_step(2, {"metric_ids": [10]}, session2)
        assert result2["status"] == "linked"
        assert result2["step_id"] == 2
        assert result2["metrics_linked"] == [10]

        # Both calls succeeded independently
        session1.add.assert_called_once()
        session2.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_empty_metric_ids_returns_400(self):
        """Empty metric_ids=[] raises HTTPException(400)."""
        session = AsyncMock()

        step_result, step = _make_step_found_result(step_id=1)
        session.execute = AsyncMock(return_value=step_result)

        with pytest.raises(HTTPException) as exc_info:
            await link_metrics_to_step(1, {"metric_ids": []}, session)

        assert exc_info.value.status_code == 400
        assert "metric_ids" in exc_info.value.detail
