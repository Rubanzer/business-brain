"""Tests for cross_event_correlator helper functions."""

from business_brain.db.discovery_models import TableProfile
from business_brain.discovery.cross_event_correlator import (
    _find_event_columns,
    _find_metric_columns,
)


class _Prof:
    """Lightweight stand-in for TableProfile."""

    def __init__(self, table_name, columns_dict):
        self.table_name = table_name
        self.row_count = 100
        self.column_classification = {
            "columns": columns_dict,
        }


# ---------------------------------------------------------------------------
# _find_event_columns
# ---------------------------------------------------------------------------


class TestFindEventColumns:
    def test_boolean_is_event(self):
        prof = _Prof("hr", {
            "is_absent": {"semantic_type": "boolean", "cardinality": 2},
        })
        assert _find_event_columns(prof) == ["is_absent"]

    def test_categorical_low_cardinality_is_event(self):
        prof = _Prof("hr", {
            "shift": {"semantic_type": "categorical", "cardinality": 3},
        })
        assert _find_event_columns(prof) == ["shift"]

    def test_categorical_high_cardinality_excluded(self):
        """Cardinality > 10 excludes the column."""
        prof = _Prof("t", {
            "product_name": {"semantic_type": "categorical", "cardinality": 50},
        })
        assert _find_event_columns(prof) == []

    def test_categorical_boundary_10_included(self):
        prof = _Prof("t", {
            "category": {"semantic_type": "categorical", "cardinality": 10},
        })
        assert _find_event_columns(prof) == ["category"]

    def test_categorical_boundary_11_excluded(self):
        prof = _Prof("t", {
            "category": {"semantic_type": "categorical", "cardinality": 11},
        })
        assert _find_event_columns(prof) == []

    def test_numeric_not_event(self):
        prof = _Prof("t", {
            "amount": {"semantic_type": "numeric_currency", "cardinality": 50},
        })
        assert _find_event_columns(prof) == []

    def test_identifier_not_event(self):
        prof = _Prof("t", {
            "employee_id": {"semantic_type": "identifier", "cardinality": 100},
        })
        assert _find_event_columns(prof) == []

    def test_temporal_not_event(self):
        prof = _Prof("t", {
            "date": {"semantic_type": "temporal", "cardinality": 30},
        })
        assert _find_event_columns(prof) == []

    def test_multiple_events(self):
        prof = _Prof("t", {
            "shift": {"semantic_type": "categorical", "cardinality": 3},
            "is_overtime": {"semantic_type": "boolean", "cardinality": 2},
            "amount": {"semantic_type": "numeric_metric", "cardinality": 50},
        })
        events = _find_event_columns(prof)
        assert "shift" in events
        assert "is_overtime" in events
        assert "amount" not in events

    def test_empty_classification(self):
        prof = _Prof("t", {})
        prof.column_classification = None
        assert _find_event_columns(prof) == []

    def test_missing_columns_key(self):
        prof = _Prof("t", {})
        prof.column_classification = {"domain_hint": "general"}
        assert _find_event_columns(prof) == []


# ---------------------------------------------------------------------------
# _find_metric_columns
# ---------------------------------------------------------------------------


class TestFindMetricColumns:
    def test_numeric_metric_found(self):
        prof = _Prof("t", {
            "output": {"semantic_type": "numeric_metric", "cardinality": 50},
        })
        assert _find_metric_columns(prof) == ["output"]

    def test_numeric_currency_found(self):
        prof = _Prof("t", {
            "revenue": {"semantic_type": "numeric_currency", "cardinality": 50},
        })
        assert _find_metric_columns(prof) == ["revenue"]

    def test_numeric_percentage_found(self):
        prof = _Prof("t", {
            "yield_pct": {"semantic_type": "numeric_percentage", "cardinality": 20},
        })
        assert _find_metric_columns(prof) == ["yield_pct"]

    def test_categorical_not_metric(self):
        prof = _Prof("t", {
            "category": {"semantic_type": "categorical", "cardinality": 5},
        })
        assert _find_metric_columns(prof) == []

    def test_boolean_not_metric(self):
        prof = _Prof("t", {
            "flag": {"semantic_type": "boolean", "cardinality": 2},
        })
        assert _find_metric_columns(prof) == []

    def test_multiple_metrics(self):
        prof = _Prof("t", {
            "revenue": {"semantic_type": "numeric_currency", "cardinality": 50},
            "qty": {"semantic_type": "numeric_metric", "cardinality": 30},
            "name": {"semantic_type": "text", "cardinality": 100},
        })
        metrics = _find_metric_columns(prof)
        assert "revenue" in metrics
        assert "qty" in metrics
        assert "name" not in metrics

    def test_empty_classification(self):
        prof = _Prof("t", {})
        prof.column_classification = None
        assert _find_metric_columns(prof) == []

    def test_missing_columns_key(self):
        prof = _Prof("t", {})
        prof.column_classification = {"domain_hint": "general"}
        assert _find_metric_columns(prof) == []
