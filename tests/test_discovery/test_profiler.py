"""Tests for the discovery profiler module."""

from business_brain.cognitive.column_classifier import classify_columns
from business_brain.db.discovery_models import TableProfile
from business_brain.discovery.profiler import generate_suggestions


class TestGenerateSuggestions:
    """Test smart question suggestion generation."""

    def _make_profile(self, table_name, columns_data, domain_hint="general"):
        """Helper to create a TableProfile-like object."""
        p = TableProfile()
        p.table_name = table_name
        p.row_count = 100
        p.domain_hint = domain_hint
        p.column_classification = columns_data
        return p

    def test_categorical_numeric_suggestion(self):
        cls = {
            "columns": {
                "PARTY": {"semantic_type": "categorical", "cardinality": 5},
                "RATE": {"semantic_type": "numeric_currency", "cardinality": 50},
            },
            "domain_hint": "procurement",
        }
        profile = self._make_profile("procurement_data", cls, "procurement")
        suggestions = generate_suggestions([profile])

        assert len(suggestions) > 0
        assert any("RATE" in s and "PARTY" in s for s in suggestions)

    def test_temporal_numeric_suggestion(self):
        cls = {
            "columns": {
                "order_date": {"semantic_type": "temporal", "cardinality": 30},
                "revenue": {"semantic_type": "numeric_currency", "cardinality": 80},
            },
            "domain_hint": "sales",
        }
        profile = self._make_profile("orders", cls, "sales")
        suggestions = generate_suggestions([profile])

        assert any("trend" in s.lower() for s in suggestions)

    def test_two_numeric_correlation_suggestion(self):
        cls = {
            "columns": {
                "price": {"semantic_type": "numeric_currency", "cardinality": 50},
                "quantity": {"semantic_type": "numeric_metric", "cardinality": 30},
            },
            "domain_hint": "sales",
        }
        profile = self._make_profile("products", cls, "sales")
        suggestions = generate_suggestions([profile])

        assert any("correlation" in s.lower() for s in suggestions)

    def test_domain_specific_suggestions(self):
        cls = {
            "columns": {
                "supplier": {"semantic_type": "categorical", "cardinality": 10},
                "cost": {"semantic_type": "numeric_currency", "cardinality": 50},
            },
            "domain_hint": "procurement",
        }
        profile = self._make_profile("suppliers", cls, "procurement")
        suggestions = generate_suggestions([profile])

        assert any("supplier" in s.lower() for s in suggestions)

    def test_max_five_suggestions(self):
        cls = {
            "columns": {
                "category": {"semantic_type": "categorical", "cardinality": 5},
                "revenue": {"semantic_type": "numeric_currency", "cardinality": 80},
                "quantity": {"semantic_type": "numeric_metric", "cardinality": 30},
                "date": {"semantic_type": "temporal", "cardinality": 30},
                "cost": {"semantic_type": "numeric_currency", "cardinality": 50},
            },
            "domain_hint": "sales",
        }
        profile = self._make_profile("data", cls, "sales")
        suggestions = generate_suggestions([profile])

        assert len(suggestions) <= 5

    def test_empty_profiles(self):
        suggestions = generate_suggestions([])
        assert suggestions == []

    def test_no_classification_data(self):
        profile = self._make_profile("empty_table", None)
        suggestions = generate_suggestions([profile])
        assert suggestions == []


class TestClassifyColumnsIntegration:
    """Verify classify_columns works as profiler expects."""

    def test_basic_classification(self):
        columns = ["id", "name", "price", "created_at"]
        sample_rows = [
            {"id": 1, "name": "Widget", "price": 29.99, "created_at": "2024-01-01"},
            {"id": 2, "name": "Gadget", "price": 49.99, "created_at": "2024-01-02"},
            {"id": 3, "name": "Gizmo", "price": 19.99, "created_at": "2024-01-03"},
        ]
        col_types = {"id": "integer", "name": "text", "price": "numeric", "created_at": "date"}

        result = classify_columns(columns, sample_rows, col_types)

        assert "columns" in result
        assert "domain_hint" in result
        assert result["columns"]["created_at"]["semantic_type"] == "temporal"
