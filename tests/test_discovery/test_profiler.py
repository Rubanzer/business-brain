"""Tests for the discovery profiler module."""

from business_brain.cognitive.column_classifier import classify_columns
from business_brain.db.discovery_models import TableProfile
from business_brain.discovery.profiler import compute_data_quality_score, generate_suggestions


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


    def test_manufacturing_domain_suggestions(self):
        cls = {
            "columns": {
                "shift": {"semantic_type": "categorical", "cardinality": 3},
                "output_tons": {"semantic_type": "numeric_metric", "cardinality": 50},
                "power_kva": {"semantic_type": "numeric_metric", "cardinality": 50},
                "timestamp": {"semantic_type": "temporal", "cardinality": 100},
            },
            "domain_hint": "manufacturing",
        }
        profile = self._make_profile("scada_readings", cls, "manufacturing")
        suggestions = generate_suggestions([profile])

        assert len(suggestions) > 0
        assert any("production" in s.lower() or "shift" in s.lower() for s in suggestions)

    def test_quality_domain_suggestions(self):
        cls = {
            "columns": {
                "grade": {"semantic_type": "categorical", "cardinality": 5},
                "rejection_pct": {"semantic_type": "numeric_percentage", "cardinality": 30},
            },
            "domain_hint": "quality",
        }
        profile = self._make_profile("quality_data", cls, "quality")
        suggestions = generate_suggestions([profile])

        assert any("rejection" in s.lower() for s in suggestions)

    def test_logistics_domain_suggestions(self):
        cls = {
            "columns": {
                "truck_no": {"semantic_type": "identifier", "cardinality": 50},
                "weight": {"semantic_type": "numeric_metric", "cardinality": 40},
            },
            "domain_hint": "logistics",
        }
        profile = self._make_profile("gate_register", cls, "logistics")
        suggestions = generate_suggestions([profile])

        assert any("truck" in s.lower() for s in suggestions)

    def test_energy_domain_suggestions(self):
        cls = {
            "columns": {
                "timestamp": {"semantic_type": "temporal", "cardinality": 100},
                "kwh": {"semantic_type": "numeric_metric", "cardinality": 80},
            },
            "domain_hint": "energy",
        }
        profile = self._make_profile("power_readings", cls, "energy")
        suggestions = generate_suggestions([profile])

        assert any("power" in s.lower() or "consumption" in s.lower() for s in suggestions)


    def test_shift_column_generates_shift_comparison(self):
        cls = {
            "columns": {
                "shift": {"semantic_type": "categorical", "cardinality": 3},
                "output_tons": {"semantic_type": "numeric_metric", "cardinality": 50},
                "power_kva": {"semantic_type": "numeric_metric", "cardinality": 50},
            },
            "domain_hint": "manufacturing",
        }
        profile = self._make_profile("production", cls, "manufacturing")
        suggestions = generate_suggestions([profile])
        assert any("shift" in s.lower() for s in suggestions)

    def test_heat_column_generates_cycle_time_suggestion(self):
        cls = {
            "columns": {
                "heat_no": {"semantic_type": "identifier", "cardinality": 100},
                "output_tons": {"semantic_type": "numeric_metric", "cardinality": 50},
            },
            "domain_hint": "manufacturing",
        }
        profile = self._make_profile("heat_log", cls, "manufacturing")
        suggestions = generate_suggestions([profile])
        assert any("heat" in s.lower() for s in suggestions)


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

    def test_manufacturing_domain_detection(self):
        """Steel plant columns should detect manufacturing domain."""
        columns = ["heat_no", "furnace_temp", "kva", "output_tonnage", "shift"]
        sample_rows = [
            {"heat_no": "H001", "furnace_temp": 1550, "kva": 420, "output_tonnage": 28.5, "shift": "A"},
            {"heat_no": "H002", "furnace_temp": 1580, "kva": 435, "output_tonnage": 30.1, "shift": "B"},
            {"heat_no": "H003", "furnace_temp": 1540, "kva": 410, "output_tonnage": 27.8, "shift": "A"},
        ]
        result = classify_columns(columns, sample_rows)
        assert result["domain_hint"] == "manufacturing"

    def test_heat_no_classified_as_identifier(self):
        """heat_no should be identified as an identifier column."""
        columns = ["heat_no", "grade", "weight"]
        sample_rows = [
            {"heat_no": "H001", "grade": "Fe500D", "weight": 28.5},
            {"heat_no": "H002", "grade": "Fe500D", "weight": 30.1},
            {"heat_no": "H003", "grade": "Fe415", "weight": 27.8},
        ]
        result = classify_columns(columns, sample_rows)
        assert result["columns"]["heat_no"]["semantic_type"] == "identifier"

    def test_logistics_domain_detection(self):
        """Gate register data should detect logistics domain."""
        columns = ["truck_no", "gate", "vehicle_type", "weight"]
        sample_rows = [
            {"truck_no": "MH12AB1234", "gate": "Gate A", "vehicle_type": "Truck", "weight": 15.5},
            {"truck_no": "MH12CD5678", "gate": "Gate B", "vehicle_type": "Truck", "weight": 18.2},
            {"truck_no": "MH12EF9012", "gate": "Gate A", "vehicle_type": "Trailer", "weight": 22.1},
        ]
        result = classify_columns(columns, sample_rows)
        assert result["domain_hint"] == "logistics"


class TestComputeDataQualityScore:
    """Test data quality score computation."""

    def _make_profile(self, columns_dict, row_count=100, domain="general"):
        p = TableProfile()
        p.table_name = "test_table"
        p.row_count = row_count
        p.domain_hint = domain
        p.column_classification = {"columns": columns_dict, "domain_hint": domain}
        return p

    def test_perfect_score(self):
        """Table with no issues should score high."""
        cols = {
            "id": {"semantic_type": "identifier", "null_count": 0, "cardinality": 100},
            "name": {"semantic_type": "categorical", "null_count": 0, "cardinality": 20},
            "amount": {"semantic_type": "numeric_currency", "null_count": 0, "cardinality": 80,
                       "stats": {"min": 10, "max": 5000, "mean": 500, "stdev": 200}},
            "date": {"semantic_type": "temporal", "null_count": 0, "cardinality": 30},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["score"] >= 80
        assert result["breakdown"]["completeness"] == 100
        assert result["breakdown"]["validity"] == 100
        assert len(result["issues"]) == 0

    def test_high_null_rate_reduces_completeness(self):
        """Columns with > 20% nulls should reduce completeness and generate issues."""
        cols = {
            "name": {"semantic_type": "categorical", "null_count": 50, "cardinality": 10},
            "value": {"semantic_type": "numeric_metric", "null_count": 40, "cardinality": 30},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["completeness"] < 100
        assert any("missing values" in i for i in result["issues"])

    def test_zero_nulls_full_completeness(self):
        cols = {
            "col_a": {"semantic_type": "categorical", "null_count": 0, "cardinality": 5},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["completeness"] == 100

    def test_constant_column_reduces_uniqueness(self):
        """A column with cardinality=1 (and row_count>1) should reduce uniqueness score."""
        cols = {
            "status": {"semantic_type": "categorical", "null_count": 0, "cardinality": 1},
            "value": {"semantic_type": "numeric_metric", "null_count": 0, "cardinality": 50},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["uniqueness"] < 100
        assert any("constant" in i.lower() for i in result["issues"])

    def test_no_constant_columns_full_uniqueness(self):
        cols = {
            "a": {"semantic_type": "categorical", "null_count": 0, "cardinality": 5},
            "b": {"semantic_type": "numeric_metric", "null_count": 0, "cardinality": 50},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["uniqueness"] == 100

    def test_negative_currency_reduces_validity(self):
        """Negative values in currency column should reduce validity."""
        cols = {
            "price": {
                "semantic_type": "numeric_currency", "null_count": 0, "cardinality": 50,
                "stats": {"min": -100, "max": 5000, "mean": 500, "stdev": 200},
            },
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["validity"] < 100
        assert any("negative" in i.lower() for i in result["issues"])

    def test_out_of_range_percentage_reduces_validity(self):
        """Percentage values outside 0-100 should reduce validity."""
        cols = {
            "rejection_rate": {
                "semantic_type": "numeric_percentage", "null_count": 0, "cardinality": 30,
                "stats": {"min": -5, "max": 120, "mean": 50, "stdev": 20},
            },
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["validity"] < 100
        assert any("percentage" in i.lower() for i in result["issues"])

    def test_diverse_column_types_increase_diversity(self):
        """Multiple distinct semantic types should score higher diversity."""
        cols = {
            "id": {"semantic_type": "identifier", "null_count": 0, "cardinality": 100},
            "name": {"semantic_type": "categorical", "null_count": 0, "cardinality": 20},
            "amount": {"semantic_type": "numeric_currency", "null_count": 0, "cardinality": 80},
            "date": {"semantic_type": "temporal", "null_count": 0, "cardinality": 30},
            "pct": {"semantic_type": "numeric_percentage", "null_count": 0, "cardinality": 20},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["diversity"] == 100  # 5 types * 25 = 125, capped at 100

    def test_single_type_low_diversity(self):
        """All columns of same type should have low diversity."""
        cols = {
            "a": {"semantic_type": "numeric_metric", "null_count": 0, "cardinality": 50},
            "b": {"semantic_type": "numeric_metric", "null_count": 0, "cardinality": 30},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["diversity"] == 25  # 1 type * 25

    def test_empty_table_returns_zero(self):
        """Empty table (row_count=0) returns score 0."""
        cols = {"a": {"semantic_type": "text", "null_count": 0, "cardinality": 0}}
        result = compute_data_quality_score(self._make_profile(cols, row_count=0))
        assert result["score"] == 0
        assert "Empty table" in result["issues"]

    def test_no_classification_returns_zero(self):
        """Profile with no column classification returns score 0."""
        p = TableProfile()
        p.table_name = "test"
        p.row_count = 100
        p.column_classification = None
        result = compute_data_quality_score(p)
        assert result["score"] == 0

    def test_missing_columns_key_returns_zero(self):
        """Classification dict without 'columns' key returns score 0."""
        p = TableProfile()
        p.table_name = "test"
        p.row_count = 100
        p.column_classification = {"domain_hint": "general"}
        result = compute_data_quality_score(p)
        assert result["score"] == 0

    def test_score_capped_at_100(self):
        """Score should never exceed 100."""
        cols = {
            "id": {"semantic_type": "identifier", "null_count": 0, "cardinality": 100},
            "name": {"semantic_type": "categorical", "null_count": 0, "cardinality": 20},
            "amount": {"semantic_type": "numeric_currency", "null_count": 0, "cardinality": 80,
                       "stats": {"min": 10, "max": 5000}},
            "date": {"semantic_type": "temporal", "null_count": 0, "cardinality": 30},
            "pct": {"semantic_type": "numeric_percentage", "null_count": 0, "cardinality": 20,
                    "stats": {"min": 0, "max": 100}},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["score"] <= 100

    def test_multiple_validity_issues_compound(self):
        """Multiple validity issues should compound the deduction."""
        cols = {
            "price": {
                "semantic_type": "numeric_currency", "null_count": 0, "cardinality": 50,
                "stats": {"min": -50, "max": 500},
            },
            "discount_pct": {
                "semantic_type": "numeric_percentage", "null_count": 0, "cardinality": 20,
                "stats": {"min": -10, "max": 150},
            },
        }
        result = compute_data_quality_score(self._make_profile(cols))
        # 2 deductions * 30 = 60 deducted from 100 → validity = 40
        assert result["breakdown"]["validity"] == 40

    def test_breakdown_keys_present(self):
        """Result should always have score, breakdown (4 keys), and issues."""
        cols = {"a": {"semantic_type": "text", "null_count": 0, "cardinality": 5}}
        result = compute_data_quality_score(self._make_profile(cols))
        assert "score" in result
        assert "breakdown" in result
        assert "issues" in result
        assert set(result["breakdown"].keys()) == {"completeness", "uniqueness", "validity", "diversity"}

    def test_50_percent_nulls_completeness_zero(self):
        """50% average null rate should drive completeness to 0."""
        cols = {
            "a": {"semantic_type": "text", "null_count": 50, "cardinality": 5},
        }
        result = compute_data_quality_score(self._make_profile(cols, row_count=100))
        assert result["breakdown"]["completeness"] == 0

    def test_all_columns_constant_uniqueness_zero(self):
        """All columns constant → uniqueness should be 0."""
        cols = {
            "a": {"semantic_type": "categorical", "null_count": 0, "cardinality": 1},
            "b": {"semantic_type": "text", "null_count": 0, "cardinality": 1},
        }
        result = compute_data_quality_score(self._make_profile(cols))
        assert result["breakdown"]["uniqueness"] == 0
