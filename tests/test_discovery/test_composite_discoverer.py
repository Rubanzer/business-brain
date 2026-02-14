"""Tests for the composite metric discoverer module."""

from business_brain.db.discovery_models import DiscoveredRelationship, TableProfile
from business_brain.discovery.composite_discoverer import (
    TEMPLATES,
    CompositeTemplate,
    _build_composite_query,
    _check_entity,
    discover_composites,
)


class TestCompositeTemplate:
    """Test CompositeTemplate matching logic."""

    def test_supplier_risk_matches(self):
        template = TEMPLATES[2]  # Supplier Risk Score
        assert template.name == "Supplier Risk Score"

        all_columns = [
            ("procurement", "rate", {"semantic_type": "numeric_currency"}),
            ("procurement", "quality_score", {"semantic_type": "numeric_metric"}),
            ("procurement", "supplier", {"semantic_type": "categorical"}),
        ]

        matched = template.match_columns(all_columns)
        assert len(matched) == 2  # rate + quality
        assert ("procurement", "rate") in matched
        assert ("procurement", "quality_score") in matched

    def test_product_profitability_matches(self):
        template = TEMPLATES[3]  # Product Profitability
        assert template.name == "Product Profitability"

        all_columns = [
            ("sales", "revenue", {"semantic_type": "numeric_currency"}),
            ("sales", "cost", {"semantic_type": "numeric_currency"}),
            ("sales", "product_id", {"semantic_type": "identifier"}),
        ]

        matched = template.match_columns(all_columns)
        assert len(matched) == 2

    def test_no_match_when_missing_signal(self):
        template = TEMPLATES[0]  # Buyer Credit Score (needs payment, order, total)
        all_columns = [
            ("orders", "payment_delay", {"semantic_type": "numeric_metric"}),
            # Missing order and total columns
        ]

        matched = template.match_columns(all_columns)
        assert matched == []  # Should not match

    def test_cross_table_match(self):
        template = TEMPLATES[3]  # Product Profitability
        all_columns = [
            ("sales", "revenue", {"semantic_type": "numeric_currency"}),
            ("expenses", "cost_per_unit", {"semantic_type": "numeric_currency"}),
        ]

        matched = template.match_columns(all_columns)
        assert len(matched) == 2
        tables = {t for t, c in matched}
        assert len(tables) == 2  # Cross-table

    def test_buyer_credit_score_match(self):
        template = TEMPLATES[0]  # Buyer Credit Score
        all_columns = [
            ("payments", "payment_delay", {"semantic_type": "numeric_metric"}),
            ("orders", "order_count", {"semantic_type": "numeric_metric"}),
            ("orders", "total_spend", {"semantic_type": "numeric_currency"}),
        ]

        matched = template.match_columns(all_columns)
        assert len(matched) == 3


class TestCheckEntity:
    """Test entity keyword matching."""

    def test_customer_entity(self):
        all_columns = [
            ("orders", "customer_id", {"semantic_type": "identifier"}),
            ("orders", "amount", {"semantic_type": "numeric_currency"}),
        ]
        assert _check_entity(["customer", "buyer"], all_columns)

    def test_no_entity_match(self):
        all_columns = [
            ("data", "value", {"semantic_type": "numeric_metric"}),
        ]
        assert not _check_entity(["customer", "buyer"], all_columns)

    def test_entity_in_table_name(self):
        all_columns = [
            ("supplier_data", "id", {"semantic_type": "identifier"}),
        ]
        assert _check_entity(["supplier", "vendor"], all_columns)

    def test_empty_columns(self):
        assert not _check_entity(["customer"], [])

    def test_empty_keywords(self):
        all_columns = [("orders", "customer_id", {})]
        assert not _check_entity([], all_columns)

    def test_case_insensitive(self):
        all_columns = [("orders", "CUSTOMER_ID", {})]
        assert _check_entity(["customer"], all_columns)

    def test_partial_keyword_in_column(self):
        all_columns = [("data", "employee_name", {})]
        assert _check_entity(["employee"], all_columns)


class TestBuildCompositeQuery:
    """Test SQL query generation for composite metrics."""

    def _make_profile(self, table_name, columns_dict):
        p = TableProfile()
        p.table_name = table_name
        p.row_count = 100
        p.column_classification = {"columns": columns_dict}
        return p

    def test_single_table_query(self):
        template = TEMPLATES[3]  # Product Profitability
        matched = [("sales", "revenue"), ("sales", "cost")]
        profiles = [self._make_profile("sales", {})]

        query = _build_composite_query(template, matched, profiles)
        assert '"sales"' in query
        assert '"revenue"' in query
        assert '"cost"' in query
        assert "LIMIT 100" in query

    def test_cross_table_query(self):
        matched = [("sales", "revenue"), ("expenses", "cost")]
        profiles = [
            self._make_profile("sales", {}),
            self._make_profile("expenses", {}),
        ]

        query = _build_composite_query(TEMPLATES[3], matched, profiles)
        assert "SELECT" in query
        assert '"revenue"' in query
        assert '"cost"' in query
        assert "LIMIT 100" in query

    def test_three_columns_single_table(self):
        matched = [("payments", "payment_delay"), ("payments", "order_count"), ("payments", "total_spend")]
        profiles = [self._make_profile("payments", {})]

        query = _build_composite_query(TEMPLATES[0], matched, profiles)
        assert '"payments"' in query
        assert '"payment_delay"' in query
        assert '"order_count"' in query
        assert '"total_spend"' in query


class TestDiscoverComposites:
    """Test full composite discovery pipeline."""

    def _make_profile(self, table_name, columns_dict, domain="general"):
        p = TableProfile()
        p.table_name = table_name
        p.row_count = 100
        p.domain_hint = domain
        p.column_classification = {
            "columns": columns_dict,
            "domain_hint": domain,
        }
        return p

    def test_discover_with_matching_template(self):
        profile = self._make_profile("sales", {
            "revenue": {"semantic_type": "numeric_currency"},
            "cost": {"semantic_type": "numeric_currency"},
            "product_id": {"semantic_type": "identifier"},
        }, "sales")

        insights = discover_composites([profile], [])

        # Should find Product Profitability
        assert any(i.composite_template == "Product Profitability" for i in insights)

    def test_discover_no_match(self):
        profile = self._make_profile("data", {
            "name": {"semantic_type": "text"},
            "id": {"semantic_type": "identifier"},
        })

        insights = discover_composites([profile], [])
        assert len(insights) == 0

    def test_discover_supplier_risk(self):
        profile = self._make_profile("procurement", {
            "supplier": {"semantic_type": "categorical"},
            "rate": {"semantic_type": "numeric_currency"},
            "quality_grade": {"semantic_type": "numeric_metric"},
        }, "procurement")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Supplier Risk Score" for i in insights)

    def test_discover_furnace_efficiency(self):
        """Steel plant SCADA data should discover Furnace Efficiency Score."""
        profile = self._make_profile("scada_readings", {
            "kva": {"semantic_type": "numeric_metric"},
            "output_tons": {"semantic_type": "numeric_metric"},
            "temperature": {"semantic_type": "numeric_metric"},
            "heat_no": {"semantic_type": "identifier"},
            "furnace_id": {"semantic_type": "identifier"},
        }, "manufacturing")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Furnace Efficiency Score" for i in insights)

    def test_discover_power_per_ton(self):
        """Power + tonnage columns should discover Power Consumption per Ton."""
        profile = self._make_profile("production", {
            "power_kwh": {"semantic_type": "numeric_metric"},
            "tonnage": {"semantic_type": "numeric_metric"},
            "date": {"semantic_type": "temporal"},
        }, "manufacturing")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Power Consumption per Ton" for i in insights)

    def test_discover_supplier_quality(self):
        """Procurement data with Fe and rejection columns should discover Supplier Quality."""
        profile = self._make_profile("supplier_quality", {
            "supplier": {"semantic_type": "categorical"},
            "fe_content": {"semantic_type": "numeric_metric"},
            "rejection_rate": {"semantic_type": "numeric_percentage"},
        }, "quality")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Supplier Quality Score" for i in insights)

    def test_discover_material_yield(self):
        """Input/output weight columns should discover Material Yield Tracker."""
        profile = self._make_profile("furnace_log", {
            "heat_no": {"semantic_type": "identifier"},
            "input_weight": {"semantic_type": "numeric_metric"},
            "output_weight": {"semantic_type": "numeric_metric"},
        }, "manufacturing")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Material Yield Tracker" for i in insights)

    def test_cross_table_composite_higher_impact(self):
        prof_a = self._make_profile("sales", {
            "revenue": {"semantic_type": "numeric_currency"},
        }, "sales")
        prof_b = self._make_profile("expenses", {
            "cost": {"semantic_type": "numeric_currency"},
            "product_id": {"semantic_type": "identifier"},
        }, "finance")

        rel = DiscoveredRelationship()
        rel.table_a = "sales"
        rel.column_a = "product_id"
        rel.table_b = "expenses"
        rel.column_b = "product_id"
        rel.relationship_type = "join_key"
        rel.confidence = 0.9

        insights = discover_composites([prof_a, prof_b], [rel])

        profitability = [i for i in insights if i.composite_template == "Product Profitability"]
        if profitability:
            evidence = profitability[0].evidence or {}
            assert evidence.get("is_cross_table") is True

    def test_discover_heat_cycle_time(self):
        """Heat data with tap and charge columns should discover Heat Cycle Time."""
        profile = self._make_profile("heat_log", {
            "heat_no": {"semantic_type": "identifier"},
            "tap_time": {"semantic_type": "temporal"},
            "charge_time": {"semantic_type": "temporal"},
            "furnace_id": {"semantic_type": "identifier"},
        }, "manufacturing")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Heat Cycle Time" for i in insights)

    def test_discover_alloy_efficiency(self):
        """Alloy + grade columns should discover Alloy Addition Efficiency."""
        profile = self._make_profile("heat_chemistry", {
            "heat_no": {"semantic_type": "identifier"},
            "alloy_addition_kg": {"semantic_type": "numeric_metric"},
            "grade_achieved": {"semantic_type": "categorical"},
        }, "manufacturing")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Alloy Addition Efficiency" for i in insights)

    def test_discover_slag_rate(self):
        """Slag + metal columns should discover Slag Rate Monitor."""
        profile = self._make_profile("heat_data", {
            "heat_no": {"semantic_type": "identifier"},
            "slag_weight": {"semantic_type": "numeric_metric"},
            "metal_weight": {"semantic_type": "numeric_metric"},
        }, "manufacturing")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Slag Rate Monitor" for i in insights)

    def test_discover_electrode_consumption(self):
        """Electrode + tonnage columns should discover Electrode Consumption Tracker."""
        profile = self._make_profile("furnace_data", {
            "heat_no": {"semantic_type": "identifier"},
            "electrode_kg": {"semantic_type": "numeric_metric"},
            "tonnage": {"semantic_type": "numeric_metric"},
        }, "manufacturing")

        insights = discover_composites([profile], [])
        assert any(i.composite_template == "Electrode Consumption Tracker" for i in insights)
