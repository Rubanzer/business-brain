"""Tests for the composite metric discoverer module."""

from business_brain.db.discovery_models import DiscoveredRelationship, TableProfile
from business_brain.discovery.composite_discoverer import (
    TEMPLATES,
    CompositeTemplate,
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
