"""Tests for the relationship finder module."""

from business_brain.discovery.relationship_finder import _names_match, _semantic_match


class TestNameMatching:
    """Test column name matching heuristics."""

    def test_exact_match(self):
        assert _names_match("customer_id", "customer_id", "orders", "payments")

    def test_table_prefix_match(self):
        # customer_id <-> id where table is "customers"
        assert _names_match("customer_id", "id", "orders", "customers")

    def test_reverse_table_prefix_match(self):
        assert _names_match("id", "order_id", "orders", "shipments")

    def test_no_match_different_names(self):
        assert not _names_match("name", "price", "products", "orders")

    def test_no_match_similar_but_different(self):
        assert not _names_match("customer_name", "customer_id", "orders", "payments")

    def test_fk_prefix_stripped(self):
        assert _names_match("fk_customer_id", "fk_customer_id", "orders", "payments")


class TestSemanticMatch:
    """Test semantic matching between columns."""

    def test_both_identifiers_similar_cardinality(self):
        info_a = {"semantic_type": "identifier", "cardinality": 100}
        info_b = {"semantic_type": "identifier", "cardinality": 80}
        assert _semantic_match(info_a, info_b)

    def test_both_identifiers_different_cardinality(self):
        info_a = {"semantic_type": "identifier", "cardinality": 100}
        info_b = {"semantic_type": "identifier", "cardinality": 10}
        assert not _semantic_match(info_a, info_b)

    def test_different_types_no_match(self):
        info_a = {"semantic_type": "identifier", "cardinality": 100}
        info_b = {"semantic_type": "categorical", "cardinality": 100}
        assert not _semantic_match(info_a, info_b)

    def test_both_categorical_no_match(self):
        info_a = {"semantic_type": "categorical", "cardinality": 5}
        info_b = {"semantic_type": "categorical", "cardinality": 5}
        assert not _semantic_match(info_a, info_b)

    def test_zero_cardinality(self):
        info_a = {"semantic_type": "identifier", "cardinality": 0}
        info_b = {"semantic_type": "identifier", "cardinality": 0}
        assert not _semantic_match(info_a, info_b)
