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

    def test_exact_boundary_ratio_0_5(self):
        """Ratio of exactly 0.5 should NOT match (> 0.5 required)."""
        info_a = {"semantic_type": "identifier", "cardinality": 50}
        info_b = {"semantic_type": "identifier", "cardinality": 100}
        assert not _semantic_match(info_a, info_b)

    def test_just_above_0_5_ratio(self):
        info_a = {"semantic_type": "identifier", "cardinality": 51}
        info_b = {"semantic_type": "identifier", "cardinality": 100}
        assert _semantic_match(info_a, info_b)

    def test_equal_cardinality(self):
        info_a = {"semantic_type": "identifier", "cardinality": 100}
        info_b = {"semantic_type": "identifier", "cardinality": 100}
        assert _semantic_match(info_a, info_b)


class TestNameMatchingEdgeCases:
    """Additional edge cases for _names_match."""

    def test_pk_prefix_stripped(self):
        assert _names_match("pk_customer_id", "pk_customer_id", "t1", "t2")

    def test_mixed_pk_fk_not_match(self):
        """fk_customer_id vs pk_customer_id — after stripping both are equal."""
        assert _names_match("fk_customer_id", "pk_customer_id", "t1", "t2")

    def test_case_insensitive_exact(self):
        assert _names_match("Customer_ID", "customer_id", "t1", "t2")

    def test_table_singular_with_es_suffix(self):
        """re.sub(r's$', '', 'addresses') → 'addresse' — won't match 'address_id'."""
        assert not _names_match("address_id", "id", "orders", "addresses")

    def test_table_prefix_with_singular_s(self):
        """orders → order, so order_id <-> id in orders works."""
        assert _names_match("id", "order_id", "orders", "shipments")

    def test_no_match_unrelated(self):
        assert not _names_match("foo", "bar", "baz", "qux")

    def test_id_in_different_table(self):
        """customer_id <-> id in 'customers' table."""
        assert _names_match("customer_id", "id", "orders", "customers")

    def test_fk_vs_plain(self):
        """fk_customer_id matches customer_id after stripping prefix."""
        assert _names_match("fk_customer_id", "customer_id", "t1", "t2")

    def test_pk_vs_plain(self):
        assert _names_match("pk_order_id", "order_id", "t1", "t2")


class TestSemanticMatchEdgeCases:
    """Additional edge cases for _semantic_match."""

    def test_missing_semantic_type(self):
        info_a = {"cardinality": 50}
        info_b = {"semantic_type": "identifier", "cardinality": 50}
        assert not _semantic_match(info_a, info_b)

    def test_missing_cardinality(self):
        info_a = {"semantic_type": "identifier"}
        info_b = {"semantic_type": "identifier"}
        assert not _semantic_match(info_a, info_b)

    def test_empty_dicts(self):
        assert not _semantic_match({}, {})

    def test_one_cardinality_zero(self):
        info_a = {"semantic_type": "identifier", "cardinality": 0}
        info_b = {"semantic_type": "identifier", "cardinality": 100}
        assert not _semantic_match(info_a, info_b)

    def test_very_close_cardinality(self):
        info_a = {"semantic_type": "identifier", "cardinality": 99}
        info_b = {"semantic_type": "identifier", "cardinality": 100}
        assert _semantic_match(info_a, info_b)

    def test_text_types_no_match(self):
        info_a = {"semantic_type": "text", "cardinality": 100}
        info_b = {"semantic_type": "text", "cardinality": 100}
        assert not _semantic_match(info_a, info_b)
