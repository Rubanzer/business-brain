"""Tests for the format matcher module."""

from business_brain.ingestion.format_matcher import (
    _normalize_col,
    compute_fingerprint,
    fuzzy_match_columns,
)


class TestNormalizeCol:
    """Test column name normalization for fingerprinting."""

    def test_lowercase(self):
        assert _normalize_col("HEAT_NO") == "heatno"

    def test_strip_spaces(self):
        assert _normalize_col("  Heat No  ") == "heatno"

    def test_remove_special_chars(self):
        assert _normalize_col("Weight (MT)") == "weightmt"

    def test_remove_underscores(self):
        assert _normalize_col("production_date") == "productiondate"

    def test_remove_hyphens(self):
        assert _normalize_col("order-id") == "orderid"


class TestComputeFingerprint:
    """Test fingerprint hash computation."""

    def test_deterministic(self):
        cols = ["id", "name", "value"]
        assert compute_fingerprint(cols) == compute_fingerprint(cols)

    def test_order_independent(self):
        cols_a = ["id", "name", "value"]
        cols_b = ["value", "name", "id"]
        assert compute_fingerprint(cols_a) == compute_fingerprint(cols_b)

    def test_case_independent(self):
        cols_a = ["ID", "Name", "VALUE"]
        cols_b = ["id", "name", "value"]
        assert compute_fingerprint(cols_a) == compute_fingerprint(cols_b)

    def test_different_columns_different_hash(self):
        cols_a = ["id", "name", "value"]
        cols_b = ["id", "name", "price"]
        assert compute_fingerprint(cols_a) != compute_fingerprint(cols_b)

    def test_empty_columns(self):
        # Should not crash on empty
        fp = compute_fingerprint([])
        assert isinstance(fp, str)
        assert len(fp) == 16  # SHA256[:16]

    def test_similar_names_different_hash(self):
        """Slightly different column names produce different fingerprints."""
        cols_a = ["customer_id", "order_date", "total"]
        cols_b = ["customer_id", "order_date", "totals"]
        assert compute_fingerprint(cols_a) != compute_fingerprint(cols_b)

    def test_special_chars_normalized(self):
        """Special chars stripped during normalization."""
        cols_a = ["Heat (No.)", "Grade", "WT (MT)"]
        cols_b = ["Heat No", "Grade", "WT MT"]
        assert compute_fingerprint(cols_a) == compute_fingerprint(cols_b)


class TestFuzzyMatchRecurring:
    """Test fuzzy matching for recurring file detection."""

    def test_identical_schemas(self):
        cols_a = ["heat_no", "grade", "weight_tons"]
        cols_b = ["heat_no", "grade", "weight_tons"]
        mapping = fuzzy_match_columns(cols_a, cols_b)
        assert len(mapping) == 3

    def test_case_variation(self):
        cols_a = ["HEAT_NO", "GRADE", "WT_TONS"]
        cols_b = ["heat_no", "grade", "wt_tons"]
        mapping = fuzzy_match_columns(cols_a, cols_b)
        assert len(mapping) == 3

    def test_column_name_variation(self):
        """Headers with different naming conventions but same meaning."""
        cols_a = ["Heat Number", "Steel Grade", "Weight"]
        cols_b = ["heat_no", "grade", "weight"]
        mapping = fuzzy_match_columns(cols_a, cols_b, threshold=0.5)
        # At minimum, "weight" should match "Weight"
        assert len(mapping) >= 1

    def test_completely_different(self):
        cols_a = ["alpha", "beta", "gamma"]
        cols_b = ["x", "y", "z"]
        mapping = fuzzy_match_columns(cols_a, cols_b)
        assert len(mapping) == 0

    def test_match_percentage(self):
        """If >80% columns match, consider it a recurring format."""
        cols_a = ["id", "name", "value", "date", "status"]
        cols_b = ["id", "name", "value", "date", "category"]
        mapping = fuzzy_match_columns(cols_a, cols_b)
        match_pct = len(mapping) / len(cols_a)
        assert match_pct >= 0.8  # 4/5 = 80%

    def test_threshold_controls_fuzziness(self):
        """Higher threshold = stricter matching."""
        cols_a = ["customer_name"]
        cols_b = ["cust_name"]
        strict = fuzzy_match_columns(cols_a, cols_b, threshold=0.9)
        lenient = fuzzy_match_columns(cols_a, cols_b, threshold=0.5)
        assert len(lenient) >= len(strict)
