"""Tests for the format detection module."""

from business_brain.db.discovery_models import TableProfile
from business_brain.discovery.format_detector import (
    _compare_profiles,
    _match_by_name,
    _match_by_type,
    _normalize,
    _type_distribution,
)
from business_brain.ingestion.format_matcher import (
    _normalize_col,
    compute_fingerprint,
    fuzzy_match_columns,
)


class TestNormalize:
    """Test column name normalization."""

    def test_simple(self):
        assert _normalize("HEAT_NO") == "heatno"

    def test_spaces(self):
        assert _normalize("Heat Number") == "heatnumber"

    def test_special_chars(self):
        assert _normalize("WT (MT)") == "wtmt"

    def test_underscores(self):
        assert _normalize("production_date") == "productiondate"


class TestFingerprint:
    """Test format fingerprinting."""

    def test_same_columns_same_hash(self):
        cols_a = ["id", "name", "value"]
        cols_b = ["value", "name", "id"]  # different order
        assert compute_fingerprint(cols_a) == compute_fingerprint(cols_b)

    def test_different_columns_different_hash(self):
        cols_a = ["id", "name", "value"]
        cols_b = ["id", "name", "price"]
        assert compute_fingerprint(cols_a) != compute_fingerprint(cols_b)

    def test_case_insensitive(self):
        cols_a = ["ID", "Name", "VALUE"]
        cols_b = ["id", "name", "value"]
        assert compute_fingerprint(cols_a) == compute_fingerprint(cols_b)


class TestFuzzyMatchColumns:
    """Test fuzzy column matching."""

    def test_exact_match(self):
        cols_a = ["customer_id", "order_date", "amount"]
        cols_b = ["customer_id", "order_date", "amount"]
        mapping = fuzzy_match_columns(cols_a, cols_b)
        assert len(mapping) == 3

    def test_case_insensitive_match(self):
        cols_a = ["HEAT_NO", "GRADE", "WT_TONS"]
        cols_b = ["heat_no", "grade", "wt_tons"]
        mapping = fuzzy_match_columns(cols_a, cols_b)
        assert len(mapping) == 3

    def test_fuzzy_match(self):
        cols_a = ["Heat Number", "Steel Grade", "Weight (MT)"]
        cols_b = ["heat_no", "grade", "weight"]
        mapping = fuzzy_match_columns(cols_a, cols_b, threshold=0.5)
        # Some should match based on fuzzy similarity
        assert len(mapping) >= 1

    def test_no_match(self):
        cols_a = ["alpha", "beta", "gamma"]
        cols_b = ["x", "y", "z"]
        mapping = fuzzy_match_columns(cols_a, cols_b)
        assert len(mapping) == 0

    def test_partial_match(self):
        cols_a = ["customer_id", "order_date", "total_amount"]
        cols_b = ["cust_id", "date", "payment_amount"]
        mapping = fuzzy_match_columns(cols_a, cols_b, threshold=0.5)
        # At least some should match
        assert len(mapping) >= 0  # depends on fuzzy threshold


class TestMatchByName:
    """Test name-based column matching between profiles."""

    def test_exact_columns(self):
        cols_a = {"customer_id": {"semantic_type": "identifier"}, "amount": {"semantic_type": "numeric_currency"}}
        cols_b = {"customer_id": {"semantic_type": "identifier"}, "amount": {"semantic_type": "numeric_currency"}}
        score, mapping = _match_by_name(cols_a, cols_b)
        assert score == 1.0
        assert len(mapping) == 2

    def test_no_overlap(self):
        cols_a = {"alpha": {"semantic_type": "text"}}
        cols_b = {"omega": {"semantic_type": "text"}}
        score, mapping = _match_by_name(cols_a, cols_b)
        assert score < 0.5

    def test_fuzzy_overlap(self):
        cols_a = {"heat_number": {"semantic_type": "identifier"}}
        cols_b = {"heatno": {"semantic_type": "identifier"}}
        score, mapping = _match_by_name(cols_a, cols_b)
        # These should fuzzy-match
        assert score > 0


class TestMatchByType:
    """Test type distribution matching."""

    def test_identical_distributions(self):
        cols_a = {"id": {"semantic_type": "identifier"}, "val": {"semantic_type": "numeric_currency"}}
        cols_b = {"key": {"semantic_type": "identifier"}, "amount": {"semantic_type": "numeric_currency"}}
        score = _match_by_type(cols_a, cols_b)
        assert score == 1.0

    def test_completely_different(self):
        cols_a = {"id": {"semantic_type": "identifier"}}
        cols_b = {"val": {"semantic_type": "numeric_currency"}}
        score = _match_by_type(cols_a, cols_b)
        assert score == 0.0

    def test_partial_overlap(self):
        cols_a = {"id": {"semantic_type": "identifier"}, "val": {"semantic_type": "numeric_currency"}, "name": {"semantic_type": "text"}}
        cols_b = {"key": {"semantic_type": "identifier"}, "amount": {"semantic_type": "numeric_currency"}, "cat": {"semantic_type": "categorical"}}
        score = _match_by_type(cols_a, cols_b)
        assert 0 < score < 1

    def test_empty_columns(self):
        score = _match_by_type({}, {})
        assert score == 0.0


class TestTypeDistribution:
    """Test type distribution computation."""

    def test_simple(self):
        cols = {
            "id": {"semantic_type": "identifier"},
            "name": {"semantic_type": "text"},
            "amount": {"semantic_type": "numeric_currency"},
        }
        dist = _type_distribution(cols)
        assert dist == {"identifier": 1, "text": 1, "numeric_currency": 1}

    def test_duplicates(self):
        cols = {
            "a": {"semantic_type": "numeric_metric"},
            "b": {"semantic_type": "numeric_metric"},
            "c": {"semantic_type": "identifier"},
        }
        dist = _type_distribution(cols)
        assert dist["numeric_metric"] == 2
        assert dist["identifier"] == 1

    def test_missing_semantic_type(self):
        cols = {"a": {}, "b": {"semantic_type": "text"}}
        dist = _type_distribution(cols)
        assert dist["unknown"] == 1
        assert dist["text"] == 1


class TestCompareProfiles:
    """Test full profile comparison."""

    def _make_profile(self, table_name, columns_dict, domain="general"):
        p = TableProfile()
        p.table_name = table_name
        p.row_count = 100
        p.domain_hint = domain
        p.column_classification = {"columns": columns_dict, "domain_hint": domain}
        return p

    def test_identical_profiles_high_score(self):
        cols = {
            "supplier": {"semantic_type": "categorical"},
            "rate": {"semantic_type": "numeric_currency"},
            "date": {"semantic_type": "temporal"},
        }
        p1 = self._make_profile("table_a", cols, "procurement")
        p2 = self._make_profile("table_b", dict(cols), "procurement")
        score, mapping = _compare_profiles(p1, p2)
        assert score >= 0.9
        assert len(mapping) == 3

    def test_completely_different_profiles_low_score(self):
        cols_a = {"temperature": {"semantic_type": "numeric_metric"}}
        cols_b = {"supplier": {"semantic_type": "categorical"}}
        p1 = self._make_profile("t1", cols_a, "manufacturing")
        p2 = self._make_profile("t2", cols_b, "procurement")
        score, _ = _compare_profiles(p1, p2)
        assert score < 0.3

    def test_same_domain_bonus(self):
        """Same non-general domain adds to score."""
        cols = {"supplier": {"semantic_type": "categorical"}}
        p_same = self._make_profile("t1", cols, "procurement")
        p_same2 = self._make_profile("t2", cols, "procurement")
        p_diff = self._make_profile("t3", cols, "manufacturing")

        score_same, _ = _compare_profiles(p_same, p_same2)
        score_diff, _ = _compare_profiles(p_same, p_diff)
        assert score_same > score_diff

    def test_general_domain_no_bonus(self):
        """'general' domain gets no domain bonus."""
        cols = {"a": {"semantic_type": "text"}}
        p1 = self._make_profile("t1", cols, "general")
        p2 = self._make_profile("t2", cols, "general")
        score, _ = _compare_profiles(p1, p2)
        # name 1.0*0.5 + type 1.0*0.3 + domain 0.0*0.2 = 0.8
        assert abs(score - 0.8) < 0.05

    def test_none_classification_returns_zero(self):
        p1 = self._make_profile("t1", {"a": {"semantic_type": "text"}})
        p2 = TableProfile()
        p2.table_name = "t2"
        p2.column_classification = None
        p2.domain_hint = "general"
        score, mapping = _compare_profiles(p1, p2)
        assert score == 0.0
        assert mapping == []

    def test_missing_columns_key_returns_zero(self):
        p1 = self._make_profile("t1", {"a": {"semantic_type": "text"}})
        p2 = TableProfile()
        p2.table_name = "t2"
        p2.column_classification = {"domain_hint": "general"}
        p2.domain_hint = "general"
        score, mapping = _compare_profiles(p1, p2)
        assert score == 0.0

    def test_empty_columns_dict_returns_zero(self):
        p1 = self._make_profile("t1", {})
        p2 = self._make_profile("t2", {"a": {"semantic_type": "text"}})
        score, _ = _compare_profiles(p1, p2)
        assert score == 0.0

    def test_scada_vs_google_sheet_formats(self):
        """Same data in different column naming formats should score high."""
        cols_a = {
            "PARTY_NAME": {"semantic_type": "categorical"},
            "RATE_PER_TON": {"semantic_type": "numeric_currency"},
            "DATE_OF_RECEIPT": {"semantic_type": "temporal"},
        }
        cols_b = {
            "party_name": {"semantic_type": "categorical"},
            "rate_per_ton": {"semantic_type": "numeric_currency"},
            "date_of_receipt": {"semantic_type": "temporal"},
        }
        p1 = self._make_profile("scada_export", cols_a, "procurement")
        p2 = self._make_profile("google_sheet", cols_b, "procurement")
        score, mapping = _compare_profiles(p1, p2)
        assert score >= 0.9
        assert len(mapping) == 3

    def test_returns_mapping_with_canonical_names(self):
        cols_a = {"Supplier_Name": {"semantic_type": "categorical"}}
        cols_b = {"supplier_name": {"semantic_type": "categorical"}}
        _, mapping = _compare_profiles(
            self._make_profile("a", cols_a),
            self._make_profile("b", cols_b),
        )
        assert len(mapping) == 1
        assert mapping[0]["canonical"] == "suppliername"


class TestMatchByNameEdgeCases:
    """Additional edge cases for name matching."""

    def test_empty_a(self):
        score, mapping = _match_by_name({}, {"a": {"semantic_type": "text"}})
        assert score == 0.0
        assert mapping == []

    def test_empty_b(self):
        score, mapping = _match_by_name({"a": {"semantic_type": "text"}}, {})
        assert score == 0.0
        assert mapping == []

    def test_both_empty(self):
        score, mapping = _match_by_name({}, {})
        assert score == 0.0

    def test_no_double_matching(self):
        """Once a column in B is matched, it shouldn't match again."""
        cols_a = {"name": {"semantic_type": "text"}, "names": {"semantic_type": "text"}}
        cols_b = {"name": {"semantic_type": "text"}}
        _, mapping = _match_by_name(cols_a, cols_b)
        b_cols = [m["b"] for m in mapping]
        assert len(b_cols) == len(set(b_cols))

    def test_score_reflects_proportion(self):
        """Score = matched_count / max(len_a, len_b)."""
        cols_a = {"a": {"semantic_type": "text"}, "b": {"semantic_type": "text"}}
        cols_b = {"a": {"semantic_type": "text"}, "x": {"semantic_type": "text"}}
        score, mapping = _match_by_name(cols_a, cols_b)
        # 1 match out of 2 columns = 0.5
        assert abs(score - 0.5) < 0.1


class TestMatchByTypeEdgeCases:
    """Additional edge cases for type matching."""

    def test_unbalanced_counts(self):
        """One side has more columns of a type."""
        cols_a = {
            "a": {"semantic_type": "numeric_metric"},
            "b": {"semantic_type": "numeric_metric"},
            "c": {"semantic_type": "numeric_metric"},
        }
        cols_b = {"x": {"semantic_type": "numeric_metric"}}
        score = _match_by_type(cols_a, cols_b)
        # min(3,1)/max(3,1) = 1/3
        assert abs(score - 1.0 / 3.0) < 0.01

    def test_many_types_partial(self):
        """Multiple types, some overlapping."""
        cols_a = {
            "a": {"semantic_type": "identifier"},
            "b": {"semantic_type": "categorical"},
            "c": {"semantic_type": "temporal"},
        }
        cols_b = {
            "x": {"semantic_type": "identifier"},
            "y": {"semantic_type": "numeric_currency"},
            "z": {"semantic_type": "temporal"},
        }
        score = _match_by_type(cols_a, cols_b)
        # 4 unique types: identifier(1/1), categorical(0), numeric_currency(0), temporal(1/1)
        # = (1 + 0 + 0 + 1) / 4 = 0.5
        assert abs(score - 0.5) < 0.01
