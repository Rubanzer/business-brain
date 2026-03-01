"""Tests for analysis/track1/enumerator.py — tiered enumeration + dedup."""

from __future__ import annotations

import pytest

from business_brain.analysis.track1.enumerator import (
    AnalysisCandidate,
    EnumerationBudget,
    _enumerate_tier0,
    _enumerate_tier1,
    _enumerate_tier2,
    _enumerate_tier3,
    _enumerate_tier4,
    _is_valid_dimension,
    _is_valid_measure,
    _make_dedup_key,
    enumerate_operations,
)
from business_brain.analysis.track1.fingerprinter import ColumnFingerprint, TableFingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fp(
    table: str = "sales",
    measures: list[str] | None = None,
    dimensions: list[str] | None = None,
    row_count: int = 1000,
) -> TableFingerprint:
    """Build a TableFingerprint for testing."""
    if measures is None:
        measures = ["revenue", "quantity", "cost"]
    if dimensions is None:
        dimensions = ["region", "category"]

    columns = {}
    for m in measures:
        columns[m] = ColumnFingerprint(
            name=m, semantic_type="numeric_metric", role="MEASURE",
            cardinality=100, null_rate=0.02,
        )
    for d in dimensions:
        columns[d] = ColumnFingerprint(
            name=d, semantic_type="categorical", role="DIMENSION",
            cardinality=10, null_rate=0.01,
        )

    return TableFingerprint(
        table_name=table, row_count=row_count, data_hash="abc123",
        domain_hint="sales", time_index=None,
        measures=measures, dimensions=dimensions, columns=columns,
    )


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


class TestQualityFilters:
    def test_valid_measure(self):
        col = ColumnFingerprint(name="x", semantic_type="numeric_metric", role="MEASURE",
                                cardinality=100, null_rate=0.1)
        assert _is_valid_measure(col, 100) is True

    def test_measure_too_many_nulls(self):
        col = ColumnFingerprint(name="x", semantic_type="numeric_metric", role="MEASURE",
                                cardinality=100, null_rate=0.6)
        assert _is_valid_measure(col, 100) is False

    def test_measure_too_few_rows(self):
        col = ColumnFingerprint(name="x", semantic_type="numeric_metric", role="MEASURE",
                                cardinality=5, null_rate=0.0)
        assert _is_valid_measure(col, 5) is False

    def test_valid_dimension(self):
        col = ColumnFingerprint(name="x", semantic_type="categorical", role="DIMENSION",
                                cardinality=10, null_rate=0.01)
        assert _is_valid_dimension(col) is True

    def test_dimension_cardinality_too_low(self):
        col = ColumnFingerprint(name="x", semantic_type="categorical", role="DIMENSION",
                                cardinality=1, null_rate=0.01)
        assert _is_valid_dimension(col) is False

    def test_dimension_cardinality_too_high(self):
        col = ColumnFingerprint(name="x", semantic_type="categorical", role="DIMENSION",
                                cardinality=500, null_rate=0.01)
        assert _is_valid_dimension(col) is False


# ---------------------------------------------------------------------------
# Dedup key
# ---------------------------------------------------------------------------


class TestDedupKey:
    def test_canonical_key_deterministic(self):
        k1 = _make_dedup_key("RANK", "sales", ["revenue"], ["region"], [])
        k2 = _make_dedup_key("RANK", "sales", ["revenue"], ["region"], [])
        assert k1 == k2

    def test_symmetric_correlate(self):
        """CORRELATE(A,B) == CORRELATE(B,A) — Gap #6."""
        k1 = _make_dedup_key("CORRELATE", "sales", ["revenue", "cost"], [], [])
        k2 = _make_dedup_key("CORRELATE", "sales", ["cost", "revenue"], [], [])
        assert k1 == k2

    def test_different_operations_different_keys(self):
        k1 = _make_dedup_key("DESCRIBE", "sales", ["revenue"], [], [])
        k2 = _make_dedup_key("DETECT_ANOMALY", "sales", ["revenue"], [], [])
        assert k1 != k2

    def test_different_segmenters_different_keys(self):
        k1 = _make_dedup_key("RANK", "sales", ["revenue"], ["region"], [])
        k2 = _make_dedup_key("RANK", "sales", ["revenue"], ["category"], [])
        assert k1 != k2

    def test_segmenter_order_doesnt_matter(self):
        k1 = _make_dedup_key("RANK", "sales", ["revenue"], ["region", "category"], [])
        k2 = _make_dedup_key("RANK", "sales", ["revenue"], ["category", "region"], [])
        assert k1 == k2

    def test_join_spec_included(self):
        k1 = _make_dedup_key("RANK", "sales", ["revenue"], [], [], {"table": "dim", "local_col": "a", "remote_col": "b"})
        k2 = _make_dedup_key("RANK", "sales", ["revenue"], [], [])
        assert k1 != k2


# ---------------------------------------------------------------------------
# Tier 0 (single-column, exhaustive)
# ---------------------------------------------------------------------------


class TestTier0:
    def test_generates_describe_for_all_measures(self):
        fp = _make_fp(measures=["m1", "m2", "m3"])
        candidates = _enumerate_tier0(fp)
        describe_candidates = [c for c in candidates if c.operation == "DESCRIBE"]
        assert len(describe_candidates) == 3

    def test_generates_detect_anomaly(self):
        fp = _make_fp(measures=["m1"])
        candidates = _enumerate_tier0(fp)
        anomaly = [c for c in candidates if c.operation == "DETECT_ANOMALY"]
        assert len(anomaly) == 1

    def test_generates_describe_categorical_for_dimensions(self):
        fp = _make_fp(dimensions=["d1", "d2"])
        candidates = _enumerate_tier0(fp)
        cat = [c for c in candidates if c.operation == "DESCRIBE_CATEGORICAL"]
        assert len(cat) == 2

    def test_skips_high_null_measures(self):
        fp = _make_fp(measures=["m1"])
        fp.columns["m1"].null_rate = 0.8
        candidates = _enumerate_tier0(fp)
        assert len([c for c in candidates if c.operation == "DESCRIBE"]) == 0

    def test_all_tier_0(self):
        fp = _make_fp()
        candidates = _enumerate_tier0(fp)
        assert all(c.tier == 0 for c in candidates)

    def test_no_anomaly_if_few_rows(self):
        fp = _make_fp(measures=["m1"], row_count=15)
        candidates = _enumerate_tier0(fp)
        assert len([c for c in candidates if c.operation == "DETECT_ANOMALY"]) == 0


# ---------------------------------------------------------------------------
# Tier 1 (pairwise, exhaustive)
# ---------------------------------------------------------------------------


class TestTier1:
    def test_correlate_all_measure_pairs(self):
        fp = _make_fp(measures=["m1", "m2", "m3"])
        candidates = _enumerate_tier1(fp)
        correlate = [c for c in candidates if c.operation == "CORRELATE"]
        # C(3,2) = 3 pairs
        assert len(correlate) == 3

    def test_rank_measure_x_dimension(self):
        fp = _make_fp(measures=["m1", "m2"], dimensions=["d1", "d2"])
        candidates = _enumerate_tier1(fp)
        rank = [c for c in candidates if c.operation == "RANK"]
        # 2 measures × 2 dimensions = 4
        assert len(rank) == 4

    def test_rank_subsumes_compare(self):
        """RANK replaces COMPARE — Gap #6."""
        fp = _make_fp()
        candidates = _enumerate_tier1(fp)
        compare = [c for c in candidates if c.operation == "COMPARE"]
        assert len(compare) == 0

    def test_correlate_is_canonical(self):
        """CORRELATE(A,B) has sorted target — Gap #6."""
        fp = _make_fp(measures=["z_measure", "a_measure"])
        candidates = _enumerate_tier1(fp)
        correlate = [c for c in candidates if c.operation == "CORRELATE"]
        assert len(correlate) == 1
        assert correlate[0].target == ["a_measure", "z_measure"]

    def test_all_tier_1(self):
        fp = _make_fp()
        candidates = _enumerate_tier1(fp)
        assert all(c.tier == 1 for c in candidates)


# ---------------------------------------------------------------------------
# Exhaustive guarantee: Tier 0+1 have NO budget cap
# ---------------------------------------------------------------------------


class TestExhaustiveGuarantee:
    def test_all_combos_generated_wide_table(self):
        """A table with 10 measures and 5 dimensions must produce ALL combos."""
        measures = [f"m{i}" for i in range(10)]
        dimensions = [f"d{i}" for i in range(5)]
        fp = _make_fp(measures=measures, dimensions=dimensions, row_count=1000)
        fingerprints = {fp.table_name: fp}

        candidates = enumerate_operations(fingerprints, [])
        t0 = [c for c in candidates if c.tier == 0]
        t1 = [c for c in candidates if c.tier == 1]

        # Tier 0: DESCRIBE(10) + DETECT_ANOMALY(10) + DESCRIBE_CATEGORICAL(5) = 25
        assert len(t0) == 25

        # Tier 1: CORRELATE C(10,2)=45 + RANK 10×5=50 = 95
        assert len(t1) == 95

    def test_no_budget_cap_on_tier01(self):
        """Even with a tiny budget, Tier 0+1 are fully enumerated."""
        fp = _make_fp(measures=[f"m{i}" for i in range(8)], dimensions=[f"d{i}" for i in range(4)])
        fingerprints = {fp.table_name: fp}
        budget = EnumerationBudget(budgeted_tier_limits={2: 1, 3: 1, 4: 1})

        candidates = enumerate_operations(fingerprints, [], budget)
        t0 = [c for c in candidates if c.tier == 0]
        t1 = [c for c in candidates if c.tier == 1]

        # Tier 0+1 should be the same regardless of budget
        expected_t0 = 8 + 8 + 4  # DESCRIBE + DETECT_ANOMALY + DESCRIBE_CATEGORICAL
        expected_t1 = 28 + 32  # C(8,2) + 8×4
        assert len(t0) == expected_t0
        assert len(t1) == expected_t1


# ---------------------------------------------------------------------------
# Budget enforcement (Tiers 2-4)
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_tier2_respects_budget(self):
        fp = _make_fp(measures=[f"m{i}" for i in range(5)], dimensions=[f"d{i}" for i in range(5)])
        fingerprints = {fp.table_name: fp}
        budget = EnumerationBudget(budgeted_tier_limits={2: 3, 3: 0, 4: 0})

        candidates = enumerate_operations(fingerprints, [], budget)
        t2 = [c for c in candidates if c.tier == 2]
        assert len(t2) <= 3

    def test_tier3_respects_budget(self):
        fp_a = _make_fp(table="a", measures=["m1", "m2"], dimensions=["d1"])
        fp_b = _make_fp(table="b", measures=["m3"], dimensions=["d2"])
        relationships = [{
            "table_a": "a", "column_a": "key", "table_b": "b", "column_b": "key",
            "confidence": 0.9,
        }]
        budget = EnumerationBudget(budgeted_tier_limits={2: 0, 3: 2, 4: 0})

        candidates = enumerate_operations({"a": fp_a, "b": fp_b}, relationships, budget)
        t3 = [c for c in candidates if c.tier == 3]
        assert len(t3) <= 2


# ---------------------------------------------------------------------------
# Cross-table enumeration (Gap #2)
# ---------------------------------------------------------------------------


class TestCrossTable:
    def test_generates_cross_table_candidates(self):
        fp_a = _make_fp(table="orders", measures=["revenue"], dimensions=["region"])
        fp_b = _make_fp(table="products", measures=["price"], dimensions=["category"])
        relationships = [{
            "table_a": "orders", "column_a": "product_id",
            "table_b": "products", "column_b": "id",
            "confidence": 0.9,
        }]

        candidates = _enumerate_tier3({"orders": fp_a, "products": fp_b}, relationships)
        assert len(candidates) > 0
        assert all(c.join_spec is not None for c in candidates)
        assert all(c.tier == 3 for c in candidates)

    def test_skips_low_confidence_relationships(self):
        fp_a = _make_fp(table="a", measures=["m1"])
        fp_b = _make_fp(table="b", measures=["m2"])
        relationships = [{
            "table_a": "a", "column_a": "k", "table_b": "b", "column_b": "k",
            "confidence": 0.3,  # below 0.7 threshold
        }]

        candidates = _enumerate_tier3({"a": fp_a, "b": fp_b}, relationships)
        assert len(candidates) == 0


# ---------------------------------------------------------------------------
# Deduplication (Gap #6)
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_no_duplicates_in_output(self):
        fp = _make_fp()
        fingerprints = {fp.table_name: fp}
        candidates = enumerate_operations(fingerprints, [])
        keys = [c.dedup_key for c in candidates]
        assert len(keys) == len(set(keys))

    def test_dedup_across_tiers(self):
        """A Tier 2 candidate that duplicates a Tier 1 should be removed."""
        fp = _make_fp(measures=["m1"], dimensions=["d1", "d2"])
        fingerprints = {fp.table_name: fp}
        candidates = enumerate_operations(fingerprints, [])

        # All keys should be unique
        keys = set()
        for c in candidates:
            assert c.dedup_key not in keys, f"Duplicate key in Tier {c.tier}: {c.operation}"
            keys.add(c.dedup_key)


# ---------------------------------------------------------------------------
# EnumerationBudget defaults
# ---------------------------------------------------------------------------


class TestEnumerationBudget:
    def test_default_budgets(self):
        b = EnumerationBudget()
        assert 0 not in b.budgeted_tier_limits
        assert 1 not in b.budgeted_tier_limits
        assert b.budgeted_tier_limits[2] == 100
        assert b.budgeted_tier_limits[3] == 50
        assert b.budgeted_tier_limits[4] == 50
