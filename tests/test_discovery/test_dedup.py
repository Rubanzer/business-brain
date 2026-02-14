"""Tests for the duplicate insight detection module."""

from business_brain.db.discovery_models import Insight
from business_brain.discovery.dedup import compute_insight_key, deduplicate_insights


def _make_insight(
    insight_type="anomaly",
    source_tables=None,
    source_columns=None,
    composite_template=None,
):
    ins = Insight()
    ins.id = "test"
    ins.insight_type = insight_type
    ins.severity = "info"
    ins.impact_score = 50
    ins.title = "Test"
    ins.description = "Test"
    ins.source_tables = source_tables or ["sales"]
    ins.source_columns = source_columns or ["revenue"]
    ins.composite_template = composite_template
    ins.evidence = {}
    ins.suggested_actions = []
    return ins


class TestComputeInsightKey:
    def test_deterministic(self):
        a = _make_insight()
        b = _make_insight()
        assert compute_insight_key(a) == compute_insight_key(b)

    def test_different_type(self):
        a = _make_insight(insight_type="anomaly")
        b = _make_insight(insight_type="trend")
        assert compute_insight_key(a) != compute_insight_key(b)

    def test_different_tables(self):
        a = _make_insight(source_tables=["sales"])
        b = _make_insight(source_tables=["orders"])
        assert compute_insight_key(a) != compute_insight_key(b)

    def test_different_columns(self):
        a = _make_insight(source_columns=["revenue"])
        b = _make_insight(source_columns=["cost"])
        assert compute_insight_key(a) != compute_insight_key(b)

    def test_different_template(self):
        a = _make_insight(composite_template="Supplier Risk Score")
        b = _make_insight(composite_template="Product Profitability")
        assert compute_insight_key(a) != compute_insight_key(b)

    def test_table_order_irrelevant(self):
        a = _make_insight(source_tables=["sales", "orders"])
        b = _make_insight(source_tables=["orders", "sales"])
        assert compute_insight_key(a) == compute_insight_key(b)

    def test_column_order_irrelevant(self):
        a = _make_insight(source_columns=["revenue", "cost"])
        b = _make_insight(source_columns=["cost", "revenue"])
        assert compute_insight_key(a) == compute_insight_key(b)

    def test_none_fields_handled(self):
        ins = _make_insight(source_tables=None, source_columns=None, composite_template=None)
        key = compute_insight_key(ins)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_key_is_hex_string(self):
        key = compute_insight_key(_make_insight())
        assert all(c in "0123456789abcdef" for c in key)


class TestDeduplicateInsights:
    def test_no_existing_keys(self):
        insights = [_make_insight(), _make_insight(insight_type="trend")]
        result = deduplicate_insights(insights, set())
        assert len(result) == 2

    def test_all_duplicates(self):
        ins = _make_insight()
        key = compute_insight_key(ins)
        result = deduplicate_insights([ins], {key})
        assert len(result) == 0

    def test_partial_duplicates(self):
        a = _make_insight(insight_type="anomaly")
        b = _make_insight(insight_type="trend")
        key_a = compute_insight_key(a)
        result = deduplicate_insights([a, b], {key_a})
        assert len(result) == 1
        assert result[0].insight_type == "trend"

    def test_duplicates_within_batch(self):
        """Two identical insights in the same batch — only first kept."""
        a = _make_insight()
        b = _make_insight()  # Same key as a
        result = deduplicate_insights([a, b], set())
        assert len(result) == 1

    def test_empty_insights(self):
        result = deduplicate_insights([], set())
        assert result == []

    def test_empty_existing_keys(self):
        ins = _make_insight()
        result = deduplicate_insights([ins], set())
        assert len(result) == 1

    def test_preserves_order(self):
        a = _make_insight(insight_type="anomaly")
        b = _make_insight(insight_type="trend")
        c = _make_insight(insight_type="composite")
        result = deduplicate_insights([a, b, c], set())
        types = [i.insight_type for i in result]
        assert types == ["anomaly", "trend", "composite"]
