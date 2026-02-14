"""Tests for the schema change tracker module."""

from business_brain.db.discovery_models import TableProfile
from business_brain.discovery.schema_tracker import (
    _compare_schemas,
    _get_columns,
    detect_schema_changes,
)


def _make_profile(table_name, columns_dict):
    p = TableProfile()
    p.table_name = table_name
    p.row_count = 100
    p.column_classification = {"columns": columns_dict}
    return p


class TestGetColumns:
    def test_normal(self):
        p = _make_profile("t", {"a": {"semantic_type": "text"}})
        assert _get_columns(p) == {"a": {"semantic_type": "text"}}

    def test_none_classification(self):
        p = TableProfile()
        p.table_name = "t"
        p.column_classification = None
        assert _get_columns(p) == {}

    def test_missing_columns_key(self):
        p = TableProfile()
        p.table_name = "t"
        p.column_classification = {"domain_hint": "general"}
        assert _get_columns(p) == {}


class TestCompareSchemas:
    def test_no_changes(self):
        prev = _make_profile("t", {"a": {"semantic_type": "text"}, "b": {"semantic_type": "numeric_metric"}})
        curr = _make_profile("t", {"a": {"semantic_type": "text"}, "b": {"semantic_type": "numeric_metric"}})
        insights = _compare_schemas(curr, prev)
        assert len(insights) == 0

    def test_column_added(self):
        prev = _make_profile("t", {"a": {"semantic_type": "text"}})
        curr = _make_profile("t", {"a": {"semantic_type": "text"}, "b": {"semantic_type": "numeric_metric"}})
        insights = _compare_schemas(curr, prev)
        assert len(insights) == 1
        assert insights[0].evidence["change_type"] == "columns_added"
        assert "b" in insights[0].evidence["columns"]

    def test_column_removed(self):
        prev = _make_profile("t", {"a": {"semantic_type": "text"}, "b": {"semantic_type": "numeric_metric"}})
        curr = _make_profile("t", {"a": {"semantic_type": "text"}})
        insights = _compare_schemas(curr, prev)
        assert len(insights) == 1
        assert insights[0].evidence["change_type"] == "columns_removed"
        assert "b" in insights[0].evidence["columns"]
        assert insights[0].severity == "warning"

    def test_type_changed(self):
        prev = _make_profile("t", {"a": {"semantic_type": "text"}})
        curr = _make_profile("t", {"a": {"semantic_type": "categorical"}})
        insights = _compare_schemas(curr, prev)
        assert len(insights) == 1
        assert insights[0].evidence["change_type"] == "type_changed"
        assert insights[0].evidence["changes"][0]["from"] == "text"
        assert insights[0].evidence["changes"][0]["to"] == "categorical"

    def test_multiple_changes(self):
        prev = _make_profile("t", {
            "a": {"semantic_type": "text"},
            "b": {"semantic_type": "numeric_metric"},
            "c": {"semantic_type": "temporal"},
        })
        curr = _make_profile("t", {
            "a": {"semantic_type": "categorical"},  # type change
            # b removed
            "d": {"semantic_type": "identifier"},  # added
        })
        insights = _compare_schemas(curr, prev)
        types = {i.evidence["change_type"] for i in insights}
        assert "columns_added" in types
        assert "columns_removed" in types
        assert "type_changed" in types

    def test_added_column_shows_semantic_type(self):
        prev = _make_profile("t", {})
        curr = _make_profile("t", {"revenue": {"semantic_type": "numeric_currency"}})
        insights = _compare_schemas(curr, prev)
        assert insights[0].evidence["types"]["revenue"] == "numeric_currency"

    def test_removed_column_impact_score(self):
        prev = _make_profile("t", {"a": {"semantic_type": "text"}})
        curr = _make_profile("t", {})
        insights = _compare_schemas(curr, prev)
        assert insights[0].impact_score == 50


class TestDetectSchemaChanges:
    def test_new_table_ignored(self):
        """A brand new table (no previous) doesn't generate change insights."""
        curr = [_make_profile("new_table", {"a": {"semantic_type": "text"}})]
        prev = []
        insights = detect_schema_changes(curr, prev)
        assert len(insights) == 0

    def test_multiple_tables(self):
        prev = [
            _make_profile("t1", {"a": {"semantic_type": "text"}}),
            _make_profile("t2", {"b": {"semantic_type": "numeric_metric"}}),
        ]
        curr = [
            _make_profile("t1", {"a": {"semantic_type": "text"}, "new_col": {"semantic_type": "identifier"}}),
            _make_profile("t2", {}),  # b removed
        ]
        insights = detect_schema_changes(curr, prev)
        assert len(insights) == 2
        tables = {i.source_tables[0] for i in insights}
        assert "t1" in tables
        assert "t2" in tables

    def test_empty_profiles(self):
        assert detect_schema_changes([], []) == []

    def test_no_changes_across_tables(self):
        profiles = [
            _make_profile("t1", {"a": {"semantic_type": "text"}}),
            _make_profile("t2", {"b": {"semantic_type": "numeric_metric"}}),
        ]
        insights = detect_schema_changes(profiles, profiles)
        assert len(insights) == 0

    def test_insight_fields(self):
        prev = [_make_profile("t", {"a": {"semantic_type": "text"}})]
        curr = [_make_profile("t", {"a": {"semantic_type": "text"}, "b": {"semantic_type": "identifier"}})]
        insights = detect_schema_changes(curr, prev)
        ins = insights[0]
        assert ins.insight_type == "schema_change"
        assert ins.id is not None
        assert ins.title is not None
        assert ins.description is not None
        assert ins.suggested_actions is not None
        assert len(ins.suggested_actions) > 0

    def test_same_name_no_type_change_if_none(self):
        """If semantic_type is None on either side, skip type change."""
        prev = _make_profile("t", {"a": {}})
        curr = _make_profile("t", {"a": {"semantic_type": "text"}})
        insights = _compare_schemas(curr, prev)
        # a has None → text, but we require both to be non-None
        assert len(insights) == 0
