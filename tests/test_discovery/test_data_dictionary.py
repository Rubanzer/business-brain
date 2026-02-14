"""Tests for data dictionary generator."""

from business_brain.discovery.data_dictionary import (
    ColumnEntry,
    DataDictionary,
    auto_describe_column,
    compare_dictionaries,
    detect_column_tags,
    format_dictionary_markdown,
    format_dictionary_text,
    generate_data_dictionary,
    infer_column_type,
)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

class TestInferColumnType:
    def test_integer(self):
        assert infer_column_type([1, 2, 3, 4, 5]) == "integer"

    def test_integer_strings(self):
        assert infer_column_type(["1", "2", "3"]) == "integer"

    def test_float(self):
        assert infer_column_type([1.5, 2.7, 3.3]) == "float"

    def test_float_strings(self):
        assert infer_column_type(["1.5", "2.7", "3.3"]) == "float"

    def test_text(self):
        vals = [f"unique_text_{i}" for i in range(50)]
        assert infer_column_type(vals) == "text"

    def test_date(self):
        assert infer_column_type(["2024-01-01", "2024-02-01", "2024-03-01"]) == "date"

    def test_boolean(self):
        assert infer_column_type([True, False, True]) == "boolean"

    def test_boolean_strings(self):
        assert infer_column_type(["yes", "no", "yes", "no"]) == "boolean"

    def test_categorical(self):
        vals = ["A", "B", "A", "C", "B", "A", "C", "B", "A", "B"]
        assert infer_column_type(vals) == "categorical"

    def test_all_none(self):
        assert infer_column_type([None, None]) == "text"

    def test_empty(self):
        assert infer_column_type([]) == "text"

    def test_mixed_with_nulls(self):
        vals = [1, 2, None, 3, None, 4]
        assert infer_column_type(vals) == "integer"


# ---------------------------------------------------------------------------
# Auto-describe
# ---------------------------------------------------------------------------

class TestAutoDescribe:
    def test_id_column(self):
        desc = auto_describe_column("customer_id", "integer", {})
        assert "Identifier" in desc or "key" in desc

    def test_date_column(self):
        desc = auto_describe_column("created_at", "date", {})
        assert "Timestamp" in desc

    def test_monetary(self):
        desc = auto_describe_column("total_cost", "float", {})
        assert "Monetary" in desc

    def test_count_column(self):
        desc = auto_describe_column("order_count", "integer", {})
        assert "Count" in desc or "quantity" in desc.lower()

    def test_percentage(self):
        desc = auto_describe_column("success_rate", "float", {})
        assert "Percentage" in desc or "ratio" in desc.lower()

    def test_name_field(self):
        desc = auto_describe_column("customer_name", "text", {})
        assert "Name" in desc or "name" in desc

    def test_email_field(self):
        desc = auto_describe_column("email", "text", {})
        assert "Email" in desc or "email" in desc.lower()

    def test_boolean_flag(self):
        desc = auto_describe_column("is_active", "boolean", {})
        assert "Boolean" in desc or "flag" in desc.lower()

    def test_generic_fallback(self):
        desc = auto_describe_column("xyz", "integer", {})
        assert "Integer" in desc


# ---------------------------------------------------------------------------
# Tag detection
# ---------------------------------------------------------------------------

class TestDetectTags:
    def test_primary_key(self):
        vals = [1, 2, 3, 4, 5]
        tags = detect_column_tags("customer_id", "integer", vals, 100.0)
        assert "primary_key" in tags

    def test_foreign_key(self):
        vals = [1, 1, 2, 2, 3]
        tags = detect_column_tags("order_id", "integer", vals, 60.0)
        assert "foreign_key" in tags

    def test_currency_tag(self):
        tags = detect_column_tags("total_cost", "float", [100, 200], 100.0)
        assert "currency" in tags

    def test_percentage_tag(self):
        tags = detect_column_tags("success_pct", "float", [50, 60], 100.0)
        assert "percentage" in tags

    def test_email_tag(self):
        vals = ["a@b.com", "c@d.org", "e@f.net"]
        tags = detect_column_tags("email", "text", vals, 100.0)
        assert "email" in tags

    def test_required_tag(self):
        vals = [1, 2, 3]
        tags = detect_column_tags("x", "integer", vals, 100.0)
        assert "required" in tags

    def test_optional_tag(self):
        vals = [1, None, None, 2, None, None, None, None, None, None]
        tags = detect_column_tags("x", "integer", vals, 50.0)
        assert "optional" in tags

    def test_no_tags_for_generic(self):
        vals = ["hello", "world"]
        tags = detect_column_tags("notes", "text", vals, 100.0)
        assert "primary_key" not in tags
        assert "foreign_key" not in tags
        assert "currency" not in tags


# ---------------------------------------------------------------------------
# Full dictionary generation
# ---------------------------------------------------------------------------

class TestGenerateDictionary:
    def test_basic(self):
        rows = [
            {"id": 1, "name": "Alice", "amount": 100.0},
            {"id": 2, "name": "Bob", "amount": 200.0},
            {"id": 3, "name": "Charlie", "amount": 150.0},
        ]
        dd = generate_data_dictionary(rows, "customers")
        assert dd is not None
        assert dd.table_name == "customers"
        assert dd.row_count == 3
        assert dd.column_count == 3

    def test_empty_returns_none(self):
        assert generate_data_dictionary([]) is None

    def test_single_row(self):
        rows = [{"id": 1, "value": 42}]
        dd = generate_data_dictionary(rows)
        assert dd is not None
        assert dd.row_count == 1

    def test_column_types_detected(self):
        rows = [
            {"id": 1, "name": "A", "amount": 100.5, "active": True, "date": "2024-01-01"},
            {"id": 2, "name": "B", "amount": 200.3, "active": False, "date": "2024-02-01"},
            {"id": 3, "name": "C", "amount": 150.0, "active": True, "date": "2024-03-01"},
        ]
        dd = generate_data_dictionary(rows)
        col_map = {c.name: c for c in dd.columns}
        assert col_map["id"].inferred_type == "integer"
        assert col_map["amount"].inferred_type == "float"

    def test_null_stats(self):
        rows = [
            {"x": 1, "y": None},
            {"x": 2, "y": None},
            {"x": 3, "y": "hello"},
        ]
        dd = generate_data_dictionary(rows)
        col_map = {c.name: c for c in dd.columns}
        assert col_map["y"].null_count == 2

    def test_sample_values(self):
        rows = [{"x": i} for i in range(20)]
        dd = generate_data_dictionary(rows)
        col_map = {c.name: c for c in dd.columns}
        assert len(col_map["x"].sample_values) == 5

    def test_relationship_hints(self):
        rows = [
            {"order_id": 1, "customer_id": 10, "amount": 100},
            {"order_id": 2, "customer_id": 20, "amount": 200},
        ]
        dd = generate_data_dictionary(rows, "orders")
        hints = dd.relationships_hint
        assert any("customer" in h for h in hints)

    def test_summary_contains_info(self):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        dd = generate_data_dictionary(rows, "test_table")
        assert "test_table" in dd.summary
        assert "2 rows" in dd.summary

    def test_all_nulls_column(self):
        rows = [{"x": 1, "y": None}, {"x": 2, "y": None}]
        dd = generate_data_dictionary(rows)
        col_map = {c.name: c for c in dd.columns}
        assert col_map["y"].null_pct == 100.0

    def test_generated_at_present(self):
        rows = [{"x": 1}]
        dd = generate_data_dictionary(rows)
        assert dd.generated_at is not None
        assert len(dd.generated_at) > 0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatMarkdown:
    def test_has_title(self):
        rows = [{"id": 1, "name": "A"}]
        dd = generate_data_dictionary(rows, "test")
        md = format_dictionary_markdown(dd)
        assert "# Data Dictionary: test" in md

    def test_has_column_table(self):
        rows = [{"id": 1, "value": 42}]
        dd = generate_data_dictionary(rows, "t")
        md = format_dictionary_markdown(dd)
        assert "| Column" in md
        assert "| id |" in md

    def test_has_details(self):
        rows = [{"x": 1}]
        dd = generate_data_dictionary(rows)
        md = format_dictionary_markdown(dd)
        assert "### x" in md


class TestFormatText:
    def test_has_header(self):
        rows = [{"id": 1}]
        dd = generate_data_dictionary(rows, "my_table")
        text = format_dictionary_text(dd)
        assert "DATA DICTIONARY: my_table" in text

    def test_has_column_info(self):
        rows = [{"x": 5, "y": "hello"}]
        dd = generate_data_dictionary(rows)
        text = format_dictionary_text(dd)
        assert "x (integer)" in text


# ---------------------------------------------------------------------------
# Compare dictionaries
# ---------------------------------------------------------------------------

class TestCompareDictionaries:
    def test_added_column(self):
        rows_a = [{"x": 1}]
        rows_b = [{"x": 1, "y": 2}]
        dd_a = generate_data_dictionary(rows_a)
        dd_b = generate_data_dictionary(rows_b)
        comp = compare_dictionaries(dd_a, dd_b)
        assert "y" in comp["added_columns"]
        assert len(comp["removed_columns"]) == 0

    def test_removed_column(self):
        rows_a = [{"x": 1, "y": 2}]
        rows_b = [{"x": 1}]
        dd_a = generate_data_dictionary(rows_a)
        dd_b = generate_data_dictionary(rows_b)
        comp = compare_dictionaries(dd_a, dd_b)
        assert "y" in comp["removed_columns"]

    def test_type_change(self):
        rows_a = [{"x": 1}, {"x": 2}, {"x": 3}]
        rows_b = [{"x": "hello"}, {"x": "world"}, {"x": "test"}]
        dd_a = generate_data_dictionary(rows_a)
        dd_b = generate_data_dictionary(rows_b)
        comp = compare_dictionaries(dd_a, dd_b)
        assert len(comp["type_changes"]) >= 1
        assert comp["type_changes"][0]["column"] == "x"

    def test_stat_change(self):
        rows_a = [{"x": 1}, {"x": 2}]
        rows_b = [{"x": 10}, {"x": 20}]
        dd_a = generate_data_dictionary(rows_a)
        dd_b = generate_data_dictionary(rows_b)
        comp = compare_dictionaries(dd_a, dd_b)
        # mean or min/max should differ
        assert len(comp["stat_changes"]) > 0

    def test_no_changes(self):
        rows = [{"x": 1}, {"x": 2}]
        dd_a = generate_data_dictionary(rows)
        dd_b = generate_data_dictionary(rows)
        comp = compare_dictionaries(dd_a, dd_b)
        assert len(comp["added_columns"]) == 0
        assert len(comp["removed_columns"]) == 0
        assert len(comp["type_changes"]) == 0
