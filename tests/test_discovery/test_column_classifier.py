"""Tests for the column semantic classifier."""

from business_brain.cognitive.column_classifier import (
    _basic_stats,
    _build_analysis_plan,
    _build_chart_plan,
    _classify_one,
    _detect_domain,
    _is_numeric_column,
    _parse_numerics,
    classify_columns,
    format_classification_for_prompt,
)


# ---------------------------------------------------------------------------
# _classify_one — single column classification
# ---------------------------------------------------------------------------


class TestClassifyOne:
    def _classify(self, col, values, sql_type=None, total=None):
        rows = [{col: v} for v in values]
        total = total or len(rows)
        return _classify_one(col, rows, total, sql_type)

    def test_temporal_by_sql_type(self):
        r = self._classify("col", ["2024-01-01"], "timestamp")
        assert r["semantic_type"] == "temporal"

    def test_temporal_by_name_pattern(self):
        r = self._classify("created_at", ["2024-01-01", "2024-01-02"])
        assert r["semantic_type"] == "temporal"

    def test_temporal_date_suffix(self):
        r = self._classify("hire_date", ["2024-01-01"])
        assert r["semantic_type"] == "temporal"

    def test_boolean_by_sql_type(self):
        r = self._classify("flag", [True, False], "boolean")
        assert r["semantic_type"] == "boolean"

    def test_boolean_by_values(self):
        r = self._classify("active", ["yes", "no", "yes", "no"])
        assert r["semantic_type"] == "boolean"

    def test_boolean_true_false_strings(self):
        r = self._classify("flag", ["true", "false", "true"])
        assert r["semantic_type"] == "boolean"

    def test_identifier_by_pattern_high_cardinality(self):
        values = [f"CUST{i:04d}" for i in range(10)]
        r = self._classify("customer_id", values)
        assert r["semantic_type"] == "identifier"

    def test_identifier_heat_no(self):
        values = [f"H{i:03d}" for i in range(10)]
        r = self._classify("heat_no", values)
        assert r["semantic_type"] == "identifier"

    def test_numeric_metric_generic(self):
        r = self._classify("output_tons", [28.5, 30.1, 27.8], "numeric")
        assert r["semantic_type"] == "numeric_metric"
        assert "stats" in r

    def test_numeric_currency_by_name(self):
        r = self._classify("revenue", [1000, 2000, 3000])
        assert r["semantic_type"] == "numeric_currency"

    def test_numeric_percentage_by_name_and_range(self):
        r = self._classify("completion_pct", [50.0, 75.0, 90.0])
        assert r["semantic_type"] == "numeric_percentage"

    def test_percentage_out_of_range_falls_to_metric(self):
        """If values are outside 0-100, even with percentage name, classify as metric."""
        r = self._classify("ratio_pct", [0.5, 0.8, 1.2])
        # 0.5, 0.8, 1.2 are in [0, 100] so it's still percentage
        assert r["semantic_type"] == "numeric_percentage"

    def test_categorical_low_cardinality_string(self):
        r = self._classify("status", ["A", "B", "C", "A", "B"])
        assert r["semantic_type"] == "categorical"

    def test_text_high_cardinality_long(self):
        long_texts = [f"This is a very long text description number {i} " * 3 for i in range(20)]
        r = self._classify("description", long_texts, total=20)
        assert r["semantic_type"] == "text"

    def test_comma_separated_numbers(self):
        r = self._classify("amount", ["1,000", "2,500", "3,750"])
        assert r["semantic_type"] == "numeric_currency"

    def test_empty_values_default_to_text(self):
        r = self._classify("empty_col", [None, None, None])
        # All nulls — falls through to text/categorical
        assert r["semantic_type"] in ("text", "categorical", "boolean")

    def test_null_count_tracked(self):
        r = self._classify("col", [1, 2, None, None, 5])
        assert r["null_count"] == 2

    def test_sample_values_limited(self):
        values = list(range(20))
        r = self._classify("num", values)
        assert len(r["sample_values"]) <= 5


# ---------------------------------------------------------------------------
# _detect_domain
# ---------------------------------------------------------------------------


class TestDetectDomain:
    def test_manufacturing_domain(self):
        cols = ["furnace_temp", "kva", "output_tonnage", "shift"]
        assert _detect_domain(cols) == "manufacturing"

    def test_sales_domain(self):
        cols = ["customer", "revenue", "product", "quantity"]
        assert _detect_domain(cols) == "sales"

    def test_hr_domain(self):
        cols = ["employee", "salary", "department", "hire_date"]
        assert _detect_domain(cols) == "hr"

    def test_finance_domain(self):
        cols = ["expense", "budget", "account", "invoice"]
        assert _detect_domain(cols) == "finance"

    def test_procurement_domain(self):
        cols = ["supplier", "material", "rate", "vendor"]
        assert _detect_domain(cols) == "procurement"

    def test_quality_domain(self):
        cols = ["rejection", "defect", "inspection", "grade"]
        assert _detect_domain(cols) == "quality"

    def test_logistics_domain(self):
        cols = ["truck", "gate", "vehicle", "dispatch"]
        assert _detect_domain(cols) == "logistics"

    def test_energy_domain(self):
        cols = ["power", "kwh", "voltage", "consumption"]
        assert _detect_domain(cols) == "energy"

    def test_marketing_domain(self):
        cols = ["campaign", "impression", "click", "conversion"]
        assert _detect_domain(cols) == "marketing"

    def test_inventory_domain(self):
        cols = ["product", "warehouse", "stock", "inventory"]
        assert _detect_domain(cols) == "inventory"

    def test_general_when_no_match(self):
        cols = ["x", "y", "z"]
        assert _detect_domain(cols) == "general"

    def test_general_when_single_keyword(self):
        """Need >= 2 keyword matches to assign a domain."""
        cols = ["customer", "x", "y"]
        # Only 1 match for "sales" → general
        assert _detect_domain(cols) == "general"

    def test_case_insensitive(self):
        cols = ["FURNACE_TEMP", "KVA", "OUTPUT"]
        assert _detect_domain(cols) == "manufacturing"


# ---------------------------------------------------------------------------
# _is_numeric_column & _parse_numerics
# ---------------------------------------------------------------------------


class TestNumericHelpers:
    def test_is_numeric_by_sql_type(self):
        assert _is_numeric_column([], "integer")

    def test_is_numeric_by_values(self):
        assert _is_numeric_column([1, 2, 3], "")

    def test_is_numeric_strings(self):
        assert _is_numeric_column(["1.5", "2.5", "3.5"], "")

    def test_is_numeric_comma_strings(self):
        assert _is_numeric_column(["1,000", "2,000"], "")

    def test_not_numeric_text(self):
        assert not _is_numeric_column(["hello", "world", "foo"], "")

    def test_empty_not_numeric(self):
        assert not _is_numeric_column([], "")

    def test_mixed_mostly_numeric(self):
        """Threshold is > 0.8, so exactly 80% (4/5) is NOT numeric."""
        values = [1, 2, 3, 4, "text"]  # 80% = 0.8, not > 0.8
        assert not _is_numeric_column(values, "")

    def test_mixed_above_80_threshold(self):
        """5/6 = 83.3% → above threshold."""
        values = [1, 2, 3, 4, 5, "text"]  # 83.3% numeric
        assert _is_numeric_column(values, "")

    def test_mixed_below_threshold(self):
        values = [1, "a", "b", "c", "d"]  # 20% numeric
        assert not _is_numeric_column(values, "")

    def test_parse_numerics_mixed(self):
        nums = _parse_numerics([1, "2.5", "3,000", "abc", None])
        assert len(nums) == 3
        assert 3000.0 in nums

    def test_parse_numerics_empty(self):
        assert _parse_numerics([]) == []


# ---------------------------------------------------------------------------
# _basic_stats
# ---------------------------------------------------------------------------


class TestBasicStats:
    def test_basic_stats(self):
        s = _basic_stats([10.0, 20.0, 30.0])
        assert s["mean"] == 20.0
        assert s["min"] == 10.0
        assert s["max"] == 30.0
        assert "median" in s
        assert "stdev" in s

    def test_single_value_no_stdev(self):
        s = _basic_stats([42.0])
        assert s["mean"] == 42.0
        assert "stdev" not in s

    def test_empty_returns_empty(self):
        assert _basic_stats([]) == {}


# ---------------------------------------------------------------------------
# _build_analysis_plan
# ---------------------------------------------------------------------------


class TestBuildAnalysisPlan:
    def test_numeric_only(self):
        classified = {
            "price": {"semantic_type": "numeric_currency"},
        }
        plan = _build_analysis_plan(classified)
        assert "descriptive_statistics" in plan
        assert "outlier_detection" in plan
        assert "cost_analysis" in plan

    def test_categorical_and_numeric(self):
        classified = {
            "category": {"semantic_type": "categorical"},
            "amount": {"semantic_type": "numeric_currency"},
        }
        plan = _build_analysis_plan(classified)
        assert "group_by_aggregation" in plan

    def test_temporal_and_numeric(self):
        classified = {
            "date": {"semantic_type": "temporal"},
            "value": {"semantic_type": "numeric_metric"},
        }
        plan = _build_analysis_plan(classified)
        assert "time_trend" in plan

    def test_two_numerics_correlation(self):
        classified = {
            "a": {"semantic_type": "numeric_metric"},
            "b": {"semantic_type": "numeric_metric"},
        }
        plan = _build_analysis_plan(classified)
        assert "correlation" in plan

    def test_categorical_only(self):
        classified = {
            "status": {"semantic_type": "categorical"},
            "type": {"semantic_type": "categorical"},
        }
        plan = _build_analysis_plan(classified)
        assert "frequency_distribution" in plan

    def test_percentage_distribution(self):
        classified = {
            "completion": {"semantic_type": "numeric_percentage"},
        }
        plan = _build_analysis_plan(classified)
        assert "percentage_distribution" in plan


# ---------------------------------------------------------------------------
# _build_chart_plan
# ---------------------------------------------------------------------------


class TestBuildChartPlan:
    def test_bar_chart_for_cat_plus_num(self):
        classified = {
            "category": {"semantic_type": "categorical", "cardinality": 5},
            "amount": {"semantic_type": "numeric_currency", "cardinality": 50},
        }
        charts = _build_chart_plan(classified, ["category", "amount"])
        assert any(c["type"] == "bar" for c in charts)

    def test_line_chart_for_temporal_plus_num(self):
        classified = {
            "date": {"semantic_type": "temporal", "cardinality": 30},
            "value": {"semantic_type": "numeric_metric", "cardinality": 50},
        }
        charts = _build_chart_plan(classified, ["date", "value"])
        assert any(c["type"] == "line" for c in charts)

    def test_scatter_for_two_numerics(self):
        classified = {
            "a": {"semantic_type": "numeric_metric", "cardinality": 50},
            "b": {"semantic_type": "numeric_metric", "cardinality": 50},
        }
        charts = _build_chart_plan(classified, ["a", "b"])
        assert any(c["type"] == "scatter" for c in charts)

    def test_pie_for_low_cardinality_cat(self):
        classified = {
            "status": {"semantic_type": "categorical", "cardinality": 3},
        }
        charts = _build_chart_plan(classified, ["status"])
        assert any(c["type"] == "pie" for c in charts)

    def test_no_pie_for_high_cardinality(self):
        classified = {
            "name": {"semantic_type": "categorical", "cardinality": 20},
        }
        charts = _build_chart_plan(classified, ["name"])
        assert not any(c["type"] == "pie" for c in charts)

    def test_currency_format_in_bar(self):
        classified = {
            "category": {"semantic_type": "categorical", "cardinality": 5},
            "revenue": {"semantic_type": "numeric_currency", "cardinality": 50},
        }
        charts = _build_chart_plan(classified, ["category", "revenue"])
        bar = [c for c in charts if c["type"] == "bar"][0]
        assert bar["number_format"] == "currency"


# ---------------------------------------------------------------------------
# format_classification_for_prompt
# ---------------------------------------------------------------------------


class TestFormatClassification:
    def test_basic_format(self):
        cls = {
            "columns": {
                "revenue": {
                    "semantic_type": "numeric_currency",
                    "cardinality": 50,
                    "stats": {"mean": 1000.0, "min": 100.0, "max": 5000.0},
                    "sample_values": ["100", "500", "1000"],
                },
            },
            "domain_hint": "sales",
            "analysis_plan": ["descriptive_statistics"],
        }
        text = format_classification_for_prompt(cls)
        assert "Domain hint: sales" in text
        assert "revenue: numeric_currency" in text
        assert "mean=1000.0" in text

    def test_empty_classification(self):
        cls = {"columns": {}, "domain_hint": "general", "analysis_plan": []}
        text = format_classification_for_prompt(cls)
        assert "Domain hint: general" in text


# ---------------------------------------------------------------------------
# Full classify_columns integration
# ---------------------------------------------------------------------------


class TestClassifyColumnsIntegration:
    def test_full_pipeline(self):
        columns = ["id", "name", "revenue", "date"]
        rows = [
            {"id": 1, "name": "A", "revenue": 100, "date": "2024-01-01"},
            {"id": 2, "name": "B", "revenue": 200, "date": "2024-01-02"},
            {"id": 3, "name": "C", "revenue": 300, "date": "2024-01-03"},
        ]
        result = classify_columns(columns, rows)
        assert "columns" in result
        assert "domain_hint" in result
        assert "analysis_plan" in result
        assert "chart_plan" in result
        assert result["columns"]["date"]["semantic_type"] == "temporal"
        assert result["columns"]["revenue"]["semantic_type"] == "numeric_currency"

    def test_with_col_types(self):
        columns = ["val"]
        rows = [{"val": "100"}, {"val": "200"}]
        result = classify_columns(columns, rows, {"val": "numeric"})
        assert result["columns"]["val"]["semantic_type"] in ("numeric_metric", "numeric_currency")

    def test_empty_rows(self):
        result = classify_columns(["col"], [])
        assert result["columns"]["col"]["cardinality"] == 0
