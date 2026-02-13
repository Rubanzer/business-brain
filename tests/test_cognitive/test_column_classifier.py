"""Tests for the column semantic classifier."""

import pytest

from business_brain.cognitive.column_classifier import (
    classify_columns,
    format_classification_for_prompt,
    _detect_domain,
    _build_analysis_plan,
)


# ---------------------------------------------------------------------------
# Sales data
# ---------------------------------------------------------------------------


class TestSalesData:
    ROWS = [
        {"customer_id": 1, "product": "Widget A", "revenue": 15000, "order_date": "2024-01-15", "region": "North"},
        {"customer_id": 2, "product": "Widget B", "revenue": 22000, "order_date": "2024-02-20", "region": "South"},
        {"customer_id": 3, "product": "Widget A", "revenue": 18000, "order_date": "2024-03-10", "region": "North"},
        {"customer_id": 4, "product": "Widget C", "revenue": 9500, "order_date": "2024-04-05", "region": "East"},
        {"customer_id": 5, "product": "Widget B", "revenue": 31000, "order_date": "2024-05-18", "region": "West"},
    ]
    COLUMNS = ["customer_id", "product", "revenue", "order_date", "region"]

    def test_customer_id_is_identifier(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["customer_id"]["semantic_type"] == "identifier"

    def test_product_is_categorical(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["product"]["semantic_type"] == "categorical"

    def test_revenue_is_currency(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["revenue"]["semantic_type"] == "numeric_currency"

    def test_order_date_is_temporal(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["order_date"]["semantic_type"] == "temporal"

    def test_region_is_categorical(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["region"]["semantic_type"] == "categorical"

    def test_domain_is_sales(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["domain_hint"] == "sales"


# ---------------------------------------------------------------------------
# HR data
# ---------------------------------------------------------------------------


class TestHRData:
    ROWS = [
        {"employee_id": 101, "name": "Alice", "department": "Engineering", "salary": 85000, "hire_date": "2020-03-01"},
        {"employee_id": 102, "name": "Bob", "department": "Marketing", "salary": 72000, "hire_date": "2021-06-15"},
        {"employee_id": 103, "name": "Carol", "department": "Engineering", "salary": 91000, "hire_date": "2019-11-20"},
        {"employee_id": 104, "name": "Dave", "department": "Sales", "salary": 68000, "hire_date": "2022-01-10"},
    ]
    COLUMNS = ["employee_id", "name", "department", "salary", "hire_date"]

    def test_employee_id_is_identifier(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["employee_id"]["semantic_type"] == "identifier"

    def test_department_is_categorical(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["department"]["semantic_type"] == "categorical"

    def test_salary_is_currency(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["salary"]["semantic_type"] == "numeric_currency"

    def test_hire_date_is_temporal(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["hire_date"]["semantic_type"] == "temporal"

    def test_domain_is_hr(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["domain_hint"] == "hr"


# ---------------------------------------------------------------------------
# Procurement data
# ---------------------------------------------------------------------------


class TestProcurementData:
    ROWS = [
        {"PARTY": "MGM", "GRADE": "A", "RATE": 27000, "Fe": 58.5, "Yield (%)": 90.2},
        {"PARTY": "AARTI", "GRADE": "B", "RATE": 26550, "Fe": 56.1, "Yield (%)": 88.7},
        {"PARTY": "MGM", "GRADE": "A", "RATE": 28000, "Fe": 59.2, "Yield (%)": 91.5},
        {"PARTY": "SHREE", "GRADE": "C", "RATE": 29500, "Fe": 55.0, "Yield (%)": 85.3},
        {"PARTY": "AARTI", "GRADE": "B", "RATE": 26800, "Fe": 57.3, "Yield (%)": 89.1},
    ]
    COLUMNS = ["PARTY", "GRADE", "RATE", "Fe", "Yield (%)"]

    def test_party_is_categorical(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["PARTY"]["semantic_type"] == "categorical"

    def test_grade_is_categorical(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["GRADE"]["semantic_type"] == "categorical"

    def test_rate_is_currency(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["RATE"]["semantic_type"] == "numeric_currency"

    def test_fe_is_numeric_metric(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["Fe"]["semantic_type"] == "numeric_metric"

    def test_yield_is_percentage(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["columns"]["Yield (%)"]["semantic_type"] == "numeric_percentage"

    def test_domain_is_procurement(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        assert result["domain_hint"] == "procurement"

    def test_stats_computed(self):
        result = classify_columns(self.COLUMNS, self.ROWS)
        rate_stats = result["columns"]["RATE"].get("stats", {})
        assert "mean" in rate_stats
        assert "min" in rate_stats
        assert "max" in rate_stats
        assert rate_stats["min"] == 26550
        assert rate_stats["max"] == 29500


# ---------------------------------------------------------------------------
# Generic / edge cases
# ---------------------------------------------------------------------------


class TestGenericData:
    def test_empty_rows(self):
        result = classify_columns(["a", "b"], [])
        assert result["columns"]["a"]["semantic_type"] is not None
        assert result["domain_hint"] == "general"

    def test_single_column(self):
        rows = [{"value": 10}, {"value": 20}, {"value": 30}]
        result = classify_columns(["value"], rows)
        assert result["columns"]["value"]["semantic_type"] == "numeric_metric"

    def test_all_numeric(self):
        rows = [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}]
        result = classify_columns(["x", "y", "z"], rows)
        for col in ["x", "y", "z"]:
            assert result["columns"][col]["semantic_type"].startswith("numeric")

    def test_no_categorical(self):
        rows = [{"amount": 100, "price": 50}, {"amount": 200, "price": 75}]
        result = classify_columns(["amount", "price"], rows)
        assert "group_by_aggregation" not in result["analysis_plan"]

    def test_boolean_column(self):
        rows = [{"active": "yes"}, {"active": "no"}, {"active": "yes"}]
        result = classify_columns(["active"], rows)
        assert result["columns"]["active"]["semantic_type"] == "boolean"

    def test_none_values_handled(self):
        rows = [{"val": None}, {"val": 10}, {"val": None}]
        result = classify_columns(["val"], rows)
        assert result["columns"]["val"]["null_count"] == 2

    def test_sql_types_respected(self):
        rows = [{"ts": "2024-01-01 00:00:00"}]
        result = classify_columns(["ts"], rows, col_types={"ts": "TIMESTAMP"})
        assert result["columns"]["ts"]["semantic_type"] == "temporal"


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------


class TestDomainDetection:
    def test_sales_keywords(self):
        assert _detect_domain(["customer", "order_id", "revenue", "product_name"]) == "sales"

    def test_finance_keywords(self):
        assert _detect_domain(["expense_type", "budget", "account_no", "cost"]) == "finance"

    def test_hr_keywords(self):
        assert _detect_domain(["employee_id", "salary", "department"]) == "hr"

    def test_marketing_keywords(self):
        assert _detect_domain(["campaign", "impression", "click", "conversion"]) == "marketing"

    def test_general_fallback(self):
        assert _detect_domain(["x", "y", "z"]) == "general"

    def test_single_keyword_not_enough(self):
        # Need >= 2 keyword matches for a domain
        assert _detect_domain(["customer"]) == "general"


# ---------------------------------------------------------------------------
# Analysis plan
# ---------------------------------------------------------------------------


class TestAnalysisPlan:
    def test_categorical_numeric_plan(self):
        rows = [
            {"cat": "A", "val": 10},
            {"cat": "B", "val": 20},
        ]
        result = classify_columns(["cat", "val"], rows)
        assert "group_by_aggregation" in result["analysis_plan"]
        assert "descriptive_statistics" in result["analysis_plan"]

    def test_temporal_numeric_plan(self):
        rows = [
            {"date": "2024-01-01", "amount": 100},
            {"date": "2024-02-01", "amount": 200},
        ]
        result = classify_columns(["date", "amount"], rows)
        assert "time_trend" in result["analysis_plan"]

    def test_multi_numeric_correlation(self):
        rows = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
        ]
        result = classify_columns(["a", "b", "c"], rows)
        assert "correlation" in result["analysis_plan"]

    def test_outlier_detection_included(self):
        rows = [{"val": 10}, {"val": 20}]
        result = classify_columns(["val"], rows)
        assert "outlier_detection" in result["analysis_plan"]


# ---------------------------------------------------------------------------
# Chart plan
# ---------------------------------------------------------------------------


class TestChartPlan:
    def test_bar_chart_for_categorical_numeric(self):
        rows = [
            {"cat": "A", "val": 10},
            {"cat": "B", "val": 20},
        ]
        result = classify_columns(["cat", "val"], rows)
        bar_charts = [c for c in result["chart_plan"] if c["type"] == "bar"]
        assert len(bar_charts) >= 1

    def test_scatter_chart_for_multi_numeric(self):
        rows = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        result = classify_columns(["x", "y"], rows)
        scatter_charts = [c for c in result["chart_plan"] if c["type"] == "scatter"]
        assert len(scatter_charts) >= 1

    def test_pie_chart_for_few_categories(self):
        rows = [{"status": "A"}, {"status": "B"}, {"status": "C"}]
        result = classify_columns(["status"], rows)
        pie_charts = [c for c in result["chart_plan"] if c["type"] == "pie"]
        assert len(pie_charts) >= 1

    def test_number_format_set(self):
        rows = [{"cat": "A", "price": 100}]
        result = classify_columns(["cat", "price"], rows)
        bar_charts = [c for c in result["chart_plan"] if c["type"] == "bar"]
        if bar_charts:
            assert "number_format" in bar_charts[0]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


class TestFormatClassification:
    def test_format_output_is_string(self):
        rows = [{"cat": "A", "val": 10}]
        result = classify_columns(["cat", "val"], rows)
        formatted = format_classification_for_prompt(result)
        assert isinstance(formatted, str)
        assert "cat" in formatted
        assert "val" in formatted

    def test_format_includes_domain(self):
        rows = [{"customer": "X", "revenue": 100}]
        result = classify_columns(["customer", "revenue"], rows)
        formatted = format_classification_for_prompt(result)
        assert "Domain hint:" in formatted

    def test_format_includes_analysis_plan(self):
        rows = [{"cat": "A", "val": 10}]
        result = classify_columns(["cat", "val"], rows)
        formatted = format_classification_for_prompt(result)
        assert "Suggested analyses:" in formatted
