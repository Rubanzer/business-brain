"""Tests for profile report generator."""

from business_brain.discovery.profile_report import (
    ProfileReport,
    ReportSection,
    compute_report_priority,
    format_report_text,
    generate_profile_report,
)


# ---------------------------------------------------------------------------
# generate_profile_report
# ---------------------------------------------------------------------------


class TestGenerateProfileReport:
    def test_basic_report(self):
        cols = {
            "id": {"semantic_type": "identifier", "cardinality": 100, "null_count": 0},
            "amount": {"semantic_type": "numeric_metric", "cardinality": 80, "null_count": 5},
        }
        report = generate_profile_report("orders", 100, cols)
        assert report.table_name == "orders"
        assert report.row_count == 100
        assert report.column_count == 2
        assert report.quality_score > 0
        assert len(report.sections) >= 3

    def test_empty_columns(self):
        report = generate_profile_report("empty", 0, {})
        assert report.column_count == 0
        assert report.quality_score == 0

    def test_with_relationships(self):
        cols = {"id": {"semantic_type": "identifier", "cardinality": 50, "null_count": 0}}
        rels = [{"table_a": "orders", "column_a": "customer_id", "table_b": "customers", "column_b": "id", "relationship_type": "join_key", "confidence": 0.9}]
        report = generate_profile_report("orders", 100, cols, relationships=rels)
        rel_section = [s for s in report.sections if s.title == "Relationships"]
        assert len(rel_section) == 1

    def test_domain_hint(self):
        cols = {"id": {"semantic_type": "identifier", "cardinality": 50, "null_count": 0}}
        report = generate_profile_report("suppliers", 100, cols, domain="procurement")
        assert report.domain == "procurement"
        assert "procurement" in report.summary

    def test_high_null_triggers_warning(self):
        cols = {
            "name": {"semantic_type": "categorical", "cardinality": 10, "null_count": 80},
        }
        report = generate_profile_report("test", 100, cols)
        quality_sections = [s for s in report.sections if s.title == "Data Quality"]
        assert quality_sections[0].severity in ("warning", "critical")

    def test_recommendations_generated(self):
        cols = {
            "val": {"semantic_type": "numeric_metric", "cardinality": 80, "null_count": 60},
        }
        report = generate_profile_report("test", 100, cols)
        # Should recommend adding identifier, temporal, and flag high null
        assert len(report.recommendations) >= 2

    def test_summary_text(self):
        cols = {"id": {"semantic_type": "identifier", "cardinality": 100, "null_count": 0}}
        report = generate_profile_report("products", 500, cols)
        assert "products" in report.summary
        assert "500" in report.summary

    def test_key_columns_section(self):
        cols = {
            "id": {"semantic_type": "identifier", "cardinality": 100, "null_count": 0},
            "date": {"semantic_type": "temporal", "cardinality": 50, "null_count": 0},
            "amount": {"semantic_type": "numeric_metric", "cardinality": 80, "null_count": 0},
        }
        report = generate_profile_report("test", 100, cols)
        key_sections = [s for s in report.sections if s.title == "Key Columns"]
        assert len(key_sections) == 1

    def test_no_key_columns_section_if_none(self):
        cols = {
            "val": {"semantic_type": "numeric_metric", "cardinality": 80, "null_count": 0},
        }
        report = generate_profile_report("test", 100, cols)
        key_sections = [s for s in report.sections if s.title == "Key Columns"]
        assert len(key_sections) == 0

    def test_small_dataset_recommendation(self):
        cols = {"id": {"semantic_type": "identifier", "cardinality": 10, "null_count": 0}}
        report = generate_profile_report("tiny", 10, cols)
        assert any("Small dataset" in r for r in report.recommendations)

    def test_low_cardinality_id_recommendation(self):
        cols = {"id": {"semantic_type": "identifier", "cardinality": 10, "null_count": 0}}
        report = generate_profile_report("test", 100, cols)
        assert any("low cardinality" in r.lower() or "duplicate" in r.lower() for r in report.recommendations)


# ---------------------------------------------------------------------------
# format_report_text
# ---------------------------------------------------------------------------


class TestFormatReportText:
    def test_basic_format(self):
        report = ProfileReport(
            table_name="test",
            row_count=100,
            column_count=5,
            domain="general",
            quality_score=85.0,
            sections=[ReportSection("Overview", "Test overview")],
            recommendations=["Fix nulls"],
            summary="Test summary.",
        )
        text = format_report_text(report)
        assert "=== Profile Report: test ===" in text
        assert "Overview" in text
        assert "Fix nulls" in text
        assert "Summary: Test summary." in text

    def test_severity_indicators(self):
        report = ProfileReport(
            table_name="test", row_count=100, column_count=5, domain="general", quality_score=50,
            sections=[
                ReportSection("Critical", "Bad stuff", severity="critical"),
                ReportSection("Warning", "Meh stuff", severity="warning"),
                ReportSection("Info", "Good stuff", severity="info"),
            ],
            recommendations=[], summary="",
        )
        text = format_report_text(report)
        assert "[!]" in text
        assert "[~]" in text


# ---------------------------------------------------------------------------
# compute_report_priority
# ---------------------------------------------------------------------------


class TestComputeReportPriority:
    def test_high_quality_low_priority(self):
        report = ProfileReport(
            table_name="test", row_count=100, column_count=5, domain="general",
            quality_score=95.0,
            sections=[ReportSection("Info", "All good")],
            recommendations=[],
            summary="",
        )
        priority = compute_report_priority(report)
        assert priority < 70

    def test_low_quality_high_priority(self):
        report = ProfileReport(
            table_name="test", row_count=100, column_count=5, domain="general",
            quality_score=30.0,
            sections=[
                ReportSection("Critical", "Bad", severity="critical"),
                ReportSection("Warning", "Meh", severity="warning"),
            ],
            recommendations=["Fix A", "Fix B", "Fix C", "Fix D"],
            summary="",
        )
        priority = compute_report_priority(report)
        assert priority > 80

    def test_max_100(self):
        report = ProfileReport(
            table_name="test", row_count=100, column_count=5, domain="general",
            quality_score=10.0,
            sections=[ReportSection("Critical", "Bad", severity="critical")] * 5,
            recommendations=["Fix"] * 20,
            summary="",
        )
        priority = compute_report_priority(report)
        assert priority <= 100
