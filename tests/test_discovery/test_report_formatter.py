"""Tests for report formatter — markdown report generation from analysis results."""

from business_brain.discovery.report_formatter import (
    FormattedReport,
    ReportSection,
    combine_reports,
    format_analysis_report,
    format_comparison_table,
    format_executive_summary,
    format_metric_card,
    format_trend_narrative,
    truncate_report,
    _count_words,
    _format_indian_number,
)


# ===================================================================
# Indian locale number formatting
# ===================================================================


class TestIndianNumberFormatting:
    def test_small_number(self):
        assert _format_indian_number(123) == "123"

    def test_thousands(self):
        assert _format_indian_number(1234) == "1,234"

    def test_ten_thousands(self):
        assert _format_indian_number(12345) == "12,345"

    def test_lakhs(self):
        assert _format_indian_number(123456) == "1,23,456"

    def test_ten_lakhs(self):
        assert _format_indian_number(1234567) == "12,34,567"

    def test_crores(self):
        assert _format_indian_number(12345678) == "1,23,45,678"

    def test_zero(self):
        assert _format_indian_number(0) == "0"

    def test_negative_number(self):
        result = _format_indian_number(-1234567)
        assert result == "-12,34,567"

    def test_float_whole(self):
        # A float that is effectively a whole number
        assert _format_indian_number(1234567.0) == "12,34,567"

    def test_float_with_decimals(self):
        result = _format_indian_number(1234567.89)
        assert result == "12,34,567.89"

    def test_small_float(self):
        result = _format_indian_number(0.99)
        assert result == "0.99"


# ===================================================================
# Word counting
# ===================================================================


class TestWordCount:
    def test_simple_sentence(self):
        assert _count_words("Hello world") == 2

    def test_excludes_markdown_hashes(self):
        assert _count_words("# Title") == 1

    def test_excludes_bold_markers(self):
        assert _count_words("**bold text**") == 2

    def test_excludes_table_pipes(self):
        assert _count_words("| col1 | col2 |") == 2

    def test_empty_string(self):
        assert _count_words("") == 0

    def test_only_markdown_syntax(self):
        assert _count_words("# ** | --- |") == 0


# ===================================================================
# Metric card formatting
# ===================================================================


class TestFormatMetricCard:
    def test_basic_metric(self):
        result = format_metric_card("Revenue", 1234567)
        assert "**Revenue**" in result
        assert "12,34,567" in result

    def test_metric_with_unit(self):
        result = format_metric_card("Revenue", 1234567, unit="INR")
        assert "INR" in result

    def test_metric_with_trend(self):
        result = format_metric_card("Revenue", 1234567, trend="\u2191 12.5%")
        assert "\u2191 12.5%" in result

    def test_metric_with_target(self):
        result = format_metric_card("Revenue", 1234567, target=1200000)
        assert "Target: 12,00,000" in result

    def test_metric_full(self):
        result = format_metric_card(
            "Revenue", 1234567, unit="INR", trend="\u2191 12.5%", target=1200000
        )
        assert "**Revenue**: 12,34,567 INR" in result
        assert "\u2191 12.5%" in result
        assert "Target: 12,00,000" in result

    def test_metric_no_extras(self):
        result = format_metric_card("Count", 42)
        assert result == "**Count**: 42"


# ===================================================================
# Comparison table formatting
# ===================================================================


class TestFormatComparisonTable:
    def test_basic_table(self):
        items = [
            {"Name": "A", "Score": 90},
            {"Name": "B", "Score": 85},
        ]
        result = format_comparison_table(items, ["Name", "Score"])
        assert "| Name | Score |" in result
        assert "| --- | --- |" in result
        assert "| A | 90 |" in result
        assert "| B | 85 |" in result

    def test_highlight_best(self):
        items = [
            {"Name": "A", "Score": 90},
            {"Name": "B", "Score": 95},
        ]
        result = format_comparison_table(items, ["Name", "Score"], highlight_best="Score")
        # Row B (index 1) should be bolded
        assert "**B**" in result
        assert "**95**" in result
        # Row A should NOT be bolded
        assert "| A | 90 |" in result

    def test_empty_items(self):
        assert format_comparison_table([], ["A", "B"]) == ""

    def test_empty_columns(self):
        assert format_comparison_table([{"A": 1}], []) == ""

    def test_missing_column_value(self):
        items = [{"Name": "A"}]
        result = format_comparison_table(items, ["Name", "Score"])
        assert "| A |  |" in result


# ===================================================================
# Executive summary formatting
# ===================================================================


class TestFormatExecutiveSummary:
    def test_full_summary(self):
        result = format_executive_summary(
            key_findings=["Revenue grew 15%", "Churn decreased"],
            action_items=["Expand marketing", "Hire more staff"],
            risk_alerts=["Cash flow risk in Q3"],
        )
        assert "### Key Findings" in result
        assert "- Revenue grew 15%" in result
        assert "### Action Items" in result
        assert "- Expand marketing" in result
        assert "### Risk Alerts" in result
        assert "- Cash flow risk in Q3" in result

    def test_summary_without_risks(self):
        result = format_executive_summary(
            key_findings=["Finding 1"],
            action_items=["Action 1"],
        )
        assert "### Key Findings" in result
        assert "### Action Items" in result
        assert "### Risk Alerts" not in result

    def test_empty_findings(self):
        result = format_executive_summary(
            key_findings=[],
            action_items=["Action 1"],
        )
        assert "### Key Findings" not in result
        assert "### Action Items" in result

    def test_all_empty(self):
        result = format_executive_summary(
            key_findings=[],
            action_items=[],
            risk_alerts=[],
        )
        assert result == ""

    def test_bullet_points_format(self):
        result = format_executive_summary(
            key_findings=["F1", "F2"],
            action_items=["A1"],
        )
        # Each finding should be on its own line with a bullet
        lines = result.split("\n")
        bullet_lines = [l for l in lines if l.startswith("- ")]
        assert len(bullet_lines) == 3  # 2 findings + 1 action


# ===================================================================
# Trend narrative generation
# ===================================================================


class TestFormatTrendNarrative:
    def test_increasing_trend(self):
        result = format_trend_narrative("Revenue", [100000, 120000, 140000, 150000])
        assert "increased" in result
        assert "1,00,000" in result
        assert "1,50,000" in result
        assert "+" in result

    def test_decreasing_trend(self):
        result = format_trend_narrative("Churn", [100, 90, 80, 70])
        assert "decreased" in result

    def test_flat_trend(self):
        result = format_trend_narrative("Metric", [100, 100, 100, 100])
        assert "remained flat" in result

    def test_with_periods(self):
        result = format_trend_narrative(
            "Revenue", [100, 200], periods=["Jan 2024", "Jun 2024"]
        )
        assert "Jan 2024" in result
        assert "Jun 2024" in result

    def test_empty_values(self):
        result = format_trend_narrative("Revenue", [])
        assert "No data available" in result

    def test_single_value(self):
        result = format_trend_narrative("Revenue", [5000])
        assert "single data point" in result
        assert "5,000" in result

    def test_fluctuating_increase(self):
        # Overall increase but with fluctuations
        result = format_trend_narrative("Sales", [100, 120, 90, 130, 110, 150])
        assert "fluctuations" in result
        assert "increased" in result

    def test_trend_arrow_increase(self):
        result = format_trend_narrative("Revenue", [100, 200])
        assert "\u2191" in result  # ↑

    def test_trend_arrow_decrease(self):
        result = format_trend_narrative("Cost", [200, 100])
        assert "\u2193" in result  # ↓


# ===================================================================
# Full analysis report generation
# ===================================================================


class TestFormatAnalysisReport:
    def test_full_report(self):
        report = format_analysis_report(
            table_name="sales",
            title="Sales Analysis",
            metrics={"Revenue": 1234567, "Count": 42},
            insights=["Sales are trending up", "Top product: Widget"],
            recommendations=["Increase marketing budget"],
            data_table=[
                {"Product": "Widget", "Sales": 100},
                {"Product": "Gadget", "Sales": 80},
            ],
        )
        assert report.title == "Sales Analysis"
        assert report.table_name == "sales"
        assert report.generated_at  # not empty
        assert "# Sales Analysis" in report.markdown
        assert report.word_count > 0
        assert len(report.sections) >= 3

    def test_report_section_ordering(self):
        report = format_analysis_report(
            table_name="t",
            title="T",
            metrics={"A": 1},
            insights=["insight"],
            recommendations=["rec"],
        )
        priorities = [s.priority for s in report.sections]
        assert priorities == sorted(priorities), "Sections should be sorted by priority"

    def test_report_metrics_only(self):
        report = format_analysis_report(
            table_name="sales",
            title="Metrics Only",
            metrics={"Revenue": 500000},
        )
        assert len(report.sections) == 1
        assert report.sections[0].section_type == "metric"
        assert "5,00,000" in report.markdown

    def test_report_empty_inputs(self):
        report = format_analysis_report(
            table_name="empty",
            title="Empty Report",
        )
        assert report.title == "Empty Report"
        assert report.table_name == "empty"
        assert len(report.sections) == 0
        assert "# Empty Report" in report.markdown

    def test_report_with_chart_data(self):
        report = format_analysis_report(
            table_name="t",
            title="Chart Report",
            chart_data={"labels": ["Jan", "Feb"], "values": [10, 20]},
        )
        assert len(report.sections) == 1
        assert report.sections[0].section_type == "chart_data"

    def test_report_contains_timestamp(self):
        report = format_analysis_report(table_name="t", title="T")
        assert "Generated:" in report.markdown
        # IST offset: +05:30
        assert "+05:30" in report.generated_at

    def test_report_insights_as_summary_section(self):
        report = format_analysis_report(
            table_name="t",
            title="T",
            insights=["Finding 1"],
        )
        assert report.sections[0].section_type == "summary"
        assert report.sections[0].priority == 1

    def test_report_string_metric(self):
        report = format_analysis_report(
            table_name="t",
            title="T",
            metrics={"Status": "Active"},
        )
        assert "Active" in report.markdown

    def test_single_section_report(self):
        report = format_analysis_report(
            table_name="t",
            title="One Section",
            recommendations=["Do something"],
        )
        assert len(report.sections) == 1
        assert report.sections[0].section_type == "recommendation"


# ===================================================================
# Report combination
# ===================================================================


class TestCombineReports:
    def test_combine_two_reports(self):
        r1 = format_analysis_report("t1", "Report 1", metrics={"A": 1})
        r2 = format_analysis_report("t2", "Report 2", metrics={"B": 2})
        combined = combine_reports([r1, r2])
        assert combined.title == "Combined Report"
        assert "Table of Contents" in combined.markdown
        assert "Report 1" in combined.markdown
        assert "Report 2" in combined.markdown
        assert "t1" in combined.table_name
        assert "t2" in combined.table_name

    def test_combine_empty_list(self):
        combined = combine_reports([])
        assert combined.title == "Combined Report"
        assert "No reports to combine" in combined.markdown

    def test_combine_single_report(self):
        r = format_analysis_report("t1", "Solo", metrics={"X": 5})
        combined = combine_reports([r])
        assert combined.title == "Solo"

    def test_combine_table_of_contents(self):
        r1 = format_analysis_report("t1", "First", metrics={"A": 1})
        r2 = format_analysis_report("t2", "Second", metrics={"B": 2})
        combined = combine_reports([r1, r2])
        assert "1. First" in combined.markdown
        assert "2. Second" in combined.markdown

    def test_combine_preserves_sections(self):
        r1 = format_analysis_report("t1", "R1", metrics={"A": 1}, insights=["I1"])
        r2 = format_analysis_report("t2", "R2", recommendations=["Rec1"])
        combined = combine_reports([r1, r2])
        # r1 has 2 sections (insights + metrics), r2 has 1 (rec)
        # plus TOC section and 2 divider sections = 6 total
        section_types = [s.section_type for s in combined.sections]
        assert "metric" in section_types
        assert "recommendation" in section_types


# ===================================================================
# Report truncation
# ===================================================================


class TestTruncateReport:
    def test_no_truncation_needed(self):
        report = format_analysis_report(
            "t", "Short", metrics={"A": 1}
        )
        truncated = truncate_report(report, max_words=5000)
        # Should return same report
        assert len(truncated.sections) == len(report.sections)

    def test_truncation_applied(self):
        # Create a wordy report
        long_insights = [f"This is insight number {i} with many many words" for i in range(50)]
        long_recs = [f"This is recommendation number {i} with many words too" for i in range(50)]
        report = format_analysis_report(
            "t", "Long Report",
            insights=long_insights,
            recommendations=long_recs,
            metrics={"Val": 42},
        )
        truncated = truncate_report(report, max_words=50)
        assert len(truncated.sections) < len(report.sections)
        assert "[Report truncated]" in truncated.markdown

    def test_truncation_keeps_highest_priority(self):
        report = format_analysis_report(
            "t", "T",
            insights=["Important insight " * 10],
            recommendations=["Low priority rec " * 10],
            metrics={"A": 1},
        )
        truncated = truncate_report(report, max_words=30)
        # The insights section (priority=1) should be kept
        types = [s.section_type for s in truncated.sections]
        assert "summary" in types  # insights have type "summary"

    def test_truncation_at_least_one_section(self):
        # Even if first section exceeds budget, it should still be included
        long_insights = [f"word " * 100]
        report = format_analysis_report(
            "t", "T",
            insights=long_insights,
            recommendations=["rec"],
        )
        truncated = truncate_report(report, max_words=5)
        assert len(truncated.sections) >= 1


# ===================================================================
# Markdown syntax correctness
# ===================================================================


class TestMarkdownSyntax:
    def test_report_starts_with_h1(self):
        report = format_analysis_report("t", "My Report")
        assert report.markdown.startswith("# My Report")

    def test_table_has_separator_row(self):
        items = [{"A": 1, "B": 2}]
        result = format_comparison_table(items, ["A", "B"])
        lines = result.strip().split("\n")
        assert len(lines) >= 3
        assert all(c in "| -" for c in lines[1])

    def test_executive_summary_uses_h3(self):
        result = format_executive_summary(
            key_findings=["F1"],
            action_items=["A1"],
        )
        assert "### Key Findings" in result
        assert "### Action Items" in result

    def test_report_sections_use_h3(self):
        report = format_analysis_report(
            "t", "T", insights=["I1"], metrics={"M": 1}
        )
        assert "### Insights" in report.markdown
        assert "### Metrics" in report.markdown
