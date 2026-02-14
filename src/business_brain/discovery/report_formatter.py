"""Report formatter — formats analysis results into clean markdown reports.

Pure functions for assembling structured markdown reports from metrics,
insights, recommendations, and tabular data. Uses Indian locale number
formatting and IST timestamps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


# IST offset: UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class ReportSection:
    """A single section within a formatted report."""

    title: str
    content: str  # markdown content
    priority: int  # 1=highest, used for ordering
    section_type: str  # "summary", "metric", "chart_data", "table", "recommendation"


@dataclass
class FormattedReport:
    """A complete formatted markdown report."""

    title: str
    sections: list[ReportSection] = field(default_factory=list)
    generated_at: str = ""  # ISO timestamp
    table_name: str = ""
    markdown: str = ""  # full rendered markdown
    word_count: int = 0


# ---------------------------------------------------------------------------
# Indian locale number formatting
# ---------------------------------------------------------------------------


def _format_indian_number(value: float | int) -> str:
    """Format a number using the Indian numbering system (1,23,456).

    The Indian system groups the last three digits, then every two digits
    after that.  For example:
        1234567  -> 12,34,567
        12345    -> 12,345
        1234     -> 1,234
        123      -> 123
    """
    if isinstance(value, float):
        # Separate integer and decimal parts
        if value == int(value) and abs(value) < 1e15:
            # Whole number stored as float — drop the decimal
            int_part = str(int(value))
            decimal_part = ""
        else:
            str_val = f"{value:.2f}"
            int_part, decimal_part = str_val.split(".")
    else:
        int_part = str(value)
        decimal_part = ""

    negative = int_part.startswith("-")
    if negative:
        int_part = int_part[1:]

    # Apply Indian grouping
    if len(int_part) <= 3:
        grouped = int_part
    else:
        # Last 3 digits stay together
        last3 = int_part[-3:]
        rest = int_part[:-3]
        # Group remaining digits in pairs from the right
        chunks: list[str] = []
        while len(rest) > 2:
            chunks.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            chunks.append(rest)
        chunks.reverse()
        grouped = ",".join(chunks) + "," + last3

    result = grouped
    if decimal_part:
        result += "." + decimal_part
    if negative:
        result = "-" + result
    return result


# ---------------------------------------------------------------------------
# Word counting
# ---------------------------------------------------------------------------


def _count_words(text: str) -> int:
    """Count words in text, excluding markdown syntax characters.

    Strips markdown formatting (# * _ | - ` > [ ] ( )) before counting.
    """
    # Remove markdown syntax
    cleaned = re.sub(r"[#*_|`>\[\]()~\-]", " ", text)
    # Remove multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return 0
    return len(cleaned.split())


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _now_ist_iso() -> str:
    """Return the current timestamp in IST as an ISO string."""
    return datetime.now(_IST).isoformat()


# ---------------------------------------------------------------------------
# Trend arrow helpers
# ---------------------------------------------------------------------------


def _trend_arrow(change_pct: float) -> str:
    """Return a trend arrow based on percentage change."""
    if change_pct > 0.5:
        return "\u2191"  # ↑
    elif change_pct < -0.5:
        return "\u2193"  # ↓
    else:
        return "\u2192"  # →


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_metric_card(
    name: str,
    value: float,
    unit: str = "",
    trend: str = "",
    target: float | None = None,
) -> str:
    """Format a single metric as a markdown card.

    Example output:
        **Revenue**: 12,34,567 INR (↑ 12.5% | Target: 12,00,000)
    """
    formatted_value = _format_indian_number(value)
    parts = [f"**{name}**: {formatted_value}"]

    if unit:
        parts[0] += f" {unit}"

    extras: list[str] = []
    if trend:
        extras.append(trend)
    if target is not None:
        extras.append(f"Target: {_format_indian_number(target)}")

    if extras:
        parts.append(f"({' | '.join(extras)})")

    return " ".join(parts)


def format_comparison_table(
    items: list[dict],
    columns: list[str],
    highlight_best: str | None = None,
) -> str:
    """Format a list of dicts as a markdown table.

    If *highlight_best* names a column, the row with the highest numeric
    value in that column has its cells wrapped in bold.
    """
    if not items or not columns:
        return ""

    # Determine which row to highlight
    best_idx: int | None = None
    if highlight_best and highlight_best in columns:
        best_val = None
        for i, item in enumerate(items):
            v = item.get(highlight_best)
            if v is not None:
                try:
                    num = float(v)
                    if best_val is None or num > best_val:
                        best_val = num
                        best_idx = i
                except (ValueError, TypeError):
                    pass

    # Header
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    # Rows
    rows: list[str] = []
    for i, item in enumerate(items):
        cells: list[str] = []
        for col in columns:
            raw = item.get(col, "")
            cell = str(raw)
            if i == best_idx:
                cell = f"**{cell}**"
            cells.append(cell)
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + rows)


def format_executive_summary(
    key_findings: list[str],
    action_items: list[str],
    risk_alerts: list[str] | None = None,
) -> str:
    """Format an executive summary section with findings, actions, and risks.

    Uses bullet points for each sub-section.
    """
    parts: list[str] = []

    if key_findings:
        parts.append("### Key Findings")
        for f in key_findings:
            parts.append(f"- {f}")
        parts.append("")

    if action_items:
        parts.append("### Action Items")
        for a in action_items:
            parts.append(f"- {a}")
        parts.append("")

    if risk_alerts:
        parts.append("### Risk Alerts")
        for r in risk_alerts:
            parts.append(f"- {r}")
        parts.append("")

    return "\n".join(parts)


def format_trend_narrative(
    metric_name: str,
    values: list[float],
    periods: list[str] | None = None,
) -> str:
    """Generate a narrative description of a metric's trend.

    Example:
        'Revenue has increased steadily over the last 6 periods, from
         1,00,000 to 1,50,000 (+50.0%).'
    """
    if not values:
        return f"No data available for {metric_name}."

    if len(values) == 1:
        return (
            f"{metric_name} has a single data point: "
            f"{_format_indian_number(values[0])}."
        )

    first = values[0]
    last = values[-1]
    n = len(values)

    if first != 0:
        change_pct = ((last - first) / abs(first)) * 100
    else:
        change_pct = 0.0 if last == 0 else 100.0

    arrow = _trend_arrow(change_pct)

    # Determine direction word
    if change_pct > 0.5:
        direction = "increased"
    elif change_pct < -0.5:
        direction = "decreased"
    else:
        direction = "remained flat"

    # Determine steadiness: count how many consecutive changes are in the
    # same direction as the overall trend.
    same_dir = 0
    for i in range(1, n):
        diff = values[i] - values[i - 1]
        if (change_pct > 0 and diff > 0) or (change_pct < 0 and diff < 0) or (change_pct == 0):
            same_dir += 1
    consistency = same_dir / (n - 1) if n > 1 else 1.0
    modifier = "steadily" if consistency >= 0.7 else "with fluctuations"

    period_label = f"the last {n} periods"
    if periods and len(periods) >= 2:
        period_label = f"{periods[0]} to {periods[-1]}"

    sign = "+" if change_pct >= 0 else ""
    formatted_first = _format_indian_number(first)
    formatted_last = _format_indian_number(last)

    return (
        f"{metric_name} has {direction} {modifier} over {period_label}, "
        f"from {formatted_first} to {formatted_last} "
        f"({sign}{change_pct:.1f}%). {arrow}"
    )


def format_analysis_report(
    table_name: str,
    title: str,
    metrics: dict[str, float | int | str] | None = None,
    insights: list[str] | None = None,
    recommendations: list[str] | None = None,
    data_table: list[dict] | None = None,
    chart_data: dict | None = None,
) -> FormattedReport:
    """Create a comprehensive markdown report from analysis results.

    Assembles sections in priority order:
        summary -> metrics -> insights -> data -> chart_data -> recommendations
    """
    sections: list[ReportSection] = []
    timestamp = _now_ist_iso()

    # 1. Summary section (from insights)
    if insights:
        content_lines = ["### Insights"]
        for ins in insights:
            content_lines.append(f"- {ins}")
        sections.append(ReportSection(
            title="Insights",
            content="\n".join(content_lines),
            priority=1,
            section_type="summary",
        ))

    # 2. Metrics section
    if metrics:
        content_lines = ["### Metrics"]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                content_lines.append(f"- **{k}**: {_format_indian_number(v)}")
            else:
                content_lines.append(f"- **{k}**: {v}")
        sections.append(ReportSection(
            title="Metrics",
            content="\n".join(content_lines),
            priority=2,
            section_type="metric",
        ))

    # 3. Data table section
    if data_table:
        if data_table:
            cols = list(data_table[0].keys())
            table_md = format_comparison_table(data_table, cols)
            sections.append(ReportSection(
                title="Data",
                content=f"### Data\n\n{table_md}",
                priority=3,
                section_type="table",
            ))

    # 4. Chart data section
    if chart_data:
        content_lines = ["### Chart Data"]
        for k, v in chart_data.items():
            content_lines.append(f"- **{k}**: {v}")
        sections.append(ReportSection(
            title="Chart Data",
            content="\n".join(content_lines),
            priority=4,
            section_type="chart_data",
        ))

    # 5. Recommendations section
    if recommendations:
        content_lines = ["### Recommendations"]
        for rec in recommendations:
            content_lines.append(f"- {rec}")
        sections.append(ReportSection(
            title="Recommendations",
            content="\n".join(content_lines),
            priority=5,
            section_type="recommendation",
        ))

    # Sort sections by priority
    sections.sort(key=lambda s: s.priority)

    # Build full markdown
    md_parts = [f"# {title}", f"*Table: {table_name} | Generated: {timestamp}*", ""]
    for section in sections:
        md_parts.append(section.content)
        md_parts.append("")

    markdown = "\n".join(md_parts)
    word_count = _count_words(markdown)

    return FormattedReport(
        title=title,
        sections=sections,
        generated_at=timestamp,
        table_name=table_name,
        markdown=markdown,
        word_count=word_count,
    )


def combine_reports(reports: list[FormattedReport]) -> FormattedReport:
    """Combine multiple reports into one master report with a table of contents."""
    if not reports:
        timestamp = _now_ist_iso()
        return FormattedReport(
            title="Combined Report",
            sections=[],
            generated_at=timestamp,
            table_name="",
            markdown="# Combined Report\n\n*No reports to combine.*\n",
            word_count=_count_words("Combined Report No reports to combine."),
        )

    if len(reports) == 1:
        return reports[0]

    timestamp = _now_ist_iso()
    all_sections: list[ReportSection] = []
    table_names: list[str] = []

    # Build table of contents
    toc_lines = ["## Table of Contents", ""]
    for i, report in enumerate(reports, 1):
        toc_lines.append(f"{i}. {report.title}")
        table_names.append(report.table_name)

    toc_section = ReportSection(
        title="Table of Contents",
        content="\n".join(toc_lines),
        priority=0,
        section_type="summary",
    )
    all_sections.append(toc_section)

    # Add each report's sections, grouped under report title
    for report in reports:
        divider_section = ReportSection(
            title=report.title,
            content=f"## {report.title}",
            priority=report.sections[0].priority if report.sections else 99,
            section_type="summary",
        )
        all_sections.append(divider_section)
        all_sections.extend(report.sections)

    # Build markdown
    combined_name = ", ".join(n for n in table_names if n)
    md_parts = [
        "# Combined Report",
        f"*Tables: {combined_name} | Generated: {timestamp}*",
        "",
    ]
    for section in all_sections:
        md_parts.append(section.content)
        md_parts.append("")

    markdown = "\n".join(md_parts)
    word_count = _count_words(markdown)

    return FormattedReport(
        title="Combined Report",
        sections=all_sections,
        generated_at=timestamp,
        table_name=combined_name,
        markdown=markdown,
        word_count=word_count,
    )


def truncate_report(
    report: FormattedReport,
    max_words: int = 500,
) -> FormattedReport:
    """Truncate a report to fit within word limit, keeping highest-priority sections.

    Sections are included in priority order (lowest number = highest priority)
    until the budget is exhausted.  If even the first section exceeds the
    budget, it is still included.
    """
    if report.word_count <= max_words:
        return report

    sorted_sections = sorted(report.sections, key=lambda s: s.priority)
    kept_sections: list[ReportSection] = []
    running_words = _count_words(report.title)

    for section in sorted_sections:
        section_words = _count_words(section.content)
        if running_words + section_words <= max_words or not kept_sections:
            kept_sections.append(section)
            running_words += section_words
        else:
            break

    # Re-sort kept sections by priority for display
    kept_sections.sort(key=lambda s: s.priority)

    # Re-render markdown
    md_parts = [
        f"# {report.title}",
        f"*Table: {report.table_name} | Generated: {report.generated_at}*",
        "",
    ]
    for section in kept_sections:
        md_parts.append(section.content)
        md_parts.append("")

    md_parts.append("*[Report truncated]*")
    markdown = "\n".join(md_parts)
    word_count = _count_words(markdown)

    return FormattedReport(
        title=report.title,
        sections=kept_sections,
        generated_at=report.generated_at,
        table_name=report.table_name,
        markdown=markdown,
        word_count=word_count,
    )
