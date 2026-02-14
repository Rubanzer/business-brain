"""Profile report generator — generates comprehensive text summaries of table profiles.

Pure functions that take profile data and produce human-readable reports
with column analysis, quality assessment, and actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileReport:
    """Complete profile report for a table."""
    table_name: str
    row_count: int
    column_count: int
    domain: str
    quality_score: float
    sections: list[ReportSection]
    recommendations: list[str]
    summary: str


@dataclass
class ReportSection:
    """A section of the profile report."""
    title: str
    content: str
    severity: str = "info"  # info, warning, critical


def generate_profile_report(
    table_name: str,
    row_count: int,
    columns: dict[str, dict],
    domain: str = "general",
    relationships: list[dict] | None = None,
) -> ProfileReport:
    """Generate a comprehensive profile report for a table.

    Args:
        table_name: Name of the table.
        row_count: Number of rows.
        columns: Column classification data (name -> info dict).
        domain: Domain hint (e.g., "procurement", "hr").
        relationships: Known relationships with other tables.

    Returns:
        ProfileReport with sections and recommendations.
    """
    sections = []
    recommendations = []

    # 1. Overview section
    sections.append(_build_overview(table_name, row_count, columns, domain))

    # 2. Column type breakdown
    sections.append(_build_type_breakdown(columns))

    # 3. Data quality section
    quality_section, quality_score = _build_quality_section(columns, row_count)
    sections.append(quality_section)

    # 4. Key columns (identifiers, temporal)
    key_section = _build_key_columns(columns)
    if key_section:
        sections.append(key_section)

    # 5. Relationships
    if relationships:
        sections.append(_build_relationships_section(relationships, table_name))

    # 6. Recommendations
    recommendations = _generate_recommendations(columns, row_count, domain, relationships)

    # Summary
    summary = _build_summary(table_name, row_count, len(columns), quality_score, domain)

    return ProfileReport(
        table_name=table_name,
        row_count=row_count,
        column_count=len(columns),
        domain=domain,
        quality_score=round(quality_score, 1),
        sections=sections,
        recommendations=recommendations,
        summary=summary,
    )


def format_report_text(report: ProfileReport) -> str:
    """Format a ProfileReport as plain text."""
    lines = [
        f"=== Profile Report: {report.table_name} ===",
        f"Rows: {report.row_count:,} | Columns: {report.column_count} | Domain: {report.domain}",
        f"Quality Score: {report.quality_score}/100",
        "",
    ]

    for section in report.sections:
        indicator = {"critical": "[!]", "warning": "[~]", "info": ""}.get(section.severity, "")
        lines.append(f"--- {section.title} {indicator} ---")
        lines.append(section.content)
        lines.append("")

    if report.recommendations:
        lines.append("--- Recommendations ---")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    lines.append(f"Summary: {report.summary}")
    return "\n".join(lines)


def compute_report_priority(report: ProfileReport) -> int:
    """Compute a priority score for the report (0-100).

    Higher score means more issues needing attention.
    """
    base = 50
    # Quality penalty
    if report.quality_score < 50:
        base += 30
    elif report.quality_score < 70:
        base += 15

    # Critical sections
    critical_count = sum(1 for s in report.sections if s.severity == "critical")
    warning_count = sum(1 for s in report.sections if s.severity == "warning")
    base += critical_count * 15 + warning_count * 5

    # Recommendation count
    base += min(len(report.recommendations) * 3, 20)

    return min(base, 100)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_overview(
    table_name: str,
    row_count: int,
    columns: dict,
    domain: str,
) -> ReportSection:
    """Build the overview section."""
    size_label = "small" if row_count < 100 else "medium" if row_count < 10000 else "large"
    content = (
        f"Table '{table_name}' contains {row_count:,} rows across {len(columns)} columns. "
        f"Dataset size: {size_label}. Domain: {domain}."
    )
    return ReportSection(title="Overview", content=content)


def _build_type_breakdown(columns: dict) -> ReportSection:
    """Build column type distribution section."""
    type_counts: dict[str, int] = {}
    for info in columns.values():
        stype = info.get("semantic_type", "unknown")
        type_counts[stype] = type_counts.get(stype, 0) + 1

    if not type_counts:
        return ReportSection(title="Column Types", content="No column type information available.")

    lines = []
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(columns) * 100
        lines.append(f"  {stype}: {count} ({pct:.0f}%)")

    content = "\n".join(lines)
    return ReportSection(title="Column Types", content=content)


def _build_quality_section(columns: dict, row_count: int) -> tuple[ReportSection, float]:
    """Build data quality section and return quality score."""
    if not columns or row_count == 0:
        return ReportSection(title="Data Quality", content="No data for quality assessment.", severity="warning"), 0.0

    total_nulls = 0
    total_cells = 0
    high_null_cols = []
    constant_cols = []

    for name, info in columns.items():
        null_count = info.get("null_count", 0) or 0
        total_nulls += null_count
        total_cells += row_count

        null_pct = (null_count / row_count * 100) if row_count > 0 else 0
        if null_pct > 20:
            high_null_cols.append((name, null_pct))

        cardinality = info.get("cardinality", 0) or 0
        if cardinality <= 1 and row_count > 10:
            constant_cols.append(name)

    completeness = (1 - total_nulls / total_cells) * 100 if total_cells > 0 else 100
    quality_score = completeness

    lines = [f"Overall completeness: {completeness:.1f}%"]

    severity = "info"
    if high_null_cols:
        lines.append(f"Columns with high null rates (>20%):")
        for col, pct in sorted(high_null_cols, key=lambda x: -x[1]):
            lines.append(f"  - {col}: {pct:.1f}% null")
        severity = "warning"

    if constant_cols:
        lines.append(f"Constant columns (cardinality <= 1): {', '.join(constant_cols)}")
        quality_score -= 5 * len(constant_cols)
        severity = "warning"

    if completeness < 70:
        severity = "critical"

    return ReportSection(title="Data Quality", content="\n".join(lines), severity=severity), max(quality_score, 0)


def _build_key_columns(columns: dict) -> ReportSection | None:
    """Build key columns section (identifiers, temporal)."""
    identifiers = []
    temporals = []
    metrics = []

    for name, info in columns.items():
        stype = info.get("semantic_type", "")
        if stype == "identifier":
            identifiers.append(name)
        elif stype == "temporal":
            temporals.append(name)
        elif stype.startswith("numeric"):
            metrics.append(name)

    if not identifiers and not temporals:
        return None

    lines = []
    if identifiers:
        lines.append(f"Identifiers: {', '.join(identifiers)}")
    if temporals:
        lines.append(f"Temporal columns: {', '.join(temporals)}")
    if metrics:
        lines.append(f"Numeric metrics: {', '.join(metrics)}")

    return ReportSection(title="Key Columns", content="\n".join(lines))


def _build_relationships_section(
    relationships: list[dict],
    table_name: str,
) -> ReportSection:
    """Build relationships section."""
    relevant = [r for r in relationships if r.get("table_a") == table_name or r.get("table_b") == table_name]

    if not relevant:
        return ReportSection(title="Relationships", content="No known relationships.")

    lines = []
    for r in relevant:
        other = r["table_b"] if r.get("table_a") == table_name else r.get("table_a", "?")
        col_a = r.get("column_a", "?")
        col_b = r.get("column_b", "?")
        rtype = r.get("relationship_type", "unknown")
        conf = r.get("confidence", 0)
        lines.append(f"  {col_a} -> {other}.{col_b} ({rtype}, confidence: {conf:.0%})")

    return ReportSection(title="Relationships", content="\n".join(lines))


def _generate_recommendations(
    columns: dict,
    row_count: int,
    domain: str,
    relationships: list[dict] | None,
) -> list[str]:
    """Generate actionable recommendations based on profile data."""
    recs = []

    # Check for missing identifiers
    has_identifier = any(info.get("semantic_type") == "identifier" for info in columns.values())
    if not has_identifier:
        recs.append("No identifier column detected. Consider adding a primary key for joins.")

    # Check for missing temporal
    has_temporal = any(info.get("semantic_type") == "temporal" for info in columns.values())
    if not has_temporal and row_count > 50:
        recs.append("No date/time column found. Adding temporal data would enable trend analysis.")

    # Check for high null columns
    for name, info in columns.items():
        null_count = info.get("null_count", 0) or 0
        if row_count > 0 and null_count / row_count > 0.5:
            recs.append(f"Column '{name}' is >50% null. Consider cleaning or removing it.")

    # Check for low cardinality identifiers (potential data issue)
    for name, info in columns.items():
        if info.get("semantic_type") == "identifier":
            card = info.get("cardinality", 0) or 0
            if row_count > 0 and card < row_count * 0.5:
                recs.append(f"Identifier '{name}' has low cardinality ({card}/{row_count}). May contain duplicates.")

    # Relationship suggestions
    if not relationships:
        if has_identifier:
            recs.append("No relationships discovered. Upload related tables to enable cross-table analysis.")

    # Small dataset warning
    if row_count < 30:
        recs.append(f"Small dataset ({row_count} rows). Statistical analyses may be unreliable.")

    return recs


def _build_summary(
    table_name: str,
    row_count: int,
    col_count: int,
    quality_score: float,
    domain: str,
) -> str:
    """Build one-line summary."""
    quality_label = "excellent" if quality_score >= 85 else "good" if quality_score >= 70 else "fair" if quality_score >= 50 else "poor"
    return (
        f"{table_name}: {row_count:,} rows, {col_count} columns, "
        f"{quality_label} quality ({quality_score:.0f}/100), domain: {domain}."
    )
