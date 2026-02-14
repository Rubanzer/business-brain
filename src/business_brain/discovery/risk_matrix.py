"""Risk assessment — likelihood x impact matrix, heatmaps, trends, exposure.

Pure functions for assessing risks via a likelihood-by-impact matrix,
generating heatmaps, tracking risk trends over time, computing financial
exposure, and producing formatted risk reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Convert a value to float safely, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


_TEXT_TO_NUMBER: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "very high": 4,
    "critical": 5,
}


def _parse_level(val) -> int | None:
    """Parse a likelihood or impact value to an integer 1-5.

    Accepts numeric values (int/float clamped to 1-5) or text labels
    such as Low, Medium, High, Very High, Critical.
    """
    if val is None:
        return None

    # Try numeric first
    numeric = _safe_float(val)
    if numeric is not None:
        clamped = max(1, min(5, int(round(numeric))))
        return clamped

    # Try text lookup
    text = str(val).strip().lower()
    return _TEXT_TO_NUMBER.get(text)


def _classify_risk(score: float) -> str:
    """Classify a risk score (likelihood * impact) into a level.

    Score range is 1-25 (from 1x1 to 5x5).
    - Critical: score >= 20
    - High: score >= 12
    - Medium: score >= 6
    - Low: score < 6
    """
    if score >= 20:
        return "Critical"
    if score >= 12:
        return "High"
    if score >= 6:
        return "Medium"
    return "Low"


def _parse_date(val) -> datetime | None:
    """Attempt to parse a date from a string or datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    text = str(val).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RiskItem:
    """A single assessed risk."""

    name: str
    likelihood: int
    impact: int
    risk_score: int
    risk_level: str
    category: str | None
    owner: str | None


@dataclass
class CategoryRiskCount:
    """Risk count breakdown for one category."""

    category: str
    count: int
    avg_score: float


@dataclass
class OwnerRiskCount:
    """Risk count breakdown for one owner."""

    owner: str
    count: int
    critical_count: int


@dataclass
class RiskAssessmentResult:
    """Complete risk assessment result."""

    risks: list[RiskItem]
    total_count: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    by_category: list[CategoryRiskCount]
    by_owner: list[OwnerRiskCount]
    summary: str


@dataclass
class HeatmapCell:
    """A single cell in the risk heatmap."""

    likelihood: int
    impact: int
    count: int
    risk_score: int


@dataclass
class RiskHeatmapResult:
    """Complete risk heatmap result."""

    matrix: list[list[int]]
    hotspots: list[HeatmapCell]
    total_risks: int
    summary: str


@dataclass
class PeriodRiskTrend:
    """Risk trend data for one time period."""

    period: str
    avg_score: float
    max_score: float
    risk_count: int


@dataclass
class RiskTrendResult:
    """Complete risk trend analysis result."""

    periods: list[PeriodRiskTrend]
    trend_direction: str  # "improving", "worsening", "stable"
    avg_score_change: float
    by_category_trend: dict[str, str]
    summary: str


@dataclass
class RiskExposureItem:
    """A single risk's financial exposure."""

    name: str
    impact_value: float
    probability: float
    expected_loss: float


@dataclass
class RiskExposureResult:
    """Complete risk exposure analysis result."""

    total_expected_loss: float
    top_risks: list[RiskExposureItem]
    risk_concentration_pct: float
    avg_expected_loss: float
    max_expected_loss: float
    summary: str


# ---------------------------------------------------------------------------
# 1. assess_risks
# ---------------------------------------------------------------------------


def assess_risks(
    rows: list[dict],
    risk_column: str,
    likelihood_column: str,
    impact_column: str,
    category_column: str | None = None,
    owner_column: str | None = None,
) -> RiskAssessmentResult | None:
    """Assess risks by computing likelihood x impact scores.

    Parses likelihood and impact as integers 1-5 (numeric or text labels),
    computes risk_score = likelihood * impact, and classifies each risk.

    Args:
        rows: Data rows as dicts.
        risk_column: Column identifying the risk name.
        likelihood_column: Column with likelihood value (1-5 or text).
        impact_column: Column with impact value (1-5 or text).
        category_column: Optional column for category breakdown.
        owner_column: Optional column for owner breakdown.

    Returns:
        RiskAssessmentResult or None if insufficient data.
    """
    if not rows:
        return None

    risks: list[RiskItem] = []
    for row in rows:
        name = row.get(risk_column)
        if name is None:
            continue
        likelihood = _parse_level(row.get(likelihood_column))
        impact = _parse_level(row.get(impact_column))
        if likelihood is None or impact is None:
            continue

        score = likelihood * impact
        level = _classify_risk(score)

        category: str | None = None
        if category_column is not None:
            cat_val = row.get(category_column)
            if cat_val is not None:
                category = str(cat_val)

        owner: str | None = None
        if owner_column is not None:
            own_val = row.get(owner_column)
            if own_val is not None:
                owner = str(own_val)

        risks.append(
            RiskItem(
                name=str(name),
                likelihood=likelihood,
                impact=impact,
                risk_score=score,
                risk_level=level,
                category=category,
                owner=owner,
            )
        )

    if not risks:
        return None

    # Sort by risk_score descending
    risks.sort(key=lambda r: r.risk_score, reverse=True)

    critical_count = sum(1 for r in risks if r.risk_level == "Critical")
    high_count = sum(1 for r in risks if r.risk_level == "High")
    medium_count = sum(1 for r in risks if r.risk_level == "Medium")
    low_count = sum(1 for r in risks if r.risk_level == "Low")

    # By category
    by_category: list[CategoryRiskCount] = []
    if category_column is not None:
        cat_map: dict[str, list[int]] = {}
        for r in risks:
            key = r.category or "Uncategorized"
            cat_map.setdefault(key, []).append(r.risk_score)
        for cat, scores in sorted(cat_map.items()):
            avg = sum(scores) / len(scores)
            by_category.append(
                CategoryRiskCount(category=cat, count=len(scores), avg_score=round(avg, 2))
            )

    # By owner
    by_owner: list[OwnerRiskCount] = []
    if owner_column is not None:
        owner_map: dict[str, dict[str, int]] = {}
        for r in risks:
            key = r.owner or "Unassigned"
            if key not in owner_map:
                owner_map[key] = {"count": 0, "critical": 0}
            owner_map[key]["count"] += 1
            if r.risk_level == "Critical":
                owner_map[key]["critical"] += 1
        for own, counts in sorted(owner_map.items()):
            by_owner.append(
                OwnerRiskCount(owner=own, count=counts["count"], critical_count=counts["critical"])
            )

    summary = (
        f"Risk assessment: {len(risks)} risks evaluated. "
        f"Critical: {critical_count}, High: {high_count}, "
        f"Medium: {medium_count}, Low: {low_count}."
    )

    return RiskAssessmentResult(
        risks=risks,
        total_count=len(risks),
        critical_count=critical_count,
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        by_category=by_category,
        by_owner=by_owner,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. compute_risk_heatmap
# ---------------------------------------------------------------------------


def compute_risk_heatmap(
    rows: list[dict],
    likelihood_column: str,
    impact_column: str,
) -> RiskHeatmapResult | None:
    """Create a 5x5 likelihood-by-impact risk heatmap.

    The matrix is indexed as matrix[likelihood-1][impact-1] where both
    likelihood and impact range from 1 to 5.  Text labels are converted
    via Low=1, Medium=2, High=3, Very High=4, Critical=5.

    Args:
        rows: Data rows as dicts.
        likelihood_column: Column with likelihood value (1-5 or text).
        impact_column: Column with impact value (1-5 or text).

    Returns:
        RiskHeatmapResult or None if insufficient data.
    """
    if not rows:
        return None

    # Initialize 5x5 matrix (all zeros)
    matrix: list[list[int]] = [[0 for _ in range(5)] for _ in range(5)]
    total_risks = 0

    for row in rows:
        likelihood = _parse_level(row.get(likelihood_column))
        impact = _parse_level(row.get(impact_column))
        if likelihood is None or impact is None:
            continue
        matrix[likelihood - 1][impact - 1] += 1
        total_risks += 1

    if total_risks == 0:
        return None

    # Build hotspots: cells with count > 0, sorted by risk_score desc
    hotspots: list[HeatmapCell] = []
    for li in range(5):
        for im in range(5):
            if matrix[li][im] > 0:
                hotspots.append(
                    HeatmapCell(
                        likelihood=li + 1,
                        impact=im + 1,
                        count=matrix[li][im],
                        risk_score=(li + 1) * (im + 1),
                    )
                )
    hotspots.sort(key=lambda c: c.risk_score, reverse=True)

    # Identify highest concentration
    top_cell = hotspots[0] if hotspots else None
    top_desc = ""
    if top_cell:
        top_desc = (
            f" Highest risk cell: likelihood={top_cell.likelihood}, "
            f"impact={top_cell.impact} ({top_cell.count} risk(s), score={top_cell.risk_score})."
        )

    summary = f"Risk heatmap: {total_risks} risks mapped to a 5x5 matrix.{top_desc}"

    return RiskHeatmapResult(
        matrix=matrix,
        hotspots=hotspots,
        total_risks=total_risks,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. analyze_risk_trends
# ---------------------------------------------------------------------------


def analyze_risk_trends(
    rows: list[dict],
    risk_column: str,
    score_column: str,
    date_column: str,
    category_column: str | None = None,
) -> RiskTrendResult | None:
    """Analyze risk score trends over time grouped by month.

    Args:
        rows: Data rows as dicts.
        risk_column: Column identifying the risk name.
        score_column: Numeric column with the risk score.
        date_column: Column with date values (for monthly grouping).
        category_column: Optional column for per-category trend.

    Returns:
        RiskTrendResult or None if insufficient data.
    """
    if not rows:
        return None

    # Parse rows into (period, score, category) tuples
    parsed: list[tuple[str, float, str | None]] = []
    for row in rows:
        name = row.get(risk_column)
        score_val = _safe_float(row.get(score_column))
        date_val = _parse_date(row.get(date_column))
        if name is None or score_val is None or date_val is None:
            continue

        period = date_val.strftime("%Y-%m")
        category: str | None = None
        if category_column is not None:
            cat_val = row.get(category_column)
            if cat_val is not None:
                category = str(cat_val)
        parsed.append((period, score_val, category))

    if not parsed:
        return None

    # Group by period
    period_map: dict[str, list[float]] = {}
    for period, score, _cat in parsed:
        period_map.setdefault(period, []).append(score)

    sorted_periods = sorted(period_map.keys())
    period_trends: list[PeriodRiskTrend] = []
    for p in sorted_periods:
        scores = period_map[p]
        avg = sum(scores) / len(scores)
        mx = max(scores)
        period_trends.append(
            PeriodRiskTrend(
                period=p,
                avg_score=round(avg, 2),
                max_score=round(mx, 2),
                risk_count=len(scores),
            )
        )

    # Determine trend direction
    if len(period_trends) >= 2:
        first_avg = period_trends[0].avg_score
        last_avg = period_trends[-1].avg_score
        avg_change = last_avg - first_avg
        if avg_change < -0.5:
            trend_direction = "improving"
        elif avg_change > 0.5:
            trend_direction = "worsening"
        else:
            trend_direction = "stable"
    else:
        avg_change = 0.0
        trend_direction = "stable"

    # By-category trend
    by_category_trend: dict[str, str] = {}
    if category_column is not None:
        cat_period_map: dict[str, dict[str, list[float]]] = {}
        for period, score, cat in parsed:
            key = cat or "Uncategorized"
            cat_period_map.setdefault(key, {}).setdefault(period, []).append(score)

        for cat, pmap in sorted(cat_period_map.items()):
            cat_sorted = sorted(pmap.keys())
            if len(cat_sorted) >= 2:
                first_scores = pmap[cat_sorted[0]]
                last_scores = pmap[cat_sorted[-1]]
                first_avg = sum(first_scores) / len(first_scores)
                last_avg = sum(last_scores) / len(last_scores)
                diff = last_avg - first_avg
                if diff < -0.5:
                    by_category_trend[cat] = "improving"
                elif diff > 0.5:
                    by_category_trend[cat] = "worsening"
                else:
                    by_category_trend[cat] = "stable"
            else:
                by_category_trend[cat] = "stable"

    summary = (
        f"Risk trend analysis over {len(period_trends)} period(s). "
        f"Direction: {trend_direction} (avg score change: {avg_change:+.2f})."
    )

    return RiskTrendResult(
        periods=period_trends,
        trend_direction=trend_direction,
        avg_score_change=round(avg_change, 2),
        by_category_trend=by_category_trend,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. compute_risk_exposure
# ---------------------------------------------------------------------------


def compute_risk_exposure(
    rows: list[dict],
    risk_column: str,
    impact_value_column: str,
    probability_column: str,
) -> RiskExposureResult | None:
    """Compute financial risk exposure (expected loss) for each risk.

    Expected loss = probability * impact_value.

    Args:
        rows: Data rows as dicts.
        risk_column: Column identifying the risk name.
        impact_value_column: Numeric column with the monetary impact value.
        probability_column: Numeric column with the probability (0-1).

    Returns:
        RiskExposureResult or None if insufficient data.
    """
    if not rows:
        return None

    items: list[RiskExposureItem] = []
    for row in rows:
        name = row.get(risk_column)
        impact_val = _safe_float(row.get(impact_value_column))
        prob_val = _safe_float(row.get(probability_column))
        if name is None or impact_val is None or prob_val is None:
            continue

        expected_loss = prob_val * impact_val
        items.append(
            RiskExposureItem(
                name=str(name),
                impact_value=round(impact_val, 2),
                probability=round(prob_val, 4),
                expected_loss=round(expected_loss, 2),
            )
        )

    if not items:
        return None

    # Sort by expected loss descending
    items.sort(key=lambda i: i.expected_loss, reverse=True)

    total_expected_loss = sum(i.expected_loss for i in items)
    avg_expected_loss = total_expected_loss / len(items)
    max_expected_loss = items[0].expected_loss

    # Top 5 risks
    top_risks = items[:5]

    # Risk concentration: what % of total expected loss comes from top 3
    top_3_loss = sum(i.expected_loss for i in items[:3])
    risk_concentration_pct = (top_3_loss / total_expected_loss * 100) if total_expected_loss > 0 else 0.0

    summary = (
        f"Risk exposure: {len(items)} risks with total expected loss of "
        f"{total_expected_loss:,.2f}. "
        f"Top 3 risks account for {risk_concentration_pct:.1f}% of total exposure. "
        f"Max single risk exposure: {max_expected_loss:,.2f}."
    )

    return RiskExposureResult(
        total_expected_loss=round(total_expected_loss, 2),
        top_risks=top_risks,
        risk_concentration_pct=round(risk_concentration_pct, 2),
        avg_expected_loss=round(avg_expected_loss, 2),
        max_expected_loss=round(max_expected_loss, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_risk_report
# ---------------------------------------------------------------------------


def format_risk_report(
    assessment: RiskAssessmentResult | None = None,
    heatmap: RiskHeatmapResult | None = None,
    trends: RiskTrendResult | None = None,
    exposure: RiskExposureResult | None = None,
) -> str:
    """Generate a combined text risk report from available analyses.

    Each section is only included if the corresponding parameter is not None.

    Args:
        assessment: Risk assessment result.
        heatmap: Risk heatmap result.
        trends: Risk trend result.
        exposure: Risk exposure result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Risk Assessment Report")
    sections.append("=" * 40)

    if assessment is not None:
        lines = ["", "Risk Assessment", "-" * 38]
        lines.append(f"  Total risks: {assessment.total_count}")
        lines.append(
            f"  Critical: {assessment.critical_count}, High: {assessment.high_count}, "
            f"Medium: {assessment.medium_count}, Low: {assessment.low_count}"
        )
        for r in assessment.risks:
            cat_str = f" [{r.category}]" if r.category else ""
            own_str = f" (owner: {r.owner})" if r.owner else ""
            lines.append(
                f"  - {r.name}: L={r.likelihood} I={r.impact} "
                f"score={r.risk_score} [{r.risk_level}]{cat_str}{own_str}"
            )
        if assessment.by_category:
            lines.append("  By category:")
            for c in assessment.by_category:
                lines.append(f"    {c.category}: {c.count} risks, avg score={c.avg_score}")
        if assessment.by_owner:
            lines.append("  By owner:")
            for o in assessment.by_owner:
                lines.append(
                    f"    {o.owner}: {o.count} risks, {o.critical_count} critical"
                )
        sections.append("\n".join(lines))

    if heatmap is not None:
        lines = ["", "Risk Heatmap (5x5)", "-" * 38]
        lines.append("  Impact ->  1    2    3    4    5")
        for li in range(5):
            row_vals = "  ".join(f"{heatmap.matrix[li][im]:3d}" for im in range(5))
            lines.append(f"  L={li + 1}:    {row_vals}")
        lines.append(f"  Total risks mapped: {heatmap.total_risks}")
        if heatmap.hotspots:
            lines.append("  Hotspots:")
            for h in heatmap.hotspots[:5]:
                lines.append(
                    f"    L={h.likelihood} I={h.impact}: "
                    f"{h.count} risk(s), score={h.risk_score}"
                )
        sections.append("\n".join(lines))

    if trends is not None:
        lines = ["", "Risk Trends", "-" * 38]
        lines.append(f"  Direction: {trends.trend_direction}")
        lines.append(f"  Avg score change: {trends.avg_score_change:+.2f}")
        for p in trends.periods:
            lines.append(
                f"  {p.period}: avg={p.avg_score}, max={p.max_score}, count={p.risk_count}"
            )
        if trends.by_category_trend:
            lines.append("  By category:")
            for cat, direction in sorted(trends.by_category_trend.items()):
                lines.append(f"    {cat}: {direction}")
        sections.append("\n".join(lines))

    if exposure is not None:
        lines = ["", "Risk Exposure", "-" * 38]
        lines.append(f"  Total expected loss: {exposure.total_expected_loss:,.2f}")
        lines.append(f"  Average expected loss: {exposure.avg_expected_loss:,.2f}")
        lines.append(f"  Maximum expected loss: {exposure.max_expected_loss:,.2f}")
        lines.append(f"  Top-3 concentration: {exposure.risk_concentration_pct:.1f}%")
        lines.append("  Top risks:")
        for r in exposure.top_risks:
            lines.append(
                f"    {r.name}: impact={r.impact_value:,.2f} "
                f"prob={r.probability:.2%} expected_loss={r.expected_loss:,.2f}"
            )
        sections.append("\n".join(lines))

    if assessment is None and heatmap is None and trends is None and exposure is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
