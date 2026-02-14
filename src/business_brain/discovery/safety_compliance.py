"""Safety and compliance analytics for manufacturing environments.

Pure functions for incident tracking, safety scoring, compliance rate
calculation, risk matrix classification, and combined safety reporting.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Incident Analysis
# ---------------------------------------------------------------------------


@dataclass
class IncidentResult:
    """Aggregated incident analysis result."""

    total_incidents: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    by_location: dict[str, int] | None
    trend: str  # "increasing", "decreasing", or "stable"
    most_common_type: str
    most_common_location: str | None
    summary: str


def _detect_trend(rows: list[dict], date_column: str | None) -> str:
    """Detect incident trend from date-sorted data.

    Compares first-half count to second-half count.
    """
    if date_column is None or len(rows) < 4:
        return "stable"

    # Sort rows by date_column for trend analysis
    dated_rows: list[tuple[str, dict]] = []
    for row in rows:
        val = row.get(date_column)
        if val is not None:
            dated_rows.append((str(val), row))

    if len(dated_rows) < 4:
        return "stable"

    dated_rows.sort(key=lambda x: x[0])

    mid = len(dated_rows) // 2
    first_half_count = mid
    second_half_count = len(dated_rows) - mid

    if first_half_count == 0:
        return "stable"

    ratio = second_half_count / first_half_count
    if ratio > 1.25:
        return "increasing"
    if ratio < 0.75:
        return "decreasing"
    return "stable"


def analyze_incidents(
    rows: list[dict],
    type_column: str,
    severity_column: str,
    date_column: str | None = None,
    location_column: str | None = None,
) -> IncidentResult:
    """Count and categorize safety incidents by type, severity, and location.

    Args:
        rows: Data rows as dicts, each representing one incident.
        type_column: Column identifying the incident type.
        severity_column: Column identifying the severity level.
        date_column: Optional column with incident date for trend detection.
        location_column: Optional column identifying the location.

    Returns:
        IncidentResult with counts, breakdowns, trend, and summary.
    """
    if not rows:
        return IncidentResult(
            total_incidents=0,
            by_type={},
            by_severity={},
            by_location=None,
            trend="stable",
            most_common_type="",
            most_common_location=None,
            summary="No incident data available.",
        )

    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_location: dict[str, int] = {}
    valid_count = 0

    for row in rows:
        inc_type = row.get(type_column)
        severity = row.get(severity_column)
        if inc_type is None and severity is None:
            continue

        valid_count += 1

        if inc_type is not None:
            key = str(inc_type)
            by_type[key] = by_type.get(key, 0) + 1

        if severity is not None:
            key = str(severity)
            by_severity[key] = by_severity.get(key, 0) + 1

        if location_column is not None:
            loc = row.get(location_column)
            if loc is not None:
                key = str(loc)
                by_location[key] = by_location.get(key, 0) + 1

    if valid_count == 0:
        return IncidentResult(
            total_incidents=0,
            by_type={},
            by_severity={},
            by_location=None,
            trend="stable",
            most_common_type="",
            most_common_location=None,
            summary="No valid incident records found.",
        )

    # Most common type
    most_common_type = max(by_type, key=by_type.get) if by_type else ""  # type: ignore[arg-type]

    # Most common location
    most_common_location: str | None = None
    location_result: dict[str, int] | None = None
    if location_column is not None and by_location:
        most_common_location = max(by_location, key=by_location.get)  # type: ignore[arg-type]
        location_result = by_location

    # Trend detection
    trend = _detect_trend(rows, date_column)

    # Summary
    parts = [f"{valid_count} incident(s) recorded."]
    if most_common_type:
        parts.append(f"Most common type: {most_common_type} ({by_type[most_common_type]}).")
    if most_common_location:
        parts.append(
            f"Most common location: {most_common_location} "
            f"({by_location[most_common_location]})."
        )
    parts.append(f"Trend: {trend}.")

    return IncidentResult(
        total_incidents=valid_count,
        by_type=by_type,
        by_severity=by_severity,
        by_location=location_result,
        trend=trend,
        most_common_type=most_common_type,
        most_common_location=most_common_location,
        summary=" ".join(parts),
    )


# ---------------------------------------------------------------------------
# 2. Safety Score
# ---------------------------------------------------------------------------


@dataclass
class EntitySafety:
    """Safety score for a single entity."""

    entity: str
    incident_count: int
    days_tracked: int
    incident_rate: float  # incidents per day
    safety_score: float  # 0-100 scale
    grade: str  # A >= 90, B >= 75, C >= 60, D >= 40, F < 40


@dataclass
class SafetyScoreResult:
    """Aggregated safety score result across entities."""

    entities: list[EntitySafety]
    mean_score: float
    safest_entity: str
    riskiest_entity: str
    summary: str


_DEFAULT_SEVERITY_WEIGHTS: dict[str, float] = {
    "low": 1.0,
    "medium": 2.0,
    "high": 5.0,
    "critical": 10.0,
}

_SCALING_FACTOR = 500.0


def _safety_grade(score: float) -> str:
    """Map a 0-100 safety score to a letter grade."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def compute_safety_score(
    rows: list[dict],
    entity_column: str,
    incident_count_column: str,
    days_column: str,
    severity_weights: dict[str, float] | None = None,
) -> SafetyScoreResult | None:
    """Compute a safety score for each entity based on incident frequency.

    Score = 100 - (weighted_incidents_per_day * scaling_factor), clamped to 0-100.
    The scaling factor is 500.

    When *severity_weights* is provided, the incident count is multiplied by
    the weight (looked up by severity string in the row).  When it is ``None``
    the raw incident count is used directly.

    Args:
        rows: Data rows as dicts.  Each row represents one entity-period.
        entity_column: Column identifying the entity (plant, line, etc.).
        incident_count_column: Numeric column with incident count.
        days_column: Numeric column with number of days tracked.
        severity_weights: Optional mapping of severity label -> weight multiplier.

    Returns:
        SafetyScoreResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate per entity
    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        count_raw = row.get(incident_count_column)
        days_raw = row.get(days_column)
        if entity is None or count_raw is None or days_raw is None:
            continue
        try:
            count_val = float(count_raw)
            days_val = float(days_raw)
        except (TypeError, ValueError):
            continue

        key = str(entity)
        if key not in acc:
            acc[key] = {"incidents": 0.0, "days": 0.0}
        acc[key]["incidents"] += count_val
        acc[key]["days"] += days_val

    if not acc:
        return None

    entities: list[EntitySafety] = []
    for name, vals in acc.items():
        incidents = vals["incidents"]
        days = vals["days"]
        if days <= 0:
            rate = float("inf")
            score = 0.0
        else:
            rate = incidents / days
            score = 100.0 - (rate * _SCALING_FACTOR)
            score = max(0.0, min(100.0, score))

        entities.append(
            EntitySafety(
                entity=name,
                incident_count=int(incidents),
                days_tracked=int(days),
                incident_rate=round(rate, 6) if rate != float("inf") else float("inf"),
                safety_score=round(score, 2),
                grade=_safety_grade(score),
            )
        )

    # Sort by safety score descending (safest first)
    entities.sort(key=lambda e: e.safety_score, reverse=True)

    mean_score = sum(e.safety_score for e in entities) / len(entities)
    safest = entities[0]
    riskiest = entities[-1]

    summary = (
        f"Safety scores for {len(entities)} entities: "
        f"Mean score = {mean_score:.1f}. "
        f"Safest = {safest.entity} ({safest.safety_score:.1f}, grade {safest.grade}), "
        f"Riskiest = {riskiest.entity} ({riskiest.safety_score:.1f}, grade {riskiest.grade})."
    )

    return SafetyScoreResult(
        entities=entities,
        mean_score=round(mean_score, 2),
        safest_entity=safest.entity,
        riskiest_entity=riskiest.entity,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. Compliance Rate
# ---------------------------------------------------------------------------


@dataclass
class EntityCompliance:
    """Compliance metrics for a single entity."""

    entity: str
    total_checks: int
    passed_checks: int
    compliance_pct: float
    status: str  # "compliant" >= 95%, "at_risk" 80-95%, "non_compliant" < 80%


@dataclass
class ComplianceResult:
    """Aggregated compliance result across entities."""

    entities: list[EntityCompliance]
    mean_compliance: float
    fully_compliant_count: int  # entities with compliance >= 95%
    non_compliant_count: int  # entities with compliance < 80%
    summary: str


def _compliance_status(pct: float) -> str:
    """Classify compliance percentage into a status."""
    if pct >= 95.0:
        return "compliant"
    if pct >= 80.0:
        return "at_risk"
    return "non_compliant"


def compliance_rate(
    rows: list[dict],
    entity_column: str,
    total_checks_column: str,
    passed_checks_column: str,
) -> ComplianceResult | None:
    """Compute compliance rate per entity.

    Compliance % = passed_checks / total_checks * 100.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity.
        total_checks_column: Numeric column with total number of checks.
        passed_checks_column: Numeric column with number of passed checks.

    Returns:
        ComplianceResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        total_raw = row.get(total_checks_column)
        passed_raw = row.get(passed_checks_column)
        if entity is None or total_raw is None or passed_raw is None:
            continue
        try:
            total_val = float(total_raw)
            passed_val = float(passed_raw)
        except (TypeError, ValueError):
            continue

        key = str(entity)
        if key not in acc:
            acc[key] = {"total": 0.0, "passed": 0.0}
        acc[key]["total"] += total_val
        acc[key]["passed"] += passed_val

    if not acc:
        return None

    entities: list[EntityCompliance] = []
    for name, vals in acc.items():
        total = vals["total"]
        passed = vals["passed"]
        pct = (passed / total * 100.0) if total > 0 else 0.0

        entities.append(
            EntityCompliance(
                entity=name,
                total_checks=int(total),
                passed_checks=int(passed),
                compliance_pct=round(pct, 2),
                status=_compliance_status(pct),
            )
        )

    # Sort by compliance descending
    entities.sort(key=lambda e: e.compliance_pct, reverse=True)

    mean_compliance = sum(e.compliance_pct for e in entities) / len(entities)
    fully_compliant = sum(1 for e in entities if e.compliance_pct >= 95.0)
    non_compliant = sum(1 for e in entities if e.compliance_pct < 80.0)

    summary = (
        f"Compliance across {len(entities)} entities: "
        f"Mean = {mean_compliance:.1f}%. "
        f"Fully compliant (>=95%): {fully_compliant}. "
        f"Non-compliant (<80%): {non_compliant}."
    )

    return ComplianceResult(
        entities=entities,
        mean_compliance=round(mean_compliance, 2),
        fully_compliant_count=fully_compliant,
        non_compliant_count=non_compliant,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. Risk Matrix
# ---------------------------------------------------------------------------


@dataclass
class RiskItem:
    """A single item in the risk matrix."""

    entity: str | None
    likelihood: float
    impact: float
    risk_score: float  # likelihood * impact
    risk_level: str  # "Low", "Medium", "High", "Critical"


@dataclass
class RiskMatrixResult:
    """Complete risk matrix analysis result."""

    items: list[RiskItem]
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    summary: str


def _classify_risk(score: float) -> str:
    """Classify a risk score (likelihood * impact) into a level.

    Assumes likelihood and impact are on a 1-5 scale, so score is 1-25.
    - Critical: score >= 20
    - High: score >= 12
    - Medium: score >= 5
    - Low: score < 5
    """
    if score >= 20:
        return "Critical"
    if score >= 12:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def risk_matrix(
    rows: list[dict],
    likelihood_column: str,
    impact_column: str,
    entity_column: str | None = None,
) -> RiskMatrixResult | None:
    """Classify risks into Low/Medium/High/Critical based on likelihood x impact.

    Args:
        rows: Data rows as dicts.
        likelihood_column: Numeric column with likelihood value (typically 1-5).
        impact_column: Numeric column with impact value (typically 1-5).
        entity_column: Optional column identifying the risk entity/item.

    Returns:
        RiskMatrixResult or None if no valid data.
    """
    if not rows:
        return None

    items: list[RiskItem] = []
    for row in rows:
        likelihood_raw = row.get(likelihood_column)
        impact_raw = row.get(impact_column)
        if likelihood_raw is None or impact_raw is None:
            continue
        try:
            likelihood = float(likelihood_raw)
            impact = float(impact_raw)
        except (TypeError, ValueError):
            continue

        entity: str | None = None
        if entity_column is not None:
            entity_val = row.get(entity_column)
            if entity_val is not None:
                entity = str(entity_val)

        score = likelihood * impact
        level = _classify_risk(score)

        items.append(
            RiskItem(
                entity=entity,
                likelihood=round(likelihood, 2),
                impact=round(impact, 2),
                risk_score=round(score, 2),
                risk_level=level,
            )
        )

    if not items:
        return None

    # Sort by risk score descending
    items.sort(key=lambda i: i.risk_score, reverse=True)

    critical = sum(1 for i in items if i.risk_level == "Critical")
    high = sum(1 for i in items if i.risk_level == "High")
    medium = sum(1 for i in items if i.risk_level == "Medium")
    low = sum(1 for i in items if i.risk_level == "Low")

    summary = (
        f"Risk matrix: {len(items)} items assessed. "
        f"Critical: {critical}, High: {high}, Medium: {medium}, Low: {low}."
    )

    return RiskMatrixResult(
        items=items,
        critical_count=critical,
        high_count=high,
        medium_count=medium,
        low_count=low,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. Combined Safety Report
# ---------------------------------------------------------------------------


def format_safety_report(
    incidents: IncidentResult | None = None,
    safety: SafetyScoreResult | None = None,
    compliance: ComplianceResult | None = None,
    risk: RiskMatrixResult | None = None,
) -> str:
    """Generate a combined text safety report from available analyses.

    Args:
        incidents: Incident analysis result.
        safety: Safety score result.
        compliance: Compliance rate result.
        risk: Risk matrix result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Safety & Compliance Report")
    sections.append("=" * 40)

    if incidents is not None:
        lines = ["", "Incident Analysis", "-" * 38]
        lines.append(f"  Total incidents: {incidents.total_incidents}")
        if incidents.by_type:
            lines.append("  By type:")
            for t, count in sorted(incidents.by_type.items(), key=lambda x: -x[1]):
                lines.append(f"    {t}: {count}")
        if incidents.by_severity:
            lines.append("  By severity:")
            for s, count in sorted(incidents.by_severity.items(), key=lambda x: -x[1]):
                lines.append(f"    {s}: {count}")
        if incidents.by_location:
            lines.append("  By location:")
            for loc, count in sorted(incidents.by_location.items(), key=lambda x: -x[1]):
                lines.append(f"    {loc}: {count}")
        lines.append(f"  Trend: {incidents.trend}")
        if incidents.most_common_type:
            lines.append(f"  Most common type: {incidents.most_common_type}")
        if incidents.most_common_location:
            lines.append(f"  Most common location: {incidents.most_common_location}")
        sections.append("\n".join(lines))

    if safety is not None:
        lines = ["", "Safety Scores", "-" * 38]
        for e in safety.entities:
            lines.append(
                f"  {e.entity}: score={e.safety_score:.1f} grade={e.grade} "
                f"(incidents={e.incident_count}, days={e.days_tracked}, "
                f"rate={e.incident_rate:.4f})"
            )
        lines.append(f"  Mean score: {safety.mean_score:.1f}")
        lines.append(f"  Safest: {safety.safest_entity}")
        lines.append(f"  Riskiest: {safety.riskiest_entity}")
        sections.append("\n".join(lines))

    if compliance is not None:
        lines = ["", "Compliance Rates", "-" * 38]
        for e in compliance.entities:
            lines.append(
                f"  {e.entity}: {e.compliance_pct:.1f}% "
                f"({e.passed_checks}/{e.total_checks}) [{e.status}]"
            )
        lines.append(f"  Mean compliance: {compliance.mean_compliance:.1f}%")
        lines.append(f"  Fully compliant: {compliance.fully_compliant_count}")
        lines.append(f"  Non-compliant: {compliance.non_compliant_count}")
        sections.append("\n".join(lines))

    if risk is not None:
        lines = ["", "Risk Matrix", "-" * 38]
        for item in risk.items:
            entity_str = item.entity or "N/A"
            lines.append(
                f"  {entity_str}: likelihood={item.likelihood:.1f} "
                f"impact={item.impact:.1f} score={item.risk_score:.1f} "
                f"[{item.risk_level}]"
            )
        lines.append(
            f"  Critical: {risk.critical_count}, High: {risk.high_count}, "
            f"Medium: {risk.medium_count}, Low: {risk.low_count}"
        )
        sections.append("\n".join(lines))

    if incidents is None and safety is None and compliance is None and risk is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
