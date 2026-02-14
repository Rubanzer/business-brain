"""Supplier scorecard — comprehensive procurement scorecards for manufacturing.

Pure functions for building weighted supplier scorecards, head-to-head
comparisons, risk detection, and concentration analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricScore:
    """One metric's contribution to a supplier's overall score."""
    metric_name: str
    raw_value: float
    normalized: float
    weight: float
    contribution: float


@dataclass
class SupplierScore:
    """A single supplier with its scorecard result."""
    supplier: str
    score: float
    grade: str
    rank: int
    metric_scores: list[MetricScore]
    strengths: list[str]
    weaknesses: list[str]


@dataclass
class ScorecardResult:
    """Complete supplier scorecard result."""
    suppliers: list[SupplierScore]
    supplier_count: int
    mean_score: float
    best_supplier: str
    worst_supplier: str
    grade_distribution: dict[str, int]
    summary: str


@dataclass
class MetricComparison:
    """Head-to-head comparison for one metric."""
    metric: str
    value_a: float
    value_b: float
    winner: str
    difference_pct: float


@dataclass
class ComparisonResult:
    """Result of comparing two suppliers."""
    supplier_a: str
    supplier_b: str
    score_a: float
    score_b: float
    winner: str
    metric_comparisons: list[MetricComparison]
    advantages_a: list[str]
    advantages_b: list[str]


@dataclass
class SupplierRisk:
    """A detected risk for a supplier."""
    supplier: str
    risk_type: str  # "declining", "below_threshold", "single_source"
    severity: str  # "low", "medium", "high"
    description: str
    affected_metric: str


@dataclass
class ConcentrationResult:
    """Supplier concentration analysis result."""
    hhi: float
    concentration_level: str  # "low", "moderate", "high"
    top_supplier_share: float
    suppliers: list[dict]
    summary: str


# ---------------------------------------------------------------------------
# Grade helpers
# ---------------------------------------------------------------------------


def _score_to_grade(score: float) -> str:
    """Map a 0-100 score to a letter grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# 1. build_scorecard
# ---------------------------------------------------------------------------


def build_scorecard(
    rows: list[dict],
    supplier_column: str,
    metrics: list[dict],
) -> ScorecardResult | None:
    """Build weighted scorecards for each supplier.

    Args:
        rows: Data rows as dicts.
        supplier_column: Column identifying the supplier.
        metrics: List of metric definitions, each a dict with keys:
            - column: str — column name in the data
            - weight: float — relative weight (all weights are normalised internally)
            - direction: str — "higher_is_better" or "lower_is_better"

    Returns:
        ScorecardResult or None if insufficient data.
    """
    if not rows or not metrics:
        return None

    # --- Normalise weights so they sum to 1 --------------------------------
    total_weight = sum(m["weight"] for m in metrics)
    if total_weight == 0:
        return None
    norm_weights = [m["weight"] / total_weight for m in metrics]

    # --- Aggregate raw values per supplier per metric (mean) ---------------
    supplier_metric_vals: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        supplier = row.get(supplier_column)
        if supplier is None:
            continue
        supplier_key = str(supplier)
        if supplier_key not in supplier_metric_vals:
            supplier_metric_vals[supplier_key] = {}
        for m in metrics:
            col = m["column"]
            val = row.get(col)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            supplier_metric_vals[supplier_key].setdefault(col, []).append(fval)

    if not supplier_metric_vals:
        return None

    # Compute mean per supplier per metric
    supplier_means: dict[str, dict[str, float]] = {}
    for sup, metric_vals in supplier_metric_vals.items():
        supplier_means[sup] = {}
        for m in metrics:
            col = m["column"]
            vals = metric_vals.get(col, [])
            if vals:
                supplier_means[sup][col] = sum(vals) / len(vals)

    # We need at least 1 supplier with data
    if not supplier_means:
        return None

    # --- Min-max normalisation per metric ----------------------------------
    metric_min: dict[str, float] = {}
    metric_max: dict[str, float] = {}
    for m in metrics:
        col = m["column"]
        values = [
            supplier_means[s][col]
            for s in supplier_means
            if col in supplier_means[s]
        ]
        if values:
            metric_min[col] = min(values)
            metric_max[col] = max(values)
        else:
            metric_min[col] = 0
            metric_max[col] = 0

    # --- Score each supplier -----------------------------------------------
    supplier_scores: list[SupplierScore] = []

    for sup in sorted(supplier_means.keys()):
        ms_list: list[MetricScore] = []
        weighted_sum = 0.0

        for i, m in enumerate(metrics):
            col = m["column"]
            direction = m.get("direction", "higher_is_better")
            weight = norm_weights[i]

            raw = supplier_means[sup].get(col)
            if raw is None:
                # Missing metric — contribute 0
                ms_list.append(MetricScore(
                    metric_name=col,
                    raw_value=0.0,
                    normalized=0.0,
                    weight=round(weight, 4),
                    contribution=0.0,
                ))
                continue

            mn = metric_min[col]
            mx = metric_max[col]
            if mx == mn:
                normalised = 1.0  # all suppliers equal
            else:
                normalised = (raw - mn) / (mx - mn)

            if direction == "lower_is_better":
                normalised = 1.0 - normalised

            contribution = normalised * weight * 100  # on 0-100 scale
            weighted_sum += contribution

            ms_list.append(MetricScore(
                metric_name=col,
                raw_value=round(raw, 4),
                normalized=round(normalised, 4),
                weight=round(weight, 4),
                contribution=round(contribution, 4),
            ))

        score = round(weighted_sum, 2)
        grade = _score_to_grade(score)

        # Strengths: metrics where normalised >= 0.8
        strengths = [
            ms.metric_name
            for ms in ms_list
            if ms.normalized >= 0.8 and ms.raw_value != 0
        ]
        # Weaknesses: metrics where normalised <= 0.3
        weaknesses = [
            ms.metric_name
            for ms in ms_list
            if ms.normalized <= 0.3 and ms.raw_value != 0
        ]

        supplier_scores.append(SupplierScore(
            supplier=sup,
            score=score,
            grade=grade,
            rank=0,  # filled below
            metric_scores=ms_list,
            strengths=strengths,
            weaknesses=weaknesses,
        ))

    # Sort by score descending and assign ranks
    supplier_scores.sort(key=lambda s: -s.score)
    for idx, ss in enumerate(supplier_scores):
        ss.rank = idx + 1

    # Grade distribution
    grade_dist: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for ss in supplier_scores:
        grade_dist[ss.grade] = grade_dist.get(ss.grade, 0) + 1

    mean_score = round(
        sum(s.score for s in supplier_scores) / len(supplier_scores), 2
    )
    best = supplier_scores[0].supplier
    worst = supplier_scores[-1].supplier

    summary = (
        f"Scorecard for {len(supplier_scores)} suppliers across "
        f"{len(metrics)} metrics. "
        f"Best: {best} ({supplier_scores[0].score}), "
        f"Worst: {worst} ({supplier_scores[-1].score}), "
        f"Mean: {mean_score}."
    )

    return ScorecardResult(
        suppliers=supplier_scores,
        supplier_count=len(supplier_scores),
        mean_score=mean_score,
        best_supplier=best,
        worst_supplier=worst,
        grade_distribution=grade_dist,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. compare_suppliers
# ---------------------------------------------------------------------------


def compare_suppliers(
    result: ScorecardResult,
    supplier_a: str,
    supplier_b: str,
) -> ComparisonResult | None:
    """Head-to-head comparison of two suppliers from a scorecard.

    Args:
        result: A ScorecardResult from build_scorecard.
        supplier_a: Name of first supplier.
        supplier_b: Name of second supplier.

    Returns:
        ComparisonResult or None if either supplier not found.
    """
    score_map: dict[str, SupplierScore] = {
        s.supplier: s for s in result.suppliers
    }
    if supplier_a not in score_map or supplier_b not in score_map:
        return None

    sa = score_map[supplier_a]
    sb = score_map[supplier_b]

    metric_comparisons: list[MetricComparison] = []
    advantages_a: list[str] = []
    advantages_b: list[str] = []

    ms_map_a = {ms.metric_name: ms for ms in sa.metric_scores}
    ms_map_b = {ms.metric_name: ms for ms in sb.metric_scores}

    all_metrics = list(ms_map_a.keys()) or list(ms_map_b.keys())
    for metric in all_metrics:
        ma = ms_map_a.get(metric)
        mb = ms_map_b.get(metric)
        val_a = ma.raw_value if ma else 0.0
        val_b = mb.raw_value if mb else 0.0

        if val_a > val_b:
            winner = supplier_a
            advantages_a.append(metric)
        elif val_b > val_a:
            winner = supplier_b
            advantages_b.append(metric)
        else:
            winner = "tie"

        max_val = max(abs(val_a), abs(val_b))
        diff_pct = (abs(val_a - val_b) / max_val * 100) if max_val != 0 else 0.0

        metric_comparisons.append(MetricComparison(
            metric=metric,
            value_a=round(val_a, 4),
            value_b=round(val_b, 4),
            winner=winner,
            difference_pct=round(diff_pct, 2),
        ))

    overall_winner = (
        supplier_a if sa.score > sb.score
        else supplier_b if sb.score > sa.score
        else "tie"
    )

    return ComparisonResult(
        supplier_a=supplier_a,
        supplier_b=supplier_b,
        score_a=sa.score,
        score_b=sb.score,
        winner=overall_winner,
        metric_comparisons=metric_comparisons,
        advantages_a=advantages_a,
        advantages_b=advantages_b,
    )


# ---------------------------------------------------------------------------
# 3. detect_supplier_risks
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {}


def detect_supplier_risks(
    rows: list[dict],
    supplier_column: str,
    metrics: list[dict],
    thresholds: dict[str, float] | None = None,
) -> list[SupplierRisk]:
    """Detect suppliers with declining metrics or below-threshold performance.

    Risk types detected:
        - **below_threshold**: supplier mean for a metric is below the given
          threshold (or below 25th percentile if no threshold given).
        - **declining**: supplier shows a declining trend across rows
          (requires ordered data; uses simple first-half vs second-half mean).
        - **single_source**: a metric has only one supplier providing data.

    Args:
        rows: Data rows as dicts.
        supplier_column: Column identifying the supplier.
        metrics: Same metric defs as build_scorecard (column, weight, direction).
        thresholds: Optional dict mapping metric column -> minimum acceptable value.

    Returns:
        List of SupplierRisk (may be empty).
    """
    if not rows or not metrics:
        return []

    thresholds = thresholds or {}

    # --- Aggregate values per supplier per metric (preserving order) --------
    supplier_metric_vals: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        supplier = row.get(supplier_column)
        if supplier is None:
            continue
        supplier_key = str(supplier)
        if supplier_key not in supplier_metric_vals:
            supplier_metric_vals[supplier_key] = {}
        for m in metrics:
            col = m["column"]
            val = row.get(col)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            supplier_metric_vals[supplier_key].setdefault(col, []).append(fval)

    if not supplier_metric_vals:
        return []

    risks: list[SupplierRisk] = []

    for m in metrics:
        col = m["column"]
        direction = m.get("direction", "higher_is_better")

        # Gather all supplier means for percentile calc
        all_means: list[float] = []
        for sup_vals in supplier_metric_vals.values():
            vals = sup_vals.get(col, [])
            if vals:
                all_means.append(sum(vals) / len(vals))

        if not all_means:
            continue

        sorted_means = sorted(all_means)
        p25 = sorted_means[max(0, len(sorted_means) // 4)]

        # Single source risk
        suppliers_with_data = [
            s for s, sv in supplier_metric_vals.items() if col in sv and sv[col]
        ]
        if len(suppliers_with_data) == 1:
            risks.append(SupplierRisk(
                supplier=suppliers_with_data[0],
                risk_type="single_source",
                severity="high",
                description=(
                    f"Only one supplier ({suppliers_with_data[0]}) provides "
                    f"data for {col}. Supply chain concentration risk."
                ),
                affected_metric=col,
            ))

        for sup, sup_vals in supplier_metric_vals.items():
            vals = sup_vals.get(col, [])
            if not vals:
                continue

            mean_val = sum(vals) / len(vals)

            # Below-threshold check
            threshold = thresholds.get(col)
            if threshold is not None:
                is_below = (
                    mean_val < threshold
                    if direction == "higher_is_better"
                    else mean_val > threshold
                )
                if is_below:
                    risks.append(SupplierRisk(
                        supplier=sup,
                        risk_type="below_threshold",
                        severity="high" if direction == "higher_is_better" and mean_val < threshold * 0.8 else "medium",
                        description=(
                            f"{sup} has {col} = {mean_val:.2f}, "
                            f"below threshold {threshold:.2f}."
                        ),
                        affected_metric=col,
                    ))
            else:
                # Use 25th percentile as implicit threshold
                is_below = (
                    mean_val <= p25
                    if direction == "higher_is_better"
                    else mean_val >= sorted_means[-(len(sorted_means) // 4 + 1)]
                )
                # Only flag if there are multiple suppliers and this one is worst
                if is_below and len(all_means) > 1 and mean_val == min(all_means):
                    risks.append(SupplierRisk(
                        supplier=sup,
                        risk_type="below_threshold",
                        severity="medium",
                        description=(
                            f"{sup} has {col} = {mean_val:.2f}, "
                            f"at the bottom of all suppliers."
                        ),
                        affected_metric=col,
                    ))

            # Declining trend check (first half vs second half)
            if len(vals) >= 4:
                mid = len(vals) // 2
                first_half_mean = sum(vals[:mid]) / mid
                second_half_mean = sum(vals[mid:]) / (len(vals) - mid)

                if direction == "higher_is_better":
                    is_declining = second_half_mean < first_half_mean * 0.9
                else:
                    is_declining = second_half_mean > first_half_mean * 1.1

                if is_declining:
                    change_pct = abs(second_half_mean - first_half_mean) / abs(first_half_mean) * 100 if first_half_mean != 0 else 0
                    severity = "high" if change_pct > 20 else "medium" if change_pct > 10 else "low"
                    risks.append(SupplierRisk(
                        supplier=sup,
                        risk_type="declining",
                        severity=severity,
                        description=(
                            f"{sup} shows declining {col}: "
                            f"{first_half_mean:.2f} -> {second_half_mean:.2f} "
                            f"({change_pct:.1f}% change)."
                        ),
                        affected_metric=col,
                    ))

    return risks


# ---------------------------------------------------------------------------
# 4. supplier_concentration
# ---------------------------------------------------------------------------


def supplier_concentration(
    rows: list[dict],
    supplier_column: str,
    value_column: str,
) -> ConcentrationResult | None:
    """Compute supplier concentration using the Herfindahl-Hirschman Index (HHI).

    HHI = sum of squared market shares (each as a percentage).
    - HHI < 1500 → low concentration
    - 1500 <= HHI < 2500 → moderate concentration
    - HHI >= 2500 → high concentration

    Args:
        rows: Data rows as dicts.
        supplier_column: Column identifying the supplier.
        value_column: Numeric column representing spend / volume.

    Returns:
        ConcentrationResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate by supplier
    totals: dict[str, float] = {}
    for row in rows:
        supplier = row.get(supplier_column)
        val = row.get(value_column)
        if supplier is None or val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        totals[str(supplier)] = totals.get(str(supplier), 0.0) + fval

    if not totals:
        return None

    grand_total = sum(totals.values())
    if grand_total == 0:
        return None

    # Shares
    supplier_list: list[dict] = []
    for sup, val in totals.items():
        share = val / grand_total * 100
        supplier_list.append({"supplier": sup, "share_pct": round(share, 2)})

    supplier_list.sort(key=lambda x: -x["share_pct"])

    # HHI
    hhi = sum(s["share_pct"] ** 2 for s in supplier_list)
    hhi = round(hhi, 2)

    if hhi < 1500:
        concentration_level = "low"
    elif hhi < 2500:
        concentration_level = "moderate"
    else:
        concentration_level = "high"

    top_share = supplier_list[0]["share_pct"]

    summary = (
        f"Supplier concentration (HHI = {hhi:.0f}, {concentration_level}). "
        f"{len(supplier_list)} suppliers. "
        f"Top supplier ({supplier_list[0]['supplier']}) holds {top_share:.1f}% share."
    )

    return ConcentrationResult(
        hhi=hhi,
        concentration_level=concentration_level,
        top_supplier_share=top_share,
        suppliers=supplier_list,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_scorecard
# ---------------------------------------------------------------------------


def format_scorecard(result: ScorecardResult) -> str:
    """Format a scorecard result as a human-readable text table.

    Args:
        result: A ScorecardResult from build_scorecard.

    Returns:
        Multi-line string with the formatted table.
    """
    lines = [
        "Supplier Scorecard",
        "=" * 72,
        result.summary,
        "",
        f"{'Rank':<6}{'Supplier':<25}{'Score':>8}{'Grade':>7}{'Strengths':>14}{'Weaknesses':>14}",
        "-" * 72,
    ]

    for ss in result.suppliers:
        strengths_str = ", ".join(ss.strengths) if ss.strengths else "-"
        weaknesses_str = ", ".join(ss.weaknesses) if ss.weaknesses else "-"
        lines.append(
            f"{ss.rank:<6}{ss.supplier:<25}{ss.score:>8.1f}{ss.grade:>7}"
            f"  {strengths_str:<12}{weaknesses_str}"
        )

    lines.append("-" * 72)
    lines.append(
        f"Grade distribution: "
        + ", ".join(f"{g}: {c}" for g, c in sorted(result.grade_distribution.items()))
    )
    lines.append(f"Mean score: {result.mean_score:.1f}")
    return "\n".join(lines)
