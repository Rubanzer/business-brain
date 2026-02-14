"""Pipeline analyzer — stage analysis, velocity, win rates, and forecasting.

Pure functions for analyzing sales pipeline data across multiple dimensions:
stage distribution, conversion rates, pipeline velocity, win/loss rates,
and weighted forecasting.
"""

from __future__ import annotations

import re
from collections import defaultdict
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


def _parse_date(val) -> datetime | None:
    """Attempt to parse a date from string or datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(val), fmt)
        except ValueError:
            continue
    return None


# Canonical stage ordering for pipeline analysis.  Stages are matched
# case-insensitively using substring / keyword matching so that various
# CRM naming conventions (e.g. "Closed Won", "closed-won", "Won") all map
# to the same canonical position.
_CANONICAL_STAGES = [
    "lead",
    "qualified",
    "proposal",
    "negotiation",
    "won",
    "lost",
]

_WON_PATTERNS = re.compile(
    r"\b(won|closed[\s\-_]?won|win|closed[\s\-_]?win)\b", re.IGNORECASE
)
_LOST_PATTERNS = re.compile(
    r"\b(lost|closed[\s\-_]?lost|lose|closed[\s\-_]?lose)\b", re.IGNORECASE
)


def _is_won(stage: str) -> bool:
    """Return True if stage name indicates a won deal."""
    return bool(_WON_PATTERNS.search(stage))


def _is_lost(stage: str) -> bool:
    """Return True if stage name indicates a lost deal."""
    return bool(_LOST_PATTERNS.search(stage))


def _canonical_order(stage: str) -> int | None:
    """Map a stage name to a canonical ordering index, or None."""
    lower = stage.lower().strip()
    for idx, canonical in enumerate(_CANONICAL_STAGES):
        if canonical in lower or lower in canonical:
            return idx
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    stage: str
    deal_count: int
    total_value: float
    avg_value: float
    pct_of_deals: float
    pct_of_value: float


@dataclass
class StageConversion:
    """Conversion rate between two adjacent pipeline stages."""
    from_stage: str
    to_stage: str
    conversion_rate: float


@dataclass
class PipelineStageResult:
    """Overall pipeline stage analysis result."""
    stages: list[StageMetrics]
    conversions: list[StageConversion]
    total_deals: int
    total_value: float
    weighted_value: float
    summary: str


@dataclass
class StageVelocity:
    """Velocity metrics for a single pipeline stage."""
    stage: str
    avg_days: float
    deal_count: int


@dataclass
class PipelineVelocityResult:
    """Pipeline velocity analysis result."""
    stage_velocities: list[StageVelocity]
    avg_cycle_days: float
    fastest_deal_days: float
    slowest_deal_days: float
    bottleneck_stage: str
    summary: str


@dataclass
class OwnerWinRate:
    """Win rate for a single deal owner."""
    owner: str
    won: int
    lost: int
    win_rate: float
    won_value: float


@dataclass
class WinRateResult:
    """Win rate analysis result."""
    overall_win_rate: float
    total_won: int
    total_lost: int
    won_value: float
    lost_value: float
    by_owner: list[OwnerWinRate]
    best_performer: str | None
    summary: str


@dataclass
class StageForecast:
    """Forecast for a single pipeline stage."""
    stage: str
    deal_count: int
    raw_value: float
    probability: float
    weighted_value: float


@dataclass
class PipelineForecastResult:
    """Pipeline forecast result."""
    stages: list[StageForecast]
    total_raw: float
    total_weighted: float
    expected_close_value: float
    coverage_ratio: float
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_pipeline_stages
# ---------------------------------------------------------------------------


def analyze_pipeline_stages(
    rows: list[dict],
    deal_column: str,
    stage_column: str,
    value_column: str,
    owner_column: str | None = None,
) -> PipelineStageResult | None:
    """Analyze pipeline by stage, computing counts, values, and conversions.

    Args:
        rows: Data rows as dicts.
        deal_column: Column identifying unique deals.
        stage_column: Column containing pipeline stage names.
        value_column: Column containing deal values.
        owner_column: Optional column for deal owner.

    Returns:
        PipelineStageResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per deal: take the latest stage and max value
    deal_data: dict[str, dict] = {}
    for row in rows:
        deal = row.get(deal_column)
        stage = row.get(stage_column)
        value = _safe_float(row.get(value_column))
        if deal is None or stage is None or value is None:
            continue

        key = str(deal)
        stage_str = str(stage)
        if key not in deal_data:
            deal_data[key] = {"stage": stage_str, "value": value}
        else:
            # Take max value, keep last-seen stage
            deal_data[key]["stage"] = stage_str
            if value > deal_data[key]["value"]:
                deal_data[key]["value"] = value

        if owner_column:
            owner = row.get(owner_column)
            if owner is not None:
                deal_data[key]["owner"] = str(owner)

    if not deal_data:
        return None

    total_deals = len(deal_data)
    total_value = sum(d["value"] for d in deal_data.values())

    # Aggregate per stage
    stage_counts: dict[str, int] = defaultdict(int)
    stage_values: dict[str, float] = defaultdict(float)

    for d in deal_data.values():
        stage_counts[d["stage"]] += 1
        stage_values[d["stage"]] += d["value"]

    # Try canonical ordering; fall back to deal-count ordering
    stage_names = list(stage_counts.keys())
    canonical_mapped = {s: _canonical_order(s) for s in stage_names}

    if all(v is not None for v in canonical_mapped.values()):
        # All stages map to canonical — use canonical order
        ordered_stages = sorted(stage_names, key=lambda s: canonical_mapped[s])
    else:
        # Auto-order by deal count descending (most deals = earliest stage)
        ordered_stages = sorted(
            stage_names, key=lambda s: stage_counts[s], reverse=True
        )

    # Build StageMetrics
    stages: list[StageMetrics] = []
    for s in ordered_stages:
        count = stage_counts[s]
        total_val = stage_values[s]
        avg_val = total_val / count if count > 0 else 0.0
        pct_deals = (count / total_deals * 100) if total_deals > 0 else 0.0
        pct_value = (total_val / total_value * 100) if total_value > 0 else 0.0
        stages.append(StageMetrics(
            stage=s,
            deal_count=count,
            total_value=round(total_val, 2),
            avg_value=round(avg_val, 2),
            pct_of_deals=round(pct_deals, 2),
            pct_of_value=round(pct_value, 2),
        ))

    # Conversion rates between adjacent stages
    conversions: list[StageConversion] = []
    for i in range(len(ordered_stages) - 1):
        from_stage = ordered_stages[i]
        to_stage = ordered_stages[i + 1]
        from_count = stage_counts[from_stage]
        to_count = stage_counts[to_stage]
        rate = (to_count / from_count * 100) if from_count > 0 else 0.0
        conversions.append(StageConversion(
            from_stage=from_stage,
            to_stage=to_stage,
            conversion_rate=round(rate, 2),
        ))

    # Weighted value: weight each stage proportionally by position
    # (later stages get higher weight)
    weighted_value = 0.0
    n_stages = len(ordered_stages)
    if n_stages > 0:
        for i, s in enumerate(ordered_stages):
            weight = (i + 1) / n_stages
            weighted_value += stage_values[s] * weight

    summary = (
        f"Pipeline: {total_deals} deals across {len(ordered_stages)} stages. "
        f"Total value: {total_value:,.2f}. "
        f"Weighted value: {weighted_value:,.2f}."
    )

    return PipelineStageResult(
        stages=stages,
        conversions=conversions,
        total_deals=total_deals,
        total_value=round(total_value, 2),
        weighted_value=round(weighted_value, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. analyze_pipeline_velocity
# ---------------------------------------------------------------------------


def analyze_pipeline_velocity(
    rows: list[dict],
    deal_column: str,
    stage_column: str,
    value_column: str,
    date_column: str,
    close_date_column: str | None = None,
) -> PipelineVelocityResult | None:
    """Calculate pipeline velocity and identify bottleneck stages.

    Args:
        rows: Data rows as dicts.
        deal_column: Column identifying unique deals.
        stage_column: Column containing pipeline stage names.
        value_column: Column containing deal values.
        date_column: Column containing entry/activity dates.
        close_date_column: Optional column for deal close date.

    Returns:
        PipelineVelocityResult or None if insufficient data.
    """
    if not rows:
        return None

    # Collect per-deal stage timestamps
    deal_stages: dict[str, dict] = {}
    for row in rows:
        deal = row.get(deal_column)
        stage = row.get(stage_column)
        value = _safe_float(row.get(value_column))
        dt = _parse_date(row.get(date_column))
        if deal is None or stage is None or value is None or dt is None:
            continue

        key = str(deal)
        stage_str = str(stage)
        if key not in deal_stages:
            deal_stages[key] = {
                "value": value,
                "stages": {},
                "dates": [],
                "close_date": None,
            }
        else:
            if value > deal_stages[key]["value"]:
                deal_stages[key]["value"] = value

        # Track earliest date for each stage per deal
        if stage_str not in deal_stages[key]["stages"]:
            deal_stages[key]["stages"][stage_str] = dt
        else:
            if dt < deal_stages[key]["stages"][stage_str]:
                deal_stages[key]["stages"][stage_str] = dt

        deal_stages[key]["dates"].append(dt)

        # Close date
        if close_date_column:
            close_dt = _parse_date(row.get(close_date_column))
            if close_dt is not None:
                existing = deal_stages[key]["close_date"]
                if existing is None or close_dt > existing:
                    deal_stages[key]["close_date"] = close_dt

    if not deal_stages:
        return None

    # Calculate time spent in each stage across all deals
    stage_durations: dict[str, list[float]] = defaultdict(list)

    for deal_info in deal_stages.values():
        stage_dates = deal_info["stages"]
        if len(stage_dates) < 2:
            continue

        sorted_stages = sorted(stage_dates.items(), key=lambda x: x[1])
        for i in range(len(sorted_stages) - 1):
            s_name, s_date = sorted_stages[i]
            _, next_date = sorted_stages[i + 1]
            days_in_stage = (next_date - s_date).days
            if days_in_stage >= 0:
                stage_durations[s_name].append(float(days_in_stage))

    # Stage velocities
    stage_velocities: list[StageVelocity] = []
    for stage_name, durations in sorted(
        stage_durations.items(),
        key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
        reverse=True,
    ):
        avg_days = sum(durations) / len(durations) if durations else 0.0
        stage_velocities.append(StageVelocity(
            stage=stage_name,
            avg_days=round(avg_days, 2),
            deal_count=len(durations),
        ))

    # Overall cycle time (creation to close)
    cycle_days_list: list[float] = []
    for deal_info in deal_stages.values():
        dates = deal_info["dates"]
        close_date = deal_info.get("close_date")
        if close_date is not None and dates:
            earliest = min(dates)
            delta = (close_date - earliest).days
            if delta >= 0:
                cycle_days_list.append(float(delta))
        elif len(dates) >= 2:
            sorted_dates = sorted(dates)
            delta = (sorted_dates[-1] - sorted_dates[0]).days
            if delta >= 0:
                cycle_days_list.append(float(delta))

    if not cycle_days_list and not stage_velocities:
        return None

    avg_cycle_days = (
        sum(cycle_days_list) / len(cycle_days_list) if cycle_days_list else 0.0
    )
    fastest_deal_days = min(cycle_days_list) if cycle_days_list else 0.0
    slowest_deal_days = max(cycle_days_list) if cycle_days_list else 0.0

    # Bottleneck = stage with longest average time
    bottleneck_stage = ""
    if stage_velocities:
        bottleneck_stage = stage_velocities[0].stage  # already sorted desc

    summary_parts = [
        f"Pipeline velocity: avg cycle {avg_cycle_days:.1f} days.",
    ]
    if fastest_deal_days != slowest_deal_days:
        summary_parts.append(
            f"Range: {fastest_deal_days:.0f}-{slowest_deal_days:.0f} days."
        )
    if bottleneck_stage:
        summary_parts.append(f"Bottleneck: {bottleneck_stage}.")

    return PipelineVelocityResult(
        stage_velocities=stage_velocities,
        avg_cycle_days=round(avg_cycle_days, 2),
        fastest_deal_days=round(fastest_deal_days, 2),
        slowest_deal_days=round(slowest_deal_days, 2),
        bottleneck_stage=bottleneck_stage,
        summary=" ".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 3. compute_win_rate
# ---------------------------------------------------------------------------


def compute_win_rate(
    rows: list[dict],
    deal_column: str,
    stage_column: str,
    value_column: str | None = None,
    owner_column: str | None = None,
) -> WinRateResult | None:
    """Compute win/loss rates overall and by owner.

    Won deals are identified by stage names containing keywords like
    ``won``, ``closed won``, ``win``.  Lost deals use ``lost``,
    ``closed lost``, etc.

    Args:
        rows: Data rows as dicts.
        deal_column: Column identifying unique deals.
        stage_column: Column containing pipeline stage names.
        value_column: Optional column for deal value.
        owner_column: Optional column for deal owner.

    Returns:
        WinRateResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per deal
    deal_data: dict[str, dict] = {}
    for row in rows:
        deal = row.get(deal_column)
        stage = row.get(stage_column)
        if deal is None or stage is None:
            continue

        key = str(deal)
        stage_str = str(stage)

        if key not in deal_data:
            deal_data[key] = {"stage": stage_str, "value": 0.0, "owner": None}

        # Keep last-seen stage
        deal_data[key]["stage"] = stage_str

        if value_column:
            value = _safe_float(row.get(value_column))
            if value is not None and value > deal_data[key]["value"]:
                deal_data[key]["value"] = value

        if owner_column:
            owner = row.get(owner_column)
            if owner is not None:
                deal_data[key]["owner"] = str(owner)

    if not deal_data:
        return None

    # Classify deals
    total_won = 0
    total_lost = 0
    won_value = 0.0
    lost_value = 0.0

    owner_stats: dict[str, dict] = defaultdict(
        lambda: {"won": 0, "lost": 0, "won_value": 0.0}
    )

    for d in deal_data.values():
        stage = d["stage"]
        value = d["value"]
        owner = d.get("owner")

        is_deal_won = _is_won(stage)
        is_deal_lost = _is_lost(stage)

        if is_deal_won:
            total_won += 1
            won_value += value
            if owner:
                owner_stats[owner]["won"] += 1
                owner_stats[owner]["won_value"] += value
        elif is_deal_lost:
            total_lost += 1
            lost_value += value
            if owner:
                owner_stats[owner]["lost"] += 1

    total_decided = total_won + total_lost
    if total_decided == 0:
        return None

    overall_win_rate = (total_won / total_decided * 100) if total_decided > 0 else 0.0

    # By owner
    by_owner: list[OwnerWinRate] = []
    if owner_column:
        for owner_name, stats in sorted(owner_stats.items()):
            decided = stats["won"] + stats["lost"]
            rate = (stats["won"] / decided * 100) if decided > 0 else 0.0
            by_owner.append(OwnerWinRate(
                owner=owner_name,
                won=stats["won"],
                lost=stats["lost"],
                win_rate=round(rate, 2),
                won_value=round(stats["won_value"], 2),
            ))

    # Best performer = owner with highest win rate (min 1 decided deal)
    best_performer: str | None = None
    if by_owner:
        best = max(by_owner, key=lambda o: (o.win_rate, o.won))
        best_performer = best.owner

    summary = (
        f"Win rate: {overall_win_rate:.1f}% "
        f"({total_won} won, {total_lost} lost). "
        f"Won value: {won_value:,.2f}, lost value: {lost_value:,.2f}."
    )
    if best_performer:
        summary += f" Best performer: {best_performer}."

    return WinRateResult(
        overall_win_rate=round(overall_win_rate, 2),
        total_won=total_won,
        total_lost=total_lost,
        won_value=round(won_value, 2),
        lost_value=round(lost_value, 2),
        by_owner=by_owner,
        best_performer=best_performer,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. forecast_pipeline
# ---------------------------------------------------------------------------

# Default probabilities assigned by stage position when no probability
# column is provided.  Index 0 = earliest stage.
_DEFAULT_PROBABILITIES = [0.10, 0.25, 0.50, 0.70, 0.90, 1.00]


def forecast_pipeline(
    rows: list[dict],
    deal_column: str,
    stage_column: str,
    value_column: str,
    probability_column: str | None = None,
) -> PipelineForecastResult | None:
    """Forecast weighted pipeline value by stage.

    When *probability_column* is not provided, default probabilities are
    assigned based on each stage's position (auto-ordered by deal count).

    Args:
        rows: Data rows as dicts.
        deal_column: Column identifying unique deals.
        stage_column: Column containing pipeline stage names.
        value_column: Column containing deal values.
        probability_column: Optional column for deal probability (0-1 or 0-100).

    Returns:
        PipelineForecastResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per deal
    deal_data: dict[str, dict] = {}
    for row in rows:
        deal = row.get(deal_column)
        stage = row.get(stage_column)
        value = _safe_float(row.get(value_column))
        if deal is None or stage is None or value is None:
            continue

        key = str(deal)
        stage_str = str(stage)

        prob: float | None = None
        if probability_column:
            prob = _safe_float(row.get(probability_column))
            if prob is not None and prob > 1:
                prob = prob / 100.0
            if prob is not None:
                prob = max(0.0, min(prob, 1.0))

        if key not in deal_data:
            deal_data[key] = {
                "stage": stage_str,
                "value": value,
                "probability": prob,
            }
        else:
            deal_data[key]["stage"] = stage_str
            if value > deal_data[key]["value"]:
                deal_data[key]["value"] = value
            if prob is not None:
                deal_data[key]["probability"] = prob

    if not deal_data:
        return None

    # Determine stage ordering (by deal count, most = earliest)
    stage_counts: dict[str, int] = defaultdict(int)
    for d in deal_data.values():
        stage_counts[d["stage"]] += 1

    ordered_stages = sorted(
        stage_counts.keys(), key=lambda s: stage_counts[s], reverse=True
    )

    # Assign default probabilities if needed
    n_stages = len(ordered_stages)
    stage_prob_map: dict[str, float] = {}
    for i, s in enumerate(ordered_stages):
        if i < len(_DEFAULT_PROBABILITIES):
            stage_prob_map[s] = _DEFAULT_PROBABILITIES[i]
        else:
            # Linearly interpolate for extra stages
            stage_prob_map[s] = min(1.0, 0.10 + 0.90 * (i / max(n_stages - 1, 1)))

    # Fill in missing probabilities with defaults
    for d in deal_data.values():
        if d["probability"] is None:
            d["probability"] = stage_prob_map.get(d["stage"], 0.5)

    # Aggregate per stage
    stage_raw: dict[str, float] = defaultdict(float)
    stage_weighted: dict[str, float] = defaultdict(float)
    stage_deal_counts: dict[str, int] = defaultdict(int)
    stage_probs: dict[str, list[float]] = defaultdict(list)

    for d in deal_data.values():
        s = d["stage"]
        stage_raw[s] += d["value"]
        stage_weighted[s] += d["value"] * d["probability"]
        stage_deal_counts[s] += 1
        stage_probs[s].append(d["probability"])

    # Build StageForecast
    stages: list[StageForecast] = []
    for s in ordered_stages:
        avg_prob = (
            sum(stage_probs[s]) / len(stage_probs[s])
            if stage_probs[s]
            else 0.0
        )
        stages.append(StageForecast(
            stage=s,
            deal_count=stage_deal_counts[s],
            raw_value=round(stage_raw[s], 2),
            probability=round(avg_prob, 4),
            weighted_value=round(stage_weighted[s], 2),
        ))

    total_raw = sum(stage_raw.values())
    total_weighted = sum(stage_weighted.values())

    # Expected close value = weighted value of stages that look like won/close
    expected_close_value = total_weighted

    # Coverage ratio = total_raw / total_weighted (how much pipeline needed
    # to produce the expected weighted value)
    coverage_ratio = (
        total_raw / total_weighted if total_weighted > 0 else 0.0
    )

    summary = (
        f"Pipeline forecast: {len(deal_data)} deals, "
        f"raw value {total_raw:,.2f}, weighted value {total_weighted:,.2f}. "
        f"Coverage ratio: {coverage_ratio:.2f}x."
    )

    return PipelineForecastResult(
        stages=stages,
        total_raw=round(total_raw, 2),
        total_weighted=round(total_weighted, 2),
        expected_close_value=round(expected_close_value, 2),
        coverage_ratio=round(coverage_ratio, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_pipeline_report
# ---------------------------------------------------------------------------


def format_pipeline_report(
    stages: PipelineStageResult | None = None,
    velocity: PipelineVelocityResult | None = None,
    win_rate: WinRateResult | None = None,
    forecast: PipelineForecastResult | None = None,
) -> str:
    """Combine pipeline analysis results into a formatted text report.

    Args:
        stages: Pipeline stage analysis result.
        velocity: Pipeline velocity result.
        win_rate: Win rate analysis result.
        forecast: Pipeline forecast result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("PIPELINE ANALYSIS REPORT")
    sections.append("=" * 60)

    if stages:
        lines = [
            "",
            "PIPELINE STAGES",
            "-" * 40,
            f"  Total Deals:     {stages.total_deals:>15}",
            f"  Total Value:     {stages.total_value:>15,.2f}",
            f"  Weighted Value:  {stages.weighted_value:>15,.2f}",
            "",
            "  Stages:",
        ]
        for s in stages.stages:
            lines.append(
                f"    {s.stage:<20} {s.deal_count:>4} deals  "
                f"{s.total_value:>12,.2f}  ({s.pct_of_deals:.1f}%)"
            )
        if stages.conversions:
            lines.append("")
            lines.append("  Conversions:")
            for c in stages.conversions:
                lines.append(
                    f"    {c.from_stage} -> {c.to_stage}: {c.conversion_rate:.1f}%"
                )
        sections.append("\n".join(lines))

    if velocity:
        lines = [
            "",
            "PIPELINE VELOCITY",
            "-" * 40,
            f"  Avg Cycle:       {velocity.avg_cycle_days:>12.1f} days",
            f"  Fastest Deal:    {velocity.fastest_deal_days:>12.1f} days",
            f"  Slowest Deal:    {velocity.slowest_deal_days:>12.1f} days",
        ]
        if velocity.bottleneck_stage:
            lines.append(f"  Bottleneck:      {velocity.bottleneck_stage:>15}")
        if velocity.stage_velocities:
            lines.append("")
            lines.append("  Stage Durations:")
            for sv in velocity.stage_velocities:
                lines.append(
                    f"    {sv.stage:<20} {sv.avg_days:>8.1f} days  "
                    f"({sv.deal_count} deals)"
                )
        sections.append("\n".join(lines))

    if win_rate:
        lines = [
            "",
            "WIN RATE ANALYSIS",
            "-" * 40,
            f"  Win Rate:        {win_rate.overall_win_rate:>14.1f}%",
            f"  Won:             {win_rate.total_won:>15}",
            f"  Lost:            {win_rate.total_lost:>15}",
            f"  Won Value:       {win_rate.won_value:>15,.2f}",
            f"  Lost Value:      {win_rate.lost_value:>15,.2f}",
        ]
        if win_rate.best_performer:
            lines.append(f"  Best Performer:  {win_rate.best_performer:>15}")
        if win_rate.by_owner:
            lines.append("")
            lines.append("  By Owner:")
            for o in win_rate.by_owner:
                lines.append(
                    f"    {o.owner:<20} {o.win_rate:>6.1f}%  "
                    f"({o.won}W/{o.lost}L)  value: {o.won_value:,.2f}"
                )
        sections.append("\n".join(lines))

    if forecast:
        lines = [
            "",
            "PIPELINE FORECAST",
            "-" * 40,
            f"  Raw Value:       {forecast.total_raw:>15,.2f}",
            f"  Weighted Value:  {forecast.total_weighted:>15,.2f}",
            f"  Expected Close:  {forecast.expected_close_value:>15,.2f}",
            f"  Coverage Ratio:  {forecast.coverage_ratio:>14.2f}x",
            "",
            "  Stage Forecast:",
        ]
        for sf in forecast.stages:
            lines.append(
                f"    {sf.stage:<20} {sf.deal_count:>4} deals  "
                f"raw: {sf.raw_value:>10,.2f}  "
                f"prob: {sf.probability:.0%}  "
                f"wtd: {sf.weighted_value:>10,.2f}"
            )
        sections.append("\n".join(lines))

    if not any([stages, velocity, win_rate, forecast]):
        sections.append("\nNo data available for report.")

    sections.append("\n" + "=" * 60)
    return "\n".join(sections)
