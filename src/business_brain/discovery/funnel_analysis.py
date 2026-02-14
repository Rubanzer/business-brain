"""Funnel analysis — track conversion through sequential stages.

Pure functions for analyzing how entities progress through a series of
stages (e.g., lead → prospect → customer → repeat customer).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FunnelStage:
    """Metrics for a single funnel stage."""
    name: str
    count: int
    pct_of_total: float  # % of initial stage count
    conversion_rate: float  # % of previous stage (100% for first)
    drop_off: int  # count lost from previous stage
    drop_off_pct: float  # % lost from previous stage


@dataclass
class FunnelResult:
    """Complete funnel analysis result."""
    stages: list[FunnelStage]
    total_stages: int
    initial_count: int
    final_count: int
    overall_conversion: float  # final / initial * 100
    biggest_drop_stage: str
    biggest_drop_pct: float
    summary: str


def analyze_funnel(
    stage_counts: list[tuple[str, int]],
) -> FunnelResult | None:
    """Analyze a funnel from stage counts.

    Args:
        stage_counts: List of (stage_name, count) tuples in order.
                      Counts should be monotonically non-increasing.

    Returns:
        FunnelResult or None if insufficient data.
    """
    if not stage_counts or len(stage_counts) < 2:
        return None

    initial = stage_counts[0][1]
    if initial <= 0:
        return None

    stages: list[FunnelStage] = []
    biggest_drop = ""
    biggest_drop_pct = 0.0

    for i, (name, count) in enumerate(stage_counts):
        count = max(0, count)
        pct_of_total = count / initial * 100

        if i == 0:
            conversion_rate = 100.0
            drop_off = 0
            drop_off_pct = 0.0
        else:
            prev_count = stage_counts[i - 1][1]
            conversion_rate = (count / prev_count * 100) if prev_count > 0 else 0.0
            drop_off = max(0, prev_count - count)
            drop_off_pct = (drop_off / prev_count * 100) if prev_count > 0 else 0.0

            if drop_off_pct > biggest_drop_pct:
                biggest_drop_pct = drop_off_pct
                biggest_drop = name

        stages.append(FunnelStage(
            name=name,
            count=count,
            pct_of_total=round(pct_of_total, 1),
            conversion_rate=round(conversion_rate, 1),
            drop_off=drop_off,
            drop_off_pct=round(drop_off_pct, 1),
        ))

    final = stage_counts[-1][1]
    overall = final / initial * 100 if initial > 0 else 0.0

    summary = (
        f"Funnel: {initial} → {final} ({overall:.1f}% overall conversion). "
        f"{len(stage_counts)} stages. "
    )
    if biggest_drop:
        summary += f"Biggest drop: {biggest_drop} ({biggest_drop_pct:.1f}% lost)."

    return FunnelResult(
        stages=stages,
        total_stages=len(stages),
        initial_count=initial,
        final_count=final,
        overall_conversion=round(overall, 1),
        biggest_drop_stage=biggest_drop,
        biggest_drop_pct=round(biggest_drop_pct, 1),
        summary=summary,
    )


def funnel_from_rows(
    rows: list[dict],
    entity_column: str,
    stage_column: str,
    stage_order: list[str],
) -> FunnelResult | None:
    """Build a funnel from row data.

    Counts unique entities that appear at each stage.
    An entity at stage N is assumed to have passed through stages 0..N-1.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (e.g., customer_id).
        stage_column: Column containing the stage name.
        stage_order: Ordered list of stage names.

    Returns:
        FunnelResult or None.
    """
    if not rows or not stage_order or len(stage_order) < 2:
        return None

    # Find the highest stage reached by each entity
    stage_index = {s: i for i, s in enumerate(stage_order)}
    entity_max_stage: dict[str, int] = {}

    for row in rows:
        entity = row.get(entity_column)
        stage = row.get(stage_column)
        if entity is None or stage is None:
            continue
        idx = stage_index.get(str(stage))
        if idx is None:
            continue
        key = str(entity)
        if key not in entity_max_stage or idx > entity_max_stage[key]:
            entity_max_stage[key] = idx

    if not entity_max_stage:
        return None

    # Count entities at each stage (entities at stage N also count for stages 0..N)
    stage_counts = []
    for i, stage_name in enumerate(stage_order):
        count = sum(1 for max_idx in entity_max_stage.values() if max_idx >= i)
        stage_counts.append((stage_name, count))

    return analyze_funnel(stage_counts)


def compare_funnels(funnel_a: FunnelResult, funnel_b: FunnelResult) -> dict:
    """Compare two funnels (e.g., two time periods or A/B test).

    Returns comparison metrics.
    """
    return {
        "overall_conversion_a": funnel_a.overall_conversion,
        "overall_conversion_b": funnel_b.overall_conversion,
        "conversion_diff": round(funnel_b.overall_conversion - funnel_a.overall_conversion, 1),
        "initial_count_a": funnel_a.initial_count,
        "initial_count_b": funnel_b.initial_count,
        "final_count_a": funnel_a.final_count,
        "final_count_b": funnel_b.final_count,
        "improved": funnel_b.overall_conversion > funnel_a.overall_conversion,
        "stage_comparison": [
            {
                "stage": sa.name,
                "conversion_a": sa.conversion_rate,
                "conversion_b": sb.conversion_rate,
                "diff": round(sb.conversion_rate - sa.conversion_rate, 1),
            }
            for sa, sb in zip(funnel_a.stages, funnel_b.stages)
        ] if len(funnel_a.stages) == len(funnel_b.stages) else [],
    }


def format_funnel_text(result: FunnelResult) -> str:
    """Format funnel as a text visualization."""
    lines = [f"Funnel Analysis", f"{'='*50}"]
    max_count = result.initial_count

    for stage in result.stages:
        bar_width = int(stage.count / max_count * 40) if max_count > 0 else 0
        bar = "█" * bar_width
        lines.append(
            f"  {stage.name:<20} {bar} {stage.count:>6} ({stage.pct_of_total:>5.1f}%)"
        )
        if stage.drop_off > 0:
            lines.append(
                f"  {'':20} ↓ {stage.drop_off} dropped ({stage.drop_off_pct:.1f}%)"
            )

    lines.append(f"{'='*50}")
    lines.append(f"Overall: {result.overall_conversion:.1f}% conversion")
    return "\n".join(lines)
